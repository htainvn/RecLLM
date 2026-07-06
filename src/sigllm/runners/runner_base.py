"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from sigllm.common.data_utils import reorg_datasets_by_split
from sigllm.common.dist_utils import *
from sigllm.common.registry import registry
from sigllm.common.utils import is_url
from sigllm.runners.utils.optimizer_builder import build_optimizer
from sigllm.runners.utils.scheduler_builder import build_scheduler
from sigllm.runners.utils.scaler_builder import build_scaler
from sigllm.runners.utils.model_wrapper import wrap_model
from sigllm.runners.utils.dataloader_builder import build_dataloaders

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id
        self.task = task
        self.datasets = datasets
        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None
        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        if self._wrapped_model is None:
            self._wrapped_model = wrap_model(
                self._model, self.config, self.device, self.use_distributed
            )
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = build_optimizer(self.model, self.config)
        return self._optimizer

    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = build_scaler(self.config)

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            self._lr_sched = build_scheduler(
                optimizer=self.optimizer,
                config=self.config,
                dataloaders=self.dataloaders,
                max_epoch=self.max_epoch,
                min_lr=self.min_lr,
                init_lr=self.init_lr,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.
        """
        if self._dataloaders is None:
            self.datasets = reorg_datasets_by_split(self.datasets)

            self._dataloaders = build_dataloaders(
                datasets=self.datasets,
                config=self.config,
                train_splits=self.train_splits,
                use_distributed=self.use_distributed,
                use_dist_eval_sampler=self.use_dist_eval_sampler
            )

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir
    
    def model_to_be_trained(self):
        if self.use_distributed:
            return self.model.module.to_be_trained()
        else:
            return self.model.to_be_trained()

    def train(self):
        start_time = time.time()
        best_agg_metric = -100000
        best_epoch = 0
        not_change = 0
        self.set_model_mode(self.config.run_cfg.mode)
    

        self.log_config()
        stop_training_flag = False
        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        if not self.evaluate_only:# with training
            for cur_epoch in range(self.start_epoch, self.max_epoch):
                # training phase
                if not self.evaluate_only and self.model_to_be_trained():
                    logging.info("Start training")
                    # having lora or IDs are used
                    train_stats = self.train_epoch(cur_epoch)
                    self.log_stats(split_name="train", stats=train_stats)
                    # torch.cuda.empty_cache()
                
                        
                # evaluation phase. run.valid_freq=N evaluates every N epochs
                # (default 1). With a 7B LLM the full-valid eval can cost 2-3x
                # the training epoch itself, so N=2 nearly halves wall-clock.
                # NOTE: the early-stop counter ticks per EVAL, so patience
                # covers valid_freq * 20 epochs.
                valid_freq = max(int(self.config.run_cfg.get("valid_freq", 1)), 1)
                run_valid = len(self.valid_splits) > 0 and (cur_epoch + 1) % valid_freq == 0
                if run_valid:
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))

                        val_log = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )
                        # torch.cuda.empty_cache()
                        
                        if val_log is not None:
                            if is_main_process():
                                assert (
                                    "agg_metrics" in val_log
                                ), "No agg_metrics found in validation log."

                                agg_metrics = val_log["agg_metrics"]
                                if agg_metrics > best_agg_metric and split_name == "valid":
                                    best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                    self._save_checkpoint(cur_epoch, is_best=True)
                                    not_change = 0
                                    
                                    
                                    # logging.info("Evaluating on {}.".format('test'))
                                    # test_log = self.eval_epoch(split_name='test', cur_epoch='best', skip_reload=True)
                                    # logging.info("testing result:", test_log)

                                val_log.update({"best_epoch": best_epoch})
                                self.log_stats(val_log, split_name)
                                not_change += 1
                                # if not_change > 20: # early stop
                                #     break
                        # torch.cuda.empty_cache()

                elif len(self.valid_splits) == 0:
                    # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                    if not self.evaluate_only:
                        self._save_checkpoint(cur_epoch, is_best=False)

                if self.evaluate_only:
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()
                if not self.model_to_be_trained():
                    break
                # Patience counts EVALS (not epochs): tolerance in epochs is
                # early_stop_patience * valid_freq. Tune together.
                es_patience = int(self.config.run_cfg.get("early_stop_patience", 20))
                if not_change > es_patience:
                    logging.info(
                        "Early stop. No valid improvement in %d consecutive evals.",
                        es_patience,
                    )
                    break

        # testing phase, would only run when evaluate_only==True
        if self.evaluate_only:
            print("training finish or just evaluation...")
            logging.info("Evaluating on {}.".format(self.test_splits[0]))
            test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
            self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
        self.set_model_mode(None) # recover to the default model

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model
    
    def set_model_mode(self,mode):
        if self.use_distributed:
            self.model.module.set_mode(mode)
        else:
            self.model.set_mode(mode)

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
