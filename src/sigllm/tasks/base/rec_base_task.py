import json
import logging
import os
from typing import Optional
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.distributed as dist
from sigllm import datasets
from sigllm.common import registry
from sigllm.common.data_utils import move_to_cuda
from sigllm.common.dist_utils import *
from sigllm.common.logger import MetricLogger, SmoothedValue
from sigllm.common.logging_utils import NotebookLogger
from sigllm.models.multimodal.qformer_rec_llm import QRecLLM
from sigllm.pipelines.rec.train_rec_baseline import calculate_user_auc, build_user_interaction_counts

LOGGER = NotebookLogger.rich_logger("sigllm.tasks.base.rec_base_task")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


class RecBaseTask:
    def __init__(self):
        super().__init__()
        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_runner(self, cfg, job_id, task, model, datasets):
        runner_cls =  registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_base"))
        return runner_cls(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    
    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)
    
    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        evaluate_only = cfg.run_cfg.evaluate

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name, dataset_config in datasets_config.items():
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(evaluate_only=evaluate_only)
            if 'train' in dataset:
                dataset['train'].name = name
                if 'sample_ratio' in dataset_config:
                    dataset['train'].sample_ratio = dataset_config.sample_ratio
            datasets[name] = dataset

        # Best-checkpoint / early-stop selection metric (default 'auc'; 'uauc' to
        # target per-user AUC). Set via run.best_metric in the config.
        self._best_metric = str(cfg.run_cfg.get("best_metric", "auc")).lower()

        # uAUC protocol filter (SeLLa-Rec averages only over users with >N
        # interactions). 0 = off (default) -> uAUC over all eligible users, as before.
        self._uauc_min_interactions = int(cfg.run_cfg.get("uauc_min_interactions", 0))
        self._uauc_counts = None
        if self._uauc_min_interactions > 0:
            data_path = None
            for _, dcfg in datasets_config.items():
                data_path = dcfg.get("path") or dcfg.get("build_info", {}).get("storage")
                if data_path:
                    break
            if data_path:
                self._uauc_counts = build_user_interaction_counts(data_path)
                logging.info(
                    f"uAUC filter ON: keep users with >{self._uauc_min_interactions} "
                    f"interactions ({len(self._uauc_counts)} users counted from {data_path})"
                )

        return datasets
    
    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss
    
    def valid_step(self, model, samples):
        outputs = model.generate_for_samples(samples)
        return outputs

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def evaluation(self, model, data_loader, cuda_enabled=True):
        return self.evaluate(model=model, data_loader=data_loader, cuda_enabled=cuda_enabled)

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError
    
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        use_amp = scaler is not None

        # Initialize logging utilities
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        iters_per_epoch = lr_scheduler.iters_per_epoch
        header = f"Train: data epoch: [{epoch}]"
        
        # Iter-based training loop
        for step in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            samples = next(data_loader)

            if cuda_enabled:
                samples = move_to_cuda(samples)

            samples.update(
                {
                    "epoch": epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": step,
                }
            )

            lr_scheduler.step(cur_epoch=epoch, cur_step=step)

            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
                loss = loss / accum_grad_iters # Chia loss để hỗ trợ Gradient Accumulation

            # 3. Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 4. Optimizer step (every accum_grad_iters)
            if (step + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # 5. Logging
            metric_logger.update(loss=loss.item() * accum_grad_iters)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Sync stats across all distributed processes (if any)
        metric_logger.synchronize_between_processes()
        log_step(f"Averaged stats: {metric_logger.global_avg()}")
        
        return {k: f"{meter.global_avg:.4f}" for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def evaluate(self, model, data_loader, cuda_enabled=True):
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        all_results = []

        for data_loader in data_loader.loaders:
            raw_outputs = self._collect_predictions(model, data_loader, metric_logger, header, cuda_enabled)
            combined_data = self._gather_distributed_data(raw_outputs)
            metrics = self._compute_metrics(combined_data)
            
            metric_logger.synchronize_between_processes()
            loss_meter = metric_logger.meters.get("loss")
            acc_meter = metric_logger.meters.get("acc")
            val_loss = loss_meter.global_avg if loss_meter is not None else 0.0
            val_acc = acc_meter.global_avg if acc_meter is not None else 0.0
            eval_summary = (
                f"Averaged stats: {metric_logger.global_avg()} "
                f"***auc: {metrics.get('auc', 0):.4f} ***uauc: {metrics.get('uauc', 0):.4f}"
            )
            logging.info(eval_summary)
            log_step(
                "Evaluation metrics",
                (
                    f"val_loss={val_loss:.6f}, "
                    f"AUC={metrics.get('auc', 0):.6f}, "
                    f"uAUC={metrics.get('uauc', 0):.6f}, "
                    f"ACC@0.5={val_acc:.6f}, "
                    f"pos_rate={metrics.get('pos_rate', 0):.4f}, "
                    f"pred_pos_rate@0.5={metrics.get('pred_pos_rate', 0):.4f}"
                ),
            )
            if metrics.get('qformer_uauc', 0):
                log_step(
                    "Q-Former standalone (rank aux head)",
                    (
                        f"qformer_AUC={metrics.get('qformer_auc', 0):.6f}, "
                        f"qformer_uAUC={metrics.get('qformer_uauc', 0):.6f} "
                        f"(vs MF baseline and LLM uAUC above)"
                    ),
                )
            log_step(
                "Score separation",
                (
                    f"pos_score_mean={metrics.get('pos_score_mean', 0):.6f}, "
                    f"neg_score_mean={metrics.get('neg_score_mean', 0):.6f}, "
                    f"score_gap={metrics.get('score_gap', 0):.6f}, "
                    f"score_mean={metrics.get('score_mean', 0):.6f}, "
                    f"score_std={metrics.get('score_std', 0):.6f}"
                ),
            )
            
            # Metric that drives best-checkpoint selection / early stop (higher=better).
            # Default 'auc' (legacy); set run.best_metric='uauc' to optimize per-user AUC.
            best_metric = getattr(self, "_best_metric", "auc")
            agg = metrics.get(best_metric)
            if agg is None:
                agg = -metric_logger.meters['loss'].global_avg
            all_results = {
            'agg_metrics': agg,
            'auc': metrics.get('auc', 0),
            'acc': val_acc,
            'loss': val_loss,
            'uauc': metrics.get('uauc', 0),
            'qformer_auc': metrics.get('qformer_auc', 0),
            'qformer_uauc': metrics.get('qformer_uauc', 0),
            'pos_rate': metrics.get('pos_rate', 0),
            'pred_pos_rate': metrics.get('pred_pos_rate', 0),
            'pos_score_mean': metrics.get('pos_score_mean', 0),
            'neg_score_mean': metrics.get('neg_score_mean', 0),
            'score_gap': metrics.get('score_gap', 0),
            'score_mean': metrics.get('score_mean', 0),
            'score_std': metrics.get('score_std', 0),
        }
        
        return all_results

    def _collect_predictions(self, model, data_loader, logger, header, cuda_enabled):
        results = {'logits': [], 'labels': [], 'users': [], 'aux': []}

        for samples in logger.log_every(data_loader, 10, header):
            if cuda_enabled:
                samples = move_to_cuda(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)

            logger.update(loss=eval_output['loss'].item())

            if 'logits' in eval_output:
                logits = eval_output['logits']
                labels = samples['label']

                results['logits'].append(logits.detach())
                results['labels'].append(labels.detach())
                results['users'].append(samples['UserID'].detach())

                # CHANGE Q2: Q-Former standalone scores from the rank aux head.
                if 'aux_logits' in eval_output:
                    results['aux'].append(eval_output['aux_logits'].detach())

                acc = ((logits > 0.5).float() == labels).float().mean()
                logger.update(acc=acc.item())
                
            torch.cuda.empty_cache()
            
        return {k: torch.cat(v, dim=0) if v else None for k, v in results.items()}


    def _gather_distributed_data(self, data):
        if data['logits'] is None:
            return data
        if not is_dist_avail_and_initialized():
            return {k: v.cpu().numpy() if v is not None else None for k, v in data.items()}
        gathered = {}
        for key, tensor in data.items():
            if tensor is None:
                gathered[key] = None
                continue
            world_size = dist.get_world_size()
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor)
            gathered[key] = torch.cat(tensor_list, dim=0).cpu().numpy()
        return gathered

    def _compute_metrics(self, data):
        if data['logits'] is None:
            return {}
        labels = np.asarray(data['labels']).astype(np.float32)
        scores = np.asarray(data['logits']).astype(np.float32)
        pos_mask = labels == 1
        neg_mask = labels == 0

        auc = roc_auc_score(labels, scores)
        uauc, _, _ = calculate_user_auc(
            data['users'], scores, labels,
            interaction_counts=getattr(self, "_uauc_counts", None),
            min_interactions=getattr(self, "_uauc_min_interactions", 0),
        )

        pos_score_mean = float(scores[pos_mask].mean()) if pos_mask.any() else 0.0
        neg_score_mean = float(scores[neg_mask].mean()) if neg_mask.any() else 0.0

        # CHANGE Q2: Q-Former standalone metrics from the rank aux head. This
        # is the direct evidence of how much ranking signal the bridge itself
        # extracts (compare against the MF baseline and the LLM uauc above).
        qf_auc, qf_uauc = 0.0, 0.0
        if data.get('aux') is not None:
            aux_scores = np.asarray(data['aux']).astype(np.float32)
            qf_auc = roc_auc_score(labels, aux_scores)
            qf_uauc, _, _ = calculate_user_auc(
                data['users'], aux_scores, labels,
                interaction_counts=getattr(self, "_uauc_counts", None),
                min_interactions=getattr(self, "_uauc_min_interactions", 0),
            )

        return {
            'auc': auc,
            'uauc': uauc,
            'qformer_auc': qf_auc,
            'qformer_uauc': qf_uauc,
            'pos_rate': float(labels.mean()) if labels.size else 0.0,
            'pred_pos_rate': float((scores > 0.5).mean()) if scores.size else 0.0,
            'pos_score_mean': pos_score_mean,
            'neg_score_mean': neg_score_mean,
            'score_gap': pos_score_mean - neg_score_mean,
            'score_mean': float(scores.mean()) if scores.size else 0.0,
            'score_std': float(scores.std()) if scores.size else 0.0,
        }
    
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        final_file = os.path.join(result_dir, f"{filename}.json")
        rank_file = os.path.join(result_dir, f"{filename}_rank{get_rank()}.json")

        with open(rank_file, "w") as f:
            json.dump(result, f)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            log_step("Merging results from all ranks...")
            combined_result = []
            for r in range(get_world_size()):
                path = os.path.join(result_dir, f"{filename}_rank{r}.json")
                with open(path, "r") as f:
                    combined_result.extend(json.load(f))

            if remove_duplicate:
                unique_res = {res[remove_duplicate]: res for res in combined_result}
                combined_result = list(unique_res.values())

            with open(final_file, "w") as f:
                json.dump(combined_result, f)
            log_step(f"Final result saved to {final_file}")

        return final_file
