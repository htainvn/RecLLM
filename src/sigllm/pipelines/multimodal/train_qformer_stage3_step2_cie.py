"""CoLLM Step 2 — train Q-Former + projection (CIE) on the full prompt.

The LoRA module is loaded from the Step 1 checkpoint and kept frozen; the
base LLM and the MF rec encoder remain frozen throughout. Only the Q-Former
and the linear projection into LLM hidden space are updated, using the full
prompt that includes the `<ItemIDList>` and `<TargetItemID>` soft-token
placeholders.

To reproduce the pre-LoRA frozen-LLM baseline, flip
`model.lora_config.use_lora=False` and clear `model.ckpt` in the YAML; the
Step 2 overrides will then run the legacy single-stage Stage 3.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.distributed.elastic.multiprocessing.errors import record

from sigllm import tasks
from sigllm.common.config import Config
from sigllm.common.dist_utils import get_rank, init_distributed_mode
from sigllm.common.utils import derive_job_id_from_llm
from sigllm.runners.runner_base_rec import RecRunnerBase  # noqa: F401  (registry side-effect)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 3 Step 2 — CIE on full prompt with frozen LoRA")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the config file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config",
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def apply_step2_overrides(cfg, slug):
    step2 = cfg.run_cfg.qformer_stage3_step2
    cfg.model_cfg.tuning_step = 2
    cfg.model_cfg.prompt_path = step2.prompt_path
    step1_out = cfg.run_cfg.qformer_stage3_step1.output_dir
    best_name = cfg.run_cfg.qformer_stage3_step1.best_ckpt_name
    step1_ckpt = os.path.join(step1_out, slug, best_name)
    if cfg.run_cfg.get("evaluate", False):
        # Eval-only: _save_checkpoint strips frozen params, so the step-1 ckpt
        # holds ONLY LoRA and the step-2 ckpt holds ONLY Q-Former/projection.
        # Load BOTH, step-1 first. A user-supplied model.ckpt names the step-2
        # weights (previously it was silently overwritten with the step-1 path,
        # so eval never saw the CIE-trained Q-Former); default to this run's
        # best checkpoint under <step2.output_dir>/<slug>/.
        user_ckpt = cfg.model_cfg.get("ckpt")
        if user_ckpt and not isinstance(user_ckpt, str):
            # Explicit list: the caller controls the full load order.
            cfg.model_cfg.ckpt = user_ckpt
        else:
            step2_ckpt = user_ckpt or os.path.join(
                step2.output_dir, slug, "checkpoint_best.pth"
            )
            cfg.model_cfg.ckpt = [step1_ckpt, step2_ckpt]
    else:
        cfg.model_cfg.ckpt = step1_ckpt
    cfg.run_cfg.output_dir = step2.output_dir
    cfg.run_cfg.init_lr = step2.init_lr
    # Honor a step-2 min_lr so the cosine schedule is not inverted by the global
    # run.min_lr (8e-5 > the 3e-5 step-2 init_lr, which made the LR RISE).
    cfg.run_cfg.min_lr = step2.get("min_lr", cfg.run_cfg.min_lr)
    cfg.run_cfg.max_epoch = step2.max_epoch


@record
def main():
    cfg = Config(parse_args())
    job_id = derive_job_id_from_llm(cfg)
    apply_step2_overrides(cfg, job_id)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    task = tasks.setup_task(cfg=cfg)
    datasets = task.build_datasets(cfg=cfg)

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    data_dir = cfg.datasets_cfg[first_dataset_key].path
    train_ = pd.read_pickle(os.path.join(data_dir, "train_ood2.pkl"))
    valid_ = pd.read_pickle(os.path.join(data_dir, "valid_ood2.pkl"))
    test_ = pd.read_pickle(os.path.join(data_dir, "test_ood2.pkl"))
    user_num = max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1
    item_num = max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1
    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    cfg.pretty_print()

    model = task.build_model(cfg=cfg)
    runner = task.build_runner(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.train()


if __name__ == "__main__":
    main()
