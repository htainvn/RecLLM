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
    cfg.model_cfg.ckpt = os.path.join(step1_out, slug, best_name)
    cfg.run_cfg.output_dir = step2.output_dir
    cfg.run_cfg.init_lr = step2.init_lr
    # min_lr must be overridden together with init_lr: the top-level min_lr
    # (8e-5) exceeds step2 init_lr (3e-5), which inverts cosine_lr_schedule
    # (LR climbs 3e-5 -> 8e-5 instead of decaying). Pull the per-stage min_lr.
    cfg.run_cfg.min_lr = step2.min_lr
    cfg.run_cfg.max_epoch = step2.max_epoch
    # SeLLa-style joint CF training at step 2 (opt-in via the step-2 block):
    # freeze_rec=False puts the MF tables in their own optimizer param group,
    # scaled by rec_lr_scale (see build_optimizer / common.optims).
    if "freeze_rec" in step2:
        cfg.model_cfg.freeze_rec = bool(step2.freeze_rec)
    if "rec_lr_scale" in step2:
        cfg.run_cfg.rec_lr_scale = step2.rec_lr_scale
    if "rec_weight_decay" in step2:
        cfg.run_cfg.rec_weight_decay = step2.rec_weight_decay


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
