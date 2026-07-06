"""CoLLM Step 1 — train LoRA only on text-only prompts.

The Q-Former, projection, MF and base LLM are all frozen; only the LoRA
adapter on the LLM is updated. The text-only prompt is identical in shape to
the full Stage 3 prompt but with the soft-token placeholders (`<ItemIDList>`,
`<TargetItemID>`) removed, so the LLM learns the recommendation task without
any collaborative-filtering signal. The best checkpoint feeds Step 2.
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
    parser = argparse.ArgumentParser(description="Train Stage 3 Step 1 — LoRA on text-only prompts")
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


def apply_step1_overrides(cfg):
    step1 = cfg.run_cfg.qformer_stage3_step1
    cfg.model_cfg.tuning_step = 1
    cfg.model_cfg.prompt_path = step1.prompt_path
    # Step 1 is pure TALLRec (text-only LoRA). The Q3 title-free mixing flag is
    # a STEP-2 mechanism — left on here it routes ~30% of batches through the
    # CF-token prompt with a task-naive frozen Q-Former, wasting LoRA training
    # on a distribution eval never sees.
    cfg.model_cfg.title_free_ratio = 0.0
    cfg.model_cfg.ckpt = None
    cfg.run_cfg.output_dir = step1.output_dir
    cfg.run_cfg.init_lr = step1.init_lr
    cfg.run_cfg.max_epoch = step1.max_epoch


@record
def main():
    cfg = Config(parse_args())
    apply_step1_overrides(cfg)
    job_id = derive_job_id_from_llm(cfg)
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
