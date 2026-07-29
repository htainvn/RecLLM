"""Stage 3 Step 3 — JOINT tuning of MF + Q-Former + projection + LoRA.

Steps 1 and 2 each freeze exactly what the other trains: Step 1 adapts LoRA
against a frozen soft-token channel, Step 2 rewrites that channel against a
frozen LoRA. Neither lets the CF encoder co-adapt with the modules that read
it, because MF is frozen in both. This step unfreezes all four at once; only
the base LLM stays frozen.

Warm-start comes from the Step-1 LoRA checkpoint, not from cold. With
everything trainable and the soft-token channel still noisy, LoRA otherwise
takes the cheaper route and learns to answer from the prompt text alone — the
same failure mode that made Step 2 flat back when Step 1 had been trained on
the text-only prompt. Point ``run.qformer_stage3_step3.ckpt_from`` at ``step2``
to warm-start from the Step-2 checkpoint instead (Q-Former/projection already
adapted, LoRA as Step 1 left it).

The load-bearing risk here is MF drift. The Stage-1/2 alignment was learned
against a FIXED MF geometry, so if MF moves quickly that alignment is
invalidated faster than the Q-Former can track it. Two controls exist:
``run.rec_lr_scale`` puts MF in its own optimizer param group at a fraction of
the global LR, and the ``mf_drift`` diagnostic logs how far it has actually
moved. Read drift together with val uAUC:

- drift rising, uAUC flat or falling -> MF LR too high, lower rec_lr_scale
- drift ~0 -> MF is effectively frozen, Step 3 buys nothing over Step 2
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
from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.utils import derive_job_id_from_llm
from sigllm.runners.runner_base_rec import RecRunnerBase  # noqa: F401  (registry side-effect)

LOGGER = NotebookLogger.rich_logger("sigllm.train_qformer_stage3_step3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage 3 Step 3 — joint MF + Q-Former + projection + LoRA"
    )
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


def resolve_warm_start(cfg, slug):
    """Path of the checkpoint Step 3 warm-starts from.

    ``ckpt_from: step1`` (default) takes the Step-1 LoRA checkpoint; ``step2``
    takes the Step-2 one. Both live under ``<stage output_dir>/<slug>/``, where
    slug is the LLM-derived job id, matching how Step 2 locates Step 1's.

    ``checkpoint_best.pth`` is the fallback name because that is what
    ``RunnerBase._save_checkpoint`` writes; only the Step-1 block happens to
    declare ``best_ckpt_name`` explicitly.
    """
    step3 = cfg.run_cfg.qformer_stage3_step3
    source = str(step3.get("ckpt_from", "step1")).lower()
    if source == "step1":
        stage = cfg.run_cfg.qformer_stage3_step1
    elif source == "step2":
        stage = cfg.run_cfg.qformer_stage3_step2
    else:
        raise ValueError(
            f"run.qformer_stage3_step3.ckpt_from must be 'step1' or 'step2', got {source!r}"
        )
    best_name = stage.get("best_ckpt_name", "checkpoint_best.pth")
    return os.path.join(stage.output_dir, slug, best_name), source


def apply_step3_overrides(cfg, slug):
    step3 = cfg.run_cfg.qformer_stage3_step3

    # tuning_step MUST be set before the model is built: _init_rec_model reads
    # it to decide whether to skip freezing MF (freezing installs a
    # `train = disabled_train` patch that is awkward to undo afterwards).
    cfg.model_cfg.tuning_step = 3
    cfg.model_cfg.prompt_path = step3.prompt_path
    # Redundant with the tuning_step=3 policy (which overrides it anyway) but
    # set explicitly so `pretty_print` shows the real intent and no reader has
    # to know about the override.
    cfg.model_cfg.freeze_rec = False

    ckpt_path, source = resolve_warm_start(cfg, slug)
    cfg.model_cfg.ckpt = ckpt_path

    cfg.run_cfg.output_dir = step3.output_dir
    cfg.run_cfg.init_lr = step3.init_lr
    # Both must be overridden together — the top-level min_lr (8e-5) is above
    # this stage's init_lr, which would invert the cosine schedule (LR climbing
    # instead of decaying). Same trap already fixed in Steps 1 and 2.
    cfg.run_cfg.min_lr = step3.min_lr
    cfg.run_cfg.max_epoch = step3.max_epoch
    # MF-specific optimizer settings, consumed by build_optimizer.
    cfg.run_cfg.rec_lr_scale = step3.rec_lr_scale
    cfg.run_cfg.rec_weight_decay = step3.rec_weight_decay

    if not os.path.exists(ckpt_path):
        LOGGER.warning(
            "Warm-start checkpoint (%s) not found at %s — Step 3 will start from "
            "the Stage-2 Q-Former/projection with a COLD LoRA. Expect LoRA to "
            "learn to answer from prompt text alone; run Step 1 first.",
            source,
            ckpt_path,
        )
    else:
        LOGGER.info("Step 3 warm-start from %s checkpoint: %s", source, ckpt_path)

    LOGGER.info(
        "Step 3 joint schedule | init_lr=%s min_lr=%s max_epoch=%s | "
        "MF: lr_scale=%s (effective init %s) weight_decay=%s",
        step3.init_lr,
        step3.min_lr,
        step3.max_epoch,
        step3.rec_lr_scale,
        float(step3.init_lr) * float(step3.rec_lr_scale),
        step3.rec_weight_decay,
    )


@record
def main():
    cfg = Config(parse_args())
    job_id = derive_job_id_from_llm(cfg)
    apply_step3_overrides(cfg, job_id)
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

    drift = model.mf_drift()
    if drift:
        LOGGER.info(
            "Final MF drift ||W-W0||/||W0|| | %s",
            ", ".join(f"{k}={v:.4f}" for k, v in sorted(drift.items())),
        )


if __name__ == "__main__":
    main()
