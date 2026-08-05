"""SeLLa-gated step 1 — LoRA on TEXT-ONLY prompts (SeLLa's TALLRec stage).

Why this exists as its own script
---------------------------------
``train_qformer_stage3_step1_lora`` cannot serve the gated branch, for a reason
that is structural rather than cosmetic: it builds ``QRecLLM``, whose
``__init__`` loads ``model.qformer_config.qformer_ckpt`` and
``llm_proj_ckpt`` UNCONDITIONALLY, and both of those are outputs of
``qformer_stage2``. The gated branch does not run Stage 1 or Stage 2, so those
files do not exist and the build dies with::

    FileNotFoundError: .../ckpt/qformer_stage2_qwen2/qformer_stage2_best_qformer.pth

This script sets both to ``not_have`` (the sentinel ``_init_qformer`` /
``_init_projection`` already understand) and points the prompt at the text-only
variant, which is what SeLLa's step 1 actually is: a TALLRec-style LoRA SFT with
NO collaborative signal at all. ``qformer_prompt_movie_text_only.txt`` carries
only ``<ItemTitleList>`` and ``<TargetItemTitle>``, so
``get_placeholder_order`` returns ``[]``, the soft-token injection branch is
never entered, and the randomly-initialised (and frozen) Q-Former is never read.
That last part matters: with a soft-token prompt and no Stage-2 checkpoint, LoRA
would be trained against a frozen RANDOM channel — the worst of both worlds.

The collaborative side is introduced for the first time in
``train_sella_gated_step3``, with this LoRA loaded and FROZEN. That is SeLLa's
arrangement.

Note the known risk, documented in the config's own step-1 history: a text-only
step 1 teaches LoRA to answer from the titles, and CoLLM's step 2 then found the
adapted model had learned to ignore the soft-token channel. SeLLa nonetheless
does exactly this, and the gated branch differs in a way that matters here — the
collaborative signal arrives as SeLLa's DIRECT ``<UserID>``/``<ItemID>`` tokens
(unreduced MF vectors, ``id_proj`` warm-started from the MF's own
``trans_1``/``trans_2``) rather than as an 8-token compressed channel, and
``id_proj``/``warm_proj``/MF all keep training in step 3.
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

LOGGER = NotebookLogger.rich_logger("sigllm.train_sella_gated_step1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SeLLa-gated step 1 — LoRA SFT on text-only prompts"
    )
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the config file.")
    parser.add_argument(
        "--options", nargs="+", help="override some settings in the used config"
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def apply_overrides(cfg):
    stage = cfg.run_cfg.sella_gated_step1

    cfg.model_cfg.tuning_step = 1          # QRecLLM policy: LoRA only
    cfg.model_cfg.prompt_path = stage.prompt_path
    cfg.model_cfg.ckpt = None

    qformer_cfg = cfg.model_cfg.get("qformer_config")
    if qformer_cfg is not None:
        # THE fix for the FileNotFoundError: these two point at qformer_stage2
        # outputs, and this branch never runs stage 2.
        cfg.model_cfg.qformer_config.qformer_ckpt = "not_have"
        cfg.model_cfg.qformer_config.llm_proj_ckpt = "not_have"
        cfg.model_cfg.qformer_config.warm_token = False
        # sem_source would also load a bank the text-only path never reads.
        cfg.model_cfg.qformer_config.sem_source = False

    # SeLLa's step 1 is a plain LM SFT. The per-user BPR auxiliaries and the
    # grouped sampler that feeds them are not part of it, and leaving them on
    # would make this stage's LoRA a different object from SeLLa's.
    if cfg.model_cfg.get("ranking_loss") is not None:
        cfg.model_cfg.ranking_loss.weight = 0.0
    if cfg.model_cfg.get("align_rank_loss") is not None:
        cfg.model_cfg.align_rank_loss.weight = 0.0
    if cfg.run_cfg.get("user_grouped_batch") is not None:
        cfg.run_cfg.user_grouped_batch.enabled = False

    cfg.run_cfg.output_dir = stage.output_dir
    cfg.run_cfg.init_lr = stage.init_lr
    # Must move together with init_lr: the top-level min_lr (8e-5) EQUALS the
    # usual step-1 init_lr, which flattens linear_warmup_cosine_lr into a
    # constant — no decay at all across the run.
    cfg.run_cfg.min_lr = stage.min_lr
    cfg.run_cfg.max_epoch = stage.max_epoch
    if "batch_size_train" in stage:
        cfg.run_cfg.batch_size_train = stage.batch_size_train
    if "iters_per_epoch" in stage:
        cfg.run_cfg.iters_per_epoch = stage.iters_per_epoch

    LOGGER.info(
        "SeLLa-gated step 1 overrides | prompt=%s | qformer_ckpt/llm_proj_ckpt="
        "not_have (stage 2 is NOT part of this branch) | warm_token=False "
        "sem_source=False ranking/align losses OFF grouped_batch OFF | "
        "output_dir=%s init_lr=%s min_lr=%s max_epoch=%s",
        stage.prompt_path, stage.output_dir, stage.init_lr, stage.min_lr, stage.max_epoch,
    )

    with open(stage.prompt_path) as f:
        active = [l for l in f.read().splitlines()
                  if l.strip() and not l.lstrip().startswith("#")]
    soft = [ph for ph in ("<UserProfile>", "<TargetItemID>", "<UserID>", "<ItemID>",
                          "<Warm_ID>", "<ItemIDList>")
            if any(ph in l for l in active)]
    if soft:
        raise ValueError(
            f"run.sella_gated_step1.prompt_path must be a TEXT-ONLY prompt, but "
            f"{stage.prompt_path} carries soft-token placeholders {soft}. With no "
            "stage-2 checkpoint the Q-Former is random AND frozen at step 1, so "
            "those slots would train LoRA against pure noise. Use "
            "prompts/qformer_prompt_movie_text_only.txt."
        )
    LOGGER.info(
        "Prompt check OK | %d text-only prompt line(s), no soft-token placeholders "
        "-> the collaborative channel is introduced for the first time in step 3.",
        len(active),
    )


@record
def main():
    cfg = Config(parse_args())
    apply_overrides(cfg)
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
    runner = task.build_runner(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

    LOGGER.info(
        "Step 1 done. Checkpoint for step 3: %s",
        os.path.join(cfg.run_cfg.output_dir, job_id,
                     cfg.run_cfg.sella_gated_step1.get("best_ckpt_name",
                                                       "checkpoint_best.pth")),
    )


if __name__ == "__main__":
    main()
