"""SeLLa-gated step 3 — the ONLY training stage of the gated-Q-Former variant.

What this replaces
------------------
The ``qformer_stage1`` (InfoNCE representation) and ``qformer_stage2``
(generative) pretraining stages are NOT part of this pipeline. The Q-Former here
is ~4.3M parameters behind a zero-init scalar gate, which is what those stages
were protecting against: the gate guarantees the run cannot start worse than
SeLLa (at ``gate = 0`` the model IS SeLLa), so a randomly-initialised module is
safe to train jointly. Dropping them also drops two documented failure modes —
Stage 1's InfoNCE gain sitting near chance and Stage 2's soft-token channel being
drowned out by the ``<ItemTitleList>`` text path.

Prerequisites, in order:

    1. ``distill_item_llm_embeddings`` (last_hidden bank -> item_llm_emb.pt)
    2. ``train_rec_baseline``          (MF WITH the SeLLa Step-2 alignment)
    3. ``train_qformer_stage3_step1_lora``  (LoRA — this stage warm-starts from it
                                             and, by default, keeps it FROZEN,
                                             which is what SeLLa does)
    4. this script

Step 3 with a COLD LoRA is a valid but different experiment; the override log
says which one is running.

What to read in the log
-----------------------
``qformer gate (step N)`` — ``gate`` and ``delta_rel = ||g*delta||/||e_user||``.
This is the reported contribution of the module. Both pinned at 0 after a few
hundred steps means the Q-Former is not earning its place and the honest result
is "no contribution"; ``delta_rel`` growing past ~1 means the gated term is
dominating the user token, which is worth a lower LR rather than a win.

``MF drift (step N)`` — only when ``freeze_rec: false``. Rising drift with flat
val uAUC means ``run.sella_gated_step3.rec_lr_scale`` is too high.
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

# Registry side-effects: importing these registers the model and the runner
# under the names the config refers to.
from sigllm.models.multimodal.sella_gated_rec_llm import SeLLaGatedRecLLM  # noqa: F401
from sigllm.runners.runner_base_rec import RecRunnerBase  # noqa: F401

LOGGER = NotebookLogger.rich_logger("sigllm.train_sella_gated_step3")

ARCH = "sella_gated_rec_llm"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SeLLa-gated step 3 (MF + id_proj + warm_proj + gated Q-Former)"
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


def resolve_warm_start(cfg, slug):
    """Path of the LoRA checkpoint this stage warm-starts from.

    ``sella_step1`` (default) is SeLLa's arrangement: a TALLRec text-only LoRA
    SFT adapted the LLM to the task, and this stage freezes it and trains the
    collaborative side. ``step1``/``step2`` point at nhánh A's checkpoints
    instead (a LoRA that WAS adapted with the soft-token channel present), which
    is a useful comparison but not SeLLa. ``self`` resumes this stage; ``none``
    means a cold LoRA.
    """
    stage_cfg = cfg.run_cfg.sella_gated_step3
    source = str(stage_cfg.get("ckpt_from", "sella_step1")).lower()
    return checkpoint_for(cfg, source, slug), source


CKPT_SOURCES = {
    "sella_step1": "sella_gated_step1",
    "step1": "qformer_stage3_step1",
    "step2": "qformer_stage3_step2",
}


def checkpoint_for(cfg, source, slug):
    """``<output_dir>/<slug>/<best_ckpt_name>`` for one named source, or None.

    Kept separate from ``resolve_warm_start`` so the LoRA source can be resolved
    independently of ``ckpt_from`` — the two differ exactly when ``ckpt_from`` is
    ``self``.
    """
    source = str(source).lower()
    if source == "none":
        return None
    if source == "self":
        stage = cfg.run_cfg.sella_gated_step3
    elif source in CKPT_SOURCES:
        stage = cfg.run_cfg.get(CKPT_SOURCES[source])
        if stage is None:
            raise ValueError(
                f"source={source!r} needs run.{CKPT_SOURCES[source]} in the config"
            )
    else:
        raise ValueError(
            "checkpoint source must be one of 'sella_step1', 'step1', 'step2', "
            f"'self', 'none' — got {source!r}"
        )
    best_name = stage.get("best_ckpt_name", "checkpoint_best.pth")
    return os.path.join(stage.output_dir, slug, best_name)


def apply_overrides(cfg, slug):
    stage = cfg.run_cfg.sella_gated_step3

    # --- model ------------------------------------------------------------
    cfg.model_cfg.arch = ARCH
    cfg.model_cfg.prompt_path = stage.prompt_path
    # This model has no CoLLM step policy — its freeze policy is fixed (base LLM
    # frozen, LoRA per sella_gated.lora_trainable, everything collaborative
    # trainable). Null it so a stale value cannot be mistaken for one being
    # applied.
    cfg.model_cfg.tuning_step = None
    # Must be set BEFORE the model is built: _init_rec_model reads it to decide
    # whether to install the freeze (which patches `train = disabled_train` on
    # the instance and is awkward to undo afterwards).
    cfg.model_cfg.freeze_rec = bool(stage.get("freeze_rec", False))

    ckpt_path, source = resolve_warm_start(cfg, slug)
    cfg.model_cfg.ckpt = ckpt_path

    # LoRA source. This stage FREEZES LoRA, and the runner strips frozen tensors
    # when saving — so a checkpoint written by this stage carries no lora_* keys.
    # When `ckpt` is such a checkpoint (ckpt_from: self, i.e. resume / eval-only /
    # ablate-at-inference), LoRA has to come from somewhere else or the adapter
    # stays at its zero-init identity and the run silently scores the unadapted
    # base model. `lora_from` names that somewhere; the model raises if nothing
    # supplies LoRA and `allow_cold_lora` is not set.
    lora_ckpt = None
    if source == "self":
        lora_source = str(stage.get("lora_from", "sella_step1")).lower()
        lora_ckpt = checkpoint_for(cfg, lora_source, slug)
        LOGGER.info(
            "ckpt_from=self -> LoRA cannot come from `ckpt` (this stage freezes "
            "LoRA, so its own checkpoints carry no lora_* keys). Taking LoRA from "
            "%s: %s",
            lora_source, lora_ckpt,
        )
        if lora_ckpt and not os.path.exists(lora_ckpt):
            LOGGER.warning(
                "LoRA source %s does not exist — the model will REFUSE to build "
                "rather than silently score an unadapted base LLM.", lora_ckpt,
            )
    cfg.model_cfg.lora_ckpt = lora_ckpt
    # ckpt_from=none is the ONLY explicit opt-in to an unadapted base LLM.
    cfg.model_cfg.allow_cold_lora = source == "none"

    # --- run --------------------------------------------------------------
    cfg.run_cfg.output_dir = stage.output_dir
    cfg.run_cfg.init_lr = stage.init_lr
    # Both must be overridden together: the top-level min_lr (8e-5) sits ABOVE
    # these stage LRs, which would invert the cosine schedule into a climb. Same
    # trap already documented for steps 1/2/3.
    cfg.run_cfg.min_lr = stage.min_lr
    cfg.run_cfg.max_epoch = stage.max_epoch
    cfg.run_cfg.rec_lr_scale = stage.rec_lr_scale
    cfg.run_cfg.rec_weight_decay = stage.rec_weight_decay
    if "batch_size_train" in stage:
        cfg.run_cfg.batch_size_train = stage.batch_size_train
    if "batch_size_eval" in stage:
        cfg.run_cfg.batch_size_eval = stage.batch_size_eval
    if "iters_per_epoch" in stage:
        cfg.run_cfg.iters_per_epoch = stage.iters_per_epoch
    # SeLLa reaches its effective batch of 300 through gradient accumulation
    # (5 x 30 x 2 GPU), and its whole run is ~109 optimizer updates. Without
    # accum_grad_iters the runner takes one update per batch, which is how the
    # first long run ended up at 7.4x SeLLa's total update count by its FIRST
    # eval — and then collapsed.
    for key in ("accum_grad_iters", "weight_decay", "warmup_steps"):
        if key in stage:
            cfg.run_cfg[key] = stage[key]

    # --- things this model deliberately does NOT use ----------------------
    # The objective is SeLLa's LM cross-entropy. The two per-user BPR auxiliaries
    # are never read by SeLLaGatedRecLLM, and `user_grouped_batch` exists only to
    # give those BPR terms same-user pos/neg pairs. Leaving grouped batching on
    # would change the batch composition relative to SeLLa for no benefit, so
    # turn it off and say so rather than letting it sit there looking active.
    grouped = cfg.run_cfg.get("user_grouped_batch", None)
    was_grouped = bool(grouped.get("enabled", False)) if grouped is not None else False
    if was_grouped:
        cfg.run_cfg.user_grouped_batch.enabled = False
    ranking_w = float((cfg.model_cfg.get("ranking_loss") or {}).get("weight", 0.0))
    align_w = float((cfg.model_cfg.get("align_rank_loss") or {}).get("weight", 0.0))

    sella_cfg = cfg.model_cfg.get("sella_gated") or {}
    LOGGER.info(
        "SeLLa-gated overrides | arch=%s prompt=%s | warm start: %s (%s) | "
        "freeze_rec=%s lora_trainable=%s | lm_loss_scope=%s | "
        "init_lr=%s min_lr=%s max_epoch=%s bs_train=%s iters/epoch=%s | "
        "MF: lr_scale=%s (effective %s) wd=%s",
        ARCH,
        stage.prompt_path,
        ckpt_path if ckpt_path else "COLD (ckpt_from=none)",
        source,
        cfg.model_cfg.freeze_rec,
        sella_cfg.get("lora_trainable", False),
        sella_cfg.get("lm_loss_scope", "full"),
        stage.init_lr,
        stage.min_lr,
        stage.max_epoch,
        cfg.run_cfg.batch_size_train,
        cfg.run_cfg.iters_per_epoch,
        stage.rec_lr_scale,
        float(stage.init_lr) * float(stage.rec_lr_scale),
        stage.rec_weight_decay,
    )
    # Budget parity against SeLLa, stated in the one unit that actually predicts
    # the collapse: optimizer updates. SeLLa step 3 = 1 epoch at effective batch
    # 300 = ~109 updates on ml-1m. Print ours next to it so an over-trained
    # configuration is visible before the run rather than after.
    accum = max(1, int(cfg.run_cfg.get("accum_grad_iters", 1)))
    eff_batch = int(cfg.run_cfg.batch_size_train) * accum
    updates = int(cfg.run_cfg.max_epoch) * int(cfg.run_cfg.iters_per_epoch) // accum
    samples = int(cfg.run_cfg.max_epoch) * int(cfg.run_cfg.iters_per_epoch) * int(cfg.run_cfg.batch_size_train)
    LOGGER.info(
        "Budget | effective batch = %d x %d = %d | %d optimizer updates over %d "
        "sample-visits | %d eval points. SeLLa step 3 = ~109 updates at effective "
        "batch 300 (1 epoch). Ratio vs SeLLa: %.1fx.",
        int(cfg.run_cfg.batch_size_train), accum, eff_batch, updates, samples,
        int(cfg.run_cfg.max_epoch), updates / 109.0,
    )
    if updates > 400:
        LOGGER.warning(
            "%d updates is %.0fx SeLLa's entire step 3. Under lm_loss_scope=full "
            "that regime erased the soft tokens (AUC 0.752 -> 0.636 between 800 "
            "and 1600 updates). If you intend to train this long, switch to "
            "model.sella_gated.lm_loss_scope=answer.",
            updates, updates / 109.0,
        )
    if was_grouped:
        LOGGER.info(
            "Disabled run.user_grouped_batch (was enabled): it exists for the "
            "per-user BPR auxiliaries, which this model does not use. Plain "
            "shuffled batching is SeLLa's."
        )
    if ranking_w > 0 or align_w > 0:
        LOGGER.warning(
            "model.ranking_loss.weight=%s and model.align_rank_loss.weight=%s are "
            "set in the config but are IGNORED by %s — the objective is SeLLa's "
            "LM cross-entropy only. Nothing to fix; this note exists so the "
            "numbers are not attributed to those terms.",
            ranking_w, align_w, ARCH,
        )

    if ckpt_path and not os.path.exists(ckpt_path):
        LOGGER.warning(
            "Warm-start checkpoint (%s) not found at %s — LoRA will be COLD. With "
            "lora_trainable=false that means the LLM is the unadapted base model "
            "and the collaborative modules have to carry everything. Run "
            "`python -m sigllm.pipelines.multimodal.train_sella_gated_step1_lora "
            "--cfg-path <cfg>` first, or set "
            "run.sella_gated_step3.ckpt_from=none to make the cold start explicit.",
            source, ckpt_path,
        )
    elif ckpt_path:
        LOGGER.info("Warm-starting LoRA from the %s checkpoint: %s", source, ckpt_path)


@record
def main():
    cfg = Config(parse_args())
    job_id = derive_job_id_from_llm(cfg)
    apply_overrides(cfg, job_id)
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

    # The headline number of this whole exercise: what the gated Q-Former ended
    # up contributing.
    stats = model.gate_stats()
    if stats:
        LOGGER.info(
            "FINAL gated Q-Former contribution | %s",
            ", ".join(f"{k}={v:.6f}" for k, v in stats.items()),
        )
    else:
        LOGGER.info("No gate stats — the Q-Former was disabled for this run.")

    drift = model.mf_drift()
    if drift:
        LOGGER.info(
            "Final MF drift ||W-W0||/||W0|| | %s",
            ", ".join(f"{k}={v:.4f}" for k, v in sorted(drift.items())),
        )


if __name__ == "__main__":
    main()
