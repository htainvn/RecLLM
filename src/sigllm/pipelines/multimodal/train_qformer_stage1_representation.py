import argparse
import contextlib
import math
import random
import torch
from torch.optim import Adam
import omegaconf
import os
import numpy as np
from typing import Optional

from sigllm.common import NotebookLogger, EarlyStopping
from sigllm.common.config import Config
from sigllm.datasets.qformer.qformer_loader import build_qformer_loader, build_qformer_loaders
from sigllm.models.rec.matrix_factorization import MatrixFactorization
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.projection.qformer_alignment_model import QRecInstructAlignmentModel
from sigllm.datasets.movie.movie_ood_dataset import MovieOODDataset
from sigllm.pipelines.multimodal.probe_cf_channel import (
    build_probe_loader,
    collate as probe_collate,
    pooling_usage,
    probe_metrics,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = NotebookLogger.rich_logger("sigllm.train_qformer_stage1")


def log_step(title: str, detail: Optional[str] = None) -> None:
    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def _init_rec_model(cfg, device):
    """
    Initializes the recommendation model, loads pretrained weights if available,
    and freezes parameters if configured.
    """
    mf_config = omegaconf.OmegaConf.create({
        # TEMP_DISABLED_USER_CF: user_num is still needed to build/load MF weights,
        # but stage 1 no longer calls mf.user_encoder().
        "user_num": int(cfg.user_num),
        "item_num": int(cfg.item_num),
        "embedding_size": int(cfg.embedding_size)
    })
    mf = MatrixFactorization(mf_config).to(device)

    pretrained_rec_path = cfg.pretrained_rec_path
    if mf is not None and os.path.exists(pretrained_rec_path):
        mf.load_state_dict(torch.load(pretrained_rec_path, map_location="cpu"))
        print(f"Successfully loaded the pretrained rec model from {pretrained_rec_path}")

    if cfg.freeze_rec and mf is not None:
        for param in mf.parameters():
            param.requires_grad = False
        mf.eval()
        mf.train = disabled_train.__get__(mf, MatrixFactorization)
        print("Freeze rec encoder completed")

    return mf


def _init_qformer(cfg, d_model, device, d_sem=None):
    """
    Initializes the Q-Former model.
    """
    qformer_output_dim = cfg.qformer_output_dim
    if qformer_output_dim is None:
        qformer_output_dim = d_model

    return HFQFormerAdapter(
        d_cf=cfg.embedding_size,
        d_model=d_model,
        num_queries=cfg.num_queries,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        output_dim=int(qformer_output_dim),
        qformer_text_model_name=cfg.qformer_text_model_name,
        max_instruction_length=cfg.get("max_instruction_length", 48),
        # Build (and pretrain) the conditioning path here so user_proj does not
        # sit at its zero init through all of Stage 1 and reach Stage 3 blind.
        user_conditioned=bool(cfg.get("user_conditioned", False)),
        # Must mirror model.qformer_config.* — these change the adapter SHAPE,
        # and the checkpoint flows stage1 -> stage2 -> stage3 under a strict load.
        # Q-Former internal dropout (hidden + attention). Was hardcoded 0.0 by
        # omission: no call site passed it, so a 3-layer transformer trained on
        # ~17k user_item pairs had no regularisation at all, which is the
        # straightforward reading of train g_ui >0.67 against val ~0.14.
        # Does NOT change the state_dict, so checkpoints stay compatible.
        dropout=float(cfg.get("qformer_dropout", 0.0)),
        candidate_fusion=bool(cfg.get("candidate_fusion", False)),
        item_residual=bool(cfg.get("item_residual", False)),
        output_residual=bool(cfg.get("output_residual", False)),
        d_user=int(cfg.embedding_size),
        d_sem=d_sem,
    ).to(device)


def _init_optimizer(model, lr, weight_decay=0.0, decoupled_weight_decay=True):
    """
    Initializes the Adam optimizer for trainable parameters.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer_cls = torch.optim.AdamW if decoupled_weight_decay else Adam
    return optimizer_cls(param_groups, lr=lr)

def _offdiag_cos(x: torch.Tensor) -> float:
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
    sim = x @ x.T
    n = sim.size(0)
    return float((sim.sum() - sim.diagonal().sum()) / (n*(n-1)))

@torch.no_grad()
def _query_anisotropy(model, loader, max_rows: int = 512):
    batch = next(iter(loader), None)
    if batch is None:
        return None
    device = next(model.parameters()).device
    item_ids = batch["i_left"][:max_rows].to(device)
    if item_ids.numel() < 4:
        return None

    was_training = model.training
    model.eval()
    pooled = model.encode_item_queries(item_ids).mean(dim=1)
    if was_training:
        model.train()
    return {
        "n": int(item_ids.numel()),
        "raw": _offdiag_cos(pooled),
        "centered": _offdiag_cos(pooled - pooled.mean(dim=0, keepdim=True))
    }

@torch.no_grad()
def _zero_init_path_norms(model):
    """Weight norms of the two zero-initialised conditioning paths.

    ``proj_sem`` and ``user_proj`` both start at exactly 0 so they act as no-ops
    at step 0 and only grow if the loss rewards them. The failure mode that
    creates is silent: if they never receive usable gradient they simply stay at
    0 forever, and every downstream metric looks "fine" because the paths are
    inert rather than wrong. Printing the norms is the only direct way to tell
    "the pretext is training this" from "the pretext is a no-op".

    A norm still at 0.0 after a few epochs means the path is dead — check the
    semantic bank coverage (for proj_sem) or the BPR margin / bpr_logit_center
    (for user_proj).
    """
    out = {}
    qformer = getattr(model, "qformer", None)
    if qformer is None:
        return out
    # item_res_proj / fuse_cf / fuse_user are the newer zero-init paths and are
    # exactly as prone to staying inert as the older two — item_res_proj is the
    # residual injection meant to bypass the body's ~45x attenuation of the
    # item-specific component, and fuse_* carry the candidate interaction, so
    # "did it leave 0" is the whole question for all three. fuse_cf/fuse_user
    # are built with bias=False, hence the None guard.
    for name in ("proj_sem", "user_proj", "item_res_proj", "fuse_cf", "fuse_user"):
        module = getattr(qformer, name, None)
        if module is None:
            continue
        out[name] = float(module.weight.detach().norm().item())
        bias = getattr(module, "bias", None)
        if bias is not None:
            out[f"{name}_bias"] = float(bias.detach().norm().item())
    # res_gain is the ONE non-zero-init path (see output_residual): it starts at
    # sqrt(d_model) so the mask-mean readout is active from step 0. Its TREND is
    # the interesting part — falling means the body is learning to carry the
    # ranking itself, rising means it is leaning harder on the readout.
    res_gain = getattr(qformer, "res_gain", None)
    if res_gain is not None:
        out["res_gain"] = float(res_gain.detach().item())
    return out


def _log_batch_preview(batch, prefix: str = "train_step"):
    """Print a compact preview of the current batch for debugging."""
    batch_size = batch["i_left"].size(0)
    type_counts = {sample_type: batch["sample_type"].count(sample_type) for sample_type in set(batch["sample_type"])}

    print(
        f"[{prefix}] batch_size={batch_size} type_counts={type_counts} "
        f"u.shape={tuple(batch['u'].shape)} i_left.shape={tuple(batch['i_left'].shape)} "
        f"i_right.shape={tuple(batch['i_right'].shape)}"
    )
    if batch_size > 0:
        print(
            f"[{prefix}] sample[0] type={batch['sample_type'][0]} u={batch['u'][0].item()} "
            f"i_left={batch['i_left'][0].item()} i_right={batch['i_right'][0].item()}"
        )
        print(f"[{prefix}] sample[0] instruction={batch['instruction'][0]}")
        print(f"[{prefix}] sample[0] text={batch['text'][0]}")


def _move_batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _subset_batch(batch, indices):
    index_list = indices.detach().cpu().tolist()
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[indices]
        else:
            out[key] = [value[i] for i in index_list]
    return out


def _indices_for_type(batch, sample_type: str, device):
    indices = [idx for idx, current in enumerate(batch["sample_type"]) if current == sample_type]
    return torch.tensor(indices, dtype=torch.long, device=device)


# Which sample type each metric depends on. A batch holding none of that type
# produces no loss for the metric, so it must be excluded from the average
# rather than contributing a zero.
METRIC_GROUPS = {
    "L_itc": "item_text", "itc_top1": "item_text",
    "L_itm": "item_text", "itm_acc": "item_text",
    "L_itg": "item_text", "itg_acc": "item_text", "itg_title_acc": "item_text",
    "L_llm": "item_text", "llm_top1": "item_text",
    "L_ii": "item_item", "ii_top1": "item_item",
    "L_ui": "user_item", "ui_top1": "user_item",
    "L_uic": "user_item", "uic_acc": "user_item",
}

SAMPLE_TYPES = ("item_text", "item_item", "user_item")

# In-batch retrieval terms and the sample type whose row count sets their chance
# level. Used to derive ``gain_*`` (nats earned over a uniform predictor).
RETRIEVAL_TERMS = (
    ("itc", "item_text"),
    ("llm", "item_text"),
    ("ii", "item_item"),
    ("ui", "user_item"),
)

# Terms summed into the selection metric L_repr. Includes the collaborative
# contrastives (L_ii, L_ui): Stage 3 consumes the Q-Former body for CF
# encoding too, and a selection metric blind to the collaborative terms can
# pick a checkpoint whose collaborative representation has already degraded.
# They previously were excluded because they converge in ~5 epochs and then
# drift up on validation, vetoing epochs still improving on text grounding —
# damp that via w_ii / w_ui rather than by dropping the terms entirely.
REPR_TERMS = ("L_itc", "L_itm", "L_itg", "L_llm", "L_ii", "L_ui")

# Metrics the fixed-n diagnostic publishes into the metrics dict as
# ``{split}_{sem_on|sem_off}_{key}``, so they can drive checkpoint selection.
# Kept small on purpose: these are measured at a FIXED n across splits, which is
# what makes them comparable, and every extra key is another name a config can
# typo into.
DIAG_EXPORTED_METRICS = (
    "gain_itc", "gain_llm", "L_itc", "L_llm", "itc_top1", "llm_top1", "itg_title_acc",
)

# Components of the composite ``val_gain_repr`` selection metric.
GAIN_REPR_ALIGN_KEY = "val_sem_off_gain_llm"
GAIN_REPR_COLLAB_KEYS = ("val_gain_ii", "val_gain_ui")


def add_gain_repr(metrics, collab_weight):
    """Add ``val_gain_repr`` = CF-only alignment gain + weighted collaborative
    gains, in place. No-op (returns False) if a component is missing.

    Every single-term selection metric tried on this model failed by being blind
    to a different collapse, and the pattern was always the same — the metric
    kept improving while something else it did not watch fell apart:

      - val_L_repr    : a weighted SUM of raw losses, dominated by the magnitude
                        of L_ii (~6.2) and L_ui (~5.5). Fell steadily while ITC
                        overfit. Blind to overfitting.
      - val_gain_itc  : ITC cannot generalise from CF at all (its target is the
                        caption's text CLS, dominated by the per-item title,
                        which MF embeddings do not carry). Selecting on it means
                        selecting on noise after ~ep2.
      - val_sem_off_gain_llm : generalises correctly and keeps rising to ep46 —
                        but it is the ONLY term that does, so it happily rode
                        past the epoch where g_ii/g_ui collapsed. Measured:
                        between ep16 and ep46 it gained +0.004 while g_ii went
                        +0.267 -> -0.015 and g_ui +0.330 -> +0.029.

    The composite fixes that by watching both halves of what Stage 2/3 consume:
    the alignment head Stage 2 warm-starts (gain_llm, measured with the semantic
    source OFF so the bank cannot supply it) and the collaborative signal Stage 3
    scores with (gain_ii, gain_ui). On the ep1-46 run its peak lands at ep16
    (0.615) with ep46 at 0.327 — i.e. it picks the checkpoint that carries both.

    All three terms are ``gain`` values (``ln(n) - L``), so they are already
    normalised by their own chance level and can be summed across losses with
    different candidate counts.
    """
    keys = (GAIN_REPR_ALIGN_KEY,) + GAIN_REPR_COLLAB_KEYS
    if any(key not in metrics for key in keys):
        return False
    collab = sum(float(metrics[key]) for key in GAIN_REPR_COLLAB_KEYS)
    metrics["val_gain_repr"] = float(metrics[GAIN_REPR_ALIGN_KEY]) + collab_weight * collab
    return True


def _selection_weights(w_itc, w_itm, w_itg, w_llm, w_ii=0.0, w_ui=0.0):
    """Weights applied to the ``L_repr`` selection metric."""
    return {
        "L_itc": float(w_itc),
        "L_itm": float(w_itm),
        "L_itg": float(w_itg),
        "L_llm": float(w_llm),
        "L_ii": float(w_ii),
        "L_ui": float(w_ui),
    }


class MetricAccumulator:
    """Average each metric over the batches where its sample type was present.

    Averaging over *all* batches scales every metric by the fraction of
    batches carrying its type, and that fraction differs sharply between
    splits: the train loader shuffles (types interleave, so every batch holds
    a few item_text rows) while val/test do not (the builder emits item_text
    as one contiguous block, so a handful of val batches are almost entirely
    item_text and the rest hold none). Diluted averages are therefore not
    comparable across splits.

    ``n_<type>`` records the mean number of rows of that type per contributing
    batch. In-batch contrastives draw negatives from exactly those rows, so
    chance accuracy is ``1 / n`` — without ``n``, a top-1 number cannot be
    read at all.
    """

    def __init__(self, weights=None):
        self.loss_total = 0.0
        self.steps = 0
        self.sums = {key: 0.0 for key in METRIC_GROUPS}
        self.active = {key: 0 for key in METRIC_GROUPS}
        self.n_sums = {group: 0.0 for group in SAMPLE_TYPES}
        self.n_active = {group: 0 for group in SAMPLE_TYPES}
        # Weights for the L_repr selection metric. Default 1.0 per term keeps the
        # old unweighted sum for callers that do not pass anything.
        self.weights = dict(weights or {})

    def update(self, loss, logs, counts):
        self.loss_total += float(loss.item())
        self.steps += 1
        for key, group in METRIC_GROUPS.items():
            if counts[group] >= 2:
                self.sums[key] += float(logs[key].item())
                self.active[key] += 1
        for group in SAMPLE_TYPES:
            if counts[group] >= 2:
                self.n_sums[group] += float(counts[group])
                self.n_active[group] += 1

    def result(self):
        out = {"loss": self.loss_total / self.steps if self.steps else 0.0}
        for key in METRIC_GROUPS:
            out[key] = self.sums[key] / self.active[key] if self.active[key] else 0.0
        for group in SAMPLE_TYPES:
            out[f"n_{group}"] = (
                self.n_sums[group] / self.n_active[group] if self.n_active[group] else 0.0
            )
            out[f"frac_{group}"] = self.n_active[group] / self.steps if self.steps else 0.0

        # Nats earned over a uniform predictor. An in-batch InfoNCE with n
        # candidates scores ln(n) at chance, so ``ln(n) - L`` is the part of the
        # loss the model actually produced. Unlike raw L_*, this is comparable
        # across splits with different batch composition: train shuffles to ~15
        # item_text rows per batch while a block-ordered eval loader gives ~250,
        # and that alone moves L_itc by ~2.8 nats at identical model quality.
        for key, group in RETRIEVAL_TERMS:
            n = out[f"n_{group}"]
            out[f"gain_{key}"] = math.log(n) - out[f"L_{key}"] if n > 1.0 else 0.0

        # Undiluted sum over the item_text objectives — exactly the terms whose
        # result Stage 2/3 inherit through the Q-Former body and the aligned
        # projection. Weighted rather than raw so that damping a term in the loss
        # also damps its vote here: w_itm < 1 exists because ITM can sit at its
        # trivial 2/3 baseline, and an unweighted L_repr let that dead term drive
        # most of the epoch-to-epoch delta while L_itc quietly degraded.
        # ``loss`` is NOT comparable to this: it stays averaged over
        # every batch, so each term is implicitly weighted by how often its
        # sample type appears, which is an artifact of builder ordering rather
        # than a design choice.
        out["L_repr"] = sum(self.weights.get(key, 1.0) * out[key] for key in REPR_TERMS)
        return out


def train_step(
    batch,
    model: QRecInstructAlignmentModel,
    w_itc: float = 1.0,
    w_itm: float = 1.0,
    w_itg: float = 1.0,
    w_ii: float = 1.0,
    w_ui: float = 0.0,
    w_llm: float = 0.0,
    tau_itc: float = 0.07,
    tau_ii: float = 0.07,
    tau_ui: float = 0.07,
    tau_llm: float = 0.07,
    ui_condition_on_item: bool = False,
    w_ui_cond: float = 0.3,
    tau_ui_cond: float = 0.2,
    ui_cond_distill_mf: bool = True,
    ui_cond_neg: str = "random",
    debug_batch: bool = False,
):
    """BLIP-2 stage-1 step: ITC + ITM + ITG on item-text samples, plus the
    SigLLM-specific item-item and (ILM-style) user-item contrastives.

    ``ui_condition_on_item=True`` ADDS (does not replace) the DIN-style
    candidate-conditioned BPR that pretrains ``user_proj``, weighted by
    ``w_ui_cond`` (see ``QRecInstructAlignmentModel.loss_user_item``)."""

    device = batch["i_left"].device

    if debug_batch:
        _log_batch_preview(batch)

    zero = next(model.parameters()).sum() * 0.0
    logs = {
        "L_itc": zero,
        "L_itm": zero,
        "L_itg": zero,
        "L_ii": zero,
        "L_ui": zero,
        "L_uic": zero,
        "L_llm": zero,
        "itc_top1": zero.detach(),
        "itm_acc": zero.detach(),
        "itg_acc": zero.detach(),
        "itg_title_acc": zero.detach(),
        "ii_top1": zero.detach(),
        "ui_top1": zero.detach(),
        "uic_acc": zero.detach(),
        "llm_top1": zero.detach(),
    }
    losses = []

    item_text_idx = _indices_for_type(batch, "item_text", device)
    if item_text_idx.numel() >= 2:
        item_text_batch = _subset_batch(batch, item_text_idx)
        item_ids = item_text_batch["i_left"]
        text_list = item_text_batch["text"]

        loss_itc, sim_matrix, itc_top1 = model.loss_itc(item_ids, text_list, tau=tau_itc)
        logs["L_itc"] = loss_itc
        logs["itc_top1"] = itc_top1.detach()
        losses.append(w_itc * loss_itc)

        if w_itm > 0.0:
            loss_itm, itm_acc = model.loss_itm(item_ids, text_list, sim_matrix)
            logs["L_itm"] = loss_itm
            logs["itm_acc"] = itm_acc.detach()
            losses.append(w_itm * loss_itm)

        if w_itg > 0.0:
            loss_itg, itg_acc, itg_title_acc = model.loss_itg(item_ids, text_list)
            logs["L_itg"] = loss_itg
            logs["itg_acc"] = itg_acc.detach()
            logs["itg_title_acc"] = itg_title_acc.detach()
            losses.append(w_itg * loss_itg)

        if w_llm > 0.00 and getattr(model, "has_llm_align", False):
            loss_llm, llm_top1 = model.loss_llm_align(item_ids, tau=tau_llm)
            logs["L_llm"] = loss_llm
            logs["llm_top1"] = llm_top1.detach()
            losses.append(w_llm * loss_llm)

    item_item_idx = _indices_for_type(batch, "item_item", device)
    if w_ii > 0.0 and item_item_idx.numel() >= 2:
        item_item_batch = _subset_batch(batch, item_item_idx)
        loss_ii, ii_top1 = model.loss_item_item_ilm(
            item_item_batch["i_left"], item_item_batch["i_right"], tau=tau_ii
        )
        logs["L_ii"] = loss_ii
        logs["ii_top1"] = ii_top1.detach()
        losses.append(w_ii * loss_ii)

    user_item_idx = _indices_for_type(batch, "user_item", device)
    if w_ui > 0.0 and user_item_idx.numel() >= 2:
        user_item_batch = _subset_batch(batch, user_item_idx)
        # History-pooled user source (multi-source cross-attention, the Stage-3
        # <UserProfile> path). The collate pads "his" into a [B, L] LongTensor
        # (0 = padding). Old pkls carry no history -> all-zero tensor -> None
        # -> single-user-vector fallback inside the loss.
        history_ids = user_item_batch.get("his")
        if isinstance(history_ids, torch.Tensor):
            if history_ids.numel() == 0 or not bool((history_ids != 0).any()):
                history_ids = None
        elif history_ids is not None and not any(history_ids):
            history_ids = None
        loss_ui, ui_top1, loss_uic, uic_acc = model.loss_user_item(
            user_item_batch["u"],
            user_item_batch["i_left"],
            tau=tau_ui,
            condition_on_item=ui_condition_on_item,
            tau_cond=tau_ui_cond,
            cond_distill_mf=ui_cond_distill_mf,
            cond_neg_mode=ui_cond_neg,
            history_ids=history_ids,
        )
        logs["L_ui"] = loss_ui
        logs["ui_top1"] = ui_top1.detach()
        losses.append(w_ui * loss_ui)
        if loss_uic is not None and w_ui_cond > 0.0:
            logs["L_uic"] = loss_uic
            logs["uic_acc"] = uic_acc.detach()
            losses.append(w_ui_cond * loss_uic)

    loss = sum(losses, zero)
    counts = {
        "item_text": int(item_text_idx.numel()),
        "item_item": int(item_item_idx.numel()),
        "user_item": int(user_item_idx.numel()),
    }
    return loss, logs, counts


def evaluate_loss(
    model,
    loader,
    w_itc=1.0,
    w_itm=1.0,
    w_itg=1.0,
    w_ii=1.0,
    w_ui=0.0,
    w_llm=0.0,
    tau_itc=0.07,
    tau_ii=0.07,
    tau_ui=0.07,
    tau_llm=0.07,
    ui_condition_on_item=False,
    w_ui_cond=0.3,
    tau_ui_cond=0.2,
    ui_cond_distill_mf=True,
    ui_cond_neg="random",
    selection_weights=None,
):
    model.eval()
    device = next(model.parameters()).device
    accumulator = MetricAccumulator(
        weights=selection_weights
        if selection_weights is not None
        else _selection_weights(w_itc, w_itm, w_itg, w_llm, w_ii, w_ui)
    )

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss, logs, counts = train_step(
                batch,
                model,
                w_itc=w_itc,
                w_itm=w_itm,
                w_itg=w_itg,
                w_ii=w_ii,
                w_ui=w_ui,
                w_llm=w_llm,
                tau_itc=tau_itc,
                tau_ii=tau_ii,
                tau_ui=tau_ui,
                tau_llm=tau_llm,
                ui_condition_on_item=ui_condition_on_item,
                w_ui_cond=w_ui_cond,
                tau_ui_cond=tau_ui_cond,
                ui_cond_distill_mf=ui_cond_distill_mf,
                ui_cond_neg=ui_cond_neg,
            )
            accumulator.update(loss, logs, counts)

    model.train()
    return accumulator.result()


def _item_text_only(dataset):
    """Subset holding just the ``item_text`` rows (for the fixed-n eval pass)."""
    from torch.utils.data import Subset

    indices = [i for i, s in enumerate(dataset.samples) if s["sample_type"] == "item_text"]
    return Subset(dataset, indices)


def build_item_text_eval_loader(cfg, data_dir, filename, batch_size):
    """Fixed-``n`` loader over the item_text rows only.

    The main loaders mix sample types, so ``n_item_text`` per batch depends on
    the split's composition: with the item-level holdout, a val batch carries a
    handful of item_text rows while a train batch carries many. In-batch
    retrieval chance is ``1/n``, so val ITC@1 looks better than train for free,
    ITM has almost no hard negative to mine from a tiny similarity matrix and
    parks at its trivial 2/3 baseline, and L_llm is capped at ``ln(n)`` well
    below the train figure.

    ``drop_last=True`` is what actually pins ``n``: without it the last batch of
    each split is whatever remains (a 205-row val block and a 2600-row train
    block yield different tail sizes), so the per-split averages would still mix
    chance levels and stay incomparable. With it, every contributing batch holds
    exactly ``batch_size`` rows on every split.

    Consequence: ``item_text_eval_batch_size`` must be **<= the SMALLEST**
    held-out item_text block, not >= the largest — a split with fewer rows than
    one batch yields zero batches and drops out of the diagnostic entirely
    (callers check ``len(loader)`` and warn).
    """

    class _BatchSizeCfg:
        def __init__(self, base, batch_size):
            self.batch_size = batch_size
            self.num_workers = int(base.num_workers)

    return build_qformer_loader(
        _BatchSizeCfg(cfg, int(batch_size)),
        filename=os.path.join(data_dir, filename),
        shuffle=False,
        filter_fn=_item_text_only,
        drop_last=True,
    )


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    val_logs,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **{f"val_{key}": value for key, value in val_logs.items()},
        },
        checkpoint_path,
    )

def _load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train_qformer_stage1_representation(cfg):
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_qformer_loaders(cfg, data_dir=cfg.data_dir)

    # Report which user-side source the ui losses will use. History pooling
    # requires a pkl rebuilt by the current QFormerAlignmentBuilder (stores
    # per-row "his"); old pkls fall back to the single MF user vector. Scan
    # the raw sample list (the flat list is block-ordered with user_item
    # last, so probing the first rows would always miss it).
    base_dataset = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    raw_samples = getattr(base_dataset, "samples", None) or []
    if any(s.get("his") for s in raw_samples if s.get("sample_type") == "user_item"):
        log_step(
            "user-item source",
            "HISTORY pooling (multi-source cross-attention, Stage-3 <UserProfile> path)",
        )
    else:
        log_step(
            "user-item source",
            "single MF user vector (S=1). Rebuild the qformer pkls with "
            "build_qformer_dataset to enable history pooling.",
        )

    mf = _init_rec_model(cfg, device)
    qformer_d_model = int(cfg.get("qformer_d_model", 768))

    w_llm = float(cfg.get("w_llm", 0.0))
    tau_llm = float(cfg.get("tau_llm", 0.07))
    llm_emb_normalize = str(cfg.get("llm_emb_normalize", "center"))
    sem_source = bool(cfg.get("sem_source", False))
    sem_dropout = float(cfg.get("sem_source_dropout", 0.5))

    # L_llm TARGET bank (input-embedding space — the space Stage 3 injects into).
    item_llm_emb = None
    d_llm = None
    item_llm_emb_path = cfg.get("item_llm_emb_path", None)
    if w_llm > 0.0:
        if not item_llm_emb_path or not os.path.exists(item_llm_emb_path):
            raise FileNotFoundError(
                "item_llm_emb_path is required and must exist when w_llm > 0.0, "
                f"but got: {item_llm_emb_path}"
            )
        blob = torch.load(item_llm_emb_path, map_location="cpu")
        item_llm_emb = blob["item_llm_emb"] if isinstance(blob, dict) else blob
        d_llm = int(item_llm_emb.size(-1))
        log_step(
            "Loaded L_llm target bank",
            f"path={item_llm_emb_path}, shape={tuple(item_llm_emb.shape)}",
        )

    # Cross-attention SOURCE bank. MUST be a different bank from the L_llm
    # target: with source == target, the model can solve L_llm by copying the
    # source through proj_sem -> queries -> out_proj -> llm_align_proj without
    # learning anything from CF — and evaluate_loss runs with model.eval()
    # (no sem dropout), so validation L_llm was a 100% leak.
    item_sem_emb = None
    d_sem = None
    item_sem_emb_path = cfg.get("item_sem_emb_path", None)
    if sem_source:
        if not item_sem_emb_path or not os.path.exists(item_sem_emb_path):
            raise FileNotFoundError(
                "item_sem_emb_path is required and must exist when sem_source "
                f"is enabled, but got: {item_sem_emb_path}"
            )
        if w_llm > 0.0 and os.path.realpath(item_sem_emb_path) == os.path.realpath(
            item_llm_emb_path
        ):
            raise ValueError(
                "item_sem_emb_path must differ from item_llm_emb_path when "
                "w_llm > 0.0: feeding the L_llm target bank as a cross-attention "
                "source lets the model copy the target through proj_sem and "
                "turns L_llm (train AND val) into a leak. Point item_sem_emb_path "
                "at the last-hidden bank (item_llm_emb.pt) or set w_llm: 0.0."
            )
        blob = torch.load(item_sem_emb_path, map_location="cpu")
        item_sem_emb = blob["item_llm_emb"] if isinstance(blob, dict) else blob
        d_sem = int(item_sem_emb.size(-1))
        # COVERAGE, not just shape. An all-zero row is treated as "no text
        # available" and masked out of cross-attention, so a bank that is mostly
        # zeros silently disables the semantic source entirely — proj_sem then
        # receives no gradient no matter what sem_source_dropout is set to, and
        # the sem_on / sem_off diagnostic pair comes out bit-identical. That is
        # the first thing to rule out when proj_sem looks dead.
        covered = int((item_sem_emb.norm(dim=-1) > 0).sum())
        total = int(item_sem_emb.size(0))
        log_step(
            "Loaded semantic source bank",
            f"path={item_sem_emb_path}, shape={tuple(item_sem_emb.shape)}, "
            f"covered_rows={covered}/{total} ({covered / max(total, 1):.1%}), "
            f"mean_norm_covered="
            f"{item_sem_emb[item_sem_emb.norm(dim=-1) > 0].norm(dim=-1).mean().item() if covered else 0.0:.4f}",
        )
        if covered == 0:
            raise ValueError(
                f"Semantic source bank {item_sem_emb_path} has ZERO non-zero rows, so "
                "every semantic token would be masked out and proj_sem could never "
                "train. Re-distill it (--space last_hidden against the base LLM, "
                "passing all splits via --data-pkl) or set sem_source: False."
            )
        if covered < total * 0.5:
            log_step(
                "WARNING",
                f"only {covered / max(total, 1):.1%} of the semantic bank is populated; "
                "uncovered items fall back to CF-only and contribute no gradient to "
                "proj_sem. Check that --data-pkl covered every split at distill time.",
            )

    qformer = _init_qformer(cfg, qformer_d_model, device, d_sem=d_sem)
    if sem_source:
        log_step(
            "Semantic cross-attention source active",
            f"d_sem={d_sem}, sem_source_dropout={sem_dropout}",
        )

    model = QRecInstructAlignmentModel(
        mf=mf,
        qformer=qformer,
        item_llm_emb=item_llm_emb,
        d_llm=d_llm,
        llm_emb_normalize=llm_emb_normalize,
        item_sem_emb=item_sem_emb,
        sem_dropout=sem_dropout,
        pair_logit_center=bool(cfg.get("pair_logit_center", False)),
        itc_logit_center=bool(cfg.get("itc_logit_center", True)),
        bpr_logit_center=bool(cfg.get("bpr_logit_center", True)),
        sem_for_text_losses=bool(cfg.get("sem_for_text_losses", False)),
        # Must match model.qformer_config.center_soft_tokens: Stage 2
        # warm-starts llm_proj from this llm_align_proj under a strict load, so
        # a mismatch here silently changes the layout the export has to fit.
        center_soft_tokens=bool(cfg.get("center_soft_tokens", True)),
    ).to(device)

    # DIN-style pretraining of the candidate-conditioning path: only possible
    # when the adapter was built with the conditioning projection. The BPR is
    # ADDED next to the ui InfoNCE (L_uic / uic_acc, chance 0.5); the InfoNCE
    # keeps its historical meaning (ui_top1 chance = 1/n). Gated on the weight
    # so w_ui_cond=0.0 also skips the extra conditioned forwards entirely.
    w_ui = float(cfg.get("w_ui", 0.0))
    tau_ui = float(cfg.get("tau_ui", 0.07))
    w_ui_cond = float(cfg.get("w_ui_cond", 0.0))
    ui_condition_on_item = bool(qformer.user_conditioned) and w_ui_cond > 0.0
    tau_ui_cond = float(cfg.get("tau_ui_cond", 0.2))
    ui_cond_distill_mf = bool(cfg.get("ui_cond_distill_mf", True))
    ui_cond_neg = str(cfg.get("ui_cond_neg", "random"))
    if ui_condition_on_item:
        log_step(
            "user_proj pretraining active",
            f"candidate-conditioned pairwise BPR added to the ui term "
            f"(w_ui_cond={w_ui_cond}, tau_ui_cond={tau_ui_cond}, "
            f"distill_mf={ui_cond_distill_mf}, neg={ui_cond_neg}, "
            f"uic_acc chance level 0.5"
            + (", ceiling ~ MF held-out pairwise acc" if ui_cond_distill_mf else "")
            + ").",
        )

    # Selection weights may deviate from the LOSS weights: the collaborative
    # contrastives belong in L_repr (they are the CF signal Stage 3 consumes)
    # but val L_ii swings ~+-1.2 nats epoch-to-epoch, and at full weight that
    # noise decided every late-epoch selection verdict. Damp its VOTE without
    # touching its gradient.
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    selection_weights = _selection_weights(
        cfg.w_itc,
        cfg.w_itm,
        cfg.w_itg,
        w_llm,
        float(cfg.get("select_w_ii", cfg.w_ii)),
        float(cfg.get("select_w_ui", w_ui)),
    )

    opt = _init_optimizer(model, cfg.lr, weight_decay=cfg.weight_decay, decoupled_weight_decay=bool(cfg.get("decoupled_weight_decay", True)))

    # Fixed-n item_text eval (see build_item_text_eval_loader) plus the CF-only
    # diagnostic. Both are measurement-only: no gradient, no effect on selection
    # or early stopping. They answer the two questions the main val line cannot:
    # "is this comparable to train?" (fixed n) and "how much of it survives
    # without the semantic source?" (sem off).
    item_text_eval_batch = int(cfg.get("item_text_eval_batch_size", 512))
    diag_every = int(cfg.get("diagnostic_every_epochs", 1))
    item_text_loaders = {}
    if diag_every > 0:
        for split, filename in (
            ("train", "train_qformer_ood2.pkl"),
            ("val", "valid_qformer_ood2.pkl"),
            ("test", "test_qformer_ood2.pkl"),
        ):
            loader = build_item_text_eval_loader(
                cfg, cfg.data_dir, filename, item_text_eval_batch
            )
            # len(loader), not len(loader.dataset): drop_last means a split with
            # fewer item_text rows than one batch produces ZERO batches, and
            # every metric would come back as a silent 0.0.
            if len(loader) > 0:
                item_text_loaders[split] = loader
            else:
                log_step(
                    "WARNING",
                    f"item_text diagnostic skipped for {split}: "
                    f"{len(loader.dataset)} item_text rows < "
                    f"item_text_eval_batch_size={item_text_eval_batch}. Lower that "
                    f"key to at most the smallest held-out block.",
                )
        if item_text_loaders:
            log_step(
                "Diagnostics enabled",
                "fixed-n item_text eval ("
                + ", ".join(
                    f"{s}={len(l) * item_text_eval_batch}/{len(l.dataset)} rows"
                    f" in {len(l)} batch(es)"
                    for s, l in item_text_loaders.items()
                )
                + f", n={item_text_eval_batch} per batch, chance="
                f"{1.0 / item_text_eval_batch:.4f}) + CF-only (sem off) pass every "
                f"{diag_every} epoch(s)",
            )

    # uAUC probe: the ONE in-training metric measured on the task the project is
    # judged on. Everything else Stage 1 optimises is a CROSS-USER retrieval
    # pretext (L_ii / L_ui: pick this user's item out of other users' items),
    # while the target metric is WITHIN-user ranking. Those come apart badly:
    # measured on the val pkl, the parameter-free readout over the frozen MF
    # embeddings scores only ~+0.05 nats on the retrieval pretext — barely above
    # chance — yet the same MF ranks held-out candidates at 0.708 uAUC. So a
    # pretext number can look degenerate while the signal is fine, and it can be
    # driven up by memorisation (train g_ui >0.67 against val ~0.14) while uAUC
    # never moves. This probe closes that blind spot without the LLM: it scores
    # pool(qformer(history)) . pool(qformer(target)) on real eval rows and
    # reports the delta against the MF dot product on the SAME rows.
    #
    # Set run.qformer_stage1.selection_metric: val_probe_gain to select the
    # checkpoint on "does the channel beat the MF it was built from".
    probe_every = int(cfg.get("probe_every_epochs", 0))
    probe_uauc_loader = None
    if probe_every > 0:
        if cfg.get("dataset_cfg") is None:
            log_step(
                "WARNING",
                "probe_every_epochs > 0 but no dataset_cfg was passed; the uAUC "
                "probe needs it to build MovieOODDataset. Probe DISABLED.",
            )
            probe_every = 0
        else:
            probe_uauc_loader = build_probe_loader(
                cfg.dataset_cfg,
                str(cfg.get("probe_split", "valid_ood2.pkl")),
                batch_size=int(cfg.get("probe_batch_size", 64)),
            )
            log_step(
                "uAUC probe enabled",
                f"split={cfg.get('probe_split', 'valid_ood2.pkl')}, "
                f"rows={len(probe_uauc_loader.dataset)}, every {probe_every} epoch(s). "
                f"Reports val_probe_uauc / _uauc_centered vs val_probe_mf_dot_uauc, "
                f"and val_probe_gain = best(channel) - MF.",
            )

    # Within-user BPR (uAUC surrogate) on real CTR rows. THE gap it fills: not
    # one existing Stage-1 loss asks "within this user's candidates, is the liked
    # one ranked higher" — they all score cross-user retrieval or caption
    # reconstruction. That is why the measured channel uAUC FELL from 0.583 to
    # 0.532 over three epochs while every internal metric rose: it was nobody's
    # objective. Needs its own loader (the qformer pkl has positives only) and a
    # USER-GROUPED sampler, or batches hold no same-user pos/neg pair and the
    # loss silently returns 0.
    w_rank = float(cfg.get("w_rank", 0.0))
    tau_rank = float(cfg.get("tau_rank", 1.0))
    rank_loader = None
    rank_iter = None
    if w_rank > 0.0:
        if cfg.get("dataset_cfg") is None:
            log_step("WARNING", "w_rank > 0 but no dataset_cfg; BPR DISABLED.")
            w_rank = 0.0
        else:
            from sigllm.common.user_grouped_sampler import UserGroupedSampler
            from torch.utils.data import DataLoader as _DL

            rank_ds = MovieOODDataset(cfg.dataset_cfg, filename=str(cfg.get("rank_split", "train_ood2.pkl")))
            uids = [int(r) for r in rank_ds.annotation["UserID"].tolist()]
            labs = [int(r) for r in rank_ds.annotation["label"].tolist()]
            rank_bs = int(cfg.get("rank_batch_size", 64))
            sampler = UserGroupedSampler(
                uids, labs, batch_size=rank_bs,
                items_per_user=int(cfg.get("rank_items_per_user", 8)),
                seed=int(cfg.get("seed", 0)),
            )
            rank_loader = _DL(
                rank_ds, batch_size=rank_bs, sampler=sampler,
                num_workers=0, collate_fn=probe_collate,
            )
            rank_iter = iter(rank_loader)
            log_step(
                "Within-user BPR ACTIVE",
                f"w_rank={w_rank}, tau={tau_rank}, split={cfg.get('rank_split', 'train_ood2.pkl')}, "
                f"rows={len(rank_ds)}, batch={rank_bs}, items_per_user="
                f"{int(cfg.get('rank_items_per_user', 8))}. Watch rank_acc (0.5 = chance) "
                f"and, more importantly, val_probe_gain — L_rank fits held-in labels so it "
                f"CAN memorise, exactly like L_ui.",
            )

    def run_uauc_probe(epoch_index):
        if probe_uauc_loader is None:
            return {}
        out = probe_metrics(
            model.qformer, model.mf, model.item_sem_emb, probe_uauc_loader, device
        )
        pool = pooling_usage(
            model.qformer, model.mf, model.item_sem_emb, probe_uauc_loader, device
        )
        out.update(pool)
        if pool:
            log_step(
                f"[DIAG ep{epoch_index}] history pooling usage",
                ", ".join(f"cos(full, last{k.split('_k')[-1]})={v:.4f}" for k, v in sorted(pool.items()))
                + " | cos ~ 1.0 at last1 means the pooling IGNORES all but the most "
                "recent item — the Q-Former is then not doing the one job it was "
                "kept for, and losing to a mask-mean is expected.",
            )
        sem_part = ""
        if "val_probe_sem_gain" in out:
            sem_part = (
                f"| sem_off={out['val_probe_uauc_sem_off']:.4f} "
                f"sem_gain={out['val_probe_sem_gain']:+.4f} "
            )
        log_step(
            f"[DIAG ep{epoch_index}] uAUC probe (within-user, the real task)",
            f"channel_uauc={out['val_probe_uauc']:.4f} "
            f"centered={out['val_probe_uauc_centered']:.4f} "
            f"dot={out['val_probe_uauc_dot']:.4f} | "
            f"MF_dot={out['val_probe_mf_dot_uauc']:.4f} "
            f"MF_hist_cos={out['val_probe_mf_hist_cos_uauc']:.4f} | "
            f"GAIN={out['val_probe_gain']:+.4f} over MF on {int(out['val_probe_rows'])} rows "
            f"| COMBINED(MF+qformer)={out['val_probe_uauc_combined']:.4f} "
            f"w={out['val_probe_combined_best_w']:g} "
            f"combined_gain={out['val_probe_combined_gain']:+.4f} "
            f"{sem_part}"
            f"| gain <= 0 means Stage 1 has not beaten the MF it was built from, "
            f"whatever g_ii/g_ui say; gain ~ 0 means pass-through, so Stage 3 can at "
            f"best re-derive MF. sem_gain ~ 0 means the semantic bank — the only "
            f"component carrying information MF lacks — is not helping this task.",
        )
        return out

    def run_diagnostics(epoch_index):
        """Fixed-n item_text metrics, with the semantic source on and off.

        Returns a flat ``{f"{split}_{tag}_{metric}": value}`` dict (e.g.
        ``val_sem_off_gain_itc``) so these can be used for CHECKPOINT SELECTION,
        not just printed. That matters: with the semantic source on, ITC/L_llm
        can be solved by reading the distilled bank — which is derived from the
        very captions ITC targets — so a selection metric measured with sem on
        rewards that shortcut. Measured on ml-1m, sem_on gave val g_itc +3.010
        while sem_off gave -0.701 in the SAME epoch: the CF path had learned
        nothing, yet the sem_on metric looked excellent. Selecting on the
        sem_off variant selects on what Stage 3 actually consumes.
        """
        probe_loader = item_text_loaders.get("val") or next(iter(item_text_loaders.values()))
        anisotropy = _query_anisotropy(model, probe_loader)
        if anisotropy is not None:
            log_step(
                f"[DIAG ep{epoch_index}] query geometry",
                f"n={anisotropy['n']} offdiag_cos_raw={anisotropy['raw']:.4f} "
                f"offdiag_cos_centered={anisotropy['centered']:.4f} "
                f"(raw near 1.0 = every item encodes to the same direction, so the "
                f"uncentered cosine losses cannot resolve items; centered is what "
                f"pair_logit_center/itc_logit_center actually score against)",
            )
        path_norms = _zero_init_path_norms(model)
        if path_norms:
            log_step(
                f"[DIAG ep{epoch_index}] zero-init paths",
                ", ".join(f"{k}={v:.6f}" for k, v in sorted(path_norms.items()))
                + " | still 0.0 = that path is inert and receiving no usable "
                "gradient (proj_sem: check bank coverage; user_proj: check "
                "L_uic against ln 2 = 0.6931 and bpr_logit_center)",
            )
        diag_metrics = {}
        # With no semantic bank the two passes are identical by construction, so
        # run once and publish it under BOTH tags. Emitting the sem_off keys
        # unconditionally keeps a sem_off selection metric valid whether or not
        # sem_source is enabled — otherwise turning sem_source off would make the
        # early stopper KeyError on a missing key.
        has_sem = model.item_sem_emb is not None
        # FACTORIES, not instances: a @contextmanager object is single-use (its
        # __enter__ deletes the stored args), so one instance reused across the
        # splits raises AttributeError on the second split. Build a fresh context
        # per pass instead.
        passes = (
            (("sem_on", contextlib.nullcontext), ("sem_off", model.sem_disabled))
            if has_sem
            else (("sem_on", contextlib.nullcontext),)
        )
        for split, loader in item_text_loaders.items():
            for tag, make_ctx in passes:
                with make_ctx():
                    logs = evaluate_loss(
                        model,
                        loader,
                        w_itc=cfg.w_itc,
                        w_itm=cfg.w_itm,
                        w_itg=cfg.w_itg,
                        w_ii=0.0,
                        w_ui=0.0,
                        w_llm=w_llm,
                        tau_itc=cfg.tau_itc,
                        tau_llm=tau_llm,
                        selection_weights=selection_weights,
                    )
                for key in DIAG_EXPORTED_METRICS:
                    diag_metrics[f"{split}_{tag}_{key}"] = float(logs[key])
                if not has_sem:
                    for key in DIAG_EXPORTED_METRICS:
                        diag_metrics[f"{split}_sem_off_{key}"] = float(logs[key])
                # ITM only when it is actually enabled: with w_itm=0 the loss is
                # never computed and the log would read "ITM_acc=0.0000", which
                # looks like an accuracy of zero rather than "not run".
                itm_part = (
                    f"L_itm={logs['L_itm']:.4f} ITM_acc={logs['itm_acc']:.4f} "
                    if float(cfg.w_itm) > 0.0
                    else "ITM=off "
                )
                log_step(
                    f"[DIAG ep{epoch_index}] {split}/{tag}",
                    f"n_it={logs['n_item_text']:.0f} "
                    f"(chance={1.0 / max(logs['n_item_text'], 1.0):.4f}) "
                    f"L_itc={logs['L_itc']:.4f} ITC@1={logs['itc_top1']:.4f} "
                    f"g_itc={logs['gain_itc']:+.3f} "
                    + itm_part
                    + f"L_itg={logs['L_itg']:.4f} ITG_title={logs['itg_title_acc']:.4f} "
                    f"L_llm={logs['L_llm']:.4f} LLM@1={logs['llm_top1']:.4f} "
                    f"g_llm={logs['gain_llm']:+.3f}",
                )
        return diag_metrics

    outdir = cfg.output_dir
    os.makedirs(outdir, exist_ok=True)
    best_checkpoint_path = os.path.join(outdir, cfg.best_checkpoint_name)
    # Default selects on the COMPOSITE val_gain_repr — the CF-only alignment gain
    # plus the collaborative gains. See add_gain_repr for the three single-term
    # metrics this replaces and what each of them was blind to. Any key in the
    # per-epoch metrics dict works (val_L_repr, val_sem_off_gain_llm, ...).
    selection_metric = cfg.get("selection_metric", "val_gain_repr")
    selection_mode = cfg.get("selection_mode", "max")
    # Weight on EACH collaborative gain inside val_gain_repr (see add_gain_repr).
    select_collab_weight = float(cfg.get("select_collab_weight", 0.5))

    # Metrics that come from (or are built on) the item_text diagnostic only
    # exist on epochs where it ran, and only if a loader was built at all. Fail
    # here, at setup, rather than on the first epoch that happens to skip it.
    # val_gain_repr counts: it reads a sem_off diagnostic key.
    DIAG_DERIVED_TAGS = ("sem_on", "sem_off", "gain_repr")
    diag_dependent = [
        name
        for name in (selection_metric, str(cfg.get("collapse_metric", selection_metric)))
        if any(tag in name for tag in DIAG_DERIVED_TAGS)
    ]
    if diag_dependent:
        if not item_text_loaders:
            raise ValueError(
                f"{diag_dependent} depend on the item_text diagnostic, but no "
                "diagnostic loader was built. Either lower "
                "item_text_eval_batch_size until every split yields >=1 batch, set "
                "diagnostic_every_epochs > 0, or pick metrics from the main "
                "loaders (e.g. val_gain_llm)."
            )
        if diag_every != 1:
            raise ValueError(
                f"{diag_dependent} need the diagnostic on EVERY evaluated epoch, "
                f"but diagnostic_every_epochs={diag_every}. Set it to 1."
            )
        # val_gain_repr always reads the val split; the others carry their split
        # in the key name.
        for name in diag_dependent:
            split_prefix = (
                "val" if "gain_repr" in name else name.split("_", 1)[0]
            )
            if split_prefix not in item_text_loaders:
                raise ValueError(
                    f"{name!r} refers to split {split_prefix!r}, which has no "
                    f"diagnostic loader. Available: {sorted(item_text_loaders)}."
                )

    stopper = EarlyStopping(
        ref_metric=selection_metric,
        monitor_mode=selection_mode,
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )
    # EMA smoothing of the selection metric before it reaches the stopper.
    # Val L_repr moves by ~0.02-0.3/epoch late in training while its
    # epoch-to-epoch noise is of the same order (val L_ii/L_ui swings), so an
    # unsmoothed stopper both burns patience on noise plateaus and crowns
    # lucky dips as "best". 0.0 disables (raw metric, old behaviour).
    selection_ema = float(cfg.get("selection_ema", 0.5))
    selection_ema_value = None
    # Threshold for collapse_metric, whatever that metric is. Named
    # min_gain_itc historically, when the guard could only watch gain_itc; the
    # old key is still honoured so existing configs keep working.
    min_collapse_gain = float(
        cfg.get("min_collapse_gain", cfg.get("min_gain_itc", 0.05))
    )
    collapse_patience = int(cfg.get("collapse_patience", 3))
    collapse_warmup = int(cfg.get("collapse_warmup_epochs", 3))
    # Defaults to the selection metric so the guard and the stopper can never
    # disagree about what "healthy" means. Only meaningful for a metric where
    # HIGHER is better and 0 is chance (the gain_* family).
    collapse_metric = cfg.get("collapse_metric", selection_metric)
    collapse_counter = 0
    aborted = False
    if collapse_patience > 0 and selection_mode != "max" and "collapse_metric" not in cfg:
        log_step(
            "WARNING",
            f"collapse guard compares {collapse_metric} < {min_collapse_gain}, but "
            f"selection_mode={selection_mode} suggests lower-is-better for that "
            "metric. Set collapse_metric to a gain_* key or collapse_patience: 0.",
        )
    log_step(
        "Training setup",
        f"seed={cfg.seed}, output_dir={outdir}, "
        f"selection={selection_metric} ({selection_mode}, ema={selection_ema}), "
        f"collapse_guard={collapse_metric} < {min_collapse_gain} "
        f"x{collapse_patience} after ep{collapse_warmup}",
    )

    for epoch in range(cfg.epoch):
        model.train()
        accumulator = MetricAccumulator(weights=selection_weights)
        train_steps = 0
        rank_loss_sum, rank_acc_sum, rank_steps, rank_acc_n = 0.0, 0.0, 0, 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            opt.zero_grad()

            loss, logs, counts = train_step(
                batch,
                model,
                w_itc=cfg.w_itc,
                w_itm=cfg.w_itm,
                w_itg=cfg.w_itg,
                w_ii=cfg.w_ii,
                w_ui=w_ui,
                w_llm=w_llm,
                tau_itc=cfg.tau_itc,
                tau_ii=cfg.tau_ii,
                tau_ui=tau_ui,
                tau_llm=tau_llm,
                ui_condition_on_item=ui_condition_on_item,
                w_ui_cond=w_ui_cond,
                tau_ui_cond=tau_ui_cond,
                ui_cond_distill_mf=ui_cond_distill_mf,
                ui_cond_neg=ui_cond_neg,
                debug_batch=cfg.debug_batch and epoch == 0 and train_steps < cfg.debug_batch_max_steps,
            )
            # Within-user BPR replay: ONE batch of real CTR rows per Q-Former
            # step, added to the same backward. This is the only Stage-1 term
            # that optimises the metric the project is judged on — see
            # loss_user_rank. Replayed from its own loader because the qformer
            # pkl carries POSITIVES only (no negatives to pair against) and the
            # BPR needs same-user pos/neg rows in one batch.
            if rank_iter is not None:
                rank_batch = next(rank_iter, None)
                if rank_batch is None:               # loader exhausted -> restart
                    rank_iter = iter(rank_loader)
                    rank_batch = next(rank_iter, None)
                if rank_batch is not None:
                    r_loss, r_acc = model.loss_user_rank(
                        rank_batch["UserID"].to(device),
                        rank_batch["TargetItemID"].to(device),
                        rank_batch["label"].to(device),
                        rank_batch["InteractedItemIDs_pad"].to(device),
                        tau=tau_rank,
                    )
                    loss = loss + w_rank * r_loss
                    # Tracked with plain locals: the accumulator only aggregates
                    # keys in METRIC_GROUPS (it is keyed by sample_type counts),
                    # so extra entries in `logs` would be silently dropped.
                    rank_loss_sum += float(r_loss.detach().item())
                    if r_acc == r_acc:            # not NaN (no pairs this batch)
                        rank_acc_sum += r_acc
                        rank_acc_n += 1
                    rank_steps += 1

            loss.backward()
            # Late-run train loss was drifting UP (12.53 -> 12.88 over the
            # last epochs of the 34-epoch run) — clip so single-batch spikes
            # cannot walk the model out of its basin.
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip_norm
                )
            opt.step()

            accumulator.update(loss, logs, counts)
            train_steps += 1

        rank_log_str = ""
        if rank_steps:
            rank_log_str = (
                f"L_rank={rank_loss_sum / rank_steps:.4f} "
                f"rank_acc={(rank_acc_sum / rank_acc_n) if rank_acc_n else float('nan'):.4f} "
                f"rank_pairs={rank_acc_n}/{rank_steps}"
            )
            # _per_user_pairwise_loss returns a graph-preserving 0 when a batch
            # holds no same-user (pos, neg) pair, so a mis-grouped sampler makes
            # the term vanish SILENTLY — it would look configured while adding
            # nothing but an extra forward. rank_pairs counts the batches that
            # actually contributed.
            if rank_acc_n == 0:
                log_step(
                    "WARNING",
                    "w_rank > 0 but NO batch contained a same-user (pos, neg) pair, so "
                    "the BPR contributed exactly nothing. Raise rank_items_per_user or "
                    "rank_batch_size — UserGroupedSampler needs several rows per user "
                    "with MIXED labels in one batch.",
                )
            elif rank_acc_n < 0.5 * rank_steps:
                log_step(
                    "WARNING",
                    f"only {rank_acc_n}/{rank_steps} BPR batches had a valid pair; the "
                    f"term is mostly inactive. Raise rank_items_per_user.",
                )
        if (epoch + 1) % cfg.log_epoch == 0:
            avg_train = accumulator.result()
            val_logs = evaluate_loss(
                model,
                val_loader,
                w_itc=cfg.w_itc,
                w_itm=cfg.w_itm,
                w_itg=cfg.w_itg,
                w_ii=cfg.w_ii,
                w_ui=w_ui,
                w_llm=w_llm,
                tau_itc=cfg.tau_itc,
                tau_ii=cfg.tau_ii,
                tau_ui=tau_ui,
                tau_llm=tau_llm,
                ui_condition_on_item=ui_condition_on_item,
                w_ui_cond=w_ui_cond,
                tau_ui_cond=tau_ui_cond,
                ui_cond_distill_mf=ui_cond_distill_mf,
                ui_cond_neg=ui_cond_neg,
                selection_weights=selection_weights,
            )
            # BEFORE building `metrics`: the diagnostic publishes the fixed-n
            # sem_on/sem_off keys that checkpoint selection may depend on.
            diag_metrics = {}
            if item_text_loaders and diag_every > 0 and (epoch + 1) % diag_every == 0:
                diag_metrics = run_diagnostics(epoch + 1)
            if probe_every > 0 and (epoch + 1) % probe_every == 0:
                diag_metrics.update(run_uauc_probe(epoch + 1))

            metrics = {
                "epoch": epoch + 1,
                **{f"val_{key}": value for key, value in val_logs.items()},
                **{f"train_{key}": value for key, value in avg_train.items()},
                **diag_metrics,
            }
            add_gain_repr(metrics, select_collab_weight)
            if selection_metric not in metrics:
                raise KeyError(
                    f"selection_metric={selection_metric!r} is not among the logged "
                    f"metrics, so the early stopper would KeyError on it. Available: "
                    f"{sorted(metrics)}"
                )
            # Raw (pre-EMA) value, kept for the log line — the stopper below may
            # see the smoothed one instead.
            raw_selection_value = float(metrics[selection_metric])
            ema_note = ""
            if selection_ema > 0.0:
                raw_value = raw_selection_value
                selection_ema_value = (
                    raw_value
                    if selection_ema_value is None
                    else selection_ema * selection_ema_value + (1.0 - selection_ema) * raw_value
                )
                # The stopper (improvement test, best-epoch choice, patience)
                # sees the SMOOTHED value; the raw value stays in the print.
                metrics[selection_metric] = selection_ema_value
                ema_note = f" ema={selection_ema_value:.4f}"

            print(
                f"epoch {epoch+1} | "
                f"Train Loss={avg_train['loss']:.4f} "
                f"L_itc={avg_train['L_itc']:.4f} L_itm={avg_train['L_itm']:.4f} "
                f"L_itg={avg_train['L_itg']:.4f} L_ii={avg_train['L_ii']:.4f} "
                f"L_ui={avg_train['L_ui']:.4f} L_uic={avg_train['L_uic']:.4f} "
                f"{rank_log_str} "
                f"L_llm={avg_train['L_llm']:.4f} "
                f"ITC@1={avg_train['itc_top1']:.4f} ITM_acc={avg_train['itm_acc']:.4f} "
                f"ITG_acc={avg_train['itg_acc']:.4f} ITG_title={avg_train['itg_title_acc']:.4f} "
                f"II@1={avg_train['ii_top1']:.4f} "
                f"UI@1={avg_train['ui_top1']:.4f} UIC_acc={avg_train['uic_acc']:.4f} "
                f"LLM@1={avg_train['llm_top1']:.4f} "
                f"g_itc={avg_train['gain_itc']:+.3f} g_llm={avg_train['gain_llm']:+.3f} "
                f"g_ii={avg_train['gain_ii']:+.3f} g_ui={avg_train['gain_ui']:+.3f} "
                f"n_it={avg_train['n_item_text']:.1f}(chance={1.0 / max(avg_train['n_item_text'], 1.0):.4f}) "
                f"n_ii={avg_train['n_item_item']:.1f} n_ui={avg_train['n_user_item']:.1f} | "
                f"Val Loss={val_logs['loss']:.4f} "
                f"L_itc={val_logs['L_itc']:.4f} L_itm={val_logs['L_itm']:.4f} "
                f"L_itg={val_logs['L_itg']:.4f} L_ii={val_logs['L_ii']:.4f} "
                f"L_ui={val_logs['L_ui']:.4f} L_uic={val_logs['L_uic']:.4f} "
                f"L_llm={val_logs['L_llm']:.4f} "
                f"ITC@1={val_logs['itc_top1']:.4f} ITM_acc={val_logs['itm_acc']:.4f} "
                f"ITG_acc={val_logs['itg_acc']:.4f} ITG_title={val_logs['itg_title_acc']:.4f} "
                f"II@1={val_logs['ii_top1']:.4f} "
                f"UI@1={val_logs['ui_top1']:.4f} UIC_acc={val_logs['uic_acc']:.4f} "
                f"LLM@1={val_logs['llm_top1']:.4f} "
                f"g_itc={val_logs['gain_itc']:+.3f} g_llm={val_logs['gain_llm']:+.3f} "
                f"g_ii={val_logs['gain_ii']:+.3f} g_ui={val_logs['gain_ui']:+.3f} "
                f"n_it={val_logs['n_item_text']:.1f}(chance={1.0 / max(val_logs['n_item_text'], 1.0):.4f}) "
                f"n_ii={val_logs['n_item_item']:.1f} n_ui={val_logs['n_user_item']:.1f} "
                f"L_repr={val_logs['L_repr']:.4f} "
                + (
                    f"g_repr={metrics['val_gain_repr']:+.4f}"
                    f"(g_llm_off={metrics.get(GAIN_REPR_ALIGN_KEY, float('nan')):+.3f}"
                    f"+{select_collab_weight:g}*[g_ii{metrics['val_gain_ii']:+.3f}"
                    f" g_ui{metrics['val_gain_ui']:+.3f}]) "
                    if "val_gain_repr" in metrics
                    else ""
                )
                + f"[SELECT] {selection_metric}({selection_mode})="
                f"{float(raw_selection_value):+.4f}{ema_note} | "
                f"w_itc={cfg.w_itc:.3f} w_itm={cfg.w_itm:.3f} w_itg={cfg.w_itg:.3f} "
                f"w_ii={cfg.w_ii:.3f} w_ui={w_ui:.3f} w_llm={w_llm:.3f} "
                f"tau_itc={cfg.tau_itc:.3f} tau_ii={cfg.tau_ii:.3f} tau_ui={tau_ui:.3f} "
                f"tau_llm={tau_llm:.3f} llm_emb_norm={llm_emb_normalize}"
            )

            # Collapse guard, watched on the SAME quantity that drives selection
            # (``collapse_metric`` defaults to selection_metric). Keeping the two
            # in sync matters: with selection on the sem_off gain, a guard still
            # reading the sem_on gain would happily let a run continue where the
            # semantic shortcut scores +3.0 while the CF path sits at -0.7.
            if collapse_patience > 0 and (epoch + 1) >= collapse_warmup:
                collapse_value = float(metrics[collapse_metric])
                collapse_counter = (
                    collapse_counter + 1 if collapse_value < min_collapse_gain else 0
                )

                if collapse_counter >= collapse_patience:
                    log_step(
                        "ABORTED - item-text objectives collapsed",
                        f"epoch={epoch + 1}, {collapse_metric}={collapse_value:+.3f} "
                        f"< {min_collapse_gain} for {collapse_counter} consecutive epochs "
                        f"(0.0 = chance). If this is the sem_off variant, the CF path "
                        f"has learned nothing and the objectives are being solved by "
                        f"reading the semantic bank — compare the [DIAG] sem_on vs "
                        f"sem_off lines and raise sem_source_dropout. Otherwise check "
                        f"[DIAG] query geometry. No weights exported.",
                    )
                    aborted = True
                    break

            improved = stopper.update(metrics)

            if improved:
                _save_checkpoint(
                    best_checkpoint_path,
                    model,
                    opt,
                    epoch + 1,
                    val_logs,
                )
                log_step("Saved new best checkpoint", f"epoch={epoch + 1}, path={best_checkpoint_path}")
            else:
                best_epoch = stopper.best_full_metric["epoch"] if stopper.best_full_metric is not None else "n/a"
                best_selected = stopper.best_metric_val
                log_step(
                    "No validation improvement",
                    f"counter={stopper.counter}, best_epoch={best_epoch}, "
                    f"best_{selection_metric}={best_selected:.4f}",
                )

            if stopper.should_stop:
                best_epoch = stopper.best_full_metric["epoch"] if stopper.best_full_metric is not None else "n/a"
                best_selected = stopper.best_metric_val
                log_step(
                    "Early stopping triggered",
                    f"epoch={epoch + 1}, best_epoch={best_epoch}, "
                    f"best_{selection_metric}={best_selected:.4f}",
                )
                break

    best_checkpoint = None
    if stopper.best_full_metric is not None and os.path.exists(best_checkpoint_path):
        best_checkpoint = _load_checkpoint(best_checkpoint_path, model)
        log_step(
            "Loaded best checkpoint",
            (
                f"epoch={best_checkpoint['epoch']}, val_loss={best_checkpoint['val_loss']:.4f}, "
                f"itc_top1={best_checkpoint.get('val_itc_top1', 0.0):.4f}, "
                f"itm_acc={best_checkpoint.get('val_itm_acc', 0.0):.4f}, "
                f"itg_acc={best_checkpoint.get('val_itg_acc', 0.0):.4f}, "
                f"ii_top1={best_checkpoint.get('val_ii_top1', 0.0):.4f}"
            ),
        )

    # Final Test
    log_step("Evaluating on Test Set")
    test_logs = evaluate_loss(
        model,
        test_loader,
        w_itc=cfg.w_itc,
        w_itm=cfg.w_itm,
        w_itg=cfg.w_itg,
        w_ii=cfg.w_ii,
        w_ui=w_ui,
        w_llm=w_llm,
        tau_itc=cfg.tau_itc,
        tau_ii=cfg.tau_ii,
        tau_ui=tau_ui,
        tau_llm=tau_llm,
        ui_condition_on_item=ui_condition_on_item,
        w_ui_cond=w_ui_cond,
        tau_ui_cond=tau_ui_cond,
        ui_cond_distill_mf=ui_cond_distill_mf,
        ui_cond_neg=ui_cond_neg,
        selection_weights=selection_weights,
    )
    log_step(
        "Test results",
        (
            f"loss={test_logs['loss']:.4f}, l_itc={test_logs['L_itc']:.4f}, "
            f"l_itm={test_logs['L_itm']:.4f}, l_itg={test_logs['L_itg']:.4f}, "
            f"l_ii={test_logs['L_ii']:.4f}, l_llm={test_logs['L_llm']:.4f}, "
            f"itc@1={test_logs['itc_top1']:.4f}, "
            f"itm_acc={test_logs['itm_acc']:.4f}, itg_acc={test_logs['itg_acc']:.4f}, "
            f"itg_title={test_logs['itg_title_acc']:.4f}, "
            f"ii@1={test_logs['ii_top1']:.4f}, llm@1={test_logs['llm_top1']:.4f}, "
            f"n_it={test_logs['n_item_text']:.1f}"
        ),
    )

    if aborted:
        log_step(
            "Skipped best QFormer export"
        )
    elif best_checkpoint is not None:
        best_qformer_path = os.path.join(outdir, cfg.best_qformer_weights_name)
        torch.save(model.qformer.state_dict(), best_qformer_path)
        log_step(
            "Exported best QFormer weights for stage2",
            f"path={best_qformer_path}, epoch={best_checkpoint['epoch']}, val_loss={best_checkpoint['val_loss']:.4f}",
        )
        if getattr(model, "has_llm_align", False):
            best_align_proj_path = os.path.join(
                outdir, cfg.get("best_align_proj_weights_name", "qformer_stage1_best_align_proj.pth")
            )
            torch.save(model.llm_align_proj.state_dict(), best_align_proj_path)
            log_step(
                "Exported aligned projection for stage2 warm-start",
                f"path={best_align_proj_path}",
            )
    else:
        log_step("Skipped best QFormer export", "No best checkpoint was selected during training")
    
    return model


def main():
    cfg = Config(parse_args())
    stage1_cfg = cfg.run_cfg.get("qformer_stage1")
    if stage1_cfg is None:
        raise KeyError("Missing 'run.qformer_stage1' section in configuration.")

    required_keys = [
        "batch_size",
        "num_workers",
        "embedding_size",
        "user_num",
        "item_num",
        "num_queries",
        "num_heads",
        "num_layers",
        "qformer_output_dim",
        "lr",
        "w_itc",
        "w_itm",
        "w_itg",
        "w_ii",
        "tau_itc",
        "tau_ii",
        "weight_decay",
        "debug_batch",
        "debug_batch_max_steps",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "best_checkpoint_name",
        "best_qformer_weights_name",
        "log_epoch",
        "epoch",
        "pretrained_rec_path",
        "freeze_rec",
        "output_dir",
        "seed",
    ]
    missing_keys = [key for key in required_keys if key not in stage1_cfg]
    if missing_keys:
        raise KeyError(
            "Missing required keys in 'run.qformer_stage1': " + ", ".join(missing_keys)
        )

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    stage1_cfg.data_dir = cfg.datasets_cfg[first_dataset_key].path
    # The uAUC probe reuses MovieOODDataset, which reads min_positive_history /
    # max_history_length / warm_definition off the DATASET config, so the whole
    # node has to travel — data_dir alone is not enough.
    stage1_cfg.dataset_cfg = cfg.datasets_cfg[first_dataset_key]

    train_qformer_stage1_representation(stage1_cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Former stage 1 representation model")
    parser.add_argument(
        "--cfg-path",
        default="configs/config.yaml",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override config settings in key=value format.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
