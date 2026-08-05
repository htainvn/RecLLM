"""Measure what the soft-token channel carries — WITHOUT the LLM.

Stage 1's own metrics (g_itc / g_llm / g_ii / g_ui, offdiag_cos, proj_sem norm)
are all *internal*: every one of them has already been observed to keep
improving while something it did not watch collapsed (see the selection_metric
comment block in configs/config.yaml). None of them answers the only question
Stage 3 cares about:

    does ``mean_pool(qformer(hist)) . mean_pool(qformer(target))`` rank a user's
    candidates better than the frozen MF it was built from?

This script answers exactly that, on the SAME rows, with the SAME history
padding/cap as Stage 3 (it reuses MovieOODDataset), and compares against the MF
dot product — the 0.7077 uAUC number the whole pipeline has to beat.

Read the result as a CEILING on Stage 3. The LLM path can only lose information
relative to this: it reads the same two vectors through a linear projection into
a frozen 7B's embedding space. So:

    probe uAUC >> MF uAUC   Stage 1/2 added real signal; a flat Stage 3 means
                            the LLM is not reading the channel -> work on the
                            injection / LoRA / prompt side.
    probe uAUC ~= MF uAUC   the Q-Former is a (lossy) pass-through. Stage 3 can
                            at best re-derive MF, which is what a 0.70 uAUC
                            looks like. Fix Stage 1 before tuning Stage 3.
    probe uAUC ~ 0.5        Stage 1 DESTROYED the CF geometry. Nothing
                            downstream can recover it; every Stage-3 experiment
                            is measuring prompt text.

The ``centered`` variant of each score subtracts the batch-mean vector before
the cosine. It exists because the Q-Former output is extremely anisotropic
(Stage 1 logs offdiag_cos_raw ~0.9998, i.e. the item-specific part is ~1.4% of
the token norm). If ``centered`` is much better than ``raw``, the discriminative
signal IS there but is buried under a shared mean direction — and since
``llm_proj`` is Linear+LayerNorm, that shared direction survives into the LLM
and the injected tokens are near-identical across items. That is a fixable
Stage-3 bug, not a Stage-1 failure, so the two numbers must be read together.

Usage
-----
    python -m sigllm.pipelines.multimodal.probe_cf_channel \\
        --cfg-path configs/config.yaml \\
        --qformer-ckpt /content/SigLLM/ckpt/qformer_stage1/qformer_stage1_best_qformer.pth \\
        --splits valid_ood2.pkl test_ood2.pkl

Run it once per Q-Former checkpoint (Stage 1 best, Stage 2 best, and the
Stage-3 checkpoint's Q-Former) to see where the channel is gained or lost.
"""

import argparse
import os
from typing import Optional

import numpy as np
import omegaconf
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from sigllm.common.config import Config
from sigllm.common.logging_utils import NotebookLogger
from sigllm.datasets.movie.movie_ood_dataset import MovieOODDataset
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.rec.matrix_factorization import MatrixFactorization
from sigllm.pipelines.rec.train_rec_baseline import calculate_user_auc

LOGGER = NotebookLogger.rich_logger("sigllm.probe_cf_channel")

# Must match QRecLLM.QFORMER_ITEM_INSTRUCTIONS[0] — that is the fixed string
# Stage 3 uses at eval, so the probe has to encode with the same conditioning.
EVAL_INSTRUCTION = "Represent this movie for recommendation using its title and genres."


def log_step(title: str, detail: Optional[str] = None) -> None:
    LOGGER.info(title if detail is None else f"{title} | {detail}")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def _offdiag_cos(x: torch.Tensor) -> float:
    """Mean pairwise cosine over distinct rows — the anisotropy figure Stage 1
    logs. 1.0 means every input encodes to the same direction."""
    x = x.float()
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
    sim = x @ x.T
    n = sim.size(0)
    return float((sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))


def build_qformer(cfg, device, d_sem):
    qcfg = cfg.model_cfg.qformer_config
    d_model = int(qcfg.get("qformer_d_model", 768))
    return HFQFormerAdapter(
        d_cf=int(cfg.model_cfg.rec_config.embedding_size),
        d_model=d_model,
        num_queries=int(qcfg.num_queries),
        num_heads=int(qcfg.num_heads),
        num_layers=int(qcfg.num_layers),
        output_dim=int(qcfg.get("qformer_output_dim") or d_model),
        qformer_text_model_name=qcfg.get("qformer_text_model_name", "bert-base-uncased"),
        max_instruction_length=int(qcfg.get("max_instruction_length", 48)),
        init_from_pretrained_text=False,
        user_conditioned=bool(qcfg.get("user_conditioned", False)),
        # These change the adapter's MODULES, so omitting them made the
        # standalone probe silently measure a different model than the one being
        # trained: output_residual=False turns _apply_output_residual into a
        # no-op, and the strict=False load would just report the weights as
        # missing. dropout is inert at eval but kept for symmetry.
        dropout=float(qcfg.get("qformer_dropout", 0.0)),
        candidate_fusion=bool(qcfg.get("candidate_fusion", False)),
        item_residual=bool(qcfg.get("item_residual", False)),
        output_residual=bool(qcfg.get("output_residual", False)),
        d_user=int(cfg.model_cfg.rec_config.embedding_size),
        d_sem=d_sem,
    ).to(device)


def load_qformer_ckpt(qformer, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith(("qformer.", "q", "out_proj")) for k in state
    ):
        state = state["model"]
    if isinstance(state, dict) and any(k.startswith("qformer.") for k in state):
        state = {k.replace("qformer.", "", 1): v for k, v in state.items()}
    msg = qformer.load_state_dict(state, strict=False)
    if msg.missing_keys:
        log_step("Q-Former ckpt MISSING keys (left at init)", ", ".join(msg.missing_keys))
    if msg.unexpected_keys:
        log_step("Q-Former ckpt unexpected keys (ignored)", ", ".join(msg.unexpected_keys))
    log_step("Loaded Q-Former", path)


def collate(rows):
    return {
        "UserID": torch.as_tensor([int(r["UserID"]) for r in rows], dtype=torch.long),
        "TargetItemID": torch.as_tensor([int(r["TargetItemID"]) for r in rows], dtype=torch.long),
        "InteractedItemIDs_pad": torch.as_tensor(
            np.stack([r["InteractedItemIDs_pad"] for r in rows]), dtype=torch.long
        ),
        "label": torch.as_tensor([int(r["label"]) for r in rows], dtype=torch.long),
    }


@torch.no_grad()
def encode_split(qformer, mf, sem_bank, loader, device):
    """Pooled Q-Former vectors + MF vectors for every row of a split."""
    out = {k: [] for k in ("profile_q", "target_q", "mf_user", "mf_item", "mf_hist", "user", "label")}

    for batch in loader:
        uid = batch["UserID"].to(device)
        iid = batch["TargetItemID"].to(device)
        hist = batch["InteractedItemIDs_pad"].to(device)
        b = uid.size(0)
        instructions = [EVAL_INSTRUCTION] * b

        target_cf = mf.item_encoder(iid)                    # [B, d]
        hist_cf = mf.item_encoder(hist)                     # [B, L, d]
        hist_mask = hist != mf.padding_index                # [B, L]

        target_sem = sem_bank[iid] if sem_bank is not None else None
        hist_sem = sem_bank[hist] if sem_bank is not None else None

        # EXACTLY the two encoder calls QRecLLM.encode_rec_features_to_llm_v2
        # makes, including the candidate conditioning of the history pooling.
        target_q = qformer(target_cf, instructions, sem_vec=target_sem)
        cond = target_cf if qformer.user_conditioned else None
        profile_q = qformer(
            hist_cf, instructions, user_cf=cond, source_mask=hist_mask, sem_vec=hist_sem,
            # e_u, matching QRecLLM: the output residual's fallback readout must
            # be cos(e_u, e_t), not cos(mask_mean(e_hist), e_t) — the latter is
            # measured BELOW the MF dot product this probe compares against.
            residual_vec=mf.user_encoder(uid),
        )

        # Mean over the Q queries — the same pooling Stage 1's L_ui scores and
        # the align_rank_loss uses. Stage 3 injects all Q tokens separately, so
        # this pooling is if anything PESSIMISTIC about the channel.
        out["profile_q"].append(profile_q.float().mean(dim=1).cpu())
        out["target_q"].append(target_q.float().mean(dim=1).cpu())

        # MF references, on the identical rows.
        mask_f = hist_mask.float().unsqueeze(-1)
        out["mf_user"].append(mf.user_encoder(uid).float().cpu())
        out["mf_item"].append(target_cf.float().cpu())
        out["mf_hist"].append(
            ((hist_cf.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)).cpu()
        )
        out["user"].append(batch["UserID"].clone())
        out["label"].append(batch["label"].clone())

    return {k: torch.cat(v, dim=0) for k, v in out.items()}


def report(name, users, labels, scores):
    users = np.asarray(users)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    auc = roc_auc_score(labels, scores)
    uauc, _, _ = calculate_user_auc(users, scores, labels)
    log_step(f"  {name:<26}", f"AUC={auc:.4f}  uAUC={uauc:.4f}")
    return auc, uauc


def probe_split(split, enc):
    users = enc["user"].numpy()
    labels = enc["label"].numpy()

    log_step(f"[{split}] geometry", "")
    # Anisotropy of what Stage 3 actually injects. >0.99 means the LLM sees
    # near-identical soft tokens for every candidate.
    sample = enc["target_q"][: min(1024, enc["target_q"].size(0))]
    log_step(
        f"  offdiag_cos(target_q)",
        f"raw={_offdiag_cos(sample):.4f}  "
        f"centered={_offdiag_cos(sample - sample.mean(0, keepdim=True)):.4f}  "
        f"(raw ~1.0 => every candidate encodes to the same direction; the "
        f"item-specific part is ~{100 * (1 - _offdiag_cos(sample)) ** 0.5:.1f}% of the norm)",
    )

    log_step(f"[{split}] scores", "")
    results = {}

    # --- MF references: the numbers to beat -------------------------------
    results["mf_dot"] = report(
        "MF dot (baseline)", users, labels,
        (enc["mf_user"] * enc["mf_item"]).sum(-1).numpy(),
    )
    results["mf_hist_cos"] = report(
        "MF mean-history cos", users, labels,
        _cosine(enc["mf_hist"], enc["mf_item"]).numpy(),
    )

    # --- The Q-Former channel ---------------------------------------------
    results["qformer_cos"] = report(
        "Q-Former cos (raw)", users, labels,
        _cosine(enc["profile_q"], enc["target_q"]).numpy(),
    )
    pc = enc["profile_q"] - enc["profile_q"].mean(0, keepdim=True)
    tc = enc["target_q"] - enc["target_q"].mean(0, keepdim=True)
    results["qformer_cos_centered"] = report(
        "Q-Former cos (centered)", users, labels, _cosine(pc, tc).numpy(),
    )
    return results


def build_probe_loader(dataset_cfg, filename, batch_size=64):
    """Loader over a MovieOOD split, shaped for ``encode_split``.

    Exposed so Stage 1 can build this once at setup and probe DURING training
    instead of only after the fact.
    """
    return DataLoader(
        MovieOODDataset(dataset_cfg, filename=filename),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )


@torch.no_grad()
def probe_metrics(qformer, mf, sem_bank, loader, device, prefix="val_probe"):
    """The probe as a flat metrics dict, for in-training use.

    WHY THIS IS THE METRIC THAT MATTERS. Every metric Stage 1 optimises is a
    CROSS-USER retrieval pretext (L_ii, L_ui: pick this user's item out of other
    users' items), but the number the project is judged on is uAUC — WITHIN-user
    ranking of one user's own candidates. Those are different tasks, and the gap
    is not academic: measured on the val pkl, the parameter-free readout over
    the frozen MF embeddings scores ~+0.05 nats on the retrieval pretext (barely
    above chance) while that same MF ranks held-out candidates at 0.708 uAUC.
    A pretext can therefore look degenerate while the underlying signal is fine,
    and conversely it can be driven up by memorisation while uAUC does not move.

    Returns ``{prefix}_uauc`` / ``_auc`` (Q-Former cosine, the channel Stage 3
    consumes), the same centered, the MF references on the IDENTICAL rows, and
    ``{prefix}_gain`` = Q-Former uAUC - MF-dot uAUC. That last one is the honest
    summary: positive means Stage 1 added something the MF it was built from did
    not already have.
    """
    was_training = qformer.training
    qformer.eval()
    try:
        enc = encode_split(qformer, mf, sem_bank, loader, device)
        # SEM OFF pass. The semantic source is the ONLY component that can carry
        # information the MF does not already have, so it is the only thing that
        # can make the channel beat MF rather than merely re-derive it. Nothing
        # else measures this on the target task: Stage 1's sem_on/sem_off
        # diagnostic covers the caption losses, and those are identical by
        # construction while sem_for_text_losses is False.
        #   sem_gain ~ 0     -> the semantic bank adds nothing to within-user
        #                       ranking; the channel is a CF pass-through and
        #                       cannot beat MF by design, not by tuning.
        #   sem_gain >> 0    -> semantics does carry ranking signal, so a channel
        #                       stuck at MF level is losing it somewhere later.
        enc_sem_off = (
            encode_split(qformer, mf, None, loader, device)
            if sem_bank is not None
            else None
        )
    finally:
        if was_training:
            qformer.train()

    users = enc["user"].numpy()
    labels = enc["label"].numpy()

    def uauc(scores):
        value, _, _ = calculate_user_auc(users, np.asarray(scores, dtype=np.float64), labels)
        return float(value)

    def auc(scores):
        return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))

    qf_raw = _cosine(enc["profile_q"], enc["target_q"]).numpy()
    pc = enc["profile_q"] - enc["profile_q"].mean(0, keepdim=True)
    tc = enc["target_q"] - enc["target_q"].mean(0, keepdim=True)
    qf_cen = _cosine(pc, tc).numpy()
    mf_dot = (enc["mf_user"] * enc["mf_item"]).sum(-1).numpy()

    # DOT as well as cosine. The cosine discards ||target_q||, and that norm
    # carries real ranking signal here: measured cos(e_u, e_t) = 0.583 against
    # e_u . e_t = 0.6437 on the same rows. A change that lands in the norm is
    # invisible to the cosine, so reporting only cosine can read as "no progress"
    # (or as a regression) while the dot readout improves.
    qf_dot = (enc["profile_q"] * enc["target_q"]).sum(-1).numpy()
    out = {
        f"{prefix}_uauc_dot": uauc(qf_dot),
        f"{prefix}_uauc": uauc(qf_raw),
        f"{prefix}_auc": auc(qf_raw),
        f"{prefix}_uauc_centered": uauc(qf_cen),
        f"{prefix}_mf_dot_uauc": uauc(mf_dot),
        f"{prefix}_mf_hist_cos_uauc": uauc(_cosine(enc["mf_hist"], enc["mf_item"]).numpy()),
        f"{prefix}_rows": float(labels.size),
    }
    # Best of raw/centered vs MF: which of the two the loss happens to favour is
    # an implementation detail of the scoring head, not of the channel.
    # Best of the three readouts: which one the channel happens to express its
    # signal in is an implementation detail of the scoring head, not of the
    # channel's information content.
    out[f"{prefix}_gain"] = (
        max(out[f"{prefix}_uauc"], out[f"{prefix}_uauc_centered"], out[f"{prefix}_uauc_dot"])
        - out[f"{prefix}_mf_dot_uauc"]
    )

    if enc_sem_off is not None:
        pc0 = enc_sem_off["profile_q"] - enc_sem_off["profile_q"].mean(0, keepdim=True)
        tc0 = enc_sem_off["target_q"] - enc_sem_off["target_q"].mean(0, keepdim=True)
        best_off = max(
            uauc(_cosine(enc_sem_off["profile_q"], enc_sem_off["target_q"]).numpy()),
            uauc(_cosine(pc0, tc0).numpy()),
        )
        out[f"{prefix}_uauc_sem_off"] = best_off
        # How much of the channel's ranking ability comes from the semantic bank.
        out[f"{prefix}_sem_gain"] = (
            max(out[f"{prefix}_uauc"], out[f"{prefix}_uauc_centered"]) - best_off
        )
    return out


def main():
    args = parse_args()
    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=args.options))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_key = list(cfg.datasets_cfg.keys())[0]
    dcfg = cfg.datasets_cfg[data_key]
    data_dir = str(dcfg.path)

    # user_num / item_num from the data, same as the Stage-3 scripts do.
    import pandas as pd

    frames = [
        pd.read_pickle(os.path.join(data_dir, f"{s}.pkl"))
        for s in ("train_ood2", "valid_ood2", "test_ood2")
    ]
    user_num = int(max(f.uid.max() for f in frames)) + 1
    item_num = int(max(f.iid.max() for f in frames)) + 1
    cfg.model_cfg.rec_config.user_num = user_num
    cfg.model_cfg.rec_config.item_num = item_num
    log_step("Catalog", f"user_num={user_num}, item_num={item_num}")

    mf = MatrixFactorization(
        omegaconf.OmegaConf.create(
            {
                "user_num": user_num,
                "item_num": item_num,
                "embedding_size": int(cfg.model_cfg.rec_config.embedding_size),
            }
        )
    ).to(device)
    mf.load_state_dict(
        torch.load(cfg.model_cfg.rec_config.pretrained_path, map_location="cpu")
    )
    mf.eval()
    log_step("Loaded MF", str(cfg.model_cfg.rec_config.pretrained_path))

    sem_bank = None
    d_sem = None
    if bool(cfg.model_cfg.qformer_config.get("sem_source", False)):
        path = cfg.model_cfg.qformer_config.item_sem_emb_path
        blob = torch.load(path, map_location="cpu")
        bank = (blob["item_llm_emb"] if isinstance(blob, dict) else blob).float()
        sem_bank = bank.to(device)
        d_sem = int(bank.size(-1))
        covered = int((bank.norm(dim=-1) > 0).sum())
        log_step(
            "Loaded semantic source bank",
            f"{path} shape={tuple(bank.shape)} covered={covered}/{bank.size(0)}",
        )

    qformer = build_qformer(cfg, device, d_sem)
    load_qformer_ckpt(qformer, args.qformer_ckpt)
    qformer.eval()

    for split in args.splits:
        dataset = MovieOODDataset(dcfg, filename=split)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        enc = encode_split(qformer, mf, sem_bank, loader, device)
        log_step("=" * 70, "")
        log_step(f"SPLIT {split}", f"{enc['label'].numel()} rows")
        probe_split(split, enc)

    log_step("=" * 70, "")
    log_step(
        "How to read this",
        "Q-Former cos >> MF dot  -> Stage 1/2 added signal, the LLM is the "
        "bottleneck. Q-Former cos ~= MF dot -> the channel is a pass-through "
        "and 0.70 uAUC is the expected Stage-3 result. Q-Former cos ~ 0.5 -> "
        "Stage 1 destroyed the CF geometry, fix it before any Stage-3 tuning. "
        "centered >> raw -> the signal exists but is buried under a shared "
        "mean direction that llm_proj's LayerNorm does not remove.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe the CF -> soft-token channel without the LLM"
    )
    parser.add_argument("--cfg-path", default="configs/config.yaml")
    parser.add_argument(
        "--qformer-ckpt",
        required=True,
        help="Q-Former weights to probe (Stage 1 best, Stage 2 best, or a "
        "Stage-3 checkpoint — run it on each to see where the channel is lost).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid_ood2.pkl", "test_ood2.pkl"],
        help="Split filenames under datasets.<key>.path. MovieOODDataset "
        "checks the path as given, so keep the .pkl extension.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--options", nargs="+", help="Config overrides, key=value.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
