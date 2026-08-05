"""Why do g_ii / g_ui sit near chance in Stage 1?

Stage 1's own numbers cannot separate the two candidate causes, because both
show up as "gain ~0.1 nats":

    (a) the co-watch / user-item signal is NOT IN the MF embeddings any more
        (a real risk since the MF is now trained with the SeLLa Step-2
        semantic InfoNCE, which competes for the same capacity), or
    (b) the signal IS there and the Q-Former destroys it — the output is
        anisotropic to the point of being constant (Stage 1 logs
        offdiag_cos_raw ~0.9998; a Stage-2 CF-only run measured
        q_pair_cos = 1.0000 exactly).

Two measurements here, both CPU-only and training-free:

A) ``proj_cf`` constant-vs-varying ratio. ``proj_cf`` is Linear(d_cf, d_model)
   WITH bias, so its output is ``W e_i + b``: a per-item part plus a constant.
   If ``||W e_i||`` is tiny next to ``||b||``, every item enters cross-attention
   as nearly the same vector and no downstream loss can tell them apart. This
   localises cause (b) to one 768x256 matrix instead of "the Q-Former".

B) The SAME InfoNCE Stage 1 runs for L_ii / L_ui, computed directly on RAW MF
   embeddings (mean-pool over a single vector is the identity, so this is the
   pooled-pair form with the encoder removed). Reported as ``gain = ln(n) - CE``
   at the same tau and the same in-batch n, so the number is directly
   comparable to the ``g_ii`` / ``g_ui`` in the epoch line. This is a CEILING:
   no encoder reading these embeddings can beat it by much.

       raw gain >> Stage 1 g_ii   -> signal exists, Q-Former loses it (cause b)
       raw gain ~= Stage 1 g_ii   -> Q-Former is a faithful pass-through; the
                                     embeddings themselves are the limit (a)
       raw gain ~ 0               -> co-watch is not linearly readable from the
                                     MF geometry at all; fix the MF first

   Each is also reported CENTERED (batch mean removed before the cosine),
   because config documents the residual as near low-rank and centering was
   only ever measured at tau=0.07, where bipolar cosines x 1/tau blow the
   logsumexp up. If ``centered`` is much better here, ``pair_logit_center=True``
   with a RAISED tau is the cheap fix — that combination has never been run.

C) Not implemented here: ``probe_cf_channel.py`` already answers "does the
   whole channel out-rank MF". Run it alongside this (see Usage).

Usage
-----
    # A + B on the current (aligned) MF:
    python -m sigllm.pipelines.multimodal.diagnose_collab_collapse \\
        --mf-ckpt /content/SigLLM/ckpt/mf/mf_model.pth \\
        --qformer-ckpt /content/SigLLM/ckpt/qformer_stage1/qformer_stage1_best_qformer.pth \\
        --qformer-pkl /content/SigLLM/data/processed/ml-1m/train_qformer_ood2.pkl

    # B on aligned VS plain MF — settles whether the SeLLa alignment ate the
    # collaborative geometry (train a plain one with
    # run.rec_baseline.item_llm_emb_path=null to a separate --save-file):
    ... --mf-ckpt <aligned>.pth --mf-ckpt-baseline <plain>.pth

    # the third measurement:
    python -m sigllm.pipelines.multimodal.probe_cf_channel \\
        --cfg-path configs/config.yaml --qformer-ckpt <same qformer ckpt>
"""

import argparse
import math
import pickle

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose Stage-1 collaborative collapse (g_ii / g_ui)")
    p.add_argument("--mf-ckpt", required=True, help="MF checkpoint (the one Stage 1 was trained on).")
    p.add_argument(
        "--mf-ckpt-baseline",
        default=None,
        help="Optional second MF checkpoint to compare against (e.g. a plain "
             "BCE-only MF, to test whether the SeLLa alignment cost co-watch signal).",
    )
    p.add_argument(
        "--qformer-ckpt",
        default=None,
        help="Stage-1 Q-Former weights, for measurement A. Skipped when omitted.",
    )
    p.add_argument(
        "--qformer-pkl",
        default=None,
        help="Built Q-Former dataset pkl holding the item_item / user_item pairs "
             "(measurement B). Skipped when omitted.",
    )
    p.add_argument("--tau-ii", type=float, default=0.07, help="Must match run.qformer_stage1.tau_ii.")
    p.add_argument("--tau-ui", type=float, default=0.07, help="Must match run.qformer_stage1.tau_ui.")
    p.add_argument(
        "--n-ii", type=int, default=610,
        help="In-batch candidates for the item_item InfoNCE. Default matches the "
             "n_ii printed in the Stage-1 epoch line so the gain is comparable.",
    )
    p.add_argument("--n-ui", type=int, default=339, help="Same, for user_item (see n_ui).")
    p.add_argument("--trials", type=int, default=50, help="Batches to average each InfoNCE over.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_state(path):
    blob = torch.load(path, map_location="cpu")
    for key in ("model", "state_dict"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            return blob[key]
    return blob


def _find(state, suffix):
    """Fetch a tensor by key suffix, so wrapper prefixes (``qformer.``,
    ``module.``) don't matter."""
    for key, value in state.items():
        if isinstance(key, str) and key.endswith(suffix) and isinstance(value, torch.Tensor):
            return value
    return None


def _offdiag_cos(x):
    """Mean pairwise cosine between DIFFERENT rows. ~1.0 means every row points
    the same way, i.e. the representation carries no per-row information."""
    if x.size(0) < 2:
        return float("nan")
    xn = F.normalize(x.float(), dim=-1)
    sim = xn @ xn.T
    n = sim.size(0)
    off = sim.sum() - sim.diagonal().sum()
    return float(off / (n * (n - 1)))


def measure_proj_cf(mf_state, qformer_state):
    """A) How much of proj_cf's output varies with the item at all."""
    item_emb = _find(mf_state, "item_embedding.weight")
    weight = _find(qformer_state, "proj_cf.weight")
    bias = _find(qformer_state, "proj_cf.bias")
    if item_emb is None or weight is None:
        print("  [skip] proj_cf.weight or item_embedding.weight not found")
        return

    # Row 0 is the padding item and is not a real item; excluding it keeps the
    # statistics from being dragged by a vector no forward pass ever means.
    emb = item_emb[1:].float()
    varying = emb @ weight.float().T                     # W e_i  (per item)
    const = bias.float() if bias is not None else torch.zeros(weight.size(0))
    out = varying + const

    var_norm = varying.norm(dim=-1).mean().item()
    const_norm = const.norm().item()
    emb_norm = emb.norm(dim=-1).mean().item()
    residual = out - out.mean(dim=0, keepdim=True)

    print(f"  mean ||e_i||                    = {emb_norm:.4f}   (MF item embedding scale)")
    print(f"  mean ||W e_i||  (item-varying)  = {var_norm:.4f}")
    print(f"  ||b||           (constant)      = {const_norm:.4f}")
    print(f"  varying / constant              = {var_norm / (const_norm + 1e-12):.4f}")
    print(f"  mean ||h - mean(h)|| / ||h||    = "
          f"{(residual.norm(dim=-1).mean() / out.norm(dim=-1).mean()).item():.4f}")
    print(f"  offdiag_cos(proj_cf out)  raw   = {_offdiag_cos(out):.4f}")
    print(f"                            centered = {_offdiag_cos(residual):.4f}")
    print("  READ: varying/constant << 0.1 (or offdiag_cos raw > 0.99) means every")
    print("        item enters cross-attention as nearly the same vector — the")
    print("        collapse starts HERE, before any Q-Former layer.")


def _infonce_gain(
    left_ids, right_ids, left_table, right_table, tau, n, trials, generator, center
):
    """Stage-1's pooled-pair InfoNCE on already-pooled vectors.

    Mirrors ``QRecInstructAlignmentModel._pooled_pair_logits``: optional batch
    centering, L2 normalise, cosine / tau, cross-entropy against the diagonal.
    Returns ``(gain, top1)`` with ``gain = ln(n) - CE`` — 0 is chance, higher
    is better, same units as the ``g_ii`` / ``g_ui`` in the Stage-1 log.

    Rows are gathered PER TRIAL rather than up front: ML1M carries on the order
    of 1e5-1e6 user_item pairs, so materialising [pairs, d_cf] for both sides
    would cost GBs to then only ever read ``n`` rows of it at a time.
    ``randperm`` (not ``randint``) keeps a batch duplicate-free — a repeated
    row would be a false negative sitting at cosine exactly 1.
    """
    total = left_ids.numel()
    n = min(n, total)
    if n < 4:
        return float("nan"), float("nan")

    labels = torch.arange(n)
    ce_sum, top1_sum = 0.0, 0.0
    for _ in range(trials):
        idx = torch.randperm(total, generator=generator)[:n]
        a = left_table[left_ids[idx]].float()
        b = right_table[right_ids[idx]].float()
        if center:
            a = a - a.mean(dim=0, keepdim=True)
            b = b - b.mean(dim=0, keepdim=True)
        logits = (F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T) / tau
        ce_sum += F.cross_entropy(logits, labels).item()
        top1_sum += (logits.argmax(dim=1) == labels).float().mean().item()
    return math.log(n) - ce_sum / trials, top1_sum / trials


def measure_raw_signal(mf_state, samples, args, label):
    """B) L_ii / L_ui computed on the raw MF embeddings — the ceiling."""
    item_emb = _find(mf_state, "item_embedding.weight")
    user_emb = _find(mf_state, "user_embedding.weight")
    if item_emb is None:
        print(f"  [skip:{label}] item_embedding.weight not found")
        return

    generator = torch.Generator().manual_seed(args.seed)

    ii = [(int(s["i_left"]), int(s["i_right"])) for s in samples
          if s.get("sample_type") == "item_item"]
    ui = [(int(s["u"]), int(s["i_left"])) for s in samples
          if s.get("sample_type") == "user_item"]

    def report(name, left_ids, right_ids, left_table, right_table, base_tau, n):
        """Sweep (center, tau), and read ``top1`` — NOT ``gain`` — as the verdict.

        ``gain`` alone cannot diagnose anything here. Simulated at Stage 1's
        measured ``offdiag_cos_raw = 0.9998``, raw cosine gives gain ~ +0.003
        in BOTH of these regimes:

            residual rank    raw gain    raw top1    centered gain / top1
            256 (isotropic)    +0.003      1.000       +6.41 / 1.000
            3   (low-rank)     +0.003      0.156       +2.14 / 0.077

        So a gain pinned at 0 says only "the margin is dead", never "the signal
        is absent". ``top1`` is what separates the two: it ignores the margin
        and reads the ORDERING. High top1 with gain ~ 0 means the information is
        fully present and merely invisible to the loss.

        Note what the same simulation says about the two tempting fixes, since
        both are weaker than they look:
          - Centering is NOT a free win. On the low-rank residual it made top1
            WORSE (0.156 -> 0.077) — the mechanism behind the real centered run
            config records at ``L_ii = 24.45`` (gain -18), i.e. the true pair
            actively scoring near the BOTTOM.
          - Raising tau does not help ``gain``: in every simulated regime gain
            falls monotonically as tau rises (centered, isotropic: +6.41 at
            0.07 down to +0.47 at 2.0). Higher tau only rescues the case where
            sharp logits invert the ranking; the sweep is here to find out
            whether that is the case at all, not on the assumption that it is.
        """
        taus = sorted({base_tau, 0.2, 0.5, 1.0, 2.0})
        ids = (left_ids, right_ids, left_table, right_table)
        for center in (False, True):
            tag = "centered" if center else "raw     "
            row = []
            for tau in taus:
                gain, top1 = _infonce_gain(
                    *ids, tau, n, args.trials, generator, center
                )
                row.append(f"tau={tau:<4g} gain={gain:+7.3f} top1={top1:.4f}")
            print(f"  {name} {tag} n={min(n, left_ids.numel())} pairs={left_ids.numel()}")
            for entry in row:
                print(f"      {entry}")

    if ii:
        report(
            "item_item",
            torch.tensor([a for a, _ in ii], dtype=torch.long),
            torch.tensor([b for _, b in ii], dtype=torch.long),
            item_emb, item_emb, args.tau_ii, args.n_ii,
        )
    else:
        print("  [skip] no item_item samples in the pkl")

    if ui and user_emb is not None:
        report(
            "user_item",
            torch.tensor([u for u, _ in ui], dtype=torch.long),
            torch.tensor([i for _, i in ui], dtype=torch.long),
            user_emb, item_emb, args.tau_ui, args.n_ui,
        )
    else:
        print("  [skip] no user_item samples (or no user_embedding) in the pkl")


def main():
    args = parse_args()
    mf_state = _load_state(args.mf_ckpt)

    has_align = any(
        isinstance(k, str) and k.startswith(("item_embedding_llm.", "trans_1.", "trans_2."))
        for k in mf_state
    )
    print(f"\n=== MF: {args.mf_ckpt}")
    print(f"  SeLLa Step-2 alignment keys present: {has_align}")

    if args.qformer_ckpt:
        print("\n=== A) proj_cf: item-varying vs constant")
        measure_proj_cf(mf_state, _load_state(args.qformer_ckpt))

    if args.qformer_pkl:
        with open(args.qformer_pkl, "rb") as handle:
            blob = pickle.load(handle)
        samples = blob["samples"] if isinstance(blob, dict) else blob

        print("\n=== B) raw-MF InfoNCE (ceiling; compare to Stage-1 g_ii / g_ui)")
        print(f"--- MF under test: {args.mf_ckpt}")
        measure_raw_signal(mf_state, samples, args, "test")

        if args.mf_ckpt_baseline:
            print(f"--- MF baseline:   {args.mf_ckpt_baseline}")
            measure_raw_signal(_load_state(args.mf_ckpt_baseline), samples, args, "baseline")
            print("  READ: baseline gain >> test gain means the SeLLa alignment InfoNCE")
            print("        traded away the collaborative geometry L_ii needs — lower")
            print("        run.rec_baseline.align_weight and retrain the MF.")
    print()


if __name__ == "__main__":
    main()
