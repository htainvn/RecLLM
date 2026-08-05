# Stage 1 / CF-channel: what is established, what is not

Written after a session that added 13 mechanisms at once and could therefore
attribute none of them. This file separates measurements that held up from
changes that have never been isolated, and gives an ordered plan.

---

## 1. Established (measured, repeated, mechanism understood)

**A shared pooling function cannot beat MF on WARM users. This is structural.**
`e_u` is 256 free parameters per user, fit by SGD directly on the CTR objective
over that user's entire training history. `<UserProfile>` is a shared function
`g(e_h1…e_hL)` trained on caption/retrieval pretexts. The free parameter can
represent any aggregation, including the optimal one, so it cannot be
out-represented on users seen in training. `combined_gain = 0` on the warm probe
is therefore the CORRECT result, not a failure — it was tracked for many epochs
as if it could go positive. Do not use the warm probe to judge the Q-Former.

**The history pooling mechanically works.** `centered cos(full, last1) = 0.18`
(≈1.0 would mean collapsed attention). `k1 0.18 / k5 0.22 / k10 0.21` means
items beyond the 10 most recent carry ~69% of the profile — `max_history_length:
50` is earning its keep, do not lower it.

**The semantic bank contributes ~nothing to within-user ranking through the
Q-Former.** `sem_gain` ≈ 0 across 10+ epochs, occasionally negative. Root cause
identified: `data_preprocessing` sets `train_["not_cold"] = 1`, so EVERY training
row is warm and `proj_sem` only ever trains where CF already works. Nothing asks
semantics to carry a ranking alone. (`cf_dropout` was added for this; unverified.)

**On COLD, MF itself is near chance** — `MF_dot = 0.5591` vs chance 0.5, with the
channel at 0.5517 (the 0.007 gap is inside noise on 3120 rows). So cold is not
"the channel loses"; it is "nothing works". The headroom there is large and
unexploited, and it is the only regime with upside.

**Two config values were measurably wrong.**
- `tau_ii`/`tau_ui = 0.07` scored WORSE THAN CHANCE on clean MF embeddings
  (item_item −2.66, user_item(hist) −0.84). Optimum ≈ 0.5 / 0.2.
- `pair_logit_center: False` was rejected on evidence collected at tau=0.07,
  where RAW is also worse than chance (−1.05) — the comparison never isolated
  centering. At tau=0.2 centering is 2.7x better on gain, 2.5x on top1.

**Q-Former internal dropout was 0.0 by omission** — no call site ever passed
`dropout` to `HFQFormerAdapter`, so a 3-layer transformer trained on ~17k
user_item pairs had no regularisation at all.

**`g_ii` is at its ceiling.** Raw-MF item_item InfoNCE tops out at +0.167 nats
(top1 8.9x chance); Stage 1 reaches +0.186. No headroom — stop optimising it.

**Anisotropy improved a lot** this session: the query-output off-diagonal cosine
went from ~0.9998 to ~0.88. Which of the changes did it is not attributed.

**Unresolved after six asks: `MF_dot` on valid is 0.6437, but config references
0.7077.** If `run.rec_baseline.align_weight` degraded MF by ~6 points, then both
the baseline AND the channel's input are worse than they should be, and every
"cannot beat MF" conclusion above was measured against a weakened MF. This is
the cheapest high-leverage item outstanding.

---

## 2. Not verified (added this session, never isolated)

All ON simultaneously, so none is attributable:

| flag | value | rationale strength |
|---|---|---|
| `output_residual` | True | ONLY change with a visible probe effect: GAIN −0.107 → −0.0475 → −0.0163 across its three versions. But the improvement is a *redundant re-derivation of MF* (the residual IS an MF readout), which is why `combined_w` stays 0. |
| `item_residual` | True | untested |
| `candidate_fusion` | True | untested |
| `memory_positional` | 50 | untested; motivated (BLIP-2 adds no positional signal to the memory, so order — the one axis `e_u` lacks — was structurally unavailable) |
| `cf_dropout` | 0.3 | untested; motivated by the all-warm-training-rows finding |
| `center_soft_tokens` | True | simulated only (injected-token offdiag 0.9998 → −0.002) |
| `qformer_dropout` | 0.1 | fixes a real omission; effect unmeasured |
| `w_rank` + `rank_replay` | 0.15 / 3 | first version was buggy (scored dot, probe scored cosine); the fixed version has never run in isolation |
| `direct_id_tokens` | True | Stage 3 only; the probe does not measure it |

---

## 3. Do these in order

**Step 0 — gating check, ~1 epoch.** Read the `COLD semantic coverage` line.
If coverage is near 0%, semantics is absent for exactly the rows that need it,
`_project_sources` masks those slots out, and cold can NEVER work regardless of
training. Rebuild `item_llm_emb.pt` over the full catalog first (the distill
script logs `"N/M item ids have NO text"`). Everything else is pointless until
this is >80%.

**Step 1 — settle the MF question.** Train a plain MF and compare `MF_dot`:
```bash
python src/sigllm/pipelines/rec/train_rec_baseline.py --options \
  run.rec_baseline.item_llm_emb_path=null \
  run.rec_baseline.save_file=/content/SigLLM/ckpt/mf/mf_plain.pth
python -m sigllm.pipelines.multimodal.diagnose_collab_collapse \
  --mf-ckpt /content/SigLLM/ckpt/mf/mf_model.pth \
  --mf-ckpt-baseline /content/SigLLM/ckpt/mf/mf_plain.pth \
  --qformer-pkl /content/SigLLM/data/processed/ml-1m/valid_qformer_ood2.pkl
```
If plain MF is materially better, lower `align_weight` and retrain before any
Stage-1 work — the foundation is what everything else is measured against.

**Step 2 — get a clean baseline.** Turn every unverified flag OFF and record
cold `GAIN` / warm `dot` / `sem_gain`. Keep only the changes with independent
evidence (`tau_ii`, `tau_ui`, `pair_logit_center`, `qformer_dropout`):
```bash
--options model.qformer_config.output_residual=False \
          model.qformer_config.item_residual=False \
          model.qformer_config.candidate_fusion=False \
          model.qformer_config.memory_positional=0 \
          model.qformer_config.center_soft_tokens=False \
          run.qformer_stage1.cf_dropout=0.0 \
          run.qformer_stage1.w_rank=0.0 \
          run.qformer_stage1.epoch=10
```

**Step 3 — enable ONE at a time**, 10 epochs each, judged on **cold GAIN** and
`sem_gain` (not warm `combined_gain`, which cannot go positive):
1. `cf_dropout=0.3` — the only change targeting the identified root cause.
2. `memory_positional=50` — the only change targeting an axis `e_u` lacks.
3. `output_residual=True` — known to improve the probe, but as MF redundancy;
   worth keeping only if it also helps COLD.
4. the rest.

**Stopping rule.** If Step 0 passes and Steps 3.1–3.2 both leave cold `GAIN` ≤ 0
and `sem_gain` < 0.01 after 10 epochs, the Q-Former does not conduct semantics
and Stage 1 should be cut. Semantics would then enter where the LLM reads it
natively — title text in the Stage-3 prompt — not through an 8-token bottleneck.

---

## 4. Diagnostics added, and the trap each one exists for

Every one of these was written because a previous reading was confounded. Check
the invariant and the floor before trusting any similarity from this model.

- `diagnose_collab_collapse.py` — raw-MF ceilings. `gain` pins at ~0 both when
  ordering is perfect and when it is absent; read `top1`.
- `probe_cf_channel.probe_metrics` — within-user uAUC vs MF on the same rows.
  Reports raw/centered/dot, because a change living in the norm is invisible to
  a cosine.
- `..._combined` — MF + Q-Former with a weight sweep. `best_w = 0` means the
  channel is redundant with MF. Uses the channel's BEST readout (using the raw
  cosine handicapped it).
- `pooling_usage` — cos(full, last k), raw AND centered plus the across-user
  floor. Raw sits on an anisotropy floor and cannot separate "ignores history"
  from "output near-constant"; and with a history-invariant `residual_vec` it
  measured the residual instead of the pooling.
- `sem_coverage` — the Step-0 gate.
- `zero-init paths` — every zero-init path's weight norm (`proj_sem`,
  `user_proj`, `item_res_proj`, `fuse_cf`, `fuse_user`) plus `res_gain`. A path
  still at 0.0 is inert, and that failure is silent.
