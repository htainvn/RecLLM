"""Gated history Q-Former — the trimmed, additive variant used by
``sella_gated_rec_llm``.

DESIGN CONTRACT (do not break)
------------------------------
1. SeLLa's three soft tokens ``<UserID>``, ``<ItemID>``, ``<Warm_ID>`` keep
   their prompt text, their positions and their embedding sources. No token is
   added, no position is added, no prompt is rewritten.
2. This module contributes exactly ONE gated additive term, and only to the
   ``<UserID>`` embedding::

       e_user  <-  e_user + g * delta(user, target, history)

   ``g`` is a single scalar, zero-initialised. At ``g == 0`` the forward pass is
   numerically identical to the no-Q-Former model (the term is a literal
   ``+ 0``); the value of ``g`` after training is the reported measure of the
   module's contribution.
3. No BERT / text branch, and no ``<UserProfile>`` / ``<TargetItemID>`` soft
   token blocks. The instruction list of this task is near-constant, so a text
   stream buys nothing and costs ~72M parameters; and a zero-init projection at
   a brand-new prompt position would put an out-of-distribution embedding into
   the LLM's input sequence, which a gated addition onto an existing token
   avoids by construction.

Compared with ``HFQFormerAdapter`` (the Stage-1/2 module this replaces):

    ================================  =========  =========
    knob                              adapter    this
    ================================  =========  =========
    text branch (BERT init, tokenizer)  yes        no
    d_model                             768        256
    num_layers                          4          2
    cross_attention_frequency           2          1
    num_queries                         8          1-2
    parameters                        ~76M       ~4.3M
    ================================  =========  =========

With ``cross_attention_frequency = 1`` and 2 layers, EVERY layer — including
the last — is refreshed from memory, so trimming depth does not starve the
readout.

WHAT THE MODULE SEES
--------------------
Cross-attention memory (2L + 1 slots for a history of length L):

    * ``L`` collaborative-filtering slots   ``proj_cf(e_j)``      per history item j
    * ``L`` semantic slots                  ``proj_sem(LN(s_j))`` per history item j
    * ``1`` user x target interaction slot  ``fuse_user([e_u ; e_u * e_t ; e_u - e_t])``

  The multiplicative block ``e_u * e_t`` is the point of the interaction slot:
  its coordinate sum *is* the MF dot product, so attention can read the
  collaborative match between this user and this candidate directly. It has to
  be its own slot rather than a delta broadcast over the other slots — adding
  the same vector to every key shifts all attention logits equally (softmax
  unchanged) and adds a slot-independent constant to the value sum, i.e. a
  broadcast degenerates into a plain output bias.

Queries: ``num_queries`` learned vectors with a residual injection of the
candidate CF vector ``e_t``. This is what makes the pooling candidate-dependent
(DIN-style target attention): the same history is pooled differently for each
candidate, so the contribution can move uAUC. A candidate-INDEPENDENT pooled
profile is constant across a user's candidates and cancels in uAUC entirely.

SHAPES
------
    user_cf   [B, d_cf]
    target_cf [B, d_cf]
    hist_cf   [B, L, d_cf]
    hist_sem  [B, L, d_sem]   (optional)
    hist_mask [B, L]  bool, True = real history item
    ->  delta [B, d_out]      already multiplied by the gate
"""

from typing import Optional

import torch
import torch.nn as nn


# How the gated term's MAGNITUDE is controlled. See the block in `forward`.
#   match_ref  ||g*delta|| == |g| * ||e_user||  -> delta_rel == |g| exactly. Default.
#   unit       ||g*delta|| == |g|               -> absolute, reference-independent.
#   none       raw out_proj output              -> the original behaviour, which
#              measured 255x the token it was added to. Kept only to reproduce it.
DELTA_SCALES = ("match_ref", "unit", "none")


class _QFormerLayer(nn.Module):
    """One Q-Former block: self-attn over queries, cross-attn into memory, FFN.

    Post-norm residual, matching the reference Q-Former block.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_self = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, memory_pad_mask):
        y, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.ln_self(x + self.drop(y))
        y, _ = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_pad_mask, need_weights=False
        )
        x = self.ln_cross(x + self.drop(y))
        x = self.ln_ffn(x + self.drop(self.ffn(x)))
        return x


class GatedHistoryQFormer(nn.Module):
    """Candidate-conditioned history encoder folded into SeLLa's ``<UserID>``."""

    def __init__(
        self,
        d_cf: int,
        d_out: int,
        d_sem: Optional[int] = None,
        d_model: int = 256,
        num_queries: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_sem: bool = True,
        use_user_slot: bool = True,
        hist_target_fusion: bool = False,
        delta_scale: str = "match_ref",
    ):
        super().__init__()
        if delta_scale not in DELTA_SCALES:
            raise ValueError(f"delta_scale must be one of {DELTA_SCALES}, got {delta_scale!r}")
        self.delta_scale = delta_scale
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.d_cf = int(d_cf)
        self.d_model = int(d_model)
        self.d_out = int(d_out)
        self.num_queries = int(num_queries)

        # --- queries: learned base + residual injection of the candidate ------
        self.q = nn.Parameter(torch.zeros(1, self.num_queries, self.d_model))
        nn.init.normal_(self.q, mean=0.0, std=0.02)
        # One distinct shift PER QUERY (reshaped to [B, Q, d_model]); a single
        # Linear(d_cf, d_model) broadcast would give every query the identical
        # view of the candidate and prevent per-query specialisation.
        self.target_proj = nn.Linear(self.d_cf, self.num_queries * self.d_model)
        # LayerNorm on the assembled query block so the ratio between the base
        # queries and the candidate residual cannot blow up the block's scale.
        self.query_norm = nn.LayerNorm(self.d_model)

        # --- memory: history CF slots ----------------------------------------
        self.proj_cf = nn.Linear(self.d_cf, self.d_model)

        # --- memory: history semantic slots ----------------------------------
        # The semantic bank holds LLM hidden states, whose norms are an order of
        # magnitude above the MF embeddings; LayerNorm first so the two slot
        # families arrive at comparable scale.
        self.d_sem = int(d_sem) if (use_sem and d_sem) else None
        if self.d_sem:
            self.sem_norm = nn.LayerNorm(self.d_sem)
            self.proj_sem = nn.Linear(self.d_sem, self.d_model)
        else:
            self.sem_norm = None
            self.proj_sem = None

        # --- memory: one user x target interaction slot -----------------------
        self.fuse_user = (
            nn.Linear(3 * self.d_cf, self.d_model) if use_user_slot else None
        )

        # --- optional ablation lever: per-slot candidate fusion ---------------
        # m_j = proj_cf(e_j) + fuse_hist([e_t ; e_j * e_t ; e_j - e_t]).
        # Off by default: candidate-dependence already enters through the query
        # residual, and this is the cheapest thing to switch on if that channel
        # turns out to be too weak.
        self.hist_target_fusion = bool(hist_target_fusion)
        self.fuse_hist = (
            nn.Linear(3 * self.d_cf, self.d_model, bias=False)
            if self.hist_target_fusion
            else None
        )

        self.mem_norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList(
            [_QFormerLayer(self.d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(self.d_model, self.d_out)

        # --- the gate ---------------------------------------------------------
        # Scalar, zero-init. Step 0 is an exact no-op, so the LLM never sees an
        # out-of-distribution input embedding before the module has learned
        # anything. Gradient still reaches ``gate`` immediately (dL/dg =
        # <grad, delta> with delta != 0 from the random init), so the gate leaves
        # 0 on the first update and only then does the Q-Former itself start
        # learning. Same dynamic as LoRA's zero-init B matrix.
        #
        # 1-D on purpose: ``build_optimizer._is_non_decay`` files every tensor
        # with ndim < 2 into the no-weight-decay group, so the gate is not pulled
        # back toward 0 by decay.
        self.gate = nn.Parameter(torch.zeros(1))

        # Diagnostics, kept as detached tensors so reading them costs no
        # host<->device sync inside the training step; ``.item()`` happens in the
        # model's throttled log hook.
        self._last_delta_norm = None
        self._last_ref_norm = None

    # ------------------------------------------------------------------ utils
    @property
    def uses_sem(self) -> bool:
        return self.proj_sem is not None

    def describe(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        parts = [
            f"d_cf={self.d_cf}",
            f"d_model={self.d_model}",
            f"d_out={self.d_out}",
            f"layers={len(self.layers)}",
            f"queries={self.num_queries}",
            "cross_attention_frequency=1",
            f"sem_slots={'on(d_sem=%d)' % self.d_sem if self.uses_sem else 'off'}",
            f"user_target_slot={'on' if self.fuse_user is not None else 'off'}",
            f"hist_target_fusion={'on' if self.hist_target_fusion else 'off'}",
            f"delta_scale={self.delta_scale}",
            "text_branch=off",
            f"params={total / 1e6:.2f}M",
        ]
        return "GatedHistoryQFormer(" + ", ".join(parts) + ")"

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        user_cf: torch.Tensor,
        target_cf: torch.Tensor,
        hist_cf: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_sem: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hist_cf.dim() != 3:
            raise ValueError(f"hist_cf must be [B, L, d_cf], got {tuple(hist_cf.shape)}")
        if user_cf.dim() != 2 or target_cf.dim() != 2:
            raise ValueError(
                "user_cf and target_cf must be [B, d_cf], got "
                f"{tuple(user_cf.shape)} and {tuple(target_cf.shape)}"
            )

        dtype = self.proj_cf.weight.dtype
        batch = hist_cf.size(0)
        device = hist_cf.device

        user_cf = user_cf.to(dtype)
        target_cf = target_cf.to(dtype)
        hist_cf = hist_cf.to(dtype)
        hist_mask = hist_mask.to(device=device, dtype=torch.bool)

        # ---- memory: history CF slots
        slots = [self.proj_cf(hist_cf)]
        masks = [hist_mask]
        if self.fuse_hist is not None:
            tgt = target_cf.unsqueeze(1).expand_as(hist_cf)
            slots[0] = slots[0] + self.fuse_hist(
                torch.cat([tgt, hist_cf * tgt, hist_cf - tgt], dim=-1)
            )

        # ---- memory: history semantic slots
        if self.proj_sem is not None and hist_sem is not None:
            hist_sem = hist_sem.to(dtype)
            if hist_sem.shape[:2] != hist_cf.shape[:2]:
                raise ValueError(
                    f"hist_sem leading shape {tuple(hist_sem.shape[:2])} must match "
                    f"hist_cf {tuple(hist_cf.shape[:2])}"
                )
            # A distilled bank leaves uncovered items at exactly zero, and the
            # padding row is zero by construction; those slots carry no
            # information, so drop them instead of feeding proj_sem's bias in as
            # a phantom key.
            sem_mask = hist_mask & (hist_sem.norm(dim=-1) > 0)
            slots.append(self.proj_sem(self.sem_norm(hist_sem)))
            masks.append(sem_mask)

        # ---- memory: the user x target interaction slot
        if self.fuse_user is not None:
            usr = user_cf.unsqueeze(1)
            tgt = target_cf.unsqueeze(1)
            slots.append(self.fuse_user(torch.cat([usr, usr * tgt, usr - tgt], dim=-1)))
            masks.append(torch.ones(batch, 1, dtype=torch.bool, device=device))

        memory = self.mem_norm(torch.cat(slots, dim=1))
        valid = torch.cat(masks, dim=1)
        # nn.MultiheadAttention returns NaN for a row whose keys are ALL masked.
        # Cannot happen while the user slot is on, but keep the guard so the
        # ablation (use_user_slot=False) stays safe on empty histories.
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            valid[empty, 0] = True

        # ---- queries: learned base + candidate residual
        queries = self.q.expand(batch, -1, -1) + self.target_proj(target_cf).view(
            batch, self.num_queries, self.d_model
        )
        x = self.query_norm(queries)
        for layer in self.layers:
            x = layer(x, memory, ~valid)

        delta = self.out_proj(x.mean(dim=1))

        # --- scale control: the difference between a gated contribution and a
        # --- replacement of the token it is added to.
        #
        # `out_proj` is an unconstrained Linear(d_model, d_out). With d_out=3584
        # its output norm is ~25 at init and grows from there, while the
        # `id_proj(e_u)` it is added to has norm ~0.27 — a measured 255x gap. A
        # single scalar gate cannot fix that: it shrank to 0.048 and the gated
        # term was still 12x the user token, i.e. `<UserID>` was not being
        # refined, it was being REPLACED by an out-of-distribution vector, and
        # with LoRA frozen the LLM could not adapt. uAUC fell ~0.10 below the
        # text-only baseline.
        #
        # `match_ref` (default) makes `out_proj` responsible for DIRECTION only
        # and the gate for magnitude, in units of the reference token:
        #     gated = g * ||e_user|| * (delta / ||delta||)
        # so ||gated|| / ||e_user|| == |g| EXACTLY. That is what makes the
        # reported `g` a real contribution ratio rather than a number whose
        # meaning depends on out_proj's arbitrary scale — g = 0.1 means "10% of
        # the user token", at any point in training.
        #
        # The norm in the denominator is NOT detached (so gradients shape the
        # direction correctly), while the reference norm IS (so the scale does
        # not leak gradient back into id_proj).
        if self.delta_scale in ("match_ref", "unit"):
            direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if self.delta_scale == "match_ref" and reference is not None:
                scale = reference.detach().to(delta.dtype).norm(dim=-1, keepdim=True)
            else:
                scale = torch.ones(1, 1, dtype=delta.dtype, device=delta.device)
            delta = direction * scale
        # "none" keeps the raw out_proj output — the original, broken behaviour,
        # retained only so the regression can be reproduced on demand.

        gated = self.gate.to(delta.dtype) * delta

        with torch.no_grad():
            self._last_delta_norm = gated.detach().float().norm(dim=-1).mean()
            self._last_ref_norm = (
                reference.detach().float().norm(dim=-1).mean()
                if reference is not None
                else None
            )
        return gated

    # ------------------------------------------------------------ diagnostics
    def pop_stats(self, gate_value: Optional[float] = None):
        """``{gate, delta_norm, delta_rel}`` for the last forward.

        ``delta_rel = ||g * delta|| / ||e_user||`` is the honest contribution
        measure — ``gate`` alone is only comparable across runs that share the
        same ``out_proj`` scale.

        ``gate_value`` lets a caller pass a pre-gathered scalar (e.g. under a
        sharded-parameter regime where reading the tensor here would see an
        empty view). ``delta_norm`` is unaffected: it is computed inside forward,
        where the parameter is materialised.
        """
        stats = {}
        if gate_value is not None:
            stats["gate"] = float(gate_value)
        else:
            gate = self.gate.detach().float().reshape(-1)
            if gate.numel() == 1:
                stats["gate"] = float(gate[0].item())
        if self._last_delta_norm is not None:
            delta_norm = float(self._last_delta_norm.item())
            stats["delta_norm"] = delta_norm
            if self._last_ref_norm is not None:
                ref_norm = float(self._last_ref_norm.item())
                stats["delta_rel"] = delta_norm / ref_norm if ref_norm > 0 else 0.0
        return stats
