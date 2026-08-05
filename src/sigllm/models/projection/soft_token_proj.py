"""The soft-token projection shared by Stage 1, 2 and 3.

Weights flow Stage 1 ``llm_align_proj`` -> Stage 2 ``llm_proj``
(``proj_ckpt_in``) -> Stage 3 ``llm_proj`` (``llm_proj_ckpt``), all loaded
``strict=True``, so the three sites MUST build the identical layout. They used
to each hand-roll ``Sequential(Linear, LayerNorm)``; this module is the single
definition so they cannot drift.
"""

import torch
import torch.nn as nn

LOGGER = __import__("logging").getLogger(__name__)


class SharedDirectionCenter(nn.Module):
    """Subtract a running estimate of the direction shared across items.

    WHY THIS EXISTS. The Q-Former output is extremely anisotropic: Stage 1 logs
    ``offdiag_cos_raw ~0.9998``, i.e. the item-specific part is ~1.4% of the
    token norm, and a Stage-2 CF-only run measured ``q_pair_cos = 1.0000``
    exactly. Measured with ``diagnose_collab_collapse.py``, ``proj_cf``'s output
    is still discriminative (raw offdiag cos 0.70), so that shared direction is
    added INSIDE the Q-Former body — learned query tokens plus layer biases
    swamping the cross-attention read. Removing the batch mean before the cosine
    is what unlocks the collaborative losses (user_item history-pooled ceiling
    on raw MF: gain +0.183 raw vs +0.497 centered, top1 6.5x vs 15.6x chance).

    Fixing only the LOSS is half a fix, though: Stage 3 injects these tokens
    into the LLM, and a shared direction that survives means the LLM sees
    near-identical vectors for every item no matter how good the Stage-1 metric
    looks.

    It works because LayerNorm cannot do this job: LayerNorm removes the mean
    ACROSS FEATURES within a single token, so a direction shared across ITEMS
    passes through it untouched. Simulated on the measured regime (input
    offdiag_cos 0.9998, H=3584, LayerNorm weight at 1/sqrt(H)), inserting this
    takes the injected tokens from ``offdiag_cos 0.9998`` to ``-0.002`` — from
    near-identical to near-orthogonal — while ``mean_l2`` stays at 1.000, so the
    injection scale contract is preserved.

    PLACEMENT: between the Linear and the LayerNorm. Note what this is and is
    not. Because the Linear is linear, centering its INPUT already propagates:
    ``W(x - m) + b = Wx + b - Wm``, so pre-Linear centering removes the same
    shared direction and (simulated) reaches the identical ``offdiag_cos
    -0.002``; at init the two are exactly equal, since the Linear's bias starts
    at zero. The reason to sit after the Linear is robustness rather than
    expressiveness: here the across-item mean is driven to exactly zero at the
    point of injection whatever the Linear's bias later drifts to, whereas
    pre-Linear centering leaves ``b`` as a residual shared component of
    training-dependent size. (Pre-Linear centering also adds no representable
    function — the bias can already express it — which is the sense in which it
    is redundant.)

    NOT after the LayerNorm: that LayerNorm's weight is initialised at
    ``1/sqrt(H)`` specifically so the injected tokens have ``mean_l2 ~ 1``,
    matching the LLM's native input-embedding scale, and subtracting afterwards
    puts the norm back out of range.

    A RUNNING mean (not the batch mean) is subtracted, always, in train and in
    eval alike. Batch statistics are unusable here: Stage 3 scores one candidate
    at a time at inference, where a batch mean would zero the vector outright,
    and a train/eval discrepancy in the injected tokens is exactly the kind of
    silent skew this pipeline has been bitten by before. The estimate is
    detached, so this contributes no gradient — it is a slowly-tracked constant
    shift, not a normalisation layer.

    The mean is tracked PER QUERY POSITION when ``num_positions`` is given. The
    Q learned queries are different vectors with different characteristic
    directions, so one global mean would leave each position's own
    shared-across-items component in place — which is the component that
    matters.

    CAVEAT worth knowing: one instance sees every tensor routed through its
    projection — in Stage 3 both the target tokens and the pooled-history tokens
    — and those have different means. What is removed is therefore the grand
    mean over whatever passed through, not a per-role mean. That still strips the
    dominant shared direction; splitting per role would need separate modules and
    would break the single-checkpoint flow between the stages.
    """

    def __init__(self, dim: int, num_positions: int = None, momentum: float = 0.01):
        super().__init__()
        self.momentum = float(momentum)
        self.num_positions = int(num_positions) if num_positions else None
        shape = (self.num_positions, dim) if self.num_positions else (dim,)
        self.register_buffer("running_mean", torch.zeros(*shape))
        # Kept as a buffer so it round-trips through state_dict: a checkpoint
        # restored mid-training must not restart the EMA from "uninitialised"
        # and blow the first batch away.
        self.register_buffer("initialized", torch.zeros(1))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        if self.num_positions and x.dim() >= 2 and x.size(-2) == self.num_positions:
            # [..., P, D] -> mean over every leading axis, per position.
            batch_mean = x.detach().reshape(-1, self.num_positions, x.size(-1)).mean(0)
        else:
            batch_mean = x.detach().reshape(-1, x.size(-1)).mean(0)
            if self.num_positions:
                # Unexpected shape (e.g. a caller projecting a single pooled
                # vector). Broadcast rather than crash, and keep the estimate
                # usable for the positional case.
                batch_mean = batch_mean.unsqueeze(0).expand(self.num_positions, -1)
        batch_mean = batch_mean.to(self.running_mean.dtype)
        if self.initialized.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.initialized.fill_(1)
        else:
            self.running_mean.mul_(1.0 - self.momentum).add_(batch_mean, alpha=self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x)
        # clone(), because ``running_mean`` is mutated IN PLACE above and one
        # training step runs this module several times (target tokens, then the
        # pooled-history tokens). A detached view shares the version counter, so a
        # later in-place update can invalidate what an earlier forward's backward
        # saved — the "[1] is at version 15; expected 13" crash the residual's
        # divisor hit. Subtraction happens not to save its operands (d/dx = 1), so
        # this path has not failed in practice; cloning is cheap insurance against
        # that detail changing, and against a crash surfacing mid-run.
        mean = self.running_mean.clone()
        if self.num_positions and not (x.dim() >= 2 and x.size(-2) == self.num_positions):
            mean = mean.mean(0)
        return x - mean.to(dtype=x.dtype, device=x.device)


def build_soft_token_projection(
    d_in: int,
    d_out: int,
    center: bool = True,
    num_positions: int = None,
    momentum: float = 0.01,
) -> nn.Sequential:
    """``Linear -> [SharedDirectionCenter] -> LayerNorm``.

    ``LayerNorm.weight`` starts at ``1/sqrt(d_out)`` so the output has
    ``mean_l2 ~ 1`` from step 0, matching the LLM's native input-embedding
    scale. The default LayerNorm init (weight=1) gives ``mean_l2 ~ sqrt(H) ~ 64``,
    which drowns the text portion of the prompt under the frozen attention.

    Index contract, relied on by the init below and by ``remap_legacy_proj_state``:
    ``[0]`` is always the Linear and the LayerNorm is LAST.
    """
    layers = [nn.Linear(d_in, d_out)]
    if center:
        layers.append(SharedDirectionCenter(d_out, num_positions=num_positions, momentum=momentum))
    layers.append(nn.LayerNorm(d_out))

    proj = nn.Sequential(*layers)
    nn.init.normal_(proj[0].weight, std=0.02)
    nn.init.zeros_(proj[0].bias)
    nn.init.constant_(proj[-1].weight, d_out ** -0.5)
    nn.init.zeros_(proj[-1].bias)
    return proj


def remap_legacy_proj_state(state_dict, target: nn.Sequential):
    """Make a pre-centering checkpoint loadable into a centered projection.

    Inserting ``SharedDirectionCenter`` at index 1 shifts the LayerNorm from
    ``1.*`` to ``2.*``, so every projection checkpoint saved before that change
    (Stage 1 ``*_best_align_proj.pth``, Stage 2 ``*_best_proj.pth``) would fail
    a strict load with "missing 2.weight / unexpected 1.weight". Migrating the
    two keys is exact — the Linear is untouched, and the fresh centering buffers
    are the correct start for the EMA.

    Returns ``(state_dict, migrated)``; a no-op when the layouts already agree.
    """
    if not isinstance(state_dict, dict):
        return state_dict, False

    layer_norm_index = len(target) - 1
    if layer_norm_index == 1:
        return state_dict, False  # centering disabled: layouts already match
    if any(isinstance(k, str) and k.startswith(f"{layer_norm_index}.") for k in state_dict):
        return state_dict, False  # already the new layout

    migrated = {}
    did_migrate = False
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("1."):
            migrated[f"{layer_norm_index}.{key[2:]}"] = value
            did_migrate = True
        else:
            migrated[key] = value
    if did_migrate:
        LOGGER.info(
            "Migrated a pre-centering projection checkpoint: LayerNorm keys 1.* -> %d.*",
            layer_norm_index,
        )
    return migrated, did_migrate
