"""CoRA-style collaborative parameter-space injection.

Implements the core idea of CoRA (Collaborative Information Perception by
LLM's Weights, arXiv:2408.10645): instead of (only) splicing collaborative
soft tokens into the prompt's *input* space, turn the Q-Former's collaborative
queries into a **per-sample low-rank weight delta** added to chosen LLM linear
layers (attention q/v projections by default). This avoids the input-space
interference between ID embeddings and text semantics that caps soft-token
methods, and keeps the collaborative signal alive deeper into the network.

Mechanism
---------
For a target linear with weight ``W : [out, in]`` and input ``x : [B, T, in]``,
we add a rank-``Q`` delta driven by the collaborative queries
``Z : [B, Q, d_model]`` (one block of ``Q`` queries per sample, produced by the
Q-Former)::

    A  = P_down(Z)            # [B, Q, in]   down factor
    Bm = P_up(Z)              # [B, Q, out]  up factor   (P_up zero-init -> delta=0 at start)
    delta = scaling * (x @ A.transpose(1, 2)) @ Bm        # [B, T, out]
    output = output + delta

``P_down`` / ``P_up`` are *shared across layers* but *separate per weight shape*
(so q_proj shares one generator across all layers, v_proj another). This keeps
the added parameter count and overfitting risk low on small datasets (~33k
samples here), while the per-sample variation comes entirely from ``Z``.

Usage
-----
    injector = CollaborativeLoRAInjector(d_model=768, target_modules=("q_proj","v_proj"), alpha=16)
    injector.attach(llm_model)                 # after LoRA is attached, if any
    ...
    injector.set_queries(cf_q)                 # cf_q: [B, Q, d_model] from the Q-Former
    with injector.enabled():
        out = llm_model(inputs_embeds=..., attention_mask=...)
    injector.clear()
"""

import contextlib
import logging
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class CollaborativeLoRAInjector(nn.Module):
    def __init__(
        self,
        d_model: int,
        target_modules: Iterable[str] = ("q_proj", "v_proj"),
        alpha: float = 16.0,
        num_queries: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.target_modules = tuple(target_modules)
        self.alpha = float(alpha)
        self.num_queries = num_queries  # used only for the scaling denominator

        # One (P_down, P_up) generator per distinct (in_features, out_features)
        # weight shape among the targets. Keyed by "in_out" string because
        # nn.ModuleDict keys must be strings.
        self.generators = nn.ModuleDict()
        self._shape_for_module: Dict[int, str] = {}  # id(module) -> shape key

        self._queries: Optional[torch.Tensor] = None  # [B, Q, d_model]
        self._enabled: bool = False
        self._hooks = []
        self._warned_batch_mismatch = False

    # -- construction ------------------------------------------------------

    @staticmethod
    def _shape_key(in_features: int, out_features: int) -> str:
        return f"{int(in_features)}_{int(out_features)}"

    def _make_generator(self, in_features: int, out_features: int) -> nn.Module:
        gen = nn.ModuleDict({
            # bias=False: a pure linear map from query space to factor space.
            "down": nn.Linear(self.d_model, int(in_features), bias=False),
            "up": nn.Linear(self.d_model, int(out_features), bias=False),
        })
        # Kaiming-ish small init on down, zero on up so the initial delta is
        # exactly zero (LoRA convention) — training starts from the unmodified
        # LLM and learns the collaborative correction.
        nn.init.normal_(gen["down"].weight, std=0.02)
        nn.init.zeros_(gen["up"].weight)
        return gen

    def _iter_target_modules(self, llm_model: nn.Module):
        target_leaf = set(self.target_modules)
        for name, module in llm_model.named_modules():
            leaf = name.split(".")[-1]
            if leaf in target_leaf and hasattr(module, "in_features") and hasattr(module, "out_features"):
                yield name, module

    def attach(self, llm_model: nn.Module) -> int:
        """Discover target modules, build per-shape generators, register hooks.

        Returns the number of modules hooked. Safe to call once; call
        ``detach()`` first to re-attach.
        """
        if self._hooks:
            raise RuntimeError("Injector already attached; call detach() first.")

        count = 0
        for name, module in self._iter_target_modules(llm_model):
            in_f = int(module.in_features)
            out_f = int(module.out_features)
            key = self._shape_key(in_f, out_f)
            if key not in self.generators:
                self.generators[key] = self._make_generator(in_f, out_f)
            self._shape_for_module[id(module)] = key
            self._hooks.append(module.register_forward_hook(self._make_hook(key)))
            count += 1

        LOGGER.info(
            "CollaborativeLoRAInjector attached: %d target modules, %d distinct shapes (%s).",
            count, len(self.generators), ", ".join(self.generators.keys()),
        )
        if count == 0:
            LOGGER.warning(
                "CollaborativeLoRAInjector found no target modules matching %s. "
                "Weight injection will be a no-op.", self.target_modules,
            )
        return count

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._shape_for_module = {}

    # -- runtime state -----------------------------------------------------

    def set_queries(self, queries: torch.Tensor) -> None:
        """Store the current batch's collaborative queries ``[B, Q, d_model]``.

        Kept attached to the autograd graph so gradients flow back into the
        Q-Former and the generators.
        """
        if queries.dim() != 3 or queries.size(-1) != self.d_model:
            raise ValueError(
                f"Expected queries [B, Q, {self.d_model}]; got {tuple(queries.shape)}"
            )
        self._queries = queries

    def clear(self) -> None:
        self._queries = None

    @contextlib.contextmanager
    def enabled(self):
        prev = self._enabled
        self._enabled = True
        try:
            yield
        finally:
            self._enabled = prev

    @property
    def scaling(self) -> float:
        q = self.num_queries
        if q is None and self._queries is not None:
            q = self._queries.size(1)
        if not q:
            q = 1
        return self.alpha / float(q)

    # -- the hook ----------------------------------------------------------

    def _make_hook(self, shape_key: str):
        def hook(module, inputs, output):
            if not self._enabled or self._queries is None:
                return output

            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() < 2:
                return output

            Z = self._queries
            if x.size(0) != Z.size(0):
                # Shapes can diverge during cached/incremental decoding; skip
                # rather than corrupt. Warn once.
                if not self._warned_batch_mismatch:
                    LOGGER.warning(
                        "CollaborativeLoRAInjector: batch mismatch x=%d vs Z=%d; "
                        "skipping weight injection for this call.",
                        x.size(0), Z.size(0),
                    )
                    self._warned_batch_mismatch = True
                return output

            gen = self.generators[shape_key]
            # Compute the low-rank correction in fp32 for stability, and on the
            # activation's device (Q-Former and LLM may live on different
            # devices under device_map="auto"), then cast back.
            z = Z.to(device=x.device, dtype=torch.float32)
            down_w = gen["down"].weight.to(device=x.device, dtype=torch.float32)
            up_w = gen["up"].weight.to(device=x.device, dtype=torch.float32)
            a = F.linear(z, down_w)   # [B, Q, in]
            b = F.linear(z, up_w)     # [B, Q, out]

            x32 = x.to(torch.float32)
            # (x @ A^T) @ B : [B,T,in]@[B,in,Q] -> [B,T,Q]; @[B,Q,out] -> [B,T,out]
            delta = torch.matmul(torch.matmul(x32, a.transpose(1, 2)), b)
            delta = self.scaling * delta
            return output + delta.to(output.dtype)

        return hook
