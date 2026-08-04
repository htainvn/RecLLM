import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import Parameter

try:
    from transformers import (
        AutoTokenizer,
        BertModel,
        InstructBlipQFormerConfig,
        InstructBlipQFormerModel,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - depends on runtime env
    AutoTokenizer = None
    BertModel = None
    InstructBlipQFormerConfig = None
    InstructBlipQFormerModel = None

LOGGER = logging.getLogger(__name__)


class HFQFormerAdapter(nn.Module):
    """Wrapper around Hugging Face InstructBLIP Q-Former.

    Exposes multiple forward modes used by the SigLLM training stages:

    - ``forward(cf_vec, text)`` — joint forward returning query hidden states
      with ``out_proj`` applied. Used by Stage 2 / Stage 3 when ``text`` is the
      InstructBLIP-style task instruction.
    - ``encode_cf(cf_vec)`` — queries only, no text branch input. Used by
      Stage 1 ITC where the CF and text streams are kept uni-modal.
    - ``encode_text(text)`` — text only, no queries, no cross-attention. Used
      by Stage 1 ITC and as a CLS pool for downstream contrastive losses.
    - ``forward_multimodal(cf_vec, text, causal_text)`` — joint forward
      returning both query and text hidden states. ``causal_text=True`` masks
      text→text attention causally for ITG; ``False`` is the default
      bidirectional mode used by ITM and by ``forward``.

    The recommendation signal enters through ``encoder_hidden_states`` (cross-
    attention to a single CF token); the text stream enters through
    ``input_ids`` (self-attention with the learned queries).
    """

    def __init__(
        self,
        d_cf: int,
        d_model: int,
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        cross_attention_frequency: int = 2,
        initializer_range: float = 0.02,
        qformer_text_model_name: str = "bert-base-uncased",
        max_instruction_length: int = 48,
        init_from_pretrained_text: bool = True,
        user_conditioned: bool = False,
        d_user: Optional[int] = None,
        d_sem: Optional[int] = None,
        candidate_fusion: bool = False,
    ):
        super().__init__()

        if AutoTokenizer is None or InstructBlipQFormerConfig is None or InstructBlipQFormerModel is None:
            raise ModuleNotFoundError(
                "transformers with InstructBLIP support is required to use HFQFormerAdapter. "
                "Please install or upgrade transformers in the runtime environment."
            )

        self.d_cf = d_cf
        self.d_model = d_model
        self.num_queries = num_queries
        self.output_dim = int(output_dim) if output_dim is not None else d_model
        self.max_instruction_length = int(max_instruction_length)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(
            qformer_text_model_name,
            truncation_side="right",
        )

        # BLIP-2 query-token init: normal with std = initializer_range (0.02),
        # matching the scale of the (BERT-initialized) embeddings the queries
        # sit next to. torch.randn (std 1.0) put the queries ~50x larger than
        # every other activation at step 0.
        self.q = Parameter(torch.zeros(1, num_queries, d_model))
        nn.init.normal_(self.q, mean=0.0, std=initializer_range)
        self.proj_cf = nn.Linear(d_cf, d_model)
        self.out_proj = nn.Identity() if self.output_dim == d_model else nn.Linear(d_model, self.output_dim)

        # Conditioned queries. When enabled, the base learnable queries `self.q`
        # are shifted per-sample by `user_proj(cond_cf)` before the Q-Former
        # forward. `cond_cf` is any vector in the MF embedding space; the
        # intended use is the CANDIDATE ITEM's CF vector (DIN-style target
        # attention: the same history is pooled differently for each candidate,
        # so the pooled profile varies within a user and can move uAUC — a
        # per-USER conditioning vector is constant across a user's candidates
        # and cancels in uAUC). The module keeps its historical name
        # `user_proj` for checkpoint compatibility.
        self.user_conditioned = bool(user_conditioned)
        if self.user_conditioned:
            d_user_eff = int(d_user) if d_user is not None else d_cf
            self.d_user = d_user_eff
            # PER-QUERY shift: Linear(d_user, num_queries * d_model), reshaped
            # to [B, Q, d_model] in _build_query_tokens. The old
            # Linear(d_user, d_model) broadcast ONE vector across all Q
            # queries, so every query saw the candidate identically and the
            # conditioning could not specialize per query. NOTE: this changes
            # the weight shape — adapter checkpoints saved before this change
            # will not strict-load; rerun Stage 1 (all stages rebuild the
            # adapter with the same shape, so the pipeline stays consistent).
            self.user_proj = nn.Linear(d_user_eff, num_queries * d_model)
            # Zero-init so the residual `queries = pretrained_q + user_proj(cond_cf)`
            # starts as a no-op at step 0 (queries == pretrained_q exactly). Gradients
            # still flow through cond_cf, so user_proj grows from 0 only if the
            # training signal rewards it. Standard pattern for LoRA / FiLM / prefix tuning.
            nn.init.zeros_(self.user_proj.weight)
            nn.init.zeros_(self.user_proj.bias)

        # Candidate-aware MULTIPLICATIVE fusion in the cross-attention MEMORY
        # (not the queries). For each memory slot j and target t, the slot
        # becomes m_j = proj_cf(e_j) + fuse_cf([e_t; e_j*e_t; e_j - e_t]) —
        # equivalent to the SeLLa-style Linear(4*d_cf -> d_model) with the
        # last three blocks zero-initialised, but expressed additively so
        # pre-fusion checkpoints keep loading and warm-start is an EXACT
        # no-op. The e_j*e_t block is the point: its coordinate sum is the MF
        # dot product, so the Q-Former can read the collaborative match
        # between the candidate and every history item (target attention).
        # fuse_user injects the analogous user-target interaction
        # [e_u; e_u*e_t; e_u - e_t], broadcast over all memory slots.
        self.candidate_fusion = bool(candidate_fusion)
        if self.candidate_fusion:
            self.fuse_cf = nn.Linear(3 * d_cf, d_model, bias=False)
            nn.init.zeros_(self.fuse_cf.weight)
            self.fuse_user = nn.Linear(3 * d_cf, d_model, bias=False)
            nn.init.zeros_(self.fuse_user.weight)

        # Optional second cross-attention source: a frozen semantic (LLM-derived)
        # item embedding next to the CF vector. Attention weighs the two sources
        # per item, so cold items (weak CF, rich text) can lean on semantics and
        # warm items on CF. All-zero semantic rows (items with no text) are
        # masked out at forward time, falling back to CF-only cleanly.
        self.d_sem = int(d_sem) if d_sem is not None else None
        self.proj_sem = nn.Linear(self.d_sem, d_model) if self.d_sem else None
        if self.proj_sem is not None:
            # Zero-init: the semantic token starts as a constant no-op register
            # (value = bias = 0) and grows only if the training signal rewards
            # it. With the default random init the sem token is an
            # always-present cross-attention key carrying pure noise, which
            # dilutes the CF signal from step 0 — a stage-1 run with random
            # init showed ITC gain collapsing from ~2.3 to ~0.65 nats.
            nn.init.zeros_(self.proj_sem.weight)
            nn.init.zeros_(self.proj_sem.bias)

        config = InstructBlipQFormerConfig(
            vocab_size=len(self.qformer_tokenizer),
            hidden_size=d_model,
            encoder_hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size or (4 * d_model),
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            cross_attention_frequency=cross_attention_frequency,
            initializer_range=initializer_range,
        )
        self.qformer = InstructBlipQFormerModel(config)
        self.vocab_size = config.vocab_size

        if init_from_pretrained_text:
            self._init_text_branch_from_pretrained_bert(qformer_text_model_name)

    def _init_text_branch_from_pretrained_bert(self, bert_model_name: str) -> int:
        """Copy embeddings, self-attention, and FFN weights from a pretrained
        BERT into the Q-Former (BLIP-2 / InstructBLIP convention).

        Cross-attention layers keep their random init since BERT has no
        cross-attention. Name remaps applied (Q-Former name -> BERT name):

        - ``.attention.attention.``    -> ``.attention.self.`` (self-attention)
        - ``.intermediate_query.``     -> ``.intermediate.``   (query-token FFN)
        - ``.output_query.``           -> ``.output.``         (query-token FFN)
        - ``embeddings.layernorm.``    -> ``embeddings.LayerNorm.``

        The query FFN and embedding LayerNorm remaps matter most: the query
        tokens flow through ``intermediate_query``/``output_query`` (NOT the
        text FFN), so without the remap the exact sublayers every query passes
        through stayed fully random while the text side got BERT weights.

        When the Q-Former has fewer layers than BERT, only the first
        ``num_layers`` of BERT are copied. Tensors with mismatched shapes
        (e.g. when ``hidden_size`` or ``num_heads`` is configured differently
        from BERT-base) are skipped, leaving them at their random init.

        Returns the number of tensors successfully transferred.
        """
        if BertModel is None:
            LOGGER.warning(
                "transformers.BertModel not available; skipping pretrained "
                "text-branch init. Q-Former text side will start from random."
            )
            return 0

        try:
            bert = BertModel.from_pretrained(bert_model_name)
        except Exception as exc:  # network / cache miss
            LOGGER.warning(
                "Could not load pretrained BERT '%s' for Q-Former text-branch "
                "init (%s). Falling back to random init.",
                bert_model_name,
                exc,
            )
            return 0

        bert_state = bert.state_dict()
        target_state = self.qformer.state_dict()

        loaded = 0
        skipped_shape = 0
        for q_key, q_tensor in target_state.items():
            bert_key = (
                q_key.replace(".attention.attention.", ".attention.self.")
                .replace(".intermediate_query.", ".intermediate.")
                .replace(".output_query.", ".output.")
                .replace("embeddings.layernorm.", "embeddings.LayerNorm.")
            )
            if bert_key not in bert_state:
                continue
            bert_tensor = bert_state[bert_key]
            if bert_tensor.shape != q_tensor.shape:
                skipped_shape += 1
                continue
            target_state[q_key] = bert_tensor.clone()
            loaded += 1

        self.qformer.load_state_dict(target_state, strict=True)
        del bert

        LOGGER.info(
            "Initialized Q-Former text branch from %s: %d tensors loaded, "
            "%d shape-mismatch skipped, %d Q-Former tensors total.",
            bert_model_name,
            loaded,
            skipped_shape,
            len(target_state),
        )
        return loaded

    def load_state_dict(self, state_dict, strict: bool = True):
        """Accept both adapter-native keys and keys where the inner HF
        Q-Former prefix was stripped by external loading code."""
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        remapped_state_dict = dict(state_dict)
        expected_keys = set(super().state_dict().keys())

        has_prefixed_qformer_keys = any(
            isinstance(k, str) and k.startswith("qformer.") for k in remapped_state_dict
        )
        if not has_prefixed_qformer_keys:
            fixed_state_dict = {}
            remapped_any_key = False
            for key, value in remapped_state_dict.items():
                if isinstance(key, str) and f"qformer.{key}" in expected_keys:
                    fixed_state_dict[f"qformer.{key}"] = value
                    remapped_any_key = True
                else:
                    fixed_state_dict[key] = value
            if remapped_any_key:
                remapped_state_dict = fixed_state_dict

        return super().load_state_dict(remapped_state_dict, strict=strict)

    @property
    def text_word_embeddings(self) -> nn.Module:
        """Q-Former's text token embedding table.

        Returned so callers (e.g. ITG language modelling head) can tie weights
        to the Q-Former tokenizer's vocabulary.
        """
        return self.qformer.embeddings.word_embeddings

    def _normalize_text_input(self, text: Union[str, list], batch_size: int) -> list:
        if isinstance(text, str):
            return [text] * batch_size
        text = list(text)
        if len(text) != batch_size:
            raise ValueError(f"Expected {batch_size} text strings, got {len(text)}")
        return text

    def _tokenize(self, text_list: list, max_len: int, device):
        tokens = self.qformer_tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def _project_cf(
        self,
        cf_vec: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        fusion_target: Optional[torch.Tensor] = None,
        fusion_user: Optional[torch.Tensor] = None,
    ):
        if cf_vec.dim() == 2:
            cf_seq = cf_vec.unsqueeze(1)
        elif cf_vec.dim() == 3:
            cf_seq = cf_vec
        else:
            raise ValueError(f"Expected cf_vec shape [B, d_cf] or [B, 1, d_cf], got {tuple(cf_vec.shape)}")
        encoder_hidden_states = self.proj_cf(cf_seq)
        if self.candidate_fusion and fusion_target is not None:
            if fusion_target.dim() != 2 or fusion_target.size(0) != cf_seq.size(0):
                raise ValueError(
                    f"Expected fusion_target shape [B, d_cf], got {tuple(fusion_target.shape)}"
                )
            target = fusion_target.unsqueeze(1).expand_as(cf_seq)                     # [B,S,d_cf]
            encoder_hidden_states = encoder_hidden_states + self.fuse_cf(
                torch.cat([target, cf_seq * target, cf_seq - target], dim=-1)
            )
            if fusion_user is not None:
                if fusion_user.dim() != 2 or fusion_user.size(0) != cf_seq.size(0):
                    raise ValueError(
                        f"Expected fusion_user shape [B, d_cf], got {tuple(fusion_user.shape)}"
                    )
                user = fusion_user.unsqueeze(1)                                       # [B,1,d_cf]
                target_1 = fusion_target.unsqueeze(1)
                user_feat = self.fuse_user(
                    torch.cat([user, user * target_1, user - target_1], dim=-1)
                )                                                                     # [B,1,d_model]
                encoder_hidden_states = encoder_hidden_states + user_feat
        S = encoder_hidden_states.size(1)
        if source_mask is None:
            encoder_attention_mask = torch.ones(cf_vec.size(0), S, dtype=torch.long, device=cf_vec.device)
        else:
            if tuple(source_mask.shape) != (cf_vec.size(0), S):
                raise ValueError(
                    f"Expected source_mask shape [B, S] where S={S}, got {tuple(source_mask.shape)}"
                )
            source_mask = source_mask.long()
            empty_rows = source_mask.sum(dim=1) == 0
            if empty_rows.any():
                source_mask = source_mask.clone()
                source_mask[empty_rows, 0] = 1
            encoder_attention_mask = source_mask
        return encoder_hidden_states, encoder_attention_mask

    def _project_sources(
        self,
        cf_vec: torch.Tensor,
        sem_vec: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        fusion_target: Optional[torch.Tensor] = None,
        fusion_user: Optional[torch.Tensor] = None,
    ):
        """Assemble the cross-attention source sequence: CF tokens first, then
        (optionally) semantic tokens aligned slot-for-slot with the CF ones.

        ``sem_vec`` must mirror ``cf_vec``'s leading shape ([B, d_sem] or
        [B, S, d_sem]). A semantic slot is valid only when its CF slot is valid
        (same ``source_mask``) AND the row is non-zero — distilled banks keep
        uncovered items at zero, so those fall back to CF-only attention.
        """
        encoder_hidden_states, encoder_attention_mask = self._project_cf(
            cf_vec, source_mask, fusion_target=fusion_target, fusion_user=fusion_user
        )
        if sem_vec is None:
            return encoder_hidden_states, encoder_attention_mask
        if self.proj_sem is None:
            raise ValueError("sem_vec passed but the adapter was built without d_sem")

        if sem_vec.dim() == 2:
            sem_seq = sem_vec.unsqueeze(1)
        elif sem_vec.dim() == 3:
            sem_seq = sem_vec
        else:
            raise ValueError(
                f"Expected sem_vec shape [B, d_sem] or [B, S, d_sem], got {tuple(sem_vec.shape)}"
            )
        if sem_seq.size(0) != encoder_hidden_states.size(0) or sem_seq.size(1) != encoder_hidden_states.size(1):
            raise ValueError(
                f"sem_vec leading shape {tuple(sem_seq.shape[:2])} must match the CF "
                f"source sequence {tuple(encoder_hidden_states.shape[:2])}"
            )

        sem_mask = (sem_seq.norm(dim=-1) > 0).long() * encoder_attention_mask
        sem_hidden = self.proj_sem(sem_seq.to(self.proj_sem.weight.dtype))
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, sem_hidden.to(encoder_hidden_states.dtype)], dim=1
        )
        encoder_attention_mask = torch.cat([encoder_attention_mask, sem_mask], dim=1)
        return encoder_hidden_states, encoder_attention_mask

    def _build_query_tokens(
        self,
        batch_size: int,
        user_cf: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Expand base learnable queries and optionally shift per-user.

        When ``user_conditioned`` is on and ``user_cf`` is provided, each query
        token is shifted by ``user_proj(user_cf)`` so the same Q tokens encode
        different aspects for different users. Otherwise queries are identical
        across the batch (vanilla Q-Former behaviour).
        """
        query_tokens = self.q.expand(batch_size, -1, -1)
        if self.user_conditioned and user_cf is not None:
            if user_cf.dim() != 2:
                raise ValueError(
                    f"Expected user_cf shape [B, d_user], got {tuple(user_cf.shape)}"
                )
            if user_cf.size(0) != batch_size:
                raise ValueError(
                    f"user_cf batch ({user_cf.size(0)}) != cf_vec batch ({batch_size})"
                )
            user_cond = self.user_proj(user_cf).view(
                -1, self.num_queries, self.d_model
            )  # [B, Q, d_model] — a distinct shift per query
            query_tokens = query_tokens + user_cond
        return query_tokens

    def _build_causal_joint_mask(
        self,
        batch_size: int,
        query_count: int,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """3D attention mask for ITG: queries see queries (bidirectional),
        text positions see queries + previous text tokens (causal)."""

        device = text_attention_mask.device
        text_len = text_attention_mask.size(1)
        seq = query_count + text_len

        mask = torch.zeros(batch_size, seq, seq, dtype=torch.long, device=device)
        mask[:, :query_count, :query_count] = 1
        mask[:, query_count:, :query_count] = 1
        causal = torch.tril(torch.ones(text_len, text_len, dtype=torch.long, device=device))
        mask[:, query_count:, query_count:] = causal.unsqueeze(0).expand(batch_size, -1, -1)

        col_pad = text_attention_mask.unsqueeze(1).expand(batch_size, seq, text_len)
        mask[:, :, query_count:] = mask[:, :, query_count:] * col_pad
        row_pad = text_attention_mask.unsqueeze(-1)
        mask[:, query_count:, :] = mask[:, query_count:, :] * row_pad
        return mask

    def encode_cf(
        self,
        cf_vec: torch.Tensor,
        user_cf: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        sem_vec: Optional[torch.Tensor] = None,
        fusion_target: Optional[torch.Tensor] = None,
        fusion_user: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Queries-only forward over a CF (collaborative filtering) vector.

        The recommendation signal enters via cross-attention to a single CF
        token; there is no text-side input. Returns query hidden states of
        shape ``[B, num_queries, d_model]``. ``out_proj`` is not applied
        here; callers decide whether they want the projected (LLM-feeding)
        or raw (contrastive) representation.
        """
        batch_size = cf_vec.size(0)
        query_tokens = self._build_query_tokens(batch_size, user_cf)
        query_attention_mask = torch.ones(
            batch_size, query_tokens.size(1), dtype=torch.long, device=cf_vec.device
        )
        encoder_hidden_states, encoder_attention_mask = self._project_sources(
            cf_vec, sem_vec, source_mask, fusion_target=fusion_target, fusion_user=fusion_user
        )

        outputs = self.qformer(
            input_ids=None,
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def encode_text(
        self,
        text: Union[str, list],
        max_length: Optional[int] = None,
    ):
        """Text-only forward — no queries, no cross-attention.

        Returns ``(text_hidden, text_cls)`` where ``text_hidden`` is
        ``[B, T, d_model]`` and ``text_cls`` is ``text_hidden[:, 0]``.
        """
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)
        device = next(self.qformer.parameters()).device
        input_ids, attention_mask = self._tokenize(
            text_list, max_length or self.max_instruction_length, device
        )
        outputs = self.qformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_embeds=None,
            encoder_hidden_states=None,
            return_dict=True,
        )
        text_hidden = outputs.last_hidden_state
        return text_hidden, text_hidden[:, 0]

    def forward_multimodal(
        self,
        cf_vec: torch.Tensor,
        text: Union[str, list],
        causal_text: bool = False,
        max_text_length: Optional[int] = None,
        user_cf: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        sem_vec: Optional[torch.Tensor] = None,
        fusion_target: Optional[torch.Tensor] = None,
        fusion_user: Optional[torch.Tensor] = None,
    ):
        """Joint forward returning ``(query_hidden, text_hidden, text_ids, text_mask)``.

        ``causal_text=True`` enables a causal mask on the text→text attention
        block (used by ITG); the default is bidirectional (used by ITM and by
        the LLM-feeding ``forward``).

        ``source_mask`` ([B, S]) marks the valid cross-attention sources; pass
        it whenever ``cf_vec`` is a padded sequence. ``None`` means "all valid",
        which is correct for the single-vector (S=1) callers.
        """
        if cf_vec.dim() not in (2, 3):
            raise ValueError(f"Expected cf_vec shape [B, d_cf] or [B, 1, d_cf], got {tuple(cf_vec.shape)}")

        batch_size = cf_vec.size(0)
        text_list = self._normalize_text_input(text, batch_size)

        query_tokens = self._build_query_tokens(batch_size, user_cf)
        query_count = query_tokens.size(1)

        text_ids, text_attention_mask = self._tokenize(
            text_list, max_text_length or self.max_instruction_length, cf_vec.device
        )

        # ``source_mask`` MUST be forwarded here: when cf_vec is a padded
        # history sequence ([B, L, d_cf]) the padded slots are not neutral —
        # ``proj_cf`` has a bias, so a zero pad embedding maps to ``proj_cf.bias``
        # and becomes a valid cross-attention key. Dropping the mask let those
        # slots absorb attention mass proportional to the padding count, i.e.
        # leaked history length into the pooled profile token.
        encoder_hidden_states, encoder_attention_mask = self._project_sources(
            cf_vec, sem_vec, source_mask, fusion_target=fusion_target, fusion_user=fusion_user
        )
        query_attention_mask = torch.ones(
            batch_size, query_count, dtype=torch.long, device=cf_vec.device
        )

        if causal_text:
            joint_attention_mask = self._build_causal_joint_mask(
                batch_size, query_count, text_attention_mask
            )
        else:
            joint_attention_mask = torch.cat([query_attention_mask, text_attention_mask], dim=1)

        outputs = self.qformer(
            input_ids=text_ids,
            attention_mask=joint_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        sequence_hidden = outputs.last_hidden_state
        query_hidden = sequence_hidden[:, :query_count]
        text_hidden = sequence_hidden[:, query_count:]
        return query_hidden, text_hidden, text_ids, text_attention_mask

    def forward(
        self,
        cf_vec: torch.Tensor,
        instruction,
        user_cf: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        sem_vec: Optional[torch.Tensor] = None,
        fusion_target: Optional[torch.Tensor] = None,
        fusion_user: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """LLM-feeding mode: queries cross-attend to ``cf_vec`` (and the
        optional ``sem_vec`` semantic source) while the text stream consumes
        ``instruction``. Returns query hidden states with ``out_proj``
        applied: ``[B, num_queries, output_dim]``."""

        query_hidden, _, _, _ = self.forward_multimodal(
            cf_vec,
            instruction,
            causal_text=False,
            user_cf=user_cf,
            source_mask=source_mask,
            sem_vec=sem_vec,
            fusion_target=fusion_target,
            fusion_user=fusion_user,
        )
        return self.out_proj(query_hidden)
