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


TYPE_USER = 0
TYPE_TARGET = 1
TYPE_HISTORY = 2


class HFQFormerAdapter(nn.Module):
    """Wrapper around Hugging Face InstructBLIP Q-Former.

    Cross-attention K/V is a 12-token sequence per sample: 1 user, 1 target,
    10 history (left-padded, mask drops empty slots). Each token is
    ``proj_cf(MF) + type_emb + (pos_emb for history)``. Queries cross-attend
    the full sequence in a single forward.

    Modes:

    - ``forward(user_cf, target_cf, history_cf, history_mask, text)`` —
      LLM-feeding joint forward with ``out_proj`` applied.
    - ``encode_cf(user_cf, target_cf, history_cf, history_mask)`` — queries
      only, no text branch.
    - ``encode_text(text)`` — text only, no queries, no cross-attention.
    - ``forward_multimodal(...)`` — joint forward returning query + text
      hidden states. ``causal_text=True`` is the ITG mask.
    """

    def __init__(
        self,
        d_cf: int,
        d_model: int,
        num_queries: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        cross_attention_frequency: int = 2,
        initializer_range: float = 0.02,
        qformer_text_model_name: str = "bert-base-uncased",
        max_instruction_length: int = 48,
        max_history_length: int = 10,
        init_from_pretrained_text: bool = True,
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
        self.max_history_length = int(max_history_length)
        self.output_dim = int(output_dim) if output_dim is not None else d_model
        self.max_instruction_length = int(max_instruction_length)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(
            qformer_text_model_name,
            truncation_side="right",
        )

        self.q = Parameter(torch.randn(1, num_queries, d_model))
        # CHANGE B: separate projectors for users vs items. User-factor and
        # item-factor geometry differ in MF space, so a single shared `proj_cf`
        # forces them into one map. `proj_user` handles the user slot;
        # `proj_item` handles the target + history slots (both are items).
        # Legacy `proj_cf` checkpoints are warm-loaded into BOTH (see
        # `load_state_dict`).
        self.proj_user = nn.Linear(d_cf, d_model)
        self.proj_item = nn.Linear(d_cf, d_model)
        self.type_emb = Parameter(torch.randn(3, d_model) * initializer_range)
        self.pos_emb = Parameter(torch.randn(self.max_history_length, d_model) * initializer_range)
        self.out_proj = nn.Identity() if self.output_dim == d_model else nn.Linear(d_model, self.output_dim)

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
        cross-attention. The Q-Former names self-attention modules
        ``encoder.layer.i.attention.attention.*`` while BERT uses
        ``encoder.layer.i.attention.self.*``; we remap accordingly. When the
        Q-Former has fewer layers than BERT, only the first ``num_layers``
        of BERT are copied. Tensors with mismatched shapes (e.g. when
        ``hidden_size`` or ``num_heads`` is configured differently from
        BERT-base) are skipped, leaving them at their random init.

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
            bert_key = q_key.replace(".attention.attention.", ".attention.self.")
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

        # CHANGE B: legacy checkpoints have a single `proj_cf.{weight,bias}`.
        # Warm-init BOTH new projectors from it so old Stage-1/2 checkpoints
        # still load. Only fill targets the checkpoint doesn't already provide.
        legacy_proj = {
            k: v for k, v in remapped_state_dict.items()
            if k == "proj_cf.weight" or k == "proj_cf.bias"
        }
        if legacy_proj:
            for legacy_key, value in legacy_proj.items():
                suffix = legacy_key.split(".", 1)[1]  # "weight" | "bias"
                for new_prefix in ("proj_user", "proj_item"):
                    new_key = f"{new_prefix}.{suffix}"
                    if new_key in expected_keys and new_key not in remapped_state_dict:
                        remapped_state_dict[new_key] = value.clone()
                remapped_state_dict.pop(legacy_key, None)

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

    def _build_multi_token_encoder(
        self,
        user_cf: torch.Tensor,
        target_cf: torch.Tensor,
        history_cf: torch.Tensor,
        history_mask: torch.Tensor,
        user_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ):
        """Build the [B, 2+H, d_model] cross-attention encoder sequence.

        Layout: token 0 = user, token 1 = target, tokens 2..2+H = history
        (chronological, left-padded). Each token is the shared ``proj_cf``
        projection plus its type embedding; history tokens also receive a
        per-slot positional embedding. ``user_mask`` and ``target_mask``
        default to all-ones (Stage 3 usage); Stage 1 single-anchor losses
        pass mask=0 to hide whichever slot is not present.
        """
        if user_cf.dim() != 2 or target_cf.dim() != 2:
            raise ValueError(
                f"Expected user_cf/target_cf as [B, d_cf]; got "
                f"{tuple(user_cf.shape)} / {tuple(target_cf.shape)}"
            )
        if history_cf.dim() != 3 or history_cf.size(1) != self.max_history_length:
            raise ValueError(
                f"Expected history_cf as [B, {self.max_history_length}, d_cf]; "
                f"got {tuple(history_cf.shape)}"
            )
        if history_mask.shape != history_cf.shape[:2]:
            raise ValueError(
                f"history_mask shape {tuple(history_mask.shape)} does not match "
                f"history_cf batch/length {tuple(history_cf.shape[:2])}"
            )

        batch_size = user_cf.size(0)
        device = user_cf.device

        user_tok = self.proj_user(user_cf) + self.type_emb[TYPE_USER]
        target_tok = self.proj_item(target_cf) + self.type_emb[TYPE_TARGET]
        history_tok = (
            self.proj_item(history_cf)
            + self.type_emb[TYPE_HISTORY]
            + self.pos_emb.unsqueeze(0)
        )

        encoder_hidden_states = torch.cat(
            [user_tok.unsqueeze(1), target_tok.unsqueeze(1), history_tok], dim=1
        )

        if user_mask is None:
            user_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        else:
            user_mask = user_mask.to(dtype=torch.long, device=device).view(batch_size, 1)
        if target_mask is None:
            target_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        else:
            target_mask = target_mask.to(dtype=torch.long, device=device).view(batch_size, 1)

        encoder_attention_mask = torch.cat(
            [user_mask, target_mask, history_mask.to(dtype=torch.long, device=device)],
            dim=1,
        )
        return encoder_hidden_states, encoder_attention_mask

    def pack_item_context(self, item_cf: torch.Tensor):
        """Stage 1 single-item helper: returns the 6-tuple
        ``(user_cf, target_cf, history_cf, history_mask, user_mask, target_mask)``
        with the item placed in the target slot and user/history masked out.
        """
        if item_cf.dim() != 2:
            raise ValueError(f"Expected item_cf as [B, d_cf]; got {tuple(item_cf.shape)}")
        batch_size = item_cf.size(0)
        device = item_cf.device
        zeros_cf = torch.zeros_like(item_cf)
        zero_hist = torch.zeros(
            batch_size, self.max_history_length, self.d_cf,
            dtype=item_cf.dtype, device=device,
        )
        zero_mask = torch.zeros(batch_size, self.max_history_length, dtype=torch.long, device=device)
        off = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        on = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        return zeros_cf, item_cf, zero_hist, zero_mask, off, on

    def pack_user_context(self, user_cf: torch.Tensor):
        """Stage 1 single-user helper: user in the user slot, target/history
        masked out."""
        if user_cf.dim() != 2:
            raise ValueError(f"Expected user_cf as [B, d_cf]; got {tuple(user_cf.shape)}")
        batch_size = user_cf.size(0)
        device = user_cf.device
        zeros_cf = torch.zeros_like(user_cf)
        zero_hist = torch.zeros(
            batch_size, self.max_history_length, self.d_cf,
            dtype=user_cf.dtype, device=device,
        )
        zero_mask = torch.zeros(batch_size, self.max_history_length, dtype=torch.long, device=device)
        on = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        off = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        return user_cf, zeros_cf, zero_hist, zero_mask, on, off

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
        user_cf: torch.Tensor,
        target_cf: torch.Tensor,
        history_cf: torch.Tensor,
        history_mask: torch.Tensor,
        user_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Queries-only forward over the multi-token CF sequence.

        Queries cross-attend the 12-token user/target/history sequence; no
        text-side input. Returns query hidden states of shape
        ``[B, num_queries, d_model]`` (``out_proj`` not applied).
        """
        batch_size = user_cf.size(0)
        query_tokens = self.q.expand(batch_size, -1, -1)
        query_attention_mask = torch.ones(
            batch_size, query_tokens.size(1), dtype=torch.long, device=user_cf.device
        )
        encoder_hidden_states, encoder_attention_mask = self._build_multi_token_encoder(
            user_cf, target_cf, history_cf, history_mask, user_mask, target_mask
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
        user_cf: torch.Tensor,
        target_cf: torch.Tensor,
        history_cf: torch.Tensor,
        history_mask: torch.Tensor,
        text: Union[str, list],
        causal_text: bool = False,
        max_text_length: Optional[int] = None,
        user_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ):
        """Joint forward returning ``(query_hidden, text_hidden, text_ids, text_mask)``.

        Queries + text self-attend together while cross-attending the
        12-token CF sequence. ``causal_text=True`` enables a causal mask on
        the text→text block (used by ITG); the default is bidirectional
        (used by ITM and by the LLM-feeding ``forward``).
        """
        batch_size = user_cf.size(0)
        text_list = self._normalize_text_input(text, batch_size)

        query_tokens = self.q.expand(batch_size, -1, -1)
        query_count = query_tokens.size(1)

        text_ids, text_attention_mask = self._tokenize(
            text_list, max_text_length or self.max_instruction_length, user_cf.device
        )

        encoder_hidden_states, encoder_attention_mask = self._build_multi_token_encoder(
            user_cf, target_cf, history_cf, history_mask, user_mask, target_mask
        )
        query_attention_mask = torch.ones(
            batch_size, query_count, dtype=torch.long, device=user_cf.device
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
        user_cf: torch.Tensor,
        target_cf: torch.Tensor,
        history_cf: torch.Tensor,
        history_mask: torch.Tensor,
        instruction,
        user_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """LLM-feeding mode: queries cross-attend the multi-token CF
        sequence while the text stream consumes ``instruction``. Returns
        query hidden states with ``out_proj`` applied: ``[B, num_queries,
        output_dim]``."""

        query_hidden, _, _, _ = self.forward_multimodal(
            user_cf,
            target_cf,
            history_cf,
            history_mask,
            instruction,
            causal_text=False,
            user_mask=user_mask,
            target_mask=target_mask,
        )
        return self.out_proj(query_hidden)
