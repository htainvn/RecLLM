import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

# Modules that only exist when the SeLLa Step-2 semantic alignment is enabled.
# Loading code uses this to reconcile checkpoints across the two variants.
ALIGNMENT_MODULE_PREFIXES = ("item_embedding_llm.", "trans_1.", "trans_2.")


class MatrixFactorization(nn.Module):
    """MF with optional SeLLa Step-2 semantic alignment.

    Alignment (SeLLa `codes/step2_train_collab/train_mf.py`) adds:
    - ``item_embedding_llm``: a TRAINABLE table initialised from the
      LLM-distilled item embeddings (``item_llm_emb.pt``), replacing the
      frozen-buffer role that bank plays elsewhere;
    - ``trans_1``/``trans_2`` (d_cf -> align_hidden -> GELU -> d_llm): the
      projection whose InfoNCE against ``item_embedding_llm`` pulls the CF
      item space toward the LLM semantic space. SeLLa Step 3 reuses these
      weights to initialise its collaborative projection
      (``pretrained_with_small=True``); the Stage-3 direct-ID projections
      here warm-start from them the same way.

    Enabled via ``rec_config.item_llm_emb_path`` (+ optional
    ``align_hidden_size``, ``align_tau``). Without it the model is exactly
    the original 2-table MF and ``forward`` is unchanged.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)

        self.align_tau = float(config.get("align_tau", 0.2)) if hasattr(config, "get") else 0.2
        item_llm_emb_path = config.get("item_llm_emb_path", None) if hasattr(config, "get") else None
        self.has_alignment = bool(item_llm_emb_path)
        if self.has_alignment:
            self._build_alignment(item_llm_emb_path)

    def _build_alignment(self, item_llm_emb_path):
        if not os.path.exists(item_llm_emb_path):
            raise FileNotFoundError(
                f"rec_config.item_llm_emb_path set but not found: {item_llm_emb_path}"
            )
        blob = torch.load(item_llm_emb_path, map_location="cpu")
        table = (blob["item_llm_emb"] if isinstance(blob, dict) else blob).float()
        if table.size(0) < self.config.item_num:
            raise ValueError(
                f"item_llm_emb table covers {table.size(0)} items, "
                f"need at least item_num={self.config.item_num}"
            )
        table = table[: self.config.item_num]

        align_hidden = int(self.config.get("align_hidden_size", 1024)) if hasattr(self.config, "get") else 1024
        d_llm = table.size(1)
        # SeLLa Step 2: trainable LLM-semantic item table + the trans MLP that
        # pulls CF item embeddings toward it.
        self.item_embedding_llm = nn.Embedding.from_pretrained(
            table, freeze=False, padding_idx=self.padding_index
        )
        self.trans_1 = nn.Linear(self.config.embedding_size, align_hidden, bias=True)
        self.gelu = nn.GELU()
        self.trans_2 = nn.Linear(align_hidden, d_llm, bias=True)
        LOGGER.info(
            "MF semantic alignment active: item_embedding_llm=%s, trans %d->%d->%d, tau=%s",
            tuple(table.shape), self.config.embedding_size, align_hidden, d_llm, self.align_tau,
        )

    def load_state_dict(self, state_dict, strict=True):
        """Reconcile checkpoints across the plain / alignment variants.

        - Plain model + alignment checkpoint (Stage 1/2/3 consume an MF
          trained with alignment): drop the alignment keys, load the tables.
        - Alignment model + plain checkpoint: load the tables, keep the
          alignment modules at init, and say so.
        """
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        has_align_keys = any(
            k.startswith(ALIGNMENT_MODULE_PREFIXES) for k in state_dict
        )
        if has_align_keys and not self.has_alignment:
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith(ALIGNMENT_MODULE_PREFIXES)
            }
            LOGGER.info(
                "MF checkpoint carries alignment keys (SeLLa Step-2 trained); "
                "this model has no alignment modules — loading tables only."
            )
        elif not has_align_keys and self.has_alignment:
            LOGGER.warning(
                "MF checkpoint has no alignment keys; item_embedding_llm/trans_* "
                "keep their init."
            )
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def user_encoder(self, user_ids):
        return self.user_embedding(user_ids)

    def item_encoder(self, item_ids):
        return self.item_embedding(item_ids)

    def compute(self):
        return None, None

    def align_item_cf_to_llm(self, item_ids):
        """trans_2(GELU(trans_1(e_i))) — the CF->LLM projection the InfoNCE trains."""
        return self.trans_2(self.gelu(self.trans_1(self.item_embedding(item_ids))))

    def alignment_loss(self, item_ids):
        """SeLLa Step-2 InfoNCE between the trainable LLM-semantic item table
        and the projected CF item embeddings (both L2-normalised, in-batch
        negatives). Matches SeLLa's ``MatrixFactorization.InfoNCE`` including
        its 0.01-power tempering of the pos/neg terms."""
        if not self.has_alignment:
            raise RuntimeError("alignment_loss called but rec_config.item_llm_emb_path is not set")
        out_1 = F.normalize(self.item_embedding_llm(item_ids), dim=1)
        out_2 = F.normalize(self.align_item_cf_to_llm(item_ids), dim=1)

        similarity = torch.exp(torch.mm(out_1, out_2.t()) / self.align_tau)
        neg = torch.sum(similarity, 1)
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.align_tau)

        neg = ((0.1 * (pos + neg)) ** 0.01) / 0.01
        pos = -(pos ** 0.01) / 0.01
        return pos.mean() + neg.mean()

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        matching = torch.mul(user_embeddings, item_embeddings).sum(dim=-1)
        return matching
