import torch
import torch.nn as nn
import torch.nn.functional as F


class QRecInstructAlignmentModel(nn.Module):
    """Stage-1 Q-Former alignment with BLIP-2 ITC + ITM + ITG objectives,
    adapted for recommendation. The "I" in ITC/ITM/ITG stands for **item**:
    queries cross-attend to a single CF (collaborative filtering) vector
    from the frozen MF encoder rather than to image patch features.

    Uses the Q-Former's own text branch for both contrastive and language
    modelling losses, removing the external BERT encoder. Heads on top of
    Q-Former outputs:

    - ITC contrastive: text [CLS] from ``encode_text`` against pooled query
      output from ``encode_cf`` (BLIP-2 uni-modal setup).
    - ITM binary: pooled query output from a joint forward (cf + text) into
      ``itm_head``. Negative pairs are drawn in-batch by hard-mining on the
      ITC similarity matrix.
    - ITG causal LM: text hidden states from a joint forward with causal
      text-side mask into ``lm_head`` (weight-tied to the Q-Former token
      embedding table).

    Item-item collaborative pairs (``loss_item_item_ilm``) remain available
    as a SigLLM-specific addition on top of the BLIP-2 head set.
    """

    def __init__(self, mf, qformer) -> None:
        super().__init__()
        self.mf = mf
        self.qformer = qformer

        d_model = qformer.d_model
        vocab_size = qformer.vocab_size

        self.itm_head = nn.Linear(d_model, 2)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = qformer.text_word_embeddings.weight

    @staticmethod
    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    def encode_item_queries(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_cf = self.mf.item_encoder(item_ids)
        ctx = self.qformer.pack_item_context(item_cf)
        return self.qformer.encode_cf(*ctx)

    def encode_user_queries(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_cf = self.mf.user_encoder(user_ids)
        ctx = self.qformer.pack_user_context(user_cf)
        return self.qformer.encode_cf(*ctx)

    def encode_text_cls(self, text_list) -> torch.Tensor:
        _, text_cls = self.qformer.encode_text(text_list)
        return text_cls

    @staticmethod
    def _select_query_by_text(query_hidden: torch.Tensor, text_vec: torch.Tensor):
        """Pick the query whose normalized similarity with the text CLS is
        highest (BLIP-2 query selection)."""

        q_norm = QRecInstructAlignmentModel.l2norm(query_hidden)
        t_norm = QRecInstructAlignmentModel.l2norm(text_vec)
        scores = torch.einsum("bqd,bd->bq", q_norm, t_norm)
        selected_idx = scores.argmax(dim=1)
        batch_idx = torch.arange(query_hidden.size(0), device=query_hidden.device)
        return query_hidden[batch_idx, selected_idx], selected_idx

    @staticmethod
    def select_pair_by_similarity(left_tokens: torch.Tensor, right_tokens: torch.Tensor):
        """Pick the closest pair of queries across two query bags (used by
        the item-item contrastive)."""

        if left_tokens.dim() != 3 or right_tokens.dim() != 3:
            raise ValueError(
                "Expected pair inputs to have shape [B, Q, D], "
                f"got left={tuple(left_tokens.shape)}, right={tuple(right_tokens.shape)}"
            )
        if left_tokens.shape != right_tokens.shape:
            raise ValueError("Pair shapes must match")

        left = QRecInstructAlignmentModel.l2norm(left_tokens)
        right = QRecInstructAlignmentModel.l2norm(right_tokens)
        scores = torch.einsum("bqd,brd->bqr", left, right)
        flat = scores.flatten(start_dim=1).argmax(dim=1)
        r = right_tokens.size(1)
        left_idx, right_idx = flat // r, flat % r
        b = torch.arange(left_tokens.size(0), device=left_tokens.device)
        return left_tokens[b, left_idx], right_tokens[b, right_idx], left_idx, right_idx

    def loss_itc(
        self,
        item_ids: torch.Tensor,
        text_list,
        tau: float = 0.07,
        symmetric: bool = True,
    ):
        """ITC: item-text contrastive with uni-modal forwards.

        Returns ``(loss, sim_matrix, accuracy)``. ``sim_matrix`` is reused
        downstream for hard-negative mining in ITM.
        """
        query_hidden = self.encode_item_queries(item_ids)
        text_cls = self.encode_text_cls(text_list)

        selected_query, _ = self._select_query_by_text(query_hidden, text_cls)

        q_norm = self.l2norm(selected_query)
        t_norm = self.l2norm(text_cls)
        sim_matrix = (q_norm @ t_norm.T) / tau

        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_q2t = F.cross_entropy(sim_matrix, labels)
        if symmetric:
            loss_t2q = F.cross_entropy(sim_matrix.T, labels)
            loss = (loss_q2t + loss_t2q) / 2.0
        else:
            loss = loss_q2t

        accuracy = (sim_matrix.argmax(dim=1) == labels).float().mean()
        return loss, sim_matrix.detach(), accuracy

    def loss_itm(
        self,
        item_ids: torch.Tensor,
        text_list,
        itc_sim_matrix: torch.Tensor,
        num_neg: int = 1,
    ):
        """ITM: binary item-text matching with hard negatives mined from the
        ITC similarity matrix.

        ``itc_sim_matrix`` is the similarity matrix returned by ``loss_itc``;
        we mask the diagonal and sample ``num_neg`` hard negative texts per item
        and ``num_neg`` hard negative items per text (CHANGE 2e). With the
        default ``num_neg=1`` this is the original BLIP-2 single-hard-negative
        ITM; larger values give the matching head more (and harder) contrastive
        pressure, which matters when item-text batches are small.
        """
        batch_size = item_ids.size(0)
        device = item_ids.device

        # Can draw at most batch_size-1 distinct off-diagonal negatives per row.
        k = max(1, min(int(num_neg), batch_size - 1))

        weights_t2q = F.softmax(itc_sim_matrix, dim=0) + 1e-4
        weights_q2t = F.softmax(itc_sim_matrix, dim=1) + 1e-4
        diag = torch.eye(batch_size, device=device, dtype=torch.bool)
        weights_t2q = weights_t2q.masked_fill(diag, 0.0)
        weights_q2t = weights_q2t.masked_fill(diag, 0.0)

        # [B, k] hard negatives (sampled without replacement within each row).
        neg_text_idx = torch.multinomial(weights_q2t, k, replacement=False)
        neg_item_idx = torch.multinomial(weights_t2q.T, k, replacement=False)

        pos_item_cf = self.mf.item_encoder(item_ids)                       # [B, d]
        # Negative-text branch: positive item repeated k times, paired with k
        # hard-negative texts. repeat_interleave matches the row-major (b, j)
        # flattening of neg_text_idx so item b lines up with its k neg texts.
        item_cf_for_neg_text = pos_item_cf.repeat_interleave(k, dim=0)     # [B*k, d]
        neg_item_cf = self.mf.item_encoder(item_ids[neg_item_idx.reshape(-1)])  # [B*k, d]

        text_list_local = list(text_list)
        pos_text = text_list_local
        neg_text_flat = [text_list_local[i] for i in neg_text_idx.reshape(-1).detach().cpu().tolist()]
        pos_text_for_neg_item = [text_list_local[b] for b in range(batch_size) for _ in range(k)]

        cf_concat = torch.cat([pos_item_cf, item_cf_for_neg_text, neg_item_cf], dim=0)
        text_concat = pos_text + neg_text_flat + pos_text_for_neg_item

        user_cf, target_cf, history_cf, history_mask, u_mask, t_mask = (
            self.qformer.pack_item_context(cf_concat)
        )
        query_hidden, _, _, _ = self.qformer.forward_multimodal(
            user_cf, target_cf, history_cf, history_mask,
            text_concat, causal_text=False,
            user_mask=u_mask, target_mask=t_mask,
        )
        pooled = query_hidden.mean(dim=1)
        logits = self.itm_head(pooled)

        labels = torch.cat(
            [
                torch.ones(batch_size, dtype=torch.long, device=device),
                torch.zeros(2 * batch_size * k, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    def loss_itg(self, item_ids: torch.Tensor, text_list):
        """ITG: causal next-token LM on the text branch, conditioned on the
        item CF vector via cross-attention through the learned queries."""

        item_cf = self.mf.item_encoder(item_ids)
        user_cf, target_cf, history_cf, history_mask, u_mask, t_mask = (
            self.qformer.pack_item_context(item_cf)
        )
        _, text_hidden, text_ids, text_attention_mask = self.qformer.forward_multimodal(
            user_cf, target_cf, history_cf, history_mask,
            text_list, causal_text=True,
            user_mask=u_mask, target_mask=t_mask,
        )

        logits = self.lm_head(text_hidden[:, :-1, :])
        targets = text_ids[:, 1:].clone()
        target_mask = text_attention_mask[:, 1:].bool()
        targets = targets.masked_fill(~target_mask, -100)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == text_ids[:, 1:]) & target_mask).float().sum()
            total = target_mask.float().sum().clamp(min=1.0)
            token_accuracy = correct / total
        return loss, token_accuracy

    def loss_item_item_ilm(self, left_ids: torch.Tensor, right_ids: torch.Tensor, tau: float = 0.07):
        """SigLLM-specific co-watch item-item contrastive."""
        left_cf = self.mf.item_encoder(left_ids)
        right_cf = self.mf.item_encoder(right_ids)
        left_q = self.qformer.encode_cf(*self.qformer.pack_item_context(left_cf))
        right_q = self.qformer.encode_cf(*self.qformer.pack_item_context(right_cf))
        left_sel, right_sel, _, _ = self.select_pair_by_similarity(left_q, right_q)

        left_norm = self.l2norm(left_sel)
        right_norm = self.l2norm(right_sel)
        logits = (left_norm @ right_norm.T) / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    def loss_user_item(self, user_ids: torch.Tensor, item_ids: torch.Tensor, tau: float = 0.07):
        """User-item contrastive (ILM-style). Pulls Q-Former representation of
        a user toward their positively-interacted item via in-batch InfoNCE.
        Mirrors ``loss_item_item_ilm`` but uses ``mf.user_encoder`` on the left
        side. On ML1M this captures the dominant CF signal (888K positive
        user-item pairs vs 3K item-text pairs in our pkls)."""
        user_cf = self.mf.user_encoder(user_ids)
        item_cf = self.mf.item_encoder(item_ids)
        user_q = self.qformer.encode_cf(*self.qformer.pack_user_context(user_cf))
        item_q = self.qformer.encode_cf(*self.qformer.pack_item_context(item_cf))
        user_sel, item_sel, _, _ = self.select_pair_by_similarity(user_q, item_q)

        user_norm = self.l2norm(user_sel)
        item_norm = self.l2norm(item_sel)
        logits = (user_norm @ item_norm.T) / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy
