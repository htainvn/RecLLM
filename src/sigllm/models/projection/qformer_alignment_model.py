from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from sigllm.models.projection.soft_token_proj import build_soft_token_projection


LLM_EMB_NORMALIZERS = ("none", "center", "whiten")


def _per_user_pairwise_loss(scores, users, labels, tau=1.0):
    """-log sigmoid((s_pos - s_neg)/tau) over same-user pairs, USER-weighted.

    Mirrors ``QRecLLM._per_user_pairwise_loss``: pair losses are averaged WITHIN
    each user first, then across users, so the objective matches uAUC (which
    weights users equally) rather than global AUC (which would weight users by
    their pair count). Returns a graph-preserving 0 when a batch holds no valid
    same-user pos/neg pair, so unlucky batches cannot NaN the run.
    """
    users = users.view(-1)
    labels = labels.view(-1).long()
    pos_mask = labels == 1
    neg_mask = labels == 0

    diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    valid = (users.unsqueeze(1) == users.unsqueeze(0)) & pos_mask.unsqueeze(1) & neg_mask.unsqueeze(0)
    if valid.sum() == 0:
        return scores.sum() * 0.0

    valid_f = valid.to(scores.dtype)
    pair_losses = F.softplus(-diff / tau) * valid_f
    row_loss = pair_losses.sum(dim=1)
    row_count = valid_f.sum(dim=1)

    uniq, inv = torch.unique(users, return_inverse=True)
    inv = inv.to(scores.device)
    loss_sum = torch.zeros(uniq.numel(), dtype=scores.dtype, device=scores.device)
    cnt = torch.zeros(uniq.numel(), dtype=scores.dtype, device=scores.device)
    loss_sum.scatter_add_(0, inv, row_loss)
    cnt.scatter_add_(0, inv, row_count)
    has = cnt > 0
    return (loss_sum[has] / cnt[has]).mean()


@torch.no_grad()
def _per_user_pairwise_acc(scores, users, labels):
    """Fraction of same-user (pos, neg) pairs ordered correctly — a direct,
    unsmoothed read of uAUC on this batch. Reported next to the loss because the
    loss value alone cannot be compared to chance; this can (0.5 = chance)."""
    users = users.view(-1)
    labels = labels.view(-1).long()
    diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    valid = (users.unsqueeze(1) == users.unsqueeze(0)) & (labels == 1).unsqueeze(1) & (labels == 0).unsqueeze(0)
    total = valid.sum()
    if total == 0:
        return float("nan")
    correct = ((diff > 0) & valid).sum() + 0.5 * ((diff == 0) & valid).sum()
    return float(correct / total)


def normalize_item_llm_emb(emb: torch.Tensor, mode: str = "center") -> torch.Tensor:
    """Condition the frozen item-LLM embedding bank for the cosine InfoNCE.

    Shared by Stage 1 (``loss_llm_align``) and Stage 2 (the ``w_llm`` alignment
    keep-alive). They MUST use the same mode: Stage 2 warm-starts ``llm_proj``
    from Stage 1's ``llm_align_proj``, so if the two stages condition the bank
    differently the keep-alive pulls the projection toward a different geometry
    than the one Stage 1 aligned to and actively undoes the alignment.

    ``center`` removes the common component. Mask-mean input embeddings share a
    large one (prompt scaffold + genre tokens; off-diag cosine ~0.88 measured on
    ml-1m), which collapses all targets into a narrow cone and leaves the
    contrastive nothing to discriminate.

    ``whiten`` additionally rescales each dimension to unit variance. Centering
    alone was not enough on ml-1m: L_llm plateaued ~0.33 nats below ln(n) on
    BOTH train and val — it could not fit even the training targets, so this is
    target geometry rather than model capacity — while ITC on the same queries
    reached ~2.3 nats. Residual per-dim variance is very uneven, so a handful of
    high-variance dims decide the cosine and the rest carry no discriminative
    budget. Full ZCA would also decorrelate, but needs a ``d_llm x d_llm``
    eigendecomposition estimated from ~3k covered items, which is not reliably
    conditioned.

    Trade-off: every mode except ``none`` moves the target off the raw LLM
    input-embedding direction, and Stage 2/3 inherit that geometry through
    ``llm_align_proj`` -> ``llm_proj``. The loss only constrains direction (both
    sides are l2-normalized) so scale is free, but fall back to ``center`` if
    soft tokens regress at Stage 3.

    Uncovered items (zero rows) are excluded from the statistics and stay zero,
    so they remain distinguishable as "no text available".

    Returns a new tensor; ``emb`` is not modified in place.
    """
    if mode not in LLM_EMB_NORMALIZERS:
        raise ValueError(f"mode must be one of {LLM_EMB_NORMALIZERS}, got {mode!r}")

    out = emb.float().clone()
    if mode == "none":
        return out

    covered = out.norm(dim=-1) > 0
    if not bool(covered.any()):
        return out

    centered = out[covered] - out[covered].mean(dim=0, keepdim=True)
    # Needs >=2 covered rows: torch's std is Bessel-corrected, so a single row
    # yields nan and would silently poison every target. One covered item is
    # already all-zero after centering, so there is nothing to rescale anyway.
    if mode == "whiten" and centered.size(0) >= 2:
        centered = centered / centered.std(dim=0, keepdim=True).clamp(min=1e-6)
    out[covered] = centered
    return out


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

    def __init__(
        self,
        mf,
        qformer,
        item_llm_emb=None,
        d_llm=None,
        llm_emb_normalize="center",
        item_sem_emb=None,
        sem_dropout=0.5,
        pair_logit_center=False,
        itc_logit_center=True,
        bpr_logit_center=True,
        sem_for_text_losses=False,
        center_soft_tokens=True,
    ) -> None:
        super().__init__()
        self.mf = mf
        self.qformer = qformer
        # Batch-center the pooled vectors of the collaborative InfoNCE terms
        # (L_ii / L_ui) before the cosine. The old default was OFF on evidence
        # collected at tau=0.07 (L_ii 24.4 against ln(n) 6.4), but that
        # comparison was confounded: measured on the raw-MF ceiling, RAW at
        # tau=0.07 is ALSO worse than chance (gain -1.05), so tau was the
        # problem, not centering. At tau=0.2 centering is 2.7x better on gain
        # and 2.5x on top1 for the history-pooled user side. Now defaults ON in
        # config, together with tau_ui=0.2 / tau_ii=0.5 — the three must move
        # together. See _pooled_pair_logits and diagnose_collab_collapse.py.
        self.pair_logit_center = bool(pair_logit_center)
        self.itc_logit_center = bool(itc_logit_center)
        # Centering for the candidate-conditioned BPR, which fails the OPPOSITE
        # way: softplus sees only s_neg - s_pos (no logsumexp to blow up), and
        # with a saturated cosine that difference is ~0, so user_proj gets no
        # gradient and L_uic sits at ln 2. Default ON. See loss_user_item.
        self.bpr_logit_center = bool(bpr_logit_center)
        # Feed the semantic bank to ITC/ITM/ITG/L_llm too? Default False: for
        # those four the bank IS the target, not side information. See
        # _sem_for_text_loss for the measured numbers.
        self.sem_for_text_losses = bool(sem_for_text_losses)
        # Optional semantic cross-attention source (see HFQFormerAdapter.d_sem):
        # the RAW distilled bank, deliberately NOT normalize_item_llm_emb'd —
        # normalization conditions the contrastive TARGET geometry; the source
        # goes through its own trainable proj_sem, which absorbs scale/offset.
        # ``sem_dropout`` zeroes whole semantic rows per sample during training
        # (zero rows are masked out downstream), forcing the CF path to stay
        # trained so warm items don't lose CF grounding to the easier semantic
        # shortcut.
        self.sem_dropout = float(sem_dropout)
        # Runtime kill switch for the semantic source, independent of
        # ``self.training``. The dropout gate is train-only, so evaluation
        # always ran with the semantic source at FULL strength — and since that
        # source is a different projection of the very captions the text/LLM
        # objectives target, a val number measured with it on cannot be read as
        # "what the CF path learned". Flip this off (see ``sem_disabled``) for a
        # CF-only diagnostic pass.
        self.sem_enabled = True
        if item_sem_emb is not None:
            sem = item_sem_emb if isinstance(item_sem_emb, torch.Tensor) else item_sem_emb.weight
            self.register_buffer("item_sem_emb", sem.float().clone(), persistent=False)
        else:
            self.item_sem_emb = None

        d_model = qformer.d_model
        vocab_size = qformer.vocab_size

        self.itm_head = nn.Linear(d_model, 2)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = qformer.text_word_embeddings.weight

        self.has_llm_align = item_llm_emb is not None
        self.llm_emb_normalize = llm_emb_normalize
        if self.has_llm_align:
            emb = item_llm_emb if isinstance(item_llm_emb, torch.Tensor) else item_llm_emb.weight
            # Stage 2's keep-alive must condition the bank identically — see
            # normalize_item_llm_emb for why, and for the mode trade-offs.
            emb = normalize_item_llm_emb(emb, llm_emb_normalize)
            d_llm = int(d_llm) if d_llm is not None else int(emb.size(-1))
            if emb.size(-1) != d_llm:
                raise ValueError(
                    f"Item LLM embedding dimension {emb.size(-1)} does not match "
                    f"specified d_llm={d_llm}"
                )
            self.register_buffer("item_llm_emb", emb, persistent=False)
            # Injection-identical head: the same Linear+LayerNorm stack (and
            # init) as the Stage-2/3 ``llm_proj``, applied per query token on
            # ``out_proj`` output. Stage-1 exports its state dict so Stage-2
            # warm-starts ``llm_proj`` from it — the alignment must live in the
            # projection the frozen LLM actually reads, not a throwaway head.
            d_q = qformer.output_dim
            # Shared factory: Stage 2 warm-starts llm_proj from this state dict
            # under a strict load, so the layout (including the centering module)
            # has to be identical on both sides.
            self.llm_align_proj = build_soft_token_projection(
                d_q, d_llm,
                center=bool(center_soft_tokens),
                num_positions=int(qformer.num_queries),
            )
        else:
            self.item_llm_emb = None
            self.llm_align_proj = None

    @staticmethod
    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    @contextmanager
    def sem_disabled(self):
        """Temporarily force the CF-only path (semantic source off).

        Used for the diagnostic eval pass: the ``sem_dropout`` gate is
        train-only, so a normal ``model.eval()`` measures every objective with
        the semantic source fully on. Comparing that against this pass separates
        "the CF representation learned this" from "the semantic source carried
        it", which matters because the source is derived from the same captions
        the text and LLM-alignment objectives target.
        """
        previous = self.sem_enabled
        self.sem_enabled = False
        try:
            yield
        finally:
            self.sem_enabled = previous

    def _sem_for(self, item_ids: torch.Tensor):
        """Semantic source rows for ``item_ids`` with training-time row dropout.

        Dropped rows become all-zero, which the adapter masks out — the sample
        falls back to CF-only cross-attention, exactly the cold-path geometry.
        """
        if self.item_sem_emb is None or not self.sem_enabled:
            return None
        sem = self.item_sem_emb[item_ids]
        if self.training and self.sem_dropout > 0.0:
            keep = (
                torch.rand(sem.shape[:-1], device=sem.device) >= self.sem_dropout
            ).to(sem.dtype).unsqueeze(-1)
            sem = sem * keep
        return sem

    def _sem_for_text_loss(self, item_ids: torch.Tensor):
        """Semantic source for the objectives whose TARGET derives from the item
        caption: ITC, ITM, ITG and L_llm.

        Returns ``None`` unless ``sem_for_text_losses`` is explicitly enabled,
        because for these four the semantic bank is not side information — it is
        the answer. The bank is the frozen LLM's own representation of the very
        caption ITC contrasts against, at 3584 dims and mean norm ~289 next to a
        256-dim MF vector, so the cheapest way to minimise these losses is to
        copy the bank through proj_sem and ignore CF entirely. Measured on ml-1m
        at epoch 1 with the bank on for every loss:

            g_itc   sem_on +3.013 / sem_off -0.633   (train)
            g_itc   sem_on +2.878 / sem_off -0.797   (val)
            g_llm   sem_on +1.051 / sem_off +0.013   (val)

        i.e. the CF path had learned nothing at all — below chance — while the
        headline numbers looked strong. Raising ``sem_dropout`` only changes the
        FRACTION of rows on which the shortcut is available, not the fact that it
        wins wherever it is; withholding the bank from these losses removes it.

        The collaborative terms (``loss_item_item_ilm``, ``loss_user_item``) keep
        the bank via ``_sem_for``: their targets are co-watch / interaction CF,
        not the caption, so there the bank is genuinely auxiliary — and that is
        enough to keep ``proj_sem`` trained for Stage 3.

        Consequence for the diagnostic: with this off (the default), the
        sem_on / sem_off item_text passes become identical BY CONSTRUCTION, and
        the selection metric ``val_sem_off_gain_itc`` measures the CF-only path
        either way. Two identical passes is the expected signature here, not a
        bug — if they ever differ, the bank has leaked back into a text loss.
        """
        if not self.sem_for_text_losses:
            return None
        return self._sem_for(item_ids)

    def encode_item_queries(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Item queries for the CAPTION-TARGETED losses (ITC via loss_itc,
        L_llm via loss_llm_align). Uses ``_sem_for_text_loss``, so by default the
        semantic bank is withheld here — see that method for why."""
        item_cf = self.mf.item_encoder(item_ids)
        # apply_residual=False: this path feeds the CAPTION-targeted losses
        # (ITC, L_llm). They see body+residual otherwise, and since the fixed
        # random projection of e_u/e_t is pure interference for "reproduce this
        # item's caption", the cheapest way for them to improve is to CANCEL it —
        # measured as the channel's within-user uAUC falling 0.583 -> 0.532 over
        # three epochs while every caption metric rose. The residual is a
        # serving-time bypass for the ranking task; the caption losses must not
        # be able to reach it.
        return self.qformer.encode_cf(
            item_cf, sem_vec=self._sem_for_text_loss(item_ids), apply_residual=False
        )

    def encode_text_cls(self, text_list) -> torch.Tensor:
        _, text_cls = self.qformer.encode_text(text_list)
        return text_cls

    @staticmethod
    def _max_query_logits(query_tokens: torch.Tensor, target_vecs: torch.Tensor, tau: float, center: bool = False,):
        """In-batch logits between a query BAG and a set of single target
        vectors: ``logits[b, c] = max_q cos(query_tokens[b, q], target_vecs[c])``.

        This is the BLIP-2 ITC shape and the ONLY unbiased max-over-queries
        form available here. Two failure modes it avoids:

        1. Selecting the best query against the POSITIVE target and reusing
           that index for the negatives handicaps the negatives — top1 and
           every retrieval number come out inflated.
        2. Taking the max over the full Q x Q pair grid between two query bags
           collapses the loss. Matching query indices sit near cosine 1
           regardless of input in this model (measured here), so ``max_{q,r}``
           picks q == r at ~1.0 for EVERY candidate, the logit row goes flat
           and the gradient vanishes — an earlier run pinned L_ui at ln 2 with
           accuracy below 0.5 for 34 epochs that way, while its
           ``1/tau``-scale gradient noise degraded the shared body.

        The fix is asymmetry: one side keeps its Q tokens, the other is
        mean-pooled to a single vector, so no query-index-matching shortcut
        exists and every candidate is still scored by its own best query.
        """
        if center and query_tokens.size(0) >= 4:
            query_tokens = query_tokens - query_tokens.mean(dim=0, keepdim=True)
            target_vecs = target_vecs - target_vecs.mean(dim=0, keepdim=True)
        q_norm = QRecInstructAlignmentModel.l2norm(query_tokens)               # [B, Q, D]
        t_norm = QRecInstructAlignmentModel.l2norm(target_vecs)                # [C, D]
        return torch.einsum("bqd,cd->bcq", q_norm, t_norm).max(dim=-1).values / tau

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

        # BLIP-2 ITC: score EVERY (item, text) candidate pair as the max over
        # the item's queries (see _max_query_logits), so negatives get the same
        # best-query treatment as the positive.
        sim_matrix = self._max_query_logits(query_hidden, text_cls, tau, center=self.itc_logit_center)

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
    ):
        """ITM: binary item-text matching with hard negatives mined from the
        ITC similarity matrix.

        ``itc_sim_matrix`` is the similarity matrix returned by ``loss_itc``;
        we mask the diagonal and sample one hard negative text per item and
        one hard negative item per text.
        """
        batch_size = item_ids.size(0)
        device = item_ids.device

        weights_t2q = F.softmax(itc_sim_matrix, dim=0) + 1e-4
        weights_q2t = F.softmax(itc_sim_matrix, dim=1) + 1e-4
        diag = torch.eye(batch_size, device=device, dtype=torch.bool)
        weights_t2q = weights_t2q.masked_fill(diag, 0.0)
        weights_q2t = weights_q2t.masked_fill(diag, 0.0)

        neg_text_idx = torch.multinomial(weights_q2t, 1).squeeze(1)
        neg_item_idx = torch.multinomial(weights_t2q.T, 1).squeeze(1)

        pos_item_cf = self.mf.item_encoder(item_ids)
        neg_item_cf = self.mf.item_encoder(item_ids[neg_item_idx])

        text_list_local = list(text_list)
        pos_text = text_list_local
        neg_text = [text_list_local[i] for i in neg_text_idx.detach().cpu().tolist()]

        cf_concat = torch.cat([pos_item_cf, pos_item_cf, neg_item_cf], dim=0)
        text_concat = pos_text + neg_text + pos_text
        ids_concat = torch.cat([item_ids, item_ids, item_ids[neg_item_idx]], dim=0)

        query_hidden, _, _, _ = self.qformer.forward_multimodal(
            cf_concat, text_concat, causal_text=False,
            sem_vec=self._sem_for_text_loss(ids_concat),
            apply_residual=False,   # ITM is caption-targeted; see encode_item_queries
        )
        pooled = query_hidden.mean(dim=1)
        logits = self.itm_head(pooled)

        labels = torch.cat(
            [
                torch.ones(batch_size, dtype=torch.long, device=device),
                torch.zeros(2 * batch_size, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    @staticmethod
    def _title_char_spans(text_list):
        """Character span of the item title inside the stage-1 caption template
        ``"Title: {title}. Genres: {...}."``. Falls back to the whole string
        when the template markers are absent."""
        spans = []
        for text in text_list:
            prefix = "Title: "
            start = len(prefix) if text.startswith(prefix) else 0
            end = text.find(". Genres:")
            if end < 0:
                end = len(text)
            spans.append((start, end))
        return spans

    def _title_target_mask(self, text_list, target_mask: torch.Tensor):
        """Boolean mask over the NEXT-TOKEN targets that lie inside the title
        span. Needs a fast tokenizer for offset mappings; returns None when
        unavailable so callers can fall back to the overall accuracy."""
        tokenizer = self.qformer.qformer_tokenizer
        if not getattr(tokenizer, "is_fast", False):
            return None
        enc = tokenizer(
            list(text_list),
            padding=True,
            truncation=True,
            max_length=self.qformer.max_instruction_length,
            return_offsets_mapping=True,
        )
        offsets = torch.tensor(enc["offset_mapping"], device=target_mask.device)  # [B, T, 2]
        spans = self._title_char_spans(text_list)
        starts = torch.tensor([s for s, _ in spans], device=target_mask.device).unsqueeze(1)
        ends = torch.tensor([e for _, e in spans], device=target_mask.device).unsqueeze(1)
        in_title = (
            (offsets[..., 0] >= starts)
            & (offsets[..., 1] <= ends)
            & (offsets[..., 1] > offsets[..., 0])  # excludes special tokens (0, 0)
        )
        return in_title[:, 1:] & target_mask

    def loss_itg(self, item_ids: torch.Tensor, text_list):
        """ITG: causal next-token LM on the text branch, conditioned on the
        item CF (and optional semantic) source via cross-attention through the
        learned queries.

        Returns ``(loss, token_accuracy, title_accuracy)``. The caption
        template ``"Title: {}. Genres: {}."`` makes most tokens (scaffold +
        genres) predictable without any item knowledge, so the overall
        ``token_accuracy`` is inflated; ``title_accuracy`` counts only the
        title-span targets — the part that actually requires item identity."""

        item_cf = self.mf.item_encoder(item_ids)
        _, text_hidden, text_ids, text_attention_mask = self.qformer.forward_multimodal(
            item_cf, text_list, causal_text=True,
            sem_vec=self._sem_for_text_loss(item_ids),
            apply_residual=False,   # ITG is caption-targeted; see encode_item_queries
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
            hits = (preds == text_ids[:, 1:])
            correct = (hits & target_mask).float().sum()
            total = target_mask.float().sum().clamp(min=1.0)
            token_accuracy = correct / total

            title_mask = self._title_target_mask(text_list, target_mask)
            if title_mask is None:
                title_accuracy = token_accuracy.clone()
            else:
                title_correct = (hits & title_mask).float().sum()
                title_total = title_mask.float().sum().clamp(min=1.0)
                title_accuracy = title_correct / title_total
        return loss, token_accuracy, title_accuracy

    def _pooled_pair_logits(self, left_tokens: torch.Tensor, right_tokens: torch.Tensor, tau: float):
        """In-batch logits between two query bags: MEAN-POOL both sides, then
        (batch-)center before the cosine.

        Both design choices are load-bearing, and the tempting alternatives are
        all worse. Simulated on this model's regime — Q tokens carrying a large
        direction shared across the batch plus a smaller input-dependent part,
        B=32, tau=0.07, chance top1 = 0.031, ln(B) = 3.47 — over the ratio of
        input-dependent to shared signal:

            form                     CE (in_sig 0.05 -> 1.0)   top1
            max over Q x R grid      3.45 -> 0.31              1.000
            max-left vs mean-right   3.46 -> 2.04              0.06 -> 0.57
            mean/mean (no center)    3.44 -> 0.15              1.000
            mean/mean + center       0.00 -> 0.00              1.000

        - Max over the full ``Q x R`` grid pins CE at ``ln(B)`` whenever the
          shared component dominates: ``max_{q,r}`` locks onto q == r, which
          sits near cosine 1 for ANY pair of inputs, so every candidate scores
          the same and the row goes flat. Ordering survives but the margin —
          and with it the gradient — does not. This is the mechanism behind the
          earlier run that pinned L_ui at ln 2 for 34 epochs.
        - Making it asymmetric instead (left keeps queries, right mean-pooled)
          is WORSE, and is the only form that also destroys the ordering:
          pooling the right side leaves every candidate pointing along the
          shared direction, so top1 falls to near chance.
        - The actual culprit is the shared component, not the choice of max.
          Removing the batch mean fixes every form, and is exactly the remedy
          this repo already applies to the L_llm target bank for the same
          reason (see ``normalize_item_llm_emb``: a large common component
          "collapses all targets into a narrow cone and leaves the contrastive
          nothing to discriminate").

        Mean-pooling costs query specialization, which is why ITC / ITG /
        L_llm keep their max-over-queries form (``_max_query_logits``) — there
        the other side is a single vector, so the ``q == r`` degeneracy cannot
        arise in the first place. For the collaborative terms, CF alignment is
        the objective and pooled scoring is what already worked for the
        conditioned BPR.

        Centering is skipped below 4 rows: at n=2 it maps the two rows to exact
        opposites, so the similarity matrix degenerates to ``[[c, -c], [-c, c]]``
        — a single degree of freedom that says nothing about the individual
        pairs — and n=3 is barely better.
        """
        left = left_tokens.mean(dim=1)                                          # [B, D]
        right = right_tokens.mean(dim=1)                                        # [C, D]
        if self.pair_logit_center and left.size(0) >= 4:
            left = left - left.mean(dim=0, keepdim=True)
            right = right - right.mean(dim=0, keepdim=True)
        return (self.l2norm(left) @ self.l2norm(right).T) / tau

    def loss_item_item_ilm(self, left_ids: torch.Tensor, right_ids: torch.Tensor, tau: float = 0.07):
        """SigLLM-specific co-watch item-item contrastive."""
        left_cf = self.mf.item_encoder(left_ids)
        right_cf = self.mf.item_encoder(right_ids)
        left_q = self.qformer.encode_cf(left_cf, sem_vec=self._sem_for(left_ids))
        right_q = self.qformer.encode_cf(right_cf, sem_vec=self._sem_for(right_ids))

        logits = self._pooled_pair_logits(left_q, right_q, tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return loss, accuracy

    def _user_source(self, user_ids: torch.Tensor, history_ids):
        """Cross-attention source for the user side of ``loss_user_item``.

        With ``history_ids``, the user is represented by their padded HISTORY
        sequence — multi-source pooling through the exact path Stage 3's
        <UserProfile> uses. Stages 1/2 otherwise only ever cross-attend to S=1
        sources, so the aggregation behaviour (softmax selection over many
        keys, padding masks, pooled output statistics) reached Stage 3
        completely untrained. Without history (old pkls), falls back to the
        single MF user vector (S=1).

        ``history_ids`` is normally the [B, L] LongTensor padded on CPU by
        ``qformer_collate`` (0 = padding) — one H2D copy per batch instead of
        one ``torch.tensor(..., device=cuda)`` per row. A list of per-row id
        lists is still accepted for direct callers.

        Returns ``(src_cf, source_mask, sem_vec)``.
        """
        if history_ids is None:
            return self.mf.user_encoder(user_ids), None, None
        if isinstance(history_ids, torch.Tensor):
            his_pad = history_ids.to(user_ids.device)
        else:
            device = user_ids.device
            batch_size = user_ids.size(0)
            max_len = max(max((len(h) for h in history_ids), default=1), 1)
            his_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
            for row, hist in enumerate(history_ids):
                if hist:
                    his_pad[row, : len(hist)] = torch.tensor(hist, dtype=torch.long)
            his_pad = his_pad.to(device)
        source_mask = his_pad != self.mf.padding_index
        return self.mf.item_encoder(his_pad), source_mask, self._sem_for(his_pad)

    def loss_user_item(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        tau: float = 0.07,
        condition_on_item: bool = False,
        tau_cond: float = 0.2,
        cond_distill_mf: bool = True,
        cond_neg_mode: str = "random",
        history_ids=None,
    ):
        """User-item objective (ILM-style). On ML1M this captures the dominant
        CF signal (888K positive user-item pairs vs 3K item-text pairs in our
        pkls).

        Always computes the in-batch InfoNCE between the user's queries and
        the positive item's queries — this is the term that pressures the
        Q-Former body to be input-dependent, and it must not be replaced.

        ``condition_on_item=True`` ADDS a DIN-style pairwise BPR that
        pretrains the candidate-conditioning path (``user_proj``): the USER
        encoding's queries are shifted by the CANDIDATE item's CF vector,
        matching how Stage 3 pools the history conditioned on the target.
        Each candidate (positive and a sampled negative — ``cond_neg_mode``
        picks uniform-random catalog items or in-batch rolls) conditions its
        OWN scoring pass, so the conditioning carries no label information
        (feeding the positive's vector into an InfoNCE row would be a
        copy-through shortcut).

        BPR scoring is MEAN-POOLED over queries, not max-pair selected: an
        independent max over the Q x Q cosine pairs on each side inflates both
        scores to the same top-pair ceiling (matching query indices sit near
        cosine 1 regardless of input), which erased the margin entirely — an
        earlier run showed L pinned at ln 2 with accuracy below 0.5 for 34
        epochs while its high-magnitude ~sigmoid'(0)/tau gradient kept
        injecting noise into the shared Q-Former body.

        ``cond_distill_mf=True`` supervises the BPR with the frozen MF's
        pairwise ORDER instead of the raw held-in label: when MF ranks the
        rolled negative above the positive for this user, the pair direction
        is flipped. Rationale: fitting held-in labels through a u x item
        interaction network is memorizable (observed: train acc 0.61, val acc
        pinned at 0.50 with val loss drifting ABOVE ln 2), while MF's dot
        product is a transferable function of the inputs — distilling its
        order forces the conditioned pathway to learn a ranking FUNCTION
        rather than a pair table. Val cond_acc is still measured against the
        true label, so its ceiling under distillation is MF's own held-out
        pairwise accuracy (~0.7 on ML1M), not 1.0.

        Returns ``(loss, top1, cond_loss, cond_acc)``; the last two are None
        when conditioning is off. ``cond_acc`` chance level is 0.5."""
        user_src, user_mask, user_sem = self._user_source(user_ids, history_ids)
        item_cf = self.mf.item_encoder(item_ids)
        item_sem = self._sem_for(item_ids)

        # residual_vec = e_u so the output residual's fallback is cos(e_u, e_t),
        # matching what QRecLLM/the probe feed. Using the pooled history here
        # instead would train the residual against a different (weaker) readout
        # than the one served — measured 0.6411 vs MF_dot 0.6437.
        user_q = self.qformer.encode_cf(
            user_src, source_mask=user_mask, sem_vec=user_sem,
            residual_vec=self.mf.user_encoder(user_ids),
        )
        item_q = self.qformer.encode_cf(item_cf, sem_vec=item_sem)

        # Every candidate is scored the same way — no query index is picked
        # against the positive first (the old bias) and no Q x Q max to collapse
        # into (see _pooled_pair_logits).
        logits = self._pooled_pair_logits(user_q, item_q, tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()

        cond_loss = None
        cond_acc = None
        if condition_on_item and getattr(self.qformer, "user_conditioned", False):
            if cond_neg_mode == "random":
                # Uniform catalog negative (row 0 is the MF padding index).
                # In-batch "roll" negatives are other users' POSITIVES —
                # popularity-biased items the user plausibly likes too, the
                # hardest possible pair type: even the frozen MF only orders
                # pos-vs-others'-pos at ~0.55-0.6, so both the training signal
                # and the metric ceiling were pinned near chance (val UIC_acc
                # 0.505 with distillation working). Random negatives restore a
                # clean margin (MF orders pos-vs-random at ~its AUC).
                neg_ids = torch.randint(
                    1, self.mf.item_embedding.num_embeddings,
                    item_ids.shape, device=item_ids.device,
                )
                neg_cf = self.mf.item_encoder(neg_ids)
                neg_q = self.qformer.encode_cf(neg_cf, sem_vec=self._sem_for(neg_ids))
                i_neg_vec = neg_q.mean(dim=1)
                perm = None
            else:  # "roll": in-batch negative — another user_item row's positive.
                perm = torch.roll(torch.arange(item_ids.size(0), device=item_ids.device), 1)
                neg_ids = item_ids[perm]
                neg_cf = item_cf[perm]
                # Conditioning-free item bags: the negatives' encodings are a
                # permutation of the positives' — no extra forward.
                i_neg_vec = None  # filled from i_pos_vec below

            # With history mode this is EXACTLY Stage 3's <UserProfile>
            # computation: pool the history sequence with queries shifted by
            # the candidate's CF vector.
            user_q_pos = self.qformer.encode_cf(
                user_src, user_cf=item_cf, source_mask=user_mask, sem_vec=user_sem
            )
            user_q_neg = self.qformer.encode_cf(
                user_src, user_cf=neg_cf, source_mask=user_mask, sem_vec=user_sem
            )
            # Pooled but NOT yet normalized — centering has to happen before the
            # l2norm to have any effect.
            u_pos_vec = user_q_pos.mean(dim=1)
            u_neg_vec = user_q_neg.mean(dim=1)
            i_pos_vec = item_q.mean(dim=1)
            if i_neg_vec is None:
                i_neg_vec = i_pos_vec[perm]

            # Centering HELPS here, unlike in the InfoNCE above — the two losses
            # fail in opposite directions and the distinction is the whole
            # reason these are separate flags.
            #
            # InfoNCE (``_pooled_pair_logits``): centering spreads the cosines,
            # and ``cross_entropy`` over n candidates divides by a small tau, so
            # the spread lands in logsumexp and drove the loss far past ln(n)
            # (measured: L_ii 24.4 vs ln(n) 6.4). Hence its default is off.
            #
            # BPR here: ``softplus`` sees only the DIFFERENCE s_neg - s_pos, so
            # there is no logsumexp to blow up. The failure is the opposite one —
            # with the query output nearly collinear across inputs (measured
            # offdiag cosine ~0.9999) both scores sit at ~1.0, where the cosine
            # is saturated, so their difference is ~0 and the gradient reaching
            # user_proj vanishes. That is exactly why L_uic sat at ln 2 = 0.6931
            # for every epoch. Removing the shared direction restores a margin
            # the pretext can actually move.
            #
            # The mean is SHARED between pos and neg on each side, deliberately.
            # Centering each by its own mean would put s_pos and s_neg in
            # different coordinate frames and their difference would stop being
            # a comparison. For "roll" the negatives are a permutation of the
            # positives, so a shared mean is also the only self-consistent
            # choice (permuting centered vectors == centering permuted ones).
            if self.bpr_logit_center and u_pos_vec.size(0) >= 4:
                u_mu = torch.cat([u_pos_vec, u_neg_vec], dim=0).mean(dim=0, keepdim=True)
                i_mu = torch.cat([i_pos_vec, i_neg_vec], dim=0).mean(dim=0, keepdim=True)
                u_pos_vec = u_pos_vec - u_mu
                u_neg_vec = u_neg_vec - u_mu
                i_pos_vec = i_pos_vec - i_mu
                i_neg_vec = i_neg_vec - i_mu

            u_pos = self.l2norm(u_pos_vec)
            u_neg = self.l2norm(u_neg_vec)
            i_pos = self.l2norm(i_pos_vec)
            i_neg = self.l2norm(i_neg_vec)

            s_pos = (u_pos * i_pos).sum(dim=-1)
            s_neg = (u_neg * i_neg).sum(dim=-1)

            valid = neg_ids != item_ids  # rolled negative can collide on tiny batches
            if valid.sum() == 0:
                cond_loss = s_pos.sum() * 0.0
                cond_acc = cond_loss.detach()
            else:
                diff = (s_neg - s_pos) / tau_cond
                if cond_distill_mf:
                    # MF margins always come from the MF USER vector,
                    # regardless of what the Q-Former's user source is.
                    with torch.no_grad():
                        mf_user = self.mf.user_encoder(user_ids)
                        m_pos = (mf_user * item_cf).sum(dim=-1)
                        m_neg = (mf_user * neg_cf).sum(dim=-1)
                        flip = m_neg > m_pos
                    diff = torch.where(flip, -diff, diff)
                cond_loss = F.softplus(diff[valid]).mean()
                # Accuracy stays measured against the TRUE label (pos beats
                # neg) so the metric is comparable whether distilling or not.
                cond_acc = (s_pos[valid] > s_neg[valid]).float().mean()

        return loss, accuracy, cond_loss, cond_acc

    def loss_user_rank(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        history_ids,
        tau: float = 1.0,
    ):
        """Per-user pairwise BPR on the channel's own readout — a uAUC surrogate.

        THE GAP THIS FILLS. Every other Stage-1 loss optimises either a CROSS-user
        retrieval pretext (L_ii, L_ui: pick this user's item out of other users'
        items) or caption reconstruction (ITC, ITG, L_llm). NOT ONE of them asks
        the question the project is judged on: within a single user's candidates,
        is the liked one ranked above the disliked one. That is why the measured
        within-user uAUC of the channel FELL while every internal metric rose —
        it was nobody's objective, so the other terms were free to spend the
        representation on theirs.

        Scored on exactly what the probe (and Stage 3) reads:
        ``pool(profile_q) . pool(target_q)``, with the user side pooled from the
        padded HISTORY through the same multi-source path <UserProfile> uses, and
        the output residual left ON (unlike the caption losses, which are now
        shielded from it) because preserving that fallback readout is the point.

        Labels are the real CTR labels of held-in rows, so this can memorise —
        watch it against the val probe rather than its own train value, exactly
        as with L_ui. The pairing is done by ``_per_user_pairwise_loss``, the same
        helper Stage 3 uses for ``ranking_loss``, so batches must carry several
        rows per user with mixed labels (see UserGroupedSampler).
        """
        user_src, user_mask, user_sem = self._user_source(user_ids, history_ids)
        item_cf = self.mf.item_encoder(item_ids)

        profile_q = self.qformer.encode_cf(
            user_src,
            user_cf=item_cf if self.qformer.user_conditioned else None,
            source_mask=user_mask,
            sem_vec=user_sem,
            residual_vec=self.mf.user_encoder(user_ids),
        )
        target_q = self.qformer.encode_cf(item_cf, sem_vec=self._sem_for(item_ids))

        # COSINE, not the dot product. Scoring the dot was a real bug: within one
        # user profile_q is near-fixed, so dot-ranking == ranking by
        # ||target_q|| * cos, and the loss can be satisfied purely by inflating
        # the norm of positive items — a channel the cosine metric cannot see,
        # while the norm inflation distorts direction as a side effect. Loss went
        # down and the probe's uAUC went DOWN with it, every epoch. The loss and
        # the metric have to read the same thing.
        #
        # tau matters here for the same reason it did for the pair logits: cosine
        # margins are O(0.01-0.1) given the anisotropy, so at tau=1.0 softplus
        # sits at ~ln 2 with almost no gradient. Use tau ~0.05.
        u_vec = F.normalize(profile_q.mean(dim=1), dim=-1)
        i_vec = F.normalize(target_q.mean(dim=1), dim=-1)
        scores = (u_vec * i_vec).sum(dim=-1)
        loss = _per_user_pairwise_loss(scores, user_ids, labels, tau)
        with torch.no_grad():
            acc = _per_user_pairwise_acc(scores, user_ids, labels)
        return loss, acc

    def loss_llm_align(self, item_ids: torch.Tensor, tau: float = 0.07, symmetric: bool = True):
        if not self.has_llm_align:
            raise RuntimeError("LLM alignment loss requested but no LLM embeddings provided")

        # Same path the soft tokens take at injection time (encode_cf ->
        # out_proj -> llm_proj, per token). BLIP-2-style max-over-queries, like
        # ITC: each (item, target) candidate pair is scored by its own best
        # query token instead of the mean over all Q tokens (mean pooling
        # pushed every query toward the same target and collapsed query
        # diversity) — and instead of the query picked against the POSITIVE
        # target only, which handicapped negatives and inflated top1.
        query_hidden = self.encode_item_queries(item_ids)                       # [B, Q, d_model]
        soft_tokens = self.llm_align_proj(self.qformer.out_proj(query_hidden))  # [B, Q, d_llm]
        t_vec = self.item_llm_emb[item_ids].to(soft_tokens.device)
        sim_matrix = self._max_query_logits(soft_tokens, t_vec, tau, center=self.itc_logit_center)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_q2t = F.cross_entropy(sim_matrix, labels)
        if symmetric:
            loss_t2q = F.cross_entropy(sim_matrix.T, labels)
            loss = (loss_q2t + loss_t2q) / 2.0
        else:
            loss = loss_q2t
        accuracy = (sim_matrix.argmax(dim=1) == labels).float().mean()
        return loss, accuracy