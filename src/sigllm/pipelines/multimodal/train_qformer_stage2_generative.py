"""Stage 2 — Generative pretraining of Q-Former + projection (BLIP-2 style).

Loads the Q-Former weights from Stage 1, attaches a ``Linear + LayerNorm``
projection into the LLM's hidden size (warm-started from Stage 1's aligned
projection when ``proj_ckpt_in`` is set, fresh otherwise), and trains
Q-Former + projection with next-token language modeling on item-text
captions while keeping the LLM fully frozen. An optional alignment
keep-alive term (``w_llm`` > 0) applies the Stage-1 InfoNCE between
mean-pooled soft tokens and distilled input-space item embeddings, so the
generative objective does not wash out the SeLLa alignment. A second,
discriminative term (``w_align`` > 0) aligns the same soft tokens to the
frozen LLM's own last-hidden caption representations via in-batch InfoNCE,
computed on the fly — grounding the exact ``llm_proj`` output Stage 3
injects in the semantic space the LLM reads from. The Q-Former runs uni-modal here (queries cross-attend
to the CF vector only, no text input on the Q-Former text branch) —
instruction-awareness is reserved for Stage 3, matching BLIP-2's stage-2
design. This produces a checkpoint usable as the starting point for
Stage 3 (instruction tuning with frozen LLM in ``QRecLLM``).

Prompt layout per sample:

    [BOS] [K soft tokens] item_text [EOS]

Loss is applied only on ``item_text [EOS]`` positions; BOS and soft tokens
are masked with -100.
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import omegaconf
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from sigllm.common import EarlyStopping, NotebookLogger
from sigllm.common.utils import resolve_hf_model_path
from sigllm.common.config import Config
from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset
from sigllm.datasets.qformer.qformer_loader import build_qformer_loader
from sigllm.models.projection.qformer_alignment_model import (
    QRecInstructAlignmentModel,
    normalize_item_llm_emb,
)
from sigllm.models.projection.soft_token_proj import (
    build_soft_token_projection,
    remap_legacy_proj_state,
)
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.rec.matrix_factorization import MatrixFactorization

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = NotebookLogger.rich_logger("sigllm.train_qformer_stage2")


def log_step(title: str, detail: Optional[str] = None) -> None:
    LOGGER.info(title if detail is None else f"{title} | {detail}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disabled_train(self, mode=True):
    return self


def _filter_item_text(dataset: QFormerAlignmentDataset) -> Subset:
    indices = [i for i, s in enumerate(dataset.samples) if s["sample_type"] == "item_text"]
    if not indices:
        raise ValueError("No item_text samples found in dataset; cannot run generative pretraining.")
    return Subset(dataset, indices)


def _filter_user_item(dataset: QFormerAlignmentDataset) -> Subset:
    """user_item rows for the ui keep-alive; may be empty (old pkls)."""
    indices = [i for i, s in enumerate(dataset.samples) if s["sample_type"] == "user_item"]
    return Subset(dataset, indices)


def _history_or_none(batch):
    """Padded [B, L] history tensor from the collate, or None when the pkl
    carries no history (all-zero) so the loss falls back to the MF user vector."""
    his = batch.get("his")
    if isinstance(his, torch.Tensor):
        if his.numel() == 0 or not bool((his != 0).any()):
            return None
        return his
    if his is not None and not any(his):
        return None
    return his


def _init_rec_model(cfg, device):
    mf_config = omegaconf.OmegaConf.create({
        "user_num": int(cfg.user_num),
        "item_num": int(cfg.item_num),
        "embedding_size": int(cfg.embedding_size),
    })
    mf = MatrixFactorization(mf_config).to(device)
    if os.path.exists(cfg.pretrained_rec_path):
        mf.load_state_dict(torch.load(cfg.pretrained_rec_path, map_location="cpu"))
        log_step("Loaded pretrained MF", cfg.pretrained_rec_path)
    for p in mf.parameters():
        p.requires_grad = False
    mf.eval()
    mf.train = disabled_train.__get__(mf, MatrixFactorization)
    return mf


def _init_qformer(cfg, device, d_sem=None):
    qformer_output_dim = cfg.get("qformer_output_dim") or cfg.qformer_d_model
    qformer = HFQFormerAdapter(
        d_cf=int(cfg.embedding_size),
        d_model=int(cfg.qformer_d_model),
        num_queries=int(cfg.num_queries),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        output_dim=int(qformer_output_dim),
        qformer_text_model_name=cfg.qformer_text_model_name,
        max_instruction_length=int(cfg.max_instruction_length),
        init_from_pretrained_text=False,
        # Must mirror Stage 1's adapter shape: its checkpoint now carries
        # user_proj (candidate conditioning) and proj_sem (semantic source),
        # and the strict load below would otherwise reject those keys. The
        # conditioning path is unused here (no user_cf is passed), but the
        # weights must ride through to Stage 3 intact.
        user_conditioned=bool(cfg.get("user_conditioned", False)),
        # Must mirror model.qformer_config.* — these change the adapter SHAPE,
        # and the checkpoint flows stage1 -> stage2 -> stage3 under a strict load.
        candidate_fusion=bool(cfg.get("candidate_fusion", False)),
        item_residual=bool(cfg.get("item_residual", False)),
        d_user=int(cfg.embedding_size),
        d_sem=d_sem,
    ).to(device)

    ckpt_path = cfg.get("qformer_ckpt_in")
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        qformer.load_state_dict(state_dict, strict=True)
        log_step("Loaded Phase 1 Q-Former", ckpt_path)
    else:
        log_step("WARNING", f"qformer_ckpt_in not found at {ckpt_path}; training from scratch")

    for p in qformer.parameters():
        p.requires_grad = True
    return qformer


def _init_llm(model_path, device):
    model_path, local_files_only = resolve_hf_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()
    return tokenizer, llm


def _build_projection(
    d_q: int,
    hidden_size: int,
    device,
    ckpt_path: Optional[str] = None,
    center: bool = True,
    num_positions: Optional[int] = None,
) -> nn.Module:
    """Linear -> [SharedDirectionCenter] -> LayerNorm.

    Built by the shared factory so Stage 1 / 2 / 3 cannot drift apart — the
    weights flow between them under a strict load. See
    ``sigllm.models.projection.soft_token_proj`` for why the centering sits
    between the Linear and the LayerNorm and why it tracks a running (not batch)
    mean.

    ``ckpt_path`` warm-starts from Stage 1's ``llm_align_proj`` (same layout),
    carrying the SeLLa alignment into the injection path.
    """
    proj = build_soft_token_projection(
        d_q, hidden_size, center=center, num_positions=num_positions
    ).to(device).float()

    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict, _ = remap_legacy_proj_state(state_dict, proj)
        msg = proj.load_state_dict(state_dict, strict=False)
        stale = [
            k for k in list(msg.missing_keys) + list(msg.unexpected_keys)
            if "running_mean" not in k and "initialized" not in k
        ]
        if stale:
            log_step("WARNING", f"proj_ckpt_in layout mismatch on keys: {stale}")
        log_step("Warm-started llm_proj from Stage 1 aligned projection", ckpt_path)
    elif ckpt_path:
        log_step("WARNING", f"proj_ckpt_in not found at {ckpt_path}; using fresh projection init")
    return proj


def _build_inputs(
    soft_tokens: torch.Tensor,
    captions: list,
    tokenizer: PreTrainedTokenizer,
    embed_layer: nn.Module,
    max_caption_length: int,
):
    """Assemble ``[BOS?][K soft tokens][caption + EOS]`` with -100 labels on
    the prefix (BOS + soft positions). Tokenizers without an explicit BOS
    (e.g. Qwen2) skip the BOS slot entirely; the loss target is unchanged."""

    device = soft_tokens.device
    batch_size, query_count, hidden = soft_tokens.shape

    captions_with_eos = [c + tokenizer.eos_token for c in captions]
    tokenizer.padding_side = "right"
    cap_tokens = tokenizer(
        captions_with_eos,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_caption_length,
        add_special_tokens=False,
    ).to(device)
    cap_ids = cap_tokens.input_ids
    cap_mask = cap_tokens.attention_mask

    cap_embeds = embed_layer(cap_ids).to(soft_tokens.dtype)
    soft_mask = torch.ones((batch_size, query_count), dtype=cap_mask.dtype, device=device)

    has_bos = tokenizer.bos_token_id is not None
    if has_bos:
        bos_ids = torch.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        bos_embeds = embed_layer(bos_ids).to(soft_tokens.dtype)
        bos_mask = torch.ones((batch_size, 1), dtype=cap_mask.dtype, device=device)
        inputs_embeds = torch.cat([bos_embeds, soft_tokens, cap_embeds], dim=1)
        attention_mask = torch.cat([bos_mask, soft_mask, cap_mask], dim=1)
        prefix_len = 1 + query_count
    else:
        inputs_embeds = torch.cat([soft_tokens, cap_embeds], dim=1)
        attention_mask = torch.cat([soft_mask, cap_mask], dim=1)
        prefix_len = query_count

    prefix_labels = torch.full(
        (batch_size, prefix_len), -100, dtype=torch.long, device=device
    )
    cap_labels = cap_ids.masked_fill(cap_ids == tokenizer.pad_token_id, -100)
    labels = torch.cat([prefix_labels, cap_labels], dim=1)

    return inputs_embeds, attention_mask, labels


def _move_batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


@torch.no_grad()
def _build_caption_target_bank(
    datasets,
    tokenizer,
    llm,
    item_num: int,
    max_caption_length: int,
    normalize_mode: str,
    device,
    chunk_size: int = 64,
):
    """Precompute the w_align targets once: one frozen-LLM last-hidden vector
    per item caption, then condition the whole bank with FIXED statistics.

    Conditioning is the load-bearing part. Raw Qwen2 last-hidden states carry
    massive shared activations (norms ~277 here): every caption target is
    nearly collinear with every other, so cosine InfoNCE sees constant rows —
    pos_sim == neg_sim, loss pinned at ln(B), zero gradient. Centering removes
    the shared component; whiten additionally evens out the per-dim variance
    the outlier dims would otherwise monopolize. Unlike the Stage-1 input-space
    bank, this space is only a discrimination reference (the LLM never reads
    these coordinates back), so bending its geometry is safe.

    Also removes the per-step no-grad LLM forward the on-the-fly path paid.
    Items appearing in several datasets keep the first caption seen (captions
    are deterministic per item, so this is a dedupe, not a choice).
    """
    caption_by_item = {}
    for subset in datasets:
        base, indices = subset.dataset, subset.indices
        for i in indices:
            sample = base.samples[i]
            caption_by_item.setdefault(int(sample["i_left"]), str(sample["text"]))

    ids = sorted(caption_by_item)
    bank = torch.zeros(item_num, llm.config.hidden_size)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        targets = _llm_caption_targets(
            [caption_by_item[i] for i in chunk], tokenizer, llm, max_caption_length
        )
        bank[torch.tensor(chunk, dtype=torch.long)] = targets.cpu()

    # Fixed-stat conditioning over covered rows; uncovered rows stay zero (they
    # are never indexed — the loaders only yield item_text samples).
    bank = normalize_item_llm_emb(bank, normalize_mode)
    log_step(
        "Built w_align caption target bank",
        f"items={len(ids)}, normalize={normalize_mode}, "
        f"mean_norm={bank[ids].norm(dim=-1).mean().item():.4f}",
    )
    return bank.to(device)


@torch.no_grad()
def _llm_caption_targets(captions, tokenizer, llm, max_caption_length: int):
    """Discriminative alignment targets from the frozen base LLM itself.

    Tokenizes the raw captions (NO soft tokens), runs the frozen LLM and
    pools the last hidden layer at the last non-pad token of each caption —
    the causal position that has seen the whole caption. Returns ``[B, H]``
    in float32 (fp16 saturates the cosine similarities downstream)."""
    device = next(llm.parameters()).device
    tokenizer.padding_side = "right"
    tokens = tokenizer(
        captions,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_caption_length,
        add_special_tokens=False,
    ).to(device)
    decoder = llm.get_decoder() if hasattr(llm, "get_decoder") else None
    with torch.amp.autocast("cuda", dtype=torch.float16):
        if decoder is not None:
            # Trunk-only forward: skips the lm_head logits ([B, T, vocab]),
            # which are pure waste here.
            hidden = decoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                return_dict=True,
            ).last_hidden_state                                          # [B, T, H]
        else:
            hidden = llm(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]                                          # [B, T, H]
    last_idx = (tokens.attention_mask.sum(dim=1) - 1).clamp(min=0)       # [B]
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, last_idx].float()                                # [B, H]


def _llm_space_align_loss(soft_tokens, target, tau: float):
    """Discriminative LLM-space alignment: symmetric in-batch InfoNCE between
    mean-pooled soft tokens (post ``llm_proj`` — the exact vectors Stage 3
    injects) and the frozen LLM's own caption representations. Unlike the
    generative caption loss, this forces the injected representation to be
    linearly separable in the space the LLM *reads from*, so the frozen LLM
    can discriminate items from the soft tokens alone. Returns
    ``(loss, stats)`` where ``stats`` carries retrieval and geometry
    diagnostics (floats, already detached). Everything is upcast to float32
    before normalize/softmax."""
    q_vec = soft_tokens.float().mean(dim=1)                      # [B, H]
    t_vec = target.to(q_vec.device).float()                      # [B, H]
    q_norm = q_vec / (q_vec.norm(dim=-1, keepdim=True) + 1e-12)
    t_norm = t_vec / (t_vec.norm(dim=-1, keepdim=True) + 1e-12)
    sim = (q_norm @ t_norm.T) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = (
        nn.functional.cross_entropy(sim, labels)
        + nn.functional.cross_entropy(sim.T, labels)
    ) / 2.0
    with torch.no_grad():
        n = sim.size(0)
        cos = sim * tau                                          # raw cosine, [-1, 1]
        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
        pos_sim = cos.diagonal().mean()
        neg_sim = cos[~eye].mean() if n > 1 else cos.new_zeros(())
        top1 = (sim.argmax(dim=1) == labels).float().mean()
        k = min(5, n)
        topk = sim.topk(k, dim=1).indices
        top5 = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        # Collapse detectors. q_pair_cos ~= 1.0 means the soft tokens are the
        # SAME vector for every item (a collapsed upstream Q-Former): the sim
        # matrix rows go identical, top1 degenerates to exactly 1/B per batch,
        # and no amount of target conditioning can help. t_pair_cos ~= 1.0 is
        # the mirror failure (collinear targets, e.g. an unnormalized
        # last-hidden bank). Healthy runs sit well below ~0.9 on both.
        q_pair = (q_norm @ q_norm.T)[~eye].mean() if n > 1 else cos.new_zeros(())
        t_pair = (t_norm @ t_norm.T)[~eye].mean() if n > 1 else cos.new_zeros(())
        stats = {
            "top1": float(top1.item()),
            "top5": float(top5.item()),
            "pos_sim": float(pos_sim.item()),
            "neg_sim": float(neg_sim.item()),
            "q_pair_cos": float(q_pair.item()),
            "t_pair_cos": float(t_pair.item()),
            "soft_norm": float(q_vec.norm(dim=-1).mean().item()),
            "target_norm": float(t_vec.norm(dim=-1).mean().item()),
            "n_candidates": n,
        }
    return loss, stats


class _RunningMeans:
    """Per-key running means over batches; keys with no observations are
    simply absent from the summary, so disabled loss terms never print."""

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, key, value):
        if value is None:
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.counts[key] = self.counts.get(key, 0) + 1

    def add_parts(self, loss, parts):
        self.add("loss", float(loss.item()))
        self.add("lm", parts["lm"])
        self.add("keepalive", parts["keepalive"])
        self.add("llm_align", parts["llm_align"])
        stats = parts["align_stats"]
        if stats is not None:
            self.add("align@1", stats["top1"])
            self.add("align@5", stats["top5"])
            self.add("pos_sim", stats["pos_sim"])
            self.add("neg_sim", stats["neg_sim"])
            self.add("sim_gap", stats["pos_sim"] - stats["neg_sim"])
            self.add("q_pair_cos", stats["q_pair_cos"])
            self.add("t_pair_cos", stats["t_pair_cos"])
            self.add("soft_norm", stats["soft_norm"])
            self.add("target_norm", stats["target_norm"])

    def mean(self, key, default=0.0):
        count = self.counts.get(key, 0)
        return self.sums.get(key, 0.0) / count if count else default

    def summary(self, keys):
        return ", ".join(
            f"{key}={self.mean(key):.4f}" for key in keys if key in self.sums
        )


# Order used by every step/epoch/val log line below. The ui_keep* keys are
# train-only (the keep-alive never runs at eval), so _RunningMeans simply
# omits them from val summaries.
_LOG_KEYS = (
    "loss", "lm", "keepalive", "llm_align",
    "align@1", "align@5", "pos_sim", "neg_sim", "sim_gap",
    "q_pair_cos", "t_pair_cos",
    "soft_norm", "target_norm",
    "ui_keep", "ui_keep@1", "uic_keep", "uic_acc",
)


def _align_loss(soft_tokens, item_ids, align_bank, tau: float):
    """Alignment keep-alive: symmetric InfoNCE between mean-pooled soft tokens
    (post ``llm_proj``, the exact vectors the LLM reads) and the distilled
    input-space item embeddings — the same objective as Stage 1's
    ``loss_llm_align``, kept on while the generative loss retrains the stack."""
    q_vec = soft_tokens.float().mean(dim=1)                      # [B, H]
    t_vec = align_bank[item_ids]                                 # [B, H]
    q_norm = q_vec / (q_vec.norm(dim=-1, keepdim=True) + 1e-12)
    t_norm = t_vec / (t_vec.norm(dim=-1, keepdim=True) + 1e-12)
    sim = (q_norm @ t_norm.T) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return (nn.functional.cross_entropy(sim, labels) + nn.functional.cross_entropy(sim.T, labels)) / 2.0


def forward_stage2(
    batch,
    mf,
    qformer,
    llm_proj,
    tokenizer,
    llm,
    max_caption_length: int,
    align_bank=None,
    w_llm: float = 0.0,
    tau_llm: float = 0.07,
    w_align: float = 0.0,
    tau_align: float = 0.07,
    sem_bank=None,
    sem_dropout: float = 0.0,
    align_target_bank=None,
):
    item_ids = batch["i_left"]
    captions = batch["text"]

    with torch.no_grad():
        item_cf = mf.item_encoder(item_ids)

    # Optional second cross-attention source (semantic item embedding). Row
    # dropout during training keeps the CF path trained; zeroed rows are
    # masked out inside the adapter.
    sem_vec = None
    if sem_bank is not None:
        sem_vec = sem_bank[item_ids]
        if qformer.training and sem_dropout > 0.0:
            keep = (
                torch.rand(sem_vec.size(0), 1, device=sem_vec.device) >= sem_dropout
            ).to(sem_vec.dtype)
            sem_vec = sem_vec * keep

    # BLIP-2-style generative pretraining: queries cross-attend to the CF
    # vector (and optional semantic source) only, no text input to the
    # Q-Former. Instruction-awareness is deferred to Stage 3.
    query_tokens = qformer.encode_cf(item_cf, sem_vec=sem_vec)
    query_tokens = qformer.out_proj(query_tokens)
    soft_tokens = llm_proj(query_tokens)

    llm_dtype = next(llm.parameters()).dtype
    soft_tokens_lm = soft_tokens.to(llm_dtype)
    embed_layer = llm.get_input_embeddings()

    inputs_embeds, attention_mask, labels = _build_inputs(
        soft_tokens_lm,
        captions,
        tokenizer,
        embed_layer,
        max_caption_length,
    )

    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    lm_loss = outputs.loss
    total = lm_loss

    keepalive = None
    if align_bank is not None and w_llm > 0.0:
        keepalive = _align_loss(soft_tokens, item_ids, align_bank, tau_llm)
        total = total + w_llm * keepalive

    llm_align = None
    align_stats = None
    if w_align > 0.0:
        if align_target_bank is not None:
            # Precomputed + fixed-stat normalized targets (see
            # _build_caption_target_bank for why raw last-hidden states give
            # this loss zero gradient).
            target = align_target_bank[item_ids]
        else:
            # Legacy on-the-fly path: RAW last-hidden targets. Known to be
            # near-collinear on Qwen2 — prefer the precomputed bank.
            target = _llm_caption_targets(captions, tokenizer, llm, max_caption_length)
        llm_align, align_stats = _llm_space_align_loss(soft_tokens, target, tau_align)
        total = total + w_align * llm_align

    parts = {
        "lm": float(lm_loss.item()),
        "keepalive": float(keepalive.item()) if keepalive is not None else None,
        "llm_align": float(llm_align.item()) if llm_align is not None else None,
        "align_stats": align_stats,
    }
    return total, parts


def evaluate(
    loader, mf, qformer, llm_proj, tokenizer, llm, device, max_caption_length: int,
    align_bank=None, w_llm: float = 0.0, tau_llm: float = 0.07,
    w_align: float = 0.0, tau_align: float = 0.07, sem_bank=None,
    align_target_bank=None,
):
    qformer.eval()
    llm_proj.eval()
    meters = _RunningMeans()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss, parts = forward_stage2(
                batch, mf, qformer, llm_proj, tokenizer, llm, max_caption_length,
                align_bank=align_bank, w_llm=w_llm, tau_llm=tau_llm,
                w_align=w_align, tau_align=tau_align, sem_bank=sem_bank,
                align_target_bank=align_target_bank,
            )
            meters.add_parts(loss, parts)
    qformer.train()
    llm_proj.train()
    return meters


def train_qformer_stage2_generative(cfg):
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_qformer_loader(
        cfg,
        filename=os.path.join(cfg.data_dir, "train_qformer_ood2.pkl"),
        shuffle=True,
        filter_fn=_filter_item_text,
    )
    valid_loader = build_qformer_loader(
        cfg,
        filename=os.path.join(cfg.data_dir, "valid_qformer_ood2.pkl"),
        shuffle=False,
        filter_fn=_filter_item_text,
    )

    mf = _init_rec_model(cfg, device)

    # Semantic cross-attention source (second source next to CF). Loaded RAW —
    # normalize_item_llm_emb conditions the contrastive TARGET bank only; the
    # source has its own trainable proj_sem. Must load before the Q-Former so
    # the adapter is built with proj_sem and can strict-load Stage 1's ckpt.
    sem_source = bool(cfg.get("sem_source", False))
    sem_dropout = float(cfg.get("sem_source_dropout", 0.5))
    sem_bank = None
    if sem_source:
        # item_sem_emb_path, NOT item_llm_emb_path: the semantic SOURCE bank
        # must stay the one Stage 1 trained proj_sem on, and must differ from
        # any alignment TARGET bank (see the Stage-1 leak guard).
        emb_path = cfg.get("item_sem_emb_path")
        if not emb_path or not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"item_sem_emb_path is required and must exist when sem_source is enabled, got: {emb_path}"
            )
        blob = torch.load(emb_path, map_location="cpu")
        sem_bank = (blob["item_llm_emb"] if isinstance(blob, dict) else blob).float().to(device)
        log_step(
            "Semantic cross-attention source active",
            f"bank={tuple(sem_bank.shape)} from {emb_path}, sem_source_dropout={sem_dropout}",
        )

    qformer = _init_qformer(
        cfg, device, d_sem=sem_bank.size(-1) if sem_bank is not None else None
    ).train()
    tokenizer, llm = _init_llm(cfg.llm_model_name, device)

    d_q = qformer.output_dim
    hidden_size = llm.config.hidden_size
    llm_proj = _build_projection(
        d_q, hidden_size, device,
        ckpt_path=cfg.get("proj_ckpt_in"),
        center=bool(cfg.get("center_soft_tokens", True)),
        num_positions=int(qformer.num_queries),
    ).train()

    w_llm = float(cfg.get("w_llm", 0.0))
    tau_llm = float(cfg.get("tau_llm", 0.07))
    # Discriminative LLM-space alignment: InfoNCE between the injected soft
    # tokens (post llm_proj) and the frozen LLM's own last-hidden caption
    # representations, computed on the fly. 0.0 = exact legacy behavior
    # (no extra LLM forward, loss unchanged).
    w_align = float(cfg.get("w_align", 0.0))
    tau_align = float(cfg.get("tau_align", 0.07))
    align_target_normalize = str(cfg.get("align_target_normalize", "whiten"))
    align_target_bank = None
    if w_align > 0.0:
        align_target_bank = _build_caption_target_bank(
            [train_loader.dataset, valid_loader.dataset],
            tokenizer,
            llm,
            item_num=int(cfg.item_num),
            max_caption_length=int(cfg.get("max_caption_length", 64)),
            normalize_mode=align_target_normalize,
            device=device,
        )
        log_step(
            "Discriminative LLM-space alignment active",
            f"w_align={w_align}, tau_align={tau_align}, "
            f"targets=precomputed bank (normalize={align_target_normalize})",
        )
    # Must match run.qformer_stage1.llm_emb_normalize — the config wires both
    # from one key. A mismatch makes the keep-alive pull llm_proj toward a
    # different target geometry than the one Stage 1 aligned it to.
    llm_emb_normalize = str(cfg.get("llm_emb_normalize", "center"))
    align_bank = None
    if w_llm > 0.0:
        emb_path = cfg.get("item_llm_emb_path")
        if not emb_path or not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"item_llm_emb_path is required and must exist when w_llm > 0.0, but got: {emb_path}"
            )
        blob = torch.load(emb_path, map_location="cpu")
        align_bank = (blob["item_llm_emb"] if isinstance(blob, dict) else blob).float().to(device)
        if align_bank.size(-1) != hidden_size:
            raise ValueError(
                f"item_llm_emb hidden size {align_bank.size(-1)} != LLM hidden size {hidden_size}; "
                "distill with --space input against the same LLM."
            )
        # Exactly the transform Stage 1's alignment model applied to the same
        # bank — shared helper so the two stages cannot drift apart.
        align_bank = normalize_item_llm_emb(align_bank, llm_emb_normalize)
        log_step(
            "Alignment keep-alive active",
            f"w_llm={w_llm}, tau_llm={tau_llm}, norm={llm_emb_normalize}, "
            f"bank={tuple(align_bank.shape)} from {emb_path}",
        )

    # User-item keep-alive. The generative objective only ever runs S=1
    # forwards (one item CF token [+ its sem token]), so a full Stage 2 run
    # retrains the Q-Former body exclusively on single-source cross-attention
    # and washes out what Stage 1 just pretrained for Stage 3's <UserProfile>:
    # multi-source history pooling (S=L with padding masks) and the
    # candidate-conditioned queries (user_proj). This term replays the Stage-1
    # user-item losses (InfoNCE + optional conditioned BPR) on the user_item
    # rows of the same train pkl, one batch per generative step, so those
    # pathways keep receiving gradient. 0.0 = exact legacy behavior.
    w_ui_keep = float(cfg.get("w_ui_keep", 0.0))
    tau_ui_keep = float(cfg.get("tau_ui_keep", 0.07))
    w_ui_cond_keep = float(cfg.get("w_ui_cond_keep", 0.0))
    tau_ui_cond = float(cfg.get("tau_ui_cond", 0.2))
    ui_cond_neg = str(cfg.get("ui_cond_neg", "roll"))
    ui_loader = None
    align_model = None
    ui_condition_on_item = False
    if w_ui_keep > 0.0:
        ui_loader = build_qformer_loader(
            cfg,
            filename=os.path.join(cfg.data_dir, "train_qformer_ood2.pkl"),
            shuffle=True,
            filter_fn=_filter_user_item,
        )
        if len(ui_loader.dataset) == 0:
            log_step(
                "WARNING",
                "w_ui_keep > 0 but the train pkl holds no user_item rows; "
                "ui keep-alive disabled (rebuild with include_user_item: True).",
            )
            ui_loader = None
        else:
            # Thin wrapper exposing loss_user_item over the SAME qformer/mf.
            # Its auxiliary heads (itm_head, lm_head) are unused here and are
            # deliberately NOT added to the optimizer.
            align_model = QRecInstructAlignmentModel(
                mf=mf,
                qformer=qformer,
                item_sem_emb=sem_bank,
                sem_dropout=sem_dropout,
                pair_logit_center=bool(cfg.get("pair_logit_center", False)),
                bpr_logit_center=bool(cfg.get("bpr_logit_center", True)),
                # This wrapper only runs loss_user_item (the ui keep-alive), which
                # is a collaborative term — the flag is irrelevant to it, but kept
                # explicit so the two stages construct the model identically.
                sem_for_text_losses=bool(cfg.get("sem_for_text_losses", False)),
                itc_logit_center=bool(cfg.get("itc_logit_center", True)),
            ).to(device)
            ui_condition_on_item = bool(qformer.user_conditioned) and w_ui_cond_keep > 0.0
            log_step(
                "ui keep-alive active",
                f"w_ui_keep={w_ui_keep}, tau_ui={tau_ui_keep}, "
                f"conditioned_bpr={ui_condition_on_item} "
                f"(w_ui_cond_keep={w_ui_cond_keep}, tau_ui_cond={tau_ui_cond}, "
                f"neg={ui_cond_neg}), rows={len(ui_loader.dataset)}",
            )

    trainable_params = [
        p for p in qformer.parameters() if p.requires_grad
    ] + list(llm_proj.parameters())
    optimizer = Adam(
        trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    scaler = torch.amp.GradScaler("cuda")
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))

    outdir = cfg.output_dir
    os.makedirs(outdir, exist_ok=True)
    qformer_out = os.path.join(outdir, cfg.qformer_ckpt_out)
    proj_out = os.path.join(outdir, cfg.proj_ckpt_out)

    stopper = EarlyStopping(
        ref_metric="val_loss",
        monitor_mode="min",
        patience=int(cfg.early_stopping_patience),
        min_delta=float(cfg.early_stopping_min_delta),
    )
    log_step(
        "Stage 2 setup",
        f"d_q={d_q}, H={hidden_size}, K={qformer.num_queries}, "
        f"train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, "
        f"output_dir={outdir}",
    )

    max_caption_length = int(cfg.get("max_caption_length", 64))
    log_epoch = int(cfg.get("log_epoch", 1))
    # In-epoch progress logging: running means over the last `log_steps`
    # batches (0 disables). Lets you watch llm_align / align@1 / sim_gap move
    # within the first epoch instead of waiting for the epoch summary.
    log_steps = int(cfg.get("log_steps", 50))

    ui_iter = iter(ui_loader) if ui_loader is not None else None

    for epoch in range(int(cfg.epoch)):
        qformer.train()
        llm_proj.train()
        if align_model is not None:
            align_model.train()
        epoch_meters = _RunningMeans()
        window_meters = _RunningMeans()
        steps_per_epoch = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss, parts = forward_stage2(
                batch, mf, qformer, llm_proj, tokenizer, llm, max_caption_length,
                align_bank=align_bank, w_llm=w_llm, tau_llm=tau_llm,
                w_align=w_align, tau_align=tau_align,
                sem_bank=sem_bank, sem_dropout=sem_dropout,
                align_target_bank=align_target_bank,
            )

            # ui keep-alive: one user_item batch per generative step, replaying
            # Stage 1's history-pooled InfoNCE (+ conditioned BPR) so the
            # multi-source and user_proj pathways keep receiving gradient.
            if ui_iter is not None:
                try:
                    ui_batch = next(ui_iter)
                except StopIteration:
                    ui_iter = iter(ui_loader)
                    ui_batch = next(ui_iter)
                ui_batch = _move_batch_to_device(ui_batch, device)
                loss_ui, ui_top1, loss_uic, uic_acc = align_model.loss_user_item(
                    ui_batch["u"],
                    ui_batch["i_left"],
                    tau=tau_ui_keep,
                    condition_on_item=ui_condition_on_item,
                    tau_cond=tau_ui_cond,
                    cond_distill_mf=False,
                    cond_neg_mode=ui_cond_neg,
                    history_ids=_history_or_none(ui_batch),
                )
                loss = loss + w_ui_keep * loss_ui
                if loss_uic is not None:
                    loss = loss + w_ui_cond_keep * loss_uic
                for meters in (epoch_meters, window_meters):
                    meters.add("ui_keep", float(loss_ui.item()))
                    meters.add("ui_keep@1", float(ui_top1.item()))
                    if loss_uic is not None:
                        meters.add("uic_keep", float(loss_uic.item()))
                        meters.add("uic_acc", float(uic_acc.item()))

            scaler.scale(loss).backward()
            # Adam + fp16 with no clipping blew the run up mid-training once
            # (loss 4.22 -> 5.74 in one epoch, never recovered); clip to keep a
            # single bad batch from ejecting the model out of its basin.
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_meters.add_parts(loss, parts)
            window_meters.add_parts(loss, parts)

            if log_steps > 0 and (step % log_steps == 0 or step == steps_per_epoch):
                log_step(
                    f"epoch {epoch + 1} step {step}/{steps_per_epoch}",
                    window_meters.summary(_LOG_KEYS),
                )
                window_meters = _RunningMeans()

        if (epoch + 1) % log_epoch != 0:
            continue

        val_meters = evaluate(
            valid_loader, mf, qformer, llm_proj, tokenizer, llm, device, max_caption_length,
            align_bank=align_bank, w_llm=w_llm, tau_llm=tau_llm,
            w_align=w_align, tau_align=tau_align, sem_bank=sem_bank,
            align_target_bank=align_target_bank,
        )
        val_loss = val_meters.mean("loss")
        print(
            f"epoch {epoch + 1} | train: {epoch_meters.summary(_LOG_KEYS)} "
            f"| val: {val_meters.summary(_LOG_KEYS)}"
        )

        improved = stopper.update({"epoch": epoch + 1, "val_loss": val_loss})
        if improved:
            torch.save(qformer.state_dict(), qformer_out)
            torch.save(llm_proj.state_dict(), proj_out)
            log_step(
                "Saved best Stage 2 checkpoint",
                f"epoch={epoch + 1}, val_loss={val_loss:.4f}, "
                f"qformer={qformer_out}, proj={proj_out}",
            )
        else:
            best_epoch = (
                stopper.best_full_metric["epoch"]
                if stopper.best_full_metric is not None
                else "n/a"
            )
            log_step(
                "No validation improvement",
                f"counter={stopper.counter}, best_epoch={best_epoch}, "
                f"best_val_loss={stopper.best_metric_val:.4f}",
            )

        if stopper.should_stop:
            log_step(
                "Early stopping triggered",
                f"epoch={epoch + 1}, best_val_loss={stopper.best_metric_val:.4f}",
            )
            break

    return qformer_out, proj_out


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Former stage 2 generative pretraining")
    parser.add_argument("--cfg-path", default="configs/config.yaml", type=str)
    parser.add_argument(
        "--options", nargs="+", help="Override config settings in key=value format."
    )
    return parser.parse_args()


def main():
    cfg = Config(parse_args())
    stage2_cfg = cfg.run_cfg.get("qformer_stage2")
    if stage2_cfg is None:
        raise KeyError("Missing 'run.qformer_stage2' section in configuration.")

    required_keys = [
        "seed",
        "batch_size",
        "num_workers",
        "embedding_size",
        "user_num",
        "item_num",
        "num_queries",
        "num_heads",
        "num_layers",
        "qformer_d_model",
        "qformer_text_model_name",
        "max_instruction_length",
        "llm_model_name",
        "lr",
        "weight_decay",
        "epoch",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "pretrained_rec_path",
        "qformer_ckpt_in",
        "qformer_ckpt_out",
        "proj_ckpt_out",
        "output_dir",
    ]
    missing = [k for k in required_keys if k not in stage2_cfg]
    if missing:
        raise KeyError("Missing required keys in 'run.qformer_stage2': " + ", ".join(missing))

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    stage2_cfg.data_dir = cfg.datasets_cfg[first_dataset_key].path

    train_qformer_stage2_generative(stage2_cfg)


if __name__ == "__main__":
    main()
