"""Stage 2 — Generative pretraining of Q-Former + projection (BLIP-2 style).

Loads the Q-Former weights from Stage 1, attaches a fresh ``nn.Linear``
projection into the LLM's hidden size, and trains Q-Former + projection
with next-token language modeling on item-text captions while keeping the
LLM fully frozen. The Q-Former runs uni-modal here (queries cross-attend
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
from sigllm.common.config import Config
from sigllm.datasets.qformer.qformer_alignment_builder import QFormerAlignmentBuilder
from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset
from sigllm.datasets.qformer.qformer_loader import build_qformer_loader
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


def _extract_qformer_state_dict(ckpt, qformer):
    """Normalize a checkpoint into a bare HFQFormerAdapter state_dict.

    Handles the three forms produced across stages:
      1. bare adapter state_dict (Stage 1 `best_qformer_weights_name`, Stage 2 output);
      2. wrapped full checkpoint {"epoch","model","optimizer",...} (Stage 1 `best_full`);
      3. Stage-1 alignment-model state where the adapter lives under a `qformer.` prefix
         alongside `mf.*` / `itm_head.*` (the inner InstructBLIP adds its own `qformer.`
         keys, so the adapter shares a name with its parent attribute).
    """
    sd = ckpt
    if isinstance(sd, dict) and "model" in sd and "epoch" in sd:
        sd = sd["model"]
    adapter_keys = set(qformer.state_dict().keys())
    if not (adapter_keys & set(sd.keys())):
        # No direct overlap -> the adapter is nested under a "qformer." prefix.
        extracted = {
            k[len("qformer."):]: v for k, v in sd.items() if k.startswith("qformer.")
        }
        if extracted:
            sd = extracted
    return sd


def _init_qformer(cfg, device):
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
    ).to(device)

    ckpt_path = cfg.get("qformer_ckpt_in")
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_qformer_state_dict(ckpt, qformer)
        qformer.load_state_dict(state_dict, strict=True)
        log_step("Loaded Phase 1 Q-Former", ckpt_path)
    else:
        log_step("WARNING", f"qformer_ckpt_in not found at {ckpt_path}; training from scratch")

    for p in qformer.parameters():
        p.requires_grad = True
    return qformer


def _init_llm(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()
    return tokenizer, llm


def _build_projection(d_q: int, hidden_size: int, device) -> nn.Module:
    """Linear + LayerNorm projection.

    LayerNorm.weight is initialised at ``1/sqrt(hidden_size)`` so the output
    has ``mean_l2 ~ 1`` from step 0, matching Vicuna's native input
    embedding scale. Default LayerNorm init (weight=1) gives mean_l2 ~
    sqrt(H) ~ 64, which drowns the text portion of the prompt under the
    frozen attention.
    """
    proj = nn.Sequential(
        nn.Linear(d_q, hidden_size),
        nn.LayerNorm(hidden_size),
    ).to(device).float()
    nn.init.normal_(proj[0].weight, std=0.02)
    nn.init.zeros_(proj[0].bias)
    nn.init.constant_(proj[1].weight, hidden_size ** -0.5)
    nn.init.zeros_(proj[1].bias)
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


def _build_stage2_instructions(batch_size: int, training: bool) -> list:
    """Item-text instructions for the Q-Former, drawn from the SAME templates
    Stage 1 used (``QFormerAlignmentBuilder.TEMPL_ITEM_TEXT``) so Stage 2 sees
    the instruction distribution it was aligned on. Fresh per row in training,
    deterministic (template 0) at eval."""
    templates = QFormerAlignmentBuilder.TEMPL_ITEM_TEXT
    if training:
        return random.choices(templates, k=batch_size)
    return [templates[0]] * batch_size


def forward_stage2(
    batch, mf, qformer, llm_proj, tokenizer, llm, max_caption_length: int,
    input_mode: str = "multimodal", training: bool = True,
):
    item_ids = batch["i_left"]
    captions = batch["text"]

    with torch.no_grad():
        item_cf = mf.item_encoder(item_ids)

    ctx = qformer.pack_item_context(item_cf)
    if input_mode == "multimodal":
        # CHANGE 2g: run the Q-Former through the SAME instruction-aware joint
        # forward used at Stage 3 (queries + instruction text cross-attending the
        # CF sequence), instead of the uni-modal queries-only ``encode_cf``. This
        # removes the Stage-2 vs Stage-3 input mismatch, so the projection the
        # LLM consumes is produced by the same code path in both stages.
        # ``qformer(...)`` already applies ``out_proj``.
        instructions = _build_stage2_instructions(item_ids.size(0), training)
        query_tokens = qformer(ctx[0], ctx[1], ctx[2], ctx[3], instructions, user_mask=ctx[4], target_mask=ctx[5])
    else:
        # Legacy BLIP-2 uni-modal pretraining (queries cross-attend the CF
        # vector only, no text input). Kept for ablation / reproducibility.
        query_tokens = qformer.encode_cf(*ctx)
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
    return outputs.loss


def evaluate(loader, mf, qformer, llm_proj, tokenizer, llm, device, max_caption_length: int,
             input_mode: str = "multimodal"):
    qformer.eval()
    llm_proj.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss = forward_stage2(
                batch, mf, qformer, llm_proj, tokenizer, llm, max_caption_length,
                input_mode=input_mode, training=False,
            )
            total += float(loss.item())
            steps += 1
    qformer.train()
    llm_proj.train()
    if steps == 0:
        return 0.0
    return total / steps


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
    qformer = _init_qformer(cfg, device).train()
    tokenizer, llm = _init_llm(cfg.llm_model_name, device)

    d_q = qformer.output_dim
    hidden_size = llm.config.hidden_size
    llm_proj = _build_projection(d_q, hidden_size, device).train()

    trainable_params = [
        p for p in qformer.parameters() if p.requires_grad
    ] + list(llm_proj.parameters())
    optimizer = Adam(
        trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    scaler = torch.amp.GradScaler("cuda")

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
    # CHANGE 2g: "multimodal" (default) matches the Stage-3 forward path;
    # "unimodal" reproduces the legacy BLIP-2 queries-only pretraining.
    input_mode = str(cfg.get("qformer_input_mode", "multimodal"))
    log_step("Stage 2 Q-Former input mode", input_mode)

    for epoch in range(int(cfg.epoch)):
        qformer.train()
        llm_proj.train()
        train_total = 0.0
        train_steps = 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = forward_stage2(
                batch, mf, qformer, llm_proj, tokenizer, llm, max_caption_length,
                input_mode=input_mode, training=True,
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.item())
            train_steps += 1

        avg_train_loss = train_total / max(train_steps, 1)
        if (epoch + 1) % log_epoch != 0:
            continue

        val_loss = evaluate(
            valid_loader, mf, qformer, llm_proj, tokenizer, llm, device, max_caption_length,
            input_mode=input_mode,
        )
        print(
            f"epoch {epoch + 1} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}"
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
