
import logging
import profile
import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sigllm.common.utils import resolve_hf_model_path

import os

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.registry import registry
from sigllm.models.multimodal.base.rec_base_model import Rec2Base
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_model")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def tensor_stat_string(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    if tensor.numel() == 0:
        return f"{name}=empty shape={tuple(tensor.shape)}"

    detached = tensor.detach().float()
    mean_val = detached.mean().item()
    std_val = detached.std(unbiased=False).item()
    norm_val = detached.norm(dim=-1).mean().item() if detached.dim() >= 2 else detached.norm().item()
    return (
        f"{name}: shape={tuple(detached.shape)}, "
        f"mean={mean_val:.4f}, std={std_val:.4f}, mean_l2={norm_val:.4f}"
    )


@registry.register_model("mini_gpt4rec_v2")
class QRecLLM(Rec2Base):
    """
    QFormer + InstructBLIP for recommendation.
    """ 
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }    
    
    # TEMP_DISABLED_USER_CF: old prompt order included a user soft-token slot.
    # PLACEHOLDERS_FOR_EMBED = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    # PLACEHOLDERS_FOR_EMBED = ["<ItemIDList>", "<TargetItemID>"]
    PLACEHOLDERS_FOR_EMBED = ["<UserProfile>", "<TargetItemID>"]

    # Item-text instructions for the Q-Former. Must match the distribution
    # the Q-Former was trained on in stage 1 (see
    # QFormerAlignmentBuilder.TEMPL_ITEM_TEXT). The verbose stage 2 prompt
    # MUST NOT be passed here — it gets truncated to max_instruction_length
    # tokens and would carry no per-item signal.
    QFORMER_ITEM_INSTRUCTIONS = [
        "Represent this movie for recommendation using its title and genres.",
        "Align this movie metadata with its collaborative filtering representation.",
        "Given the movie metadata, extract recommendation-relevant item features.",
        "Use the title and genres to describe this movie in the item embedding space.",
        "Map this movie's textual attributes to its collaborative recommendation signal.",
        "Identify the movie preferences implied by its title and genre metadata.",
        "Create a language-aligned representation of this movie for recommendation.",
        "Summarize this movie as an item a recommender system can compare.",
        "Based on the title and genres, represent what kind of users may like this movie.",
        "Encode the semantic information of this movie for item-language alignment.",
        "Use a few metadata cues to align this movie with behavioral item signals.",
        "Produce a recommendation-aware representation from this movie description.",
    ]

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        pretrained_qformer=None,
        pretrained_llm_proj=None,
        freeze_rec=True,
        llm_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=1024,
        end_sym='\n',
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        num_queries=8,
        num_heads=8,
        num_layers=2,
        qformer_d_model=768,
        qformer_output_dim=None,
        qformer_text_model_name="bert-base-uncased",
        max_instruction_length=48,
        freeze_proj=False,
        ablate_soft_tokens=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=("q_proj", "v_proj"),
        lora_dropout=0.05,
        tuning_step=None,
        user_conditioned=False,
        warm_token=False,
        direct_id_tokens=False,
        candidate_fusion=False,
        pretrained_item_llm_emb=None,
        ranking_loss_weight=0.0,
        ranking_loss_tau=1.0,
        align_rank_loss_weight=0.0,
        align_rank_loss_tau=1.0,
        sem_source=False,
        item_sem_emb_path=None,
        sem_source_dropout=0.5,
        mf_drift_log_steps=200,
    ):
        super().__init__()

        # How often (training steps) to log the joint-tuning MF drift
        # diagnostic. Only fires when tuning_step=3 took a weight snapshot;
        # 0 disables. See _maybe_log_mf_drift.
        self.mf_drift_log_steps = int(mf_drift_log_steps)
        self.proj_token_num = proj_token_num
        self._has_logged_trainable_stats = False
        self._flow_log_steps = 0
        self._max_flow_log_steps = 3
        self._has_logged_prompt_injection_stats = False
        self._eval_pred_log_count = 0
        self._max_eval_pred_log_batches = 20

        self.ablate_soft_tokens = bool(ablate_soft_tokens)
        if self.ablate_soft_tokens:
            log_step(
                "ABLATION ACTIVE",
                "ablate_soft_tokens=True → target_llm and the <UserProfile> "
                "hisotry-pooled tokens will be zeroed before injection (Information flow log will show "
                "target_llm mean/std=0).",
            )

        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_dropout = float(lora_dropout)
        self.tuning_step = tuning_step
        self.user_conditioned = bool(user_conditioned)
        self.warm_token = bool(warm_token)
        self.direct_id_tokens = bool(direct_id_tokens)
        self.candidate_fusion = bool(candidate_fusion)
        self.embed_placeholders = list(self.PLACEHOLDERS_FOR_EMBED)
        if self.warm_token and "<Warm_ID>" not in self.embed_placeholders:
            self.embed_placeholders = ["<UserProfile>", "<Warm_ID>", "<TargetItemID>"]
        if self.direct_id_tokens:
            # SeLLa-style direct path, PARALLEL to the Q-Former: <UserID> and
            # <ItemID> inject MLP(e_u) / MLP(e_i) as ONE soft token each, so
            # the LLM sees both raw CF vectors and can in principle recover
            # the MF dot product e_u . e_i (>= MF baseline), on top of the
            # semantic channel.
            for ph in ("<UserID>", "<ItemID>"):
                if ph not in self.embed_placeholders:
                    self.embed_placeholders.append(ph)
            log_step(
                "DIRECT ID TOKENS ACTIVE",
                "<UserID> = user_id_proj(e_u), <ItemID> = item_id_proj(e_i); "
                "1 soft token each, injected next to the Q-Former tokens.",
            )
        if self.candidate_fusion:
            log_step(
                "CANDIDATE FUSION ACTIVE",
                "cross-attention memory slots become m_j = proj_cf(e_j) + "
                "fuse_cf([e_t; e_j*e_t; e_j-e_t]) (+ fuse_user([e_u; e_u*e_t; "
                "e_u-e_t])); zero-init, so warm start is a no-op.",
            )

        # uAUC-aligned auxiliary losses (opt-in). ranking_loss shapes the LLM's
        # Yes/No margin; align_rank_loss shapes the aligned CF soft tokens. Both
        # are per-user BPR (a uAUC surrogate) and need a user-grouped batch
        # sampler to have same-user pos/neg pairs. weight=0.0 -> disabled.
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_loss_tau = float(ranking_loss_tau)
        if self.ranking_loss_weight > 0.0:
            log_step(
                "uAUC ranking loss ACTIVE",
                f"L = BCE + {self.ranking_loss_weight} * per-user BPR "
                f"(tau={self.ranking_loss_tau}). Needs a user-grouped batch sampler "
                f"to be effective.",
            )
        self.align_rank_loss_weight = float(align_rank_loss_weight)
        self.align_rank_loss_tau = float(align_rank_loss_tau)
        self.align_rank_head = None

        if self.user_conditioned:
            log_step(
                "CANDIDATE-CONDITIONED MODE",
                "user_conditioned=True → the <UserProfile> history pooling is "
                "conditioned on the TARGET ITEM's CF vector (user_proj(target_cf) "
                "shifts the Q tokens), so the pooled profile varies per candidate "
                "and can move uAUC. A per-user shift would be constant across a "
                "user's candidates and cancel in uAUC.",
            )

        log_step("Running MiniGPT4Rec_v2 initialization")

        self.rec_model_type = rec_model

        # Initialize components
        self._init_rec_model(rec_model, rec_config, pretrained_rec, freeze_rec)
        self._init_llm_model(llm_model)
        self._init_sem_source(sem_source, item_sem_emb_path, sem_source_dropout)
        self._init_qformer(
            d_cf=rec_config.embedding_size,
            d_model=qformer_d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            qformer_output_dim=qformer_output_dim,
            pretrained_qformer=pretrained_qformer,
            freeze_qformer=False,
            qformer_text_model_name=qformer_text_model_name,
            max_instruction_length=max_instruction_length,
            user_conditioned=self.user_conditioned,
            d_user=rec_config.embedding_size,
            d_sem=self.item_sem_emb.size(-1) if self.item_sem_emb is not None else None,
            candidate_fusion=self.candidate_fusion,
        )
        self._init_projection(proj_token_num, freeze_proj, pretrained_llm_proj)
        self._init_warm_token(pretrained_item_llm_emb)
        self._init_direct_id_proj(rec_config.embedding_size, pretrained_rec)
        self._init_prompts(prompt_path, prompt_template, max_txt_len, end_sym)
        self._apply_tuning_step_policy()
        # Built after the LLM (reads hidden_size) and after the step policy so it
        # co-trains with the unfrozen Q-Former/projection at Step 2.
        self._init_align_rank_head()

    def _init_rec_model(self, rec_model, rec_config, pretrained_rec, freeze_rec):
        log_step("Loading Rec_model")
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config)

        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            log_step("Successfully loaded the pretrained model")

        # ``freeze_rec`` is the single source of truth, including at step 3.
        # That keeps the two independent questions separable: "does LoRA +
        # Q-Former + projection co-adapting fix the 2-step failure?" and "does
        # letting MF move help?". Step 3 with freeze_rec=True answers the first
        # alone; the step-3 script sets freeze_rec=False by default so the
        # documented joint-with-MF behaviour is unchanged.
        #
        # Skipping the freeze (rather than undoing it later) is deliberate:
        # freezing installs a ``train = disabled_train`` patch on the instance,
        # and un-patching is easy to get subtly wrong — the attribute shadows the
        # class method, so it must be deleted, not reassigned.
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            log_step("Freeze rec encoder")

        log_step("Loading Rec_model Done")

    def _init_llm_model(self, llm_path):
        log_step(f"Loading LLM: {llm_path}")
        model_path = llm_path if llm_path else "./content/ckpt/llm/base"
        model_path, local_files_only = resolve_hf_model_path(model_path)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self._resolve_soft_token_placeholder()
        log_step(
            "Loading LLM Done",
            f"hidden_size={self.llm_model.config.hidden_size}, "
            f"pad_token_id={self.llm_tokenizer.pad_token_id}, "
            f"soft_token_id={self._soft_token_id} ('{self._soft_token_str}')",
        )

        if self.use_lora:
            self._attach_lora()

    def _resolve_soft_token_placeholder(self):
        tok = self.llm_tokenizer
        if tok.unk_token_id is not None:
            self._soft_token_str = tok.unk_token
            self._soft_token_id = tok.unk_token_id
            return

        skip_ids = {tok.eos_token_id, tok.pad_token_id, tok.bos_token_id}
        skip_ids.discard(None)

        hardcoded = (
            "<|extra_0|>", "<|reserved_0|>", "<|fim_pad|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>",
            "<|im_start|>",
        )
        for candidate in hardcoded:
            ids = tok(candidate, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] not in skip_ids:
                self._soft_token_str = candidate
                self._soft_token_id = ids[0]
                return

        added = getattr(tok, "added_tokens_decoder", None) or {}
        for token_id, added_token in added.items():
            if token_id in skip_ids:
                continue
            content = getattr(added_token, "content", str(added_token))
            ids = tok(content, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] == token_id:
                self._soft_token_str = content
                self._soft_token_id = token_id
                return

        log_step(
            "Soft-token fallback",
            "no unk_token and no safe single-token candidate; using eos_token "
            "as soft-slot placeholder. Soft slots will COLLIDE with padding if "
            "pad_token == eos_token — Step 2 may corrupt embeddings silently.",
        )
        self._soft_token_str = tok.eos_token
        self._soft_token_id = tok.eos_token_id

    def _attach_lora(self):
        from peft import LoraConfig, TaskType, get_peft_model

        log_step(
            "Attaching LoRA to LLM",
            f"r={self.lora_r}, alpha={self.lora_alpha}, "
            f"target_modules={list(self.lora_target_modules)}, dropout={self.lora_dropout}",
        )
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.lora_target_modules),
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        log_step(
            "LoRA attached",
            f"trainable LoRA params={count_trainable_parameters(self.llm_model)}",
        )

    def _apply_tuning_step_policy(self):
        step = self.tuning_step
        if step is None:
            return

        # The direct-ID projections follow the same schedule as warm_proj:
        # frozen at step 1 (LoRA-only), trainable at steps 2/3.
        id_projs = [
            m for m in (getattr(self, "user_id_proj", None), getattr(self, "item_id_proj", None))
            if m is not None
        ]

        if int(step) == 1:
            for p in self.qformer.parameters():
                p.requires_grad = False
            for p in self.llm_proj.parameters():
                p.requires_grad = False
            self.qformer.eval()
            self.qformer.train = disabled_train
            self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            if getattr(self, "warm_proj", None) is not None:
                for p in self.warm_proj.parameters():
                    p.requires_grad = False
                self.warm_proj.eval()
                self.warm_proj.train = disabled_train
            for m in id_projs:
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()
                m.train = disabled_train
            log_step(
                "Tuning step 1",
                "LoRA trainable; Q-Former, projection, warm_proj, direct-ID projections, "
                "MF and base LLM all frozen.",
            )

        elif int(step) == 2:
            # CoLLM Equation (5), Ω = ϕ variant: Q-Former + projection trainable,
            # LoRA + base LLM + MF frozen. Q-Former was briefly frozen here as an
            # α2 experiment (theory: 8-layer Q-Former too large for 33k samples);
            # plateaued at uAUC ~0.694, below baseline 0.708, because llm_proj
            # alone (~2.8M params) lacks capacity to fix the CIE channel. Unfrozen
            # again to let Q-Former co-adapt with projection on the recommendation
            # task — matches the original CoLLM recipe.
            if hasattr(self.llm_model, "peft_config"):
                for n, p in self.llm_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = False
            for p in self.qformer.parameters():
                p.requires_grad = True
            self.qformer.train()
            for p in self.llm_proj.parameters():
                p.requires_grad = True
            self.llm_proj.train()
            if getattr(self, "warm_proj", None) is not None:
                for p in self.warm_proj.parameters():
                    p.requires_grad = True
                self.warm_proj.train()
            for m in id_projs:
                for p in m.parameters():
                    p.requires_grad = True
                m.train()
            # MF follows freeze_rec here too (SeLLa retrains the CF embeddings
            # jointly; freeze_rec=False in the step-2 block opts in). Snapshot
            # for the mf_drift diagnostic exactly as step 3 does — an unfrozen
            # MF without the drift log is flying blind.
            mf_trainable = self.rec_encoder is not None and any(
                p.requires_grad for p in self.rec_encoder.parameters()
            )
            if mf_trainable:
                self.rec_encoder.train()
                self._snapshot_mf_weights()
            log_step(
                "Tuning step 2",
                "Q-Former + projection + warm_proj + direct-ID projections trainable; "
                "LoRA, base LLM frozen; MF "
                + ("TRAINABLE (freeze_rec=False) — watch mf_drift" if mf_trainable else "frozen"),
            )

        elif int(step) == 3:
            # JOINT tuning: LoRA + Q-Former + projection + MF all trainable,
            # base LLM still frozen. Steps 1 and 2 each freeze exactly what the
            # other trains, so neither can co-adapt the CF encoder with the
            # channel that reads it; this step does.
            #
            # Warm-start from the Step-1 LoRA checkpoint rather than from cold:
            # with everything trainable at once and the soft-token channel still
            # noisy, LoRA takes the cheaper route and learns to answer from the
            # prompt text alone — the same failure that made Step 2 flat when
            # Step 1 had been trained on the text-only prompt.
            #
            # MF carries the geometry Stage 1/2 aligned against, so it must move
            # SLOWLY: see run.rec_lr_scale (its own optimizer param group) and
            # the mf_drift diagnostic. A fast-moving MF invalidates the
            # alignment those stages produced, which is the same class of
            # failure as the collapsed-body A/B run (b).
            if hasattr(self.llm_model, "peft_config"):
                for n, p in self.llm_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True

            # The Q-Former stays in the joint set whenever MF is trainable, and
            # that is not optional: Stage 1/2 taught it to read a SPECIFIC MF
            # geometry, so if MF moves and the Q-Former is frozen, a fixed
            # mapping is being fed a changed input. They have to move together
            # or not at all.
            for p in self.qformer.parameters():
                p.requires_grad = True
            self.qformer.train()
            for p in self.llm_proj.parameters():
                p.requires_grad = True
            self.llm_proj.train()
            if getattr(self, "warm_proj", None) is not None:
                for p in self.warm_proj.parameters():
                    p.requires_grad = True
                self.warm_proj.train()
            for m in id_projs:
                for p in m.parameters():
                    p.requires_grad = True
                m.train()

            # MF follows freeze_rec, so joint tuning comes in two flavours and
            # the log has to say which one is running — the difference decides
            # whether mf_drift means anything and whether the Stage-1/2
            # alignment is being held fixed or moved underneath.
            mf_trainable = self.rec_encoder is not None and any(
                p.requires_grad for p in self.rec_encoder.parameters()
            )
            if mf_trainable:
                self.rec_encoder.train()
                self._snapshot_mf_weights()

            log_step(
                "Tuning step 3 (JOINT)",
                "LoRA + Q-Former + projection + warm_proj trainable"
                + (
                    " + MF (freeze_rec=False) — watch mf_drift against "
                    "run.rec_lr_scale; if drift stays ~0 the MF arm of this "
                    "experiment is not actually running."
                    if mf_trainable
                    else " ; MF FROZEN (freeze_rec=True)."
                )
                + " Base LLM frozen either way.",
            )

        else:
            log_step("Tuning step", f"unrecognized value '{step}', no policy applied")

    def _init_sem_source(self, sem_source, item_sem_emb_path, sem_source_dropout):
        """Frozen semantic item embeddings used as the SECOND cross-attention
        source next to CF (see HFQFormerAdapter.d_sem). Loaded RAW: the bank
        goes through the adapter's trainable proj_sem, so no normalization is
        applied here. Must be the SAME bank Stage 1/2 trained proj_sem on
        (input-space distill), not the last-hidden warm-token bank."""
        self.sem_source_dropout = float(sem_source_dropout)
        if not sem_source:
            self.item_sem_emb = None
            return
        if not item_sem_emb_path or not os.path.exists(item_sem_emb_path):
            raise FileNotFoundError(
                f"sem_source=True but item_sem_emb_path not found: {item_sem_emb_path}"
            )
        blob = torch.load(item_sem_emb_path, map_location="cpu")
        bank = (blob["item_llm_emb"] if isinstance(blob, dict) else blob).float()
        self.register_buffer("item_sem_emb", bank.to(self.device), persistent=False)
        log_step(
            "Semantic cross-attention source active",
            f"bank={tuple(bank.shape)} from {item_sem_emb_path}, "
            f"dropout={self.sem_source_dropout} (training only)",
        )

    def _sem_for_items(self, item_ids):
        """Semantic rows for ``item_ids`` (any leading shape). Training-time
        row dropout zeroes whole rows; the adapter masks zero rows out, so a
        dropped row falls back to CF-only cross-attention."""
        if self.item_sem_emb is None:
            return None
        sem = self.item_sem_emb[item_ids]
        if self.training and self.sem_source_dropout > 0.0:
            keep = (
                torch.rand(sem.shape[:-1], device=sem.device) >= self.sem_source_dropout
            ).to(sem.dtype).unsqueeze(-1)
            sem = sem * keep
        return sem

    def _init_qformer(
        self,
        d_cf,
        d_model,
        num_queries,
        num_heads,
        num_layers,
        qformer_output_dim,
        pretrained_qformer: str,
        freeze_qformer: bool,
        qformer_text_model_name: str,
        max_instruction_length: int,
        user_conditioned: bool = False,
        d_user: int = None,
        d_sem: int = None,
        candidate_fusion: bool = False,
    ):
        log_step("Loading QFormer")
        log_step(
            "Using Q-Former tokenizer for instructions",
            f"tokenizer={qformer_text_model_name}, hidden_size={d_model}",
        )

        self.qformer = HFQFormerAdapter(
            d_cf=d_cf,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=qformer_output_dim or d_model,
            qformer_text_model_name=qformer_text_model_name,
            max_instruction_length=max_instruction_length,
            init_from_pretrained_text=False,
            user_conditioned=user_conditioned,
            d_user=d_user,
            d_sem=d_sem,
            candidate_fusion=candidate_fusion,
        ).to(self.device)

        if pretrained_qformer and pretrained_qformer != "not_have":
            ckpt = torch.load(pretrained_qformer, map_location="cpu")
            state_dict = ckpt
            if isinstance(state_dict, dict) and any(k.startswith("qformer.") for k in state_dict.keys()):
                state_dict = {k.replace("qformer.", "", 1): v for k, v in state_dict.items()}
            load_msg = self.qformer.load_state_dict(state_dict, strict=False)
            if load_msg.missing_keys:
                log_step(
                    "QFormer ckpt missing keys (kept at init)",
                    ", ".join(load_msg.missing_keys),
                )
            if load_msg.unexpected_keys:
                log_step(
                    "QFormer ckpt unexpected keys (ignored)",
                    ", ".join(load_msg.unexpected_keys),
                )
            log_step("Successfully loaded QFormer checkpoint", pretrained_qformer)

        # 3) freeze / train tiếp
        if freeze_qformer:
            for p in self.qformer.parameters():
                p.requires_grad = False
            self.qformer.eval()
            self.qformer.train = disabled_train
            log_step("Freeze QFormer")
        else:
            for p in self.qformer.parameters():
                p.requires_grad = True
            self.qformer.train()
            log_step("Train QFormer in stage 3")

        log_step("Loading QFormer Done")
        return self.qformer

    def _init_projection(self, proj_token_num, freeze_proj, pretrained_llm_proj=None):
        """
        Stage 3 projection: map Q-Former output tokens -> LLM hidden tokens.
        Input  : qformer_out [B, Q, d_q]
        Output : llm_tokens  [B, Q, H]

        Matches InstructBLIP: a single ``nn.Linear`` from Q-Former hidden size
        to LLM hidden size, applied per token. If ``pretrained_llm_proj``
        points to a state dict (e.g. from Stage 2 generative pretraining), it
        is loaded before any freezing.
        """
        log_step("Loading Projection (QFormer -> LLM)")

        if self.qformer is None:
            raise ValueError("qformer is None. Please init/load Q-Former before init projection.")
        if not hasattr(self.qformer, "q"):
            raise ValueError("qformer.q (learned query tokens) is required to infer num_queries.")
        if self.llm_model is None:
            raise ValueError("llm_model is None. Please init LLM backbone before init projection.")

        d_q = self.qformer.output_dim
        Q = int(self.qformer.q.shape[-2])
        H = int(self.llm_model.config.hidden_size)

        # luôn sync theo Q-Former để tránh lệch số <unk> khi inject
        self.proj_token_num = Q
        if proj_token_num is not None and int(proj_token_num) != Q:
            log_step("WARNING",
                    f"proj_token_num({proj_token_num}) != qformer.num_queries({Q}). "
                    f"Using Q={Q} to keep injection consistent.")

        self.llm_proj = nn.Sequential(
            nn.Linear(d_q, H),
            nn.LayerNorm(H),
        )
        nn.init.normal_(self.llm_proj[0].weight, std=0.02)
        nn.init.zeros_(self.llm_proj[0].bias)
        nn.init.constant_(self.llm_proj[1].weight, H ** -0.5)
        nn.init.zeros_(self.llm_proj[1].bias)

        if pretrained_llm_proj and pretrained_llm_proj != "not_have" and os.path.exists(pretrained_llm_proj):
            state_dict = torch.load(pretrained_llm_proj, map_location="cpu")
            self.llm_proj.load_state_dict(state_dict, strict=True)
            log_step("Loaded Stage 2 projection", pretrained_llm_proj)

        if freeze_proj:
            for p in self.llm_proj.parameters():
                p.requires_grad = False
            self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            log_step("Freeze llm_proj")

        log_step("Loading Projection Done",
                f"d_q={d_q}, H={H}, Q={self.proj_token_num}")

    def _init_warm_token(self, pretrained_item_llm_emb):
        if not self.warm_token:
            self.item_llm_emb = None
            self.warm_proj = None
            return

        if not pretrained_item_llm_emb or not os.path.exists(pretrained_item_llm_emb):
            raise FileNotFoundError(f"warm_token=True but pretrained_item_llm_emb not found: {pretrained_item_llm_emb}")

        H = int(self.llm_model.config.hidden_size)
        blob = torch.load(pretrained_item_llm_emb, map_location="cpu")
        table = blob["item_llm_emb"] if isinstance(blob, dict) else blob
        table = table.float()

        if table.size(-1) != H:
            raise ValueError(f"pretrained_item_llm_emb has hidden size {table.size(-1)}, expected {H}")

        self.register_buffer("item_llm_emb", table.to(self.device), persistent=False)
        self.warm_proj = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H)).to(self.device)
        nn.init.normal_(self.warm_proj[0].weight, std=0.02)
        nn.init.zeros_(self.warm_proj[0].bias)
        nn.init.constant_(self.warm_proj[1].weight, H ** -0.5)
        nn.init.zeros_(self.warm_proj[1].bias)
        log_step(
            "Warm token active", 
            f"<Warm_ID> injects warm_proj(e^L_item) [table={tuple(table.shape)}] into LLM embedding space. "
            f"Carries LLM semantic knowledge for cold-start items."
        )

    def _init_direct_id_proj(self, d_cf, pretrained_rec):
        """Build the SeLLa-style direct projections for <UserID> / <ItemID>.

        Each is Linear(d_cf, hidden) -> GELU -> Linear(hidden, H), mirroring
        SeLLa's ``LinearProjection``. When the pretrained MF checkpoint was
        trained WITH the Step-2 semantic alignment (carries trans_1/trans_2),
        both MLPs warm-start from those weights — SeLLa's
        ``pretrained_with_small=True``, the full CL + Projection setting the
        paper describes (the uploaded SeLLa code ships with False).
        """
        if not self.direct_id_tokens:
            self.user_id_proj = None
            self.item_id_proj = None
            return

        H = int(self.llm_model.config.hidden_size)

        def build_mlp(hidden):
            mlp = nn.Sequential(nn.Linear(d_cf, hidden), nn.GELU(), nn.Linear(hidden, H))
            nn.init.normal_(mlp[0].weight, std=0.02)
            nn.init.zeros_(mlp[0].bias)
            nn.init.normal_(mlp[2].weight, std=0.02)
            nn.init.zeros_(mlp[2].bias)
            return mlp

        hidden = 1024
        trans_state = None
        if pretrained_rec and pretrained_rec != "not_have" and os.path.exists(pretrained_rec):
            mf_state = torch.load(pretrained_rec, map_location="cpu")
            if "trans_1.weight" in mf_state and "trans_2.weight" in mf_state:
                if int(mf_state["trans_2.weight"].shape[0]) == H:
                    hidden = int(mf_state["trans_1.weight"].shape[0])
                    trans_state = mf_state
                else:
                    log_step(
                        "Direct ID proj: SKIPPED trans warm-start",
                        f"MF trans_2 outputs {int(mf_state['trans_2.weight'].shape[0])} "
                        f"dims but LLM hidden is {H} — different base LLM? "
                        f"Falling back to fresh init.",
                    )

        self.user_id_proj = build_mlp(hidden).to(self.device)
        self.item_id_proj = build_mlp(hidden).to(self.device)

        if trans_state is not None:
            for mlp in (self.user_id_proj, self.item_id_proj):
                mlp[0].weight.data.copy_(trans_state["trans_1.weight"])
                mlp[0].bias.data.copy_(trans_state["trans_1.bias"])
                mlp[2].weight.data.copy_(trans_state["trans_2.weight"])
                mlp[2].bias.data.copy_(trans_state["trans_2.bias"])
            log_step(
                "Direct ID proj warm-started from MF trans_1/trans_2",
                f"(SeLLa pretrained_with_small=True) {d_cf}->{hidden}->{H}",
            )
        else:
            log_step("Direct ID proj fresh init", f"{d_cf}->{hidden}->{H}")

    def _snapshot_mf_weights(self):
        """Snapshot the MF weights for the ``mf_drift`` diagnostic.

        Taken during ``__init__`` (from the step policy), which is BEFORE
        ``from_config`` applies any ``model.ckpt``. The reference is therefore
        the **pretrained** MF — precisely the geometry Stage 1/2 aligned the
        Q-Former against — and not "wherever this run happened to resume from".
        That is the intended baseline: the question drift answers is how far MF
        has moved away from the alignment those stages were built on, so a
        resumed run should keep measuring against the same origin.

        Costs one extra copy of the embedding tables (~4 MB at 4k ids x 256
        dims), kept on CPU so it never competes with activations for GPU
        memory. Only taken for tuning_step=3; every other step has MF frozen,
        where drift is zero by construction.
        """
        self._mf_init_weights = {
            name: param.detach().to("cpu", copy=True)
            for name, param in self.rec_encoder.named_parameters()
        }

    @torch.no_grad()
    def mf_drift(self):
        """Relative movement of the MF weights away from the pretrained MF, as
        ``{name: ||W - W0|| / ||W0||}`` plus an ``overall`` figure. See
        ``_snapshot_mf_weights`` for what W0 is.

        This is the early-warning signal for the main risk of joint tuning: the
        Stage-1/2 alignment was learned against a FIXED MF geometry, so if MF
        moves far, that alignment is being invalidated while the LLM-side
        channel tries to chase it. Read it together with val uAUC — rising
        drift with flat or falling uAUC means the MF LR is too high (lower
        ``run.rec_lr_scale``); near-zero drift means MF is effectively frozen
        and Step 3 is not buying anything over Step 2.

        Returns an empty dict when no snapshot exists (any step but 3).
        """
        snapshot = getattr(self, "_mf_init_weights", None)
        if not snapshot or self.rec_encoder is None:
            return {}

        drift = {}
        total_sq, base_sq = 0.0, 0.0
        for name, param in self.rec_encoder.named_parameters():
            if name not in snapshot:
                continue
            initial = snapshot[name].to(param.device, dtype=param.dtype)
            delta_sq = (param.detach() - initial).pow(2).sum().item()
            initial_sq = initial.pow(2).sum().item()
            drift[name] = (delta_sq ** 0.5) / (initial_sq ** 0.5 + 1e-12)
            total_sq += delta_sq
            base_sq += initial_sq
        drift["overall"] = (total_sq ** 0.5) / (base_sq ** 0.5 + 1e-12)
        return drift

    def _maybe_log_mf_drift(self):
        """Log ``mf_drift`` every ``mf_drift_log_steps`` training steps.

        Drift is only useful as an EARLY warning — knowing at the end of a
        200-epoch run that MF walked too far is knowing too late. Throttled
        because it touches every MF parameter; at the default interval the cost
        is two reductions over ~1M params plus a 4 MB host-to-device copy of
        the snapshot, which is noise next to one LLM forward.
        """
        if not self.training or not getattr(self, "_mf_init_weights", None):
            return
        interval = int(getattr(self, "mf_drift_log_steps", 200))
        if interval <= 0:
            return
        self._mf_drift_step = getattr(self, "_mf_drift_step", 0) + 1
        if self._mf_drift_step % interval != 0:
            return
        drift = self.mf_drift()
        if drift:
            log_step(
                f"MF drift (step {self._mf_drift_step})",
                ", ".join(f"{k}={v:.5f}" for k, v in sorted(drift.items()))
                + " | ||W-W0||/||W0||; rising drift with flat val uAUC means "
                "run.rec_lr_scale is too high",
            )

    def _log_trainable_module_stats(self):
        if self._has_logged_trainable_stats:
            return

        llm_total = (
            count_trainable_parameters(self.llm_model) if self.llm_model is not None else 0
        )
        lora_total = 0
        if self.llm_model is not None and hasattr(self.llm_model, "peft_config"):
            lora_total = sum(
                p.numel()
                for n, p in self.llm_model.named_parameters()
                if "lora_" in n and p.requires_grad
            )
        stats = [
            f"rec_encoder={count_trainable_parameters(self.rec_encoder) if self.rec_encoder is not None else 0}",
            f"qformer={count_trainable_parameters(self.qformer) if self.qformer is not None else 0}",
            f"llm_proj={count_trainable_parameters(self.llm_proj) if hasattr(self, 'llm_proj') else 0}",
            f"llm_model={llm_total}",
            f"llm_lora={lora_total}",
        ]
        log_step("Trainable parameter counts", ", ".join(stats))
        self._has_logged_trainable_stats = True

    def _log_information_flow(self, user_q, target_q, user_llm, target_llm, merged_flat):
        if self._flow_log_steps >= self._max_flow_log_steps:
            return

        log_step(
            "Information flow",
            " | ".join(
                [
                    tensor_stat_string("user_q", user_q),
                    tensor_stat_string("target_q", target_q),
                    tensor_stat_string("user_llm", user_llm),
                    tensor_stat_string("target_llm", target_llm),
                    tensor_stat_string("merged_embs", merged_flat),
                ]
            ),
        )
        self._flow_log_steps += 1

    def _init_prompts(self, prompt_path, prompt_template, max_txt_len, end_sym):
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            # TEMP_DISABLED_USER_CF: keep old prompts in the file with this marker,
            # but do not sample them while user CF is disabled.
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts
                if raw_prompt.strip() and not raw_prompt.lstrip().startswith("#")
            ]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            log_step(f"Load {len(self.prompt_list)} training prompts")
            log_step(f"Prompt List: \n{self.prompt_list}")
        else:
            self.prompt_list = []

    def _sample_prompt(self):
        return random.choices(
            self.prompt_list,
            weights=[5] * (len(self.prompt_list) - 1) + [1],
            k=1,
        )[0]

    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode

    def to_be_trained(self):
        # TEMP_DISABLED_USER_CF: old trainable placeholders included "<UserID>".
        # id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        id_terms = ["<UserProfile>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>", "<UserID>", "<ItemID>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True

        if self.llm_model is not None and hasattr(self.llm_model, "peft_config"):
            for n, p in self.llm_model.named_parameters():
                if "lora_" in n and p.requires_grad:
                    return True

        return False

    def set_answer_type(self,mode):
        if mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llm_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llm_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            log_step("answer token ids: pos:{}, neg ids:{}".format(pos_ans_id, neg_ans_id))
            
        else:
            raise NotImplementedError("not implement this types of answers")

    def print_prompt(self):
        log_step('Prompt Pos Example \n{} {} or {}'.format(self._sample_prompt(),self.pos_ans[0],self.neg_ans[0]))
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def get_placeholder_order(self, prompt: str, placeholders=None):
        if placeholders is None:
            placeholders = getattr(self, "embed_placeholders", self.PLACEHOLDERS_FOR_EMBED)
        positions = []
        for ph in placeholders:
            pos = prompt.find(ph)
            if pos >= 0:
                positions.append((pos, ph))
        positions.sort(key=lambda x: x[0])
        return [ph for _, ph in positions]

    def encode_rec_features_to_llm_v2(self, batch_data, feature_order=None, instruction_list=None):
        """
        Encodes recommendation features (History, Target) into LLM embedding space.
        
        Args:
            batch_data (dict): Dictionary containing:
                - 'UserID': (B,)
                - 'TargetItemID': (B,)
                - 'InteractedItemIDs_pad': (B, L)
            feature_order (list): Order of features, e.g., ["<UserProfile>", "<TargetItemID>"]
            
        Returns:
            rec_embeds (dict):
                - 'User_emb': None while TEMP_DISABLED_USER_CF is active
                - 'TargetItem_emb': (B, Q, H) - Target item read out by Q queries
                - 'UserProfile_emb': (B, Q, H) - History POOLED into Q queries
                  (padding excluded via source_mask), or None
                - 'merged_embs': (N, H) - Flattened & filtered valid tokens for LLM input
            rec_atts: None (Placeholder for future attention masks)
        """
        if self.rec_encoder is None:
            return None, None

        self._log_trainable_module_stats()
        self._maybe_log_mf_drift()

        device = batch_data["UserID"].device
        B = batch_data["UserID"].shape[0]
        Q = self.proj_token_num
        H = self.llm_model.config.hidden_size

        if instruction_list is None:
            instruction_list = batch_data.get(
                "instruction",
                self._build_qformer_instructions(B),
            )
        if isinstance(instruction_list, str):
            ins_list = [instruction_list] * B
        else:
            ins_list = list(instruction_list)
        if len(ins_list) != B:
            raise ValueError(f"Expected {B} instructions, got {len(ins_list)}")

        with self.maybe_autocast():
            # Stage-2 uses the in-tree rec encoder API: direct embedding lookup from ids.
            # TEMP_DISABLED_USER_CF: old path injected a user CF token into the LLM prompt.
            user_q = None
            user_llm = None
            target_cf = self.rec_encoder.item_encoder(batch_data["TargetItemID"])  # [B,d_cf]
            target_sem = self._sem_for_items(batch_data["TargetItemID"])           # [B,d_sem] or None

            # The raw user CF vector is needed by the direct <UserID> token and
            # by the user-target fusion term; neither path goes through the
            # Q-Former queries.
            user_cf = None
            if self.direct_id_tokens or self.candidate_fusion:
                user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])       # [B,d_cf]

            # SeLLa-style direct path (1 token each): the LLM sees e_u and e_i
            # unreduced, so it can in principle reconstruct the MF dot product.
            user_id_llm = None
            item_id_llm = None
            if self.direct_id_tokens:
                user_id_llm = self.user_id_proj(user_cf).unsqueeze(1)               # [B,1,H]
                item_id_llm = self.item_id_proj(target_cf).unsqueeze(1)             # [B,1,H]
                if self.ablate_soft_tokens:
                    user_id_llm = torch.zeros_like(user_id_llm)
                    item_id_llm = torch.zeros_like(item_id_llm)

            # Candidate conditioning (DIN-style): the TARGET item's CF vector
            # shifts the Q tokens when pooling the history below, so the same
            # history yields a different <UserProfile> per candidate. The old
            # per-USER conditioning made profile_q constant across a user's
            # candidates — it cancelled in uAUC and only added cross-user noise.
            target_cond = target_cf if self.user_conditioned else None

            # 2) QFormer outputs (instruction-conditioned). The target encode is
            # NOT conditioned on itself; its candidate signal already enters as
            # the cross-attention source.
            # user_q = self.qformer(user_cf, ins_list)        # [B,Q,d_model]
            target_q = self.qformer(target_cf, ins_list, sem_vec=target_sem)  # [B,Q,d_model]

            # 3) Project to LLM hidden per token
            # user_llm = self.llm_proj(user_q)               # [B,Q,H]
            target_llm = self.llm_proj(target_q)           # [B,Q,H]

            if self.ablate_soft_tokens:
                target_llm = torch.zeros_like(target_llm)

            warm_llm = None
            if self.warm_token and self.warm_proj is not None:
                warm_cf = self.item_llm_emb[batch_data["TargetItemID"]]  # [B,H]
                warm_llm = self.warm_proj(warm_cf).unsqueeze(1)
                if self.ablate_soft_tokens:
                    warm_llm = torch.zeros_like(warm_llm)

            profile_llm = None
            merged_flat = None

            has_interacted = "InteractedItemIDs_pad" in batch_data
            need_merge = (
                has_interacted
                and feature_order is not None
                and "<UserProfile>" in feature_order
                and "<TargetItemID>" in feature_order
            )

            if need_merge:
                # The whole history is POOLED into Q soft tokens, not L*Q: the
                # L item vectors are the cross-attention source sequence, and
                # the Q learned queries read them out. hist_mask keeps padded
                # slots out of that attention.
                ids = batch_data["InteractedItemIDs_pad"]  # [B,L]
                hist_cf = self.rec_encoder.item_encoder(ids)                   # [B,L,d_cf]
                hist_mask = (ids != self.rec_encoder.padding_index)              # [B,L]
                hist_sem = self._sem_for_items(ids)                              # [B,L,d_sem] or None

                # Conditioned on the CANDIDATE (target_cond), not the user:
                # this is what makes <UserProfile> vary per candidate. See the
                # conditioning comment above target_q. candidate_fusion adds
                # the MULTIPLICATIVE variant on the memory side: each history
                # slot carries [e_t; e_j*e_t; e_j-e_t] (zero-init, no-op at
                # warm start), so the Q-Former reads the collaborative match
                # between the candidate and every history item directly.
                profile_q = self.qformer(
                    hist_cf, ins_list, user_cf=target_cond, source_mask=hist_mask,
                    sem_vec=hist_sem,
                    fusion_target=target_cf if self.candidate_fusion else None,
                    fusion_user=user_cf if self.candidate_fusion else None,
                )

                profile_llm = self.llm_proj(profile_q)                          # [B,Q,H]
                
                if self.ablate_soft_tokens:
                    profile_llm = torch.zeros_like(profile_llm)

                # mask expand theo Q
                ones_q = torch.ones((B, Q), device=device, dtype=torch.long)

                ph2emb = {
                    # TEMP_DISABLED_USER_CF: old merge map included "<UserID>": user_llm.
                    # "<UserID>": user_llm,                 # [B,Q,H]
                    "<UserProfile>": profile_llm,  # [B,Q,H] — history pooled into Q tokens
                    "<TargetItemID>": target_llm          # [B,Q,H]
                }
                ph2mask = {
                    # "<UserID>": ones_q,
                    "<UserProfile>": ones_q,
                    "<TargetItemID>": ones_q
                }
                if self.warm_token and warm_llm is not None:
                    ph2emb["<Warm_ID>"] = warm_llm          # [B,1,H]
                    ph2mask["<Warm_ID>"] = torch.ones((B, 1), device=device, dtype=torch.long)
                if self.direct_id_tokens:
                    ones_1 = torch.ones((B, 1), device=device, dtype=torch.long)
                    if user_id_llm is not None:
                        ph2emb["<UserID>"] = user_id_llm    # [B,1,H]
                        ph2mask["<UserID>"] = ones_1
                    if item_id_llm is not None:
                        ph2emb["<ItemID>"] = item_id_llm    # [B,1,H]
                        ph2mask["<ItemID>"] = ones_1

                merged_embeds = torch.cat([ph2emb[ph] for ph in feature_order], dim=1)    # [B, sum_slots, H]
                full_mask = torch.cat([ph2mask[ph] for ph in feature_order], dim=1)       # [B, sum_slots]

                idx = torch.nonzero(full_mask, as_tuple=False)                            # [N,2]
                merged_flat = merged_embeds[idx[:, 0], idx[:, 1]]                         # [N,H]

            rec_embeds = {
                "User_emb": user_llm,                 # None while TEMP_DISABLED_USER_CF is active
                "TargetItem_emb": target_llm,         # [B,Q,H]
                "UserProfile_emb": profile_llm,  # [B,Q,H] or None
                "Warm_emb": warm_llm,                  # [B,1,H] or None
                "UserID_emb": user_id_llm,             # [B,1,H] or None (direct path)
                "ItemID_emb": item_id_llm,             # [B,1,H] or None (direct path)
                "merged_embs": merged_flat,             # [N,H] or None
            }
            self._log_information_flow(user_q, target_q, user_llm, target_llm, merged_flat)

        return rec_embeds, None

    def wrap_prompt_with_soft_tokens_v2(self, rec_embeds, rec_atts, batch_data, prompt_template):
        if not prompt_template:
             return None, None
        
        prompt_ori = prompt_template
        batch_size = batch_data['UserID'].shape[0]
        bos = self.llm_tokenizer.bos_token if self.llm_tokenizer.bos_token else ""

        unk_token = self._soft_token_str
        unk_seq = " ".join([unk_token] * self.proj_token_num)
        
        prompt_template = bos + prompt_template
        if self.direct_id_tokens:
            # SeLLa direct path: <UserID>/<ItemID> are ONE soft token each.
            prompt_template = prompt_template.replace("<UserID>", unk_token)
            prompt_template = prompt_template.replace("<ItemID>", unk_token)
        else:
            # TEMP_DISABLED_USER_CF: old prompt path replaced <UserID> with soft tokens.
            prompt_template = prompt_template.replace("<UserID>", "")
            prompt_template = prompt_template.replace("<ItemID>", "")
        prompt_template = prompt_template.replace("<TargetItemID>", unk_seq)
        prompt_template = prompt_template.replace("<UserProfile>", unk_seq)
        # prompt_template = prompt_template.replace("<DCNFeature>", unk_seq)

        if self.warm_token:
            prompt_template = prompt_template.replace("<Warm_ID>", unk_token)
        else:
            prompt_template = prompt_template.replace("<Warm_ID>", "")

        prompt_list = []
        for k in range(batch_size):
            current_prompt = prompt_template
            
            if 'InteractedItemIDs_pad' in batch_data:
                valid_items = (batch_data['InteractedItemIDs_pad'][k] != self.rec_encoder.padding_index).sum().item()
                item_list_placeholder = " ".join([unk_seq] * valid_items)
                current_prompt = current_prompt.replace('<ItemIDList>', item_list_placeholder)

            if "<ItemTitleList>" in current_prompt and 'InteractedItemTitles' in batch_data:
                 current_prompt = current_prompt.replace("<ItemTitleList>", str(batch_data['InteractedItemTitles'][k]))
            
            if "<TargetItemTitle>" in current_prompt and 'TargetItemTitle' in batch_data:
                 current_prompt = current_prompt.replace("<TargetItemTitle>", str(batch_data['TargetItemTitle'][k]))

            prompt_list.append(current_prompt)
        
        if not self.has_print_prompt:
            preview_parts = []
            if "<UserProfile>" in prompt_ori and 'InteractedItemIDs_pad' in batch_data:
                history_ids = batch_data['InteractedItemIDs_pad'][0].detach().cpu().tolist()
                history_ids = [int(i) for i in history_ids if int(i) != self.rec_encoder.padding_index]
                preview_parts.append(
                    # soft_tokens is Q regardless of history length: the L items
                    # are pooled by the Q queries, NOT expanded to L*Q slots.
                    f"[UserProfile pooled_over={len(history_ids)} history items soft_tokens={self.proj_token_num}]",
                )
            if "<TargetItemID>" in prompt_ori and 'TargetItemID' in batch_data:
                target_id = int(batch_data['TargetItemID'][0].detach().cpu().item())
                preview_parts.append(
                    f"[TargetItemID id={target_id} soft_tokens={self.proj_token_num}]",
                )
            log_step("prompt injection preview:", " | ".join(preview_parts))
            self.has_print_prompt = True

        self.llm_tokenizer.padding_side = "left"
        prompts_tokens = self.llm_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(batch_data['UserID'].device)

        unk_token_id = self._soft_token_id
        
        embed_layer = self.llm_model.get_input_embeddings()
        inputs_embeds = embed_layer(prompts_tokens.input_ids)

        replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)

        has_history_placeholder = "<UserProfile>" in prompt_ori
        has_target_placeholder = "<TargetItemID>" in prompt_ori

        if has_history_placeholder and has_target_placeholder and rec_embeds.get('merged_embs') is not None:
            inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = rec_embeds['merged_embs'].to(inputs_embeds)

        elif has_target_placeholder:
            # TEMP_DISABLED_USER_CF: old target-only branch concatenated user and target tokens.
            # emb_to_inject = torch.cat([rec_embeds['User_emb'], rec_embeds['TargetItem_emb']], dim=1)
            emb_to_inject = rec_embeds['TargetItem_emb']
            emb_to_inject = emb_to_inject.reshape(-1, emb_to_inject.shape[-1])

            # The prompt can contain both history and target placeholder tokens.
            # Inject the target embeddings only into the final Q soft-token slots
            # for each sample, which correspond to the target placeholder.
            target_count = self.proj_token_num
            for b in range(batch_size):
                sample_positions = torch.nonzero(prompts_tokens.input_ids[b] == unk_token_id, as_tuple=False).squeeze(-1)
                if sample_positions.numel() < target_count:
                    continue
                start = sample_positions.numel() - target_count
                target_positions = sample_positions[start:]
                if target_positions.numel() == 0:
                    continue
                src = emb_to_inject[b * target_count:(b + 1) * target_count].to(inputs_embeds.dtype)
                inputs_embeds[b, target_positions] = src

        elif "<DCNFeature>" in prompt_ori:
            raise NotImplementedError("<DCNFeature> is not implemented in this version")

        if not self._has_logged_prompt_injection_stats:
            valid_history_items = 0
            if 'InteractedItemIDs_pad' in batch_data:
                valid_history_items = int(
                    (batch_data['InteractedItemIDs_pad'][0] != self.rec_encoder.padding_index).sum().item()
                )

            target_soft_tokens = self.proj_token_num if "<TargetItemID>" in prompt_ori else 0
            history_soft_tokens = self.proj_token_num if "<UserProfile>" in prompt_ori else 0
            warm_soft_tokens = 1 if (self.warm_token and "<Warm_ID>" in prompt_ori) else 0
            direct_soft_tokens = 0
            if self.direct_id_tokens:
                direct_soft_tokens = int("<UserID>" in prompt_ori) + int("<ItemID>" in prompt_ori)
            total_soft_tokens = (
                target_soft_tokens + history_soft_tokens + warm_soft_tokens + direct_soft_tokens
            )
            sample_unk_slots = int((prompts_tokens.input_ids[0] == unk_token_id).sum().item())

            log_step(
                "Prompt injection stats",
                (
                    f"valid_history_items={valid_history_items}, "
                    f"history_soft_tokens={history_soft_tokens}, "
                    f"target_soft_tokens={target_soft_tokens}, "
                    f"sample_soft_tokens={total_soft_tokens}, "
                    f"sample_unk_slots={sample_unk_slots}, "
                    f"batch_unk_slots={replaced_idx.shape[0]}"
                ),
            )
            self._has_logged_prompt_injection_stats = True

        return inputs_embeds, prompts_tokens.attention_mask

    def assemble_llm_sequences(self, input_embeds, input_atts, label_embeds, label_atts):
        full_embeds = torch.cat([input_embeds, label_embeds], dim=1)
        full_atts = torch.cat([input_atts, label_atts], dim=1)
        return full_embeds, full_atts

    def prepare_llm_targets(self, input_atts, label_tokens):
        batch_size, input_len = input_atts.shape
        device = input_atts.device
        
        empty_targets = torch.full((batch_size, input_len), -100, device=device)
        
        label_targets = label_tokens.input_ids.masked_fill(
            label_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        
        return torch.cat([empty_targets, label_targets], dim=1)

    def execute_llm_forward(self, embeds, atts, targets):
        with self.maybe_autocast():
            return self.llm_model(
                inputs_embeds=embeds,
                attention_mask=atts,
                return_dict=True,
            )

    def _init_align_rank_head(self):
        """Build the auxiliary head for the rank-preserving alignment loss.

        Reads a scalar CTR-like score off the aligned CF soft tokens
        (``cf_emb``, ``[B, Q, H]`` pooled over queries) so a per-user BPR term
        can force the ALIGNED representation to preserve within-user ordering.
        Only built when the loss is enabled; a fresh ``nn.Linear`` is trainable
        by default, and the step-2 policy already leaves non-LoRA modules
        unfrozen, so it co-trains with the Q-Former/projection at Step 2.
        """
        if self.align_rank_loss_weight <= 0.0:
            self.align_rank_head = None
            return
        H = int(self.llm_model.config.hidden_size)
        self.align_rank_head = nn.Linear(H, 1)
        log_step(
            "Rank-preserving alignment loss ACTIVE",
            f"aux per-user BPR on the aligned CF tokens "
            f"(weight={self.align_rank_loss_weight}, tau={self.align_rank_loss_tau}). "
            f"Needs a user-grouped batch sampler, same as ranking_loss.",
        )

    def _per_user_pairwise_loss(self, scores, users, labels, tau=None):
        """Per-user pairwise BPR — a differentiable surrogate for uAUC.

        For every (positive, negative) pair belonging to the SAME user inside
        the batch, push s_pos above s_neg via -log sigmoid((s_pos - s_neg)/tau).
        Returns a scalar; 0 (graph-preserving) when the batch holds no valid
        same-user pos/neg pair, so it never NaNs on unlucky batches.

        The aggregation is USER-WEIGHTED to match uAUC: pair losses are first
        averaged within each user, then averaged across users. A flat mean over
        all pairs would weight users by their pair count (users with many items
        dominate the gradient), which optimises a pair-weighted objective ≈
        global AUC rather than the user-weighted uAUC.

        NOTE: effectiveness depends on batches containing multiple items per
        user (mixed labels). With purely random batching same-user pairs are
        rare — pair this with a user-grouped batch sampler.
        """
        users = users.view(-1)
        labels = labels.view(-1).long()
        pos_mask = labels == 1
        neg_mask = labels == 0

        # diff[i, j] = s_i - s_j ; valid when i is a positive and j a negative
        # of the same user.
        diff = scores.unsqueeze(1) - scores.unsqueeze(0)                  # [B, B]
        same_user = users.unsqueeze(1) == users.unsqueeze(0)             # [B, B]
        valid = same_user & pos_mask.unsqueeze(1) & neg_mask.unsqueeze(0)

        if valid.sum() == 0:
            return scores.sum() * 0.0  # no pairs this batch -> 0, keep the graph

        # -log sigmoid(x) = softplus(-x), numerically stable. Zero out the
        # invalid entries so they contribute nothing to the per-user sums.
        valid_f = valid.to(scores.dtype)
        tau = self.ranking_loss_tau if tau is None else tau
        pair_losses = nn.functional.softplus(-diff / tau) * valid_f

        # Each pair (i, j) belongs to user users[i] (== users[j]). Collapse the
        # neg axis, then scatter-add rows into their user bucket so every user
        # gets its own (sum, count) -> within-user mean.
        row_loss_sum = pair_losses.sum(dim=1)                            # [B]
        row_pair_count = valid_f.sum(dim=1)                             # [B]

        uniq_users, inv = torch.unique(users, return_inverse=True)
        n_users = uniq_users.numel()
        # scatter_add_ requires self, index and src to share device and (for
        # self/src) dtype. `inv` follows `users`, `row_*` follow `scores`; force
        # all three onto the accumulator's device/dtype so a device_map-placed
        # head (fp32, possibly off the batch device) can't break the scatter.
        acc_device, acc_dtype = scores.device, scores.dtype
        inv = inv.to(acc_device)
        user_loss_sum = torch.zeros(n_users, dtype=acc_dtype, device=acc_device)
        user_pair_count = torch.zeros(n_users, dtype=acc_dtype, device=acc_device)
        user_loss_sum.scatter_add_(0, inv, row_loss_sum.to(device=acc_device, dtype=acc_dtype))
        user_pair_count.scatter_add_(0, inv, row_pair_count.to(device=acc_device, dtype=acc_dtype))

        has_pairs = user_pair_count > 0
        per_user_mean = user_loss_sum[has_pairs] / user_pair_count[has_pairs]
        return per_user_mean.mean()

    def calculate_recommendation_loss(self, outputs, label_tokens, batch_data, ans_map, cf_emb=None):
        pos_id = self.llm_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        neg_id = self.llm_tokenizer(ans_map[0], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]

        prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
        binary_logits = torch.stack(
            [prediction_logits[:, neg_id], prediction_logits[:, pos_id]],
            dim=1,
        ).float()
        labels = batch_data['label'].long()

        loss = nn.functional.cross_entropy(binary_logits, labels)

        # uAUC-aligned auxiliary term (opt-in, training only). Disabled by
        # default so this is bit-for-bit the original BCE unless
        # ranking_loss_weight > 0. Skipped at eval so val_loss stays comparable
        # to the BCE baseline (eval batches aren't user-grouped anyway).
        if self.training and self.ranking_loss_weight > 0.0 and 'UserID' in batch_data:
            margin = binary_logits[:, 1] - binary_logits[:, 0]   # score s = logit(Yes) - logit(No)
            bpr = self._per_user_pairwise_loss(margin, batch_data['UserID'], labels)
            loss = loss + self.ranking_loss_weight * bpr

        # Rank-preserving ALIGNMENT term (opt-in, training only). Reads a scalar
        # off the aligned CF soft tokens (cf_emb: [B, Q, H], pooled over queries)
        # and imposes the SAME per-user BPR on THEM — so the collaborative
        # alignment itself, not just the LLM's Yes/No verdict, is pushed to
        # preserve within-user ordering. Targets uAUC at the alignment level.
        # fp32 score keeps the ranking tie-free; needs a user-grouped sampler.
        # Steps 2 and 3: both leave the Q-Former + projection (the alignment)
        # trainable, which is what this term shapes. At Step 1 they are frozen,
        # so the gradient would have nowhere to go except the aux head.
        if (self.training and self.tuning_step in (2, 3)
                and self.align_rank_loss_weight > 0.0
                and self.align_rank_head is not None and cf_emb is not None
                and 'UserID' in batch_data):
            pooled = cf_emb.mean(dim=1).float()                       # [B, Q, H] -> [B, H]
            align_score = self.align_rank_head(pooled).squeeze(-1)    # [B]
            align_bpr = self._per_user_pairwise_loss(
                align_score, batch_data['UserID'], labels,
                tau=self.align_rank_loss_tau,
            )
            loss = loss + self.align_rank_loss_weight * align_bpr

        return loss

    def recommendation_scores(self, outputs, label_tokens, ans_map):
        pos_id = self.llm_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        neg_id = self.llm_tokenizer(ans_map[0], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]

        prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
        binary_logits = torch.stack(
            [prediction_logits[:, neg_id], prediction_logits[:, pos_id]],
            dim=1,
        )
        return torch.softmax(binary_logits, dim=1)[:, 1]

    def build_llm_outputs_from_labels(self, batch_data):
        device = batch_data['UserID'].device
        ans_map = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        text_labels = [ans_map[int(label)] for label in batch_data["label"]]

        self.llm_tokenizer.padding_side = "right"
        label_tokens = self.llm_tokenizer(
            text_labels,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        embed_layer = self.llm_model.get_input_embeddings()
        label_embeds = embed_layer(label_tokens.input_ids)

        return label_embeds, label_tokens, ans_map

    def _build_qformer_instructions(self, batch_size: int) -> list:
        """Build short item-text instructions for the Q-Former.

        Matches the distribution the Q-Former was trained on in stage 1: a
        fresh sample per row during training, a deterministic fixed string
        during eval/inference so the same input maps to the same embedding.
        """
        if self.training:
            return random.choices(self.QFORMER_ITEM_INSTRUCTIONS, k=batch_size)
        return [self.QFORMER_ITEM_INSTRUCTIONS[0]] * batch_size

    def build_llm_inputs_from_prompt_v2(self, prompt_template, batch_data):
        feature_order = self.get_placeholder_order(prompt_template) if prompt_template else None
        batch_size = batch_data["UserID"].shape[0]

        if not feature_order:
            self._log_trainable_module_stats()
            rec_embeds = {
                "User_emb": None,
                "TargetItem_emb": None,
                "InteractedItems_embs": None,
                "merged_embs": None,
            }
            rec_atts = None
        else:
            instruction_list = self._build_qformer_instructions(batch_size)
            rec_embeds, rec_atts = self.encode_rec_features_to_llm_v2(
                batch_data,
                feature_order=feature_order,
                instruction_list=instruction_list,
            )

        llm_embeds, llm_atts = self.wrap_prompt_with_soft_tokens_v2(rec_embeds, rec_atts, batch_data, prompt_template)
        # Expose the aligned target-item CF soft tokens ([B, Q, H] or None) so the
        # rank-preserving alignment loss can read a per-user score off them.
        cf_emb = rec_embeds.get("TargetItem_emb")
        return llm_embeds, llm_atts, cf_emb

    def generate_for_samples(self, samples, return_all=False):
        prompt = self.prompt_list[0]
        # cf_emb unused at eval: the align loss is gated on self.training, so
        # eval loss stays pure BCE and comparable to the baseline.
        input_embeds, input_atts, _ = self.build_llm_inputs_from_prompt_v2(prompt, samples)
        label_embeds, label_tokens, ans_map = self.build_llm_outputs_from_labels(samples)

        full_embeds, full_atts = self.assemble_llm_sequences(
            input_embeds, input_atts, label_embeds, label_tokens.attention_mask
        )

        targets = self.prepare_llm_targets(input_atts, label_tokens)

        outputs = self.execute_llm_forward(full_embeds, full_atts, targets)
        loss = self.calculate_recommendation_loss(outputs, label_tokens, samples, ans_map)

        logits = self.recommendation_scores(outputs, label_tokens, ans_map)

        self._maybe_log_predictions(samples, logits, ans_map)

        if return_all:
            return outputs, logits

        return {"loss": loss, "logits": logits}

    def _maybe_log_predictions(self, samples, prob_yes, ans_map, samples_per_batch=3):
        """Emit a compact per-sample log of (UserID, TargetItemID, label,
        predicted Yes/No, probability). Bounded by
        ``_max_eval_pred_log_batches`` to avoid log spam; the counter resets
        at the start of every eval pass (see ``eval``)."""

        if self._eval_pred_log_count >= self._max_eval_pred_log_batches:
            return

        user_ids = samples["UserID"].detach().cpu().tolist()
        item_ids = samples["TargetItemID"].detach().cpu().tolist()
        labels = samples["label"].detach().cpu().tolist()
        probs = prob_yes.detach().float().cpu().tolist()

        n = min(len(user_ids), samples_per_batch)
        rows = []
        for i in range(n):
            pred = ans_map[1] if probs[i] >= 0.5 else ans_map[0]
            gt = ans_map[int(labels[i])]
            outcome = "OK" if pred == gt else "WRONG"
            rows.append(
                f"user={user_ids[i]} item={item_ids[i]} "
                f"gt={gt} prob_yes={probs[i]:.3f} pred={pred} {outcome}"
            )

        log_step(
            f"[stage3 eval batch #{self._eval_pred_log_count}]",
            " | ".join(rows),
        )
        self._eval_pred_log_count += 1

    def eval(self):
        """Reset the prediction-log counter so each eval pass starts logging
        from sample 0 again. ``train(True)`` does not reset, so logs stay
        scoped to eval calls."""
        self._eval_pred_log_count = 0
        return super().eval()

    def forward_v2(self, batch_data):
        prompt = self._sample_prompt()
        input_embeds, input_atts, cf_emb = self.build_llm_inputs_from_prompt_v2(prompt, batch_data)
        label_embeds, label_tokens, ans_map = self.build_llm_outputs_from_labels(batch_data)

        full_embeds, full_atts = self.assemble_llm_sequences(
            input_embeds, input_atts, label_embeds, label_tokens.attention_mask
        )

        targets = self.prepare_llm_targets(input_atts, label_tokens)

        outputs = self.execute_llm_forward(full_embeds, full_atts, targets)
        loss = self.calculate_recommendation_loss(outputs, label_tokens, batch_data, ans_map, cf_emb=cf_emb)

        return {"loss": loss}

    def forward(self, samples):
        if self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("Only forward_v2 is implemented in this version")

    @classmethod
    def from_config(cls, cfg):
        rec_model = cfg.get('rec_model',"MF")
        freeze_rec = cfg.get("freeze_rec",True)
        rec_config = cfg.get("rec_config")
        qformer_config = cfg.get("qformer_config") or {}
        llm_model = cfg.get("llm_model")
        proj_token_num = cfg.get("proj_token_num")
        freeze_proj = cfg.get("freeze_proj")
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 1024)
        end_sym = cfg.get("end_sym", '\n')
        num_queries = qformer_config.get("num_queries", 8)
        num_heads = qformer_config.get("num_heads", 8)
        num_layers = qformer_config.get("num_layers", 2)
        qformer_d_model = qformer_config.get("qformer_d_model", 768)
        qformer_output_dim = qformer_config.get("qformer_output_dim")
        pretrained_qformer = qformer_config.get("qformer_ckpt")
        qformer_text_model_name = qformer_config.get("qformer_text_model_name", "bert-base-uncased")
        max_instruction_length = qformer_config.get("max_instruction_length", 48)
        pretrained_llm_proj = qformer_config.get("llm_proj_ckpt")
        user_conditioned = bool(qformer_config.get("user_conditioned", False))
        warm_token = bool(qformer_config.get("warm_token", False))
        direct_id_tokens = bool(cfg.get("direct_id_tokens", False))
        candidate_fusion = bool(qformer_config.get("candidate_fusion", False))
        pretrained_item_llm_emb = qformer_config.get("item_llm_emb_path", None)
        sem_source = bool(qformer_config.get("sem_source", False))
        item_sem_emb_path = qformer_config.get("item_sem_emb_path", None)
        sem_source_dropout = float(qformer_config.get("sem_source_dropout", 0.5))
        ablate_soft_tokens = cfg.get("ablate_soft_tokens", False)

        lora_cfg = cfg.get("lora_config") or {}
        use_lora = bool(lora_cfg.get("use_lora", False))
        lora_r = int(lora_cfg.get("r", 8))
        lora_alpha = int(lora_cfg.get("alpha", 16))
        lora_target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        lora_dropout = float(lora_cfg.get("dropout", 0.05))
        tuning_step = cfg.get("tuning_step", None)

        ranking_cfg = cfg.get("ranking_loss") or {}
        ranking_loss_weight = float(ranking_cfg.get("weight", 0.0))
        ranking_loss_tau = float(ranking_cfg.get("tau", 1.0))
        align_rank_cfg = cfg.get("align_rank_loss") or {}
        align_rank_loss_weight = float(align_rank_cfg.get("weight", 0.0))
        align_rank_loss_tau = float(align_rank_cfg.get("tau", 1.0))

        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec=rec_config['pretrained_path'],
            pretrained_qformer=pretrained_qformer,
            pretrained_llm_proj=pretrained_llm_proj,
            freeze_rec=freeze_rec,
            llm_model=llm_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            proj_token_num=proj_token_num,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            qformer_d_model=qformer_d_model,
            qformer_output_dim=qformer_output_dim,
            qformer_text_model_name=qformer_text_model_name,
            max_instruction_length=max_instruction_length,
            freeze_proj=freeze_proj,
            ablate_soft_tokens=ablate_soft_tokens,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            tuning_step=tuning_step,
            user_conditioned=user_conditioned,
            warm_token=warm_token,
            direct_id_tokens=direct_id_tokens,
            candidate_fusion=candidate_fusion,
            pretrained_item_llm_emb=pretrained_item_llm_emb,
            ranking_loss_weight=ranking_loss_weight,
            ranking_loss_tau=ranking_loss_tau,
            align_rank_loss_weight=align_rank_loss_weight,
            align_rank_loss_tau=align_rank_loss_tau,
            sem_source=sem_source,
            item_sem_emb_path=item_sem_emb_path,
            sem_source_dropout=sem_source_dropout,
            mf_drift_log_steps=int(cfg.get("mf_drift_log_steps", 200)),
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            log_step("Load QRecLLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            log_step("loading message, msg.... {}".format(msg))

            # Restore the pretrained MF ONLY when the checkpoint does not carry
            # its own copy. The runner strips non-trainable tensors when saving,
            # so a Step-1/2 checkpoint has no ``rec_encoder.*`` keys and the
            # reload is what puts MF back after the strict=False load; but a
            # JOINT (Step-3) checkpoint does carry them, and reloading
            # mf_model.pth over those would silently throw away every MF update
            # the joint run produced — including at eval time, where the model
            # scored would no longer be the model trained.
            #
            # Keyed on the checkpoint contents, not on ``freeze_rec``: the two
            # only coincided by accident (freeze_rec=True <=> MF absent from the
            # checkpoint), and eval flows commonly leave freeze_rec at its
            # config default while loading a joint checkpoint — exactly the case
            # the old condition got wrong.
            ckpt_has_rec = any(
                isinstance(k, str) and k.startswith("rec_encoder.") for k in ckpt["model"]
            )
            if ckpt_has_rec:
                log_step(
                    "Kept MF weights from the checkpoint",
                    "checkpoint carries rec_encoder.* (joint tuning); NOT reloading "
                    f"{rec_config['pretrained_path']}",
                )
            elif os.path.exists(rec_config["pretrained_path"]):
                model.rec_encoder.load_state_dict(
                    torch.load(rec_config["pretrained_path"], map_location="cpu")
                )
                log_step(
                    "Restored pretrained MF",
                    f"checkpoint carried no rec_encoder.* keys; loaded "
                    f"{rec_config['pretrained_path']}",
                )

        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model
