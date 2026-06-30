
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.registry import registry
from sigllm.models.multimodal.base.rec_base_model import Rec2Base
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.projection.collaborative_lora_injector import CollaborativeLoRAInjector

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
    
    # Multi-token CF: one joint Q-Former forward over user + target + history
    # produces a single block of Q soft tokens, injected at <CFTokens>.
    PLACEHOLDERS_FOR_EMBED = ["<CFTokens>"]

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
        cf_injection_mode="soft_token",
        cora_alpha=16.0,
        cora_target_modules=("q_proj", "v_proj"),
    ):
        super().__init__()

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
                "ablate_soft_tokens=True → target_llm and interacted_llm_flat "
                "will be zeroed before injection (Information flow log will show "
                "target_llm mean/std=0).",
            )

        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_dropout = float(lora_dropout)
        self.tuning_step = tuning_step

        # CHANGE A: collaborative injection mode.
        self.cf_injection_mode = str(cf_injection_mode or "soft_token").lower()
        if self.cf_injection_mode not in ("soft_token", "lora_weight", "both"):
            raise ValueError(
                f"cf_injection_mode must be one of soft_token|lora_weight|both; "
                f"got '{self.cf_injection_mode}'"
            )
        self.cora_alpha = float(cora_alpha)
        self.cora_target_modules = tuple(cora_target_modules)
        self.cf_injector = None

        log_step("Running MiniGPT4Rec_v2 initialization")

        self.rec_model_type = rec_model

        # Initialize components
        self._init_rec_model(rec_model, rec_config, pretrained_rec, freeze_rec)
        self._init_llm_model(llm_model)
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
        )
        self._init_projection(proj_token_num, freeze_proj, pretrained_llm_proj)
        self._init_prompts(prompt_path, prompt_template, max_txt_len, end_sym)
        self._init_cf_injection()
        self._apply_tuning_step_policy()

    def _init_rec_model(self, rec_model, rec_config, pretrained_rec, freeze_rec):
        log_step("Loading Rec_model")
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config)
        
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            log_step("Successfully loaded the pretrained model")
        
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

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True,
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
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

        # No unk token (e.g. Qwen2). Register a DEDICATED reserved placeholder
        # token that by construction cannot appear in the prompt body or the
        # ChatML scaffolding. This is the standard BLIP-2/LLaVA approach and is
        # collision-proof. The previous strategy scanned for an unused special
        # token and, finding none on Qwen2-Instruct, fell back to eos_token
        # (<|im_end|>) — which the chat template uses, so soft-slot matching
        # also hit the scaffolding markers and the embedding-injection counts
        # diverged (256 vs 288).
        placeholder = "<|cf_slot|>"
        try:
            num_added = tok.add_tokens([placeholder], special_tokens=True)
            new_id = tok.convert_tokens_to_ids(placeholder)
            if new_id is not None and new_id != tok.unk_token_id:
                emb = self.llm_model.get_input_embeddings()
                emb_rows = emb.weight.size(0) if emb is not None else 0
                if num_added > 0 and len(tok) > emb_rows:
                    # Grow embeddings only when the new id exceeds existing rows
                    # (Qwen2 already has spare rows, so this usually no-ops), then
                    # re-freeze the (frozen base) embeddings.
                    self.llm_model.resize_token_embeddings(len(tok))
                    in_emb = self.llm_model.get_input_embeddings()
                    if in_emb is not None:
                        for p in in_emb.parameters():
                            p.requires_grad = False
                    out_emb = self.llm_model.get_output_embeddings()
                    if out_emb is not None:
                        for p in out_emb.parameters():
                            p.requires_grad = False
                self._soft_token_str = placeholder
                self._soft_token_id = new_id
                return
        except Exception as exc:  # pragma: no cover - defensive
            log_step("Soft-token placeholder add failed", str(exc))

        # Last-resort fallback (should not be reached): eos.
        log_step(
            "Soft-token fallback",
            "could not register a dedicated placeholder; using eos_token. Soft "
            "slots may COLLIDE with chat/padding tokens and corrupt embeddings.",
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

    def _cf_weight_enabled(self) -> bool:
        """True when the CoRA-style weight delta is part of the CF pathway."""
        return self.cf_injection_mode in ("lora_weight", "both")

    def _init_cf_injection(self):
        """CHANGE A: build and attach the collaborative weight injector.

        No-op for the default ``soft_token`` mode. For ``lora_weight`` / ``both``
        the Q-Former queries are turned into a per-sample low-rank delta on the
        LLM's attention projections. The injector is attached AFTER LoRA so it
        wraps the (possibly PEFT-wrapped) target modules.
        """
        if not self._cf_weight_enabled():
            log_step("CF injection", f"mode={self.cf_injection_mode} (no weight injector)")
            return

        d_model = int(self.qformer.d_model)
        self.cf_injector = CollaborativeLoRAInjector(
            d_model=d_model,
            target_modules=self.cora_target_modules,
            alpha=self.cora_alpha,
            num_queries=int(self.qformer.q.shape[-2]),
        )
        n = self.cf_injector.attach(self.llm_model)
        self.cf_injector = self.cf_injector.to(self.device)
        log_step(
            "CF injection",
            f"mode={self.cf_injection_mode}, hooked={n} modules, "
            f"alpha={self.cora_alpha}, targets={list(self.cora_target_modules)}, "
            f"trainable_params={count_trainable_parameters(self.cf_injector)}",
        )

    def _apply_tuning_step_policy(self):
        step = self.tuning_step
        if step is None:
            return

        if int(step) == 1:
            for p in self.qformer.parameters():
                p.requires_grad = False
            for p in self.llm_proj.parameters():
                p.requires_grad = False
            self.qformer.eval()
            self.qformer.train = disabled_train
            self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            if self.cf_injector is not None:
                for p in self.cf_injector.parameters():
                    p.requires_grad = False
                self.cf_injector.eval()
                self.cf_injector.train = disabled_train
            log_step(
                "Tuning step 1",
                "LoRA trainable; Q-Former, projection, CF-injector, MF and base LLM all frozen.",
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
            if self.cf_injector is not None:
                for p in self.cf_injector.parameters():
                    p.requires_grad = True
                self.cf_injector.train()
            log_step(
                "Tuning step 2",
                "Q-Former + projection + CF-injector trainable; LoRA, base LLM and MF frozen.",
            )

        else:
            log_step("Tuning step", f"unrecognized value '{step}', no policy applied")

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
        ).to(self.device)

        if pretrained_qformer and pretrained_qformer != "not_have":
            ckpt = torch.load(pretrained_qformer, map_location="cpu")
            # Normalize: unwrap a {"epoch","model",...} checkpoint, and if the
            # adapter is nested under a "qformer." prefix (Stage-1 alignment
            # state), extract just that submodule. A bare adapter state_dict
            # (Stage 2 output) is used as-is. The previous global
            # replace("qformer.", "", 1) corrupted bare adapters, whose inner
            # InstructBLIP legitimately uses "qformer.*" keys.
            state_dict = ckpt
            if isinstance(state_dict, dict) and "model" in state_dict and "epoch" in state_dict:
                state_dict = state_dict["model"]
            adapter_keys = set(self.qformer.state_dict().keys())
            if not (adapter_keys & set(state_dict.keys())):
                extracted = {
                    k[len("qformer."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("qformer.")
                }
                if extracted:
                    state_dict = extracted
            self.qformer.load_state_dict(state_dict, strict=True)
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

    def _log_information_flow(self, user_q, cf_q, user_llm, cf_llm, merged_flat):
        if self._flow_log_steps >= self._max_flow_log_steps:
            return

        log_step(
            "Information flow",
            " | ".join(
                [
                    tensor_stat_string("cf_q", cf_q),
                    tensor_stat_string("cf_llm", cf_llm),
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
                if raw_prompt.strip() and not raw_prompt.lstrip().startswith("# DISABLED_USER_CF")
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
        id_terms = ["<CFTokens>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True

        if self.llm_model is not None and hasattr(self.llm_model, "peft_config"):
            for n, p in self.llm_model.named_parameters():
                if "lora_" in n and p.requires_grad:
                    return True

        # CHANGE A: the CoRA weight-injection path can be the only trainable CF
        # route when using a text-only prompt with cf_injection_mode=lora_weight.
        if self.cf_injector is not None:
            for p in self.cf_injector.parameters():
                if p.requires_grad:
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
    
    def get_placeholder_order(self, prompt: str, placeholders=PLACEHOLDERS_FOR_EMBED):
        positions = []
        for ph in placeholders:
            pos = prompt.find(ph)
            if pos >= 0:
                positions.append((pos, ph))
        positions.sort(key=lambda x: x[0])
        return [ph for _, ph in positions]

    def encode_rec_features_to_llm_v2(self, batch_data, feature_order=None, instruction_list=None):
        """Run the Q-Former once over the joint user+target+history CF sequence
        and return ``Q`` LLM-space soft tokens.

        Args:
            batch_data (dict): ``UserID`` [B], ``TargetItemID`` [B],
                ``InteractedItemIDs_pad`` [B, H_max].
            feature_order (list[str]): subset of ``PLACEHOLDERS_FOR_EMBED``
                actually present in the prompt template.
        """
        if self.rec_encoder is None:
            return None, None

        self._log_trainable_module_stats()

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
            user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])           # [B, d_cf]
            target_cf = self.rec_encoder.item_encoder(batch_data["TargetItemID"])   # [B, d_cf]
            history_ids = batch_data["InteractedItemIDs_pad"]                       # [B, H_max]
            history_cf = self.rec_encoder.item_encoder(history_ids)                 # [B, H_max, d_cf]
            history_mask = (history_ids != self.rec_encoder.padding_index).long()   # [B, H_max]

            cf_q = self.qformer(user_cf, target_cf, history_cf, history_mask, ins_list)  # [B, Q, d_model]
            cf_llm = self.llm_proj(cf_q)                                                 # [B, Q, H]

            # CHANGE A: feed the collaborative queries to the weight injector.
            # These are the pre-projection Q-Former outputs (d_model), which the
            # injector maps to per-sample low-rank deltas during the LLM forward.
            if self._cf_weight_enabled() and self.cf_injector is not None:
                self.cf_injector.set_queries(cf_q)

            if self.ablate_soft_tokens:
                cf_llm = torch.zeros_like(cf_llm)

            merged_flat = None
            if feature_order and "<CFTokens>" in feature_order:
                merged_flat = cf_llm.reshape(B * Q, H)

            rec_embeds = {
                "CF_emb": cf_llm,            # [B, Q, H]
                "merged_embs": merged_flat,  # [B*Q, H] or None
            }
            self._log_information_flow(None, cf_q, None, cf_llm, merged_flat)

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
        prompt_template = prompt_template.replace("<CFTokens>", unk_seq)

        prompt_list = []
        for k in range(batch_size):
            current_prompt = prompt_template

            if "<ItemTitleList>" in current_prompt and 'InteractedItemTitles' in batch_data:
                current_prompt = current_prompt.replace("<ItemTitleList>", str(batch_data['InteractedItemTitles'][k]))

            if "<TargetItemTitle>" in current_prompt and 'TargetItemTitle' in batch_data:
                current_prompt = current_prompt.replace("<TargetItemTitle>", str(batch_data['TargetItemTitle'][k]))

            prompt_list.append(current_prompt)

        if not self.has_print_prompt:
            preview_parts = []
            if "<CFTokens>" in prompt_ori and 'InteractedItemIDs_pad' in batch_data:
                history_ids = batch_data['InteractedItemIDs_pad'][0].detach().cpu().tolist()
                history_ids = [int(i) for i in history_ids if int(i) != self.rec_encoder.padding_index]
                target_id = int(batch_data['TargetItemID'][0].detach().cpu().item())
                user_id = int(batch_data['UserID'][0].detach().cpu().item())
                preview_parts.append(
                    f"[CFTokens user={user_id} target={target_id} history={history_ids} "
                    f"soft_tokens={self.proj_token_num}]",
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

        if "<CFTokens>" in prompt_ori and rec_embeds.get('merged_embs') is not None:
            merged = rec_embeds['merged_embs']
            n_slots = replaced_idx.shape[0]
            if n_slots != merged.shape[0]:
                raise RuntimeError(
                    f"Soft-token slot/embedding mismatch: found {n_slots} "
                    f"'{self._soft_token_str}' slots in the tokenized prompt but have "
                    f"{merged.shape[0]} soft embeddings ({batch_size} x {self.proj_token_num}). "
                    f"The soft-token placeholder (id={self._soft_token_id}) likely collides "
                    f"with a token used elsewhere in the prompt/chat template."
                )
            inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = merged.to(inputs_embeds)

        if not self._has_logged_prompt_injection_stats:
            cf_soft_tokens = self.proj_token_num if "<CFTokens>" in prompt_ori else 0
            sample_unk_slots = int((prompts_tokens.input_ids[0] == unk_token_id).sum().item())

            log_step(
                "Prompt injection stats",
                (
                    f"cf_soft_tokens={cf_soft_tokens}, "
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
            # CHANGE A: enable the CoRA-style weight delta only for this forward,
            # then clear the stored queries so they cannot leak into a later
            # call that has not set them.
            if self._cf_weight_enabled() and self.cf_injector is not None:
                with self.cf_injector.enabled():
                    out = self.llm_model(
                        inputs_embeds=embeds,
                        attention_mask=atts,
                        return_dict=True,
                    )
                self.cf_injector.clear()
                return out
            return self.llm_model(
                inputs_embeds=embeds,
                attention_mask=atts,
                return_dict=True,
            )

    def calculate_recommendation_loss(self, outputs, label_tokens, batch_data, ans_map):
        pos_id = self.llm_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        neg_id = self.llm_tokenizer(ans_map[0], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]
        
        prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
        binary_logits = torch.stack(
            [prediction_logits[:, neg_id], prediction_logits[:, pos_id]],
            dim=1,
        )
        labels = batch_data['label'].long()
        
        loss = nn.functional.cross_entropy(binary_logits, labels)
        
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

        if not feature_order and not self._cf_weight_enabled():
            self._log_trainable_module_stats()
            rec_embeds = {
                "CF_emb": None,
                "merged_embs": None,
            }
            rec_atts = None
        else:
            instruction_list = self._build_qformer_instructions(batch_size)
            rec_embeds, rec_atts = self.encode_rec_features_to_llm_v2(
                batch_data,
                feature_order=feature_order or [],
                instruction_list=instruction_list,
            )

        llm_embeds, llm_atts = self.wrap_prompt_with_soft_tokens_v2(rec_embeds, rec_atts, batch_data, prompt_template)
        return llm_embeds, llm_atts

    def generate_for_samples(self, samples, return_all=False):
        prompt = self.prompt_list[0]
        input_embeds, input_atts = self.build_llm_inputs_from_prompt_v2(prompt, samples)
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
        input_embeds, input_atts = self.build_llm_inputs_from_prompt_v2(prompt, batch_data)
        label_embeds, label_tokens, ans_map = self.build_llm_outputs_from_labels(batch_data)

        full_embeds, full_atts = self.assemble_llm_sequences(
            input_embeds, input_atts, label_embeds, label_tokens.attention_mask
        )
        
        targets = self.prepare_llm_targets(input_atts, label_tokens)
        
        outputs = self.execute_llm_forward(full_embeds, full_atts, targets)
        loss = self.calculate_recommendation_loss(outputs, label_tokens, batch_data, ans_map)

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
        ablate_soft_tokens = cfg.get("ablate_soft_tokens", False)

        lora_cfg = cfg.get("lora_config") or {}
        use_lora = bool(lora_cfg.get("use_lora", False))
        lora_r = int(lora_cfg.get("r", 8))
        lora_alpha = int(lora_cfg.get("alpha", 16))
        lora_target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        lora_dropout = float(lora_cfg.get("dropout", 0.05))
        tuning_step = cfg.get("tuning_step", None)

        cf_injection_mode = cfg.get("cf_injection_mode", "soft_token")
        cora_alpha = float(cfg.get("cora_alpha", 16.0))
        cora_target_modules = cfg.get("cora_target_modules", lora_target_modules)

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
            cf_injection_mode=cf_injection_mode,
            cora_alpha=cora_alpha,
            cora_target_modules=cora_target_modules,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            log_step("Load QRecLLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            log_step("loading message, msg.... {}".format(msg))
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))

        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model
