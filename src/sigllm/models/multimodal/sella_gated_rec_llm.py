"""SeLLa-parity recommender with a GATED history Q-Former folded into ``<UserID>``.

Why this exists as its own model rather than a flag on ``QRecLLM``
------------------------------------------------------------------
``QRecLLM`` puts the Q-Former in the prompt as ``2 * num_queries`` NEW soft-token
positions (``<UserProfile>`` + ``<TargetItemID>``), trains against a 2-way
cross-entropy on the Yes/No logits at one position, and depends on Stage 1
(InfoNCE representation pretraining) and Stage 2 (generative pretraining) to make
those positions mean anything. This model takes the opposite stance on all three:

1. **Three soft positions, exactly SeLLa's.** ``<UserID>``, ``<ItemID>``,
   ``<Warm_ID>``, in the prompt positions they already occupy. Nothing is added,
   nothing is moved, nothing is rewritten. The Q-Former is an ADDITIVE, GATED
   contribution to the ``<UserID>`` embedding::

       e_user  <-  id_proj(e_u) + g * qformer(user, target, history)

   ``g`` is a single scalar, zero-initialised, so **at g = 0 this model is
   numerically identical to SeLLa** — the term is a literal ``+ 0``, not a small
   perturbation. ``g`` after training is therefore a direct readout of what the
   module contributed, and it is reported in the logs (together with
   ``delta_rel = ||g*delta|| / ||e_user||``, which is the scale-free version).

   The alternative — a new prompt position fed by a zero-init projection — puts a
   zero vector into the LLM's input sequence, which is out of distribution for
   every position embedding around it. Folding into an existing token avoids that
   by construction, and it keeps the semantics honest: the Q-Former is refining
   the USER representation using history conditioned on the candidate, which is
   what a user token is for.

2. **SeLLa's eval; SeLLa's loss available but NOT the default.** Scoring is
   SeLLa's: the Yes/No softmax at the answer position, with AUC/uAUC computed
   from it by ``RecBaseTask``. The auxiliary ``ranking_loss`` /
   ``align_rank_loss`` terms from the config are deliberately NOT read here
   (``__init__`` logs that they are ignored).

   The training objective started as SeLLa's full-sequence LM cross-entropy and
   was changed to ``lm_loss_scope='answer'`` on evidence. Measured:

       epoch 0   val_loss 1.589809   AUC 0.752483   uAUC 0.697958
       epoch 1   val_loss 1.583828   AUC 0.635533   uAUC 0.618170

   The loss went DOWN while AUC/uAUC collapsed. ``full`` supervises ~86 positions
   of which exactly ONE is the answer; the other ~85 ask the model to predict the
   next PROMPT token, and this prompt is near-identical across samples, so the
   cheapest way to make it more predictable is to drive the soft tokens toward a
   CONSTANT — i.e. to erase the per-user / per-item information the channel
   exists to carry. ``scope='full'`` is still selectable for the parity claim,
   and note that epoch 0 under it BEAT the baseline (0.752/0.698): the channel
   works, the objective then trains it away.

3. **No Stage 1, no Stage 2.** The Q-Former trains jointly in this step from a
   fresh init. With ~4.3M parameters behind a zero-init gate there is nothing for
   a representation-pretraining stage to protect: the gate already guarantees the
   run cannot start worse than SeLLa, and the two dropped stages were each
   carrying a documented failure (Stage 1's InfoNCE gain sitting near chance,
   Stage 2's channel being drowned out by the ``<ItemTitleList>`` text path).

Everything else — dataset, builder, runner, task, optimizer groups, checkpoint
format, ``mf_drift`` — is the existing SigLLM machinery, unchanged.
"""

import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.registry import registry
from sigllm.common.utils import resolve_hf_model_path
from sigllm.models.multimodal.base.rec_base_model import Rec2Base
from sigllm.models.q_former.gated_history_qformer import GatedHistoryQFormer

LOGGER = NotebookLogger.rich_logger("sigllm.sella_gated_rec_llm")

# Scopes for the training loss.
#   full         SeLLa parity — every non-padding position is supervised,
#                INCLUDING the three soft-token positions (SeLLa does exactly
#                this: its target_ids are input_ids with only pad masked, and
#                its <User_ID>/<Item_ID>/<Warm_ID> ids stay in the targets).
#   full_no_soft full-sequence CE with the soft-token positions masked out. The
#                clean variant: predicting the placeholder token id is not a
#                meaningful objective, and here all three slots share ONE
#                reserved id, so `full` spends gradient on "emit the placeholder".
#   answer       (DEFAULT) only the Yes/No answer token(s). The objective the
#                0.729/0.690 baseline used, and the only scope that removes the
#                erase-the-soft-tokens pressure documented in the class docstring.
LM_LOSS_SCOPES = ("full", "full_no_soft", "answer")


def log_step(title: str, detail: Optional[str] = None) -> None:
    LOGGER.info(title if detail is None else f"{title} | {detail}")


def disabled_train(self, mode=True):
    """Replacement for ``module.train`` that pins a frozen module to eval."""
    return self


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@registry.register_model("sella_gated_rec_llm")
class SeLLaGatedRecLLM(Rec2Base):
    """SeLLa's three soft tokens + a gated history Q-Former on ``<UserID>``."""

    PRETRAINED_MODEL_CONFIG_DICT = {}

    # Order here is irrelevant — injection order follows the PROMPT (see
    # ``_placeholder_order``), which is what keeps the three tokens where they are.
    PLACEHOLDERS_FOR_EMBED = ["<UserID>", "<ItemID>", "<Warm_ID>"]

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        freeze_rec=False,
        llm_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=1024,
        end_sym="\n",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=("q_proj", "v_proj"),
        lora_dropout=0.05,
        lora_trainable=False,
        warm_token=True,
        pretrained_item_llm_emb=None,
        item_sem_emb_path=None,
        use_qformer=True,
        qformer_d_model=256,
        qformer_num_queries=2,
        qformer_num_heads=4,
        qformer_num_layers=2,
        qformer_dropout=0.0,
        qformer_use_sem=True,
        qformer_use_user_slot=True,
        qformer_hist_target_fusion=False,
        qformer_delta_scale="match_ref",
        qformer_per_sample_magnitude=True,
        qformer_gate_init=0.0,
        ablate_qformer=False,
        id_proj_norm=False,
        id_proj_warm_start=True,
        id_proj_init="default",
        lm_loss_scope="full",
        gate_log_steps=50,
        mf_drift_log_steps=200,
    ):
        super().__init__()

        if lm_loss_scope not in LM_LOSS_SCOPES:
            raise ValueError(
                f"lm_loss_scope must be one of {LM_LOSS_SCOPES}, got {lm_loss_scope!r}"
            )
        self.lm_loss_scope = lm_loss_scope
        self.warm_token = bool(warm_token)
        self.use_qformer = bool(use_qformer)
        self.ablate_qformer = bool(ablate_qformer)
        self.lora_trainable = bool(lora_trainable)
        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_dropout = float(lora_dropout)
        self.id_proj_norm = bool(id_proj_norm)
        self.id_proj_warm_start = bool(id_proj_warm_start)
        if id_proj_init not in ("default", "small"):
            raise ValueError(f"id_proj_init must be 'default' or 'small', got {id_proj_init!r}")
        self.id_proj_init = id_proj_init
        self.gate_log_steps = int(gate_log_steps)
        self.mf_drift_log_steps = int(mf_drift_log_steps)
        self._item_sem_emb_path = item_sem_emb_path

        self._step_count = 0
        self._has_logged_trainable_stats = False
        self._has_logged_injection_stats = False
        self._has_logged_loss_scope = False
        self.run_mode_ = None

        log_step(
            "SeLLa-gated model init",
            "3 soft tokens (<UserID>, <ItemID>, <Warm_ID>); Q-Former enters as a "
            "zero-init GATED addition to <UserID>. Auxiliary ranking_loss / "
            "align_rank_loss from the config are IGNORED by this model — the "
            f"objective is SeLLa's LM cross-entropy (scope={self.lm_loss_scope}).",
        )

        self._init_rec_model(rec_model, rec_config, pretrained_rec, freeze_rec)
        self._init_llm_model(llm_model)
        self._init_warm_token(pretrained_item_llm_emb)
        self._init_id_proj(rec_config.embedding_size, pretrained_rec)
        self._init_history_qformer(
            d_cf=rec_config.embedding_size,
            d_model=qformer_d_model,
            num_queries=qformer_num_queries,
            num_heads=qformer_num_heads,
            num_layers=qformer_num_layers,
            dropout=qformer_dropout,
            use_sem=qformer_use_sem,
            use_user_slot=qformer_use_user_slot,
            hist_target_fusion=qformer_hist_target_fusion,
            delta_scale=qformer_delta_scale,
            per_sample_magnitude=qformer_per_sample_magnitude,
            gate_init=qformer_gate_init,
        )
        self._init_prompts(prompt_path, prompt_template, max_txt_len, end_sym)
        self._apply_freeze_policy()

    # ------------------------------------------------------------- components
    def _init_rec_model(self, rec_model, rec_config, pretrained_rec, freeze_rec):
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config)
        if self.rec_encoder is not None and pretrained_rec and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            log_step("Loaded pretrained MF", pretrained_rec)

        # Freezing installs a ``train = disabled_train`` patch on the instance,
        # which shadows the class method and is awkward to undo — so SKIP it
        # rather than applying and reverting.
        if freeze_rec and self.rec_encoder is not None:
            for p in self.rec_encoder.parameters():
                p.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            log_step("MF frozen", "freeze_rec=True")

    def _init_llm_model(self, llm_path):
        model_path = llm_path if llm_path else "./content/ckpt/llm/base"
        model_path, local_files_only = resolve_hf_model_path(model_path)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=True, local_files_only=local_files_only,
        )
        for p in self.llm_model.parameters():
            p.requires_grad = False

        self._resolve_soft_token_placeholder()
        log_step(
            "LLM loaded",
            f"hidden_size={self.llm_model.config.hidden_size}, "
            f"vocab={self.llm_model.config.vocab_size}, "
            f"soft_token_id={self._soft_token_id} ('{self._soft_token_str}')",
        )
        if self.use_lora:
            self._attach_lora()

    def _resolve_soft_token_placeholder(self):
        """Pick a single-token id to stand in for each soft slot in the prompt.

        Same policy as ``QRecLLM``: prefer ``unk``, else a reserved special
        token, and only fall back to ``eos`` (which collides with padding) with a
        loud warning.
        """
        tok = self.llm_tokenizer
        if tok.unk_token_id is not None:
            self._soft_token_str = tok.unk_token
            self._soft_token_id = tok.unk_token_id
            return

        skip_ids = {tok.eos_token_id, tok.pad_token_id, tok.bos_token_id}
        skip_ids.discard(None)
        for candidate in (
            "<|extra_0|>", "<|reserved_0|>", "<|fim_pad|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>", "<|im_start|>",
        ):
            ids = tok(candidate, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] not in skip_ids:
                self._soft_token_str = candidate
                self._soft_token_id = ids[0]
                return

        for token_id, added_token in (getattr(tok, "added_tokens_decoder", None) or {}).items():
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
            "no unk_token and no safe reserved single token; using eos as the "
            "soft-slot placeholder. Slots will COLLIDE with padding.",
        )
        self._soft_token_str = tok.eos_token
        self._soft_token_id = tok.eos_token_id

    def _attach_lora(self):
        from peft import LoraConfig, TaskType, get_peft_model

        self.llm_model = get_peft_model(
            self.llm_model,
            LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=list(self.lora_target_modules),
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        log_step(
            "LoRA attached",
            f"r={self.lora_r}, alpha={self.lora_alpha}, "
            f"targets={list(self.lora_target_modules)}, "
            f"params={count_trainable_parameters(self.llm_model)}",
        )

    def _init_warm_token(self, pretrained_item_llm_emb):
        """``<Warm_ID>`` source, and the semantic bank the Q-Former reads.

        Prefers the MF's TRAINED ``item_embedding_llm`` table over the raw
        distilled bank: when the MF was trained with the SeLLa Step-2 alignment
        that table started FROM the bank and was then trained jointly with the
        InfoNCE, so reading the file instead would throw that training away.
        Reading the live module also keeps the table trainable under
        ``freeze_rec=False``, matching SeLLa's trainable ``<Warm_ID>`` source.
        """
        self.warm_from_rec = False
        self.warm_proj = None
        self.sem_dim = None
        # NOTE: `item_llm_emb` is registered as a BUFFER below, and only on the
        # branch that needs it. Do not pre-assign it to None here —
        # ``nn.Module.register_buffer`` refuses a name that already exists as a
        # plain attribute, so the two paths have to stay mutually exclusive.

        H = int(self.llm_model.config.hidden_size)
        rec_table = getattr(self.rec_encoder, "item_embedding_llm", None)
        source = None
        table_shape = None

        if rec_table is not None:
            self.sem_dim = int(rec_table.weight.size(-1))
            if self.sem_dim == H:
                self.warm_from_rec = True
                source = "rec_encoder.item_embedding_llm (SeLLa Step-2 trained, live module)"
                table_shape = tuple(rec_table.weight.shape)

        if not self.warm_from_rec:
            path = pretrained_item_llm_emb or self._item_sem_emb_path
            if not path or not os.path.exists(path):
                self.item_llm_emb = None
                if self.warm_token:
                    raise FileNotFoundError(
                        "warm_token=True but neither rec_encoder.item_embedding_llm "
                        f"(LLM-hidden sized) nor a bank file was available: {path!r}"
                    )
                log_step(
                    "No semantic bank",
                    "warm_token=False and no bank file — the Q-Former will run "
                    "with CF slots only (semantic slots off).",
                )
                self.sem_dim = None
                return
            blob = torch.load(path, map_location="cpu")
            table = (blob["item_llm_emb"] if isinstance(blob, dict) else blob).float()
            self.register_buffer("item_llm_emb", table.to(self.device), persistent=False)
            self.sem_dim = int(table.size(-1))
            source = f"raw bank {path} (frozen buffer)"
            table_shape = tuple(table.shape)
            if self.warm_token and self.sem_dim != H:
                raise ValueError(
                    f"warm_token=True but the bank hidden size is {self.sem_dim}, "
                    f"expected the LLM hidden size {H}"
                )
        else:
            self.item_llm_emb = None

        if self.warm_token:
            self.warm_proj = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H)).to(self.device)
            nn.init.normal_(self.warm_proj[0].weight, std=0.02)
            nn.init.zeros_(self.warm_proj[0].bias)
            nn.init.constant_(self.warm_proj[1].weight, H ** -0.5)
            nn.init.zeros_(self.warm_proj[1].bias)
            log_step(
                "<Warm_ID> active",
                f"warm_proj(e^L_item), table={table_shape}, source={source}",
            )
        else:
            log_step(
                "<Warm_ID> off",
                "the placeholder is stripped from the prompt; the semantic bank "
                f"is still used for the Q-Former's semantic slots (dim={self.sem_dim}).",
            )

    def _warm_table_rows(self, item_ids):
        """Rows of the semantic (LLM-distilled) item table for ``item_ids``.

        Works for any leading shape, so the same call serves the ``<Warm_ID>``
        target lookup ``[B]`` and the Q-Former's history slots ``[B, L]``.
        """
        if self.warm_from_rec:
            return self.rec_encoder.item_embedding_llm(item_ids)
        if self.item_llm_emb is None:
            return None
        return self.item_llm_emb[item_ids]

    def _init_id_proj(self, d_cf, pretrained_rec):
        """SeLLa's ``LinearProjection``: ONE shared MLP for ``<UserID>`` and
        ``<ItemID>`` (``prepare_collm_prompt`` runs user and item embeddings
        through the same ``projection_model``).

        Two knobs, both about the SCALE of what lands in the prompt:

        ``id_proj_warm_start`` (default True) copies the MF checkpoint's
        ``trans_1``/``trans_2``. Note what those were trained for: the InfoNCE in
        the MF's alignment pulls ``trans_2(GELU(trans_1(e_i)))`` toward
        ``item_embedding_llm``, i.e. toward the LLM's **LAST-HIDDEN** space. Here
        the output is injected as an **INPUT** embedding. In a 7B model those two
        spaces differ in norm by an order of magnitude, so the warm start can put
        a wildly out-of-scale vector into the prompt — and with LoRA frozen the
        LLM cannot adapt to it. SeLLa's own shipped code uses
        ``pretrained_with_small=False``, i.e. no warm start at all; the warm start
        is a SigLLM addition.

        ``id_proj_norm`` (default False) appends a LayerNorm scaled to
        ``H ** -0.5``, exactly the treatment ``warm_proj`` already gets, which
        pins the output norm near a real token embedding's regardless of what the
        preceding layers produce. Turn it on if the "Soft-token scale vs real
        token embeddings" log line shows a ratio far from 1.
        """
        H = int(self.llm_model.config.hidden_size)
        hidden = 1024
        trans_state = None

        if not self.id_proj_warm_start:
            log_step(
                "id_proj warm start DISABLED",
                "id_proj_warm_start=False — fresh init, which is what SeLLa's "
                "shipped code does (pretrained_with_small=False).",
            )
        elif pretrained_rec and pretrained_rec != "not_have" and os.path.exists(pretrained_rec):
            mf_state = torch.load(pretrained_rec, map_location="cpu")
            if "trans_1.weight" in mf_state and "trans_2.weight" in mf_state:
                if int(mf_state["trans_2.weight"].shape[0]) == H:
                    hidden = int(mf_state["trans_1.weight"].shape[0])
                    trans_state = mf_state
                else:
                    log_step(
                        "id_proj: SKIPPED trans warm-start",
                        f"MF trans_2 outputs {int(mf_state['trans_2.weight'].shape[0])} "
                        f"dims but LLM hidden is {H} — different base LLM?",
                    )

        layers = [nn.Linear(d_cf, hidden), nn.GELU(), nn.Linear(hidden, H)]
        if self.id_proj_norm:
            layers.append(nn.LayerNorm(H))
        self.id_proj = nn.Sequential(*layers)
        if self.id_proj_init == "small":
            # SigLLM's original: std=0.02 on both Linears. Shrinks the output —
            # measured ||id_proj(e_u)|| = 0.27 vs ~1.2 for a real Qwen2 embedding.
            nn.init.normal_(self.id_proj[0].weight, std=0.02)
            nn.init.zeros_(self.id_proj[0].bias)
            nn.init.normal_(self.id_proj[2].weight, std=0.02)
            nn.init.zeros_(self.id_proj[2].bias)
        # "default" leaves PyTorch's Linear init untouched, which is exactly what
        # SeLLa's LinearProjection uses (it never re-initialises). Each default
        # Linear preserves per-coordinate std up to 1/sqrt(3), so the output norm
        # lands near ||e_u|| rather than far below it.
        if self.id_proj_norm:
            # Same scaling warm_proj uses: LayerNorm output has unit variance per
            # dim, so weight = H**-0.5 puts the vector's norm near 1 — the order
            # of magnitude of a real Qwen2 input embedding.
            nn.init.constant_(self.id_proj[3].weight, H ** -0.5)
            nn.init.zeros_(self.id_proj[3].bias)
        self.id_proj = self.id_proj.to(self.device)

        if trans_state is not None:
            self.id_proj[0].weight.data.copy_(trans_state["trans_1.weight"])
            self.id_proj[0].bias.data.copy_(trans_state["trans_1.bias"])
            self.id_proj[2].weight.data.copy_(trans_state["trans_2.weight"])
            self.id_proj[2].bias.data.copy_(trans_state["trans_2.bias"])
            log_step(
                "id_proj warm-started from MF trans_1/trans_2",
                f"(SeLLa pretrained_with_small=True) {d_cf}->{hidden}->{H}",
            )
        else:
            log_step("id_proj fresh init", f"{d_cf}->{hidden}->{H}")

    def _init_history_qformer(
        self, d_cf, d_model, num_queries, num_heads, num_layers, dropout,
        use_sem, use_user_slot, hist_target_fusion, delta_scale,
        per_sample_magnitude, gate_init,
    ):
        if not self.use_qformer:
            self.history_qformer = None
            log_step(
                "History Q-Former DISABLED",
                "use_qformer=False — this is the SeLLa baseline arm of the A/B.",
            )
            return

        d_sem = self.sem_dim if (use_sem and self.sem_dim) else None
        if use_sem and not d_sem:
            log_step(
                "Q-Former semantic slots off",
                "no semantic bank is available; memory is CF slots + the "
                "user-target slot only.",
            )

        self.history_qformer = GatedHistoryQFormer(
            d_cf=d_cf,
            d_out=int(self.llm_model.config.hidden_size),
            d_sem=d_sem,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_sem=use_sem,
            use_user_slot=use_user_slot,
            hist_target_fusion=hist_target_fusion,
            delta_scale=delta_scale,
            per_sample_magnitude=per_sample_magnitude,
            gate_init=gate_init,
        ).to(self.device)

        log_step("History Q-Former built", self.history_qformer.describe())
        log_step(
            "Gate",
            "gate=0.0 at init -> e_user is EXACTLY id_proj(e_u) on step 0 "
            "(SeLLa, bit for bit). Watch qformer/gate and qformer/delta_rel in "
            "the training log: they are the reported contribution of this module.",
        )

    def _init_prompts(self, prompt_path, prompt_template, max_txt_len, end_sym):
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False
        self.prompt_list = []

        if prompt_path:
            with open(prompt_path, "r") as f:
                raw = f.read().splitlines()
            kept = [p for p in raw if p.strip() and not p.lstrip().startswith("#")]
            self.prompt_list = [prompt_template.format(p) for p in kept]
            log_step(f"Loaded {len(self.prompt_list)} prompts", prompt_path)

        if not self.prompt_list:
            raise ValueError("prompt_path produced no usable prompts")

        missing = [
            ph for ph in ("<UserID>", "<ItemID>")
            if not all(ph in p for p in self.prompt_list)
        ]
        if missing:
            raise ValueError(
                f"every prompt must carry {missing} — this model injects the "
                "collaborative signal only through SeLLa's soft tokens, so a "
                "prompt without them silently trains a text-only model."
            )
        if self.warm_token and not all("<Warm_ID>" in p for p in self.prompt_list):
            raise ValueError("warm_token=True but some prompts lack <Warm_ID>")

    def _apply_freeze_policy(self):
        """SeLLa step-3 policy: the base LLM is frozen; LoRA comes warm from the
        step-1 checkpoint and stays FROZEN by default (SeLLa freezes it — step 1
        already adapted the LLM to the task); MF follows ``freeze_rec``; the
        projections and the Q-Former train.
        """
        if self.use_lora and hasattr(self.llm_model, "peft_config"):
            for n, p in self.llm_model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = self.lora_trainable

        unfrozen = []
        for name, module in (
            ("id_proj", self.id_proj),
            ("warm_proj", self.warm_proj),
            ("history_qformer", self.history_qformer),
        ):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = True
            module.train()
            unfrozen.append(name)

        mf_trainable = self.rec_encoder is not None and any(
            p.requires_grad for p in self.rec_encoder.parameters()
        )
        if mf_trainable:
            self.rec_encoder.train()
            self._snapshot_mf_weights()

        # Name only the modules that actually exist — this line is what a reader
        # uses to confirm which arm of the experiment is running, so it must not
        # claim a module that was never built.
        log_step(
            "Freeze policy",
            "base LLM frozen; LoRA "
            + ("TRAINABLE" if self.lora_trainable else "frozen (SeLLa parity)")
            + f"; trainable modules: {' + '.join(unfrozen) if unfrozen else 'none'}"
            + "; MF "
            + ("TRAINABLE — watch MF drift" if mf_trainable else "frozen"),
        )

    # ------------------------------------------------------------ diagnostics
    def _snapshot_mf_weights(self):
        self._mf_init_weights = {
            name: param.detach().to("cpu", copy=True)
            for name, param in self.rec_encoder.named_parameters()
        }

    @torch.no_grad()
    def mf_drift(self):
        """``{name: ||W - W0|| / ||W0||}`` plus ``overall``, against the
        PRETRAINED MF. Empty when MF is frozen (no snapshot taken)."""
        snapshot = getattr(self, "_mf_init_weights", None)
        if not snapshot or self.rec_encoder is None:
            return {}
        drift, total_sq, base_sq = {}, 0.0, 0.0
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

    def gate_stats(self):
        """``{gate, delta_norm, delta_rel}`` for the last forward, or ``{}``."""
        if self.history_qformer is None:
            return {}
        return self.history_qformer.pop_stats()

    def _log_trainable_stats(self):
        if self._has_logged_trainable_stats:
            return
        lora_total = 0
        if self.llm_model is not None and hasattr(self.llm_model, "peft_config"):
            lora_total = sum(
                p.numel() for n, p in self.llm_model.named_parameters()
                if "lora_" in n and p.requires_grad
            )
        log_step(
            "Trainable parameter counts",
            ", ".join([
                f"rec_encoder={count_trainable_parameters(self.rec_encoder) if self.rec_encoder is not None else 0}",
                f"id_proj={count_trainable_parameters(self.id_proj)}",
                f"warm_proj={count_trainable_parameters(self.warm_proj) if self.warm_proj is not None else 0}",
                f"history_qformer={count_trainable_parameters(self.history_qformer) if self.history_qformer is not None else 0}",
                f"llm_lora={lora_total}",
            ]),
        )
        self._has_logged_trainable_stats = True

    def _maybe_log_periodic(self):
        """Gate + MF-drift diagnostics, throttled, training only."""
        if not self.training:
            return
        self._step_count += 1

        if self.gate_log_steps > 0 and self._step_count % self.gate_log_steps == 0:
            stats = self.gate_stats()
            if stats:
                log_step(
                    f"qformer gate (step {self._step_count})",
                    ", ".join(f"{k}={v:.6f}" for k, v in stats.items())
                    + " | gate is the reported contribution; delta_rel = "
                    "||g*delta||/||e_user||. Both flat at 0 means the module is "
                    "not earning its place.",
                )
                # delta_rel > 1 means the gated term is LARGER than the token it
                # is added to: <UserID> is no longer being refined, it is being
                # replaced by a vector the frozen LLM has never seen. This is not
                # a tuning issue, it is the difference between an addition and an
                # overwrite, and it costs uAUC immediately. It went unnoticed for
                # a whole run (measured delta_rel = 12.2) because nothing shouted.
                rel = stats.get("delta_rel")
                if rel is not None and rel > 1.0:
                    log_step(
                        "!! GATED TERM IS OVERWRITING <UserID>",
                        f"delta_rel={rel:.2f} — the added term is {rel:.1f}x the "
                        "norm of id_proj(e_u). With delta_scale=match_ref this "
                        "cannot exceed |gate|, so seeing it here means "
                        "delta_scale='none' (the unnormalised out_proj output). "
                        "Set model.sella_gated.delta_scale=match_ref.",
                    )

        if self.mf_drift_log_steps > 0 and self._step_count % self.mf_drift_log_steps == 0:
            drift = self.mf_drift()
            if drift:
                log_step(
                    f"MF drift (step {self._step_count})",
                    ", ".join(f"{k}={v:.5f}" for k, v in sorted(drift.items()))
                    + " | rising drift with flat val uAUC means run.rec_lr_scale "
                    "is too high",
                )

    # ------------------------------------------------------------- run config
    def set_mode(self, mode):
        self.run_mode_ = mode

    def to_be_trained(self):
        return True

    def set_answer_type(self, mode):
        if mode == "v2":
            self.pos_ans = ["Yes"]
            self.neg_ans = ["No"]
        elif mode == "v1":
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        else:
            raise NotImplementedError(f"unsupported ans_type: {mode}")
        pos_id = self.llm_tokenizer(self.pos_ans[0], add_special_tokens=False).input_ids
        neg_id = self.llm_tokenizer(self.neg_ans[0], add_special_tokens=False).input_ids
        if len(pos_id) != 1 or len(neg_id) != 1:
            raise ValueError(
                f"answer words must be single tokens; got {self.pos_ans[0]}->{pos_id}, "
                f"{self.neg_ans[0]}->{neg_id}. The Yes/No softmax reads ONE position."
            )
        log_step("Answer tokens", f"pos={self.pos_ans[0]}:{pos_id[0]}, neg={self.neg_ans[0]}:{neg_id[0]}")

    def print_prompt(self):
        log_step("Prompt example", f"{self.prompt_list[0]} {self.pos_ans[0]} or {self.neg_ans[0]}")

    def _sample_prompt(self):
        return random.choice(self.prompt_list) if self.training else self.prompt_list[0]

    def _placeholder_order(self, prompt):
        """Active soft-token placeholders, ordered by position IN THE PROMPT.

        The injection scatter relies on this: ``torch.nonzero`` returns the
        placeholder positions in increasing order per row, so the embeddings must
        be concatenated in prompt order for the two to line up.
        """
        active = []
        for ph in self.PLACEHOLDERS_FOR_EMBED:
            if ph == "<Warm_ID>" and not self.warm_token:
                continue
            pos = prompt.find(ph)
            if pos >= 0:
                active.append((pos, ph))
        active.sort(key=lambda x: x[0])
        return [ph for _, ph in active]

    # ------------------------------------------------------------- soft tokens
    def _soft_token_embeddings(self, batch_data):
        """The three SeLLa soft-token embeddings, ``<UserID>`` gated-augmented.

        Returns ``{placeholder: [B, 1, H]}``.
        """
        self._log_trainable_stats()

        user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])        # [B,d_cf]
        target_cf = self.rec_encoder.item_encoder(batch_data["TargetItemID"])  # [B,d_cf]

        user_llm = self.id_proj(user_cf)                                     # [B,H]
        item_llm = self.id_proj(target_cf)                                   # [B,H]

        # ---- the ONLY change relative to SeLLa: a gated addition on <UserID>.
        if self.history_qformer is not None and not self.ablate_qformer:
            ids = batch_data.get("InteractedItemIDs_pad")
            if ids is None:
                raise KeyError(
                    "the history Q-Former needs 'InteractedItemIDs_pad'; the "
                    "dataset produced no history column"
                )
            ids = ids.long()
            hist_cf = self.rec_encoder.item_encoder(ids)                     # [B,L,d_cf]
            # padding_index is the MF's own convention (0). The dataset LEFT-pads
            # the history with 0, so this mask is what keeps padded slots out of
            # the cross-attention.
            hist_mask = ids != self.rec_encoder.padding_index                # [B,L]
            hist_sem = (
                self._warm_table_rows(ids) if self.history_qformer.uses_sem else None
            )
            delta = self.history_qformer(
                user_cf=user_cf,
                target_cf=target_cf,
                hist_cf=hist_cf,
                hist_mask=hist_mask,
                hist_sem=hist_sem,
                reference=user_llm,
            )
            user_llm = user_llm + delta.to(user_llm.dtype)

        embeds = {
            "<UserID>": user_llm.unsqueeze(1),
            "<ItemID>": item_llm.unsqueeze(1),
        }
        if self.warm_token and self.warm_proj is not None:
            warm_rows = self._warm_table_rows(batch_data["TargetItemID"])
            embeds["<Warm_ID>"] = self.warm_proj(warm_rows).unsqueeze(1)
        return embeds

    @torch.no_grad()
    def _log_soft_token_scale(self, embeds, order, inputs_embeds, text_mask):
        """Compare each injected soft token's L2 norm against the REAL token
        embeddings in the same batch, and complain if they are not comparable.

        This is the failure that produced nothing but a worse number the first
        time round, with no error anywhere. ``id_proj`` warm-starts from the MF's
        ``trans_1``/``trans_2``, and those were trained to land in the LLM's
        LAST-HIDDEN space (the distilled ``item_llm_emb`` bank is
        ``space=last_hidden``) — but the output is injected at an INPUT-embedding
        position. Those two spaces differ in scale by an order of magnitude or
        more in a 7B model, and unlike ``warm_proj`` (Linear + LayerNorm)
        ``id_proj`` has no normalisation to absorb it. A token that is 30x too
        large is out of distribution for every position around it, and with LoRA
        frozen the LLM cannot adapt to it.

        A ratio near 1 means the token is in-distribution. Far from 1 means the
        collaborative channel is being injected as noise, and the honest read of
        a bad uAUC is "the injection is broken", not "collaborative signal does
        not help".
        """
        text = inputs_embeds[text_mask].detach().float()
        if text.numel() == 0:
            return
        text_norm = text.norm(dim=-1).mean().item()
        parts = [f"text_tokens_mean_l2={text_norm:.3f}"]
        worst_name, worst_ratio = None, 1.0
        for ph in order:
            emb = embeds[ph].detach().float()
            n = emb.norm(dim=-1).mean().item()
            ratio = n / text_norm if text_norm > 0 else float("inf")
            parts.append(f"{ph}={n:.3f} (x{ratio:.2f})")
            if abs(ratio - 1.0) > abs(worst_ratio - 1.0):
                worst_name, worst_ratio = ph, ratio
        log_step("Soft-token scale vs real token embeddings", ", ".join(parts))
        if worst_name is not None and (worst_ratio > 5.0 or worst_ratio < 0.2):
            log_step(
                "!! SOFT-TOKEN SCALE MISMATCH",
                f"{worst_name} is {worst_ratio:.1f}x the norm of a real token "
                "embedding. The LLM sees an out-of-distribution vector at that "
                "position and, with LoRA frozen, cannot adapt to it — expect uAUC "
                "BELOW the text-only step-1 baseline. Most likely cause: id_proj "
                "warm-started from MF trans_1/trans_2, which map into the LLM's "
                "LAST-HIDDEN space, not its input-embedding space. Fix by "
                "training with model.sella_gated.id_proj_norm=True (adds a "
                "LayerNorm on id_proj's output, same treatment warm_proj already "
                "gets), or drop the warm start.",
            )

    def _build_prompt_inputs(self, prompt_template, batch_data):
        """Render the prompt, inject the three soft tokens, return
        ``(inputs_embeds, attention_mask, input_ids)``.

        ``input_ids`` is returned because the full-sequence LM loss needs the
        prompt targets — SeLLa supervises the whole non-padding sequence.
        """
        order = self._placeholder_order(prompt_template)
        embeds = self._soft_token_embeddings(batch_data)
        device = batch_data["UserID"].device
        batch_size = batch_data["UserID"].shape[0]

        unk = self._soft_token_str
        bos = self.llm_tokenizer.bos_token if self.llm_tokenizer.bos_token else ""

        text = bos + prompt_template
        for ph in self.PLACEHOLDERS_FOR_EMBED:
            text = text.replace(ph, unk if ph in order else "")

        prompts = []
        for k in range(batch_size):
            current = text
            if "<ItemTitleList>" in current and "InteractedItemTitles" in batch_data:
                current = current.replace(
                    "<ItemTitleList>", str(batch_data["InteractedItemTitles"][k])
                )
            if "<TargetItemTitle>" in current and "TargetItemTitle" in batch_data:
                current = current.replace(
                    "<TargetItemTitle>", str(batch_data["TargetItemTitle"][k])
                )
            prompts.append(current)

        self.llm_tokenizer.padding_side = "left"
        tokens = self.llm_tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(device)

        inputs_embeds = self.llm_model.get_input_embeddings()(tokens.input_ids)

        slots_mask = tokens.input_ids == self._soft_token_id
        slot_idx = torch.nonzero(slots_mask, as_tuple=False)
        expected = batch_size * len(order)
        if slot_idx.shape[0] != expected:
            # Silence here would mean the soft tokens land in the wrong places
            # (or not at all) and the collaborative channel is quietly dead —
            # the usual cause is right-truncation eating the tail of the prompt.
            raise RuntimeError(
                f"expected {expected} soft-token slots ({len(order)} per sample x "
                f"{batch_size}), found {slot_idx.shape[0]}. Placeholders={order}, "
                f"max_txt_len={self.max_txt_len}, longest prompt="
                f"{int(tokens.attention_mask.sum(dim=1).max().item())} tokens. "
                "Raise max_txt_len or shorten <ItemTitleList>."
            )

        merged = torch.cat([embeds[ph] for ph in order], dim=1)          # [B, n_ph, H]
        inputs_embeds[slot_idx[:, 0], slot_idx[:, 1]] = merged.reshape(
            -1, merged.shape[-1]
        ).to(inputs_embeds)

        if not self._has_logged_injection_stats:
            log_step(
                "Prompt injection stats",
                f"placeholders={order}, slots_per_sample={len(order)}, "
                f"batch_slots={slot_idx.shape[0]}, "
                f"prompt_tokens={int(tokens.input_ids.shape[1])}, "
                f"valid_history[0]={int((batch_data['InteractedItemIDs_pad'][0] != self.rec_encoder.padding_index).sum().item()) if 'InteractedItemIDs_pad' in batch_data else 0}",
            )
            self._log_soft_token_scale(embeds, order, inputs_embeds, ~slots_mask)
            self._has_logged_injection_stats = True

        if not self.has_print_prompt:
            log_step("Rendered prompt[0]", prompts[0])
            self.has_print_prompt = True

        return inputs_embeds, tokens.attention_mask, tokens.input_ids

    def _build_answer(self, batch_data):
        device = batch_data["UserID"].device
        ans_map = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        texts = [ans_map[int(label)] for label in batch_data["label"]]

        self.llm_tokenizer.padding_side = "right"
        tokens = self.llm_tokenizer(
            texts, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_txt_len, add_special_tokens=False,
        ).to(device)
        embeds = self.llm_model.get_input_embeddings()(tokens.input_ids)
        return embeds, tokens, ans_map

    # -------------------------------------------------------------- objectives
    def _lm_loss(self, logits, prompt_ids, prompt_atts, answer_tokens):
        """SeLLa's objective: LM cross-entropy over the whole sequence.

        SeLLa builds ``target_ids = input_ids.clone()`` and masks only the
        padding, so the instruction, the user turn AND the soft-token positions
        are all supervised. ``scope='full'`` reproduces that exactly;
        ``full_no_soft`` additionally masks the soft slots (whose target is the
        placeholder id, which is not a meaningful thing to predict and, unlike
        SeLLa's three distinct ids, is the SAME id at all three slots here);
        ``answer`` supervises only the Yes/No token.

        Only the supervised rows are gathered before the fp32 upcast — a
        full-vocabulary fp32 logit tensor over a padded sequence is several GB at
        this batch size, and most of it would be thrown away by ``ignore_index``.
        """
        if self.lm_loss_scope == "answer":
            prompt_targets = torch.full_like(prompt_ids, -100)
        else:
            prompt_targets = prompt_ids.masked_fill(prompt_atts == 0, -100)
            if self.lm_loss_scope == "full_no_soft":
                prompt_targets = prompt_targets.masked_fill(
                    prompt_ids == self._soft_token_id, -100
                )

        answer_targets = answer_tokens.input_ids.masked_fill(
            answer_tokens.attention_mask == 0, -100
        )
        targets = torch.cat([prompt_targets, answer_targets], dim=1)

        # The prompt is LEFT-padded (SeLLa right-pads), so without this the target
        # at the FIRST real token would be supervised from the last PAD position —
        # a position whose attention row is fully masked and whose hidden state is
        # therefore meaningless. Require the predicting position (t-1) to be real.
        # Under right padding this is a no-op, which is why SeLLa needs no
        # equivalent; here it is what makes "supervise the whole sequence" mean
        # the same thing.
        full_atts = torch.cat([prompt_atts, answer_tokens.attention_mask], dim=1)
        predictable = torch.zeros_like(full_atts, dtype=torch.bool)
        predictable[:, 1:] = full_atts[:, :-1] != 0
        targets = targets.masked_fill(~predictable, -100)

        shift_logits = logits[:, :-1, :]
        shift_labels = targets[:, 1:]
        keep = shift_labels != -100
        if not bool(keep.any()):
            raise RuntimeError("LM loss has no supervised positions")

        selected_logits = shift_logits[keep].float()      # [N, V]
        selected_labels = shift_labels[keep]              # [N]

        if not self._has_logged_loss_scope:
            log_step(
                "LM loss",
                f"scope={self.lm_loss_scope}, supervised_positions={int(keep.sum().item())}"
                f"/{int(shift_labels.numel())}, vocab={selected_logits.shape[-1]}",
            )
            self._has_logged_loss_scope = True

        return F.cross_entropy(selected_logits, selected_labels)

    def recommendation_scores(self, logits, answer_tokens, ans_map):
        """P(Yes) from the softmax over the two answer tokens at the answer
        position — SeLLa's eval readout, unchanged."""
        pos_id = self.llm_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        neg_id = self.llm_tokenizer(ans_map[0], add_special_tokens=False).input_ids[0]
        answer_len = answer_tokens.input_ids.shape[-1]
        prediction_logits = logits[:, -(answer_len + 1), :]
        binary = torch.stack(
            [prediction_logits[:, neg_id], prediction_logits[:, pos_id]], dim=1
        ).float()
        return torch.softmax(binary, dim=1)[:, 1]

    # ---------------------------------------------------------------- forward
    def _forward_llm(self, prompt_template, batch_data):
        input_embeds, input_atts, input_ids = self._build_prompt_inputs(
            prompt_template, batch_data
        )
        answer_embeds, answer_tokens, ans_map = self._build_answer(batch_data)

        full_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
        full_atts = torch.cat([input_atts, answer_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=full_embeds, attention_mask=full_atts, return_dict=True
            )
        return outputs.logits, input_ids, input_atts, answer_tokens, ans_map

    def forward_v2(self, batch_data):
        with self.maybe_autocast():
            logits, input_ids, input_atts, answer_tokens, _ = self._forward_llm(
                self._sample_prompt(), batch_data
            )
        loss = self._lm_loss(logits, input_ids, input_atts, answer_tokens)
        self._maybe_log_periodic()
        return {"loss": loss}

    def forward(self, samples):
        if self.run_mode_ == "v2":
            return self.forward_v2(samples)
        raise NotImplementedError("only run.mode='v2' is implemented")

    def generate_for_samples(self, samples, return_all=False):
        with self.maybe_autocast():
            logits, input_ids, input_atts, answer_tokens, ans_map = self._forward_llm(
                self.prompt_list[0], samples
            )
        loss = self._lm_loss(logits, input_ids, input_atts, answer_tokens)
        scores = self.recommendation_scores(logits, answer_tokens, ans_map)
        if return_all:
            return logits, scores
        return {"loss": loss, "logits": scores}

    # ------------------------------------------------------------- from_config
    @classmethod
    def from_config(cls, cfg):
        rec_config = cfg.get("rec_config")
        sella = cfg.get("sella_gated") or {}
        qformer_cfg = cfg.get("qformer_config") or {}
        lora_cfg = cfg.get("lora_config") or {}

        # The semantic bank: `sella_gated.item_llm_emb_path` if given, else the
        # bank the rest of the config already points at.
        bank = sella.get("item_llm_emb_path") or qformer_cfg.get("item_llm_emb_path")

        model = cls(
            rec_model=cfg.get("rec_model", "MF"),
            rec_config=rec_config,
            pretrained_rec=rec_config["pretrained_path"],
            freeze_rec=bool(cfg.get("freeze_rec", False)),
            llm_model=cfg.get("llm_model"),
            prompt_path=cfg.get("prompt_path", ""),
            prompt_template=cfg.get("prompt_template", ""),
            max_txt_len=cfg.get("max_txt_len", 1024),
            end_sym=cfg.get("end_sym", "\n"),
            use_lora=bool(lora_cfg.get("use_lora", True)),
            lora_r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            lora_trainable=bool(sella.get("lora_trainable", False)),
            warm_token=bool(sella.get("warm_token", True)),
            pretrained_item_llm_emb=bank,
            item_sem_emb_path=qformer_cfg.get("item_sem_emb_path"),
            use_qformer=bool(sella.get("use_qformer", True)),
            qformer_d_model=int(sella.get("d_model", 256)),
            qformer_num_queries=int(sella.get("num_queries", 2)),
            qformer_num_heads=int(sella.get("num_heads", 4)),
            qformer_num_layers=int(sella.get("num_layers", 2)),
            qformer_dropout=float(sella.get("dropout", 0.0)),
            qformer_use_sem=bool(sella.get("use_sem", True)),
            qformer_use_user_slot=bool(sella.get("use_user_slot", True)),
            qformer_hist_target_fusion=bool(sella.get("hist_target_fusion", False)),
            qformer_delta_scale=str(sella.get("delta_scale", "match_ref")),
            qformer_per_sample_magnitude=bool(sella.get("per_sample_magnitude", True)),
            qformer_gate_init=float(sella.get("gate_init", 0.0)),
            ablate_qformer=bool(sella.get("ablate_qformer", False)),
            id_proj_norm=bool(sella.get("id_proj_norm", False)),
            id_proj_warm_start=bool(sella.get("id_proj_warm_start", True)),
            id_proj_init=str(sella.get("id_proj_init", "default")),
            lm_loss_scope=str(sella.get("lm_loss_scope", "full")),
            gate_log_steps=int(sella.get("gate_log_steps", 50)),
            mf_drift_log_steps=int(cfg.get("mf_drift_log_steps", 200)),
        )

        # --- LoRA source, loaded BEFORE the main checkpoint ------------------
        # The runner strips every requires_grad=False tensor when saving, and on
        # this branch LoRA is frozen (lora_trainable=False, SeLLa parity). So a
        # checkpoint produced by THIS stage carries no lora_* keys at all: it
        # holds the collaborative modules and nothing else. Loading it alone
        # leaves LoRA at its init, where lora_B is zeros — i.e. LoRA is the
        # identity and the "adapted" LLM is silently the raw base model.
        #
        # `lora_ckpt` names the checkpoint that DOES carry LoRA (normally the
        # step-1 one). It is loaded first so the main checkpoint can still
        # overwrite anything it legitimately owns.
        lora_loaded = 0
        lora_source = None
        lora_ckpt_path = cfg.get("lora_ckpt", None)
        if lora_ckpt_path and os.path.exists(lora_ckpt_path):
            blob = torch.load(lora_ckpt_path, map_location="cpu")
            state = blob.get("model", blob) if isinstance(blob, dict) else blob
            lora_state = {k: v for k, v in state.items()
                          if isinstance(k, str) and "lora_" in k}
            if lora_state:
                model.load_state_dict(lora_state, strict=False)
                lora_loaded = len(lora_state)
                lora_source = lora_ckpt_path
                log_step(
                    "LoRA restored from",
                    f"{lora_ckpt_path} ({lora_loaded} lora_* tensors)",
                )
            else:
                log_step(
                    "LoRA source carries NO lora_* keys",
                    f"{lora_ckpt_path} — check that it is a step-1 checkpoint.",
                )
        elif lora_ckpt_path:
            log_step("LoRA source not found", str(lora_ckpt_path))

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            log_step("Loading checkpoint", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt["model"], strict=False)
            log_step("load_state_dict", str(msg))
            main_lora = sum(
                1 for k in ckpt["model"] if isinstance(k, str) and "lora_" in k
            )
            if main_lora:
                lora_loaded += main_lora
                lora_source = ckpt_path
                log_step("LoRA restored from", f"{ckpt_path} ({main_lora} lora_* tensors)")

            # Restore the pretrained MF ONLY when the checkpoint does not carry
            # its own: the runner strips frozen tensors when saving, so a step-1
            # (LoRA-only) checkpoint has no rec_encoder.* keys and the reload is
            # what puts MF back after the strict=False load. A checkpoint from
            # THIS step does carry them, and reloading mf_model.pth over those
            # would throw away every MF update the run produced.
            ckpt_has_rec = any(
                isinstance(k, str) and k.startswith("rec_encoder.") for k in ckpt["model"]
            )
            gated_keys = [
                k for k in ckpt["model"]
                if isinstance(k, str) and k.startswith("history_qformer.")
            ]
            if ckpt_has_rec:
                log_step(
                    "Kept MF from the checkpoint",
                    "checkpoint carries rec_encoder.*; NOT reloading "
                    f"{rec_config['pretrained_path']}",
                )
            elif os.path.exists(rec_config["pretrained_path"]):
                model.rec_encoder.load_state_dict(
                    torch.load(rec_config["pretrained_path"], map_location="cpu")
                )
                log_step("Restored pretrained MF", rec_config["pretrained_path"])
            log_step(
                "Q-Former warm start",
                f"{len(gated_keys)} history_qformer.* tensors in the checkpoint"
                + (
                    " — gate resumes at its trained value."
                    if gated_keys
                    else " — FRESH Q-Former, gate starts at 0 (expected when "
                    "warm-starting from a step-1 LoRA checkpoint)."
                ),
            )

        # --- the guard that makes the failure impossible to miss -------------
        # A frozen, never-loaded LoRA is the one failure mode of this branch
        # that costs nothing at build time, raises no error, and simply reports
        # worse numbers — the base model wearing an identity adapter. It has to
        # be an exception, not a warning. `allow_cold_lora` is the explicit
        # opt-in (ckpt_from: none sets it).
        if model.use_lora and not model.lora_trainable and lora_loaded == 0:
            if bool(cfg.get("allow_cold_lora", False)):
                log_step(
                    "COLD LoRA (explicitly allowed)",
                    "LoRA is frozen at its zero-init, i.e. the identity — the LLM "
                    "is the unadapted base model. Only meaningful as a deliberate "
                    "ablation.",
                )
            else:
                raise RuntimeError(
                    "LoRA is frozen (lora_trainable=False) but NO lora_* tensors "
                    "were loaded, so lora_B is still zeros and the adapter is the "
                    "identity — this would evaluate the unadapted base model and "
                    "silently report worse numbers.\n"
                    f"  model.ckpt      = {ckpt_path or None}\n"
                    f"  model.lora_ckpt = {lora_ckpt_path or None}\n"
                    "Fix: point run.sella_gated_step3.lora_from at the stage whose "
                    "checkpoint carries LoRA (default 'sella_step1'), or set "
                    "run.sella_gated_step3.ckpt_from=none to opt into a cold LoRA."
                )
        elif model.use_lora and not model.lora_trainable:
            log_step(
                "LoRA check OK",
                f"{lora_loaded} lora_* tensors loaded from {lora_source}; frozen "
                "for training (SeLLa parity).",
            )

        model.set_answer_type(mode=cfg.get("ans_type", "v2"))
        model.print_prompt()
        return model
