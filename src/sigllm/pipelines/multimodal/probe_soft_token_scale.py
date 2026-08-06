"""Is the collaborative channel being injected as noise?

Answers one question in about a minute, without training anything: do the three
SeLLa soft tokens land at a norm comparable to a REAL Qwen2 input embedding?

Why it matters. ``<UserID>``/``<ItemID>`` are ``id_proj(e_u)`` / ``id_proj(e_i)``,
and ``id_proj`` warm-starts from the MF checkpoint's ``trans_1``/``trans_2``.
Those were trained by the MF's InfoNCE to match ``item_embedding_llm``, i.e. to
land in the LLM's **last-hidden** space — but they are injected as **input**
embeddings. Those two spaces differ in norm by an order of magnitude in a 7B
model, and ``id_proj`` (unlike ``warm_proj``) has no LayerNorm to absorb it. With
LoRA frozen the LLM cannot adapt to an out-of-distribution token, so the symptom
is simply a uAUC below the text-only step-1 baseline — with no error anywhere.

A ratio near 1 clears id_proj as a suspect. A ratio of 10x+ means the channel is
broken and the run says nothing about whether collaborative signal helps.

Usage:
    python -m sigllm.pipelines.multimodal.probe_soft_token_scale \\
        --cfg-path configs/config.yaml
"""

import argparse

import torch

from sigllm.common.config import Config
from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.utils import resolve_hf_model_path

LOGGER = NotebookLogger.rich_logger("sigllm.probe_soft_token_scale")


def parse_args():
    p = argparse.ArgumentParser(description="Probe soft-token norms vs real token embeddings")
    p.add_argument("--cfg-path", type=str, required=True)
    p.add_argument("--options", nargs="+")
    return p.parse_args()


def _norms(x):
    n = x.float().norm(dim=-1)
    return n.mean().item(), n.median().item(), n.max().item()


@torch.no_grad()
def main():
    cfg = Config(parse_args())
    model_cfg = cfg.model_cfg
    rec_cfg = model_cfg.rec_config
    sella = model_cfg.get("sella_gated") or {}

    # ---- the LLM's input embedding table: the distribution we must match -----
    from transformers import AutoModelForCausalLM

    path, local_only = resolve_hf_model_path(model_cfg.llm_model)
    llm = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, trust_remote_code=True,
        local_files_only=local_only, device_map="cpu",
    )
    emb = llm.get_input_embeddings().weight
    H = emb.shape[1]
    e_mean, e_med, e_max = _norms(emb)
    LOGGER.info(
        "LLM input embeddings | shape=%s  L2 mean=%.4f median=%.4f max=%.4f",
        tuple(emb.shape), e_mean, e_med, e_max,
    )
    del llm

    # ---- the MF checkpoint --------------------------------------------------
    mf = torch.load(rec_cfg.pretrained_path, map_location="cpu")
    LOGGER.info("MF checkpoint | %s | keys=%d", rec_cfg.pretrained_path, len(mf))

    user_e = mf["user_embedding.weight"].float()
    item_e = mf["item_embedding.weight"].float()
    LOGGER.info("MF user emb L2 mean=%.4f | item emb L2 mean=%.4f",
                _norms(user_e)[0], _norms(item_e)[0])

    has_trans = "trans_1.weight" in mf and "trans_2.weight" in mf
    if not has_trans:
        LOGGER.warning(
            "MF checkpoint has no trans_1/trans_2 — id_proj gets a FRESH init, so "
            "the last-hidden-space mismatch cannot occur. This probe has nothing "
            "to flag; the scale question is settled."
        )
        return

    # ---- what id_proj would emit at step 0, WITH the warm start -------------
    def id_proj_warm(x):
        h = torch.nn.functional.linear(x, mf["trans_1.weight"], mf["trans_1.bias"])
        h = torch.nn.functional.gelu(h)
        return torch.nn.functional.linear(h, mf["trans_2.weight"], mf["trans_2.bias"])

    u_out = id_proj_warm(user_e)
    i_out = id_proj_warm(item_e)
    u_mean = _norms(u_out)[0]
    i_mean = _norms(i_out)[0]

    LOGGER.info("=" * 70)
    LOGGER.info(
        "<UserID> = id_proj(e_u) | L2 mean=%.4f  -> %.1fx a real token embedding",
        u_mean, u_mean / e_mean,
    )
    LOGGER.info(
        "<ItemID> = id_proj(e_i) | L2 mean=%.4f  -> %.1fx a real token embedding",
        i_mean, i_mean / e_mean,
    )

    if "item_embedding_llm.weight" in mf:
        sem = mf["item_embedding_llm.weight"].float()
        s_mean = _norms(sem)[0]
        LOGGER.info(
            "MF item_embedding_llm (the InfoNCE target, LAST-HIDDEN space) | "
            "L2 mean=%.4f -> %.1fx a real token embedding",
            s_mean, s_mean / e_mean,
        )
        LOGGER.info(
            "<Warm_ID> = warm_proj(that) — warm_proj ends in LayerNorm(weight=H**-0.5), "
            "so its output norm is ~%.3f regardless of the input scale.", H ** 0.5 * (H ** -0.5),
        )

    worst = max(u_mean / e_mean, i_mean / e_mean)
    LOGGER.info("=" * 70)
    if worst > 5.0 or worst < 0.2:
        LOGGER.error(
            "VERDICT: BROKEN. The direct-ID tokens are %.1fx off the input-embedding "
            "scale. With LoRA frozen the LLM cannot adapt to them, so a uAUC below "
            "the text-only step-1 baseline is EXPECTED and says nothing about "
            "collaborative signal. Rerun step 3 with either\n"
            "    --options model.sella_gated.id_proj_norm=True\n"
            "  or\n"
            "    --options model.sella_gated.id_proj_warm_start=False   (SeLLa's own default)",
            worst,
        )
    else:
        LOGGER.info(
            "VERDICT: id_proj scale is fine (worst %.2fx). Look elsewhere for the "
            "regression — start with model.sella_gated.lm_loss_scope=answer, since "
            "`full` spends ~99%% of its gradient on predicting prompt tokens rather "
            "than the Yes/No answer.", worst,
        )


if __name__ == "__main__":
    main()
