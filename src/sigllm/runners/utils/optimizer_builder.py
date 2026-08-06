import logging

import torch

# Parameters whose names contain this marker get their own optimizer group: the
# MF (collaborative) encoder. Matched as a substring so it works whether or not
# the model is DDP-wrapped (``module.rec_encoder.*``).
REC_ENCODER_MARKER = "rec_encoder."

# The gated history Q-Former (sella_gated_rec_llm). Gets its own group for a
# reason specific to a zero-init gate on a short budget: with Adam the per-step
# change in a scalar parameter is ~lr, so after N updates |gate| cannot exceed
# ~N * lr. At SeLLa's step-3 budget (110 updates, lr 2e-4) that ceiling is 0.022
# — and a measured run ended at 0.0103, i.e. 47% of the ceiling. The gate was not
# stuck; it was moving as fast as the budget allowed and ran out of steps. SeLLa's
# 110 updates were sized for a WARM-STARTED projection, not a from-scratch module
# behind a gate that has to climb from 0.
#
# run.qformer_lr_scale multiplies this group's LR (see the lr_scale handling in
# sigllm.common.optims, since the schedulers overwrite every group's lr each
# step). Empty for every other model, and empty groups are not appended, so
# nothing changes for QRecLLM.
QFORMER_MARKER = "history_qformer."


def _is_non_decay(name, param):
    """Biases, norms and any 1-D tensor are excluded from weight decay."""
    return param.ndim < 2 or "bias" in name or "ln" in name or "bn" in name


def build_optimizer(model, config):
    """Build the AdamW optimizer, with the MF encoder in its own param group.

    The MF encoder needs to be separable from everything else for two reasons,
    both of which only bite once joint tuning (``tuning_step: 3``) makes it
    trainable:

    1. **Learning rate.** The Stage-1/2 alignment was trained against a FIXED
       MF geometry, so MF has to move much more slowly than the modules that
       read it, or that alignment is invalidated faster than the Q-Former can
       track it. The LR schedulers overwrite ``param_group["lr"]`` on every
       group at every step, so a per-group *base* LR cannot survive on its own —
       the group carries an ``lr_scale`` that the schedulers multiply in (see
       ``sigllm.common.optims``). ``run.rec_lr_scale`` sets it.
    2. **Weight decay.** MF is a pair of embedding tables, i.e. 2-D, so the
       generic rule below files it under full ``weight_decay`` (1e-3 here) —
       an order of magnitude above the 1e-4 the MF baseline was trained with,
       applied to embeddings where decay pulls unvisited rows toward zero.
       ``run.rec_weight_decay`` gives it its own value.

    Groups are only created when they are non-empty, so for Steps 1 and 2
    (MF frozen, hence filtered out by ``requires_grad``) this produces exactly
    the same two groups as before.
    """
    weight_decay = float(config.run_cfg.weight_decay)
    rec_weight_decay = float(config.run_cfg.get("rec_weight_decay", 1e-4))
    rec_lr_scale = float(config.run_cfg.get("rec_lr_scale", 1.0))
    qformer_lr_scale = float(config.run_cfg.get("qformer_lr_scale", 1.0))

    p_wd, p_non_wd = [], []
    p_rec_wd, p_rec_non_wd = [], []
    p_qf_wd, p_qf_non_wd = [], []
    num_parameters = 0
    num_rec_parameters = 0
    num_qf_parameters = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if QFORMER_MARKER in name:
            (p_qf_non_wd if _is_non_decay(name, param) else p_qf_wd).append(param)
            num_qf_parameters += param.data.nelement()
        elif REC_ENCODER_MARKER in name:
            (p_rec_non_wd if _is_non_decay(name, param) else p_rec_wd).append(param)
            num_rec_parameters += param.data.nelement()
        else:
            (p_non_wd if _is_non_decay(name, param) else p_wd).append(param)
        num_parameters += param.data.nelement()

    # The first two groups are emitted unconditionally, even if empty, while the
    # rec groups are only added when populated. That asymmetry is deliberate:
    # ``Optimizer.load_state_dict`` matches groups positionally and rejects a
    # different group COUNT, so keeping the original two in place means an
    # optimizer state saved before this change still resumes. The rec groups are
    # new, and only exist once MF is trainable (tuning_step=3) — for steps 1/2
    # this function produces exactly the two groups it always did.
    optim_params = [
        {"params": p_wd, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": p_non_wd, "weight_decay": 0.0, "lr_scale": 1.0},
    ]
    if p_rec_wd:
        optim_params.append(
            {"params": p_rec_wd, "weight_decay": rec_weight_decay, "lr_scale": rec_lr_scale}
        )
    if p_rec_non_wd:
        optim_params.append(
            {"params": p_rec_non_wd, "weight_decay": 0.0, "lr_scale": rec_lr_scale}
        )
    if p_qf_wd:
        optim_params.append(
            {"params": p_qf_wd, "weight_decay": weight_decay, "lr_scale": qformer_lr_scale}
        )
    if p_qf_non_wd:
        # The gate lives here (ndim < 2 -> non-decay), which is the whole point.
        optim_params.append(
            {"params": p_qf_non_wd, "weight_decay": 0.0, "lr_scale": qformer_lr_scale}
        )

    logging.info("number of trainable parameters: %d", num_parameters)
    if num_qf_parameters:
        logging.info(
            "gated Q-Former is TRAINABLE: %d params in its own group "
            "(lr_scale=%s -> effective lr %s). With a zero-init gate the reachable "
            "|gate| after N Adam updates is ~N * effective_lr.",
            num_qf_parameters, qformer_lr_scale,
            float(config.run_cfg.init_lr) * qformer_lr_scale,
        )
    if num_rec_parameters:
        logging.info(
            "rec encoder is TRAINABLE: %d params in its own group "
            "(lr_scale=%s, weight_decay=%s)",
            num_rec_parameters,
            rec_lr_scale,
            rec_weight_decay,
        )

    beta2 = config.run_cfg.get("beta2", 0.999)
    # The top-level lr/weight_decay stay as AdamW defaults for any group that
    # does not override them; every group above sets weight_decay explicitly.
    return torch.optim.AdamW(
        optim_params,
        lr=float(config.run_cfg.init_lr),
        weight_decay=weight_decay,
        betas=(0.9, beta2),
    )
