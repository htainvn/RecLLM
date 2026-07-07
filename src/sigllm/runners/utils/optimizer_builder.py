import torch
import logging


def _is_qformer_bridge_param(name: str) -> bool:
    """True for the Q-Former + projection ("bridge") parameters.

    These are the only trainable modules in CoLLM Step 2, and the gradient that
    reaches them through the frozen LLM (from a 1-bit Yes/No loss) is weak.
    The rank head (CHANGE Q2) rides with the bridge: it is trained by the same
    auxiliary loss and should follow the same lr multiplier.
    """
    return (
        name.startswith("qformer.")
        or "llm_proj" in name
        or "cf_injector" in name
        or "rank_head" in name
        or "target_id_head" in name
    )


def build_optimizer(model, config):
    """
    Build optimizer from config.

    CHANGE 2i: give the Q-Former/projection bridge its own parameter group with
    ``lr = init_lr * qformer_lr_mult`` (default 1.0 = unchanged). Under a frozen
    LLM the bridge receives a weak gradient, so a larger step on these params
    (paired with the title-free prompt, CHANGE 2a, which routes the loss through
    the CF tokens) helps the Q-Former actually move. Weight decay is still
    disabled for biases / 1-D params in every group.
    """
    qformer_lr_mult = float(config.run_cfg.get("qformer_lr_mult", 1.0))
    base_lr = float(config.run_cfg.init_lr)
    weight_decay = float(config.run_cfg.weight_decay)

    # 4 buckets: {bridge, other} x {weight-decay, no-decay}
    groups = {
        "bridge_wd": [], "bridge_nowd": [],
        "other_wd": [], "other_nowd": [],
    }
    num_parameters = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        no_wd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
        bridge = _is_qformer_bridge_param(n)
        key = f"{'bridge' if bridge else 'other'}_{'nowd' if no_wd else 'wd'}"
        groups[key].append(p)
        num_parameters += p.data.nelement()

    logging.info("number of trainable parameters: %d" % num_parameters)
    if qformer_lr_mult != 1.0:
        n_bridge = sum(p.numel() for k in ("bridge_wd", "bridge_nowd") for p in groups[k])
        logging.info(
            "qformer_lr_mult=%.3f -> bridge lr=%.3e on %d params",
            qformer_lr_mult, base_lr * qformer_lr_mult, n_bridge,
        )

    # ``lr_scale`` is honored by the LR scheduler (see common/optims.py) so the
    # bridge multiplier persists across warmup/cosine steps.
    optim_params = [
        {"params": groups["other_wd"], "weight_decay": weight_decay, "lr": base_lr, "lr_scale": 1.0},
        {"params": groups["other_nowd"], "weight_decay": 0.0, "lr": base_lr, "lr_scale": 1.0},
        {"params": groups["bridge_wd"], "weight_decay": weight_decay,
         "lr": base_lr * qformer_lr_mult, "lr_scale": qformer_lr_mult},
        {"params": groups["bridge_nowd"], "weight_decay": 0.0,
         "lr": base_lr * qformer_lr_mult, "lr_scale": qformer_lr_mult},
    ]
    # Drop empty groups so schedulers that iterate param_groups stay clean.
    optim_params = [g for g in optim_params if len(g["params"]) > 0]

    beta2 = config.run_cfg.get("beta2", 0.999)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, beta2),
    )

    return optimizer
