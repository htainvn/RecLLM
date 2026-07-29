import math

from sigllm.common.registry import registry


@registry.register_lr_scheduler("linear_warmup_cosine_lr")
class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=total_cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def _apply_lr(optimizer, lr):
    """Write ``lr`` to every param group, scaled by that group's ``lr_scale``.

    These schedulers overwrite ``param_group["lr"]`` outright on every step, so
    a per-group base LR set at construction time cannot survive: it is
    clobbered on the first step. ``lr_scale`` is the supported way to give one
    group a different LR — it is re-applied here on every write.

    Used by joint tuning (``tuning_step: 3``) to keep the MF encoder on a much
    smaller LR than the modules that read it (``run.rec_lr_scale``); groups
    without the key default to 1.0, which is the previous behaviour exactly.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    _apply_lr(optimizer, lr)


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    _apply_lr(optimizer, lr)
