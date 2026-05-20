
# Trapezoidal LR multiplier: linear warmup, constant plateau, linear warmdown.
#
# Two calling conventions are supported:
#   - get_lr(step, num_step)
#       → no warmup; warmdown over the last `cooldown_frac` of training.
#   - get_lr(step, num_step, warmup_iters=W, warmdown_iters=D)
#       → matches the modded-nanogpt reference schedule (warmup + warmdown
#         in absolute iterations).
def get_lr(step, num_step, cooldown_frac=0.4,
           warmup_iters=0, warmdown_iters=None):
    if warmdown_iters is None:
        warmdown_iters = int(cooldown_frac * num_step)

    if warmup_iters > 0 and step < warmup_iters:
        return (step + 1) / warmup_iters
    if step < num_step - warmdown_iters:
        return 1.0
    return max(0.0, (num_step - step) / warmdown_iters)
