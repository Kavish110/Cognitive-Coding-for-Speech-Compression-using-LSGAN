import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


# --------------------------
# Optimizer builders
# --------------------------

@dataclass
class OptimConfig:
    lr: float = 2.5e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 0.0
    eps: float = 1e-8


def build_adamw(params: Iterable, cfg: OptimConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, eps=cfg.eps)


# --------------------------
# Schedulers
# --------------------------

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup to base LR, then cosine decay to min_lr.
    Args:
        optimizer: torch optimizer
        warmup_steps: int
        total_steps: int
        min_lr: float (as absolute floor, not ratio)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # next step index
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                # linear warmup from 0 -> base_lr
                lr = base_lr * step / self.warmup_steps
            else:
                # cosine from base_lr -> min_lr
                progress = (step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
                cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cos
            lrs.append(lr)
        return lrs


class ConstantLR(torch.optim.lr_scheduler._LRScheduler):
    """Keeps LR fixed (wrapper so you can plug a 'scheduler' without branching)."""
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr for lr in self.base_lrs]


# --------------------------
# High-level helpers
# --------------------------

def build_optimizers_and_schedulers(
    g_params: Iterable,
    d_params: Iterable,
    base_lr: float = 2.5e-4,
    total_steps: Optional[int] = None,
    warmup_steps: int = 5000,
    min_lr: float = 1e-5,
    use_cosine: bool = True,
    betas=(0.8, 0.99),
    weight_decay: float = 0.0,
) -> Dict[str, object]:
    """
    Returns:
        {
          'opt_g': AdamW,
          'opt_d': AdamW,
          'sch_g': LRScheduler,
          'sch_d': LRScheduler
        }
    """
    g_opt = torch.optim.AdamW(g_params, lr=base_lr, betas=betas, weight_decay=weight_decay)
    d_opt = torch.optim.AdamW(d_params, lr=base_lr, betas=betas, weight_decay=weight_decay)

    if use_cosine and total_steps is not None:
        g_sch = WarmupCosine(g_opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
        d_sch = WarmupCosine(d_opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
    else:
        g_sch = ConstantLR(g_opt)
        d_sch = ConstantLR(d_opt)

    return {"opt_g": g_opt, "opt_d": d_opt, "sch_g": g_sch, "sch_d": d_sch}


def step_schedulers(schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler]):
    """Call .step() on any provided schedulers if present in dict."""
    for k in ("sch_g", "sch_d"):
        sch = schedulers.get(k)
        if sch is not None:
            sch.step()
