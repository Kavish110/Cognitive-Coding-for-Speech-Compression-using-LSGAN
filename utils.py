import os
import random
import math
import yaml
import torch
import numpy as np
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional


# --------------------------
# Repro / device / params
# --------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For stricter determinism (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_requires_grad(modules: Iterable[torch.nn.Module], flag: bool):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


# --------------------------
# Config / checkpoint I/O
# --------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, **state):
    """
    save_checkpoint("checkpoints/codec.pt", dec=dec.state_dict(), step=123)
    """
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


# --------------------------
# Grad / mixed precision
# --------------------------

def clip_gradients(params, max_norm: float):
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


@contextmanager
def autocast(enabled: bool = True, dtype=torch.float16):
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=dtype):
            yield
    else:
        yield


# --------------------------
# Simple EMA
# --------------------------

class ExponentialMovingAverage:
    """
    Keeps a shadow copy of parameters: shadow = decay*shadow + (1-decay)*param
    Usage:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        for each step:
            ... optimize ...
            ema.update(model.parameters())
        # Swap to eval weights:
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        ... eval ...
        ema.restore(model.parameters())
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for p in params:
            if p.requires_grad:
                self.shadow[p] = p.data.clone()

    @torch.no_grad()
    def update(self, params: Iterable[torch.nn.Parameter]):
        for p in params:
            if p.requires_grad:
                assert p in self.shadow
                self.shadow[p].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, params: Iterable[torch.nn.Parameter]):
        for p in params:
            if p.requires_grad:
                p.data.copy_(self.shadow[p])

    def store(self, params: Iterable[torch.nn.Parameter]):
        self.backup = {p: p.data.clone() for p in params if p.requires_grad}

    def restore(self, params: Iterable[torch.nn.Parameter]):
        for p in params:
            if p.requires_grad:
                p.data.copy_(self.backup[p])
        self.backup = {}


# --------------------------
# Misc logging helpers
# --------------------------

def format_size(num_params: int) -> str:
    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    return str(num_params)


def lr_now(optim: torch.optim.Optimizer) -> float:
    for pg in optim.param_groups:
        return pg.get("lr", 0.0)
    return 0.0
