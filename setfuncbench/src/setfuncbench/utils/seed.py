from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set Python, NumPy, and PyTorch RNG seeds.

    Args:
      seed: Base seed for RNGs.
      deterministic: Debug-mode switch. If True, enables deterministic algorithms in PyTorch.
        This can reduce performance and may raise runtime errors on GPU for some ops/kernels.
        If False (default), uses GPU-friendly settings.

    Notes:
      - Our synthetic datasets are deterministic w.r.t. cfg.seed via a local torch.Generator.
      - This function mainly affects model initialization and any code using global RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism / performance toggles (global state)
    if deterministic:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not support toggling this; ignore.
            pass
    else:
        # Fast defaults
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def make_torch_generator(seed: int) -> torch.Generator:
    """
    Create a CPU generator seeded deterministically.

    We generate random tensors on CPU with this generator then move to target device.
    This makes determinism easier to reason about across devices.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


@dataclass(frozen=True)
class SeedSequence:
    """
    Deterministic seed stream:
    - start from a base seed
    - produce per-step seeds in a stable way
    """
    base_seed: int

    def seed_for_step(self, step: int, offset: int = 0) -> int:
        # Simple, explicit scheme. Good enough for synthetic data.
        return int(self.base_seed) + int(offset) + int(step)
