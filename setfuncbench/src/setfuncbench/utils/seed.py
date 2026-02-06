from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set Python, NumPy, and PyTorch RNG seeds.

    Notes:
    - This affects model init and any code using global RNG.
    - Datasets in this repo are deterministic w.r.t. cfg.seed by using a local torch.Generator,
      but we still set global seed for reproducible model init and training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Safe defaults; some ops may be nondeterministic on GPU depending on your setup.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not support this; ignore.
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
