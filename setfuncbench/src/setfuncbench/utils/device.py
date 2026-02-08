from __future__ import annotations


def default_device() -> str:
    """Return the preferred default compute device string."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
