from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from setfuncbench.config import DatasetConfig

# Import dataset entrypoints
from setfuncbench.data.datasets.dataset1_shared_quadratic import sample_batch as _sample_d1
from setfuncbench.data.datasets.dataset2_mixture_curvatures import sample_batch as _sample_d2
from setfuncbench.data.datasets.dataset3_hidden_pairing import sample_batch as _sample_d3
from setfuncbench.data.datasets.dataset4_pointer_chasing import sample_batch as _sample_d4


Batch = Dict[str, torch.Tensor]
DatasetFn = Callable[[DatasetConfig, torch.device], Batch]


DATASET_REGISTRY: Dict[str, DatasetFn] = {
    "dataset1_shared_quadratic": _sample_d1,
    "dataset2_mixture_curvatures": _sample_d2,
    "dataset3_hidden_pairing": _sample_d3,
    "dataset4_pointer_chasing": _sample_d4,
}


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def _assert_batch(batch: Dict[str, Any], cfg: DatasetConfig) -> None:
    required = ["ctx_x", "ctx_y", "qry_x", "qry_y"]
    for k in required:
        if k not in batch:
            raise KeyError(f"Batch missing required key '{k}'")

    ctx_x = batch["ctx_x"]
    ctx_y = batch["ctx_y"]
    qry_x = batch["qry_x"]
    qry_y = batch["qry_y"]

    # Shape assertions
    B, K, M, one = ctx_x.shape
    assert (B, K, M, one) == (cfg.batch_size, cfg.K, cfg.M, 1), f"ctx_x shape {ctx_x.shape} != {(cfg.batch_size, cfg.K, cfg.M, 1)}"
    assert ctx_y.shape == (cfg.batch_size, cfg.K, cfg.M, 1)
    assert qry_x.shape == (cfg.batch_size, cfg.K, cfg.Q, 1)
    assert qry_y.shape == (cfg.batch_size, cfg.K, cfg.Q, 1)

    # Dtype conventions
    assert ctx_x.dtype == torch.float32
    assert ctx_y.dtype == torch.float32
    assert qry_x.dtype == torch.float32
    assert qry_y.dtype == torch.float32

    # Device conventions: all main tensors on same device
    dev = ctx_x.device
    assert ctx_y.device == dev and qry_x.device == dev and qry_y.device == dev


def sample_batch(cfg: DatasetConfig, device: torch.device | str) -> Batch:
    """
    Unified dataset API.

    Returns dict with tensors:
      ctx_x, ctx_y: (B,K,M,1)
      qry_x, qry_y: (B,K,Q,1)
    plus optional eval-only latents.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if cfg.name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{cfg.name}'. Available: {list_datasets()}")

    batch = DATASET_REGISTRY[cfg.name](cfg, device)
    _assert_batch(batch, cfg)
    return batch
