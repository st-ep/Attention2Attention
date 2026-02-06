from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param
from setfuncbench.utils.seed import make_torch_generator


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 1: shared quadratic core + per-function offsets.

    Returns:
      ctx_x, ctx_y: (B,K,M,1)
      qry_x, qry_y: (B,K,Q,1)
      latents: eval-only dict
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    dtype = torch.float32
    g = make_torch_generator(cfg.seed)

    # Hyperparameters (defaults match earlier discussion)
    sigma_y = float(get_param(cfg.params, "sigma_y", 0.01))
    sigma_a = float(get_param(cfg.params, "sigma_a", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 1.0))
    sigma_c = float(get_param(cfg.params, "sigma_c", 0.5))

    # --- latents (CPU) ---
    a = torch.randn((B, 1, 1, 1), generator=g, dtype=dtype) * sigma_a
    b = torch.randn((B, 1, 1, 1), generator=g, dtype=dtype) * sigma_b
    c = torch.randn((B, K, 1, 1), generator=g, dtype=dtype) * sigma_c

    # --- inputs (CPU) ---
    # ctx_x: (B,K,M,1) in [-1,1]
    ctx_x = (2.0 * torch.rand((B, K, M, 1), generator=g, dtype=dtype) - 1.0)
    # qry_x: (B,K,Q,1) in [-1,1]
    qry_x = (2.0 * torch.rand((B, K, Q, 1), generator=g, dtype=dtype) - 1.0)

    # --- outputs (CPU) ---
    ctx_y = a * ctx_x**2 + b * ctx_x + c
    if sigma_y > 0:
        ctx_y = ctx_y + sigma_y * torch.randn(ctx_y.shape, generator=g, dtype=dtype)

    qry_y = a * qry_x**2 + b * qry_x + c  # usually noiseless targets

    # --- permute functions within each sample to enforce set structure ---
    perm = torch.argsort(torch.rand((B, K), generator=g, dtype=dtype), dim=1)  # (B,K) long indices

    def permute(t: torch.Tensor, N: int) -> torch.Tensor:
        # t: (B,K,N,1)
        return t.gather(1, perm[:, :, None, None].expand(-1, -1, N, 1))

    ctx_x = permute(ctx_x, M)
    ctx_y = permute(ctx_y, M)
    qry_x = permute(qry_x, Q)
    qry_y = permute(qry_y, Q)

    # Move to device
    ctx_x = ctx_x.to(device)
    ctx_y = ctx_y.to(device)
    qry_x = qry_x.to(device)
    qry_y = qry_y.to(device)

    latents = {
        "a": a.squeeze(-1).squeeze(-1).squeeze(-1),  # (B,)
        "b": b.squeeze(-1).squeeze(-1).squeeze(-1),  # (B,)
        "c": c.squeeze(-1).squeeze(-1),              # (B,K)
    }
    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
