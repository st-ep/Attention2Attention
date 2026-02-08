from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param


def _make_local_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a local torch.Generator on the target device."""
    try:
        g = torch.Generator(device=device)
    except Exception:
        g = torch.Generator(device=torch.device(device.type))
    g.manual_seed(int(seed))
    return g


def _permute_K(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute K dimension for a tensor whose dim-1 is K."""
    assert t.dim() >= 2 and perm.dim() == 2
    B, K = perm.shape
    assert t.shape[0] == B and t.shape[1] == K
    index = perm
    for _ in range(t.dim() - 2):
        index = index.unsqueeze(-1)
    index = index.expand_as(t)
    return t.gather(1, index)


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 1: shared quadratic core + per-function offsets.

    Latent convention (repo-wide):
      - Any per-function latent returned in `latents` is aligned with the returned (permuted) function order.
      - Global/sample-level latents are unaffected by permutation.

    Returns:
      ctx_x, ctx_y: (B,K,M,1)
      qry_x, qry_y: (B,K,Q,1)
      latents: eval-only dict
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    dtype = torch.float32
    g = _make_local_generator(cfg.seed, device)

    # Hyperparameters
    sigma_y = float(get_param(cfg.params, "sigma_y", 0.01))
    sigma_a = float(get_param(cfg.params, "sigma_a", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 1.0))
    sigma_c = float(get_param(cfg.params, "sigma_c", 0.5))

    # Latents
    a = torch.randn((B, 1, 1, 1), generator=g, device=device, dtype=dtype) * sigma_a
    b = torch.randn((B, 1, 1, 1), generator=g, device=device, dtype=dtype) * sigma_b
    c = torch.randn((B, K, 1, 1), generator=g, device=device, dtype=dtype) * sigma_c

    # Inputs in [-1, 1]
    ctx_x = 2.0 * torch.rand((B, K, M, 1), generator=g, device=device, dtype=dtype) - 1.0
    qry_x = 2.0 * torch.rand((B, K, Q, 1), generator=g, device=device, dtype=dtype) - 1.0

    # Outputs
    ctx_y = a * ctx_x**2 + b * ctx_x + c
    if sigma_y > 0:
        ctx_y = ctx_y + sigma_y * torch.randn(ctx_y.shape, generator=g, device=device, dtype=dtype)

    qry_y = a * qry_x**2 + b * qry_x + c

    # Permute functions within each sample to enforce set structure
    perm = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)
    ctx_x = _permute_K(ctx_x, perm)
    ctx_y = _permute_K(ctx_y, perm)
    qry_x = _permute_K(qry_x, perm)
    qry_y = _permute_K(qry_y, perm)

    # Align per-function latent c with returned order (repo-wide convention)
    c_perm = _permute_K(c, perm)

    latents = {
        "a": a.squeeze(-1).squeeze(-1).squeeze(-1),
        "b": b.squeeze(-1).squeeze(-1).squeeze(-1),
        "c": c_perm.squeeze(-1).squeeze(-1),
    }
    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
