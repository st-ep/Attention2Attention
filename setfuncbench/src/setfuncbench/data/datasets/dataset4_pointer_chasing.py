from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param
from setfuncbench.utils.seed import make_torch_generator


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 4: Key–Value Pointer Chasing (single-cycle, H-hop retrieval).

    NOTE:
      - The agreed dataset uses M=4 sentinel context x-values {-3,-2,-1,0}.
      - We enforce M==4 here for clarity.

    TODO:
      - Add optional distractors / near-collisions and stronger noise modes.
      - Provide perm-consistent next/t indices if needed for deeper debugging.
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    assert M == 4, "Dataset 4 uses exactly M=4 sentinel context points."
    dtype = torch.float32
    g = make_torch_generator(cfg.seed)

    H = int(get_param(cfg.params, "H", 2))

    sigma_u = float(get_param(cfg.params, "sigma_u", 1.0))
    sigma_v = float(get_param(cfg.params, "sigma_v", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 0.5))

    sigma_u_obs = float(get_param(cfg.params, "sigma_u_obs", 0.01))
    sigma_v_obs = float(get_param(cfg.params, "sigma_v_obs", 0.01))
    sigma_ptr = float(get_param(cfg.params, "sigma_ptr", 0.02))
    sigma_b_obs = float(get_param(cfg.params, "sigma_b_obs", 0.01))

    # Keys/values/intercepts
    u = torch.randn((B, K), generator=g, dtype=dtype) * sigma_u
    v = torch.randn((B, K), generator=g, dtype=dtype) * sigma_v
    b0 = torch.randn((B, K), generator=g, dtype=dtype) * sigma_b

    # Single-cycle successor mapping via random permutation pi
    pi = torch.argsort(torch.rand((B, K), generator=g, dtype=dtype), dim=1)  # (B,K)
    next_idx = torch.empty((B, K), dtype=torch.long)
    next_idx.scatter_(1, pi, torch.roll(pi, shifts=-1, dims=1))  # next(pi[i]) = pi[i+1]

    # H-hop target
    t = torch.arange(K, dtype=torch.long)[None, :].expand(B, K).clone()
    for _ in range(H):
        t = next_idx.gather(1, t)

    slope = v.gather(1, t)  # (B,K)

    # Context x: sentinel values
    sentinel = torch.tensor([-3.0, -2.0, -1.0, 0.0], dtype=dtype)
    ctx_x = sentinel[None, None, :, None].expand(B, K, 4, 1).clone()  # (B,K,4,1)

    # Context y encodes: own key, own value, successor key, intercept
    y_key = u + sigma_u_obs * torch.randn(u.shape, generator=g, dtype=dtype)
    y_val = v + sigma_v_obs * torch.randn(v.shape, generator=g, dtype=dtype)
    u_next = u.gather(1, next_idx)
    y_ptr = u_next + sigma_ptr * torch.randn(u_next.shape, generator=g, dtype=dtype)
    y_b = b0 + sigma_b_obs * torch.randn(b0.shape, generator=g, dtype=dtype)

    ctx_y = torch.stack([y_key, y_val, y_ptr, y_b], dim=2)[..., None]  # (B,K,4,1)

    # Queries
    qry_x = torch.rand((B, K, Q, 1), generator=g, dtype=dtype)
    qry_y = slope[:, :, None, None] * qry_x + b0[:, :, None, None]     # (B,K,Q,1)

    # Permute functions (set)
    perm = torch.argsort(torch.rand((B, K), generator=g, dtype=dtype), dim=1)

    def permute(tensor: torch.Tensor, N: int) -> torch.Tensor:
        return tensor.gather(1, perm[:, :, None, None].expand(-1, -1, N, 1))

    ctx_x = permute(ctx_x, 4)
    ctx_y = permute(ctx_y, 4)
    qry_x = permute(qry_x, Q)
    qry_y = permute(qry_y, Q)

    # Move to device
    ctx_x = ctx_x.to(device)
    ctx_y = ctx_y.to(device)
    qry_x = qry_x.to(device)
    qry_y = qry_y.to(device)

    latents = {
        "slope": slope.gather(1, perm),  # (B,K) eval-only convenience
        "H": H,
    }
    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
