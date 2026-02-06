from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param
from setfuncbench.utils.seed import make_torch_generator


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 3: hidden pairing (group size 2).

    TODO:
      - Vectorize pairing assignment (currently uses a small Python loop over batch for clarity).
      - Add additional difficulty knobs (pair collisions, etc.) via cfg.params.
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    assert K % 2 == 0, "Dataset 3 requires even K."
    dtype = torch.float32
    g = make_torch_generator(cfg.seed)

    sigma_y = float(get_param(cfg.params, "sigma_y", 0.02))
    sigma_a = float(get_param(cfg.params, "sigma_a", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 1.0))
    sigma_c = float(get_param(cfg.params, "sigma_c", 0.5))

    P = K // 2

    # Random pairing per sample via permutation
    perm0 = torch.argsort(torch.rand((B, K), generator=g, dtype=dtype), dim=1)  # (B,K)
    pair_id = torch.zeros((B, K), dtype=torch.long)

    for b in range(B):
        for p in range(P):
            i = int(perm0[b, 2 * p].item())
            j = int(perm0[b, 2 * p + 1].item())
            pair_id[b, i] = p
            pair_id[b, j] = p

    a_p = torch.randn((B, P, 1, 1), generator=g, dtype=dtype) * sigma_a
    b_p = torch.randn((B, P, 1, 1), generator=g, dtype=dtype) * sigma_b
    c_k = torch.randn((B, K, 1, 1), generator=g, dtype=dtype) * sigma_c

    a = a_p.gather(1, pair_id[:, :, None, None].expand(-1, -1, 1, 1))  # (B,K,1,1)
    b_ = b_p.gather(1, pair_id[:, :, None, None].expand(-1, -1, 1, 1)) # (B,K,1,1)

    ctx_x = (2.0 * torch.rand((B, K, M, 1), generator=g, dtype=dtype) - 1.0)
    qry_x = (2.0 * torch.rand((B, K, Q, 1), generator=g, dtype=dtype) - 1.0)

    ctx_y = a * ctx_x**2 + b_ * ctx_x + c_k
    if sigma_y > 0:
        ctx_y = ctx_y + sigma_y * torch.randn(ctx_y.shape, generator=g, dtype=dtype)

    qry_y = a * qry_x**2 + b_ * qry_x + c_k

    # Set permutation
    perm = torch.argsort(torch.rand((B, K), generator=g, dtype=dtype), dim=1)

    def permute(t: torch.Tensor, N: int) -> torch.Tensor:
        return t.gather(1, perm[:, :, None, None].expand(-1, -1, N, 1))

    ctx_x = permute(ctx_x, M)
    ctx_y = permute(ctx_y, M)
    qry_x = permute(qry_x, Q)
    qry_y = permute(qry_y, Q)
    pair_id = pair_id.gather(1, perm)

    # Move to device
    ctx_x = ctx_x.to(device)
    ctx_y = ctx_y.to(device)
    qry_x = qry_x.to(device)
    qry_y = qry_y.to(device)

    latents = {"pair_id": pair_id}  # (B,K)
    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
