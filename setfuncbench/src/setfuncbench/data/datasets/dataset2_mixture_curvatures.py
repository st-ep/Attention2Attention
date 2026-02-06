from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param
from setfuncbench.utils.seed import make_torch_generator


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 2: Mixture-of-Curvatures Quadratics (hidden clusters) + left/right extrapolation.

    This file is intentionally minimal, but already produces correct-shaped tensors and is deterministic
    w.r.t. cfg.seed. It is also close to the agreed spec.

    TODO:
      - Add more rigorous outlier modes / additional knobs from the final dataset description.
      - Optionally add query noise, heteroscedasticity, and richer group-size distributions.
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    dtype = torch.float32
    g = make_torch_generator(cfg.seed)

    # Defaults (kept lightweight; override via cfg.params)
    G = int(get_param(cfg.params, "G", 3))
    x_ctx_max = float(get_param(cfg.params, "x_ctx_max", 0.2))
    x_qry_min = float(get_param(cfg.params, "x_qry_min", 0.6))

    sigma_y = float(get_param(cfg.params, "sigma_y", 0.02))
    sigma_a = float(get_param(cfg.params, "sigma_a", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 1.0))
    sigma_c = float(get_param(cfg.params, "sigma_c", 0.5))
    p_out = float(get_param(cfg.params, "p_out", 0.1))
    sigma_a_out = float(get_param(cfg.params, "sigma_a_out", sigma_a))

    # --- hidden groups (balanced assignment) ---
    base = torch.arange(G).repeat((K + G - 1) // G)[:K]          # (K,)
    g_ids = base[torch.argsort(torch.rand((B, K), generator=g), dim=1)]  # (B,K) long
    g_ids = g_ids.to(torch.long)

    # --- outlier mask ---
    outlier = (torch.rand((B, K), generator=g) < p_out)  # (B,K) bool

    # --- curvatures ---
    a_g = torch.randn((B, G, 1, 1), generator=g, dtype=dtype) * sigma_a  # (B,G,1,1)
    a_in = a_g.gather(1, g_ids[:, :, None, None].expand(-1, -1, 1, 1))   # (B,K,1,1)
    a_out = torch.randn((B, K, 1, 1), generator=g, dtype=dtype) * sigma_a_out
    a_k = torch.where(outlier[:, :, None, None], a_out, a_in)            # (B,K,1,1)

    # --- slopes with within-group zero-sum constraint for non-outliers ---
    b_raw = torch.randn((B, K), generator=g, dtype=dtype) * sigma_b      # (B,K)
    non_out = (~outlier).to(dtype)                                       # (B,K) float 0/1

    sum_b = torch.zeros((B, G), dtype=dtype)
    cnt_b = torch.zeros((B, G), dtype=dtype)
    sum_b = sum_b.scatter_add(1, g_ids, b_raw * non_out)
    cnt_b = cnt_b.scatter_add(1, g_ids, non_out)

    mean_b = sum_b / cnt_b.clamp(min=1.0)                                # (B,G)
    b_adj = b_raw - mean_b.gather(1, g_ids) * non_out                    # (B,K)
    b_k = b_adj[:, :, None, None]                                        # (B,K,1,1)

    # --- intercepts ---
    c_k = (torch.randn((B, K, 1, 1), generator=g, dtype=dtype) * sigma_c)  # (B,K,1,1)

    # --- context x: left-only ---
    # Spec uses M=2 with x={0, x_ctx_max}. We support M>=2 for convenience.
    ctx_x = torch.empty((B, K, M, 1), dtype=dtype)
    ctx_x[:, :, 0, 0] = 0.0
    if M >= 2:
        ctx_x[:, :, 1, 0] = x_ctx_max
    if M > 2:
        ctx_x[:, :, 2:, 0] = torch.rand((B, K, M - 2), generator=g, dtype=dtype) * x_ctx_max

    # --- query x: right-only ---
    qry_x = x_qry_min + (1.0 - x_qry_min) * torch.rand((B, K, Q, 1), generator=g, dtype=dtype)

    def f(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # a,b,c broadcast over last dims
        return a * x**2 + b * x + c

    ctx_y = f(a_k, b_k, c_k, ctx_x)
    if sigma_y > 0:
        ctx_y = ctx_y + sigma_y * torch.randn(ctx_y.shape, generator=g, dtype=dtype)

    qry_y = f(a_k, b_k, c_k, qry_x)  # usually noiseless targets

    # --- permute functions (set structure) ---
    perm = torch.argsort(torch.rand((B, K), generator=g, dtype=dtype), dim=1)

    def permute(t: torch.Tensor, N: int) -> torch.Tensor:
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
        "g": g_ids.gather(1, perm),            # (B,K)
        "outlier": outlier.gather(1, perm),    # (B,K)
        # Store minimal latents; add more if needed
    }
    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
