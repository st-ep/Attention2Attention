from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param


def _make_local_generator(seed: int, device: torch.device) -> torch.Generator:
    """
    Create a local torch.Generator on the target device for fast generation.

    Determinism requirement:
      - For a fixed cfg.seed and a fixed device, sampling is deterministic.
      - CPU tests rely on this; GPU runs remain deterministic per-device, but not necessarily
        bitwise identical to CPU (which is fine for our purposes).
    """
    try:
        g = torch.Generator(device=device)
    except Exception:
        # Fallback for older PyTorch / device strings: match device type only.
        g = torch.Generator(device=torch.device(device.type))
    g.manual_seed(int(seed))
    return g


def _permute_K(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Permute the K dimension of a tensor shaped (B,K,*,*).

    Args:
      t: Tensor (B,K,N,1) or (B,K,1,1) or similar with K at dim=1
      perm: Long tensor (B,K) where perm[b, i] is the source index for output position i.

    Returns:
      Tensor with permuted K dimension, same shape as input.
    """
    assert t.dim() >= 2 and perm.dim() == 2
    B, K = perm.shape
    assert t.shape[0] == B and t.shape[1] == K

    # Build expand shape for gather along dim=1.
    # Example: t (B,K,M,1) -> index (B,K,M,1)
    index = perm
    for _ in range(t.dim() - 2):
        index = index.unsqueeze(-1)
    index = index.expand_as(t)
    return t.gather(1, index)


def _permute_K_2d(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute a (B,K) tensor by a (B,K) permutation."""
    return t.gather(1, perm)


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 2: Mixture-of-Curvatures Quadratics (hidden clusters) with:
      - group-shared curvature a_g
      - per-function slope b_k and intercept c_k
      - within-group constraint sum_{k in group, non-outlier} b_k = 0
      - optional outliers with independent curvature and no constraint
      - left/right split: context at x={0, x_ctx_max}, queries x ~ U[x_qry_min, 1]

    Repo-wide latent convention:
      - Any per-function latent returned is aligned with the returned (permuted) function order.
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    if M < 2:
        raise ValueError("Dataset 2 requires M >= 2 (context includes x=0 and x=x_ctx_max).")
    dtype = torch.float32
    g = _make_local_generator(cfg.seed, device)

    # Core knobs
    G = int(get_param(cfg.params, "G", 3))
    x_ctx_max = float(get_param(cfg.params, "x_ctx_max", 0.2))
    x_qry_min = float(get_param(cfg.params, "x_qry_min", 0.6))
    if not (0.0 <= x_ctx_max < x_qry_min <= 1.0):
        raise ValueError(f"Expected 0 <= x_ctx_max < x_qry_min <= 1, got x_ctx_max={x_ctx_max}, x_qry_min={x_qry_min}")

    sigma_y = float(get_param(cfg.params, "sigma_y", 0.02))
    sigma_q = float(get_param(cfg.params, "sigma_q", 0.0))  # optional query noise
    sigma_a = float(get_param(cfg.params, "sigma_a", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 1.0))
    sigma_c = float(get_param(cfg.params, "sigma_c", 0.5))
    p_out = float(get_param(cfg.params, "p_out", 0.1))
    sigma_a_out = float(get_param(cfg.params, "sigma_a_out", sigma_a))

    # Robustness knobs
    min_non_out = int(get_param(cfg.params, "min_non_outliers_per_group", 2))
    max_tries = int(get_param(cfg.params, "max_outlier_resample_tries", 50))

    # Optional controlled imbalance (still guarantees >=2 functions per group).
    # group_imbalance=0 -> balanced.
    group_imbalance = float(get_param(cfg.params, "group_imbalance", 0.0))

    if K < 2 * G:
        raise ValueError(f"Dataset 2 requires K >= 2*G to ensure >=2 functions per group. Got K={K}, G={G}.")
    if min_non_out < 2:
        raise ValueError("min_non_outliers_per_group must be >= 2 for identifiability in Dataset 2.")

    # -------------------------------------------------------------------------
    # Group assignment (B,K), long in [0, G-1], with >=2 members/group guaranteed.
    # -------------------------------------------------------------------------
    if group_imbalance <= 0.0:
        # Balanced by construction: counts differ by at most 1.
        base = torch.arange(G, device=device, dtype=torch.long).repeat((K + G - 1) // G)[:K]  # (K,)
        # Randomize order per sample to avoid positional biases.
        perm_g = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)
        g_ids = base[None, :].expand(B, K).gather(1, perm_g)  # (B,K)
    else:
        # Controlled imbalance while enforcing >=2 per group.
        # Allocate 2 per group, distribute remaining R according to weights.
        min_count = 2
        R = K - min_count * G
        weights = torch.ones((G,), device=device, dtype=dtype)
        weights[0] = 1.0 + group_imbalance * float(G - 1)
        probs = (weights / weights.sum()).clamp(min=0.0)

        fixed = torch.arange(G, device=device, dtype=torch.long).repeat_interleave(min_count)  # (2G,)
        fixed = fixed[None, :].expand(B, fixed.numel())  # (B,2G)

        if R > 0:
            extra = torch.multinomial(probs.expand(B, G), num_samples=R, replacement=True, generator=g)  # (B,R)
            g_ids = torch.cat([fixed, extra], dim=1)  # (B,K)
        else:
            g_ids = fixed

        # Shuffle within each sample
        perm_g = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)
        g_ids = g_ids.gather(1, perm_g)

    # -------------------------------------------------------------------------
    # Outlier mask resampling to guarantee >=min_non_out non-outliers per group.
    # -------------------------------------------------------------------------
    outlier = (torch.rand((B, K), generator=g, device=device, dtype=dtype) < p_out)  # (B,K) bool

    def non_out_counts(mask_out: torch.Tensor) -> torch.Tensor:
        # Returns counts (B,G) of non-outliers per group using scatter_add.
        non_out = (~mask_out).to(dtype)  # (B,K) float
        cnt = torch.zeros((B, G), device=device, dtype=dtype)
        cnt = cnt.scatter_add(1, g_ids, non_out)
        return cnt

    cnt_non_out = non_out_counts(outlier)
    ok = (cnt_non_out >= float(min_non_out)).all(dim=1)  # (B,)

    for _ in range(max_tries):
        if bool(ok.all()):
            break
        need = ~ok  # (B,)
        # Resample only failing samples
        outlier_new = (torch.rand((int(need.sum().item()), K), generator=g, device=device, dtype=dtype) < p_out)
        outlier = outlier.clone()
        outlier[need] = outlier_new
        cnt_non_out = non_out_counts(outlier)
        ok = (cnt_non_out >= float(min_non_out)).all(dim=1)

    if not bool(ok.all()):
        # Clean fallback: disable outliers for failing samples (keeps spec satisfiable and avoids hangs).
        outlier = outlier.clone()
        outlier[~ok] = False

    # -------------------------------------------------------------------------
    # Sample latents and build functions
    # -------------------------------------------------------------------------
    # Group curvatures
    a_g = torch.randn((B, G, 1, 1), generator=g, device=device, dtype=dtype) * sigma_a  # (B,G,1,1)
    a_in = a_g.gather(1, g_ids[:, :, None, None].expand(-1, -1, 1, 1))  # (B,K,1,1)

    # Outlier curvatures (independent per outlier function)
    a_out = torch.randn((B, K, 1, 1), generator=g, device=device, dtype=dtype) * sigma_a_out
    a_k = torch.where(outlier[:, :, None, None], a_out, a_in)  # (B,K,1,1)

    # Slopes with within-group zero-sum over non-outliers
    b_raw = torch.randn((B, K), generator=g, device=device, dtype=dtype) * sigma_b  # (B,K)
    non_out = (~outlier).to(dtype)  # (B,K)

    sum_b = torch.zeros((B, G), device=device, dtype=dtype).scatter_add(1, g_ids, b_raw * non_out)
    cnt_b = torch.zeros((B, G), device=device, dtype=dtype).scatter_add(1, g_ids, non_out)
    mean_b = sum_b / cnt_b.clamp(min=1.0)  # (B,G)

    b_adj = b_raw - mean_b.gather(1, g_ids) * non_out  # (B,K)
    b_k = b_adj[:, :, None, None]  # (B,K,1,1)

    # Intercepts
    c_k = torch.randn((B, K, 1, 1), generator=g, device=device, dtype=dtype) * sigma_c  # (B,K,1,1)

    # -------------------------------------------------------------------------
    # Context/query sampling with left/right split
    # -------------------------------------------------------------------------
    # ctx_x: (B,K,M,1) where first two are fixed: x=0, x=x_ctx_max
    ctx_x = torch.empty((B, K, M, 1), device=device, dtype=dtype)
    ctx_x[:, :, 0, 0] = 0.0
    ctx_x[:, :, 1, 0] = x_ctx_max
    if M > 2:
        # Remaining points uniform in [0, x_ctx_max]
        ctx_x[:, :, 2:, 0] = x_ctx_max * torch.rand((B, K, M - 2), generator=g, device=device, dtype=dtype)

    # qry_x: (B,K,Q,1) uniform in [x_qry_min, 1]
    qry_x = x_qry_min + (1.0 - x_qry_min) * torch.rand((B, K, Q, 1), generator=g, device=device, dtype=dtype)

    # Evaluate quadratics
    ctx_y = a_k * ctx_x**2 + b_k * ctx_x + c_k  # (B,K,M,1)
    qry_y = a_k * qry_x**2 + b_k * qry_x + c_k  # (B,K,Q,1)

    if sigma_y > 0.0:
        ctx_y = ctx_y + sigma_y * torch.randn(ctx_y.shape, generator=g, device=device, dtype=dtype)
    if sigma_q > 0.0:
        qry_y = qry_y + sigma_q * torch.randn(qry_y.shape, generator=g, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Permute functions within each sample to enforce set structure
    # -------------------------------------------------------------------------
    perm = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)  # (B,K) long

    ctx_x = _permute_K(ctx_x, perm)
    ctx_y = _permute_K(ctx_y, perm)
    qry_x = _permute_K(qry_x, perm)
    qry_y = _permute_K(qry_y, perm)

    # Align per-function latents with returned order (repo-wide convention)
    g_ids_p = _permute_K_2d(g_ids, perm)  # (B,K)
    out_p = _permute_K_2d(outlier.to(torch.long), perm).to(torch.bool)  # (B,K)
    a_k_p = _permute_K(a_k, perm).squeeze(-1).squeeze(-1)  # (B,K)
    b_k_p = _permute_K(b_k, perm).squeeze(-1).squeeze(-1)  # (B,K)
    c_k_p = _permute_K(c_k, perm).squeeze(-1).squeeze(-1)  # (B,K)

    latents: Dict[str, Any] = {
        "g": g_ids_p,  # (B,K) group id aligned
        "outlier": out_p,  # (B,K) aligned
        "a_g": a_g.squeeze(-1).squeeze(-1),  # (B,G) group curvature (not per-function)
        "a_k": a_k_p,  # (B,K) aligned
        "b_k": b_k_p,  # (B,K) aligned
        "c_k": c_k_p,  # (B,K) aligned
        "x_ctx_max": torch.tensor(x_ctx_max, device=device, dtype=dtype),
        "x_qry_min": torch.tensor(x_qry_min, device=device, dtype=dtype),
    }

    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
