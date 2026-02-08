from __future__ import annotations

from typing import Any, Dict

import torch

from setfuncbench.config import DatasetConfig, get_param


def _make_local_generator(seed: int, device: torch.device) -> torch.Generator:
    try:
        g = torch.Generator(device=device)
    except Exception:
        g = torch.Generator(device=torch.device(device.type))
    g.manual_seed(int(seed))
    return g


def _permute_K(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
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
    Dataset 3: Hidden pairing (group size 2) quadratics.

    - K must be even.
    - Pairing is hidden; only the set contains the information needed to disambiguate.
    - Permutation invariance is enforced by a final random permutation along K.

    Repo-wide latent convention:
      - per-function latents (e.g., pair_id) are aligned with returned function order.
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    if K % 2 != 0:
        raise ValueError(f"Dataset 3 requires even K. Got K={K}.")
    dtype = torch.float32
    g = _make_local_generator(cfg.seed, device)

    sigma_y = float(get_param(cfg.params, "sigma_y", 0.02))
    sigma_a = float(get_param(cfg.params, "sigma_a", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 1.0))
    sigma_c = float(get_param(cfg.params, "sigma_c", 0.5))
    sigma_q = float(get_param(cfg.params, "sigma_q", 0.0))  # optional query noise

    P = K // 2

    # -------------------------------------------------------------------------
    # Build a random pairing via a random permutation perm0 and adjacent pairing
    # -------------------------------------------------------------------------
    perm0 = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)  # (B,K)

    # Pair ids in perm-space: [0,0,1,1,...,P-1,P-1], shape (K,)
    pair_ids_in_perm = torch.arange(P, device=device, dtype=torch.long).repeat_interleave(2)  # (K,)
    pair_ids_in_perm = pair_ids_in_perm[None, :].expand(B, K)  # (B,K)

    # Scatter to original index space: pair_id[b, perm0[b, i]] = pair_ids_in_perm[b, i]
    pair_id = torch.empty((B, K), device=device, dtype=torch.long)
    pair_id.scatter_(1, perm0, pair_ids_in_perm)  # (B,K) pair id for each function index

    # -------------------------------------------------------------------------
    # Sample pair-level and function-level latents
    # -------------------------------------------------------------------------
    a_p = torch.randn((B, P, 1, 1), generator=g, device=device, dtype=dtype) * sigma_a  # (B,P,1,1)
    b_p = torch.randn((B, P, 1, 1), generator=g, device=device, dtype=dtype) * sigma_b  # (B,P,1,1)
    c_k = torch.randn((B, K, 1, 1), generator=g, device=device, dtype=dtype) * sigma_c  # (B,K,1,1)

    # Gather shared pair params per function: (B,K,1,1)
    a_k = a_p.gather(1, pair_id[:, :, None, None].expand(-1, -1, 1, 1))
    b_k = b_p.gather(1, pair_id[:, :, None, None].expand(-1, -1, 1, 1))

    # -------------------------------------------------------------------------
    # Sample context/query x and generate y
    # -------------------------------------------------------------------------
    ctx_x = 2.0 * torch.rand((B, K, M, 1), generator=g, device=device, dtype=dtype) - 1.0  # [-1,1]
    qry_x = 2.0 * torch.rand((B, K, Q, 1), generator=g, device=device, dtype=dtype) - 1.0  # [-1,1]

    ctx_y = a_k * ctx_x**2 + b_k * ctx_x + c_k
    qry_y = a_k * qry_x**2 + b_k * qry_x + c_k

    if sigma_y > 0.0:
        ctx_y = ctx_y + sigma_y * torch.randn(ctx_y.shape, generator=g, device=device, dtype=dtype)
    if sigma_q > 0.0:
        qry_y = qry_y + sigma_q * torch.randn(qry_y.shape, generator=g, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Final permutation to present as an unordered set (hides pairing)
    # -------------------------------------------------------------------------
    perm = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)  # (B,K)

    ctx_x = _permute_K(ctx_x, perm)
    ctx_y = _permute_K(ctx_y, perm)
    qry_x = _permute_K(qry_x, perm)
    qry_y = _permute_K(qry_y, perm)

    # Align per-function latents with returned order
    pair_id_p = pair_id.gather(1, perm)  # (B,K)
    a_k_p = _permute_K(a_k, perm).squeeze(-1).squeeze(-1)  # (B,K)
    b_k_p = _permute_K(b_k, perm).squeeze(-1).squeeze(-1)  # (B,K)
    c_k_p = _permute_K(c_k, perm).squeeze(-1).squeeze(-1)  # (B,K)

    latents: Dict[str, Any] = {
        "pair_id": pair_id_p,  # (B,K) aligned
        "a_k": a_k_p,  # (B,K) aligned
        "b_k": b_k_p,  # (B,K) aligned
        "c_k": c_k_p,  # (B,K) aligned
    }

    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
