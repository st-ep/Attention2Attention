from __future__ import annotations

from typing import Any, Dict, Literal

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


def _permute_K_2d(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return t.gather(1, perm)


def _make_inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    """
    perm: (B,K), where perm[b, i] is the old index placed at new position i.
    returns inv: (B,K), where inv[b, old] = new_position.
    """
    B, K = perm.shape
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(K, device=perm.device, dtype=perm.dtype)[None, :].expand(B, K))
    return inv


def _make_keys(
    B: int,
    K: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    g: torch.Generator,
    mode: Literal["linspace", "gaussian", "bucket"],
    sigma_u: float,
    key_jitter: float,
    num_key_buckets: int,
) -> torch.Tensor:
    """
    Key generation for Dataset 4.

    Clean-by-default behavior:
      - mode='linspace' produces well-separated keys by construction (shuffled linspace + tiny jitter).

    Explicit collision knobs:
      - mode='gaussian' uses N(0, sigma_u^2) keys (more chance of near-collisions)
      - mode='bucket' uses discrete buckets (intentional collisions)
    """
    if mode == "linspace":
        base = torch.linspace(-1.0, 1.0, K, device=device, dtype=dtype) * float(sigma_u)  # (K,)
        # Shuffle per sample
        perm_keys = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)  # (B,K)
        u = base[None, :].expand(B, K).gather(1, perm_keys)
        if key_jitter > 0.0:
            u = u + float(key_jitter) * float(sigma_u) * torch.randn((B, K), generator=g, device=device, dtype=dtype)
        return u

    if mode == "gaussian":
        return torch.randn((B, K), generator=g, device=device, dtype=dtype) * float(sigma_u)

    if mode == "bucket":
        nb = int(max(2, num_key_buckets))
        bucket_vals = torch.linspace(-1.0, 1.0, nb, device=device, dtype=dtype) * float(sigma_u)  # (nb,)
        idx = torch.randint(low=0, high=nb, size=(B, K), generator=g, device=device, dtype=torch.long)
        u = bucket_vals.gather(0, idx.reshape(-1)).reshape(B, K)
        if key_jitter > 0.0:
            u = u + float(key_jitter) * float(sigma_u) * torch.randn((B, K), generator=g, device=device, dtype=dtype)
        return u

    raise ValueError(f"Unknown keys_mode={mode}")


def sample_batch(cfg: DatasetConfig, device: torch.device) -> Dict[str, Any]:
    """
    Dataset 4: Key-Value Pointer Chasing.

    Context uses sentinel x-values {-3, -2, -1, 0} encoding:
      x=-3 -> own key u_k
      x=-2 -> own value v_k
      x=-1 -> successor key u_next(k)
      x= 0 -> intercept b_k

    Queries: x ~ U[0,1], target y = v_{next^H(k)} * x + b_k.

    Clean-by-default:
      - keys_mode='linspace' to avoid accidental near-collisions.

    Repo-wide latent convention:
      - per-function latents returned are aligned with returned function order.
      - we also return mapping info (next_index, t_index) in returned order for debugging.
    """
    B, K, M, Q = cfg.batch_size, cfg.K, cfg.M, cfg.Q
    if M != 4:
        raise ValueError("Dataset 4 requires M==4 (sentinel context x in {-3,-2,-1,0}).")
    dtype = torch.float32
    g = _make_local_generator(cfg.seed, device)

    # Core knobs
    H = int(get_param(cfg.params, "H", 2))
    if H < 0:
        raise ValueError("H must be >= 0")

    sigma_u = float(get_param(cfg.params, "sigma_u", 1.0))
    sigma_v = float(get_param(cfg.params, "sigma_v", 1.0))
    sigma_b = float(get_param(cfg.params, "sigma_b", 0.5))

    # Token noise knobs
    sigma_u_obs = float(get_param(cfg.params, "sigma_u_obs", 0.01))
    sigma_v_obs = float(get_param(cfg.params, "sigma_v_obs", 0.01))
    sigma_ptr = float(get_param(cfg.params, "sigma_ptr", 0.02))
    sigma_b_obs = float(get_param(cfg.params, "sigma_b_obs", 0.01))

    # Key generation knobs (clean by default)
    keys_mode = str(get_param(cfg.params, "keys_mode", "linspace"))
    key_jitter = float(get_param(cfg.params, "key_jitter", 1e-3))
    num_key_buckets = int(get_param(cfg.params, "num_key_buckets", max(2, K // 2)))

    # -------------------------------------------------------------------------
    # Sample keys/values/intercepts in the "original" order (index space 0..K-1)
    # -------------------------------------------------------------------------
    u = _make_keys(
        B,
        K,
        device=device,
        dtype=dtype,
        g=g,
        mode=keys_mode,  # type: ignore[arg-type]
        sigma_u=sigma_u,
        key_jitter=key_jitter,
        num_key_buckets=num_key_buckets,
    )  # (B,K)

    v = torch.randn((B, K), generator=g, device=device, dtype=dtype) * sigma_v  # (B,K)
    b0 = torch.randn((B, K), generator=g, device=device, dtype=dtype) * sigma_b  # (B,K)

    # -------------------------------------------------------------------------
    # Build a single-cycle successor mapping next(k) via a random permutation pi
    # -------------------------------------------------------------------------
    pi = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)  # (B,K)
    next_idx = torch.empty((B, K), device=device, dtype=torch.long)
    next_idx.scatter_(1, pi, torch.roll(pi, shifts=-1, dims=1))  # next(pi[i]) = pi[i+1]

    # H-hop target indices t = next^H(k)
    t = torch.arange(K, device=device, dtype=torch.long)[None, :].expand(B, K).clone()
    for _ in range(H):
        t = next_idx.gather(1, t)

    slope = v.gather(1, t)  # (B,K)

    # -------------------------------------------------------------------------
    # Build context tensors (sentinel x-values and token-encoded y-values)
    # -------------------------------------------------------------------------
    sentinel = torch.tensor([-3.0, -2.0, -1.0, 0.0], device=device, dtype=dtype)  # (4,)
    ctx_x = sentinel[None, None, :, None].expand(B, K, 4, 1).clone()  # (B,K,4,1)

    # Token encoding + noise
    y_key = u + sigma_u_obs * torch.randn((B, K), generator=g, device=device, dtype=dtype)
    y_val = v + sigma_v_obs * torch.randn((B, K), generator=g, device=device, dtype=dtype)
    u_next = u.gather(1, next_idx)
    y_ptr = u_next + sigma_ptr * torch.randn((B, K), generator=g, device=device, dtype=dtype)
    y_b = b0 + sigma_b_obs * torch.randn((B, K), generator=g, device=device, dtype=dtype)

    # ctx_y stacked in the same order as sentinel x values
    ctx_y = torch.stack([y_key, y_val, y_ptr, y_b], dim=2)[..., None]  # (B,K,4,1)

    # -------------------------------------------------------------------------
    # Queries on [0,1]
    # -------------------------------------------------------------------------
    qry_x = torch.rand((B, K, Q, 1), generator=g, device=device, dtype=dtype)  # (B,K,Q,1)
    qry_y = slope[:, :, None, None] * qry_x + b0[:, :, None, None]  # (B,K,Q,1)

    # -------------------------------------------------------------------------
    # Final permutation to present as an unordered set
    # -------------------------------------------------------------------------
    perm = torch.argsort(torch.rand((B, K), generator=g, device=device, dtype=dtype), dim=1)  # (B,K)
    inv_perm = _make_inverse_perm(perm)  # (B,K), inv_perm[old] = new_pos

    # Permute main tensors
    ctx_x = _permute_K(ctx_x, perm)
    ctx_y = _permute_K(ctx_y, perm)
    qry_x = _permute_K(qry_x, perm)
    qry_y = _permute_K(qry_y, perm)

    # Align per-function latents to returned order
    u_p = _permute_K_2d(u, perm)  # (B,K)
    v_p = _permute_K_2d(v, perm)  # (B,K)
    b_p = _permute_K_2d(b0, perm)  # (B,K)
    slope_p = _permute_K_2d(slope, perm)  # (B,K)

    # Mapping info in returned order:
    # For each new position i (old index perm[i]), compute successor/t target positions.
    old_idx_for_new = perm  # (B,K) old index at each new position
    old_next = next_idx.gather(1, old_idx_for_new)  # (B,K) old successor index for each new position
    next_index_new = inv_perm.gather(1, old_next)  # (B,K) successor position in returned order

    old_t_for_new = t.gather(1, old_idx_for_new)  # (B,K) old H-hop target for each new position
    t_index_new = inv_perm.gather(1, old_t_for_new)  # (B,K) H-hop target position in returned order

    latents: Dict[str, Any] = {
        "u": u_p,  # (B,K) keys aligned
        "v": v_p,  # (B,K) values aligned
        "b": b_p,  # (B,K) intercepts aligned
        "slope": slope_p,  # (B,K) slope aligned
        "next_index": next_index_new,  # (B,K) successor index in returned order
        "t_index": t_index_new,  # (B,K) H-hop target index in returned order
        "H": H,
        "keys_mode": keys_mode,
    }

    return {"ctx_x": ctx_x, "ctx_y": ctx_y, "qry_x": qry_x, "qry_y": qry_y, "latents": latents}
