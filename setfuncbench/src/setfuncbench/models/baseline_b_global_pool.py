from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from setfuncbench.config import ModelConfig, get_param
from setfuncbench.models.baseline_a_no_talk import _mlp


class BaselineBGlobalPool(nn.Module):
    """
    Baseline B: global (non-selective) pooling communication.

    Structure:
      - per-function encoder -> h_k
      - global pooled embedding g = pool_k(h_k) where pool is mean or sum
      - decoder uses [h_k, g, x_q]

    This baseline is intentionally non-selective; it should help on globally-shared-latent problems,
    but degrade on selective communication datasets.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = int(get_param(cfg.params, "hidden_dim", 128))
        enc_depth = int(get_param(cfg.params, "enc_depth", 2))
        dec_depth = int(get_param(cfg.params, "dec_depth", 2))

        pool = str(get_param(cfg.params, "pool", "mean")).lower()
        if pool not in ("mean", "sum"):
            raise ValueError("BaselineBGlobalPool param 'pool' must be 'mean' or 'sum'.")
        self.pool = pool

        # Point encoder: (x,y) -> hidden
        self.point_encoder = _mlp(in_dim=2, hidden_dim=hidden_dim, out_dim=hidden_dim, depth=enc_depth)
        # Decoder: [h_k, g, x_q] -> y_q
        self.decoder = _mlp(in_dim=(hidden_dim + hidden_dim + 1), hidden_dim=hidden_dim, out_dim=1, depth=dec_depth)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx_x = batch["ctx_x"]  # (B,K,M,1)
        ctx_y = batch["ctx_y"]  # (B,K,M,1)
        qry_x = batch["qry_x"]  # (B,K,Q,1)

        # ctx_feat: (B,K,M,2)
        ctx_feat = torch.cat([ctx_x, ctx_y], dim=-1)

        B, K, M, _ = ctx_feat.shape
        Q = qry_x.shape[2]

        # Encode points -> (B,K,M,H)
        enc = self.point_encoder(ctx_feat.reshape(B * K * M, 2)).reshape(B, K, M, -1)
        # Per-function token -> (B,K,H)
        h = enc.mean(dim=2)

        # Global pooling -> (B,1,H)
        if self.pool == "mean":
            g = h.mean(dim=1, keepdim=True)
        else:  # sum
            g = h.sum(dim=1, keepdim=True)

        # Decode queries -> (B,K,Q,1)
        hq = h[:, :, None, :].expand(B, K, Q, h.shape[-1])  # (B,K,Q,H)
        gq = g[:, :, None, :].expand(B, K, Q, g.shape[-1])  # (B,K,Q,H)
        dec_in = torch.cat([hq, gq, qry_x], dim=-1)  # (B,K,Q,2H+1)

        pred = self.decoder(dec_in.reshape(B * K * Q, -1)).reshape(B, K, Q, 1)
        return pred
