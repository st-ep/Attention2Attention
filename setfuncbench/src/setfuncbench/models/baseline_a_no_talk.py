from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from setfuncbench.config import ModelConfig, get_param


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    """
    Simple MLP with ReLU.
    depth=2 => Linear->ReLU->Linear
    """
    layers = []
    d = in_dim
    for _ in range(max(depth - 1, 1)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class BaselineANoTalk(nn.Module):
    """
    Baseline A: independent per-function model (no cross-function communication).

    API:
      forward(batch) -> pred_qry_y of shape (B,K,Q,1)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = int(get_param(cfg.params, "hidden_dim", 128))
        enc_depth = int(get_param(cfg.params, "enc_depth", 2))
        dec_depth = int(get_param(cfg.params, "dec_depth", 2))

        # Point encoder: (x,y) -> hidden
        self.point_encoder = _mlp(in_dim=2, hidden_dim=hidden_dim, out_dim=hidden_dim, depth=enc_depth)
        # Decoder: [h_k, x_q] -> y_q
        self.decoder = _mlp(in_dim=hidden_dim + 1, hidden_dim=hidden_dim, out_dim=1, depth=dec_depth)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx_x = batch["ctx_x"]  # (B,K,M,1)
        ctx_y = batch["ctx_y"]  # (B,K,M,1)
        qry_x = batch["qry_x"]  # (B,K,Q,1)

        # Build per-point features
        # ctx_feat: (B,K,M,2)
        ctx_feat = torch.cat([ctx_x, ctx_y], dim=-1)

        B, K, M, _ = ctx_feat.shape
        Q = qry_x.shape[2]

        # Encode points independently
        # Flatten: (B*K*M, 2) -> (B*K*M, hidden)
        ctx_flat = ctx_feat.reshape(B * K * M, 2)
        enc_flat = self.point_encoder(ctx_flat)  # (B*K*M, hidden)
        enc = enc_flat.reshape(B, K, M, -1)      # (B,K,M,hidden)

        # Pool over context points to get per-function embedding h_k
        h = enc.mean(dim=2)  # (B,K,hidden)

        # Decode queries
        # Expand h to (B,K,Q,hidden)
        hq = h[:, :, None, :].expand(B, K, Q, h.shape[-1])
        dec_in = torch.cat([hq, qry_x], dim=-1)  # (B,K,Q,hidden+1)
        pred = self.decoder(dec_in.reshape(B * K * Q, -1)).reshape(B, K, Q, 1)
        return pred
