from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from setfuncbench.config import ModelConfig, get_param
from setfuncbench.models.baseline_a_no_talk import _mlp


class _SelfAttnBlock(nn.Module):
    """
    Minimal Transformer-style block over function tokens.
    Input/output: (B, K, D)
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        a, _ = self.attn(x, x, x, need_weights=False)  # (B,K,D)
        x = self.ln1(x + a)
        # FFN
        x = self.ln2(x + self.ff(x))
        return x


class BaselineCCommTransformer(nn.Module):
    """
    Baseline C: simple communication model via self-attention over functions.

    Structure:
      - per-function encoder -> token h_k
      - L layers MHSA/FFN over {h_k}
      - decoder uses [h_k^{(L)}, x_q]

    TODO:
      - Add richer Set Transformer modules (e.g., inducing points, cross-attention to context tokens).
      - Add optional recurrence for multi-hop tasks (Dataset 4) as an ablation.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = int(get_param(cfg.params, "hidden_dim", 128))
        num_layers = int(get_param(cfg.params, "num_layers", 2))
        num_heads = int(get_param(cfg.params, "num_heads", 4))
        enc_depth = int(get_param(cfg.params, "enc_depth", 2))
        dec_depth = int(get_param(cfg.params, "dec_depth", 2))

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.point_encoder = _mlp(in_dim=2, hidden_dim=hidden_dim, out_dim=hidden_dim, depth=enc_depth)
        self.comm = nn.ModuleList([_SelfAttnBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder = _mlp(in_dim=hidden_dim + 1, hidden_dim=hidden_dim, out_dim=1, depth=dec_depth)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx_x = batch["ctx_x"]  # (B,K,M,1)
        ctx_y = batch["ctx_y"]  # (B,K,M,1)
        qry_x = batch["qry_x"]  # (B,K,Q,1)

        ctx_feat = torch.cat([ctx_x, ctx_y], dim=-1)  # (B,K,M,2)
        B, K, M, _ = ctx_feat.shape
        Q = qry_x.shape[2]

        # Per-function token: mean over context points
        enc = self.point_encoder(ctx_feat.reshape(B * K * M, 2)).reshape(B, K, M, -1)
        h = enc.mean(dim=2)  # (B,K,D)

        # Communication across functions
        for blk in self.comm:
            h = blk(h)  # (B,K,D)

        # Decode queries
        hq = h[:, :, None, :].expand(B, K, Q, h.shape[-1])  # (B,K,Q,D)
        dec_in = torch.cat([hq, qry_x], dim=-1)            # (B,K,Q,D+1)
        pred = self.decoder(dec_in.reshape(B * K * Q, -1)).reshape(B, K, Q, 1)
        return pred
