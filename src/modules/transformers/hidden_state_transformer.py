import math
from typing import List
import torch
import torch.nn as nn


class _SelfAttentionHead(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int, dropout: float) -> None:
        super().__init__()
        self.q = nn.Linear(input_dim, attn_dim)
        self.k = nn.Linear(input_dim, attn_dim)
        self.v = nn.Linear(input_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)          # [B,N,attn_dim]
        attn = torch.softmax(q @ k.transpose(1, 2) / self.scale, dim=-1)  # [B,N,N]
        attn = self.dropout(attn)
        return attn @ v                                     # [B,N,attn_dim]


class _TransformerBlock(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int, n_heads: int,
                 ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            _SelfAttentionHead(input_dim, attn_dim, dropout) for _ in range(n_heads)
        ])
        self.attn_dropout = nn.Dropout(dropout)

        # (중요) 멀티헤드 합산 -> input_dim으로 사상해 residual에 더함
        self.out_proj = nn.Linear(attn_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        ff_hidden = input_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # heads: list of [B,N,attn_dim] -> sum: [B,N,attn_dim]
        attn_sum = torch.stack([h(x) for h in self.heads], dim=0).sum(dim=0)
        attn_out = self.out_proj(attn_sum)                  # [B,N,input_dim]
        x = self.norm1(x + self.attn_dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class HiddenStateTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 rect_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 1,
                 attn_dim: int = 64,
                 ff_multiplier: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if n_heads < 1 or n_layers < 1:
            raise ValueError("n_heads and n_layers must be >= 1")

        self.blocks = nn.ModuleList([
            _TransformerBlock(
                input_dim=input_dim,
                attn_dim=attn_dim,            # ← 이름 통일
                n_heads=n_heads,
                ff_multiplier=ff_multiplier,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])
        self.out_mlp = nn.Sequential(
            nn.Linear(input_dim, rect_dim),
            nn.ReLU(),
            nn.Linear(rect_dim, rect_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.detach()         # AERIAL: detach 권장
        for blk in self.blocks:
            x = blk(x)                     # [B, N, input_dim]
        pooled = x.mean(dim=1)             # [B, input_dim]
        rect = self.out_mlp(pooled)        # [B, rect_dim]
        return rect
