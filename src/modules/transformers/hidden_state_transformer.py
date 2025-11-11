import math
from typing import List
import torch
import torch.nn as nn


class _SelfAttentionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(1, 2) / self.scale, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int,
                 ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            _SelfAttentionHead(hidden_dim, dropout) for _ in range(n_heads)
        ])
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        ff_hidden = hidden_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_sum = torch.stack([h(x) for h in self.heads], dim=0).sum(dim=0)
        x = self.norm1(x + self.attn_dropout(attn_sum))
        x = self.norm2(x + self.ffn(x))
        return x


class HiddenStateTransformer(nn.Module):
    """
    Aggregate agent RNN hiddens (B, N, H) -> rect (B, R)
    AERIAL: agent-axis attention -> pool over agents -> small MLP -> rect.
    """
    def __init__(self,
                 input_dim: int,          # H (agent RNN hidden size)
                 rect_dim: int = 64,      # R (output)
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
                n_heads=n_heads,
                attn_dim=attn_dim,
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
        """
        hidden_states: [B, N, H]  (detach 권장)
        returns: rect: [B, R]
        """
        x = hidden_states.detach()  # AERIAL 구현 관행: 분리
        for blk in self.blocks:
            x = blk(x)              # [B, N, H]
        pooled = x.mean(dim=1)      # agent-axis pooling -> [B, H]
        rect = self.out_mlp(pooled) # [B, R]
        return rect
