# modules/mixers/qmix_rect.py

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QMixerRect(nn.Module):
    """
    QMIX with rect (AERIAL shared latent) as the conditioning input
    instead of environment state.

    - Keeps the same structure/behavior as the original QMixer:
      * hypernet_layers in {1,2}
      * abs() on hyper weights to preserve monotonicity
      * V(·) as state-dependent (here: rect-dependent) bias
      * ELU hidden
    """

    def __init__(self, args, rect_dim=None):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents

        # rect_dim은 Transformer 출력 차원으로 고정
        if rect_dim is None:
            rect_dim = getattr(args, "rect_dim", None) or getattr(args, "hidden_state_transformer_dim", None)
            assert rect_dim is not None, "rect_dim 또는 hidden_state_transformer_dim이 지정되어야 합니다."
    
        # rect_dim을 그대로 사용
        self.rect_dim = int(rect_dim)
        self.embed_dim = args.mixing_embed_dim

        # Hypernetwork setup (same options as original QMixer)
        n_hlayers = int(getattr(args, "hypernet_layers", 1))
        if n_hlayers == 1:
            self.hyper_w_1 = nn.Linear(self.rect_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.rect_dim, self.embed_dim)
        elif n_hlayers == 2:
            hypernet_embed = int(getattr(self.args, "hypernet_embed", 128))
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.rect_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.rect_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif n_hlayers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # Rect-dependent bias for the hidden layer
        self.hyper_b_1 = nn.Linear(self.rect_dim, self.embed_dim)

        # V(rect) instead of a bias for the last layer
        self.V = nn.Sequential(
            nn.Linear(self.rect_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: th.Tensor, rects: th.Tensor) -> th.Tensor:
        """
        Args:
            agent_qs: (B, T, N) or (B, T, N, 1) or (B, N)  — we will view as (-1, 1, N)
            rects:    (B, T, R) or (B, R)

        Returns:
            q_tot: (B, T, 1) (or (B, 1) if input had no time dimension)
        """
        bs = agent_qs.size(0)

        # Normalize agent_qs to shape (-1, 1, N)
        if agent_qs.dim() == 4:
            # (B, T, N, 1)
            B, T, N, _ = agent_qs.shape
            assert N == self.n_agents
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            time_len = T
        elif agent_qs.dim() == 3:
            # (B, T, N)
            B, T, N = agent_qs.shape
            assert N == self.n_agents
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            time_len = T
        elif agent_qs.dim() == 2:
            # (B, N)
            B, N = agent_qs.shape
            assert N == self.n_agents
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            time_len = 1
        else:
            raise ValueError(f"agent_qs shape not supported: {agent_qs.shape}")

        # Normalize rects to shape (-1, R) aligned with time if present
        if rects.dim() == 3:
            # (B, T, R)  -> (-1, R)
            rects = rects.view(-1, self.rect_dim)
        elif rects.dim() == 2:
            # (B, R) -> broadcast across time if needed
            if time_len > 1:
                rects = rects.unsqueeze(1).expand(bs, time_len, -1).contiguous()
                rects = rects.view(-1, self.rect_dim)
            else:
                rects = rects.view(-1, self.rect_dim)
        else:
            raise ValueError(f"rects shape not supported: {rects.shape}")

        # First layer
        w1 = th.abs(self.hyper_w_1(rects))                       # (-1, embed_dim * N)
        b1 = self.hyper_b_1(rects)                               # (-1, embed_dim)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)          # (-1, N, E)
        b1 = b1.view(-1, 1, self.embed_dim)                      # (-1, 1, E)

        hidden = F.elu(th.bmm(agent_qs, w1) + b1)                # (-1, 1, E)

        # Second (final) layer
        w_final = th.abs(self.hyper_w_final(rects))              # (-1, E)
        w_final = w_final.view(-1, self.embed_dim, 1)            # (-1, E, 1)

        # Rect-dependent V
        v = self.V(rects).view(-1, 1, 1)                         # (-1, 1, 1)

        # Final output
        y = th.bmm(hidden, w_final) + v                          # (-1, 1, 1)

        # Reshape back to (B, T, 1)
        q_tot = y.view(bs, time_len, 1)
        return q_tot
