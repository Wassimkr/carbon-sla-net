"""Custom actor-critic network for Carbon-SLA-Net.

Designed for use with MaskablePPO (sb3-contrib) and independently testable
by BCTrainer for supervised behavior cloning.  Uses LayerNorm rather than
BatchNorm so the network works correctly at batch size 1 during PPO rollouts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    """Single encoder layer: Linear → LayerNorm → ReLU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout_p: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.relu(self.norm(self.linear(x))))


class CarbonSLANetPolicy(nn.Module):
    """Shared-trunk actor-critic network.

    Architecture::

        Input (obs_dim)
          ↓
        _EncoderBlock(obs_dim → hidden[0])   ← linear + layernorm + relu + dropout
        _EncoderBlock(hidden[0] → hidden[1]) ← linear + layernorm + relu + dropout
          ↓                    ↓
        actor Linear          critic Linear
        (hidden[-1] → n_actions)  (hidden[-1] → 1)
    """

    def __init__(
        self,
        obs_dim: int = 74,
        n_actions: int = 11,
        hidden_sizes: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        # Shared encoder trunk
        blocks: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            blocks.append(_EncoderBlock(in_dim, h, dropout))
            in_dim = h
        self.encoder = nn.Sequential(*blocks)

        # Actor and critic heads
        self.actor = nn.Linear(hidden_sizes[-1], n_actions)
        self.critic = nn.Linear(hidden_sizes[-1], 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (standard PPO orthogonal init)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        gain_trunk = math.sqrt(2)
        for block in self.encoder:
            nn.init.orthogonal_(block.linear.weight, gain=gain_trunk)
            nn.init.constant_(block.linear.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, state_value). Shapes: (B, n_actions), (B, 1)."""
        features = self.encoder(obs)
        return self.actor(features), self.critic(features)

    def get_action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits only. Shape: (B, n_actions)."""
        return self.actor(self.encoder(obs))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state value only. Shape: (B, 1)."""
        return self.critic(self.encoder(obs))
