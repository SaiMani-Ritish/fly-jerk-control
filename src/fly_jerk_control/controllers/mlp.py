"""Small MLP residual actor with a separate critic for PPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from fly_jerk_control.controllers.residual_base import ResidualController, ResidualResourceReport, count_params, obs_tensor


class MLPResidual(ResidualController, nn.Module):
    def __init__(
        self,
        obs_dim: int = 6,
        hidden_sizes: list[int] | tuple[int, ...] = (32, 32),
        log_std_init: float = -0.5,
        device: str = "cpu",
    ) -> None:
        nn.Module.__init__(self)
        self.device = torch.device(device)
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.Tanh()])
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, 1)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([log_std_init], dtype=torch.float32))
        self.to(self.device)

    def reset(self) -> None:
        return None

    def _dist(self, obs_t: torch.Tensor) -> Normal:
        h = self.backbone(obs_t)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def value(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = obs_tensor(obs, self.device)
        return self.critic(obs).squeeze(-1)

    def raw_action(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[float, dict[str, Any]]:
        obs_t = obs_tensor(obs, self.device)
        dist = self._dist(obs_t)
        raw_t = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(raw_t).sum(-1)
        value = self.value(obs_t)
        raw = float(raw_t.squeeze().detach().cpu())
        return raw, {
            "logp": float(logp.detach().cpu()),
            "value": float(value.detach().cpu()),
            "kind": "mlp",
        }

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return logp, entropy, value

    def resource_report(self) -> ResidualResourceReport:
        return ResidualResourceReport(
            n_units=0,
            n_directed_edges=0,
            n_trainable_params=count_params(self),
            architecture="mlp",
            notes="Feedforward residual. Neuron/edge counts are not comparable to graph models.",
        )
