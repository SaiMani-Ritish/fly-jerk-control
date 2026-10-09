"""Leaky-integrator recurrent residual with a dense (random) map.

h_next = (1-alpha)*h + alpha*tanh(W_graph h + W_in obs + b)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from fly_jerk_control.controllers.residual_base import ResidualController, ResidualResourceReport, count_params, obs_tensor


class RecurrentResidual(ResidualController, nn.Module):
    def __init__(
        self,
        obs_dim: int = 6,
        n_units: int = 8,
        alpha: float = 0.5,
        log_std_init: float = -0.5,
        mask: np.ndarray | None = None,
        device: str = "cpu",
        architecture: str = "random_recurrent",
    ) -> None:
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.n_units = int(n_units)
        self.obs_dim = int(obs_dim)
        self.alpha = float(alpha)
        self.architecture = architecture
        if mask is None:
            mask_np = np.ones((self.n_units, self.n_units), dtype=np.float32)
        else:
            mask_np = np.asarray(mask, dtype=np.float32)
            if mask_np.shape != (self.n_units, self.n_units):
                raise ValueError(f"Mask shape {mask_np.shape} != {(self.n_units, self.n_units)}")
        # mask[j, i] = 1 iff neuron i influences neuron j (edge i -> j).
        self.register_buffer("mask", torch.from_numpy(mask_np))
        self.w_graph = nn.Parameter(torch.zeros(self.n_units, self.n_units))
        self.w_in = nn.Parameter(torch.zeros(self.n_units, self.obs_dim))
        self.bias = nn.Parameter(torch.zeros(self.n_units))
        nn.init.xavier_uniform_(self.w_graph)
        nn.init.xavier_uniform_(self.w_in)
        with torch.no_grad():
            self.w_graph.mul_(self.mask)
        self.readout = nn.Linear(self.n_units, 1)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([log_std_init], dtype=torch.float32))
        self.h = torch.zeros(self.n_units, device=self.device)
        self.to(self.device)

    def reset(self) -> None:
        self.h = torch.zeros(self.n_units, device=self.device)

    def get_recurrent_state(self) -> torch.Tensor:
        return self.h.detach().clone()

    def set_recurrent_state(self, state: Any) -> None:
        if state is None:
            self.reset()
        else:
            self.h = torch.as_tensor(state, device=self.device, dtype=torch.float32).reshape(self.n_units)

    def effective_w(self) -> torch.Tensor:
        return self.w_graph * self.mask

    def step_hidden(self, obs_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """obs_t: (..., obs_dim), h: (..., n_units)."""
        w = self.effective_w()
        pre = torch.matmul(h, w.T) + torch.matmul(obs_t, self.w_in.T) + self.bias
        return (1.0 - self.alpha) * h + self.alpha * torch.tanh(pre)

    def value(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = obs_tensor(obs, self.device)
        return self.critic(obs).squeeze(-1)

    def _forward_one(self, obs_t: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        self.h = self.step_hidden(obs_t.view(-1), self.h)
        mu = self.readout(self.h.unsqueeze(0))
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std), self.h

    def raw_action(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[float, dict[str, Any]]:
        obs_t = obs_tensor(obs, self.device)
        dist, _ = self._forward_one(obs_t)
        raw_t = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(raw_t).sum(-1)
        value = self.critic(obs_t).squeeze(-1)
        return float(raw_t.squeeze().detach().cpu()), {
            "logp": float(logp.detach().cpu()),
            "value": float(value.detach().cpu()),
            "kind": self.architecture,
            "hidden": self.h.detach().cpu().numpy().copy(),
        }

    def sequence_forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        h0: torch.Tensor,
        resets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Truncated BPTT over a sequence.

        obs: (T, B, obs_dim), actions: (T, B, 1), h0: (B, n), resets: (T, B)
        resets[t]=1 means the hidden state is zeroed **before** step t (episode boundary).
        """
        t_len, batch, _ = obs.shape
        h = h0
        logps = []
        values = []
        entropies = []
        for t in range(t_len):
            reset = resets[t].view(batch, 1)
            h = h * (1.0 - reset)
            h = self.step_hidden(obs[t], h)
            mu = self.readout(h)
            std = torch.exp(self.log_std).expand_as(mu)
            dist = Normal(mu, std)
            logps.append(dist.log_prob(actions[t]).sum(-1))
            entropies.append(dist.entropy().sum(-1))
            values.append(self.critic(obs[t]).squeeze(-1))
        return torch.stack(logps, 0), torch.stack(entropies, 0), torch.stack(values, 0)

    def resource_report(self) -> ResidualResourceReport:
        n_edges = int(self.mask.sum().item())
        return ResidualResourceReport(
            n_units=self.n_units,
            n_directed_edges=n_edges,
            n_trainable_params=count_params(self),
            architecture=self.architecture,
            notes=(
                "W_graph is masked; masked-out entries are multiplied by 0 so they "
                "neither contribute to the pre-activation nor receive gradient. "
                "Parameter count includes encoder/decoder/critic, not just graph weights. "
                "Edge i->j means unit i influences unit j; stored as mask[j,i]."
            ),
        )
