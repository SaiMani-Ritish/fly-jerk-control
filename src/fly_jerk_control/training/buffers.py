"""Rollout storage for feedforward and truncated-BPTT recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)  # 1 at episode end (term or trunc)
    values: list[float] = field(default_factory=list)
    hiddens: list[np.ndarray] = field(default_factory=list)
    resets: list[float] = field(default_factory=list)  # 1 if this step starts a new episode

    def add(self, **kwargs) -> None:
        self.obs.append(np.asarray(kwargs["obs"], dtype=np.float32))
        self.actions.append(np.asarray(kwargs["action"], dtype=np.float32).reshape(1))
        self.logp.append(float(kwargs["logp"]))
        self.rewards.append(float(kwargs["reward"]))
        self.dones.append(float(kwargs["done"]))
        self.values.append(float(kwargs["value"]))
        hid = kwargs.get("hidden")
        if hid is None:
            hid = np.zeros(1, dtype=np.float32)
        self.hiddens.append(np.asarray(hid, dtype=np.float32))
        self.resets.append(float(kwargs.get("reset", 0.0)))

    def as_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(np.stack(self.obs)),
            "actions": torch.from_numpy(np.stack(self.actions)),
            "logp": torch.tensor(self.logp, dtype=torch.float32),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
            "hiddens": torch.from_numpy(np.stack(self.hiddens)),
            "resets": torch.tensor(self.resets, dtype=torch.float32),
        }

    def clear(self) -> None:
        self.__init__()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = rewards.shape[0]
    adv = torch.zeros(t)
    last_gae = 0.0
    next_value = last_value
    for i in reversed(range(t)):
        next_nonterminal = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * next_value * next_nonterminal - values[i]
        last_gae = float(delta + gamma * gae_lambda * next_nonterminal * last_gae)
        adv[i] = last_gae
        next_value = float(values[i])
    ret = adv + values
    return adv, ret
