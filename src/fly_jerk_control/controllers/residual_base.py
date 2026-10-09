"""Residual controller interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ResidualResourceReport:
    n_units: int
    n_directed_edges: int
    n_trainable_params: int
    architecture: str
    notes: str = ""


class ResidualController(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def raw_action(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[float, dict[str, Any]]:
        """Return unbounded raw residual (pre-tanh) and extras (logp, value, hidden)."""
        raise NotImplementedError

    @abstractmethod
    def resource_report(self) -> ResidualResourceReport:
        raise NotImplementedError

    def get_recurrent_state(self) -> Any:
        return None

    def set_recurrent_state(self, state: Any) -> None:
        return None


class ZeroResidual(ResidualController):
    def reset(self) -> None:
        return None

    def raw_action(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[float, dict[str, Any]]:
        return 0.0, {"logp": 0.0, "value": 0.0, "kind": "zero"}

    def resource_report(self) -> ResidualResourceReport:
        return ResidualResourceReport(0, 0, 0, "zero", "No learned residual.")


def count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def bounded_residual(raw: float, limit: float) -> float:
    return float(limit) * float(np.tanh(raw))


def obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).view(1, -1)
