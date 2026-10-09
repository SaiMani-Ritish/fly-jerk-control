"""Compose nominal tracker + residual + shared actuator (logging helper)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_jerk_control.controllers.residual_base import ResidualController, bounded_residual
from fly_jerk_control.envs.actuator import ActuatorConfig, CommandLimitResult, limit_command


@dataclass
class ComposedCommand:
    f_nominal: float
    f_residual: float
    f_requested: float
    limit: CommandLimitResult
    extras: dict[str, Any]


class ComposedController:
    """Residual is applied in the environment; this helper is for offline composition tests."""

    def __init__(self, residual: ResidualController, residual_limit: float) -> None:
        self.residual = residual
        self.residual_limit = residual_limit

    def reset(self) -> None:
        self.residual.reset()

    def command(
        self,
        obs: np.ndarray,
        f_nominal: float,
        f_prev_limited: float,
        dt: float,
        actuator: ActuatorConfig,
        *,
        deterministic: bool = True,
    ) -> ComposedCommand:
        raw, extras = self.residual.raw_action(obs, deterministic=deterministic)
        f_res = bounded_residual(raw, self.residual_limit)
        f_req = float(f_nominal) + f_res
        lim = limit_command(f_req, f_prev_limited, dt, actuator)
        return ComposedCommand(f_nominal, f_res, f_req, lim, extras)
