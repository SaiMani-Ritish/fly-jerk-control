"""Shared actuator command limits and first-order lag.

Slew-rate limits apply to the **command** `F_cmd`, not to plant jerk.
They do not guarantee a hard physical jerk bound under disturbances or
time-varying mass/damping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActuatorConfig:
    tau: float
    f_max: float
    slew_max: float
    """Maximum |dF_cmd/dt| in N/s. Use a non-finite value to disable slew limiting."""


@dataclass
class CommandLimitResult:
    f_limited: float
    saturated: bool
    slew_limited: bool
    f_requested: float


def limit_command(
    f_requested: float,
    f_prev_limited: float,
    dt: float,
    cfg: ActuatorConfig,
) -> CommandLimitResult:
    """Apply magnitude clip, then slew limit toward the clipped request."""
    f_req = float(f_requested)
    f_mag = float(np.clip(f_req, -cfg.f_max, cfg.f_max))
    saturated = abs(f_req) > cfg.f_max + 1e-12
    f_out = f_mag
    slew_limited = False
    if np.isfinite(cfg.slew_max) and cfg.slew_max >= 0.0:
        max_delta = cfg.slew_max * float(dt)
        delta = f_mag - f_prev_limited
        if abs(delta) > max_delta + 1e-12:
            f_out = f_prev_limited + np.clip(delta, -max_delta, max_delta)
            slew_limited = True
        else:
            f_out = f_mag
        f_out = float(np.clip(f_out, -cfg.f_max, cfg.f_max))
    return CommandLimitResult(
        f_limited=float(f_out),
        saturated=saturated,
        slew_limited=slew_limited,
        f_requested=f_req,
    )


def actuator_derivative(f_act: float, f_cmd_limited: float, tau: float) -> float:
    """First-order lag: tau * dF_act/dt = F_cmd - F_act.

    If tau <= 0, the lag is disabled and the derivative is unused (algebraic set).
    """
    if tau <= 0.0:
        return 0.0
    return (f_cmd_limited - f_act) / tau
