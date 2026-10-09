"""One-dimensional mass-damper plant with RK4 integration.

Plant:
    m * dv/dt = F_act - b * v + F_disturbance
    dx/dt = v

Actuator lag is integrated in the same state vector:
    tau * dF_act/dt = F_cmd_limited - F_act

If tau <= 0, F_act is set equal to F_cmd_limited at the start of each substep.

Within-episode mass changes, if enabled, are **idealized parameter switches**:
velocity is continuous and mass jumps. This is not a payload-pickup model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_jerk_control.envs.actuator import ActuatorConfig, actuator_derivative


@dataclass
class PlantState:
    x: float
    v: float
    f_act: float
    t: float
    mass: float
    damping: float


def plant_acceleration(state: PlantState, f_disturbance: float) -> float:
    return (state.f_act - state.damping * state.v + f_disturbance) / state.mass


def _deriv(
    t: float,
    y: np.ndarray,
    *,
    mass: float,
    damping: float,
    tau: float,
    f_cmd: float,
    f_disturbance: float,
) -> np.ndarray:
    x, v, f_act = float(y[0]), float(y[1]), float(y[2])
    dx = v
    dv = (f_act - damping * v + f_disturbance) / mass
    df = actuator_derivative(f_act, f_cmd, tau)
    return np.array([dx, dv, df], dtype=float)


def rk4_step(
    state: PlantState,
    dt: float,
    f_cmd_limited: float,
    f_disturbance_fn,
    actuator: ActuatorConfig,
) -> PlantState:
    """Advance plant+actuator with classical RK4.

    `f_disturbance_fn(t)` is evaluated at RK4 stage times. Mass and damping are
    held constant over the step (switches belong at substep boundaries).
    """
    y = np.array([state.x, state.v, state.f_act], dtype=float)
    mass, damping, tau = state.mass, state.damping, actuator.tau
    t0 = state.t
    if tau <= 0.0:
        y[2] = f_cmd_limited

    def f(t: float, yy: np.ndarray) -> np.ndarray:
        return _deriv(
            t,
            yy,
            mass=mass,
            damping=damping,
            tau=tau,
            f_cmd=f_cmd_limited,
            f_disturbance=float(f_disturbance_fn(t)),
        )

    k1 = f(t0, y)
    k2 = f(t0 + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t0 + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t0 + dt, y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if tau <= 0.0:
        y_next[2] = f_cmd_limited
    return PlantState(
        x=float(y_next[0]),
        v=float(y_next[1]),
        f_act=float(y_next[2]),
        t=t0 + dt,
        mass=mass,
        damping=damping,
    )
