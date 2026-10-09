"""Plant and actuator tests."""

from __future__ import annotations

import numpy as np

from fly_jerk_control.envs.actuator import ActuatorConfig, limit_command
from fly_jerk_control.envs.plant import PlantState, rk4_step


def _free_force_fn(_t: float) -> float:
    return 0.0


def test_constant_velocity_no_force_no_damping() -> None:
    act = ActuatorConfig(tau=0.0, f_max=10.0, slew_max=np.inf)
    state = PlantState(x=0.0, v=1.5, f_act=0.0, t=0.0, mass=2.0, damping=0.0)
    dt = 0.01
    for _ in range(100):
        state = rk4_step(state, dt, 0.0, _free_force_fn, act)
    assert abs(state.v - 1.5) < 1e-10
    assert abs(state.x - 1.5) < 1e-8


def test_constant_force_no_damping_quadratic_position() -> None:
    act = ActuatorConfig(tau=0.0, f_max=100.0, slew_max=np.inf)
    mass = 2.0
    f = 4.0
    state = PlantState(x=0.0, v=0.0, f_act=f, t=0.0, mass=mass, damping=0.0)
    dt = 0.001
    t_end = 0.5
    n = int(t_end / dt)
    for _ in range(n):
        state = rk4_step(state, dt, f, _free_force_fn, act)
    a = f / mass
    assert abs(state.v - a * t_end) < 1e-6
    assert abs(state.x - 0.5 * a * t_end**2) < 1e-6


def test_damping_exponential_velocity() -> None:
    act = ActuatorConfig(tau=0.0, f_max=10.0, slew_max=np.inf)
    m, b, v0 = 1.0, 2.0, 3.0
    state = PlantState(x=0.0, v=v0, f_act=0.0, t=0.0, mass=m, damping=b)
    t_end = 0.4
    dt = 0.0005
    for _ in range(int(t_end / dt)):
        state = rk4_step(state, dt, 0.0, _free_force_fn, act)
    expected = v0 * np.exp(-b / m * t_end)
    assert abs(state.v - expected) < 1e-4


def test_rk4_timestep_convergence() -> None:
    act = ActuatorConfig(tau=0.0, f_max=100.0, slew_max=np.inf)

    def simulate(dt: float) -> float:
        state = PlantState(x=0.0, v=0.0, f_act=1.0, t=0.0, mass=1.0, damping=0.3)
        t_end = 0.5
        for _ in range(int(round(t_end / dt))):
            state = rk4_step(state, dt, 1.0, _free_force_fn, act)
        return state.x

    x_coarse = simulate(0.02)
    x_fine = simulate(0.005)
    x_finer = simulate(0.001)
    err_fine = abs(x_fine - x_finer)
    err_coarse = abs(x_coarse - x_finer)
    assert err_fine < err_coarse


def test_actuator_lag_first_order() -> None:
    tau = 0.1
    act = ActuatorConfig(tau=tau, f_max=100.0, slew_max=np.inf)
    state = PlantState(x=0.0, v=0.0, f_act=0.0, t=0.0, mass=1.0, damping=0.0)
    f_cmd = 5.0
    t_end = 0.2
    dt = 0.0005
    for _ in range(int(t_end / dt)):
        state = rk4_step(state, dt, f_cmd, _free_force_fn, act)
    expected = f_cmd * (1.0 - np.exp(-t_end / tau))
    assert abs(state.f_act - expected) < 1e-3


def test_actuator_magnitude_and_slew() -> None:
    cfg = ActuatorConfig(tau=0.05, f_max=2.0, slew_max=10.0)
    r1 = limit_command(50.0, 0.0, 0.1, cfg)
    assert r1.saturated
    assert r1.slew_limited
    assert abs(r1.f_limited - 1.0) < 1e-12  # slew 10 N/s * 0.1 s
    r2 = limit_command(0.5, 0.0, 0.1, cfg)
    assert not r2.saturated
    assert not r2.slew_limited
    assert abs(r2.f_limited - 0.5) < 1e-12
