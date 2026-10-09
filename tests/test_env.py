"""Environment stop logic, termination, delay, shared actuator path."""

from __future__ import annotations

import numpy as np

from fly_jerk_control.config import default_config
from fly_jerk_control.controllers.residual_base import ZeroResidual
from fly_jerk_control.envs.mass_stop_env import MassStopEnv
from fly_jerk_control.evaluation.metrics import compute_episode_metrics
from fly_jerk_control.evaluation.rollout import rollout_episode


def _cfg(**kwargs):
    cfg = default_config()
    for k, v in kwargs.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def test_nominal_stops_stationary_fails() -> None:
    cfg = _cfg()
    env = MassStopEnv(cfg)
    m_ok, extra = rollout_episode(env, ZeroResidual(), seed=0, episode_index=0)
    assert m_ok.success
    assert extra["info"]["termination_reason"] in ("success", "timeout")
    assert extra["info"]["termination_reason"] == "success" or m_ok.success

    cfg_f = _cfg()
    cfg_f["env"]["command_mode"] = "force"
    env_f = MassStopEnv(cfg_f)
    obs, _ = env_f.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _r, terminated, truncated, info = env_f.step(np.array([0.0], dtype=np.float32))
    assert not info["success"]
    assert info["termination_reason"] == "timeout"


def test_unsafe_termination_reason() -> None:
    cfg = _cfg()
    cfg["env"]["command_mode"] = "force"
    cfg["env"]["x_max"] = 0.2
    cfg["env"]["f_max"] = 80.0
    cfg["env"]["slew_max"] = 1e6
    env = MassStopEnv(cfg)
    obs, _ = env.reset(seed=0)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        obs, _r, terminated, truncated, info = env.step(np.array([80.0], dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "unsafe"


def test_constant_force_does_not_false_success() -> None:
    cfg = _cfg()
    cfg["env"]["command_mode"] = "force"
    cfg["env"]["slew_max"] = 1e6
    env = MassStopEnv(cfg)
    obs, _ = env.reset(seed=1)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _r, terminated, truncated, info = env.step(np.array([3.0], dtype=np.float32))
    assert not info["success"]


def test_dwell_requires_hold() -> None:
    cfg = _cfg()
    cfg["env"]["dwell_time"] = 0.5
    cfg["env"]["pos_tol"] = 0.05
    cfg["env"]["vel_tol"] = 0.05
    env = MassStopEnv(cfg)
    m, extra = rollout_episode(env, ZeroResidual(), seed=0)
    if extra["info"]["success"]:
        assert extra["info"]["success_time"] is not None
        assert extra["info"]["success_time"] >= 0.5 - 1e-6


def test_observation_delay_no_future_leak() -> None:
    cfg = _cfg()
    cfg["env"]["obs_delay_steps"] = 5
    cfg["env"]["obs_noise_pos"] = 0.0
    cfg["env"]["obs_noise_vel"] = 0.0
    env = MassStopEnv(cfg)
    obs, _ = env.reset(seed=0)
    xs = []
    delayed_pos_err = []
    for _ in range(20):
        xs.append(env._state.x)
        delayed_pos_err.append(float(obs[0]) * float(cfg["residual"]["obs_scales"][0]))
        obs, _r, term, trunc, _i = env.step(np.array([0.0], dtype=np.float32))
        if term or trunc:
            break
    # Delayed pos error uses x_meas - x_ref at delayed time, not current x.
    # After several steps, current x has moved; first delayed errors stay near the initial error.
    assert abs(delayed_pos_err[0]) < 0.02
    assert abs(xs[-1] - xs[0]) > 0.01


def test_shared_actuator_path_logs() -> None:
    cfg = _cfg()
    env = MassStopEnv(cfg)
    _m, extra = rollout_episode(env, ZeroResidual(), seed=0)
    log = extra["log"]
    assert len(log.f_nom) == len(log.f_act) == len(log.f_cmd)
    assert max(abs(f) for f in log.f_cmd) <= cfg["env"]["f_max"] + 1e-8


def test_metric_nan_if_never_stops() -> None:
    cfg = _cfg()
    cfg["env"]["command_mode"] = "force"
    env = MassStopEnv(cfg)
    obs, _ = env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _r, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    m = compute_episode_metrics(
        env.traces(),
        x_target=1.0,
        success=False,
        success_time=None,
        termination_reason=info["termination_reason"],
        dt_sim=env.dt_sim,
        dwell_time=0.2,
        disturbance_window=(1.2, 1.8),
    )
    assert not m.post_stop_oscillation_valid
    assert np.isnan(m.post_stop_oscillation)


def test_reproducible_scenarios() -> None:
    cfg = _cfg()
    cfg["disturbance"]["kind"] = "mass_change"
    cfg["disturbance"]["mass_values"] = [0.8, 1.2]
    env = MassStopEnv(cfg)
    _, i1 = env.reset(seed=0, options={"episode_index": 3})
    m1 = i1["mass"]
    _, i2 = env.reset(seed=0, options={"episode_index": 3})
    assert i2["mass"] == m1
