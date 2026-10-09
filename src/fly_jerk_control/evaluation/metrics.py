"""Episode metrics with explicit validity flags.

Post-stop oscillation is undefined (NaN) if the episode never achieved a
successful stop. Smoothness rankings must report success rate first.
This module does not implement any prior JAS formula.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fly_jerk_control.envs.mass_stop_env import EpisodeLog


@dataclass
class EpisodeMetrics:
    success: bool
    termination_reason: str
    endpoint_position_error: float
    endpoint_velocity: float
    completion_time: float
    settling_time: float
    settling_time_valid: bool
    peak_abs_jerk: float
    integrated_squared_jerk: float
    peak_abs_jerk_disturbance_window: float
    integrated_squared_jerk_disturbance_window: float
    post_stop_oscillation: float
    post_stop_oscillation_valid: bool
    residual_l2: float
    residual_mean_abs: float
    force_saturation_fraction: float
    action_variation: float
    inference_latency_s: float
    n_units: int
    n_directed_edges: int
    n_trainable_params: int
    planner_feasible: bool
    dt_sim: float
    jerk_estimator: str
    seed: int
    split: str
    controller: str
    smoke_label: str


def _window_mask(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    if window is None:
        return np.ones_like(t, dtype=bool)
    t0, t1 = window
    return (t >= t0) & (t <= t1)


def compute_episode_metrics(
    log: EpisodeLog,
    *,
    x_target: float,
    success: bool,
    success_time: float | None,
    termination_reason: str,
    dt_sim: float,
    dwell_time: float,
    disturbance_window: tuple[float, float] | None,
    inference_latency_s: float = float("nan"),
    n_units: int = 0,
    n_directed_edges: int = 0,
    n_trainable_params: int = 0,
    planner_feasible: bool = True,
    seed: int = 0,
    split: str = "eval",
    controller: str = "unknown",
    smoke_label: str = "",
) -> EpisodeMetrics:
    t = np.asarray(log.t, dtype=float)
    x = np.asarray(log.x, dtype=float)
    v = np.asarray(log.v, dtype=float)
    jerk = np.asarray(log.jerk, dtype=float)
    f_res = np.asarray(log.f_res, dtype=float)
    sat = np.asarray(log.saturated, dtype=float)
    if t.size == 0:
        raise ValueError("empty episode log")
    peak_j = float(np.max(np.abs(jerk)))
    isj = float(np.sum(jerk**2) * dt_sim)
    mask = _window_mask(t, disturbance_window)
    if mask.any():
        peak_j_w = float(np.max(np.abs(jerk[mask])))
        isj_w = float(np.sum(jerk[mask] ** 2) * dt_sim)
    else:
        peak_j_w = float("nan")
        isj_w = float("nan")

    settling_time = float("nan")
    settling_valid = False
    osc = float("nan")
    osc_valid = False
    if success and success_time is not None:
        settling_time = float(success_time)
        settling_valid = True
        post = t >= success_time
        if post.any():
            osc = float(np.max(np.abs(x[post] - x_target)))
            osc_valid = True
    completion = float(t[-1])
    action_var = float(np.mean(np.abs(np.diff(f_res)))) if f_res.size > 1 else 0.0
    return EpisodeMetrics(
        success=bool(success),
        termination_reason=str(termination_reason),
        endpoint_position_error=float(abs(x[-1] - x_target)),
        endpoint_velocity=float(v[-1]),
        completion_time=completion,
        settling_time=settling_time,
        settling_time_valid=settling_valid,
        peak_abs_jerk=peak_j,
        integrated_squared_jerk=isj,
        peak_abs_jerk_disturbance_window=peak_j_w,
        integrated_squared_jerk_disturbance_window=isj_w,
        post_stop_oscillation=osc,
        post_stop_oscillation_valid=osc_valid,
        residual_l2=float(np.sqrt(np.mean(f_res**2))),
        residual_mean_abs=float(np.mean(np.abs(f_res))),
        force_saturation_fraction=float(np.mean(sat)),
        action_variation=action_var,
        inference_latency_s=float(inference_latency_s),
        n_units=int(n_units),
        n_directed_edges=int(n_directed_edges),
        n_trainable_params=int(n_trainable_params),
        planner_feasible=bool(planner_feasible),
        dt_sim=float(dt_sim),
        jerk_estimator="finite_difference_of_plant_acceleration",
        seed=int(seed),
        split=str(split),
        controller=str(controller),
        smoke_label=str(smoke_label),
    )


def metrics_to_dict(m: EpisodeMetrics) -> dict[str, Any]:
    return asdict(m)
