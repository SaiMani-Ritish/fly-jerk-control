"""Rollouts with explicit termination reasons and no training-state leakage."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from fly_jerk_control.controllers.residual_base import ResidualController
from fly_jerk_control.envs.mass_stop_env import MassStopEnv
from fly_jerk_control.evaluation.metrics import EpisodeMetrics, compute_episode_metrics


def rollout_episode(
    env: MassStopEnv,
    residual: ResidualController,
    *,
    deterministic: bool = True,
    seed: int | None = None,
    episode_index: int = 0,
    split: str = "eval",
    measure_latency: bool = True,
) -> tuple[EpisodeMetrics, dict[str, Any]]:
    residual.reset()
    obs, info = env.reset(seed=seed, options={"episode_index": episode_index, "split": split})
    latencies: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        raw, _extras = residual.raw_action(obs, deterministic=deterministic)
        if measure_latency:
            latencies.append(time.perf_counter() - t0)
        obs, _reward, terminated, truncated, info = env.step(np.array([raw], dtype=np.float32))
    log = env.traces()
    report = residual.resource_report()
    win = tuple(env.cfg.get("evaluation", {}).get("disturbance_window", [1.2, 1.8]))
    metrics = compute_episode_metrics(
        log,
        x_target=env.x_target,
        success=bool(info["success"]),
        success_time=info.get("success_time"),
        termination_reason=str(info["termination_reason"]),
        dt_sim=env.dt_sim,
        dwell_time=float(env.cfg["env"]["dwell_time"]),
        disturbance_window=(float(win[0]), float(win[1])),
        inference_latency_s=float(np.mean(latencies)) if latencies else float("nan"),
        n_units=report.n_units,
        n_directed_edges=report.n_directed_edges,
        n_trainable_params=report.n_trainable_params,
        planner_feasible=bool(info.get("planner_feasible", True)),
        seed=int(env.cfg.get("seed", 0)),
        split=split,
        controller=report.architecture,
        smoke_label=str(env.cfg.get("label", "")),
    )
    return metrics, {"info": info, "log": log, "obs_terminal": obs}


def rollout_many(
    env: MassStopEnv,
    residual: ResidualController,
    n_episodes: int,
    **kwargs: Any,
) -> list[tuple[EpisodeMetrics, dict[str, Any]]]:
    out = []
    for i in range(n_episodes):
        out.append(rollout_episode(env, residual, episode_index=i, **kwargs))
    return out
