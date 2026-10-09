"""Degeneracy diagnostics: constant action, no motion, smoothness-without-success."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_jerk_control.envs.mass_stop_env import EpisodeLog
from fly_jerk_control.evaluation.metrics import EpisodeMetrics


@dataclass
class DegeneracyReport:
    constant_residual: bool
    near_stationary_state: bool
    success: bool
    smoothness_without_success: bool
    notes: str


def diagnose(log: EpisodeLog, metrics: EpisodeMetrics, x0: float, pos_tol: float) -> DegeneracyReport:
    f_res = np.asarray(log.f_res, dtype=float)
    x = np.asarray(log.x, dtype=float)
    constant = bool(f_res.size > 1 and np.max(np.abs(f_res - f_res[0])) < 1e-8)
    stationary = bool(np.max(np.abs(x - x0)) < pos_tol)
    bad_smooth = (not metrics.success) and np.isfinite(metrics.integrated_squared_jerk)
    notes = []
    if constant:
        notes.append("residual is constant")
    if stationary:
        notes.append("state remained near initial position")
    if bad_smooth:
        notes.append("failed task; smoothness must not be used as a win condition")
    return DegeneracyReport(
        constant_residual=constant,
        near_stationary_state=stationary,
        success=metrics.success,
        smoothness_without_success=bad_smooth,
        notes="; ".join(notes) if notes else "no degeneracy flags",
    )
