"""Jerk estimator documentation and ranking safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fly_jerk_control.config import default_config
from fly_jerk_control.controllers.residual_base import ZeroResidual
from fly_jerk_control.envs.mass_stop_env import MassStopEnv
from fly_jerk_control.evaluation.aggregation import rank_controllers
from fly_jerk_control.evaluation.rollout import rollout_episode


def test_jerk_finite_difference_uses_sim_dt() -> None:
    cfg = default_config()
    env = MassStopEnv(cfg)
    m, extra = rollout_episode(env, ZeroResidual(), seed=0)
    assert m.jerk_estimator == "finite_difference_of_plant_acceleration"
    assert abs(m.dt_sim - env.dt_sim) < 1e-15
    jerk = np.asarray(extra["log"].jerk)
    a = np.asarray(extra["log"].a)
    # Skip the first sample (undefined / zeroed at t=0).
    recon = np.diff(a) / env.dt_sim
    assert np.allclose(jerk[1:], recon, atol=1e-8)


def test_ranking_prefers_success() -> None:
    df = pd.DataFrame(
        {
            "controller": ["fail_smooth", "ok_rougher"],
            "success": [False, True],
            "integrated_squared_jerk": [0.01, 10.0],
            "seed": [0, 0],
        }
    )
    ranked = rank_controllers(df)
    assert ranked.iloc[0]["controller"] == "ok_rougher"
    assert ranked.iloc[0]["success_rate"] == 1.0
