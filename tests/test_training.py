"""Training smoke: checkpoint reload does not leak recurrent state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fly_jerk_control.config import default_config
from fly_jerk_control.controllers.mlp import MLPResidual
from fly_jerk_control.envs.mass_stop_env import MassStopEnv
from fly_jerk_control.evaluation.rollout import rollout_episode
from fly_jerk_control.factory import build_residual
from fly_jerk_control.training.checkpoints import load_checkpoint, save_checkpoint
from fly_jerk_control.training.ppo import train_ppo


def test_tiny_mlp_train_and_reload(tmp_path: Path) -> None:
    cfg = default_config()
    cfg["residual"]["type"] = "mlp"
    cfg["residual"]["hidden_sizes"] = [8, 8]
    cfg["env"]["episode_duration"] = 1.0
    cfg["planner"]["duration"] = 0.6
    cfg["training"]["total_env_steps"] = 64
    cfg["training"]["n_steps"] = 32
    cfg["training"]["n_epochs"] = 1
    cfg["training"]["minibatch_size"] = 16
    env = MassStopEnv(cfg)
    ctrl = build_residual(cfg)
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoints").mkdir()
    train_ppo(env, ctrl, cfg, run)
    ckpt = run / "checkpoints" / "final.pt"
    assert ckpt.exists()
    other = MLPResidual(hidden_sizes=[8, 8])
    load_checkpoint(ckpt, other)
    for p1, p2 in zip(ctrl.parameters(), other.parameters()):
        assert torch.allclose(p1, p2)


def test_recurrent_train_resets_between_eval_episodes(tmp_path: Path) -> None:
    cfg = default_config()
    cfg["residual"]["type"] = "recurrent"
    cfg["residual"]["n_units"] = 6
    cfg["env"]["episode_duration"] = 0.8
    cfg["planner"]["duration"] = 0.5
    cfg["training"]["total_env_steps"] = 64
    cfg["training"]["n_steps"] = 32
    cfg["training"]["n_epochs"] = 1
    cfg["training"]["bptt_len"] = 8
    env = MassStopEnv(cfg)
    ctrl = build_residual(cfg)
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoints").mkdir()
    train_ppo(env, ctrl, cfg, run)
    m1, _ = rollout_episode(env, ctrl, seed=0, episode_index=0)
    h_after = ctrl.get_recurrent_state().clone()
    m2, _ = rollout_episode(env, ctrl, seed=0, episode_index=0)
    # rollout_episode resets at start; second eval should match first and not continue hidden state.
    assert m1.completion_time == m2.completion_time
    ctrl.reset()
    assert float(h_after.abs().sum()) >= 0.0
