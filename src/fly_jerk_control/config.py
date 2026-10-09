"""YAML configuration loading with documented defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def default_config() -> dict[str, Any]:
    return {
        "seed": 0,
        "run_name": "run",
        "label": "unlabeled",
        "output_root": "outputs",
        "env": {
            "mass": 1.0,
            "damping": 0.5,
            "tau_act": 0.05,
            "f_max": 40.0,
            "slew_max": 200.0,
            "x_min": -0.5,
            "x_max": 1.5,
            "dt_ctrl": 0.01,
            "n_substeps": 4,
            "episode_duration": 4.0,
            "x0": 0.0,
            "v0": 0.0,
            "x_target": 1.0,
            "v_target": 0.0,
            "a_target": 0.0,
            "obs_noise_pos": 0.0,
            "obs_noise_vel": 0.0,
            "obs_delay_steps": 0,
            "pos_tol": 0.05,
            "vel_tol": 0.05,
            "dwell_time": 0.2,
            "terminate_on_success": False,
            "command_mode": "residual",
            "log_substeps": True,
        },
        "planner": {
            "type": "quintic",
            "duration": 2.0,
            "v0": 0.0,
            "a0": 0.0,
            "vf": 0.0,
            "af": 0.0,
            "v_max": 2.0,
            "a_max": 5.0,
            "j_max": 40.0,
            "enforce_limits": False,
        },
        "nominal": {
            "mass": 1.0,
            "damping": 0.5,
            "kp": 40.0,
            "kd": 15.0,
        },
        "residual": {
            "type": "zero",
            "limit": 8.0,
            "hidden_sizes": [32, 32],
            "n_units": 8,
            "alpha": 0.5,
            "init": "topology_random",
            "graph_nodes": "data/fixtures/synthetic_nodes.csv",
            "graph_edges": "data/fixtures/synthetic_edges.csv",
            "allow_synthetic": True,
            "obs_scales": [1.0, 1.0, 1.0, 5.0, 8.0, 1.0],
        },
        "disturbance": {
            "kind": "none",
            "pulse_t0": 1.2,
            "pulse_duration": 0.4,
            "pulse_amplitude": 4.0,
            "mass_values": [1.0],
            "damping_values": [0.5],
            "tau_values": [0.05],
            "within_episode_mass_switch": False,
            "mass_switch_time": 1.5,
            "mass_after_switch": 1.4,
        },
        "reward": {
            "w_pos": 4.0,
            "w_vel": 1.0,
            "w_residual": 0.01,
            "w_dresidual": 0.02,
            "success_bonus": 5.0,
            "unsafe_penalty": 20.0,
        },
        "training": {
            "algo": "ppo",
            "total_env_steps": 2048,
            "n_steps": 256,
            "n_epochs": 4,
            "minibatch_size": 64,
            "bptt_len": 16,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "lr": 3.0e-4,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 1.0,
            "log_std_init": -0.5,
            "normalize_obs": False,
            "split": "train",
            "recurrent_supported": True,
        },
        "evaluation": {
            "n_episodes": 3,
            "split": "eval",
            "deterministic": True,
            "disturbance_window": [1.2, 1.8],
        },
        "scenario_splits": {
            "train": {"mass_values": [1.0], "damping_values": [0.5], "tau_values": [0.05]},
            "val": {"mass_values": [1.0, 1.1], "damping_values": [0.5], "tau_values": [0.05]},
            "test": {"mass_values": [1.3], "damping_values": [0.8], "tau_values": [0.08]},
        },
    }


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_config()
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {path} must be a mapping")
        cfg = _deep_update(cfg, loaded)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
