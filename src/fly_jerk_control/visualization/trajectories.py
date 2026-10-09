"""Saved diagnostic plots. Smoke outputs must be labeled as such."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fly_jerk_control.envs.mass_stop_env import EpisodeLog


def _title(base: str, smoke_label: str) -> str:
    if smoke_label:
        return f"{base} [{smoke_label}]"
    return base


def plot_trajectories(log: EpisodeLog, path: Path, *, smoke_label: str = "") -> None:
    t = np.asarray(log.t)
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(t, log.x, label="x")
    axes[0].plot(t, log.x_ref, "--", label="x_ref")
    axes[0].set_ylabel("position (m)")
    axes[0].legend()
    axes[0].set_title(_title("Position", smoke_label))
    axes[1].plot(t, log.v, label="v")
    axes[1].plot(t, log.v_ref, "--", label="v_ref")
    axes[1].set_ylabel("velocity (m/s)")
    axes[2].plot(t, log.a, label="a")
    axes[2].plot(t, log.a_ref, "--", label="a_ref")
    axes[2].set_ylabel("acceleration (m/s^2)")
    axes[3].plot(t, log.jerk, label="jerk")
    axes[3].plot(t, log.j_ref, "--", label="j_ref")
    axes[3].set_ylabel("jerk (m/s^3)")
    axes[3].set_xlabel("time (s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forces(log: EpisodeLog, path: Path, *, smoke_label: str = "") -> None:
    t = np.asarray(log.t)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, log.f_nom, label="F_nominal")
    ax.plot(t, log.f_res, label="F_residual")
    ax.plot(t, log.f_req, label="F_requested")
    ax.plot(t, log.f_cmd, label="F_cmd_limited")
    ax.plot(t, log.f_act, label="F_act")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force (N)")
    ax.set_title(_title("Forces (shared actuator path)", smoke_label))
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_success_vs_jerk(df: pd.DataFrame, path: Path, *, smoke_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for ctrl, g in df.groupby("controller"):
        ax.scatter(g["integrated_squared_jerk"], g["success"].astype(int), label=str(ctrl), alpha=0.7)
    ax.set_xlabel("integrated squared jerk")
    ax.set_ylabel("success (1/0)")
    ax.set_title(_title("Task success vs jerk (not a ranking)", smoke_label))
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_learning_curves(df: pd.DataFrame, path: Path, *, smoke_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "seed" in df.columns:
        for seed, g in df.groupby("seed"):
            ax.plot(g["steps"], g["ep_return_mean"], label=f"seed {seed}")
    else:
        ax.plot(df["steps"], df["ep_return_mean"])
    ax.set_xlabel("env steps")
    ax.set_ylabel("episode return (training)")
    ax.set_title(_title("Learning curves (not scientific results)", smoke_label))
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
