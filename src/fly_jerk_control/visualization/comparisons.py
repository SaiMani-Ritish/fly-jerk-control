"""Comparison plots across runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fly_jerk_control.visualization.trajectories import _title


def plot_performance_vs_disturbance(df: pd.DataFrame, path: Path, *, smoke_label: str = "") -> None:
    if "pulse_amplitude" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for ctrl, g in df.groupby("controller"):
        agg = g.groupby("pulse_amplitude")["success"].mean()
        ax.plot(agg.index, agg.values, marker="o", label=str(ctrl))
    ax.set_xlabel("pulse amplitude (N)")
    ax.set_ylabel("success rate")
    ax.set_title(_title("Success vs disturbance strength", smoke_label))
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
