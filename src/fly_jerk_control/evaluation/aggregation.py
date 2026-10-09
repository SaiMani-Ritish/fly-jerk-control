"""Per-seed aggregation. Episodes from one checkpoint are not independent train replications."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from fly_jerk_control.evaluation.metrics import EpisodeMetrics, metrics_to_dict


def episodes_to_frame(metrics: Iterable[EpisodeMetrics]) -> pd.DataFrame:
    return pd.DataFrame([metrics_to_dict(m) for m in metrics])


def rank_controllers(df: pd.DataFrame, smoothness_col: str = "integrated_squared_jerk") -> pd.DataFrame:
    """Success rate first. Low-jerk failures are not winners."""
    rows = []
    for ctrl, g in df.groupby("controller"):
        success_rate = float(g["success"].mean())
        succ = g.loc[g["success"]]
        rows.append(
            {
                "controller": ctrl,
                "n_episodes": int(len(g)),
                "n_seeds": int(g["seed"].nunique()),
                "success_rate": success_rate,
                "smoothness_successful_only": float(succ[smoothness_col].mean()) if len(succ) else float("nan"),
                "smoothness_unconditional": float(g[smoothness_col].mean()),
                "n_success": int(len(succ)),
                "n_fail": int((~g["success"].astype(bool)).sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["success_rate", "smoothness_successful_only"], ascending=[False, True])


def hierarchical_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap over groups (training seeds), resampling episodes within seed."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        sample = []
        for g in chosen:
            idx = np.flatnonzero(groups == g)
            draw = rng.choice(idx, size=len(idx), replace=True)
            sample.append(values[draw])
        boots.append(float(np.nanmean(np.concatenate(sample))))
    boots_a = np.asarray(boots, dtype=float)
    lo = float(np.nanpercentile(boots_a, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots_a, 100 * (1 - alpha / 2)))
    return {"mean": float(np.nanmean(values)), "ci_low": lo, "ci_high": hi, "n_groups": int(len(uniq))}


def per_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["controller", "seed", "split"]
    agg = {
        "success": "mean",
        "peak_abs_jerk": "mean",
        "integrated_squared_jerk": "mean",
        "settling_time": "mean",
        "endpoint_position_error": "mean",
        "n_trainable_params": "first",
        "n_units": "first",
        "n_directed_edges": "first",
    }
    present = {k: v for k, v in agg.items() if k in df.columns}
    return df.groupby(keys, dropna=False).agg(present).reset_index()
