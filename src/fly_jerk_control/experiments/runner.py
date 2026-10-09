"""Experiment runner: evaluate, train (tiny), compare."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fly_jerk_control.config import dump_config
from fly_jerk_control.envs.mass_stop_env import MassStopEnv
from fly_jerk_control.evaluation.aggregation import episodes_to_frame, per_seed_summary, rank_controllers
from fly_jerk_control.evaluation.diagnostics import diagnose
from fly_jerk_control.evaluation.metrics import metrics_to_dict
from fly_jerk_control.evaluation.rollout import rollout_episode
from fly_jerk_control.experiments.manifests import timestamp_run_dir, write_manifest
from fly_jerk_control.factory import build_residual
from fly_jerk_control.training.checkpoints import load_checkpoint
from fly_jerk_control.training.ppo import train_ppo
from fly_jerk_control.visualization.comparisons import plot_performance_vs_disturbance
from fly_jerk_control.visualization.trajectories import (
    plot_forces,
    plot_learning_curves,
    plot_success_vs_jerk,
    plot_trajectories,
)


def _write_report(run_dir: Path, lines: list[str]) -> None:
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_run(cfg: dict[str, Any], *, root: Path | None = None, residual=None) -> Path:
    root = root or Path.cwd()
    run_dir = timestamp_run_dir(cfg.get("output_root", "outputs"), cfg.get("run_name", "eval"))
    dump_config(cfg, run_dir / "resolved_config.yaml")
    write_manifest(run_dir, cfg, extra={"mode": "evaluate"})
    env = MassStopEnv(cfg)
    ctrl = residual if residual is not None else build_residual(cfg, root=root)
    rows = []
    diag_lines = ["# Evaluation report", ""]
    if cfg.get("label"):
        diag_lines.append(f"**Label:** {cfg['label']}")
        diag_lines.append("")
    diag_lines.append("This report is not a scientific claim. Numbers are from the configured run only.")
    diag_lines.append("")
    n = int(cfg.get("evaluation", {}).get("n_episodes", 1))
    split = str(cfg.get("evaluation", {}).get("split", "eval"))
    det = bool(cfg.get("evaluation", {}).get("deterministic", True))
    for i in range(n):
        metrics, extra = rollout_episode(
            env,
            ctrl,
            deterministic=det,
            seed=int(cfg.get("seed", 0)),
            episode_index=i,
            split=split,
        )
        rows.append(metrics_to_dict(metrics))
        log = extra["log"]
        if i == 0:
            plot_trajectories(log, run_dir / "plots" / "trajectory.png", smoke_label=str(cfg.get("label", "")))
            plot_forces(log, run_dir / "plots" / "forces.png", smoke_label=str(cfg.get("label", "")))
            pd.DataFrame({k: getattr(log, k) for k in vars(log)}).to_csv(
                run_dir / "trajectories" / "episode_000.csv", index=False
            )
        deg = diagnose(log, metrics, float(cfg["env"]["x0"]), float(cfg["env"]["pos_tol"]))
        diag_lines.append(
            f"- episode {i}: success={metrics.success} reason={metrics.termination_reason} "
            f"degeneracy={deg.notes}"
        )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "episode_metrics.csv", index=False)
    per_seed_summary(df).to_csv(run_dir / "aggregate_metrics.csv", index=False)
    rank_controllers(df).to_csv(run_dir / "ranking_success_first.csv", index=False)
    plot_success_vs_jerk(df, run_dir / "plots" / "success_vs_jerk.png", smoke_label=str(cfg.get("label", "")))
    _write_report(run_dir, diag_lines)
    return run_dir


def train_run(cfg: dict[str, Any], *, root: Path | None = None) -> Path:
    root = root or Path.cwd()
    run_dir = timestamp_run_dir(cfg.get("output_root", "outputs"), cfg.get("run_name", "train"))
    dump_config(cfg, run_dir / "resolved_config.yaml")
    write_manifest(run_dir, cfg, extra={"mode": "train"})
    env = MassStopEnv(cfg)
    ctrl = build_residual(cfg, root=root)
    summary = train_ppo(env, ctrl, cfg, run_dir)
    curves = pd.DataFrame(summary["curves"])
    if not curves.empty:
        plot_learning_curves(curves, run_dir / "plots" / "learning_curves.png", smoke_label=str(cfg.get("label", "")))
    # evaluate with the trained controller; reset recurrent state in rollout
    eval_dir_note = evaluate_into(run_dir, cfg, ctrl, root=root)
    _write_report(
        run_dir,
        [
            "# Training report",
            "",
            f"Label: {cfg.get('label')}",
            f"Updates: {summary['updates']}",
            f"Steps: {summary['steps']}",
            "This is a training utility run, not a biological finding.",
            f"Post-train evaluation artifacts: {eval_dir_note}",
        ],
    )
    return run_dir


def evaluate_into(run_dir: Path, cfg: dict[str, Any], ctrl, *, root: Path | None) -> str:
    env = MassStopEnv(cfg)
    rows = []
    n = int(cfg.get("evaluation", {}).get("n_episodes", 1))
    for i in range(n):
        metrics, extra = rollout_episode(
            env,
            ctrl,
            deterministic=True,
            seed=int(cfg.get("seed", 0)),
            episode_index=i,
            split=str(cfg.get("evaluation", {}).get("split", "eval")),
        )
        rows.append(metrics_to_dict(metrics))
        if i == 0:
            plot_trajectories(extra["log"], run_dir / "plots" / "post_train_trajectory.png", smoke_label=str(cfg.get("label", "")))
    pd.DataFrame(rows).to_csv(run_dir / "post_train_episode_metrics.csv", index=False)
    return "post_train_episode_metrics.csv"


def evaluate_checkpoint(checkpoint: Path, cfg: dict[str, Any], *, root: Path | None = None) -> Path:
    ctrl = build_residual(cfg, root=root)
    load_checkpoint(checkpoint, ctrl)  # type: ignore[arg-type]
    cfg = dict(cfg)
    cfg["run_name"] = cfg.get("run_name", "eval_ckpt")
    return evaluate_run(cfg, root=root, residual=ctrl)


def compare_runs(run_dirs: list[Path], output: Path | None = None) -> Path:
    frames = []
    for d in run_dirs:
        csv_path = Path(d) / "episode_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        df["run_dir"] = str(d)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    out = output or (Path(run_dirs[0]).parent / "compare")
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "episode_metrics.csv", index=False)
    rank_controllers(all_df).to_csv(out / "ranking_success_first.csv", index=False)
    plot_success_vs_jerk(all_df, out / "success_vs_jerk.png", smoke_label="comparison")
    plot_performance_vs_disturbance(all_df, out / "vs_disturbance.png", smoke_label="comparison")
    (out / "report.md").write_text(
        "\n".join(
            [
                "# Comparison",
                "",
                "Success rate is ranked first. A low-jerk failure is not a winner.",
                "This comparison is only as meaningful as the listed run directories.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out
