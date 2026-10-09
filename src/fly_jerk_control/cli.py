"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from fly_jerk_control.config import load_config
from fly_jerk_control.connectome.loader import load_graph_tables
from fly_jerk_control.connectome.synthetic import SYNTHETIC_LABEL, is_synthetic_graph
from fly_jerk_control.connectome.transforms import directed_degree_preserving_shuffle
from fly_jerk_control.connectome.validation import component_summary
from fly_jerk_control.experiments.runner import compare_runs, evaluate_checkpoint, evaluate_run, train_run


def _root() -> Path:
    return Path.cwd()


def cmd_smoke(_args: argparse.Namespace) -> int:
    cfg_path = _root() / "configs" / "smoke.yaml"
    cfg = load_config(cfg_path)
    cfg["label"] = "SMOKE TEST — not a research result"
    cfg["run_name"] = "smoke"
    run_dir = evaluate_run(cfg, root=_root())
    print(f"Smoke evaluation wrote {run_dir}")
    return 0


def cmd_evaluate_baseline(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.setdefault("run_name", "baseline")
    run_dir = evaluate_run(cfg, root=_root())
    print(run_dir)
    return 0


def cmd_validate_graph(args: argparse.Namespace) -> int:
    tables = load_graph_tables(args.nodes, args.edges)
    summary = component_summary(tables)
    print("provenance:", tables.provenance.as_dict())
    print("components:", summary)
    if is_synthetic_graph(tables):
        print("WARNING:", SYNTHETIC_LABEL)
    shuffled, report = directed_degree_preserving_shuffle(tables, seed=0)
    print("shuffle:", report)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.setdefault("run_name", "train")
    run_dir = train_run(cfg, root=_root())
    print(run_dir)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    run_dir = evaluate_checkpoint(Path(args.checkpoint), cfg, root=_root())
    print(run_dir)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    out = compare_runs([Path(p) for p in args.runs])
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fly_jerk_control")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="Nominal zero-residual smoke evaluation").set_defaults(func=cmd_smoke)
    b = sub.add_parser("evaluate-baseline")
    b.add_argument("--config", default="configs/baseline.yaml")
    b.set_defaults(func=cmd_evaluate_baseline)
    g = sub.add_parser("validate-graph")
    g.add_argument("--nodes", required=True)
    g.add_argument("--edges", required=True)
    g.set_defaults(func=cmd_validate_graph)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--config", required=True)
    e.set_defaults(func=cmd_evaluate)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
