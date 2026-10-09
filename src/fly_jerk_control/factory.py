"""Build residual controllers from config without silent graph/planner substitutions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fly_jerk_control.connectome.loader import load_graph_tables
from fly_jerk_control.connectome.synthetic import SYNTHETIC_LABEL, is_synthetic_graph, make_synthetic_tables
from fly_jerk_control.connectome.transforms import directed_degree_preserving_shuffle
from fly_jerk_control.connectome.validation import validate_graph_tables
from fly_jerk_control.controllers.graph_recurrent import GraphRecurrentResidual
from fly_jerk_control.controllers.mlp import MLPResidual
from fly_jerk_control.controllers.recurrent import RecurrentResidual
from fly_jerk_control.controllers.residual_base import ResidualController, ZeroResidual


def build_residual(cfg: dict[str, Any], *, root: Path | None = None) -> ResidualController:
    r = cfg["residual"]
    kind = str(r.get("type", "zero")).lower()
    obs_dim = 6
    log_std = float(cfg.get("training", {}).get("log_std_init", -0.5))
    if kind == "zero":
        return ZeroResidual()
    if kind == "mlp":
        return MLPResidual(obs_dim=obs_dim, hidden_sizes=list(r.get("hidden_sizes", [32, 32])), log_std_init=log_std)
    if kind == "recurrent":
        n = int(r.get("n_units", 8))
        return RecurrentResidual(obs_dim=obs_dim, n_units=n, alpha=float(r.get("alpha", 0.5)), log_std_init=log_std)
    if kind in ("graph", "shuffled_graph", "connectome"):
        return _build_graph_residual(cfg, kind, root=root, log_std=log_std)
    raise ValueError(f"Unknown residual.type={kind}")


def _resolve(path: str, root: Path | None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = root if root is not None else Path.cwd()
    return (base / p).resolve()


def _build_graph_residual(
    cfg: dict[str, Any],
    kind: str,
    *,
    root: Path | None,
    log_std: float,
) -> ResidualController:
    r = cfg["residual"]
    allow_synthetic = bool(r.get("allow_synthetic", False))
    nodes = r.get("graph_nodes")
    edges = r.get("graph_edges")
    if not nodes or not edges:
        if allow_synthetic:
            tables = make_synthetic_tables(int(r.get("n_units", 8)))
            tables = validate_graph_tables(tables)
        else:
            raise RuntimeError(
                "No graph files provided. A biological-connectome run requires nodes.csv and "
                f"edges.csv. Synthetic fixtures are labeled {SYNTHETIC_LABEL!r} and are not a substitute."
            )
    else:
        tables = load_graph_tables(_resolve(str(nodes), root), _resolve(str(edges), root))
    if kind == "connectome" and is_synthetic_graph(tables) and not allow_synthetic:
        raise RuntimeError(
            "Real-connectome experiment refused: only synthetic fixture data is available. "
            f"Graph is labeled {SYNTHETIC_LABEL!r}. Re-run with residual.allow_synthetic=true "
            "only for an explicit synthetic smoke test."
        )
    arch = {
        "graph": "graph_recurrent",
        "shuffled_graph": "shuffled_graph",
        "connectome": "connectome",
    }[kind]
    if kind == "shuffled_graph":
        tables, report = directed_degree_preserving_shuffle(tables, seed=int(cfg.get("seed", 0)))
        if not report.in_degree_preserved or not report.out_degree_preserved:
            raise RuntimeError(f"Shuffle failed to preserve degree: {report}")
    return GraphRecurrentResidual(
        tables,
        obs_dim=6,
        alpha=float(r.get("alpha", 0.5)),
        log_std_init=log_std,
        init=str(r.get("init", "topology_random")),
        architecture=arch,
        allow_synthetic=allow_synthetic or kind != "connectome",
    )


def random_dense_mask(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=np.float32)
