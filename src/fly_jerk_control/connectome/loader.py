"""Load and hash connectome tables. Never fabricates biological IDs or downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fly_jerk_control.connectome.schema import (
    EDGE_COLUMNS,
    NODE_COLUMNS,
    GraphProvenance,
    GraphTables,
    empty_provenance,
)
from fly_jerk_control.connectome.synthetic import SYNTHETIC_LABEL, is_synthetic_frame
from fly_jerk_control.connectome.validation import validate_graph_tables


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; required {columns}")
    return df[list(columns)].copy()


def load_graph_tables(
    nodes_path: str | Path,
    edges_path: str | Path,
    *,
    provenance: GraphProvenance | None = None,
    allow_self_loops: bool = False,
    duplicate_policy: str = "sum",
) -> GraphTables:
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    nodes = _read_csv(nodes_path, NODE_COLUMNS)
    edges = _read_csv(edges_path, EDGE_COLUMNS)
    synthetic = is_synthetic_frame(nodes) or is_synthetic_frame(edges)
    prov = provenance or empty_provenance(
        synthetic=synthetic,
        name=SYNTHETIC_LABEL if synthetic else nodes_path.stem,
        source=str(nodes_path.parent),
    )
    prov.nodes_sha256 = sha256_file(nodes_path)
    prov.edges_sha256 = sha256_file(edges_path)
    prov.synthetic = synthetic
    if synthetic:
        prov.dataset_name = SYNTHETIC_LABEL
        prov.notes = (prov.notes + " " + SYNTHETIC_LABEL).strip()
    tables = GraphTables(nodes=nodes, edges=edges, provenance=prov)
    return validate_graph_tables(
        tables,
        allow_self_loops=allow_self_loops,
        duplicate_policy=duplicate_policy,
    )
