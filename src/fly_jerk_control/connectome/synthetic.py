"""Synthetic fixture graph — not biological connectome data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fly_jerk_control.connectome.schema import GraphProvenance, GraphTables, empty_provenance

SYNTHETIC_LABEL = "synthetic fixture — not biological connectome data"


def is_synthetic_frame(df: pd.DataFrame) -> bool:
    if "provenance" not in df.columns:
        return False
    return bool(df["provenance"].astype(str).str.contains("synthetic fixture", case=False).any())


def is_synthetic_graph(tables: GraphTables) -> bool:
    return bool(tables.provenance.synthetic) or is_synthetic_frame(tables.nodes) or is_synthetic_frame(tables.edges)


def make_synthetic_tables(n: int = 8) -> GraphTables:
    """Small directed chain-plus-feedback graph for tests."""
    node_ids = [f"syn_{i:02d}" for i in range(n)]
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "cell_type": [""] * n,
            "region": ["fixture"] * n,
            "neurotransmitter": [""] * n,
            "provenance": [SYNTHETIC_LABEL] * n,
        }
    )
    edges_src = []
    edges_tgt = []
    syn = []
    for i in range(n - 1):
        edges_src.append(node_ids[i])
        edges_tgt.append(node_ids[i + 1])
        syn.append(float(i + 1))
    # A few extra edges so shuffling is nontrivial.
    extra = [(0, 3), (2, 5), (5, 1), (7, 4), (3, 6)]
    for a, b in extra:
        if a < n and b < n:
            edges_src.append(node_ids[a])
            edges_tgt.append(node_ids[b])
            syn.append(2.0)
    edges = pd.DataFrame(
        {
            "source_id": edges_src,
            "target_id": edges_tgt,
            "synapse_count": syn,
            "provenance": [SYNTHETIC_LABEL] * len(edges_src),
        }
    )
    prov = empty_provenance(synthetic=True, name=SYNTHETIC_LABEL, source="in-memory fixture")
    prov.original_n_nodes = n
    prov.original_n_edges = len(edges)
    prov.retained_n_nodes = n
    prov.retained_n_edges = len(edges)
    prov.notes = SYNTHETIC_LABEL
    return GraphTables(nodes=nodes, edges=edges, provenance=prov)


def write_synthetic_fixture(directory: str | Path, n: int = 8) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = make_synthetic_tables(n)
    nodes_path = directory / "synthetic_nodes.csv"
    edges_path = directory / "synthetic_edges.csv"
    tables.nodes.to_csv(nodes_path, index=False)
    tables.edges.to_csv(edges_path, index=False)
    return nodes_path, edges_path
