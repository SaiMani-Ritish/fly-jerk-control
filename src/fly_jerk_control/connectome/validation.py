"""Graph validation: IDs, endpoints, orientation, synapse counts, components."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from fly_jerk_control.connectome.schema import GraphTables


class GraphValidationError(ValueError):
    pass


def validate_graph_tables(
    tables: GraphTables,
    *,
    allow_self_loops: bool = False,
    duplicate_policy: str = "sum",
    component_policy: str = "keep_all",
) -> GraphTables:
    """Validate and normalize tables.

    component_policy:
      - keep_all: do not drop nodes (default). Circuit boundaries are preserved.
      - weakly_giant: keep the largest weakly connected component (explicit opt-in).
    """
    nodes = tables.nodes.copy()
    edges = tables.edges.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    if nodes["node_id"].duplicated().any():
        dups = nodes.loc[nodes["node_id"].duplicated(), "node_id"].tolist()
        raise GraphValidationError(f"Duplicate node IDs: {dups[:10]}")
    ids = set(nodes["node_id"])
    unknown_src = set(edges["source_id"]) - ids
    unknown_tgt = set(edges["target_id"]) - ids
    if unknown_src or unknown_tgt:
        raise GraphValidationError(
            f"Edge endpoints missing from nodes.csv: sources={sorted(unknown_src)[:10]} "
            f"targets={sorted(unknown_tgt)[:10]}"
        )
    syn = pd.to_numeric(edges["synapse_count"], errors="coerce")
    if syn.isna().any() or not np.isfinite(syn.to_numpy(dtype=float)).all():
        raise GraphValidationError("synapse_count must be finite")
    if (syn < 0).any():
        raise GraphValidationError("synapse_count must be nonnegative")
    edges["synapse_count"] = syn.astype(float)
    self_loops = edges["source_id"] == edges["target_id"]
    if self_loops.any() and not allow_self_loops:
        edges = edges.loc[~self_loops].copy()
        tables.provenance.notes = (
            tables.provenance.notes + " self-loops dropped (allow_self_loops=False)"
        ).strip()
    key = ["source_id", "target_id"]
    dup_mask = edges.duplicated(key, keep=False)
    if dup_mask.any():
        if duplicate_policy == "sum":
            agg = {c: "first" for c in edges.columns if c not in key + ["synapse_count"]}
            agg["synapse_count"] = "sum"
            edges = edges.groupby(key, as_index=False).agg(agg)
            tables.provenance.notes = (
                tables.provenance.notes + " duplicate edges summed"
            ).strip()
        elif duplicate_policy == "error":
            raise GraphValidationError("Duplicate directed edges present")
        else:
            raise GraphValidationError(f"Unknown duplicate_policy {duplicate_policy}")
    if component_policy == "weakly_giant":
        g = to_networkx(nodes, edges)
        giant = max(nx.weakly_connected_components(g), key=len)
        nodes = nodes.loc[nodes["node_id"].isin(giant)].copy()
        edges = edges.loc[edges["source_id"].isin(giant) & edges["target_id"].isin(giant)].copy()
        tables.provenance.pruning_or_aggregation = "weakly_giant"
        tables.provenance.notes = (
            tables.provenance.notes
            + " EXPLICIT weakly_giant extraction; not applied by default because it can remove circuit boundaries."
        ).strip()
    elif component_policy != "keep_all":
        raise GraphValidationError(f"Unknown component_policy {component_policy}")
    nodes = nodes.reset_index(drop=True)
    tables.nodes = nodes
    tables.edges = edges.reset_index(drop=True)
    tables.node_index = {nid: i for i, nid in enumerate(nodes["node_id"].astype(str))}
    tables.provenance.retained_n_nodes = int(len(nodes))
    tables.provenance.retained_n_edges = int(len(edges))
    if tables.provenance.original_n_nodes == 0:
        tables.provenance.original_n_nodes = int(len(nodes))
        tables.provenance.original_n_edges = int(len(edges))
    return tables


def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes["node_id"].astype(str).tolist())
    for _, row in edges.iterrows():
        g.add_edge(str(row["source_id"]), str(row["target_id"]), synapse_count=float(row["synapse_count"]))
    return g


def component_summary(tables: GraphTables) -> dict[str, int]:
    g = to_networkx(tables.nodes, tables.edges)
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_weak": nx.number_weakly_connected_components(g),
        "n_strong": nx.number_strongly_connected_components(g),
    }


def directed_degrees(tables: GraphTables) -> tuple[Counter, Counter]:
    out_deg: Counter = Counter()
    in_deg: Counter = Counter()
    for nid in tables.nodes["node_id"].astype(str):
        out_deg[nid] = 0
        in_deg[nid] = 0
    for _, row in tables.edges.iterrows():
        out_deg[str(row["source_id"])] += 1
        in_deg[str(row["target_id"])] += 1
    return out_deg, in_deg
