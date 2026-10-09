"""Adjacency masks, synapse-based init, and directed degree-preserving shuffle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fly_jerk_control.connectome.schema import GraphTables
from fly_jerk_control.connectome.validation import directed_degrees, validate_graph_tables


def adjacency_mask(tables: GraphTables) -> np.ndarray:
    """mask[j, i] = 1 if edge i -> j (neuron i influences neuron j)."""
    n = len(tables.nodes)
    idx = tables.node_index or {str(nid): i for i, nid in enumerate(tables.nodes["node_id"].astype(str))}
    mask = np.zeros((n, n), dtype=np.float32)
    for _, row in tables.edges.iterrows():
        i = idx[str(row["source_id"])]
        j = idx[str(row["target_id"])]
        mask[j, i] = 1.0
    return mask


def init_weights_from_synapses(tables: GraphTables) -> np.ndarray:
    """Normalized log1p(synapse_count) on allowed edges.

    Engineering assumption only. Not a claim that synapse counts are functional weights.
    """
    n = len(tables.nodes)
    idx = tables.node_index or {str(nid): i for i, nid in enumerate(tables.nodes["node_id"].astype(str))}
    w = np.zeros((n, n), dtype=np.float32)
    for _, row in tables.edges.iterrows():
        i = idx[str(row["source_id"])]
        j = idx[str(row["target_id"])]
        w[j, i] = np.log1p(float(row["synapse_count"]))
    scale = float(np.max(np.abs(w))) or 1.0
    return w / scale


@dataclass
class ShuffleReport:
    n_attempted: int
    n_swaps: int
    n_rejected: int
    changed: bool
    in_degree_preserved: bool
    out_degree_preserved: bool
    seed: int
    notes: str


def directed_degree_preserving_shuffle(
    tables: GraphTables,
    *,
    n_swaps: int | None = None,
    seed: int = 0,
    max_attempts_factor: int = 20,
) -> tuple[GraphTables, ShuffleReport]:
    """Maslov–Sneppen style directed edge swaps preserving in- and out-degree.

    Rejects swaps that would create existing edges or self-loops.
    """
    rng = np.random.default_rng(seed)
    edges = tables.edges.copy().reset_index(drop=True)
    edge_list = [(str(s), str(t)) for s, t in zip(edges["source_id"], edges["target_id"])]
    syn = {(str(s), str(t)): float(c) for s, t, c in zip(edges["source_id"], edges["target_id"], edges["synapse_count"])}
    prov_map = {(str(s), str(t)): str(p) for s, t, p in zip(edges["source_id"], edges["target_id"], edges["provenance"])}
    m = len(edge_list)
    if m < 2:
        report = ShuffleReport(0, 0, 0, False, True, True, seed, "too few edges")
        return tables, report
    target_swaps = n_swaps if n_swaps is not None else 10 * m
    max_attempts = max(target_swaps * max_attempts_factor, target_swaps)
    attempted = 0
    swapped = 0
    rejected = 0
    original = set(edge_list)
    edge_set = set(edge_list)
    while swapped < target_swaps and attempted < max_attempts:
        attempted += 1
        i, j = rng.choice(m, size=2, replace=False)
        a, b = edge_list[i]
        c, d = edge_list[j]
        if len({a, b, c, d}) < 4:
            rejected += 1
            continue
        new1, new2 = (a, d), (c, b)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            rejected += 1
            continue
        if new1 in edge_set or new2 in edge_set:
            rejected += 1
            continue
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(new1)
        edge_set.add(new2)
        syn[new1] = syn.pop((a, b))
        syn[new2] = syn.pop((c, d))
        prov_map[new1] = prov_map.pop((a, b))
        prov_map[new2] = prov_map.pop((c, d))
        edge_list[i] = new1
        edge_list[j] = new2
        swapped += 1
    new_edges = pd.DataFrame(
        {
            "source_id": [e[0] for e in edge_list],
            "target_id": [e[1] for e in edge_list],
            "synapse_count": [syn[e] for e in edge_list],
            "provenance": [prov_map[e] for e in edge_list],
        }
    )
    shuffled = GraphTables(
        nodes=tables.nodes.copy(),
        edges=new_edges,
        provenance=deepcopy(tables.provenance),
        node_index=dict(tables.node_index),
    )
    shuffled.provenance.notes = (
        shuffled.provenance.notes + f" degree-preserving shuffle seed={seed} swaps={swapped}"
    ).strip()
    shuffled = validate_graph_tables(shuffled, allow_self_loops=False, duplicate_policy="error")
    out0, in0 = directed_degrees(tables)
    out1, in1 = directed_degrees(shuffled)
    report = ShuffleReport(
        n_attempted=attempted,
        n_swaps=swapped,
        n_rejected=rejected,
        changed=set(edge_list) != original,
        in_degree_preserved=in0 == in1,
        out_degree_preserved=out0 == out1,
        seed=seed,
        notes=f"unsuccessful_swaps={rejected}",
    )
    return shuffled, report
