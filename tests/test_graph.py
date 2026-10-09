"""Graph schema, orientation, masking, shuffle, provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fly_jerk_control.config import default_config
from fly_jerk_control.connectome.loader import load_graph_tables
from fly_jerk_control.connectome.schema import GraphTables
from fly_jerk_control.connectome.synthetic import SYNTHETIC_LABEL, make_synthetic_tables
from fly_jerk_control.connectome.transforms import adjacency_mask, directed_degree_preserving_shuffle
from fly_jerk_control.connectome.validation import GraphValidationError, directed_degrees, validate_graph_tables
from fly_jerk_control.controllers.graph_recurrent import GraphRecurrentResidual
from fly_jerk_control.factory import build_residual


def test_synthetic_label() -> None:
    t = make_synthetic_tables()
    assert SYNTHETIC_LABEL in t.provenance.dataset_name
    assert t.provenance.synthetic


def test_schema_rejects_bad_endpoints(tmp_path) -> None:
    nodes = tmp_path / "n.csv"
    edges = tmp_path / "e.csv"
    nodes.write_text("node_id,cell_type,region,neurotransmitter,provenance\na,,,," + SYNTHETIC_LABEL + "\n")
    edges.write_text(
        "source_id,target_id,synapse_count,provenance\na,missing,1," + SYNTHETIC_LABEL + "\n"
    )
    with pytest.raises(GraphValidationError):
        load_graph_tables(nodes, edges)


def test_schema_rejects_negative_synapses() -> None:
    t = make_synthetic_tables()
    t.edges.loc[0, "synapse_count"] = -1
    with pytest.raises(GraphValidationError):
        validate_graph_tables(t)


def test_edge_orientation_mask() -> None:
    nodes = pd.DataFrame(
        {
            "node_id": ["i", "j"],
            "cell_type": ["", ""],
            "region": ["", ""],
            "neurotransmitter": ["", ""],
            "provenance": [SYNTHETIC_LABEL, SYNTHETIC_LABEL],
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": ["i"],
            "target_id": ["j"],
            "synapse_count": [3.0],
            "provenance": [SYNTHETIC_LABEL],
        }
    )
    tables = validate_graph_tables(GraphTables(nodes=nodes, edges=edges, provenance=make_synthetic_tables().provenance))
    mask = adjacency_mask(tables)
    # edge i -> j => mask[j, i] = 1
    assert mask[1, 0] == 1.0
    assert mask[0, 1] == 0.0


def test_masked_weights_do_not_affect_forward() -> None:
    t = validate_graph_tables(make_synthetic_tables(6))
    model = GraphRecurrentResidual(t, architecture="graph_recurrent")
    obs = np.zeros(6, dtype=np.float32)
    model.reset()
    raw1, _ = model.raw_action(obs)
    with torch.no_grad():
        # Write garbage onto masked-out entries.
        model.w_graph.add_(1.0 - model.mask)
    model.reset()
    raw2, _ = model.raw_action(obs)
    assert abs(raw1 - raw2) < 1e-6
    # Gradients on masked weights remain ~0.
    model.reset()
    obs_t = torch.zeros(6)
    h = torch.zeros(model.n_units)
    h2 = model.step_hidden(obs_t, h)
    loss = h2.sum()
    loss.backward()
    masked_grad = model.w_graph.grad * (1.0 - model.mask)
    assert float(masked_grad.abs().max()) < 1e-8


def test_degree_preserving_shuffle() -> None:
    t = validate_graph_tables(make_synthetic_tables(8))
    shuffled, report = directed_degree_preserving_shuffle(t, seed=1, n_swaps=50)
    assert report.out_degree_preserved
    assert report.in_degree_preserved
    assert report.changed
    o0, i0 = directed_degrees(t)
    o1, i1 = directed_degrees(shuffled)
    assert o0 == o1 and i0 == i1


def test_connectome_refuses_synthetic_without_flag() -> None:
    cfg = default_config()
    cfg["residual"]["type"] = "connectome"
    cfg["residual"]["allow_synthetic"] = False
    cfg["residual"]["graph_nodes"] = "data/fixtures/synthetic_nodes.csv"
    cfg["residual"]["graph_edges"] = "data/fixtures/synthetic_edges.csv"
    with pytest.raises(RuntimeError, match="synthetic"):
        build_residual(cfg)


def test_recurrent_reset_no_leak() -> None:
    t = validate_graph_tables(make_synthetic_tables(6))
    model = GraphRecurrentResidual(t)
    obs = np.ones(6, dtype=np.float32)
    model.reset()
    for _ in range(5):
        model.raw_action(obs, deterministic=True)
    h_mid = model.get_recurrent_state().clone()
    model.reset()
    assert torch.allclose(model.get_recurrent_state(), torch.zeros_like(h_mid))
    raw_a, _ = model.raw_action(obs)
    model.reset()
    raw_b, _ = model.raw_action(obs)
    assert abs(raw_a - raw_b) < 1e-7
