"""Connectome-derived graph residual (same encoder/decoder as other graph models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fly_jerk_control.connectome.loader import load_graph_tables
from fly_jerk_control.connectome.schema import GraphTables
from fly_jerk_control.connectome.synthetic import SYNTHETIC_LABEL, is_synthetic_graph
from fly_jerk_control.connectome.transforms import adjacency_mask, init_weights_from_synapses
from fly_jerk_control.controllers.recurrent import RecurrentResidual


class GraphRecurrentResidual(RecurrentResidual):
    def __init__(
        self,
        tables: GraphTables,
        obs_dim: int = 6,
        alpha: float = 0.5,
        log_std_init: float = -0.5,
        init: str = "topology_random",
        device: str = "cpu",
        architecture: str = "graph_recurrent",
        allow_synthetic: bool = True,
    ) -> None:
        if is_synthetic_graph(tables) and architecture == "connectome" and not allow_synthetic:
            raise RuntimeError(
                "Refusing a biological-connectome controller: the loaded graph is "
                f"labeled {SYNTHETIC_LABEL!r}. Pass a real export or set residual.allow_synthetic "
                "only for explicit synthetic smoke tests."
            )
        mask = adjacency_mask(tables)
        super().__init__(
            obs_dim=obs_dim,
            n_units=mask.shape[0],
            alpha=alpha,
            log_std_init=log_std_init,
            mask=mask,
            device=device,
            architecture=architecture,
        )
        if init == "log1p_synapses":
            w0 = init_weights_from_synapses(tables)
            with torch.no_grad():
                self.w_graph.copy_(torch.from_numpy(w0).to(self.w_graph.device))
                self.w_graph.mul_(self.mask)
            self._init_note = (
                "Engineering assumption: initial W_graph from normalized log1p(synapse_count). "
                "Synapse counts are structural measurements, not recovered functional weights."
            )
        else:
            self._init_note = "Topology-only random initialization on allowed edges."
        self.tables = tables

    def resource_report(self):
        report = super().resource_report()
        report.notes = f"{report.notes} {self._init_note} provenance={self.tables.provenance.dataset_name}"
        return report


def load_graph_residual_from_paths(
    nodes: str | Path,
    edges: str | Path,
    **kwargs: Any,
) -> GraphRecurrentResidual:
    tables = load_graph_tables(nodes, edges)
    return GraphRecurrentResidual(tables, **kwargs)
