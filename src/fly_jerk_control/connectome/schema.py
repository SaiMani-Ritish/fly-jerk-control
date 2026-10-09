"""Generic directed connectome table schema.

Missing annotations are allowed (empty strings). Synapse counts are structural
measurements, not recovered functional weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


NODE_COLUMNS = ("node_id", "cell_type", "region", "neurotransmitter", "provenance")
EDGE_COLUMNS = ("source_id", "target_id", "synapse_count", "provenance")


@dataclass
class GraphProvenance:
    dataset_name: str
    source: str
    version: str
    extraction_criteria: str
    extraction_date: str
    original_n_nodes: int
    original_n_edges: int
    retained_n_nodes: int
    retained_n_edges: int
    pruning_or_aggregation: str
    missing_annotations: str
    nodes_sha256: str
    edges_sha256: str
    synthetic: bool
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class GraphTables:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    provenance: GraphProvenance
    node_index: dict[str, int] = field(default_factory=dict)

    def ordered_node_ids(self) -> list[str]:
        return [str(x) for x in self.nodes["node_id"].tolist()]


def empty_provenance(*, synthetic: bool, name: str, source: str) -> GraphProvenance:
    return GraphProvenance(
        dataset_name=name,
        source=source,
        version="unspecified",
        extraction_criteria="unspecified",
        extraction_date="unspecified",
        original_n_nodes=0,
        original_n_edges=0,
        retained_n_nodes=0,
        retained_n_edges=0,
        pruning_or_aggregation="none",
        missing_annotations="allowed",
        nodes_sha256="",
        edges_sha256="",
        synthetic=synthetic,
        notes="",
    )
