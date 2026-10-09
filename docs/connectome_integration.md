# Connectome integration

## Schema

Directed graph tables:

| File | Columns |
| --- | --- |
| nodes.csv | node_id, cell_type, region, neurotransmitter, provenance |
| edges.csv | source_id, target_id, synapse_count, provenance |

Missing annotations are empty strings. `synapse_count` is a **structural** measurement. It is not a recovered synaptic weight.

**Orientation:** an edge `i -> j` means neuron `i` influences neuron `j`. The recurrent mask stores this as `mask[j, i] = 1` so that `(W h)[j] += W[j, i] * h[i]`.

Self-loops are dropped by default. Duplicate directed edges are summed. Weakly connected giant-component extraction is **opt-in** (`component_policy="weakly_giant"`) because it can erase circuit boundaries.

## Synthetic fixture

`data/fixtures/synthetic_*.csv` is labeled **synthetic fixture — not biological connectome data**. Loaders propagate `provenance.synthetic=True`.

`residual.type: connectome` with only synthetic data raises unless `residual.allow_synthetic: true`.

This repository does not fabricate biological neuron IDs, dataset credentials, or downloads.

## Future MANC / neuPrint (or other) export

Suggested conversion (to be done by the experimenter, outside this repo’s defaults):

1. Export a documented neuron set (e.g., a named MANC / neuPrint query) to two tables matching the schema.
2. Map identifiers to `node_id` strings; keep original IDs in an extra file if needed.
3. Leave neurotransmitter blank unless a **written, justified** sign mapping is added later. Do not infer E/I from names by default.
4. Fill a provenance record:
   - dataset name and version
   - source URL or DOI
   - extraction criteria (cell classes, neuropils, synapse threshold)
   - extraction date
   - original vs retained node/edge counts
   - pruning or aggregation
   - missing-annotation summary
   - SHA-256 of `nodes.csv` and `edges.csv` (computed by `load_graph_tables`)
5. Store the export under `data/imports/<dataset>_<version>/` (git-lfs or external), not as if it were the synthetic fixture.

Optional weight init `log1p(synapse_count)` is an engineering assumption and must be compared with topology-only random init.

## Controls

Directed degree-preserving edge swaps (`connectome/transforms.py`) must preserve in- and out-degree, log rejected swaps, and verify the edge set actually changed.
