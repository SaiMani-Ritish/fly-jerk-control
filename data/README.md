# Connectome data

This directory holds **fixtures and user-supplied exports**. The repository does not download or fabricate biological neuron IDs.

## Schema

`nodes.csv`:
`node_id,cell_type,region,neurotransmitter,provenance`

`edges.csv`:
`source_id,target_id,synapse_count,provenance`

Empty annotation fields are allowed. `synapse_count` must be finite and nonnegative.

## Synthetic fixture

`fixtures/synthetic_*.csv` is labeled:

**synthetic fixture — not biological connectome data**

A `residual.type: connectome` experiment **refuses** to run on synthetic data unless `residual.allow_synthetic: true` is set for an explicit smoke test.

## Importing a future MANC / neuPrint export

See `docs/connectome_integration.md`. Record dataset name, version, source, extraction criteria, date, sizes, pruning policy, missing annotations, and SHA-256 hashes of the imported files.
