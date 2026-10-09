# Experiment registry

Status values: `harness_only` (code exists, no scientific result), `blocked` (cannot run as labeled), `not_started`.

| ID | Config | Question | Status |
| --- | --- | --- | --- |
| M1-smoke | `configs/smoke.yaml` | Does the nominal pipeline stop on the reachable task? | harness_only |
| M1-baseline | `configs/baseline.yaml` | Zero residual under a smooth force pulse | harness_only |
| M2-topology | `configs/topology_ablation.yaml` | Graph vs shuffled vs random recurrent (synthetic fixture unless real tables provided) | harness_only |
| M3-train-smoke | `configs/train_smoke.yaml` | Tiny PPO smoke; checkpoint reload | harness_only |
| M4-holdout | `configs/generalization.yaml` | Held-out mass/damping/tau split | harness_only |
| BIO-connectome | user-supplied nodes/edges | Same as research question with a real annotated export | blocked until a non-synthetic graph with provenance exists |

Do not paste smoke-test scalars into papers or slides as connectome results.

When a real export is added, record dataset, version, hashes, extraction criteria, and the git commit of this package in the run `manifest.json`.
