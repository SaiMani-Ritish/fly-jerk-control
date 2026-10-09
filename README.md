# Fly-Connectome-Informed Residual Control for Smooth Motion Under Changing Dynamics

Independent research repository. It does **not** re-test the prior capstone SNN/STDP hypothesis.

**Question:** Does a small fly-connectome-informed recurrent residual controller improve smooth, accurate stopping under changing dynamics compared with matched nonbiological residual controllers?

Treat controller comparisons as **hypotheses**. A negative result is a valid outcome. Smoke-test numbers are labeled and are not findings.

## Setup (Python 3.11+)

```powershell
cd fly-jerk-control
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python -m fly_jerk_control.cli smoke
```

## Commands

```text
python -m fly_jerk_control.cli smoke
python -m fly_jerk_control.cli evaluate-baseline --config configs/baseline.yaml
python -m fly_jerk_control.cli validate-graph --nodes data/fixtures/synthetic_nodes.csv --edges data/fixtures/synthetic_edges.csv
python -m fly_jerk_control.cli train --config configs/train_smoke.yaml
python -m fly_jerk_control.cli evaluate --checkpoint <path-to.pt> --config configs/train_smoke.yaml
python -m fly_jerk_control.cli compare --runs outputs/<run_a> outputs/<run_b>
```

Do not launch long training runs from the default configs. `configs/train_smoke.yaml` is intentionally tiny.

## Architecture

Target → quintic planner → nominal PD+FF tracker.  
Tracking errors → bounded residual (`limit * tanh(raw)`).  
Nominal + residual → **shared** actuator limits and lag → mass-damper plant (RK4).

Residual observations: position error, velocity error, measured velocity, reference acceleration, previous residual, phase. True mass, damping, future disturbances, and noiseless hidden state are not exposed.

## Data policy

`data/fixtures/` is a **synthetic fixture — not biological connectome data**. A `connectome` residual type refuses synthetic data unless `residual.allow_synthetic: true`.

## Docs

- `docs/research_plan.md` — milestones and acceptance criteria
- `docs/methodology.md` — plant, metrics, training
- `docs/connectome_integration.md` — schema and future imports
- `docs/limitations.md`
- `docs/experiment_registry.md`
