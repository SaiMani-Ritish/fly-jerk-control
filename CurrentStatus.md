### Research Repository Built: `fly-jerk-control`

The research repository [fly-jerk-control](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control) has been created. The codebase provides an experimental pipeline for investigating whether connectome-informed recurrent residual controllers improve smooth, accurate stopping under changing dynamics compared with matched nonbiological residual controllers.

---

### 1. What Was Implemented

#### **Milestone 1 — Baseline Physical Simulation & Control Pipeline**
- **1D Mass-Damper Plant** ([plant.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/envs/plant.py)): Continuous RK4 numerical integration with zero-order hold force actuation and safety bounds.
- **Actuator Dynamics** ([actuator.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/envs/actuator.py)): Exact first-order lag solution ($\tau_{\text{act}} \cdot \dot{F}_{\text{act}} = F_{\text{cmd}} - F_{\text{act}}$) with slew-rate limiting and force saturation bounds. All controllers share this identical actuator pipeline.
- **Disturbance Modeling** ([disturbances.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/envs/disturbances.py)): Smooth Gaussian force pulses, parameter step changes (mass/damping switches), and observation noise.
- **Quintic Trajectory Planner** ([quintic.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/planners/quintic.py)): Analytic quintic boundary-condition solver computing position, velocity, acceleration, and analytic reference jerk.
- **Tracking Controllers**:
  - `NominalController` ([nominal.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/controllers/nominal.py)): Model-based feedforward + PD tracking.
  - `ComposedController` ([composed.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/controllers/composed.py)): $F_{\text{cmd}} = F_{\text{nominal}} + F_{\text{residual}}$, logging force saturation and slew events.
- **Task-Validity Gated Metrics** ([metrics.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/evaluation/metrics.py)): Finite-difference plant jerk, settling dwell time, post-stop oscillation (marked `NaN` on failed episodes), endpoint errors, and force saturation fraction.

#### **Milestone 2 — Graph Pipeline & Anti-Fabrication Data Integration**
- **Data Schema** ([schema.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/connectome/schema.py)): `nodes.csv` and `edges.csv` definitions with SHA-256 provenance tracking.
- **Synthetic Test Fixtures** ([synthetic.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/connectome/synthetic.py)): Explicitly tagged with `"synthetic fixture — not biological connectome data"`.
- **Safety Loader Guard** ([loader.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/connectome/loader.py)): Refuses to execute real connectome experiments without explicit `--allow-synthetic` flags when test data is present.
- **Directed Degree-Preserving Randomization** ([transforms.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/connectome/transforms.py)): Double-edge swap algorithm strictly preserving individual in-degrees and out-degrees as a matched topological null control.
- **Graph Recurrent Residual Controller** ([graph_recurrent.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/controllers/graph_recurrent.py)): Masked recurrent neural network ($h_{t+1} = (1-\alpha)h_t + \alpha \tanh((W_{\text{rec}} \odot M)h_t + W_{\text{in}}x_t + b)$) with gradient masking hooks.

#### **Milestone 3 — PPO Learning & Training Engine**
- **PPO Trainer** ([ppo.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/training/ppo.py)): Trajectory-aware PPO with separate Value Critic, GAE advantage calculation ([buffers.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/training/buffers.py)), fixed observation scaling, and transparent reward decomposition.
- **Checkpoints** ([checkpoints.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/training/checkpoints.py)): PyTorch model weights and JSON metadata serialization.

#### **Milestone 4 — Scientific Comparison & Visualization**
- **Experiment Runner** ([runner.py](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/src/fly_jerk_control/experiments/runner.py)): Multi-seed comparison across 5 matched controller architectures (`nominal_only`, `mlp_residual`, `random_rnn`, `shuffled_graph`, `connectome_graph`) with separate resource accounting (units, directed edges, parameters).
- **Interactive Visualizer Dashboard** ([dashboard.html](file:///C:/Users/sriha/.gemini/antigravity/brain/066675b7-c3ed-4295-b2aa-ac42b1a89314/dashboard.html)): Self-contained dashboard with real-time physics waveforms, disturbance sliders, connectome graph visualizer, and trade-off comparison table.

---

### 2. Commands You Can Run

All commands are runnable from the repository root:

```powershell
# 1. Run minimal smoke test (verifies nominal success & stationary failure)
python -m fly_jerk_control.cli smoke

# 2. Evaluate baseline across disturbance scenarios
python -m fly_jerk_control.cli evaluate-baseline --config configs/baseline.yaml

# 3. Validate connectome CSV schema & verify degree-preserving shuffle
python -m fly_jerk_control.cli validate-graph --nodes data/fixtures/synthetic_nodes.csv --edges data/fixtures/synthetic_edges.csv --shuffle-test

# 4. Train a residual controller with PPO (e.g. MLP or Graph RNN)
python -m fly_jerk_control.cli train --controller-type mlp --timesteps 2000 --residual-limit 2.0
python -m fly_jerk_control.cli train --controller-type graph_recurrent --timesteps 2000 --residual-limit 2.0

# 5. Evaluate a trained checkpoint
python -m fly_jerk_control.cli evaluate --checkpoint outputs/train_mlp_.../checkpoint.pt --n-episodes 10

# 6. Run full multi-seed comparative benchmark across all matched architectures
python -m fly_jerk_control.cli compare --config configs/topology_ablation.yaml --allow-synthetic

# 7. Run full test suite
python -m pytest -v
```

---

### 3. Test Suite Execution Results

All **65 unit and integration tests** pass:

```
tests/test_milestone1.py (51 tests):
  ✓ Plant stationary state, acceleration, damping steady-state, convergence, safety bounds, time advancement
  ✓ Actuator force saturation, slew limiting, first-order lag, zero tau, saturation fraction
  ✓ Quintic boundary conditions, non-zero velocities, analytic jerk, feasibility checks
  ✓ Environment reset/step, stationary failure, nominal success, safety violations, timeout truncation
  ✓ Dwell tolerance logic & NaN post-stop oscillation flags for failed episodes
  ✓ Residual controllers (MLP, Vanilla RNN hidden state reset/persistence, Graph RNN mask & gradient hook)
  ✓ Disturbance pulse, step force, sensor noise, and seed reproducibility

tests/test_milestone2.py (10 tests):
  ✓ Synthetic fixture creation & explicit warning labels
  ✓ Missing node endpoints and negative synapse rejection
  ✓ Directed degree-preserving shuffle (exact in/out-degree equality verification)
  ✓ Connectome loader safety guard blocking unflagged synthetic experiments
  ✓ Synaptic weight normalization into Graph RNN

tests/test_milestone3.py (4 tests):
  ✓ TrajectoryBuffer GAE return estimation
  ✓ Checkpoint serialization & weight restoration
  ✓ PPO training smoke test for MLP and Graph RNN residual controllers
```

---

### 4. Empirical Smoke & Benchmark Outcomes

#### **CLI Smoke Test Output (`outputs/smoke_...`)**
- Nominal Controller: **3/3 Episodes Succeeded** (Endpoint position error: $0.0005\,\text{m}$, Peak Jerk: $8.45\,\text{m/s}^3$).
- Stationary Zero-Force Controller: **Correctly Failed** with `timeout` (never reached target dwell).

#### **Comparative Benchmark Summary (`outputs/topology_ablation_...`)**

| Controller | Architecture Type | Units | Edges | Trainable Params | Stopping Success | Mean Settling Time | Peak Jerk | Integrated Jerk | Pos Error |
|---|---|---|---|---|---|---|---|---|---|
| `nominal_only` | Feedforward + PD | 0 | 0 | 0 | **100%** | $0.110\,\text{s}$ | $8.45\,\text{m/s}^3$ | $5.27$ | $0.0005\,\text{m}$ |
| `mlp_residual` | Dense MLP [16,16] | 32 | 368 | 401 | **100%** | $0.110\,\text{s}$ | $10.07\,\text{m/s}^3$ | $4.85$ | $0.0003\,\text{m}$ |
| `random_rnn` | Fully-Connected RNN | 20 | 520 | 560 | **100%** | $0.110\,\text{s}$ | $9.28\,\text{m/s}^3$ | $6.49$ | $0.0004\,\text{m}$ |
| `shuffled_graph` | Shuffled Null Control | 20 | 101 | 261 | **100%** | $0.100\,\text{s}$ | $7.65\,\text{m/s}^3$ | $7.42$ | $0.0025\,\text{m}$ |
| `connectome_graph` | Graph Recurrent (Untrained) | 20 | 101 | 261 | **0%** | **N/A** | $28.21\,\text{m/s}^3$ | $80.38$ | $0.0206\,\text{m}$ |

**Key Diagnostic Finding**:
The untrained connectome graph with raw synaptic initial values commands a residual bias that pushes position error ($0.0206\,\text{m}$) outside the strict $0.01\,\text{m}$ tolerance band. Under our task-validity gating, it is reported as **0% success** with settling time `N/A`, preventing an inaccurate controller from appearing smooth.

---

### 5. Known Limitations & Engineering Assumptions

1. **Synapse Count Functional Mapping**: Biological synapse counts measure physical contact area/counts, not signed synaptic conductance or functional weights. Normalized $\log(1 + \text{synapses})$ is an initialization heuristic, not a recovered biological circuit weight.
2. **1D Lumped Plant**: The current environment isolates 1D mass-damper dynamics. Multi-joint kinematic coupling (such as Drosophila leg or wing hinge mechanics) is deferred to future extensions.
3. **Idealized Parameter Switches**: Changes in mass or damping across episodes represent idealized parameter variations rather than continuous payload-pickup dynamics.

---

### 6. Requirements Before Real Biological Connectome Experiments

To run experiments with Drosophila datasets (such as MANC or Hemibrain):
1. **Biological Export Schema**: Place `nodes.csv` and `edges.csv` from an annotated export in `data/connectome/` following the format documented in [connectome_integration.md](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/docs/connectome_integration.md).
2. **Metadata Registration**: Record the dataset name, release version (e.g. `MANC_v1.0`), extraction criteria, and SHA-256 hashes in [experiment_registry.md](file:///C:/Users/sriha/.gemini/antigravity/scratch/fly-jerk-control/docs/experiment_registry.md).
3. **Run Validation**: Execute `python -m fly_jerk_control.cli validate-graph --nodes data/connectome/nodes.csv --edges data/connectome/edges.csv --shuffle-test` to ensure topological consistency before launching training runs.