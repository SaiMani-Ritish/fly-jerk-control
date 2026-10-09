# Research plan and implementation acceptance criteria

**Project:** Fly-Connectome-Informed Residual Control for Smooth Motion Under Changing Dynamics

**Question:** Does a small fly-connectome-informed recurrent residual controller improve smooth, accurate stopping under changing dynamics compared with matched nonbiological residual controllers?

This repository does **not** re-test the prior capstone hypothesis (cerebellar SNN + Hebbian STDP vs quintic baseline). Results are hypotheses until experiments are run. Smoke outputs are labeled and are not scientific findings.

## Architecture (fixed)

Target and constraints → trajectory planner → nominal tracker.
Measured state and tracking errors → residual controller.
Nominal force + residual force → shared actuator limits and lag → simulated plant.

The residual is bounded: `F_residual = residual_limit * tanh(raw)`.
All controllers share the same actuator pipeline.

## Milestone 1 — Baseline pipeline (this increment)

Deliver: installable package, 1-D mass-damper stop env, RK4 plant + lagged actuator, quintic planner, PD+FF nominal tracker, metrics, smoke CLI, tests.

Acceptance:

1. A reachable nominal task (`x: 0 → 1 m`) stops successfully under the documented tolerances and dwell.
2. A stationary policy that holds the initial position fails the stop task.
3. Early unsafe termination is never ranked as a smoothness win (success rate is reported first; failed episodes do not get a dummy oscillation of 0).
4. Smoke evaluation writes timestamped outputs labeled as smoke tests, reproducibly for a fixed seed.

## Milestone 2 — Graph pipeline

Synthetic fixture only until a real export exists. Schema validation, masked graph RNN, degree-preserving shuffle, provenance refusal for “biological” runs without real data.

## Milestone 3 — Learning

PPO for MLP residuals. Recurrent PPO on sequences with truncated BPTT (not shuffled i.i.d. transitions). Tiny smoke train only; no long runs by default.

## Milestone 4 — Scientific comparison harness

Matched configs, train/val/held-out splits, per-seed aggregation, degeneracy diagnostics. No overstated biology.

## Milestone 5 — deferred

Ruckig, LIF, plasticity, multi-joint arm: stubs or explicit “not implemented,” not silent stand-ins.

## Engineering decisions (Milestone 1)

- Integrator: classical RK4 on `[x, v, F_act]` with zero-order hold on limited `F_cmd` over each control interval; parameter switches applied at substep boundaries.
- Within-episode mass change (when enabled) is an **idealized parameter switch**: velocity is continuous; mass jumps. This is **not** a payload-pickup model.
- Quintic references can violate configured jerk/velocity/acceleration limits; the planner reports feasibility rather than claiming a hard jerk bound.
- Force slew limits constrain the **command**, not a guaranteed plant jerk bound under disturbances or mass switches.
- Evaluation prefers full-horizon rollouts except on unsafe termination, so early success does not shrink integrated metrics by truncation.
- Jerk: analytic for the reference; plant jerk is a finite difference of the plant acceleration trace `(F_act - b v + F_dist) / m`.
