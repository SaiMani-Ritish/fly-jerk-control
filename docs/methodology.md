# Methodology

## Plant and integrator

State `[x, v, F_act]` is integrated with classical RK4. The limited force command is zero-order held over each control interval (`dt_ctrl`) with `n_substeps` simulation steps.

```
m dv/dt = F_act - b v + F_disturbance
dx/dt = v
tau_act dF_act/dt = F_cmd_limited - F_act
```

If `tau_act <= 0`, `F_act` is set equal to `F_cmd_limited` (algebraic, lag disabled).

**Within-episode mass change** (optional) is an idealized parameter switch: `v` is continuous, `m` jumps at a configured time at substep boundaries. This is **not** a payload-pickup model.

Slew limits constrain `F_cmd`. They do **not** imply a hard plant jerk bound under disturbances or parameter switches.

## Planner and nominal control

The first planner is a quintic polynomial with specified start/end position, velocity, and acceleration. Analytic jerk is reported for the **reference**. Feasibility checks compare sampled peaks against configured `v_max`, `a_max`, `j_max`. Feasibility is not a plant jerk guarantee.

Nominal force:

`F_nominal = m_nom a_ref + b_nom v_ref + Kp (x_ref - x) + Kd (v_ref - v)`

`m_nom` and `b_nom` are the controller’s assumed plant, not necessarily truth.

Ruckig is optional and **not** implemented. Selecting `planner.type: ruckig` errors; it never silently substitutes quintic under a Ruckig label.

## Residual and observations

`F_residual = residual_limit * tanh(raw)`  
`F_cmd_requested = F_nominal + F_residual` then shared magnitude and slew limits.

Observations (fixed scales in config; not fit on test data):

1. Position error (measured − reference)
2. Velocity error
3. Measured velocity
4. Reference acceleration
5. Previous residual command
6. Phase `clip(t / T_move, 0, 1)`

The **reward** may use true `x, v` (privileged training signal). That does not put true mass or future disturbance into the policy observation.

## Stop criterion

Success: `|x - x_target| < pos_tol` and `|v| < vel_tol` held for `dwell_time` at the control rate. Evaluation default does **not** terminate early on success (full horizon metrics). Unsafe position bounds terminate the episode (`terminated=True`, reason `unsafe`). Time limit truncates (`truncated=True`, reason `timeout` or `success` if dwell already met).

A controller that stays at `x=0` fails. Early unsafe termination is never a smoothness win: success rate is ranked first; post-stop oscillation is NaN if dwell never occurred.

## Plant jerk

Plant acceleration is `(F_act - b v + F_dist) / m`. Plant jerk is a forward finite difference of that acceleration at `dt_sim`. Settings are stored on each metrics row. Reference jerk is analytic.

Full-episode and configured disturbance-window jerk are both reported. No undocumented smoothing.

## PPO

MLP: clipped PPO, separate critic, GAE, shuffled minibatches of transitions.

Recurrent/graph: same GAE, but policy/value updates use **truncated BPTT windows** along the collected trajectory with stored hidden states and episode-boundary resets. Masked graph weights are multiplied by the adjacency mask every update.

Training / validation / held-out splits are named in config (`scenario_splits`). Do not tune on `test`.

## Statistics

Training **seeds** are the independent units for learned models. Many episodes from one checkpoint are repeated measures, not replications. `evaluation/aggregation.py` provides per-seed tables and a hierarchical bootstrap helper.
