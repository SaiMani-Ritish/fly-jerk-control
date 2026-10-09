# Limitations

- One-dimensional mass-damper stopping is not fly locomotion and not a whole-brain model.
- Continuous leaky-integrator units are not spiking neurons. STDP / LIF / neuromodulation are not implemented.
- Nominal PD+FF with a quintic reference can succeed on the reachable `0 → 1 m` task; that is a **sanity check**, not evidence for connectome topology.
- Matched “neuron count” is not matched parameter count. Resource reports include units, directed edges, trainable parameters, and inference latency.
- Reward terms are not identical to evaluation jerk. A direct-jerk-penalty run would be a named ablation, not the default.
- Observation delay is a discrete buffer of noisy measurements, not a biophysical sensor model.
- Parameter-switch mass changes are idealized and must not be described as grasping a payload.
- Recurrent PPO uses truncated BPTT on a single collected trajectory (batch size 1 in the first prototype). Scale and variance estimates will be limited until multi-env collection exists.
- Hierarchical bootstrap is provided as a tool; default smoke runs do not constitute inference.
- No JAS metric from prior work is implemented; none was specified.
- Optional Ruckig, LIF graph controllers, plasticity, and multi-joint arms are out of scope until Milestones 1–4 are solid.
