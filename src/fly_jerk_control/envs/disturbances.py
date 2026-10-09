"""External force and parameter-switch disturbances.

Smooth pulses are preferred for the primary smoothness benchmark so that
controller-induced jerk can be distinguished from unavoidable pulse edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def raised_cosine_pulse(t: float, t0: float, duration: float, amplitude: float) -> float:
    """C1-smooth pulse: 0.5 * A * (1 - cos(2 pi (t-t0)/T)) on [t0, t0+T]."""
    if duration <= 0.0:
        return 0.0
    if t < t0 or t > t0 + duration:
        return 0.0
    phase = 2.0 * np.pi * (t - t0) / duration
    return 0.5 * amplitude * (1.0 - np.cos(phase))


@dataclass
class DisturbanceConfig:
    kind: str = "none"
    pulse_t0: float = 1.2
    pulse_duration: float = 0.4
    pulse_amplitude: float = 4.0
    mass_values: tuple[float, ...] = (1.0,)
    damping_values: tuple[float, ...] = (0.5,)
    tau_values: tuple[float, ...] = (0.05,)
    within_episode_mass_switch: bool = False
    mass_switch_time: float = 1.5
    mass_after_switch: float = 1.4
    episode_index: int = 0
    seed: int = 0


def sample_episode_params(cfg: DisturbanceConfig) -> dict[str, Any]:
    """Deterministic per-episode plant parameters from configured discrete sets."""
    rng = np.random.default_rng(int(cfg.seed) + 10007 * int(cfg.episode_index) + 17)
    mass = float(rng.choice(np.asarray(cfg.mass_values, dtype=float)))
    damping = float(rng.choice(np.asarray(cfg.damping_values, dtype=float)))
    tau = float(rng.choice(np.asarray(cfg.tau_values, dtype=float)))
    kind = cfg.kind
    return {
        "kind": kind,
        "mass": mass,
        "damping": damping,
        "tau_act": tau,
        "within_episode_mass_switch": bool(cfg.within_episode_mass_switch),
        "mass_switch_time": float(cfg.mass_switch_time),
        "mass_after_switch": float(cfg.mass_after_switch),
        "pulse_t0": float(cfg.pulse_t0),
        "pulse_duration": float(cfg.pulse_duration),
        "pulse_amplitude": float(cfg.pulse_amplitude),
        "semantics": (
            "idealized_parameter_switch"
            if cfg.within_episode_mass_switch
            else "episode_level_parameter"
        ),
    }


def force_at_time(t: float, params: dict[str, Any]) -> float:
    kind = params.get("kind", "none")
    if kind in ("none", "sensor_noise", "obs_delay", "mass_change", "damping_change", "tau_change"):
        return 0.0
    if kind in ("force_pulse", "smooth_pulse"):
        return raised_cosine_pulse(
            t,
            float(params["pulse_t0"]),
            float(params["pulse_duration"]),
            float(params["pulse_amplitude"]),
        )
    raise ValueError(f"Unknown disturbance kind: {kind}")


def apply_parameter_switch(t: float, state_mass: float, params: dict[str, Any]) -> float:
    """Apply idealized mass switch at substep boundaries if enabled."""
    if not params.get("within_episode_mass_switch", False):
        return state_mass
    if t + 1e-12 >= float(params["mass_switch_time"]):
        return float(params["mass_after_switch"])
    return state_mass


def disturbance_from_env_cfg(env_disturbance: dict[str, Any], seed: int, episode_index: int) -> DisturbanceConfig:
    return DisturbanceConfig(
        kind=str(env_disturbance.get("kind", "none")),
        pulse_t0=float(env_disturbance.get("pulse_t0", 1.2)),
        pulse_duration=float(env_disturbance.get("pulse_duration", 0.4)),
        pulse_amplitude=float(env_disturbance.get("pulse_amplitude", 4.0)),
        mass_values=tuple(env_disturbance.get("mass_values", [1.0])),
        damping_values=tuple(env_disturbance.get("damping_values", [0.5])),
        tau_values=tuple(env_disturbance.get("tau_values", [0.05])),
        within_episode_mass_switch=bool(env_disturbance.get("within_episode_mass_switch", False)),
        mass_switch_time=float(env_disturbance.get("mass_switch_time", 1.5)),
        mass_after_switch=float(env_disturbance.get("mass_after_switch", 1.4)),
        episode_index=int(episode_index),
        seed=int(seed),
    )


ForceFn = Callable[[float], float]
