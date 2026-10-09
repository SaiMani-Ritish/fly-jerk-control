"""Gymnasium mass-damper stopping environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fly_jerk_control.controllers.nominal import NominalTracker
from fly_jerk_control.envs.actuator import ActuatorConfig, limit_command
from fly_jerk_control.envs.disturbances import (
    apply_parameter_switch,
    disturbance_from_env_cfg,
    force_at_time,
    sample_episode_params,
)
from fly_jerk_control.envs.plant import PlantState, plant_acceleration, rk4_step
from fly_jerk_control.planners.quintic import QuinticPlanner
from fly_jerk_control.planners.ruckig_adapter import RuckigAdapter


RESIDUAL_OBS_KEYS = (
    "pos_error",
    "vel_error",
    "vel_meas",
    "a_ref",
    "prev_residual",
    "phase",
)


@dataclass
class StopTracker:
    pos_tol: float
    vel_tol: float
    dwell_time: float
    dt: float
    consecutive: int = 0
    first_success_time: float | None = None

    def update(self, x: float, v: float, x_target: float, t: float) -> bool:
        ok = abs(x - x_target) <= self.pos_tol and abs(v) <= self.vel_tol
        if ok:
            self.consecutive += 1
            needed = int(np.ceil(self.dwell_time / self.dt - 1e-12))
            needed = max(needed, 1)
            if self.consecutive >= needed and self.first_success_time is None:
                self.first_success_time = t
                return True
        else:
            self.consecutive = 0
        return False

    @property
    def succeeded(self) -> bool:
        return self.first_success_time is not None


@dataclass
class EpisodeLog:
    t: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    a: list[float] = field(default_factory=list)
    jerk: list[float] = field(default_factory=list)
    x_ref: list[float] = field(default_factory=list)
    v_ref: list[float] = field(default_factory=list)
    a_ref: list[float] = field(default_factory=list)
    j_ref: list[float] = field(default_factory=list)
    f_nom: list[float] = field(default_factory=list)
    f_res: list[float] = field(default_factory=list)
    f_req: list[float] = field(default_factory=list)
    f_cmd: list[float] = field(default_factory=list)
    f_act: list[float] = field(default_factory=list)
    f_dist: list[float] = field(default_factory=list)
    saturated: list[int] = field(default_factory=list)
    slew_limited: list[int] = field(default_factory=list)
    mass: list[float] = field(default_factory=list)


class MassStopEnv(gym.Env):
    """1-D stopping task with shared actuator constraints.

    Observations given to residual policies never include true mass, damping,
    future disturbance values, or noiseless hidden state.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict[str, Any], *, episode_index: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.episode_index = episode_index
        e = cfg["env"]
        self.dt = float(e["dt_ctrl"])
        self.n_substeps = int(e["n_substeps"])
        self.dt_sim = self.dt / self.n_substeps
        self.episode_duration = float(e["episode_duration"])
        self.x_min = float(e["x_min"])
        self.x_max = float(e["x_max"])
        self.x_target = float(e["x_target"])
        self.command_mode = str(e.get("command_mode", "residual"))
        self.residual_limit = float(cfg["residual"]["limit"])
        self.obs_noise_pos = float(e.get("obs_noise_pos", 0.0))
        self.obs_noise_vel = float(e.get("obs_noise_vel", 0.0))
        self.obs_delay_steps = int(e.get("obs_delay_steps", 0))
        self.log_substeps = bool(e.get("log_substeps", True))
        self.terminate_on_success = bool(e.get("terminate_on_success", False))
        self.reward_cfg = cfg["reward"]
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self._state: PlantState | None = None
        self._planner: QuinticPlanner | None = None
        self._nominal: NominalTracker | None = None
        self._log = EpisodeLog()
        self._delay_buf: deque[tuple[float, float]] = deque()
        self._prev_residual = 0.0
        self._prev_cmd = 0.0
        self._prev_a: float | None = None
        self._success = StopTracker(
            pos_tol=float(e["pos_tol"]),
            vel_tol=float(e["vel_tol"]),
            dwell_time=float(e["dwell_time"]),
            dt=self.dt,
        )
        self._params: dict[str, Any] = {}
        self._reward_parts: dict[str, float] = {}
        self._gave_success_bonus = False
        self.termination_reason = "not_started"

    def _build_planner(self) -> None:
        p = self.cfg["planner"]
        e = self.cfg["env"]
        ptype = str(p.get("type", "quintic")).lower()
        if ptype == "ruckig":
            self._planner = RuckigAdapter()  # raises by design
            return
        if ptype != "quintic":
            raise ValueError(f"Unknown planner type {ptype}")
        self._planner = QuinticPlanner(
            x0=float(e["x0"]),
            xf=float(e["x_target"]),
            duration=float(p["duration"]),
            v0=float(p.get("v0", 0.0)),
            a0=float(p.get("a0", 0.0)),
            vf=float(p.get("vf", 0.0)),
            af=float(p.get("af", 0.0)),
            v_max=p.get("v_max"),
            a_max=p.get("a_max"),
            j_max=p.get("j_max"),
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        if options and "episode_index" in options:
            self.episode_index = int(options["episode_index"])
        e = self.cfg["env"]
        dist_cfg = disturbance_from_env_cfg(self.cfg["disturbance"], int(self.cfg.get("seed", 0)), self.episode_index)
        split = (options or {}).get("split")
        if split:
            split_over = self.cfg.get("scenario_splits", {}).get(split, {})
            for key in ("mass_values", "damping_values", "tau_values"):
                if key in split_over:
                    setattr(dist_cfg, key, tuple(split_over[key]))
        self._params = sample_episode_params(dist_cfg)
        kind = self._params["kind"]
        if kind == "mass_change":
            pass  # mass already sampled from mass_values
        elif kind == "damping_change":
            pass
        elif kind == "tau_change":
            pass
        self._build_planner()
        n = self.cfg["nominal"]
        self._nominal = NominalTracker(
            mass=float(n["mass"]),
            damping=float(n["damping"]),
            kp=float(n["kp"]),
            kd=float(n["kd"]),
        )
        self._state = PlantState(
            x=float(e["x0"]),
            v=float(e["v0"]),
            f_act=0.0,
            t=0.0,
            mass=float(self._params["mass"]),
            damping=float(self._params["damping"]),
        )
        tau = float(self._params["tau_act"])
        self._actuator = ActuatorConfig(
            tau=tau,
            f_max=float(e["f_max"]),
            slew_max=float(e["slew_max"]),
        )
        self._log = EpisodeLog()
        self._prev_residual = 0.0
        self._prev_cmd = 0.0
        self._prev_a = None
        self._success = StopTracker(
            pos_tol=float(e["pos_tol"]),
            vel_tol=float(e["vel_tol"]),
            dwell_time=float(e["dwell_time"]),
            dt=self.dt,
        )
        self._gave_success_bonus = False
        self.termination_reason = "running"
        x_m, v_m = self._measure(self._state.x, self._state.v)
        self._delay_buf = deque([ (x_m, v_m) ] * (self.obs_delay_steps + 1), maxlen=self.obs_delay_steps + 1)
        self._append_log_point(self._state, 0.0, 0.0, 0.0, 0.0, False, False, 0.0)
        obs = self._residual_obs()
        return obs, self._info(truncated=False, terminated=False)

    def _measure(self, x: float, v: float) -> tuple[float, float]:
        x_m = x + float(self._rng.normal(0.0, self.obs_noise_pos)) if self.obs_noise_pos > 0 else x
        v_m = v + float(self._rng.normal(0.0, self.obs_noise_vel)) if self.obs_noise_vel > 0 else v
        return x_m, v_m

    def _delayed(self, x_m: float, v_m: float) -> tuple[float, float]:
        self._delay_buf.append((x_m, v_m))
        return self._delay_buf[0]

    def _residual_obs(self) -> np.ndarray:
        assert self._state is not None and self._planner is not None
        # Oldest buffer entry is the delayed measurement. With delay_steps=0, maxlen=1 so this is current.
        x_m, v_m = self._delay_buf[0]
        ref = self._planner.reference(self._state.t)
        duration = max(self._planner.coeff.duration, 1e-9)
        phase = float(np.clip(self._state.t / duration, 0.0, 1.0))
        obs = np.array(
            [
                x_m - ref.x,
                v_m - ref.v,
                v_m,
                ref.a,
                self._prev_residual,
                phase,
            ],
            dtype=np.float32,
        )
        scales = np.asarray(self.cfg["residual"].get("obs_scales", [1, 1, 1, 1, 1, 1]), dtype=np.float32)
        obs = obs / np.maximum(scales, 1e-6)
        return obs

    def _append_log_point(
        self,
        state: PlantState,
        f_nom: float,
        f_res: float,
        f_req: float,
        f_cmd: float,
        sat: bool,
        slew: bool,
        f_dist: float,
    ) -> None:
        a = plant_acceleration(state, f_dist)
        if self._prev_a is None:
            jerk = 0.0
        else:
            jerk = (a - self._prev_a) / self.dt_sim
        self._prev_a = a
        ref = self._planner.reference(state.t) if self._planner else None
        self._log.t.append(state.t)
        self._log.x.append(state.x)
        self._log.v.append(state.v)
        self._log.a.append(a)
        self._log.jerk.append(jerk)
        self._log.x_ref.append(ref.x if ref else 0.0)
        self._log.v_ref.append(ref.v if ref else 0.0)
        self._log.a_ref.append(ref.a if ref else 0.0)
        self._log.j_ref.append(ref.jerk if ref else 0.0)
        self._log.f_nom.append(f_nom)
        self._log.f_res.append(f_res)
        self._log.f_req.append(f_req)
        self._log.f_cmd.append(f_cmd)
        self._log.f_act.append(state.f_act)
        self._log.f_dist.append(f_dist)
        self._log.saturated.append(int(sat))
        self._log.slew_limited.append(int(slew))
        self._log.mass.append(state.mass)

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None and self._planner is not None and self._nominal is not None
        action_f = float(np.asarray(action).reshape(-1)[0])
        ref = self._planner.reference(self._state.t)
        x_m, v_m = self._delay_buf[0]
        f_nom = self._nominal.force(x_m, v_m, ref)
        if self.command_mode == "force":
            f_res = 0.0
            f_req = action_f
        else:
            f_res = self.residual_limit * float(np.tanh(action_f))
            f_req = f_nom + f_res
        lim = limit_command(f_req, self._prev_cmd, self.dt, self._actuator)
        self._prev_cmd = lim.f_limited
        prev_res_for_reward = self._prev_residual
        self._prev_residual = f_res

        for _ in range(self.n_substeps):
            self._state.mass = apply_parameter_switch(self._state.t, self._state.mass, self._params)
            self._state = rk4_step(
                self._state,
                self.dt_sim,
                lim.f_limited,
                lambda tt, p=self._params: force_at_time(tt, p),
                self._actuator,
            )
            f_dist = force_at_time(self._state.t, self._params)
            if self.log_substeps:
                self._append_log_point(
                    self._state, f_nom, f_res, f_req, lim.f_limited, lim.saturated, lim.slew_limited, f_dist
                )
        if not self.log_substeps:
            f_dist = force_at_time(self._state.t, self._params)
            self._append_log_point(
                self._state, f_nom, f_res, f_req, lim.f_limited, lim.saturated, lim.slew_limited, f_dist
            )

        x_meas, v_meas = self._measure(self._state.x, self._state.v)
        self._delayed(x_meas, v_meas)
        just_succeeded = self._success.update(self._state.x, self._state.v, self.x_target, self._state.t)

        unsafe = self._state.x < self.x_min or self._state.x > self.x_max
        time_up = self._state.t >= self.episode_duration - 1e-12
        terminated = bool(unsafe)
        truncated = bool(time_up and not unsafe)
        if unsafe:
            self.termination_reason = "unsafe"
        elif self.terminate_on_success and self._success.succeeded:
            terminated = True
            truncated = False
            self.termination_reason = "success"
        elif time_up:
            self.termination_reason = "success" if self._success.succeeded else "timeout"
        else:
            self.termination_reason = "running"

        reward = self._reward(f_res, prev_res_for_reward, just_succeeded, unsafe)
        obs = self._residual_obs()
        return obs, reward, terminated, truncated, self._info(terminated=terminated, truncated=truncated)

    def _reward(self, f_res: float, prev_residual: float, just_succeeded: bool, unsafe: bool) -> float:
        assert self._state is not None and self._planner is not None
        rc = self.reward_cfg
        ref = self._planner.reference(self._state.t)
        pos_err = self._state.x - ref.x
        vel_err = self._state.v - ref.v
        dres = f_res - prev_residual
        parts = {
            "pos": -float(rc["w_pos"]) * pos_err**2,
            "vel": -float(rc["w_vel"]) * vel_err**2,
            "residual": -float(rc["w_residual"]) * f_res**2,
            "dresidual": -float(rc["w_dresidual"]) * dres**2,
            "success": float(rc["success_bonus"]) if just_succeeded and not self._gave_success_bonus else 0.0,
            "unsafe": -float(rc["unsafe_penalty"]) if unsafe else 0.0,
        }
        if parts["success"] > 0:
            self._gave_success_bonus = True
        self._reward_parts = parts
        return float(sum(parts.values()))

    def _info(self, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        assert self._planner is not None and self._state is not None
        feas = self._planner.feasibility()
        return {
            "termination_reason": self.termination_reason,
            "success": bool(self._success.succeeded),
            "success_time": self._success.first_success_time,
            "t": self._state.t,
            "x": self._state.x,
            "v": self._state.v,
            "f_act": self._state.f_act,
            "mass": self._state.mass,
            "damping": self._state.damping,
            "tau_act": self._actuator.tau,
            "disturbance_kind": self._params.get("kind"),
            "disturbance_semantics": self._params.get("semantics"),
            "planner_feasible": feas.feasible,
            "planner_reasons": list(feas.reasons),
            "reward_parts": dict(self._reward_parts),
            "command_mode": self.command_mode,
            "episode_log": self._log if (terminated or truncated) else None,
            "true_mass_hidden": True,
        }

    def traces(self) -> EpisodeLog:
        return self._log
