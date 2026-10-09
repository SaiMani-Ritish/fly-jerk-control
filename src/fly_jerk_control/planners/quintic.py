"""Quintic reference planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_jerk_control.planners.base import Planner, Reference, TrajectoryFeasibility


@dataclass(frozen=True)
class QuinticCoefficients:
    a: tuple[float, float, float, float, float, float]
    duration: float


def _solve_coefficients(
    x0: float,
    v0: float,
    a0: float,
    xf: float,
    vf: float,
    af: float,
    duration: float,
) -> QuinticCoefficients:
    T = float(duration)
    if T <= 0.0:
        raise ValueError("Quintic duration must be positive")
    a0c, a1, a2 = float(x0), float(v0), 0.5 * float(a0)
    A = np.array(
        [
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ],
        dtype=float,
    )
    b = np.array(
        [
            xf - (x0 + v0 * T + 0.5 * a0 * T**2),
            vf - (v0 + a0 * T),
            af - a0,
        ],
        dtype=float,
    )
    a3, a4, a5 = np.linalg.solve(A, b)
    return QuinticCoefficients(a=(a0c, a1, a2, float(a3), float(a4), float(a5)), duration=T)


def eval_quintic(coeff: QuinticCoefficients, t: float) -> Reference:
    T = coeff.duration
    tt = float(np.clip(t, 0.0, T))
    a0, a1, a2, a3, a4, a5 = coeff.a
    x = a0 + a1 * tt + a2 * tt**2 + a3 * tt**3 + a4 * tt**4 + a5 * tt**5
    v = a1 + 2 * a2 * tt + 3 * a3 * tt**2 + 4 * a4 * tt**3 + 5 * a5 * tt**4
    acc = 2 * a2 + 6 * a3 * tt + 12 * a4 * tt**2 + 20 * a5 * tt**3
    jerk = 6 * a3 + 24 * a4 * tt + 60 * a5 * tt**2
    if t > T:
        # Hold terminal rest/accel after the motion window.
        x = a0 + a1 * T + a2 * T**2 + a3 * T**3 + a4 * T**4 + a5 * T**5
        v = a1 + 2 * a2 * T + 3 * a3 * T**2 + 4 * a4 * T**3 + 5 * a5 * T**4
        acc = 2 * a2 + 6 * a3 * T + 12 * a4 * T**2 + 20 * a5 * T**3
        jerk = 0.0
    return Reference(t=float(t), x=float(x), v=float(v), a=float(acc), jerk=float(jerk))


class QuinticPlanner(Planner):
    """Fifth-order polynomial with specified start/end pos, vel, accel.

    A feasible flag checks configured reference limits over a dense sample.
    This does **not** mean the reference satisfies arbitrary hard jerk limits
    of the plant or actuator.
    """

    def __init__(
        self,
        *,
        x0: float,
        xf: float,
        duration: float,
        v0: float = 0.0,
        a0: float = 0.0,
        vf: float = 0.0,
        af: float = 0.0,
        v_max: float | None = None,
        a_max: float | None = None,
        j_max: float | None = None,
        n_check: int = 401,
    ) -> None:
        self.coeff = _solve_coefficients(x0, v0, a0, xf, vf, af, duration)
        self.boundary = {
            "x0": x0,
            "v0": v0,
            "a0": a0,
            "xf": xf,
            "vf": vf,
            "af": af,
            "duration": duration,
        }
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.n_check = n_check
        self._feasibility = self._check_feasibility()

    def reference(self, t: float) -> Reference:
        return eval_quintic(self.coeff, t)

    def feasibility(self) -> TrajectoryFeasibility:
        return self._feasibility

    def _check_feasibility(self) -> TrajectoryFeasibility:
        ts = np.linspace(0.0, self.coeff.duration, self.n_check)
        refs = [eval_quintic(self.coeff, float(t)) for t in ts]
        peak_v = max(abs(r.v) for r in refs)
        peak_a = max(abs(r.a) for r in refs)
        peak_j = max(abs(r.jerk) for r in refs)
        reasons: list[str] = []
        feasible = True
        if self.v_max is not None and peak_v > self.v_max + 1e-9:
            feasible = False
            reasons.append(f"peak_|v|={peak_v:.4g} exceeds v_max={self.v_max}")
        if self.a_max is not None and peak_a > self.a_max + 1e-9:
            feasible = False
            reasons.append(f"peak_|a|={peak_a:.4g} exceeds a_max={self.a_max}")
        if self.j_max is not None and peak_j > self.j_max + 1e-9:
            feasible = False
            reasons.append(f"peak_|j|={peak_j:.4g} exceeds j_max={self.j_max}")
        start = eval_quintic(self.coeff, 0.0)
        end = eval_quintic(self.coeff, self.coeff.duration)
        b = self.boundary
        if abs(start.x - b["x0"]) > 1e-8 or abs(start.v - b["v0"]) > 1e-8 or abs(start.a - b["a0"]) > 1e-8:
            feasible = False
            reasons.append("start boundary failed")
        if abs(end.x - b["xf"]) > 1e-8 or abs(end.v - b["vf"]) > 1e-8 or abs(end.a - b["af"]) > 1e-8:
            feasible = False
            reasons.append("end boundary failed")
        return TrajectoryFeasibility(
            feasible=feasible,
            peak_abs_velocity=peak_v,
            peak_abs_acceleration=peak_a,
            peak_abs_jerk=peak_j,
            reasons=tuple(reasons),
        )
