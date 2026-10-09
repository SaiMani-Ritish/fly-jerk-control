"""Nominal feedforward-plus-PD tracker.

Uses the **nominal** mass and damping, not the true plant parameters.
The same actuator constraint pipeline is applied after composing with any residual.
"""

from __future__ import annotations

from dataclasses import dataclass

from fly_jerk_control.planners.base import Reference


@dataclass
class NominalTracker:
    mass: float
    damping: float
    kp: float
    kd: float

    def force(self, x_meas: float, v_meas: float, ref: Reference) -> float:
        return (
            self.mass * ref.a
            + self.damping * ref.v
            + self.kp * (ref.x - x_meas)
            + self.kd * (ref.v - v_meas)
        )
