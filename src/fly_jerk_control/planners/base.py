"""Planner interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    t: float
    x: float
    v: float
    a: float
    jerk: float


@dataclass(frozen=True)
class TrajectoryFeasibility:
    feasible: bool
    peak_abs_velocity: float
    peak_abs_acceleration: float
    peak_abs_jerk: float
    reasons: tuple[str, ...]


class Planner(ABC):
    @abstractmethod
    def reference(self, t: float) -> Reference:
        raise NotImplementedError

    @abstractmethod
    def feasibility(self) -> TrajectoryFeasibility:
        raise NotImplementedError
