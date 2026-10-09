"""Optional Ruckig adapter.

This planner is disabled until the optional extra is installed. It never
silently falls back to quintic while labeling a run as Ruckig.
"""

from __future__ import annotations

from fly_jerk_control.planners.base import Planner, Reference, TrajectoryFeasibility


class RuckigUnavailableError(RuntimeError):
    pass


class RuckigAdapter(Planner):
    def __init__(self, *args, **kwargs) -> None:
        try:
            import ruckig  # noqa: F401
        except ImportError as exc:
            raise RuckigUnavailableError(
                "Ruckig is not installed. Install the optional extra "
                "`pip install fly-jerk-control[ruckig]` or select planner.type=quintic. "
                "This run will not substitute a quintic trajectory under a Ruckig label."
            ) from exc
        raise NotImplementedError(
            "Ruckig adapter is an optional Milestone 5 extension and is not implemented yet."
        )

    def reference(self, t: float) -> Reference:
        raise RuckigUnavailableError("Ruckig adapter is not implemented")

    def feasibility(self) -> TrajectoryFeasibility:
        raise RuckigUnavailableError("Ruckig adapter is not implemented")
