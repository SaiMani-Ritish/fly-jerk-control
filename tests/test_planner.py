"""Quintic planner tests."""

from __future__ import annotations

import numpy as np
import pytest

from fly_jerk_control.planners.quintic import QuinticPlanner
from fly_jerk_control.planners.ruckig_adapter import RuckigAdapter, RuckigUnavailableError


def test_quintic_boundaries() -> None:
    p = QuinticPlanner(x0=0.0, xf=1.0, duration=2.0, v0=0.0, a0=0.0, vf=0.0, af=0.0)
    s = p.reference(0.0)
    e = p.reference(2.0)
    assert abs(s.x) < 1e-10 and abs(s.v) < 1e-10 and abs(s.a) < 1e-10
    assert abs(e.x - 1.0) < 1e-10 and abs(e.v) < 1e-10 and abs(e.a) < 1e-10
    assert p.feasibility().feasible


def test_quintic_infeasible_when_limits_tight() -> None:
    p = QuinticPlanner(
        x0=0.0,
        xf=1.0,
        duration=0.2,
        v_max=0.01,
        a_max=0.01,
        j_max=0.01,
    )
    feas = p.feasibility()
    assert not feas.feasible
    assert feas.reasons


def test_quintic_hold_after_duration() -> None:
    p = QuinticPlanner(x0=0.0, xf=1.0, duration=1.0)
    late = p.reference(5.0)
    assert abs(late.x - 1.0) < 1e-8
    assert abs(late.jerk) < 1e-12


def test_ruckig_does_not_silent_fallback() -> None:
    with pytest.raises((RuckigUnavailableError, NotImplementedError)):
        RuckigAdapter()
