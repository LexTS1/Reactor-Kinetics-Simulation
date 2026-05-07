"""Composable reactivity functions for point-kinetics scenarios."""

from __future__ import annotations

from typing import Callable

import numpy as np


ReactivityFunction = Callable[[float], float]


def zero_reactivity() -> ReactivityFunction:
    """Return rho(t) = 0."""

    return lambda t: 0.0


def step_reactivity(rho_step: float, t_step: float) -> ReactivityFunction:
    """Return a positive or signed step reactivity insertion.

    Args:
        rho_step: Reactivity after the step, dimensionless.
        t_step: Step time [s].
    """

    if not np.isfinite(rho_step):
        raise ValueError("rho_step must be finite.")
    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite.")

    rho_step = float(rho_step)
    t_step = float(t_step)
    return lambda t: 0.0 if t < t_step else rho_step


def negative_step_reactivity(rho_step: float, t_step: float) -> ReactivityFunction:
    """Return a negative step insertion with magnitude abs(rho_step)."""

    return step_reactivity(rho_step=-abs(float(rho_step)), t_step=t_step)


def ramp_reactivity(
    rho_start: float,
    rho_end: float,
    t_start: float,
    t_end: float,
) -> ReactivityFunction:
    """Return a piecewise-linear reactivity ramp."""

    values = np.array([rho_start, rho_end, t_start, t_end], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("ramp reactivity inputs must be finite.")
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start for a ramp.")

    rho_start = float(rho_start)
    rho_end = float(rho_end)
    t_start = float(t_start)
    t_end = float(t_end)

    def rho(t: float) -> float:
        if t <= t_start:
            return rho_start
        if t >= t_end:
            return rho_end
        fraction = (t - t_start) / (t_end - t_start)
        return rho_start + fraction * (rho_end - rho_start)

    return rho


def combined_reactivity(*rho_functions: ReactivityFunction) -> ReactivityFunction:
    """Return the sum of any number of reactivity functions.

    This is the MVP hook for later rod, boron, external, fuel, and moderator
    components. Each component is simply a function of time returning
    dimensionless reactivity.
    """

    if not rho_functions:
        return zero_reactivity()

    def rho(t: float) -> float:
        return float(sum(rho_function(t) for rho_function in rho_functions))

    return rho
