"""solve_ivp wrapper for the point-kinetics MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .point_kinetics import (
    KineticsParameters,
    ReactivityFunction,
    ThermalFeedbackParameters,
    critical_initial_state,
    default_u235_parameters,
    initial_state_with_thermal,
    point_kinetics_rhs,
    point_kinetics_thermal_rhs,
    thermal_feedback_reactivity,
)
from .scenarios import zero_reactivity


@dataclass(frozen=True)
class SimulationResult:
    """Formatted simulation result returned by solve_point_kinetics."""

    time: NDArray[np.float64]
    neutron_population: NDArray[np.float64]
    precursors: NDArray[np.float64]
    reactivity: NDArray[np.float64]
    params: KineticsParameters
    title: str = "Point-kinetics simulation"
    fuel_temperature: Optional[NDArray[np.float64]] = None
    moderator_temperature: Optional[NDArray[np.float64]] = None
    raw_solution: Any = None

    @property
    def has_thermal_feedback(self) -> bool:
        """Return True if fuel and moderator temperatures are present."""

        return self.fuel_temperature is not None and self.moderator_temperature is not None


def _validate_t_eval(
    t_span: tuple[float, float],
    t_eval: Optional[Sequence[float] | NDArray[np.float64]],
) -> NDArray[np.float64]:
    if t_span[1] <= t_span[0]:
        raise ValueError("t_span end must be greater than t_span start.")

    if t_eval is None:
        return np.linspace(t_span[0], t_span[1], 500)

    t_eval_array = np.asarray(t_eval, dtype=float)
    if t_eval_array.ndim != 1 or t_eval_array.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(t_eval_array)):
        raise ValueError("t_eval values must be finite.")
    if np.any(np.diff(t_eval_array) < 0.0):
        raise ValueError("t_eval must be sorted in nondecreasing order.")
    if t_eval_array[0] < t_span[0] or t_eval_array[-1] > t_span[1]:
        raise ValueError("t_eval values must lie within t_span.")
    return t_eval_array


def _validate_initial_state(
    initial_state: NDArray[np.float64],
    expected_length: int,
) -> NDArray[np.float64]:
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (expected_length,):
        raise ValueError(f"initial_state must have length {expected_length}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state values must be finite.")
    if state[0] <= 0.0:
        raise ValueError("initial neutron population must be positive.")
    return state


def solve_point_kinetics(
    *,
    params: Optional[KineticsParameters] = None,
    reactivity: Optional[ReactivityFunction] = None,
    t_span: tuple[float, float] = (0.0, 10.0),
    t_eval: Optional[Sequence[float] | NDArray[np.float64]] = None,
    initial_state: Optional[Sequence[float] | NDArray[np.float64]] = None,
    n0: float = 1.0,
    thermal_params: Optional[ThermalFeedbackParameters] = None,
    method: str = "BDF",
    rtol: float = 1.0e-8,
    atol: float | Sequence[float] = 1.0e-10,
    max_step: Optional[float] = None,
    title: str = "Point-kinetics simulation",
) -> SimulationResult:
    """Solve the point-kinetics equations and return a formatted result.

    Args:
        params: Six-group kinetics parameters. Defaults to U-235 values.
        reactivity: External reactivity rho(t), dimensionless.
        t_span: Start and end time [s].
        t_eval: Times at which to store the solution [s].
        initial_state: Optional full initial state vector.
        n0: Initial neutron population used when initial_state is omitted.
        thermal_params: Optional lumped thermal feedback parameters.
        method: solve_ivp method. Use "BDF" or "Radau" for stiff transients.
        rtol: Relative tolerance.
        atol: Absolute tolerance, scalar or component-wise.
        max_step: Optional maximum integration step [s].
        title: Human-readable result title.
    """

    params = default_u235_parameters() if params is None else params
    reactivity = zero_reactivity() if reactivity is None else reactivity
    t_eval_array = _validate_t_eval(t_span=t_span, t_eval=t_eval)

    if thermal_params is None:
        if initial_state is None:
            y0 = critical_initial_state(params=params, n0=n0)
        else:
            y0 = _validate_initial_state(np.asarray(initial_state, dtype=float), 7)

        rhs = lambda t, y: point_kinetics_rhs(  # noqa: E731
            t=t,
            state=y,
            params=params,
            reactivity=reactivity,
        )
    else:
        if initial_state is None:
            y0 = initial_state_with_thermal(
                params=params,
                thermal_params=thermal_params,
                n0=n0,
            )
        else:
            y0 = _validate_initial_state(np.asarray(initial_state, dtype=float), 9)

        rhs = lambda t, y: point_kinetics_thermal_rhs(  # noqa: E731
            t=t,
            state=y,
            params=params,
            external_reactivity=reactivity,
            thermal_params=thermal_params,
        )

    solve_kwargs: dict[str, Any] = {
        "fun": rhs,
        "t_span": t_span,
        "y0": y0,
        "method": method,
        "t_eval": t_eval_array,
        "rtol": rtol,
        "atol": atol,
    }
    if max_step is not None:
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be finite and positive when provided.")
        solve_kwargs["max_step"] = float(max_step)

    solution = solve_ivp(**solve_kwargs)
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")

    neutron_population = solution.y[0]
    precursors = solution.y[1:7]

    if thermal_params is None:
        reactivity_values = np.array([reactivity(t) for t in solution.t], dtype=float)
        fuel_temperature = None
        moderator_temperature = None
    else:
        fuel_temperature = solution.y[7]
        moderator_temperature = solution.y[8]
        reactivity_values = np.array(
            [
                reactivity(t)
                + thermal_feedback_reactivity(
                    fuel_temperature=fuel_temperature[index],
                    moderator_temperature=moderator_temperature[index],
                    thermal_params=thermal_params,
                )
                for index, t in enumerate(solution.t)
            ],
            dtype=float,
        )

    return SimulationResult(
        time=solution.t,
        neutron_population=neutron_population,
        precursors=precursors,
        reactivity=reactivity_values,
        params=params,
        title=title,
        fuel_temperature=fuel_temperature,
        moderator_temperature=moderator_temperature,
        raw_solution=solution,
    )
