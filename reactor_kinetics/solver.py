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
    coupled_initial_state,
    critical_initial_state,
    default_u235_parameters,
    evaluate_reactivity,
    point_kinetics_rhs,
    point_kinetics_thermal_rhs,
    thermal_feedback_reactivity,
)
from .scenarios import zero_reactivity
from .thermal import ThermalParameters


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
    reactivity_components: Optional[dict[str, NDArray[np.float64]]] = None
    raw_solution: Any = None

    @property
    def has_thermal_feedback(self) -> bool:
        """Return True if fuel and moderator temperatures are present."""

        return self.fuel_temperature is not None and self.moderator_temperature is not None

    @property
    def has_reactivity_components(self) -> bool:
        """Return True if individual reactivity components are present."""

        return self.reactivity_components is not None

    @property
    def rho_total(self) -> NDArray[np.float64]:
        """Total reactivity, dimensionless delta-k/k."""

        return self.reactivity

    def _component_or_zero(self, name: str) -> NDArray[np.float64]:
        if self.reactivity_components is None:
            return np.zeros_like(self.reactivity)
        return self.reactivity_components.get(name, np.zeros_like(self.reactivity))

    @property
    def rho_external(self) -> NDArray[np.float64]:
        """External reactivity component."""

        return self._component_or_zero("external")

    @property
    def rho_control_rods(self) -> NDArray[np.float64]:
        """Control-rod reactivity component."""

        return self._component_or_zero("control_rods")

    @property
    def rho_boron(self) -> NDArray[np.float64]:
        """Soluble-boron reactivity component."""

        return self._component_or_zero("boron")

    @property
    def rho_fuel_temperature(self) -> NDArray[np.float64]:
        """Fuel-temperature reactivity component."""

        return self._component_or_zero("fuel_temperature")

    @property
    def rho_moderator_temperature(self) -> NDArray[np.float64]:
        """Moderator-temperature reactivity component."""

        return self._component_or_zero("moderator_temperature")

    @property
    def rho_feedback(self) -> NDArray[np.float64]:
        """Total fuel plus moderator temperature feedback reactivity."""

        return self.rho_fuel_temperature + self.rho_moderator_temperature


def _reactivity_components_at_times(
    reactivity: ReactivityFunction,
    times: NDArray[np.float64],
    states: NDArray[np.float64],
) -> Optional[dict[str, NDArray[np.float64]]]:
    components_method = getattr(reactivity, "components", None)
    if not callable(components_method):
        return None

    component_rows: list[dict[str, float]] = []
    for index, t in enumerate(times):
        components = components_method(float(t), states[:, index])
        if hasattr(components, "as_dict"):
            component_rows.append(components.as_dict())
        else:
            component_rows.append(dict(components))

    keys = sorted({key for row in component_rows for key in row})
    return {
        key: np.array([row.get(key, 0.0) for row in component_rows], dtype=float)
        for key in keys
    }


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
    thermal_params: Optional[ThermalParameters] = None,
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
            y0 = coupled_initial_state(
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
            reactivity=reactivity,
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
        reactivity_values = np.array(
            [
                evaluate_reactivity(reactivity, float(t), solution.y[:, index])
                for index, t in enumerate(solution.t)
            ],
            dtype=float,
        )
        fuel_temperature = None
        moderator_temperature = None
    else:
        fuel_temperature = solution.y[7]
        moderator_temperature = solution.y[8]
        reactivity_values = np.array(
            [
                evaluate_reactivity(reactivity, float(t), solution.y[:, index])
                + thermal_feedback_reactivity(
                    fuel_temperature=fuel_temperature[index],
                    moderator_temperature=moderator_temperature[index],
                    thermal_params=thermal_params,
                )
                for index, t in enumerate(solution.t)
            ],
            dtype=float,
        )
    reactivity_components = _reactivity_components_at_times(
        reactivity=reactivity,
        times=solution.t,
        states=solution.y,
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
        reactivity_components=reactivity_components,
        raw_solution=solution,
    )
