"""Composable reactivity components for educational point kinetics.

All reactivity values are dimensionless delta-k/k. Positive reactivity
increases neutron population, while negative reactivity suppresses it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray


InputTrajectory = Callable[[float], float]
InputValue = Union[float, InputTrajectory]


@dataclass(frozen=True)
class ReactivityCoefficients:
    """Reference values and linear coefficients for reactivity components.

    total_control_rod_worth is positive. With the convention used here,
    rod_insertion_fraction = 1.0 is fully inserted and 0.0 is fully withdrawn,
    so withdrawing rods adds positive reactivity relative to the reference
    position.
    """

    total_control_rod_worth: float
    boron_worth_pcm_per_ppm: float
    alpha_fuel: float
    alpha_moderator: float
    reference_boron_ppm: float
    reference_fuel_temperature: float
    reference_moderator_temperature: float
    reference_rod_insertion_fraction: float = 1.0

    @property
    def control_rod_worth(self) -> float:
        """Alias for total rod worth, kept for concise formulas."""

        return self.total_control_rod_worth


@dataclass(frozen=True)
class ReactivityInputs:
    """Static values or time trajectories for reactivity channels."""

    external_rho: InputValue = 0.0
    rod_insertion_fraction: InputValue = 1.0
    boron_ppm: InputValue = 1000.0
    fuel_temperature: InputValue | None = None
    moderator_temperature: InputValue | None = None


@dataclass(frozen=True)
class ReactivityComponentValues:
    """Individual reactivity components and their total."""

    external: float
    control_rods: float
    boron: float
    fuel_temperature: float
    moderator_temperature: float

    @property
    def total(self) -> float:
        """Return the sum of all components."""

        return (
            self.external
            + self.control_rods
            + self.boron
            + self.fuel_temperature
            + self.moderator_temperature
        )

    def as_dict(self) -> dict[str, float]:
        """Return component values in a plotting-friendly mapping."""

        return {
            "external": self.external,
            "control_rods": self.control_rods,
            "boron": self.boron,
            "fuel_temperature": self.fuel_temperature,
            "moderator_temperature": self.moderator_temperature,
            "total": self.total,
        }


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _resolve_input(value: InputValue, t: float) -> float:
    if callable(value):
        return _require_finite("trajectory value", value(t))
    return _require_finite("input value", value)


def _temperature_from_state(
    state: NDArray[np.float64] | None,
    index: int,
    fallback: float,
) -> float:
    if state is None or len(state) <= index:
        return fallback
    return _require_finite("state temperature", float(state[index]))


def pcm_to_rho(pcm: float) -> float:
    """Convert pcm to dimensionless reactivity."""

    return _require_finite("pcm", pcm) * 1.0e-5


def rho_to_pcm(rho: float) -> float:
    """Convert dimensionless reactivity to pcm."""

    return _require_finite("rho", rho) * 1.0e5


def rho_external(external_rho: float) -> float:
    """Return a generic external reactivity perturbation."""

    return _require_finite("external_rho", external_rho)


def rho_control_rods(
    rod_insertion_fraction: float,
    reference_rod_insertion_fraction: float,
    total_rod_worth: float,
) -> float:
    """Return control-rod reactivity relative to the reference position.

    More insertion is more negative. With positive total worth, withdrawing
    rods from the reference position inserts positive reactivity.
    """

    insertion = _require_finite("rod_insertion_fraction", rod_insertion_fraction)
    reference = _require_finite(
        "reference_rod_insertion_fraction",
        reference_rod_insertion_fraction,
    )
    worth = _require_finite("total_rod_worth", total_rod_worth)
    if worth < 0.0:
        raise ValueError("total_rod_worth must be non-negative.")
    return worth * (reference - insertion)


def rho_boron(
    boron_ppm: float,
    reference_boron_ppm: float,
    boron_worth_pcm_per_ppm: float,
) -> float:
    """Return soluble-boron reactivity relative to reference concentration.

    Boron is a neutron absorber. Increasing boron concentration adds negative
    reactivity; dilution adds positive reactivity.
    """

    concentration = _require_finite("boron_ppm", boron_ppm)
    reference = _require_finite("reference_boron_ppm", reference_boron_ppm)
    worth = _require_finite("boron_worth_pcm_per_ppm", boron_worth_pcm_per_ppm)
    if worth < 0.0:
        raise ValueError("boron_worth_pcm_per_ppm must be non-negative.")
    return -pcm_to_rho(worth * (concentration - reference))


def rho_fuel_temperature(
    fuel_temperature: float,
    reference_fuel_temperature: float,
    alpha_fuel: float,
) -> float:
    """Return linear fuel-temperature feedback reactivity."""

    return _require_finite("alpha_fuel", alpha_fuel) * (
        _require_finite("fuel_temperature", fuel_temperature)
        - _require_finite("reference_fuel_temperature", reference_fuel_temperature)
    )


def rho_moderator_temperature(
    moderator_temperature: float,
    reference_moderator_temperature: float,
    alpha_moderator: float,
) -> float:
    """Return linear moderator-temperature feedback reactivity."""

    return _require_finite("alpha_moderator", alpha_moderator) * (
        _require_finite("moderator_temperature", moderator_temperature)
        - _require_finite(
            "reference_moderator_temperature",
            reference_moderator_temperature,
        )
    )


def constant_input(value: float) -> InputTrajectory:
    """Return a time-independent input trajectory."""

    constant = _require_finite("value", value)
    return lambda t: constant


def linear_ramp_input(
    start_value: float,
    end_value: float,
    t_start: float,
    t_end: float,
) -> InputTrajectory:
    """Return a piecewise-linear input trajectory."""

    start = _require_finite("start_value", start_value)
    end = _require_finite("end_value", end_value)
    start_time = _require_finite("t_start", t_start)
    end_time = _require_finite("t_end", t_end)
    if end_time <= start_time:
        raise ValueError("t_end must be greater than t_start.")

    def trajectory(t: float) -> float:
        time = float(t)
        if time <= start_time:
            return start
        if time >= end_time:
            return end
        fraction = (time - start_time) / (end_time - start_time)
        return start + fraction * (end - start)

    return trajectory


def step_input(
    initial_value: float,
    final_value: float,
    t_step: float,
) -> InputTrajectory:
    """Return an input trajectory with one step change."""

    initial = _require_finite("initial_value", initial_value)
    final = _require_finite("final_value", final_value)
    step_time = _require_finite("t_step", t_step)
    return lambda t: initial if t < step_time else final


class ReactivityModel:
    """Callable model that sums external, rod, boron, and temperature effects."""

    def __init__(
        self,
        coefficients: ReactivityCoefficients,
        inputs: ReactivityInputs | Mapping[str, Any] | None = None,
    ) -> None:
        self.coefficients = coefficients
        self.inputs = _coerce_inputs(inputs)

    def components(
        self,
        t: float,
        state: NDArray[np.float64] | None = None,
        inputs: ReactivityInputs | Mapping[str, Any] | None = None,
    ) -> ReactivityComponentValues:
        """Return individual reactivity components at time t."""

        active_inputs = _coerce_inputs(inputs) if inputs is not None else self.inputs
        coeffs = self.coefficients

        fuel_temperature = (
            _resolve_input(active_inputs.fuel_temperature, t)
            if active_inputs.fuel_temperature is not None
            else _temperature_from_state(state, 7, coeffs.reference_fuel_temperature)
        )
        moderator_temperature = (
            _resolve_input(active_inputs.moderator_temperature, t)
            if active_inputs.moderator_temperature is not None
            else _temperature_from_state(
                state,
                8,
                coeffs.reference_moderator_temperature,
            )
        )

        return ReactivityComponentValues(
            external=rho_external(_resolve_input(active_inputs.external_rho, t)),
            control_rods=rho_control_rods(
                rod_insertion_fraction=_resolve_input(
                    active_inputs.rod_insertion_fraction,
                    t,
                ),
                reference_rod_insertion_fraction=(
                    coeffs.reference_rod_insertion_fraction
                ),
                total_rod_worth=coeffs.total_control_rod_worth,
            ),
            boron=rho_boron(
                boron_ppm=_resolve_input(active_inputs.boron_ppm, t),
                reference_boron_ppm=coeffs.reference_boron_ppm,
                boron_worth_pcm_per_ppm=coeffs.boron_worth_pcm_per_ppm,
            ),
            fuel_temperature=rho_fuel_temperature(
                fuel_temperature=fuel_temperature,
                reference_fuel_temperature=coeffs.reference_fuel_temperature,
                alpha_fuel=coeffs.alpha_fuel,
            ),
            moderator_temperature=rho_moderator_temperature(
                moderator_temperature=moderator_temperature,
                reference_moderator_temperature=(
                    coeffs.reference_moderator_temperature
                ),
                alpha_moderator=coeffs.alpha_moderator,
            ),
        )

    def __call__(
        self,
        t: float,
        state: NDArray[np.float64] | None = None,
        inputs: ReactivityInputs | Mapping[str, Any] | None = None,
    ) -> float:
        """Return total reactivity at time t."""

        return self.components(t=t, state=state, inputs=inputs).total


def rho_total(
    coefficients: ReactivityCoefficients,
    inputs: ReactivityInputs | Mapping[str, Any] | None = None,
    t: float = 0.0,
    state: NDArray[np.float64] | None = None,
) -> float:
    """Convenience function returning total reactivity for one state."""

    return ReactivityModel(coefficients=coefficients, inputs=inputs)(t, state)


def _coerce_inputs(
    inputs: ReactivityInputs | Mapping[str, Any] | None,
) -> ReactivityInputs:
    if inputs is None:
        return ReactivityInputs()
    if isinstance(inputs, ReactivityInputs):
        return inputs
    return ReactivityInputs(**dict(inputs))
