"""Six-group point-kinetics equations and thermal feedback helpers.

The state vector for kinetics-only simulations is:

    [n, C1, C2, C3, C4, C5, C6]

With lumped thermal feedback enabled, the state vector is:

    [n, C1, C2, C3, C4, C5, C6, Tf, Tm]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


ReactivityFunction = Callable[[float], float]


@dataclass(frozen=True)
class KineticsParameters:
    """Point-kinetics parameters for a six delayed-neutron group model.

    Attributes:
        beta: Delayed neutron fractions for the six groups, dimensionless.
        lambdas: Delayed-neutron precursor decay constants [1/s].
        Lambda: Prompt neutron generation time [s].
    """

    beta: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    Lambda: float

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=float)
        lambdas = np.asarray(self.lambdas, dtype=float)

        if beta.shape != (6,):
            raise ValueError("beta_i must contain exactly 6 delayed-neutron groups.")
        if lambdas.shape != (6,):
            raise ValueError("lambda_i must contain exactly 6 decay constants.")
        if not np.all(np.isfinite(beta)) or np.any(beta < 0.0):
            raise ValueError("beta_i values must be finite and non-negative.")
        if not np.all(np.isfinite(lambdas)) or np.any(lambdas <= 0.0):
            raise ValueError("lambda_i values must be finite and positive.")
        if not np.isfinite(self.Lambda) or self.Lambda <= 0.0:
            raise ValueError("Lambda must be finite and positive.")
        if float(np.sum(beta)) <= 0.0:
            raise ValueError("beta_eff must be positive.")

        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "Lambda", float(self.Lambda))

    @property
    def beta_eff(self) -> float:
        """Effective delayed neutron fraction, beta_eff."""

        return float(np.sum(self.beta))


@dataclass(frozen=True)
class ThermalFeedbackParameters:
    """Lumped two-temperature thermal feedback parameters.

    Temperatures are in kelvin. The power scale is arbitrary but must be
    consistent with the heat capacities and heat transfer coefficients.
    """

    fuel_heat_capacity: float
    moderator_heat_capacity: float
    fuel_moderator_heat_transfer: float
    moderator_coolant_heat_transfer: float
    coolant_temperature: float
    initial_fuel_temperature: float
    initial_moderator_temperature: float
    fuel_temperature_coefficient: float
    moderator_temperature_coefficient: float
    nominal_power: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = {
            "fuel_heat_capacity": self.fuel_heat_capacity,
            "moderator_heat_capacity": self.moderator_heat_capacity,
            "fuel_moderator_heat_transfer": self.fuel_moderator_heat_transfer,
            "moderator_coolant_heat_transfer": self.moderator_coolant_heat_transfer,
            "nominal_power": self.nominal_power,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        temperature_fields = {
            "coolant_temperature": self.coolant_temperature,
            "initial_fuel_temperature": self.initial_fuel_temperature,
            "initial_moderator_temperature": self.initial_moderator_temperature,
        }
        for name, value in temperature_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        coefficient_fields = {
            "fuel_temperature_coefficient": self.fuel_temperature_coefficient,
            "moderator_temperature_coefficient": self.moderator_temperature_coefficient,
        }
        for name, value in coefficient_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")


def default_u235_parameters(Lambda: float = 1.0e-5) -> KineticsParameters:
    """Return a representative six-group U-235 delayed-neutron dataset.

    The beta_i and lambda_i values are standard classroom-scale values for
    thermal-fission U-235 point-kinetics demonstrations. Reactivity is
    dimensionless, time is in seconds, and lambda_i values are in 1/s.
    """

    beta = np.array(
        [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
        dtype=float,
    )
    lambdas = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01], dtype=float)
    return KineticsParameters(beta=beta, lambdas=lambdas, Lambda=Lambda)


def critical_initial_state(
    params: KineticsParameters,
    n0: float = 1.0,
) -> NDArray[np.float64]:
    """Return the critical steady-state kinetics state for rho = 0.

    At critical steady state, dC_i/dt = 0, so:

        C_i0 = beta_i / (Lambda * lambda_i) * n0
    """

    if not np.isfinite(n0) or n0 <= 0.0:
        raise ValueError("initial neutron population n0 must be finite and positive.")

    precursors = params.beta / (params.Lambda * params.lambdas) * float(n0)
    return np.concatenate(([float(n0)], precursors))


def initial_state_with_thermal(
    params: KineticsParameters,
    thermal_params: ThermalFeedbackParameters,
    n0: float = 1.0,
) -> NDArray[np.float64]:
    """Return a critical kinetics state with initial fuel and moderator temperatures."""

    kinetics_state = critical_initial_state(params=params, n0=n0)
    return np.concatenate(
        (
            kinetics_state,
            [
                thermal_params.initial_fuel_temperature,
                thermal_params.initial_moderator_temperature,
            ],
        )
    )


def thermal_feedback_reactivity(
    fuel_temperature: float,
    moderator_temperature: float,
    thermal_params: ThermalFeedbackParameters,
) -> float:
    """Return temperature feedback reactivity from lumped temperatures."""

    return (
        thermal_params.fuel_temperature_coefficient
        * (fuel_temperature - thermal_params.initial_fuel_temperature)
        + thermal_params.moderator_temperature_coefficient
        * (moderator_temperature - thermal_params.initial_moderator_temperature)
    )


def point_kinetics_rhs(
    t: float,
    state: NDArray[np.float64],
    params: KineticsParameters,
    reactivity: ReactivityFunction,
) -> NDArray[np.float64]:
    """Evaluate the six-group point-kinetics right-hand side."""

    n = state[0]
    precursors = state[1:7]
    rho = float(reactivity(t))

    derivatives = np.empty(7, dtype=float)
    derivatives[0] = (
        ((rho - params.beta_eff) / params.Lambda) * n
        + np.dot(params.lambdas, precursors)
    )
    derivatives[1:7] = (params.beta / params.Lambda) * n - params.lambdas * precursors
    return derivatives


def point_kinetics_thermal_rhs(
    t: float,
    state: NDArray[np.float64],
    params: KineticsParameters,
    external_reactivity: ReactivityFunction,
    thermal_params: ThermalFeedbackParameters,
) -> NDArray[np.float64]:
    """Evaluate point kinetics coupled to two lumped thermal equations."""

    n = state[0]
    precursors = state[1:7]
    fuel_temperature = state[7]
    moderator_temperature = state[8]

    rho = float(external_reactivity(t)) + thermal_feedback_reactivity(
        fuel_temperature=fuel_temperature,
        moderator_temperature=moderator_temperature,
        thermal_params=thermal_params,
    )

    derivatives = np.empty(9, dtype=float)
    derivatives[0] = (
        ((rho - params.beta_eff) / params.Lambda) * n
        + np.dot(params.lambdas, precursors)
    )
    derivatives[1:7] = (params.beta / params.Lambda) * n - params.lambdas * precursors

    relative_power = thermal_params.nominal_power * n
    fuel_to_moderator = thermal_params.fuel_moderator_heat_transfer * (
        fuel_temperature - moderator_temperature
    )
    moderator_to_coolant = thermal_params.moderator_coolant_heat_transfer * (
        moderator_temperature - thermal_params.coolant_temperature
    )

    derivatives[7] = (
        relative_power - fuel_to_moderator
    ) / thermal_params.fuel_heat_capacity
    derivatives[8] = (
        fuel_to_moderator - moderator_to_coolant
    ) / thermal_params.moderator_heat_capacity
    return derivatives
