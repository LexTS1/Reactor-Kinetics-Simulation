"""Lumped two-node thermal model for coupled point kinetics.

The thermal state is represented by fuel and moderator temperatures in kelvin:

    Cf dTf/dt = P(t) - Hfm (Tf - Tm)
    Cm dTm/dt = Hfm (Tf - Tm) - Hmc (Tm - Tc)

Power is coupled to point kinetics through relative neutron population:

    P(t) = power_scale * n(t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, init=False)
class ThermalParameters:
    """Parameters for a lumped fuel-moderator thermal model.

    Args:
        fuel_heat_capacity: Effective fuel heat capacity [J/K].
        moderator_heat_capacity: Effective moderator heat capacity [J/K].
        fuel_moderator_htc: Fuel-to-moderator heat transfer coefficient [W/K].
        moderator_coolant_htc: Moderator-to-coolant heat transfer coefficient [W/K].
        coolant_temperature: Coolant/reference sink temperature [K].
        reference_fuel_temperature: Reference fuel temperature [K].
        reference_moderator_temperature: Reference moderator temperature [K].
        power_scale: Nominal thermal power corresponding to n = 1 [W].

    The initializer accepts older Phase 1/2 compact names as aliases:
    ``Cf``, ``Cm``, ``Hfm``, ``Hmc``, ``Tc``, ``Tf0``, ``Tm0``, and
    ``nominal_power``. Optional legacy temperature coefficients are retained as
    attributes for external callers, but Phase 4 feedback should come from the
    reactivity component model.
    """

    fuel_heat_capacity: float
    moderator_heat_capacity: float
    fuel_moderator_htc: float
    moderator_coolant_htc: float
    coolant_temperature: float
    reference_fuel_temperature: float
    reference_moderator_temperature: float
    power_scale: float
    fuel_temperature_coefficient: float | None
    moderator_temperature_coefficient: float | None

    def __init__(
        self,
        fuel_heat_capacity: float | None = None,
        moderator_heat_capacity: float | None = None,
        fuel_moderator_htc: float | None = None,
        moderator_coolant_htc: float | None = None,
        coolant_temperature: float | None = None,
        reference_fuel_temperature: float | None = None,
        reference_moderator_temperature: float | None = None,
        power_scale: float | None = None,
        *,
        Cf: float | None = None,
        Cm: float | None = None,
        Hfm: float | None = None,
        Hmc: float | None = None,
        Tc: float | None = None,
        Tf0: float | None = None,
        Tm0: float | None = None,
        fuel_moderator_heat_transfer: float | None = None,
        moderator_coolant_heat_transfer: float | None = None,
        initial_fuel_temperature: float | None = None,
        initial_moderator_temperature: float | None = None,
        nominal_power: float | None = None,
        fuel_temperature_coefficient: float | None = None,
        moderator_temperature_coefficient: float | None = None,
    ) -> None:
        values = {
            "fuel_heat_capacity": _first_present(fuel_heat_capacity, Cf),
            "moderator_heat_capacity": _first_present(moderator_heat_capacity, Cm),
            "fuel_moderator_htc": _first_present(
                fuel_moderator_htc,
                fuel_moderator_heat_transfer,
                Hfm,
            ),
            "moderator_coolant_htc": _first_present(
                moderator_coolant_htc,
                moderator_coolant_heat_transfer,
                Hmc,
            ),
            "coolant_temperature": _first_present(coolant_temperature, Tc),
            "reference_fuel_temperature": _first_present(
                reference_fuel_temperature,
                initial_fuel_temperature,
                Tf0,
            ),
            "reference_moderator_temperature": _first_present(
                reference_moderator_temperature,
                initial_moderator_temperature,
                Tm0,
            ),
            "power_scale": _first_present(power_scale, nominal_power),
        }

        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise TypeError(f"missing thermal parameter(s): {', '.join(missing)}")

        for name in (
            "fuel_heat_capacity",
            "moderator_heat_capacity",
            "fuel_moderator_htc",
            "moderator_coolant_htc",
            "power_scale",
            "coolant_temperature",
            "reference_fuel_temperature",
            "reference_moderator_temperature",
        ):
            object.__setattr__(self, name, float(values[name]))

        object.__setattr__(
            self,
            "fuel_temperature_coefficient",
            None
            if fuel_temperature_coefficient is None
            else _require_finite("fuel_temperature_coefficient", fuel_temperature_coefficient),
        )
        object.__setattr__(
            self,
            "moderator_temperature_coefficient",
            None
            if moderator_temperature_coefficient is None
            else _require_finite(
                "moderator_temperature_coefficient",
                moderator_temperature_coefficient,
            ),
        )
        self._validate()

    def _validate(self) -> None:
        positive_fields = {
            "fuel_heat_capacity": self.fuel_heat_capacity,
            "moderator_heat_capacity": self.moderator_heat_capacity,
            "fuel_moderator_htc": self.fuel_moderator_htc,
            "moderator_coolant_htc": self.moderator_coolant_htc,
            "power_scale": self.power_scale,
            "coolant_temperature": self.coolant_temperature,
            "reference_fuel_temperature": self.reference_fuel_temperature,
            "reference_moderator_temperature": self.reference_moderator_temperature,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    @property
    def fuel_moderator_heat_transfer(self) -> float:
        """Backward-compatible alias for ``fuel_moderator_htc``."""

        return self.fuel_moderator_htc

    @property
    def moderator_coolant_heat_transfer(self) -> float:
        """Backward-compatible alias for ``moderator_coolant_htc``."""

        return self.moderator_coolant_htc

    @property
    def initial_fuel_temperature(self) -> float:
        """Backward-compatible alias for ``reference_fuel_temperature``."""

        return self.reference_fuel_temperature

    @property
    def initial_moderator_temperature(self) -> float:
        """Backward-compatible alias for ``reference_moderator_temperature``."""

        return self.reference_moderator_temperature

    @property
    def nominal_power(self) -> float:
        """Backward-compatible alias for ``power_scale``."""

        return self.power_scale


ThermalFeedbackParameters = ThermalParameters


def _first_present(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def thermal_power_from_neutron_population(n: float, power_scale: float) -> float:
    """Return thermal power [W] from relative neutron population."""

    neutron_population = _require_finite("n", n)
    scale = _require_finite("power_scale", power_scale)
    if scale <= 0.0:
        raise ValueError("power_scale must be positive.")
    return scale * neutron_population


def thermal_derivatives(
    fuel_temperature: float,
    moderator_temperature: float,
    neutron_population: float,
    params: ThermalParameters,
) -> tuple[float, float]:
    """Return ``(dTf_dt, dTm_dt)`` for the two-node thermal model."""

    fuel = _require_finite("fuel_temperature", fuel_temperature)
    moderator = _require_finite("moderator_temperature", moderator_temperature)
    power = thermal_power_from_neutron_population(
        n=neutron_population,
        power_scale=params.power_scale,
    )

    fuel_to_moderator = params.fuel_moderator_htc * (fuel - moderator)
    moderator_to_coolant = params.moderator_coolant_htc * (
        moderator - params.coolant_temperature
    )

    dTf_dt = (power - fuel_to_moderator) / params.fuel_heat_capacity
    dTm_dt = (
        fuel_to_moderator - moderator_to_coolant
    ) / params.moderator_heat_capacity
    return float(dTf_dt), float(dTm_dt)


def thermal_initial_state(params: ThermalParameters) -> tuple[float, float]:
    """Return initial fuel and moderator temperatures [K]."""

    return (
        params.reference_fuel_temperature,
        params.reference_moderator_temperature,
    )
