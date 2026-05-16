from __future__ import annotations

import numpy as np
import pytest

from reactor_kinetics.point_kinetics import default_u235_parameters
from reactor_kinetics.reactivity import (
    ReactivityCoefficients,
    ReactivityInputs,
    ReactivityModel,
    rho_fuel_temperature,
    rho_moderator_temperature,
    step_input,
)
from reactor_kinetics.solver import solve_point_kinetics
from reactor_kinetics.thermal import ThermalParameters, thermal_derivatives


@pytest.fixture()
def thermal_params() -> ThermalParameters:
    return ThermalParameters(
        fuel_heat_capacity=2.0e7,
        moderator_heat_capacity=8.0e7,
        fuel_moderator_htc=9.375e6,
        moderator_coolant_htc=1.5e8,
        coolant_temperature=560.0,
        reference_fuel_temperature=900.0,
        reference_moderator_temperature=580.0,
        power_scale=3.0e9,
    )


@pytest.fixture()
def coefficients() -> ReactivityCoefficients:
    return ReactivityCoefficients(
        reference_rod_insertion_fraction=0.5,
        total_control_rod_worth=0.08,
        reference_boron_ppm=1000.0,
        boron_worth_pcm_per_ppm=7.0,
        reference_fuel_temperature=900.0,
        reference_moderator_temperature=580.0,
        alpha_fuel=-3.0e-5,
        alpha_moderator=-5.0e-5,
    )


def _model(
    coefficients: ReactivityCoefficients,
    external_rho: float,
    *,
    feedback_enabled: bool,
) -> ReactivityModel:
    return ReactivityModel(
        coefficients,
        ReactivityInputs(
            external_rho=step_input(0.0, external_rho, 0.2),
            rod_insertion_fraction=coefficients.reference_rod_insertion_fraction,
            boron_ppm=coefficients.reference_boron_ppm,
            fuel_temperature=(
                None if feedback_enabled else coefficients.reference_fuel_temperature
            ),
            moderator_temperature=(
                None
                if feedback_enabled
                else coefficients.reference_moderator_temperature
            ),
        ),
    )


def test_thermal_derivative_sanity(thermal_params: ThermalParameters) -> None:
    dTf_dt, dTm_dt = thermal_derivatives(
        fuel_temperature=900.0,
        moderator_temperature=580.0,
        neutron_population=1.2,
        params=thermal_params,
    )

    assert np.isfinite(dTf_dt)
    assert np.isfinite(dTm_dt)


def test_temperature_rises_under_power(
    thermal_params: ThermalParameters,
    coefficients: ReactivityCoefficients,
) -> None:
    params = default_u235_parameters()
    result = solve_point_kinetics(
        params=params,
        reactivity=_model(coefficients, 0.2 * params.beta_eff, feedback_enabled=True),
        thermal_params=thermal_params,
        t_span=(0.0, 3.0),
        t_eval=np.linspace(0.0, 3.0, 250),
        max_step=0.01,
    )

    assert result.fuel_temperature[-1] > result.fuel_temperature[0]
    assert result.moderator_temperature[-1] >= result.moderator_temperature[0]


def test_negative_feedback_sign(coefficients: ReactivityCoefficients) -> None:
    fuel_feedback = rho_fuel_temperature(930.0, 900.0, coefficients.alpha_fuel)
    moderator_feedback = rho_moderator_temperature(
        585.0,
        580.0,
        coefficients.alpha_moderator,
    )

    assert fuel_feedback < 0.0
    assert moderator_feedback < 0.0
    assert fuel_feedback + moderator_feedback < 0.0


def test_coupled_positive_insertion_behaviour(
    thermal_params: ThermalParameters,
    coefficients: ReactivityCoefficients,
) -> None:
    params = default_u235_parameters()
    result = solve_point_kinetics(
        params=params,
        reactivity=_model(coefficients, 0.3 * params.beta_eff, feedback_enabled=True),
        thermal_params=thermal_params,
        t_span=(0.0, 5.0),
        t_eval=np.linspace(0.0, 5.0, 350),
        max_step=0.01,
    )

    early = result.time <= 1.0
    assert np.max(result.neutron_population[early]) > result.neutron_population[0]
    assert result.fuel_temperature[-1] > result.fuel_temperature[0]
    assert result.moderator_temperature[-1] > result.moderator_temperature[0]
    assert np.min(result.rho_feedback) < 0.0


def test_feedback_reduces_power_relative_to_no_feedback(
    thermal_params: ThermalParameters,
    coefficients: ReactivityCoefficients,
) -> None:
    params = default_u235_parameters()
    t_eval = np.linspace(0.0, 5.0, 350)
    no_feedback = solve_point_kinetics(
        params=params,
        reactivity=_model(coefficients, 0.3 * params.beta_eff, feedback_enabled=False),
        t_span=(0.0, 5.0),
        t_eval=t_eval,
        max_step=0.01,
    )
    feedback = solve_point_kinetics(
        params=params,
        reactivity=_model(coefficients, 0.3 * params.beta_eff, feedback_enabled=True),
        thermal_params=thermal_params,
        t_span=(0.0, 5.0),
        t_eval=t_eval,
        max_step=0.01,
    )

    assert feedback.neutron_population[-1] < no_feedback.neutron_population[-1]
    assert np.max(feedback.neutron_population) <= np.max(no_feedback.neutron_population)
