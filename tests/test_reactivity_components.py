from __future__ import annotations

import numpy as np
import pytest

from reactor_kinetics.point_kinetics import default_u235_parameters
from reactor_kinetics.reactivity import (
    ReactivityCoefficients,
    ReactivityInputs,
    ReactivityModel,
    linear_ramp_input,
    pcm_to_rho,
    rho_boron,
    rho_control_rods,
    rho_fuel_temperature,
    rho_moderator_temperature,
    rho_to_pcm,
)
from reactor_kinetics.solver import solve_point_kinetics


@pytest.fixture()
def coefficients() -> ReactivityCoefficients:
    return ReactivityCoefficients(
        reference_rod_insertion_fraction=0.5,
        total_control_rod_worth=0.01,
        reference_boron_ppm=1000.0,
        boron_worth_pcm_per_ppm=7.0,
        reference_fuel_temperature=900.0,
        reference_moderator_temperature=580.0,
        alpha_fuel=-3.0e-5,
        alpha_moderator=-5.0e-5,
    )


def test_pcm_conversion() -> None:
    assert pcm_to_rho(100.0) == pytest.approx(0.001)
    assert rho_to_pcm(0.001) == pytest.approx(100.0)


def test_control_rod_sign_convention() -> None:
    worth = 0.01

    assert rho_control_rods(0.4, 0.5, worth) > 0.0
    assert rho_control_rods(0.6, 0.5, worth) < 0.0
    assert rho_control_rods(0.5, 0.5, worth) == pytest.approx(0.0)


def test_boron_sign_convention() -> None:
    worth = 7.0

    assert rho_boron(950.0, 1000.0, worth) > 0.0
    assert rho_boron(1050.0, 1000.0, worth) < 0.0
    assert rho_boron(1000.0, 1000.0, worth) == pytest.approx(0.0)


def test_temperature_feedback_sign_convention() -> None:
    assert rho_fuel_temperature(910.0, 900.0, -3.0e-5) < 0.0
    assert rho_fuel_temperature(890.0, 900.0, -3.0e-5) > 0.0
    assert rho_fuel_temperature(900.0, 900.0, -3.0e-5) == pytest.approx(0.0)

    assert rho_moderator_temperature(590.0, 580.0, -5.0e-5) < 0.0
    assert rho_moderator_temperature(570.0, 580.0, -5.0e-5) > 0.0
    assert rho_moderator_temperature(580.0, 580.0, -5.0e-5) == pytest.approx(0.0)


def test_total_reactivity_is_sum_of_components(
    coefficients: ReactivityCoefficients,
) -> None:
    model = ReactivityModel(
        coefficients,
        ReactivityInputs(
            external_rho=0.0002,
            rod_insertion_fraction=0.45,
            boron_ppm=980.0,
            fuel_temperature=905.0,
            moderator_temperature=578.0,
        ),
    )

    components = model.components(t=0.0)

    assert model(0.0) == pytest.approx(
        components.external
        + components.control_rods
        + components.boron
        + components.fuel_temperature
        + components.moderator_temperature
    )
    assert components.as_dict()["total"] == pytest.approx(model(0.0))


def _final_power_for_inputs(
    coefficients: ReactivityCoefficients,
    inputs: ReactivityInputs,
) -> float:
    params = default_u235_parameters()
    result = solve_point_kinetics(
        params=params,
        reactivity=ReactivityModel(coefficients, inputs),
        t_span=(0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 400),
        max_step=0.05,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert result.reactivity_components is not None
    return float(result.neutron_population[-1])


def test_rod_withdrawal_and_insertion_change_power(
    coefficients: ReactivityCoefficients,
) -> None:
    withdrawal = _final_power_for_inputs(
        coefficients,
        ReactivityInputs(
            rod_insertion_fraction=linear_ramp_input(0.5, 0.4, 1.0, 5.0),
            boron_ppm=1000.0,
            fuel_temperature=900.0,
            moderator_temperature=580.0,
        ),
    )
    insertion = _final_power_for_inputs(
        coefficients,
        ReactivityInputs(
            rod_insertion_fraction=linear_ramp_input(0.5, 0.6, 1.0, 5.0),
            boron_ppm=1000.0,
            fuel_temperature=900.0,
            moderator_temperature=580.0,
        ),
    )

    assert withdrawal > 1.0
    assert insertion < 1.0


def test_boron_dilution_and_addition_change_power(
    coefficients: ReactivityCoefficients,
) -> None:
    dilution = _final_power_for_inputs(
        coefficients,
        ReactivityInputs(
            rod_insertion_fraction=0.5,
            boron_ppm=linear_ramp_input(1000.0, 950.0, 1.0, 5.0),
            fuel_temperature=900.0,
            moderator_temperature=580.0,
        ),
    )
    addition = _final_power_for_inputs(
        coefficients,
        ReactivityInputs(
            rod_insertion_fraction=0.5,
            boron_ppm=linear_ramp_input(1000.0, 1050.0, 1.0, 5.0),
            fuel_temperature=900.0,
            moderator_temperature=580.0,
        ),
    )

    assert dilution > 1.0
    assert addition < 1.0
