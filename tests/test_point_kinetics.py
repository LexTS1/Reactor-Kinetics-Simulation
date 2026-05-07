from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reactor_kinetics.config import load_reactor_config
from reactor_kinetics.point_kinetics import (
    critical_initial_state,
    default_u235_parameters,
    point_kinetics_rhs,
)
from reactor_kinetics.scenarios import zero_reactivity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_parameters_load_correctly() -> None:
    params = default_u235_parameters()

    assert params.Lambda > 0.0
    assert params.beta_eff > 0.0
    assert np.all(params.beta >= 0.0)
    assert np.all(params.lambdas > 0.0)


def test_reference_config_loads_valid_six_group_parameters() -> None:
    config = load_reactor_config(PROJECT_ROOT / "configs" / "pwr_reference.yaml")

    assert config.thermal_enabled
    assert config.thermal is not None
    assert config.kinetics.beta.shape == (6,)
    assert config.kinetics.lambdas.shape == (6,)
    assert config.kinetics.beta_eff == pytest.approx(float(np.sum(config.kinetics.beta)))


def test_beta_and_lambda_arrays_have_length_six() -> None:
    params = default_u235_parameters()

    assert len(params.beta) == 6
    assert len(params.lambdas) == 6


def test_beta_eff_equals_sum_beta() -> None:
    params = default_u235_parameters()

    assert params.beta_eff == pytest.approx(float(np.sum(params.beta)))


def test_critical_initial_state_has_expected_entries() -> None:
    params = default_u235_parameters()
    state = critical_initial_state(params)

    assert state.shape == (7,)
    assert state[0] == pytest.approx(1.0)


def test_critical_initial_precursors_are_positive() -> None:
    params = default_u235_parameters()
    state = critical_initial_state(params)

    assert np.all(state[1:] > 0.0)


def test_zero_reactivity_derivative_is_zero_at_critical_state() -> None:
    params = default_u235_parameters()
    state = critical_initial_state(params)
    derivative = point_kinetics_rhs(
        t=0.0,
        state=state,
        params=params,
        reactivity=zero_reactivity(),
    )

    assert np.allclose(derivative, np.zeros(7), atol=1.0e-12)
