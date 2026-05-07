from __future__ import annotations

import pytest

from reactor_kinetics.scenarios import (
    combined_reactivity,
    ramp_reactivity,
    step_reactivity,
    zero_reactivity,
)


def test_zero_reactivity_returns_zero() -> None:
    rho = zero_reactivity()

    assert rho(0.0) == pytest.approx(0.0)
    assert rho(100.0) == pytest.approx(0.0)


def test_step_reactivity_is_zero_before_insertion_time() -> None:
    rho = step_reactivity(rho_step=0.003, t_step=2.0)

    assert rho(1.999) == pytest.approx(0.0)


def test_step_reactivity_returns_step_after_insertion_time() -> None:
    rho = step_reactivity(rho_step=0.003, t_step=2.0)

    assert rho(2.0) == pytest.approx(0.003)
    assert rho(10.0) == pytest.approx(0.003)


def test_ramp_reactivity_starts_at_rho_start() -> None:
    rho = ramp_reactivity(rho_start=0.001, rho_end=0.004, t_start=2.0, t_end=8.0)

    assert rho(0.0) == pytest.approx(0.001)
    assert rho(2.0) == pytest.approx(0.001)


def test_ramp_reactivity_ends_at_rho_end() -> None:
    rho = ramp_reactivity(rho_start=0.001, rho_end=0.004, t_start=2.0, t_end=8.0)

    assert rho(8.0) == pytest.approx(0.004)
    assert rho(20.0) == pytest.approx(0.004)


def test_combined_reactivity_returns_component_sum() -> None:
    first = step_reactivity(rho_step=0.002, t_step=1.0)
    second = ramp_reactivity(rho_start=0.001, rho_end=0.003, t_start=0.0, t_end=2.0)
    rho = combined_reactivity(first, second)

    assert rho(2.0) == pytest.approx(0.005)
