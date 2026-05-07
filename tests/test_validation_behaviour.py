from __future__ import annotations

from pathlib import Path

from reactor_kinetics.config import load_reactor_config
from reactor_kinetics.validation import (
    validate_delayed_supercritical_growth,
    validate_inhour_comparison,
    validate_negative_reactivity_decay,
    validate_prompt_jump,
    validate_prompt_supercritical_growth_rate,
    validate_zero_reactivity_steady_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_reactor_config(PROJECT_ROOT / "configs" / "pwr_reference.yaml").kinetics


def test_zero_reactivity_gives_steady_power() -> None:
    result = validate_zero_reactivity_steady_state(PARAMS)

    assert result.passed
    assert result.metrics["relative_max_deviation"] < result.metrics["tolerance"]


def test_negative_reactivity_decays() -> None:
    result = validate_negative_reactivity_decay(PARAMS)

    assert result.passed
    assert result.metrics["final_fraction"] < 0.8


def test_positive_reactivity_grows() -> None:
    result = validate_delayed_supercritical_growth(PARAMS)

    assert result.passed
    assert result.metrics["final_fraction"] > 1.0
    assert result.metrics["growth_rate"] > 0.0


def test_prompt_supercritical_grows_faster_than_delayed_supercritical() -> None:
    result = validate_prompt_supercritical_growth_rate(PARAMS)

    assert result.passed
    assert result.metrics["prompt_growth_rate"] > result.metrics["delayed_growth_rate"]


def test_prompt_jump_matches_theory_with_engineering_tolerance() -> None:
    result = validate_prompt_jump(PARAMS)

    assert result.passed
    assert result.metrics["relative_error"] < result.metrics["tolerance"]


def test_inhour_period_matches_extracted_numerical_period() -> None:
    result = validate_inhour_comparison(PARAMS)

    assert result.passed
    assert result.metrics["relative_error"] < result.metrics["tolerance"]
    assert result.metrics["r_squared"] > 0.999
