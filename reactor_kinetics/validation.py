"""Validation utilities for six-group point-kinetics behaviour."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .point_kinetics import KineticsParameters
from .scenarios import step_reactivity, zero_reactivity
from .solver import SimulationResult, solve_point_kinetics


@dataclass(frozen=True)
class PeriodEstimate:
    """Exponential period estimate from a log-power fit."""

    period: float
    growth_rate: float
    r_squared: float


@dataclass(frozen=True)
class ValidationResult:
    """Result for one validation check."""

    name: str
    passed: bool
    criterion: str
    message: str
    metrics: Mapping[str, float]
    simulation: SimulationResult | None = None
    period_estimate: PeriodEstimate | None = None


def estimate_reactor_period(
    time: NDArray[np.float64],
    neutron_population: NDArray[np.float64],
    t_min: float | None = None,
    t_max: float | None = None,
) -> PeriodEstimate:
    """Estimate reactor period from a linear fit of log(n) versus time.

    Args:
        time: Time coordinates [s].
        neutron_population: Positive neutron population or relative power.
        t_min: Optional lower time bound for the fit [s].
        t_max: Optional upper time bound for the fit [s].

    Returns:
        PeriodEstimate containing period [s], inverse period [1/s], and R2.
    """

    time_array = np.asarray(time, dtype=float)
    power_array = np.asarray(neutron_population, dtype=float)
    if time_array.shape != power_array.shape or time_array.ndim != 1:
        raise ValueError("time and neutron_population must be matching 1D arrays.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(power_array)):
        raise ValueError("time and neutron_population must be finite.")
    if np.any(power_array <= 0.0):
        raise ValueError("neutron_population must be positive for a logarithmic fit.")

    mask = np.ones_like(time_array, dtype=bool)
    if t_min is not None:
        mask &= time_array >= float(t_min)
    if t_max is not None:
        mask &= time_array <= float(t_max)
    if np.count_nonzero(mask) < 3:
        raise ValueError("period fit requires at least three selected points.")

    fit_time = time_array[mask]
    log_power = np.log(power_array[mask])
    growth_rate, intercept = np.polyfit(fit_time, log_power, 1)
    fitted = growth_rate * fit_time + intercept
    residual_sum = float(np.sum((log_power - fitted) ** 2))
    total_sum = float(np.sum((log_power - float(np.mean(log_power))) ** 2))
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    period = np.inf if growth_rate == 0.0 else 1.0 / growth_rate
    return PeriodEstimate(
        period=float(period),
        growth_rate=float(growth_rate),
        r_squared=float(r_squared),
    )


def inhour_omega(rho: float, params: KineticsParameters) -> float:
    """Solve the point-kinetics Inhour equation for positive inverse period.

    The relation solved is:

        rho = omega Lambda + sum_i beta_i omega / (omega + lambda_i)
    """

    rho_value = float(rho)
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("inhour_omega currently supports finite positive rho only.")

    def residual(omega: float) -> float:
        return float(
            omega * params.Lambda
            + np.sum(params.beta * omega / (omega + params.lambdas))
            - rho_value
        )

    lower = 0.0
    upper = max(1.0, rho_value / params.Lambda)
    while residual(upper) <= 0.0:
        upper *= 2.0
    return float(brentq(residual, lower, upper, xtol=1.0e-13, rtol=1.0e-12))


def inhour_period(rho: float, params: KineticsParameters) -> float:
    """Return the positive reactor period from the Inhour equation."""

    return 1.0 / inhour_omega(rho=rho, params=params)


def validate_zero_reactivity_steady_state(
    params: KineticsParameters,
    tolerance: float = 1.0e-5,
) -> ValidationResult:
    """Validate that rho = 0 preserves the critical steady state."""

    result = solve_point_kinetics(
        params=params,
        reactivity=zero_reactivity(),
        t_span=(0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 500),
        max_step=0.1,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Zero reactivity steady state",
    )
    initial = float(result.neutron_population[0])
    relative_deviation = float(
        np.max(np.abs(result.neutron_population - initial)) / initial
    )
    passed = relative_deviation < tolerance
    return ValidationResult(
        name="Zero reactivity steady state",
        passed=passed,
        criterion=f"max(abs(n(t) - n0)) / n0 < {tolerance:g}",
        message=f"relative deviation = {relative_deviation:.3e}",
        metrics={
            "relative_max_deviation": relative_deviation,
            "tolerance": float(tolerance),
        },
        simulation=result,
    )


def validate_negative_reactivity_decay(params: KineticsParameters) -> ValidationResult:
    """Validate that a negative half-beta step causes power to decay."""

    rho = -0.5 * params.beta_eff
    result = solve_point_kinetics(
        params=params,
        reactivity=step_reactivity(rho_step=rho, t_step=1.0),
        t_span=(0.0, 30.0),
        t_eval=np.linspace(0.0, 30.0, 700),
        max_step=0.1,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Negative reactivity decay: rho = -0.5 beta_eff",
    )
    initial = float(result.neutron_population[0])
    final = float(result.neutron_population[-1])
    final_fraction = final / initial
    passed = final < initial and final_fraction < 0.8
    return ValidationResult(
        name="Negative reactivity decay",
        passed=passed,
        criterion="n_final < 0.8 n_initial for rho = -0.5 beta_eff",
        message=f"final / initial = {final_fraction:.3f}",
        metrics={
            "rho": rho,
            "rho_over_beta_eff": rho / params.beta_eff,
            "final_fraction": final_fraction,
        },
        simulation=result,
    )


def validate_delayed_supercritical_growth(
    params: KineticsParameters,
) -> ValidationResult:
    """Validate delayed-supercritical growth for rho = 0.5 beta_eff."""

    rho = 0.5 * params.beta_eff
    result = solve_point_kinetics(
        params=params,
        reactivity=step_reactivity(rho_step=rho, t_step=1.0),
        t_span=(0.0, 30.0),
        t_eval=np.linspace(0.0, 30.0, 700),
        max_step=0.05,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Delayed-supercritical growth: rho = 0.5 beta_eff",
    )
    initial = float(result.neutron_population[0])
    final = float(result.neutron_population[-1])
    period = estimate_reactor_period(
        result.time,
        result.neutron_population,
        t_min=10.0,
        t_max=30.0,
    )
    passed = final > initial and period.growth_rate > 0.0
    return ValidationResult(
        name="Delayed-supercritical growth",
        passed=passed,
        criterion="n_final > n_initial for rho = 0.5 beta_eff",
        message=(
            f"final / initial = {final / initial:.3g}, "
            f"period = {period.period:.3g} s"
        ),
        metrics={
            "rho": rho,
            "rho_over_beta_eff": rho / params.beta_eff,
            "final_fraction": final / initial,
            "growth_rate": period.growth_rate,
            "period": period.period,
            "r_squared": period.r_squared,
        },
        simulation=result,
        period_estimate=period,
    )


def validate_prompt_supercritical_growth_rate(
    params: KineticsParameters,
) -> ValidationResult:
    """Validate that prompt-supercritical growth is faster than delayed growth."""

    delayed_rho = 0.5 * params.beta_eff
    prompt_rho = 1.2 * params.beta_eff
    delayed = solve_point_kinetics(
        params=params,
        reactivity=step_reactivity(rho_step=delayed_rho, t_step=0.0),
        t_span=(0.0, 40.0),
        t_eval=np.linspace(0.0, 40.0, 900),
        max_step=0.05,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Delayed-supercritical comparison case",
    )
    prompt = solve_point_kinetics(
        params=params,
        reactivity=step_reactivity(rho_step=prompt_rho, t_step=0.0),
        t_span=(0.0, 0.08),
        t_eval=np.linspace(0.0, 0.08, 900),
        max_step=1.0e-4,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Prompt-supercritical growth: rho = 1.2 beta_eff",
    )
    delayed_period = estimate_reactor_period(
        delayed.time,
        delayed.neutron_population,
        t_min=20.0,
        t_max=40.0,
    )
    prompt_period = estimate_reactor_period(
        prompt.time,
        prompt.neutron_population,
        t_min=0.02,
        t_max=0.08,
    )
    passed = prompt_period.growth_rate > delayed_period.growth_rate
    return ValidationResult(
        name="Prompt-supercritical growth-rate comparison",
        passed=passed,
        criterion="growth_rate_prompt > growth_rate_delayed",
        message=(
            f"prompt growth rate = {prompt_period.growth_rate:.3g} 1/s, "
            f"delayed growth rate = {delayed_period.growth_rate:.3g} 1/s"
        ),
        metrics={
            "prompt_rho": prompt_rho,
            "prompt_rho_over_beta_eff": prompt_rho / params.beta_eff,
            "delayed_growth_rate": delayed_period.growth_rate,
            "prompt_growth_rate": prompt_period.growth_rate,
            "growth_rate_ratio": prompt_period.growth_rate / delayed_period.growth_rate,
            "prompt_period": prompt_period.period,
            "prompt_r_squared": prompt_period.r_squared,
            "delayed_period": delayed_period.period,
            "delayed_r_squared": delayed_period.r_squared,
        },
        simulation=prompt,
        period_estimate=prompt_period,
    )


def validate_prompt_jump(
    params: KineticsParameters,
    tolerance: float = 0.10,
) -> ValidationResult:
    """Compare a sub-prompt-critical step with the prompt-jump approximation."""

    t_step = 1.0
    observation_delay = 0.02
    rho = 0.2 * params.beta_eff
    expected_ratio = params.beta_eff / (params.beta_eff - rho)
    t_eval = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 0.99, 80),
                np.linspace(0.995, 1.04, 300),
                np.array([t_step, t_step + observation_delay]),
                np.linspace(1.05, 2.0, 80),
            )
        )
    )
    result = solve_point_kinetics(
        params=params,
        reactivity=step_reactivity(rho_step=rho, t_step=t_step),
        t_span=(0.0, 2.0),
        t_eval=t_eval,
        max_step=1.0e-3,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Prompt jump validation: rho = 0.2 beta_eff",
    )
    n_minus = float(result.neutron_population[result.time < t_step][-1])
    plus_index = int(np.argmin(np.abs(result.time - (t_step + observation_delay))))
    n_plus = float(result.neutron_population[plus_index])
    observed_ratio = n_plus / n_minus
    relative_error = abs(observed_ratio - expected_ratio) / expected_ratio
    passed = relative_error < tolerance
    return ValidationResult(
        name="Prompt jump approximation",
        passed=passed,
        criterion=f"relative error < {tolerance:g}",
        message=(
            f"observed ratio = {observed_ratio:.4f}, "
            f"theory = {expected_ratio:.4f}, "
            f"relative error = {relative_error:.3%}"
        ),
        metrics={
            "rho": rho,
            "rho_over_beta_eff": rho / params.beta_eff,
            "t_step": t_step,
            "observation_delay": observation_delay,
            "n_minus": n_minus,
            "n_plus": n_plus,
            "observed_ratio": observed_ratio,
            "expected_ratio": expected_ratio,
            "relative_error": relative_error,
            "tolerance": float(tolerance),
        },
        simulation=result,
    )


def validate_inhour_comparison(
    params: KineticsParameters,
    tolerance: float = 0.10,
) -> ValidationResult:
    """Compare numerical log-slope growth with the Inhour equation."""

    rho = 0.5 * params.beta_eff
    result = solve_point_kinetics(
        params=params,
        reactivity=step_reactivity(rho_step=rho, t_step=0.0),
        t_span=(0.0, 40.0),
        t_eval=np.linspace(0.0, 40.0, 900),
        max_step=0.05,
        rtol=1.0e-9,
        atol=1.0e-11,
        title="Inhour comparison: rho = 0.5 beta_eff",
    )
    period = estimate_reactor_period(
        result.time,
        result.neutron_population,
        t_min=20.0,
        t_max=40.0,
    )
    omega_numerical = period.growth_rate
    omega_inhour = inhour_omega(rho=rho, params=params)
    relative_error = abs(omega_numerical - omega_inhour) / omega_inhour
    passed = relative_error < tolerance and period.r_squared > 0.999
    return ValidationResult(
        name="Inhour equation comparison",
        passed=passed,
        criterion=f"relative omega error < {tolerance:g}",
        message=(
            f"numerical omega = {omega_numerical:.5g} 1/s, "
            f"Inhour omega = {omega_inhour:.5g} 1/s, "
            f"relative error = {relative_error:.3%}"
        ),
        metrics={
            "rho": rho,
            "rho_over_beta_eff": rho / params.beta_eff,
            "omega_numerical": omega_numerical,
            "omega_inhour": omega_inhour,
            "period_numerical": period.period,
            "period_inhour": 1.0 / omega_inhour,
            "relative_error": relative_error,
            "tolerance": float(tolerance),
            "r_squared": period.r_squared,
        },
        simulation=result,
        period_estimate=period,
    )


def run_validation_suite(params: KineticsParameters) -> list[ValidationResult]:
    """Run the standard Phase 2 validation checks."""

    return [
        validate_zero_reactivity_steady_state(params),
        validate_negative_reactivity_decay(params),
        validate_delayed_supercritical_growth(params),
        validate_prompt_supercritical_growth_rate(params),
        validate_prompt_jump(params),
        validate_inhour_comparison(params),
    ]
