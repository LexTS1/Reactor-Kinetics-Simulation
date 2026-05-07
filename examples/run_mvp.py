"""Run the Physics MVP point-kinetics demonstration cases."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "reactor_kinetics_mpl_cache"),
)

from matplotlib import pyplot as plt  # noqa: E402

from reactor_kinetics.plotting import plot_summary  # noqa: E402
from reactor_kinetics.point_kinetics import (  # noqa: E402
    ThermalFeedbackParameters,
    default_u235_parameters,
)
from reactor_kinetics.scenarios import (  # noqa: E402
    negative_step_reactivity,
    ramp_reactivity,
    step_reactivity,
)
from reactor_kinetics.solver import SimulationResult, solve_point_kinetics  # noqa: E402


def run_cases() -> list[SimulationResult]:
    """Run the required MVP demonstration transients."""

    params = default_u235_parameters()
    beta_eff = params.beta_eff

    delayed_step_rho = 0.5 * beta_eff
    negative_step_rho = 0.5 * beta_eff
    prompt_step_rho = 1.2 * beta_eff
    ramp_end_rho = 0.8 * beta_eff

    thermal_initial_moderator_temperature = 600.0
    thermal_initial_fuel_temperature = 610.0
    thermal_coolant_temperature = 560.0

    thermal_params = ThermalFeedbackParameters(
        fuel_heat_capacity=20.0,
        moderator_heat_capacity=200.0,
        fuel_moderator_heat_transfer=0.10,
        moderator_coolant_heat_transfer=0.025,
        coolant_temperature=thermal_coolant_temperature,
        initial_fuel_temperature=thermal_initial_fuel_temperature,
        initial_moderator_temperature=thermal_initial_moderator_temperature,
        fuel_temperature_coefficient=-4.0e-5,
        moderator_temperature_coefficient=-2.0e-5,
        nominal_power=1.0,
    )

    cases = [
        {
            "title": "Delayed-supercritical positive step: rho = 0.5 beta_eff",
            "reactivity": step_reactivity(delayed_step_rho, t_step=1.0),
            "t_span": (0.0, 30.0),
            "t_eval": np.linspace(0.0, 30.0, 700),
            "max_step": 0.2,
            "thermal_params": None,
        },
        {
            "title": "Subcritical negative step: rho = -0.5 beta_eff",
            "reactivity": negative_step_reactivity(negative_step_rho, t_step=1.0),
            "t_span": (0.0, 30.0),
            "t_eval": np.linspace(0.0, 30.0, 700),
            "max_step": 0.2,
            "thermal_params": None,
        },
        {
            "title": "Prompt-supercritical step: rho = 1.2 beta_eff",
            "reactivity": step_reactivity(prompt_step_rho, t_step=0.005),
            "t_span": (0.0, 0.05),
            "t_eval": np.linspace(0.0, 0.05, 500),
            "max_step": 1.0e-4,
            "thermal_params": None,
        },
        {
            "title": "Ramp insertion: rho = 0 to 0.8 beta_eff",
            "reactivity": ramp_reactivity(
                rho_start=0.0,
                rho_end=ramp_end_rho,
                t_start=2.0,
                t_end=20.0,
            ),
            "t_span": (0.0, 20.0),
            "t_eval": np.linspace(0.0, 20.0, 700),
            "max_step": 0.2,
            "thermal_params": None,
        },
        {
            "title": "Thermal feedback: positive step with negative temperature coefficients",
            "reactivity": step_reactivity(0.6 * beta_eff, t_step=1.0),
            "t_span": (0.0, 200.0),
            "t_eval": np.linspace(0.0, 200.0, 1200),
            "max_step": 0.5,
            "thermal_params": thermal_params,
        },
    ]

    results: list[SimulationResult] = []
    for case in cases:
        result = solve_point_kinetics(
            params=params,
            reactivity=case["reactivity"],
            t_span=case["t_span"],
            t_eval=case["t_eval"],
            thermal_params=case["thermal_params"],
            method="BDF",
            rtol=1.0e-8,
            atol=1.0e-10,
            max_step=case["max_step"],
            title=case["title"],
        )
        results.append(result)
    return results


def print_case_summary(results: list[SimulationResult]) -> None:
    """Print a compact numerical summary of the demonstration cases."""

    for result in results:
        final_power = result.neutron_population[-1]
        peak_power = float(np.max(result.neutron_population))
        final_reactivity = result.reactivity[-1]
        print(
            f"{result.title}\n"
            f"  final relative power = {final_power:.6g}\n"
            f"  peak relative power  = {peak_power:.6g}\n"
            f"  final net rho        = {final_reactivity:.6g}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create figures without opening an interactive Matplotlib window.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory where summary figures are saved as PNG files.",
    )
    args = parser.parse_args()

    results = run_cases()
    print_case_summary(results)

    figures = [plot_summary(result) for result in results]

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for index, figure in enumerate(figures, start=1):
            figure.savefig(args.save_dir / f"mvp_case_{index}.png", dpi=160)

    if not args.no_show and "agg" in plt.get_backend().lower():
        print("Matplotlib backend is non-interactive; use --save-dir to write PNGs.")
    elif not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
