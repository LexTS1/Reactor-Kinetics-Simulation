"""Run coupled point-kinetics and lumped thermal-feedback examples."""

from __future__ import annotations

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

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from reactor_kinetics.config import load_reactor_config  # noqa: E402
from reactor_kinetics.plotting import (  # noqa: E402
    plot_summary,
    plot_thermal_feedback_summary,
)
from reactor_kinetics.point_kinetics import coupled_initial_state  # noqa: E402
from reactor_kinetics.reactivity import (  # noqa: E402
    ReactivityInputs,
    ReactivityModel,
    step_input,
)
from reactor_kinetics.solver import SimulationResult, solve_point_kinetics  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "thermal_outputs"


def _reactivity_model(
    config_reactivity,
    external_rho: float,
    *,
    feedback_enabled: bool,
) -> ReactivityModel:
    fuel_temperature = None if feedback_enabled else config_reactivity.reference_fuel_temperature
    moderator_temperature = (
        None
        if feedback_enabled
        else config_reactivity.reference_moderator_temperature
    )
    return ReactivityModel(
        config_reactivity,
        ReactivityInputs(
            external_rho=step_input(0.0, external_rho, 0.5),
            rod_insertion_fraction=config_reactivity.reference_rod_insertion_fraction,
            boron_ppm=config_reactivity.reference_boron_ppm,
            fuel_temperature=fuel_temperature,
            moderator_temperature=moderator_temperature,
        ),
    )


def _run_case(
    *,
    title: str,
    external_rho: float,
    feedback_enabled: bool,
    filename: str,
) -> SimulationResult:
    config = load_reactor_config(PROJECT_ROOT / "configs" / "pwr_reference.yaml")
    if config.reactivity is None:
        raise ValueError("reference configuration must include reactivity coefficients.")
    if feedback_enabled and config.thermal is None:
        raise ValueError("reference configuration must include thermal parameters.")

    if feedback_enabled:
        _ = coupled_initial_state(config.kinetics, config.thermal)

    result = solve_point_kinetics(
        params=config.kinetics,
        reactivity=_reactivity_model(
            config.reactivity,
            external_rho,
            feedback_enabled=feedback_enabled,
        ),
        thermal_params=config.thermal if feedback_enabled else None,
        t_span=(0.0, 8.0),
        t_eval=np.linspace(0.0, 8.0, 700),
        max_step=0.01,
        rtol=1.0e-8,
        atol=1.0e-10,
        title=title,
    )

    save_path = OUTPUT_DIR / filename
    if result.has_thermal_feedback:
        plot_thermal_feedback_summary(result, save_path=save_path, show=False, close=True)
    else:
        plot_summary(result, save_path=save_path, show=False, close=True)
    return result


def _plot_comparison(results: list[SimulationResult], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for result in results:
        axes[0].plot(result.time, result.neutron_population, label=result.title)
        axes[1].plot(result.time, result.rho_total, label=result.title)

    axes[0].set_ylabel("Relative power [-]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize="small")

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Reactivity Δk/k [-]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def run_cases() -> list[SimulationResult]:
    """Run Phase 4 thermal-feedback examples and save plots."""

    config = load_reactor_config(PROJECT_ROOT / "configs" / "pwr_reference.yaml")
    beta_eff = config.kinetics.beta_eff
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        _run_case(
            title="No feedback: positive 0.3 beta step",
            external_rho=0.3 * beta_eff,
            feedback_enabled=False,
            filename="no_feedback_positive_step.png",
        ),
        _run_case(
            title="Thermal feedback: positive 0.3 beta step",
            external_rho=0.3 * beta_eff,
            feedback_enabled=True,
            filename="feedback_positive_step.png",
        ),
        _run_case(
            title="Thermal feedback: positive 0.5 beta step",
            external_rho=0.5 * beta_eff,
            feedback_enabled=True,
            filename="feedback_stronger_step.png",
        ),
    ]

    _plot_comparison(results, OUTPUT_DIR / "feedback_comparison.png")
    return results


def main() -> None:
    results = run_cases()
    for result in results:
        feedback = result.rho_feedback[-1] if result.has_thermal_feedback else 0.0
        print(
            f"{result.title}\n"
            f"  final relative power = {result.neutron_population[-1]:.6g}\n"
            f"  final total rho      = {result.rho_total[-1]:.6g}\n"
            f"  final feedback rho   = {feedback:.6g}\n"
        )
    print(f"Saved thermal-feedback plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
