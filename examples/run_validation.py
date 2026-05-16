"""Run Phase 2 validation checks and save validation plots."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

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
    plot_log_neutron_population,
    plot_neutron_population,
)
from reactor_kinetics.validation import (  # noqa: E402
    ValidationResult,
    run_validation_suite,
)


def _result_by_name(
    results: list[ValidationResult],
    name: str,
) -> ValidationResult:
    for result in results:
        if result.name == name:
            return result
    raise KeyError(name)


def _print_report(config_name: str, results: list[ValidationResult]) -> None:
    print(f"Validation configuration: {config_name}")
    print("Phase 2 validation report")
    for result in results:
        status = "PASS" if result.passed else "CHECK"
        print(f"  [{status}] {result.name}: {result.message}")
        print(f"        criterion: {result.criterion}")


def _save_prompt_jump_plot(result: ValidationResult, output_path: Path) -> None:
    simulation = result.simulation
    if simulation is None:
        return

    t_step = result.metrics["t_step"]
    n_minus = result.metrics["n_minus"]
    t_window = (
        simulation.time >= t_step - 0.03
    ) & (simulation.time <= t_step + 0.08)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        simulation.time[t_window],
        simulation.neutron_population[t_window] / n_minus,
        label="Numerical n / n_minus",
    )
    ax.axvline(t_step, color="black", linewidth=0.8, linestyle="--", label="Step")
    ax.axhline(
        result.metrics["expected_ratio"],
        color="tab:red",
        linestyle=":",
        label="Prompt-jump approximation",
    )
    ax.scatter(
        [t_step + result.metrics["observation_delay"]],
        [result.metrics["observed_ratio"]],
        color="tab:orange",
        zorder=3,
        label="Extraction point",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("n / n_minus [-]")
    ax.set_title("Prompt jump validation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_inhour_plot(result: ValidationResult, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Numerical log-slope", "Inhour equation"]
    values = [result.metrics["omega_numerical"], result.metrics["omega_inhour"]]
    ax.bar(labels, values, color=["tab:blue", "tab:green"])
    ax.set_ylabel("Inverse period omega [1/s]")
    ax.set_title("Inhour comparison for rho = 0.5 beta_eff")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_standard_plots(results: list[ValidationResult], output_dir: Path) -> None:
    plot_specs = [
        (
            "Zero reactivity steady state",
            output_dir / "zero_reactivity_steady_state.png",
            plot_neutron_population,
        ),
        (
            "Negative reactivity decay",
            output_dir / "negative_reactivity_decay.png",
            plot_neutron_population,
        ),
        (
            "Delayed-supercritical growth",
            output_dir / "delayed_supercritical_growth.png",
            plot_log_neutron_population,
        ),
        (
            "Prompt-supercritical growth-rate comparison",
            output_dir / "prompt_supercritical_growth.png",
            plot_log_neutron_population,
        ),
    ]
    for name, path, plotter in plot_specs:
        validation_result = _result_by_name(results, name)
        if validation_result.simulation is not None:
            plotter(validation_result.simulation, save_path=path, close=True)

    _save_prompt_jump_plot(
        _result_by_name(results, "Prompt jump approximation"),
        output_dir / "prompt_jump_validation.png",
    )
    _save_inhour_plot(
        _result_by_name(results, "Inhour equation comparison"),
        output_dir / "inhour_comparison.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pwr_reference.yaml",
        help="YAML model configuration to validate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "validation" / "outputs",
        help="Directory where validation plots are saved.",
    )
    args = parser.parse_args()

    config = load_reactor_config(args.config)
    results = run_validation_suite(config.kinetics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_standard_plots(results, args.output_dir)

    _print_report(config.name, results)
    print(f"Saved validation plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
