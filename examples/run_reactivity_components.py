"""Run Phase 3 reactivity-component demonstration cases."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

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
from reactor_kinetics.reactivity import (  # noqa: E402
    ReactivityInputs,
    ReactivityModel,
    linear_ramp_input,
    rho_to_pcm,
)
from reactor_kinetics.solver import SimulationResult, solve_point_kinetics  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "reactivity_outputs"


def _load_input_schema() -> dict[str, object]:
    schema_path = PROJECT_ROOT / "schemas" / "input_schema.yaml"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    if not isinstance(schema, dict) or "inputs" not in schema:
        raise ValueError("input schema must contain an 'inputs' mapping.")
    return schema


def _plot_case(
    result: SimulationResult,
    save_path: Path,
    temperature_traces: dict[str, np.ndarray] | None = None,
) -> None:
    has_temperatures = temperature_traces is not None
    row_count = 4 if has_temperatures else 3
    fig, axes = plt.subplots(row_count, 1, figsize=(9, 2.9 * row_count), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(result.time, result.neutron_population, color="tab:blue")
    axes[0].set_ylabel("n / n0 [-]")
    axes[0].set_title(result.title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.time, [rho_to_pcm(value) for value in result.reactivity])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(
        rho_to_pcm(result.params.beta_eff),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="beta_eff",
    )
    axes[1].set_ylabel("total rho [pcm]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize="small")

    if result.reactivity_components is not None:
        for name, values in result.reactivity_components.items():
            if name == "total":
                continue
            axes[2].plot(
                result.time,
                [rho_to_pcm(value) for value in values],
                label=name.replace("_", " "),
            )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("component rho [pcm]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize="small", ncol=2)

    if temperature_traces is not None:
        axes[3].plot(
            result.time,
            temperature_traces["fuel_temperature"],
            label="fuel",
        )
        axes[3].plot(
            result.time,
            temperature_traces["moderator_temperature"],
            label="moderator",
        )
        axes[3].set_ylabel("temperature [K]")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(fontsize="small")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _run_case(
    title: str,
    inputs: ReactivityInputs,
    t_end: float,
    filename: str,
    temperature_traces: dict[str, np.ndarray] | None = None,
) -> SimulationResult:
    config = load_reactor_config(PROJECT_ROOT / "configs" / "pwr_reference.yaml")
    if config.reactivity is None:
        raise ValueError("reference configuration must include reactivity coefficients.")

    result = solve_point_kinetics(
        params=config.kinetics,
        reactivity=ReactivityModel(config.reactivity, inputs),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, 700),
        max_step=min(0.02, t_end / 100.0),
        rtol=1.0e-8,
        atol=1.0e-10,
        title=title,
    )
    _plot_case(result, OUTPUT_DIR / filename, temperature_traces=temperature_traces)
    return result


def run_cases() -> list[SimulationResult]:
    """Run all Phase 3 examples and save their plots."""

    _load_input_schema()
    config = load_reactor_config(PROJECT_ROOT / "configs" / "pwr_reference.yaml")
    if config.reactivity is None:
        raise ValueError("reference configuration must include reactivity coefficients.")
    beta_eff = config.kinetics.beta_eff

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[SimulationResult] = []
    cases.append(
        _run_case(
            title="Rod withdrawal: insertion fraction 0.5 to 0.45",
            inputs=ReactivityInputs(
                rod_insertion_fraction=linear_ramp_input(0.5, 0.45, 2.0, 10.0),
                boron_ppm=1000.0,
                fuel_temperature=900.0,
                moderator_temperature=580.0,
            ),
            t_end=30.0,
            filename="rod_withdrawal.png",
        )
    )
    cases.append(
        _run_case(
            title="Rod insertion: insertion fraction 0.5 to 0.55",
            inputs=ReactivityInputs(
                rod_insertion_fraction=linear_ramp_input(0.5, 0.55, 2.0, 10.0),
                boron_ppm=1000.0,
                fuel_temperature=900.0,
                moderator_temperature=580.0,
            ),
            t_end=30.0,
            filename="rod_insertion.png",
        )
    )
    cases.append(
        _run_case(
            title="Boron dilution: 1000 ppm to 950 ppm",
            inputs=ReactivityInputs(
                rod_insertion_fraction=0.5,
                boron_ppm=linear_ramp_input(1000.0, 950.0, 2.0, 10.0),
                fuel_temperature=900.0,
                moderator_temperature=580.0,
            ),
            t_end=30.0,
            filename="boron_dilution.png",
        )
    )
    cases.append(
        _run_case(
            title="Boron addition: 1000 ppm to 1050 ppm",
            inputs=ReactivityInputs(
                rod_insertion_fraction=0.5,
                boron_ppm=linear_ramp_input(1000.0, 1050.0, 2.0, 10.0),
                fuel_temperature=900.0,
                moderator_temperature=580.0,
            ),
            t_end=30.0,
            filename="boron_addition.png",
        )
    )

    t_temperature = np.linspace(0.0, 40.0, 700)
    fuel_temperature = np.array(
        [linear_ramp_input(900.0, 930.0, 5.0, 25.0)(t) for t in t_temperature],
        dtype=float,
    )
    moderator_temperature = np.array(
        [linear_ramp_input(580.0, 585.0, 5.0, 25.0)(t) for t in t_temperature],
        dtype=float,
    )
    cases.append(
        _run_case(
            title="Prescribed temperature increase with negative feedback",
            inputs=ReactivityInputs(
                external_rho=linear_ramp_input(0.0, 0.3 * beta_eff, 1.0, 8.0),
                rod_insertion_fraction=0.5,
                boron_ppm=1000.0,
                fuel_temperature=linear_ramp_input(900.0, 930.0, 5.0, 25.0),
                moderator_temperature=linear_ramp_input(580.0, 585.0, 5.0, 25.0),
            ),
            t_end=40.0,
            filename="temperature_feedback.png",
            temperature_traces={
                "fuel_temperature": fuel_temperature,
                "moderator_temperature": moderator_temperature,
            },
        )
    )
    cases.append(
        _run_case(
            title="External ramp: rho = 0 to 0.5 beta_eff",
            inputs=ReactivityInputs(
                external_rho=linear_ramp_input(0.0, 0.5 * beta_eff, 2.0, 20.0),
                rod_insertion_fraction=0.5,
                boron_ppm=1000.0,
                fuel_temperature=900.0,
                moderator_temperature=580.0,
            ),
            t_end=30.0,
            filename="external_ramp.png",
        )
    )
    return cases


def main() -> None:
    results = run_cases()
    for result in results:
        print(
            f"{result.title}\n"
            f"  final relative power = {result.neutron_population[-1]:.6g}\n"
            f"  final total rho      = {result.reactivity[-1]:.6g}\n"
        )
    print(f"Saved reactivity-component plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
