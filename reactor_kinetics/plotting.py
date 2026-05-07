"""Matplotlib plotting helpers for point-kinetics results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .solver import SimulationResult


def _finalize_figure(
    fig: Figure,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Save, show, and/or close a Matplotlib figure."""

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


def plot_neutron_population(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Plot neutron population / relative power."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.time, result.neutron_population, label="Relative power")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("n / n0 [-]")
    ax.set_title(result.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finalize_figure(fig, save_path=save_path, show=show, close=close)


def plot_log_neutron_population(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Plot neutron population / relative power on a logarithmic axis."""

    fig, ax = plt.subplots(figsize=(8, 4))
    positive_power = np.clip(result.neutron_population, 1.0e-300, None)
    ax.semilogy(result.time, positive_power, label="Relative power")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("n / n0 [-]")
    ax.set_title(f"{result.title}: logarithmic power")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finalize_figure(fig, save_path=save_path, show=show, close=close)


def plot_precursors(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Plot delayed neutron precursor concentrations."""

    fig, ax = plt.subplots(figsize=(8, 4))
    for group_index, precursor in enumerate(result.precursors, start=1):
        ax.plot(result.time, precursor, label=f"C{group_index}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Precursor concentration [relative]")
    ax.set_title(f"{result.title}: delayed neutron precursors")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()
    return _finalize_figure(fig, save_path=save_path, show=show, close=close)


def plot_reactivity(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Plot net reactivity."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.time, result.reactivity, label="Net reactivity")
    ax.axhline(result.params.beta_eff, color="tab:red", linestyle="--", label="beta_eff")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reactivity rho [-]")
    ax.set_title(f"{result.title}: reactivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finalize_figure(fig, save_path=save_path, show=show, close=close)


def plot_temperatures(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Plot fuel and moderator temperatures for thermal-feedback results."""

    if not result.has_thermal_feedback:
        raise ValueError("result does not contain thermal-feedback temperatures.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.time, result.fuel_temperature, label="Fuel temperature")
    ax.plot(result.time, result.moderator_temperature, label="Moderator temperature")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"{result.title}: temperatures")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finalize_figure(fig, save_path=save_path, show=show, close=close)


def plot_summary(
    result: SimulationResult,
    save_path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
) -> Figure:
    """Create a multi-panel summary plot for one simulation result."""

    panel_count = 5 if result.has_thermal_feedback else 4
    fig, axes = plt.subplots(panel_count, 1, figsize=(9, 2.7 * panel_count), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(result.time, result.neutron_population, color="tab:blue")
    axes[0].set_ylabel("n / n0 [-]")
    axes[0].set_title(result.title)
    axes[0].grid(True, alpha=0.3)

    positive_power = np.clip(result.neutron_population, 1.0e-300, None)
    axes[1].semilogy(result.time, positive_power, color="tab:orange")
    axes[1].set_ylabel("n / n0 [-]")
    axes[1].set_title("Log relative power")
    axes[1].grid(True, which="both", alpha=0.3)

    for group_index, precursor in enumerate(result.precursors, start=1):
        axes[2].plot(result.time, precursor, label=f"C{group_index}")
    axes[2].set_ylabel("C_i [relative]")
    axes[2].set_title("Delayed neutron precursors")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=3, fontsize="small")

    axes[3].plot(result.time, result.reactivity, color="tab:green", label="Net rho")
    axes[3].axhline(
        result.params.beta_eff,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="beta_eff",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel("rho [-]")
    axes[3].set_title("Reactivity")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize="small")

    if result.has_thermal_feedback:
        axes[4].plot(result.time, result.fuel_temperature, label="Fuel")
        axes[4].plot(result.time, result.moderator_temperature, label="Moderator")
        axes[4].set_ylabel("Temperature [K]")
        axes[4].set_title("Lumped temperatures")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(fontsize="small")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    return _finalize_figure(fig, save_path=save_path, show=show, close=close)
