"""YAML configuration loading for reactor kinetics simulations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .point_kinetics import KineticsParameters, ThermalFeedbackParameters


@dataclass(frozen=True)
class ReactorConfig:
    """Validated model configuration loaded from a YAML file."""

    name: str
    description: str
    kinetics: KineticsParameters
    thermal_enabled: bool
    thermal: ThermalFeedbackParameters | None = None


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration field '{key}' must be a mapping.")
    return value


def _require_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration field '{key}' must be a non-empty string.")
    return value


def _require_float(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"configuration field '{key}' must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configuration field '{key}' must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"configuration field '{key}' must be finite.")
    return number


def _require_positive_float(data: Mapping[str, Any], key: str) -> float:
    number = _require_float(data, key)
    if number <= 0.0:
        raise ValueError(f"configuration field '{key}' must be positive.")
    return number


def _require_float_array(data: Mapping[str, Any], key: str) -> np.ndarray:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"configuration field '{key}' must be a list.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configuration field '{key}' must contain numbers.") from exc
    if array.shape != (6,):
        raise ValueError(f"configuration field '{key}' must contain exactly 6 values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"configuration field '{key}' must contain finite values.")
    return array


def _load_kinetics(data: Mapping[str, Any]) -> KineticsParameters:
    kinetics = _require_mapping(data, "kinetics")
    params = KineticsParameters(
        beta=_require_float_array(kinetics, "beta"),
        lambdas=_require_float_array(kinetics, "lambda"),
        Lambda=_require_positive_float(kinetics, "Lambda"),
    )
    if params.beta_eff <= 0.0:
        raise ValueError("configuration kinetics beta_eff must be positive.")
    return params


def _load_thermal(data: Mapping[str, Any]) -> tuple[bool, ThermalFeedbackParameters | None]:
    thermal = data.get("thermal")
    if thermal is None:
        return False, None
    if not isinstance(thermal, Mapping):
        raise ValueError("configuration field 'thermal' must be a mapping.")

    enabled = thermal.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("configuration field 'thermal.enabled' must be true or false.")
    if not enabled:
        return False, None

    thermal_params = ThermalFeedbackParameters(
        fuel_heat_capacity=_require_positive_float(thermal, "Cf"),
        moderator_heat_capacity=_require_positive_float(thermal, "Cm"),
        fuel_moderator_heat_transfer=_require_positive_float(thermal, "Hfm"),
        moderator_coolant_heat_transfer=_require_positive_float(thermal, "Hmc"),
        coolant_temperature=_require_positive_float(thermal, "Tc"),
        initial_fuel_temperature=_require_positive_float(thermal, "Tf0"),
        initial_moderator_temperature=_require_positive_float(thermal, "Tm0"),
        fuel_temperature_coefficient=_require_float(thermal, "alpha_f"),
        moderator_temperature_coefficient=_require_float(thermal, "alpha_m"),
        nominal_power=_require_positive_float(thermal, "power_scale"),
    )
    return True, thermal_params


def load_reactor_config(path: str | Path) -> ReactorConfig:
    """Load and validate a reactor kinetics YAML configuration file.

    The loader validates the required top-level fields, converts delayed-neutron
    group data to NumPy arrays, and maps the compact YAML thermal field names to
    the package's thermal-feedback dataclass.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, Mapping):
        raise ValueError("configuration file must contain a top-level mapping.")

    kinetics = _load_kinetics(loaded)
    thermal_enabled, thermal = _load_thermal(loaded)
    return ReactorConfig(
        name=_require_text(loaded, "name"),
        description=_require_text(loaded, "description"),
        kinetics=kinetics,
        thermal_enabled=thermal_enabled,
        thermal=thermal,
    )
