"""Build a deterministic engine plugin for the synthetic datasets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as numpy_module
import yaml

from copernican.lib.engine_adapter import build_engine_plugin
from tests.project import filesystem_helpers

_DATA_DIR = Path(__file__).parent
_MODEL_PATH = _DATA_DIR / "model.yml"


def _comoving_distance(
    redshift_values: numpy_module.ndarray,
    hubble_constant: float,
    matter_density_0: float,
) -> numpy_module.ndarray:
    redshift_array = numpy_module.asarray(redshift_values, dtype=float)
    return 2997.92458 * redshift_array / hubble_constant


def _mu_model(
    redshift_values: numpy_module.ndarray,
    hubble_constant: float,
    matter_density_0: float,
) -> numpy_module.ndarray:
    luminosity_distance = _luminosity_distance(
        redshift_values,
        hubble_constant,
        matter_density_0,
    )
    return 5.0 * numpy_module.log10(luminosity_distance) + 25.0


def _luminosity_distance(
    redshift_values: numpy_module.ndarray,
    hubble_constant: float,
    matter_density_0: float,
) -> numpy_module.ndarray:
    return _comoving_distance(
        redshift_values,
        hubble_constant,
        matter_density_0,
    ) * (1.0 + numpy_module.asarray(redshift_values, float))


def _angular_distance(
    redshift_values: numpy_module.ndarray,
    hubble_constant: float,
    matter_density_0: float,
) -> numpy_module.ndarray:
    return _comoving_distance(
        redshift_values,
        hubble_constant,
        matter_density_0,
    ) / (1.0 + numpy_module.asarray(redshift_values, float))


def _hubble_rate(
    redshift_values: numpy_module.ndarray,
    hubble_constant: float,
    matter_density_0: float,
) -> numpy_module.ndarray:
    redshift_array = numpy_module.asarray(redshift_values, dtype=float)
    return hubble_constant * numpy_module.sqrt(
        matter_density_0 * numpy_module.power(1.0 + redshift_array, 3)
        + (1.0 - matter_density_0)
    )


def _volume_distance(
    redshift_values: numpy_module.ndarray,
    hubble_constant: float,
    matter_density_0: float,
) -> numpy_module.ndarray:
    comoving_distance = _comoving_distance(
        redshift_values,
        hubble_constant,
        matter_density_0,
    )
    hubble_rate = _hubble_rate(
        redshift_values,
        hubble_constant,
        matter_density_0,
    )
    return numpy_module.power(
        comoving_distance * comoving_distance * (2997.92458 / hubble_rate),
        1.0 / 3.0,
    )


def _sound_horizon(hubble_constant: float, matter_density_0: float) -> float:
    return 147.0 / numpy_module.sqrt(1.0 + matter_density_0)


def build_plugin():
    """Build the synthetic plugin through the native CMB adapter."""

    model_data = yaml.safe_load(filesystem_helpers.read_text(_MODEL_PATH))
    model_data["filename"] = os.fspath(_MODEL_PATH)

    functions = {
        "distance_modulus_model": _mu_model,
        "get_comoving_distance_Mpc": _comoving_distance,
        "get_luminosity_distance_Mpc": _luminosity_distance,
        "get_angular_diameter_distance_Mpc": _angular_distance,
        "get_Hz_per_Mpc": _hubble_rate,
        "get_DV_Mpc": _volume_distance,
        "get_sound_horizon_rs_Mpc": _sound_horizon,
    }
    return build_engine_plugin(model_data, functions)


__all__ = ["build_plugin", "_MODEL_PATH"]
