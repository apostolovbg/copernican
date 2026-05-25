"""Build a deterministic engine plugin for the synthetic datasets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from copernican_lib.engine_adapter import build_engine_plugin

_DATA_DIR = Path(__file__).parent
_MODEL_PATH = _DATA_DIR / "model.yml"


def _comoving_distance(
    z: np.ndarray, h0: float, omega_m0: float
) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    return 2997.92458 * z_arr / h0


def _mu_model(z: np.ndarray, h0: float, omega_m0: float) -> np.ndarray:
    dist = _luminosity_distance(z, h0, omega_m0)
    return 5.0 * np.log10(dist) + 25.0


def _luminosity_distance(
    z: np.ndarray, h0: float, omega_m0: float
) -> np.ndarray:
    return _comoving_distance(z, h0, omega_m0) * (1.0 + np.asarray(z, float))


def _angular_distance(z: np.ndarray, h0: float, omega_m0: float) -> np.ndarray:
    return _comoving_distance(z, h0, omega_m0) / (1.0 + np.asarray(z, float))


def _hz(z: np.ndarray, h0: float, omega_m0: float) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    return h0 * np.sqrt(omega_m0 * np.power(1.0 + z_arr, 3) + (1.0 - omega_m0))


def _dv(z: np.ndarray, h0: float, omega_m0: float) -> np.ndarray:
    dm = _comoving_distance(z, h0, omega_m0)
    hz = _hz(z, h0, omega_m0)
    return np.power(dm * dm * (2997.92458 / hz), 1.0 / 3.0)


def _rs(h0: float, omega_m0: float) -> float:
    return 147.0 / np.sqrt(1.0 + omega_m0)


def _cmb_spectrum(camb_params: dict, ells, spectra=("TT",)):
    ell_arr = np.asarray(list(ells), dtype=float)
    template = 1200.0 / (ell_arr + 3.0)
    if len(spectra) == 1:
        return template
    return {spec: template.copy() for spec in spectra}


def build_plugin():
    with _MODEL_PATH.open("r", encoding="utf-8") as handle:
        model_data = yaml.safe_load(handle)
    model_data["filename"] = os.fspath(_MODEL_PATH)
    functions = {
        "distance_modulus_model": _mu_model,
        "get_comoving_distance_Mpc": _comoving_distance,
        "get_luminosity_distance_Mpc": _luminosity_distance,
        "get_angular_diameter_distance_Mpc": _angular_distance,
        "get_Hz_per_Mpc": _hz,
        "get_DV_Mpc": _dv,
        "get_sound_horizon_rs_Mpc": _rs,
        "compute_cmb_spectrum": _cmb_spectrum,
        "compute_cmb_spectrum_from_dict": _cmb_spectrum,
    }
    return build_engine_plugin(model_data, functions)


__all__ = ["build_plugin", "_MODEL_PATH"]
