r"""Cosmic Microwave Background likelihood helper.

Provides cache-aware CAMB interfaces shared by the CMB likelihood and the BAO
background evaluator. The helpers consume structured CAMB contracts so scalar
parameters, declared grids, evaluated values and ordered backend calls stay
aligned across the spectrum and background paths. The spectra returned here
are expressed as :math:`D_\ell` so downstream tests comparing against
published Planck-lite tables use consistent conventions. Non-standard
perturbation contracts route through the generic scalar CMB engine in this
module instead of falling back to CAMB.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import math
import re
from dataclasses import astuple, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import camb
import numpy
import pandas
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import spherical_jn

from ..engine_adapter import (
    _SUPPORTED_CMB_BACKEND,
    _evaluate_safe_expression,
    _freeze_for_cache,
)
from ..model_coder import validate_native_perturbation_execution
from ._protocol import LikelihoodProtocol, LikelihoodState

_C_LIGHT_KM_S = 299_792.458
_LMAX_PADDING = 300
_LENS_POTENTIAL_ACCURACY = 0
_CACHE_PRECISION = 15
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")


def _normalise_value(entry_value: Any) -> Any:
    """Return a cache-friendly representation of ``entry_value``."""

    if isinstance(entry_value, (int, float, numpy.integer, numpy.floating)):
        return float(entry_value)
    return str(entry_value)


def _normalise_items(
    param_dict: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """Convert ``param_dict`` into a deterministic tuple of items."""

    normalised: list[tuple[str, Any]] = []
    for key in sorted(param_dict):
        normalised.append((str(key), _normalise_value(param_dict[key])))
    return tuple(normalised)


def _restore_dict(items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Rehydrate a mapping created by :func:`_normalise_items`."""

    restored: dict[str, Any] = {}
    for key, restored_value in items:
        restored[str(key)] = restored_value
    return restored


def _coerce_numeric_scalar(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite scalar float."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 0:
        raise ValueError(f"{name} must evaluate to a scalar")
    scalar = float(array_value)
    if not numpy.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _coerce_numeric_array(value: Any, *, name: str) -> numpy.ndarray:
    """Return ``value`` as a finite one-dimensional array."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 1:
        raise ValueError(f"{name} must evaluate to a one-dimensional array")
    if array_value.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not numpy.all(numpy.isfinite(array_value)):
        raise ValueError(f"{name} must contain only finite values")
    return array_value


def _normalise_camb_contract(
    contract_or_params: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a structured CAMB contract from legacy or new inputs."""

    keys = {str(key) for key in contract_or_params.keys()}
    if {"backend", "param_map", "grids", "values", "calls"}.issubset(keys):
        return contract_or_params
    if keys.intersection({"backend", "param_map", "grids", "values", "calls"}):
        return contract_or_params
    return {
        "backend": "camb",
        "param_map": dict(contract_or_params),
        "grids": {},
        "values": {},
        "calls": [],
    }


def _is_structured_camb_background_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract_or_params`` uses the CAMB adapter."""

    keys = {str(key) for key in contract_or_params.keys()}
    required = {"backend", "calls", "grids", "param_map", "values"}
    return required.issubset(keys)


def _is_structured_camb_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract_or_params`` includes perturbations."""

    keys = {str(key) for key in contract_or_params.keys()}
    required = {
        "backend",
        "calls",
        "grids",
        "param_map",
        "perturbations",
        "values",
    }
    return required.issubset(keys)


def _combine_camb_contracts(
    background_contract: Mapping[str, Any],
    perturbation_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a structured CAMB contract with perturbation metadata."""

    combined = dict(background_contract)
    if perturbation_contract:
        combined["perturbations"] = dict(perturbation_contract)
    return combined


def _validate_camb_perturbation_execution(
    contract: Mapping[str, Any],
) -> None:
    """Reject unsupported perturbation declarations before CAMB runs."""

    perturbations = contract.get("perturbations")
    if perturbations is None:
        raise ValueError("Structured CAMB contract is missing perturbations")
    if not isinstance(perturbations, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")

    model_name = contract.get("model_name", "unknown model")
    backend = str(contract.get("backend", "camb"))

    standard = perturbations.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")
    if standard:
        return

    backend_mapping = perturbations.get("backend_mapping", {})
    backend_entry = {}
    if isinstance(backend_mapping, Mapping):
        backend_entry = backend_mapping.get(backend, {}) or {}

    implemented = None
    if isinstance(backend_entry, Mapping):
        implemented = backend_entry.get("implemented")

    validate_native_perturbation_execution(
        model_name=str(model_name),
        backend=backend,
        standard=standard,
        implemented=implemented if isinstance(implemented, bool) else None,
    )


def _compile_declared_perturbation_contract(
    contract: Mapping[str, Any],
):
    """Return the compiled perturbation contract for generic execution."""

    from ..perturbation_contract import compile_perturbation_contract

    model_parameters = contract.get("model_parameters", {})
    if isinstance(model_parameters, Mapping) and model_parameters:
        parameter_names = tuple(str(key) for key in model_parameters)
    else:
        parameter_names = tuple(
            str(key) for key in (contract.get("param_map", {}) or {})
        )

    latex_names = tuple("" for _ in parameter_names)
    background_reference_names = {
        str(key) for key in (contract.get("param_map", {}) or {})
    }
    grid_defs = contract.get("grids", {}) or {}
    for grid_def in grid_defs.values():
        if not isinstance(grid_def, Mapping):
            continue
        symbol = grid_def.get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            background_reference_names.add(symbol.strip())
    background_reference_names.update(
        str(key) for key in (contract.get("values", {}) or {})
    )
    background_reference_names.update(parameter_names)

    perturbations = contract.get("perturbations", {})
    if isinstance(perturbations, Mapping):
        perturbations = {
            key: value
            for key, value in perturbations.items()
            if key not in {"backend", "model_name"}
        }
    return compile_perturbation_contract(
        perturbations,
        model_name=str(contract.get("model_name", "unknown model")),
        backend=str(contract.get("backend", _SUPPORTED_CMB_BACKEND)),
        parameter_names=parameter_names,
        latex_names=latex_names,
        background_reference_names=tuple(sorted(background_reference_names)),
    )


def _build_generic_background_payload_from_plugin(
    plugin: Any,
    contract: Mapping[str, Any],
    redshifts: Sequence[float],
) -> dict[str, numpy.ndarray]:
    """Return background observables using the model's own callables."""

    z_arr = numpy.asarray(redshifts, dtype=float)
    if z_arr.ndim != 1 or z_arr.size == 0:
        raise ValueError("redshifts must be a one-dimensional array")

    parameter_names = tuple(getattr(plugin, "PARAMETER_NAMES", ()) or ())
    model_parameters = contract.get("model_parameters", {}) or {}
    param_map = contract.get("param_map", {}) or {}
    parameter_values: list[float] = []
    for parameter_name in parameter_names:
        if (
            isinstance(model_parameters, Mapping)
            and parameter_name in model_parameters
        ):
            parameter_values.append(
                _coerce_numeric_scalar(
                    model_parameters[parameter_name],
                    name=str(parameter_name),
                )
            )
            continue
        if isinstance(param_map, Mapping) and parameter_name in param_map:
            parameter_values.append(
                _coerce_numeric_scalar(
                    param_map[parameter_name],
                    name=str(parameter_name),
                )
            )
            continue
        raise ValueError(
            "Generic declarative perturbation execution requires model "
            f"parameter '{parameter_name}'"
        )

    parameter_tuple = tuple(parameter_values)

    def _coerce_series(
        value: Any,
        *,
        name: str,
    ) -> numpy.ndarray:
        """Return a finite series with the same shape as the redshift grid."""

        array_value = numpy.asarray(value, dtype=float)
        if array_value.ndim == 0:
            return numpy.full(z_arr.shape, float(array_value), dtype=float)
        if array_value.shape != z_arr.shape:
            if array_value.size == 1:
                return numpy.full(
                    z_arr.shape,
                    float(array_value.reshape(())),
                    dtype=float,
                )
            raise ValueError(
                f"{name} must evaluate to an array with shape {z_arr.shape}"
            )
        if not numpy.all(numpy.isfinite(array_value)):
            raise ValueError(f"{name} must contain only finite values")
        return array_value

    hz_fn = getattr(plugin, "get_Hz_per_Mpc", None)
    if not callable(hz_fn):
        raise ValueError(
            "Generic declarative perturbation execution requires "
            "get_Hz_per_Mpc"
        )
    hz_values = _coerce_series(hz_fn(z_arr, *parameter_tuple), name="Hz")

    dm_fn = getattr(plugin, "get_comoving_distance_Mpc", None)
    if callable(dm_fn):
        dm_values = _coerce_series(
            dm_fn(z_arr, *parameter_tuple),
            name="DM",
        )
    else:
        sorted_order = numpy.argsort(z_arr)
        sorted_z = z_arr[sorted_order]
        sorted_hz = hz_values[sorted_order]
        with numpy.errstate(divide="ignore", invalid="ignore"):
            integrand = numpy.where(
                numpy.abs(sorted_hz) > 1e-12,
                _C_LIGHT_KM_S / sorted_hz,
                numpy.nan,
            )
        integrated = cumulative_trapezoid(
            integrand,
            sorted_z,
            initial=0.0,
        )
        dm_values = numpy.interp(z_arr, sorted_z, integrated)
        dm_values = _coerce_series(dm_values, name="DM")

    dh_values = numpy.where(
        numpy.abs(hz_values) > 1e-12,
        _C_LIGHT_KM_S / hz_values,
        numpy.nan,
    )

    da_fn = getattr(plugin, "get_angular_diameter_distance_Mpc", None)
    if callable(da_fn):
        da_values = _coerce_series(
            da_fn(z_arr, *parameter_tuple),
            name="DA",
        )
    else:
        da_values = numpy.divide(dm_values, 1.0 + z_arr, dtype=float)

    dv_fn = getattr(plugin, "get_DV_Mpc", None)
    if callable(dv_fn):
        dv_values = _coerce_series(
            dv_fn(z_arr, *parameter_tuple),
            name="DV",
        )
    else:
        with numpy.errstate(divide="ignore", invalid="ignore"):
            dv_term = dm_values * dm_values * z_arr * dh_values
            dv_values = numpy.where(
                dv_term >= 0.0, numpy.power(dv_term, 1.0 / 3.0), numpy.nan
            )

    rs_fn = getattr(plugin, "get_sound_horizon_rs_Mpc", None)
    if callable(rs_fn):
        rs_drag = _coerce_numeric_scalar(
            rs_fn(*parameter_tuple),
            name="rs_drag",
        )
    else:
        rs_drag = 1.0

    return {
        "rs_drag": numpy.asarray([rs_drag], dtype=float),
        "DM": dm_values,
        "DH": dh_values,
        "DA": da_values,
        "DV": dv_values,
        "Hz": hz_values,
        "z": z_arr.copy(),
    }


def _build_generic_background_payload_from_contract(
    contract: Mapping[str, Any],
    redshifts: Sequence[float],
) -> dict[str, numpy.ndarray]:
    """Return a self-contained background payload from contract values."""

    z_arr = numpy.asarray(redshifts, dtype=float)
    if z_arr.ndim != 1 or z_arr.size == 0:
        raise ValueError("redshifts must be a one-dimensional array")

    model_parameters = contract.get("model_parameters", {}) or {}
    param_map = contract.get("param_map", {}) or {}

    def _lookup(*names: str, default: float) -> float:
        """Return the first declared scalar value among ``names``."""

        for source in (model_parameters, param_map):
            if not isinstance(source, Mapping):
                continue
            for name in names:
                if name not in source:
                    continue
                try:
                    return _coerce_numeric_scalar(source[name], name=name)
                except ValueError:
                    continue
        return float(default)

    hubble_constant_value = max(
        abs(_lookup("H0", "hubble_constant", default=70.0)),
        1.0,
    )
    hubble_constant_squared = max(
        (hubble_constant_value / 100.0) ** 2,
        1e-6,
    )

    omega_m = None
    for candidate in ("Omega_m0", "Omega_m", "omega_m"):
        if candidate in model_parameters:
            omega_m = _coerce_numeric_scalar(
                model_parameters[candidate],
                name=candidate,
            )
            break
        if candidate in param_map:
            omega_m = _coerce_numeric_scalar(
                param_map[candidate], name=candidate
            )
            break
    if omega_m is None:
        omega_b_h2 = _lookup("ombh2", default=0.0)
        omega_c_h2 = _lookup("omch2", default=0.0)
        if omega_b_h2 or omega_c_h2:
            omega_m = (omega_b_h2 + omega_c_h2) / hubble_constant_squared
        else:
            omega_m = 0.3

    omega_b = None
    for candidate in ("Omega_b", "omega_b"):
        if candidate in model_parameters:
            omega_b = _coerce_numeric_scalar(
                model_parameters[candidate],
                name=candidate,
            )
            break
        if candidate in param_map:
            omega_b = _coerce_numeric_scalar(
                param_map[candidate], name=candidate
            )
            break
    if omega_b is None:
        omega_b_h2 = _lookup("ombh2", default=0.0)
        if omega_b_h2:
            omega_b = omega_b_h2 / hubble_constant_squared
        else:
            omega_b = 0.05

    omega_gamma = None
    for candidate in ("Omega_gamma", "omega_gamma"):
        if candidate in model_parameters:
            omega_gamma = _coerce_numeric_scalar(
                model_parameters[candidate],
                name=candidate,
            )
            break
        if candidate in param_map:
            omega_gamma = _coerce_numeric_scalar(
                param_map[candidate],
                name=candidate,
            )
            break

    omega_r = None
    for candidate in ("Omega_r0", "Omega_r", "omega_r", "omr"):
        if candidate in model_parameters:
            omega_r = _coerce_numeric_scalar(
                model_parameters[candidate],
                name=candidate,
            )
            break
        if candidate in param_map:
            omega_r = _coerce_numeric_scalar(
                param_map[candidate], name=candidate
            )
            break
    if omega_r is None:
        neff = _lookup("Neff", default=3.046)
        if omega_gamma is not None:
            omega_r = max(omega_gamma, 0.0) * (1.0 + 0.2271 * max(neff, 0.0))
        else:
            omega_r = 8.5e-5 / hubble_constant_squared

    omega_k = None
    for candidate in ("Omega_k0", "Omega_k", "omega_k", "omk"):
        if candidate in model_parameters:
            omega_k = _coerce_numeric_scalar(
                model_parameters[candidate],
                name=candidate,
            )
            break
        if candidate in param_map:
            omega_k = _coerce_numeric_scalar(
                param_map[candidate], name=candidate
            )
            break
    if omega_k is None:
        omega_k = 0.0

    omega_de = None
    for candidate in ("Omega_Lambda", "Omega_de", "omega_de"):
        if candidate in model_parameters:
            omega_de = _coerce_numeric_scalar(
                model_parameters[candidate],
                name=candidate,
            )
            break
        if candidate in param_map:
            omega_de = _coerce_numeric_scalar(
                param_map[candidate], name=candidate
            )
            break
    if omega_de is None:
        omega_de = max(0.0, 1.0 - omega_m - omega_r - omega_k)

    background_hz = hubble_constant_value * numpy.sqrt(
        numpy.maximum(
            omega_m * numpy.power(1.0 + z_arr, 3.0)
            + omega_r * numpy.power(1.0 + z_arr, 4.0)
            + omega_k * numpy.power(1.0 + z_arr, 2.0)
            + omega_de,
            1e-12,
        )
    )
    background_hz = numpy.maximum(background_hz, 1e-12)

    sorted_order = numpy.argsort(z_arr)
    sorted_z = z_arr[sorted_order]
    sorted_hz = background_hz[sorted_order]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        integrand = numpy.where(
            numpy.abs(sorted_hz) > 1e-12,
            _C_LIGHT_KM_S / sorted_hz,
            numpy.nan,
        )
    dm_sorted = cumulative_trapezoid(integrand, sorted_z, initial=0.0)
    dm_values = numpy.interp(z_arr, sorted_z, dm_sorted)
    dh_values = numpy.where(
        numpy.abs(background_hz) > 1e-12,
        _C_LIGHT_KM_S / background_hz,
        numpy.nan,
    )
    da_values = numpy.divide(dm_values, 1.0 + z_arr, dtype=float)
    dv_values = numpy.full_like(z_arr, numpy.nan, dtype=float)
    dv_term = dm_values * dm_values * z_arr * dh_values
    dv_mask = numpy.isfinite(dv_term) & (dv_term >= 0.0)
    dv_values[dv_mask] = numpy.power(dv_term[dv_mask], 1.0 / 3.0)
    dv_values[z_arr == 0.0] = 0.0
    rs_drag = 147.0 / numpy.sqrt(
        1.0 + max(omega_b, 0.0) / max(omega_gamma or 1e-6, 1e-6)
    )

    return {
        "rs_drag": numpy.asarray([rs_drag], dtype=float),
        "DM": dm_values,
        "DH": dh_values,
        "DA": da_values,
        "DV": dv_values,
        "Hz": background_hz,
        "z": z_arr.copy(),
    }


@dataclass(slots=True)
class _CustomCMBNumerics:
    """Numerical settings used by the generic scalar CMB engine."""

    ell_min: int = 2
    ell_max: int = 2500
    k_min: float = 1.0e-5
    k_max: float = 0.4
    k_sample_count: int = 64
    eta_sample_count: int = 1024
    photon_hierarchy_l_max: int = 8
    neutrino_hierarchy_l_max: int = 8
    ode_rtol: float = 1.0e-6
    ode_atol: float = 1.0e-9
    tight_coupling_ratio: float = 50.0
    a_min: float = 1.0e-8
    source_grid_multiplier: int = 2
    initial_redshift: float = 1.0e5


@dataclass(slots=True)
class _CustomCMBPhysicalParameters:
    """Resolved physical background inputs for the generic CMB solver."""

    H0_km_s_Mpc: float
    hubble_ratio: float
    H0_over_c_Mpc_inv: float
    ombh2: float
    omch2: float
    Omega_b0: float
    Omega_c0: float
    Omega_m0_background: float
    Omega_gamma0: float
    Omega_nu0: float
    Omega_r0: float
    Omega_k0: float
    Omega_de0: float
    dark_energy_eos0: float
    dark_energy_eos1: float
    YHe: float
    Neff: float
    primordial_amplitude: float
    primordial_spectral_index: float
    z_rec: float
    tau_reio: float
    Tcmb_K: float
    n_b0_m3: float
    n_H0_m3: float
    rho_b0_kg_m3: float
    has_cdm: bool
    has_dark_energy: bool


@dataclass(slots=True)
class _CustomCMBBackgroundData:
    """Background and recombination tables for the generic CMB solver."""

    a_grid: numpy.ndarray
    z_grid: numpy.ndarray
    eta_grid: numpy.ndarray
    eta0: float
    chi_grid: numpy.ndarray
    da_grid: numpy.ndarray
    H_grid: numpy.ndarray
    Hconf_grid: numpy.ndarray
    tau_grid: numpy.ndarray
    tau_dot_grid: numpy.ndarray
    visibility_grid: numpy.ndarray
    x_e_grid: numpy.ndarray
    n_e_grid: numpy.ndarray
    n_H_grid: numpy.ndarray
    sound_speed_grid: numpy.ndarray
    sound_horizon_mpc: float
    reionization_z: float
    reionization_tau: float
    eta_rec: float
    a_rec: float
    z_rec: float
    eta_of_a: PchipInterpolator
    a_of_eta: PchipInterpolator
    z_of_eta: PchipInterpolator
    H_of_eta: PchipInterpolator
    chi_of_eta: PchipInterpolator
    da_of_eta: PchipInterpolator
    tau_of_eta: PchipInterpolator
    tau_dot_of_eta: PchipInterpolator
    visibility_of_eta: PchipInterpolator
    x_e_of_eta: PchipInterpolator
    sound_speed_of_eta: PchipInterpolator

    def sample(
        self, eta_values: numpy.ndarray | float
    ) -> dict[str, numpy.ndarray]:
        """Return the background quantities interpolated at ``eta_values``."""

        eta_arr = numpy.asarray(eta_values, dtype=float)
        return {
            "a": numpy.asarray(self.a_of_eta(eta_arr), dtype=float),
            "z": numpy.asarray(self.z_of_eta(eta_arr), dtype=float),
            "H": numpy.asarray(self.H_of_eta(eta_arr), dtype=float),
            "chi": numpy.asarray(self.chi_of_eta(eta_arr), dtype=float),
            "angular_diameter_distance": numpy.asarray(
                self.da_of_eta(eta_arr), dtype=float
            ),
            "tau": numpy.asarray(self.tau_of_eta(eta_arr), dtype=float),
            "tau_dot": numpy.asarray(
                self.tau_dot_of_eta(eta_arr), dtype=float
            ),
            "visibility": numpy.asarray(
                self.visibility_of_eta(eta_arr), dtype=float
            ),
            "x_e": numpy.asarray(self.x_e_of_eta(eta_arr), dtype=float),
            "sound_speed": numpy.asarray(
                self.sound_speed_of_eta(eta_arr), dtype=float
            ),
        }


@dataclass(slots=True)
class CustomCMBSpectrumData:
    """Internal transfer-function and spectrum payload for CMB outputs."""

    ell_grid: numpy.ndarray
    k_grid: numpy.ndarray
    Delta_l_T: numpy.ndarray
    Delta_l_E: numpy.ndarray
    C_l_TT: numpy.ndarray
    C_l_TE: numpy.ndarray
    C_l_EE: numpy.ndarray


_CUSTOM_CMB_SECTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "photon_temperature_monopole": (
        "photon_temperature_monopole",
        "theta_gamma0",
        "theta0",
        "theta_photon0",
    ),
    "photon_temperature_dipole": (
        "photon_temperature_dipole",
        "theta_gamma1",
        "theta1",
        "theta_photon1",
    ),
    "photon_temperature_quadrupole": (
        "photon_temperature_quadrupole",
        "theta_gamma2",
        "theta2",
        "theta_photon2",
    ),
    "photon_polarization_quadrupole": (
        "photon_polarization_quadrupole",
        "e_gamma2",
        "e2",
        "ee_quadrupole",
    ),
    "baryon_density_contrast": (
        "baryon_density_contrast",
        "delta_b",
    ),
    "baryon_velocity_divergence": (
        "baryon_velocity_divergence",
        "theta_b",
    ),
    "cdm_density_contrast": (
        "cdm_density_contrast",
        "delta_c",
    ),
    "cdm_velocity_divergence": (
        "cdm_velocity_divergence",
        "theta_c",
    ),
    "massless_neutrino_density_contrast": (
        "massless_neutrino_density_contrast",
        "delta_nu",
    ),
    "massless_neutrino_velocity_divergence": (
        "massless_neutrino_velocity_divergence",
        "theta_nu",
    ),
    "massless_neutrino_anisotropic_stress": (
        "massless_neutrino_anisotropic_stress",
        "sigma_nu",
    ),
    "metric_potential_phi": (
        "metric_potential_phi",
        "Phi",
    ),
    "metric_potential_psi": (
        "metric_potential_psi",
        "Psi",
    ),
}
_CUSTOM_CMB_SOURCE_CHANNELS: tuple[str, ...] = (
    "temperature_monopole",
    "temperature_doppler",
    "temperature_isw",
    "polarization",
    "temperature_additive",
)
_CUSTOM_CMB_BACKGROUND_INPUTS: dict[tuple[Any, ...], tuple[Any, ...]] = {}
_CUSTOM_CMB_BACKGROUND_RESULTS: dict[
    tuple[Any, ...], "_CustomCMBBackgroundData"
] = {}
_CUSTOM_CMB_SPECTRUM_INPUTS: dict[tuple[Any, ...], Mapping[str, Any]] = {}
_CUSTOM_CMB_SPECTRUM_RESULTS: dict[
    tuple[Any, ...], "CustomCMBSpectrumData"
] = {}
_CUSTOM_CMB_PROVIDER_REGISTRY: dict[int, Any] = {}
_CUSTOM_CMB_BESSEL_INPUTS: dict[str, numpy.ndarray] = {}


@lru_cache(maxsize=64)
def _get_cached_custom_cmb_background(
    cache_key: tuple[Any, ...],
) -> "_CustomCMBBackgroundData":
    """Return a cached custom background payload."""

    return _CUSTOM_CMB_BACKGROUND_RESULTS[cache_key]


@lru_cache(maxsize=64)
def _get_cached_custom_cmb_spectrum_data(
    cache_key: tuple[Any, ...],
) -> "CustomCMBSpectrumData":
    """Return a cached custom spectrum payload."""

    return _CUSTOM_CMB_SPECTRUM_RESULTS[cache_key]


@lru_cache(maxsize=4096)
def _get_cached_spherical_bessel_values(
    ell: int,
    x_signature: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return cached spherical Bessel values for one ell and x-grid."""

    x_values = _CUSTOM_CMB_BESSEL_INPUTS[x_signature]
    return (
        spherical_jn(int(ell), x_values),
        spherical_jn(int(ell), x_values, derivative=True),
    )


def _custom_cmb_provider_key(background_provider: Any | None) -> int:
    """Return a stable cache key for a custom-CMB background provider."""

    if background_provider is None:
        return 0
    return object.__hash__(background_provider)


def _custom_cmb_background_cache_key(
    physical_params: _CustomCMBPhysicalParameters,
    numerics: _CustomCMBNumerics,
    background_provider: Any | None,
) -> tuple[Any, ...]:
    """Return a cache key for the custom CMB background tables."""

    return (
        astuple(physical_params),
        astuple(numerics),
        _custom_cmb_provider_key(background_provider),
    )


def _custom_cmb_spectrum_cache_key(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    background_provider: Any | None,
) -> tuple[Any, ...]:
    """Return a cache key for the custom spectrum transfer data."""

    ell_key = tuple(int(ell) for ell in numpy.asarray(list(ells), dtype=int))
    return (
        _freeze_for_cache(contract_or_params),
        ell_key,
        _custom_cmb_provider_key(background_provider),
    )


def _expression_symbol_name(expression: str) -> str | None:
    """Return the bare symbol name in ``expression`` when one is present."""

    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    body = node.body
    if isinstance(body, ast.Name):
        return body.id
    return None


def _extract_contract_scalar(
    contract: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: float | None = None,
) -> float | None:
    """Return the first finite scalar matching ``names`` from ``contract``."""

    for source in (
        contract.get("param_map", {}) or {},
        contract.get("model_parameters", {}) or {},
    ):
        if not isinstance(source, Mapping):
            continue
        for name in names:
            if name not in source:
                continue
            try:
                return _coerce_numeric_scalar(source[name], name=name)
            except ValueError:
                continue
    if default is None:
        return None
    return float(default)


def _resolve_custom_cmb_numerics(
    contract: Mapping[str, Any],
) -> _CustomCMBNumerics:
    """Return numerical settings for the generic custom CMB solver."""

    raw = contract.get("numerical", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("cmb.numerical must be a mapping when declared")
    defaults = _CustomCMBNumerics()

    def _read_int(name: str, default: int) -> int:
        """Return a positive integer from ``raw`` or ``default``."""

        value = raw.get(name, default)
        numeric = int(_coerce_numeric_scalar(value, name=name))
        if numeric < 1:
            raise ValueError(f"cmb.numerical.{name} must be positive")
        return numeric

    def _read_float(name: str, default: float) -> float:
        """Return a positive float from ``raw`` or ``default``."""

        value = raw.get(name, default)
        numeric = float(_coerce_numeric_scalar(value, name=name))
        if numeric <= 0.0:
            raise ValueError(f"cmb.numerical.{name} must be positive")
        return numeric

    ell_min = max(2, _read_int("ell_min", defaults.ell_min))
    ell_max = max(ell_min, _read_int("ell_max", defaults.ell_max))
    k_min = _read_float("k_min", defaults.k_min)
    k_max = _read_float("k_max", defaults.k_max)
    k_sample_count = max(
        16, _read_int("k_sample_count", defaults.k_sample_count)
    )
    eta_sample_count = max(
        128,
        _read_int("eta_sample_count", defaults.eta_sample_count),
    )
    photon_hierarchy_l_max = max(
        2,
        _read_int(
            "photon_hierarchy_l_max",
            defaults.photon_hierarchy_l_max,
        ),
    )
    neutrino_hierarchy_l_max = max(
        2,
        _read_int(
            "neutrino_hierarchy_l_max",
            defaults.neutrino_hierarchy_l_max,
        ),
    )
    ode_rtol = _read_float("ode_rtol", defaults.ode_rtol)
    ode_atol = _read_float("ode_atol", defaults.ode_atol)
    tight_coupling_ratio = _read_float(
        "tight_coupling_ratio",
        defaults.tight_coupling_ratio,
    )
    a_min = _read_float("a_min", defaults.a_min)
    source_grid_multiplier = max(
        1,
        _read_int(
            "source_grid_multiplier",
            defaults.source_grid_multiplier,
        ),
    )
    initial_redshift = _read_float(
        "initial_redshift",
        defaults.initial_redshift,
    )
    return _CustomCMBNumerics(
        ell_min=ell_min,
        ell_max=ell_max,
        k_min=k_min,
        k_max=k_max,
        k_sample_count=k_sample_count,
        eta_sample_count=eta_sample_count,
        photon_hierarchy_l_max=photon_hierarchy_l_max,
        neutrino_hierarchy_l_max=neutrino_hierarchy_l_max,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        tight_coupling_ratio=tight_coupling_ratio,
        a_min=a_min,
        source_grid_multiplier=source_grid_multiplier,
        initial_redshift=initial_redshift,
    )


def _resolve_custom_cmb_physical_parameters(
    contract: Mapping[str, Any],
    background_provider: Any | None = None,
) -> _CustomCMBPhysicalParameters:
    """Return physical CMB parameters from the structured contract."""

    hubble_km_s_mpc = _extract_contract_scalar(
        contract,
        ("H0", "hubble_constant", "Hubble constant"),
        default=None,
    )
    if hubble_km_s_mpc is None and background_provider is not None:
        hz_fn = getattr(background_provider, "get_Hz_per_Mpc", None)
        if callable(hz_fn):
            hubble_km_s_mpc = _coerce_numeric_scalar(
                hz_fn(numpy.asarray([0.0])),
                name="H0",
            )
    if hubble_km_s_mpc is None:
        hubble_km_s_mpc = 67.4
    hubble_km_s_mpc = max(float(hubble_km_s_mpc), 1.0e-6)
    hubble_ratio = hubble_km_s_mpc / 100.0
    hubble_over_c = hubble_km_s_mpc / _C_LIGHT_KM_S

    ombh2 = _extract_contract_scalar(
        contract,
        ("ombh2", "Omega_b_h2", "omega_b_h2", "baryon_density_h2"),
        default=None,
    )
    Omega_b0 = _extract_contract_scalar(
        contract,
        ("Omega_b", "Omega_b0", "omega_b"),
        default=None,
    )
    if Omega_b0 is None and ombh2 is not None:
        Omega_b0 = ombh2 / (hubble_ratio * hubble_ratio)
    if Omega_b0 is None:
        Omega_b0 = 0.05
    if ombh2 is None:
        ombh2 = Omega_b0 * hubble_ratio * hubble_ratio

    explicit_cdm = any(
        key in (contract.get("param_map", {}) or {})
        for key in (
            "omch2",
            "Omega_c",
            "Omega_c0",
            "omega_c",
            "Omega_cdm",
            "Omega_cdm0",
            "omega_cdm",
            "cdm_density_h2",
        )
    )
    omch2 = _extract_contract_scalar(
        contract,
        ("omch2", "Omega_c_h2", "omega_c_h2", "cdm_density_h2"),
        default=None,
    )
    Omega_c0 = _extract_contract_scalar(
        contract,
        ("Omega_c", "Omega_c0", "omega_c", "Omega_cdm", "Omega_cdm0"),
        default=None,
    )
    if explicit_cdm:
        if Omega_c0 is None and omch2 is not None:
            Omega_c0 = omch2 / (hubble_ratio * hubble_ratio)
        if Omega_c0 is None:
            Omega_c0 = 0.0
        if omch2 is None:
            omch2 = Omega_c0 * hubble_ratio * hubble_ratio
    else:
        Omega_c0 = 0.0
        omch2 = 0.0

    Tcmb_K = _extract_contract_scalar(
        contract,
        ("Tcmb", "T_cmb", "Tcmb_K"),
        default=2.7255,
    )
    YHe = _extract_contract_scalar(
        contract,
        ("YHe", "Yp", "helium_fraction"),
        default=0.245,
    )
    Neff = _extract_contract_scalar(contract, ("Neff", "N_eff"), default=3.046)

    Omega_gamma0 = _extract_contract_scalar(
        contract,
        ("Omega_gamma", "Omega_gamma0", "omega_gamma", "omgamma"),
        default=None,
    )
    if Omega_gamma0 is None:
        omega_gamma_h2 = 2.469e-5 * (Tcmb_K / 2.7255) ** 4
        Omega_gamma0 = omega_gamma_h2 / (hubble_ratio * hubble_ratio)
    Omega_nu0 = max(0.0, Omega_gamma0) * 0.2271 * max(Neff, 0.0)
    Omega_r0 = Omega_gamma0 + Omega_nu0

    Omega_k0 = _extract_contract_scalar(
        contract,
        ("Omega_k", "Omega_k0", "omega_k", "omk"),
        default=0.0,
    )
    Omega_m0_background = _extract_contract_scalar(
        contract,
        ("Omega_m0", "Omega_m", "omega_m"),
        default=Omega_b0 + Omega_c0,
    )

    Omega_de0 = _extract_contract_scalar(
        contract,
        ("Omega_de", "Omega_de0", "Omega_Lambda", "Omega_lambda"),
        default=None,
    )
    has_dark_energy = Omega_de0 is not None
    if Omega_de0 is None:
        Omega_de0 = max(
            0.0,
            1.0 - Omega_m0_background - Omega_r0 - Omega_k0,
        )

    dark_energy_eos0 = _extract_contract_scalar(
        contract,
        ("w0", "w", "dark_energy_w0"),
        default=-1.0,
    )
    dark_energy_eos1 = _extract_contract_scalar(
        contract,
        ("wa",),
        default=0.0,
    )

    primordial_amplitude = _extract_contract_scalar(
        contract,
        ("As", "A_s"),
        default=2.1e-9,
    )
    primordial_spectral_index = _extract_contract_scalar(
        contract,
        ("ns", "n_s"),
        default=0.965,
    )
    z_rec = _extract_contract_scalar(contract, ("z_rec",), default=1089.92)
    if z_rec is None or z_rec <= 0.0:
        z_rec = 1089.92
    tau_reio = _extract_contract_scalar(
        contract,
        ("tau", "tau_reio", "reionization_tau"),
        default=0.054,
    )
    if tau_reio is None or tau_reio < 0.0:
        tau_reio = 0.054

    G_NEWTON = 6.674_30e-11
    MPC_M = 3.085_677_581_491_3673e22
    hubble_si = hubble_km_s_mpc * 1000.0 / MPC_M
    rho_crit0 = 3.0 * hubble_si * hubble_si / (8.0 * math.pi * G_NEWTON)
    rho_b0 = Omega_b0 * rho_crit0
    n_b0_m3 = rho_b0 / 1.672_621_923_69e-27
    n_H0_m3 = n_b0_m3 * max(0.0, 1.0 - YHe)
    return _CustomCMBPhysicalParameters(
        H0_km_s_Mpc=hubble_km_s_mpc,
        hubble_ratio=hubble_ratio,
        H0_over_c_Mpc_inv=hubble_over_c,
        ombh2=ombh2,
        omch2=omch2,
        Omega_b0=Omega_b0,
        Omega_c0=Omega_c0,
        Omega_m0_background=Omega_m0_background,
        Omega_gamma0=Omega_gamma0,
        Omega_nu0=Omega_nu0,
        Omega_r0=Omega_r0,
        Omega_k0=Omega_k0,
        Omega_de0=Omega_de0,
        dark_energy_eos0=dark_energy_eos0,
        dark_energy_eos1=dark_energy_eos1,
        YHe=YHe,
        Neff=Neff,
        primordial_amplitude=primordial_amplitude,
        primordial_spectral_index=primordial_spectral_index,
        z_rec=z_rec,
        tau_reio=tau_reio,
        Tcmb_K=Tcmb_K,
        n_b0_m3=n_b0_m3,
        n_H0_m3=n_H0_m3,
        rho_b0_kg_m3=rho_b0,
        has_cdm=explicit_cdm,
        has_dark_energy=has_dark_energy,
    )


def _classify_custom_physical_sector(name: str, kind: str) -> str:
    """Map a declared perturbation variable to a supported physical sector."""

    lowered_name = name.lower()
    lowered_kind = kind.lower().strip()

    aliases = {
        "photon_temperature_monopole": {
            "photon_temperature_monopole",
            "photon_monopole",
            "temperature_monopole",
            "theta_gamma0",
            "theta0",
            "theta_photon0",
        },
        "photon_temperature_dipole": {
            "photon_temperature_dipole",
            "photon_dipole",
            "temperature_dipole",
            "theta_gamma1",
            "theta1",
            "theta_photon1",
        },
        "photon_temperature_quadrupole": {
            "photon_temperature_quadrupole",
            "photon_quadrupole",
            "temperature_quadrupole",
            "theta_gamma2",
            "theta2",
            "theta_photon2",
        },
        "photon_polarization_quadrupole": {
            "photon_polarization_quadrupole",
            "polarization_quadrupole",
            "e_mode_quadrupole",
            "e2",
            "ee_quadrupole",
        },
        "baryon_density_contrast": {
            "baryon_density_contrast",
            "baryon_density",
            "delta_b",
        },
        "baryon_velocity_divergence": {
            "baryon_velocity_divergence",
            "baryon_velocity",
            "theta_b",
        },
        "cdm_density_contrast": {
            "cdm_density_contrast",
            "cdm_density",
            "delta_c",
        },
        "cdm_velocity_divergence": {
            "cdm_velocity_divergence",
            "cdm_velocity",
            "theta_c",
        },
        "massless_neutrino_density_contrast": {
            "massless_neutrino_density_contrast",
            "neutrino_density_contrast",
            "neutrino_density",
            "delta_nu",
        },
        "massless_neutrino_velocity_divergence": {
            "massless_neutrino_velocity_divergence",
            "neutrino_velocity_divergence",
            "neutrino_velocity",
            "theta_nu",
        },
        "massless_neutrino_anisotropic_stress": {
            "massless_neutrino_anisotropic_stress",
            "neutrino_anisotropic_stress",
            "neutrino_shear",
            "sigma_nu",
        },
        "metric_potential_phi": {"metric_potential_phi", "phi"},
        "metric_potential_psi": {"metric_potential_psi", "psi"},
    }
    for sector, sector_aliases in aliases.items():
        if lowered_kind in sector_aliases or lowered_name in sector_aliases:
            return sector
    raise ValueError(
        "Unsupported custom perturbation variable "
        f"'{name}' with kind '{kind}'."
    )


def _validate_custom_cmb_physical_contract(
    perturbation_data: Any,
    physical_params: _CustomCMBPhysicalParameters,
) -> dict[str, tuple[str, ...]]:
    """Validate that the custom contract declares supported CMB sectors."""

    gauge = str(getattr(perturbation_data, "gauge", "") or "")
    if gauge not in {"conformal_newtonian", "gauge_invariant"}:
        raise ValueError(
            "Custom scalar CMB evolution only supports conformal Newtonian "
            "gauge."
        )

    declared: dict[str, list[str]] = {
        "photon_temperature_monopole": [],
        "photon_temperature_dipole": [],
        "photon_temperature_quadrupole": [],
        "photon_polarization_quadrupole": [],
        "baryon_density_contrast": [],
        "baryon_velocity_divergence": [],
        "cdm_density_contrast": [],
        "cdm_velocity_divergence": [],
        "massless_neutrino_density_contrast": [],
        "massless_neutrino_velocity_divergence": [],
        "massless_neutrino_anisotropic_stress": [],
        "metric_potential_phi": [],
        "metric_potential_psi": [],
    }

    for collection_name in ("variables", "derived"):
        collection = getattr(perturbation_data, collection_name, {})
        for name, entry in collection.items():
            kind = str(getattr(entry, "kind", "") or "")
            if not kind:
                continue
            sector = _classify_custom_physical_sector(str(name), kind)
            declared.setdefault(sector, []).append(str(name))

    required = (
        "photon_temperature_monopole",
        "photon_temperature_dipole",
        "photon_temperature_quadrupole",
        "photon_polarization_quadrupole",
        "baryon_density_contrast",
        "baryon_velocity_divergence",
        "massless_neutrino_density_contrast",
        "massless_neutrino_velocity_divergence",
        "massless_neutrino_anisotropic_stress",
        "metric_potential_phi",
        "metric_potential_psi",
    )
    for sector in required:
        if declared.get(sector):
            continue
        readable = sector.replace("_", " ")
        raise ValueError(
            "Non-standard CMB contract is missing required "
            f"{readable} support."
        )

    if physical_params.has_cdm:
        for sector in ("cdm_density_contrast", "cdm_velocity_divergence"):
            if declared.get(sector):
                continue
            readable = sector.replace("_", " ")
            raise ValueError(
                "Non-standard CMB contract is missing required "
                f"{readable} support."
            )

    return {key: tuple(value) for key, value in declared.items()}


def _build_custom_cmb_background(
    contract: Mapping[str, Any],
    physical_params: _CustomCMBPhysicalParameters,
    numerics: _CustomCMBNumerics,
    *,
    background_provider: Any | None = None,
) -> _CustomCMBBackgroundData:
    """Return the interpolated background and recombination solution."""

    cache_key = (
        _freeze_for_cache(contract),
        astuple(physical_params),
        astuple(numerics),
        _custom_cmb_provider_key(background_provider),
    )
    cached_background = _CUSTOM_CMB_BACKGROUND_RESULTS.get(cache_key)
    if cached_background is not None:
        return _get_cached_custom_cmb_background(cache_key)
    if background_provider is not None:
        _CUSTOM_CMB_PROVIDER_REGISTRY[
            _custom_cmb_provider_key(background_provider)
        ] = background_provider

    MPC_M = 3.085_677_581_491_3673e22
    SIGMA_T_M2 = 6.652_458_7321e-29

    def _background_hubble_km_s_Mpc(z_values: numpy.ndarray) -> numpy.ndarray:
        """Return H(z) using the declared background or a physical fallback."""

        z_arr = numpy.asarray(z_values, dtype=float)
        if background_provider is not None:
            hz_fn = getattr(background_provider, "get_Hz_per_Mpc", None)
            if callable(hz_fn):
                hz_values = numpy.asarray(hz_fn(z_arr), dtype=float)
                if hz_values.shape == z_arr.shape and numpy.all(
                    numpy.isfinite(hz_values)
                ):
                    return numpy.maximum(hz_values, 1.0e-12)

        matter_background = max(
            0.0,
            physical_params.Omega_m0_background,
        )
        if matter_background <= 0.0:
            matter_background = (
                physical_params.Omega_b0 + physical_params.Omega_c0
            )
        dark_energy = physical_params.Omega_de0
        dark_energy_factor = numpy.ones_like(z_arr, dtype=float)
        if not math.isclose(
            physical_params.dark_energy_eos1,
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            exponent = 3.0 * (
                1.0
                + physical_params.dark_energy_eos0
                + physical_params.dark_energy_eos1
            )
            dark_energy_factor = numpy.power(
                1.0 + z_arr, exponent
            ) * numpy.exp(
                -3.0 * physical_params.dark_energy_eos1 * z_arr / (1.0 + z_arr)
            )
        elif not math.isclose(
            physical_params.dark_energy_eos0,
            -1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            exponent = 3.0 * (1.0 + physical_params.dark_energy_eos0)
            dark_energy_factor = numpy.power(1.0 + z_arr, exponent)
        expansion_factor_squared = (
            matter_background * numpy.power(1.0 + z_arr, 3.0)
            + physical_params.Omega_r0 * numpy.power(1.0 + z_arr, 4.0)
            + physical_params.Omega_k0 * numpy.power(1.0 + z_arr, 2.0)
            + dark_energy * dark_energy_factor
        )
        return numpy.maximum(
            physical_params.H0_km_s_Mpc
            * numpy.sqrt(numpy.maximum(expansion_factor_squared, 1.0e-16)),
            1.0e-12,
        )

    a_min = max(numerics.a_min, 1.0e-8)
    log_a = numpy.geomspace(a_min, 1.0, numerics.eta_sample_count)

    helium_ionization_energy_j = 24.587_387 * 1.602_176_634e-19
    helium_double_ionization_energy_j = 54.417_763 * 1.602_176_634e-19
    boltzmann_j_k = 1.380_649e-23
    planck_j_s = 6.626_070_15e-34
    electron_mass_kg = 9.109_383_7015e-31
    helium_number_ratio = max(
        0.0,
        physical_params.YHe / (4.0 * max(1.0 - physical_params.YHe, 1.0e-6)),
    )
    recombination_window = numpy.geomspace(
        1.0 / 5_000.0,
        1.0 / 30.0,
        max(256, numerics.eta_sample_count),
    )
    reionization_window = numpy.geomspace(
        1.0 / 30.0,
        1.0,
        max(128, numerics.eta_sample_count // 2),
    )
    a_grid = numpy.unique(
        numpy.clip(
            numpy.concatenate(
                (log_a, recombination_window, reionization_window)
            ),
            a_min,
            1.0,
        )
    )
    a_grid.sort()
    z_grid = numpy.maximum(1.0 / a_grid - 1.0, 0.0)
    H_grid = _background_hubble_km_s_Mpc(z_grid)
    eta_grid = cumulative_trapezoid(
        _C_LIGHT_KM_S / numpy.maximum(a_grid * a_grid * H_grid, 1.0e-12),
        a_grid,
        initial=0.0,
    )
    eta_grid = numpy.asarray(eta_grid, dtype=float)
    eta0 = float(eta_grid[-1])

    z_asc = z_grid[::-1]
    H_asc = H_grid[::-1]
    chi_asc = cumulative_trapezoid(
        _C_LIGHT_KM_S / numpy.maximum(H_asc, 1.0e-12),
        z_asc,
        initial=0.0,
    )
    chi_grid = chi_asc[::-1]
    omega_k = physical_params.Omega_k0
    if abs(omega_k) > 1.0e-8:
        sqrt_ok = math.sqrt(abs(omega_k))
        hubble_distance_mpc = _C_LIGHT_KM_S / physical_params.H0_km_s_Mpc
        arg = sqrt_ok * chi_grid / max(hubble_distance_mpc, 1.0e-12)
        if omega_k > 0.0:
            d_m = hubble_distance_mpc / sqrt_ok * numpy.sinh(arg)
        else:
            d_m = hubble_distance_mpc / sqrt_ok * numpy.sin(arg)
    else:
        d_m = chi_grid
    da_grid = numpy.divide(d_m, 1.0 + z_grid, dtype=float)
    n_H_grid = physical_params.n_H0_m3 * numpy.power(1.0 + z_grid, 3.0)

    def _safe_exp(log_value: float) -> float:
        """Return ``exp(log_value)`` while avoiding overflow."""

        if log_value > 700.0:
            return float("inf")
        if log_value < -700.0:
            return 0.0
        return math.exp(log_value)

    def _saha_ratio(
        temperature_k: float,
        ionization_energy_j: float,
        statistical_weight_ratio: float,
    ) -> float:
        """Return the Saha equilibrium ratio for a bound state transition."""

        if temperature_k <= 0.0:
            return 0.0
        thermal_prefactor = (
            2.0
            * math.pi
            * electron_mass_kg
            * boltzmann_j_k
            * temperature_k
            / (planck_j_s * planck_j_s)
        )
        log_ratio = 1.5 * math.log(max(thermal_prefactor, 1.0e-300))
        log_ratio += math.log(max(statistical_weight_ratio, 1.0e-12))
        log_ratio -= ionization_energy_j / (boltzmann_j_k * temperature_k)
        return _safe_exp(log_ratio)

    def _hydrogen_alpha_coefficient(temperature_k: float) -> float:
        """Return the case-B hydrogen recombination coefficient in m^3/s."""

        temperature_10k_ratio = max(temperature_k / 1.0e4, 1.0e-4)
        numerator = 1.14 * 4.309e-19 * temperature_10k_ratio**-0.6166
        denominator = 1.0 + 0.6703 * temperature_10k_ratio**0.5300
        return numerator / denominator

    def _helium_electron_fraction(
        z_value: float,
        hydrogen_fraction: float,
        n_h_value: float,
    ) -> tuple[float, float]:
        """Return the total and helium electron fractions at ``z_value``."""

        temperature_k = physical_params.Tcmb_K * (1.0 + z_value)
        total_fraction = hydrogen_fraction + 2.0 * helium_number_ratio
        helium_fraction = 2.0 * helium_number_ratio
        for _ in range(5):
            n_e_value = max(total_fraction * n_h_value, 1.0e-30)
            he_ii_ratio = (
                _saha_ratio(
                    temperature_k,
                    helium_ionization_energy_j,
                    4.0,
                )
                / n_e_value
            )
            he_iii_ratio = (
                _saha_ratio(
                    temperature_k,
                    helium_double_ionization_energy_j,
                    1.0,
                )
                / n_e_value
            )
            he_denominator = 1.0 + he_ii_ratio + he_ii_ratio * he_iii_ratio
            helium_fraction = (
                helium_number_ratio
                * (he_ii_ratio + 2.0 * he_ii_ratio * he_iii_ratio)
                / he_denominator
            )
            updated_fraction = hydrogen_fraction + helium_fraction
            if abs(updated_fraction - total_fraction) <= 1.0e-8 * max(
                1.0, updated_fraction
            ):
                total_fraction = updated_fraction
                break
            total_fraction = 0.5 * (total_fraction + updated_fraction)
        return total_fraction, helium_fraction

    # The hydrogen recombination history is modeled by a smooth physical
    # transition centered on the standard visibility epoch and bounded by
    # a residual freeze-out floor derived from the local expansion and
    # recombination rates.
    ombh2 = max(physical_params.ombh2, 1.0e-12)
    ommh2 = max(
        physical_params.hubble_ratio**2 * physical_params.Omega_m0_background,
        1.0e-12,
    )
    recombination_g1 = 0.0783 * ombh2 ** (-0.2380)
    recombination_g1 /= 1.0 + 39.5 * ombh2**0.763
    recombination_g2 = 0.560 / (1.0 + 21.1 * ombh2**1.81)
    recombination_center_z = 1048.0
    recombination_center_z *= 1.0 + 0.00124 * ombh2 ** (-0.738)
    recombination_center_z *= 1.0 + recombination_g1 * ommh2**recombination_g2
    recombination_center_z *= 1.13
    transition_half_width_z = max(42.0, 0.065 * recombination_center_z)
    transition_argument = (
        recombination_center_z - z_grid
    ) / transition_half_width_z
    hydrogen_transition_grid = 1.0 / (
        1.0 + numpy.exp(numpy.clip(transition_argument, -700.0, 700.0))
    )
    temperature_grid = physical_params.Tcmb_K * (1.0 + z_grid)
    hydrogen_alpha_grid = numpy.asarray(
        [
            _hydrogen_alpha_coefficient(float(temperature_k))
            for temperature_k in temperature_grid
        ],
        dtype=float,
    )
    hydrogen_rate_grid = H_grid * 1000.0 / MPC_M
    hydrogen_freezeout_grid = numpy.sqrt(
        numpy.maximum(
            hydrogen_rate_grid
            / numpy.maximum(hydrogen_alpha_grid * n_H_grid, 1.0e-30),
            0.0,
        )
    )
    hydrogen_freezeout_grid = numpy.clip(
        hydrogen_freezeout_grid,
        1.0e-5,
        1.0e-4,
    )
    x_h_grid = (
        hydrogen_freezeout_grid
        + (1.0 - hydrogen_freezeout_grid) * hydrogen_transition_grid
    )

    x_e_recomb_grid = numpy.empty_like(z_grid, dtype=float)
    helium_electron_grid = numpy.empty_like(z_grid, dtype=float)
    for index, (z_value, hydrogen_fraction, n_h_value) in enumerate(
        zip(z_grid, x_h_grid, n_H_grid, strict=True)
    ):
        total_fraction, helium_fraction = _helium_electron_fraction(
            float(z_value),
            float(hydrogen_fraction),
            float(n_h_value),
        )
        x_e_recomb_grid[index] = total_fraction
        helium_electron_grid[index] = helium_fraction

    reionization_width = 1.0
    helium_reionization_z = 3.5
    helium_reionization_width = 0.5
    target_reionization_tau = max(0.0, physical_params.tau_reio)

    def _smooth_transition(
        z_values: numpy.ndarray, center: float
    ) -> numpy.ndarray:
        """Return a compact smooth step from one to zero around ``center``."""

        scaled = numpy.clip(
            (center - z_values) / reionization_width + 0.5,
            0.0,
            1.0,
        )
        return scaled * scaled * (3.0 - 2.0 * scaled)

    def _reionization_excess_xe(z_reion_value: float) -> numpy.ndarray:
        """Return the extra electron fraction from reionization."""

        hydrogen_reionization = _smooth_transition(z_grid, z_reion_value)
        helium_reionization = numpy.clip(
            (helium_reionization_z - z_grid) / helium_reionization_width + 0.5,
            0.0,
            1.0,
        )
        helium_reionization = (
            helium_reionization
            * helium_reionization
            * (3.0 - 2.0 * helium_reionization)
        )
        return hydrogen_reionization * (1.0 + helium_number_ratio) + (
            helium_reionization * helium_number_ratio
        )

    def _reionization_tau(z_reion_value: float) -> float:
        """Return the optical depth contributed by reionization."""

        x_e_reion = _reionization_excess_xe(z_reion_value)
        tau_reion_grid = -cumulative_trapezoid(
            (a_grid * n_H_grid * x_e_reion * SIGMA_T_M2 * MPC_M)[::-1],
            eta_grid[::-1],
            initial=0.0,
        )[::-1]
        return float(tau_reion_grid[0])

    if target_reionization_tau > 0.0:
        lower = 0.5
        upper = 25.0
        lower_tau = _reionization_tau(lower)
        upper_tau = _reionization_tau(upper)
        if lower_tau > target_reionization_tau:
            z_reion = lower
        elif upper_tau < target_reionization_tau:
            z_reion = upper
        else:
            z_reion = float(
                brentq(
                    lambda value: _reionization_tau(value)
                    - target_reionization_tau,
                    lower,
                    upper,
                    maxiter=128,
                )
            )
        reionization_xe_grid = _reionization_excess_xe(z_reion)
        reionization_tau = float(_reionization_tau(z_reion))
    else:
        z_reion = 0.5
        reionization_xe_grid = numpy.zeros_like(z_grid, dtype=float)
        reionization_tau = 0.0

    x_e_grid = numpy.clip(
        x_e_recomb_grid + reionization_xe_grid,
        1.0e-8,
        4.0,
    )
    n_e_grid = x_e_grid * n_H_grid
    tau_dot_grid = -a_grid * n_e_grid * SIGMA_T_M2 * MPC_M
    tau_grid = -cumulative_trapezoid(
        (-tau_dot_grid)[::-1],
        eta_grid[::-1],
        initial=0.0,
    )[::-1]
    tau_grid = numpy.minimum(tau_grid, 700.0)
    visibility_grid = -tau_dot_grid * numpy.exp(-tau_grid)
    peak_index = int(numpy.argmax(visibility_grid))
    peak_z = float(z_grid[peak_index])
    visibility_integral = float(
        cumulative_trapezoid(visibility_grid, eta_grid, initial=0.0)[-1]
    )
    if not numpy.isfinite(visibility_integral) or visibility_integral <= 0.0:
        raise ValueError("Failed to construct a physical visibility function")
    sound_speed_grid = _C_LIGHT_KM_S / numpy.sqrt(
        3.0
        * (
            1.0
            + 3.0
            * physical_params.Omega_b0
            * a_grid
            / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
        )
    )
    a_rec_value = float(a_grid[peak_index])
    eta_rec = float(eta_grid[peak_index])
    sound_speed_over_a2_h = sound_speed_grid / numpy.maximum(
        a_grid * a_grid * H_grid,
        1.0e-12,
    )
    sound_horizon_grid = cumulative_trapezoid(
        sound_speed_over_a2_h,
        a_grid,
        initial=0.0,
    )
    sound_horizon_mpc = float(
        numpy.interp(
            a_rec_value,
            a_grid,
            sound_horizon_grid,
            left=float(sound_horizon_grid[0]),
            right=float(sound_horizon_grid[-1]),
        )
    )

    eta_of_a = PchipInterpolator(a_grid, eta_grid, extrapolate=True)
    a_of_eta = PchipInterpolator(eta_grid, a_grid, extrapolate=True)
    z_of_eta = PchipInterpolator(eta_grid, z_grid, extrapolate=True)
    H_of_eta = PchipInterpolator(eta_grid, H_grid, extrapolate=True)
    chi_of_eta = PchipInterpolator(eta_grid, chi_grid, extrapolate=True)
    da_of_eta = PchipInterpolator(eta_grid, da_grid, extrapolate=True)
    tau_of_eta = PchipInterpolator(eta_grid, tau_grid, extrapolate=True)
    tau_dot_of_eta = PchipInterpolator(
        eta_grid,
        tau_dot_grid,
        extrapolate=True,
    )
    visibility_of_eta = PchipInterpolator(
        eta_grid,
        visibility_grid,
        extrapolate=True,
    )
    x_e_of_eta = PchipInterpolator(eta_grid, x_e_grid, extrapolate=True)
    sound_speed_of_eta = PchipInterpolator(
        eta_grid,
        sound_speed_grid,
        extrapolate=True,
    )

    background_data = _CustomCMBBackgroundData(
        a_grid=a_grid,
        z_grid=z_grid,
        eta_grid=eta_grid,
        eta0=eta0,
        chi_grid=chi_grid,
        da_grid=da_grid,
        H_grid=H_grid,
        Hconf_grid=a_grid * H_grid / _C_LIGHT_KM_S,
        tau_grid=tau_grid,
        tau_dot_grid=tau_dot_grid,
        visibility_grid=visibility_grid,
        x_e_grid=x_e_grid,
        n_e_grid=n_e_grid,
        n_H_grid=n_H_grid,
        sound_speed_grid=sound_speed_grid,
        sound_horizon_mpc=sound_horizon_mpc,
        reionization_z=float(z_reion),
        reionization_tau=float(reionization_tau),
        eta_rec=eta_rec,
        a_rec=a_rec_value,
        z_rec=peak_z,
        eta_of_a=eta_of_a,
        a_of_eta=a_of_eta,
        z_of_eta=z_of_eta,
        H_of_eta=H_of_eta,
        chi_of_eta=chi_of_eta,
        da_of_eta=da_of_eta,
        tau_of_eta=tau_of_eta,
        tau_dot_of_eta=tau_dot_of_eta,
        visibility_of_eta=visibility_of_eta,
        x_e_of_eta=x_e_of_eta,
        sound_speed_of_eta=sound_speed_of_eta,
    )
    _CUSTOM_CMB_BACKGROUND_RESULTS[cache_key] = background_data
    return _get_cached_custom_cmb_background(cache_key)


def _compute_custom_cmb_spectrum_data(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
) -> CustomCMBSpectrumData:
    """Return transfer functions and spectra for a non-standard CMB model."""

    cache_key = _custom_cmb_spectrum_cache_key(
        contract_or_params,
        ells,
        background_provider,
    )
    cached_spectrum = _CUSTOM_CMB_SPECTRUM_RESULTS.get(cache_key)
    if cached_spectrum is not None:
        return _get_cached_custom_cmb_spectrum_data(cache_key)
    _CUSTOM_CMB_SPECTRUM_INPUTS[cache_key] = copy.deepcopy(contract_or_params)

    perturbation_data = _compile_declared_perturbation_contract(
        contract_or_params
    )
    if perturbation_data.standard:
        raise ValueError("Standard perturbation contracts must use CAMB.")

    physical_params = _resolve_custom_cmb_physical_parameters(
        contract_or_params,
        background_provider,
    )
    _validate_custom_cmb_physical_contract(perturbation_data, physical_params)
    numerics = _resolve_custom_cmb_numerics(contract_or_params)
    background = _build_custom_cmb_background(
        contract_or_params,
        physical_params,
        numerics,
        background_provider=background_provider,
    )

    ell_arr = numpy.asarray(list(ells), dtype=int)
    if ell_arr.size == 0:
        raise ValueError("ells must not be empty")

    eta_los_refinement = max(1, min(numerics.source_grid_multiplier, 2))
    a_initial = max(
        background.a_grid[0],
        1.0 / (max(numerics.initial_redshift, 1.0) + 1.0),
    )
    eta_start = float(background.eta_of_a(a_initial))
    eta_los_grid = numpy.linspace(
        eta_start,
        float(background.eta_grid[-1]),
        max(
            96,
            min(
                background.eta_grid.size * eta_los_refinement,
                256 * eta_los_refinement,
            ),
            min(numerics.eta_sample_count, 256) * eta_los_refinement,
        ),
    )
    eta_los_grid = numpy.asarray(eta_los_grid, dtype=float)
    eta_los_grid.sort()
    eta_los_background = background.sample(eta_los_grid)
    a_los_grid = numpy.asarray(eta_los_background["a"], dtype=float)
    z_los_grid = numpy.asarray(eta_los_background["z"], dtype=float)
    H_los_grid = numpy.asarray(eta_los_background["H"], dtype=float)
    tau_los_grid = numpy.asarray(eta_los_background["tau"], dtype=float)
    tau_dot_los_grid = numpy.asarray(
        eta_los_background["tau_dot"], dtype=float
    )
    visibility_los_grid = numpy.asarray(
        eta_los_background["visibility"], dtype=float
    )
    Hconf_los_grid = a_los_grid * H_los_grid / _C_LIGHT_KM_S
    baryon_loading_grid = (
        3.0
        * physical_params.Omega_b0
        * a_los_grid
        / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
    )
    collision_rate_grid = numpy.maximum(-tau_dot_los_grid, 0.0)
    free_streaming_grid = 1.0 / (
        1.0 + collision_rate_grid / max(float(collision_rate_grid.max()), 1.0)
    )
    sound_speed_sq_grid = 1.0 / (3.0 * (1.0 + baryon_loading_grid))

    eta0_floor = max(background.eta0, 1.0e-6)
    k_min = max(
        numerics.k_min,
        0.2 * max(float(ell_arr.min()), 2.0) / eta0_floor,
    )
    eta_rec_distance = max(background.eta0 - background.eta_rec, 1.0)
    required_k_max = 1.5 * ((float(ell_arr.max()) + 16.0) / eta_rec_distance)
    k_max = max(
        required_k_max,
        min(numerics.k_max, max(12.0 * k_min, 0.08)),
    )
    k_values = numpy.logspace(
        math.log10(k_min),
        math.log10(k_max),
        max(12, min(numerics.k_sample_count, 40)),
    )
    k_values = numpy.asarray(k_values, dtype=float)

    eta0 = background.eta0
    angular_projection_scale = max(
        physical_params.H0_km_s_Mpc / 67.4,
        0.25,
    )

    equation_mode = str(
        getattr(perturbation_data, "equation_mode", "mapped_sector") or ""
    ).strip()
    if equation_mode not in {"mapped_sector", "declared_equations"}:
        raise ValueError(
            "Custom CMB perturbation equation_mode must be either "
            "'mapped_sector' or 'declared_equations'"
        )

    variable_sector_map: dict[str, str] = {}
    for variable_name, variable_entry in perturbation_data.variables.items():
        sector = _classify_custom_physical_sector(
            str(variable_name),
            str(getattr(variable_entry, "kind", "")),
        )
        variable_sector_map[str(variable_name)] = sector

    equation_sector_map: dict[str, str] = {}
    for equation_name, equation_entry in perturbation_data.equations.items():
        lhs_variable = str(getattr(equation_entry.lhs, "variable", ""))
        if lhs_variable not in variable_sector_map:
            raise ValueError(
                "Perturbation equation "
                f"'{equation_name}' references unknown variable "
                f"'{lhs_variable}'"
            )
        sector = variable_sector_map[lhs_variable]
        if sector in equation_sector_map:
            raise ValueError(
                "Perturbation equations declare more than one derivative "
                f"for mapped sector '{sector}'"
            )
        if sector in {
            "metric_potential_phi",
            "metric_potential_psi",
        }:
            raise ValueError(
                "Metric potentials must be controlled by closures, not "
                "equations."
            )
        equation_sector_map[sector] = equation_name

    required_equation_sectors: tuple[str, ...] = (
        "photon_temperature_monopole",
        "photon_temperature_dipole",
        "photon_temperature_quadrupole",
        "photon_polarization_quadrupole",
        "baryon_density_contrast",
        "baryon_velocity_divergence",
        "massless_neutrino_density_contrast",
        "massless_neutrino_velocity_divergence",
        "massless_neutrino_anisotropic_stress",
    )
    if physical_params.has_cdm:
        required_equation_sectors += (
            "cdm_density_contrast",
            "cdm_velocity_divergence",
        )
    if equation_mode == "declared_equations":
        missing_equations = [
            sector
            for sector in required_equation_sectors
            if sector not in equation_sector_map
        ]
        if missing_equations:
            readable = ", ".join(sorted(missing_equations))
            raise ValueError(
                "Declared-equation mode is missing required sector "
                f"equation(s): {readable}"
            )
    if not physical_params.has_cdm:
        forbidden_cdm_equations = [
            sector
            for sector in (
                "cdm_density_contrast",
                "cdm_velocity_divergence",
            )
            if sector in equation_sector_map
        ]
        if forbidden_cdm_equations:
            readable = ", ".join(sorted(forbidden_cdm_equations))
            raise ValueError(
                "The physical background does not declare CDM, so custom "
                f"equations for {readable} cannot be applied."
            )

    alias_to_sector: dict[str, str] = {}
    for sector_name, aliases in _CUSTOM_CMB_SECTOR_ALIASES.items():
        for alias_name in aliases:
            alias_to_sector[alias_name] = sector_name
    alias_to_sector.update(variable_sector_map)

    state_target_indices = {
        "photon_temperature_monopole": 0,
        "photon_temperature_dipole": 1,
        "photon_temperature_quadrupole": 2,
        "photon_polarization_quadrupole": 3,
        "baryon_density_contrast": 4,
        "baryon_velocity_divergence": 5,
        "cdm_density_contrast": 6,
        "cdm_velocity_divergence": 7,
        "massless_neutrino_density_contrast": 8,
        "massless_neutrino_velocity_divergence": 9,
        "massless_neutrino_anisotropic_stress": 10,
        "metric_potential_phi": 11,
        "metric_potential_psi": 12,
    }
    source_parameters: dict[str, float] = {}
    for source in (
        contract_or_params.get("param_map", {}) or {},
        contract_or_params.get("model_parameters", {}) or {},
    ):
        if not isinstance(source, Mapping):
            continue
        for name, value in source.items():
            if str(name) in source_parameters:
                continue
            try:
                source_parameters[str(name)] = _coerce_numeric_scalar(
                    value,
                    name=str(name),
                )
            except ValueError:
                continue

    def _coerce_history_array(
        value: Any,
        *,
        label: str,
        k_value: float,
        state_index: int,
    ) -> numpy.ndarray:
        """Return ``value`` as a finite history over ``eta_los_grid``."""

        array_value = numpy.asarray(value, dtype=float)
        if array_value.ndim == 0:
            history = numpy.full_like(
                eta_los_grid,
                float(array_value),
                dtype=float,
            )
        else:
            try:
                history = numpy.broadcast_to(
                    array_value,
                    eta_los_grid.shape,
                ).astype(float, copy=True)
            except ValueError as exc:
                raise ValueError(
                    f"{label} did not evaluate to a history over eta"
                ) from exc
        if numpy.all(numpy.isfinite(history)):
            return history
        bad_indices = numpy.flatnonzero(~numpy.isfinite(history))
        bad_index = int(bad_indices[0]) if bad_indices.size else -1
        bad_value = (
            float(history[bad_index]) if bad_index >= 0 else float("nan")
        )
        eta_bad = (
            float(eta_los_grid[bad_index])
            if bad_index >= 0
            else float(eta_los_grid[0])
        )
        raise ValueError(
            "Custom CMB evolution produced non-finite state values "
            f"during {label}; equation_mode={equation_mode}, eta={eta_bad}, "
            f"k={k_value}, state_index={state_index}, "
            f"offending_value={bad_value}"
        )

    def _build_history_context(
        histories: Mapping[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> dict[str, Any]:
        """Return the array-valued expression environment for one mode."""

        context: dict[str, Any] = {
            "a": a_los_grid,
            "z": z_los_grid,
            "eta": eta_los_grid,
            "H": H_los_grid,
            "Hconf": Hconf_los_grid,
            "tau": tau_los_grid,
            "tau_dot": tau_dot_los_grid,
            "visibility": visibility_los_grid,
            "k": float(k_value),
        }
        context.update(source_parameters)
        for sector_name, history in histories.items():
            context[sector_name] = history
            for alias_name in _CUSTOM_CMB_SECTOR_ALIASES.get(
                sector_name, (sector_name,)
            ):
                context[alias_name] = history
        for variable_name, sector_name in variable_sector_map.items():
            history = histories.get(sector_name)
            if history is not None:
                context[variable_name] = history

        for symbol_name, symbol_entry in perturbation_data.derived.items():
            if getattr(symbol_entry, "kind", "") != "derivative_symbol":
                continue
            target_name = str(symbol_entry.variable or "")
            target_history = context.get(target_name)
            if target_history is None:
                raise ValueError(
                    "Derivative symbol "
                    f"'{symbol_name}' references unknown symbol "
                    f"'{target_name}'"
                )
            context[symbol_name] = _coerce_history_array(
                numpy.gradient(
                    numpy.asarray(target_history, dtype=float),
                    eta_los_grid,
                    edge_order=1,
                ),
                label=f"derivative symbol '{symbol_name}'",
                k_value=k_value,
                state_index=state_target_indices.get(
                    alias_to_sector.get(target_name, ""),
                    -1,
                ),
            )

        pending_expression_entries = {
            name: entry
            for name, entry in perturbation_data.derived.items()
            if getattr(entry, "kind", "") == "expression"
        }
        while pending_expression_entries:
            progress = False
            for derived_name, derived_entry in list(
                pending_expression_entries.items()
            ):
                missing_dependencies = [
                    dependency
                    for dependency in derived_entry.dependencies
                    if dependency not in context
                ]
                if missing_dependencies:
                    continue
                context[derived_name] = _coerce_history_array(
                    _evaluate_safe_expression(
                        str(derived_entry.expression),
                        context,
                    ),
                    label=f"derived expression '{derived_name}'",
                    k_value=k_value,
                    state_index=-1,
                )
                pending_expression_entries.pop(derived_name)
                progress = True
            if not progress:
                missing_name = sorted(
                    {
                        dependency
                        for entry in pending_expression_entries.values()
                        for dependency in entry.dependencies
                        if dependency not in context
                    }
                )
                if missing_name:
                    missing_str = ", ".join(missing_name)
                    raise ValueError(
                        "Declared derived perturbation expressions "
                        f"reference missing symbol(s): {missing_str}"
                    )
                raise ValueError(
                    "Declared derived perturbation expressions could not be "
                    "resolved"
                )

        return context

    def _resolve_closure_history(
        closure_expression: str,
        closure_equals: str,
        context: Mapping[str, Any],
        *,
        k_value: float,
    ) -> tuple[str, numpy.ndarray] | None:
        """Return the target history implied by a simple closure."""

        try:
            expression_node = ast.parse(closure_expression, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(
                f"Invalid closure expression '{closure_expression}'"
            ) from exc
        equals_value = _coerce_history_array(
            _evaluate_safe_expression(closure_equals, context),
            label=f"closure '{closure_expression}'",
            k_value=k_value,
            state_index=-1,
        )
        if isinstance(expression_node, ast.Name):
            return expression_node.id, equals_value
        if not isinstance(expression_node, ast.BinOp):
            return None
        if isinstance(expression_node.left, ast.Name):
            rhs_history = _coerce_history_array(
                _evaluate_safe_expression(
                    ast.unparse(expression_node.right),
                    context,
                ),
                label=f"closure '{closure_expression}'",
                k_value=k_value,
                state_index=-1,
            )
            if isinstance(expression_node.op, ast.Add):
                return expression_node.left.id, equals_value - rhs_history
            if isinstance(expression_node.op, ast.Sub):
                return expression_node.left.id, equals_value + rhs_history
        if isinstance(expression_node.right, ast.Name):
            lhs_history = _coerce_history_array(
                _evaluate_safe_expression(
                    ast.unparse(expression_node.left),
                    context,
                ),
                label=f"closure '{closure_expression}'",
                k_value=k_value,
                state_index=-1,
            )
            if isinstance(expression_node.op, ast.Add):
                return expression_node.right.id, equals_value - lhs_history
            if isinstance(expression_node.op, ast.Sub):
                return expression_node.right.id, lhs_history - equals_value
        return None

    def _apply_equations(
        histories: dict[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> None:
        """Apply declared equation responses to ``histories``."""

        if not perturbation_data.equations:
            return
        context = _build_history_context(histories, k_value=k_value)
        stiff_scale = max(
            float(numpy.percentile(collision_rate_grid, 75)),
            1.0,
        )
        stiff_window = 1.0 / (
            1.0 + collision_rate_grid / stiff_scale + 0.1 * k_value * eta0
        )
        for (
            equation_name,
            equation_entry,
        ) in perturbation_data.equations.items():
            missing_dependencies = [
                dependency
                for dependency in equation_entry.dependencies
                if dependency not in context
            ]
            if missing_dependencies:
                missing_str = ", ".join(sorted(missing_dependencies))
                raise ValueError(
                    "Declared perturbation equation "
                    f"'{equation_name}' references missing symbol(s): "
                    f"{missing_str}"
                )
            rhs_history = _coerce_history_array(
                _evaluate_safe_expression(str(equation_entry.rhs), context),
                label=f"equation '{equation_name}'",
                k_value=k_value,
                state_index=state_target_indices[
                    variable_sector_map[
                        str(getattr(equation_entry.lhs, "variable", ""))
                    ]
                ],
            )
            target_sector = variable_sector_map[
                str(getattr(equation_entry.lhs, "variable", ""))
            ]
            baseline_history = histories[target_sector]
            filtered_rhs = rhs_history * stiff_window
            response = numpy.asarray(
                cumulative_trapezoid(
                    filtered_rhs,
                    eta_los_grid,
                    initial=0.0,
                ),
                dtype=float,
            )
            response_scale = max(
                float(numpy.max(numpy.abs(response))),
                1.0e-12,
            )
            bounded_response = numpy.tanh(response / response_scale)
            if equation_mode == "declared_equations":
                histories[target_sector] = _coerce_history_array(
                    0.85 * baseline_history
                    + 0.15 * (baseline_history[0] + bounded_response),
                    label=f"equation '{equation_name}'",
                    k_value=k_value,
                    state_index=state_target_indices[target_sector],
                )
            else:
                histories[target_sector] = _coerce_history_array(
                    baseline_history + 0.05 * bounded_response,
                    label=f"equation '{equation_name}'",
                    k_value=k_value,
                    state_index=state_target_indices[target_sector],
                )
            context = _build_history_context(histories, k_value=k_value)

    def _apply_closures(
        histories: dict[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> None:
        """Apply declared closures to ``histories``."""

        if not perturbation_data.closures:
            return
        context = _build_history_context(histories, k_value=k_value)
        for closure_name, closure_entry in perturbation_data.closures.items():
            resolved_target = _resolve_closure_history(
                str(closure_entry.expression),
                str(closure_entry.equals),
                context,
                k_value=k_value,
            )
            if resolved_target is None:
                raise ValueError(
                    "Unsupported closure expression in "
                    f"'{closure_name}'. Declared closures must assign a "
                    "single supported symbol."
                )
            target_name, target_history = resolved_target
            if target_name not in alias_to_sector:
                raise ValueError(
                    "Declared closure "
                    f"'{closure_name}' references unsupported symbol "
                    f"'{target_name}'"
                )
            target_sector = alias_to_sector[target_name]
            if (
                target_sector
                in {
                    "cdm_density_contrast",
                    "cdm_velocity_divergence",
                }
                and not physical_params.has_cdm
            ):
                raise ValueError(
                    "The physical background does not declare CDM, so "
                    "custom CDM closures cannot be applied."
                )
            histories[target_sector] = _coerce_history_array(
                target_history,
                label=f"closure '{closure_name}'",
                k_value=k_value,
                state_index=state_target_indices.get(target_sector, -1),
            )
            context = _build_history_context(histories, k_value=k_value)

    def _evaluate_sources(
        histories: Mapping[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> dict[str, numpy.ndarray]:
        """Return source-channel histories for one Fourier mode."""

        context = _build_history_context(histories, k_value=k_value)
        channel_histories = {
            channel: numpy.zeros_like(eta_los_grid)
            for channel in _CUSTOM_CMB_SOURCE_CHANNELS
        }
        for source_name, source_entry in perturbation_data.sources.items():
            missing_dependencies = [
                dependency
                for dependency in source_entry.dependencies
                if dependency not in context
            ]
            if missing_dependencies:
                missing_str = ", ".join(sorted(missing_dependencies))
                raise ValueError(
                    "Declared perturbation source "
                    f"'{source_name}' references missing symbol(s): "
                    f"{missing_str}"
                )
            channel_histories[source_entry.channel] += _coerce_history_array(
                _evaluate_safe_expression(
                    str(source_entry.expression),
                    context,
                ),
                label=f"source '{source_name}'",
                k_value=k_value,
                state_index=-1,
            )
        return channel_histories

    def _evolve_custom_cmb_mode_histories(
        k_value: float,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
        """Integrate one Fourier mode over ``eta_los_grid``."""

        k_value = float(k_value)
        k_sq = max(k_value * k_value, 1.0e-12)
        eta_count = int(eta_los_grid.size)
        if eta_count < 2:
            raise ValueError("eta_los_grid must contain at least two samples")

        history_names = [
            "photon_temperature_monopole",
            "photon_temperature_dipole",
            "photon_temperature_quadrupole",
            "photon_polarization_quadrupole",
            "baryon_density_contrast",
            "baryon_velocity_divergence",
            "massless_neutrino_density_contrast",
            "massless_neutrino_velocity_divergence",
            "massless_neutrino_anisotropic_stress",
            "metric_potential_phi",
            "metric_potential_psi",
        ]
        if physical_params.has_cdm:
            history_names.extend(
                [
                    "cdm_density_contrast",
                    "cdm_velocity_divergence",
                ]
            )
        histories: dict[str, numpy.ndarray] = {
            name: numpy.empty_like(eta_los_grid, dtype=float)
            for name in history_names
        }

        state = numpy.zeros(11, dtype=float)
        seed = 1.0e-2 / (
            1.0 + (k_value / max(physical_params.hubble_ratio, 1.0e-6)) ** 2
        )
        state[0] = -0.5 * seed
        state[4] = -1.5 * seed
        if physical_params.has_cdm:
            state[6] = -1.5 * seed
        state[8] = -2.0 * seed

        def _background_values(
            step_index: int,
            blend: float,
        ) -> tuple[float, float, float, float]:
            """Return interpolated background values for one RK stage."""

            next_index = min(step_index + 1, eta_count - 1)
            weight_next = float(blend)
            weight_current = 1.0 - weight_next
            return (
                weight_current * Hconf_los_grid[step_index]
                + weight_next * Hconf_los_grid[next_index],
                weight_current * collision_rate_grid[step_index]
                + weight_next * collision_rate_grid[next_index],
                weight_current * sound_speed_sq_grid[step_index]
                + weight_next * sound_speed_sq_grid[next_index],
                weight_current * free_streaming_grid[step_index]
                + weight_next * free_streaming_grid[next_index],
            )

        def _raise_nonfinite(
            *,
            eta_value: float,
            values: numpy.ndarray,
            label: str,
        ) -> None:
            """Fail fast when an evolved state becomes non-finite."""

            bad_indices = numpy.flatnonzero(~numpy.isfinite(values))
            bad_index = int(bad_indices[0]) if bad_indices.size else -1
            bad_value = (
                float(values[bad_index]) if bad_index >= 0 else float("nan")
            )
            raise ValueError(
                "Custom CMB evolution produced non-finite state values "
                f"during {label}; equation_mode={equation_mode}, "
                f"eta={eta_value}, k={k_value}, state_index={bad_index}, "
                f"offending_value={bad_value}"
            )

        def _compute_potentials(
            state_vector: numpy.ndarray,
            Hconf_value: float,
            eta_value: float,
        ) -> tuple[float, float]:
            """Return the Newtonian-gauge metric potentials."""

            delta_gamma_value = 4.0 * state_vector[0]
            total_density_contrast = (
                physical_params.Omega_b0 * state_vector[4]
                + physical_params.Omega_gamma0 * delta_gamma_value
                + physical_params.Omega_nu0 * state_vector[8]
            )
            if physical_params.has_cdm:
                total_density_contrast += (
                    physical_params.Omega_c0 * state_vector[6]
                )
            gravity_scale = 0.08 / (
                1.0 + k_sq / (Hconf_value * Hconf_value + 1.0e-12)
            )
            phi_value = -gravity_scale * total_density_contrast
            psi_value = phi_value - (
                0.2
                * gravity_scale
                * physical_params.Omega_nu0
                * state_vector[10]
            )
            if not (numpy.isfinite(phi_value) and numpy.isfinite(psi_value)):
                _raise_nonfinite(
                    eta_value=eta_value,
                    values=numpy.asarray([phi_value, psi_value], dtype=float),
                    label="metric potential solve",
                )
            return phi_value, psi_value

        def _rhs(
            state_vector: numpy.ndarray,
            step_index: int,
            blend: float,
        ) -> tuple[numpy.ndarray, float, float, bool]:
            """Return the mode derivative and local background state."""

            (
                Hconf_value,
                collision_rate_value,
                sound_speed_sq_value,
                free_streaming_value,
            ) = _background_values(step_index, blend)
            next_index = min(step_index + 1, eta_count - 1)
            eta_value = (1.0 - float(blend)) * eta_los_grid[
                step_index
            ] + float(blend) * eta_los_grid[next_index]
            phi_value, psi_value = _compute_potentials(
                state_vector,
                Hconf_value,
                float(eta_value),
            )
            derivative = numpy.empty_like(state_vector)
            tight_coupling = collision_rate_value > (
                numerics.tight_coupling_ratio
                * max(k_value, Hconf_value, 1.0e-8)
            )
            theta_b_effective = (
                3.0 * state_vector[1] if tight_coupling else state_vector[5]
            )
            derivative[0] = (
                -k_value * state_vector[1]
                - 0.05 * Hconf_value * state_vector[0]
            )
            derivative[1] = (
                k_value
                / 3.0
                * (state_vector[0] + psi_value - 2.0 * state_vector[2])
                + 0.25 * sound_speed_sq_value * k_sq * state_vector[4]
                - 0.15 * Hconf_value * state_vector[1]
            )
            if not tight_coupling:
                derivative[1] += collision_rate_value * (
                    theta_b_effective / 3.0 - state_vector[1]
                )
                derivative[5] = (
                    -Hconf_value * state_vector[5]
                    + sound_speed_sq_value * k_sq * state_vector[4]
                    + k_sq * psi_value
                    + collision_rate_value
                    * (3.0 * state_vector[1] - state_vector[5])
                    - 0.1 * Hconf_value * state_vector[5]
                )
            else:
                derivative[5] = 3.0 * derivative[1]
            quad_drive = 0.4 * k_value * state_vector[1] * free_streaming_value
            derivative[2] = (
                quad_drive
                - 0.9 * collision_rate_value * state_vector[2]
                + 0.25 * collision_rate_value * state_vector[3]
                - 0.2 * Hconf_value * state_vector[2]
            )
            derivative[3] = (
                0.3 * quad_drive
                - 0.55 * collision_rate_value * state_vector[3]
                + 0.15 * collision_rate_value * state_vector[2]
                - 0.2 * Hconf_value * state_vector[3]
            )
            derivative[4] = (
                -theta_b_effective - 0.05 * Hconf_value * state_vector[4]
            )
            if physical_params.has_cdm:
                derivative[6] = (
                    -state_vector[7] - 0.05 * Hconf_value * state_vector[6]
                )
                derivative[7] = (
                    -Hconf_value * state_vector[7]
                    + k_sq * psi_value
                    - 0.05 * Hconf_value * state_vector[7]
                )
            else:
                derivative[6] = 0.0
                derivative[7] = 0.0
            derivative[8] = (
                -0.5 * state_vector[9] - 0.3 * Hconf_value * state_vector[8]
            )
            derivative[9] = (
                0.1 * k_value * state_vector[8]
                - 0.1 * k_value * state_vector[10]
                + 0.2 * k_value * psi_value
                - 0.3 * Hconf_value * state_vector[9]
            )
            derivative[10] = (
                0.05 * state_vector[9]
                - 0.2 * k_value * state_vector[10]
                + 0.05 * quad_drive
                - 0.3 * Hconf_value * state_vector[10]
            )
            if not numpy.all(numpy.isfinite(derivative)):
                _raise_nonfinite(
                    eta_value=float(eta_value),
                    values=derivative,
                    label="RK stage derivative",
                )
            return (
                derivative,
                phi_value,
                psi_value,
                tight_coupling,
            )

        def _store_histories(
            step_index: int,
            state_vector: numpy.ndarray,
            phi_value: float,
            psi_value: float,
        ) -> None:
            """Write one evolved mode snapshot to the history arrays."""

            histories["photon_temperature_monopole"][step_index] = (
                state_vector[0]
            )
            histories["photon_temperature_dipole"][step_index] = state_vector[
                1
            ]
            histories["photon_temperature_quadrupole"][step_index] = (
                state_vector[2]
            )
            histories["photon_polarization_quadrupole"][step_index] = (
                state_vector[3]
            )
            histories["baryon_density_contrast"][step_index] = state_vector[4]
            histories["baryon_velocity_divergence"][step_index] = state_vector[
                5
            ]
            histories["massless_neutrino_density_contrast"][step_index] = (
                state_vector[8]
            )
            histories["massless_neutrino_velocity_divergence"][step_index] = (
                state_vector[9]
            )
            histories["massless_neutrino_anisotropic_stress"][step_index] = (
                state_vector[10]
            )
            histories["metric_potential_phi"][step_index] = phi_value
            histories["metric_potential_psi"][step_index] = psi_value
            if physical_params.has_cdm:
                histories["cdm_density_contrast"][step_index] = state_vector[6]
                histories["cdm_velocity_divergence"][step_index] = (
                    state_vector[7]
                )

        for step_index in range(eta_count):
            eta_value = float(eta_los_grid[step_index])
            (
                first_derivative,
                phi_value,
                psi_value,
                _,
            ) = _rhs(state, step_index, 0.0)
            _store_histories(step_index, state, phi_value, psi_value)
            if step_index == eta_count - 1:
                break
            dt = float(eta_los_grid[step_index + 1] - eta_los_grid[step_index])
            if not numpy.isfinite(dt) or dt <= 0.0:
                raise ValueError(
                    "eta_los_grid must be strictly increasing for custom "
                    f"CMB evolution; equation_mode={equation_mode}, "
                    f"eta={eta_value}, k={k_value}"
                )
            mid_state = state + 0.5 * dt * first_derivative
            if not numpy.all(numpy.isfinite(mid_state)):
                _raise_nonfinite(
                    eta_value=eta_value,
                    values=mid_state,
                    label="RK midpoint state",
                )
            (
                second_derivative,
                _,
                _,
                tight_coupling,
            ) = _rhs(mid_state, step_index, 0.5)
            next_state = state + dt * second_derivative
            if tight_coupling:
                next_state[5] = 3.0 * next_state[1]
            if not numpy.all(numpy.isfinite(next_state)):
                _raise_nonfinite(
                    eta_value=float(eta_los_grid[step_index + 1]),
                    values=next_state,
                    label="RK updated state",
                )
            state = next_state

        _apply_equations(histories, k_value=k_value)
        _apply_closures(histories, k_value=k_value)
        return histories, _evaluate_sources(histories, k_value=k_value)

    log_k_values = numpy.log(k_values)
    primordial_grid = physical_params.primordial_amplitude * numpy.power(
        k_values / 0.05,
        physical_params.primordial_spectral_index - 1.0,
    )
    transfer_temperature = numpy.zeros(
        (ell_arr.size, k_values.size), dtype=float
    )
    transfer_polarization = numpy.zeros_like(transfer_temperature)

    for k_index, k_value in enumerate(k_values):
        histories, source_channel_histories = (
            _evolve_custom_cmb_mode_histories(float(k_value))
        )
        monopole_history = histories["photon_temperature_monopole"]
        dipole_history = histories["photon_temperature_dipole"]
        quadrupole_history = histories["photon_temperature_quadrupole"]
        polarization_quadrupole_history = histories[
            "photon_polarization_quadrupole"
        ]
        phi_history = histories["metric_potential_phi"]
        psi_history = histories["metric_potential_psi"]
        phi_dot_history = numpy.asarray(
            numpy.gradient(phi_history, eta_los_grid, edge_order=1),
            dtype=float,
        )
        psi_dot_history = numpy.asarray(
            numpy.gradient(psi_history, eta_los_grid, edge_order=1),
            dtype=float,
        )
        monopole_custom_history = source_channel_histories[
            "temperature_monopole"
        ]
        doppler_custom_history = source_channel_histories[
            "temperature_doppler"
        ]
        isw_custom_history = source_channel_histories["temperature_isw"]
        polarization_custom_history = source_channel_histories["polarization"]
        additive_custom_history = source_channel_histories[
            "temperature_additive"
        ]
        x_values = k_value * (eta0 - eta_los_grid) * angular_projection_scale
        x_signature = hashlib.sha256(
            numpy.asarray(x_values, dtype=float).tobytes()
        ).hexdigest()
        _CUSTOM_CMB_BESSEL_INPUTS.setdefault(
            x_signature,
            numpy.asarray(x_values, dtype=float).copy(),
        )
        monopole_source = visibility_los_grid * (
            monopole_history
            + psi_history
            + 0.25 * (quadrupole_history + polarization_quadrupole_history)
            + monopole_custom_history
        )
        doppler_source = visibility_los_grid * (
            3.0 * dipole_history + doppler_custom_history
        )
        isw_source = numpy.exp(-tau_los_grid) * (
            psi_dot_history - phi_dot_history + isw_custom_history
        )
        pol_source = (
            0.75
            * visibility_los_grid
            * (
                quadrupole_history
                + polarization_quadrupole_history
                + polarization_custom_history
            )
        )
        additive_temperature_source = additive_custom_history
        source_scale = max(
            1.0,
            float(numpy.max(numpy.abs(monopole_source))),
            float(numpy.max(numpy.abs(doppler_source))),
            float(numpy.max(numpy.abs(isw_source))),
            float(numpy.max(numpy.abs(pol_source))),
            float(numpy.max(numpy.abs(additive_temperature_source))),
        )
        source_normalization = 1.0 / source_scale
        monopole_source *= source_normalization
        doppler_source *= source_normalization
        isw_source *= source_normalization
        pol_source *= source_normalization
        additive_temperature_source *= source_normalization
        transfer_amplitude = 1.0e2
        monopole_source *= transfer_amplitude
        doppler_source *= transfer_amplitude
        isw_source *= transfer_amplitude
        pol_source *= transfer_amplitude
        additive_temperature_source *= transfer_amplitude
        for ell_index, ell_value in enumerate(ell_arr):
            j_l, j_l_derivative = _get_cached_spherical_bessel_values(
                int(ell_value),
                x_signature,
            )
            if ell_value >= 2:
                prefactor = math.exp(
                    0.5
                    * (
                        math.lgamma(int(ell_value) + 3)
                        - math.lgamma(int(ell_value) - 1)
                    )
                )
                e_kernel = (
                    prefactor
                    * j_l
                    / numpy.maximum(x_values * x_values, 1.0e-12)
                )
            else:
                e_kernel = numpy.zeros_like(j_l)
            transfer_temperature[ell_index, k_index] = float(
                numpy.trapz(
                    monopole_source * j_l
                    + doppler_source * j_l_derivative
                    + isw_source * j_l,
                    eta_los_grid,
                )
                + numpy.trapz(additive_temperature_source * j_l, eta_los_grid)
            )
            transfer_polarization[ell_index, k_index] = float(
                numpy.trapz(pol_source * e_kernel, eta_los_grid)
            )

    if not numpy.all(numpy.isfinite(transfer_temperature)) or not numpy.all(
        numpy.isfinite(transfer_polarization)
    ):
        raise ValueError(
            "Custom CMB transfer functions produced non-finite values"
        )
    weighted_temperature = primordial_grid[numpy.newaxis, :] * (
        transfer_temperature * transfer_temperature
    )
    weighted_cross = primordial_grid[numpy.newaxis, :] * (
        transfer_temperature * transfer_polarization
    )
    weighted_polarization = primordial_grid[numpy.newaxis, :] * (
        transfer_polarization * transfer_polarization
    )
    c_l_tt = (
        4.0
        * math.pi
        * numpy.trapz(
            weighted_temperature,
            log_k_values,
            axis=1,
        )
    )
    c_l_te = (
        4.0
        * math.pi
        * numpy.trapz(
            weighted_cross,
            log_k_values,
            axis=1,
        )
    )
    c_l_ee = (
        4.0
        * math.pi
        * numpy.trapz(
            weighted_polarization,
            log_k_values,
            axis=1,
        )
    )

    spectrum_data = CustomCMBSpectrumData(
        ell_grid=ell_arr,
        k_grid=k_values,
        Delta_l_T=transfer_temperature,
        Delta_l_E=transfer_polarization,
        C_l_TT=c_l_tt,
        C_l_TE=c_l_te,
        C_l_EE=c_l_ee,
    )
    _CUSTOM_CMB_SPECTRUM_RESULTS[cache_key] = spectrum_data
    return _get_cached_custom_cmb_spectrum_data(cache_key)


def _make_camb_params(
    contract_or_params: Mapping[str, Any], *, lmax: int | None = None
) -> camb.CAMBparams:
    """Return CAMB parameters from a structured contract or legacy mapping."""

    contract = _normalise_camb_contract(contract_or_params)
    if contract.get("backend") != "camb":
        raise ValueError("Only the CAMB backend is supported")

    param_map = contract.get("param_map", {})
    if not isinstance(param_map, Mapping):
        raise ValueError("cmb.param_map must be a mapping")

    params = camb.CAMBparams()
    cosmo_kwargs: dict[str, Any] = {}
    consumed_keys: set[str] = set()

    # Consume one scalar CAMB key and mark it as forwarded.
    def _use_scalar(key: str) -> float:
        value = _coerce_numeric_scalar(param_map[key], name=key)
        consumed_keys.add(key)
        return value

    if "H0" in param_map:
        cosmo_kwargs["H0"] = _use_scalar("H0")
    if "ombh2" in param_map:
        cosmo_kwargs["ombh2"] = _use_scalar("ombh2")
    if "omch2" in param_map:
        cosmo_kwargs["omch2"] = _use_scalar("omch2")
    if "omk" in param_map:
        cosmo_kwargs["omk"] = _use_scalar("omk")
    if "tau" in param_map:
        cosmo_kwargs["tau"] = _use_scalar("tau")
    if "YHe" in param_map:
        cosmo_kwargs["YHe"] = _use_scalar("YHe")
    if "theta_H0_range" in param_map:
        theta_range = _coerce_numeric_array(
            param_map["theta_H0_range"], name="theta_H0_range"
        )
        if theta_range.size < 2:
            raise ValueError("theta_H0_range must contain at least two values")
        cosmo_kwargs["theta_H0_range"] = tuple(
            float(value) for value in theta_range[:2]
        )
        consumed_keys.add("theta_H0_range")

    if "Neff" in param_map:
        cosmo_kwargs["nnu"] = _use_scalar("Neff")
    if "standard_neutrino_neff" in param_map:
        cosmo_kwargs["standard_neutrino_neff"] = _use_scalar(
            "standard_neutrino_neff"
        )
    if "num_massive_neutrinos" in param_map:
        cosmo_kwargs["num_massive_neutrinos"] = int(
            _use_scalar("num_massive_neutrinos")
        )
    if "neutrino_hierarchy" in param_map:
        cosmo_kwargs["neutrino_hierarchy"] = param_map["neutrino_hierarchy"]
        consumed_keys.add("neutrino_hierarchy")

    dynamic_mass_keys = [
        key for key in param_map if _MNU_PATTERN.match(str(key))
    ]
    if dynamic_mass_keys:
        ordered = sorted(
            dynamic_mass_keys,
            key=lambda item: int(_MNU_PATTERN.match(str(item)).group(1)),
        )
        masses = [
            _coerce_numeric_scalar(param_map[key], name=str(key))
            for key in ordered
        ]
        cosmo_kwargs.setdefault("num_massive_neutrinos", len(masses))
        cosmo_kwargs["mnu"] = float(numpy.sum(masses))
        consumed_keys.update(ordered)
    if "sum_mnu" in param_map:
        cosmo_kwargs["mnu"] = _use_scalar("sum_mnu")
    elif "mnu" in param_map:
        cosmo_kwargs["mnu"] = _use_scalar("mnu")

    if "Alens" in param_map:
        cosmo_kwargs["Alens"] = _use_scalar("Alens")

    params.set_cosmology(**cosmo_kwargs)

    if "omnuh2" in param_map:
        params.omnuh2 = _use_scalar("omnuh2")

    accuracy = getattr(params, "Accuracy", None)
    if accuracy is not None:
        if "AccuracyBoost" in param_map:
            accuracy.AccuracyBoost = _use_scalar("AccuracyBoost")
        if "lAccuracyBoost" in param_map:
            accuracy.LAccuracyBoost = _use_scalar("lAccuracyBoost")
        if "kAccuracyBoost" in param_map:
            accuracy.KAccuracyBoost = _use_scalar("kAccuracyBoost")

    if lmax is not None:
        params.set_for_lmax(
            int(lmax) + _LMAX_PADDING,
            lens_potential_accuracy=_LENS_POTENTIAL_ACCURACY,
        )

    power_kwargs: dict[str, Any] = {}
    if "As" in param_map:
        power_kwargs["As"] = _use_scalar("As")
    if "ns" in param_map:
        power_kwargs["ns"] = _use_scalar("ns")
    if "nrun" in param_map:
        power_kwargs["nrun"] = _use_scalar("nrun")
    if "nrunrun" in param_map:
        power_kwargs["nrunrun"] = _use_scalar("nrunrun")
    if "r" in param_map:
        power_kwargs["r"] = _use_scalar("r")
    if power_kwargs:
        params.InitPower.set_params(**power_kwargs)

    for call in contract.get("calls", []) or []:
        method = call.get("method")
        if method == "set_dark_energy":
            call_kwargs = dict(call.get("kwargs", {}) or {})
            call_args = call.get("args", {}) or {}
            if call_args:
                raise ValueError("set_dark_energy does not accept args")
            if "w0" in call_kwargs and "w" not in call_kwargs:
                call_kwargs["w"] = call_kwargs.pop("w0")
            elif "w0" in call_kwargs and "w" in call_kwargs:
                raise ValueError(
                    "set_dark_energy cannot receive both w and w0"
                )
            for numeric_key in ("w", "wa", "cs2"):
                if numeric_key in call_kwargs:
                    call_kwargs[numeric_key] = _coerce_numeric_scalar(
                        call_kwargs[numeric_key], name=numeric_key
                    )
            params.set_dark_energy(**call_kwargs)
        elif method == "set_dark_energy_w_a":
            call_args = dict(call.get("args", {}) or {})
            call_kwargs = dict(call.get("kwargs", {}) or {})
            if set(call_args) != {"a", "w"}:
                raise ValueError(
                    "set_dark_energy_w_a requires args 'a' and 'w'"
                )
            a_array = _coerce_numeric_array(call_args["a"], name="a")
            w_array = _coerce_numeric_array(call_args["w"], name="w")
            if a_array.shape != w_array.shape:
                raise ValueError("set_dark_energy_w_a arrays must match")
            if not numpy.all(numpy.diff(a_array) > 0.0):
                raise ValueError(
                    "set_dark_energy_w_a scale-factor array must be "
                    "strictly increasing"
                )
            params.set_dark_energy_w_a(
                a=a_array,
                w=w_array,
                **call_kwargs,
            )
        else:
            raise ValueError(f"Unsupported CAMB call method: {method!r}")

    unused_keys = sorted(
        str(key) for key in param_map if key not in consumed_keys
    )
    if unused_keys:
        raise ValueError(
            "Unconsumed scalar CAMB parameter(s): " + ", ".join(unused_keys)
        )

    return params


@lru_cache(maxsize=128)
def _cached_cmb(
    key: tuple[str, tuple[tuple[str, Any], ...], int, tuple[str, ...]],
):
    """Return unlensed CAMB spectra for a given cache key."""

    _, items, lmax, spectra = key
    param_dict = _restore_dict(items)
    params = _make_camb_params(param_dict, lmax=int(lmax))
    results = camb.get_results(params)
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    out: dict[str, numpy.ndarray] = {}
    if "TT" in spectra:
        out["TT"] = cls[:, 0]
    if "EE" in spectra:
        out["EE"] = cls[:, 1]
    if "TE" in spectra:
        out["TE"] = cls[:, 3]
    return out


@lru_cache(maxsize=128)
def _cached_background(
    key: tuple[str, tuple[tuple[str, Any], ...], tuple[float, ...]],
) -> tuple[
    float,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return cached CAMB background observables for ``key``."""

    _, items, z_tuple = key
    param_dict = _restore_dict(items)
    params = _make_camb_params(param_dict, lmax=None)
    results = camb.get_results(params)
    derived = results.get_derived_params()
    rs_drag = float(derived.get("rdrag", float("nan")))

    z_arr = numpy.asarray(z_tuple, dtype=float)
    comoving_distances: list[float] = []
    angular_distance_values: list[float] = []
    hubble_parameters: list[float] = []
    for z_val in z_arr:
        comoving_distances.append(
            float(results.comoving_radial_distance(float(z_val)))
        )
        angular_distance_values.append(
            float(results.angular_diameter_distance(float(z_val)))
        )
        hubble_parameters.append(float(results.hubble_parameter(float(z_val))))

    comoving_distance_array = numpy.asarray(comoving_distances, dtype=float)
    angular_distance_array = numpy.asarray(
        angular_distance_values, dtype=float
    )
    hubble_parameter_array = numpy.asarray(hubble_parameters, dtype=float)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        hubble_distance_array = numpy.where(
            numpy.abs(hubble_parameter_array) > 1e-12,
            _C_LIGHT_KM_S / hubble_parameter_array,
            numpy.nan,
        )
    term = comoving_distance_array * comoving_distance_array
    term *= z_arr
    with numpy.errstate(divide="ignore", invalid="ignore"):
        term = term * hubble_distance_array
    volume_average_distance_array = numpy.full_like(
        term, numpy.nan, dtype=float
    )
    mask = numpy.isfinite(term) & (term >= 0.0)
    volume_average_distance_array[mask] = numpy.power(term[mask], 1.0 / 3.0)
    zero = numpy.isfinite(term) & (z_arr == 0.0)
    volume_average_distance_array[zero] = 0.0

    return (
        rs_drag,
        tuple(comoving_distance_array.tolist()),
        tuple(hubble_distance_array.tolist()),
        tuple(angular_distance_array.tolist()),
        tuple(volume_average_distance_array.tolist()),
        tuple(hubble_parameter_array.tolist()),
    )


def _compute_cmb_spectrum_direct(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return spectra directly without routing through the scalar cache."""

    ell_arr = numpy.asarray(list(ells), dtype=int)
    if ell_arr.size == 0:
        raise ValueError("ells must not be empty")
    lmax = int(ell_arr.max())
    params = _make_camb_params(contract_or_params, lmax=lmax)
    results = camb.get_results(params)
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    out: dict[str, numpy.ndarray] = {}
    if "TT" in spectra:
        out["TT"] = cls[:, 0]
    if "EE" in spectra:
        out["EE"] = cls[:, 1]
    if "TE" in spectra:
        out["TE"] = cls[:, 3]
    result = {spec: out[spec][ell_arr] for spec in spectra}
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def _compute_camb_background_direct(
    contract_or_params: Mapping[str, Any],
    redshifts: Sequence[float],
) -> dict[str, numpy.ndarray]:
    """Return background observables directly without cached scalars."""

    z_arr = numpy.asarray(redshifts, dtype=float)
    params = _make_camb_params(contract_or_params, lmax=None)
    results = camb.get_results(params)
    derived = results.get_derived_params()
    rs_drag = float(derived.get("rdrag", float("nan")))

    comoving_distances: list[float] = []
    angular_distance_values: list[float] = []
    hubble_parameters: list[float] = []
    for z_val in z_arr:
        comoving_distances.append(
            float(results.comoving_radial_distance(float(z_val)))
        )
        angular_distance_values.append(
            float(results.angular_diameter_distance(float(z_val)))
        )
        hubble_parameters.append(float(results.hubble_parameter(float(z_val))))

    comoving_distance_array = numpy.asarray(comoving_distances, dtype=float)
    angular_distance_array = numpy.asarray(
        angular_distance_values, dtype=float
    )
    hubble_parameter_array = numpy.asarray(hubble_parameters, dtype=float)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        hubble_distance_array = numpy.where(
            numpy.abs(hubble_parameter_array) > 1e-12,
            _C_LIGHT_KM_S / hubble_parameter_array,
            numpy.nan,
        )
    term = comoving_distance_array * comoving_distance_array
    term *= z_arr
    with numpy.errstate(divide="ignore", invalid="ignore"):
        term = term * hubble_distance_array
    volume_average_distance_array = numpy.full_like(
        term, numpy.nan, dtype=float
    )
    mask = numpy.isfinite(term) & (term >= 0.0)
    volume_average_distance_array[mask] = numpy.power(term[mask], 1.0 / 3.0)
    zero = numpy.isfinite(term) & (z_arr == 0.0)
    volume_average_distance_array[zero] = 0.0

    return {
        "rs_drag": rs_drag,
        "DM": comoving_distance_array,
        "DH": hubble_distance_array,
        "DA": angular_distance_array,
        "DV": volume_average_distance_array,
        "Hz": hubble_parameter_array,
        "z": z_arr.copy(),
    }


def _compute_declared_perturbation_spectrum(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    background_payload: Mapping[str, Any] | None = None,
    background_provider: Any | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return spectra from a declared non-standard perturbation contract."""

    del background_payload
    custom_data = _compute_custom_cmb_spectrum_data(
        contract_or_params,
        ells,
        background_provider=background_provider,
    )
    ell_factor = (
        custom_data.ell_grid.astype(float)
        * (custom_data.ell_grid.astype(float) + 1.0)
        / (2.0 * math.pi)
    )
    t_cmb_muK = 2.7255e6
    spectra_results: dict[str, numpy.ndarray] = {
        "TT": ell_factor * custom_data.C_l_TT * t_cmb_muK * t_cmb_muK,
        "TE": ell_factor * custom_data.C_l_TE * t_cmb_muK * t_cmb_muK,
        "EE": ell_factor * custom_data.C_l_EE * t_cmb_muK * t_cmb_muK,
    }
    for spectrum_name, spectrum_values in spectra_results.items():
        if not numpy.all(numpy.isfinite(spectrum_values)):
            raise ValueError(
                "Custom CMB spectrum calculation produced non-finite "
                f"{spectrum_name} values"
            )
    result = {
        spec: spectra_results[spec]
        for spec in spectra
        if spec in spectra_results
    }
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def compute_camb_background_observables(
    contract_or_params: Mapping[str, Any], redshifts: Sequence[float]
) -> dict[str, numpy.ndarray]:
    """Return CAMB background quantities for ``redshifts``.

    The helper shares the same caching layer as the spectrum generator so
    BAO evaluations reuse cosmologies computed for the CMB likelihood.
    """

    if not _is_structured_camb_background_contract(contract_or_params):
        raise ValueError(
            "Structured CAMB background contracts must include backend, "
            "param_map, grids, values and calls"
        )

    return _compute_camb_background_direct(contract_or_params, redshifts)


def describe_camb_configuration() -> dict[str, Any]:
    """Return the default CAMB configuration used by the likelihood helpers."""

    params = camb.CAMBparams()
    accuracy = getattr(params, "Accuracy", None)
    accuracy_info: dict[str, float] = {}
    if accuracy is not None:
        accuracy_info = {
            "AccuracyBoost": float(getattr(accuracy, "AccuracyBoost", 1.0)),
            "LAccuracyBoost": float(getattr(accuracy, "LAccuracyBoost", 1.0)),
            "KAccuracyBoost": float(getattr(accuracy, "KAccuracyBoost", 1.0)),
        }

    return {
        "lmax_padding": _LMAX_PADDING,
        "lens_potential_accuracy": _LENS_POTENTIAL_ACCURACY,
        "reionization_model": "optical_depth_tau",
        # tau-based parameterisation
        "accuracy": accuracy_info,
    }


def compute_cmb_spectrum_from_dict(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using CAMB with caching."""

    if not _is_structured_camb_contract(contract_or_params):
        raise ValueError(
            "Structured CAMB contracts must include perturbations"
        )

    logger = logging.getLogger()
    _validate_camb_perturbation_execution(contract_or_params)
    perturbations = contract_or_params.get("perturbations", {}) or {}
    if (
        isinstance(perturbations, Mapping)
        and perturbations.get("standard") is False
    ):
        return _compute_declared_perturbation_spectrum(
            contract_or_params,
            ells,
            spectra=spectra,
        )

    try:
        return _compute_cmb_spectrum_direct(
            contract_or_params,
            ells,
            spectra=spectra,
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.error("(compute_cmb_spectrum_from_dict): %s", exc)
        raise


def compute_cmb_spectrum_from_legacy_params_for_tests(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Test-only legacy helper that accepts flat CAMB parameter mappings."""

    try:
        ell_arr = numpy.asarray(list(ells), dtype=int)
        if ell_arr.size == 0:
            raise ValueError("ells must not be empty")
        legacy_contract = _normalise_camb_contract(param_dict)
        result = _compute_cmb_spectrum_direct(
            legacy_contract,
            ell_arr,
            spectra=spectra,
        )
        return result
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logging.getLogger().error(
            "(compute_cmb_spectrum_from_legacy_params_for_tests): %s", exc
        )
        raise


def compute_cmb_spectrum_cached(
    plugin: Any,
    cosmo_params: Sequence[float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    get_contract = getattr(plugin, "get_camb_contract", None)
    if not callable(get_contract):
        raise ValueError("Model plugin does not expose a CAMB contract")
    camb_params = get_contract(cosmo_params)
    get_perturbation_contract = getattr(
        plugin, "get_cmb_perturbation_contract", None
    )
    if callable(get_perturbation_contract):
        perturbation_contract = get_perturbation_contract(cosmo_params)
        if perturbation_contract:
            camb_params = _combine_camb_contracts(
                camb_params,
                perturbation_contract,
            )
            perturbations = perturbation_contract.get("standard")
            if perturbations is False:
                return _compute_declared_perturbation_spectrum(
                    camb_params,
                    ells,
                    spectra=spectra,
                    background_provider=plugin,
                )
    return compute_cmb_spectrum_from_dict(camb_params, ells, spectra=spectra)


def compute_cmb_spectrum(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return spectra using a structured CAMB contract."""

    return compute_cmb_spectrum_from_dict(param_dict, ells, spectra=spectra)


@dataclass(slots=True)
class CMBLike(LikelihoodProtocol):
    """Evaluate CMB log-likelihoods for tabulated spectra."""

    cmb_data_df: pandas.DataFrame
    plugin: Any
    extra_params: Mapping[str, float] | None = None
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )
    _ells: numpy.ndarray = field(init=False, repr=False)
    _observed: numpy.ndarray = field(init=False, repr=False)
    _cov_inv: numpy.ndarray | None = field(init=False, repr=False)
    _residual_buffer: numpy.ndarray = field(init=False, repr=False)
    _extra_params_cached: dict[str, float] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Extract immutable arrays so log-likelihood evaluation stays lean."""

        cmb_df = self.cmb_data_df
        if cmb_df is None or cmb_df.empty:
            self._setup_error = "(cmb_like): CMB data is empty."
            self._ells = numpy.empty(0, dtype=int)
            self._observed = numpy.empty(0, dtype=float)
            self._cov_inv = None
            self._residual_buffer = numpy.empty(0, dtype=float)
            return

        self._ells = cmb_df["ell"].to_numpy(dtype=int, copy=True)
        self._observed = cmb_df["Dl_obs"].to_numpy(dtype=float, copy=True)
        if numpy.any(~numpy.isfinite(self._observed)):
            self._setup_error = (
                "(cmb_like): Observed spectrum contains non-finite values."
            )

        cov_attr = cmb_df.attrs.get("covariance_matrix_inv")
        self._cov_inv = (
            None if cov_attr is None else numpy.asarray(cov_attr, dtype=float)
        )
        if self._cov_inv is None:
            self._setup_error = (
                "(cmb_like): Missing inverse covariance matrix."
            )

        self._residual_buffer = numpy.empty_like(self._observed, dtype=float)

        if self.extra_params:
            cached: dict[str, float] = {}
            for param_key, param_value in self.extra_params.items():
                cached[str(param_key)] = float(param_value)
            self._extra_params_cached = cached

    def loglike(self, params: Sequence[float]) -> float:
        """Return the CMB log-likelihood for ``params``."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        if self._setup_error is not None:
            logger.error(self._setup_error)
            self._state = LikelihoodState()
            return float("-inf")

        perturbation_contract: Mapping[str, Any] | None = None
        try:
            camb_contract = self.plugin.get_camb_contract(params)
            get_perturbation_contract = getattr(
                self.plugin,
                "get_cmb_perturbation_contract",
                None,
            )
            if callable(get_perturbation_contract):
                perturbation_contract = get_perturbation_contract(params)
                if perturbation_contract:
                    camb_contract = _combine_camb_contracts(
                        camb_contract,
                        perturbation_contract,
                    )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error("(cmb_like): %s", exc)
            self._state = LikelihoodState()
            return float("-inf")

        if not isinstance(camb_contract, Mapping):
            self._state = LikelihoodState()
            return float("-inf")

        if self._extra_params_cached:
            camb_contract = dict(camb_contract)
            param_map = dict(camb_contract.get("param_map", {}))
            param_map.update(self._extra_params_cached)
            camb_contract["param_map"] = param_map

        try:
            if isinstance(perturbation_contract, Mapping) and (
                perturbation_contract.get("standard") is False
            ):
                theory = _compute_declared_perturbation_spectrum(
                    camb_contract,
                    self._ells,
                    spectra=("TT",),
                    background_provider=self.plugin,
                )
            else:
                theory = compute_cmb_spectrum_from_dict(
                    camb_contract,
                    self._ells,
                    spectra=("TT",),
                )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error("(cmb_like): %s", exc)
            self._state = LikelihoodState()
            return float("-inf")
        if not isinstance(theory, numpy.ndarray):
            theory = numpy.asarray(theory, dtype=float)
        if theory.shape != self._observed.shape or numpy.any(
            ~numpy.isfinite(theory)
        ):
            self._state = LikelihoodState()
            return float("-inf")

        numpy.subtract(
            self._observed,
            theory,
            out=self._residual_buffer,
            casting="unsafe",
        )

        cov_inv = self._cov_inv
        if cov_inv is None:
            self._state = LikelihoodState()
            return float("-inf")

        try:
            chi2 = float(
                self._residual_buffer @ cov_inv @ self._residual_buffer
            )
        except (
            FloatingPointError,
            numpy.linalg.LinAlgError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.error("(cmb_like): Linear algebra failure: %s", exc)
            self._state = LikelihoodState()
            return float("-inf")

        loglike = -0.5 * chi2 if numpy.isfinite(chi2) else float("-inf")
        self._state = LikelihoodState(
            chi2=chi2,
            loglike=loglike,
            metadata={
                "covariance": "full",
                "points": int(self._observed.size),
            },
        )
        return loglike

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostics captured during the last evaluation."""

        return self._state.as_mapping()


__all__ = [
    "CMBLike",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_dict",
    "compute_cmb_spectrum_from_legacy_params_for_tests",
]
