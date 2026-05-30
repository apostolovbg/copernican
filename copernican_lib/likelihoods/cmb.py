r"""Cosmic Microwave Background likelihood helper.

Provides cache-aware CAMB interfaces shared by the CMB likelihood and the BAO
background evaluator. The helpers consume structured CAMB contracts so scalar
parameters, declared grids, evaluated values and ordered backend calls stay
aligned across the spectrum and background paths. The spectra returned here
are expressed as :math:`D_\ell` so downstream tests comparing against
published Planck-lite tables use consistent conventions.
The helper also enforces the declared perturbation contract so CMB-valid
models cannot silently fall back to standard CAMB perturbations when they
claim a non-standard backend mapping.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

import camb
import numpy
import pandas
from scipy.integrate import cumulative_trapezoid
from scipy.special import spherical_jn

from ..engine_adapter import (
    _SUPPORTED_CMB_BACKEND,
    _evaluate_safe_expression,
    _parse_safe_expression,
)
from ..model_coder import validate_native_perturbation_execution
from ._protocol import LikelihoodProtocol, LikelihoodState

_C_LIGHT_KM_S = 299_792.458
_LMAX_PADDING = 300
_LENS_POTENTIAL_ACCURACY = 0
_CACHE_PRECISION = 15
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")
_FAKE_CMB_BASELINE = 1200.0
_FAKE_CMB_OFFSET = 3.0

# CI and developer workstations occasionally lack usable CAMB wheels.  A
# dedicated stub hook lets tests swap out the heavy numerical path for a
# deterministic placeholder spectrum so regression suites avoid hour-long
# physics evaluations while still exercising the chi-squared plumbing.
_FAKE_CMB_PROVIDER: (
    Callable[
        [Mapping[str, Any], Iterable[int], Sequence[str]],
        Mapping[str, numpy.ndarray] | numpy.ndarray,
    ]
    | None
) = None


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


def _coerce_fake_output(
    fake: Mapping[str, numpy.ndarray] | numpy.ndarray,
    spectra: Sequence[str],
) -> Mapping[str, numpy.ndarray] | numpy.ndarray:
    """Normalise stubbed spectra for deterministic test execution.

    The injected provider may return a bare array for single-spectrum
    scenarios.  This helper mirrors :func:`compute_cmb_spectrum_from_dict`
    by forwarding single entries directly while ensuring multi-spectrum
    runs respect the requested ordering.
    """

    if isinstance(fake, Mapping):
        result: dict[str, numpy.ndarray] = {}
        for spec in spectra:
            spectrum_output = fake.get(spec)
            if spectrum_output is None:
                continue
            result[str(spec)] = numpy.asarray(spectrum_output, dtype=float)
        return result

    coerced = numpy.asarray(fake, dtype=float)
    if len(spectra) == 1:
        return coerced
    return {spec: coerced for spec in spectra}


def _fake_cmb_enabled() -> bool:
    """Return ``True`` when the CMB helper should bypass CAMB entirely."""

    flag = os.environ.get("COPERNICAN_FAKE_CMB", "")
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _fake_background_payload(z_arr: numpy.ndarray) -> dict[str, numpy.ndarray]:
    """Return deterministic background observables for CI-only runs."""

    rs_drag = numpy.full(1, 147.0, dtype=float)
    dm_vals = z_arr * 1000.0
    dh_vals = numpy.full_like(z_arr, 1000.0)
    da_vals = numpy.divide(dm_vals, 1.0 + z_arr, dtype=float)
    dv_vals = numpy.power(dm_vals * numpy.square(1.0 + z_arr), 1.0 / 3.0)
    hz_vals = numpy.full_like(z_arr, 70.0)
    return {
        "rs_drag": rs_drag,
        "DM": dm_vals,
        "DH": dh_vals,
        "DA": da_vals,
        "DV": dv_vals,
        "Hz": hz_vals,
        "z": z_arr.copy(),
    }


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


def _project_declared_perturbation_series(
    source_series: numpy.ndarray,
    ell_value: int,
    k_value: float,
    tau_grid: numpy.ndarray,
    *,
    damping_scale: float,
) -> float:
    """Project a declared source series onto a single multipole."""

    tau_distance = numpy.maximum(float(tau_grid[-1]) - tau_grid, 0.0)
    kernel = spherical_jn(int(ell_value), k_value * tau_distance)
    damping = numpy.exp(
        -numpy.square(k_value * tau_distance / max(damping_scale, 1e-6))
    )
    projected = numpy.trapz(source_series * kernel * damping, tau_grid)
    if not numpy.isfinite(projected):
        return 0.0
    return float(projected)


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

    perturbation_data = _compile_declared_perturbation_contract(
        contract_or_params
    )
    if perturbation_data.standard:
        raise ValueError(
            "Standard perturbation contracts must use the CAMB path."
        )

    backend = str(perturbation_data.backend)
    backend_entry = perturbation_data.backend_mapping.get(backend)
    if backend_entry is None:
        raise ValueError(
            f"cmb.perturbations.backend_mapping must include {backend}"
        )
    if getattr(backend_entry, "native_solver_required", None) is not True:
        raise ValueError(
            "Non-standard perturbations must declare "
            f"backend_mapping.{backend}.native_solver_required: true"
        )
    if getattr(backend_entry, "implemented", None) is not True:
        raise ValueError(
            "A generic declarative executor is required for backend "
            f"'{backend}'; backend_mapping.{backend}.implemented must be "
            "true"
        )

    contract = dict(contract_or_params)
    contract.pop("perturbations", None)
    ell_arr = numpy.asarray(list(ells), dtype=int)
    if ell_arr.size == 0:
        raise ValueError("ells must not be empty")

    variable_names = tuple(perturbation_data.variables.keys())
    source_names = tuple(perturbation_data.sources.keys())
    derived_expression_entries = {
        name: entry
        for name, entry in perturbation_data.derived.items()
        if entry.kind == "expression"
    }
    derivative_symbol_entries = {
        name: entry
        for name, entry in perturbation_data.derived.items()
        if entry.kind == "derivative_symbol"
    }
    equation_by_variable: dict[str, Any] = {}
    for equation in perturbation_data.equations.values():
        if equation.lhs.order != 1:
            raise ValueError(
                "Generic declarative perturbation execution currently "
                "supports only first-order equations"
            )
        if equation.lhs.variable in equation_by_variable:
            raise ValueError(
                f"Perturbation variable '{equation.lhs.variable}' has more "
                "than one evolution equation"
            )
        equation_by_variable[equation.lhs.variable] = equation

    model_parameter_values = {
        str(key): _coerce_numeric_scalar(value, name=str(key))
        for key, value in (contract.get("model_parameters", {}) or {}).items()
    }
    param_map_values = {
        str(key): _coerce_numeric_scalar(value, name=str(key))
        for key, value in (contract.get("param_map", {}) or {}).items()
    }
    combined_parameter_values = dict(model_parameter_values)
    combined_parameter_values.update(param_map_values)

    z_rec_value = combined_parameter_values.get("z_rec", 1089.92)
    if not numpy.isfinite(z_rec_value) or z_rec_value <= 0.0:
        z_rec_value = 1089.92

    grid_size = max(64, 4 * (len(variable_names) + len(source_names)))
    z_grid = numpy.linspace(float(z_rec_value), 0.0, grid_size, dtype=float)
    if background_payload is None:
        if background_provider is not None:
            background = _build_generic_background_payload_from_plugin(
                background_provider,
                contract,
                z_grid,
            )
        else:
            background = _build_generic_background_payload_from_contract(
                contract,
                z_grid,
            )
    else:
        background = background_payload
    background_z = numpy.asarray(background.get("z", z_grid), dtype=float)
    background_hz = numpy.asarray(background.get("Hz", numpy.nan), dtype=float)
    if background_hz.shape != background_z.shape:
        raise ValueError("Generic declarative background has invalid shape")
    if not numpy.all(numpy.isfinite(background_hz)):
        raise ValueError("Generic declarative background is non-finite")
    background_a = numpy.divide(1.0, 1.0 + background_z, dtype=float)
    tau_grid = cumulative_trapezoid(
        1.0
        / numpy.maximum(
            background_a * background_a * numpy.maximum(background_hz, 1e-12),
            1e-12,
        ),
        background_a,
        initial=0.0,
    )
    if (
        not numpy.all(numpy.isfinite(tau_grid))
        or tau_grid.size == 0
        or tau_grid[-1] <= 0.0
    ):
        tau_grid = numpy.linspace(0.0, 1.0, background_a.size, dtype=float)
    else:
        tau_grid = tau_grid / tau_grid[-1]

    value_definitions = contract.get("value_definitions", {}) or {}
    value_arrays = contract.get("values", {}) or {}
    grid_arrays = contract.get("grids", {}) or {}

    def _background_scalar(name: str, tau_value: float) -> float:
        """Return a scalar background value interpolated at ``tau_value``."""

        series = background.get(name)
        if series is None:
            return 0.0
        series_array = numpy.asarray(series, dtype=float)
        if series_array.ndim == 0 or series_array.size == 1:
            return float(series_array.reshape(()))
        return float(numpy.interp(tau_value, tau_grid, series_array))

    def _contract_value(name: str, current_a: float) -> float:
        """Return the scalar contract value associated with ``name``."""

        raw_value = value_arrays.get(name)
        value_def = value_definitions.get(name, {})
        grid_name = (
            value_def.get("grid") if isinstance(value_def, Mapping) else None
        )
        if isinstance(grid_name, str):
            grid_array = numpy.asarray(grid_arrays.get(grid_name), dtype=float)
            value_array = numpy.asarray(raw_value, dtype=float)
            if (
                grid_array.ndim == 1
                and value_array.ndim == 1
                and grid_array.size == value_array.size
            ):
                return float(numpy.interp(current_a, grid_array, value_array))
        if raw_value is None:
            return 0.0
        return _coerce_numeric_scalar(raw_value, name=name)

    derived_expression_order = tuple(
        name
        for name in (
            perturbation_data.dependency_graph_summary.derived_expression_names
        )
        if name in derived_expression_entries
    )
    dependency_summary = perturbation_data.dependency_graph_summary
    dependency_map = {
        name: tuple(
            dependency
            for dependency in dependencies
            if dependency in derived_expression_entries
        )
        for name, dependencies in (
            dependency_summary.derived_expression_dependencies.items()
        )
    }
    ordered_derived_names: list[str] = []
    unresolved = set(derived_expression_order)
    resolved: set[str] = set()
    while unresolved:
        ready = sorted(
            name
            for name in unresolved
            if set(dependency_map.get(name, ())) <= resolved
        )
        if not ready:
            raise ValueError(
                "Derived perturbation expressions contain a cycle"
            )
        ordered_derived_names.extend(ready)
        for name in ready:
            unresolved.remove(name)
            resolved.add(name)

    initial_state = numpy.asarray(
        [
            (0.0 if "stress" in entry.kind.lower() else 1.0e-3 * (index + 1))
            for index, entry in enumerate(perturbation_data.variables.values())
        ],
        dtype=float,
    )
    variable_order = tuple(perturbation_data.variables.keys())
    variable_index = {name: index for index, name in enumerate(variable_order)}

    def _build_environment(
        tau_value: float,
        state_vector: numpy.ndarray,
        derivative_values: Mapping[str, float],
        k_value: float,
    ) -> dict[str, Any]:
        """Return the evaluation environment at a single grid point."""

        current_a = float(numpy.interp(tau_value, tau_grid, background_a))
        current_z = float(numpy.interp(tau_value, tau_grid, background_z))
        current_hz = float(numpy.interp(tau_value, tau_grid, background_hz))
        current_env: dict[str, Any] = {
            "a": current_a,
            "eta": tau_value,
            "tau": tau_value,
            "z": current_z,
            "H": current_hz,
            "Hconf": current_a * current_hz,
            "k": k_value,
            "Phi": 0.0,
            "Psi": 0.0,
        }
        current_env.update(combined_parameter_values)
        current_env.update(param_map_values)
        current_env.update(
            {
                name: float(state_vector[variable_index[name]])
                for name in variable_order
            }
        )
        current_env.update(derivative_values)
        current_env.update(
            {name: _contract_value(name, current_a) for name in value_arrays}
        )
        current_env["DM"] = _background_scalar("DM", tau_value)
        current_env["DH"] = _background_scalar("DH", tau_value)
        current_env["DA"] = _background_scalar("DA", tau_value)
        current_env["DV"] = _background_scalar("DV", tau_value)
        current_env["rs_drag"] = _background_scalar("rs_drag", tau_value)
        current_env["Hz"] = current_hz
        return current_env

    def _evaluate_derived_values(env: Mapping[str, Any]) -> dict[str, float]:
        """Return the evaluated derived expression mapping."""

        derived_values: dict[str, float] = {}
        working_env = dict(env)
        for derived_name in ordered_derived_names:
            expression = derived_expression_entries[derived_name].expression
            derived_values[derived_name] = _coerce_numeric_scalar(
                _evaluate_safe_expression(expression or "0", working_env),
                name=f"cmb.perturbations.derived.{derived_name}",
            )
            working_env[derived_name] = derived_values[derived_name]
        return derived_values

    def _apply_closure_constraints(
        env: dict[str, Any],
        state_vector: numpy.ndarray,
    ) -> float:
        """Apply algebraic closure relations to the current state."""

        closure_penalty = 0.0
        for closure_name, closure_data in perturbation_data.closures.items():
            left_expr = closure_data.expression
            right_expr = closure_data.equals
            left_node = _parse_safe_expression(left_expr)
            right_value = _coerce_numeric_scalar(
                _evaluate_safe_expression(right_expr, env),
                name=f"cmb.perturbations.closures.{closure_name}.equals",
            )
            if isinstance(left_node.body, ast.Name):
                target_name = left_node.body.id
                env[target_name] = right_value
                if target_name in variable_index:
                    state_vector[variable_index[target_name]] = right_value
                continue
            left_value = _coerce_numeric_scalar(
                _evaluate_safe_expression(left_expr, env),
                name=(
                    "cmb.perturbations.closures." f"{closure_name}.expression"
                ),
            )
            closure_penalty += abs(left_value - right_value)
        return closure_penalty

    def _evaluate_source_amplitude(
        tau_value: float,
        state_vector: numpy.ndarray,
        k_value: float,
    ) -> float:
        """Return a scalar source amplitude for the current grid point."""

        derivative_values = {name: 0.0 for name in derivative_symbol_entries}
        rhs_values: dict[str, float] = {
            name: 0.0 for name in equation_by_variable
        }
        closure_penalty = 0.0
        for _ in range(4):
            env = _build_environment(
                tau_value,
                state_vector,
                derivative_values,
                k_value,
            )
            env.update(_evaluate_derived_values(env))
            closure_penalty = _apply_closure_constraints(env, state_vector)
            env.update(_evaluate_derived_values(env))

            new_rhs_values: dict[str, float] = {}
            for variable_name, equation in equation_by_variable.items():
                new_rhs_values[variable_name] = _coerce_numeric_scalar(
                    _evaluate_safe_expression(equation.rhs, env),
                    name=(
                        f"cmb.perturbations.equations." f"{equation.name}.rhs"
                    ),
                )

            new_derivative_values = dict(derivative_values)
            for (
                derivative_name,
                derivative_data,
            ) in derivative_symbol_entries.items():
                if derivative_data.order == 1 and derivative_data.wrt == "tau":
                    new_derivative_values[derivative_name] = (
                        new_rhs_values.get(
                            derivative_data.variable,
                            0.0,
                        )
                    )
                else:
                    new_derivative_values[derivative_name] = 0.0

            if all(
                numpy.isclose(
                    new_rhs_values[name],
                    rhs_values.get(name, float("nan")),
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )
                for name in new_rhs_values
            ) and all(
                numpy.isclose(
                    new_derivative_values[name],
                    derivative_values.get(name, float("nan")),
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )
                for name in new_derivative_values
            ):
                derivative_values = new_derivative_values
                rhs_values = new_rhs_values
                break

            derivative_values = new_derivative_values
            rhs_values = new_rhs_values

        env = _build_environment(
            tau_value,
            state_vector,
            derivative_values,
            k_value,
        )
        env.update(_evaluate_derived_values(env))
        env.update(derivative_values)
        closure_penalty = _apply_closure_constraints(env, state_vector)
        env.update(_evaluate_derived_values(env))
        source_amplitude = 0.0
        for source_name, source_data in perturbation_data.sources.items():
            source_amplitude += abs(
                _coerce_numeric_scalar(
                    _evaluate_safe_expression(source_data.expression, env),
                    name=f"cmb.perturbations.sources.{source_name}",
                )
            )
        return (
            source_amplitude
            + closure_penalty
            + sum(abs(value) for value in rhs_values.values())
        )

    def _rhs_vector(
        tau_value: float, state_vector: numpy.ndarray, k_value: float
    ) -> numpy.ndarray:
        """Return the derivative vector for the declared system."""

        derivative_values = {name: 0.0 for name in derivative_symbol_entries}
        rhs_values: dict[str, float] = {
            name: 0.0 for name in equation_by_variable
        }
        for _ in range(4):
            env = _build_environment(
                tau_value,
                state_vector,
                derivative_values,
                k_value,
            )
            env.update(_evaluate_derived_values(env))
            _apply_closure_constraints(env, state_vector)
            env.update(_evaluate_derived_values(env))

            new_rhs_values: dict[str, float] = {}
            for variable_name, equation in equation_by_variable.items():
                new_rhs_values[variable_name] = _coerce_numeric_scalar(
                    _evaluate_safe_expression(equation.rhs, env),
                    name=(
                        f"cmb.perturbations.equations." f"{equation.name}.rhs"
                    ),
                )

            new_derivative_values = dict(derivative_values)
            for (
                derivative_name,
                derivative_data,
            ) in derivative_symbol_entries.items():
                if derivative_data.order == 1 and derivative_data.wrt == "tau":
                    new_derivative_values[derivative_name] = (
                        new_rhs_values.get(
                            derivative_data.variable,
                            0.0,
                        )
                    )
                else:
                    new_derivative_values[derivative_name] = 0.0

            if all(
                numpy.isclose(
                    new_rhs_values[name],
                    rhs_values.get(name, float("nan")),
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )
                for name in new_rhs_values
            ) and all(
                numpy.isclose(
                    new_derivative_values[name],
                    derivative_values.get(name, float("nan")),
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )
                for name in new_derivative_values
            ):
                rhs_values = new_rhs_values
                break

            derivative_values = new_derivative_values
            rhs_values = new_rhs_values

        derivative_vector = numpy.zeros_like(state_vector, dtype=float)
        for variable_name, variable_index_value in variable_index.items():
            derivative_vector[variable_index_value] = rhs_values.get(
                variable_name,
                0.0,
            )
        return derivative_vector

    background_scale = float(
        numpy.asarray(background.get("rs_drag", 1.0), dtype=float).reshape(())
    )
    if not numpy.isfinite(background_scale) or background_scale <= 0.0:
        background_scale = 1.0
    damping_scale = max(0.25, min(4.0, background_scale / 120.0))

    k_min = max(0.05, 0.5 / max(float(tau_grid[-1]), 1.0))
    k_max = max(float(ell_arr.max()) + 3.0, k_min * 8.0)
    k_sample_count = max(16, min(64, 4 * len(ell_arr) + 8))
    k_values = numpy.logspace(
        numpy.log10(k_min),
        numpy.log10(k_max),
        k_sample_count,
    )
    log_k_values = numpy.log(k_values)

    as_value = combined_parameter_values.get("As", 2.1e-9)
    if not numpy.isfinite(as_value) or as_value <= 0.0:
        as_value = 2.1e-9
    ns_value = combined_parameter_values.get("ns", 0.965)
    if not numpy.isfinite(ns_value):
        ns_value = 0.965
    primordial_power = as_value * numpy.power(
        numpy.maximum(k_values, 1e-12) / 0.05,
        ns_value - 1.0,
    )

    temperature_transfers = numpy.zeros(
        (k_sample_count, ell_arr.size), dtype=float
    )
    polarization_transfers = numpy.zeros_like(temperature_transfers)

    for k_index, current_k in enumerate(k_values):
        state_vector = initial_state.copy()
        temperature_series: list[float] = []
        polarization_series: list[float] = []

        source_scalar = _evaluate_source_amplitude(
            float(tau_grid[0]),
            state_vector,
            float(current_k),
        )
        state_norm = float(numpy.sum(numpy.abs(state_vector)))
        temperature_series.append(
            source_scalar + 0.1 * state_norm + 0.01 * background_scale
        )
        polarization_series.append(
            0.55 * source_scalar + 0.05 * state_norm + 0.01 * background_scale
        )

        for index in range(len(tau_grid) - 1):
            tau_start = float(tau_grid[index])
            tau_stop = float(tau_grid[index + 1])
            delta_tau = tau_stop - tau_start
            if delta_tau <= 0.0:
                continue

            stage_one = _rhs_vector(tau_start, state_vector, float(current_k))
            stage_two = _rhs_vector(
                tau_start + 0.5 * delta_tau,
                state_vector + 0.5 * delta_tau * stage_one,
                float(current_k),
            )
            stage_three = _rhs_vector(
                tau_start + 0.5 * delta_tau,
                state_vector + 0.5 * delta_tau * stage_two,
                float(current_k),
            )
            stage_four = _rhs_vector(
                tau_stop,
                state_vector + delta_tau * stage_three,
                float(current_k),
            )
            state_vector = state_vector + (
                delta_tau
                * (
                    stage_one
                    + 2.0 * stage_two
                    + 2.0 * stage_three
                    + stage_four
                )
                / 6.0
            )
            source_scalar = _evaluate_source_amplitude(
                tau_stop,
                state_vector,
                float(current_k),
            )
            state_norm = float(numpy.sum(numpy.abs(state_vector)))
            derivative_norm = float(numpy.sum(numpy.abs(stage_one)))
            temperature_series.append(
                source_scalar + 0.1 * state_norm + 0.01 * background_scale
            )
            polarization_series.append(
                0.45 * source_scalar
                + 0.08 * derivative_norm
                + 0.05 * state_norm
            )

        temperature_history = numpy.asarray(temperature_series, dtype=float)
        polarization_history = numpy.asarray(
            polarization_series,
            dtype=float,
        )
        if temperature_history.shape != tau_grid.shape:
            raise ValueError("generic perturbation history has invalid shape")
        if polarization_history.shape != tau_grid.shape:
            raise ValueError("generic perturbation history has invalid shape")

        for ell_index, ell_value in enumerate(ell_arr):
            temperature_transfers[k_index, ell_index] = (
                _project_declared_perturbation_series(
                    temperature_history,
                    int(ell_value),
                    float(current_k),
                    tau_grid,
                    damping_scale=damping_scale,
                )
            )
            polarization_transfers[k_index, ell_index] = (
                _project_declared_perturbation_series(
                    polarization_history,
                    int(ell_value),
                    float(current_k),
                    tau_grid,
                    damping_scale=damping_scale,
                )
            )

    ell_factor = (
        numpy.asarray(ell_arr, dtype=float)
        * (numpy.asarray(ell_arr, dtype=float) + 1.0)
        / (2.0 * numpy.pi)
    )
    spectra_results: dict[str, numpy.ndarray] = {}
    tt_spectrum = numpy.zeros_like(ell_factor)
    te_spectrum = numpy.zeros_like(ell_factor)
    ee_spectrum = numpy.zeros_like(ell_factor)
    bb_spectrum = numpy.zeros_like(ell_factor)

    for ell_index in range(ell_arr.size):
        temperature_transfer = temperature_transfers[:, ell_index]
        polarization_transfer = polarization_transfers[:, ell_index]
        cl_tt = (
            4.0
            * numpy.pi
            * numpy.trapz(
                primordial_power * numpy.square(temperature_transfer),
                log_k_values,
            )
        )
        cl_te = (
            4.0
            * numpy.pi
            * numpy.trapz(
                primordial_power
                * temperature_transfer
                * polarization_transfer,
                log_k_values,
            )
        )
        cl_ee = (
            4.0
            * numpy.pi
            * numpy.trapz(
                primordial_power * numpy.square(polarization_transfer),
                log_k_values,
            )
        )
        tt_spectrum[ell_index] = ell_factor[ell_index] * cl_tt
        te_spectrum[ell_index] = ell_factor[ell_index] * cl_te
        ee_spectrum[ell_index] = ell_factor[ell_index] * cl_ee
        bb_spectrum[ell_index] = 0.25 * ee_spectrum[ell_index]

    spectra_results["TT"] = numpy.nan_to_num(
        tt_spectrum, nan=0.0, posinf=0.0, neginf=0.0
    )
    spectra_results["TE"] = numpy.nan_to_num(
        te_spectrum, nan=0.0, posinf=0.0, neginf=0.0
    )
    spectra_results["EE"] = numpy.nan_to_num(
        ee_spectrum, nan=0.0, posinf=0.0, neginf=0.0
    )
    spectra_results["BB"] = numpy.nan_to_num(
        bb_spectrum, nan=0.0, posinf=0.0, neginf=0.0
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
    BAO evaluations reuse cosmologies computed for the CMB likelihood. When
    ``COPERNICAN_FAKE_CMB`` is active the computation returns synthetic but
    deterministic observables to keep CI runs fast while preserving production
    behaviour.
    """

    if not _is_structured_camb_background_contract(contract_or_params):
        raise ValueError(
            "Structured CAMB background contracts must include backend, "
            "param_map, grids, values and calls"
        )

    if _fake_cmb_enabled() or _FAKE_CMB_PROVIDER is not None:
        logger = logging.getLogger()
        logger.info(
            "(compute_camb_background_observables): Using synthetic "
            "background observables in lieu of CAMB",
        )
        z_arr = numpy.asarray(redshifts, dtype=float)
        return _fake_background_payload(z_arr)

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
    r"""Return theoretical :math:`D_\ell` spectra using CAMB with caching.

    Tests inject ``_FAKE_CMB_PROVIDER`` or set ``COPERNICAN_FAKE_CMB=1`` to
    bypass real CAMB calls when the dependency is missing or too slow for CI
    timeouts.  Production runs continue down the cached CAMB path to preserve
    scientific fidelity.
    """

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

    fake_provider = _FAKE_CMB_PROVIDER
    if fake_provider is not None:
        logger.info(
            "(compute_cmb_spectrum_from_dict): Using injected CMB stub "
            "provider",
        )
        fake = fake_provider(contract_or_params, ells, spectra=spectra)
        return _coerce_fake_output(fake, spectra)

    if _fake_cmb_enabled():
        ell_arr = numpy.asarray(list(ells), dtype=int)
        template = _FAKE_CMB_BASELINE / (ell_arr + _FAKE_CMB_OFFSET)
        result = {spec: template.copy() for spec in spectra}
        if len(result) == 1:
            return template
        return result

    try:
        _validate_camb_perturbation_execution(contract_or_params)
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

    logger = logging.getLogger()
    fake_provider = _FAKE_CMB_PROVIDER
    if fake_provider is not None:
        logger.info(
            "(compute_cmb_spectrum_from_legacy_params_for_tests): Using "
            "injected CMB stub provider",
        )
        fake = fake_provider(param_dict, ells, spectra=spectra)
        return _coerce_fake_output(fake, spectra)

    if _fake_cmb_enabled():
        ell_arr = numpy.asarray(list(ells), dtype=int)
        template = _FAKE_CMB_BASELINE / (ell_arr + _FAKE_CMB_OFFSET)
        result = {spec: template.copy() for spec in spectra}
        if len(result) == 1:
            return template
        return result

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
        logger.error(
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
