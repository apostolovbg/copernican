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

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

import camb
import numpy
import pandas

from ..engine_adapter import CMB_BACKEND_CAPABILITIES
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


def _is_structured_camb_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract_or_params`` uses the new contract."""

    keys = {str(key) for key in contract_or_params.keys()}
    return bool(
        keys.intersection(
            {
                "backend",
                "calls",
                "grids",
                "param_map",
                "perturbations",
                "values",
            }
        )
    )


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
    backend_capabilities = CMB_BACKEND_CAPABILITIES.get(backend, {})
    if not isinstance(backend_capabilities, Mapping):
        backend_capabilities = {}

    standard = perturbations.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")
    if standard:
        return

    backend_mapping = perturbations.get("backend_mapping", {})
    backend_entry = {}
    if isinstance(backend_mapping, Mapping):
        backend_entry = backend_mapping.get(backend, {}) or {}

    if not backend_capabilities.get("native_nonstandard_perturbations", False):
        raise ValueError(
            "Model "
            f"'{model_name}' declares non-standard perturbations for backend "
            f"'{backend}' (standard={standard}), but the backend capability "
            "registry does not support native non-standard perturbations. "
            "A native backend implementation is required."
        )

    if not isinstance(backend_entry, Mapping) or not backend_entry.get(
        "implemented", False
    ):
        raise ValueError(
            "Model "
            f"'{model_name}' declares non-standard perturbations for backend "
            f"'{backend}' (standard={standard}), but the declared backend "
            "mapping does not mark a native implementation as available. "
            "A native backend implementation is required."
        )


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

    if _fake_cmb_enabled() or _FAKE_CMB_PROVIDER is not None:
        logger = logging.getLogger()
        logger.info(
            "(compute_camb_background_observables): Using synthetic "
            "background observables in lieu of CAMB",
        )
        z_arr = numpy.asarray(redshifts, dtype=float)
        return _fake_background_payload(z_arr)

    if _is_structured_camb_contract(contract_or_params):
        return _compute_camb_background_direct(contract_or_params, redshifts)

    z_arr = numpy.asarray(redshifts, dtype=float)
    z_tuple = tuple(
        float(f"{float(value):.{_CACHE_PRECISION}g}") for value in z_arr
    )
    items = _normalise_items(contract_or_params)
    (
        rs_drag,
        comoving_distance_tuple,
        hubble_distance_tuple,
        angular_distance_tuple,
        volume_average_distance_tuple,
        hubble_parameter_tuple,
    ) = _cached_background(("background", items, z_tuple))
    return {
        "rs_drag": float(rs_drag),
        "DM": numpy.asarray(comoving_distance_tuple, dtype=float),
        "DH": numpy.asarray(hubble_distance_tuple, dtype=float),
        "DA": numpy.asarray(angular_distance_tuple, dtype=float),
        "DV": numpy.asarray(volume_average_distance_tuple, dtype=float),
        "Hz": numpy.asarray(hubble_parameter_tuple, dtype=float),
        "z": numpy.asarray(z_tuple, dtype=float),
    }


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

    logger = logging.getLogger()
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
        if _is_structured_camb_contract(contract_or_params):
            _validate_camb_perturbation_execution(contract_or_params)
            return _compute_cmb_spectrum_direct(
                contract_or_params,
                ells,
                spectra=spectra,
            )

        ell_arr = numpy.asarray(list(ells), dtype=int)
        if ell_arr.size == 0:
            raise ValueError("ells must not be empty")
        items = _normalise_items(contract_or_params)
        lmax = int(ell_arr.max())
        cache_key = ("dict", items, lmax, tuple(sorted(spectra)))
        full = _cached_cmb(cache_key)
        result = {spec: full[spec][ell_arr] for spec in spectra}
        if len(result) == 1:
            return next(iter(result.values()))
        return result
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


def compute_cmb_spectrum_cached(
    plugin: Any,
    cosmo_params: Sequence[float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    get_contract = getattr(plugin, "get_camb_contract", None)
    if callable(get_contract):
        camb_params = get_contract(cosmo_params)
    else:
        camb_params = plugin.get_camb_params(cosmo_params)
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
    return compute_cmb_spectrum_from_dict(camb_params, ells, spectra=spectra)


def compute_cmb_spectrum(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Backward-compatible wrapper accepting a CAMB parameter dictionary."""

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
]
