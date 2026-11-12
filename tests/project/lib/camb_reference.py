"""Independent CAMB reference helpers for scientific tests.

Nothing in the production package imports this module. Its calculations
provide an external comparison surface for declared solver acceptance tests.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import camb
import numpy

_C_LIGHT_KM_S = 299_792.458
_LMAX_PADDING = 300
_LENS_POTENTIAL_ACCURACY = 0
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")
CAMB_REFERENCE_IDENTITY = f"camb:{camb.__version__}"


def experimental_reference_identity(source_digest: str) -> str:
    """Encode the complete SHA-256 source identity compactly and losslessly."""
    raw = bytes.fromhex(source_digest)
    if len(raw) != 32:
        raise ValueError("Experimental source digest must be SHA-256")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"camb:experimental:{camb.__version__}:{encoded}"


# This contract is intentionally frozen in the test-owned reference surface.
# It mirrors the bundled LambdaCDM initial point but never enters production
# CCMBS code or becomes a backend fallback.
FIXED_LCDM_REFERENCE_CONTRACT = {
    "backend": "camb",
    "param_map": {
        "H0": 75.0,
        "ombh2": 0.0309375,
        "omch2": 0.18,
        "tau": 0.054,
        "As": 2.1e-9,
        "ns": 0.965,
        "Neff": 3.0,
        "mnu": 0.0,
        "num_massive_neutrinos": 0,
        "YHe": 0.245,
    },
    "grids": {},
    "values": {},
    "calls": [],
}
FIXED_LCDM_REFERENCE_ELL_VALUES = (2, 20, 100, 200, 500, 1000, 1500, 2000)
FIXED_LCDM_REFERENCE_SPECTRA = ("TT", "TE", "EE")
FIXED_LCDM_FULL_REFERENCE_SPECTRA = (
    "TT",
    "TE",
    "EE",
    "BB",
    "PP",
    "TP",
    "EP",
    "lensed_TT",
    "lensed_TE",
    "lensed_EE",
    "lensed_BB",
)
FIXED_LCDM_REFERENCE_TOLERANCES = {
    "TT": 0.02,
    "TE": 0.03,
    "EE": 0.02,
}
FIXED_LCDM_FULL_REFERENCE_TOLERANCES = {
    "TT": 0.02,
    "TE": 0.03,
    "EE": 0.02,
    "BB": 0.02,
    "PP": 0.03,
    "TP": 0.05,
    "EP": 0.05,
    "lensed_TT": 0.02,
    "lensed_TE": 0.03,
    "lensed_EE": 0.02,
    "lensed_BB": 0.05,
}
CAMB_PARITY_ELL_VALUES = (
    2,
    20,
    100,
    200,
    500,
    800,
    1200,
    1500,
    2000,
    2500,
)
CAMB_PARITY_SPECTRA = FIXED_LCDM_FULL_REFERENCE_SPECTRA
CAMB_PARITY_RELATIVE_TOLERANCES = FIXED_LCDM_FULL_REFERENCE_TOLERANCES
FULL_LCDM_REFERENCE_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "camb_lcdm_reference.json"
)
LCDM_HISTORY_REFERENCE_FIXTURE_PATH = (
    FULL_LCDM_REFERENCE_FIXTURE_PATH.with_name("camb_lcdm_histories.json")
)
LCDM_HISTORY_ETA_VALUES = (
    1.0,
    2.0,
    5.0,
    10.0,
    50.0,
    100.0,
    200.0,
    300.0,
    500.0,
    1000.0,
)
LCDM_HISTORY_K_VALUES = (0.0001, 0.02)
LCDM_HISTORY_VARIABLES = (
    "delta_baryon",
    "delta_photon",
    "delta_neutrino",
    "pi_photon",
    "E_2",
    "E_source",
    "a",
    "H",
)
_CMB_SPECTRUM_COLUMNS = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
_LENSING_SPECTRUM_COLUMNS = {"PP": 0, "TP": 1, "EP": 2}


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


def _is_structured_camb_background_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract_or_params`` uses the CAMB adapter."""

    keys = {str(key) for key in contract_or_params.keys()}
    required = {"backend", "calls", "grids", "param_map", "values"}
    return required.issubset(keys)


def _make_camb_params(
    contract_or_params: Mapping[str, Any], *, lmax: int | None = None
) -> camb.CAMBparams:
    """Return CAMB parameters from a structured reference contract."""

    contract = contract_or_params
    if not _is_structured_camb_background_contract(contract):
        raise ValueError(
            "Structured CAMB reference contracts must include backend, "
            "param_map, grids, values and calls"
        )
    if contract.get("backend") != "camb":
        raise ValueError("Only the CAMB backend is supported")

    param_map = contract.get("param_map", {})
    if not isinstance(param_map, Mapping):
        raise ValueError("cmb.param_map must be a mapping")

    params = camb.CAMBparams()
    model_kwargs: dict[str, Any] = {}
    consumed_keys: set[str] = set()

    def _use_scalar(key: str) -> float:
        """Return one scalar CAMB parameter and mark it as consumed."""

        value = _coerce_numeric_scalar(param_map[key], name=key)
        consumed_keys.add(key)
        return value

    if "H0" in param_map:
        model_kwargs["H0"] = _use_scalar("H0")
    if "ombh2" in param_map:
        model_kwargs["ombh2"] = _use_scalar("ombh2")
    if "omch2" in param_map:
        model_kwargs["omch2"] = _use_scalar("omch2")
    if "omk" in param_map:
        model_kwargs["omk"] = _use_scalar("omk")
    if "tau" in param_map:
        model_kwargs["tau"] = _use_scalar("tau")
    if "YHe" in param_map:
        model_kwargs["YHe"] = _use_scalar("YHe")
    if "TCMB" in param_map:
        model_kwargs["TCMB"] = _use_scalar("TCMB")
    if "theta_H0_range" in param_map:
        theta_range = _coerce_numeric_array(
            param_map["theta_H0_range"], name="theta_H0_range"
        )
        if theta_range.size < 2:
            raise ValueError("theta_H0_range must contain at least two values")
        model_kwargs["theta_H0_range"] = tuple(
            float(value) for value in theta_range[:2]
        )
        consumed_keys.add("theta_H0_range")

    if "Neff" in param_map:
        model_kwargs["nnu"] = _use_scalar("Neff")
    if "standard_neutrino_neff" in param_map:
        model_kwargs["standard_neutrino_neff"] = _use_scalar(
            "standard_neutrino_neff"
        )
    if "num_massive_neutrinos" in param_map:
        model_kwargs["num_massive_neutrinos"] = int(
            _use_scalar("num_massive_neutrinos")
        )
    if "neutrino_hierarchy" in param_map:
        model_kwargs["neutrino_hierarchy"] = param_map["neutrino_hierarchy"]
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
        model_kwargs.setdefault("num_massive_neutrinos", len(masses))
        model_kwargs["mnu"] = float(numpy.sum(masses))
        consumed_keys.update(ordered)
    if "sum_mnu" in param_map:
        if "mnu" in param_map:
            consumed_keys.add("mnu")
        model_kwargs["mnu"] = _use_scalar("sum_mnu")
    elif "mnu" in param_map:
        model_kwargs["mnu"] = _use_scalar("mnu")

    if "Alens" in param_map:
        model_kwargs["Alens"] = _use_scalar("Alens")

    params.set_cosmology(**model_kwargs)

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


def resolve_camb_parameters(
    contract_or_params: Mapping[str, Any],
) -> dict[str, float | int]:
    """Return CAMB's resolved physical parameters after defaults apply."""

    params = _make_camb_params(contract_or_params)
    return {
        "H0": float(params.H0),
        "ombh2": float(params.ombh2),
        "omch2": float(params.omch2),
        "omk": float(params.omk),
        "YHe": float(params.YHe),
        "tau": float(params.Reion.optical_depth),
        "As": float(params.InitPower.As),
        "ns": float(params.InitPower.ns),
        "num_nu_massless": float(params.num_nu_massless),
        "num_nu_massive": int(params.num_nu_massive),
        "omnuh2": float(params.omnuh2),
    }


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
    lensed_cls = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    lensing_cls = results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK")
    out: dict[str, numpy.ndarray] = {}
    for spectrum_name in spectra:
        name = str(spectrum_name)
        if name in _CMB_SPECTRUM_COLUMNS:
            out[name] = cls[:, _CMB_SPECTRUM_COLUMNS[name]]
        elif name.startswith("lensed_") and name[7:] in _CMB_SPECTRUM_COLUMNS:
            out[name] = lensed_cls[:, _CMB_SPECTRUM_COLUMNS[name[7:]]]
        elif name in _LENSING_SPECTRUM_COLUMNS:
            out[name] = lensing_cls[:, _LENSING_SPECTRUM_COLUMNS[name]]
        else:
            raise ValueError(f"Unsupported CAMB reference spectrum: {name}")
    result = {spec: out[str(spec)][ell_arr] for spec in spectra}
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def compute_camb_scalar_time_evolution(
    contract_or_params: Mapping[str, Any],
    k_value: float,
    eta_values: Sequence[float] | numpy.ndarray,
    *,
    variables: Sequence[str] = (
        "delta_cdm",
        "delta_baryon",
        "delta_photon",
        "delta_neutrino",
        "v_newtonian_baryon",
        "v_photon",
        "pi_photon",
        "E_2",
        "Weyl",
        "T_source",
        "E_source",
    ),
    frame: str = "Newtonian",
    l_accuracy_boost: int = 4,
    accuracy_boost: float = 1.0,
) -> dict[str, numpy.ndarray]:
    """Return independent CAMB scalar histories on the requested eta grid.

    Native named density outputs retain CAMB's CDM-comoving synchronous
    gauge. ``frame`` controls symbolic-expression transformations only.
    """

    k_scalar = _coerce_numeric_scalar(k_value, name="k_value")
    if k_scalar <= 0.0:
        raise ValueError("k_value must be positive")
    eta_array = _coerce_numeric_array(eta_values, name="eta_values")
    variable_names = tuple(str(name) for name in variables)
    if not variable_names:
        raise ValueError("variables must not be empty")
    if int(l_accuracy_boost) < 1:
        raise ValueError("l_accuracy_boost must be positive")
    accuracy = _coerce_numeric_scalar(accuracy_boost, name="accuracy_boost")
    if accuracy < 1.0:
        raise ValueError("accuracy_boost must be at least one")
    params = _make_camb_params(contract_or_params, lmax=32)
    params.Accuracy.AccuracyBoost = accuracy
    # Time evolution needs thermal/background data, not angular spectra.
    results = camb.get_background(params)
    conformal_age = _coerce_numeric_scalar(
        results.conformal_time(0.0), name="CAMB conformal_age"
    )
    if numpy.any(eta_array <= 0.0) or numpy.any(eta_array > conformal_age):
        raise ValueError(
            "eta_values must lie in the open CAMB conformal-time domain "
            f"(0, {conformal_age:.12g}]"
        )
    values = numpy.asarray(
        results.get_time_evolution(
            k_scalar,
            eta_array,
            vars=list(variable_names),
            lAccuracyBoost=int(l_accuracy_boost),
            frame=str(frame),
        ),
        dtype=float,
    )
    if values.ndim == 1:
        values = values[numpy.newaxis, :]
    expected_shape = (eta_array.size, len(variable_names))
    if values.shape != expected_shape:
        raise ValueError(
            "CAMB scalar time evolution returned shape "
            f"{values.shape}; expected {expected_shape}"
        )
    if not numpy.all(numpy.isfinite(values)):
        raise ValueError("CAMB scalar time evolution returned non-finite data")
    return {
        name: numpy.asarray(values[:, index], dtype=float)
        for index, name in enumerate(variable_names)
    }


def camb_histories_in_newtonian_gauge(
    native: Mapping[str, Any],
    k_value: float,
    *,
    curvature_sign: int = 1,
) -> dict[str, numpy.ndarray]:
    """Transform native CAMB histories without compiling custom Fortran.

    CAMB 1.6.0 equations.f90 exports V_c=-k*sigma/Hconf and
    Weyl=k^2*(Phi+Psi)/2. Its synchronous densities transform by
    delta_N=delta_sync-3*(1+w)*Hconf*sigma/k. Photon brightness and
    polarization quadrupoles are four times temperature multipoles.
    Use curvature_sign=-1 for the declared seed Phi-Theta0=+1; CAMB's
    native unit-curvature seed has Phi-Theta0=-1. This fixed convention
    change does not fit amplitudes or phases to the computed histories.
    """

    k = _coerce_numeric_scalar(k_value, name="k_value")
    if k <= 0.0 or curvature_sign not in (-1, 1):
        raise ValueError("Newtonian histories require k>0 and sign +/-1")
    fields = {
        name: _coerce_numeric_array(native[name], name=name)
        for name in (
            "H",
            "etak",
            "Weyl",
            "v_newtonian_cdm",
            "v_newtonian_baryon",
            "delta_cdm",
            "delta_baryon",
            "delta_photon",
            "delta_neutrino",
            "pi_photon",
            "E_2",
        )
    }
    if len({array.shape for array in fields.values()}) != 1:
        raise ValueError("Native CAMB history grids must match")
    hubble = fields["H"]
    shift = (hubble / k) ** 2 * fields["v_newtonian_cdm"]
    phi = fields["etak"] / k + shift
    histories = {
        "Phi": phi,
        "Psi": 2.0 * fields["Weyl"] / k**2 - phi,
        "delta_b": fields["delta_baryon"] + 3.0 * shift,
        "delta_c": fields["delta_cdm"] + 3.0 * shift,
        "delta_nu": fields["delta_neutrino"] + 4.0 * shift,
        "theta_gamma0": fields["delta_photon"] / 4.0 + shift,
        "theta_b": -hubble * fields["v_newtonian_baryon"],
        "theta_c": -hubble * fields["v_newtonian_cdm"],
        "theta_gamma2": fields["pi_photon"] / 4.0,
        "e_gamma2": fields["E_2"] / 4.0,
    }
    return {
        name: float(curvature_sign) * values
        for name, values in histories.items()
    }


def compare_scalar_history_waves(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    eta_values: Sequence[float] | numpy.ndarray,
    *,
    variable_map: Mapping[str, str] | None = None,
    k_value: float | None = None,
) -> dict[str, Any]:
    """Compare same-grid scalar histories without changing their units.

    The report deliberately separates structural wave evidence from parity.
    It records zero crossings, centered correlation, best-fit amplitude, and
    a sample-grid phase lag so a later solver slice can locate the first
    physical disagreement without turning this diagnostic into a fallback.
    """

    eta_array = _coerce_numeric_array(eta_values, name="eta_values")
    if eta_array.size < 3 or numpy.any(numpy.diff(eta_array) <= 0.0):
        raise ValueError(
            "eta_values must contain at least three increasing points"
        )
    if variable_map is None:
        variable_map = {str(name): str(name) for name in actual}
    if not variable_map:
        raise ValueError("variable_map must not be empty")

    def _zero_crossings(values: numpy.ndarray) -> int:
        signs = numpy.signbit(values)
        exact_zero = values == 0.0
        return int(numpy.count_nonzero(signs[1:] != signs[:-1])) + int(
            numpy.count_nonzero(exact_zero)
        )

    eta_step = float(numpy.median(numpy.diff(eta_array)))
    variables: dict[str, dict[str, Any]] = {}
    for actual_name, reference_name in variable_map.items():
        if actual_name not in actual:
            raise KeyError(f"Missing actual history '{actual_name}'")
        if reference_name not in reference:
            raise KeyError(f"Missing reference history '{reference_name}'")
        actual_values = _coerce_numeric_array(
            actual[actual_name], name=f"actual.{actual_name}"
        )
        reference_values = _coerce_numeric_array(
            reference[reference_name], name=f"reference.{reference_name}"
        )
        if actual_values.shape != eta_array.shape:
            raise ValueError(
                f"Actual history '{actual_name}' shape does not match eta"
            )
        if reference_values.shape != eta_array.shape:
            raise ValueError(
                f"Reference history '{reference_name}' shape does not "
                "match eta"
            )
        actual_centered = actual_values - numpy.mean(actual_values)
        reference_centered = reference_values - numpy.mean(reference_values)
        actual_norm = float(numpy.linalg.norm(actual_centered))
        reference_norm = float(numpy.linalg.norm(reference_centered))
        if actual_norm <= 1.0e-30 or reference_norm <= 1.0e-30:
            raise ValueError(
                f"History '{actual_name}' must contain a non-constant wave"
            )
        scale = float(
            numpy.dot(actual_centered, reference_centered)
            / numpy.dot(reference_centered, reference_centered)
        )
        residual = actual_centered - scale * reference_centered
        correlation = float(
            numpy.dot(actual_centered, reference_centered)
            / (actual_norm * reference_norm)
        )
        cross_correlation = numpy.correlate(
            actual_centered / actual_norm,
            reference_centered / reference_norm,
            mode="full",
        )
        peak_index = int(numpy.argmax(cross_correlation))
        lag_samples = peak_index - (eta_array.size - 1)
        variables[str(actual_name)] = {
            "reference_variable": str(reference_name),
            "finite": True,
            "actual_amplitude": float(numpy.ptp(actual_values)),
            "reference_amplitude": float(numpy.ptp(reference_values)),
            "actual_zero_crossings": _zero_crossings(actual_centered),
            "reference_zero_crossings": _zero_crossings(reference_centered),
            "best_fit_scale": scale,
            "normalized_rms_residual": float(
                numpy.linalg.norm(residual) / actual_norm
            ),
            "max_normalized_residual": float(
                numpy.max(numpy.abs(residual)) / max(actual_norm, 1.0e-30)
            ),
            "phase_correlation": correlation,
            "phase_lag_samples": int(lag_samples),
            "phase_lag_eta": float(lag_samples * eta_step),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "available": True,
        "finite": True,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "k_value": (
            None
            if k_value is None
            else _coerce_numeric_scalar(k_value, name="k_value")
        ),
        "eta_count": int(eta_array.size),
        "eta_start": float(eta_array[0]),
        "eta_end": float(eta_array[-1]),
        "eta_sha256": hashlib.sha256(
            numpy.asarray(eta_array, dtype="<f8").tobytes()
        ).hexdigest(),
        "variables": variables,
        "wave_count": int(
            sum(
                metric["actual_zero_crossings"] > 0
                for metric in variables.values()
            )
        ),
        "phase_coherent": bool(
            all(
                abs(metric["phase_correlation"]) >= 0.5
                for metric in variables.values()
            )
        ),
    }
    return report


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


def compute_cmb_spectrum_from_camb_contract(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return standard CAMB spectra from a structured contract."""

    logger = logging.getLogger()
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
        logger.error("(compute_cmb_spectrum_from_camb_contract): %s", exc)
        raise


def compute_camb_background_observables(
    contract_or_params: Mapping[str, Any], redshifts: Sequence[float]
) -> dict[str, numpy.ndarray]:
    """Return CAMB background quantities for ``redshifts``."""

    if not _is_structured_camb_background_contract(contract_or_params):
        raise ValueError(
            "Structured CAMB background contracts must include backend, "
            "param_map, grids, values and calls"
        )

    return _compute_camb_background_direct(contract_or_params, redshifts)


def describe_camb_configuration() -> dict[str, Any]:
    """Return the default CAMB configuration used by the CMB helpers."""

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
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "lmax_padding": _LMAX_PADDING,
        "lens_potential_accuracy": _LENS_POTENTIAL_ACCURACY,
        "reionization_model": "optical_depth_tau",
        "accuracy": accuracy_info,
    }


def _reference_jsonable(value: Any) -> Any:
    """Convert NumPy values into deterministic fixture JSON values."""

    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _reference_jsonable(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_reference_jsonable(item) for item in value]
    return value


def reference_fixture_sha256(fixture: Mapping[str, Any]) -> str:
    """Return the deterministic digest for one frozen reference fixture."""

    payload = json.dumps(
        _reference_jsonable(fixture), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_lcdm_history_reference_fixture() -> dict[str, Any]:
    """Freeze compact native CAMB histories without calculating C_ell.

    Named CAMB density outputs stay in the CDM-comoving synchronous gauge;
    get_time_evolution's frame argument transforms symbolic expressions only.
    This fixture is independent reference data, not CCMBS parity evidence.
    """

    params = _make_camb_params(FIXED_LCDM_REFERENCE_CONTRACT, lmax=32)
    results = camb.get_background(params)
    eta = numpy.asarray(LCDM_HISTORY_ETA_VALUES)
    values = numpy.asarray(
        results.get_time_evolution(
            LCDM_HISTORY_K_VALUES,
            eta,
            vars=list(LCDM_HISTORY_VARIABLES),
            lAccuracyBoost=4,
        ),
        dtype=float,
    )
    expected_shape = (
        len(LCDM_HISTORY_K_VALUES),
        len(eta),
        len(LCDM_HISTORY_VARIABLES),
    )
    if values.shape != expected_shape or not numpy.all(numpy.isfinite(values)):
        raise ValueError("CAMB returned invalid compact histories")
    thermal = results.get_background_time_evolution(
        eta, vars=["x_e", "opacity", "visibility"]
    )
    variables = {
        name: {
            "gauge": (
                "CDM_comoving_synchronous"
                if name.startswith("delta_")
                else (
                    "gauge_invariant"
                    if name in {"pi_photon", "E_2", "E_source"}
                    else "background"
                )
            ),
            "units": (
                "Mpc^-1" if name in {"H", "E_source"} else "dimensionless"
            ),
            "normalization": {
                "delta_baryon": "delta_rho_b/rho_b",
                "delta_photon": "4*Theta_0 (same gauge)",
                "delta_neutrino": "delta_rho_massless_nu/rho_massless_nu",
                "pi_photon": "CAMB brightness quadrupole = 4*Theta_gamma2",
                "E_2": "CAMB polarization quadrupole = 4*e_gamma2",
                "E_source": "Temperature E / (k chi)^2; no spin-2 factorial",
                "a": "scale factor, a_today=1",
                "H": "conformal H = a*H_physical/c",
            }[name],
        }
        for name in LCDM_HISTORY_VARIABLES
    }
    fixture = {
        "schema_version": 1,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "contract": FIXED_LCDM_REFERENCE_CONTRACT,
        "resolved_parameters": resolve_camb_parameters(
            FIXED_LCDM_REFERENCE_CONTRACT
        ),
        "physical_inputs": {
            "TCMB_K": float(params.TCMB),
            "YHe": float(params.YHe),
            "H0_km_s_Mpc": float(params.H0),
            "omk": float(params.omk),
            "As": float(params.InitPower.As),
            "ns": float(params.InitPower.ns),
            "reionization_model": type(params.Reion).__name__,
            "optical_depth": float(params.Reion.optical_depth),
            "delta_redshift": float(params.Reion.delta_redshift),
            "recombination_model": type(params.Recomb).__name__,
            "RECFAST_fudge": float(params.Recomb.RECFAST_fudge),
        },
        "eta_mpc": list(LCDM_HISTORY_ETA_VALUES),
        "k_mpc_inv": list(LCDM_HISTORY_K_VALUES),
        "eta0_mpc": float(results.conformal_time(0.0)),
        "variables": variables,
        "l_accuracy_boost": 4,
        "primordial_normalization": "CAMB unit curvature; no rescaling",
        "recombination": type(params.Recomb).__name__,
        "reionization": type(params.Reion).__name__,
        "thermal_units": {
            "x_e": "electrons_per_hydrogen_nucleus",
            "opacity": "Mpc^-1",
            "visibility": "Mpc^-1",
        },
        "thermal": {
            name: numpy.asarray(array).tolist()
            for name, array in thermal.items()
        },
        "redshift": numpy.asarray(
            results.redshift_at_conformal_time(eta)
        ).tolist(),
        "H_km_s_Mpc": numpy.asarray(
            results.hubble_parameter(results.redshift_at_conformal_time(eta))
        ).tolist(),
        "histories": {
            name: values[:, :, index].tolist()
            for index, name in enumerate(LCDM_HISTORY_VARIABLES)
        },
        "mapping": {
            "entropy": "delta_b - 0.75*delta_gamma (gauge invariant)",
            "absolute_density": "Transform densities to a common gauge first.",
            "polarization": "theta_gamma2=pi_photon/4; e_gamma2=E_2/4",
            "polter": "CAMB polter=0.4*(theta_gamma2+6*e_gamma2)",
        },
        "source": "https://github.com/cmbant/CAMB/tree/1.6.0",
    }
    fixture["fixture_sha256"] = reference_fixture_sha256(fixture)
    return fixture


def write_lcdm_history_reference_fixture(
    path: str | Path = LCDM_HISTORY_REFERENCE_FIXTURE_PATH,
) -> Path:
    """Explicitly regenerate the independent compact reference payload."""

    destination = Path(path)
    content = json.dumps(
        build_lcdm_history_reference_fixture(), indent=2, sort_keys=True
    )
    # Keep the exact 64-character digest on its own readable JSON line.
    content = content.replace('"fixture_sha256": ', '"fixture_sha256":\n    ')
    destination.write_text(content + "\n", encoding="utf-8")
    return destination


def load_lcdm_history_reference_fixture(
    path: str | Path = LCDM_HISTORY_REFERENCE_FIXTURE_PATH,
) -> dict[str, Any]:
    """Read compact histories and enforce their exact stored digest."""

    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = fixture.get("fixture_sha256")
    payload = {
        name: value
        for name, value in fixture.items()
        if name != "fixture_sha256"
    }
    if (
        not isinstance(digest, str)
        or reference_fixture_sha256(payload) != digest
    ):
        raise ValueError("CAMB history fixture digest mismatch")
    if fixture["reference_identity"] != CAMB_REFERENCE_IDENTITY:
        raise ValueError("CAMB history fixture version mismatch")
    eta = _coerce_numeric_array(fixture["eta_mpc"], name="eta_mpc")
    k_values = _coerce_numeric_array(fixture["k_mpc_inv"], name="k_mpc_inv")
    if (
        fixture.get("schema_version") != 1
        or numpy.any(eta <= 0)
        or numpy.any(numpy.diff(eta) <= 0)
        or numpy.any(k_values <= 0)
        or numpy.any(numpy.diff(k_values) <= 0)
    ):
        raise ValueError("CAMB history fixture has an invalid schema or grid")
    if set(fixture["histories"]) != set(LCDM_HISTORY_VARIABLES):
        raise ValueError("CAMB history fixture variable set mismatch")
    for name, values in fixture["histories"].items():
        array = numpy.asarray(values, dtype=float)
        if array.shape != (len(k_values), len(eta)) or not numpy.all(
            numpy.isfinite(array)
        ):
            raise ValueError(f"CAMB history fixture has invalid {name} data")
    return fixture


def build_lcdm_reference_fixture(
    ells: Iterable[int] = FIXED_LCDM_REFERENCE_ELL_VALUES,
    *,
    spectra: Sequence[str] = FIXED_LCDM_REFERENCE_SPECTRA,
) -> dict[str, Any]:
    """Return the fixed CAMB LCDM spectra and auditable provenance."""

    ell_values = tuple(int(value) for value in ells)
    requested_spectra = tuple(str(value).upper() for value in spectra)
    computed = compute_cmb_spectrum_from_camb_contract(
        FIXED_LCDM_REFERENCE_CONTRACT,
        ell_values,
        spectra=requested_spectra,
    )
    if not isinstance(computed, Mapping):
        computed = {requested_spectra[0]: computed}
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "normalization": "unlensed_scalar_D_ell_microkelvin_squared",
        "ell_values": ell_values,
        "spectra": {
            name: numpy.asarray(computed[name], dtype=float)
            for name in requested_spectra
        },
        "contract": FIXED_LCDM_REFERENCE_CONTRACT,
        "resolved_parameters": resolve_camb_parameters(
            FIXED_LCDM_REFERENCE_CONTRACT
        ),
        "tolerances": {
            name: float(FIXED_LCDM_REFERENCE_TOLERANCES[name])
            for name in requested_spectra
            if name in FIXED_LCDM_REFERENCE_TOLERANCES
        },
        "provenance": describe_camb_configuration(),
    }
    fixture["fixture_sha256"] = reference_fixture_sha256(fixture)
    return fixture


def build_camb_full_reference_fixture(
    contract: Mapping[str, Any],
    ells: Iterable[int] = CAMB_PARITY_ELL_VALUES,
    *,
    model_name: str = "declared_model",
    fixed_point: str = "initial",
    spectra: Sequence[str] = CAMB_PARITY_SPECTRA,
) -> dict[str, Any]:
    """Build one complete independent CAMB fixture for a fixed point.

    CAMB is called once for the complete requested surface.  The returned
    payload keeps both native raw ``C_ell`` values and native ``D_ell``
    values, allowing the production diagnostics to compare units and signs
    without interpolating or solving once per graph panel.
    """

    ell_values = tuple(int(value) for value in ells)
    if not ell_values or any(value < 2 for value in ell_values):
        raise ValueError("ells must contain values at or above 2")
    if tuple(sorted(set(ell_values))) != ell_values:
        raise ValueError("ells must be sorted and unique")
    requested = tuple(str(value) for value in spectra)
    if not requested:
        raise ValueError("spectra must not be empty")
    ell_array = numpy.asarray(ell_values, dtype=int)
    lmax = int(ell_array.max())
    params = _make_camb_params(contract, lmax=lmax)
    results = camb.get_results(params)
    arrays = {
        "unlensed_d": numpy.asarray(
            results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
            dtype=float,
        ),
        "unlensed_c": numpy.asarray(
            results.get_unlensed_scalar_cls(
                lmax=lmax, CMB_unit="muK", raw_cl=True
            ),
            dtype=float,
        ),
        "lensed_d": numpy.asarray(
            results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
            dtype=float,
        ),
        "lensed_c": numpy.asarray(
            results.get_lensed_scalar_cls(
                lmax=lmax, CMB_unit="muK", raw_cl=True
            ),
            dtype=float,
        ),
        "lensing_d": numpy.asarray(
            results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK"),
            dtype=float,
        ),
        "lensing_c": numpy.asarray(
            results.get_lens_potential_cls(
                lmax=lmax, CMB_unit="muK", raw_cl=True
            ),
            dtype=float,
        ),
    }
    spectra_payload: dict[str, dict[str, list[float]]] = {}
    for name in requested:
        token = name.upper()
        if token in _CMB_SPECTRUM_COLUMNS:
            column = _CMB_SPECTRUM_COLUMNS[token]
            d_values = arrays["unlensed_d"][ell_array, column]
            c_values = arrays["unlensed_c"][ell_array, column]
        elif token.startswith("LENSED_"):
            base_name = token[7:]
            if base_name not in _CMB_SPECTRUM_COLUMNS:
                raise ValueError(
                    f"Unsupported CAMB reference spectrum: {name}"
                )
            column = _CMB_SPECTRUM_COLUMNS[base_name]
            d_values = arrays["lensed_d"][ell_array, column]
            c_values = arrays["lensed_c"][ell_array, column]
        elif token in _LENSING_SPECTRUM_COLUMNS:
            column = _LENSING_SPECTRUM_COLUMNS[token]
            d_values = arrays["lensing_d"][ell_array, column]
            c_values = arrays["lensing_c"][ell_array, column]
        else:
            raise ValueError(f"Unsupported CAMB reference spectrum: {name}")
        if not numpy.all(numpy.isfinite(d_values)) or not numpy.all(
            numpy.isfinite(c_values)
        ):
            raise ValueError(f"CAMB returned non-finite {name} reference")
        spectra_payload[name] = {
            "C_ell": [float(value) for value in c_values],
            "D_ell": [float(value) for value in d_values],
        }
    fixture: dict[str, Any] = {
        "schema_version": 3,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "backend": "camb",
        "model": str(model_name),
        "fixed_point": str(fixed_point),
        "sector": "scalar",
        "ell_values": ell_values,
        "spectra": spectra_payload,
        "declared_observables": requested,
        "conventions": {
            "cmb": "D_ell = ell*(ell+1)*C_ell/(2*pi)",
            "PP": "D_ell = ell^2*(ell+1)^2*C_ell/(2*pi)",
            "TP_EP": "D_ell = [ell*(ell+1)]^(3/2)*C_ell/(2*pi)",
            "source": "CAMB get_*_cls raw_cl and native D_ell outputs",
        },
        "contract": contract,
        "resolved_parameters": resolve_camb_parameters(contract),
        "tolerances": {
            name: float(FIXED_LCDM_FULL_REFERENCE_TOLERANCES.get(name, 0.05))
            for name in requested
        },
        "provenance": describe_camb_configuration(),
    }
    fixture["fixture_sha256"] = reference_fixture_sha256(fixture)
    return fixture


def build_camb_parity_reference_set(
    model_contracts: Mapping[str, Mapping[str, Any]],
    *,
    ells: Iterable[int] = CAMB_PARITY_ELL_VALUES,
    spectra: Sequence[str] = CAMB_PARITY_SPECTRA,
    numerical_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build multi-model, multi-point CAMB evidence without shared solves.

    ``model_contracts`` is keyed by model filename.  Each value is either one
    fixed contract or a mapping of fixed-point labels to contracts.  This
    keeps massive-neutrino masses and response points explicit in the report.

    The optional ``numerical_plan`` is evidence metadata for the corresponding
    declared CCMBS request.  It is retained beside every raw CAMB row so a
    parity report cannot lose the exact bounded request that produced it.
    """

    ell_values = tuple(int(value) for value in ells)
    if (
        not ell_values
        or any(value < 2 for value in ell_values)
        or tuple(sorted(set(ell_values))) != ell_values
    ):
        raise ValueError("ells must be sorted, unique, and at least 2")
    selected_spectra = tuple(str(value) for value in spectra)
    if not selected_spectra or len(set(selected_spectra)) != len(
        selected_spectra
    ):
        raise ValueError("spectra must be non-empty and unique")
    plan = {
        str(key): _reference_jsonable(value)
        for key, value in (numerical_plan or {}).items()
    }
    models: dict[str, Any] = {}
    for filename, contracts in model_contracts.items():
        if not isinstance(contracts, Mapping):
            raise TypeError(
                f"CAMB contracts for '{filename}' must be mappings"
            )
        if "param_map" in contracts:
            fixed_points = {"initial": contracts}
        else:
            fixed_points = dict(contracts)
        model_rows: dict[str, Any] = {}
        for label, contract in fixed_points.items():
            fixture = build_camb_full_reference_fixture(
                contract,
                ell_values,
                model_name=str(filename),
                fixed_point=str(label),
                spectra=selected_spectra,
            )
            surface_hashes = {
                str(name): reference_fixture_sha256(
                    {
                        "C_ell": values["C_ell"],
                        "D_ell": values["D_ell"],
                    }
                )
                for name, values in fixture["spectra"].items()
            }
            fixture["parity_row"] = {
                "model_filename": str(filename),
                "fixed_point": str(label),
                "reference_identity": CAMB_REFERENCE_IDENTITY,
                "ell_values": ell_values,
                "spectra": selected_spectra,
                "numerical_plan": plan,
                "contract_sha256": reference_fixture_sha256(
                    {"contract": contract}
                ),
                "surface_sha256": surface_hashes,
                "raw_artifact_sha256": reference_fixture_sha256(
                    fixture["spectra"]
                ),
            }
            model_rows[str(label)] = fixture
        models[str(filename)] = model_rows
    report: dict[str, Any] = {
        "schema_version": 2,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "ell_values": ell_values,
        "spectra": selected_spectra,
        "numerical_plan": plan,
        "models": models,
    }
    report["report_sha256"] = reference_fixture_sha256(report)
    return report


def build_fixed_lcdm_camb_parity_row(
    ells: Iterable[int] = CAMB_PARITY_ELL_VALUES,
) -> dict[str, Any]:
    """Build the canonical sparse fixed-LCDM TT/TE/EE CAMB row."""

    return build_camb_full_reference_fixture(
        FIXED_LCDM_REFERENCE_CONTRACT,
        ells,
        model_name="model_lcdm.yml",
        fixed_point="initial",
        spectra=FIXED_LCDM_REFERENCE_SPECTRA,
    )


def build_lcdm_full_reference_fixture(
    ells: Iterable[int] = FIXED_LCDM_REFERENCE_ELL_VALUES,
) -> dict[str, Any]:
    """Return the complete fixed scalar CAMB comparison fixture.

    The fixture stores both raw ``C_ell`` and CAMB-native ``D_ell`` values.
    CAMB's lens-potential spectra use their documented deflection-potential
    convention; the per-observable conversion metadata makes that distinction
    explicit instead of treating lensing outputs as CMB temperature spectra.
    """

    ell_values = tuple(int(value) for value in ells)
    if not ell_values or any(value < 2 for value in ell_values):
        raise ValueError("ells must contain values at or above 2")
    if tuple(sorted(set(ell_values))) != ell_values:
        raise ValueError("ells must be sorted and unique")
    requested = FIXED_LCDM_FULL_REFERENCE_SPECTRA
    ell_array = numpy.asarray(ell_values, dtype=int)
    lmax = int(ell_array.max())
    params = _make_camb_params(FIXED_LCDM_REFERENCE_CONTRACT, lmax=lmax)
    results = camb.get_results(params)
    unlensed_d = numpy.asarray(
        results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=float,
    )
    unlensed_c = numpy.asarray(
        results.get_unlensed_scalar_cls(
            lmax=lmax, CMB_unit="muK", raw_cl=True
        ),
        dtype=float,
    )
    lensed_d = numpy.asarray(
        results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=float,
    )
    lensed_c = numpy.asarray(
        results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=True),
        dtype=float,
    )
    lensing_d = numpy.asarray(
        results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK"),
        dtype=float,
    )
    lensing_c = numpy.asarray(
        results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK", raw_cl=True),
        dtype=float,
    )
    spectra: dict[str, dict[str, list[float]]] = {}
    for name in requested:
        if name in _CMB_SPECTRUM_COLUMNS:
            column = _CMB_SPECTRUM_COLUMNS[name]
            d_values = unlensed_d[ell_array, column]
            c_values = unlensed_c[ell_array, column]
        elif name.startswith("lensed_"):
            base_name = name[7:]
            column = _CMB_SPECTRUM_COLUMNS[base_name]
            d_values = lensed_d[ell_array, column]
            c_values = lensed_c[ell_array, column]
        else:
            column = _LENSING_SPECTRUM_COLUMNS[name]
            d_values = lensing_d[ell_array, column]
            c_values = lensing_c[ell_array, column]
        if not numpy.all(numpy.isfinite(d_values)) or not numpy.all(
            numpy.isfinite(c_values)
        ):
            raise ValueError(f"CAMB returned non-finite {name} reference")
        spectra[name] = {
            "C_ell": [float(value) for value in c_values],
            "D_ell": [float(value) for value in d_values],
        }
    fixture: dict[str, Any] = {
        "schema_version": 2,
        "reference_identity": CAMB_REFERENCE_IDENTITY,
        "backend": "camb",
        "model": "LambdaCDM",
        "sector": "scalar",
        "ell_values": ell_values,
        "spectra": spectra,
        "declared_observables": requested,
        "applicability": {
            "scalar": {
                "included": requested,
                "omitted": (),
                "reason": "All scalar CMB and lensing observables requested",
            },
            "vector": {
                "included": (),
                "omitted": requested,
                "reason": "No vector sector is declared by bundled models",
            },
            "tensor": {
                "included": (),
                "omitted": requested,
                "reason": (
                    "Tensor sectors require a separate nonzero-r fixture"
                ),
            },
        },
        "conventions": {
            "cmb": "D_ell = ell*(ell+1)*C_ell/(2*pi)",
            "PP": "D_ell = ell^2*(ell+1)^2*C_ell/(2*pi)",
            "TP_EP": ("D_ell = [ell*(ell+1)]^(3/2)*C_ell/(2*pi)"),
            "temperature_polarization_unit": "microkelvin_squared",
            "lensing_potential_unit": "dimensionless_PP_and_microkelvin_cross",
            "source": "CAMB get_*_cls raw_cl and native D_ell outputs",
        },
        "contract": FIXED_LCDM_REFERENCE_CONTRACT,
        "resolved_parameters": resolve_camb_parameters(
            FIXED_LCDM_REFERENCE_CONTRACT
        ),
        "tolerances": {
            name: float(FIXED_LCDM_FULL_REFERENCE_TOLERANCES[name])
            for name in requested
        },
        "provenance": describe_camb_configuration(),
    }
    fixture["fixture_sha256"] = reference_fixture_sha256(fixture)
    return fixture


def write_lcdm_full_reference_fixture(
    path: str | Path = FULL_LCDM_REFERENCE_FIXTURE_PATH,
) -> Path:
    """Write one canonical full LCDM fixture and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_lcdm_full_reference_fixture()
    destination.write_text(
        json.dumps(_reference_jsonable(fixture), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_lcdm_full_reference_fixture(
    path: str | Path = FULL_LCDM_REFERENCE_FIXTURE_PATH,
) -> dict[str, Any]:
    """Load and integrity-check the tracked full LCDM fixture."""

    source = Path(path)
    fixture = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("CAMB fixture root must be an object")
    stored_digest = fixture.get("fixture_sha256")
    if not isinstance(stored_digest, str):
        raise ValueError("CAMB fixture is missing fixture_sha256")
    actual_digest = reference_fixture_sha256(
        {
            key: value
            for key, value in fixture.items()
            if key != "fixture_sha256"
        }
    )
    if stored_digest != actual_digest:
        raise ValueError(
            "CAMB fixture digest mismatch: "
            f"{stored_digest} != {actual_digest}"
        )
    return fixture


def compare_lcdm_reference_spectra(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    representation: str = "D_ell",
) -> dict[str, dict[str, float]]:
    """Compare aligned fixture spectra without interpolation or fallback."""

    if representation not in {"C_ell", "D_ell"}:
        raise ValueError("representation must be C_ell or D_ell")
    metrics: dict[str, dict[str, float]] = {}
    for name, reference_entry in reference.items():
        if name not in actual:
            raise KeyError(f"Missing spectrum '{name}' for comparison")
        if not isinstance(reference_entry, Mapping):
            raise ValueError(f"Reference spectrum '{name}' is not structured")
        expected = numpy.asarray(reference_entry[representation], dtype=float)
        observed = numpy.asarray(actual[name], dtype=float)
        if expected.shape != observed.shape:
            raise ValueError(
                f"Spectrum '{name}' shape mismatch: "
                f"{observed.shape} != {expected.shape}"
            )
        if not numpy.all(numpy.isfinite(expected)) or not numpy.all(
            numpy.isfinite(observed)
        ):
            raise ValueError(f"Spectrum '{name}' contains non-finite values")
        delta = numpy.abs(observed - expected)
        scale = numpy.maximum(numpy.abs(expected), 1.0e-30)
        metrics[name] = {
            "max_absolute": float(numpy.max(delta, initial=0.0)),
            "max_fractional": float(numpy.max(delta / scale, initial=0.0)),
            "sample_count": float(expected.size),
        }
    return metrics


__all__ = [
    "build_lcdm_history_reference_fixture",
    "load_lcdm_history_reference_fixture",
    "write_lcdm_history_reference_fixture",
    "CAMB_REFERENCE_IDENTITY",
    "CAMB_PARITY_ELL_VALUES",
    "CAMB_PARITY_RELATIVE_TOLERANCES",
    "CAMB_PARITY_SPECTRA",
    "FIXED_LCDM_REFERENCE_CONTRACT",
    "FIXED_LCDM_REFERENCE_ELL_VALUES",
    "FIXED_LCDM_REFERENCE_SPECTRA",
    "FIXED_LCDM_REFERENCE_TOLERANCES",
    "FIXED_LCDM_FULL_REFERENCE_SPECTRA",
    "FIXED_LCDM_FULL_REFERENCE_TOLERANCES",
    "FULL_LCDM_REFERENCE_FIXTURE_PATH",
    "build_lcdm_reference_fixture",
    "build_lcdm_full_reference_fixture",
    "build_camb_full_reference_fixture",
    "build_camb_parity_reference_set",
    "build_fixed_lcdm_camb_parity_row",
    "compare_lcdm_reference_spectra",
    "compare_scalar_history_waves",
    "compute_camb_background_observables",
    "compute_camb_scalar_time_evolution",
    "camb_histories_in_newtonian_gauge",
    "compute_cmb_spectrum_from_camb_contract",
    "describe_camb_configuration",
    "load_lcdm_full_reference_fixture",
    "reference_fixture_sha256",
    "resolve_camb_parameters",
    "write_lcdm_full_reference_fixture",
]


def completion_reference_windows(fixture: Mapping[str, Any]) -> list[dict]:
    """Freeze contiguous feature windows from independent dense CAMB arrays.

    Acoustic extrema above ell 50 exclude the reionization bump. Three TT
    peaks/troughs, EE peaks, and TE crossings cover phase and sign; fixed
    high-ell windows constrain damping. Unit spacing prevents interpolation
    from hiding a spike between the sparse range anchors.
    """

    ell = numpy.asarray(fixture["ell_values"], dtype=int)
    if not numpy.array_equal(ell, numpy.arange(2, 2501)):
        raise ValueError("Completion windows require dense ell 2 through 2500")
    features = []
    for name, kind in (
        ("TT", "peak"),
        ("TT", "trough"),
        ("EE", "peak"),
        ("TE", "zero_crossing"),
    ):
        values = numpy.asarray(fixture["spectra"][name]["D_ell"])
        if kind == "zero_crossing":
            indices = numpy.flatnonzero(values[:-1] * values[1:] <= 0)
        else:
            sign = 1 if kind == "peak" else -1
            indices = (
                numpy.flatnonzero(
                    (sign * numpy.diff(values)[:-1] > 0)
                    & (sign * numpy.diff(values)[1:] < 0)
                )
                + 1
            )
        centers = [int(ell[index]) for index in indices if ell[index] >= 50][
            :3
        ]
        if len(centers) != 3:
            raise ValueError(
                f"Independent reference lacks three {name} {kind}s"
            )
        for center in centers:
            features.append(
                {
                    "surface": name,
                    "kind": kind,
                    "ell": center,
                    "lower": center - 8,
                    "upper": center + 8,
                }
            )
    return features


def build_completion_camb_reference(
    contract: Mapping[str, Any],
    *,
    model_name: str,
    fixed_point: str,
    requirement: Mapping[str, Any] | None = None,
) -> dict:
    """Retain resolved defaults and the explicitly frozen acoustic windows.

    Updating a physical reference must not move its tested windows or drop
    surfaces. Initial contract creation alone selects windows from CAMB.
    """

    if requirement is not None:
        reference = build_camb_full_reference_fixture(
            contract,
            requirement["ells"],
            model_name=model_name,
            fixed_point=fixed_point,
            spectra=tuple(
                name
                for name in CAMB_PARITY_SPECTRA
                if name in requirement["surfaces"]
            ),
        )
        reference["ell_values"] = list(requirement["ells"])
        reference["features"] = copy.deepcopy(requirement["features"])
        reference["damping_windows"] = copy.deepcopy(
            requirement["damping_windows"]
        )
        reference["resolved_defaults"] = str(
            _make_camb_params(contract, lmax=max(requirement["ells"]))
        )
        reference.pop("fixture_sha256")
        reference["fixture_sha256"] = reference_fixture_sha256(reference)
        return reference

    dense = build_camb_full_reference_fixture(
        contract,
        range(2, 2501),
        model_name=model_name,
        fixed_point=fixed_point,
    )
    features = completion_reference_windows(dense)
    selected = set(CAMB_PARITY_ELL_VALUES)
    for feature in features:
        selected.update(range(feature["lower"], feature["upper"] + 1))
    selected.update(range(1988, 2013))
    selected.update(range(2480, 2501))
    ells = sorted(selected)
    indices = numpy.asarray(ells) - 2
    dense["ell_values"] = ells
    for surface in dense["spectra"].values():
        for representation in ("C_ell", "D_ell"):
            surface[representation] = numpy.asarray(surface[representation])[
                indices
            ].tolist()
    dense["features"] = features
    dense["damping_windows"] = [[1988, 2012], [2480, 2500]]
    dense["resolved_defaults"] = str(_make_camb_params(contract, lmax=2500))
    dense.pop("fixture_sha256")
    dense["fixture_sha256"] = reference_fixture_sha256(dense)
    return dense


def build_experimental_completion_reference(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    requirement: Mapping[str, Any],
    *,
    input_tables: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind independently refined experimental arrays to preserved windows.

    The supplied candidates must contain dense ell 2..2500. Original required
    multipoles remain sampled while new feature windows follow the changed
    physical reference. Both complete candidate calculations and their
    physical tables survive so acceptance can recompute reference uncertainty.
    """
    from copernican.lib.likelihoods.cmb import diagnostics

    identity = str(fine.get("reference_identity", ""))
    if not identity.startswith("camb:experimental:"):
        raise ValueError("Experimental reference identity is required")
    for label, candidate in (("coarse", coarse), ("fine", fine)):
        body = {
            name: value
            for name, value in candidate.items()
            if name != "fixture_sha256"
        }
        if (
            candidate.get("fixture_sha256") != reference_fixture_sha256(body)
            or candidate.get("reference_identity") != identity
            or candidate.get("contract") != fine.get("contract")
            or candidate.get("ell_values") != list(range(2, 2501))
            or set(candidate.get("spectra", {}))
            != set(requirement["surfaces"])
            or candidate.get("experimental_inputs", {}).get(
                "physical_inputs_sha256"
            )
            != fine.get("experimental_inputs", {}).get(
                "physical_inputs_sha256"
            )
        ):
            raise ValueError(f"Invalid {label} experimental candidate")

    retained_tables = {}
    for label, candidate in (("coarse", coarse), ("fine", fine)):
        retained_tables[label] = {}
        for name, columns in (("recombination", 3), ("reionization", 2)):
            table = numpy.asarray(input_tables[label][name], dtype=float)
            if (
                table.ndim != 2
                or table.shape[0] < 2
                or table.shape[1] != columns
                or not numpy.all(numpy.isfinite(table))
                or numpy.any(numpy.diff(table[:, 0]) <= 0)
                or numpy.any(table[:, 1:] < 0)
            ):
                raise ValueError(f"Invalid {label} {name} table")
            serialized = io.StringIO()
            numpy.savetxt(serialized, table, fmt="%.17e")
            digest = hashlib.sha256(serialized.getvalue().encode()).hexdigest()
            if (
                digest
                != candidate["experimental_inputs"][f"{name}_table_sha256"]
            ):
                raise ValueError(f"Changed {label} {name} table")
            retained_tables[label][name] = table.tolist()
    for name in ("recombination", "reionization"):
        if len(retained_tables["fine"][name]) <= len(
            retained_tables["coarse"][name]
        ):
            raise ValueError(f"{name} refinement did not increase sampling")

    features = completion_reference_windows(fine)
    ells = set(int(value) for value in requirement["ells"])
    for feature in features:
        ells.update(range(feature["lower"], feature["upper"] + 1))
    if not ells or min(ells) < 2 or max(ells) > 2500:
        raise ValueError("Experimental feature windows exceed dense coverage")
    selected = sorted(ells)
    indices = numpy.asarray(selected, dtype=int) - 2

    def select_spectra(candidate):
        """Select the same raw and public samples from one dense solve."""
        return {
            name: {
                representation: numpy.asarray(values)[indices].tolist()
                for representation, values in surface.items()
            }
            for name, surface in candidate["spectra"].items()
        }

    result = copy.deepcopy(dict(fine))
    source_digest = result["experimental_inputs"].get("source_identity_sha256")
    if source_digest is not None:
        identity = experimental_reference_identity(source_digest)
        result["reference_identity"] = identity
        result["provenance"]["reference_identity"] = identity
    result["ell_values"] = selected
    result["spectra"] = select_spectra(fine)
    result["features"] = features
    result["damping_windows"] = copy.deepcopy(requirement["damping_windows"])
    result["experimental_inputs"]["previous_feature_windows"] = copy.deepcopy(
        requirement["features"]
    )
    result["reference_refinement"] = {
        "ell_values": selected,
        "spectra": select_spectra(coarse),
        "input_tables": retained_tables,
        "table_sha256": {
            label: {
                name: candidate["experimental_inputs"][f"{name}_table_sha256"]
                for name in ("recombination", "reionization")
            }
            for label, candidate in (("coarse", coarse), ("fine", fine))
        },
        "candidate_payloads": {"coarse": coarse, "fine": fine},
        "coarse_fixture_sha256": coarse["fixture_sha256"],
        "fine_fixture_sha256": fine["fixture_sha256"],
    }
    result.pop("fixture_sha256", None)
    result["fixture_sha256"] = reference_fixture_sha256(result)
    updated = copy.deepcopy(dict(requirement))
    updated.update(
        ells=selected,
        features=features,
        reference_identity=identity,
        reference_sha256=result["fixture_sha256"],
        resolved_reference_inputs=copy.deepcopy(result["resolved_parameters"]),
    )
    comparison = diagnostics.compare_cmb_completion_spectra(
        result, result, updated
    )
    if not comparison["accepted"]:
        raise ValueError(
            "Experimental reference refinement or features failed: "
            + "; ".join(comparison["issues"])
        )
    return result, updated
