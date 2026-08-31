"""Physics tests for the declared-graph CMB likelihood helpers."""

from __future__ import annotations

import ast
import copy
import inspect
import re
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence
from unittest import mock

import camb
import numpy
import pandas
from scipy.integrate import quad as scipy_quad
from scipy.linalg import expm

from copernican.lib import model_coder
from copernican.lib.likelihoods import cmb
from copernican.lib.likelihoods.cmb.orchestrators import ccmbs as cmb_solver
from copernican.lib.likelihoods.cmb.runtime import background as cmb_background
from copernican.lib.likelihoods.cmb.runtime import (
    cache,
    convergence,
    evolution,
)
from copernican.lib.likelihoods.cmb.runtime import projection as cmb_projection
from tests.project.lib import camb_reference


def _named_limit_message(
    quantity: str,
    measured: float,
    tolerance: float,
) -> str:
    """Return a consistent failure message for validation thresholds."""

    return (
        f"{quantity} exceeded tolerance: measured={measured:.6g}, "
        f"tolerance={tolerance:.6g}"
    )


def _full_width_at_half_max(
    grid: numpy.ndarray,
    values: numpy.ndarray,
) -> float:
    """Return the width of ``values`` at half of its maximum."""

    grid_values = numpy.asarray(grid, dtype=float)
    profile = numpy.asarray(values, dtype=float)
    half_max = 0.5 * float(numpy.max(profile))
    support = numpy.flatnonzero(profile >= half_max)
    if support.size < 2:
        raise ValueError("Half-maximum support requires at least two points.")
    return float(grid_values[support[-1]] - grid_values[support[0]])


def _local_extrema_ells(
    ells: numpy.ndarray,
    values: numpy.ndarray,
    *,
    kind: str,
    ell_min: int,
    ell_max: int,
) -> list[int]:
    """Return local-extremum multipoles inside ``[ell_min, ell_max]``."""

    ell_values = numpy.asarray(ells, dtype=int)
    spectrum = numpy.asarray(values, dtype=float)
    extrema: list[int] = []
    for index in range(1, spectrum.size - 1):
        ell_value = int(ell_values[index])
        if ell_value < ell_min or ell_value > ell_max:
            continue
        if kind == "max":
            keep = spectrum[index - 1] < spectrum[index] >= spectrum[index + 1]
        elif kind == "min":
            keep = spectrum[index - 1] > spectrum[index] <= spectrum[index + 1]
        else:  # pragma: no cover - helper misuse
            raise ValueError(f"Unsupported extremum kind: {kind}")
        if keep:
            extrema.append(ell_value)
    return extrema


def _zero_crossing_ells(
    ells: numpy.ndarray,
    values: numpy.ndarray,
    *,
    ell_min: int,
    ell_max: int,
) -> list[int]:
    """Return the multipoles where ``values`` changes sign."""

    ell_values = numpy.asarray(ells, dtype=int)
    spectrum = numpy.asarray(values, dtype=float)
    zero_crossings: list[int] = []
    for index in range(1, spectrum.size):
        ell_value = int(ell_values[index])
        if ell_value < ell_min or ell_value > ell_max:
            continue
        if spectrum[index - 1] == 0.0 or spectrum[index] == 0.0:
            zero_crossings.append(ell_value)
            continue
        if (spectrum[index - 1] < 0.0) != (spectrum[index] < 0.0):
            zero_crossings.append(ell_value)
    return zero_crossings


def _max_relative_delta(
    baseline: numpy.ndarray,
    changed: numpy.ndarray,
) -> float:
    """Return the maximum relative change between two arrays."""

    baseline_values = numpy.asarray(baseline, dtype=numpy.longdouble)
    changed_values = numpy.asarray(changed, dtype=numpy.longdouble)
    baseline_scale = numpy.max(numpy.abs(baseline_values), initial=0.0)
    if baseline_scale == 0.0:
        baseline_scale = numpy.longdouble(1.0)
    return float(
        numpy.max(
            numpy.abs(changed_values - baseline_values),
            initial=0.0,
        )
        / baseline_scale
    )


def _slice_nine_spectrum_metrics(
    actual: Mapping[str, numpy.ndarray],
    reference: Mapping[str, numpy.ndarray],
    *,
    spectra: Sequence[str],
    auto_spectrum_floor: float = 1.0e-10,
) -> dict[str, dict[str, float]]:
    """Return absolute auto and normalized cross-spectrum error metrics.

    Auto spectra are summarized with fractional errors only where the
    reference is above a relative floor. Cross spectra use an RMS error
    normalized by the reference RMS, so sign changes and zero crossings do
    not create artificial fractional singularities.
    """

    auto_spectra = {
        "TT",
        "EE",
        "BB",
        "PP",
        "lensed_TT",
        "lensed_EE",
        "lensed_BB",
    }
    metrics: dict[str, dict[str, float]] = {}
    for spectrum_name in spectra:
        name = str(spectrum_name)
        if name not in actual or name not in reference:
            raise KeyError(f"Missing spectrum '{name}' for comparison")
        actual_values = numpy.asarray(actual[name], dtype=numpy.longdouble)
        reference_values = numpy.asarray(
            reference[name],
            dtype=numpy.longdouble,
        )
        if actual_values.shape != reference_values.shape:
            raise ValueError(
                f"Spectrum '{name}' has incompatible comparison shapes: "
                f"{actual_values.shape} != {reference_values.shape}"
            )
        finite = numpy.isfinite(actual_values) & numpy.isfinite(
            reference_values
        )
        if not numpy.any(finite):
            raise ValueError(
                f"Spectrum '{name}' has no finite comparison data"
            )
        if name in auto_spectra:
            reference_scale = numpy.max(
                numpy.abs(reference_values[finite]),
                initial=numpy.longdouble(0.0),
            )
            floor = max(
                numpy.longdouble("1.0e-30"),
                numpy.longdouble(auto_spectrum_floor) * reference_scale,
            )
            supported = finite & (numpy.abs(reference_values) > floor)
            if not numpy.any(supported):
                raise ValueError(
                    f"Spectrum '{name}' has no values above the comparison "
                    "floor"
                )
            fractional = numpy.abs(
                (actual_values[supported] - reference_values[supported])
                / reference_values[supported]
            )
            metrics[name] = {
                "median_fractional": float(numpy.median(fractional)),
                "p90_fractional": float(numpy.percentile(fractional, 90.0)),
                "max_fractional": float(numpy.max(fractional)),
                "sample_count": float(fractional.size),
            }
            continue
        delta = actual_values[finite] - reference_values[finite]
        reference_rms = numpy.sqrt(
            numpy.mean(numpy.square(reference_values[finite]))
        )
        if reference_rms <= numpy.longdouble("1.0e-30"):
            normalized_rms = numpy.longdouble(0.0)
            if numpy.any(numpy.abs(delta) > numpy.longdouble("1.0e-30")):
                normalized_rms = numpy.longdouble(numpy.inf)
        else:
            normalized_rms = (
                numpy.sqrt(numpy.mean(numpy.square(delta))) / reference_rms
            )
        metrics[name] = {
            "normalized_rms": float(normalized_rms),
            "sample_count": float(delta.size),
        }
    return metrics


SLICE_NINE_NEUTRAL_MODEL = MappingProxyType(
    {
        "H0": 67.4,
        "ombh2": 0.02237,
        "omch2": 0.12,
        "Tcmb_K": 2.7255,
        "YHe": 0.245,
        "Neff": 3.046,
        "As": 2.1e-9,
        "ns": 0.965,
        "tau": 0.054,
    }
)

SLICE_NINE_DECLARED_NUMERICAL_CONTROLS = MappingProxyType(
    {
        "ell_min": 2,
        "ell_max": 2000,
        "k_min": 1.0e-4,
        "k_max": 0.3,
        "k_sample_count": 128,
        "eta_sample_count": 2048,
        "evolution_eta_sample_count": 2048,
        "ode_rtol": 1.0e-6,
        "ode_atol": 1.0e-9,
        "tight_coupling_ratio": 80.0,
        "a_min": 1.0e-6,
        "source_grid_multiplier": 2,
        "initial_redshift": 2.0e4,
        "photon_hierarchy_l_max": 16,
        "neutrino_hierarchy_l_max": 12,
    }
)

SLICE_NINE_ACCEPTANCE_RANGES = MappingProxyType(
    {
        "scalar_ell": (2, 2000),
        "potential_ell": (10, 1500),
        "lensing_ell": (2, 2000),
    }
)

SLICE_NINE_ACCEPTANCE_SPECTRA = (
    "TT",
    "TE",
    "EE",
    "PP",
    "TP",
    "EP",
    "lensed_TT",
    "lensed_TE",
    "lensed_EE",
    "lensed_BB",
)

SLICE_NINE_ACCEPTANCE_THRESHOLDS = MappingProxyType(
    {
        "conformal_age_fraction": numpy.longdouble("0.002"),
        "sound_horizon_fraction": numpy.longdouble("0.002"),
        "visibility_peak_fraction": numpy.longdouble("0.005"),
        "visibility_width_fraction": numpy.longdouble("0.03"),
        "recombination_median_fraction": numpy.longdouble("0.02"),
        "recombination_p90_fraction": numpy.longdouble("0.05"),
        "tau_reio_fraction": numpy.longdouble("0.01"),
        "tt_fractional_median": numpy.longdouble("0.05"),
        "tt_fractional_p90": numpy.longdouble("0.10"),
        "ee_fractional_median": numpy.longdouble("0.05"),
        "ee_fractional_p90": numpy.longdouble("0.10"),
        "te_normalized_rms": numpy.longdouble("0.05"),
        "acoustic_feature_ell": numpy.longdouble("3.0"),
        "pp_fractional_median": numpy.longdouble("0.10"),
        "pp_fractional_p90": numpy.longdouble("0.20"),
        "lensed_bb_fractional_median": numpy.longdouble("0.15"),
        "tensor_fractional_median": numpy.longdouble("0.10"),
        "massive_neutrino_response_fractional": numpy.longdouble("0.10"),
        "gauge_equivalent_fractional": numpy.longdouble("0.001"),
    }
)


def _slice_nine_declared_acceptance_contract() -> dict[str, object]:
    """Return the fixed declared contract used by later Slice Nine tests."""

    contract = _declared_scalar_hierarchy_contract(sum_mnu=0.0)
    contract["model_name"] = "SliceNineNeutralDeclared"
    contract["param_map"] = {
        name: float(value)
        for name, value in SLICE_NINE_NEUTRAL_MODEL.items()
        if name != "Tcmb_K"
    }
    contract["model_parameters"] = {
        "Tcmb_K": SLICE_NINE_NEUTRAL_MODEL["Tcmb_K"]
    }
    numerical = dict(SLICE_NINE_DECLARED_NUMERICAL_CONTROLS)
    contract["numerical"] = copy.deepcopy(numerical)
    perturbations = contract["perturbations"]
    if not isinstance(perturbations, dict):  # pragma: no cover - fixture guard
        raise TypeError("Declared acceptance perturbations must be a mapping")
    perturbations["numerics"] = copy.deepcopy(numerical)
    accuracy_controls = dict(perturbations.get("accuracy_controls", {}))
    accuracy_controls["scalar_reference_ells"] = [2, 2000]
    accuracy_controls["phase_aware_k_quadrature"] = True
    accuracy_controls["runtime_envelope"] = "bounded"
    perturbations["accuracy_controls"] = accuracy_controls
    return contract


def _slice_nine_reference_backend_name() -> str:
    """Return the independent reference backend name."""

    return "CAMB"


def _slice_nine_camb_reference_contract(*, lmax: int) -> dict[str, object]:
    """Return metadata for a direct CAMB reference calculation."""

    if int(lmax) < 2:
        raise ValueError("lmax must be at least 2")
    return {
        "backend": "camb",
        "standard": True,
        "lmax": int(lmax),
        "model_values": dict(SLICE_NINE_NEUTRAL_MODEL),
    }


def _slice_nine_build_camb_params(
    *,
    lmax: int,
    sum_mnu: float = 0.0,
    num_massive_neutrinos: int = 3,
    want_tensors: bool = False,
    tensor_ratio: float = 0.0,
    tensor_tilt: float = 0.0,
) -> object:
    """Build CAMB parameters without using a production solver path."""

    if camb is None:
        raise RuntimeError("CAMB is not installed")
    params = camb.CAMBparams()
    params.set_cosmology(
        H0=float(SLICE_NINE_NEUTRAL_MODEL["H0"]),
        ombh2=float(SLICE_NINE_NEUTRAL_MODEL["ombh2"]),
        omch2=float(SLICE_NINE_NEUTRAL_MODEL["omch2"]),
        tau=float(SLICE_NINE_NEUTRAL_MODEL["tau"]),
        YHe=float(SLICE_NINE_NEUTRAL_MODEL["YHe"]),
        nnu=float(SLICE_NINE_NEUTRAL_MODEL["Neff"]),
        TCMB=float(SLICE_NINE_NEUTRAL_MODEL["Tcmb_K"]),
        mnu=float(sum_mnu),
        num_massive_neutrinos=int(num_massive_neutrinos),
    )
    params.InitPower.set_params(
        As=float(SLICE_NINE_NEUTRAL_MODEL["As"]),
        ns=float(SLICE_NINE_NEUTRAL_MODEL["ns"]),
        r=float(tensor_ratio),
        nt=float(tensor_tilt),
    )
    params.WantTensors = bool(want_tensors)
    params.set_for_lmax(int(lmax) + 300, lens_potential_accuracy=1)
    return params


def _slice_nine_camb_background_reference(
    eta_grid: numpy.ndarray,
) -> dict[str, object]:
    """Return direct CAMB background and recombination reference data."""

    eta_values = numpy.asarray(eta_grid, dtype=float)
    if eta_values.ndim != 1 or eta_values.size == 0:
        raise ValueError("eta_grid must be a non-empty one-dimensional array")
    lmax = 32
    params = _slice_nine_build_camb_params(lmax=lmax)
    results = camb.get_results(params)
    histories = results.get_background_time_evolution(
        eta_values,
        vars=["x_e", "visibility", "opacity"],
        format="dict",
    )
    reionization_eta_grid = numpy.linspace(
        float(results.conformal_time(50.0)),
        float(results.conformal_time(0.0)),
        max(2048, eta_values.size),
    )
    reionization_opacity = results.get_background_time_evolution(
        reionization_eta_grid,
        vars=["opacity"],
        format="dict",
    )["opacity"]
    tau_reio = numpy.trapz(
        numpy.asarray(reionization_opacity, dtype=float),
        reionization_eta_grid,
    )
    peak_eta = float(results.tau_maxvis)
    peak_z = float(results.redshift_at_conformal_time(peak_eta))
    return {
        "eta0": float(results.conformal_time(0.0)),
        "peak_eta": peak_eta,
        "peak_z": peak_z,
        "sound_horizon": float(results.sound_horizon(peak_z)),
        "x_e": numpy.asarray(histories["x_e"], dtype=float),
        "visibility": numpy.asarray(histories["visibility"], dtype=float),
        "tau_reio": float(tau_reio),
    }


def _slice_nine_camb_reference_spectra(
    ells: Sequence[int] | numpy.ndarray,
    *,
    spectra: Sequence[str] = SLICE_NINE_ACCEPTANCE_SPECTRA,
    sum_mnu: float = 0.0,
    num_massive_neutrinos: int = 3,
) -> dict[str, numpy.ndarray]:
    """Return direct CAMB reference spectra at requested multipoles."""

    ell_grid = numpy.asarray(tuple(ells), dtype=int)
    if ell_grid.size == 0 or numpy.any(ell_grid < 2):
        raise ValueError("ells must contain values at or above 2")
    lmax = int(numpy.max(ell_grid))
    params = _slice_nine_build_camb_params(
        lmax=lmax,
        sum_mnu=float(sum_mnu),
        num_massive_neutrinos=int(num_massive_neutrinos),
    )
    results = camb.get_results(params)
    unlensed = numpy.asarray(
        results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=numpy.longdouble,
    )
    lensed = numpy.asarray(
        results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=numpy.longdouble,
    )
    lensing = numpy.asarray(
        results.get_lens_potential_cls(lmax=lmax),
        dtype=numpy.longdouble,
    )
    columns = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
    outputs: dict[str, numpy.ndarray] = {}
    for spectrum_name in spectra:
        name = str(spectrum_name)
        if name in columns:
            values = unlensed[:, columns[name]]
        elif name.startswith("lensed_") and name[7:] in columns:
            values = lensed[:, columns[name[7:]]]
        elif name in {"PP", "TP", "EP"}:
            values = lensing[:, {"PP": 0, "TP": 1, "EP": 2}[name]]
        else:
            raise ValueError(f"Unsupported CAMB reference spectrum: {name}")
        outputs[name] = numpy.asarray(values[ell_grid], dtype=numpy.longdouble)
    return outputs


def _slice_thirty_camb_tensor_reference_spectra(
    ells: Sequence[int] | numpy.ndarray,
    *,
    sum_mnu: float = 0.0,
    tensor_ratio: float = 0.1,
    tensor_tilt: float = 0.0,
) -> dict[str, numpy.ndarray]:
    """Return direct CAMB unlensed and lensed tensor spectra."""

    ell_grid = numpy.asarray(tuple(ells), dtype=int)
    if ell_grid.size == 0 or numpy.any(ell_grid < 2):
        raise ValueError("ells must contain values at or above 2")
    lmax = int(numpy.max(ell_grid))
    params = _slice_nine_build_camb_params(
        lmax=lmax,
        sum_mnu=float(sum_mnu),
        want_tensors=True,
        tensor_ratio=float(tensor_ratio),
        tensor_tilt=float(tensor_tilt),
    )
    results = camb.get_results(params)
    tensor_cls = numpy.asarray(
        results.get_tensor_cls(
            lmax=lmax,
            CMB_unit="muK",
        ),
        dtype=numpy.longdouble,
    )
    total_cls = numpy.asarray(
        results.get_total_cls(lmax=lmax, CMB_unit="muK"),
        dtype=numpy.longdouble,
    )
    lensed_scalar_cls = numpy.asarray(
        results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK"),
        dtype=numpy.longdouble,
    )
    lensed_tensor_cls = total_cls - lensed_scalar_cls
    outputs = {
        name: numpy.asarray(
            tensor_cls[ell_grid, column],
            dtype=numpy.longdouble,
        )
        for name, column in (("TT", 0), ("EE", 1), ("BB", 2))
    }
    outputs.update(
        {
            f"lensed_{name}": numpy.asarray(
                lensed_tensor_cls[ell_grid, column],
                dtype=numpy.longdouble,
            )
            for name, column in (("TT", 0), ("EE", 1), ("BB", 2))
        }
    )
    return outputs


def _rename_declared_contract_tokens(
    value: object,
    rename_map: Mapping[str, str],
) -> object:
    """Return ``value`` with declared symbol references renamed."""

    if isinstance(value, str):
        if not rename_map:
            return value
        pattern = re.compile(
            r"\b("
            + "|".join(
                re.escape(name)
                for name in sorted(rename_map, key=len, reverse=True)
            )
            + r")\b"
        )
        return pattern.sub(lambda match: rename_map[match.group(0)], value)
    if isinstance(value, list):
        return [
            _rename_declared_contract_tokens(item, rename_map)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rename_declared_contract_tokens(item, rename_map)
            for item in value
        )
    if isinstance(value, dict):
        renamed: dict[object, object] = {}
        for key, item in value.items():
            renamed_key = (
                rename_map.get(key, key) if isinstance(key, str) else key
            )
            renamed[renamed_key] = _rename_declared_contract_tokens(
                item,
                rename_map,
            )
        return renamed
    return value


def _split_collision_contract(
    *,
    rename_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a declared graph that routes Thomson through exact metadata."""

    contract = _speedup_contract(_custom_contract())
    perturbations = contract["perturbations"]
    perturbations["derived"]["photon_baryon_momentum_ratio"] = {
        "expression": "(4.0 * Omega_gamma0) / (3.0 * Omega_b0 * a)",
        "description": "Photon-to-baryon momentum-transfer ratio.",
    }
    perturbations["derived"]["baryon_thomson_drag"] = {
        "expression": (
            "-3.0 * k * photon_baryon_momentum_ratio * thomson_drag"
        ),
        "description": "Baryon counterpart for the exact Thomson block.",
    }
    perturbations["equations"]["evolve_theta_gamma1"]["rhs"] = (
        "(k / 3.0) * (theta_gamma0 + Psi - 2.0 * theta_gamma2) "
        "+ thomson_drag"
    )
    perturbations["equations"]["evolve_theta_gamma2"]["rhs"] = (
        "(2.0 / 5.0) * k * theta_gamma1 " "- (3.0 / 5.0) * k * theta_gamma3"
    )
    perturbations["equations"]["evolve_e_gamma0"]["rhs"] = (
        "-k * e_gamma1 - collision_rate * "
        "(e_gamma0 - 0.5 * polarization_moment)"
    )
    perturbations["equations"]["evolve_e_gamma1"]["rhs"] = (
        "(k / 3.0) * (e_gamma0 - 2.0 * e_gamma2) - "
        "collision_rate * e_gamma1"
    )
    perturbations["equations"]["evolve_e_gamma2"]["rhs"] = (
        "(2.0 / 5.0) * k * e_gamma1 " "- (3.0 / 5.0) * k * e_gamma3"
    )
    polarization_moment = perturbations["derived"]["polarization_moment"]
    polarization_moment["expression"] = "theta_gamma2 + e_gamma0 + e_gamma2"
    perturbations["equations"]["evolve_theta_b"]["rhs"] = (
        "-Hconf * theta_b + k * k * sound_speed_sq * delta_b "
        "+ baryon_thomson_drag + k * k * Psi"
    )
    perturbations["collision_operators"] = {
        "thomson_drag": {
            "expression": (
                "collision_rate * ((theta_b / k) / 3.0 - theta_gamma1)"
            ),
            "counterpart": "baryon_thomson_drag",
            "integration_strategy": "exact",
            "activation_strategy": "tight_coupling",
            "rate_expression": "collision_rate",
            "exact_form": {
                "targets": [
                    {"kind": "photon_temperature_dipole"},
                    {"kind": "baryon_velocity_divergence"},
                    {"kind": "photon_temperature_quadrupole"},
                    {"kind": "photon_polarization_monopole"},
                    {"kind": "photon_polarization_dipole"},
                    {"kind": "photon_polarization_quadrupole"},
                ],
                "matrix": [
                    [
                        "-1.0",
                        "1.0 / (3.0 * k)",
                        "0.0",
                        "0.0",
                        "0.0",
                        "0.0",
                    ],
                    [
                        "3.0 * k * photon_baryon_momentum_ratio",
                        "-photon_baryon_momentum_ratio",
                        "0.0",
                        "0.0",
                        "0.0",
                        "0.0",
                    ],
                    ["0.0", "0.0", "-0.9", "0.1", "0.0", "0.1"],
                    ["0.0", "0.0", "0.5", "-0.5", "0.0", "0.5"],
                    ["0.0", "0.0", "0.0", "0.0", "-1.0", "0.0"],
                    ["0.0", "0.0", "0.1", "0.1", "0.0", "-0.9"],
                ],
                "damping_targets": [
                    {"kind": "photon_temperature_octopole"},
                    {"kind": "photon_polarization_octopole"},
                ],
                "damping_coefficient": "-1.0",
                "activation_strategy": "tight_coupling",
            },
        }
    }
    perturbations["conservation_rules"] = {
        "thomson_drag_balance": {
            "kind": "absolute_max",
            "expression": (
                "3.0 * k * photon_baryon_momentum_ratio * thomson_drag + "
                "baryon_thomson_drag"
            ),
            "tolerance": 1.0e-12,
            "domain": "scalar",
        }
    }
    if rename_map:
        contract["perturbations"] = _rename_declared_contract_tokens(
            perturbations,
            rename_map,
        )
    return contract


def _declared_graph_perturbations(
    *,
    baryon_rhs: str = (
        "-Hconf * theta_b + k * k * sound_speed_sq * delta_b "
        "+ collision_rate * (3.0 * k * theta_gamma1 - theta_b) "
        "+ k * k * Psi"
    ),
    photon_monopole_rhs: str = "-k * theta_gamma1 + (k * Psi) / 3.0",
    metric_closure_expression: str = "Phi",
    additive_source_expression: str = "0.0",
    include_bb: bool = False,
    include_lensing: bool = False,
    include_vector: bool = False,
) -> dict[str, object]:
    """Return a physically structured declared-math CMB graph."""

    perturbations: dict[str, object] = {
        "contract_version": 2,
        "gauge": "conformal_newtonian",
        "variables": {
            "theta_gamma0": {
                "kind": "photon_temperature_monopole",
                "tensor_character": "scalar_like",
            },
            "theta_gamma1": {
                "kind": "photon_temperature_dipole",
                "tensor_character": "scalar_like",
            },
            "theta_gamma2": {
                "kind": "photon_temperature_quadrupole",
                "tensor_character": "scalar_like",
            },
            "theta_gamma3": {
                "kind": "photon_temperature_octopole",
                "tensor_character": "scalar_like",
            },
            "e_gamma0": {
                "kind": "photon_polarization_monopole",
                "tensor_character": "scalar_like",
            },
            "e_gamma1": {
                "kind": "photon_polarization_dipole",
                "tensor_character": "scalar_like",
            },
            "e_gamma2": {
                "kind": "photon_polarization_quadrupole",
                "tensor_character": "scalar_like",
            },
            "e_gamma3": {
                "kind": "photon_polarization_octopole",
                "tensor_character": "scalar_like",
            },
            "delta_b": {"kind": "baryon_density_contrast"},
            "theta_b": {"kind": "baryon_velocity_divergence"},
            "delta_c": {"kind": "cdm_density_contrast"},
            "theta_c": {"kind": "cdm_velocity_divergence"},
            "delta_nu": {
                "kind": "massless_neutrino_density_contrast",
            },
            "theta_nu": {
                "kind": "massless_neutrino_velocity_divergence",
            },
            "sigma_nu": {
                "kind": "massless_neutrino_anisotropic_stress",
            },
            "nu_l3": {
                "kind": "massless_neutrino_multipole",
                "tensor_character": "scalar_like",
            },
            "Phi": {
                "kind": "metric_potential_phi",
                "gauge_role": "curvature_potential",
            },
            "Psi": {
                "kind": "metric_potential_psi",
                "gauge_role": "newtonian_potential",
            },
        },
        "derived": {
            "polarization_moment": {
                "expression": "theta_gamma2 + 6.0 * e_gamma2",
                "description": "Quadrupole source for polarization.",
            },
            "total_matter_density": {
                "expression": "Omega_c0 * delta_c + Omega_b0 * delta_b",
                "description": "Matter density source for the metric.",
            },
            "total_radiation_density": {
                "expression": (
                    "4.0 * Omega_gamma0 * theta_gamma0 + Omega_nu0 * delta_nu"
                ),
                "description": "Radiation density source for the metric.",
            },
            "total_momentum_density": {
                "expression": (
                    "Omega_b0 * theta_b + Omega_c0 * theta_c + "
                    "4.0 * Omega_gamma0 * theta_gamma1 + Omega_nu0 * theta_nu"
                ),
                "description": "Momentum source for the metric.",
            },
            "total_neutrino_shear": {
                "expression": "Omega_nu0 * sigma_nu",
                "description": "Neutrino shear source for the metric.",
            },
            "metric_denominator": {
                "expression": "k * k + 3.0 * Hconf * Hconf",
                "description": "Regularized Poisson denominator.",
            },
            "einstein_gravity_strength": {
                "expression": "H0_over_c_Mpc_inv * H0_over_c_Mpc_inv",
                "description": (
                    "Background gravity scale used by the scalar Einstein "
                    "constraints."
                ),
            },
        },
        "equations": {
            "evolve_theta_gamma0": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma0",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": photon_monopole_rhs,
                "role": "continuity",
            },
            "evolve_theta_gamma1": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma1",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(k / 3.0) * (theta_gamma0 + Psi - 2.0 * theta_gamma2) "
                    "+ collision_rate * ((theta_b / k) / 3.0 - theta_gamma1)"
                ),
                "role": "euler",
            },
            "evolve_theta_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(2.0 / 5.0) * k * theta_gamma1 "
                    "- (3.0 / 5.0) * k * theta_gamma3 "
                    "- collision_rate * "
                    "(theta_gamma2 - 0.1 * polarization_moment)"
                ),
                "role": "hierarchy",
            },
            "evolve_theta_gamma3": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma3",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(3.0 / 7.0) * k * theta_gamma2 "
                    "- (4.0 / 7.0) * k * theta_gamma3"
                ),
                "role": "hierarchy",
            },
            "evolve_e_gamma0": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma0",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-k * e_gamma1",
                "role": "polarization",
            },
            "evolve_e_gamma1": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma1",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "(k / 3.0) * (e_gamma0 - 2.0 * e_gamma2)",
                "role": "polarization",
            },
            "evolve_e_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(2.0 / 5.0) * k * e_gamma1 "
                    "- (3.0 / 5.0) * k * e_gamma3 "
                    "- collision_rate * "
                    "(e_gamma2 - 0.1 * polarization_moment)"
                ),
                "role": "polarization",
            },
            "evolve_e_gamma3": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma3",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(3.0 / 7.0) * k * e_gamma2 "
                    "- (4.0 / 7.0) * k * e_gamma3"
                ),
                "role": "polarization",
            },
            "evolve_delta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_b",
                "role": "continuity",
            },
            "evolve_theta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": baryon_rhs,
                "role": "euler",
            },
            "evolve_delta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_c",
                "role": "continuity",
            },
            "evolve_theta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-Hconf * theta_c + k * k * Psi",
                "role": "euler",
            },
            "evolve_delta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-(4.0 / 3.0) * theta_nu",
                "role": "continuity",
            },
            "evolve_theta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": ("k * k * (0.25 * delta_nu - sigma_nu + Psi)"),
                "role": "euler",
            },
            "evolve_sigma_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(4.0 / 15.0) * theta_nu " "- (3.0 / 5.0) * k * nu_l3"
                ),
                "role": "hierarchy",
            },
            "evolve_nu_l3": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "nu_l3",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(3.0 / 7.0) * k * sigma_nu " "- (4.0 / 7.0) * k * nu_l3"
                ),
                "role": "hierarchy",
            },
        },
        "constraints": {
            "phi_constraint": {
                "target": "Phi",
                "expression": (
                    "-1.5 * einstein_gravity_strength * "
                    "(total_matter_density + total_radiation_density) "
                    "/ metric_denominator"
                ),
                "role": "constraint",
            }
        },
        "closures": {
            "psi_closure": {
                "target": "Psi",
                "expression": (
                    "("
                    + metric_closure_expression
                    + ") - 3.0 * einstein_gravity_strength * "
                    "Omega_nu0 * sigma_nu / metric_denominator"
                ),
                "role": "closure",
            }
        },
        "sources": {
            "temperature_monopole": {
                "expression": (
                    "visibility * (theta_gamma0 + Psi "
                    "+ 0.25 * polarization_moment)"
                ),
                "role": "monopole",
            },
            "temperature_doppler": {
                "expression": "visibility * 3.0 * theta_gamma1",
                "role": "doppler",
            },
            "temperature_isw": {
                "expression": "exp(-tau) * (Psi - Phi)",
                "role": "isw",
            },
            "polarization_source": {
                "expression": "0.75 * visibility * polarization_moment",
                "role": "polarization",
            },
            "temperature_additive": {
                "expression": additive_source_expression,
                "role": "additive",
            },
        },
        "observables": {
            "temperature": {
                "kind": "transfer_component",
                "projection": "line_of_sight_temperature",
                "source_terms": {
                    "monopole": "temperature_monopole",
                    "doppler": "temperature_doppler",
                    "isw": "temperature_isw",
                    "additive": "temperature_additive",
                },
            },
            "polarization_e": {
                "kind": "transfer_component",
                "projection": "line_of_sight_polarization_e",
                "source_terms": {"polarization": "polarization_source"},
            },
            "TT": {
                "kind": "angular_power_spectrum",
                "primary": "temperature",
                "secondary": "temperature",
            },
            "TE": {
                "kind": "angular_power_spectrum",
                "primary": "temperature",
                "secondary": "polarization_e",
            },
            "EE": {
                "kind": "angular_power_spectrum",
                "primary": "polarization_e",
                "secondary": "polarization_e",
            },
        },
        "initial_conditions": {
            "theta_gamma3_seed": {
                "target": {
                    "variable": "theta_gamma3",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "theta_gamma0_seed": {
                "target": {
                    "variable": "theta_gamma0",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-0.5 * seed",
            },
            "theta_gamma1_seed": {
                "target": {
                    "variable": "theta_gamma1",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "theta_gamma2_seed": {
                "target": {
                    "variable": "theta_gamma2",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": (
                    "(k * eta_initial) * (k * eta_initial) * seed / 30.0"
                ),
            },
            "e_gamma2_seed": {
                "target": {
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "e_gamma0_seed": {
                "target": {
                    "variable": "e_gamma0",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "e_gamma1_seed": {
                "target": {
                    "variable": "e_gamma1",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "e_gamma3_seed": {
                "target": {
                    "variable": "e_gamma3",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "delta_b_seed": {
                "target": {
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-1.5 * seed",
            },
            "theta_b_seed": {
                "target": {
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "delta_c_seed": {
                "target": {
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-1.5 * seed",
            },
            "theta_c_seed": {
                "target": {
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "delta_nu_seed": {
                "target": {
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-2.0 * seed",
            },
            "theta_nu_seed": {
                "target": {
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "sigma_nu_seed": {
                "target": {
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": (
                    "(k * eta_initial) * (k * eta_initial) * seed / 15.0"
                ),
            },
            "nu_l3_seed": {
                "target": {
                    "variable": "nu_l3",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
        },
        "boundary_conditions": {},
        "numerics": {
            "ell_min": 20,
            "ell_max": 90,
            "k_min": 1.0e-4,
            "k_max": 0.3,
            "k_sample_count": 10,
            "eta_sample_count": 768,
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
            "tight_coupling_ratio": 50.0,
            "a_min": 1.0e-6,
            "source_grid_multiplier": 2,
            "initial_redshift": 2.0e4,
        },
        "validity": {
            "regimes": ["linear", "scalar_like"],
            "notes": "Synthetic declared graph for runtime tests.",
        },
    }
    if include_bb:
        perturbations["variables"]["tensor_b"] = {
            "kind": "custom_tensor_polarization_source",
            "rank": 2,
            "spin": 2.0,
            "parity": "odd",
            "tensor_character": "tensor_like",
        }
        perturbations["equations"]["evolve_tensor_b"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": ("0.2 * k * polarization_moment - 0.4 * Hconf * tensor_b"),
            "role": "odd_parity_polarization",
        }
        perturbations["initial_conditions"]["tensor_b_seed"] = {
            "target": {
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "(k * eta_initial) * seed / 120.0",
        }
        perturbations["sources"]["polarization_b_source"] = {
            "expression": "visibility * tensor_b",
            "role": "polarization_b",
        }
        perturbations["observables"]["polarization_b"] = {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {
                "polarization_b": "polarization_b_source",
            },
        }
        perturbations["observables"]["BB"] = {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
        }
    if include_lensing:
        perturbations["sources"]["lensing_potential"] = {
            "expression": "exp(-tau) * (Phi + Psi)",
            "role": "potential",
        }
        perturbations["observables"]["lensing_potential"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_lensing_potential",
            "source_terms": {"potential": "lensing_potential"},
        }
        perturbations["observables"]["PP"] = {
            "kind": "angular_power_spectrum",
            "primary": "lensing_potential",
            "secondary": "lensing_potential",
        }
        perturbations["observables"]["TP"] = {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "lensing_potential",
        }
        perturbations["observables"]["EP"] = {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "lensing_potential",
        }
    if include_vector:
        perturbations["variables"]["vector_signal"] = {
            "kind": "custom_vector_mode",
            "spin": 1.0,
            "parity": "even",
            "tensor_character": "vector_like",
        }
        perturbations["equations"]["evolve_vector_signal"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "vector_signal",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": ("0.15 * k * theta_gamma1 - 0.25 * Hconf * vector_signal"),
            "role": "vector_coupling",
        }
        perturbations["initial_conditions"]["vector_signal_seed"] = {
            "target": {
                "variable": "vector_signal",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "(k * eta_initial) * seed / 90.0",
        }
        perturbations["sources"]["vector_source"] = {
            "expression": "visibility * vector_signal",
            "role": "signal",
        }
        perturbations["observables"]["vector_signal"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_signal",
            "source_terms": {"signal": "vector_source"},
        }
        perturbations["observables"]["VV"] = {
            "kind": "angular_power_spectrum",
            "primary": "vector_signal",
            "secondary": "vector_signal",
        }
    return perturbations


def _declared_background() -> dict[str, object]:
    """Return the declared background and reionization contract."""

    return {
        "derived": {
            "h": "H0 / 100.0",
            "H0_over_c_Mpc_inv": "H0 / 299792.458",
            "Omega_k0": "0.0",
            "Tcmb": "Tcmb_K",
            "Omega_b0": "ombh2 / (h * h)",
            "Omega_c0": "omch2 / (h * h)",
            "Omega_gamma0": ("2.469e-5 * ((Tcmb_K / 2.7255) ** 4) / (h * h)"),
            "Omega_nu0": "0.22710731766 * Neff * Omega_gamma0",
            "Omega_r0": "Omega_gamma0 + Omega_nu0",
            "Omega_m0": "Omega_b0 + Omega_c0",
            "Omega_de0": "1.0 - Omega_m0 - Omega_r0 - Omega_k0",
            "H": (
                "H0 * sqrt("
                "Omega_r0 / (a ** 4) + "
                "Omega_m0 / (a ** 3) + "
                "Omega_k0 / (a ** 2) + "
                "Omega_de0"
                ")"
            ),
        },
        "reionization": {
            "calibration": {
                "symbol": "reionization_log10_amplitude",
                "target_optical_depth": "tau",
                "lower": -24.0,
                "upper": 32.0,
            },
            "quantities": {
                "stellar_temperature_K": 5.0e4,
                "quasar_temperature_K": 1.5e5,
                "hydrogen_temperature_K": 1.0e4,
                "helium_temperature_K": 1.0e4,
                "helium_double_temperature_K": 2.0e4,
                "collapse_threshold": 1.686,
                "collapse_source": ("exp(-collapse_threshold / (a + 1.0e-6))"),
                "hard_source": "collapse_source * collapse_source",
                "stellar_helium_hardness": (
                    "exp(-("
                    "(24.587387 - 13.605693122994) * 1.602176634e-19"
                    ") / (1.380649e-23 * stellar_temperature_K))"
                ),
                "quasar_helium_hardness": (
                    "exp(-("
                    "(54.417763 - 13.605693122994) * 1.602176634e-19"
                    ") / (1.380649e-23 * quasar_temperature_K))"
                ),
                "hydrogen_ionization_rate": (
                    "(10 ** reionization_log10_amplitude) "
                    "* H_SI * collapse_source"
                ),
                "helium_ionization_rate": (
                    "hydrogen_ionization_rate * stellar_helium_hardness"
                ),
                "helium_double_ionization_rate": (
                    "(10 ** reionization_log10_amplitude) "
                    "* H_SI * hard_source * quasar_helium_hardness"
                ),
            },
        },
    }


def _base_custom_cmb_contract(
    **perturbation_kwargs: object,
) -> dict[str, object]:
    """Return a synthetic non-standard CMB contract used by the tests."""

    return {
        "model_name": "SyntheticCustomCMB",
        "param_map": {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
        },
        "model_parameters": {
            "Tcmb_K": 2.7255,
        },
        "background": _declared_background(),
        "grids": {},
        "values": {},
        "calls": [],
        "perturbations": _declared_graph_perturbations(
            **perturbation_kwargs,
        ),
        "numerical": {
            "ell_min": 20,
            "ell_max": 90,
            "k_min": 1.0e-4,
            "k_max": 0.3,
            "k_sample_count": 10,
            "eta_sample_count": 768,
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
            "tight_coupling_ratio": 50.0,
            "a_min": 1.0e-6,
            "source_grid_multiplier": 2,
            "initial_redshift": 2.0e4,
        },
    }


def _camb_reference_contract(
    param_map: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return an explicit test-only CAMB reference contract."""

    reference_params = (
        {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
        }
        if param_map is None
        else dict(param_map)
    )
    return {
        "backend": "camb",
        "param_map": reference_params,
        "grids": {},
        "values": {},
        "calls": [],
    }


def _custom_contract(**perturbation_kwargs: object) -> dict[str, object]:
    """Return a deep-copied non-standard CMB fixture."""

    return copy.deepcopy(_base_custom_cmb_contract(**perturbation_kwargs))


def _speedup_contract(contract: dict[str, object]) -> dict[str, object]:
    """Return ``contract`` with lighter numerics for governed behavior tests.

    Fast runtime-response coverage uses this helper so the full governed suite
    stays integral while the separate scientific reference checks keep their
    production-like numerics unchanged.
    """

    contract["numerical"].update(
        {
            "eta_sample_count": 128,
            "evolution_eta_sample_count": 64,
            "source_grid_multiplier": 1,
        }
    )
    return contract


def _prepare_declared_contract(
    contract: dict[str, object],
) -> dict[str, object]:
    """Return ``contract`` with its declared runtime prepared upstream."""

    return model_coder.prepare_declared_cmb_execution_contract(
        copy.deepcopy(contract)
    )


def _with_prepared_numerical_overrides(
    contract: Mapping[str, object],
    **overrides: object,
) -> dict[str, object]:
    """Return a rebound request without mutating frozen runtime assets."""

    rebound = dict(contract)
    numerical = dict(contract.get("numerical", {}) or {})
    numerical.update(overrides)
    rebound["numerical"] = numerical
    return rebound


def _ensure_prepared_declared_contract(
    contract: dict[str, object],
) -> dict[str, object]:
    """Return a prepared declared contract from raw or prepared input."""

    if contract.get("perturbation_data") is not None:
        return copy.deepcopy(contract)
    return _prepare_declared_contract(contract)


def _raw_declared_spectrum_data(
    contract: dict[str, object],
    ells: numpy.ndarray,
) -> cmb_projection.CustomCMBSpectrumData:
    """Return unclipped declared spectrum data for one declared contract."""

    return cmb_projection._compute_custom_cmb_spectrum_data(
        _ensure_prepared_declared_contract(contract),
        numpy.asarray(ells, dtype=int),
    )


def _raw_declared_public_spectra(
    contract: dict[str, object],
    ells: numpy.ndarray,
    *,
    spectra: tuple[str, ...],
) -> dict[str, numpy.ndarray]:
    """Return unclipped public spectra for one prepared declared contract."""

    prepared = _ensure_prepared_declared_contract(contract)
    requested_ell_grid = numpy.asarray(tuple(ells), dtype=int)
    canonical_requested = tuple(
        cmb_solver._canonical_spectrum_name(name) for name in spectra
    )
    needs_lensing = any(
        cmb_solver._is_lensed_requested_spectrum(spectrum_name)
        for spectrum_name in canonical_requested
    )
    if needs_lensing:
        analysis_ell_grid = numpy.arange(
            int(requested_ell_grid.max()) + 1,
            dtype=int,
        )
        output_indices = requested_ell_grid
    else:
        analysis_ell_grid = requested_ell_grid
        output_indices = numpy.arange(requested_ell_grid.size, dtype=int)
    custom_data = cmb_projection._compute_custom_cmb_spectrum_data(
        prepared,
        analysis_ell_grid,
        requested_spectra=cmb_solver._requested_base_spectra(
            canonical_requested
        ),
    )
    ell_grid = numpy.asarray(custom_data.ell_grid, dtype=numpy.longdouble)
    ell_factor = (
        ell_grid * (ell_grid + 1.0) / (2.0 * numpy.longdouble(numpy.pi))
    )
    t_cmb_muK = numpy.longdouble("2.7255e6")
    perturbation_data = prepared["perturbation_data"]
    spectra_results: dict[str, numpy.ndarray] = {}
    for spectrum_name, spectrum_values in custom_data.spectra.items():
        canonical_name = cmb_solver._canonical_spectrum_name(spectrum_name)
        scale = numpy.asarray(
            cmb_solver._power_spectrum_scale_factor(
                perturbation_data,
                canonical_name,
                ell_factor=ell_factor,
                t_cmb_muK=float(t_cmb_muK),
                lensing_mode=needs_lensing,
            ),
            dtype=numpy.longdouble,
        )
        spectra_results[canonical_name] = scale * numpy.asarray(
            spectrum_values, dtype=numpy.longdouble
        )
    if needs_lensing:
        lensing_inputs = cmb_solver._normalize_lensing_input_spectra(
            spectra_results
        )
        lmax = int(numpy.max(numpy.asarray(custom_data.ell_grid, dtype=int)))
        base_cls = numpy.zeros((lmax + 1, 4), dtype=numpy.longdouble)
        base_cls[:, 0] = numpy.asarray(
            lensing_inputs["TT"][: lmax + 1],
            dtype=numpy.longdouble,
        )
        base_cls[:, 1] = numpy.asarray(
            lensing_inputs["EE"][: lmax + 1],
            dtype=numpy.longdouble,
        )
        base_cls[:, 2] = numpy.asarray(
            lensing_inputs.get(
                "BB",
                numpy.zeros(lmax + 1, dtype=numpy.longdouble),
            )[: lmax + 1],
            dtype=numpy.longdouble,
        )
        base_cls[:, 3] = numpy.asarray(
            lensing_inputs["TE"][: lmax + 1],
            dtype=numpy.longdouble,
        )
        clpp = cmb_solver._lensing_potential_clpp(
            numpy.asarray(
                lensing_inputs["PP"][: lmax + 1],
                dtype=numpy.longdouble,
            )
        )
        lensed_cls = cmb_solver._lensed_cls(
            base_cls,
            clpp,
            lmax=lmax,
            lmax_lensed=lmax,
        )
        spectra_results.update(
            {
                "lensed_TT": numpy.asarray(
                    lensed_cls[:, 0], dtype=numpy.longdouble
                ),
                "lensed_EE": numpy.asarray(
                    lensed_cls[:, 1], dtype=numpy.longdouble
                ),
                "lensed_BB": numpy.asarray(
                    lensed_cls[:, 2], dtype=numpy.longdouble
                ),
                "lensed_TE": numpy.asarray(
                    lensed_cls[:, 3], dtype=numpy.longdouble
                ),
            }
        )
    return {
        original_name: numpy.asarray(
            spectra_results[canonical_name],
            dtype=numpy.longdouble,
        )[output_indices]
        for original_name, canonical_name in zip(spectra, canonical_requested)
        if canonical_name in spectra_results
    }


def _capture_visible_scalar_monopole_history(
    contract: dict[str, object],
    *,
    ells: Sequence[int] = (40,),
    spectra: tuple[str, ...] = ("TT",),
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, numpy.ndarray]]:
    """Return a scalar history and spectra captured from one solver run."""

    cache.clear_cmb_parameter_caches()
    captured: list[tuple[numpy.ndarray, numpy.ndarray]] = []
    original = cmb_projection._evaluate_compiled_expression_noerr

    def _capture_monopole_history(
        expression_data: object,
        env: Mapping[str, object],
    ) -> object:
        """Record the visible monopole history once."""

        expression = str(getattr(expression_data, "expression", ""))
        if (
            expression
            in {
                "visibility * (observable_theta_gamma0 + Psi)",
                "visibility * (observable_theta_gamma0 + Psi "
                "+ 0.25 * polarization_moment)",
            }
            and not captured
        ):
            captured.append(
                (
                    numpy.asarray(env["eta"], dtype=float).copy(),
                    numpy.asarray(
                        env["theta_gamma0"],
                        dtype=float,
                    ).copy(),
                )
            )
        return original(expression_data, env)

    with mock.patch.object(
        cmb_projection,
        "_evaluate_compiled_expression_noerr",
        side_effect=_capture_monopole_history,
    ):
        public_spectra = _raw_declared_public_spectra(
            contract,
            numpy.asarray(tuple(ells), dtype=int),
            spectra=spectra,
        )

    if len(captured) != 1:
        raise AssertionError("Expected one visible scalar monopole history")
    return captured[0][0], captured[0][1], public_spectra


def _capture_tensor_source_histories(
    contract: dict[str, object],
) -> dict[tuple[str, float], numpy.ndarray]:
    """Return tensor source histories keyed by source name and k mode."""

    cache.clear_cmb_result_caches()
    perturbation_data = contract["perturbation_data"]
    expression_names = {
        entry.expression: str(name)
        for name, entry in perturbation_data.sources.items()
    }
    captured: dict[tuple[str, float], numpy.ndarray] = {}
    original = cmb_projection._evaluate_compiled_expression_noerr

    def _capture_source_history(
        expression_data: object,
        env: Mapping[str, object],
    ) -> object:
        """Record each evaluated tensor source history once."""

        value = original(expression_data, env)
        expression = getattr(expression_data, "expression", "")
        values = numpy.asarray(value)
        if expression in expression_names and values.ndim == 1:
            key = (
                expression_names[expression],
                round(float(env["k"]), 15),
            )
            captured.setdefault(key, numpy.asarray(values, dtype=float).copy())
        return value

    with mock.patch.object(
        cmb_projection,
        "_evaluate_compiled_expression_noerr",
        side_effect=_capture_source_history,
    ):
        cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.asarray((40,), dtype=int),
            requested_spectra=("TT", "EE", "BB"),
        )

    if not captured:
        raise AssertionError("Expected tensor source histories")
    return captured


def _resolved_declared_scalar_context(
    contract: dict[str, object],
    *,
    a_value: float = 0.5,
    state_updates: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return one resolved scalar runtime context for a prepared contract."""

    perturbation_data = contract["perturbation_data"]
    physical_params = cmb_background._resolve_custom_cmb_physical_parameters(
        contract
    )
    context = cmb_background._physical_runtime_scalars(physical_params)
    for source_name in ("param_map", "model_parameters"):
        source = contract.get(source_name, {}) or {}
        for name, value in source.items():
            if isinstance(value, (int, float, numpy.integer, numpy.floating)):
                context[str(name)] = float(value)
    context.update(
        {
            "a": float(a_value),
            "z": (1.0 / float(a_value)) - 1.0,
            "eta": 1.0,
            "H": 120.0,
            "Hconf": 0.02,
            "Hconf_tau": -2.0e-4,
            "tau": 0.4,
            "tau_dot": -0.08,
            "visibility": 0.03,
            "chi": 13_800.0,
            "angular_diameter_distance": 13_100.0,
            "sound_speed": 3.0**-0.5,
            "sound_speed_sq": 1.0 / 3.0,
            "baryon_sound_speed_sq": 1.0e-9,
            "collision_rate": 0.12,
            "free_streaming": 1.0,
            "tight_coupling_drag": 0.08,
            "sound_horizon": 145.0,
            "k": 0.1,
            "seed": 1.0,
            "delta_b": 0.3,
            "theta_b": 0.04,
            "delta_c": 0.25,
            "theta_c": 0.03,
            "theta_gamma0": 0.2,
            "theta_gamma1": 0.015,
            "theta_gamma2": 0.01,
            "theta_gamma3": 0.005,
            "e_gamma0": 0.0,
            "e_gamma1": 0.0,
            "e_gamma2": 0.002,
            "e_gamma3": 0.001,
            "delta_nu": 0.18,
            "theta_nu": 0.035,
            "sigma_nu": 0.01,
            "nu_l3": 0.004,
            "Phi": 0.02,
            "Phi_gi": 0.02,
            "Psi": 0.018,
            "gauge_shift_alpha_tau": 0.0177,
        }
    )
    if state_updates:
        context.update(
            {str(name): float(value) for name, value in state_updates.items()}
        )
    context.update(
        evolution._declared_momentum_grid_context(
            perturbation_data,
            model_parameters=contract["param_map"],
            physical_params=physical_params,
            scale_factor=float(a_value),
        )
    )
    execution_plan = evolution._compile_declared_graph_execution_plan(
        perturbation_data
    )
    if (
        getattr(perturbation_data, "gauge", "") == "synchronous"
        and state_updates is None
    ):
        provisional = evolution._resolve_declared_graph_context(
            dict(context),
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
        )
        gauge_shift_alpha = 0.015
        phi_value = float(provisional["Phi"])
        context.update(
            {
                "gauge_shift_alpha": gauge_shift_alpha,
                "eta_sync_metric": (
                    phi_value + float(context["Hconf"]) * gauge_shift_alpha
                ),
                "h_sync_metric": (
                    float(context["k"])
                    * float(context["eta"])
                    * float(context["k"])
                    * float(context["eta"])
                    * phi_value
                ),
            }
        )
    return evolution._resolve_declared_graph_context(
        context,
        perturbation_data,
        allow_partial=True,
        eta_grid=None,
        execution_plan=execution_plan,
    )


def _declared_scalar_hierarchy_contract(
    *,
    gauge: str = "conformal_newtonian",
    initial_mode: str = "adiabatic_scalar",
    include_massive_neutrino: bool = False,
    include_lensing: bool = False,
    sum_mnu: float = 0.06,
) -> dict[str, object]:
    """Return a scalar declared hierarchy fixture."""

    numerics = {
        "ell_min": 20,
        "ell_max": 120,
        "k_min": 1.0e-4,
        "k_max": 0.3,
        "k_sample_count": 18,
        "eta_sample_count": 192,
        "evolution_eta_sample_count": 128,
        "evolution_phase_step": 2.0,
        "ode_rtol": 1.0e-5,
        "ode_atol": 1.0e-8,
        "tight_coupling_ratio": 80.0,
        "a_min": 1.0e-6,
        "source_grid_multiplier": 1,
        "initial_redshift": 2.0e4,
        "photon_hierarchy_l_max": 8,
        "neutrino_hierarchy_l_max": 5,
    }
    if include_massive_neutrino:
        numerics["massive_neutrino_hierarchy_l_max"] = 5
        numerics["momentum_grids"] = {
            "massive_neutrino_default": {
                "count": 8,
                "q_min": 0.05,
                "q_max": 18.0,
                "mass_parameter": "sum_mnu",
            }
        }
    initial_condition_families = {
        initial_mode: {
            "sector": "scalar",
            "members": [],
        }
    }
    scalar_species = [
        "photon",
        "baryon",
        "cdm",
        "massless_neutrino",
    ]
    scalar_hierarchy_families = [
        "photon_temperature",
        "photon_polarization_e",
        "massless_neutrino",
    ]
    species = {
        "photon": {
            "sector": "scalar",
            "hierarchy_family": "photon_temperature",
            "collision_operators": ["thomson_drag"],
            "background_reference": "Omega_gamma0",
        },
        "baryon": {
            "sector": "scalar",
            "collision_operators": ["thomson_drag"],
            "background_reference": "Omega_b0",
        },
        "cdm": {
            "sector": "scalar",
            "background_reference": "Omega_c0",
        },
        "massless_neutrino": {
            "sector": "scalar",
            "hierarchy_family": "massless_neutrino",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        },
    }
    hierarchy_families = {
        "photon_temperature": {
            "sector": "scalar",
            "species": ["photon"],
            "closure": "free_streaming_scalar",
            "default_l_max": 8,
            "multipole_symbol": "theta_gamma_l",
        },
        "photon_polarization_e": {
            "sector": "scalar",
            "species": ["photon"],
            "closure": "free_streaming_scalar",
            "default_l_max": 8,
            "multipole_symbol": "e_gamma_l",
        },
        "massless_neutrino": {
            "sector": "scalar",
            "species": ["massless_neutrino"],
            "closure": "free_streaming_scalar",
            "default_l_max": 5,
            "multipole_symbol": "nu_l",
        },
    }
    if include_massive_neutrino:
        scalar_species.append("massive_neutrino")
        scalar_hierarchy_families.append("massive_neutrino")
        species["massive_neutrino"] = {
            "sector": "scalar",
            "hierarchy_family": "massive_neutrino",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        }
        hierarchy_families["massive_neutrino"] = {
            "sector": "scalar",
            "species": ["massive_neutrino"],
            "closure": "free_streaming_scalar",
            "default_l_max": 5,
            "multipole_symbol": "nu_massive_l",
            "momentum_grid": "massive_neutrino_default",
        }
    if include_lensing:
        hierarchy_families["lensing_potential"] = {
            "sector": "scalar",
            "species": ["photon", "massless_neutrino"],
            "closure": "line_of_sight_lensing",
            "default_l_max": 8,
            "multipole_symbol": "phi_l",
        }
    background = _declared_background()
    if include_massive_neutrino:
        background_derived = dict(background["derived"])
        background_derived.update(
            {
                "Omega_nu_massive0": ("sum_mnu / (93.14 * h * h)"),
                "massive_neutrino_transition_a": (
                    "(3.151 * ((4.0 / 11.0) ** (1.0 / 3.0)) * "
                    "8.617333262145e-5 * Tcmb_K) / "
                    "(sum_mnu / num_massive_neutrinos + 1.0e-30)"
                ),
                "Omega_nu_massive_rel0": (
                    "Omega_nu_massive0 * massive_neutrino_transition_a"
                ),
                "Omega_nu_massless0": (
                    "0.5 * (Omega_nu0 - Omega_nu_massive_rel0 + "
                    "abs(Omega_nu0 - Omega_nu_massive_rel0))"
                ),
                "Omega_de0": ("1.0 - Omega_m0 - Omega_r0 - Omega_nu_massive0"),
                "H": (
                    "H0 * sqrt("
                    "(Omega_gamma0 + Omega_nu_massless0) / (a ** 4) + "
                    "Omega_nu_massive0 * sqrt("
                    "a * a + massive_neutrino_transition_a * "
                    "massive_neutrino_transition_a) / "
                    "(sqrt(1.0 + massive_neutrino_transition_a * "
                    "massive_neutrino_transition_a) * (a ** 4)) + "
                    "Omega_m0 / (a ** 3) + "
                    "Omega_k0 / (a ** 2) + Omega_de0"
                    ")"
                ),
            }
        )
        background["derived"] = background_derived
    return {
        "model_name": "DeclaredScalarHierarchy",
        "param_map": {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
            "sum_mnu": sum_mnu,
            "num_massive_neutrinos": 3,
        },
        "model_parameters": {
            "Tcmb_K": 2.7255,
        },
        "background": background,
        "grids": {},
        "values": {},
        "calls": [],
        "numerical": dict(numerics),
        "perturbations": {
            "contract_version": 2,
            "gauge": gauge,
            "variables": {},
            "derived": {},
            "equations": {},
            "constraints": {},
            "closures": {},
            "collision_operators": {
                "thomson_drag": {
                    "sector": "scalar",
                    "species": ["photon", "baryon"],
                    "expression": (
                        "collision_rate * "
                        "((theta_b / acoustic_k) / 3.0 - theta_gamma1)"
                    ),
                    "counterpart": "baryon_thomson_drag",
                }
            },
            "conservation_rules": {
                "thomson_drag_balance": {
                    "kind": "absolute_max",
                    "expression": (
                        "3.0 * acoustic_k * "
                        "photon_baryon_momentum_ratio * thomson_drag + "
                        "baryon_thomson_drag"
                    ),
                    "tolerance": 1.0e-12,
                    "domain": "scalar",
                }
            },
            "initial_conditions": {},
            "initial_condition_families": initial_condition_families,
            "boundary_conditions": {},
            "sectors": {
                "scalar": {
                    "description": "Declared scalar hierarchy sector.",
                    "species": scalar_species,
                    "hierarchy_families": scalar_hierarchy_families,
                    "supported_gauges": [
                        "conformal_newtonian",
                        "synchronous",
                    ],
                    "tensor_character": "scalar_like",
                }
            },
            "species": species,
            "hierarchy_families": hierarchy_families,
            "projection_typing": {
                "temperature_line_of_sight": {
                    "sector": "scalar",
                    "kernel": "temperature_mixed_window",
                    "source_roles": ["monopole", "doppler", "isw"],
                    "observable_kinds": ["transfer_component"],
                    "parity": "even",
                    "spin": 0.0,
                }
            },
            "accuracy_controls": {
                "scalar_reference_ells": [20, 60, 120],
                "runtime_envelope": "bounded",
            },
            "sources": {},
            "observables": {},
            "numerics": dict(numerics),
            "validity": {
                "regimes": ["linear", "declared_scalar_hierarchy"],
                "notes": "Metadata-only declared scalar hierarchy route.",
            },
        },
    }


def _declared_vector_hierarchy_contract() -> dict[str, object]:
    """Return a vector declared hierarchy fixture."""

    numerics = {
        "ell_min": 20,
        "ell_max": 120,
        "k_min": 1.0e-4,
        "k_max": 0.3,
        "k_sample_count": 18,
        "eta_sample_count": 192,
        "evolution_eta_sample_count": 128,
        "evolution_phase_step": 2.0,
        "ode_rtol": 1.0e-5,
        "ode_atol": 1.0e-8,
        "tight_coupling_ratio": 80.0,
        "a_min": 1.0e-6,
        "source_grid_multiplier": 1,
        "initial_redshift": 2.0e4,
        "photon_hierarchy_l_max": 8,
        "photon_polarization_hierarchy_l_max": 8,
        "neutrino_hierarchy_l_max": 5,
    }
    vector_species = [
        "photon",
        "baryon",
        "cdm",
        "massless_neutrino",
    ]
    vector_hierarchy_families = [
        "photon_temperature_vector",
        "photon_polarization_e_vector",
        "photon_polarization_b_vector",
        "massless_neutrino_vector",
    ]
    return {
        "model_name": "DeclaredVectorHierarchy",
        "param_map": {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
            "sum_mnu": 0.06,
            "num_massive_neutrinos": 3,
        },
        "model_parameters": {
            "Tcmb_K": 2.7255,
        },
        "background": _declared_background(),
        "grids": {},
        "values": {},
        "calls": [],
        "numerical": dict(numerics),
        "perturbations": {
            "contract_version": 2,
            "gauge": "conformal_newtonian",
            "variables": {},
            "derived": {},
            "equations": {},
            "constraints": {},
            "closures": {},
            "collision_operators": {},
            "conservation_rules": {},
            "initial_conditions": {},
            "initial_condition_families": {
                "regular_vector_mode": {
                    "sector": "vector",
                    "members": [],
                }
            },
            "boundary_conditions": {},
            "sectors": {
                "vector": {
                    "description": "Declared vector hierarchy sector.",
                    "species": vector_species,
                    "hierarchy_families": vector_hierarchy_families,
                    "supported_gauges": ["conformal_newtonian"],
                    "tensor_character": "vector_like",
                }
            },
            "species": {
                "photon": {
                    "sector": "vector",
                    "hierarchy_family": "photon_temperature_vector",
                    "collision_operators": ["thomson_vector_drag"],
                    "background_reference": "Omega_gamma0",
                },
                "baryon": {
                    "sector": "vector",
                    "collision_operators": ["thomson_vector_drag"],
                    "background_reference": "Omega_b0",
                },
                "cdm": {
                    "sector": "vector",
                    "background_reference": "Omega_c0",
                },
                "massless_neutrino": {
                    "sector": "vector",
                    "hierarchy_family": "massless_neutrino_vector",
                    "background_reference": "Omega_nu0",
                    "anisotropic_stress": "supported",
                },
            },
            "hierarchy_families": {
                "photon_temperature_vector": {
                    "sector": "vector",
                    "species": ["photon"],
                    "closure": "free_streaming_vector",
                    "default_l_max": 8,
                    "multipole_symbol": "theta_gamma_vl",
                },
                "photon_polarization_e_vector": {
                    "sector": "vector",
                    "species": ["photon"],
                    "closure": "free_streaming_vector",
                    "default_l_max": 8,
                    "multipole_symbol": "e_gamma_vl",
                },
                "photon_polarization_b_vector": {
                    "sector": "vector",
                    "species": ["photon"],
                    "closure": "free_streaming_vector",
                    "default_l_max": 8,
                    "multipole_symbol": "b_gamma_vl",
                },
                "massless_neutrino_vector": {
                    "sector": "vector",
                    "species": ["massless_neutrino"],
                    "closure": "free_streaming_vector",
                    "default_l_max": 5,
                    "multipole_symbol": "nu_vl",
                },
            },
            "projection_typing": {},
            "accuracy_controls": {
                "vector_reference_ells": [20, 60, 120],
                "runtime_envelope": "bounded",
            },
            "sources": {},
            "observables": {},
            "numerics": dict(numerics),
            "validity": {
                "regimes": ["linear", "declared_vector_hierarchy"],
                "notes": "Metadata-only declared vector hierarchy route.",
            },
        },
    }


def _declared_tensor_hierarchy_contract() -> dict[str, object]:
    """Return a tensor declared hierarchy fixture."""

    numerics = {
        "ell_min": 20,
        "ell_max": 120,
        "k_min": 1.0e-4,
        "k_max": 0.3,
        "k_sample_count": 18,
        "eta_sample_count": 192,
        "evolution_eta_sample_count": 128,
        "evolution_phase_step": 2.0,
        "ode_rtol": 1.0e-5,
        "ode_atol": 1.0e-8,
        "a_min": 1.0e-6,
        "source_grid_multiplier": 1,
        "initial_redshift": 2.0e4,
        "photon_hierarchy_l_max": 8,
        "photon_polarization_hierarchy_l_max": 8,
        "neutrino_hierarchy_l_max": 5,
    }
    tensor_species = ["photon", "massless_neutrino"]
    tensor_hierarchy_families = [
        "photon_temperature_tensor",
        "photon_polarization_e_tensor",
        "photon_polarization_b_tensor",
        "massless_neutrino_tensor",
    ]
    return {
        "model_name": "DeclaredTensorHierarchy",
        "param_map": {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
            "sum_mnu": 0.0,
            "num_massive_neutrinos": 3,
            "r": 0.1,
            "nt": 0.0,
        },
        "model_parameters": {
            "Tcmb_K": 2.7255,
        },
        "background": _declared_background(),
        "grids": {},
        "values": {},
        "calls": [],
        "numerical": dict(numerics),
        "perturbations": {
            "contract_version": 2,
            "gauge": "conformal_newtonian",
            "variables": {},
            "derived": {},
            "equations": {},
            "constraints": {},
            "closures": {},
            "collision_operators": {},
            "conservation_rules": {},
            "initial_conditions": {},
            "initial_condition_families": {
                "tensor_mode": {
                    "sector": "tensor",
                    "members": [],
                }
            },
            "boundary_conditions": {},
            "sectors": {
                "tensor": {
                    "description": "Declared tensor hierarchy sector.",
                    "species": tensor_species,
                    "hierarchy_families": tensor_hierarchy_families,
                    "supported_gauges": ["conformal_newtonian"],
                    "tensor_character": "tensor_like",
                }
            },
            "species": {
                "photon": {
                    "sector": "tensor",
                    "hierarchy_family": "photon_temperature_tensor",
                    "background_reference": "Omega_gamma0",
                },
                "massless_neutrino": {
                    "sector": "tensor",
                    "hierarchy_family": "massless_neutrino_tensor",
                    "background_reference": "Omega_nu0",
                    "anisotropic_stress": "supported",
                },
            },
            "hierarchy_families": {
                "photon_temperature_tensor": {
                    "sector": "tensor",
                    "species": ["photon"],
                    "closure": "free_streaming_tensor",
                    "default_l_max": 8,
                    "multipole_symbol": "theta_gamma_tl",
                },
                "photon_polarization_e_tensor": {
                    "sector": "tensor",
                    "species": ["photon"],
                    "closure": "free_streaming_tensor",
                    "default_l_max": 8,
                    "multipole_symbol": "e_gamma_tl",
                },
                "photon_polarization_b_tensor": {
                    "sector": "tensor",
                    "species": ["photon"],
                    "closure": "free_streaming_tensor",
                    "default_l_max": 8,
                    "multipole_symbol": "b_gamma_tl",
                },
                "massless_neutrino_tensor": {
                    "sector": "tensor",
                    "species": ["massless_neutrino"],
                    "closure": "free_streaming_tensor",
                    "default_l_max": 5,
                    "multipole_symbol": "nu_tl",
                },
            },
            "projection_typing": {},
            "accuracy_controls": {
                "tensor_reference_ells": [40, 50, 70],
                "runtime_envelope": "bounded",
            },
            "sources": {},
            "observables": {},
            "numerics": dict(numerics),
            "validity": {
                "regimes": ["linear", "declared_tensor_hierarchy"],
                "notes": "Metadata-only declared tensor hierarchy route.",
            },
        },
    }


def _custom_perturbations(**perturbation_kwargs: object) -> dict[str, object]:
    """Return the non-standard perturbation graph from the fixture."""

    return copy.deepcopy(
        _base_custom_cmb_contract(**perturbation_kwargs)["perturbations"]
    )


def _strip_perturbations(contract: dict[str, object]) -> dict[str, object]:
    """Return ``contract`` without the nested perturbation declaration."""

    stripped = copy.deepcopy(contract)
    stripped.pop("perturbations", None)
    return stripped


def _strip_declared_runtime_sections(
    contract: dict[str, object],
) -> dict[str, object]:
    """Return a metadata-only declared scalar hierarchy contract."""

    stripped = copy.deepcopy(contract)
    perturbations = stripped.get("perturbations")
    if isinstance(perturbations, dict):
        for section_name in (
            "variables",
            "derived",
            "equations",
            "constraints",
            "closures",
            "sources",
            "observables",
            "initial_conditions",
            "boundary_conditions",
        ):
            perturbations.pop(section_name, None)
    return stripped


def _analytic_signal_contract(
    *,
    source_scale: float = 1.0,
    closure_scale: float = 1.0,
    decay_rate: float = 0.02,
) -> dict[str, object]:
    """Return a one-mode graph with exact source and closure scaling."""

    contract = _base_custom_cmb_contract()
    contract["model_name"] = "AnalyticSignalCMB"
    model_parameters = dict(contract.get("model_parameters", {}))
    model_parameters.update(
        {
            "source_scale": source_scale,
            "closure_scale": closure_scale,
            "decay_rate": decay_rate,
        }
    )
    contract["model_parameters"] = model_parameters
    contract["numerical"].update(
        {
            "ell_max": 40,
            "k_sample_count": 6,
            "eta_sample_count": 128,
            "source_grid_multiplier": 1,
        }
    )
    contract["perturbations"] = {
        "contract_version": 2,
        "gauge": "conformal_newtonian",
        "variables": {
            "signal_mode": {
                "kind": "photon_temperature_monopole",
                "tensor_character": "scalar_like",
            },
        },
        "derived": {
            "closure_drive": {
                "expression": "closure_scale * signal_mode",
            },
        },
        "equations": {
            "evolve_signal_mode": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "signal_mode",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-decay_rate * signal_mode",
                "role": "continuity",
            },
        },
        "constraints": {},
        "closures": {},
        "sources": {
            "signal_source": {
                "expression": "source_scale * closure_drive",
                "role": "signal",
            },
        },
        "observables": {
            "signal_transfer": {
                "kind": "transfer_component",
                "projection": "line_of_sight_signal",
                "source_terms": {"signal": "signal_source"},
            },
            "TT": {
                "kind": "angular_power_spectrum",
                "primary": "signal_transfer",
                "secondary": "signal_transfer",
            },
        },
        "initial_conditions": {
            "signal_seed": {
                "target": {
                    "variable": "signal_mode",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "seed",
            },
        },
        "boundary_conditions": {},
        "validity": {
            "regimes": ["analytic_signal"],
        },
    }
    return contract


def _generic_background_custom_contract() -> dict[str, object]:
    """Return a custom CMB fixture with non-LCDM-style parameter names."""

    contract = _base_custom_cmb_contract()
    contract["model_name"] = "GenericBackgroundCustomCMB"
    contract["param_map"] = {
        "H0": 67.4,
        "tau": 0.054,
        "primordial_power_amplitude": 2.1e-9,
        "primordial_power_tilt": 0.965,
        "baryon_fraction_today": 0.049,
        "cold_dark_matter_fraction_today": 0.262,
        "photon_fraction_today": 5.38e-5,
        "relativistic_neutrino_fraction_today": 3.65e-5,
        "curvature_density_fraction": -0.01,
        "dark_fluid_eos_today": -0.85,
    }
    contract["model_parameters"] = {
        "cmb_temperature_K": 2.7255,
        "helium_mass_fraction": 0.245,
    }
    reionization = copy.deepcopy(_declared_background()["reionization"])
    contract["background"] = {
        "derived": {
            "Omega_b0": "baryon_fraction_today",
            "Omega_c0": "cold_dark_matter_fraction_today",
            "Omega_gamma0": "photon_fraction_today",
            "Omega_nu0": "relativistic_neutrino_fraction_today",
            "Omega_r0": "Omega_gamma0 + Omega_nu0",
            "Omega_k0": "curvature_density_fraction",
            "Omega_m0": "Omega_b0 + Omega_c0",
            "w0": "dark_fluid_eos_today",
            "Omega_de0": "1.0 - Omega_m0 - Omega_r0 - Omega_k0",
            "dark_energy_pressure_today": "w0 * Omega_de0",
            "H": (
                "H0 * sqrt("
                "Omega_r0 / (a ** 4) + "
                "Omega_m0 / (a ** 3) + "
                "Omega_k0 / (a ** 2) + "
                "Omega_de0 * (a ** (-3.0 * (1.0 + w0)))"
                ")"
            ),
        },
        "reionization": reionization,
    }
    return contract


def _critical_density_today_kg_m3(hubble_km_s_mpc: float) -> float:
    """Return the present-day critical density for ``hubble_km_s_mpc``."""

    hubble_si = float(hubble_km_s_mpc) * 1000.0 / 3.085_677_581_491_3673e22
    return float(3.0 * hubble_si * hubble_si / (8.0 * numpy.pi * 6.674_30e-11))


def _physical_density_custom_contract() -> dict[str, object]:
    """Return a declared CMB fixture driven by direct physical densities."""

    contract = _base_custom_cmb_contract()
    contract["model_name"] = "PhysicalDensityCustomCMB"
    expansion_rate_today = 67.4
    baryon_fraction_today = 0.049
    dark_component_fraction_today = 0.262
    critical_density_today = _critical_density_today_kg_m3(
        expansion_rate_today
    )
    perturbations = _declared_graph_perturbations()
    perturbations["derived"]["total_matter_density"]["expression"] = (
        "dark_component_budget_today * delta_c + "
        "baryon_budget_today * delta_b"
    )
    perturbations["derived"]["total_radiation_density"]["expression"] = (
        "4.0 * photon_fraction_today * theta_gamma0 + "
        "relativistic_neutrino_fraction_today * delta_nu"
    )
    contract["param_map"] = {
        "expansion_rate_today": expansion_rate_today,
        "reionization_depth": 0.054,
        "scalar_power_amplitude": 2.1e-9,
        "scalar_tilt_index": 0.965,
        "critical_mass_density_today_kg_m3": critical_density_today,
        "baryon_rest_mass_density_today": (
            baryon_fraction_today * critical_density_today
        ),
        "cold_dark_matter_rest_mass_density_today": (
            dark_component_fraction_today * critical_density_today
        ),
        "photon_fraction_today": 5.38e-5,
        "relativistic_neutrino_fraction_today": 3.65e-5,
        "curvature_fraction_today": -0.01,
        "dark_component_eos_today": -0.85,
    }
    contract["model_parameters"] = {
        "cmb_temperature_kelvin": 2.7255,
        "helium_mass_fraction": 0.245,
    }
    contract["background"] = {
        "derived": {
            "baryon_budget_today": (
                "baryon_rest_mass_density_today / "
                "critical_mass_density_today_kg_m3"
            ),
            "dark_component_budget_today": (
                "cold_dark_matter_rest_mass_density_today / "
                "critical_mass_density_today_kg_m3"
            ),
            "light_budget_today": (
                "photon_fraction_today + relativistic_neutrino_fraction_today"
            ),
            "matter_budget_today": (
                "baryon_budget_today + dark_component_budget_today"
            ),
            "vacuum_budget_today": (
                "1.0 - matter_budget_today - light_budget_today - "
                "curvature_fraction_today"
            ),
            "dark_energy_pressure_today": (
                "dark_component_eos_today * vacuum_budget_today"
            ),
            "H": (
                "expansion_rate_today * sqrt("
                "light_budget_today / (a ** 4) + "
                "matter_budget_today / (a ** 3) + "
                "curvature_fraction_today / (a ** 2) + "
                "vacuum_budget_today * ("
                "a ** (-3.0 * (1.0 + dark_component_eos_today))"
                "))"
            ),
        },
        "reionization": copy.deepcopy(_declared_background()["reionization"]),
    }
    contract["background"]["reionization"]["calibration"][
        "target_optical_depth"
    ] = "reionization_depth"
    contract["perturbations"] = perturbations
    return contract


class _CustomCMBPlugin:
    """Plugin stub that exposes the synthetic declared graph fixture."""

    INITIAL_GUESSES = (
        67.4,
        0.02237,
        0.12,
        0.054,
        2.1e-9,
        0.965,
        3.046,
        0.245,
        1090.0,
    )

    def get_cmb_contract(self, _params):
        """Reject the unprepared contract path during declared execution."""

        raise AssertionError("declared runtime should bypass get_cmb_contract")

    def get_cmb_declared_runtime(self, _params):
        """Return the synthetic declared-runtime contract."""

        return _speedup_contract(_custom_contract())

    def get_cmb_perturbation_contract(self, _params):
        """Return the synthetic declared perturbation graph."""

        return _custom_perturbations()


class _ContractFallbackOnlyPlugin:
    """Expose only the removed unprepared-contract path."""

    def get_cmb_contract(self, _params):
        """Fail if declared resolution attempts the removed fallback."""

        raise AssertionError("removed backend fallback was called")


class SliceNineReferenceContractTestCase(unittest.TestCase):
    """Exercise the fixed Slice Nine independent-reference surface."""

    def test_neutral_model_is_fixed_and_declared(self) -> None:
        """The acceptance fixture must be route-neutral and declared."""

        contract = _slice_nine_declared_acceptance_contract()

        self.assertNotIn("backend", contract)
        self.assertNotIn("standard", contract["perturbations"])
        self.assertNotIn("backend_mapping", contract["perturbations"])
        self.assertEqual(contract["model_name"], "SliceNineNeutralDeclared")
        model_values = dict(contract["param_map"])
        model_values.update(contract["model_parameters"])
        self.assertEqual(model_values, dict(SLICE_NINE_NEUTRAL_MODEL))
        self.assertEqual(
            contract["numerical"],
            dict(SLICE_NINE_DECLARED_NUMERICAL_CONTROLS),
        )

    def test_plugin_resolution_requires_declared_runtime(self) -> None:
        """Plugin resolution must not call an unprepared contract fallback."""

        with self.assertRaisesRegex(ValueError, "declared CMB runtime"):
            cmb.compute_cmb_spectrum_cached(
                _ContractFallbackOnlyPlugin(),
                (),
                (20,),
            )

    def test_acceptance_ranges_and_thresholds_are_explicit(self) -> None:
        """Reference ranges and PLAN thresholds must be machine-readable."""

        ranges = SLICE_NINE_ACCEPTANCE_RANGES
        thresholds = SLICE_NINE_ACCEPTANCE_THRESHOLDS

        self.assertEqual(ranges["scalar_ell"], (2, 2000))
        self.assertEqual(ranges["lensing_ell"], (2, 2000))
        self.assertEqual(ranges["potential_ell"], (10, 1500))
        self.assertEqual(
            thresholds["tt_fractional_p90"], numpy.longdouble("0.10")
        )
        self.assertEqual(
            thresholds["lensed_bb_fractional_median"],
            numpy.longdouble("0.15"),
        )

    def test_absolute_reference_metrics_handle_cross_zeroes(self) -> None:
        """Reference metrics must remain finite across cross-spectrum zeros."""

        actual = {
            "TT": numpy.asarray((1.1, 1.8, 4.4, 8.8), dtype=numpy.longdouble),
            "TE": numpy.asarray((1.2, -1.8, 0.4, 2.7), dtype=numpy.longdouble),
        }
        reference = {
            "TT": numpy.asarray((1.0, 2.0, 4.0, 8.0), dtype=numpy.longdouble),
            "TE": numpy.asarray((1.0, -2.0, 0.0, 3.0), dtype=numpy.longdouble),
        }
        metrics = _slice_nine_spectrum_metrics(
            actual,
            reference,
            spectra=("TT", "TE"),
        )
        self.assertAlmostEqual(metrics["TT"]["median_fractional"], 0.1)
        self.assertAlmostEqual(metrics["TT"]["p90_fractional"], 0.1)
        self.assertIn("normalized_rms", metrics["TE"])
        self.assertTrue(numpy.isfinite(metrics["TE"]["normalized_rms"]))

    def test_absolute_reference_metrics_cover_lensing_cross_surfaces(
        self,
    ) -> None:
        """Metrics must cover scalar, potential, and every lensed surface."""

        reference = {
            name: numpy.asarray((1.0, 2.0, 3.0, 4.0), dtype=numpy.longdouble)
            for name in SLICE_NINE_ACCEPTANCE_SPECTRA
        }
        actual = {
            name: values * numpy.longdouble("1.01")
            for name, values in reference.items()
        }
        metrics = _slice_nine_spectrum_metrics(
            actual,
            reference,
            spectra=SLICE_NINE_ACCEPTANCE_SPECTRA,
        )
        self.assertEqual(set(metrics), set(SLICE_NINE_ACCEPTANCE_SPECTRA))
        for name in ("TT", "EE", "PP", "lensed_TT", "lensed_BB"):
            self.assertAlmostEqual(
                metrics[name]["median_fractional"],
                0.01,
            )
        for name in ("TE", "TP", "EP", "lensed_TE"):
            self.assertAlmostEqual(metrics[name]["normalized_rms"], 0.01)

    def test_declared_k_grid_scales_to_requested_multipoles(self) -> None:
        """Declared projection surfaces stay fixed across request shapes."""

        raw_contract = _declared_scalar_hierarchy_contract(sum_mnu=0.0)
        raw_contract["numerical"].update(
            {
                "ell_max": 2000,
                "k_sample_count": 16,
                "eta_sample_count": 64,
            }
        )
        raw_contract["perturbations"]["accuracy_controls"] = {
            "scalar_reference_ells": [2, 2000]
        }
        contract = _prepare_declared_contract(raw_contract)
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical_params,
            numerics,
        )
        low_ell_grid = cmb_projection._build_projection_k_grid(
            ell_arr=numpy.asarray((20, 60, 120), dtype=int),
            background=background_data,
            numerics=numerics,
            perturbation_data=contract["perturbation_data"],
        )
        full_ell_grid = cmb_projection._build_projection_k_grid(
            ell_arr=numpy.asarray((2, 2000), dtype=int),
            background=background_data,
            numerics=numerics,
            perturbation_data=contract["perturbation_data"],
        )
        numpy.testing.assert_array_equal(
            low_ell_grid,
            full_ell_grid,
            err_msg=(
                "The same declared numerical surface must be used for low "
                "and full multipole requests."
            ),
        )

    def test_declared_scalar_absolute_parity_surface_is_fixed(self) -> None:
        """The scalar parity fixture must use one declared graph."""

        prepared = _prepare_declared_contract(
            _slice_nine_declared_acceptance_contract()
        )
        perturbation_data = prepared["perturbation_data"]
        manifest = perturbation_data.manifest_summary
        route = manifest["execution_route"]

        self.assertEqual(
            route,
            {
                "solver_id": "ccmbs_numpy",
                "solver_label": (
                    "CCMBS — Copernican Cosmic Microwave Background Solver"
                ),
                "runtime_module": (
                    "copernican.lib.likelihoods.cmb.orchestrators.ccmbs"
                ),
                "ready": True,
            },
        )
        self.assertEqual(
            set(manifest["angular_power_spectrum_targets"]),
            {"TT", "TE", "EE", "BB", "PP", "TP", "EP"},
        )
        self.assertEqual(
            set(manifest["transfer_component_contracts"]),
            {
                "temperature",
                "polarization_e",
                "polarization_b",
                "lensing_potential",
            },
        )
        self.assertEqual(
            tuple(manifest["validity_regimes"]),
            ("linear", "declared_scalar_hierarchy"),
        )
        self.assertTrue(
            prepared["perturbations"]["accuracy_controls"][
                "phase_aware_k_quadrature"
            ]
        )
        self.assertEqual(
            SLICE_NINE_ACCEPTANCE_RANGES["scalar_ell"],
            (2, 2000),
        )
        self.assertEqual(
            SLICE_NINE_ACCEPTANCE_RANGES["potential_ell"],
            (10, 1500),
        )

    def test_camb_reference_is_test_only_and_independent(self) -> None:
        """CAMB reference construction stays independent."""

        self.assertEqual(_slice_nine_reference_backend_name(), "CAMB")
        reference_contract = _slice_nine_camb_reference_contract(lmax=32)
        self.assertTrue(reference_contract["standard"])
        self.assertEqual(reference_contract["lmax"], 32)
        self.assertEqual(
            reference_contract["model_values"],
            dict(SLICE_NINE_NEUTRAL_MODEL),
        )
        self.assertNotIn("declared", reference_contract)

    def test_generated_source_history_bundle_is_digest_addressed(self) -> None:
        """Generated source evidence must be complete and reproducible."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_scalar_hierarchy_contract())
        )
        cache.clear_cmb_result_caches()
        first = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
            workload="fixed_parameter_diagnostic_slice_nine",
        )
        bundle = first.runtime_envelope["source_history_bundle_digest"]
        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["status"], "complete")
        self.assertRegex(bundle["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(int(bundle["mode_count"]), 0)
        self.assertGreater(int(bundle["sample_count"]), 0)
        self.assertEqual(
            set(bundle["included_fields"]),
            {
                "source_history_residual_samples_by_k",
                "hierarchy_equation_residuals_by_k",
                "initial_state_diagnostics_by_k",
                "metric_history_gradient_residual_by_k",
                "source_history_refinement",
                "declared_source_history_convergence",
                "source_history_derivative_provenance",
                "source_residual_audit_controls",
                "independent_source_residual_audit",
                "generated_scalar_source_closure",
            },
        )
        refinement = first.runtime_envelope["source_history_refinement"]
        self.assertEqual(refinement["axis"], "eta")
        self.assertEqual(
            refinement["fine_sample_count"],
            first.runtime_envelope["eta_sample_count"],
        )
        self.assertEqual(
            refinement["coarse_sample_count"],
            len(refinement["coarse_eta"]),
        )
        self.assertEqual(
            refinement["fine_sample_count"],
            len(refinement["fine_eta"]),
        )

        cache.clear_cmb_result_caches()
        second = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
            workload="fixed_parameter_diagnostic_slice_nine",
        )
        self.assertEqual(
            second.runtime_envelope["source_history_bundle_digest"]["sha256"],
            bundle["sha256"],
        )

    def test_camb_reference_returns_requested_finite_cls(self) -> None:
        """The independent reference path must return finite requested data."""

        reference = _slice_nine_camb_reference_spectra(
            numpy.asarray((2, 10, 32), dtype=int),
            spectra=("TT", "TE", "EE", "lensed_BB", "PP", "TP", "EP"),
        )

        self.assertEqual(
            set(reference),
            {"TT", "TE", "EE", "lensed_BB", "PP", "TP", "EP"},
        )
        for values in reference.values():
            self.assertEqual(values.shape, (3,))
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_reference_helpers_do_not_call_production_cmb_solver(self) -> None:
        """The test reference must remain independent of declared execution."""

        source = inspect.getsource(_slice_nine_camb_reference_spectra)
        self.assertNotIn("compute_cmb_spectrum_from_contract", source)
        self.assertNotIn("projection", source)


class CMBScientificReferenceValidationTestCase(unittest.TestCase):
    """CAMB-backed scientific reference checks for the CMB surface."""

    def test_camb_tensor_reference_returns_absolute_cls(self) -> None:
        """Tensor references must expose unlensed and lensed spectra."""

        reference = _slice_thirty_camb_tensor_reference_spectra(
            numpy.asarray((40, 50, 70), dtype=int),
        )
        self.assertEqual(
            set(reference),
            {
                "TT",
                "EE",
                "BB",
                "lensed_TT",
                "lensed_EE",
                "lensed_BB",
            },
        )
        for values in reference.values():
            self.assertEqual(values.shape, (3,))
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(float(numpy.max(reference["TT"])), 0.0)
        self.assertGreater(float(numpy.max(reference["EE"])), 0.0)
        self.assertGreater(float(numpy.max(reference["BB"])), 0.0)
        source = inspect.getsource(_slice_thirty_camb_tensor_reference_spectra)
        self.assertIn("get_tensor_cls", source)
        self.assertIn("get_total_cls", source)
        self.assertIn("get_lensed_scalar_cls", source)
        self.assertNotIn("projection", source)
        self.assertNotIn("compute_cmb_spectrum", source)

    def test_declared_tensor_spectra_match_absolute_camb_anchors(self) -> None:
        """Declared tensor spectra must match fixed CAMB anchors."""

        ells = numpy.asarray((40, 50, 70), dtype=int)
        analysis_ells = numpy.arange(int(numpy.max(ells)) + 1, dtype=int)
        tensor_contract = _declared_tensor_hierarchy_contract()
        tensor_contract["numerical"]["k_sample_count"] = 96
        declared_tensor = _raw_declared_public_spectra(
            _prepare_declared_contract(tensor_contract),
            analysis_ells,
            spectra=("TT", "TE", "EE", "BB"),
        )
        scalar_contract = _speedup_contract(
            _declared_scalar_hierarchy_contract()
        )
        lensing = _raw_declared_public_spectra(
            _prepare_declared_contract(scalar_contract),
            analysis_ells,
            spectra=("PP",),
        )
        lensed_tensor = cmb_solver._assemble_exact_lensed_spectra(
            {
                **declared_tensor,
                "PP": lensing["PP"],
            },
            analysis_ells,
        )
        reference = _slice_thirty_camb_tensor_reference_spectra(ells)
        declared = {
            spectrum_name: numpy.asarray(values)[ells]
            for spectrum_name, values in {
                **declared_tensor,
                **lensed_tensor,
            }.items()
            if spectrum_name
            in {
                "TT",
                "EE",
                "BB",
                "lensed_TT",
                "lensed_EE",
                "lensed_BB",
            }
        }
        compared_spectra = (
            "TT",
            "EE",
            "BB",
            "lensed_TT",
            "lensed_EE",
            "lensed_BB",
        )
        for spectrum_name in compared_spectra:
            self.assertEqual(declared[spectrum_name].shape, (3,))
            self.assertTrue(numpy.all(numpy.isfinite(declared[spectrum_name])))
        metrics = _slice_nine_spectrum_metrics(
            declared,
            reference,
            spectra=compared_spectra,
            auto_spectrum_floor=1.0e-6,
        )
        for spectrum_name in compared_spectra:
            self.assertLessEqual(
                metrics[spectrum_name]["median_fractional"],
                float(
                    SLICE_NINE_ACCEPTANCE_THRESHOLDS[
                        "tensor_fractional_median"
                    ]
                ),
                msg=f"{spectrum_name} metrics: {metrics[spectrum_name]}",
            )
        self.assertGreater(
            float(numpy.max(numpy.asarray(declared["lensed_BB"]))),
            0.0,
        )

    def test_camb_massive_neutrino_references_are_fixed_cosmologies(
        self,
    ) -> None:
        """Massive-neutrino references must compare absolute spectra."""

        ells = numpy.asarray((20, 60, 120), dtype=int)
        spectra = ("TT", "TE", "EE")
        light_reference = _slice_nine_camb_reference_spectra(
            ells,
            spectra=spectra,
            sum_mnu=0.0,
        )
        heavy_reference = _slice_nine_camb_reference_spectra(
            ells,
            spectra=spectra,
            sum_mnu=0.6,
        )
        for spectrum_name in spectra:
            light_values = light_reference[spectrum_name]
            heavy_values = heavy_reference[spectrum_name]
            self.assertTrue(numpy.all(numpy.isfinite(light_values)))
            self.assertTrue(numpy.all(numpy.isfinite(heavy_values)))
            self.assertGreater(
                float(numpy.max(numpy.abs(light_values))),
                0.0,
            )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(heavy_reference["TT"] - light_reference["TT"])
                )
            ),
            0.0,
        )

    def test_massive_neutrino_absolute_source_spectra_match_reference(
        self,
    ) -> None:
        """Fixed q hierarchies must match absolute physical source moments."""

        def _reference_moments(
            *,
            q_min: float,
            q_max: float,
            mass_ratio_today: float,
            scale_factors: numpy.ndarray,
        ) -> dict[str, numpy.ndarray]:
            def _occupation(log_q: float) -> float:
                q_value = float(numpy.exp(log_q))
                return float(1.0 / (1.0 + numpy.exp(q_value)))

            def _moment(
                scale_factor: float,
                power: float,
                denominator: str,
            ) -> float:
                mass_ratio = float(scale_factor) * mass_ratio_today

                def _integrand(log_q: float) -> float:
                    q_value = float(numpy.exp(log_q))
                    epsilon = float(
                        numpy.sqrt(q_value * q_value + mass_ratio**2)
                    )
                    if denominator == "epsilon":
                        factor = epsilon
                    elif denominator == "three_epsilon":
                        factor = 1.0 / (3.0 * epsilon)
                    elif denominator == "inverse_epsilon":
                        factor = 1.0 / epsilon
                    else:  # pragma: no cover - fixture-only guard
                        raise AssertionError(denominator)
                    return (
                        _occupation(log_q) * numpy.exp(power * log_q) * factor
                    )

                return float(
                    scipy_quad(
                        _integrand,
                        numpy.log(q_min),
                        numpy.log(q_max),
                        epsabs=1.0e-11,
                        epsrel=1.0e-11,
                    )[0]
                )

            density_today = _moment(1.0, 3.0, "epsilon")
            result: dict[str, numpy.ndarray] = {}
            for name, power, denominator in (
                ("density_fraction", 3.0, "epsilon"),
                ("pressure_fraction", 5.0, "three_epsilon"),
                ("momentum_fraction", 3.0, "epsilon"),
                ("shear_fraction", 5.0, "inverse_epsilon"),
            ):
                moment_values = numpy.asarray(
                    [
                        _moment(float(a_value), power, denominator)
                        for a_value in scale_factors
                    ],
                    dtype=float,
                )
                result[name] = moment_values / density_today
            return result

        fixed_cosmologies = (
            (1.0e-6, numpy.asarray((0.01, 0.02, 0.05))),
            (60.0, numpy.asarray((0.25, 0.5, 1.0))),
        )
        for sum_mnu, scale_factors in fixed_cosmologies:
            with self.subTest(sum_mnu=sum_mnu):
                raw_contract = _declared_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                    sum_mnu=sum_mnu,
                )
                raw_contract["numerical"]["momentum_grids"][
                    "massive_neutrino_default"
                ]["count"] = 12
                contract = _prepare_declared_contract(raw_contract)
                physical_params = (
                    cmb_background._resolve_custom_cmb_physical_parameters(
                        contract
                    )
                )
                runtime = evolution._resolve_declared_momentum_grid_runtimes(
                    contract["perturbation_data"],
                    model_parameters=contract["param_map"],
                    physical_params=physical_params,
                )[0]
                context = evolution._declared_momentum_grid_context(
                    contract["perturbation_data"],
                    model_parameters=contract["param_map"],
                    physical_params=physical_params,
                    scale_factor=scale_factors,
                )
                mass_ratio_today = float(
                    context["massive_neutrino_mass_eV"]
                ) / float(context["neutrino_temperature_eV"])
                reference = _reference_moments(
                    q_min=float(runtime.points[0]),
                    q_max=float(runtime.points[-1]),
                    mass_ratio_today=mass_ratio_today,
                    scale_factors=scale_factors,
                )
                omega_nu = sum_mnu / (
                    93.14 * float(physical_params.hubble_ratio) ** 2
                )
                for name in reference:
                    expected = (
                        omega_nu
                        * numpy.power(
                            scale_factors,
                            -4.0,
                        )
                        * reference[name]
                    )
                    actual = numpy.asarray(
                        context[f"massive_neutrino_{name}"],
                        dtype=float,
                    )
                    numpy.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=0.10,
                        atol=1.0e-30,
                        err_msg=(
                            f"absolute {name} mismatch for sum_mnu="
                            f"{sum_mnu}"
                        ),
                    )

    def test_batched_projection_bessel_values_match_reference(self) -> None:
        """Batched radial kernels must preserve SciPy reference values."""

        ell_signature = (2, 10, 40, 200, 1000)
        x_values = numpy.asarray((0.7, 5.0, 50.0, 500.0, 1900.0))
        values, derivatives = cmb_background._compute_spherical_bessel_batch(
            ell_signature,
            x_values,
        )
        ell_array = numpy.asarray(ell_signature, dtype=int)[:, None]
        expected_values = cmb_background.spherical_jn(
            ell_array,
            x_values[None, :],
        )
        expected_derivatives = cmb_background.spherical_jn(
            ell_array,
            x_values[None, :],
            derivative=True,
        )
        numpy.testing.assert_allclose(
            values,
            expected_values,
            rtol=1.0e-10,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            derivatives,
            expected_derivatives,
            rtol=1.0e-10,
            atol=1.0e-14,
        )

    def test_tensor_projection_uses_tensor_radial_kernel(self) -> None:
        """Tensor sources must not be projected with scalar Bessel windows."""

        x_values = numpy.asarray((0.7, 1.3, 2.1), dtype=float)
        x_signature = "slice-nine-item5-tensor-radial-kernel"
        cache.store_bessel_inputs(x_signature, x_values)
        kernel_batch = (
            cmb_background._get_cached_declared_projection_kernel_batch(
                (2,),
                x_signature,
            )
        )
        source = numpy.asarray((1.0, 2.0, 3.0), dtype=float)
        actual = cmb_projection._declared_graph_projection(
            projection="line_of_sight_signal",
            kernel="spherical_bessel_window",
            sector="tensor",
            kernel_batch=kernel_batch,
            k_value=1.0,
            eta_weights=numpy.ones(3, dtype=float),
            chi_grid=numpy.zeros(3, dtype=float),
            source_chi=1.0,
            source_histories={"signal": source},
        )
        numpy.testing.assert_allclose(
            actual,
            kernel_batch.tensor_temperature @ source,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.tensor_temperature[0],
            numpy.sqrt(3.0 / 8.0)
            * numpy.sqrt(24.0)
            * cmb_background.spherical_jn(2, x_values)
            / numpy.maximum(x_values, 1.0e-12) ** 2,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        j_l = cmb_background.spherical_jn(2, x_values)
        j_l_derivative = cmb_background.spherical_jn(
            2,
            x_values,
            derivative=True,
        )
        inverse_x = 1.0 / x_values
        j_l_second = (
            2.0 * 3.0 * inverse_x**2 - 1.0
        ) * j_l - 2.0 * inverse_x * j_l_derivative
        numpy.testing.assert_allclose(
            kernel_batch.tensor_e[0],
            0.25
            * (
                -j_l
                + j_l_second
                + 2.0 * j_l * inverse_x**2
                + 4.0 * j_l_derivative * inverse_x
            ),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.tensor_b[0],
            0.5 * (j_l_derivative + 2.0 * j_l * inverse_x),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        for projection, kernel, expected in (
            (
                "spin2_e_mode",
                "spin2_e_window",
                kernel_batch.tensor_e,
            ),
            (
                "spin2_b_mode",
                "spin2_b_window",
                kernel_batch.tensor_b,
            ),
        ):
            actual = cmb_projection._declared_graph_projection(
                projection=projection,
                kernel=kernel,
                sector="tensor",
                kernel_batch=kernel_batch,
                k_value=1.0,
                eta_weights=numpy.ones(3, dtype=float),
                chi_grid=numpy.zeros(3, dtype=float),
                source_chi=1.0,
                source_histories={"signal": source},
            )
            numpy.testing.assert_allclose(
                actual,
                expected @ source,
                rtol=1.0e-14,
                atol=1.0e-14,
            )

    def test_vector_projection_kernels_match_flat_space_limits(self) -> None:
        """Vector radial kernels must match the declared flat limits."""

        x_values = numpy.asarray((0.7, 1.3, 2.1), dtype=float)
        x_signature = "slice-nine-item5-vector-radial-kernel"
        cache.store_bessel_inputs(x_signature, x_values)
        kernel_batch = (
            cmb_background._get_cached_declared_projection_kernel_batch(
                (3,),
                x_signature,
            )
        )
        j_l = cmb_background.spherical_jn(3, x_values)
        j_l_derivative = cmb_background.spherical_jn(
            3,
            x_values,
            derivative=True,
        )
        inverse_x = 1.0 / x_values
        numpy.testing.assert_allclose(
            kernel_batch.vector_temperature_1[0],
            numpy.sqrt(3.0 * 4.0 / 2.0) * j_l * inverse_x,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.vector_temperature_2[0],
            numpy.sqrt(3.0 * 3.0 * 4.0 / 2.0)
            * (j_l_derivative * inverse_x - j_l * inverse_x**2),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.vector_e[0],
            0.5
            * numpy.sqrt(2.0 * 5.0)
            * (j_l * inverse_x**2 + j_l_derivative * inverse_x),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.vector_b[0],
            0.5 * numpy.sqrt(2.0 * 5.0) * j_l * inverse_x,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        source = numpy.asarray((1.0, 2.0, 3.0), dtype=float)
        residuals = []
        for projection, expected in (
            (
                "line_of_sight_vector_temperature",
                kernel_batch.vector_temperature_1,
            ),
            (
                "line_of_sight_vector_polarization_e",
                kernel_batch.vector_e,
            ),
            (
                "line_of_sight_vector_polarization_b",
                kernel_batch.vector_b,
            ),
        ):
            actual = cmb_projection._declared_graph_projection(
                projection=projection,
                kernel="spherical_bessel_window",
                sector="vector",
                kernel_batch=kernel_batch,
                k_value=1.0,
                eta_weights=numpy.ones(3, dtype=float),
                chi_grid=numpy.zeros(3, dtype=float),
                source_chi=1.0,
                source_histories={"signal": source},
            )
            expected_values = expected @ source
            numpy.testing.assert_allclose(
                actual,
                expected_values,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
            residuals.append(
                numpy.max(
                    numpy.abs(actual - expected_values)
                    / numpy.maximum(numpy.abs(expected_values), 1.0e-30)
                )
            )
        self.assertLess(max(residuals), 1.0e-12)

    def test_lensing_normalization_and_absolute_remapping_match_camb(
        self,
    ) -> None:
        """The declared remapper must match CAMB on lensed scalar spectra."""

        lmax = SLICE_NINE_ACCEPTANCE_RANGES["lensing_ell"][1]
        params = _slice_nine_build_camb_params(lmax=lmax)
        results = camb.get_results(params)
        unlensed = numpy.asarray(
            results.get_unlensed_scalar_cls(
                lmax=lmax,
                CMB_unit="muK",
            ),
            dtype=numpy.longdouble,
        )
        lensed_reference = numpy.asarray(
            results.get_lensed_scalar_cls(
                lmax=lmax,
                CMB_unit="muK",
            ),
            dtype=numpy.longdouble,
        )
        lensing = numpy.asarray(
            results.get_lens_potential_cls(lmax=lmax),
            dtype=numpy.longdouble,
        )
        raw_lensing = numpy.asarray(
            results.get_lens_potential_cls(lmax=lmax, raw_cl=True),
            dtype=numpy.longdouble,
        )
        ell_values = numpy.arange(lmax + 1, dtype=numpy.longdouble)
        ell_factor = ell_values * (ell_values + 1.0)
        numpy.testing.assert_allclose(
            lensing[2:, 0],
            raw_lensing[2:, 0]
            * ell_factor[2:] ** 2
            / (2.0 * numpy.longdouble(numpy.pi)),
            rtol=1.0e-12,
            atol=1.0e-30,
            err_msg="CAMB PP must use the declared deflection convention.",
        )
        actual = cmb_solver._assemble_exact_lensed_spectra(
            {
                "TT": unlensed[:, 0],
                "EE": unlensed[:, 1],
                "BB": unlensed[:, 2],
                "TE": unlensed[:, 3],
                "PP": lensing[:, 0],
            },
            numpy.arange(lmax + 1, dtype=int),
        )
        thresholds = SLICE_NINE_ACCEPTANCE_THRESHOLDS
        reference = {
            name: numpy.asarray(
                lensed_reference[
                    :,
                    {
                        "lensed_TT": 0,
                        "lensed_EE": 1,
                        "lensed_BB": 2,
                        "lensed_TE": 3,
                    }[name],
                ],
                dtype=numpy.longdouble,
            )
            for name in actual
        }
        metrics = _slice_nine_spectrum_metrics(
            {
                name: numpy.asarray(values, dtype=numpy.longdouble)
                for name, values in actual.items()
            },
            reference,
            spectra=(
                "lensed_TT",
                "lensed_EE",
                "lensed_TE",
                "lensed_BB",
            ),
        )
        self.assertLessEqual(
            metrics["lensed_TT"]["median_fractional"],
            float(thresholds["tt_fractional_median"]),
        )
        self.assertLessEqual(
            metrics["lensed_TT"]["p90_fractional"],
            float(thresholds["tt_fractional_p90"]),
        )
        self.assertLessEqual(
            metrics["lensed_EE"]["median_fractional"],
            float(thresholds["ee_fractional_median"]),
        )
        self.assertLessEqual(
            metrics["lensed_EE"]["p90_fractional"],
            float(thresholds["ee_fractional_p90"]),
        )
        self.assertLessEqual(
            metrics["lensed_TE"]["normalized_rms"],
            float(thresholds["te_normalized_rms"]),
        )
        self.assertLessEqual(
            metrics["lensed_BB"]["median_fractional"],
            float(thresholds["lensed_bb_fractional_median"]),
        )

    def test_camb_reference_lensing_cross_surfaces_use_declared_conventions(
        self,
    ) -> None:
        """The independent PP, TP, and EP surfaces use declared units."""

        ells = numpy.asarray((10, 100, 1500), dtype=int)
        reference = _slice_nine_camb_reference_spectra(
            ells,
            spectra=("PP", "TP", "EP"),
        )
        params = _slice_nine_build_camb_params(lmax=int(ells.max()))
        results = camb.get_results(params)
        lensing = numpy.asarray(
            results.get_lens_potential_cls(lmax=int(ells.max())),
            dtype=numpy.longdouble,
        )
        for name, column in (("PP", 0), ("TP", 1), ("EP", 2)):
            numpy.testing.assert_allclose(
                reference[name],
                lensing[ells, column],
                rtol=1.0e-12,
                atol=1.0e-30,
                err_msg=f"Independent CAMB {name} reference changed units.",
            )

    def test_slice_nine_neutral_background_matches_camb(self) -> None:
        """The fixed declared background must meet all CAMB thresholds."""

        contract = _prepare_declared_contract(
            _slice_nine_declared_acceptance_contract()
        )
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        reference = _slice_nine_camb_background_reference(
            background_data.eta_grid
        )
        reference_peak_eta = float(reference["peak_eta"])
        reference_peak_z = float(reference["peak_z"])
        reference_eta0 = float(reference["eta0"])
        reference_sound_horizon = float(reference["sound_horizon"])
        reference_x_e = numpy.asarray(reference["x_e"], dtype=float)
        reference_visibility = numpy.asarray(
            reference["visibility"],
            dtype=float,
        )
        thresholds = SLICE_NINE_ACCEPTANCE_THRESHOLDS

        peak_index = int(numpy.argmax(background_data.visibility_grid))
        peak_z = float(background_data.z_grid[peak_index])
        peak_eta = float(background_data.eta_grid[peak_index])
        recombination_band = (background_data.z_grid >= 800.0) & (
            background_data.z_grid <= 1600.0
        )
        recombination_median_x_e_error = float(
            numpy.median(
                numpy.abs(
                    background_data.x_e_grid[recombination_band]
                    - reference_x_e[recombination_band]
                )
                / numpy.maximum(reference_x_e[recombination_band], 1.0e-8)
            )
        )
        recombination_p90_error = float(
            numpy.percentile(
                numpy.abs(
                    background_data.x_e_grid[recombination_band]
                    - reference_x_e[recombination_band]
                )
                / numpy.maximum(reference_x_e[recombination_band], 1.0e-8),
                90.0,
            )
        )
        reionization_transition_band = (background_data.z_grid >= 6.0) & (
            background_data.z_grid <= 10.0
        )
        reionization_transition_error = float(
            numpy.median(
                numpy.abs(
                    background_data.x_e_grid[reionization_transition_band]
                    - reference_x_e[reionization_transition_band]
                )
            )
        )
        visibility_width_eta = _full_width_at_half_max(
            background_data.eta_grid,
            background_data.visibility_grid,
        )
        reference_visibility_width_eta = _full_width_at_half_max(
            background_data.eta_grid,
            reference_visibility,
        )
        visibility_width_z = _full_width_at_half_max(
            background_data.z_grid[::-1],
            background_data.visibility_grid[::-1],
        )
        reference_visibility_width_z = _full_width_at_half_max(
            background_data.z_grid[::-1],
            reference_visibility[::-1],
        )
        max_ionized_fraction = 1.0 + (
            physical.YHe / (2.0 * max(1.0 - physical.YHe, 1.0e-6))
        )

        self.assertTrue(numpy.all(numpy.isfinite(background_data.x_e_grid)))
        self.assertTrue(numpy.all(background_data.x_e_grid >= 0.0))
        self.assertTrue(
            numpy.all(
                background_data.x_e_grid <= max_ionized_fraction + 1.0e-6
            )
        )
        self.assertTrue(numpy.all(numpy.isfinite(reference_x_e)))
        self.assertTrue(numpy.all(numpy.isfinite(reference_visibility)))
        self.assertTrue(
            numpy.all(numpy.diff(background_data.tau_grid) <= 1.0e-8)
        )

        peak_z_error = abs(peak_z - reference_peak_z) / reference_peak_z
        self.assertLess(
            peak_z_error,
            thresholds["visibility_peak_fraction"],
            _named_limit_message(
                "visibility peak redshift",
                peak_z_error,
                thresholds["visibility_peak_fraction"],
            ),
        )
        peak_eta_error = (
            abs(peak_eta - reference_peak_eta) / reference_peak_eta
        )
        self.assertLess(
            peak_eta_error,
            thresholds["visibility_peak_fraction"],
            _named_limit_message(
                "visibility peak conformal time",
                peak_eta_error,
                thresholds["visibility_peak_fraction"],
            ),
        )
        visibility_width_eta_error = (
            abs(visibility_width_eta - reference_visibility_width_eta)
            / reference_visibility_width_eta
        )
        self.assertLess(
            visibility_width_eta_error,
            thresholds["visibility_width_fraction"],
            _named_limit_message(
                "visibility FWHM in conformal time",
                visibility_width_eta_error,
                thresholds["visibility_width_fraction"],
            ),
        )
        visibility_width_z_error = (
            abs(visibility_width_z - reference_visibility_width_z)
            / reference_visibility_width_z
        )
        self.assertLess(
            visibility_width_z_error,
            thresholds["visibility_width_fraction"],
            _named_limit_message(
                "visibility FWHM in redshift",
                visibility_width_z_error,
                thresholds["visibility_width_fraction"],
            ),
        )
        eta0_error = (
            abs(background_data.eta0 - reference_eta0) / reference_eta0
        )
        self.assertLess(
            eta0_error,
            thresholds["conformal_age_fraction"],
            _named_limit_message(
                "eta0",
                eta0_error,
                thresholds["conformal_age_fraction"],
            ),
        )
        sound_horizon_error = (
            abs(background_data.sound_horizon_mpc - reference_sound_horizon)
            / reference_sound_horizon
        )
        self.assertLess(
            sound_horizon_error,
            thresholds["sound_horizon_fraction"],
            _named_limit_message(
                "sound horizon at visibility peak",
                sound_horizon_error,
                thresholds["sound_horizon_fraction"],
            ),
        )
        self.assertLess(
            recombination_median_x_e_error,
            thresholds["recombination_median_fraction"],
            _named_limit_message(
                "recombination x_e median relative error",
                recombination_median_x_e_error,
                thresholds["recombination_median_fraction"],
            ),
        )
        self.assertLess(
            recombination_p90_error,
            thresholds["recombination_p90_fraction"],
            _named_limit_message(
                "recombination x_e p90 relative error",
                recombination_p90_error,
                thresholds["recombination_p90_fraction"],
            ),
        )
        self.assertLess(
            reionization_transition_error,
            0.08,
            _named_limit_message(
                "reionization-band x_e median absolute error",
                reionization_transition_error,
                0.08,
            ),
        )
        tau_error = abs(
            background_data.reionization_tau - float(reference["tau_reio"])
        ) / max(float(reference["tau_reio"]), 1.0e-12)
        self.assertLess(
            tau_error,
            thresholds["tau_reio_fraction"],
            _named_limit_message(
                "reionization optical depth",
                tau_error,
                thresholds["tau_reio_fraction"],
            ),
        )

    def test_camb_reference_contract_cannot_enter_production(self) -> None:
        """Test-only CAMB contracts must fail at the production boundary."""

        reference_contract = _camb_reference_contract()
        with self.assertRaisesRegex(ValueError, "removed route key"):
            cmb.compute_cmb_spectrum_from_contract(
                reference_contract,
                (20,),
            )

        params = camb_reference._make_camb_params(
            reference_contract,
            lmax=20,
        )
        self.assertIsInstance(params, camb.CAMBparams)

    def test_declared_scalar_hierarchy_amplitude_response_tracks_camb(
        self,
    ) -> None:
        """Declared scalar hierarchy preserves CAMB amplitude response."""

        base_contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        shifted_contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        shifted_contract["param_map"]["As"] *= 1.1
        ells = numpy.asarray((30, 60, 90, 120), dtype=int)
        base_declared = cmb.compute_cmb_spectrum_from_contract(
            base_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )
        shifted_declared = cmb.compute_cmb_spectrum_from_contract(
            shifted_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )

        base_reference_contract = _camb_reference_contract(
            base_contract["param_map"]
        )
        shifted_reference_contract = _camb_reference_contract(
            shifted_contract["param_map"]
        )
        base_params = camb_reference._make_camb_params(
            base_reference_contract,
            lmax=int(ells.max()),
        )
        shifted_params = camb_reference._make_camb_params(
            shifted_reference_contract,
            lmax=int(ells.max()),
        )
        base_reference = camb.get_results(base_params).get_unlensed_scalar_cls(
            lmax=int(ells.max()),
            CMB_unit="muK",
        )
        shifted_reference = camb.get_results(
            shifted_params
        ).get_unlensed_scalar_cls(
            lmax=int(ells.max()),
            CMB_unit="muK",
        )

        for spectrum_name, column_index in (("TT", 0), ("EE", 1), ("TE", 3)):
            declared_base = numpy.asarray(
                base_declared[spectrum_name], dtype=float
            )
            declared_shifted = numpy.asarray(
                shifted_declared[spectrum_name],
                dtype=float,
            )
            reference_base = numpy.asarray(
                base_reference[:, column_index][ells],
                dtype=float,
            )
            reference_shifted = numpy.asarray(
                shifted_reference[:, column_index][ells],
                dtype=float,
            )
            declared_ratio = numpy.divide(
                numpy.abs(declared_shifted),
                numpy.maximum(numpy.abs(declared_base), 1.0e-30),
            )
            reference_ratio = numpy.divide(
                numpy.abs(reference_shifted),
                numpy.maximum(numpy.abs(reference_base), 1.0e-30),
            )
            numpy.testing.assert_allclose(
                declared_ratio,
                reference_ratio,
                rtol=5.0e-2,
                atol=5.0e-2,
                err_msg=(
                    f"{spectrum_name} amplitude-response mismatch for the "
                    "declared scalar hierarchy route."
                ),
            )


class CMBCustomAnalyticValidationTestCase(unittest.TestCase):
    """Analytic observable checks for the declared-graph runtime."""

    def test_custom_source_expression_changes_observable(self) -> None:
        """Source scalings should map to the expected quadratic TT response."""

        baseline = _analytic_signal_contract(source_scale=1.0)
        changed = _analytic_signal_contract(source_scale=1.5)
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, 1.5 * 1.5),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared source scaling should produce the exact quadratic "
                "TT power response."
            ),
        )

    def test_custom_closures_change_spectrum(self) -> None:
        """Closure scalings should map to quadratic TT power."""

        baseline = _analytic_signal_contract(closure_scale=1.0)
        changed = _analytic_signal_contract(closure_scale=1.7)
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, 1.7 * 1.7),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared closure scaling should produce the exact quadratic "
                "TT power response."
            ),
        )

    def test_custom_equations_change_spectrum(self) -> None:
        """Stronger damping should suppress the raw analytic TT response."""

        low_decay = _prepare_declared_contract(
            _analytic_signal_contract(decay_rate=0.01)
        )
        high_decay = _prepare_declared_contract(
            _analytic_signal_contract(decay_rate=0.05)
        )
        low_decay = _with_prepared_numerical_overrides(
            low_decay,
            eta_sample_count=256,
            source_grid_multiplier=2,
        )
        high_decay = _with_prepared_numerical_overrides(
            high_decay,
            eta_sample_count=256,
            source_grid_multiplier=2,
        )
        ells = numpy.arange(20, 30, dtype=int)
        low_decay_tt = numpy.asarray(
            cmb_projection._compute_custom_cmb_spectrum_data(
                low_decay,
                ells,
            ).spectra["TT"],
            dtype=float,
        )
        high_decay_tt = numpy.asarray(
            cmb_projection._compute_custom_cmb_spectrum_data(
                high_decay,
                ells,
            ).spectra["TT"],
            dtype=float,
        )
        self.assertTrue(
            numpy.all(numpy.abs(high_decay_tt) < numpy.abs(low_decay_tt)),
            (
                "Increasing the declared decay coefficient should reduce "
                "the TT amplitude for every tested multipole."
            ),
        )

    def test_declared_interactions_change_observable(self) -> None:
        """Interaction terms should feed declared sources before projection."""

        baseline = _analytic_signal_contract()
        changed = _analytic_signal_contract()
        for contract, coefficient in (
            (baseline, 0.5),
            (changed, 1.0),
        ):
            contract["perturbations"]["interactions"] = {
                "signal_bridge": {
                    "expression": f"{coefficient:.16g} * signal_mode",
                }
            }
            contract["perturbations"]["sources"]["signal_source"][
                "expression"
            ] = "closure_drive + signal_bridge"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, (2.0 / 1.5) ** 2),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared interactions should contribute to the compiled "
                "source pipeline before the TT projection."
            ),
        )

    def test_custom_projection_kernel_changes_observable(self) -> None:
        """Custom kernels should change the projected transfer response."""

        spherical = _analytic_signal_contract()
        derivative = _analytic_signal_contract()
        spherical_observable = spherical["perturbations"]["observables"][
            "signal_transfer"
        ]
        derivative_observable = derivative["perturbations"]["observables"][
            "signal_transfer"
        ]
        spherical_observable["projection"] = "custom_line_of_sight"
        spherical_observable["kernel"] = "spherical_bessel_window"
        derivative_observable["projection"] = "custom_line_of_sight"
        derivative_observable["kernel"] = "spherical_bessel_derivative_window"
        ells = numpy.arange(20, 30, dtype=int)
        spherical_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                spherical,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        derivative_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                derivative,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(derivative_tt - spherical_tt))),
            1.0e-12,
        )

    def test_projection_extension_alias_changes_observable(self) -> None:
        """Projection extensions should route through the reviewed alias."""

        baseline = _analytic_signal_contract()
        extended = _analytic_signal_contract()
        extended["perturbations"]["projection_extensions"] = {
            "signal_derivative_alias": {
                "base_projection": "custom_line_of_sight",
                "kernel": "spherical_bessel_derivative_window",
                "required_roles": ["signal"],
                "allowed_roles": ["signal"],
            }
        }
        extended["perturbations"]["observables"]["signal_transfer"][
            "projection"
        ] = "signal_derivative_alias"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        extended_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                extended,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(extended_tt - baseline_tt))),
            1.0e-12,
        )

    def test_spin2_e_projection_sums_declared_sources(self) -> None:
        """Spin-2 E projections should not hide one declared source term."""

        baseline = _analytic_signal_contract()
        changed = _analytic_signal_contract()
        baseline_observable = baseline["perturbations"]["observables"][
            "signal_transfer"
        ]
        changed_observable = changed["perturbations"]["observables"][
            "signal_transfer"
        ]
        baseline_observable["projection"] = "spin2_e_mode"
        changed_observable["projection"] = "spin2_e_mode"
        baseline_observable["source_terms"] = {"signal": "signal_source"}
        changed["perturbations"]["sources"]["polarization_source"] = {
            "expression": "0.5 * closure_drive",
            "role": "polarization",
        }
        changed_observable["source_terms"] = {
            "signal": "signal_source",
            "polarization": "polarization_source",
        }
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, 1.5 * 1.5),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Spin-2 E projections should sum every declared source term "
                "instead of silently substituting one role for another."
            ),
        )

    def test_lensing_source_changes_pp(self) -> None:
        """Lensing-source scalings should map to exact quadratic PP power."""

        baseline = _speedup_contract(_custom_contract(include_lensing=True))
        changed = _speedup_contract(_custom_contract(include_lensing=True))
        changed["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.35 * exp(-tau) * (Phi + Psi)"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_pp = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("PP",),
        )["PP"]
        changed_pp = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("PP",),
        )["PP"]
        numpy.testing.assert_allclose(
            changed_pp / baseline_pp,
            numpy.full_like(baseline_pp, 1.35 * 1.35),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared lensing-source scaling should produce the exact "
                "quadratic PP power response."
            ),
        )

    def test_custom_lensing_kernel_changes_pp(self) -> None:
        """Custom lensing kernels should preserve the declared PP response."""

        baseline = _speedup_contract(_custom_contract(include_lensing=True))
        changed = _speedup_contract(_custom_contract(include_lensing=True))
        for contract in (baseline, changed):
            contract["perturbations"]["observables"]["lensing_potential"] = {
                "kind": "transfer_component",
                "projection": "custom_line_of_sight",
                "kernel": "lensing_potential_window",
                "source_terms": {"potential": "lensing_potential"},
            }
        changed["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.35 * exp(-tau) * (Phi + Psi)"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_pp = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("PP",),
        )["PP"]
        changed_pp = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("PP",),
        )["PP"]
        numpy.testing.assert_allclose(
            changed_pp / baseline_pp,
            numpy.full_like(baseline_pp, 1.35 * 1.35),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Custom lensing kernels should preserve the exact quadratic "
                "PP response to declared source scaling."
            ),
        )

    def test_b_mode_source_changes_bb(self) -> None:
        """B-mode source scalings should map to exact quadratic BB power."""

        baseline = _speedup_contract(_custom_contract(include_bb=True))
        changed = _speedup_contract(_custom_contract(include_bb=True))
        changed["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "1.25 * visibility * tensor_b"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_bb = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("BB",),
        )["BB"]
        changed_bb = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("BB",),
        )["BB"]
        numpy.testing.assert_allclose(
            changed_bb / baseline_bb,
            numpy.full_like(baseline_bb, 1.25 * 1.25),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared odd-parity source scaling should produce the "
                "exact quadratic BB power response."
            ),
        )

    def test_lensed_b_mode_source_changes_lensed_bb(self) -> None:
        """Declared BB sources should survive the exact lensed remapper."""

        baseline = _speedup_contract(
            _custom_contract(include_bb=True, include_lensing=True)
        )
        changed = _speedup_contract(
            _custom_contract(include_bb=True, include_lensing=True)
        )
        changed["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "1.25 * visibility * tensor_b"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_lensed_bb = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("lensed_BB",),
        )["lensed_BB"]
        changed_lensed_bb = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("lensed_BB",),
        )["lensed_BB"]
        self.assertGreater(
            float(
                numpy.max(numpy.abs(changed_lensed_bb - baseline_lensed_bb))
            ),
            0.0,
        )

    def test_custom_b_mode_kernel_changes_bb(self) -> None:
        """Custom B-mode kernels should preserve the declared BB response."""

        baseline = _speedup_contract(_custom_contract(include_bb=True))
        changed = _speedup_contract(_custom_contract(include_bb=True))
        for contract in (baseline, changed):
            contract["perturbations"]["observables"]["polarization_b"] = {
                "kind": "transfer_component",
                "projection": "custom_line_of_sight",
                "kernel": "spin2_b_window",
                "source_terms": {"polarization_b": "polarization_b_source"},
                "required_projection_roles": ["b_mode"],
            }
        changed["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "1.25 * visibility * tensor_b"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_bb = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("BB",),
        )["BB"]
        changed_bb = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("BB",),
        )["BB"]
        numpy.testing.assert_allclose(
            changed_bb / baseline_bb,
            numpy.full_like(baseline_bb, 1.25 * 1.25),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Custom B-mode kernels should preserve the exact quadratic "
                "BB response to declared odd-parity source scaling."
            ),
        )


class CMBCustomRuntimeBehaviorTestCase(unittest.TestCase):
    """Fast runtime-response coverage for declared-graph execution."""

    def test_declared_los_simpson_weights_integrate_nonuniform_quadratic(
        self,
    ) -> None:
        """Declared LOS weights should preserve quadratic histories."""

        eta_grid = numpy.asarray((0.0, 0.3, 0.9, 1.4, 2.1), dtype=float)
        history = 2.0 * eta_grid**2 + 3.0 * eta_grid + 1.0
        weights = cmb_projection._simpson_weights(eta_grid)
        expected = (
            (2.0 / 3.0) * eta_grid[-1] ** 3
            + (3.0 / 2.0) * eta_grid[-1] ** 2
            + eta_grid[-1]
        )
        self.assertAlmostEqual(
            float(numpy.dot(weights, history)),
            float(expected),
            places=12,
        )

    def test_declared_los_weights_stay_positive_across_clustered_anchors(
        self,
    ) -> None:
        """Clustered physical anchors must not destabilize LOS weights."""

        eta_grid = numpy.asarray(
            (0.0, 1.0e-6, 0.2, 0.4, 1.0),
            dtype=float,
        )
        weights = cmb_projection._simpson_weights(eta_grid)

        self.assertTrue(numpy.all(numpy.isfinite(weights)))
        self.assertTrue(numpy.all(weights >= 0.0))
        self.assertAlmostEqual(
            float(numpy.sum(weights)),
            float(eta_grid[-1] - eta_grid[0]),
            places=12,
        )

    def test_declared_los_phase_grid_records_effective_resolution(
        self,
    ) -> None:
        """A bounded generated request exposes its phase-aware LOS grid."""

        contract = _declared_scalar_hierarchy_contract(sum_mnu=0.0)
        contract["perturbations"]["accuracy_controls"] = {
            "los_phase_quadrature": {
                "enabled": True,
                "minimum_nodes": 512,
                "maximum_nodes": 512,
                "phase_points_per_cycle": 4.0,
            },
            "runtime_envelope": "bounded",
        }
        spectrum_data = _raw_declared_spectrum_data(
            contract,
            numpy.asarray((20, 30), dtype=int),
        )
        envelope = spectrum_data.runtime_envelope

        self.assertTrue(envelope["los_phase_quadrature_enabled"])
        self.assertTrue(envelope["los_phase_quadrature_applied"])
        self.assertEqual(envelope["los_phase_eta_sample_count"], 512)
        self.assertEqual(envelope["eta_sample_count"], 512)
        self.assertEqual(envelope["los_phase_minimum_nodes"], 512)
        self.assertEqual(envelope["los_phase_maximum_nodes"], 512)
        self.assertEqual(envelope["los_phase_points_per_cycle"], 4.0)
        self.assertGreater(envelope["los_phase_eta_min_step"], 0.0)
        self.assertGreaterEqual(
            envelope["los_phase_eta_max_step"],
            envelope["los_phase_eta_min_step"],
        )

    def test_declared_nonuniform_gradient_preserves_quadratic_derivative(
        self,
    ) -> None:
        """Declared history derivatives should remain second-order at edges."""

        eta_grid = numpy.asarray((0.0, 0.3, 0.9, 1.4, 2.1), dtype=float)
        history = 2.0 * eta_grid**2 + 3.0 * eta_grid + 1.0
        derivative = evolution._nonuniform_gradient(
            history,
            eta_grid,
        )
        numpy.testing.assert_allclose(
            derivative,
            4.0 * eta_grid + 3.0,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_temperature_projection_uses_declared_histories_directly(
        self,
    ) -> None:
        """Temperature LOS projection should not add acoustic phase weights."""

        kernel_batch = cmb_background._DeclaredProjectionKernelBatch(
            j_l=numpy.asarray(((1.0, 0.5, 0.25), (0.0, 1.0, 0.5))),
            j_l_derivative=numpy.asarray(((0.2, 0.1, 0.0), (0.0, 0.3, 0.4))),
            e_kernel=numpy.zeros((2, 3), dtype=float),
            b_kernel=numpy.zeros((2, 3), dtype=float),
            j_l_second_derivative=numpy.zeros((2, 3), dtype=float),
            vector_temperature_1=numpy.zeros((2, 3), dtype=float),
            vector_temperature_2=numpy.zeros((2, 3), dtype=float),
            vector_e=numpy.zeros((2, 3), dtype=float),
            vector_b=numpy.zeros((2, 3), dtype=float),
            tensor_temperature=numpy.zeros((2, 3), dtype=float),
            tensor_e=numpy.zeros((2, 3), dtype=float),
            tensor_b=numpy.zeros((2, 3), dtype=float),
        )
        eta_weights = numpy.asarray((0.25, 0.5, 0.25), dtype=float)
        source_histories = {
            "monopole": numpy.asarray((1.0, 2.0, 3.0), dtype=float),
            "doppler": numpy.asarray((0.5, 1.0, 1.5), dtype=float),
            "isw": numpy.asarray((2.0, 1.0, 0.0), dtype=float),
            "additive": numpy.asarray((1.0, 1.0, 1.0), dtype=float),
        }

        projected = cmb_projection._declared_graph_projection(
            projection="line_of_sight_temperature",
            kernel=None,
            kernel_batch=kernel_batch,
            k_value=0.2,
            eta_weights=eta_weights,
            chi_grid=numpy.asarray((1.0, 2.0, 4.0), dtype=float),
            source_chi=8.0,
            source_histories=source_histories,
        )

        expected = kernel_batch.j_l @ (
            eta_weights
            * (
                source_histories["monopole"]
                + source_histories["isw"]
                + source_histories["additive"]
            )
        ) + kernel_batch.j_l_derivative @ (
            eta_weights * source_histories["doppler"]
        )
        numpy.testing.assert_allclose(
            projected,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_temperature_derivative_source_uses_second_radial_kernel(
        self,
    ) -> None:
        """The integrated-by-parts source must use the second Bessel path."""

        kernel_batch = cmb_background._DeclaredProjectionKernelBatch(
            j_l=numpy.asarray(((1.0, 0.5, 0.25), (0.0, 1.0, 0.5))),
            j_l_derivative=numpy.zeros((2, 3), dtype=float),
            j_l_second_derivative=numpy.asarray(
                ((7.0, 11.0, 13.0), (17.0, 19.0, 23.0)),
                dtype=float,
            ),
            e_kernel=numpy.zeros((2, 3), dtype=float),
            b_kernel=numpy.zeros((2, 3), dtype=float),
            vector_temperature_1=numpy.zeros((2, 3), dtype=float),
            vector_temperature_2=numpy.zeros((2, 3), dtype=float),
            vector_e=numpy.zeros((2, 3), dtype=float),
            vector_b=numpy.zeros((2, 3), dtype=float),
            tensor_temperature=numpy.zeros((2, 3), dtype=float),
            tensor_e=numpy.zeros((2, 3), dtype=float),
            tensor_b=numpy.zeros((2, 3), dtype=float),
        )
        eta_weights = numpy.asarray((0.25, 0.5, 0.25), dtype=float)
        derivative_source = numpy.asarray((1.0, 2.0, 3.0), dtype=float)
        projected = cmb_projection._declared_graph_projection(
            projection="line_of_sight_temperature",
            kernel=None,
            kernel_batch=kernel_batch,
            k_value=0.2,
            eta_weights=eta_weights,
            chi_grid=numpy.asarray((1.0, 2.0, 4.0), dtype=float),
            source_chi=8.0,
            source_histories={"additive_derivative": derivative_source},
        )
        numpy.testing.assert_allclose(
            projected,
            kernel_batch.j_l_second_derivative
            @ (eta_weights * derivative_source),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_declared_projection_rejects_missing_source_histories(
        self,
    ) -> None:
        """A projection must not publish fabricated zero transfer values."""

        kernel_batch = cmb_background._DeclaredProjectionKernelBatch(
            j_l=numpy.ones((1, 2), dtype=float),
            j_l_derivative=numpy.ones((1, 2), dtype=float),
            j_l_second_derivative=numpy.ones((1, 2), dtype=float),
            e_kernel=numpy.ones((1, 2), dtype=float),
            b_kernel=numpy.ones((1, 2), dtype=float),
            vector_temperature_1=numpy.ones((1, 2), dtype=float),
            vector_temperature_2=numpy.ones((1, 2), dtype=float),
            vector_e=numpy.ones((1, 2), dtype=float),
            vector_b=numpy.ones((1, 2), dtype=float),
            tensor_temperature=numpy.ones((1, 2), dtype=float),
            tensor_e=numpy.ones((1, 2), dtype=float),
            tensor_b=numpy.ones((1, 2), dtype=float),
        )
        with self.assertRaisesRegex(ValueError, "no available source"):
            cmb_projection._declared_graph_projection(
                projection="line_of_sight_temperature",
                kernel="temperature_mixed_window",
                kernel_batch=kernel_batch,
                k_value=0.2,
                eta_weights=numpy.asarray((0.5, 0.5), dtype=float),
                chi_grid=numpy.asarray((1.0, 2.0), dtype=float),
                source_chi=8.0,
                source_histories={},
            )

    def test_lensing_projection_uses_declared_potential_source(self) -> None:
        """Lensing projection applies the declared Weyl-source sign."""

        kernel_batch = cmb_background._DeclaredProjectionKernelBatch(
            j_l=numpy.asarray(((1.0, 0.5, 0.25),), dtype=float),
            j_l_derivative=numpy.zeros((1, 3), dtype=float),
            j_l_second_derivative=numpy.zeros((1, 3), dtype=float),
            e_kernel=numpy.zeros((1, 3), dtype=float),
            b_kernel=numpy.zeros((1, 3), dtype=float),
            vector_temperature_1=numpy.zeros((1, 3), dtype=float),
            vector_temperature_2=numpy.zeros((1, 3), dtype=float),
            vector_e=numpy.zeros((1, 3), dtype=float),
            vector_b=numpy.zeros((1, 3), dtype=float),
            tensor_temperature=numpy.zeros((1, 3), dtype=float),
            tensor_e=numpy.zeros((1, 3), dtype=float),
            tensor_b=numpy.zeros((1, 3), dtype=float),
        )
        chi_grid = numpy.asarray((1.0, 2.0, 4.0), dtype=float)
        eta_weights = numpy.asarray((0.25, 0.5, 0.25), dtype=float)
        source = numpy.asarray((1.0, 2.0, 3.0), dtype=float)
        geometry = numpy.asarray((0.875, 0.375, 0.125), dtype=float)
        projected = cmb_projection._declared_graph_projection(
            projection="line_of_sight_lensing_potential",
            kernel="lensing_potential_window",
            kernel_batch=kernel_batch,
            k_value=0.2,
            eta_weights=eta_weights,
            chi_grid=chi_grid,
            source_chi=8.0,
            source_histories={"potential": source},
        )
        expected = -kernel_batch.j_l @ (eta_weights * geometry * source)
        numpy.testing.assert_allclose(
            projected,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_tight_coupling_regime_has_explicit_hysteresis(self) -> None:
        """Tight coupling should enter and exit through named thresholds."""

        entry_rate = evolution._tight_coupling_entry_rate(
            k_value=0.2,
            tight_coupling_ratio=50.0,
        )
        exit_rate = evolution._tight_coupling_exit_rate(
            k_value=0.2,
            tight_coupling_ratio=50.0,
        )

        self.assertGreater(entry_rate, exit_rate)
        self.assertTrue(
            evolution._tight_coupling_is_active(
                active=False,
                collision_rate=1.01 * entry_rate,
                k_value=0.2,
                tight_coupling_ratio=50.0,
            )
        )
        self.assertTrue(
            evolution._tight_coupling_is_active(
                active=True,
                collision_rate=0.5 * (entry_rate + exit_rate),
                k_value=0.2,
                tight_coupling_ratio=50.0,
            )
        )
        self.assertFalse(
            evolution._tight_coupling_is_active(
                active=True,
                collision_rate=0.99 * exit_rate,
                k_value=0.2,
                tight_coupling_ratio=50.0,
            )
        )

        declared_exit_ratio = 0.25
        declared_exit_rate = evolution._tight_coupling_exit_rate(
            k_value=0.2,
            tight_coupling_ratio=50.0,
            exit_ratio=declared_exit_ratio,
        )
        self.assertAlmostEqual(
            declared_exit_rate,
            declared_exit_ratio * entry_rate,
        )
        self.assertTrue(
            evolution._tight_coupling_is_active(
                active=True,
                collision_rate=0.5 * (declared_exit_rate + entry_rate),
                k_value=0.2,
                tight_coupling_ratio=50.0,
                exit_ratio=declared_exit_ratio,
            )
        )
        self.assertFalse(
            evolution._tight_coupling_is_active(
                active=True,
                collision_rate=0.99 * declared_exit_rate,
                k_value=0.2,
                tight_coupling_ratio=50.0,
                exit_ratio=declared_exit_ratio,
            )
        )

    def test_tight_coupling_exit_ratio_must_be_below_entry(self) -> None:
        """A hysteresis exit threshold must remain below entry."""

        with self.assertRaisesRegex(
            ValueError,
            r"tight-coupling exit ratio must be in \(0, 1\)",
        ):
            evolution._tight_coupling_exit_rate(
                k_value=0.2,
                tight_coupling_ratio=50.0,
                exit_ratio=1.0,
            )

    def test_source_file_does_not_contain_fake_or_legacy_hacks(self) -> None:
        """The production module should not contain old compatibility code."""

        source_text = "\n".join(
            Path(module.__file__).read_text(encoding="utf-8")
            for module in (
                cmb_solver,
                cmb_background,
                evolution,
                cmb_projection,
            )
        )
        for needle in (
            "equation_mode",
            "mapped_sector",
            "declared_equations",
            "source_normalization",
            "transfer_amplitude",
            "angular_projection_scale",
            "_evolve_custom_cmb_mode_histories",
            "_CUSTOM_CMB_SOURCE_CHANNELS",
            "_CUSTOM_CMB_SECTOR_ALIASES",
            "_classify_custom_physical_sector",
            "visibility shift",
            "visibility rescale",
            "_smooth_transition",
            "_gaussian_smooth_spectrum",
            "_assemble_approximate_lensed_spectra",
            "Keep the declared PP response visible",
        ):
            self.assertNotIn(needle, source_text)

    def test_custom_graph_runs_and_transfer_payloads_are_finite(self) -> None:
        """Transfer components and declared spectra should stay finite."""

        contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        ells = numpy.arange(20, 45, dtype=int)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
        )

        self.assertIsInstance(
            spectrum_data,
            cmb_projection.CustomCMBSpectrumData,
        )
        self.assertTrue(numpy.array_equal(spectrum_data.ell_grid, ells))
        self.assertEqual(
            spectrum_data.transfer_components["temperature"].shape,
            (ells.size, spectrum_data.k_grid.size),
        )
        self.assertEqual(
            spectrum_data.transfer_components["polarization_e"].shape,
            (ells.size, spectrum_data.k_grid.size),
        )
        self.assertEqual(set(spectrum_data.spectra), {"TT", "TE", "EE"})
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.Delta_l_T,
                spectrum_data.transfer_components["temperature"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.Delta_l_E,
                spectrum_data.transfer_components["polarization_e"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.C_l_TT,
                spectrum_data.spectra["TT"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.C_l_TE,
                spectrum_data.spectra["TE"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.C_l_EE,
                spectrum_data.spectra["EE"],
            )
        )
        for array in (
            spectrum_data.k_grid,
            spectrum_data.transfer_components["temperature"],
            spectrum_data.transfer_components["polarization_e"],
            spectrum_data.spectra["TT"],
            spectrum_data.spectra["TE"],
            spectrum_data.spectra["EE"],
        ):
            self.assertTrue(numpy.all(numpy.isfinite(array)))

    def test_declared_scalar_sources_use_runtime_optical_depth_history(
        self,
    ) -> None:
        """Generated scalar sources should see the background tau history."""

        cache.clear_cmb_parameter_caches()
        contract_data = _declared_scalar_hierarchy_contract()
        contract_data["numerical"].update(
            {
                "k_min": 0.02,
                "k_max": 0.02,
                "k_sample_count": 1,
                "eta_sample_count": 48,
                "source_grid_multiplier": 1,
            }
        )
        contract = _prepare_declared_contract(contract_data)
        captured_tau: list[numpy.ndarray] = []
        original = cmb_projection._evaluate_compiled_expression_noerr

        def _capture_tau(
            expression_data: object,
            env: Mapping[str, object],
        ) -> object:
            """Record the tau history seen by the scalar ISW source."""

            if (
                getattr(expression_data, "expression", "")
                == "exp(-tau) * (Phi_history_tau + Psi_tau)"
                and not captured_tau
            ):
                captured_tau.append(
                    numpy.asarray(env["tau"], dtype=float).copy()
                )
            return original(expression_data, env)

        with mock.patch.object(
            cmb_projection,
            "_evaluate_compiled_expression_noerr",
            side_effect=_capture_tau,
        ):
            cmb_projection._compute_custom_cmb_spectrum_data(
                contract,
                numpy.asarray((40,), dtype=int),
                requested_spectra=("TT",),
            )

        self.assertEqual(len(captured_tau), 1)
        tau_history = captured_tau[0]
        self.assertEqual(tau_history.ndim, 1)
        self.assertGreater(float(tau_history.max()), 100.0)
        self.assertLess(float(tau_history.min()), 1.0)
        self.assertGreater(
            float(numpy.max(numpy.abs(numpy.diff(tau_history)))),
            1.0e-6,
        )

    def test_declared_scalar_adiabatic_sources_use_hidden_superhorizon_prefix(
        self,
    ) -> None:
        """Adiabatic sources should evolve before the LOS grid start."""

        contract_data = _declared_scalar_hierarchy_contract()
        contract_data["numerical"].update(
            {
                "k_min": 0.01987357845532738,
                "k_max": 0.01987357845532738,
                "k_sample_count": 1,
                "eta_sample_count": 48,
                "source_grid_multiplier": 1,
                "initial_redshift": 2.0e4,
            }
        )
        contract = _prepare_declared_contract(contract_data)
        eta_history, theta_gamma0_history, _ = (
            _capture_visible_scalar_monopole_history(contract)
        )
        self.assertGreater(float(eta_history[0]), 20.0)
        self.assertTrue(numpy.isfinite(theta_gamma0_history[0]))
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        neutrino_fraction = physical.Omega_nu0 / max(
            physical.Omega_gamma0 + physical.Omega_nu0,
            1.0e-30,
        )
        initial_potential = 10.0 / (15.0 + 4.0 * neutrino_fraction)
        # The first visible sample is taken after the hidden prefix has
        # already evolved the regular seed.  Require a material displacement
        # while leaving the stronger early-start tracking assertion below to
        # bound the actual history.
        self.assertGreater(
            abs(float(theta_gamma0_history[0]) + 0.5 * initial_potential),
            0.05,
        )

    def test_declared_scalar_adiabatic_hidden_prefix_tracks_early_start(
        self,
    ) -> None:
        """Hidden evolution should stay close to an early start."""

        contract_data = _declared_scalar_hierarchy_contract()
        contract_data["numerical"].update(
            {
                "k_min": 0.01987357845532738,
                "k_max": 0.01987357845532738,
                "k_sample_count": 1,
                "eta_sample_count": 48,
                "source_grid_multiplier": 1,
                "initial_redshift": 2.0e4,
            }
        )
        contract = _prepare_declared_contract(contract_data)
        reference_data = copy.deepcopy(contract_data)
        reference_data["numerical"]["initial_redshift"] = 1.0e6
        reference = _prepare_declared_contract(reference_data)

        eta_history, theta_gamma0_history, _ = (
            _capture_visible_scalar_monopole_history(contract)
        )
        reference_eta, reference_theta_gamma0, _ = (
            _capture_visible_scalar_monopole_history(reference)
        )
        reference_interp = numpy.interp(
            eta_history,
            reference_eta,
            reference_theta_gamma0,
        )
        support = numpy.abs(reference_interp) >= (
            0.05 * float(numpy.max(numpy.abs(reference_interp)))
        )
        self.assertTrue(bool(numpy.any(support)))
        relative_error = numpy.abs(
            theta_gamma0_history[support] - reference_interp[support]
        ) / numpy.maximum(numpy.abs(reference_interp[support]), 1.0e-12)

        self.assertLess(float(numpy.median(relative_error)), 0.05)
        self.assertLess(float(numpy.max(relative_error)), 0.05)

    def test_declared_scalar_source_grid_preserves_visibility_refinement(
        self,
    ) -> None:
        """Scalar source grids should keep visibility-era clustering."""

        contract_data = _declared_scalar_hierarchy_contract()
        contract_data["numerical"].update(
            {
                "k_min": 0.01987357845532738,
                "k_max": 0.01987357845532738,
                "k_sample_count": 1,
                "eta_sample_count": 48,
                "source_grid_multiplier": 1,
                "initial_redshift": 2.0e4,
            }
        )
        contract = _prepare_declared_contract(contract_data)
        eta_history, _, _ = _capture_visible_scalar_monopole_history(contract)
        eta_steps = numpy.diff(eta_history)

        self.assertGreater(int(eta_history.size), 40)
        self.assertLess(
            float(numpy.min(eta_steps)) / float(numpy.max(eta_steps)),
            0.5,
        )

    def test_custom_spectra_have_structure_and_parameter_response(
        self,
    ) -> None:
        """Declared spectra should be finite, structured, and responsive."""

        contract = _speedup_contract(_custom_contract())
        ells = numpy.arange(20, 90, dtype=int)
        base = _raw_declared_public_spectra(
            contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )
        hi_as_contract = _speedup_contract(_custom_contract())
        hi_as_contract["param_map"]["As"] = 4.2e-9
        hi_as = _raw_declared_public_spectra(
            hi_as_contract,
            ells,
            spectra=("TT",),
        )["TT"]
        hi_h0_contract = _speedup_contract(_custom_contract())
        hi_h0_contract["param_map"]["H0"] = 74.0
        hi_h0 = _raw_declared_public_spectra(
            hi_h0_contract,
            ells,
            spectra=("TT",),
        )["TT"]

        base_tt = numpy.asarray(base["TT"], dtype=numpy.longdouble)
        base_te = numpy.asarray(base["TE"], dtype=numpy.longdouble)
        base_ee = numpy.asarray(base["EE"], dtype=numpy.longdouble)
        hi_as_tt = numpy.asarray(hi_as, dtype=numpy.longdouble)
        hi_h0_tt = numpy.asarray(hi_h0, dtype=numpy.longdouble)

        self.assertTrue(numpy.all(numpy.isfinite(base_tt)))
        self.assertTrue(numpy.all(numpy.isfinite(base_te)))
        self.assertTrue(numpy.all(numpy.isfinite(base_ee)))
        self.assertGreater(numpy.max(base_tt) - numpy.min(base_tt), 0.0)
        self.assertGreater(
            float(numpy.max(numpy.abs(base_te))),
            0.0,
        )
        self.assertTrue(numpy.all(base_ee >= 0.0))
        self.assertAlmostEqual(
            float(numpy.mean(hi_as_tt / base_tt)),
            2.0,
            places=2,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(hi_h0_tt - base_tt))),
            0.0,
        )

    def test_primordial_tilt_changes_temperature_shape(self) -> None:
        """Primordial tilt should reshape the declared temperature spectrum."""

        ells = numpy.arange(20, 90, dtype=int)
        low_ns_contract = _speedup_contract(_custom_contract())
        high_ns_contract = _speedup_contract(_custom_contract())
        low_ns_contract["param_map"]["ns"] = 0.85
        high_ns_contract["param_map"]["ns"] = 1.15
        low_ns_tt = numpy.asarray(
            _raw_declared_spectrum_data(low_ns_contract, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        high_ns_tt = numpy.asarray(
            _raw_declared_spectrum_data(high_ns_contract, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        low_shape = float(
            numpy.mean(numpy.abs(low_ns_tt[ells >= 60]))
            / numpy.mean(numpy.abs(low_ns_tt[ells <= 35]))
        )
        high_shape = float(
            numpy.mean(numpy.abs(high_ns_tt[ells >= 60]))
            / numpy.mean(numpy.abs(high_ns_tt[ells <= 35]))
        )
        self.assertGreater(high_shape, low_shape)

    def test_reionization_tau_changes_background_and_temperature(self) -> None:
        """The physical reionization ODE should feed the spectrum response."""

        low_tau_contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        high_tau_contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        low_tau_contract["param_map"]["tau"] = 0.03
        high_tau_contract["param_map"]["tau"] = 0.08
        low_physical = cmb_background._resolve_custom_cmb_physical_parameters(
            low_tau_contract
        )
        high_physical = cmb_background._resolve_custom_cmb_physical_parameters(
            high_tau_contract
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(
            low_tau_contract
        )
        low_background = cmb_background._build_custom_cmb_background(
            low_tau_contract,
            low_physical,
            numerics,
        )
        high_background = cmb_background._build_custom_cmb_background(
            high_tau_contract,
            high_physical,
            numerics,
        )
        z_probe = 8.0
        low_probe = float(
            low_background.x_e_grid[
                numpy.argmin(numpy.abs(low_background.z_grid - z_probe))
            ]
        )
        high_probe = float(
            high_background.x_e_grid[
                numpy.argmin(numpy.abs(high_background.z_grid - z_probe))
            ]
        )
        ells = numpy.arange(20, 30, dtype=int)
        low_tau_tt = _raw_declared_public_spectra(
            low_tau_contract,
            ells,
            spectra=("TT",),
        )["TT"]
        high_tau_tt = _raw_declared_public_spectra(
            high_tau_contract,
            ells,
            spectra=("TT",),
        )["TT"]
        self.assertLess(
            abs(low_background.reionization_tau - low_physical.tau_reio),
            0.01,
        )
        self.assertLess(
            abs(high_background.reionization_tau - high_physical.tau_reio),
            0.01,
        )
        self.assertGreater(high_probe, low_probe)
        self.assertGreater(
            float(numpy.max(numpy.abs(high_tau_tt - low_tau_tt))),
            1.0e-12,
        )

    def test_vectorized_reionization_quantities_match_scalar_context(self):
        """Reionization grid evaluation must preserve declared stage values."""

        contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        a_values = numpy.asarray((0.05, 0.1, 0.2), dtype=float)
        z_values = 1.0 / a_values - 1.0
        n_h_values = numpy.asarray((10.0, 20.0, 30.0), dtype=float)
        x_h_floor_values = numpy.asarray((0.1, 0.2, 0.3), dtype=float)
        helium_floor_values = numpy.asarray((0.01, 0.02, 0.03), dtype=float)
        x_e_floor_values = numpy.asarray((0.11, 0.22, 0.33), dtype=float)
        hubble_rates = numpy.asarray((1.0e-18, 2.0e-18, 3.0e-18))
        grid_context = (
            cmb_background._resolve_declared_reionization_quantity_grids(
                contract,
                a_values=a_values,
                z_values=z_values,
                n_h_values=n_h_values,
                x_h_floor_values=x_h_floor_values,
                helium_electron_floor_values=helium_floor_values,
                x_e_floor_values=x_e_floor_values,
                hubble_rates=hubble_rates,
                helium_number_ratio=0.08,
                hubble0_si=2.0e-18,
                calibration_symbol="reionization_log10_amplitude",
                calibration_value=4.0,
            )
        )
        quantity_names = (
            "hydrogen_ionization_rate",
            "helium_ionization_rate",
            "helium_double_ionization_rate",
            "hydrogen_temperature_K",
            "helium_temperature_K",
            "helium_double_temperature_K",
        )

        for index, (a_value, z_value) in enumerate(
            zip(a_values, z_values, strict=True)
        ):
            scalar_context = (
                cmb_background._resolve_declared_background_context(
                    contract,
                    a_values=float(a_value),
                    z_values=float(z_value),
                )
            )
            scalar_context.update(
                {
                    "n_H": float(n_h_values[index]),
                    "x_h_floor": float(x_h_floor_values[index]),
                    "helium_electron_floor": float(helium_floor_values[index]),
                    "x_e_floor": float(x_e_floor_values[index]),
                    "neutral_h_floor": max(
                        1.0 - float(x_h_floor_values[index]),
                        0.0,
                    ),
                    "neutral_he_floor": max(
                        0.08 - float(helium_floor_values[index]),
                        0.0,
                    ),
                    "helium_number_ratio": 0.08,
                    "H_SI": float(hubble_rates[index]),
                    "H0_SI": 2.0e-18,
                    "reionization_log10_amplitude": 4.0,
                }
            )
            scalar_values = (
                cmb_background._resolve_declared_reionization_context(
                    contract,
                    base_context=scalar_context,
                )
            )
            for name in quantity_names:
                self.assertAlmostEqual(
                    float(grid_context[name][index]),
                    float(scalar_values[name]),
                )

    def test_bb_and_lensing_targets_run_when_declared(self) -> None:
        """Additional observable targets should run through the graph."""

        contract = _speedup_contract(
            _custom_contract(include_bb=True, include_lensing=True)
        )
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("TT", "BB", "PP"),
        )

        self.assertEqual(set(spectra), {"TT", "BB", "PP"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["BB"], dtype=float)))
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["PP"], dtype=float)))
            ),
            0.0,
        )

    def test_lensing_cross_targets_run_when_declared(self) -> None:
        """Temperature and E-mode lensing cross terms should run declaredly."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("TP", "EP", "PP"),
        )

        self.assertEqual(set(spectra), {"TP", "EP", "PP"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["TP"], dtype=float)))
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["EP"], dtype=float)))
            ),
            0.0,
        )

    def test_vector_sector_targets_run_when_declared(self) -> None:
        """Vector-like transfer components should run declaredly."""

        contract = _speedup_contract(_custom_contract(include_vector=True))
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("VV",),
        )

        self.assertTrue(numpy.all(numpy.isfinite(spectra)))
        self.assertGreater(
            float(numpy.max(numpy.abs(numpy.asarray(spectra, dtype=float)))),
            0.0,
        )

    def test_sector_mismatch_cross_spectrum_fails_before_runtime(self) -> None:
        """Mixed scalar and vector transfer spectra should fail early."""

        contract = _speedup_contract(_custom_contract(include_vector=True))
        contract["perturbations"]["observables"]["TV"] = {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "vector_signal",
        }

        with self.assertRaisesRegex(
            ValueError,
            "incompatible sectors",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TV",),
            )

    def test_lensed_spectra_change_with_declared_lensing_strength(
        self,
    ) -> None:
        """Exact declared lensed outputs should respond to PP strength."""

        baseline = _speedup_contract(_custom_contract(include_lensing=True))
        changed = _speedup_contract(_custom_contract(include_lensing=True))
        baseline["numerical"]["k_sample_count"] = 16
        changed["numerical"]["k_sample_count"] = 16
        baseline["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.0e4 * exp(-tau) * (Phi + Psi)"
        changed["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.6 * 1.0e4 * exp(-tau) * (Phi + Psi)"
        ells = numpy.arange(20, 60, dtype=int)
        baseline_unlensed = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("TT", "EE", "TE"),
        )
        baseline_lensed = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=(
                "PP",
                "lensed_TT",
                "lensed_EE",
                "lensed_TE",
                "lensed_BB",
            ),
        )
        changed_lensed = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=(
                "PP",
                "lensed_TT",
                "lensed_EE",
                "lensed_TE",
                "lensed_BB",
            ),
        )

        self.assertEqual(
            set(baseline_lensed),
            {"PP", "lensed_TT", "lensed_EE", "lensed_TE", "lensed_BB"},
        )
        for values in baseline_lensed.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            baseline_lensed["lensed_BB"],
                            dtype=numpy.longdouble,
                        )
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            baseline_lensed["lensed_TT"],
                            dtype=numpy.longdouble,
                        )
                        - numpy.asarray(
                            baseline_unlensed["TT"],
                            dtype=numpy.longdouble,
                        )
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            changed_lensed["PP"],
                            dtype=numpy.longdouble,
                        )
                        - numpy.asarray(
                            baseline_lensed["PP"],
                            dtype=numpy.longdouble,
                        )
                    )
                )
            ),
            0.0,
        )
        for spectrum_name in (
            "lensed_TT",
            "lensed_TE",
            "lensed_EE",
            "lensed_BB",
        ):
            with self.subTest(spectrum_name=spectrum_name):
                self.assertGreater(
                    float(
                        numpy.max(
                            numpy.abs(
                                numpy.asarray(
                                    changed_lensed[spectrum_name],
                                    dtype=numpy.longdouble,
                                )
                                - numpy.asarray(
                                    baseline_lensed[spectrum_name],
                                    dtype=numpy.longdouble,
                                )
                            )
                        )
                    ),
                    0.0,
                )

    def test_lensed_sparse_requests_match_contiguous_remapping(self) -> None:
        """Lensing must interpolate on a contiguous analysis grid once."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        spectra = (
            "lensed_TT",
            "lensed_TE",
            "lensed_EE",
            "lensed_BB",
        )
        sparse_ells = numpy.asarray((20, 27, 44), dtype=int)
        dense_ells = numpy.arange(20, 45, dtype=int)
        sparse = _raw_declared_public_spectra(
            contract,
            sparse_ells,
            spectra=spectra,
        )
        dense = _raw_declared_public_spectra(
            contract,
            dense_ells,
            spectra=spectra,
        )
        dense_indices = sparse_ells - dense_ells[0]
        for name in spectra:
            numpy.testing.assert_allclose(
                sparse[name],
                dense[name][dense_indices],
                rtol=1.0e-12,
                atol=1.0e-30,
                err_msg=(
                    f"Sparse {name} requests must preserve remapped "
                    "multipoles."
                ),
            )

    def test_unlensed_sparse_requests_match_contiguous_remapping(self) -> None:
        """Unlensed spectra must be invariant to sparse ell requests."""

        contract = _speedup_contract(_custom_contract())
        prepared = _prepare_declared_contract(contract)
        spectra = ("TT", "TE", "EE")
        sparse_ells = numpy.asarray((20, 27, 44), dtype=int)
        dense_ells = numpy.arange(20, 45, dtype=int)
        sparse = cmb_solver._compute_declared_perturbation_spectrum(
            prepared,
            sparse_ells,
            spectra=spectra,
        )
        dense = cmb_solver._compute_declared_perturbation_spectrum(
            prepared,
            dense_ells,
            spectra=spectra,
        )
        dense_indices = sparse_ells - dense_ells[0]
        for name in spectra:
            numpy.testing.assert_allclose(
                sparse[name],
                dense[name][dense_indices],
                rtol=1.0e-12,
                atol=1.0e-30,
                err_msg=(
                    f"Sparse {name} requests must preserve contiguous "
                    "multipoles."
                ),
            )

    def test_unlensed_low_ell_is_stable_when_high_ell_is_requested(
        self,
    ) -> None:
        """High-ell requests must not change already requested low ells."""

        contract = _speedup_contract(_custom_contract())
        prepared = _prepare_declared_contract(contract)
        spectra = ("TT", "TE", "EE")
        low_ells = numpy.asarray((20, 27, 44), dtype=int)
        high_ell_surface = numpy.arange(20, 81, dtype=int)
        low = cmb_solver._compute_declared_perturbation_spectrum(
            prepared,
            low_ells,
            spectra=spectra,
        )
        high = cmb_solver._compute_declared_perturbation_spectrum(
            prepared,
            high_ell_surface,
            spectra=spectra,
        )
        high_indices = low_ells - high_ell_surface[0]
        for name in spectra:
            numpy.testing.assert_allclose(
                low[name],
                high[name][high_indices],
                rtol=1.0e-12,
                atol=1.0e-30,
                err_msg=(
                    f"High-ell {name} requests must not perturb low "
                    "multipoles."
                ),
            )

    def test_unlensed_low_ell_is_stable_when_request_starts_later(
        self,
    ) -> None:
        """Low multipoles must not change when a request omits ell=2."""

        contract = _speedup_contract(_custom_contract())
        prepared = _prepare_declared_contract(contract)
        spectra = ("TT", "TE", "EE")
        low_ells = numpy.asarray((32, 37, 44), dtype=int)
        full_ells = numpy.arange(2, 45, dtype=int)
        low = cmb_solver._compute_declared_perturbation_spectrum(
            prepared,
            low_ells,
            spectra=spectra,
        )
        full = cmb_solver._compute_declared_perturbation_spectrum(
            prepared,
            full_ells,
            spectra=spectra,
        )
        low_indices = low_ells - full_ells[0]
        for name in spectra:
            numpy.testing.assert_allclose(
                low[name],
                full[name][low_indices],
                rtol=1.0e-12,
                atol=1.0e-30,
                err_msg=(
                    f"Late-start {name} requests must preserve low "
                    "multipoles."
                ),
            )

    def test_exact_collision_action_matches_matrix_exponential(self) -> None:
        """The accelerated action must retain matrix-exponential values."""

        operator_matrix = numpy.asarray(
            (
                (-1.0, 0.25, 0.0),
                (2.0, -0.5, 0.5),
                (0.0, -1.0, -0.75),
            ),
            dtype=float,
        )
        target_state = numpy.asarray((0.5, -0.25, 0.75), dtype=float)
        dt = 0.125
        actual = cmb_projection._exact_linear_collision_step(
            operator_matrix=operator_matrix,
            dt=dt,
            target_state=target_state,
        )
        expected = expm(operator_matrix * dt) @ target_state
        numpy.testing.assert_allclose(
            actual, expected, rtol=1.0e-12, atol=1.0e-12
        )

    def test_exact_two_state_collision_action_matches_matrix_exponential(
        self,
    ) -> None:
        """The tensor Thomson two-state action must remain exact."""

        operator_matrix = numpy.asarray(
            ((-0.9, 0.6), (0.1, -0.4)),
            dtype=float,
        )
        target_state = numpy.asarray((0.35, -0.2), dtype=float)
        dt = 0.375
        actual = cmb_projection._exact_linear_collision_step(
            operator_matrix=operator_matrix,
            dt=dt,
            target_state=target_state,
        )
        expected = expm(operator_matrix * dt) @ target_state
        numpy.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_exact_collision_action_applies_operator_scale_to_two_state_block(
        self,
    ) -> None:
        """Scaled two-state blocks must retain their physical rate."""

        operator_matrix = numpy.asarray(
            ((-0.9, 0.6), (0.1, -0.4)),
            dtype=float,
        )
        target_state = numpy.asarray((0.35, -0.2), dtype=float)
        dt = 0.375
        operator_scale = 0.12
        actual = cmb_projection._exact_linear_collision_step(
            operator_matrix=operator_matrix,
            dt=dt,
            target_state=target_state,
            operator_scale=operator_scale,
        )
        expected = expm(operator_matrix * operator_scale * dt) @ target_state
        numpy.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_exact_block_collision_action_matches_matrix_exponential(self):
        """Independent exact collision blocks must retain their action."""

        operator_matrix = numpy.asarray(
            (
                (-1.0, 0.25, 0.0, 0.0),
                (2.0, -0.5, 0.0, 0.0),
                (0.0, 0.0, -0.9, 0.6),
                (0.0, 0.0, 0.1, -0.4),
            ),
            dtype=float,
        )
        target_state = numpy.asarray((0.5, -0.25, 0.35, -0.2), dtype=float)
        dt = 0.375
        actual = cmb_projection._exact_linear_collision_step(
            operator_matrix=operator_matrix,
            dt=dt,
            target_state=target_state,
        )
        expected = expm(operator_matrix * dt) @ target_state
        numpy.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_projection_kernel_batches_reuse_across_scalar_rebinds(
        self,
    ) -> None:
        """Projection-kernel caches should survive scalar parameter rebinds."""

        cache.clear_cmb_caches()
        baseline = _prepare_declared_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted = _prepare_declared_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted["param_map"]["As"] *= 1.1
        ells = numpy.arange(20, 45, dtype=int)
        cmb.compute_cmb_spectrum_from_contract(
            baseline,
            ells,
            spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )
        first_stats = cache.cmb_cache_stats()[
            "declared_projection_kernel_batch"
        ]
        cmb.compute_cmb_spectrum_from_contract(
            shifted,
            ells,
            spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )
        second_stats = cache.cmb_cache_stats()[
            "declared_projection_kernel_batch"
        ]

        self.assertEqual(second_stats["entries"], first_stats["entries"])
        self.assertGreater(second_stats["hits"], first_stats["hits"])

    def test_warm_projection_reuses_radial_kernel_batches(self) -> None:
        """Warm parameter requests must skip repeated Bessel batches."""

        cache.clear_cmb_caches()
        baseline = _prepare_declared_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted = _prepare_declared_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted["param_map"]["As"] *= 1.1
        ells = numpy.arange(20, 45, dtype=int)
        first = cmb_projection._compute_custom_cmb_spectrum_data(
            baseline,
            ells,
            requested_spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )
        second = cmb_projection._compute_custom_cmb_spectrum_data(
            shifted,
            ells,
            requested_spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )

        self.assertGreater(
            int(first.runtime_envelope["projection_bessel_batch_count"]),
            0,
        )
        self.assertEqual(
            int(second.runtime_envelope["projection_bessel_batch_count"]),
            0,
        )
        self.assertGreater(
            int(second.runtime_envelope["projection_kernel_cache_hits"]),
            0,
        )

    def test_warm_parameter_reuses_transfer_products(self) -> None:
        """Primordial rebinds must reuse evolution and projection products."""

        cache.clear_cmb_caches()
        baseline = _prepare_declared_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted = _prepare_declared_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted["param_map"]["As"] *= 1.1
        ells = numpy.arange(20, 45, dtype=int)
        spectra = ("TT", "TE", "EE", "PP", "TP", "EP")
        first = cmb_projection._compute_custom_cmb_spectrum_data(
            baseline,
            ells,
            requested_spectra=spectra,
        )
        second = cmb_projection._compute_custom_cmb_spectrum_data(
            shifted,
            ells,
            requested_spectra=spectra,
        )

        self.assertTrue(second.runtime_envelope["transfer_cache_hit"])
        self.assertEqual(
            float(second.runtime_envelope["evolution_seconds"]),
            0.0,
        )
        self.assertEqual(
            float(second.runtime_envelope["projection_seconds"]),
            0.0,
        )
        self.assertFalse(
            numpy.array_equal(
                numpy.asarray(first.spectra["TT"]),
                numpy.asarray(second.spectra["TT"]),
            )
        )

    def test_bb_requires_declared_b_mode_transfer_component(self) -> None:
        """BB should fail clearly when no odd-parity transfer is declared."""

        contract = _speedup_contract(_custom_contract())
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported CMB observable 'BB'",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("BB",),
            )

    def test_start_boundary_conditions_can_seed_missing_state(self) -> None:
        """Start-anchored conditions should seed the declared solver."""

        contract = _speedup_contract(_custom_contract())
        theta_b_seed = contract["perturbations"]["initial_conditions"].pop(
            "theta_b_seed"
        )
        theta_b_seed["anchor"] = "start"
        boundary_conditions = contract["perturbations"]["boundary_conditions"]
        boundary_conditions["theta_b_start"] = theta_b_seed
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_initial_conditions_can_resolve_relation_targets(self) -> None:
        """Initial-condition expressions may depend on solved relations."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["initial_conditions"]["theta_b_seed"][
            "expression"
        ] = "(k * eta_initial / 6.0) * seed + 0.25 * Phi"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_relation_target_derivative_sources_run(self) -> None:
        """Array-valued sources may differentiate algebraic metric targets."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["derived"]["Phi_tau"] = {
            "kind": "derivative_symbol",
            "variable": "Phi",
            "wrt": "tau",
            "order": 1,
        }
        contract["perturbations"]["derived"]["Psi_tau"] = {
            "kind": "derivative_symbol",
            "variable": "Psi",
            "wrt": "tau",
            "order": 1,
        }
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "0.02 * (Psi_tau - Phi_tau)"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_end_boundary_conditions_can_seed_missing_state(self) -> None:
        """End-anchored boundary conditions should drive the shooter."""

        contract = _analytic_signal_contract(decay_rate=1.0e-4)
        contract["perturbations"]["initial_conditions"].pop("signal_seed")
        contract["perturbations"]["boundary_conditions"]["signal_end"] = {
            "target": {
                "variable": "signal_mode",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.5",
            "anchor": "end",
        }
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_declared_gauge_metadata_is_not_restricted(self) -> None:
        """The declared graph solver should accept declared gauge metadata."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["gauge"] = "synchronous"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_declared_background_symbols_feed_declared_equations(self) -> None:
        """Declared background symbols should flow into perturbation math."""

        cache.clear_cmb_caches()
        baseline = _speedup_contract(_custom_contract())
        changed = _speedup_contract(_custom_contract())
        baseline["background"]["derived"]["metric_drive"] = "0.25 * Omega_b0"
        changed["background"]["derived"]["metric_drive"] = "0.75 * Omega_b0"
        baseline_baryon_equation = baseline["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        changed_baryon_equation = changed["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        baseline_baryon_equation["rhs"] += " + metric_drive * k * k * Psi"
        changed_baryon_equation["rhs"] += " + metric_drive * k * k * Psi"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("TT",),
        )["TT"]
        changed_tt = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("TT",),
        )["TT"]
        self.assertGreater(
            float(numpy.max(numpy.abs(changed_tt - baseline_tt))),
            1.0e-12,
        )

    def test_declared_exact_collision_matrix_changes_temperature_response(
        self,
    ) -> None:
        """Exact collision-matrix changes should alter the spectrum."""

        ells = numpy.arange(20, 45, dtype=int)
        baseline = _speedup_contract(_split_collision_contract())
        changed = _speedup_contract(_split_collision_contract())
        changed["perturbations"]["collision_operators"]["thomson_drag"][
            "exact_form"
        ]["matrix"][0][0] = "-1.2"
        baseline_tt = numpy.asarray(
            _raw_declared_spectrum_data(baseline, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        changed_tt = numpy.asarray(
            _raw_declared_spectrum_data(changed, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        self.assertGreater(
            _max_relative_delta(baseline_tt, changed_tt),
            1.0e-6,
        )

    def test_exact_split_collision_phase_refinement_converges(self) -> None:
        """Exact collision half-steps converge under phase refinement."""

        ells = numpy.arange(20, 45, dtype=int)
        baseline = _speedup_contract(_split_collision_contract())
        refined = _speedup_contract(_split_collision_contract())
        refined["numerical"]["evolution_phase_step"] = 1.0
        baseline_data = _raw_declared_spectrum_data(baseline, ells)
        refined_data = _raw_declared_spectrum_data(refined, ells)
        for spectrum_name in ("TT", "TE", "EE"):
            with self.subTest(spectrum=spectrum_name):
                self.assertLess(
                    _max_relative_delta(
                        numpy.asarray(
                            baseline_data.spectra[spectrum_name],
                            dtype=numpy.longdouble,
                        ),
                        numpy.asarray(
                            refined_data.spectra[spectrum_name],
                            dtype=numpy.longdouble,
                        ),
                    ),
                    1.0e-3,
                )

    def test_explicit_collision_operator_survives_exact_split_step(
        self,
    ) -> None:
        """Explicit collision terms should still run beside exact splits."""

        ells = numpy.arange(20, 45, dtype=int)
        baseline = _speedup_contract(_split_collision_contract())
        changed = _speedup_contract(_split_collision_contract())
        changed["perturbations"]["collision_operators"]["custom_drag"] = {
            "expression": "0.2 * collision_rate * theta_gamma0",
        }
        changed["perturbations"]["equations"]["evolve_theta_gamma1"][
            "rhs"
        ] += " + custom_drag"
        baseline_tt = numpy.asarray(
            _raw_declared_spectrum_data(baseline, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        changed_tt = numpy.asarray(
            _raw_declared_spectrum_data(changed, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        self.assertGreater(
            _max_relative_delta(baseline_tt, changed_tt),
            1.0e-7,
        )

    def test_implicit_collision_operator_coexists_with_exact_thomson(
        self,
    ) -> None:
        """Exact and implicit collision operators should coexist cleanly."""

        ells = numpy.arange(20, 45, dtype=int)
        baseline = _speedup_contract(_split_collision_contract())
        changed = _speedup_contract(_split_collision_contract())
        changed["perturbations"]["collision_operators"]["baryon_drag"] = {
            "expression": "-0.25 * collision_rate * theta_b",
            "integration_strategy": "implicit",
            "rate_expression": "collision_rate",
            "linear_block": {
                "targets": [{"kind": "baryon_velocity_divergence"}],
                "matrix": [["-0.25"]],
            },
        }
        changed_theta_b_equation = changed["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        changed_theta_b_equation["rhs"] += " + baryon_drag"
        baseline_tt = numpy.asarray(
            _raw_declared_spectrum_data(baseline, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        changed_tt = numpy.asarray(
            _raw_declared_spectrum_data(changed, ells).spectra["TT"],
            dtype=numpy.longdouble,
        )
        self.assertGreater(
            _max_relative_delta(baseline_tt, changed_tt),
            1.0e-7,
        )

    def test_split_collision_contract_renamed_states_match_baseline(
        self,
    ) -> None:
        """Exact collision targets should resolve by metadata, not names."""

        ells = numpy.arange(20, 45, dtype=int)
        baseline = _raw_declared_spectrum_data(
            _speedup_contract(_split_collision_contract()),
            ells,
        )
        renamed = _raw_declared_spectrum_data(
            _speedup_contract(
                _split_collision_contract(
                    rename_map={
                        "theta_gamma1": "temp_dipole",
                        "theta_b": "baryon_velocity",
                        "theta_gamma2": "temp_quadrupole",
                        "e_gamma2": "pol_quadrupole",
                        "theta_gamma3": "temp_octopole",
                        "e_gamma3": "pol_octopole",
                    }
                )
            ),
            ells,
        )

        for component_name in ("temperature", "polarization_e"):
            numpy.testing.assert_allclose(
                numpy.asarray(
                    renamed.transfer_components[component_name],
                    dtype=numpy.longdouble,
                ),
                numpy.asarray(
                    baseline.transfer_components[component_name],
                    dtype=numpy.longdouble,
                ),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        for spectrum_name in ("TT", "TE", "EE"):
            numpy.testing.assert_allclose(
                numpy.asarray(
                    renamed.spectra[spectrum_name],
                    dtype=numpy.longdouble,
                ),
                numpy.asarray(
                    baseline.spectra[spectrum_name],
                    dtype=numpy.longdouble,
                ),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    def test_split_collision_contract_requires_exact_form_before_evolution(
        self,
    ) -> None:
        """Exact split operators should fail before evolution if incomplete."""

        contract = _speedup_contract(_split_collision_contract())
        contract["perturbations"]["collision_operators"]["thomson_drag"].pop(
            "exact_form"
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires a compiled exact_form",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_split_collision_contract_requires_linear_block_before_evolution(
        self,
    ) -> None:
        """Implicit split operators should fail if their block is missing."""

        contract = _speedup_contract(_split_collision_contract())
        contract["perturbations"]["collision_operators"]["baryon_drag"] = {
            "expression": "-0.25 * collision_rate * theta_b",
            "integration_strategy": "implicit",
            "rate_expression": "collision_rate",
        }
        contract_theta_b_equation = contract["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        contract_theta_b_equation["rhs"] += " + baryon_drag"
        with self.assertRaisesRegex(
            ValueError,
            "requires a compiled linear_block",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_split_collision_conservation_rule_failure_raises(self) -> None:
        """Compiled collision conservation checks should stay enforced."""

        contract = _speedup_contract(_split_collision_contract())
        contract["perturbations"]["conservation_rules"][
            "thomson_drag_balance"
        ]["expression"] = (
            "3.0 * k * photon_baryon_momentum_ratio * thomson_drag + "
            "baryon_thomson_drag + 1.0e-2"
        )
        contract["perturbations"]["conservation_rules"][
            "thomson_drag_balance"
        ]["tolerance"] = 1.0e-6
        with self.assertRaisesRegex(
            ValueError,
            "conservation rule exceeded tolerance",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_multiple_declared_coordinates_preserve_runtime_response(
        self,
    ) -> None:
        """Mixed `tau`/`a` equations should run through the transform."""

        baseline = _analytic_signal_contract(decay_rate=0.025)
        transformed = _analytic_signal_contract(decay_rate=0.025)
        transformed_equation = transformed["perturbations"]["equations"][
            "evolve_signal_mode"
        ]
        transformed_equation["lhs"]["wrt"] = "a"
        transformed_equation["rhs"] = (
            f"({transformed_equation['rhs']})"
            + " * (299792.458 / (a * a * H))"
        )
        transformed["perturbations"]["initial_conditions"]["signal_seed"][
            "target"
        ]["wrt"] = "a"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        transformed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                transformed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            transformed_tt,
            baseline_tt,
            rtol=1.0e-2,
            atol=1.0e-10,
        )

    def test_generic_background_aliases_run_without_lcdm_named_inputs(
        self,
    ) -> None:
        """Generic background aliases should supply the declared solver."""

        contract = _speedup_contract(_generic_background_custom_contract())
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT", "TE", "EE"),
        )
        self.assertTrue(physical.has_cdm)
        self.assertIsNone(physical.Neff)
        self.assertAlmostEqual(physical.Omega_c0, 0.262)
        self.assertAlmostEqual(physical.primordial_amplitude, 2.1e-9)
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_physical_density_inputs_run_without_lcdm_density_aliases(
        self,
    ) -> None:
        """Direct physical densities should drive the declared solver."""

        contract = _speedup_contract(_physical_density_custom_contract())
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT", "TE", "EE"),
        )
        self.assertTrue(physical.has_cdm)
        self.assertAlmostEqual(physical.Omega_b0, 0.049, places=6)
        self.assertAlmostEqual(physical.Omega_c0, 0.262, places=6)
        self.assertAlmostEqual(
            physical.rho_b0_kg_m3,
            float(contract["param_map"]["baryon_rest_mass_density_today"]),
            places=20,
        )
        self.assertAlmostEqual(
            physical.rho_c0_kg_m3,
            float(
                contract["param_map"][
                    "cold_dark_matter_rest_mass_density_today"
                ]
            ),
            places=20,
        )
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_declared_source_refinement_above_two_changes_runtime_grid(
        self,
    ) -> None:
        """Source refinement above two should stay active at runtime."""

        baseline = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        refined = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        baseline = _with_prepared_numerical_overrides(
            baseline,
            source_grid_multiplier=3,
        )
        refined = _with_prepared_numerical_overrides(
            refined,
            source_grid_multiplier=4,
        )
        ells = numpy.arange(20, 30, dtype=int)
        baseline_data = cmb_projection._compute_custom_cmb_spectrum_data(
            baseline,
            ells,
        )
        refined_data = cmb_projection._compute_custom_cmb_spectrum_data(
            refined,
            ells,
        )
        self.assertGreater(
            int(refined_data.runtime_envelope["eta_sample_count"]),
            int(baseline_data.runtime_envelope["eta_sample_count"]),
        )

    def test_adaptive_surfaces_record_convergence(self) -> None:
        """Declared transfer, source, and LOS refinement stay observable."""

        contract = _speedup_contract(_analytic_signal_contract())
        contract["perturbations"]["accuracy_controls"] = {
            "adaptive_transfer": {
                "minimum_nodes": 8,
                "maximum_nodes": 24,
                "relative_tolerance": 2.0,
                "absolute_tolerance": 1.0e-12,
            },
            "adaptive_source": {
                "minimum_nodes": 282,
                "maximum_nodes": 512,
                "relative_tolerance": 2.0,
                "absolute_tolerance": 1.0e-12,
            },
            "adaptive_projection": {
                "relative_tolerance": 2.0,
                "absolute_tolerance": 1.0e-12,
            },
            "phase_points_per_cycle": 8,
            "fail_on_nonconvergence": True,
        }
        contract = _prepare_declared_contract(contract)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 24, dtype=int),
        )

        envelope = spectrum_data.runtime_envelope
        self.assertTrue(bool(envelope["adaptive_transfer_enabled"]))
        self.assertTrue(bool(envelope["adaptive_source_enabled"]))
        self.assertTrue(bool(envelope["adaptive_projection_enabled"]))
        self.assertEqual(
            tuple(envelope["declared_source_history_roles"]),
            ("signal_transfer:signal",),
        )
        self.assertEqual(
            int(envelope["declared_source_history_sample_count"]),
            int(envelope["eta_sample_count"]),
        )
        self.assertGreater(
            int(envelope["declared_source_history_mode_count"]),
            0,
        )
        self.assertTrue(bool(envelope["declared_source_history_finite"]))
        self.assertTrue(
            numpy.isfinite(
                float(
                    envelope["declared_source_history_max_abs"][
                        "signal_transfer:signal"
                    ]
                )
            )
        )
        history_convergence = envelope["declared_source_history_convergence"]
        self.assertGreater(
            int(history_convergence["refinement_mode_count"]),
            0,
        )
        self.assertLessEqual(
            float(history_convergence["relative_error"]),
            2.0,
        )
        self.assertGreater(
            int(envelope["adaptive_transfer_refinement_levels"]), 0
        )
        self.assertGreater(
            int(envelope["adaptive_source_refinement_levels"]), 0
        )
        self.assertGreater(
            int(envelope["adaptive_projection_refinement_levels"]),
            0,
        )
        self.assertLessEqual(
            float(envelope["adaptive_transfer_relative_error"]),
            2.0,
        )
        self.assertLessEqual(
            float(envelope["adaptive_source_relative_error"]),
            2.0,
        )
        self.assertLessEqual(
            float(envelope["adaptive_projection_relative_error"]),
            2.0,
        )
        for values in spectrum_data.spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_declared_scalar_evolution_refinement_reports_anchor_errors(
        self,
    ) -> None:
        """Scalar state and source refinements report physical anchors."""

        contract = _speedup_contract(
            _declared_scalar_hierarchy_contract(sum_mnu=0.0)
        )
        contract["model_name"] = "DeclaredScalarEvolutionRefinement"
        contract["numerical"].update(
            {
                "k_sample_count": 1,
                "eta_sample_count": 64,
                "evolution_eta_sample_count": 64,
                "ell_max": 24,
            }
        )
        contract["perturbations"]["numerics"] = dict(contract["numerical"])
        contract["perturbations"]["accuracy_controls"] = {
            "adaptive_evolution": {
                "minimum_nodes": 64,
                "maximum_nodes": 128,
                "relative_tolerance": 1.0e-2,
                "absolute_tolerance": 1.0e-12,
            },
            "runtime_envelope": "bounded",
            "fail_on_nonconvergence": False,
        }
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            _prepare_declared_contract(contract),
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
        )

        report = spectrum_data.runtime_envelope["scalar_evolution_convergence"]
        self.assertGreater(int(report["mode_count"]), 0)
        self.assertEqual(
            report["tier_order"],
            ("coarse", "intermediate", "reference"),
        )
        self.assertGreater(
            int(report["fine_sample_count"]),
            int(report["intermediate_sample_count"]),
        )
        self.assertGreater(
            int(report["intermediate_sample_count"]),
            int(report["coarse_sample_count"]),
        )
        self.assertEqual(
            set(report["anchor_relative_errors"]),
            {"early", "recombination", "late"},
        )
        evidence = report["refinement_evidence"]
        self.assertTrue(evidence["same_model"])
        self.assertEqual(
            set(evidence["tiers"]),
            {"coarse", "intermediate", "reference"},
        )
        self.assertIn("coarse_to_intermediate", evidence)
        self.assertIn("intermediate_to_reference", evidence)
        self.assertEqual(
            int(
                spectrum_data.runtime_envelope[
                    "adaptive_evolution_refinement_levels"
                ]
            ),
            2,
        )
        self.assertGreater(float(report["relative_error"]), 1.0e-2)
        self.assertTrue(numpy.isfinite(float(report["absolute_error"])))

    def test_declared_scalar_evolution_converges_at_physical_anchors(
        self,
    ) -> None:
        """A declared scalar history meets the one-percent anchor bound."""

        contract = _speedup_contract(_analytic_signal_contract())
        contract["model_name"] = "DeclaredScalarEvolutionConvergence"
        contract["numerical"].update(
            {
                "k_sample_count": 1,
                "eta_sample_count": 64,
                "evolution_eta_sample_count": 64,
                "ell_max": 24,
            }
        )
        contract["perturbations"]["numerics"] = dict(contract["numerical"])
        contract["perturbations"]["accuracy_controls"] = {
            "adaptive_evolution": {
                "minimum_nodes": 32,
                "maximum_nodes": 128,
                "relative_tolerance": 1.0e-2,
                "absolute_tolerance": 1.0e-12,
            },
            "runtime_envelope": "bounded",
        }
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            _prepare_declared_contract(contract),
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
        )

        report = spectrum_data.runtime_envelope["scalar_evolution_convergence"]
        self.assertEqual(
            report["tier_order"],
            ("coarse", "intermediate", "reference"),
        )
        self.assertGreater(
            int(report["reference_sample_count"]),
            int(report["intermediate_sample_count"]),
        )
        self.assertGreater(
            int(report["intermediate_sample_count"]),
            int(report["coarse_sample_count"]),
        )
        self.assertLessEqual(float(report["relative_error"]), 1.0e-2)
        self.assertLessEqual(
            max(report["anchor_relative_errors"].values()),
            1.0e-2,
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectrum_data.spectra["TT"])))

    def test_declared_scalar_hierarchy_depth_converges_at_anchor_surface(
        self,
    ) -> None:
        """Refined scalar hierarchy depths change accepted spectra by <1%."""

        spectra_by_depth = []
        for depth in (10, 12):
            contract = _speedup_contract(
                _declared_scalar_hierarchy_contract(sum_mnu=0.0)
            )
            contract["model_name"] = f"DeclaredScalarHierarchyDepth{depth}"
            contract["numerical"].update(
                {
                    "k_sample_count": 6,
                    "eta_sample_count": 128,
                    "photon_hierarchy_l_max": depth,
                    "photon_polarization_hierarchy_l_max": depth,
                    "neutrino_hierarchy_l_max": depth - 3,
                    "ell_max": 120,
                }
            )
            contract["perturbations"]["numerics"].update(contract["numerical"])
            spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 60, 120), dtype=int),
                requested_spectra=("TT", "EE"),
            )
            spectra_by_depth.append(spectrum_data.spectra)

        for spectrum_name in ("TT", "EE"):
            metric = convergence.evaluate_control_refinement(
                spectra_by_depth[0][spectrum_name],
                spectra_by_depth[1][spectrum_name],
                name=f"scalar {spectrum_name} hierarchy",
                tolerance=(convergence.FINAL_HIERARCHY_RELATIVE_TOLERANCE),
            )
            convergence.require_convergence(metric)

    def test_adaptive_projection_refines_line_of_sight_grid(
        self,
    ) -> None:
        """Projection refinement changes the physical LOS sampling grid."""

        baseline = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        refined = _speedup_contract(_analytic_signal_contract())
        refined["perturbations"]["accuracy_controls"] = {
            "adaptive_projection": {
                "minimum_nodes": 282,
                "maximum_nodes": 512,
                "relative_tolerance": 2.0,
                "absolute_tolerance": 1.0e-12,
            },
            "phase_points_per_cycle": 8,
        }
        refined = _prepare_declared_contract(refined)
        ells = numpy.arange(20, 24, dtype=int)
        baseline_data = cmb_projection._compute_custom_cmb_spectrum_data(
            baseline,
            ells,
        )
        refined_data = cmb_projection._compute_custom_cmb_spectrum_data(
            refined,
            ells,
        )

        self.assertGreater(
            int(refined_data.runtime_envelope["eta_sample_count"]),
            int(baseline_data.runtime_envelope["eta_sample_count"]),
        )
        self.assertLessEqual(
            float(
                refined_data.runtime_envelope[
                    "adaptive_projection_relative_error"
                ]
            ),
            2.0,
        )

    def test_adaptive_projection_compares_coarsened_eta_surface(self) -> None:
        """Projection convergence compares distinct radial resolutions."""

        contract = _speedup_contract(_analytic_signal_contract())
        contract["model_name"] = "AdaptiveProjectionCoarsenedEtaSurface"
        contract["perturbations"]["accuracy_controls"] = {
            "adaptive_projection": {
                "minimum_nodes": 282,
                "maximum_nodes": 512,
                "relative_tolerance": 2.0,
                "absolute_tolerance": 1.0e-12,
            },
            "phase_points_per_cycle": 8,
        }
        prepared = _prepare_declared_contract(contract)
        eta_sizes: list[int] = []
        original_projection = cmb_projection._declared_graph_projection

        def _record_projection(*args: object, **kwargs: object) -> object:
            """Record the radial resolution used by each projection."""

            kernel_batch = kwargs["kernel_batch"]
            eta_sizes.append(int(kernel_batch.j_l.shape[1]))
            return original_projection(*args, **kwargs)

        with mock.patch.object(
            cmb_projection,
            "_declared_graph_projection",
            side_effect=_record_projection,
        ):
            cmb_projection._compute_custom_cmb_spectrum_data(
                prepared,
                numpy.asarray((20, 23), dtype=int),
                requested_spectra=("TT",),
            )

        self.assertGreater(len(eta_sizes), 1)
        self.assertLess(min(eta_sizes), max(eta_sizes))

    def test_requested_unknown_spectrum_fails_before_runtime(self) -> None:
        """Unknown requested spectra fail instead of returning an empty set."""

        contract = _speedup_contract(_custom_contract())
        contract["model_name"] = "UnknownRequestedSpectrum"
        with self.assertRaisesRegex(
            ValueError,
            "does not provide requested spectra: XX",
        ):
            cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 23), dtype=int),
                requested_spectra=("XX",),
            )

    def test_solver_rejects_unknown_spectrum_before_background(self) -> None:
        """Unsupported spectra fail before declared background construction."""

        contract = _speedup_contract(_custom_contract())
        contract["model_name"] = "UnknownSpectrumPreflight"
        prepared = _prepare_declared_contract(contract)
        with mock.patch.object(
            cmb_solver,
            "_compute_custom_cmb_spectrum_data",
            side_effect=AssertionError("background must not be constructed"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported CMB observable 'XX'",
            ):
                cmb_solver._compute_declared_perturbation_spectrum(
                    prepared,
                    numpy.asarray((20, 23), dtype=int),
                    spectra=("XX",),
                )

    def test_requested_spectrum_filters_unrelated_source_evaluation(
        self,
    ) -> None:
        """A TT request does not evaluate an unused lensing source."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        contract["model_name"] = "RequestedSpectrumSourceFiltering"
        evaluated_expressions: list[str] = []
        original_evaluator = cmb_projection._evaluate_compiled_expression_noerr

        def _record_expression(
            expression_data: object,
            env: Mapping[str, object],
        ) -> object:
            """Record source expressions evaluated by the declared graph."""

            evaluated_expressions.append(
                str(getattr(expression_data, "expression", ""))
            )
            return original_evaluator(expression_data, env)

        with mock.patch.object(
            cmb_projection,
            "_evaluate_compiled_expression_noerr",
            side_effect=_record_expression,
        ):
            spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 23), dtype=int),
                requested_spectra=("TT",),
            )

        self.assertEqual(set(spectrum_data.spectra), {"TT"})
        self.assertNotIn("exp(-tau) * (Phi + Psi)", evaluated_expressions)

    def test_scalar_unlensed_surfaces_converge_within_one_percent(
        self,
    ) -> None:
        """Scalar TT, TE, EE, and PP surfaces meet the accuracy bound."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        contract["model_name"] = "ScalarUnlensedSurfaceConvergence"
        contract["numerical"].update(
            {
                "k_sample_count": 64,
                "eta_sample_count": 512,
                "ell_max": 40,
            }
        )
        contract["perturbations"]["accuracy_controls"] = {
            "adaptive_transfer": {
                "minimum_nodes": 64,
                "maximum_nodes": 128,
                "relative_tolerance": 1.0e-2,
                "absolute_tolerance": 1.0e-12,
            },
            "adaptive_source": {
                "minimum_nodes": 1133,
                "maximum_nodes": 4096,
                "relative_tolerance": 1.0e-2,
                "absolute_tolerance": 1.0e-12,
            },
            "adaptive_projection": {
                "minimum_nodes": 1133,
                "maximum_nodes": 4096,
                "relative_tolerance": 1.0e-2,
                "absolute_tolerance": 1.0e-12,
            },
            "phase_points_per_cycle": 8,
            "fail_on_nonconvergence": True,
        }
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            _prepare_declared_contract(contract),
            numpy.asarray((20, 30, 40), dtype=int),
            requested_spectra=("TT", "TE", "EE", "PP"),
        )

        self.assertEqual(
            set(spectrum_data.spectra),
            {"TT", "TE", "EE", "PP"},
        )
        envelope = spectrum_data.runtime_envelope
        for surface in ("transfer", "source", "projection"):
            self.assertLessEqual(
                float(envelope[f"adaptive_{surface}_relative_error"]),
                1.0e-2,
            )
        for values in spectrum_data.spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_final_scalar_k_and_source_refinements_converge(self) -> None:
        """Independent k and source-grid refinements meet final bounds."""

        spectra_by_grid = {}
        for label, k_count, source_multiplier in (
            ("baseline", 64, 2),
            ("k_refined", 96, 2),
            ("source_refined", 64, 4),
        ):
            contract = _speedup_contract(
                _custom_contract(include_lensing=True)
            )
            contract["model_name"] = f"FinalScalarGrid{label}"
            contract["numerical"].update(
                {
                    "k_sample_count": k_count,
                    "eta_sample_count": 512,
                    "ell_max": 40,
                    "source_grid_multiplier": source_multiplier,
                }
            )
            contract["perturbations"]["numerics"] = copy.deepcopy(
                contract["numerical"]
            )
            cache.clear_cmb_caches()
            spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 30, 40), dtype=int),
                requested_spectra=("TT", "TE", "EE", "PP"),
            )
            spectra_by_grid[label] = spectrum_data.spectra

        for refined_label in ("k_refined", "source_refined"):
            with self.subTest(refinement=refined_label):
                report = convergence.evaluate_spectrum_refinement(
                    spectra_by_grid["baseline"],
                    spectra_by_grid[refined_label],
                    required_spectra=("TT", "TE", "EE", "PP"),
                )
                convergence.require_convergence(report)

    def test_declared_eta_sample_count_changes_background_resolution(
        self,
    ) -> None:
        """Declared `eta_sample_count` should affect the declared grid."""

        coarse = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        refined = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        coarse = _with_prepared_numerical_overrides(
            coarse,
            eta_sample_count=128,
        )
        refined = _with_prepared_numerical_overrides(
            refined,
            eta_sample_count=256,
        )
        coarse_background = cmb_background._build_custom_cmb_background(
            coarse,
            cmb_background._resolve_custom_cmb_physical_parameters(coarse),
            cmb_background._resolve_custom_cmb_numerics(coarse),
        )
        refined_background = cmb_background._build_custom_cmb_background(
            refined,
            cmb_background._resolve_custom_cmb_physical_parameters(refined),
            cmb_background._resolve_custom_cmb_numerics(refined),
        )
        self.assertGreater(
            int(refined_background.eta_grid.size),
            int(coarse_background.eta_grid.size),
        )

    def test_background_refinement_meets_final_bound(self) -> None:
        """Background and recombination converge below one percent."""

        backgrounds = []
        for eta_count in (384, 767):
            contract = _speedup_contract(_custom_contract())
            contract["model_name"] = f"BackgroundRefinement{eta_count}"
            contract["numerical"]["eta_sample_count"] = eta_count
            contract["perturbations"]["numerics"] = copy.deepcopy(
                contract["numerical"]
            )
            prepared = _prepare_declared_contract(contract)
            physical = cmb_background._resolve_custom_cmb_physical_parameters(
                prepared
            )
            numerics = cmb_background._resolve_custom_cmb_numerics(prepared)
            backgrounds.append(
                cmb_background._build_custom_cmb_background(
                    prepared,
                    physical,
                    numerics,
                )
            )

        coarse, refined = backgrounds
        for name in (
            "eta0",
            "sound_horizon_mpc",
            "eta_rec",
            "reionization_z",
            "reionization_tau",
        ):
            metric = convergence.evaluate_control_refinement(
                (getattr(coarse, name),),
                (getattr(refined, name),),
                name=f"background {name}",
                tolerance=0.01,
            )
            convergence.require_convergence(metric)
        scale_factor = numpy.geomspace(1.0e-4, 1.0, 512)
        for name in ("visibility_of_eta", "x_e_of_eta"):
            coarse_history = getattr(coarse, name)(
                coarse.eta_of_a(scale_factor)
            )
            refined_history = getattr(refined, name)(
                refined.eta_of_a(scale_factor)
            )
            metric = convergence.evaluate_control_refinement(
                coarse_history,
                refined_history,
                name=f"background {name}",
                tolerance=0.01,
            )
            convergence.require_convergence(metric)

    def test_background_keeps_pre_grid_conformal_time(self) -> None:
        """Declared eta and sound-horizon grids start at the big bang."""

        contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        radiation_density = max(
            float(physical.Omega_r0 or 0.0),
            float(physical.Omega_gamma0),
            1.0e-30,
        )
        expected_eta_start = float(background_data.a_grid[0]) / (
            float(physical.H0_over_c_Mpc_inv) * numpy.sqrt(radiation_density)
        )

        self.assertGreater(float(background_data.eta_grid[0]), 0.0)
        self.assertAlmostEqual(
            float(background_data.eta_grid[0]),
            expected_eta_start,
            places=12,
        )
        self.assertGreater(float(background_data.sound_horizon_mpc), 0.0)

    def test_declared_recombination_uses_post_decoupling_matter_temperature(
        self,
    ) -> None:
        """Declared recombination should not keep matter coupled to the CMB."""

        contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        low_redshift = (background_data.z_grid >= 20.0) & (
            background_data.z_grid <= 100.0
        )
        self.assertTrue(
            numpy.all(background_data.x_e_grid[low_redshift] > 0.0)
        )
        self.assertLess(
            float(numpy.min(background_data.x_e_grid[low_redshift])),
            0.001,
        )

    def test_background_separates_baryon_and_acoustic_sound_speeds(
        self,
    ) -> None:
        """Baryon pressure should not use the photon-baryon sound speed."""

        contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )
        physical = cmb_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        baryon_speed_sq = numpy.asarray(
            background_data.baryon_sound_speed_sq_grid,
            dtype=float,
        )
        acoustic_speed_sq = numpy.square(
            numpy.asarray(background_data.sound_speed_grid, dtype=float)
            / cmb_background._C_LIGHT_KM_S
        )

        self.assertTrue(bool(numpy.all(numpy.isfinite(baryon_speed_sq))))
        self.assertGreater(float(numpy.min(baryon_speed_sq)), 0.0)
        self.assertLess(
            float(numpy.max(baryon_speed_sq)),
            float(numpy.min(acoustic_speed_sq)),
        )

    def test_requested_declared_k_sample_count_is_not_capped(self) -> None:
        """Declared `k_sample_count` should honor declared values above 48."""

        contract = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        contract = _with_prepared_numerical_overrides(
            contract,
            k_sample_count=64,
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )
        self.assertEqual(int(spectrum_data.k_grid.size), 64)

    def test_declared_k_grid_includes_declared_scalar_reference_scales(
        self,
    ) -> None:
        """Declared k sampling should include scalar reference scales."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical_params,
            numerics,
        )
        ells = numpy.arange(20, 121, dtype=int)
        k_grid = cmb_projection._build_projection_k_grid(
            ell_arr=ells,
            background=background_data,
            numerics=numerics,
            perturbation_data=contract["perturbation_data"],
        )
        eta_rec_distance = max(
            float(background_data.eta0) - float(background_data.eta_rec),
            1.0,
        )

        self.assertEqual(int(k_grid.size), int(numerics.k_sample_count))
        for ell_value in (20, 60, 120):
            expected_k = (float(ell_value) + 0.5) / eta_rec_distance
            self.assertTrue(
                bool(
                    numpy.any(
                        numpy.isclose(
                            k_grid,
                            expected_k,
                            rtol=1.0e-12,
                            atol=1.0e-12,
                        )
                    )
                )
            )

    def test_joint_mcmc_keeps_final_phase_grid_floor(self) -> None:
        """Joint likelihood evaluation must not use the smoke k ladder."""

        raw_contract = _declared_scalar_hierarchy_contract()
        raw_contract["numerical"].update(
            {
                "k_sample_count": 64,
                "eta_sample_count": 64,
                "evolution_eta_sample_count": 64,
            }
        )
        raw_contract["perturbations"]["accuracy_controls"] = {
            "accuracy_tier": "final",
            "scalar_reference_ells": [2, 120],
            "runtime_envelope": "bounded",
        }
        contract = _prepare_declared_contract(raw_contract)
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical_params,
            numerics,
        )
        k_grid = cmb_projection._build_projection_k_grid(
            ell_arr=numpy.asarray((20, 60, 120), dtype=int),
            background=background_data,
            numerics=numerics,
            perturbation_data=contract["perturbation_data"],
            allow_final_production_floor=True,
        )
        self.assertGreaterEqual(int(k_grid.size), 512)

    def test_declared_tensor_k_grid_covers_spin2_tail(self) -> None:
        """Tensor k sampling must retain the spin-2 projection tail."""

        contract = _prepare_declared_contract(
            _declared_tensor_hierarchy_contract()
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        background_data = cmb_background._build_custom_cmb_background(
            contract,
            physical_params,
            numerics,
        )
        ells = numpy.asarray((40, 50, 70), dtype=int)
        k_grid = cmb_projection._build_projection_k_grid(
            ell_arr=ells,
            background=background_data,
            numerics=numerics,
            perturbation_data=contract["perturbation_data"],
        )
        eta_rec_distance = max(
            float(background_data.eta0) - float(background_data.eta_rec),
            1.0,
        )
        required_k_max = 1.5 * (70.0 + 16.0) / eta_rec_distance

        self.assertGreaterEqual(float(k_grid[-1]), 5.0 * required_k_max)

    def test_accuracy_controls_reject_underresolved_scalar_numerics(
        self,
    ) -> None:
        """Declared minimum controls should reject under-resolved numerics."""

        base_contract = _speedup_contract(_custom_contract())
        base_contract["numerical"].update(
            {
                "ell_max": 40,
                "photon_hierarchy_l_max": 2,
                "neutrino_hierarchy_l_max": 2,
            }
        )
        cases = (
            ("minimum_ell_max", 61, "ell_max"),
            ("minimum_k_sample_count", 17, "k_sample_count"),
            ("minimum_eta_sample_count", 129, "eta_sample_count"),
            ("minimum_source_grid_multiplier", 2, "source_grid_multiplier"),
            (
                "minimum_photon_hierarchy_l_max",
                3,
                "photon_hierarchy_l_max",
            ),
            (
                "minimum_neutrino_hierarchy_l_max",
                3,
                "neutrino_hierarchy_l_max",
            ),
        )

        for control_name, minimum_value, message in cases:
            contract = copy.deepcopy(base_contract)
            contract["perturbations"]["accuracy_controls"] = {
                control_name: minimum_value,
            }
            with self.subTest(control_name=control_name):
                with self.assertRaisesRegex(ValueError, message):
                    cmb_background._resolve_custom_cmb_numerics(contract)

    def test_runtime_envelope_records_governed_work_units(self) -> None:
        """Declared spectra should carry the governed runtime envelope."""

        cache.clear_cmb_caches()
        contract = _speedup_contract(_analytic_signal_contract())
        contract["perturbations"]["accuracy_controls"] = {
            "runtime_envelope": {
                "maximum_total_work_units": 200000,
            }
        }
        contract = _prepare_declared_contract(contract)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )

        self.assertGreater(
            int(spectrum_data.runtime_envelope["total_work_units"]),
            0,
        )
        self.assertLessEqual(
            int(spectrum_data.runtime_envelope["total_work_units"]),
            200000,
        )
        self.assertIn("evolution_work_units", spectrum_data.runtime_envelope)
        self.assertIn("projection_work_units", spectrum_data.runtime_envelope)
        numerical_envelope = spectrum_data.runtime_envelope[
            "numerical_envelope"
        ]
        self.assertIsNone(numerical_envelope["accuracy_tier"])
        self.assertEqual(numerical_envelope["sectors"], ["scalar"])
        self.assertEqual(
            numerical_envelope["spectrum_relative_tolerances"]["lensed_BB"],
            0.05,
        )
        for phase_name in (
            "compilation",
            "background",
            "evolution",
            "projection",
            "power_spectrum",
        ):
            self.assertGreaterEqual(
                float(spectrum_data.runtime_envelope[f"{phase_name}_seconds"]),
                0.0,
            )

        cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )
        performance = cache.cmb_performance_stats()
        self.assertEqual(int(performance["requests"]), 2)
        self.assertEqual(int(performance["cache_hits"]), 1)
        self.assertIn("total_seconds", performance["phase_seconds"])

    def test_final_accuracy_tier_fails_before_background_work(self) -> None:
        """An under-resolved final request must fail before integration."""

        raw_contract = _speedup_contract(_analytic_signal_contract())
        raw_contract["perturbations"]["accuracy_controls"] = {
            "accuracy_tier": "final",
            "runtime_envelope": "bounded",
        }
        contract = _prepare_declared_contract(raw_contract)
        with mock.patch.object(
            cmb_projection,
            "_build_custom_cmb_background",
        ) as background_builder:
            with self.assertRaisesRegex(ValueError, "under-resolved"):
                cmb_projection._compute_custom_cmb_spectrum_data(
                    contract,
                    numpy.arange(20, 25, dtype=int),
                )
        background_builder.assert_not_called()

    def test_declared_runtime_prepares_graph_once_per_spectrum(self) -> None:
        """Static graph preparation must not scale with Fourier modes."""

        cache.clear_cmb_caches()
        contract = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )
        envelope = spectrum_data.runtime_envelope
        self.assertEqual(int(envelope["static_graph_preparations"]), 1)
        self.assertEqual(
            int(envelope["dynamic_mode_count"]),
            int(envelope["k_sample_count"]),
        )

    def test_declared_runtime_reuse_is_exact_and_parameter_safe(self) -> None:
        """Warm structure must preserve results without stale parameters."""

        cache.clear_cmb_caches()
        raw_contract = _speedup_contract(_analytic_signal_contract())
        contract = _prepare_declared_contract(raw_contract)
        ells = numpy.arange(20, 25, dtype=int)

        cold = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
            requested_spectra=("TT",),
        )
        cold_record = cache.latest_cmb_performance_record()
        cache.clear_cmb_result_caches()
        warm = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
            requested_spectra=("TT",),
        )
        warm_record = cache.latest_cmb_performance_record()

        numpy.testing.assert_array_equal(
            numpy.asarray(cold.spectra["TT"]),
            numpy.asarray(warm.spectra["TT"]),
        )
        self.assertEqual(cold_record["cache_state"], "cold")
        self.assertEqual(warm_record["cache_state"], "warm")
        self.assertEqual(
            int(warm.runtime_envelope["static_graph_preparations"]),
            0,
        )
        self.assertGreater(
            float(cold_record["phase_seconds"]["compilation_seconds"]),
            float(warm_record["phase_seconds"]["compilation_seconds"]),
        )

        exact = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
            requested_spectra=("TT",),
        )
        exact_record = cache.latest_cmb_performance_record()
        numpy.testing.assert_array_equal(
            numpy.asarray(exact.spectra["TT"]),
            numpy.asarray(warm.spectra["TT"]),
        )
        self.assertEqual(exact_record["cache_state"], "exact_cache_hit")
        self.assertEqual(
            float(exact_record["phase_seconds"]["evolution_seconds"]),
            0.0,
        )
        self.assertEqual(
            float(exact_record["phase_seconds"]["projection_seconds"]),
            0.0,
        )

        shifted_raw = _speedup_contract(
            _analytic_signal_contract(source_scale=1.25)
        )
        shifted_contract = _prepare_declared_contract(shifted_raw)
        shifted = cmb_projection._compute_custom_cmb_spectrum_data(
            shifted_contract,
            ells,
            requested_spectra=("TT",),
        )
        shifted_record = cache.latest_cmb_performance_record()
        self.assertEqual(
            contract["runtime_signature"],
            shifted_contract["runtime_signature"],
        )
        self.assertEqual(shifted_record["cache_state"], "warm")
        self.assertEqual(
            int(shifted.runtime_envelope["static_graph_preparations"]),
            0,
        )
        self.assertFalse(
            numpy.array_equal(
                numpy.asarray(shifted.spectra["TT"]),
                numpy.asarray(warm.spectra["TT"]),
            )
        )

    def test_projection_batches_radial_recurrence_work(self) -> None:
        """Projection telemetry must show shared radial mode preparation."""

        cache.clear_cmb_caches()
        contract = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
            requested_spectra=("TT",),
        )
        envelope = spectrum_data.runtime_envelope
        self.assertGreater(
            int(envelope["projection_bessel_batch_count"]),
            0,
        )
        self.assertEqual(
            int(envelope["projection_bessel_mode_count"]),
            int(envelope["k_sample_count"]),
        )
        self.assertLess(
            int(envelope["projection_bessel_batch_count"]),
            int(envelope["projection_bessel_mode_count"]),
        )

    def test_runtime_cache_state_requires_matching_request_shape(self) -> None:
        """A changed spectrum request is cold, not a warm parameter rebound."""

        cache.clear_cmb_caches()
        contract = _prepare_declared_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.asarray((20, 25), dtype=int),
            requested_spectra=("TT",),
        )
        cache.clear_cmb_result_caches()
        cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.asarray((20, 30), dtype=int),
            requested_spectra=("TT",),
        )
        record = cache.latest_cmb_performance_record()
        self.assertIsNotNone(record)
        self.assertEqual(record["cache_state"], "cold")

    def test_declared_spectrum_cache_identity_covers_runtime_inputs(self):
        """Every result-affecting surface must change cache identity."""

        contract = _analytic_signal_contract()

        def _identity(
            candidate,
            *,
            ells=(20, 30),
            requested=("TT",),
        ):
            return cmb_background._custom_cmb_spectrum_cache_key(
                candidate,
                ells,
                None,
                requested_spectra=requested,
            )

        identities = [_identity(contract)]
        structure_changed = copy.deepcopy(contract)
        structure_changed["perturbations"]["equations"]["evolve_signal_mode"][
            "rhs"
        ] = "-2.0 * decay_rate * signal_mode"
        identities.append(_identity(structure_changed))
        parameters_changed = copy.deepcopy(contract)
        parameters_changed["model_parameters"]["source_scale"] = 2.0
        identities.append(_identity(parameters_changed))
        grid_changed = copy.deepcopy(contract)
        grid_changed["numerical"]["k_sample_count"] = 7
        identities.append(_identity(grid_changed))
        accuracy_changed = copy.deepcopy(contract)
        accuracy_changed["perturbations"]["accuracy_controls"] = {
            "adaptive": {"source_relative_tolerance": 0.02}
        }
        identities.append(_identity(accuracy_changed))
        identities.append(_identity(contract, requested=("EE",)))
        identities.append(_identity(contract, ells=(20, 20, 30)))

        self.assertEqual(len(set(identities)), len(identities))

    def test_declared_spectrum_payload_distinguishes_availability_states(self):
        """Computed, unrequested, and physical-zero outputs stay distinct."""

        contract = _analytic_signal_contract()
        observables = contract["perturbations"]["observables"]
        for name in ("TE", "EE", "PP"):
            observables[name] = {
                "kind": "angular_power_spectrum",
                "primary": "signal_transfer",
                "secondary": "signal_transfer",
            }
        prepared = _prepare_declared_contract(contract)
        tt_only = cmb_projection._compute_custom_cmb_spectrum_data(
            prepared,
            numpy.asarray((20, 30), dtype=int),
            requested_spectra=("TT",),
        )
        lensing_inputs = cmb_projection._compute_custom_cmb_spectrum_data(
            prepared,
            numpy.asarray((20, 30), dtype=int),
            requested_spectra=("TT", "TE", "EE", "BB", "PP"),
        )

        self.assertEqual(tt_only.spectrum_availability["TT"], "computed")
        self.assertEqual(tt_only.spectrum_availability["EE"], "unrequested")
        self.assertNotIn("EE", tt_only.spectra)
        self.assertEqual(
            lensing_inputs.spectrum_availability["BB"],
            "physical_zero",
        )
        self.assertNotIn("BB", lensing_inputs.spectra)
        self.assertEqual(
            lensing_inputs.runtime_envelope["spectrum_availability"]["BB"],
            "physical_zero",
        )
        with self.assertRaises(ValueError):
            tt_only.spectra["TT"][0] = 0.0

    def test_declared_component_spectrum_executes_by_exact_name(self) -> None:
        """Explicit component observables must not collapse into aliases."""

        contract = _analytic_signal_contract()
        observables = contract["perturbations"]["observables"]
        observables["scalar_TT"] = observables.pop("TT")
        prepared = _prepare_declared_contract(contract)
        ells = numpy.asarray((20, 30, 40), dtype=int)

        spectrum = cmb.compute_cmb_spectrum_from_contract(
            prepared,
            ells,
            spectra=("scalar_TT",),
        )

        self.assertTrue(numpy.all(numpy.isfinite(spectrum)))
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported CMB observable 'TT'",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                prepared,
                ells,
                spectra=("TT",),
            )

    def test_public_spectrum_request_rejects_empty_names(self):
        """A public request must identify at least one output."""

        prepared = _prepare_declared_contract(_analytic_signal_contract())
        ells = numpy.asarray((20, 30), dtype=int)

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            cmb.compute_cmb_spectrum_from_contract(
                prepared,
                ells,
                spectra=(),
            )

    def test_declared_scalar_and_vector_aliases_match_total_spectra(self):
        """Single-sector component and total aliases must be identical."""

        cases = (
            ("scalar", _declared_scalar_hierarchy_contract(), "TT"),
            ("vector", _declared_vector_hierarchy_contract(), "BB"),
        )
        ells = numpy.asarray((20, 30, 40), dtype=int)
        for sector, raw_contract, base_name in cases:
            with self.subTest(sector=sector):
                prepared = _prepare_declared_contract(
                    _speedup_contract(raw_contract)
                )
                spectra = cmb.compute_cmb_spectrum_from_contract(
                    prepared,
                    ells,
                    spectra=(
                        base_name,
                        f"{sector}_{base_name}",
                        f"total_{base_name}",
                    ),
                )
                numpy.testing.assert_allclose(
                    spectra[base_name],
                    spectra[f"{sector}_{base_name}"],
                )
                numpy.testing.assert_allclose(
                    spectra[base_name],
                    spectra[f"total_{base_name}"],
                )

    def test_mixed_sector_total_alias_requires_a_total_observable(self):
        """A sector output must not masquerade as a mixed total spectrum."""

        perturbation_data = SimpleNamespace(
            manifest_summary={"sector_names": ("scalar", "tensor")},
            observables={"TT": SimpleNamespace(sector="scalar")},
        )
        available = {"TT": numpy.ones(2)}

        self.assertEqual(
            cmb_solver._resolve_available_spectrum_name(
                "scalar_TT",
                perturbation_data=perturbation_data,
                available_spectra=available,
            ),
            "TT",
        )
        self.assertIsNone(
            cmb_solver._resolve_available_spectrum_name(
                "total_TT",
                perturbation_data=perturbation_data,
                available_spectra=available,
            )
        )

    def test_declared_generated_modes_use_declared_graph_evolution(
        self,
    ) -> None:
        """Generated modes should use one finite declared-graph runtime."""

        cache.clear_cmb_caches()
        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(
                initial_mode="cdm_isocurvature",
            )
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
            requested_spectra=("TT",),
        )
        envelope = spectrum_data.runtime_envelope
        self.assertEqual(int(envelope["contract_static_preparations"]), 1)
        self.assertEqual(int(envelope["model_static_preparations"]), 1)
        self.assertEqual(int(envelope["request_specific_preparations"]), 1)
        self.assertEqual(int(envelope["batch_count"]), 1)
        self.assertEqual(
            int(envelope["batch_mode_count"]),
            int(envelope["k_sample_count"]),
        )
        self.assertGreater(int(envelope["batched_rk_stage_count"]), 0)
        self.assertTrue(
            numpy.all(
                numpy.isfinite(numpy.asarray(spectrum_data.spectra["TT"]))
            )
        )
        repeated = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
            requested_spectra=("TT",),
        )
        numpy.testing.assert_array_equal(
            numpy.asarray(repeated.spectra["TT"]),
            numpy.asarray(spectrum_data.spectra["TT"]),
        )
        self.assertGreater(
            cache.cmb_cache_stats()["declared_spectrum"]["hits"],
            0,
        )

    def test_batched_exact_collision_matches_scalar_exact_steps(self) -> None:
        """Vectorized collision blocks must preserve scalar exact updates."""

        matrices = numpy.asarray(
            (
                (
                    (
                        (-1.0, 0.5, 0.0, 0.0),
                        (0.25, -0.125, 0.0, 0.0),
                        (0.0, 0.0, -0.9, 0.6),
                        (0.0, 0.0, 0.1, -0.4),
                    ),
                    (
                        (-1.0, 0.25, 0.0, 0.0),
                        (0.5, -0.25, 0.0, 0.0),
                        (0.0, 0.0, -0.9, 0.6),
                        (0.0, 0.0, 0.1, -0.4),
                    ),
                )
            ),
            dtype=float,
        ).reshape(2, 4, 4)
        states = numpy.asarray(
            ((0.2, -0.4, 0.1, -0.2), (-0.3, 0.6, -0.5, 0.4)),
            dtype=float,
        )
        scales = numpy.asarray((0.75, 1.25), dtype=float)
        batched = cmb_projection._exact_batched_linear_collision_step(
            operator_matrices=matrices,
            dt=0.125,
            target_states=states,
            operator_scales=scales,
        )
        expected = numpy.asarray(
            [
                cmb_projection._exact_linear_collision_step(
                    operator_matrix=matrix,
                    dt=0.125,
                    target_state=state,
                    operator_scale=scale,
                )
                for matrix, state, scale in zip(matrices, states, scales)
            ],
            dtype=float,
        )

        numpy.testing.assert_allclose(batched, expected, rtol=1.0e-12)

    def test_compiled_equation_program_is_a_reusable_executor(self) -> None:
        """Compiled equation plans must run without per-stage exec dispatch."""

        program = evolution._compile_equation_program(
            (
                (0, "eta", "signal + 1.0", None),
                (1, "eta", None, 0),
            )
        )
        state_vector = numpy.asarray(
            ((1.0, 2.0), (3.0, 4.0)),
            dtype=float,
        )
        derivative = numpy.zeros_like(state_vector)

        program(
            {"signal": numpy.asarray((1.0, 2.0), dtype=float)},
            state_vector,
            derivative,
            {"eta": 2.0},
        )

        numpy.testing.assert_array_equal(
            derivative,
            numpy.asarray(((4.0, 6.0), (2.0, 4.0)), dtype=float),
        )

    def test_compiled_equation_program_hoists_repeated_context_reads(
        self,
    ) -> None:
        """One batched stage should read each declared input once."""

        class _ReadCountingContext(dict):
            """Record lookup counts while preserving mapping semantics."""

            def __init__(self, *args, **kwargs):
                """Initialize the input mapping and empty read counters."""

                super().__init__(*args, **kwargs)
                self.reads: dict[str, int] = {}

            def __getitem__(self, key):
                """Record one declared-symbol lookup."""

                normalized = str(key)
                self.reads[normalized] = self.reads.get(normalized, 0) + 1
                return super().__getitem__(key)

        context = _ReadCountingContext(
            repeated=numpy.asarray((2.0, 3.0), dtype=float),
            offset=numpy.asarray((1.0, 1.0), dtype=float),
        )
        coordinate_rates = _ReadCountingContext(
            eta=2.0,
        )
        program = evolution._compile_equation_program(
            (
                (0, "eta", "repeated * repeated + offset", None),
                (1, "eta", "repeated - offset", None),
            )
        )
        derivative = numpy.zeros((2, 2), dtype=float)

        program(
            context,
            numpy.zeros((2, 2), dtype=float),
            derivative,
            coordinate_rates,
        )

        numpy.testing.assert_array_equal(
            derivative,
            numpy.asarray(((10.0, 20.0), (2.0, 4.0)), dtype=float),
        )
        self.assertEqual(context.reads, {"offset": 1, "repeated": 1})
        self.assertEqual(coordinate_rates.reads, {"eta": 1})

    def test_batched_row_equation_program_matches_vector_equations(
        self,
    ) -> None:
        """Scalar row execution must preserve vectorized equation outputs."""

        specifications = (
            (0, "eta", "signal + offset", None),
            (1, "eta", "signal * signal", None),
            (2, "eta", None, 1),
        )
        context = {
            "signal": numpy.asarray((1.0, 2.0, 3.0), dtype=float),
            "offset": 0.5,
        }
        state = numpy.asarray(
            (
                (0.0, 0.0, 0.0),
                (4.0, 5.0, 6.0),
                (0.0, 0.0, 0.0),
            ),
            dtype=float,
        )
        coordinate_rates = {"eta": 2.0}
        vector_derivative = numpy.zeros_like(state)
        row_derivative = numpy.zeros_like(state)

        evolution._compile_equation_program(specifications)(
            context,
            state,
            vector_derivative,
            coordinate_rates,
        )
        evolution._compile_batched_row_equation_program(
            specifications,
            ("signal",),
        )(
            context,
            state,
            row_derivative,
            coordinate_rates,
        )

        numpy.testing.assert_array_equal(row_derivative, vector_derivative)

    def test_compiled_context_program_is_a_reusable_executor(self) -> None:
        """Compiled context plans must retain declared suppression behavior."""

        program = evolution._compile_ordered_context_program(
            (
                ("double_signal", "2.0 * signal"),
                ("shifted_signal", "double_signal + 1.0"),
            )
        )
        context = {"signal": numpy.asarray((1.0, 2.0), dtype=float)}
        suppressed = {"double_signal": numpy.zeros(2, dtype=float)}
        context.update(suppressed)

        program(context, suppressed)

        numpy.testing.assert_array_equal(
            context["double_signal"],
            numpy.zeros(2, dtype=float),
        )
        numpy.testing.assert_array_equal(
            context["shifted_signal"],
            numpy.ones(2, dtype=float),
        )

    def test_compiled_context_program_can_overwrite_stale_relation(
        self,
    ) -> None:
        """Recompute relation targets from the current state context."""

        program = evolution._compile_ordered_context_program(
            (("double_signal", "2.0 * signal"),),
            overwrite_outputs=("double_signal",),
        )
        context = {"signal": 3.0, "double_signal": -99.0}

        program(context, {})

        self.assertEqual(context["double_signal"], 6.0)

    def test_compiled_expression_tuple_reuses_one_context_executor(
        self,
    ) -> None:
        """Compiled expression tuples must preserve declared value order."""

        program = evolution._compile_expression_tuple_program(
            ("signal + 1.0", "2.0 * signal"),
        )

        values = program({"signal": numpy.asarray((1.0, 2.0), dtype=float)})

        numpy.testing.assert_array_equal(
            values[0],
            numpy.asarray((2.0, 3.0), dtype=float),
        )
        numpy.testing.assert_array_equal(
            values[1],
            numpy.asarray((2.0, 4.0), dtype=float),
        )

    def test_batched_declared_evolution_matches_scalar_modes(self) -> None:
        """Batched declared modes must preserve scalar-spectrum convergence."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(
                initial_mode="cdm_isocurvature",
            )
        )
        ells = numpy.asarray((20, 60, 120), dtype=int)
        cache.clear_cmb_caches()
        batched = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
            requested_spectra=("TT",),
        )
        cache.clear_cmb_caches()
        with mock.patch.object(
            cmb_projection,
            "_can_batch_declared_evolution",
            return_value=False,
        ):
            scalar = cmb_projection._compute_custom_cmb_spectrum_data(
                contract,
                ells,
                requested_spectra=("TT",),
            )

        numpy.testing.assert_allclose(
            numpy.asarray(batched.spectra["TT"]),
            numpy.asarray(scalar.spectra["TT"]),
            rtol=2.0e-5,
            atol=1.0e-14,
        )

    def test_batched_rhs_excludes_diagnostic_context_steps(self) -> None:
        """Batched RK stages must evaluate only equation-required values."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.asarray((20, 60, 120), dtype=int),
            requested_spectra=("TT",),
        )
        envelope = spectrum_data.runtime_envelope

        self.assertLess(
            int(envelope["batched_rhs_value_step_count"]),
            int(envelope["batched_diagnostic_value_step_count"]),
        )

    def test_adaptive_k_projection_reuses_batched_mode_evolution(self) -> None:
        """Adaptive k projection must not fall back to scalar mode solves."""

        raw_contract = _declared_scalar_hierarchy_contract()
        accuracy_controls = raw_contract["perturbations"]["accuracy_controls"]
        accuracy_controls["adaptive_k_quadrature"] = {
            "mode": "source",
            "node_count": 64,
            "ell_min": 2,
            "ell_stride": 1,
            "eta_stride": 4,
        }
        contract = _prepare_declared_contract(raw_contract)
        cache.clear_cmb_caches()
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.asarray((20, 60, 120), dtype=int),
            requested_spectra=("TT",),
        )

        envelope = spectrum_data.runtime_envelope
        self.assertGreaterEqual(int(envelope["batch_count"]), 1)
        self.assertTrue(
            bool(numpy.all(numpy.isfinite(spectrum_data.spectra["TT"])))
        )

    def test_runtime_envelope_records_explicit_work_hints(self) -> None:
        """Explicit work hints should not reject a valid physical request."""

        contract = _speedup_contract(_analytic_signal_contract())
        contract["perturbations"]["accuracy_controls"] = {
            "runtime_envelope": {
                "maximum_total_work_units": 1000,
            }
        }
        contract = _prepare_declared_contract(contract)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )
        self.assertEqual(
            spectrum_data.runtime_envelope["work_limits"],
            {"maximum_total_work_units": 1000},
        )
        self.assertFalse(
            spectrum_data.runtime_envelope["work_limits_enforced"]
        )

    def test_conservation_rule_violation_fails_loudly(self) -> None:
        """Conservation residuals should fail once they exceed tolerance."""

        contract = _analytic_signal_contract()
        contract["perturbations"]["conservation_rules"] = {
            "signal_balance": {
                "kind": "absolute_max",
                "expression": "signal_mode",
                "tolerance": 1.0e-12,
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "conservation rule exceeded tolerance",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_declared_recombination_quantities_change_background(
        self,
    ) -> None:
        """Declared recombination hooks should alter the solved background."""

        def _recombination_contract(peebles_c: float) -> dict[str, object]:
            contract = _speedup_contract(_custom_contract())
            contract["background"]["recombination"] = {
                "quantities": {
                    "hydrogen_temperature_K": "2.7255 * (1.0 + z)",
                    "hydrogen_alpha_B": (
                        "1.0e-19 * "
                        "((hydrogen_temperature_K / 3000.0) ** -0.5)"
                    ),
                    "beta_continuum": (
                        "5.0e-20 * "
                        "((hydrogen_temperature_K / 3000.0) ** 0.5)"
                    ),
                    "peebles_c": f"{peebles_c:.16g}",
                }
            }
            return _prepare_declared_contract(contract)

        baseline = _recombination_contract(0.4)
        changed = _recombination_contract(0.9)
        baseline_background = cmb_background._build_custom_cmb_background(
            baseline,
            cmb_background._resolve_custom_cmb_physical_parameters(baseline),
            cmb_background._resolve_custom_cmb_numerics(baseline),
        )
        changed_background = cmb_background._build_custom_cmb_background(
            changed,
            cmb_background._resolve_custom_cmb_physical_parameters(changed),
            cmb_background._resolve_custom_cmb_numerics(changed),
        )

        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        changed_background.x_e_grid
                        - baseline_background.x_e_grid
                    )
                )
            ),
            1.0e-8,
        )

    def test_declared_recombination_quantities_require_full_hook_set(
        self,
    ) -> None:
        """Partial recombination hooks should fail with a named error."""

        contract = _speedup_contract(_custom_contract())
        contract["background"]["recombination"] = {
            "quantities": {
                "hydrogen_temperature_K": "3000.0",
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "must define 'hydrogen_alpha_B'",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_momentum_grid_accuracy_control_rejects_short_grid(
        self,
    ) -> None:
        """Massive-neutrino grids should honor declared count floors."""

        contract = _speedup_contract(
            _declared_scalar_hierarchy_contract(
                include_massive_neutrino=True,
            )
        )
        contract["perturbations"]["numerics"]["momentum_grids"] = {
            "massive_neutrino_default": {
                "count": 4,
            }
        }
        contract["perturbations"]["accuracy_controls"][
            "minimum_momentum_grid_count"
        ] = {
            "massive_neutrino_default": 5,
        }
        with self.assertRaisesRegex(
            ValueError,
            "momentum_grids.massive_neutrino_default.count >= 5",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                _prepare_declared_contract(contract),
                numpy.arange(20, 24, dtype=int),
                spectra=("TT",),
            )

    def test_massive_neutrino_q_moments_match_independent_quadrature(
        self,
    ) -> None:
        """Resolved q moments should match an independent log-q integral."""

        contract_data = _declared_scalar_hierarchy_contract(
            include_massive_neutrino=True,
            sum_mnu=0.5,
        )
        contract_data["numerical"]["momentum_grids"][
            "massive_neutrino_default"
        ].update(
            {
                "quadrature_order": 2,
            }
        )
        contract = _prepare_declared_contract(contract_data)
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        runtime = evolution._resolve_declared_momentum_grid_runtimes(
            contract["perturbation_data"],
            model_parameters=contract["param_map"],
            physical_params=physical_params,
        )[0]
        self.assertEqual(runtime.quadrature_order, 2)
        self.assertTrue(numpy.all(numpy.diff(runtime.points) > 0.0))
        self.assertTrue(numpy.all(runtime.weights > 0.0))
        self.assertAlmostEqual(
            float(numpy.sum(runtime.weights)),
            float(numpy.log(runtime.points[-1] / runtime.points[0])),
            places=12,
        )

        scale_factor = 0.7
        context = evolution._declared_momentum_grid_context(
            contract["perturbation_data"],
            model_parameters=contract["param_map"],
            physical_params=physical_params,
            scale_factor=scale_factor,
        )
        neutrino_temperature = float(context["neutrino_temperature_eV"])
        mass_eV = float(context["massive_neutrino_mass_eV"])
        mass_ratio = scale_factor * mass_eV / neutrino_temperature
        log_q_min = float(numpy.log(runtime.points[0]))
        log_q_max = float(numpy.log(runtime.points[-1]))

        def _occupation(log_q: float) -> float:
            q_value = float(numpy.exp(log_q))
            return float(1.0 / (1.0 + numpy.exp(q_value)))

        def _energy(log_q: float) -> float:
            q_value = float(numpy.exp(log_q))
            return float(numpy.sqrt(q_value * q_value + mass_ratio**2))

        reference_density = scipy_quad(
            lambda log_q: _occupation(log_q)
            * numpy.exp(3.0 * log_q)
            * _energy(log_q),
            log_q_min,
            log_q_max,
            epsabs=1.0e-11,
            epsrel=1.0e-11,
        )[0]
        reference_pressure = scipy_quad(
            lambda log_q: _occupation(log_q)
            * numpy.exp(5.0 * log_q)
            / (3.0 * _energy(log_q)),
            log_q_min,
            log_q_max,
            epsabs=1.0e-11,
            epsrel=1.0e-11,
        )[0]
        measured_density = float(
            context["massive_neutrino_background_density_moment"]
        )
        measured_pressure = float(
            context["massive_neutrino_background_pressure_moment"]
        )
        self.assertLess(
            abs(measured_density / reference_density - 1.0),
            2.0e-3,
        )
        self.assertLess(
            abs(
                measured_pressure / measured_density
                - reference_pressure / reference_density
            ),
            2.0e-3,
        )

    def test_massive_neutrino_q_grid_rejects_invalid_definition(self) -> None:
        """Invalid q bounds, counts, and rules must fail before evolution."""

        for invalid_definition, message in (
            ({"count": 1}, "count must be an integer >= 2"),
            ({"q_min": 0.0}, "requires 0 < q_min < q_max"),
            ({"q_max": 0.01}, "requires 0 < q_min < q_max"),
            ({"quadrature_order": 4}, "quadrature_order must be 2"),
        ):
            with self.subTest(invalid_definition=invalid_definition):
                contract_data = _declared_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                )
                contract_data["numerical"]["momentum_grids"][
                    "massive_neutrino_default"
                ].update(invalid_definition)
                contract = _prepare_declared_contract(contract_data)
                physical_params = (
                    cmb_background._resolve_custom_cmb_physical_parameters(
                        contract
                    )
                )
                with self.assertRaisesRegex(ValueError, message):
                    evolution._declared_momentum_grid_context(
                        contract["perturbation_data"],
                        model_parameters=contract["param_map"],
                        physical_params=physical_params,
                        scale_factor=0.5,
                    )

    def test_non_massive_contract_has_no_q_runtime(self) -> None:
        """Momentum grids must stay inert without a massive species."""

        contract_data = _declared_scalar_hierarchy_contract()
        contract_data["numerical"]["momentum_grids"] = {
            "unused_grid": {
                "count": 8,
                "q_min": 0.05,
                "q_max": 18.0,
            }
        }
        contract_data["perturbations"]["hierarchy_families"][
            "photon_temperature"
        ]["momentum_grid"] = "unused_grid"
        contract_data["perturbations"]["numerics"]["momentum_grids"] = {
            "unused_grid": {
                "count": 8,
                "q_min": 0.05,
                "q_max": 18.0,
            }
        }
        contract = _prepare_declared_contract(contract_data)
        perturbation_data = contract["perturbation_data"]
        self.assertFalse(
            any("_q" in name for name in perturbation_data.variables)
        )
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        context = evolution._declared_momentum_grid_context(
            perturbation_data,
            model_parameters=contract["param_map"],
            physical_params=physical_params,
            scale_factor=0.5,
        )
        self.assertEqual(context, {})

    def test_background_pressure_and_curvature_symbols_change_outputs(
        self,
    ) -> None:
        """Pressure/equation-of-state background outputs should feed math."""

        baseline = _speedup_contract(_generic_background_custom_contract())
        changed = _speedup_contract(_generic_background_custom_contract())
        source_adjustment = (
            " + 0.05 * (dark_energy_pressure_today + Omega_k0) * k * Psi"
        )
        baseline_equations = baseline["perturbations"]["equations"]
        changed_equations = changed["perturbations"]["equations"]
        baseline_equations["evolve_theta_b"]["rhs"] += source_adjustment
        changed_equations["evolve_theta_b"]["rhs"] += source_adjustment
        changed_background = changed["background"]["derived"]
        changed_background["dark_energy_pressure_today"] = (
            "1.4 * w0 * Omega_de0"
        )
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("TT",),
        )["TT"]
        changed_tt = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("TT",),
        )["TT"]
        self.assertGreater(
            float(numpy.max(numpy.abs(changed_tt - baseline_tt))),
            1.0e-12,
        )

    def test_missing_declared_background_h_fails_loudly(self) -> None:
        """Declared contracts must provide the background H."""

        contract = _speedup_contract(_custom_contract())
        contract["background"]["derived"].pop("H", None)
        with self.assertRaisesRegex(
            ValueError,
            "must provide a derived expansion history",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_incomplete_background_declarations_fail_early(self) -> None:
        """Missing required physical background inputs should fail early."""

        contract = _speedup_contract(_generic_background_custom_contract())
        contract["background"]["derived"].pop("Omega_b0", None)
        contract["background"]["derived"].pop("Omega_m0", None)
        contract["param_map"].pop("baryon_fraction_today", None)
        derived_background = contract["background"]["derived"]
        derived_background["matter_total"] = "0.311"
        derived_background["Omega_c0"] = "matter_total"
        derived_background["Omega_de0"] = (
            "1.0 - matter_total - Omega_r0 - Omega_k0"
        )
        derived_background["H"] = (
            "H0 * sqrt("
            "Omega_r0 / (a ** 4) + "
            "matter_total / (a ** 3) + "
            "Omega_k0 / (a ** 2) + "
            "Omega_de0 * (a ** (-3.0 * (1.0 + w0)))"
            ")"
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires explicit baryon density",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_invalid_background_curvature_override_fails_loudly(self) -> None:
        """Invalid declared curvature histories should fail before runtime."""

        contract = _speedup_contract(_generic_background_custom_contract())
        derived_background = contract["background"]["derived"]
        derived_background["chi"] = "z"
        derived_background["angular_diameter_distance"] = "-0.5 * z"
        with self.assertRaisesRegex(
            ValueError,
            "curvature histories must stay non-negative",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_multi_hop_derived_context_resolves_fully(self) -> None:
        """Derived chains should resolve across more than one dependency."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["derived"]["temperature_drive_mid"] = {
            "expression": "theta_gamma0 + Phi",
        }
        contract["perturbations"]["derived"]["temperature_drive_outer"] = {
            "expression": "temperature_drive_mid + Psi",
        }
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "0.15 * temperature_drive_outer"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_missing_initial_conditions_fail_loudly(self) -> None:
        """Missing initial conditions should fail before evolution."""

        contract = _speedup_contract(_custom_contract())
        del contract["perturbations"]["initial_conditions"]["theta_b_seed"]
        with self.assertRaisesRegex(
            ValueError,
            "missing required initial conditions",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_missing_observable_mappings_fail_loudly(self) -> None:
        """Missing observable mappings should fail clearly."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["observables"] = {}
        with self.assertRaisesRegex(ValueError, "must declare observables"):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_unsupported_projection_fails_loudly(self) -> None:
        """Unsupported projections should be rejected before evolution."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["observables"]["temperature"][
            "projection"
        ] = "bogus_projection"
        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_missing_projection_source_role_fails_loudly(self) -> None:
        """Projection-role mismatches should fail during compilation."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["observables"]["polarization_e"][
            "source_terms"
        ] = {"signal": "polarization_source"}
        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("EE",),
            )

    def test_b_mode_projection_requires_odd_parity_source(self) -> None:
        """BB should fail before runtime on scalar-like B-source plumbing."""

        contract = _speedup_contract(_custom_contract(include_bb=True))
        contract["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "visibility * theta_gamma2"
        with self.assertRaisesRegex(
            ValueError,
            "requires an odd-parity declared source ancestry",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("BB",),
            )

    def test_custom_b_mode_projection_rejects_scalar_multi_source(
        self,
    ) -> None:
        """Custom B-mode kernels should reject scalar-only extra sources."""

        contract = _speedup_contract(_custom_contract(include_bb=True))
        contract["perturbations"]["sources"]["polarization_aux_source"] = {
            "expression": "visibility * theta_gamma2",
            "role": "polarization_aux",
        }
        contract["perturbations"]["observables"]["polarization_b"] = {
            "kind": "transfer_component",
            "projection": "custom_line_of_sight",
            "kernel": "spin2_b_window",
            "source_terms": {
                "polarization_b": "polarization_b_source",
                "polarization_aux": "polarization_aux_source",
            },
            "required_projection_roles": ["b_mode"],
        }
        with self.assertRaisesRegex(
            ValueError,
            "requires source 'polarization_aux_source' to provide declared "
            "projection roles: b_mode",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("BB",),
            )

    def test_lensing_projection_requires_potential_role(self) -> None:
        """PP should fail before runtime without a declared potential role."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        contract["perturbations"]["observables"]["lensing_potential"][
            "source_terms"
        ] = {"signal": "lensing_potential"}
        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles: potential",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("PP",),
            )

    def test_nonfinite_expression_results_fail_loudly(self) -> None:
        """Non-finite source expressions should fail clearly."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "sqrt(-1)"
        with self.assertRaisesRegex(
            ValueError,
            "Declared source term produced non-finite values",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_nonfinite_evolution_states_fail_loudly(self) -> None:
        """Non-finite evolution states should fail clearly."""

        contract = _speedup_contract(_custom_contract())
        theta_b_equation = contract["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        theta_b_equation["rhs"] = "1.0 / (delta_b - delta_b)"
        with self.assertRaisesRegex(
            ValueError,
            "must be finite",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_custom_cached_path_does_not_call_camb(self) -> None:
        """The cached plugin route should not use the standard CAMB path."""

        plugin = _CustomCMBPlugin()
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch.object(
            camb_reference,
            "_compute_cmb_spectrum_direct",
            side_effect=AssertionError("standard CAMB path should not run"),
        ):
            with mock.patch.object(
                camb_reference.camb,
                "get_results",
                side_effect=AssertionError(
                    "CAMB prediction path should not run"
                ),
            ):
                result = cmb.compute_cmb_spectrum_cached(
                    plugin,
                    plugin.INITIAL_GUESSES,
                    ells,
                    spectra=("TT", "TE", "EE"),
                )

        self.assertEqual(set(result), {"TT", "TE", "EE"})
        for spectrum in result.values():
            self.assertTrue(numpy.all(numpy.isfinite(spectrum)))
            self.assertEqual(spectrum.shape, (ells.size,))

    def test_custom_cached_path_uses_precompiled_declared_runtime(
        self,
    ) -> None:
        """The cached route should reuse precompiled declared runtime data."""

        contract = _prepare_declared_contract(
            _speedup_contract(_custom_contract())
        )

        class _PrecompiledRuntimePlugin(_CustomCMBPlugin):
            """Plugin stub exposing one precompiled declared runtime."""

            def get_cmb_declared_runtime(self, _params):
                return contract

        plugin = _PrecompiledRuntimePlugin()
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            side_effect=AssertionError(
                "declared runtime should reuse precompiled perturbation data"
            ),
        ):
            result = cmb.compute_cmb_spectrum_cached(
                plugin,
                plugin.INITIAL_GUESSES,
                ells,
                spectra=("TT",),
            )

        self.assertTrue(numpy.all(numpy.isfinite(result)))
        self.assertEqual(result.shape, (ells.size,))

    def test_declared_scalar_hierarchy_materializes_generated_hierarchy(
        self,
    ) -> None:
        """The declared scalar route should compile hierarchy data."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]

        self.assertTrue(
            perturbation_data.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertIn("theta_gamma8", perturbation_data.variables)
        self.assertIn("e_gamma8", perturbation_data.variables)
        self.assertIn("nu_l5", perturbation_data.variables)
        self.assertIn("TT", perturbation_data.observables)
        self.assertIn("TE", perturbation_data.observables)
        self.assertIn("EE", perturbation_data.observables)
        self.assertIn("PP", perturbation_data.observables)
        self.assertIn("TP", perturbation_data.observables)
        self.assertIn("EP", perturbation_data.observables)
        self.assertEqual(
            perturbation_data.derived["polarization_moment"].expression,
            "theta_gamma2 + e_gamma0 + e_gamma2",
        )
        self.assertEqual(
            perturbation_data.initial_conditions["e_gamma2_seed"].expression,
            "theta_gamma2 / 4.0",
        )
        self.assertEqual(
            perturbation_data.initial_conditions["e_gamma3_seed"].expression,
            "(3.0 / 28.0) * acoustic_k * theta_gamma2 / collision_rate",
        )
        self.assertIn(
            "acoustic_k * theta_gamma2 / collision_rate",
            perturbation_data.initial_conditions[
                "theta_gamma3_seed"
            ].expression,
        )
        self.assertIn(
            "acoustic_k * theta_gamma2 / collision_rate",
            perturbation_data.initial_conditions["e_gamma3_seed"].expression,
        )
        self.assertEqual(
            perturbation_data.sources["temperature_quadrupole"].expression,
            "0.0",
        )
        self.assertEqual(
            perturbation_data.sources[
                "temperature_quadrupole_derivative"
            ].expression,
            "0.0",
        )
        self.assertEqual(
            perturbation_data.sources["polarization_source"].expression,
            "0.75 * visibility * polarization_moment",
        )
        self.assertEqual(
            perturbation_data.sources["lensing_potential"].expression,
            "Phi + Psi",
        )
        self.assertEqual(
            perturbation_data.equations["evolve_theta_gamma0"].rhs,
            "-acoustic_k * theta_gamma1 - Phi_tau",
        )
        self.assertIn(
            "- 0.6 * acoustic_k * nu_l3",
            perturbation_data.equations["evolve_sigma_nu"].rhs,
        )
        self.assertIn(
            "0.4285714285714285 * acoustic_k * sigma_nu",
            perturbation_data.equations["evolve_nu_l3"].rhs,
        )
        self.assertIn(
            "scalar_lapse_seed / 15.0",
            perturbation_data.initial_conditions["sigma_nu_seed"].expression,
        )

    def test_declared_scalar_tight_coupling_uses_declared_collision_block(
        self,
    ) -> None:
        """Scalar tight coupling must come from declared collision metadata."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        collision = contract["perturbation_data"].collision_operators[
            "thomson_drag"
        ]
        self.assertEqual(collision.integration_strategy, "exact")
        self.assertEqual(
            tuple(target.kind for target in collision.exact_form.targets),
            (
                "photon_temperature_dipole",
                "baryon_velocity_divergence",
                "photon_temperature_quadrupole",
                "photon_polarization_quadrupole",
            ),
        )
        self.assertEqual(collision.exact_form.matrix[2][2], "-0.8")
        self.assertEqual(collision.exact_form.matrix[2][3], "0.1")
        self.assertEqual(collision.exact_form.matrix[3][2], "0.05")
        self.assertEqual(collision.exact_form.matrix[3][3], "-0.25")

    def test_declared_vector_hierarchy_materializes_generated_hierarchy(
        self,
    ) -> None:
        """The declared vector route should compile hierarchy data."""

        contract = _prepare_declared_contract(
            _declared_vector_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]

        self.assertTrue(
            perturbation_data.manifest_summary["generated_vector_hierarchy"]
        )
        self.assertFalse(
            perturbation_data.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertIn("sigma_vector", perturbation_data.variables)
        self.assertIn("theta_gamma_v8", perturbation_data.variables)
        self.assertIn("e_gamma_v8", perturbation_data.variables)
        self.assertIn("b_gamma_v8", perturbation_data.variables)
        self.assertIn("nu_v5", perturbation_data.variables)
        self.assertIn("vector_temperature_source", perturbation_data.sources)
        self.assertIn("BB", perturbation_data.observables)
        self.assertEqual(
            perturbation_data.observables["polarization_b"].parity,
            "odd",
        )

    def test_declared_vector_polarization_terminals_absorb_free_streaming(
        self,
    ) -> None:
        """Vector E and B terminals must use the flat-space closure."""

        contract = _prepare_declared_contract(
            _declared_vector_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]
        context = {
            "acoustic_k": 0.2,
            "collision_rate": 0.0,
            "vector_eta_safe": 10.0,
            "e_gamma_v7": 0.2,
            "e_gamma_v8": 0.07,
            "b_gamma_v7": -0.1,
            "b_gamma_v8": 0.04,
        }
        e_terminal = evolution._evaluate_compiled_expression_noerr(
            perturbation_data.equations["evolve_e_gamma_v8"].compiled_rhs,
            context,
        )
        b_terminal = evolution._evaluate_compiled_expression_noerr(
            perturbation_data.equations["evolve_b_gamma_v8"].compiled_rhs,
            context,
        )

        self.assertAlmostEqual(
            float(e_terminal),
            8.0 / 7.0 * 0.2 * 0.2
            - 10.0 * 0.07 / 10.0
            + 2.0 / (8.0 * 9.0) * 0.2 * 0.04,
        )
        self.assertAlmostEqual(
            float(b_terminal),
            8.0 / 7.0 * 0.2 * -0.1
            - 10.0 * 0.04 / 10.0
            - 2.0 / (8.0 * 9.0) * 0.2 * 0.07,
        )

    def test_declared_vector_hierarchy_spectra_converge_below_one_percent(
        self,
    ) -> None:
        """Vector TT, EE, and BB meet the hierarchy refinement bound."""

        spectra_by_depth = []
        for depth in (8, 10):
            contract = _speedup_contract(_declared_vector_hierarchy_contract())
            controls = {
                "k_sample_count": 8,
                "eta_sample_count": 192,
                "photon_hierarchy_l_max": depth,
                "photon_polarization_hierarchy_l_max": depth,
                "neutrino_hierarchy_l_max": depth - 3,
            }
            contract["model_name"] = f"VectorHierarchyRefinement{depth}"
            contract["numerical"].update(controls)
            contract["perturbations"]["numerics"].update(controls)
            spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 60, 120), dtype=int),
                requested_spectra=("TT", "EE", "BB"),
            )
            spectra_by_depth.append(spectrum_data.spectra)

        for spectrum_name in ("TT", "EE", "BB"):
            metric = convergence.evaluate_control_refinement(
                spectra_by_depth[0][spectrum_name],
                spectra_by_depth[1][spectrum_name],
                name=f"vector {spectrum_name} hierarchy",
                tolerance=(convergence.FINAL_HIERARCHY_RELATIVE_TOLERANCE),
            )
            convergence.require_convergence(metric)

    def test_declared_vector_manifest_records_physical_sector_roles(
        self,
    ) -> None:
        """Vector manifests must identify physical roles and parity kernels."""

        vector = _prepare_declared_contract(
            _declared_vector_hierarchy_contract()
        )
        vector_summary = vector["perturbation_data"].manifest_summary[
            "vector_hierarchy"
        ]
        self.assertTrue(vector_summary["implemented"])
        self.assertEqual(vector_summary["sector"], "vector")
        self.assertEqual(vector_summary["metric_state"], "sigma_vector")
        self.assertEqual(vector_summary["closure"], "free_streaming_vector")
        self.assertEqual(vector_summary["parity"], ("even", "odd"))
        self.assertEqual(
            vector_summary["radial_kernels"],
            (
                "vector_temperature_1",
                "vector_temperature_2",
                "vector_e",
                "vector_b",
            ),
        )
        self.assertTrue(vector_summary["temperature_states"])
        self.assertTrue(vector_summary["polarization_e_states"])
        self.assertTrue(vector_summary["polarization_b_states"])
        self.assertTrue(vector_summary["neutrino_states"])

        scalar = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        scalar_summary = scalar["perturbation_data"].manifest_summary[
            "vector_hierarchy"
        ]
        self.assertFalse(scalar_summary["implemented"])
        self.assertEqual(scalar_summary["sector"], None)
        self.assertEqual(scalar_summary["radial_kernels"], ())
        scalar_manifest = scalar["perturbation_data"].manifest_summary
        self.assertFalse(
            any("vector" in name for name in scalar_manifest["variable_names"])
        )
        self.assertFalse(
            any("vector" in name for name in scalar_manifest["source_names"])
        )

    def test_declared_vector_hierarchy_transfer_payloads_are_finite(
        self,
    ) -> None:
        """Physical vector transfer outputs should stay finite."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_vector_hierarchy_contract())
        )
        ells = numpy.arange(20, 45, dtype=int)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
        )

        self.assertEqual(
            set(spectrum_data.transfer_components),
            {"temperature", "polarization_b", "polarization_e"},
        )
        self.assertEqual(set(spectrum_data.spectra), {"TT", "TE", "EE", "BB"})
        for array in (
            spectrum_data.transfer_components["temperature"],
            spectrum_data.transfer_components["polarization_e"],
            spectrum_data.transfer_components["polarization_b"],
            spectrum_data.spectra["TT"],
            spectrum_data.spectra["TE"],
            spectrum_data.spectra["EE"],
            spectrum_data.spectra["BB"],
        ):
            self.assertTrue(numpy.all(numpy.isfinite(array)))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        spectrum_data.transfer_components["polarization_b"]
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(spectrum_data.spectra["BB"]))),
            0.0,
        )

    def test_declared_vector_b_mode_survives_exact_lensing_remapper(
        self,
    ) -> None:
        """Exact lensing should preserve physical vector primordial BB."""

        scalar = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        vector = _prepare_declared_contract(
            _speedup_contract(_declared_vector_hierarchy_contract())
        )
        ells = numpy.arange(0, 121, dtype=int)
        scalar_spectra = _raw_declared_public_spectra(
            scalar,
            ells,
            spectra=("TT", "TE", "EE", "PP"),
        )
        vector_bb = _raw_declared_public_spectra(
            vector,
            ells,
            spectra=("BB",),
        )["BB"]

        lensed_without_vector = cmb_solver._assemble_exact_lensed_spectra(
            {
                **scalar_spectra,
                "BB": numpy.zeros_like(vector_bb, dtype=float),
            },
            ells,
        )["lensed_BB"]
        lensed_with_vector = cmb_solver._assemble_exact_lensed_spectra(
            {
                **scalar_spectra,
                "BB": vector_bb,
            },
            ells,
        )["lensed_BB"]

        self.assertTrue(numpy.all(numpy.isfinite(lensed_with_vector)))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(lensed_with_vector - lensed_without_vector)
                )
            ),
            0.0,
        )

    def test_declared_tensor_hierarchy_materializes_generated_hierarchy(
        self,
    ) -> None:
        """The declared tensor route should compile hierarchy data."""

        contract = _prepare_declared_contract(
            _declared_tensor_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]

        self.assertTrue(
            perturbation_data.manifest_summary["generated_tensor_hierarchy"]
        )
        self.assertFalse(
            perturbation_data.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertFalse(
            perturbation_data.manifest_summary["generated_vector_hierarchy"]
        )
        self.assertIn("h_tensor", perturbation_data.variables)
        self.assertIn("h_tensor_tau", perturbation_data.variables)
        self.assertNotIn("delta_gamma_tensor", perturbation_data.variables)
        self.assertNotIn("theta_gamma_tensor", perturbation_data.variables)
        self.assertNotIn("delta_nu_tensor", perturbation_data.variables)
        self.assertNotIn("theta_nu_tensor", perturbation_data.variables)
        self.assertNotIn("e_gamma_t0", perturbation_data.variables)
        self.assertNotIn("e_gamma_t1", perturbation_data.variables)
        self.assertIn("theta_gamma_t8", perturbation_data.variables)
        self.assertIn("e_gamma_t8", perturbation_data.variables)
        self.assertIn("b_gamma_t8", perturbation_data.variables)
        self.assertIn("nu_t5", perturbation_data.variables)
        self.assertIn("tensor_temperature_source", perturbation_data.sources)
        self.assertIn("BB", perturbation_data.observables)
        self.assertEqual(
            perturbation_data.observables["polarization_b"].parity,
            "odd",
        )

    def test_declared_tensor_initial_series_satisfies_declared_constraints(
        self,
    ) -> None:
        """Tensor initial data should satisfy the regular series."""

        contract = _prepare_declared_contract(
            _declared_tensor_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]
        self.assertEqual(
            set(perturbation_data.constraints),
            {
                "tensor_initial_collision_constraint",
                "tensor_initial_metric_constraint",
                "tensor_initial_neutrino_constraint",
            },
        )
        self.assertEqual(
            perturbation_data.derived[
                "tensor_initial_series_denominator"
            ].expression,
            "15.0 + 4.0 * tensor_free_streaming_fraction",
        )
        validator = getattr(
            evolution,
            "_validate_generated_tensor_initial_constraints",
            None,
        )
        self.assertIsNotNone(validator)

        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        numerics = cmb_background._resolve_custom_cmb_numerics(contract)
        model_parameters = {
            **contract["param_map"],
            **contract["model_parameters"],
        }
        k_value = 0.02
        eta_value = 1.0e-3
        base_context = evolution._build_declared_base_context(
            perturbation_data=perturbation_data,
            model_parameters=model_parameters,
            physical_params=physical_params,
            numerics=numerics,
            k_value=k_value,
            eta_value=eta_value,
            background_scalars={
                "a": 1.0e-8,
                "z": 1.0e8 - 1.0,
                "eta": eta_value,
                "Hconf": 1.0 / eta_value,
                "collision_rate": 1.0e8,
                "free_streaming": 0.0,
                "sound_horizon": eta_value / numpy.sqrt(3.0),
                "sound_speed_sq": 1.0 / 3.0,
            },
        )
        execution_plan = evolution._compile_declared_graph_execution_plan(
            perturbation_data
        )
        initial_state, _ = evolution._evaluate_declared_initial_state(
            perturbation_data=perturbation_data,
            execution_plan=execution_plan,
            base_context=base_context,
        )
        context = dict(base_context)
        for slot in execution_plan.runtime_spec.state_slots:
            value = float(initial_state[slot.index])
            if slot.order == 0:
                context[slot.variable] = value
            else:
                context[f"__d{slot.order}_{slot.variable}_{slot.wrt}"] = value
        context = evolution._resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
        )
        validator(
            perturbation_data=perturbation_data,
            context=context,
            k_value=k_value,
        )
        broken_context = dict(context)
        broken_context["tensor_initial_metric_residual"] = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "tensor_initial_metric_residual",
        ):
            validator(
                perturbation_data=perturbation_data,
                context=broken_context,
                k_value=k_value,
            )

    def test_declared_tensor_sources_match_independent_normalization(
        self,
    ) -> None:
        """Tensor source expressions should match the analytic convention."""

        contract = _prepare_declared_contract(
            _declared_tensor_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]
        context = {
            "tau": 0.4,
            "visibility": 0.03,
            "h_tensor_tau": -0.2,
            "pi_gamma_tensor": 0.05,
            "e_gamma_t2": -0.01,
            "b_gamma_t2": 0.02,
        }
        polarization_moment = (
            0.1 * context["pi_gamma_tensor"] + 0.6 * context["e_gamma_t2"]
        )
        context["tensor_polarization_moment"] = polarization_moment
        expected = {
            "tensor_temperature_source": (
                -numpy.exp(-context["tau"]) * context["h_tensor_tau"]
                + (15.0 / 8.0) * context["visibility"] * polarization_moment
            ),
            "tensor_polarization_e_source": (
                (15.0 / 2.0)
                * numpy.sqrt(3.0 / 8.0)
                * context["visibility"]
                * polarization_moment
            ),
            "tensor_polarization_b_source": (
                (15.0 / 2.0)
                * numpy.sqrt(3.0 / 8.0)
                * context["visibility"]
                * polarization_moment
            ),
        }
        for name, expected_value in expected.items():
            actual = evolution._evaluate_compiled_expression_noerr(
                perturbation_data.sources[name].compiled_expression,
                context,
            )
            self.assertAlmostEqual(float(actual), float(expected_value))

    def test_declared_tensor_terminal_closures_match_analytic_limits(
        self,
    ) -> None:
        """Tensor hierarchy terminals should use the flat-space limits."""

        contract = _prepare_declared_contract(
            _declared_tensor_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]
        context = {
            "acoustic_k": 0.2,
            "tensor_eta_safe": 10.0,
            "theta_gamma_t7": 0.3,
            "theta_gamma_t8": 0.1,
            "nu_t4": 0.25,
            "nu_t5": 0.08,
            "e_gamma_t7": 0.2,
            "e_gamma_t8": 0.07,
            "b_gamma_t7": -0.1,
            "b_gamma_t8": 0.04,
        }
        temperature = evolution._evaluate_compiled_expression_noerr(
            perturbation_data.equations["evolve_theta_gamma_t8"].compiled_rhs,
            context,
        )
        self.assertAlmostEqual(
            float(temperature),
            0.2 * 8.0 / 6.0 * 0.3 - 11.0 * 0.1 / 10.0,
        )
        neutrino = evolution._evaluate_compiled_expression_noerr(
            perturbation_data.equations["evolve_nu_t5"].compiled_rhs,
            context,
        )
        self.assertAlmostEqual(
            float(neutrino),
            0.2 * 5.0 / 3.0 * 0.25 - 8.0 * 0.08 / 10.0,
        )
        e_terminal = evolution._evaluate_compiled_expression_noerr(
            perturbation_data.equations["evolve_e_gamma_t8"].compiled_rhs,
            context,
        )
        b_terminal = evolution._evaluate_compiled_expression_noerr(
            perturbation_data.equations["evolve_b_gamma_t8"].compiled_rhs,
            context,
        )
        self.assertAlmostEqual(
            float(e_terminal),
            8.0 / 6.0 * 0.2 * 0.2
            - 11.0 * 0.07 / 10.0
            + 4.0 / (8.0 * 9.0) * 0.2 * 0.04,
        )
        self.assertAlmostEqual(
            float(b_terminal),
            8.0 / 6.0 * 0.2 * -0.1
            - 11.0 * 0.04 / 10.0
            - 4.0 / (8.0 * 9.0) * 0.2 * 0.07,
        )

    def test_declared_tensor_hierarchy_depth_converges_source_histories(
        self,
    ) -> None:
        """Deeper tensor hierarchies should change sources by under 1%."""

        contracts = []
        for photon_depth, polarization_depth, neutrino_depth in (
            (8, 8, 5),
            (10, 10, 7),
        ):
            raw_contract = _speedup_contract(
                _declared_tensor_hierarchy_contract()
            )
            controls = {
                "k_sample_count": 8,
                "photon_hierarchy_l_max": photon_depth,
                "photon_polarization_hierarchy_l_max": polarization_depth,
                "neutrino_hierarchy_l_max": neutrino_depth,
            }
            raw_contract["numerical"].update(controls)
            raw_contract["perturbations"]["numerics"].update(controls)
            contracts.append(_prepare_declared_contract(raw_contract))

        baseline = _capture_tensor_source_histories(contracts[0])
        refined = _capture_tensor_source_histories(contracts[1])
        self.assertEqual(set(baseline), set(refined))
        for key in sorted(baseline):
            scale = max(
                float(numpy.max(numpy.abs(refined[key]))),
                1.0e-30,
            )
            relative_error = float(
                numpy.max(numpy.abs(baseline[key] - refined[key])) / scale
            )
            self.assertLessEqual(
                relative_error,
                0.01,
                msg=f"{key} hierarchy-depth error: {relative_error}",
            )

    def test_declared_tensor_hierarchy_spectra_converge_below_one_percent(
        self,
    ) -> None:
        """Tensor TT, EE, and BB meet the hierarchy refinement bound."""

        spectra_by_depth = []
        for depth in (12, 14):
            contract = _speedup_contract(_declared_tensor_hierarchy_contract())
            controls = {
                "k_sample_count": 8,
                "eta_sample_count": 192,
                "photon_hierarchy_l_max": depth,
                "photon_polarization_hierarchy_l_max": depth,
                "neutrino_hierarchy_l_max": depth - 3,
            }
            contract["model_name"] = f"TensorHierarchyRefinement{depth}"
            contract["numerical"].update(controls)
            contract["perturbations"]["numerics"].update(controls)
            spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 60, 120), dtype=int),
                requested_spectra=("TT", "EE", "BB"),
            )
            spectra_by_depth.append(spectrum_data.spectra)

        for spectrum_name in ("TT", "EE", "BB"):
            metric = convergence.evaluate_control_refinement(
                spectra_by_depth[0][spectrum_name],
                spectra_by_depth[1][spectrum_name],
                name=f"tensor {spectrum_name} hierarchy",
                tolerance=(convergence.FINAL_HIERARCHY_RELATIVE_TOLERANCE),
            )
            convergence.require_convergence(metric)

    def test_declared_tensor_hierarchy_transfer_payloads_are_finite(
        self,
    ) -> None:
        """Physical tensor transfer outputs should stay finite."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_tensor_hierarchy_contract())
        )
        ells = numpy.asarray((20, 60, 120), dtype=int)
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
        )

        self.assertEqual(
            set(spectrum_data.transfer_components),
            {"temperature", "polarization_b", "polarization_e"},
        )
        self.assertEqual(set(spectrum_data.spectra), {"TT", "TE", "EE", "BB"})
        for array in (
            spectrum_data.transfer_components["temperature"],
            spectrum_data.transfer_components["polarization_e"],
            spectrum_data.transfer_components["polarization_b"],
            spectrum_data.spectra["TT"],
            spectrum_data.spectra["TE"],
            spectrum_data.spectra["EE"],
            spectrum_data.spectra["BB"],
        ):
            self.assertTrue(numpy.all(numpy.isfinite(array)))
        self.assertGreater(
            float(numpy.max(numpy.abs(spectrum_data.spectra["TT"]))),
            0.0,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(spectrum_data.spectra["BB"]))),
            0.0,
        )

    def test_declared_tensor_hierarchy_amplitude_response_scales_linearly(
        self,
    ) -> None:
        """Tensor primordial amplitude should scale tensor spectra linearly."""

        baseline = _prepare_declared_contract(
            _speedup_contract(_declared_tensor_hierarchy_contract())
        )
        changed_contract = _speedup_contract(
            _declared_tensor_hierarchy_contract()
        )
        changed_contract["param_map"]["r"] = 0.15
        changed = _prepare_declared_contract(changed_contract)
        ells = numpy.arange(20, 36, dtype=int)
        baseline_bb = _raw_declared_public_spectra(
            baseline,
            ells,
            spectra=("BB",),
        )["BB"]
        changed_bb = _raw_declared_public_spectra(
            changed,
            ells,
            spectra=("BB",),
        )["BB"]
        numpy.testing.assert_allclose(
            changed_bb / baseline_bb,
            numpy.full_like(baseline_bb, 1.5),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_declared_tensor_hierarchy_tilt_changes_bb_shape(
        self,
    ) -> None:
        """Tensor tilt should reshape the declared B-mode spectrum."""

        red_contract = _speedup_contract(_declared_tensor_hierarchy_contract())
        blue_contract = _speedup_contract(
            _declared_tensor_hierarchy_contract()
        )
        red_contract["param_map"]["nt"] = -0.6
        blue_contract["param_map"]["nt"] = 0.6
        red = _prepare_declared_contract(red_contract)
        blue = _prepare_declared_contract(blue_contract)
        ells = numpy.asarray((20, 30, 50, 80, 120), dtype=int)
        red_bb = _raw_declared_public_spectra(
            red,
            ells,
            spectra=("BB",),
        )["BB"]
        blue_bb = _raw_declared_public_spectra(
            blue,
            ells,
            spectra=("BB",),
        )["BB"]
        red_shape = float(red_bb[-1] / red_bb[0])
        blue_shape = float(blue_bb[-1] / blue_bb[0])

        self.assertGreater(blue_shape, red_shape)

    def test_declared_tensor_hierarchy_neutrino_stress_changes_bb(
        self,
    ) -> None:
        """Tensor neutrino stress should alter the declared B-mode output."""

        with_neutrinos = _prepare_declared_contract(
            _speedup_contract(_declared_tensor_hierarchy_contract())
        )
        without_neutrinos_contract = _speedup_contract(
            _declared_tensor_hierarchy_contract()
        )
        without_neutrinos_contract["param_map"]["Neff"] = 0.0
        without_neutrinos = _prepare_declared_contract(
            without_neutrinos_contract
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        baseline_bb = _raw_declared_public_spectra(
            with_neutrinos,
            ells,
            spectra=("BB",),
        )["BB"]
        changed_bb = _raw_declared_public_spectra(
            without_neutrinos,
            ells,
            spectra=("BB",),
        )["BB"]

        self.assertTrue(numpy.all(numpy.isfinite(baseline_bb)))
        self.assertTrue(numpy.all(numpy.isfinite(changed_bb)))
        self.assertGreater(
            float(numpy.max(numpy.abs(changed_bb - baseline_bb))),
            1.0e-18,
        )

    def test_declared_tensor_component_aliases_match_total_spectra(
        self,
    ) -> None:
        """Tensor and total aliases should resolve to the same spectra."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_tensor_hierarchy_contract())
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("TT", "tensor_TT", "total_TT", "BB", "tensor_BB"),
        )

        numpy.testing.assert_allclose(
            numpy.asarray(spectra["TT"], dtype=float),
            numpy.asarray(spectra["tensor_TT"], dtype=float),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["TT"], dtype=float),
            numpy.asarray(spectra["total_TT"], dtype=float),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["BB"], dtype=float),
            numpy.asarray(spectra["tensor_BB"], dtype=float),
        )

    def test_declared_tensor_b_mode_survives_exact_lensing_remapper(
        self,
    ) -> None:
        """Exact lensing should preserve physical tensor primordial BB."""

        scalar = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        tensor = _prepare_declared_contract(
            _speedup_contract(_declared_tensor_hierarchy_contract())
        )
        ells = numpy.arange(0, 121, dtype=int)
        scalar_spectra = _raw_declared_public_spectra(
            scalar,
            ells,
            spectra=("TT", "TE", "EE", "PP"),
        )
        tensor_bb = _raw_declared_public_spectra(
            tensor,
            ells,
            spectra=("BB",),
        )["BB"]

        lensed_without_tensor = cmb_solver._assemble_exact_lensed_spectra(
            {
                **scalar_spectra,
                "BB": numpy.zeros_like(tensor_bb, dtype=float),
            },
            ells,
        )["lensed_BB"]
        lensed_with_tensor = cmb_solver._assemble_exact_lensed_spectra(
            {
                **scalar_spectra,
                "BB": tensor_bb,
            },
            ells,
        )["lensed_BB"]

        self.assertTrue(numpy.all(numpy.isfinite(lensed_with_tensor)))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(lensed_with_tensor - lensed_without_tensor)
                )
            ),
            0.0,
        )

    def test_declared_scalar_hierarchy_materializes_massive_neutrinos(
        self,
    ) -> None:
        """The declared scalar route should expose aggregate aliases only."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        perturbation_data = contract["perturbation_data"]

        self.assertNotIn("delta_nu_massive", perturbation_data.variables)
        self.assertNotIn("theta_nu_massive", perturbation_data.variables)
        self.assertNotIn("sigma_nu_massive", perturbation_data.variables)
        self.assertNotIn("nu_massive_l5", perturbation_data.variables)
        self.assertIn("delta_nu_massive", perturbation_data.derived)
        self.assertIn("theta_nu_massive", perturbation_data.derived)
        self.assertIn("sigma_nu_massive", perturbation_data.derived)
        self.assertIn("nu_massive_l5", perturbation_data.derived)
        self.assertEqual(
            perturbation_data.hierarchy_families[
                "massive_neutrino"
            ].momentum_grid,
            "massive_neutrino_default",
        )

    def test_declared_scalar_hierarchy_materializes_massive_neutrino_q_bins(
        self,
    ) -> None:
        """Massive-neutrino q bins should materialize resolved states."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        perturbation_data = contract["perturbation_data"]

        self.assertIn("delta_nu_massive_q0", perturbation_data.variables)
        self.assertIn("theta_nu_massive_q0", perturbation_data.variables)
        self.assertIn("sigma_nu_massive_q0", perturbation_data.variables)
        self.assertIn("nu_massive_q0_l5", perturbation_data.variables)

    def test_declared_scalar_hierarchy_materializer_uses_q_weights(
        self,
    ) -> None:
        """Generated q bins should drive the only massive-neutrino states."""

        contract = _prepare_declared_contract(
            _strip_declared_runtime_sections(
                _declared_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                )
            )
        )
        perturbation_data = contract["perturbation_data"]

        self.assertTrue(
            perturbation_data.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertIn(
            "massive_neutrino_q0_density_weight * delta_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_density_q0"
            ].expression,
        )
        self.assertIn(
            "massive_neutrino_q0_pressure_weight * delta_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_pressure_q0"
            ].expression,
        )
        self.assertIn(
            "acoustic_k * massive_neutrino_q0_momentum_weight * "
            "theta_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_momentum_q0"
            ].expression,
        )
        self.assertIn(
            "massive_neutrino_q0_shear_weight * sigma_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_shear_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_weight",
            perturbation_data.derived[
                "massive_neutrino_metric_density_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_velocity_ratio",
            perturbation_data.derived[
                "massive_neutrino_metric_momentum_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_pressure_ratio",
            perturbation_data.derived[
                "massive_neutrino_metric_shear_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_delta_nu_massive_q0"].rhs,
        )
        self.assertIn(
            "massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_theta_nu_massive_q0"].rhs,
        )
        self.assertIn(
            "massive_neutrino_q0_streaming_speed * "
            "massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_theta_nu_massive_q0"].rhs,
        )
        self.assertNotIn(
            "1.0 / massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_theta_nu_massive_q0"].rhs,
        )
        self.assertNotIn(
            "evolve_delta_nu_massive",
            perturbation_data.equations,
        )
        self.assertNotIn(
            "evolve_theta_nu_massive",
            perturbation_data.equations,
        )
        self.assertNotIn(
            "evolve_sigma_nu_massive",
            perturbation_data.equations,
        )
        self.assertNotIn(
            "evolve_nu_massive_l3",
            perturbation_data.equations,
        )
        self.assertIn(
            "sqrt((acoustic_k * eta) * (acoustic_k * eta) + 6 * 6)",
            perturbation_data.equations["evolve_nu_l5"].rhs,
        )
        self.assertIn(
            "massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_nu_massive_q0_l5"].rhs,
        )
        self.assertIn(
            "sqrt((acoustic_k * eta) * (acoustic_k * eta) + 6 * 6)",
            perturbation_data.equations["evolve_nu_massive_q0_l5"].rhs,
        )
        self.assertIn(
            "massive_neutrino_momentum_source",
            perturbation_data.derived["total_momentum_source"].expression,
        )
        self.assertEqual(
            perturbation_data.derived["baryon_thomson_drag"].expression,
            "- 3.0 * acoustic_k * photon_baryon_momentum_ratio * "
            "thomson_drag",
        )
        self.assertIn(
            "thomson_drag_balance",
            perturbation_data.conservation_rules,
        )
        self.assertEqual(
            perturbation_data.collision_operators["thomson_drag"].counterpart,
            "baryon_thomson_drag",
        )
        self.assertEqual(
            perturbation_data.conservation_rules[
                "thomson_drag_balance"
            ].expression,
            "3.0 * acoustic_k * photon_baryon_momentum_ratio * "
            "thomson_drag + "
            "baryon_thomson_drag",
        )
        self.assertIn("einstein_energy_residual", perturbation_data.derived)
        self.assertIn("einstein_momentum_residual", perturbation_data.derived)
        self.assertIn("einstein_shear_residual", perturbation_data.derived)
        self.assertIn(
            "Phi_tau",
            perturbation_data.equations["evolve_delta_b"].rhs,
        )
        self.assertEqual(
            perturbation_data.derived["Psi_tau"].variable,
            "Psi",
        )
        self.assertIsNone(perturbation_data.derived["Psi_tau"].expression)
        self.assertEqual(
            perturbation_data.equations["evolve_Phi"].rhs,
            "Phi_tau",
        )
        self.assertEqual(
            perturbation_data.initial_conditions["Phi_seed"].expression,
            "(scalar_potential_seed) + metric_shear_correction",
        )
        self.assertEqual(
            perturbation_data.derived["metric_constraint_scale"].expression,
            "acoustic_k_sq",
        )
        self.assertEqual(
            perturbation_data.closures["psi_closure"].expression,
            "Phi - metric_shear_correction",
        )
        self.assertIn(
            "photon_velocity_divergence",
            perturbation_data.derived["total_momentum_source"].expression,
        )
        self.assertIn(
            "massive_neutrino_density_source",
            perturbation_data.derived["total_density_source"].expression,
        )
        self.assertIn(
            "neutrino_temperature_eV",
            perturbation_data.derived[
                "massive_neutrino_q0_streaming_speed"
            ].expression,
        )
        self.assertIn(
            "(4.0 / 3.0) * a * a",
            perturbation_data.derived[
                "massive_neutrino_momentum_source"
            ].expression,
        )
        self.assertIn(
            "massive_neutrino_shear_source",
            perturbation_data.derived["total_shear_source"].expression,
        )
        self.assertIn(
            "Omega_gamma0 * observable_theta_gamma2",
            perturbation_data.derived["total_shear_source"].expression,
        )
        self.assertIn(
            "baryon_thomson_drag",
            perturbation_data.equations["evolve_theta_b"].rhs,
        )
        self.assertEqual(
            perturbation_data.sources["temperature_doppler"].expression,
            "visibility * observable_theta_b / acoustic_k",
        )
        self.assertIn(
            "theta_gamma3",
            perturbation_data.equations["evolve_theta_gamma2"].rhs,
        )
        self.assertNotIn(
            "collision_rate",
            perturbation_data.equations["evolve_theta_gamma2"].rhs,
        )
        self.assertIn(
            "e_gamma3",
            perturbation_data.equations["evolve_e_gamma2"].rhs,
        )
        self.assertNotIn(
            "collision_rate",
            perturbation_data.equations["evolve_theta_gamma3"].rhs,
        )
        self.assertNotIn(
            "collision_rate",
            perturbation_data.equations["evolve_e_gamma2"].rhs,
        )
        self.assertNotIn(
            "collision_rate",
            perturbation_data.equations["evolve_e_gamma3"].rhs,
        )
        self.assertEqual(
            perturbation_data.equations["evolve_e_gamma0"].rhs,
            "0.0",
        )
        self.assertNotIn(
            "tight_coupling_drag",
            perturbation_data.equations["evolve_theta_gamma3"].rhs,
        )
        self.assertNotIn(
            "tight_coupling_drag",
            perturbation_data.equations["evolve_e_gamma3"].rhs,
        )
        self.assertEqual(
            perturbation_data.equations["evolve_theta_gamma8"].rhs,
            "1 * acoustic_k * theta_gamma7 - acoustic_k * 9 * theta_gamma8 / "
            "sqrt((acoustic_k * eta) * (acoustic_k * eta) + 9 * 9)",
        )
        self.assertEqual(
            perturbation_data.equations["evolve_e_gamma8"].rhs,
            "1.333333333333333 * acoustic_k * e_gamma7 - "
            "acoustic_k * 11 * e_gamma8 / "
            "sqrt((acoustic_k * eta) * (acoustic_k * eta) + 11 * 11)",
        )
        self.assertEqual(
            perturbation_data.derived["delta_nu_massive"].expression,
            "massive_neutrino_metric_density",
        )
        self.assertEqual(
            perturbation_data.derived["theta_nu_massive"].expression,
            "massive_neutrino_metric_momentum",
        )
        self.assertEqual(
            perturbation_data.derived["sigma_nu_massive"].expression,
            "massive_neutrino_metric_shear",
        )
        self.assertIn(
            "massive_neutrino_metric_l3_q0",
            perturbation_data.derived["nu_massive_l3"].expression,
        )
        self.assertIn(
            "massive_neutrino_q0_shear_weight * nu_massive_q0_l3",
            perturbation_data.derived[
                "massive_neutrino_metric_l3_q0"
            ].expression,
        )
        self.assertIn(
            "acoustic_k * scalar_initial_conformal_time / 6.0",
            perturbation_data.initial_conditions[
                "theta_gamma1_seed"
            ].expression,
        )
        self.assertIn(
            "acoustic_k_sq * scalar_initial_conformal_time / 2.0",
            perturbation_data.initial_conditions["theta_b_seed"].expression,
        )
        self.assertIn(
            "distribution_log_derivative",
            perturbation_data.initial_conditions[
                "delta_nu_massive_q0_seed"
            ].expression,
        )
        self.assertIn(
            "acoustic_k * scalar_initial_conformal_time / 8.0",
            perturbation_data.initial_conditions[
                "theta_nu_massive_q0_seed"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_velocity_ratio",
            perturbation_data.initial_conditions[
                "theta_nu_massive_q0_seed"
            ].expression,
        )
        self.assertNotIn(
            "delta_nu_massive_seed",
            perturbation_data.initial_conditions,
        )
        self.assertNotIn(
            "theta_nu_massive_seed",
            perturbation_data.initial_conditions,
        )
        self.assertNotIn(
            "sigma_nu_massive_seed",
            perturbation_data.initial_conditions,
        )
        self.assertNotIn(
            "nu_massive_l3_seed",
            perturbation_data.initial_conditions,
        )

    def test_declared_scalar_hierarchy_q_integrated_aliases_match_bins(
        self,
    ) -> None:
        """Resolved q-bin aliases should stay locked to physical moments."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        q_count = int(
            contract["numerical"]["momentum_grids"][
                "massive_neutrino_default"
            ]["count"]
        )
        state_updates: dict[str, float] = {}
        for q_index in range(q_count):
            state_updates[f"delta_nu_massive_q{q_index}"] = 1.0
            state_updates[f"theta_nu_massive_q{q_index}"] = 2.0
            state_updates[f"sigma_nu_massive_q{q_index}"] = 3.0
            for moment in range(3, 6):
                state_updates[f"nu_massive_q{q_index}_l{moment}"] = (
                    float(moment) + 1.0
                )

        context = _resolved_declared_scalar_context(
            contract,
            state_updates=state_updates,
        )

        self.assertAlmostEqual(
            float(context["massive_neutrino_metric_density"]), 1.0
        )
        self.assertAlmostEqual(
            float(context["massive_neutrino_metric_pressure"]), 1.0
        )
        self.assertAlmostEqual(
            float(context["massive_neutrino_metric_momentum"]),
            0.2,
        )
        self.assertAlmostEqual(
            float(context["massive_neutrino_metric_shear"]), 3.0
        )
        self.assertAlmostEqual(float(context["delta_nu_massive"]), 1.0)
        self.assertAlmostEqual(float(context["theta_nu_massive"]), 0.2)
        self.assertAlmostEqual(float(context["sigma_nu_massive"]), 3.0)
        self.assertAlmostEqual(float(context["nu_massive_l3"]), 4.0)
        self.assertAlmostEqual(float(context["nu_massive_l4"]), 5.0)
        self.assertAlmostEqual(float(context["nu_massive_l5"]), 6.0)

    def test_declared_scalar_hierarchy_metric_sources_respond_to_inputs(
        self,
    ) -> None:
        """Matter and radiation inputs should alter Einstein source weights."""

        baseline = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        heavier_baryons = _declared_scalar_hierarchy_contract()
        heavier_baryons["param_map"]["ombh2"] = 0.03
        heavier_baryons = _prepare_declared_contract(heavier_baryons)
        hotter_radiation = _declared_scalar_hierarchy_contract()
        hotter_radiation["model_parameters"]["Tcmb_K"] = 3.0
        hotter_radiation = _prepare_declared_contract(hotter_radiation)

        baseline_context = _resolved_declared_scalar_context(baseline)
        matter_context = _resolved_declared_scalar_context(heavier_baryons)
        radiation_context = _resolved_declared_scalar_context(hotter_radiation)

        self.assertGreater(
            float(matter_context["matter_density_source"]),
            float(baseline_context["matter_density_source"]),
        )
        self.assertGreater(
            float(radiation_context["radiation_density_source"]),
            float(baseline_context["radiation_density_source"]),
        )
        self.assertGreater(
            float(matter_context["total_momentum_source"]),
            float(baseline_context["total_momentum_source"]),
        )

    def test_declared_scalar_hierarchy_rejects_broken_einstein_residuals(
        self,
    ) -> None:
        """Runtime Einstein residual diagnostics should fail bad contracts."""

        contract = _speedup_contract(_declared_scalar_hierarchy_contract())
        contract["perturbations"]["conservation_rules"] = {
            "broken_einstein_probe": {
                "kind": "absolute_max",
                "expression": "einstein_energy_residual + 0.1",
                "tolerance": 5.0e-2,
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "conservation rule exceeded tolerance",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_declared_scalar_hierarchy_records_constraint_anchor_diagnostics(
        self,
    ) -> None:
        """Accepted scalar histories should expose full-history diagnostics."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_scalar_hierarchy_contract())
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
        )

        diagnostics = spectrum_data.runtime_envelope[
            "scalar_constraint_diagnostics"
        ]
        self.assertTrue(
            spectrum_data.runtime_envelope["generated_scalar_hierarchy"]
        )
        derivative_validation = spectrum_data.runtime_envelope[
            "metric_history_derivative_validation"
        ]
        self.assertEqual(
            derivative_validation["required"],
            ("Phi_tau", "Psi_tau", "Phi_history_tau"),
        )
        self.assertGreater(int(derivative_validation["mode_count"]), 0)
        self.assertTrue(derivative_validation["finite"])
        self.assertEqual(
            set(diagnostics),
            {
                "einstein_energy_residual",
                "einstein_momentum_residual",
                "einstein_shear_residual",
            },
        )
        for metrics in diagnostics.values():
            self.assertGreater(int(metrics["mode_count"]), 0)
            self.assertGreater(int(metrics["sample_count"]), 0)
            self.assertEqual(
                set(metrics["anchors"]),
                {"early", "recombination", "late"},
            )
            self.assertEqual(
                set(metrics["normalized_anchors"]),
                {"early", "recombination", "late"},
            )
            self.assertGreaterEqual(float(metrics["maximum_eta"]), 0.0)
            self.assertIn(
                metrics["physical_regime"],
                {"radiation", "recombination", "matter", "late"},
            )
            self.assertGreater(float(metrics["normalization_scale"]), 0.0)
            self.assertTrue(metrics["normalization_terms"])
            self.assertEqual(
                metrics["normalization_source"],
                "sum_abs_declared_einstein_terms",
            )
            self.assertIn(
                metrics["tolerance_kind"],
                {"absolute", "normalized"},
            )
            self.assertEqual(metrics["resolution_status"], "reference")
            self.assertIn(
                metrics["physical_judgement"],
                {"evaluated", "deferred"},
            )

    def test_generated_scalar_initial_preflight_covers_sorted_k_grid(
        self,
    ) -> None:
        """Every requested scalar mode should pass one pre-ODE preflight."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_scalar_hierarchy_contract())
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
        )

        preflight = spectrum_data.runtime_envelope[
            "scalar_initial_constraint_preflight"
        ]
        k_values = tuple(float(value) for value in spectrum_data.k_grid)
        self.assertTrue(preflight["performed"])
        self.assertEqual(preflight["failure_order"], "ascending_k")
        self.assertEqual(preflight["k_values"], tuple(sorted(k_values)))
        self.assertEqual(preflight["mode_count"], len(set(k_values)))
        self.assertEqual(
            set(preflight["residuals"]),
            {
                "einstein_energy_residual",
                "einstein_momentum_residual",
                "einstein_shear_residual",
            },
        )
        for metrics in preflight["residuals"].values():
            self.assertLessEqual(
                float(metrics["maximum_normalized"]),
                float(metrics["tolerance"]),
            )
            self.assertGreater(float(metrics["normalization_scale"]), 0.0)
            self.assertTrue(metrics["normalization_terms"])
            self.assertEqual(
                metrics["normalization_source"],
                "sum_abs_declared_einstein_terms",
            )

    def test_generated_scalar_initial_metric_seed_is_preserved(
        self,
    ) -> None:
        """Generated scalar evolution must retain its declared metric seed."""

        contract = _prepare_declared_contract(
            _speedup_contract(_declared_scalar_hierarchy_contract())
        )
        spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 24, dtype=int),
            requested_spectra=("TT",),
        )

        state_history = spectrum_data.runtime_envelope[
            "state_history_max_abs_by_k"
        ]
        self.assertTrue(state_history)
        self.assertGreater(
            max(float(values["Phi"]) for values in state_history.values()),
            0.1,
        )

    def test_projection_k_grid_rejects_undeclared_high_k_request(
        self,
    ) -> None:
        """Projection preflight must not extend beyond declared k limits."""

        with self.assertRaisesRegex(
            ValueError,
            "exceeds declared numerical limits",
        ):
            cmb_projection._build_projection_k_grid(
                ell_arr=numpy.asarray((2, 2_000), dtype=int),
                background=SimpleNamespace(
                    eta0=14_000.0,
                    eta_rec=280.0,
                    sound_horizon_mpc=145.0,
                ),
                numerics=SimpleNamespace(
                    ell_min=2,
                    k_min=1.0e-4,
                    k_max=1.0e-3,
                    k_sample_count=8,
                ),
                perturbation_data=SimpleNamespace(
                    accuracy_controls={},
                    manifest_summary={},
                ),
            )

    def test_scalar_constraint_acceptance_is_resolution_aware(
        self,
    ) -> None:
        """Reference tolerances apply only to sufficiently resolved grids."""

        context = {
            "einstein_energy_residual": numpy.full(8, 5.0e-3),
        }
        controls = {
            "scalar_constraint_reference_eta_samples": 16,
            "scalar_constraint_tolerances": {
                "einstein_energy_residual": 1.0e-3,
            },
        }
        diagnostics = cmb_projection._validate_scalar_constraint_histories(
            perturbation_data=SimpleNamespace(conservation_rules={}),
            context=context,
            eta_grid=numpy.arange(8, dtype=float),
            accuracy_controls=controls,
            k_value=0.1,
        )

        self.assertFalse(diagnostics["einstein_energy_residual"]["enforced"])
        self.assertFalse(
            diagnostics["einstein_energy_residual"]["reference_resolution_met"]
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["resolution_status"],
            "under_resolved",
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["physical_judgement"],
            "deferred",
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["normalization_source"],
            "residual_magnitude_fallback",
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["tolerance_kind"],
            "normalized",
        )
        self.assertGreater(
            float(diagnostics["einstein_energy_residual"]["maximum_absolute"]),
            float(diagnostics["einstein_energy_residual"]["tolerance"]),
        )

        controls["scalar_constraint_reference_eta_samples"] = 8
        with self.assertRaisesRegex(
            ValueError,
            "Scalar Einstein constraint exceeded tolerance",
        ):
            cmb_projection._validate_scalar_constraint_histories(
                perturbation_data=SimpleNamespace(conservation_rules={}),
                context=context,
                eta_grid=numpy.arange(8, dtype=float),
                accuracy_controls=controls,
                k_value=0.1,
            )

    def test_declared_power_spectrum_scale_factor_is_physical(
        self,
    ) -> None:
        """Declared spectrum scaling should stay on physical units only."""

        ell_factor = numpy.asarray((2.0, 6.0, 12.0), dtype=numpy.longdouble)
        t_cmb_muK = numpy.longdouble("2.7255e6")

        tt_scale = cmb_solver._power_spectrum_scale_factor(
            None,
            "TT",
            ell_factor=ell_factor,
            t_cmb_muK=float(t_cmb_muK),
            lensing_mode=False,
        )
        pp_scale = cmb_solver._power_spectrum_scale_factor(
            None,
            "PP",
            ell_factor=ell_factor,
            t_cmb_muK=float(t_cmb_muK),
            lensing_mode=True,
        )

        numpy.testing.assert_allclose(
            numpy.asarray(tt_scale, dtype=numpy.longdouble),
            ell_factor * t_cmb_muK * t_cmb_muK,
        )
        numpy.testing.assert_allclose(
            numpy.asarray(pp_scale, dtype=numpy.longdouble),
            2.0 * numpy.longdouble(numpy.pi) * ell_factor * ell_factor,
        )

    def test_declared_power_spectrum_uses_log_k_simpson_quadrature(
        self,
    ) -> None:
        """Primordial transfer products should use the declared log-k rule."""

        log_k = numpy.asarray((-2.0, -0.5, 1.0), dtype=numpy.longdouble)
        primordial = numpy.square(log_k)
        unit_transfer = numpy.ones((1, log_k.size), dtype=numpy.longdouble)
        actual = cmb_projection._integrate_power_spectrum(
            primordial,
            log_k,
            unit_transfer,
            unit_transfer,
        )
        expected = numpy.asarray(
            (4.0 * numpy.longdouble(numpy.pi)) * 3.0,
            dtype=numpy.longdouble,
        )

        numpy.testing.assert_allclose(actual, expected)

    def test_declared_scalar_spectrum_aliases_round_trip(self) -> None:
        """Declared phiphi, Tphi, and Ephi aliases should round-trip."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        ells = numpy.asarray((20, 40, 60, 90, 120), dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("PP", "phiphi", "TP", "Tphi", "EP", "Ephi"),
        )

        self.assertEqual(
            set(spectra),
            {"PP", "phiphi", "TP", "Tphi", "EP", "Ephi"},
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["PP"], dtype=float),
            numpy.asarray(spectra["phiphi"], dtype=float),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["TP"], dtype=float),
            numpy.asarray(spectra["Tphi"], dtype=float),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["EP"], dtype=float),
            numpy.asarray(spectra["Ephi"], dtype=float),
        )

    def test_declared_scalar_hierarchy_synchronous_matches_newtonian(
        self,
    ) -> None:
        """Generated synchronous and Newtonian routes should agree."""

        newtonian = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        synchronous = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(gauge="synchronous")
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        spectra = (
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
        baseline = cmb.compute_cmb_spectrum_from_contract(
            newtonian,
            ells,
            spectra=spectra,
        )
        comparison = cmb.compute_cmb_spectrum_from_contract(
            synchronous,
            ells,
            spectra=spectra,
        )

        for spectrum_name in spectra:
            numpy.testing.assert_allclose(
                numpy.asarray(comparison[spectrum_name], dtype=float),
                numpy.asarray(baseline[spectrum_name], dtype=float),
                rtol=1.0e-10,
                atol=1.0e-10,
            )

    def test_declared_scalar_hierarchy_gauge_invariant_matches_newtonian(
        self,
    ) -> None:
        """Gauge-invariant declared runs should match Newtonian observables."""

        newtonian = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        gauge_invariant = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(gauge="gauge_invariant")
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        spectra = (
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
        baseline = cmb.compute_cmb_spectrum_from_contract(
            newtonian,
            ells,
            spectra=spectra,
        )
        comparison = cmb.compute_cmb_spectrum_from_contract(
            gauge_invariant,
            ells,
            spectra=spectra,
        )

        for spectrum_name in spectra:
            numpy.testing.assert_allclose(
                numpy.asarray(comparison[spectrum_name], dtype=float),
                numpy.asarray(baseline[spectrum_name], dtype=float),
                rtol=1.0e-10,
                atol=1.0e-10,
            )

    def test_declared_scalar_gauge_routes_preserve_history_and_manifest(
        self,
    ) -> None:
        """Gauge routes must expose explicit bridges and shared observables."""

        threshold = float(
            SLICE_NINE_ACCEPTANCE_THRESHOLDS["gauge_equivalent_fractional"]
        )
        ells = numpy.asarray((20, 30, 40), dtype=int)
        spectra = SLICE_NINE_ACCEPTANCE_SPECTRA
        routes = (
            "conformal_newtonian",
            "synchronous",
            "gauge_invariant",
        )
        prepared_routes = {
            gauge: _prepare_declared_contract(
                _speedup_contract(
                    _declared_scalar_hierarchy_contract(gauge=gauge)
                )
            )
            for gauge in routes
        }
        manifests = {
            gauge: prepared["perturbation_data"].manifest_summary
            for gauge, prepared in prepared_routes.items()
        }
        self.assertEqual(
            manifests["conformal_newtonian"]["gauge_equivalence"][
                "transformation"
            ],
            "observable_identity",
        )
        self.assertEqual(
            manifests["synchronous"]["gauge_equivalence"]["transformation"],
            "scalar_first_order",
        )
        self.assertEqual(
            manifests["gauge_invariant"]["gauge_equivalence"][
                "transformation"
            ],
            "bardeen_invariant",
        )
        for gauge in routes:
            equivalence = manifests[gauge]["gauge_equivalence"]
            self.assertEqual(equivalence["observable_basis"], "newtonian")
            self.assertTrue(equivalence["explicit"])
            self.assertTrue(
                set(equivalence["metric_state_names"]).issubset(
                    set(manifests[gauge]["variable_names"])
                )
            )
            self.assertTrue(
                set(equivalence["derived_transform_names"]).issubset(
                    set(manifests[gauge]["derived_names"])
                )
            )
        self.assertNotEqual(
            manifests["conformal_newtonian"]["variable_names"],
            manifests["synchronous"]["variable_names"],
        )
        self.assertNotEqual(
            manifests["synchronous"]["variable_names"],
            manifests["gauge_invariant"]["variable_names"],
        )

        captured_routes = {
            gauge: _capture_visible_scalar_monopole_history(
                prepared,
                ells=ells,
                spectra=spectra,
            )
            for gauge, prepared in prepared_routes.items()
        }
        histories = {
            gauge: captured[:2] for gauge, captured in captured_routes.items()
        }
        baseline_eta, baseline_monopole = histories[routes[0]]
        for gauge in routes[1:]:
            eta_grid, monopole = histories[gauge]
            numpy.testing.assert_array_equal(eta_grid, baseline_eta)
            self.assertLess(
                _max_relative_delta(baseline_monopole, monopole),
                threshold,
            )

        spectra_by_gauge = {
            gauge: captured[2] for gauge, captured in captured_routes.items()
        }
        baseline = spectra_by_gauge[routes[0]]
        for gauge in routes[1:]:
            metrics = _slice_nine_spectrum_metrics(
                spectra_by_gauge[gauge],
                baseline,
                spectra=spectra,
            )
            for spectrum_name in spectra:
                metric = metrics[spectrum_name]
                error = (
                    metric["median_fractional"]
                    if "median_fractional" in metric
                    else metric["normalized_rms"]
                )
                self.assertLess(error, threshold, spectrum_name)

    def test_declared_scalar_hierarchy_sync_transform_matches_observables(
        self,
    ) -> None:
        """Synchronous histories should reconstruct the observable basis."""

        synchronous = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(gauge="synchronous")
        )
        context = _resolved_declared_scalar_context(synchronous)

        self.assertAlmostEqual(
            float(context["Phi_from_synchronous"]),
            float(context["Phi"]),
        )
        self.assertAlmostEqual(
            float(context["Psi_from_synchronous"]),
            float(context["Psi"]),
        )
        self.assertNotEqual(
            float(context["eta_sync_metric"]),
            float(context["Phi"]),
        )

    def test_declared_scalar_hierarchy_standard_modes_generate_distinct(
        self,
    ) -> None:
        """Generated standard initial-condition modes should change TT."""

        adiabatic = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(
                initial_mode="adiabatic_scalar"
            )
        )
        cdm_mode = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(
                initial_mode="cdm_isocurvature"
            )
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        adiabatic_tt = numpy.asarray(
            cmb_projection._compute_custom_cmb_spectrum_data(
                adiabatic,
                ells,
            ).spectra["TT"],
            dtype=numpy.longdouble,
        )
        cdm_tt = numpy.asarray(
            cmb_projection._compute_custom_cmb_spectrum_data(
                cdm_mode,
                ells,
            ).spectra["TT"],
            dtype=numpy.longdouble,
        )

        self.assertTrue(numpy.all(numpy.isfinite(adiabatic_tt)))
        self.assertTrue(numpy.all(numpy.isfinite(cdm_tt)))
        self.assertGreater(
            float(numpy.max(numpy.abs(adiabatic_tt - cdm_tt))),
            1.0e-12,
        )
        self.assertEqual(
            cdm_mode["perturbation_data"]
            .initial_conditions["delta_c_seed"]
            .expression,
            "seed",
        )
        self.assertEqual(
            adiabatic["perturbation_data"]
            .initial_conditions["delta_c_seed"]
            .expression,
            "-1.5 * scalar_lapse_seed",
        )

    def test_declared_scalar_hierarchy_modes_seed_all_supported_families(
        self,
    ) -> None:
        """Every supported scalar mode should expose absolute seeds."""

        modes = (
            "adiabatic_scalar",
            "baryon_isocurvature",
            "cdm_isocurvature",
            "neutrino_density_isocurvature",
            "neutrino_velocity_isocurvature",
        )
        seed_expressions = {}
        for mode in modes:
            contract = _prepare_declared_contract(
                _declared_scalar_hierarchy_contract(
                    initial_mode=mode,
                    include_massive_neutrino=True,
                )
            )
            initial_conditions = contract[
                "perturbation_data"
            ].initial_conditions
            seed_expressions[mode] = {
                name: initial_conditions[name].expression
                for name in (
                    "delta_b_seed",
                    "delta_c_seed",
                    "delta_nu_seed",
                    "delta_nu_massive_q0_seed",
                    "theta_nu_massive_q0_seed",
                )
            }

        self.assertIn(
            "distribution_log_derivative",
            seed_expressions["adiabatic_scalar"]["delta_nu_massive_q0_seed"],
        )
        for mode in modes[1:]:
            self.assertNotIn(
                "distribution_log_derivative",
                seed_expressions[mode]["delta_nu_massive_q0_seed"],
            )
        self.assertEqual(
            seed_expressions["neutrino_velocity_isocurvature"][
                "delta_nu_massive_q0_seed"
            ],
            "0.0",
        )
        self.assertEqual(
            seed_expressions["neutrino_velocity_isocurvature"][
                "theta_nu_massive_q0_seed"
            ],
            "acoustic_k * seed",
        )
        self.assertEqual(
            len(
                {tuple(values.items()) for values in seed_expressions.values()}
            ),
            len(modes),
        )

    def test_declared_scalar_hierarchy_rejects_initial_collision_violation(
        self,
    ) -> None:
        """Fast-manifold initial states must satisfy collision constraints."""

        contract_data = _declared_scalar_hierarchy_contract()
        contract_data["perturbations"]["initial_conditions"] = {
            "theta_b_seed": {
                "target": {
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            }
        }
        contract = _prepare_declared_contract(_speedup_contract(contract_data))

        with self.assertRaisesRegex(
            ValueError,
            "initial collision constraint exceeded tolerance",
        ):
            cmb_projection._compute_custom_cmb_spectrum_data(
                contract,
                numpy.arange(20, 24, dtype=int),
            )

    def test_declared_scalar_modes_generate_distinct_source_histories(
        self,
    ) -> None:
        """Supported scalar modes should produce distinct histories."""

        modes = (
            "adiabatic_scalar",
            "baryon_isocurvature",
            "cdm_isocurvature",
            "neutrino_density_isocurvature",
            "neutrino_velocity_isocurvature",
        )
        histories = {}
        for mode in modes:
            contract_data = _declared_scalar_hierarchy_contract(
                initial_mode=mode
            )
            contract_data["numerical"].update(
                {
                    "k_min": 0.02,
                    "k_max": 0.02,
                    "k_sample_count": 1,
                    "eta_sample_count": 16,
                    "source_grid_multiplier": 1,
                    "initial_redshift": 2.0e4,
                }
            )
            contract = _prepare_declared_contract(contract_data)
            _, history, _ = _capture_visible_scalar_monopole_history(contract)
            self.assertTrue(numpy.all(numpy.isfinite(history)))
            histories[mode] = history

        reference = histories["adiabatic_scalar"]
        for mode in modes[1:]:
            self.assertGreater(
                float(numpy.max(numpy.abs(histories[mode] - reference))),
                1.0e-8,
                msg=f"mode {mode} collapsed to the adiabatic history",
            )

    def test_declared_scalar_hierarchy_massive_neutrino_response(
        self,
    ) -> None:
        """Massive-neutrino masses should alter physical moments and TT."""

        light_contract = _speedup_contract(
            _declared_scalar_hierarchy_contract(
                include_massive_neutrino=True,
                sum_mnu=0.06,
            )
        )
        light_contract["numerical"].update(
            {
                "eta_sample_count": 48,
                "k_sample_count": 8,
            }
        )
        light_contract["numerical"]["momentum_grids"][
            "massive_neutrino_default"
        ]["count"] = 6
        light = _prepare_declared_contract(light_contract)
        heavy_contract = _speedup_contract(
            _declared_scalar_hierarchy_contract(
                include_massive_neutrino=True,
                sum_mnu=6.0,
            )
        )
        heavy_contract["numerical"].update(
            {
                "eta_sample_count": 48,
                "k_sample_count": 8,
            }
        )
        heavy_contract["numerical"]["momentum_grids"][
            "massive_neutrino_default"
        ]["count"] = 6
        heavy = _prepare_declared_contract(heavy_contract)
        light_physical = (
            cmb_background._resolve_custom_cmb_physical_parameters(light)
        )
        heavy_physical = (
            cmb_background._resolve_custom_cmb_physical_parameters(heavy)
        )
        light_q_context = evolution._declared_momentum_grid_context(
            light["perturbation_data"],
            model_parameters=light["param_map"],
            physical_params=light_physical,
            scale_factor=0.5,
        )
        heavy_q_context = evolution._declared_momentum_grid_context(
            heavy["perturbation_data"],
            model_parameters=heavy["param_map"],
            physical_params=heavy_physical,
            scale_factor=0.5,
        )
        light_streaming_speed = numpy.asarray(
            light_q_context["massive_neutrino_q0_streaming_speed"],
            dtype=float,
        )
        heavy_streaming_speed = numpy.asarray(
            heavy_q_context["massive_neutrino_q0_streaming_speed"],
            dtype=float,
        )
        light_mass_fraction = numpy.asarray(
            light_q_context["massive_neutrino_q0_mass_fraction"],
            dtype=float,
        )
        heavy_mass_fraction = numpy.asarray(
            heavy_q_context["massive_neutrino_q0_mass_fraction"],
            dtype=float,
        )

        self.assertTrue(numpy.isfinite(light_streaming_speed))
        self.assertTrue(numpy.isfinite(heavy_streaming_speed))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(light_streaming_speed - heavy_streaming_speed)
                )
            ),
            1.0e-12,
        )
        self.assertGreater(
            float(
                numpy.max(numpy.abs(light_mass_fraction - heavy_mass_fraction))
            ),
            1.0e-12,
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        light_tt = numpy.asarray(
            cmb_projection._compute_custom_cmb_spectrum_data(
                light,
                ells,
            ).spectra["TT"],
            dtype=numpy.longdouble,
        )
        heavy_tt = numpy.asarray(
            cmb_projection._compute_custom_cmb_spectrum_data(
                heavy,
                ells,
            ).spectra["TT"],
            dtype=numpy.longdouble,
        )

        self.assertTrue(numpy.all(numpy.isfinite(light_tt)))
        self.assertTrue(numpy.all(numpy.isfinite(heavy_tt)))
        self.assertGreater(
            float(numpy.max(numpy.abs(light_tt - heavy_tt))),
            1.0e-15,
        )

    def test_massive_neutrino_grid_uses_physical_temperature_and_sources(
        self,
    ) -> None:
        """Massive q bins must use T_nu0 and physical Einstein weights."""

        raw_contract = _declared_scalar_hierarchy_contract(
            include_massive_neutrino=True,
            sum_mnu=0.06,
        )
        contract = _prepare_declared_contract(raw_contract)
        physical_params = (
            cmb_background._resolve_custom_cmb_physical_parameters(contract)
        )
        context = evolution._declared_momentum_grid_context(
            contract["perturbation_data"],
            model_parameters=contract["param_map"],
            physical_params=physical_params,
            scale_factor=1.0,
        )

        self.assertLess(
            float(context["massive_neutrino_q0_streaming_speed"]),
            0.2,
        )
        self.assertGreater(
            float(context["massive_neutrino_density_fraction"]),
            0.0,
        )
        expected_omega_nu = 0.06 / (93.14 * (67.4 / 100.0) ** 2)
        self.assertAlmostEqual(
            float(context["massive_neutrino_density_fraction"]),
            expected_omega_nu,
            delta=expected_omega_nu * 0.02,
        )
        self.assertGreater(
            float(context["massive_neutrino_density_fraction"])
            - float(context["massive_neutrino_pressure_fraction"]),
            0.0,
        )

    def test_synchronous_route_evolves_only_synchronous_metric_states(
        self,
    ) -> None:
        """Synchronous execution must expose explicit gauge metric states."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(gauge="synchronous")
        )
        perturbation_data = contract["perturbation_data"]

        self.assertNotIn("evolve_Phi", perturbation_data.equations)
        self.assertIn("evolve_Phi_gi", perturbation_data.equations)
        self.assertIn("evolve_h_sync_metric", perturbation_data.equations)
        self.assertIn("evolve_eta_sync_metric", perturbation_data.equations)
        self.assertIn("evolve_gauge_shift_alpha", perturbation_data.equations)
        self.assertEqual(
            perturbation_data.closures["phi_closure"].target,
            "Phi",
        )
        self.assertEqual(
            perturbation_data.closures["phi_closure"].expression,
            "Phi_gi",
        )
        self.assertEqual(
            perturbation_data.closures["psi_closure"].target,
            "Psi",
        )

    def test_gauge_invariant_route_evolves_bardeen_metric_states(self) -> None:
        """Gauge-invariant execution must use its own metric states."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(gauge="gauge_invariant")
        )
        perturbation_data = contract["perturbation_data"]

        self.assertIn("evolve_Phi_gi", perturbation_data.equations)
        self.assertNotIn("evolve_Phi", perturbation_data.equations)
        self.assertEqual(
            perturbation_data.constraints["observable_phi_constraint"].target,
            "Phi",
        )
        self.assertEqual(
            perturbation_data.constraints[
                "observable_phi_constraint"
            ].expression,
            "Phi_gi",
        )
        self.assertEqual(
            perturbation_data.closures["observable_psi_closure"].target,
            "Psi",
        )
        self.assertEqual(
            perturbation_data.closures["observable_psi_closure"].expression,
            "Psi_gi",
        )

    def test_declared_scalar_hierarchy_momentum_grid_limits_and_convergence(
        self,
    ) -> None:
        """Physical q moments should respect limits and converge with count."""

        def _momentum_context(
            *,
            sum_mnu: float,
            q_count: int,
            a_value: float,
        ) -> dict[str, object]:
            contract_data = _declared_scalar_hierarchy_contract(
                include_massive_neutrino=True,
                sum_mnu=sum_mnu,
            )
            contract_data["numerical"]["momentum_grids"][
                "massive_neutrino_default"
            ]["count"] = q_count
            contract = _prepare_declared_contract(contract_data)
            physical_params = (
                cmb_background._resolve_custom_cmb_physical_parameters(
                    contract
                )
            )
            return evolution._declared_momentum_grid_context(
                contract["perturbation_data"],
                model_parameters=contract["param_map"],
                physical_params=physical_params,
                scale_factor=a_value,
            )

        relativistic = _momentum_context(
            sum_mnu=1.0e-6, q_count=8, a_value=0.01
        )
        nonrelativistic = _momentum_context(
            sum_mnu=60.0, q_count=8, a_value=1.0
        )
        coarse = _momentum_context(sum_mnu=1.5, q_count=4, a_value=0.5)
        medium = _momentum_context(sum_mnu=1.5, q_count=6, a_value=0.5)
        fine = _momentum_context(sum_mnu=1.5, q_count=8, a_value=0.5)

        for prefix in (
            "massive_neutrino_q0_density_weight",
            "massive_neutrino_q0_pressure_weight",
            "massive_neutrino_q0_momentum_weight",
            "massive_neutrino_q0_shear_weight",
        ):
            self.assertGreater(float(relativistic[prefix]), 0.0)
        for context in (relativistic, nonrelativistic):
            for suffix in (
                "density_weight",
                "pressure_weight",
                "momentum_weight",
                "shear_weight",
            ):
                total = 0.0
                for q_index in range(8):
                    total += float(
                        context[f"massive_neutrino_q{q_index}_{suffix}"]
                    )
                self.assertAlmostEqual(total, 1.0, places=12)

        self.assertAlmostEqual(
            float(relativistic["massive_neutrino_pressure_ratio"]),
            1.0 / 3.0,
            delta=5.0e-4,
        )
        self.assertLess(
            float(relativistic["massive_neutrino_mass_fraction"]),
            1.0e-5,
        )
        self.assertLess(
            float(nonrelativistic["massive_neutrino_pressure_ratio"]),
            5.0e-2,
        )
        self.assertGreater(
            float(nonrelativistic["massive_neutrino_mass_fraction"]),
            0.9,
        )
        self.assertGreater(
            float(relativistic["massive_neutrino_q0_streaming_speed"]),
            float(nonrelativistic["massive_neutrino_q0_streaming_speed"]),
        )

        coarse_pressure = float(coarse["massive_neutrino_pressure_ratio"])
        medium_pressure = float(medium["massive_neutrino_pressure_ratio"])
        fine_pressure = float(fine["massive_neutrino_pressure_ratio"])
        coarse_mass_fraction = float(coarse["massive_neutrino_mass_fraction"])
        medium_mass_fraction = float(medium["massive_neutrino_mass_fraction"])
        fine_mass_fraction = float(fine["massive_neutrino_mass_fraction"])

        self.assertLess(
            abs(fine_pressure - medium_pressure),
            abs(medium_pressure - coarse_pressure),
        )
        self.assertLess(
            abs(fine_mass_fraction - medium_mass_fraction),
            abs(medium_mass_fraction - coarse_mass_fraction),
        )

    def test_massive_neutrino_q_and_hierarchy_spectra_converge(self) -> None:
        """Massive-neutrino q and hierarchy refinements meet final bounds."""

        spectra = []
        for q_count, hierarchy_l_max in ((16, 5), (20, 5), (20, 7)):
            contract = _speedup_contract(
                _declared_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                    sum_mnu=0.5,
                )
            )
            contract["model_name"] = (
                f"MassiveNeutrinoQ{q_count}L{hierarchy_l_max}"
            )
            contract["numerical"].update(
                {
                    "k_sample_count": 6,
                    "eta_sample_count": 128,
                    "massive_neutrino_hierarchy_l_max": hierarchy_l_max,
                }
            )
            contract["numerical"]["momentum_grids"][
                "massive_neutrino_default"
            ]["count"] = q_count
            contract["perturbations"]["numerics"] = copy.deepcopy(
                contract["numerical"]
            )
            cache.clear_cmb_caches()
            spectrum_data = cmb_projection._compute_custom_cmb_spectrum_data(
                _prepare_declared_contract(contract),
                numpy.asarray((20, 40), dtype=int),
                requested_spectra=("TT",),
            )
            spectra.append(spectrum_data.spectra["TT"])

        q_metric = convergence.evaluate_control_refinement(
            spectra[0],
            spectra[1],
            name="massive-neutrino q grid",
            tolerance=convergence.FINAL_Q_GRID_RELATIVE_TOLERANCE,
        )
        hierarchy_metric = convergence.evaluate_control_refinement(
            spectra[1],
            spectra[2],
            name="massive-neutrino hierarchy",
            tolerance=(convergence.FINAL_HIERARCHY_RELATIVE_TOLERANCE),
        )
        convergence.require_convergence(q_metric)
        convergence.require_convergence(hierarchy_metric)

    def test_declared_scalar_hierarchy_momentum_grid_cache_reuses(
        self,
    ) -> None:
        """Momentum-grid quadrature should reuse bounded cache entries."""

        cache.clear_cmb_caches()
        baseline = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        shifted = _declared_scalar_hierarchy_contract(
            include_massive_neutrino=True
        )
        shifted["param_map"]["As"] *= 1.1
        shifted_contract = _prepare_declared_contract(shifted)
        baseline_physical = (
            cmb_background._resolve_custom_cmb_physical_parameters(
                baseline,
            )
        )
        shifted_physical = (
            cmb_background._resolve_custom_cmb_physical_parameters(
                shifted_contract
            )
        )
        evolution._declared_momentum_grid_context(
            baseline["perturbation_data"],
            model_parameters=baseline["param_map"],
            physical_params=baseline_physical,
            scale_factor=0.5,
        )
        first_stats = cache.cmb_cache_stats()["declared_momentum_grid"]
        first_topology_stats = cache.cmb_cache_stats()[
            "declared_momentum_topology"
        ]
        evolution._declared_momentum_grid_context(
            shifted_contract["perturbation_data"],
            model_parameters=shifted_contract["param_map"],
            physical_params=shifted_physical,
            scale_factor=0.5,
        )
        second_stats = cache.cmb_cache_stats()["declared_momentum_grid"]
        second_topology_stats = cache.cmb_cache_stats()[
            "declared_momentum_topology"
        ]

        self.assertEqual(first_stats["entries"], second_stats["entries"])
        self.assertGreaterEqual(second_stats["hits"], first_stats["hits"] + 1)
        self.assertEqual(
            first_topology_stats["entries"],
            second_topology_stats["entries"],
        )
        self.assertGreaterEqual(
            second_topology_stats["hits"],
            first_topology_stats["hits"] + 1,
        )

    def test_declared_scalar_hierarchy_reuses_structural_runtime_bundle(
        self,
    ) -> None:
        """Scalar runtime signatures should survive bound-value changes."""

        baseline = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        shifted = _declared_scalar_hierarchy_contract()
        shifted["param_map"]["As"] *= 1.1
        shifted["param_map"]["H0"] = 68.1
        shifted_prepared = _prepare_declared_contract(shifted)

        self.assertEqual(
            baseline["runtime_signature"],
            shifted_prepared["runtime_signature"],
        )

    def test_declared_scalar_hierarchy_runs_camb_free(self) -> None:
        """The declared scalar route should stay CAMB-free at runtime."""

        contract = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        ells = numpy.arange(20, 60, dtype=int)
        with mock.patch.object(
            camb_reference.camb,
            "get_results",
            side_effect=AssertionError(
                "declared scalar hierarchy should not call CAMB"
            ),
        ):
            spectra = cmb.compute_cmb_spectrum_from_contract(
                contract,
                ells,
                spectra=("TT", "TE", "EE"),
            )

        self.assertEqual(set(spectra), {"TT", "TE", "EE"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_direct_custom_path_does_not_call_camb(self) -> None:
        """The direct declared-graph route should stay CAMB-free."""

        contract = _speedup_contract(_custom_contract())
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch.object(
            camb_reference.camb,
            "get_results",
            side_effect=AssertionError("CAMB prediction path should not run"),
        ):
            result = cmb.compute_cmb_spectrum_from_contract(
                contract,
                ells,
                spectra=("TT", "TE", "EE"),
            )

        self.assertEqual(set(result), {"TT", "TE", "EE"})
        for spectrum in result.values():
            self.assertTrue(numpy.all(numpy.isfinite(spectrum)))
            self.assertEqual(spectrum.shape, (ells.size,))


class SliceSixteenRuntimeAuthorityTestCase(unittest.TestCase):
    """Protect the declared scalar graph from alternate physics solvers."""

    def test_scalar_runtime_has_one_compiled_evolution_authority(self) -> None:
        """The production scalar entry point must use the declared graph."""

        boundary_source = inspect.getsource(
            cmb_projection._compute_custom_cmb_spectrum_data
        )
        self.assertIn(
            "_compute_custom_cmb_spectrum_data_impl",
            boundary_source,
        )
        source = inspect.getsource(
            cmb_projection._compute_custom_cmb_spectrum_data_impl
        )
        self.assertIn("_mode_rhs", source)
        self.assertIn("_integrate_declared_state_history", source)
        for forbidden in (
            "_integrate_generated_scalar_history_fast",
            "_integrate_generated_scalar_history_batch",
            "_batch_generated_source_histories",
        ):
            self.assertNotIn(forbidden, source)

    def test_fast_collision_projection_uses_declared_linear_algebra(
        self,
    ) -> None:
        """Fast-manifold projection must preserve a declared invariant."""

        wave_number = 0.2
        baryon_loading = 0.15
        matrix = numpy.asarray(
            (
                (-1.0, 1.0 / (3.0 * wave_number)),
                (3.0 * wave_number * baryon_loading, -baryon_loading),
            ),
            dtype=float,
        )
        current = numpy.asarray((0.7, 0.03), dtype=float)
        result = cmb_projection._solve_declared_fast_collision_target(
            matrix,
            numpy.zeros(2, dtype=float),
            current,
            100.0,
        )
        invariant = numpy.asarray((3.0 * wave_number * baryon_loading, 1.0))
        numpy.testing.assert_allclose(
            float(invariant @ result),
            float(invariant @ current),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        numpy.testing.assert_allclose(
            matrix @ result,
            numpy.zeros(2, dtype=float),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_declared_scalar_layout_has_no_implicit_state_slots(self) -> None:
        """Every production state slot must originate in a declared
        equation."""

        prepared = _prepare_declared_contract(
            _declared_scalar_hierarchy_contract()
        )
        perturbation_data = prepared["perturbation_data"]
        runtime_spec = evolution._prepare_declared_graph_runtime_spec(
            perturbation_data
        )
        equation_variables = {
            str(entry.lhs.variable)
            for entry in perturbation_data.equations.values()
        }
        state_variables = {
            str(slot.variable)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
        }
        self.assertEqual(state_variables, equation_variables)

    def test_generated_scalar_execution_compiles_declared_equations(
        self,
    ) -> None:
        """Generated scalar output must execute its compiled equation
        program."""

        contract = _declared_scalar_hierarchy_contract()
        contract["numerical"].update(
            {
                "ell_max": 120,
                "k_sample_count": 3,
                "eta_sample_count": 64,
                "evolution_eta_sample_count": 64,
            }
        )
        prepared = _prepare_declared_contract(contract)
        evolution._compile_equation_program.cache_clear()
        with mock.patch.object(
            cmb_projection,
            "_compile_equation_program",
            wraps=evolution._compile_equation_program,
        ) as compile_program:
            spectra = cmb.compute_cmb_spectrum_from_contract(
                prepared,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )
        self.assertGreaterEqual(compile_program.call_count, 1)
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the public CMB helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        """The module should export the expected public helpers."""

        self.assertEqual(
            set(cmb.__all__),
            {
                "CMBLike",
                "CMBBatchResult",
                "CMBError",
                "ConstraintViolationError",
                "ContractError",
                "ConvergenceError",
                "ImplementationError",
                "InitialPointError",
                "NonFiniteEvolutionError",
                "ParameterDomainError",
                "UnsupportedCapabilityError",
                "compute_cmb_spectrum",
                "compute_cmb_spectrum_batch",
                "compute_cmb_spectrum_cached",
                "compute_cmb_spectrum_from_contract",
                "CMBContractAudit",
                "CMBModelDeclarationDecision",
                "CMBSourceGraphAudit",
                "CMB_CORPUS_BASELINE_REQUEST",
                "CMB_USMF2_BASELINE_TIERS",
                "CMBCorpusBaselineRow",
                "CMBModelDiagnostic",
                "assess_physical_spectrum_shape",
                "audit_source_history_residuals",
                "build_cmb_corpus_baseline_report",
                "build_cmb_certification_report",
                "compare_cmb_spectra_to_reference",
                "assert_bundled_cmb_contracts",
                "assert_bundled_cmb_declarations",
                "assert_bundled_cmb_source_graphs",
                "audit_bundled_cmb_declarations",
                "audit_bundled_cmb_contracts",
                "audit_bundled_cmb_source_graphs",
                "discover_bundled_cmb_plugins",
                "run_bundled_cmb_corpus_baseline",
                "run_bundled_cmb_diagnostics",
                "run_cmb_model_diagnostic",
                "write_cmb_corpus_baseline_report",
                "write_cmb_certification_report",
            },
        )
        self.assertTrue(hasattr(cmb, "CMBLike"))
        self.assertTrue(callable(cmb.compute_cmb_spectrum))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_batch))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_cached))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_from_contract))
        self.assertTrue(callable(cmb.build_cmb_certification_report))
        self.assertTrue(callable(cmb.write_cmb_certification_report))
        self.assertFalse(hasattr(cmb, "compute_camb_background_observables"))
        self.assertFalse(
            hasattr(cmb, "compute_cmb_spectrum_from_legacy_params_for_tests")
        )
        self.assertFalse(hasattr(cmb, "describe_camb_configuration"))
        self.assertFalse(hasattr(cmb, "_CustomCMBBackgroundData"))
        self.assertFalse(hasattr(cmb, "_make_camb_params"))
        self.assertFalse(hasattr(cmb, "camb"))
        self.assertFalse(hasattr(cmb, "compute_cmb_spectrum_from_dict"))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        """The likelihood protocol symbols should remain available."""

        self.assertTrue(callable(cmb.CMBLike.loglike))
        self.assertTrue(callable(cmb.CMBLike.loglike_batch))
        self.assertTrue(callable(cmb.CMBLike.prepare_worker_runtime))
        self.assertTrue(callable(cmb.CMBLike.preflight))
        self.assertTrue(hasattr(cmb.CMBLike.state, "__get__"))

    def test_production_package_has_no_reference_solver_imports(self) -> None:
        """Production modules must not import CAMB or CLASS."""

        package_root = Path(cmb.__file__).resolve().parents[3]
        forbidden_modules = {"camb", "classy"}
        violations = []
        for source_path in sorted(package_root.rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported_modules = ()
                if isinstance(node, ast.Import):
                    imported_modules = tuple(
                        alias.name.split(".", maxsplit=1)[0]
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules = (node.module.split(".", maxsplit=1)[0],)
                if forbidden_modules.intersection(imported_modules):
                    relative_path = source_path.relative_to(package_root)
                    violations.append(f"{relative_path}:{node.lineno}")
        self.assertEqual(violations, [])
        removed_solver = package_root / "lib/likelihoods/cmb/camb_solver.py"
        self.assertFalse(removed_solver.exists())


class CMBLikeMultiSpectrumTestCase(unittest.TestCase):
    """Exercise the public CMB likelihood surface on spectrum blocks."""

    def test_loglike_preserves_interleaved_repeated_spectrum_rows(
        self,
    ) -> None:
        """Likelihood rows and covariance must retain dataset ordering."""

        class _BlockSpectrumPlugin:
            """Return a prepared declared runtime for likelihood testing."""

            def get_cmb_declared_runtime(self, _params):
                return {
                    "background": {},
                    "background_runtime": object(),
                    "calls": [],
                    "grids": {},
                    "model_name": "BlockSpectrum",
                    "model_parameters": {},
                    "numerical": {},
                    "param_map": {},
                    "perturbation_data": object(),
                    "perturbations": {},
                    "values": {},
                }

        cmb_data = pandas.DataFrame(
            {
                "ell": [30, 20, 30, 20, 30],
                "spectrum": ["TE", "TT", "TT", "TE", "TE"],
                "Dl_obs": [0.5, 1.0, 2.0, 0.25, 0.125],
            }
        )
        cmb_data.attrs["covariance_matrix_inv"] = numpy.asarray(
            [
                [2.0, 0.1, 0.0, 0.0, 0.0],
                [0.1, 2.0, 0.1, 0.0, 0.0],
                [0.0, 0.1, 2.0, 0.1, 0.0],
                [0.0, 0.0, 0.1, 2.0, 0.1],
                [0.0, 0.0, 0.0, 0.1, 2.0],
            ],
            dtype=float,
        )
        theory = {
            "TT": numpy.asarray([99.0, 1.0, 2.0, 99.0, 99.0]),
            "TE": numpy.asarray([0.5, 99.0, 99.0, 0.25, 0.125]),
        }
        with mock.patch(
            (
                "copernican.lib.likelihoods.cmb.cmb."
                "_compute_declared_perturbation_spectrum"
            ),
            return_value=theory,
        ) as compute_spectrum:
            likelihood = cmb.CMBLike(cmb_data, _BlockSpectrumPlugin())
            loglike = likelihood.loglike(())

        self.assertEqual(likelihood.state["chi2"], 0.0)
        self.assertEqual(loglike, 0.0)
        numpy.testing.assert_array_equal(
            likelihood._ells,
            [30, 20, 30, 20, 30],
        )
        self.assertEqual(likelihood._observed.shape, (5,))
        self.assertEqual(
            compute_spectrum.call_args.kwargs["spectra"], ("TE", "TT")
        )
        numpy.testing.assert_array_equal(
            compute_spectrum.call_args.args[1],
            [30, 20, 30, 20, 30],
        )


if __name__ == "__main__":
    unittest.main()
