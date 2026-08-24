"""Fixed-parameter diagnostics for declared CCMBS model contracts.

The diagnostic path deliberately runs outside plotting and sampling.  It
captures the raw transfer products and runtime envelope produced by CCMBS,
then repeats the same fixed-point request on a doubled wave-number grid so
resolution evidence is available before any likelihood or figure consumes a
spectrum.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy

from ... import model_adapter, model_coder, model_spec_validator
from ...cmb_contract import audit_cmb_capabilities
from .errors import CMBError, ConvergenceError
from .runtime.projection import _compute_custom_cmb_spectrum_data

_DEFAULT_SPECTRA = ("TT", "TE", "EE")
_DEFAULT_ELL_VALUES = (2, 20, 100)
_DEFAULT_RELATIVE_TOLERANCES = {
    "TT": 1.0e-2,
    "TE": 2.0e-2,
    "EE": 1.0e-2,
}

_AUTO_REFERENCE_SPECTRA = {
    "TT",
    "EE",
    "BB",
    "PP",
    "LENSED_TT",
    "LENSED_EE",
    "LENSED_BB",
}


def _public_spectrum_values(
    ell_values: Sequence[int],
    raw_spectra: Mapping[str, Any],
) -> dict[str, numpy.ndarray]:
    """Convert CCMBS ``C_ell`` products to the public ``D_ell`` units.

    The direct projection runtime intentionally returns dimensionless
    primordial ``C_ell`` values.  The public solver applies the common
    ``ell(ell+1)/(2*pi)`` and CMB-temperature normalization afterwards.
    Diagnostics must make that same boundary explicit before comparing with
    the frozen CAMB fixture or describing spectra as public products.
    """

    ells = numpy.asarray(tuple(ell_values), dtype=numpy.longdouble)
    ell_factor = ells * (ells + 1.0) / (2.0 * numpy.longdouble(numpy.pi))
    temperature_scale = numpy.longdouble("2.7255e6") ** 2
    converted: dict[str, numpy.ndarray] = {}
    for name, values in raw_spectra.items():
        token = str(name).upper()
        scale = (
            ell_factor * temperature_scale
            if token in {"TT", "TE", "EE", "BB"}
            else ell_factor
        )
        converted[str(name)] = numpy.asarray(
            numpy.asarray(values, dtype=numpy.longdouble) * scale,
            dtype=float,
        )
    return converted


def assess_physical_spectrum_shape(
    ell_values: Iterable[int],
    spectra: Mapping[str, Any],
) -> dict[str, Any]:
    """Check finite, positive auto spectra and reject quadrature spikes.

    The check is intentionally performed on raw public arrays, before any
    plotting.  It does not prescribe a cosmological amplitude or peak
    locations; it only rejects non-finite power, materially negative auto
    spectra, and adjacent-multipole jumps too large to be a resolved acoustic
    surface.  Sparse diagnostic requests receive the finite/positivity checks
    but no smoothness verdict.
    """

    ells = numpy.asarray(tuple(ell_values), dtype=int)
    if ells.ndim != 1 or ells.size == 0 or numpy.any(numpy.diff(ells) <= 0):
        raise ValueError(
            "Spectrum-shape ell values must be strictly increasing"
        )
    result: dict[str, Any] = {
        "finite": True,
        "auto_spectra_nonnegative": True,
        "smooth": None,
        "maximum_log_jump": 0.0,
        "issues": [],
    }
    issues: list[str] = result["issues"]
    for name, values in spectra.items():
        array = numpy.asarray(values, dtype=float)
        if array.shape != ells.shape or not numpy.all(numpy.isfinite(array)):
            result["finite"] = False
            issues.append(f"{name} is non-finite or has the wrong shape")
            continue
        if str(name).upper() in {"TT", "EE", "BB"}:
            scale = max(float(numpy.max(numpy.abs(array), initial=0.0)), 1.0)
            if float(numpy.min(array, initial=0.0)) < -1.0e-10 * scale:
                result["auto_spectra_nonnegative"] = False
                issues.append(f"{name} contains materially negative power")
    if ells.size >= 16 and "TT" in spectra:
        tt_spectrum = numpy.asarray(spectra["TT"], dtype=float)
        positive = numpy.maximum(tt_spectrum, numpy.finfo(float).tiny)
        jumps = numpy.abs(numpy.diff(numpy.log(positive)))
        maximum_log_jump = float(numpy.max(jumps, initial=0.0))
        result["maximum_log_jump"] = maximum_log_jump
        # A five-to-ten-fold jump between adjacent requested multipoles is a
        # quadrature alias, not an acoustic peak.  The threshold is relaxed
        # for sparse grids by scaling with the largest ell spacing.
        spacing = max(int(numpy.max(numpy.diff(ells))), 1)
        threshold = numpy.log(6.0) * max(1.0, spacing / 5.0)
        result["smooth"] = bool(maximum_log_jump <= threshold)
        if not result["smooth"]:
            issues.append("TT contains an unresolved adjacent-multipole spike")
    if issues:
        result["issues"] = tuple(issues)
    else:
        result["issues"] = ()
    return result


def compare_cmb_spectra_to_reference(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    relative_tolerances: Mapping[str, float] | None = None,
    auto_spectrum_floor: float = 1.0e-10,
) -> dict[str, Any]:
    """Compare raw public spectra with an independent fixed-point reference.

    Auto spectra use a p90 fractional error above a relative reference floor;
    cross spectra use an RMS error normalized by the reference RMS so sign
    changes remain well-defined.  The function is backend-neutral: tests may
    pass CAMB/CLASS data while the production package remains reference-solver
    free.
    """

    tolerances = dict(_DEFAULT_RELATIVE_TOLERANCES)
    tolerances.update(relative_tolerances or {})
    metrics: dict[str, dict[str, Any]] = {}
    missing = sorted(set(reference) - set(actual))
    if missing:
        raise ValueError(
            "Missing spectra for reference comparison: " + ", ".join(missing)
        )
    for name, reference_values in reference.items():
        if name not in actual:
            continue
        actual_array = numpy.asarray(actual[name], dtype=numpy.longdouble)
        reference_array = numpy.asarray(
            reference_values, dtype=numpy.longdouble
        )
        if actual_array.shape != reference_array.shape:
            raise ValueError(
                f"Spectrum '{name}' has incompatible comparison shapes: "
                f"{actual_array.shape} != {reference_array.shape}"
            )
        finite = numpy.isfinite(actual_array) & numpy.isfinite(reference_array)
        if not numpy.any(finite):
            raise ValueError(
                f"Spectrum '{name}' has no finite comparison data"
            )
        tolerance = float(tolerances.get(str(name).upper(), 1.0e-2))
        if str(name).upper() in _AUTO_REFERENCE_SPECTRA:
            scale = numpy.max(numpy.abs(reference_array[finite]), initial=0.0)
            floor = max(
                numpy.longdouble("1.0e-30"),
                numpy.longdouble(auto_spectrum_floor) * scale,
            )
            supported = finite & (numpy.abs(reference_array) > floor)
            if not numpy.any(supported):
                raise ValueError(
                    f"Spectrum '{name}' has no values above the "
                    "comparison floor"
                )
            fractional = numpy.abs(
                (actual_array[supported] - reference_array[supported])
                / reference_array[supported]
            )
            error = float(numpy.percentile(fractional, 90.0))
            metrics[str(name)] = {
                "kind": "auto",
                "median_fractional": float(numpy.median(fractional)),
                "p90_fractional": error,
                "max_fractional": float(numpy.max(fractional)),
                "tolerance": tolerance,
                "converged": bool(error <= tolerance),
            }
        else:
            delta = actual_array[finite] - reference_array[finite]
            reference_rms = numpy.sqrt(
                numpy.mean(numpy.square(reference_array[finite]))
            )
            if reference_rms <= numpy.longdouble("1.0e-30"):
                error = (
                    0.0
                    if numpy.all(
                        numpy.abs(delta) <= numpy.longdouble("1.0e-30")
                    )
                    else float("inf")
                )
            else:
                error = float(
                    numpy.sqrt(numpy.mean(numpy.square(delta))) / reference_rms
                )
            metrics[str(name)] = {
                "kind": "cross",
                "normalized_rms": error,
                "tolerance": tolerance,
                "converged": bool(error <= tolerance),
            }
    return {
        "metrics": metrics,
        "converged": bool(metrics)
        and all(bool(metric["converged"]) for metric in metrics.values()),
    }


_SOURCE_RESIDUAL_DEFINITIONS = {
    "einstein_energy": (
        "acoustic_k_sq",
        "Phi",
        "Hconf",
        "Phi_tau",
        "Psi",
        "einstein_gravity_strength",
        "total_density_source",
    ),
    "einstein_momentum": (
        "acoustic_k_sq",
        "Phi_tau",
        "Hconf",
        "Psi",
        "einstein_gravity_strength",
        "total_momentum_source",
    ),
    "einstein_shear": (
        "acoustic_k_sq",
        "metric_shear_correction",
        "einstein_gravity_strength",
        "total_shear_source",
    ),
    "visibility_monopole": (
        "visibility",
        "observable_theta_gamma0",
        "Psi",
        "temperature_monopole",
    ),
    "visibility_quadrupole": (
        "visibility",
        "polarization_moment",
        "temperature_quadrupole",
    ),
    "visibility_quadrupole_derivative": (
        "visibility",
        "polarization_moment",
        "temperature_quadrupole_derivative",
    ),
    "visibility_doppler": (
        "visibility",
        "observable_theta_b",
        "acoustic_k",
        "temperature_doppler",
    ),
    "polarization": (
        "visibility",
        "polarization_moment",
        "polarization_source",
    ),
    "isw": (
        "tau",
        "Phi_history_tau",
        "Psi_tau",
        "temperature_isw",
    ),
}

_DEFAULT_SOURCE_RESIDUAL_ABSOLUTE_TOLERANCES = {
    "einstein_energy": 3.0e-3,
    "einstein_momentum": 1.0e-6,
    "einstein_shear": 1.0e-6,
    "visibility_monopole": 1.0e-6,
    "visibility_quadrupole": 1.0e-6,
    "visibility_quadrupole_derivative": 1.0e-6,
    "visibility_doppler": 1.0e-6,
    "polarization": 1.0e-6,
    "isw": 1.0e-6,
}
_DEFAULT_HIERARCHY_RELATIVE_TOLERANCE = 1.0
_DEFAULT_HIERARCHY_ABSOLUTE_TOLERANCE = 1.25e-2


def resolve_source_residual_audit_controls(
    accuracy_controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve explicit tolerances for independent generated-source audits.

    The normalized Einstein measure is ill-conditioned when all source terms
    are close to zero, while finite-difference hierarchy checks have a genuine
    absolute integration error.  Audits therefore retain both criteria and
    accept a sample when either declared bound is met.  Contracts may override
    these defaults with ``source_residual_audit`` controls.
    """

    controls = accuracy_controls or {}
    if not isinstance(controls, Mapping):
        raise ValueError("accuracy_controls must be a mapping")
    relative_tolerances = {
        name: 1.0e-8 for name in _SOURCE_RESIDUAL_DEFINITIONS
    }
    absolute_tolerances = dict(_DEFAULT_SOURCE_RESIDUAL_ABSOLUTE_TOLERANCES)
    hierarchy = {
        "relative_tolerance": _DEFAULT_HIERARCHY_RELATIVE_TOLERANCE,
        "absolute_tolerance": _DEFAULT_HIERARCHY_ABSOLUTE_TOLERANCE,
    }
    declared = controls.get("source_residual_audit")
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("source_residual_audit must be a mapping")
        for key, target in (
            ("relative_tolerances", relative_tolerances),
            ("absolute_tolerances", absolute_tolerances),
        ):
            values = declared.get(key)
            if values is None:
                continue
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"source_residual_audit.{key} must be a mapping"
                )
            for name, value in values.items():
                name = str(name)
                if name not in target:
                    raise ValueError(
                        f"Unknown source residual audit tolerance: {name}"
                    )
                value = float(value)
                if not numpy.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        "Source residual audit tolerances must be finite "
                        "and positive"
                    )
                target[name] = value
        declared_hierarchy = declared.get("hierarchy_equations")
        if declared_hierarchy is not None:
            if not isinstance(declared_hierarchy, Mapping):
                raise ValueError(
                    "source_residual_audit.hierarchy_equations must be "
                    "a mapping"
                )
            for name in hierarchy:
                if name not in declared_hierarchy:
                    continue
                value = float(declared_hierarchy[name])
                if not numpy.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        "Hierarchy residual audit tolerances must be finite "
                        "and positive"
                    )
                hierarchy[name] = value
    return {
        "schema_version": 1,
        "criterion": "relative_or_absolute",
        "relative_tolerances": relative_tolerances,
        "absolute_tolerances": absolute_tolerances,
        "hierarchy_equations": hierarchy,
        "provenance": (
            "accuracy_controls.source_residual_audit"
            if declared is not None
            else "CCMBS declared diagnostic defaults"
        ),
    }


def audit_source_history_residuals(
    runtime_envelope: Mapping[str, Any],
    *,
    relative_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Recompute source and metric closures from raw runtime samples.

    CCMBS records a compact set of source-history terms at deterministic eta
    anchors.  This audit deliberately does not consume the solver-owned
    ``scalar_constraint_diagnostics`` aggregates; it reconstructs every
    residual from the recorded terms so a physically inconsistent source
    graph cannot appear valid merely because its internal validator agrees
    with itself.
    """

    raw_modes = runtime_envelope.get("source_history_residual_samples_by_k")
    if not isinstance(raw_modes, Mapping) or not raw_modes:
        return {
            "schema_version": 1,
            "independent_recomputation": True,
            "available": False,
            "mode_count": 0,
            "sample_count": 0,
            "metrics": {},
            "converged": False,
            "issues": ("CCMBS did not expose source-history audit samples",),
        }

    controls = runtime_envelope.get("source_residual_audit_controls")
    if isinstance(controls, Mapping):
        relative_tolerances = {
            name: float(
                controls.get("relative_tolerances", {}).get(
                    name, relative_tolerance
                )
            )
            for name in _SOURCE_RESIDUAL_DEFINITIONS
        }
        absolute_tolerances = {
            name: float(controls.get("absolute_tolerances", {}).get(name, 0.0))
            for name in _SOURCE_RESIDUAL_DEFINITIONS
        }
        hierarchy_controls = controls.get("hierarchy_equations", {})
        hierarchy_relative_tolerance = float(
            hierarchy_controls.get("relative_tolerance", 1.0)
        )
        hierarchy_absolute_tolerance = float(
            hierarchy_controls.get("absolute_tolerance", 0.0)
        )
    else:
        relative_tolerances = {
            name: float(relative_tolerance)
            for name in _SOURCE_RESIDUAL_DEFINITIONS
        }
        absolute_tolerances = {
            name: 0.0 for name in _SOURCE_RESIDUAL_DEFINITIONS
        }
        hierarchy_relative_tolerance = 1.0e-2
        hierarchy_absolute_tolerance = 0.0
    metrics = {
        name: {
            "maximum_absolute": 0.0,
            "maximum_normalized": 0.0,
            "tolerance": relative_tolerances[name],
            "absolute_tolerance": absolute_tolerances[name],
            "criterion": "relative_or_absolute",
            "sample_count": 0,
            "available": True,
            "converged": True,
        }
        for name in _SOURCE_RESIDUAL_DEFINITIONS
    }
    issues: list[str] = []
    mode_count = 0
    sample_count = 0

    def _residual(name: str, sample: Mapping[str, Any]) -> tuple[float, float]:
        """Return absolute and normalized residuals for one raw sample."""

        values = {
            key: float(sample[key])
            for key in _SOURCE_RESIDUAL_DEFINITIONS[name]
        }
        if name == "einstein_energy":
            terms = (
                values["acoustic_k_sq"] * values["Phi"],
                3.0
                * values["Hconf"]
                * (values["Phi_tau"] + values["Hconf"] * values["Psi"]),
                1.5
                * values["einstein_gravity_strength"]
                * values["total_density_source"],
            )
        elif name == "einstein_momentum":
            terms = (
                values["acoustic_k_sq"]
                * (values["Phi_tau"] + values["Hconf"] * values["Psi"]),
                -1.5
                * values["einstein_gravity_strength"]
                * values["total_momentum_source"],
            )
        elif name == "einstein_shear":
            terms = (
                values["acoustic_k_sq"] * values["metric_shear_correction"],
                -3.0
                * values["einstein_gravity_strength"]
                * values["total_shear_source"],
            )
        elif name == "visibility_monopole":
            terms = (
                values["visibility"]
                * (values["observable_theta_gamma0"] + values["Psi"]),
                -values["temperature_monopole"],
            )
        elif name == "visibility_quadrupole":
            terms = (
                2.5 * values["visibility"] * values["polarization_moment"],
                -values["temperature_quadrupole"],
            )
        elif name == "visibility_quadrupole_derivative":
            terms = (
                7.5 * values["visibility"] * values["polarization_moment"],
                -values["temperature_quadrupole_derivative"],
            )
        elif name == "visibility_doppler":
            terms = (
                values["visibility"]
                * values["observable_theta_b"]
                / values["acoustic_k"],
                -values["temperature_doppler"],
            )
        elif name == "polarization":
            terms = (
                7.5 * values["visibility"] * values["polarization_moment"],
                -values["polarization_source"],
            )
        else:
            terms = (
                numpy.exp(-values["tau"])
                * (values["Phi_history_tau"] + values["Psi_tau"]),
                -values["temperature_isw"],
            )
        residual = float(numpy.sum(numpy.asarray(terms, dtype=float)))
        scale = max(float(numpy.sum(numpy.abs(terms))), 1.0e-30)
        return abs(residual), abs(residual) / scale

    for mode in raw_modes.values():
        if not isinstance(mode, Mapping):
            issues.append("Malformed source-history mode audit payload")
            continue
        mode_count += 1
        samples = mode.get("samples", ())
        if not isinstance(samples, Sequence):
            issues.append("Malformed source-history sample collection")
            continue
        for sample in samples:
            if not isinstance(sample, Mapping):
                issues.append("Malformed source-history sample")
                continue
            sample_count += 1
            for name, required in _SOURCE_RESIDUAL_DEFINITIONS.items():
                if not all(field_name in sample for field_name in required):
                    metrics[name]["available"] = False
                    metrics[name]["converged"] = False
                    continue
                try:
                    absolute, normalized = _residual(name, sample)
                except (ArithmeticError, TypeError, ValueError):
                    metrics[name]["available"] = False
                    metrics[name]["converged"] = False
                    continue
                metric = metrics[name]
                metric["sample_count"] += 1
                metric["maximum_absolute"] = max(
                    float(metric["maximum_absolute"]), absolute
                )
                metric["maximum_normalized"] = max(
                    float(metric["maximum_normalized"]), normalized
                )

    for name, metric in metrics.items():
        if not metric["available"]:
            issues.append(f"{name} source terms are incomplete")
        else:
            relative_pass = float(metric["maximum_normalized"]) <= float(
                metric["tolerance"]
            )
            absolute_pass = float(
                metric["absolute_tolerance"]
            ) > 0.0 and float(metric["maximum_absolute"]) <= float(
                metric["absolute_tolerance"]
            )
            metric["relative_converged"] = relative_pass
            metric["absolute_converged"] = absolute_pass
            if relative_pass:
                metric["convergence_basis"] = "relative"
            elif absolute_pass:
                metric["convergence_basis"] = "absolute"
            else:
                metric["convergence_basis"] = "none"
                metric["converged"] = False
                issues.append(
                    f"{name} residual exceeds relative tolerance "
                    f"{metric['tolerance']:.3g} and absolute tolerance "
                    f"{metric['absolute_tolerance']:.3g}"
                )
        if not metric["available"]:
            metric["converged"] = False
        elif metric["converged"] and not (
            bool(metric.get("relative_converged"))
            or bool(metric.get("absolute_converged"))
        ):
            metric["converged"] = False

    hierarchy_modes = runtime_envelope.get("hierarchy_equation_residuals_by_k")
    if hierarchy_modes is not None:
        hierarchy_metric = {
            "maximum_absolute": 0.0,
            "maximum_normalized": 0.0,
            "tolerance": hierarchy_relative_tolerance,
            "absolute_tolerance": hierarchy_absolute_tolerance,
            "criterion": "relative_or_absolute",
            "sample_count": 0,
            "equation_count": 0,
            "available": bool(hierarchy_modes),
            "converged": bool(hierarchy_modes),
        }
        if not isinstance(hierarchy_modes, Mapping) or not hierarchy_modes:
            hierarchy_metric["available"] = False
            hierarchy_metric["converged"] = False
            issues.append("hierarchy equation residuals are unavailable")
        else:
            for mode in hierarchy_modes.values():
                if not isinstance(mode, Mapping):
                    hierarchy_metric["available"] = False
                    hierarchy_metric["converged"] = False
                    continue
                hierarchy_metric["sample_count"] += int(
                    mode.get("sample_count", 0)
                )
                equations = mode.get("equations", {})
                if not isinstance(equations, Mapping):
                    hierarchy_metric["available"] = False
                    hierarchy_metric["converged"] = False
                    continue
                for equation in equations.values():
                    if not isinstance(equation, Mapping):
                        hierarchy_metric["available"] = False
                        hierarchy_metric["converged"] = False
                        continue
                    hierarchy_metric["equation_count"] += 1
                    hierarchy_metric["maximum_absolute"] = max(
                        float(hierarchy_metric["maximum_absolute"]),
                        float(equation.get("maximum_absolute", numpy.inf)),
                    )
                    hierarchy_metric["maximum_normalized"] = max(
                        float(hierarchy_metric["maximum_normalized"]),
                        float(equation.get("maximum_normalized", numpy.inf)),
                    )
            if (
                int(hierarchy_metric["sample_count"]) <= 0
                or int(hierarchy_metric["equation_count"]) <= 0
            ):
                hierarchy_metric["available"] = False
                hierarchy_metric["converged"] = False
            else:
                relative_pass = float(
                    hierarchy_metric["maximum_normalized"]
                ) <= float(hierarchy_metric["tolerance"])
                absolute_tolerance = float(
                    hierarchy_metric["absolute_tolerance"]
                )
                absolute_pass = (
                    absolute_tolerance > 0.0
                    and float(hierarchy_metric["maximum_absolute"])
                    <= absolute_tolerance
                )
                hierarchy_metric["relative_converged"] = relative_pass
                hierarchy_metric["absolute_converged"] = absolute_pass
                if relative_pass:
                    hierarchy_metric["convergence_basis"] = "relative"
                elif absolute_pass:
                    hierarchy_metric["convergence_basis"] = "absolute"
                else:
                    hierarchy_metric["convergence_basis"] = "none"
                    hierarchy_metric["converged"] = False
                    issues.append(
                        "hierarchy equation residual exceeds relative "
                        f"tolerance {hierarchy_metric['tolerance']:.3g} and "
                        "absolute tolerance "
                        f"{hierarchy_metric['absolute_tolerance']:.3g}"
                    )
            if not hierarchy_metric["converged"]:
                hierarchy_metric["converged"] = False
        metrics["hierarchy_equations"] = hierarchy_metric
    initial_modes = runtime_envelope.get("initial_state_diagnostics_by_k")
    if initial_modes is not None:
        initial_metric = {
            "maximum_absolute": 0.0,
            "maximum_normalized": 0.0,
            "tolerance": 1.0e-2,
            "absolute_tolerance": 1.0e-10,
            "criterion": "relative_or_absolute",
            "sample_count": 0,
            "available": bool(initial_modes),
            "converged": bool(initial_modes),
        }
        if not isinstance(initial_modes, Mapping) or not initial_modes:
            initial_metric["available"] = False
            initial_metric["converged"] = False
            issues.append("initial-condition diagnostics are unavailable")
        else:
            for mode in initial_modes.values():
                if not isinstance(mode, Mapping):
                    initial_metric["available"] = False
                    initial_metric["converged"] = False
                    continue
                constraints = mode.get("constraint_diagnostics", {})
                if not isinstance(constraints, Mapping):
                    initial_metric["available"] = False
                    initial_metric["converged"] = False
                    continue
                for constraint in constraints.values():
                    if not isinstance(constraint, Mapping):
                        initial_metric["available"] = False
                        initial_metric["converged"] = False
                        continue
                    initial_metric["sample_count"] += 1
                    initial_metric["maximum_absolute"] = max(
                        float(initial_metric["maximum_absolute"]),
                        abs(
                            float(
                                constraint.get("absolute_residual", numpy.inf)
                            )
                        ),
                    )
                    initial_metric["maximum_normalized"] = max(
                        float(initial_metric["maximum_normalized"]),
                        abs(
                            float(
                                constraint.get(
                                    "normalized_residual", numpy.inf
                                )
                            )
                        ),
                    )
            if int(initial_metric["sample_count"]) <= 0:
                initial_metric["available"] = False
                initial_metric["converged"] = False
            else:
                relative_pass = float(
                    initial_metric["maximum_normalized"]
                ) <= float(initial_metric["tolerance"])
                absolute_pass = float(
                    initial_metric["maximum_absolute"]
                ) <= float(initial_metric["absolute_tolerance"])
                initial_metric["relative_converged"] = relative_pass
                initial_metric["absolute_converged"] = absolute_pass
                if relative_pass:
                    initial_metric["convergence_basis"] = "relative"
                elif absolute_pass:
                    initial_metric["convergence_basis"] = "absolute"
                else:
                    initial_metric["convergence_basis"] = "none"
                    initial_metric["converged"] = False
                    issues.append(
                        "initial-condition residual exceeds relative "
                        f"tolerance {initial_metric['tolerance']:.3g} and "
                        "absolute tolerance "
                        f"{initial_metric['absolute_tolerance']:.3g}"
                    )
        metrics["initial_conditions"] = initial_metric
    available = any(
        bool(metric["available"]) and int(metric["sample_count"]) > 0
        for metric in metrics.values()
    )
    return {
        "schema_version": 1,
        "independent_recomputation": True,
        "available": available,
        "mode_count": mode_count,
        "sample_count": sample_count,
        "metrics": metrics,
        "controls": controls,
        "converged": bool(
            mode_count > 0
            and sample_count > 0
            and not issues
            and all(metric["converged"] for metric in metrics.values())
        ),
        "issues": tuple(issues),
    }


def _jsonable(value: Any) -> Any:
    """Return one deterministic JSON-compatible diagnostic value."""

    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, CMBError):
        return value.diagnostic()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]
    return value


def _diagnostic_failure(error: BaseException) -> dict[str, Any]:
    """Return a stable failure payload without hiding diagnostic context."""

    if isinstance(error, CMBError):
        return error.diagnostic()
    return {
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _bound_contract(
    contract: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a contract while replacing declared numerical controls only."""

    bound = dict(contract)
    numerical = dict(bound.get("numerical", {}) or {})
    for name, value in (overrides or {}).items():
        if isinstance(value, bool):
            raise ValueError(
                f"Diagnostic numerical override {name} is boolean"
            )
        numerical[str(name)] = value
    bound["numerical"] = numerical
    return bound


def _relative_error(reference: Any, refined: Any) -> float:
    """Return a scale-safe maximum relative difference between arrays."""

    reference_array = numpy.asarray(reference, dtype=float)
    refined_array = numpy.asarray(refined, dtype=float)
    if reference_array.shape != refined_array.shape:
        return float("inf")
    scale = max(float(numpy.max(numpy.abs(refined_array))), 1.0e-30)
    return float(numpy.max(numpy.abs(refined_array - reference_array)) / scale)


@dataclass(frozen=True, slots=True)
class CMBModelDiagnostic:
    """Raw fixed-point CCMBS evidence for one bundled CMB model."""

    model_filename: str
    model_name: str
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    requested_ells: tuple[int, ...]
    requested_spectra: tuple[str, ...]
    spectra: Mapping[str, numpy.ndarray] = field(default_factory=dict)
    raw_spectra: Mapping[str, numpy.ndarray] = field(default_factory=dict)
    raw_transfer_components: Mapping[str, numpy.ndarray] = field(
        default_factory=dict
    )
    runtime_envelope: Mapping[str, Any] = field(default_factory=dict)
    refinement: Mapping[str, Any] = field(default_factory=dict)
    shape: Mapping[str, Any] = field(default_factory=dict)
    source_residual_audit: Mapping[str, Any] = field(default_factory=dict)
    reference_comparison: Mapping[str, Any] = field(default_factory=dict)
    failure: Mapping[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Return whether the fixed-point request produced finite spectra."""

        return self.failure is None

    def to_dict(self) -> dict[str, Any]:
        """Return complete serializable evidence, including raw arrays."""

        return {
            "failure": _jsonable(self.failure),
            "model_filename": self.model_filename,
            "model_name": self.model_name,
            "parameter_names": self.parameter_names,
            "parameter_values": self.parameter_values,
            "raw_spectra": _jsonable(self.raw_spectra),
            "raw_transfer_components": _jsonable(self.raw_transfer_components),
            "refinement": _jsonable(self.refinement),
            "reference_comparison": _jsonable(self.reference_comparison),
            "source_residual_audit": _jsonable(self.source_residual_audit),
            "shape": _jsonable(self.shape),
            "requested_ells": self.requested_ells,
            "requested_spectra": self.requested_spectra,
            "runtime_envelope": _jsonable(self.runtime_envelope),
            "spectra": _jsonable(self.spectra),
            "success": self.success,
        }


def _certification_evidence_status(
    report: CMBModelDiagnostic,
    *,
    required_spectra: Sequence[str],
    require_reference: bool,
) -> tuple[bool, tuple[str, ...]]:
    """Check one report against the final scientific evidence contract."""

    issues: list[str] = []
    required = tuple(str(name).upper() for name in required_spectra)
    missing_spectra = sorted(set(required) - set(report.spectra))
    if missing_spectra:
        issues.append("missing public spectra: " + ", ".join(missing_spectra))
    missing_raw = sorted(set(required) - set(report.raw_spectra))
    if missing_raw:
        issues.append("missing raw spectra: " + ", ".join(missing_raw))
    if report.failure is not None:
        issues.append(
            "solver failure: "
            + str(report.failure.get("error_type", "unknown"))
        )
    shape = report.shape
    if not bool(shape.get("finite", False)):
        issues.append("spectrum shape is non-finite")
    if not bool(shape.get("auto_spectra_nonnegative", False)):
        issues.append("auto spectrum contains negative power")
    if shape.get("smooth") is False:
        issues.append("TT shape contains unresolved quadrature structure")
    refinement = report.refinement
    if refinement.get("converged") is not True:
        issues.append("doubled-grid convergence is unavailable or failed")
    residuals = report.source_residual_audit
    if not bool(residuals.get("available", False)):
        issues.append("independent source residual audit is unavailable")
    elif not bool(residuals.get("converged", False)):
        issues.append("independent source residual audit failed")
    reference = report.reference_comparison
    if require_reference and not bool(reference.get("available", False)):
        issues.append("independent reference comparison is unavailable")
    elif require_reference and not bool(reference.get("converged", False)):
        issues.append("independent reference comparison failed")
    return not issues, tuple(issues)


def build_cmb_certification_report(
    reports: Iterable[CMBModelDiagnostic],
    *,
    required_model_filenames: Iterable[str] | None = None,
    required_spectra: Sequence[str] = _DEFAULT_SPECTRA,
    require_reference: bool = True,
) -> dict[str, Any]:
    """Build a deterministic final certification record from raw reports.

    The record is deliberately strict: unavailable source evidence and an
    absent independent comparison are rejected rather than interpreted as a
    successful smoke test.  Raw arrays remain nested under each report so the
    record can be audited without recreating the solver run.
    """

    ordered_reports = tuple(
        sorted(reports, key=lambda item: str(item.model_filename))
    )
    expected = tuple(
        sorted(
            str(name)
            for name in (
                required_model_filenames
                if required_model_filenames is not None
                else (item.model_filename for item in ordered_reports)
            )
        )
    )
    required = tuple(str(name).upper() for name in required_spectra)
    seen = {str(item.model_filename) for item in ordered_reports}
    missing_models = sorted(set(expected) - seen)
    model_records: list[dict[str, Any]] = []
    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    for report in ordered_reports:
        valid, issues = _certification_evidence_status(
            report,
            required_spectra=required,
            require_reference=require_reference,
        )
        model_name = str(report.model_filename)
        model_records.append(
            {
                "model_filename": model_name,
                "accepted": valid,
                "issues": list(issues),
                "report": report.to_dict(),
            }
        )
        if valid:
            accepted.append(model_name)
        else:
            rejected[model_name] = list(issues)
    for model_name in missing_models:
        rejected[model_name] = ["model is missing from diagnostic matrix"]
    complete = not missing_models and set(seen) == set(expected)
    success = complete and len(accepted) == len(expected) and not rejected
    record: dict[str, Any] = {
        "schema_version": 1,
        "required_spectra": list(required),
        "required_models": list(expected),
        "complete": complete,
        "success": success,
        "accepted_models": sorted(accepted),
        "rejected_models": {name: rejected[name] for name in sorted(rejected)},
        "reports": model_records,
    }
    canonical = json.dumps(
        _jsonable(record), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    return record


def write_cmb_certification_report(
    reports: Iterable[CMBModelDiagnostic],
    destination: str | Path,
    *,
    required_model_filenames: Iterable[str] | None = None,
    required_spectra: Sequence[str] = _DEFAULT_SPECTRA,
    require_reference: bool = True,
) -> dict[str, Any]:
    """Serialize a certification record and return the written payload."""

    record = build_cmb_certification_report(
        reports,
        required_model_filenames=required_model_filenames,
        required_spectra=required_spectra,
        require_reference=require_reference,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def run_cmb_model_diagnostic(
    plugin: Any,
    *,
    ells: Iterable[int] = _DEFAULT_ELL_VALUES,
    spectra: Sequence[str] = _DEFAULT_SPECTRA,
    numerical_overrides: Mapping[str, Any] | None = None,
    refine_wave_number_grid: bool = True,
    relative_tolerances: Mapping[str, float] | None = None,
    reference_spectra: Mapping[str, Any] | None = None,
    reference_tolerances: Mapping[str, float] | None = None,
) -> CMBModelDiagnostic:
    """Collect raw CCMBS evidence for one fixed initial-guess parameter point.

    The report contains both unscaled transfer products and public spectra.
    When requested, the second run doubles ``k_sample_count`` and records the
    component-wise relative difference; it never converts a failed or
    non-converged result into a successful report.
    """

    requested_ells = tuple(int(value) for value in ells)
    requested_spectra = tuple(str(value) for value in spectra)
    if not requested_ells:
        raise ValueError("Diagnostic ell grid must not be empty")
    if not requested_spectra:
        raise ValueError("Diagnostic spectra must not be empty")
    if not getattr(plugin, "valid_for_cmb", False):
        raise ValueError("Diagnostic plugin must declare valid_for_cmb")

    model_filename = str(
        getattr(plugin, "MODEL_FILENAME", None)
        or f"{getattr(plugin, 'MODEL_NAME', 'model')}.yml"
    )
    model_name = str(getattr(plugin, "MODEL_NAME", model_filename))
    parameter_names = tuple(
        str(value) for value in getattr(plugin, "PARAMETER_NAMES", ())
    )
    parameter_values = tuple(
        float(value) for value in getattr(plugin, "INITIAL_GUESSES", ())
    )
    tolerances = dict(_DEFAULT_RELATIVE_TOLERANCES)
    tolerances.update(relative_tolerances or {})

    base_contract = plugin.get_cmb_declared_runtime(parameter_values)
    base = _bound_contract(base_contract, numerical_overrides)
    try:
        raw_data = _compute_custom_cmb_spectrum_data(
            base,
            requested_ells,
            requested_spectra=requested_spectra,
            workload="fixed_parameter_diagnostic",
        )
        # The raw CCMBS result contains dimensionless C_ell products. Apply
        # the same deterministic public normalization as the solver without
        # evolving every mode a second time.
        raw_spectra = raw_data.spectra
        if not isinstance(raw_spectra, Mapping):
            raw_spectra = {requested_spectra[0]: raw_spectra}
        public_spectra = _public_spectrum_values(
            requested_ells,
            raw_spectra,
        )
        refinement: dict[str, Any] = {
            "axis": "k_sample_count",
            "base_count": int(raw_data.k_grid.size),
            "declared_base_count": int(
                base["numerical"].get("k_sample_count", 0)
            ),
            "refined_count": None,
            "declared_refined_count": None,
            "metrics": {},
            "converged": None,
        }
        if refine_wave_number_grid:
            base_k_count = int(base["numerical"].get("k_sample_count", 0))
            if base_k_count < 1:
                raise ValueError("Diagnostic k_sample_count must be positive")
            refined = _bound_contract(
                base,
                {"k_sample_count": base_k_count * 2},
            )
            refined_raw_data = _compute_custom_cmb_spectrum_data(
                refined,
                requested_ells,
                requested_spectra=requested_spectra,
                workload="fixed_parameter_diagnostic_refinement",
            )
            refined_raw_spectra = refined_raw_data.spectra
            if not isinstance(refined_raw_spectra, Mapping):
                refined_raw_spectra = {
                    requested_spectra[0]: refined_raw_spectra
                }
            refined_spectra = _public_spectrum_values(
                requested_ells,
                refined_raw_spectra,
            )
            metrics = {}
            for name in requested_spectra:
                error = _relative_error(
                    public_spectra[name], refined_spectra[name]
                )
                tolerance = float(tolerances.get(name, 1.0e-2))
                metrics[name] = {
                    "relative_error": error,
                    "tolerance": tolerance,
                    "converged": bool(
                        numpy.isfinite(error) and error <= tolerance
                    ),
                }
            refinement["refined_count"] = int(refined_raw_data.k_grid.size)
            refinement["declared_refined_count"] = base_k_count * 2
            refinement["metrics"] = metrics
            refinement["converged"] = bool(
                metrics and all(item["converged"] for item in metrics.values())
            )
        failure = None
        if refine_wave_number_grid and not refinement["converged"]:
            failure = _diagnostic_failure(
                ConvergenceError(
                    "Fixed-parameter CCMBS spectrum did not converge under "
                    "the doubled k-grid",
                    context={
                        "base_k_sample_count": refinement["base_count"],
                        "refined_k_sample_count": refinement["refined_count"],
                        "metrics": refinement["metrics"],
                        "model_name": model_name,
                    },
                )
            )
        shape = assess_physical_spectrum_shape(requested_ells, public_spectra)
        source_residual_audit = audit_source_history_residuals(
            raw_data.runtime_envelope
        )
        generated_scalar = bool(
            raw_data.runtime_envelope.get("generated_scalar_hierarchy", False)
        )
        reference_comparison: Mapping[str, Any] = {
            "available": False,
            "converged": True,
            "metrics": {},
        }
        if reference_spectra is not None:
            reference_comparison = compare_cmb_spectra_to_reference(
                public_spectra,
                reference_spectra,
                relative_tolerances=reference_tolerances,
            )
        if (
            failure is None
            and reference_spectra is not None
            and not reference_comparison["converged"]
        ):
            failure = {
                "error_type": "reference_mismatch",
                "message": (
                    "Fixed-parameter CCMBS spectra failed the independent "
                    "reference comparison"
                ),
                "reference_comparison": _jsonable(reference_comparison),
            }
        if (
            failure is None
            and generated_scalar
            and not bool(source_residual_audit.get("converged", False))
        ):
            failure = {
                "error_type": "source_residual_failure",
                "message": (
                    "Generated CCMBS source histories failed the independent "
                    "physical residual audit"
                ),
                "source_residual_audit": _jsonable(source_residual_audit),
            }
        # Source-closure metrics are recorded with this runtime report.  The
        # generated-hierarchy slice owns their scientific acceptance; keeping
        # them diagnostic here lets Slice One certify raw projection without
        # silently discarding the failed residual evidence.
        if failure is None and shape["issues"]:
            failure = {
                "error_type": "spectrum_shape_failure",
                "message": "; ".join(shape["issues"]),
                "shape": _jsonable(shape),
            }
        return CMBModelDiagnostic(
            model_filename=model_filename,
            model_name=model_name,
            parameter_names=parameter_names,
            parameter_values=parameter_values,
            requested_ells=requested_ells,
            requested_spectra=requested_spectra,
            spectra={
                str(name): numpy.asarray(values, dtype=float)
                for name, values in public_spectra.items()
            },
            raw_spectra={
                str(name): numpy.asarray(values, dtype=float)
                for name, values in raw_data.spectra.items()
            },
            raw_transfer_components={
                str(name): numpy.asarray(values, dtype=float)
                for name, values in raw_data.transfer_components.items()
            },
            runtime_envelope=dict(raw_data.runtime_envelope),
            refinement=refinement,
            shape=shape,
            source_residual_audit=source_residual_audit,
            reference_comparison=reference_comparison,
            failure=failure,
        )
    except (
        ArithmeticError,
        AttributeError,
        CMBError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return CMBModelDiagnostic(
            model_filename=model_filename,
            model_name=model_name,
            parameter_names=parameter_names,
            parameter_values=parameter_values,
            requested_ells=requested_ells,
            requested_spectra=requested_spectra,
            failure=_diagnostic_failure(error),
        )


def discover_bundled_cmb_plugins(
    model_directory: str | Path | None = None,
) -> tuple[Any, ...]:
    """Build every bundled model plugin whose contract enables CMB support."""

    package_root = Path(__file__).resolve().parents[3]
    models_path = Path(model_directory or package_root / "models")
    plugins: list[Any] = []
    for model_path in sorted(models_path.glob("model_*.yml")):
        with tempfile.TemporaryDirectory(prefix="ccmbs-model-") as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                model_path,
                cache_dir,
            )
            functions, model_data = model_coder.generate_callables(cache_path)
        if not bool(model_data.get("valid_for_cmb", False)):
            continue
        plugin = model_adapter.build_plugin(model_data, functions)
        plugin.MODEL_FILENAME = model_path.name
        plugins.append(plugin)
    return tuple(plugins)


def run_bundled_cmb_diagnostics(
    *,
    model_directory: str | Path | None = None,
    ells: Iterable[int] = _DEFAULT_ELL_VALUES,
    spectra: Sequence[str] = _DEFAULT_SPECTRA,
    numerical_overrides: Mapping[str, Any] | None = None,
    refine_wave_number_grid: bool = True,
    reference_spectra_by_model: Mapping[str, Mapping[str, Any]] | None = None,
    reference_tolerances_by_model: (
        Mapping[str, Mapping[str, float]] | None
    ) = None,
) -> tuple[CMBModelDiagnostic, ...]:
    """Collect fixed-point raw evidence for every bundled CMB model.

    ``reference_spectra_by_model`` is intentionally keyed by the model
    filename.  This prevents an independent LCDM fixture from being reused
    for another theory and preserves an explicit unavailable state when a
    model has no scientifically justified reference.
    """

    requested_ells = tuple(int(value) for value in ells)
    requested_spectra = tuple(str(value) for value in spectra)
    reports = []
    for plugin in discover_bundled_cmb_plugins(model_directory):
        supported = set(
            audit_cmb_capabilities(
                plugin.CMB_PERTURBATION_DATA
            ).supported_observables
        )
        unsupported = sorted(set(requested_spectra) - supported)
        if unsupported:
            reports.append(
                CMBModelDiagnostic(
                    model_filename=str(plugin.MODEL_FILENAME),
                    model_name=str(plugin.MODEL_NAME),
                    parameter_names=tuple(plugin.PARAMETER_NAMES),
                    parameter_values=tuple(plugin.INITIAL_GUESSES),
                    requested_ells=requested_ells,
                    requested_spectra=requested_spectra,
                    failure={
                        "error_type": "UnsupportedCapabilityError",
                        "message": (
                            "Bundled model lacks requested spectra: "
                            + ", ".join(unsupported)
                        ),
                    },
                )
            )
            continue
        reports.append(
            run_cmb_model_diagnostic(
                plugin,
                ells=requested_ells,
                spectra=requested_spectra,
                numerical_overrides=numerical_overrides,
                refine_wave_number_grid=refine_wave_number_grid,
                reference_spectra=(reference_spectra_by_model or {}).get(
                    str(plugin.MODEL_FILENAME)
                ),
                reference_tolerances=(reference_tolerances_by_model or {}).get(
                    str(plugin.MODEL_FILENAME)
                ),
            )
        )
    return tuple(reports)


__all__ = [
    "CMBModelDiagnostic",
    "assess_physical_spectrum_shape",
    "audit_source_history_residuals",
    "build_cmb_certification_report",
    "compare_cmb_spectra_to_reference",
    "discover_bundled_cmb_plugins",
    "run_bundled_cmb_diagnostics",
    "run_cmb_model_diagnostic",
    "write_cmb_certification_report",
]
