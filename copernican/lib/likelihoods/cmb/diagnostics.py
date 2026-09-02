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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy

from ... import model_adapter, model_coder, model_spec_validator
from ...cmb_contract import audit_cmb_capabilities
from ...cmb_output import canonical_cmb_spectrum_name, describe_cmb_spectrum
from .contracts_audit import (
    audit_bundled_cmb_contracts,
    audit_bundled_cmb_declarations,
    audit_bundled_cmb_source_graphs,
)
from .errors import CMBError, ConvergenceError
from .runtime import cache
from .runtime.projection import _compute_custom_cmb_spectrum_data

_DEFAULT_SPECTRA = ("TT", "TE", "EE")
_DEFAULT_ELL_VALUES = (2, 20, 100)
_DEFAULT_RELATIVE_TOLERANCES = {
    "TT": 1.0e-2,
    "TE": 2.0e-2,
    "EE": 1.0e-2,
}

# The bundled corpus is a scientific input, not an incidental directory
# listing.  Keeping its filename order frozen makes missing, duplicate, or
# newly introduced model rows visible in every matrix digest.
BUNDLED_CMB_MODEL_FILENAMES = (
    "model_lcdm.yml",
    "model_lcdm_mnu.yml",
    "model_qauc.yml",
    "model_qrsf.yml",
    "model_ref_planck2018.yml",
    "model_tog.yml",
    "model_torg.yml",
    "model_usmf2.yml",
    "model_w0wa.yml",
    "model_wcdm.yml",
)

# This is deliberately a named, reproducible request.  It is a bounded
# evidence tier for the matrix and is never substituted for a model's
# production numerical declaration.
CMB_CERTIFICATION_TIER = {
    "id": "ccmbs-slice-six-final-certification-v1",
    # The first acoustic feature is near ell=200.  A certification surface
    # that stops below it can only test finiteness, not the physical acoustic
    # structure required by the matrix acceptance contract.
    "ells": tuple(range(2, 301)),
    "spectra": _DEFAULT_SPECTRA,
    "refine_wave_number_grid": True,
    "numerical_overrides": {"k_sample_count": 1024},
}

_FINAL_CERTIFICATION_INTEGRITY_KEYS = (
    "no_camb_fallback",
    "no_surrogate_spectra",
    "no_delayed_acceptance",
    "no_hidden_aliases",
    "no_arbitrary_timeout",
    "no_unchecked_declaration_bridge",
    "no_machine_local_paths",
    "raw_evidence_used",
)

# The baseline deliberately predates final scientific certification.  It is
# one named, fixed, direct request that records the corpus' pre-repair state,
# including rejected and unfinished outcomes, without turning either into a
# passing result.  ``eta_sample_count`` is explicit because a k-only label
# cannot describe the source history used to make the projected spectra.
CMB_CORPUS_BASELINE_REQUEST = {
    "schema_version": 1,
    "id": "ccmbs-corpus-baseline-v1",
    "parameter_source": "model_initial_guesses",
    "ells": tuple(range(2, 301)),
    "spectra": _DEFAULT_SPECTRA,
    "numerical_overrides": {
        "k_sample_count": 1024,
        "eta_sample_count": 192,
    },
    "source_anchor_policy": "quartiles-plus-visibility-peak-v1",
    "refinement": {
        "axis": "k_sample_count",
        "factor": 2,
        "required": True,
    },
}

# USMF2 is run through finite, node-count-defined tiers.  The progression has
# no time-based exit: a caller that provides only a prefix receives an honest
# ``unclassified`` record with the remaining declared work, never a timeout
# relabelled as model unavailability.
CMB_USMF2_BASELINE_TIERS = (
    {
        "id": "usmf2-baseline-probe-v1",
        "numerical_overrides": {
            "k_sample_count": 64,
            "eta_sample_count": 192,
        },
        "refine_wave_number_grid": False,
    },
    {
        "id": "usmf2-baseline-intermediate-v1",
        "numerical_overrides": {
            "k_sample_count": 256,
            "eta_sample_count": 192,
        },
        "refine_wave_number_grid": False,
    },
    {
        "id": "usmf2-baseline-request-v1",
        "numerical_overrides": {
            "k_sample_count": 1024,
            "eta_sample_count": 192,
        },
        "refine_wave_number_grid": True,
    },
)

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
            else (
                ell_factor
                * numpy.sqrt(2.0 * numpy.pi * ell_factor)
                * numpy.longdouble("2.7255e6")
                if token in {"TP", "EP"}
                else (
                    2.0 * numpy.longdouble(numpy.pi) * ell_factor * ell_factor
                    if token == "PP"
                    else ell_factor
                )
            )
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


def assess_acoustic_structure(
    ell_values: Iterable[int],
    spectra: Mapping[str, Any],
    *,
    minimum_peak_prominence: float = 1.0e-3,
) -> dict[str, Any]:
    """Extract deterministic acoustic-shape evidence from public spectra.

    The projection runtime can produce finite numbers while still aliasing
    the radial Bessel phase.  This audit therefore records the observable
    features that a reference comparison needs: TT peak and trough ordering,
    a high-to-low damping ratio, TE zero crossings and signs, and EE peak
    locations.  It never smooths, interpolates, or rescales the input arrays;
    sparse requests are reported as incomplete evidence instead of being
    promoted to a successful shape decision.
    """

    ells = numpy.asarray(tuple(ell_values), dtype=int)
    if ells.ndim != 1 or ells.size == 0 or numpy.any(numpy.diff(ells) <= 0):
        raise ValueError("Acoustic-shape ell values must be increasing")
    result: dict[str, Any] = {
        "schema_version": 1,
        "available": False,
        "peak_ordered": False,
        "damping_ratio": None,
        "tt": {},
        "te": {},
        "ee": {},
        "issues": [],
    }
    issues: list[str] = result["issues"]
    if ells.size < 16:
        issues.append("acoustic structure requires at least 16 multipoles")
        result["issues"] = tuple(issues)
        return result

    def _finite_values(name: str) -> numpy.ndarray | None:
        """Return one finite spectrum with the requested ell shape."""

        if name not in spectra:
            issues.append(f"{name} spectrum is unavailable")
            return None
        values = numpy.asarray(spectra[name], dtype=float)
        if values.shape != ells.shape or not numpy.all(numpy.isfinite(values)):
            issues.append(f"{name} spectrum is non-finite or has wrong shape")
            return None
        return values

    def _extrema(values: numpy.ndarray, *, maximum: bool) -> list[int]:
        """Return separated, materially prominent extrema.

        A raw one-multipole scan mistakes integration noise for acoustic
        structure.  The audit therefore ranks local candidates by their
        immediate prominence and keeps only one candidate in each physical
        separation window.  This is a selection rule on the supplied samples,
        not a smoothing or interpolation step, so a jagged quadrature result
        cannot pass merely because it was visually smoothed first.
        """

        scale = max(float(numpy.max(numpy.abs(values))), 1.0e-30)
        dynamic_range = float(numpy.max(values) - numpy.min(values))
        prominence = max(
            float(minimum_peak_prominence) * scale,
            0.02 * dynamic_range,
            1.0e-30,
        )
        candidates: list[tuple[float, int]] = []
        for index in range(1, values.size - 1):
            left, current, right = values[index - 1 : index + 2]
            candidate = (
                current > left and current >= right
                if maximum
                else current < left and current <= right
            )
            if not candidate:
                continue
            local_prominence = (
                float(current - max(left, right))
                if maximum
                else float(min(left, right) - current)
            )
            if local_prominence >= prominence:
                candidates.append((local_prominence, index))
        if not candidates:
            return []
        minimum_separation = max(
            1,
            int(round(float(ells[-1] - ells[0]) / 12.0)),
        )
        selected: list[int] = []
        for _, index in sorted(candidates, reverse=True):
            ell = int(ells[index])
            if all(
                abs(ell - int(ells[other])) >= minimum_separation
                for other in selected
            ):
                selected.append(index)
        return [int(ells[index]) for index in sorted(selected)]

    tt_spectrum = _finite_values("TT")
    if tt_spectrum is not None:
        peaks = _extrema(tt_spectrum, maximum=True)
        troughs = _extrema(tt_spectrum, maximum=False)
        positive = numpy.maximum(tt_spectrum, numpy.finfo(float).tiny)
        split = max(2, positive.size // 3)
        high = float(numpy.median(positive[-split:]))
        acoustic_start = max(
            int(ells[0]) + max(1, int(round((ells[-1] - ells[0]) / 12.0))),
            20,
        )
        acoustic_peak_indices = numpy.flatnonzero(
            numpy.isin(ells, numpy.asarray(peaks, dtype=int))
            & (ells >= acoustic_start)
        )
        if acoustic_peak_indices.size:
            # Use the strongest resolved acoustic crest as the envelope
            # anchor.  The first few multipoles contain the Sachs--Wolfe
            # plateau and are not part of the damping-tail comparison.
            anchor = float(numpy.max(positive[acoustic_peak_indices]))
        else:
            anchor = float(numpy.max(positive))
        damping_ratio = high / max(anchor, numpy.finfo(float).tiny)
        result["tt"] = {
            "peak_ells": tuple(peaks),
            "trough_ells": tuple(troughs),
            "peak_count": len(peaks),
            "trough_count": len(troughs),
        }
        result["damping_ratio"] = damping_ratio
        relevant_peaks = [peak for peak in peaks if peak >= acoustic_start]
        relevant_troughs = [
            trough for trough in troughs if trough >= acoustic_start
        ]
        first_interval_has_trough = bool(
            len(relevant_peaks) >= 2
            and any(
                relevant_peaks[0] < trough < relevant_peaks[1]
                for trough in relevant_troughs
            )
        )
        result["peak_ordered"] = bool(
            len(relevant_peaks) >= 2
            and all(
                left < right
                for left, right in zip(relevant_peaks, relevant_peaks[1:])
            )
            and first_interval_has_trough
        )
        if not result["peak_ordered"]:
            issues.append("TT acoustic peaks are missing or unordered")
        if not numpy.isfinite(damping_ratio) or damping_ratio >= 1.0:
            issues.append("TT damping tail does not decrease")

    te_spectrum = _finite_values("TE")
    if te_spectrum is not None:
        signs = numpy.signbit(te_spectrum).astype(int)
        crossings = numpy.flatnonzero(numpy.diff(signs) != 0)
        result["te"] = {
            "zero_crossing_ells": tuple(
                int(ells[index]) for index in crossings
            ),
            "sign_change_count": int(crossings.size),
            "finite": True,
        }
        if crossings.size == 0:
            issues.append("TE has no resolved sign changes")

    ee_spectrum = _finite_values("EE")
    if ee_spectrum is not None:
        ee_peaks = _extrema(ee_spectrum, maximum=True)
        result["ee"] = {
            "peak_ells": tuple(ee_peaks),
            "peak_count": len(ee_peaks),
            "finite": True,
        }
        if not ee_peaks:
            issues.append("EE has no resolved acoustic peaks")

    result["available"] = not issues
    result["issues"] = tuple(issues)
    return result


def compare_cmb_spectra_to_reference(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    relative_tolerances: Mapping[str, float] | None = None,
    auto_spectrum_floor: float = 1.0e-10,
    representation: str = "D_ell",
) -> dict[str, Any]:
    """Compare raw public spectra with an independent fixed-point reference.

    Auto spectra use a p90 fractional error above a relative reference floor;
    cross spectra use an RMS error normalized by the reference RMS so sign
    changes remain well-defined.  The function is backend-neutral: tests may
    pass CAMB/CLASS data while the production package remains reference-solver
    free.
    """

    selected_representation = str(representation).upper()
    if selected_representation not in _FULL_PARITY_REPRESENTATIONS:
        raise ValueError("representation must be C_ell or D_ell")
    reference_payload = reference.get("spectra", reference)
    if not isinstance(reference_payload, Mapping):
        raise TypeError("Reference spectra must be a mapping")
    normalized_reference: dict[str, Any] = {}
    for name, value in reference_payload.items():
        if isinstance(value, Mapping):
            candidate = value.get(
                "C_ell" if selected_representation == "C_ELL" else "D_ell"
            )
            if candidate is None:
                raise ValueError(
                    f"Reference spectrum '{name}' lacks "
                    f"{representation} values"
                )
            normalized_reference[str(name)] = candidate
        else:
            normalized_reference[str(name)] = value
    reference = normalized_reference
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


_FULL_PARITY_AUTO_SPECTRA = frozenset(
    {
        "TT",
        "EE",
        "BB",
        "PP",
        "LENSED_TT",
        "LENSED_EE",
        "LENSED_BB",
    }
)
_FULL_PARITY_SECTORS = frozenset({"scalar", "vector", "tensor", "total"})
_FULL_PARITY_REPRESENTATIONS = frozenset({"C_ELL", "D_ELL"})


def _canonical_parity_surface(sector: str, name: str) -> str:
    """Return a stable ``sector:observable`` parity row identifier."""

    canonical = canonical_cmb_spectrum_name(name)
    metadata = describe_cmb_spectrum(canonical)
    component = metadata.component
    selected_sector = str(sector or "scalar").casefold()
    if component in _FULL_PARITY_SECTORS:
        if selected_sector == "scalar" or selected_sector == "total":
            selected_sector = str(component)
        prefix = f"{component}_"
        if canonical.casefold().startswith(prefix):
            canonical = canonical[len(prefix) :]
    if selected_sector not in _FULL_PARITY_SECTORS:
        raise ValueError(f"Unsupported parity sector '{selected_sector}'")
    return f"{selected_sector}:{canonical}"


def _flatten_parity_surfaces(
    payload: Mapping[str, Any],
    *,
    default_sector: str = "scalar",
) -> dict[str, tuple[str, Any]]:
    """Flatten direct, sector-nested, and fixture ``spectra`` payloads."""

    if not isinstance(payload, Mapping):
        raise TypeError("Parity spectra must be supplied as a mapping")
    root_sector = str(payload.get("sector", default_sector))
    source: Mapping[str, Any] = payload
    nested = payload.get("spectra")
    if isinstance(nested, Mapping):
        source = nested
    flattened: dict[str, tuple[str, Any]] = {}
    for raw_name, value in source.items():
        name = str(raw_name)
        if name in {"sector", "spectra", "ell_values", "declared_observables"}:
            continue
        if name.casefold() in _FULL_PARITY_SECTORS and isinstance(
            value, Mapping
        ):
            for nested_name, nested_value in value.items():
                row_name = _canonical_parity_surface(name, str(nested_name))
                if row_name in flattened:
                    raise ValueError(f"Duplicate parity surface '{row_name}'")
                flattened[row_name] = (str(nested_name), nested_value)
            continue
        row_name = _canonical_parity_surface(root_sector, name)
        if row_name in flattened:
            raise ValueError(f"Duplicate parity surface '{row_name}'")
        flattened[row_name] = (name, value)
    if not flattened:
        raise ValueError("Parity spectra payload contains no observables")
    return flattened


def _parity_representation_values(
    entry: Any,
    *,
    representation: str,
) -> tuple[numpy.ndarray, str, Mapping[str, Any] | None]:
    """Select one representation without silently changing its units."""

    selected = str(representation).upper()
    if selected not in _FULL_PARITY_REPRESENTATIONS:
        raise ValueError("representation must be C_ell or D_ell")
    if not isinstance(entry, Mapping):
        return numpy.asarray(entry, dtype=numpy.longdouble), selected, None
    normalized = {str(key).upper(): value for key, value in entry.items()}
    selected_value = normalized.get(selected)
    if selected_value is None:
        if "VALUES" in normalized:
            selected_value = normalized["VALUES"]
        else:
            raise ValueError(
                f"Parity entry has no declared {selected} representation"
            )
    return (
        numpy.asarray(selected_value, dtype=numpy.longdouble),
        selected,
        normalized,
    )


def _parity_conversion_factor(
    observable: str,
    ell_values: numpy.ndarray,
) -> numpy.ndarray:
    """Return the CAMB C-to-D factor for one observable family."""

    base_name = describe_cmb_spectrum(observable).base_spectrum
    ell_product = ell_values * (ell_values + 1.0)
    if base_name in {"TT", "TE", "EE", "BB"}:
        return ell_product / (2.0 * numpy.longdouble(numpy.pi))
    if base_name == "PP":
        return numpy.square(ell_product) / (2.0 * numpy.longdouble(numpy.pi))
    if base_name in {"TP", "EP"}:
        return numpy.power(ell_product, 1.5) / (
            2.0 * numpy.longdouble(numpy.pi)
        )
    raise ValueError(
        f"No C_ell/D_ell convention is declared for '{observable}'"
    )


def _parity_row_shape(
    observable: str,
    values: numpy.ndarray,
    ell_values: numpy.ndarray,
) -> dict[str, Any]:
    """Record finite, sign, and resolved-structure evidence for one row."""

    finite = bool(
        values.ndim == 1
        and values.shape == ell_values.shape
        and numpy.all(numpy.isfinite(values))
    )
    base_name = describe_cmb_spectrum(observable).base_spectrum
    auto = base_name in {"TT", "EE", "BB", "PP"}
    nonnegative = bool(
        not auto or not finite or numpy.min(values) >= numpy.longdouble(0.0)
    )
    scale = max(float(numpy.max(numpy.abs(values), initial=0.0)), 1.0e-30)
    sign_threshold = numpy.longdouble("1.0e-3") * scale
    supported = finite & (numpy.abs(values) > sign_threshold)
    sign_changes = int(
        numpy.count_nonzero(
            numpy.diff(numpy.signbit(values[supported]).astype(int))
        )
        if numpy.count_nonzero(supported) > 1
        else 0
    )
    return {
        "finite": finite,
        "auto": auto,
        "nonnegative": nonnegative,
        "sign_change_count": sign_changes,
        "maximum_abs": float(numpy.max(numpy.abs(values), initial=0.0)),
        "ell_min": int(ell_values[0]),
        "ell_max": int(ell_values[-1]),
    }


def compare_full_cmb_observable_parity(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    ell_values: Iterable[int] | None = None,
    representation: str = "D_ell",
    relative_tolerances: Mapping[str, float] | None = None,
    auto_spectrum_floor: float = 1.0e-10,
    zero_tolerance: float = 1.0e-12,
    refinement: Mapping[str, Any] | None = None,
    fixture_digest: str | None = None,
    require_refinement: bool = True,
    require_fixture_digest: bool = False,
) -> dict[str, Any]:
    """Compare every raw sector/observable row without interpolation.

    The input may be a flat spectrum mapping, a sector mapping, or the
    structured CAMB fixture used by the scientific tests.  Every expected
    row is retained, including missing and unexpected rows.  If both
    ``C_ell`` and ``D_ell`` are supplied, their declared conversion is
    checked before the selected representation is compared.
    """

    selected_token = str(representation).upper()
    if selected_token not in _FULL_PARITY_REPRESENTATIONS:
        raise ValueError("representation must be C_ell or D_ell")
    selected_representation = "C_ell" if selected_token == "C_ELL" else "D_ell"
    actual_rows = _flatten_parity_surfaces(actual)
    reference_rows = _flatten_parity_surfaces(reference)
    reference_ells = reference.get("ell_values")
    actual_ells = actual.get("ell_values")
    if ell_values is None:
        ell_values = (
            reference_ells if reference_ells is not None else actual_ells
        )
    if ell_values is None:
        raise ValueError("Full parity comparison requires ell_values")
    ell_array = numpy.asarray(tuple(ell_values), dtype=int)
    if (
        ell_array.ndim != 1
        or ell_array.size == 0
        or numpy.any(numpy.diff(ell_array) <= 0)
    ):
        raise ValueError("Parity ell_values must be strictly increasing")
    for label, candidate in (
        ("actual", actual_ells),
        ("reference", reference_ells),
    ):
        if candidate is not None and not numpy.array_equal(
            numpy.asarray(tuple(candidate), dtype=int), ell_array
        ):
            raise ValueError(f"{label} ell_values do not match parity grid")

    tolerances = dict(_DEFAULT_RELATIVE_TOLERANCES)
    tolerances.update(
        {
            "BB": 0.02,
            "PP": 0.03,
            "TP": 0.05,
            "EP": 0.05,
            "LENSED_TT": 0.02,
            "LENSED_TE": 0.03,
            "LENSED_EE": 0.02,
            "LENSED_BB": 0.05,
        }
    )
    tolerances.update(
        {
            str(name).upper(): float(value)
            for name, value in (relative_tolerances or {}).items()
        }
    )
    rows: list[dict[str, Any]] = []
    all_names = sorted(set(reference_rows) | set(actual_rows))
    for row_name in all_names:
        actual_entry = actual_rows.get(row_name)
        reference_entry = reference_rows.get(row_name)
        sector, observable = row_name.split(":", 1)
        row: dict[str, Any] = {
            "sector": sector,
            "observable": observable,
            "row": row_name,
            "representation": selected_representation,
            "status": "rejected",
            "issues": [],
        }
        issues: list[str] = row["issues"]
        if actual_entry is None:
            issues.append("missing actual spectrum")
            rows.append(row)
            continue
        if reference_entry is None:
            issues.append("unexpected actual spectrum")
            rows.append(row)
            continue
        try:
            actual_values, _, actual_mapping = _parity_representation_values(
                actual_entry[1], representation=selected_representation
            )
            reference_values, _, reference_mapping = (
                _parity_representation_values(
                    reference_entry[1], representation=selected_representation
                )
            )
        except (TypeError, ValueError) as exc:
            issues.append(str(exc))
            rows.append(row)
            continue
        if actual_values.shape != ell_array.shape:
            issues.append("actual spectrum has the wrong shape")
        if reference_values.shape != ell_array.shape:
            issues.append("reference spectrum has the wrong shape")
        if issues:
            row["actual"] = _jsonable(actual_values)
            row["reference"] = _jsonable(reference_values)
            rows.append(row)
            continue
        actual_shape = _parity_row_shape(observable, actual_values, ell_array)
        reference_shape = _parity_row_shape(
            observable, reference_values, ell_array
        )
        row["actual_shape"] = actual_shape
        row["reference_shape"] = reference_shape
        row["actual"] = _jsonable(actual_values)
        row["reference"] = _jsonable(reference_values)
        if not actual_shape["finite"] or not reference_shape["finite"]:
            issues.append("spectrum contains non-finite values")
        if not actual_shape["nonnegative"]:
            issues.append("actual auto-spectrum contains negative power")
        if not reference_shape["nonnegative"]:
            issues.append("reference auto-spectrum contains negative power")
        for mapping_name, mapping in (
            ("actual", actual_mapping),
            ("reference", reference_mapping),
        ):
            if mapping is not None:
                if "C_ELL" not in mapping or "D_ELL" not in mapping:
                    continue
                c_values = numpy.asarray(
                    mapping["C_ELL"], dtype=numpy.longdouble
                )
                d_values = numpy.asarray(
                    mapping["D_ELL"], dtype=numpy.longdouble
                )
                factor = _parity_conversion_factor(observable, ell_array)
                if (
                    c_values.shape != ell_array.shape
                    or d_values.shape != ell_array.shape
                ):
                    issues.append(f"{mapping_name} C_ell/D_ell shapes differ")
                    continue
                expected_d = c_values * factor
                conversion_scale = max(
                    float(numpy.max(numpy.abs(d_values), initial=0.0)),
                    1.0e-30,
                )
                conversion_error = float(
                    numpy.max(numpy.abs(expected_d - d_values), initial=0.0)
                    / conversion_scale
                )
                row.setdefault("representation_consistency", {})[
                    mapping_name
                ] = {
                    "maximum_relative_error": conversion_error,
                    "converged": bool(conversion_error <= 1.0e-10),
                }
                if conversion_error > 1.0e-10:
                    issues.append(
                        f"{mapping_name} C_ell/D_ell conversion is "
                        "inconsistent"
                    )
        name_key = observable.upper()
        tolerance = float(
            tolerances.get(
                f"{sector.upper()}:{name_key}",
                tolerances.get(name_key, 1.0e-2),
            )
        )
        reference_scale = numpy.max(
            numpy.abs(reference_values), initial=numpy.longdouble(0.0)
        )
        if (
            name_key in _FULL_PARITY_AUTO_SPECTRA
            and reference_scale <= numpy.longdouble("1.0e-30")
        ):
            maximum_error = float(
                numpy.max(numpy.abs(actual_values), initial=0.0)
            )
            metric = {
                "kind": "zero_auto",
                "max_absolute": maximum_error,
                "tolerance": float(zero_tolerance),
                "converged": bool(maximum_error <= float(zero_tolerance)),
            }
        elif name_key in _FULL_PARITY_AUTO_SPECTRA:
            floor = max(
                numpy.longdouble("1.0e-30"),
                numpy.longdouble(auto_spectrum_floor) * reference_scale,
            )
            supported = numpy.abs(reference_values) > floor
            fractional = numpy.abs(
                (actual_values[supported] - reference_values[supported])
                / reference_values[supported]
            )
            p90 = float(numpy.percentile(fractional, 90.0))
            metric = {
                "kind": "auto",
                "median_fractional": float(numpy.median(fractional)),
                "p90_fractional": p90,
                "max_fractional": float(numpy.max(fractional, initial=0.0)),
                "tolerance": tolerance,
                "converged": bool(p90 <= tolerance),
            }
        else:
            delta = actual_values - reference_values
            reference_rms = numpy.sqrt(
                numpy.mean(numpy.square(reference_values))
            )
            normalized_rms = float(
                numpy.sqrt(numpy.mean(numpy.square(delta)))
                / max(reference_rms, numpy.longdouble("1.0e-30"))
            )
            sign_floor = max(
                numpy.longdouble("1.0e-30"),
                numpy.longdouble(auto_spectrum_floor) * reference_scale,
            )
            supported = numpy.abs(reference_values) > sign_floor
            sign_mismatches = int(
                numpy.count_nonzero(
                    numpy.signbit(actual_values[supported])
                    != numpy.signbit(reference_values[supported])
                )
            )
            metric = {
                "kind": "cross",
                "normalized_rms": normalized_rms,
                "sign_mismatch_count": sign_mismatches,
                "tolerance": tolerance,
                "converged": bool(
                    normalized_rms <= tolerance and sign_mismatches == 0
                ),
            }
        row["metric"] = metric
        if not metric["converged"]:
            issues.append("raw-array parity tolerance failed")
        row_refinement = refinement
        if isinstance(refinement, Mapping) and row_name in refinement:
            candidate = refinement[row_name]
            row_refinement = (
                candidate if isinstance(candidate, Mapping) else {}
            )
        row["refinement"] = _jsonable(row_refinement or {})
        if require_refinement and not bool(
            isinstance(row_refinement, Mapping)
            and row_refinement.get("converged") is True
        ):
            issues.append(
                "independent base/refined convergence is unavailable"
            )
        row["status"] = "accepted" if not issues else "rejected"
        rows.append(row)
    digest = str(fixture_digest or "")
    digest_valid = bool(
        digest
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )
    issues = []
    if set(reference_rows) - set(actual_rows):
        issues.append("one or more reference observables are missing")
    if set(actual_rows) - set(reference_rows):
        issues.append("unexpected observables were supplied")
    if require_fixture_digest and not digest_valid:
        issues.append("fixture digest is missing or malformed")
    if not rows:
        issues.append("no parity rows were compared")
    accepted = not issues and all(row["status"] == "accepted" for row in rows)
    report = {
        "schema_version": 1,
        "representation": selected_representation,
        "ell_values": tuple(int(value) for value in ell_array),
        "rows": rows,
        "row_count": len(rows),
        "fixture_digest": digest or None,
        "fixture_digest_valid": digest_valid,
        "refinement_required": bool(require_refinement),
        "issues": tuple(issues),
        "converged": bool(accepted),
        "accepted": bool(accepted),
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def build_cmb_parity_report(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    parameter_points: Mapping[str, Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a canonical full-observable report with response points."""

    report = compare_full_cmb_observable_parity(actual, reference, **kwargs)
    points: dict[str, Any] = {}
    for label, point in (parameter_points or {}).items():
        if not isinstance(point, Mapping):
            raise TypeError(f"Parameter point '{label}' must be a mapping")
        if "actual" not in point or "reference" not in point:
            raise ValueError(
                f"Parameter point '{label}' requires actual and reference"
            )
        point_kwargs = dict(kwargs)
        point_kwargs["require_fixture_digest"] = False
        point_kwargs["fixture_digest"] = point.get("fixture_digest")
        if "ell_values" in point:
            point_kwargs["ell_values"] = point["ell_values"]
        if "refinement" in point:
            point_kwargs["refinement"] = point["refinement"]
        points[str(label)] = compare_full_cmb_observable_parity(
            point["actual"], point["reference"], **point_kwargs
        )
    report["parameter_points"] = points
    report["parameter_point_count"] = len(points) + 1
    report["response_points_converged"] = all(
        bool(point["accepted"]) for point in points.values()
    )
    report["accepted"] = bool(
        report["accepted"] and report["response_points_converged"]
    )
    report["converged"] = report["accepted"]
    report["report_sha256"] = _canonical_sha256(report)
    return report


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
        "polarization_moment",
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
                * (
                    values["observable_theta_gamma0"]
                    + values["Psi"]
                    + 0.25 * values["polarization_moment"]
                ),
                -values["temperature_monopole"],
            )
        elif name == "visibility_quadrupole":
            terms = (
                0.0,
                -values["temperature_quadrupole"],
            )
        elif name == "visibility_quadrupole_derivative":
            terms = (
                0.0,
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
                0.75 * values["visibility"] * values["polarization_moment"],
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
        # ``ndarray.tolist`` preserves extended-precision scalar classes
        # (notably ``longdouble``), which the standard JSON encoder cannot
        # serialize.  Recurse through the resulting containers so every
        # scalar is normalized at the same boundary.
        return _jsonable(value.tolist())
    if isinstance(value, numpy.generic):
        scalar = value.item()
        if isinstance(scalar, numpy.generic):
            if isinstance(value, numpy.complexfloating):
                return complex(value)
            return float(value)
        return scalar
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


def _contract_identity(plugin: Any) -> dict[str, Any]:
    """Return a stable identity for one compiled model contract."""

    perturbation_data = getattr(plugin, "CMB_PERTURBATION_DATA", None)
    runtime_signature = str(
        getattr(perturbation_data, "runtime_signature", "")
        or getattr(plugin, "CMB_RUNTIME_SIGNATURE", "")
    )
    get_runtime = getattr(plugin, "get_cmb_declared_runtime", None)
    if not runtime_signature and callable(get_runtime):
        try:
            runtime = get_runtime(getattr(plugin, "INITIAL_GUESSES", ()))
            runtime_signature = str(runtime.get("runtime_signature", ""))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            runtime_signature = ""
    contract = {
        "model_filename": str(getattr(plugin, "MODEL_FILENAME", "")),
        "model_name": str(getattr(plugin, "MODEL_NAME", "")),
        "contract_version": int(
            getattr(perturbation_data, "contract_version", 0) or 0
        ),
        "gauge": str(getattr(perturbation_data, "gauge", "")),
        "runtime_signature": runtime_signature,
        "sectors": sorted(
            str(name)
            for name in (getattr(perturbation_data, "sectors", {}) or {})
        ),
        "spectra": sorted(
            str(name).upper()
            for name in (getattr(perturbation_data, "observables", {}) or {})
            if str(
                getattr(
                    (getattr(perturbation_data, "observables", {}) or {}).get(
                        name
                    ),
                    "kind",
                    "",
                )
            ).lower()
            == "angular_power_spectrum"
        ),
    }
    canonical = json.dumps(
        _jsonable(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    contract["sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def declared_cmb_spectrum_names(plugin_or_contract: Any) -> tuple[str, ...]:
    """Return every angular spectrum explicitly declared by a CMB graph.

    The corpus matrix must derive its request from the model contract rather
    than from a solver-wide default.  Transfer components are intentionally
    excluded, while component and lensed names are preserved through the
    canonical output-name normalizer.
    """

    perturbation_data = getattr(
        plugin_or_contract, "CMB_PERTURBATION_DATA", plugin_or_contract
    )
    observables = getattr(perturbation_data, "observables", {}) or {}
    names = {
        canonical_cmb_spectrum_name(name)
        for name, entry in observables.items()
        if str(getattr(entry, "kind", "")).casefold()
        == "angular_power_spectrum"
    }
    return tuple(sorted(names))


def _unavailable_report(
    plugin: Any,
    *,
    requested_ells: tuple[int, ...],
    requested_spectra: tuple[str, ...],
    error_type: str,
    message: str,
    category: str = "unavailable",
) -> "CMBModelDiagnostic":
    """Create an explicit matrix row when a fixed point was not measured."""

    return CMBModelDiagnostic(
        model_filename=str(getattr(plugin, "MODEL_FILENAME", "<unknown>")),
        model_name=str(getattr(plugin, "MODEL_NAME", "<unknown>")),
        parameter_names=tuple(
            str(value) for value in getattr(plugin, "PARAMETER_NAMES", ())
        ),
        parameter_values=tuple(
            float(value) for value in getattr(plugin, "INITIAL_GUESSES", ())
        ),
        requested_ells=requested_ells,
        requested_spectra=requested_spectra,
        availability="unavailable",
        contract_identity=_contract_identity(plugin),
        failure={
            "error_type": error_type,
            "category": category,
            "message": message,
        },
    )


def _cache_identity_payload() -> dict[str, Any]:
    """Return a path-free, stable record of the latest CCMBS cache key.

    Runtime cache keys may contain compiled callables and large frozen
    contracts, so serializing their ``repr`` would both leak implementation
    details and make reports needlessly large.  The payload hashes each
    semantic key component and preserves the selected solver identity.  It is
    therefore sufficient to prove which direct base/refined requests were
    distinct without exposing a machine-local representation.
    """

    identity = cache.latest_cmb_request_identity()
    if identity is None:
        return {
            "available": False,
            "identity_schema": "ccmbs-runtime-cache-identity-sha256-v1",
            "reason": "CCMBS did not publish a spectrum cache identity",
        }
    components = {
        "contract_static": repr(identity.contract_static),
        "model_static": repr(identity.model_static),
        "request_specific": repr(identity.request_specific),
        "execution_solver": str(identity.execution_solver),
    }
    payload = {
        "available": True,
        "identity_schema": "ccmbs-runtime-cache-identity-sha256-v1",
        "contract_static_sha256": hashlib.sha256(
            components["contract_static"].encode("utf-8")
        ).hexdigest(),
        "model_static_sha256": hashlib.sha256(
            components["model_static"].encode("utf-8")
        ).hexdigest(),
        "request_specific_sha256": hashlib.sha256(
            components["request_specific"].encode("utf-8")
        ).hexdigest(),
        "execution_solver": components["execution_solver"],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


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
    acoustic_structure: Mapping[str, Any] = field(default_factory=dict)
    source_residual_audit: Mapping[str, Any] = field(default_factory=dict)
    reference_comparison: Mapping[str, Any] = field(default_factory=dict)
    availability: str = "measured"
    contract_identity: Mapping[str, Any] = field(default_factory=dict)
    cache_identity: Mapping[str, Any] = field(default_factory=dict)
    scalar_batch_evidence: Mapping[str, Any] = field(default_factory=dict)
    cache_isolation_evidence: Mapping[str, Any] = field(default_factory=dict)
    failure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate the explicit measured, rejected, or unavailable state."""

        availability = str(self.availability).lower()
        if availability not in {"measured", "unavailable", "rejected"}:
            raise ValueError(
                "Diagnostic availability must be measured, unavailable, "
                "or rejected"
            )
        object.__setattr__(self, "availability", availability)

    @property
    def success(self) -> bool:
        """Return whether the fixed-point request produced finite spectra."""

        return self.failure is None

    def to_dict(self) -> dict[str, Any]:
        """Return complete serializable evidence, including raw arrays."""

        return {
            "availability": self.availability,
            "cache_identity": _jsonable(self.cache_identity),
            "cache_isolation_evidence": _jsonable(
                self.cache_isolation_evidence
            ),
            "contract_identity": _jsonable(self.contract_identity),
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
            "acoustic_structure": _jsonable(self.acoustic_structure),
            "requested_ells": self.requested_ells,
            "requested_spectra": self.requested_spectra,
            "runtime_envelope": _jsonable(self.runtime_envelope),
            "scalar_batch_evidence": _jsonable(self.scalar_batch_evidence),
            "spectra": _jsonable(self.spectra),
            "success": self.success,
        }


def _canonical_sha256(payload: Any) -> str:
    """Return the canonical SHA-256 digest for JSON diagnostic evidence."""

    canonical = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _infer_solver_identities(
    reports: Sequence[CMBModelDiagnostic],
) -> dict[str, Any]:
    """Collect path-free solver identities from diagnostic envelopes."""

    identities: dict[str, str] = {}
    for report in reports:
        envelope = report.runtime_envelope
        solver_id = envelope.get("solver_id")
        if not solver_id:
            provenance = envelope.get("solver_provenance", {})
            if isinstance(provenance, Mapping):
                solver_id = provenance.get("solver_id")
        if solver_id:
            identities[str(report.model_filename)] = str(solver_id)
    return {
        "models": {name: identities[name] for name in sorted(identities)},
        "unique": sorted(set(identities.values())),
    }


def _positive_node_count(value: Any, *, field_name: str) -> int:
    """Normalize one explicit numerical node count for a baseline request."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive integer") from error
    if count < 1 or count != value:
        raise ValueError(f"{field_name} must be a positive integer")
    return count


def _normalize_corpus_baseline_request(
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the versioned direct request shared by the frozen corpus."""

    supplied = dict(request or {})
    unknown_fields = sorted(set(supplied) - set(CMB_CORPUS_BASELINE_REQUEST))
    if unknown_fields:
        raise ValueError(
            "Corpus baseline request has unknown fields: "
            + ", ".join(unknown_fields)
        )
    for field_name, expected_fields in (
        ("numerical_overrides", {"k_sample_count", "eta_sample_count"}),
        ("refinement", {"axis", "factor", "required"}),
    ):
        supplied_fields = supplied.get(field_name, {}) or {}
        if not isinstance(supplied_fields, Mapping):
            raise ValueError(f"Corpus baseline {field_name} must be a mapping")
        unexpected_fields = sorted(set(supplied_fields) - expected_fields)
        if unexpected_fields:
            raise ValueError(
                f"Corpus baseline {field_name} has unknown fields: "
                + ", ".join(unexpected_fields)
            )
    normalized = dict(CMB_CORPUS_BASELINE_REQUEST)
    normalized.update(supplied)
    numerical = dict(CMB_CORPUS_BASELINE_REQUEST["numerical_overrides"])
    numerical.update(supplied.get("numerical_overrides", {}) or {})
    refinement = dict(CMB_CORPUS_BASELINE_REQUEST["refinement"])
    refinement.update(supplied.get("refinement", {}) or {})
    requested_ells = tuple(int(value) for value in normalized["ells"])
    if (
        not requested_ells
        or requested_ells[0] < 2
        or any(
            later <= earlier
            for earlier, later in zip(requested_ells, requested_ells[1:])
        )
    ):
        raise ValueError(
            "Corpus baseline multipoles must be increasing integers from "
            "ell=2"
        )
    requested_spectra = tuple(
        str(value).upper() for value in normalized["spectra"]
    )
    if requested_spectra != _DEFAULT_SPECTRA:
        raise ValueError(
            "Corpus baseline must request ordered TT, TE, and EE spectra"
        )
    if int(normalized.get("schema_version", 0)) != 1:
        raise ValueError("Corpus baseline request schema_version must be 1")
    if not str(normalized.get("id", "")).strip():
        raise ValueError("Corpus baseline request must define a versioned id")
    if normalized.get("parameter_source") != "model_initial_guesses":
        raise ValueError(
            "Corpus baseline parameters must use model_initial_guesses"
        )
    if (
        normalized.get("source_anchor_policy")
        != "quartiles-plus-visibility-peak-v1"
    ):
        raise ValueError(
            "Corpus baseline must use the declared source-anchor policy"
        )
    if refinement.get("axis") != "k_sample_count":
        raise ValueError("Corpus baseline refinement must use k_sample_count")
    if (
        _positive_node_count(
            refinement.get("factor"), field_name="refinement.factor"
        )
        != 2
    ):
        raise ValueError("Corpus baseline refinement factor must equal 2")
    if refinement.get("required") is not True:
        raise ValueError(
            "Corpus baseline must require doubled-grid refinement"
        )
    normalized_numerical = {
        "k_sample_count": _positive_node_count(
            numerical.get("k_sample_count"),
            field_name="numerical_overrides.k_sample_count",
        ),
        "eta_sample_count": _positive_node_count(
            numerical.get("eta_sample_count"),
            field_name="numerical_overrides.eta_sample_count",
        ),
    }
    result = {
        "schema_version": 1,
        "id": str(normalized["id"]),
        "parameter_source": "model_initial_guesses",
        "ells": requested_ells,
        "spectra": requested_spectra,
        "numerical_overrides": normalized_numerical,
        "source_anchor_policy": "quartiles-plus-visibility-peak-v1",
        "refinement": {
            "axis": "k_sample_count",
            "factor": 2,
            "required": True,
        },
    }
    frozen = {
        "schema_version": 1,
        "id": str(CMB_CORPUS_BASELINE_REQUEST["id"]),
        "parameter_source": "model_initial_guesses",
        "ells": tuple(CMB_CORPUS_BASELINE_REQUEST["ells"]),
        "spectra": tuple(CMB_CORPUS_BASELINE_REQUEST["spectra"]),
        "numerical_overrides": {
            "k_sample_count": int(
                CMB_CORPUS_BASELINE_REQUEST["numerical_overrides"][
                    "k_sample_count"
                ]
            ),
            "eta_sample_count": int(
                CMB_CORPUS_BASELINE_REQUEST["numerical_overrides"][
                    "eta_sample_count"
                ]
            ),
        },
        "source_anchor_policy": "quartiles-plus-visibility-peak-v1",
        "refinement": {
            "axis": "k_sample_count",
            "factor": 2,
            "required": True,
        },
    }
    if result != frozen:
        raise ValueError(
            "Corpus baseline request is fixed; use the named v1 request"
        )
    return result


def _normalize_usmf2_baseline_tiers(
    tiers: Iterable[Mapping[str, Any]] | None,
    *,
    baseline_request: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate finite USMF2 work tiers without introducing time limits."""

    canonical_tiers = tuple(CMB_USMF2_BASELINE_TIERS)
    values = tuple(canonical_tiers if tiers is None else tuple(tiers))
    if not values:
        raise ValueError("USMF2 baseline progression must contain a tier")
    if len(values) > len(canonical_tiers):
        raise ValueError("USMF2 baseline progression exceeds the named tiers")
    identifiers: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, value in enumerate(values, start=1):
        if not isinstance(value, Mapping):
            raise ValueError("USMF2 baseline tiers must be mappings")
        identifier = str(value.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("USMF2 baseline tier ids must be unique")
        canonical_tier = canonical_tiers[position - 1]
        if identifier != str(canonical_tier["id"]):
            raise ValueError(
                "USMF2 baseline tiers must be an ordered prefix of the "
                "named progression"
            )
        identifiers.add(identifier)
        overrides = dict(baseline_request["numerical_overrides"])
        overrides.update(value.get("numerical_overrides", {}) or {})
        normalized_tier = {
            "position": position,
            "id": identifier,
            "numerical_overrides": {
                "k_sample_count": _positive_node_count(
                    overrides.get("k_sample_count"),
                    field_name=("USMF2 " f"tier {identifier} k_sample_count"),
                ),
                "eta_sample_count": _positive_node_count(
                    overrides.get("eta_sample_count"),
                    field_name=(
                        "USMF2 " f"tier {identifier} eta_sample_count"
                    ),
                ),
            },
            "refine_wave_number_grid": bool(
                value.get("refine_wave_number_grid", False)
            ),
        }
        expected_overrides = dict(baseline_request["numerical_overrides"])
        expected_overrides.update(canonical_tier["numerical_overrides"])
        expected_refinement = bool(canonical_tier["refine_wave_number_grid"])
        if (
            normalized_tier["numerical_overrides"] != expected_overrides
            or normalized_tier["refine_wave_number_grid"]
            != expected_refinement
        ):
            raise ValueError(
                "USMF2 baseline tiers must retain the named node counts and "
                "refinement settings"
            )
        normalized.append(normalized_tier)
    return tuple(normalized)


def _baseline_projection_metadata(
    report: CMBModelDiagnostic,
) -> dict[str, Any]:
    """Extract the declared and effective pre-plot projection metadata."""

    envelope = report.runtime_envelope
    keys = (
        "configured_numerical_controls",
        "effective_numerical_controls",
        "declared_k_sample_count",
        "k_grid_actual_count",
        "dynamic_mode_count",
        "phase_aware_k_enabled",
        "phase_required_nodes",
        "phase_grid_status",
        "source_grid_count",
        "declared_source_history_convergence",
        "source_history_mode_count",
        "source_history_cache_hit_count",
        "source_history_cache_miss_count",
    )
    return {key: _jsonable(envelope[key]) for key in keys if key in envelope}


def _baseline_source_history_metadata(
    report: CMBModelDiagnostic,
) -> dict[str, Any]:
    """Index raw source-history samples retained by the nested report."""

    envelope = report.runtime_envelope
    histories = envelope.get("source_history_residual_samples_by_k", {})
    if not isinstance(histories, Mapping):
        histories = {}
    sample_count = 0
    for values in histories.values():
        if isinstance(values, Mapping):
            sample_count += int(values.get("sample_count", 0))
    return {
        "available": bool(histories),
        "raw_data_path": (
            "diagnostic.runtime_envelope."
            "source_history_residual_samples_by_k"
        ),
        "sample_schema": envelope.get("source_history_residual_sample_schema"),
        "mode_count": len(histories),
        "sample_count": sample_count,
        "bundle_digest": _jsonable(
            envelope.get("source_history_bundle_digest", {})
        ),
    }


def _baseline_work_estimate(
    baseline_request: Mapping[str, Any],
    *,
    report: CMBModelDiagnostic | None = None,
) -> dict[str, Any]:
    """Describe finite numerical work in grid products, never wall time."""

    controls = dict(baseline_request["numerical_overrides"])
    if report is not None:
        effective = report.runtime_envelope.get(
            "effective_numerical_controls", {}
        )
        if isinstance(effective, Mapping):
            controls.update(
                {
                    name: effective[name]
                    for name in ("k_sample_count", "eta_sample_count")
                    if name in effective
                }
            )
    k_count = _positive_node_count(
        controls["k_sample_count"], field_name="work k_sample_count"
    )
    eta_count = _positive_node_count(
        controls["eta_sample_count"], field_name="work eta_sample_count"
    )
    ell_count = len(tuple(baseline_request["ells"]))
    spectrum_count = len(tuple(baseline_request["spectra"]))
    source_anchor_count = 6
    refinement = dict(baseline_request["refinement"])

    def _work_for_nodes(nodes: int) -> dict[str, int]:
        """Return lower-bound grid products for one direct solve tier."""

        evolution_cells = nodes * eta_count
        projection_cells = nodes * ell_count * spectrum_count
        anchor_values = nodes * source_anchor_count
        return {
            "k_mode_count": nodes,
            "eta_nodes_per_mode": eta_count,
            "evolution_grid_cells": evolution_cells,
            "projection_surface_cells": projection_cells,
            "source_anchor_values": anchor_values,
            "lower_bound_work_units": (
                evolution_cells + projection_cells + anchor_values
            ),
        }

    base = _work_for_nodes(k_count)
    refined = None
    total = int(base["lower_bound_work_units"])
    if refinement["required"]:
        refined = _work_for_nodes(k_count * int(refinement["factor"]))
        total += int(refined["lower_bound_work_units"])
    return {
        "schema_version": 1,
        "unit": "grid_product_lower_bound",
        "not_a_wall_clock_estimate": True,
        "source_anchor_policy": baseline_request["source_anchor_policy"],
        "base": base,
        "refined": refined,
        "total_lower_bound_work_units": total,
    }


def _baseline_decision(report: CMBModelDiagnostic) -> str:
    """Map a completed direct diagnostic to one honest baseline decision."""

    if report.availability == "unavailable":
        return "unavailable"
    if report.failure is None and report.availability == "measured":
        return "accepted"
    return "rejected"


@dataclass(frozen=True, slots=True)
class CMBCorpusBaselineRow:
    """One canonical pre-repair CCMBS corpus measurement record.

    The nested diagnostic retains raw source histories and spectra.  The row
    adds corpus-level audits, request identity, bounded work metadata, and the
    explicit decision vocabulary needed to distinguish an unfinished USMF2
    request from a solver or model failure.
    """

    model_filename: str
    model_name: str
    decision: str
    diagnostic: CMBModelDiagnostic
    contract_audit: Mapping[str, Any] = field(default_factory=dict)
    source_graph_audit: Mapping[str, Any] = field(default_factory=dict)
    request_identity: Mapping[str, Any] = field(default_factory=dict)
    projection_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_history_metadata: Mapping[str, Any] = field(default_factory=dict)
    work_estimate: Mapping[str, Any] = field(default_factory=dict)
    completion_state: str = "completed"
    progression: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    decision_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep completion state and baseline decision vocabulary aligned."""

        decision = str(self.decision).lower()
        if decision not in {
            "accepted",
            "rejected",
            "unavailable",
            "unclassified",
        }:
            raise ValueError("Corpus baseline decision is invalid")
        completion_state = str(self.completion_state).lower()
        if completion_state not in {"completed", "incomplete"}:
            raise ValueError("Corpus baseline completion state is invalid")
        if (decision == "unclassified") != (completion_state == "incomplete"):
            raise ValueError(
                "Unclassified baseline rows must have incomplete execution"
            )
        if self.model_filename != self.diagnostic.model_filename:
            raise ValueError(
                "Corpus baseline row and diagnostic filenames must match"
            )
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "completion_state", completion_state)

    def to_dict(self) -> dict[str, Any]:
        """Return one hashable row with raw evidence nested exactly once."""

        record: dict[str, Any] = {
            "model_filename": self.model_filename,
            "model_name": self.model_name,
            "decision": self.decision,
            "completion_state": self.completion_state,
            "decision_context": _jsonable(self.decision_context),
            "request_identity": _jsonable(self.request_identity),
            "contract_audit": _jsonable(self.contract_audit),
            "source_graph_audit": _jsonable(self.source_graph_audit),
            "projection_metadata": _jsonable(self.projection_metadata),
            "source_history_metadata": _jsonable(self.source_history_metadata),
            "work_estimate": _jsonable(self.work_estimate),
            "progression": _jsonable(self.progression),
            "diagnostic": self.diagnostic.to_dict(),
        }
        record["row_sha256"] = _canonical_sha256(record)
        return record


def _baseline_evidence_issues(
    row: CMBCorpusBaselineRow,
    *,
    baseline_request: Mapping[str, Any],
) -> tuple[str, ...]:
    """Identify missing baseline evidence without reclassifying an outcome."""

    issues: list[str] = []
    report = row.diagnostic
    if row.decision in {"accepted", "rejected"}:
        required = set(baseline_request["spectra"])
        missing_raw = sorted(required - set(report.raw_spectra))
        missing_public = sorted(required - set(report.spectra))
        if missing_raw:
            issues.append("missing raw spectra: " + ", ".join(missing_raw))
        if missing_public:
            issues.append(
                "missing public spectra: " + ", ".join(missing_public)
            )
        if not report.raw_transfer_components:
            issues.append("missing raw transfer components")
        if not bool(row.source_history_metadata.get("available", False)):
            issues.append("missing raw source-history samples")
        if not report.source_residual_audit:
            issues.append("missing source residual vectors")
        metadata = row.projection_metadata
        if "configured_numerical_controls" not in metadata:
            issues.append("missing configured numerical controls")
        if "effective_numerical_controls" not in metadata:
            issues.append("missing effective numerical controls")
        base_identity = report.cache_identity.get("base", {})
        if not bool(base_identity.get("available", False)):
            issues.append("missing base cache identity")
        if baseline_request["refinement"]["required"]:
            refined_identity = report.cache_identity.get("refined", {})
            if not bool(refined_identity.get("available", False)):
                issues.append("missing refined cache identity")
    elif row.decision == "unavailable" and report.failure is None:
        issues.append("unavailable row has no typed failure")
    elif row.decision == "unclassified":
        remaining = row.decision_context.get("remaining_tiers", ())
        if not remaining:
            issues.append("unclassified row has no remaining work record")
    return tuple(issues)


def _baseline_row_order_key(
    row: CMBCorpusBaselineRow,
) -> tuple[str, str]:
    """Sort a row by filename and its canonical evidence digest."""

    return row.model_filename, str(row.to_dict()["row_sha256"])


def build_cmb_corpus_baseline_report(
    rows: Iterable[CMBCorpusBaselineRow],
    *,
    baseline_request: Mapping[str, Any] | None = None,
    required_model_filenames: Iterable[str] = BUNDLED_CMB_MODEL_FILENAMES,
    discovered_model_filenames: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the deterministic ten-row pre-repair CCMBS baseline report.

    Structural completeness means every frozen filename has one row.  It is
    deliberately independent from scientific acceptance: rejected,
    unavailable, and honestly unclassified rows remain part of the baseline
    instead of disappearing from the digest.
    """

    request = _normalize_corpus_baseline_request(baseline_request)
    expected_values = tuple(str(value) for value in required_model_filenames)
    expected_duplicates = tuple(
        sorted(
            name
            for name in set(expected_values)
            if expected_values.count(name) > 1
        )
    )
    expected = tuple(dict.fromkeys(expected_values))
    ordered_rows = tuple(sorted(rows, key=_baseline_row_order_key))
    seen_values = tuple(row.model_filename for row in ordered_rows)
    seen = set(seen_values)
    duplicate_models = tuple(
        sorted(name for name in seen if seen_values.count(name) > 1)
    )
    missing_models = tuple(sorted(set(expected) - seen))
    unexpected_models = tuple(sorted(seen - set(expected)))
    discovered = tuple(
        sorted(
            str(value)
            for value in (
                expected
                if discovered_model_filenames is None
                else discovered_model_filenames
            )
        )
    )
    discovery_missing = tuple(sorted(set(expected) - set(discovered)))
    discovery_unexpected = tuple(sorted(set(discovered) - set(expected)))
    records: list[dict[str, Any]] = []
    outcome_counts = {
        "accepted": 0,
        "rejected": 0,
        "unavailable": 0,
        "unclassified": 0,
    }
    for row in ordered_rows:
        record = row.to_dict()
        evidence_issues = _baseline_evidence_issues(
            row,
            baseline_request=request,
        )
        record["evidence_complete"] = not evidence_issues
        record["evidence_issues"] = list(evidence_issues)
        records.append(record)
        outcome_counts[row.decision] += 1
    complete = (
        not expected_duplicates
        and not duplicate_models
        and not missing_models
        and not unexpected_models
        and not discovery_missing
        and not discovery_unexpected
        and len(ordered_rows) == len(expected)
    )
    evidence_complete = bool(
        complete and all(record["evidence_complete"] for record in records)
    )
    decision_complete = bool(complete and not outcome_counts["unclassified"])
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ccmbs_corpus_pre_repair_baseline",
        "baseline_request": _jsonable(request),
        "baseline_request_sha256": _canonical_sha256(request),
        "required_model_filenames": list(expected),
        "discovered_model_filenames": list(discovered),
        "missing_models": list(missing_models),
        "unexpected_models": list(unexpected_models),
        "duplicate_models": list(duplicate_models),
        "required_model_duplicates": list(expected_duplicates),
        "discovery_missing_models": list(discovery_missing),
        "discovery_unexpected_models": list(discovery_unexpected),
        "complete": complete,
        "evidence_complete": evidence_complete,
        "decision_complete": decision_complete,
        "outcome_counts": outcome_counts,
        "accepted_models": [
            row.model_filename
            for row in ordered_rows
            if row.decision == "accepted"
        ],
        "rejected_models": [
            row.model_filename
            for row in ordered_rows
            if row.decision == "rejected"
        ],
        "unavailable_models": [
            row.model_filename
            for row in ordered_rows
            if row.decision == "unavailable"
        ],
        "unclassified_models": [
            row.model_filename
            for row in ordered_rows
            if row.decision == "unclassified"
        ],
        "rows": records,
    }
    record["record_sha256"] = _canonical_sha256(record)
    return record


def write_cmb_corpus_baseline_report(
    rows: Iterable[CMBCorpusBaselineRow],
    destination: str | Path,
    *,
    baseline_request: Mapping[str, Any] | None = None,
    required_model_filenames: Iterable[str] = BUNDLED_CMB_MODEL_FILENAMES,
    discovered_model_filenames: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write one canonical pre-repair corpus baseline JSON record."""

    record = build_cmb_corpus_baseline_report(
        rows,
        baseline_request=baseline_request,
        required_model_filenames=required_model_filenames,
        discovered_model_filenames=discovered_model_filenames,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


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
    acoustic = report.acoustic_structure
    if acoustic and not bool(acoustic.get("available", False)):
        issues.append("acoustic peak and phase evidence is unavailable")
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


def _matrix_evidence_status(
    report: CMBModelDiagnostic,
    *,
    required_spectra: Sequence[str],
    require_reference: bool,
    contract_audit: Mapping[str, Any] | None,
    source_graph_audit: Mapping[str, Any] | None,
    declaration_audit: Mapping[str, Any] | None,
) -> tuple[bool, tuple[str, ...]]:
    """Apply the complete bundled-model matrix acceptance contract."""

    accepted, base_issues = _certification_evidence_status(
        report,
        required_spectra=required_spectra,
        require_reference=require_reference,
    )
    issues = list(base_issues)
    if report.availability != "measured":
        issues.append(
            "fixed-point status is " f"{report.availability}, not measured"
        )
    if not report.contract_identity:
        issues.append("compiled contract identity is unavailable")
    elif not str(report.contract_identity.get("sha256", "")):
        issues.append("compiled contract identity has no digest")
    if contract_audit is None:
        issues.append("contract declaration audit is unavailable")
    elif not bool(contract_audit.get("valid", False)):
        issues.append("contract declaration audit failed")
    if source_graph_audit is None:
        issues.append("source graph audit is unavailable")
    elif not bool(source_graph_audit.get("valid", False)):
        issues.append("source graph audit failed")
    if declaration_audit is None:
        issues.append("declaration audit is unavailable")
    elif not bool(declaration_audit.get("valid", False)):
        issues.append("declaration audit failed")
    # Batch and cache parity are acceptance evidence for rows that reached a
    # measured fixed point.  A row rejected by the scalar solver must retain
    # its typed scientific reason without being misclassified as a second,
    # unrelated batch/cache failure; Slice Eight promotes only measured rows.
    measured = report.availability == "measured" and report.failure is None
    if measured:
        batch = report.scalar_batch_evidence
        if not bool(batch.get("available", False)):
            issues.append("scalar/batch equivalence evidence is unavailable")
        elif not bool(batch.get("converged", False)):
            issues.append("scalar/batch equivalence failed")
        cache_evidence = report.cache_isolation_evidence
        if not bool(cache_evidence.get("available", False)):
            issues.append("cache-isolation evidence is unavailable")
        elif not bool(cache_evidence.get("isolated", False)):
            issues.append("cache identities are not isolated")
    return accepted and not issues, tuple(issues)


def build_cmb_certification_report(
    reports: Iterable[CMBModelDiagnostic],
    *,
    required_model_filenames: Iterable[str] | None = None,
    required_spectra: Sequence[str] = _DEFAULT_SPECTRA,
    require_reference: bool = True,
    matrix_mode: bool = False,
    certification_tier: Mapping[str, Any] | None = None,
    contract_audits: Mapping[str, Mapping[str, Any]] | None = None,
    source_graph_audits: Mapping[str, Mapping[str, Any]] | None = None,
    declaration_audits: Mapping[str, Mapping[str, Any]] | None = None,
    solver_identity: Mapping[str, Any] | str | None = None,
    dataset_identities: Mapping[str, Any] | None = None,
    fixture_hashes: Mapping[str, Any] | None = None,
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
    expected_values = tuple(
        str(name)
        for name in (
            required_model_filenames
            if required_model_filenames is not None
            else (item.model_filename for item in ordered_reports)
        )
    )
    expected = tuple(sorted(set(expected_values)))
    expected_duplicates = sorted(
        name
        for name in set(expected_values)
        if expected_values.count(name) > 1
    )
    required_values = tuple(str(name).upper() for name in required_spectra)
    if len(set(required_values)) != len(required_values):
        raise ValueError("Certification spectra must be unique")
    required = required_values
    seen_values = tuple(str(item.model_filename) for item in ordered_reports)
    seen = set(seen_values)
    duplicate_models = sorted(
        name for name in seen if seen_values.count(name) > 1
    )
    missing_models = sorted(set(expected) - seen)
    unexpected_models = sorted(seen - set(expected))
    model_records: list[dict[str, Any]] = []
    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    for report in ordered_reports:
        audit = (contract_audits or {}).get(str(report.model_filename))
        graph_audit = (source_graph_audits or {}).get(
            str(report.model_filename)
        )
        declaration_audit = (declaration_audits or {}).get(
            str(report.model_filename)
        )
        if matrix_mode:
            valid, issues = _matrix_evidence_status(
                report,
                required_spectra=required,
                require_reference=require_reference,
                contract_audit=audit,
                source_graph_audit=graph_audit,
                declaration_audit=declaration_audit,
            )
        else:
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
                "availability": report.availability,
                "raw_evidence_sha256": _canonical_sha256(report.to_dict()),
                "contract_audit": _jsonable(audit),
                "declaration_audit": _jsonable(declaration_audit),
                "source_graph_audit": _jsonable(graph_audit),
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
    for model_name in unexpected_models:
        rejected[model_name] = ["model is not in the frozen CMB corpus"]
    for model_name in duplicate_models:
        rejected.setdefault(model_name, []).append(
            "model appears more than once in diagnostic matrix"
        )
    for model_name in expected_duplicates:
        rejected.setdefault(model_name, []).append(
            "required model list contains duplicate entries"
        )
    complete = (
        not missing_models
        and not duplicate_models
        and not expected_duplicates
        and seen == set(expected)
        and len(ordered_reports) == len(expected)
    )
    # Matrix completeness is a decision property, not a claim that every
    # model is scientifically accepted.  A complete report may legitimately
    # contain typed rejected/unavailable rows; those rows feed the later
    # repair/certification slice and are never counted in ``success``.
    decision_complete = (
        complete
        and all(
            bool(item["accepted"]) or bool(item["issues"])
            for item in model_records
        )
        and len(rejected) == len(expected) - len(accepted)
    )
    success = complete and len(accepted) == len(expected) and not rejected
    record: dict[str, Any] = {
        "schema_version": 1,
        "required_spectra": list(required),
        "required_models": list(expected),
        "certification_tier": _jsonable(certification_tier or {}),
        "matrix_mode": bool(matrix_mode),
        "complete": complete,
        "decision_complete": bool(decision_complete),
        "success": success,
        "accepted_models": sorted(accepted),
        "rejected_models": {name: rejected[name] for name in sorted(rejected)},
        "contract_audits": {
            name: _jsonable((contract_audits or {}).get(name))
            for name in sorted(contract_audits or {})
        },
        "declaration_audits": {
            name: _jsonable((declaration_audits or {}).get(name))
            for name in sorted(declaration_audits or {})
        },
        "source_graph_audits": {
            name: _jsonable((source_graph_audits or {}).get(name))
            for name in sorted(source_graph_audits or {})
        },
        "provenance": {
            "schema_version": 1,
            "solver_identity": _jsonable(
                solver_identity
                if solver_identity is not None
                else _infer_solver_identities(ordered_reports)
            ),
            "dataset_identities": _jsonable(dataset_identities or {}),
            "fixture_hashes": _jsonable(fixture_hashes or {}),
            "raw_evidence_digests": {
                str(item.model_filename): _canonical_sha256(item.to_dict())
                for item in ordered_reports
            },
        },
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
    matrix_mode: bool = False,
    certification_tier: Mapping[str, Any] | None = None,
    contract_audits: Mapping[str, Mapping[str, Any]] | None = None,
    source_graph_audits: Mapping[str, Mapping[str, Any]] | None = None,
    declaration_audits: Mapping[str, Mapping[str, Any]] | None = None,
    solver_identity: Mapping[str, Any] | str | None = None,
    dataset_identities: Mapping[str, Any] | None = None,
    fixture_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a certification record and return the written payload."""

    record = build_cmb_certification_report(
        reports,
        required_model_filenames=required_model_filenames,
        required_spectra=required_spectra,
        require_reference=require_reference,
        matrix_mode=matrix_mode,
        certification_tier=certification_tier,
        contract_audits=contract_audits,
        source_graph_audits=source_graph_audits,
        declaration_audits=declaration_audits,
        solver_identity=solver_identity,
        dataset_identities=dataset_identities,
        fixture_hashes=fixture_hashes,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def build_final_cmb_certification_report(
    reports: Iterable[CMBModelDiagnostic],
    *,
    required_model_filenames: Iterable[str] = BUNDLED_CMB_MODEL_FILENAMES,
    required_spectra: Sequence[str] = _DEFAULT_SPECTRA,
    reference_required_models: Iterable[str] = ("model_lcdm.yml",),
    certification_tier: Mapping[str, Any] | None = None,
    contract_audits: Mapping[str, Mapping[str, Any]] | None = None,
    source_graph_audits: Mapping[str, Mapping[str, Any]] | None = None,
    declaration_audits: Mapping[str, Mapping[str, Any]] | None = None,
    solver_identity: Mapping[str, Any] | str | None = None,
    dataset_identities: Mapping[str, Any] | None = None,
    fixture_hashes: Mapping[str, Any] | None = None,
    integrity_checks: Mapping[str, Any] | None = None,
    bao_isolation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the strict, final corpus decision and its provenance.

    The ordinary certification builder remains useful for intermediate
    evidence.  This boundary additionally requires the frozen corpus, a
    declared reference for selected models, provenance metadata, all runtime
    integrity checks, and an independent BAO result.  An explicitly disabled
    CMB model is reported as ``unavailable``; an enabled model with a failed
    run remains rejected.
    """

    ordered_reports = tuple(reports)
    required_models = tuple(str(name) for name in required_model_filenames)
    reference_models = tuple(str(name) for name in reference_required_models)
    base = build_cmb_certification_report(
        ordered_reports,
        required_model_filenames=required_models,
        required_spectra=required_spectra,
        require_reference=False,
        matrix_mode=True,
        certification_tier=certification_tier or CMB_CERTIFICATION_TIER,
        contract_audits=contract_audits,
        source_graph_audits=source_graph_audits,
        declaration_audits=declaration_audits,
        solver_identity=solver_identity,
        dataset_identities=dataset_identities,
        fixture_hashes=fixture_hashes,
    )
    report_by_name = {
        str(report.model_filename): report for report in ordered_reports
    }
    accepted = set(str(name) for name in base["accepted_models"])
    rejected = {
        str(name): list(issues)
        for name, issues in base["rejected_models"].items()
    }
    unavailable: set[str] = set()
    for name in sorted(set(required_models) & set(report_by_name)):
        report = report_by_name[name]
        row = next(
            item
            for item in base["reports"]
            if str(item["model_filename"]) == name
        )
        declaration = row.get("declaration_audit") or {}
        explicitly_unavailable = (
            report.availability == "unavailable"
            and str((report.failure or {}).get("category", ""))
            == "unavailable"
            and str(declaration.get("decision", "")) == "unavailable"
            and bool(declaration.get("valid", False))
        )
        if explicitly_unavailable:
            accepted.discard(name)
            rejected.pop(name, None)
            unavailable.add(name)

    for name in reference_models:
        report = report_by_name.get(name)
        comparison = report.reference_comparison if report else {}
        if report is None:
            rejected.setdefault(name, []).append(
                "required independent reference model is missing"
            )
        elif not bool(comparison.get("available", False)):
            accepted.discard(name)
            rejected.setdefault(name, []).append(
                "required independent reference comparison is unavailable"
            )
        elif not bool(comparison.get("converged", False)):
            accepted.discard(name)
            rejected.setdefault(name, []).append(
                "required independent reference comparison failed"
            )

    checks = {
        key: bool((integrity_checks or {}).get(key, False))
        for key in _FINAL_CERTIFICATION_INTEGRITY_KEYS
    }
    integrity_issues = [
        f"integrity check failed: {key}"
        for key, passed in checks.items()
        if not passed
    ]
    provenance = base.get("provenance", {})
    metadata_issues: list[str] = []
    if not provenance.get("solver_identity"):
        metadata_issues.append("solver identity is missing")
    if not provenance.get("dataset_identities"):
        metadata_issues.append("dataset identities are missing")
    if not provenance.get("fixture_hashes"):
        metadata_issues.append("independent fixture hashes are missing")
    bao_evidence = dict(bao_isolation or {})
    bao_passed = bool(
        bao_evidence.get("available", False)
        and bao_evidence.get("converged", False)
    )
    if not bao_passed:
        metadata_issues.append("BAO CMB-isolation evidence is unavailable")

    global_issues = integrity_issues + metadata_issues
    final_success = bool(
        base["complete"]
        and base["decision_complete"]
        and not rejected
        and not global_issues
    )
    base["accepted_models"] = sorted(accepted)
    base["rejected_models"] = {
        name: rejected[name] for name in sorted(rejected)
    }
    base["success"] = final_success
    base["final_certification"] = {
        "schema_version": 1,
        "status": "certified" if final_success else "rejected",
        "accepted_models": sorted(accepted),
        "unavailable_models": sorted(unavailable),
        "rejected_models": {name: rejected[name] for name in sorted(rejected)},
        "reference_required_models": sorted(reference_models),
        "integrity_checks": checks,
        "bao_isolation": _jsonable(bao_evidence),
        "issues": sorted(global_issues),
    }
    canonical = json.dumps(
        _jsonable(base), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    base["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    return base


def write_final_cmb_certification_report(
    reports: Iterable[CMBModelDiagnostic],
    destination: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write one final certification decision as deterministic JSON."""

    record = build_final_cmb_certification_report(reports, **kwargs)
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
    matrix_fast_path: bool = False,
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
    contract_identity = _contract_identity(plugin)
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
    if matrix_fast_path:
        # Matrix certification uses the declared tier's physical node count
        # directly.  The production floor is an optional sampling safeguard
        # for long likelihood runs, not part of fixed-point evidence, and
        # would otherwise multiply every model audit by eight.
        base["_diagnostic_matrix_fast_path"] = True
    try:
        raw_data = _compute_custom_cmb_spectrum_data(
            base,
            requested_ells,
            requested_spectra=requested_spectra,
            workload="fixed_parameter_diagnostic",
        )
        base_cache_identity = _cache_identity_payload()
        refined_cache_identity: Mapping[str, Any] | None = None
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
            "base_raw_spectra": _jsonable(raw_spectra),
            "base_public_spectra": _jsonable(public_spectra),
            "refined_raw_spectra": None,
            "refined_public_spectra": None,
        }
        phase_resolution = {
            "status": raw_data.runtime_envelope.get(
                "phase_resolution_status", "not_applicable"
            ),
            "grid_status": _jsonable(
                raw_data.runtime_envelope.get("phase_grid_status", {})
            ),
            "required_nodes": int(
                raw_data.runtime_envelope.get("phase_required_nodes", 0)
            ),
            "actual_nodes": int(
                raw_data.runtime_envelope.get("k_grid_actual_count", 0)
            ),
        }
        refinement["phase_resolution"] = phase_resolution
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
            refined_cache_identity = _cache_identity_payload()
            refined_raw_spectra = refined_raw_data.spectra
            if not isinstance(refined_raw_spectra, Mapping):
                refined_raw_spectra = {
                    requested_spectra[0]: refined_raw_spectra
                }
            refined_spectra = _public_spectrum_values(
                requested_ells,
                refined_raw_spectra,
            )
            refinement["refined_raw_spectra"] = _jsonable(refined_raw_spectra)
            refinement["refined_public_spectra"] = _jsonable(refined_spectra)
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
        if phase_resolution["status"] == "under_resolved":
            failure = _diagnostic_failure(
                ConvergenceError(
                    (
                        "Fixed-parameter CCMBS projection k-grid is "
                        "under-resolved"
                    ),
                    context={"phase_resolution": phase_resolution},
                )
            )
        if (
            failure is None
            and refine_wave_number_grid
            and not refinement["converged"]
        ):
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
        acoustic_structure = assess_acoustic_structure(
            requested_ells,
            public_spectra,
        )
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
        if failure is None and acoustic_structure["issues"]:
            failure = {
                "error_type": "acoustic_shape_failure",
                "message": "; ".join(acoustic_structure["issues"]),
                "acoustic_structure": _jsonable(acoustic_structure),
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
                str(name): numpy.asarray(values, dtype=numpy.longdouble)
                for name, values in raw_data.spectra.items()
            },
            raw_transfer_components={
                str(name): numpy.asarray(values, dtype=float)
                for name, values in raw_data.transfer_components.items()
            },
            runtime_envelope=dict(raw_data.runtime_envelope),
            refinement=refinement,
            shape=shape,
            acoustic_structure=acoustic_structure,
            source_residual_audit=source_residual_audit,
            reference_comparison=reference_comparison,
            availability="rejected" if failure is not None else "measured",
            contract_identity=contract_identity,
            cache_identity={
                "base": base_cache_identity,
                "refined": refined_cache_identity,
            },
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
            availability="unavailable",
            contract_identity=contract_identity,
            failure=_diagnostic_failure(error),
        )


_BATCH_PARITY_METADATA_KEYS = frozenset(
    {
        "spectrum",
        "spectra",
        "raw_spectra",
        "requested_ells",
        "requested_spectra",
        "failure",
        "solver_id",
        "solver_label",
        "diagnostics",
        "phase_timings",
    }
)


def _batch_parity_payload(value: Any) -> dict[str, Any]:
    """Normalize scalar mappings and ordered batch items for parity checks."""

    if isinstance(value, Mapping):
        has_metadata = bool(
            _BATCH_PARITY_METADATA_KEYS.intersection(value.keys())
        )
        spectrum = value.get("spectrum", value.get("spectra"))
        if spectrum is None and not has_metadata:
            spectrum = value
        elif spectrum is None:
            spectrum = {}
        payload = {
            "spectrum": spectrum,
            "raw_spectra": value.get("raw_spectra"),
            "requested_ells": value.get("requested_ells"),
            "requested_spectra": value.get("requested_spectra"),
            "failure": value.get("failure"),
            "solver_id": value.get("solver_id"),
            "solver_label": value.get("solver_label"),
            "diagnostics": value.get("diagnostics"),
            "phase_timings": value.get("phase_timings"),
        }
    else:
        payload = {
            "spectrum": getattr(value, "spectrum", None),
            "raw_spectra": getattr(value, "raw_spectra", None),
            "requested_ells": getattr(value, "requested_ells", None),
            "requested_spectra": getattr(value, "requested_spectra", None),
            "failure": getattr(value, "failure", None),
            "solver_id": getattr(value, "solver_id", None),
            "solver_label": getattr(value, "solver_label", None),
            "diagnostics": getattr(value, "diagnostics", None),
            "phase_timings": getattr(value, "phase_timings", None),
        }
    spectrum = payload["spectrum"]
    if isinstance(spectrum, numpy.ndarray):
        spectrum = {"TT": spectrum}
    elif spectrum is not None and not isinstance(spectrum, Mapping):
        spectrum = {"TT": numpy.asarray(spectrum)}
    payload["spectrum"] = spectrum
    return payload


def _batch_failure_signature(value: Any) -> Any:
    """Keep typed failure comparisons stable across runtime diagnostics."""

    if value is None:
        return None
    if isinstance(value, CMBError):
        diagnostic = value.diagnostic()
        return {
            "error_type": type(value).__name__,
            "category": diagnostic.get("category"),
            "message": diagnostic.get("message"),
        }
    if isinstance(value, Mapping):
        return {
            "error_type": _jsonable(
                value.get("error_type") or value.get("type")
            ),
            "category": _jsonable(value.get("category")),
            "message": _jsonable(value.get("message")),
        }
    return {"error_type": type(value).__name__, "message": str(value)}


def _batch_arrays_equal(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    """Compare two named spectrum payloads without tolerances or coercion."""

    if left is None or right is None:
        return left is None and right is None
    if set(left) != set(right):
        return False
    return all(
        numpy.array_equal(
            numpy.asarray(left[name]), numpy.asarray(right[name])
        )
        for name in left
    )


def _stable_batch_diagnostics(value: Any) -> Any:
    """Remove per-call counters and timings before metadata comparison."""

    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    performance = normalized.get("performance_record")
    if isinstance(performance, Mapping):
        stable = {
            key: performance[key]
            for key in ("workload", "outcome", "stop_phase", "failure")
            if key in performance
        }
        context = performance.get("context")
        if isinstance(context, Mapping):
            stable["context"] = context
        normalized["performance_record"] = stable
    normalized.pop("elapsed_seconds", None)
    normalized.pop("cache_state", None)
    return _jsonable(normalized)


def assess_scalar_batch_cache_evidence(
    scalar_spectra: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    batch_results: Sequence[Any] | None,
    *,
    expected_indices: Sequence[int] | None = None,
    expected_requested_ells: Sequence[int] | None = None,
    expected_requested_spectra: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit exact scalar/batch equality, metadata, and cache separation.

    Public scalar spectrum mappings remain supported for compatibility.  A
    richer mapping may additionally provide raw spectra, requested grids,
    solver metadata, and a typed failure; fields present in either result
    are compared exactly.  Missing cache identities or metadata required by
    the caller are reported as non-passing evidence.
    """

    evidence: dict[str, Any] = {
        "schema_version": 2,
        "available": False,
        "converged": False,
        "ordering_preserved": False,
        "spectra_equal": False,
        "raw_spectra_equal": None,
        "metadata_equal": False,
        "cache_isolated": False,
        "issues": [],
    }
    issues: list[str] = evidence["issues"]
    if not batch_results:
        issues.append("scalar and ordered batch results are required")
        evidence["issues"] = tuple(issues)
        return evidence
    if isinstance(scalar_spectra, Mapping):
        scalar_payloads = tuple(
            scalar_spectra for _ in range(len(batch_results))
        )
    elif isinstance(scalar_spectra, Sequence) and not isinstance(
        scalar_spectra, (str, bytes)
    ):
        scalar_payloads = tuple(scalar_spectra)
        if len(scalar_payloads) != len(batch_results):
            issues.append("scalar and batch result counts do not match")
            evidence["issues"] = tuple(issues)
            return evidence
    else:
        issues.append("scalar and ordered batch results are required")
        evidence["issues"] = tuple(issues)
        return evidence
    if not all(isinstance(payload, Mapping) for payload in scalar_payloads):
        issues.append("scalar results must contain spectrum mappings")
        evidence["issues"] = tuple(issues)
        return evidence
    expected = tuple(
        range(len(batch_results))
        if expected_indices is None
        else tuple(int(value) for value in expected_indices)
    )
    if len(expected) != len(batch_results):
        issues.append("expected batch indices do not match result count")
    observed_indices: list[int] = []
    identities: list[str] = []
    equal = True
    raw_equal = True
    raw_measured = False
    metadata_equal = True
    for position, result in enumerate(batch_results):
        scalar = _batch_parity_payload(scalar_payloads[position])
        batch = _batch_parity_payload(result)
        observed_indices.append(int(getattr(result, "index", position)))
        scalar_failure = _batch_failure_signature(scalar["failure"])
        batch_failure = _batch_failure_signature(batch["failure"])
        if scalar_failure != batch_failure:
            issues.append(f"batch item {position} typed failure differs")
            equal = False
        if scalar_failure is None:
            if not _batch_arrays_equal(scalar["spectrum"], batch["spectrum"]):
                issues.append(f"batch item {position} public spectra differ")
                equal = False
            scalar_raw = scalar["raw_spectra"]
            batch_raw = batch["raw_spectra"]
            if scalar_raw is not None:
                raw_measured = True
                if not _batch_arrays_equal(scalar_raw, batch_raw):
                    issues.append(f"batch item {position} raw spectra differ")
                    raw_equal = False
            elif batch_raw is not None:
                issues.append(
                    "batch item "
                    f"{position} has raw spectra without scalar data"
                )
                raw_equal = False
        for key, expected_value in (
            ("requested_ells", expected_requested_ells),
            ("requested_spectra", expected_requested_spectra),
        ):
            if expected_value is None:
                continue
            actual = batch[key]
            if actual is None:
                issues.append(f"batch item {position} omits {key}")
                metadata_equal = False
            elif tuple(actual) != tuple(expected_value):
                issues.append(f"batch item {position} differs for {key}")
                metadata_equal = False
            scalar_value = scalar[key]
            if scalar_value is not None and tuple(scalar_value) != tuple(
                expected_value
            ):
                issues.append(f"scalar item {position} differs for {key}")
                metadata_equal = False
        for key in ("solver_id", "solver_label"):
            scalar_value = scalar[key]
            if scalar_value is not None and scalar_value != batch[key]:
                issues.append(f"batch item {position} differs for {key}")
                metadata_equal = False
        if scalar["diagnostics"] is not None and _stable_batch_diagnostics(
            scalar["diagnostics"]
        ) != _stable_batch_diagnostics(batch["diagnostics"]):
            issues.append(f"batch item {position} differs for diagnostics")
            metadata_equal = False
        if (
            scalar["phase_timings"] is not None
            and batch["phase_timings"] is None
        ):
            issues.append(f"batch item {position} differs for phase_timings")
            metadata_equal = False
        provenance = getattr(result, "cache_provenance", {}) or {}
        identity = provenance.get("cache_identity")
        if identity is not None:
            identities.append(repr(identity))
    ordering = tuple(observed_indices) == expected
    if not ordering:
        issues.append("batch result ordering does not match input ordering")
    isolated = len(identities) == len(batch_results) and len(
        set(identities)
    ) == len(identities)
    if not isolated:
        issues.append("batch cache identities are missing or cross-talked")
    evidence.update(
        {
            "available": not issues,
            "converged": not issues,
            "ordering_preserved": ordering,
            "spectra_equal": equal,
            "raw_spectra_equal": raw_equal if raw_measured else None,
            "metadata_equal": metadata_equal,
            "cache_isolated": isolated,
            "scalar_count": len(scalar_payloads),
            "identity_count": len(identities),
            "issues": tuple(issues),
        }
    )
    return evidence


def _matrix_batch_parameter_points(
    plugin: Any,
) -> tuple[tuple[tuple[float, ...], ...], str | None]:
    """Choose two bounded points for an exact scalar/batch comparison."""

    initial = tuple(
        float(value) for value in getattr(plugin, "INITIAL_GUESSES", ())
    )
    bounds = tuple(getattr(plugin, "PARAMETER_BOUNDS", ()) or ())
    if not initial or len(bounds) != len(initial):
        return (initial,), "model parameter bounds are unavailable"
    varied = list(initial)
    for index, (value, bound) in enumerate(zip(initial, bounds)):
        if bound is None or len(bound) != 2:
            continue
        lower, upper = bound
        if lower is None or upper is None:
            continue
        lower_value = float(lower)
        upper_value = float(upper)
        if not numpy.isfinite(lower_value) or not numpy.isfinite(upper_value):
            continue
        span = upper_value - lower_value
        if span <= 0.0:
            continue
        step = max(span * 1.0e-2, numpy.finfo(float).eps)
        candidate = value + step
        if candidate > upper_value:
            candidate = value - step
        if candidate < lower_value or candidate == value:
            continue
        varied[index] = candidate
        return (initial, tuple(varied)), None
    return (initial,), "model contract has no independently variable parameter"


def _run_scalar_batch_cache_check(
    plugin: Any,
    report: CMBModelDiagnostic,
    *,
    ells: Sequence[int],
    spectra: Sequence[str],
    numerical_overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare scalar and ordered batch results at two parameter points."""

    unavailable = {
        "available": False,
        "converged": False,
        "status": "unavailable",
        "reason": "fixed-point scalar evidence is unavailable",
    }
    cache_unavailable = {
        "available": False,
        "isolated": False,
        "status": "unavailable",
        "reason": "fixed-point scalar evidence is unavailable",
    }
    if not report.spectra and report.failure is not None:
        return unavailable, cache_unavailable
    points, reason = _matrix_batch_parameter_points(plugin)
    if reason is not None:
        unavailable = dict(unavailable)
        unavailable["reason"] = reason
        cache_unavailable = dict(cache_unavailable)
        cache_unavailable["reason"] = reason
        return unavailable, cache_unavailable
    try:
        from . import cmb as cmb_api
        from .cmb import (
            compute_cmb_spectrum_batch,
            compute_cmb_spectrum_cached,
        )

        # A diagnostic can reject a finite solve for shape or residual
        # evidence.  Parity still has to compare its scalar spectrum with the
        # ordered batch item; only a report with no spectrum at all carries a
        # failure into this execution audit.
        scalar_payloads: list[Mapping[str, Any]] = [
            {
                "spectrum": report.spectra or None,
                "raw_spectra": report.raw_spectra or None,
                "requested_ells": report.requested_ells,
                "requested_spectra": report.requested_spectra,
                "failure": None if report.spectra else report.failure,
            }
        ]
        contracts: list[Mapping[str, Any]] = []
        for index, parameters in enumerate(points):
            contract = plugin.get_cmb_declared_runtime(parameters)
            if numerical_overrides:
                contract = _bound_contract(contract, numerical_overrides)
            contract = dict(contract)
            contract["_diagnostic_matrix_fast_path"] = True
            contracts.append(contract)
            if index:
                try:
                    cmb_api._LAST_CMB_RESULT.set(None)
                    scalar = compute_cmb_spectrum_cached(
                        plugin,
                        parameters,
                        ells,
                        spectra=spectra,
                        workload="matrix_batch_reference",
                        numerical_overrides=numerical_overrides,
                        diagnostic_matrix_fast_path=True,
                    )
                # DEVCOV_ALLOW_BROAD_ONCE: isolate scalar batch failures.
                except Exception as error:
                    scalar_result = cmb_api._LAST_CMB_RESULT.get()
                    scalar_payloads.append(
                        {
                            "spectrum": None,
                            "raw_spectra": None,
                            "requested_ells": ells,
                            "requested_spectra": spectra,
                            "failure": (
                                None
                                if scalar_result is None
                                else scalar_result.failure
                            )
                            or _diagnostic_failure(error),
                        }
                    )
                    continue
                scalar_result = cmb_api._LAST_CMB_RESULT.get()
                if scalar_result is not None:
                    scalar_payloads.append(
                        {
                            "spectrum": scalar_result.spectra,
                            "raw_spectra": scalar_result.raw_spectra,
                            "requested_ells": scalar_result.requested_ells,
                            "requested_spectra": (
                                scalar_result.requested_spectra
                            ),
                            "failure": scalar_result.failure,
                            "solver_id": scalar_result.solver_id,
                            "solver_label": scalar_result.solver_label,
                            "diagnostics": scalar_result.diagnostics,
                            "phase_timings": scalar_result.phase_timings,
                        }
                    )
                elif isinstance(scalar, Mapping):
                    scalar_payloads.append(
                        {
                            "spectrum": scalar,
                            "requested_ells": ells,
                            "requested_spectra": spectra,
                        }
                    )
                else:
                    scalar_payloads.append(
                        {
                            "spectrum": {spectra[0]: scalar},
                            "requested_ells": ells,
                            "requested_spectra": spectra,
                        }
                    )
        batch_results = compute_cmb_spectrum_batch(
            contracts,
            ells,
            background_provider=plugin,
            requested_spectra=spectra,
            workload="matrix_batch_reference",
        )
        batch_metadata_available = all(
            bool(getattr(item, "requested_ells", ()))
            and bool(getattr(item, "requested_spectra", ()))
            for item in batch_results
        )
        evidence = assess_scalar_batch_cache_evidence(
            scalar_payloads,
            batch_results,
            expected_indices=range(len(points)),
            expected_requested_ells=(
                ells if batch_metadata_available else None
            ),
            expected_requested_spectra=(
                spectra if batch_metadata_available else None
            ),
        )
        evidence = dict(evidence)
        evidence.update(
            {
                "status": "passed" if evidence["available"] else "failed",
                "parameter_points": [list(point) for point in points],
                "batch_results": [
                    _jsonable(result.to_dict()) for result in batch_results
                ],
            }
        )
        cache_evidence = {
            "available": bool(evidence["available"]),
            "isolated": bool(evidence["cache_isolated"]),
            "status": evidence["status"],
            "identity_count": int(evidence.get("identity_count", 0)),
            "issues": tuple(evidence.get("issues", ())),
        }
        return evidence, cache_evidence
    except (
        ArithmeticError,
        AttributeError,
        CMBError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        failure = _diagnostic_failure(error)
        unavailable = dict(unavailable)
        unavailable.update(
            {
                "reason": "scalar/batch check raised a typed failure",
                "failure": failure,
            }
        )
        cache_unavailable = dict(cache_unavailable)
        cache_unavailable.update(
            {
                "reason": "cache isolation could not be measured",
                "failure": failure,
            }
        )
        return unavailable, cache_unavailable


def build_bundled_cmb_matrix_report(
    reports: Iterable[CMBModelDiagnostic],
    *,
    required_model_filenames: Iterable[str] = BUNDLED_CMB_MODEL_FILENAMES,
    required_spectra: Sequence[str] = _DEFAULT_SPECTRA,
    certification_tier: Mapping[str, Any] | None = None,
    contract_audits: Mapping[str, Mapping[str, Any]] | None = None,
    source_graph_audits: Mapping[str, Mapping[str, Any]] | None = None,
    declaration_audits: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the strict filename-keyed scientific matrix for Slice Five."""

    return build_cmb_certification_report(
        reports,
        required_model_filenames=required_model_filenames,
        required_spectra=required_spectra,
        require_reference=False,
        matrix_mode=True,
        certification_tier=certification_tier or CMB_CERTIFICATION_TIER,
        contract_audits=contract_audits,
        source_graph_audits=source_graph_audits,
        declaration_audits=declaration_audits,
    )


def build_bundled_cmb_full_matrix_report(
    reports: Iterable[CMBModelDiagnostic],
    *,
    required_model_filenames: Iterable[str] = BUNDLED_CMB_MODEL_FILENAMES,
    declared_spectra_by_model: Mapping[str, Sequence[str]] | None = None,
    certification_tier: Mapping[str, Any] | None = None,
    contract_audits: Mapping[str, Mapping[str, Any]] | None = None,
    source_graph_audits: Mapping[str, Mapping[str, Any]] | None = None,
    declaration_audits: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the complete per-model observable certification matrix.

    Unlike the historical TT/TE/EE matrix, this report derives required
    spectra per model and records an explicit classification for every row.
    A declaration-level ``unavailable`` decision is the only acceptable
    unavailable outcome; an enabled model that fails execution is rejected.
    Thus no missing output can be hidden by a corpus-wide default request.
    """

    ordered_reports = tuple(
        sorted(reports, key=lambda item: str(item.model_filename))
    )
    expected_values = tuple(str(name) for name in required_model_filenames)
    expected = tuple(sorted(set(expected_values)))
    expected_duplicates = sorted(
        name
        for name in set(expected_values)
        if expected_values.count(name) > 1
    )
    seen_values = tuple(str(item.model_filename) for item in ordered_reports)
    seen = set(seen_values)
    duplicate_models = sorted(
        name for name in seen if seen_values.count(name) > 1
    )
    missing_models = sorted(set(expected) - seen)
    unexpected_models = sorted(seen - set(expected))
    declared_map = {
        str(name): tuple(
            canonical_cmb_spectrum_name(value) for value in values
        )
        for name, values in (declared_spectra_by_model or {}).items()
    }
    records: list[dict[str, Any]] = []
    accepted: list[str] = []
    unavailable: list[str] = []
    rejected: dict[str, list[str]] = {}
    classifications: dict[str, str] = {}
    for report in ordered_reports:
        filename = str(report.model_filename)
        declared = declared_map.get(filename, ())
        issues: list[str] = []
        declaration = (declaration_audits or {}).get(filename) or {}
        declaration_decision = str(declaration.get("decision", ""))
        if filename not in declared_map:
            issues.append("declared observable inventory is unavailable")
        if tuple(report.requested_spectra) != tuple(declared):
            issues.append(
                "diagnostic request does not exactly match declared spectra"
            )
        if declaration_decision == "unavailable":
            if report.availability != "unavailable":
                issues.append(
                    "unavailable declaration must retain a typed unavailable "
                    "diagnostic outcome"
                )
            outcome = "unavailable" if not issues else "rejected"
        else:
            valid, evidence_issues = _matrix_evidence_status(
                report,
                required_spectra=declared,
                require_reference=False,
                contract_audit=(contract_audits or {}).get(filename),
                source_graph_audit=(source_graph_audits or {}).get(filename),
                declaration_audit=declaration,
            )
            issues.extend(evidence_issues)
            outcome = "accepted" if valid and not issues else "rejected"
        classifications[filename] = outcome
        if outcome == "accepted":
            accepted.append(filename)
        elif outcome == "unavailable":
            unavailable.append(filename)
        else:
            rejected[filename] = list(issues)
        records.append(
            {
                "model_filename": filename,
                "classification": outcome,
                "accepted": outcome == "accepted",
                "availability": report.availability,
                "declared_spectra": list(declared),
                "raw_evidence_sha256": _canonical_sha256(report.to_dict()),
                "contract_audit": _jsonable(
                    (contract_audits or {}).get(filename)
                ),
                "declaration_audit": _jsonable(declaration),
                "source_graph_audit": _jsonable(
                    (source_graph_audits or {}).get(filename)
                ),
                "issues": list(issues),
                "report": report.to_dict(),
            }
        )
    for filename in missing_models:
        classifications[filename] = "rejected"
        rejected[filename] = ["model is missing from diagnostic matrix"]
    for filename in unexpected_models:
        classifications[filename] = "rejected"
        rejected[filename] = ["model is not in the frozen CMB corpus"]
    for filename in duplicate_models:
        classifications[filename] = "rejected"
        rejected.setdefault(filename, []).append(
            "model appears more than once in diagnostic matrix"
        )
    for filename in expected_duplicates:
        classifications[filename] = "rejected"
        rejected.setdefault(filename, []).append(
            "required model list contains duplicate entries"
        )
    complete = (
        not missing_models
        and not unexpected_models
        and not duplicate_models
        and not expected_duplicates
        and seen == set(expected)
        and len(ordered_reports) == len(expected)
    )
    decision_complete = bool(
        complete
        and all(
            classifications.get(filename)
            in {"accepted", "rejected", "unavailable"}
            for filename in expected
        )
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ccmbs_bundled_full_observable_matrix",
        "required_models": list(expected),
        "declared_spectra_by_model": {
            name: list(declared_map.get(name, ())) for name in expected
        },
        "certification_tier": _jsonable(certification_tier or {}),
        "complete": complete,
        "decision_complete": decision_complete,
        "no_unclassified_models": decision_complete,
        "success": bool(complete and not rejected),
        "accepted_models": sorted(accepted),
        "unavailable_models": sorted(unavailable),
        "rejected_models": {name: rejected[name] for name in sorted(rejected)},
        "classifications": {
            name: classifications[name] for name in sorted(classifications)
        },
        "reports": records,
    }
    record["record_sha256"] = _canonical_sha256(record)
    return record


def run_bundled_cmb_full_matrix(
    *,
    model_directory: str | Path | None = None,
    certification_tier: Mapping[str, Any] | None = None,
    numerical_overrides: Mapping[str, Any] | None = None,
    reference_spectra_by_model: Mapping[str, Mapping[str, Any]] | None = None,
    reference_tolerances_by_model: (
        Mapping[str, Mapping[str, float]] | None
    ) = None,
    execute_batch_checks: bool = True,
) -> dict[str, Any]:
    """Execute the full declared-observable matrix for every CMB model.

    Requests are taken from each compiled model's angular-spectrum
    declarations.  This keeps BB/PP/cross and future lensed observables in
    the same raw-evidence path, while retaining typed unavailable outcomes for
    models that explicitly disable CMB output.
    """

    tier = dict(CMB_CERTIFICATION_TIER)
    tier.update(certification_tier or {})
    requested_ells = tuple(int(value) for value in tier.get("ells", ()))
    if not requested_ells:
        raise ValueError("Certification tier must declare ells")
    tier_overrides = dict(tier.get("numerical_overrides", {}) or {})
    tier_overrides.update(numerical_overrides or {})
    plugins = discover_bundled_cmb_plugins(model_directory)
    default_corpus = model_directory is None
    expected = (
        BUNDLED_CMB_MODEL_FILENAMES
        if default_corpus
        else tuple(sorted(str(plugin.MODEL_FILENAME) for plugin in plugins))
    )
    contract_audits = {
        audit.model_filename: audit.to_dict()
        for audit in audit_bundled_cmb_contracts(model_directory)
    }
    source_graph_audits = {
        audit.model_filename: audit.to_dict()
        for audit in audit_bundled_cmb_source_graphs(model_directory)
    }
    declaration_audits = {
        decision.model_filename: decision.to_dict()
        for decision in audit_bundled_cmb_declarations(model_directory)
    }
    declared_map = {
        str(plugin.MODEL_FILENAME): declared_cmb_spectrum_names(plugin)
        for plugin in plugins
    }
    reports: list[CMBModelDiagnostic] = []
    for plugin in plugins:
        filename = str(plugin.MODEL_FILENAME)
        requested_spectra = declared_map.get(filename, ())
        declaration = declaration_audits.get(filename, {})
        if not requested_spectra:
            reports.append(
                _unavailable_report(
                    plugin,
                    requested_ells=requested_ells,
                    requested_spectra=(),
                    error_type="NoDeclaredCMBObservable",
                    message="Model declares no angular CMB spectrum",
                    category="unavailable",
                )
            )
            continue
        if declaration.get("decision") == "unavailable":
            reports.append(
                _unavailable_report(
                    plugin,
                    requested_ells=requested_ells,
                    requested_spectra=requested_spectra,
                    error_type="CMBCapabilityUnavailable",
                    message="Model declaration explicitly disables CMB output",
                    category="unavailable",
                )
            )
            continue
        try:
            report = run_cmb_model_diagnostic(
                plugin,
                ells=requested_ells,
                spectra=requested_spectra,
                numerical_overrides=tier_overrides,
                refine_wave_number_grid=bool(
                    tier.get("refine_wave_number_grid", True)
                ),
                reference_spectra=(reference_spectra_by_model or {}).get(
                    filename
                ),
                reference_tolerances=(reference_tolerances_by_model or {}).get(
                    filename
                ),
                matrix_fast_path=True,
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
            report = _unavailable_report(
                plugin,
                requested_ells=requested_ells,
                requested_spectra=requested_spectra,
                error_type=type(error).__name__,
                message=str(error),
                category="rejected",
            )
            report = replace(report, availability="rejected")
        scalar_batch = report.scalar_batch_evidence
        cache_isolation = report.cache_isolation_evidence
        if execute_batch_checks and report.availability == "measured":
            scalar_batch, cache_isolation = _run_scalar_batch_cache_check(
                plugin,
                report,
                ells=requested_ells,
                spectra=requested_spectra,
                numerical_overrides=tier_overrides,
            )
        elif not execute_batch_checks:
            scalar_batch = {
                "available": False,
                "converged": False,
                "status": "not_measured",
                "reason": "exact ordered batch check was not executed",
            }
            cache_isolation = {
                "available": False,
                "isolated": False,
                "status": "not_measured",
                "reason": "cache identity comparison was not executed",
            }
        reports.append(
            replace(
                report,
                contract_identity=report.contract_identity
                or _contract_identity(plugin),
                scalar_batch_evidence=scalar_batch,
                cache_isolation_evidence=cache_isolation,
            )
        )
    return build_bundled_cmb_full_matrix_report(
        reports,
        required_model_filenames=expected,
        declared_spectra_by_model=declared_map,
        certification_tier=tier,
        contract_audits=contract_audits,
        source_graph_audits=source_graph_audits,
        declaration_audits=declaration_audits,
    )


def write_bundled_cmb_full_matrix_report(
    reports: Iterable[CMBModelDiagnostic],
    destination: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write one deterministic full-observable corpus matrix report."""

    record = build_bundled_cmb_full_matrix_report(reports, **kwargs)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def write_bundled_cmb_matrix_report(
    reports: Iterable[CMBModelDiagnostic],
    destination: str | Path,
    *,
    required_model_filenames: Iterable[str] = BUNDLED_CMB_MODEL_FILENAMES,
    required_spectra: Sequence[str] = _DEFAULT_SPECTRA,
    certification_tier: Mapping[str, Any] | None = None,
    contract_audits: Mapping[str, Mapping[str, Any]] | None = None,
    source_graph_audits: Mapping[str, Mapping[str, Any]] | None = None,
    declaration_audits: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a deterministic bundled-model matrix report to JSON."""

    record = build_bundled_cmb_matrix_report(
        reports,
        required_model_filenames=required_model_filenames,
        required_spectra=required_spectra,
        certification_tier=certification_tier,
        contract_audits=contract_audits,
        source_graph_audits=source_graph_audits,
        declaration_audits=declaration_audits,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def run_bundled_cmb_matrix(
    *,
    model_directory: str | Path | None = None,
    certification_tier: Mapping[str, Any] | None = None,
    numerical_overrides: Mapping[str, Any] | None = None,
    reference_spectra_by_model: Mapping[str, Mapping[str, Any]] | None = None,
    reference_tolerances_by_model: (
        Mapping[str, Mapping[str, float]] | None
    ) = None,
    execute_batch_checks: bool = False,
) -> dict[str, Any]:
    """Run the fixed-point matrix and preserve every explicit outcome.

    ``execute_batch_checks`` is opt-in because the exact scalar-to-batch
    adapter intentionally repeats the expensive solver call.  When it is
    false, the matrix records a typed unavailable evidence decision instead
    of claiming scalar/batch parity.
    """

    tier = dict(CMB_CERTIFICATION_TIER)
    tier.update(certification_tier or {})
    requested_ells = tuple(int(value) for value in tier.get("ells", ()))
    requested_spectra = tuple(
        str(value).upper() for value in tier.get("spectra", _DEFAULT_SPECTRA)
    )
    if not requested_ells or not requested_spectra:
        raise ValueError("Certification tier must declare ells and spectra")
    tier_overrides = dict(tier.get("numerical_overrides", {}) or {})
    tier_overrides.update(numerical_overrides or {})
    plugins = discover_bundled_cmb_plugins(model_directory)
    default_corpus = model_directory is None
    expected = (
        BUNDLED_CMB_MODEL_FILENAMES
        if default_corpus
        else tuple(sorted(str(plugin.MODEL_FILENAME) for plugin in plugins))
    )
    contract_audits = {
        audit.model_filename: audit.to_dict()
        for audit in audit_bundled_cmb_contracts(model_directory)
    }
    source_graph_audits = {
        audit.model_filename: audit.to_dict()
        for audit in audit_bundled_cmb_source_graphs(model_directory)
    }
    declaration_audits = {
        decision.model_filename: decision.to_dict()
        for decision in audit_bundled_cmb_declarations(model_directory)
    }
    reports: list[CMBModelDiagnostic] = []
    for plugin in plugins:
        filename = str(plugin.MODEL_FILENAME)
        try:
            report = run_cmb_model_diagnostic(
                plugin,
                ells=requested_ells,
                spectra=requested_spectra,
                numerical_overrides=tier_overrides,
                refine_wave_number_grid=bool(
                    tier.get("refine_wave_number_grid", True)
                ),
                reference_spectra=(reference_spectra_by_model or {}).get(
                    filename
                ),
                reference_tolerances=(reference_tolerances_by_model or {}).get(
                    filename
                ),
                matrix_fast_path=True,
            )
        except (
            ArithmeticError,
            AttributeError,
            CMBError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:  # pragma: no cover - defensive boundary
            report = _unavailable_report(
                plugin,
                requested_ells=requested_ells,
                requested_spectra=requested_spectra,
                error_type=type(error).__name__,
                message=str(error),
            )
        scalar_batch = report.scalar_batch_evidence
        cache_isolation = report.cache_isolation_evidence
        if execute_batch_checks:
            scalar_batch, cache_isolation = _run_scalar_batch_cache_check(
                plugin,
                report,
                ells=requested_ells,
                spectra=requested_spectra,
                numerical_overrides=tier_overrides,
            )
        else:
            scalar_batch = {
                "available": False,
                "converged": False,
                "status": "not_measured",
                "reason": "exact ordered batch check was not executed",
            }
            cache_isolation = {
                "available": False,
                "isolated": False,
                "status": "not_measured",
                "reason": "cache identity comparison was not executed",
            }
        report = replace(
            report,
            contract_identity=report.contract_identity
            or _contract_identity(plugin),
            scalar_batch_evidence=scalar_batch,
            cache_isolation_evidence=cache_isolation,
        )
        reports.append(report)
    return build_bundled_cmb_matrix_report(
        reports,
        required_model_filenames=expected,
        required_spectra=requested_spectra,
        certification_tier=tier,
        contract_audits=contract_audits,
        source_graph_audits=source_graph_audits,
        declaration_audits=declaration_audits,
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


def _build_corpus_baseline_row(
    report: CMBModelDiagnostic,
    *,
    baseline_request: Mapping[str, Any],
    contract_audit: Mapping[str, Any] | None,
    source_graph_audit: Mapping[str, Any] | None,
    decision: str | None = None,
    completion_state: str = "completed",
    progression: Sequence[Mapping[str, Any]] = (),
    decision_context: Mapping[str, Any] | None = None,
) -> CMBCorpusBaselineRow:
    """Attach corpus-level evidence to one direct diagnostic report."""

    context = {
        "reported_availability": report.availability,
        "typed_failure": _jsonable(report.failure),
    }
    context.update(decision_context or {})
    request_identity = {
        "baseline_request_sha256": _canonical_sha256(baseline_request),
        "parameter_source": baseline_request["parameter_source"],
        "parameter_names": list(report.parameter_names),
        "parameter_values": list(report.parameter_values),
    }
    return CMBCorpusBaselineRow(
        model_filename=report.model_filename,
        model_name=report.model_name,
        decision=decision or _baseline_decision(report),
        diagnostic=report,
        contract_audit=dict(contract_audit or {}),
        source_graph_audit=dict(source_graph_audit or {}),
        request_identity=request_identity,
        projection_metadata=_baseline_projection_metadata(report),
        source_history_metadata=_baseline_source_history_metadata(report),
        work_estimate=_baseline_work_estimate(
            baseline_request,
            report=report,
        ),
        completion_state=completion_state,
        progression=tuple(progression),
        decision_context=context,
    )


def _missing_corpus_baseline_report(
    filename: str,
    *,
    baseline_request: Mapping[str, Any],
) -> CMBModelDiagnostic:
    """Represent a missing frozen corpus plugin as a typed unavailable row."""

    return CMBModelDiagnostic(
        model_filename=filename,
        model_name=filename,
        parameter_names=(),
        parameter_values=(),
        requested_ells=tuple(baseline_request["ells"]),
        requested_spectra=tuple(baseline_request["spectra"]),
        availability="unavailable",
        failure={
            "error_type": "CorpusDiscoveryError",
            "category": "unavailable",
            "message": "Frozen CMB corpus model was not discovered",
        },
    )


def _tier_baseline_request(
    baseline_request: Mapping[str, Any],
    tier: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one USMF2 tier to the common baseline dimensions."""

    tier_request = dict(baseline_request)
    tier_request["numerical_overrides"] = dict(tier["numerical_overrides"])
    tier_request["refinement"] = {
        "axis": "k_sample_count",
        "factor": 2,
        "required": bool(tier["refine_wave_number_grid"]),
    }
    return tier_request


def _completed_usmf2_progression(
    completed_tiers: Sequence[Mapping[str, Any]],
    required_tiers: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether a progression executed every named USMF2 tier."""

    return tuple(tier["id"] for tier in completed_tiers) == tuple(
        tier["id"] for tier in required_tiers
    )


def _run_usmf2_corpus_baseline(
    plugin: Any,
    *,
    baseline_request: Mapping[str, Any],
    tiers: Sequence[Mapping[str, Any]],
    required_tiers: Sequence[Mapping[str, Any]],
    contract_audit: Mapping[str, Any] | None,
    source_graph_audit: Mapping[str, Any] | None,
) -> CMBCorpusBaselineRow:
    """Measure USMF2 by finite tiers and preserve incomplete obligations."""

    progression: list[dict[str, Any]] = []
    terminal_report: CMBModelDiagnostic | None = None
    for tier in tiers:
        tier_request = _tier_baseline_request(baseline_request, tier)
        try:
            report = run_cmb_model_diagnostic(
                plugin,
                ells=baseline_request["ells"],
                spectra=baseline_request["spectra"],
                numerical_overrides=tier["numerical_overrides"],
                refine_wave_number_grid=bool(tier["refine_wave_number_grid"]),
                matrix_fast_path=True,
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
            report = _unavailable_report(
                plugin,
                requested_ells=tuple(baseline_request["ells"]),
                requested_spectra=tuple(baseline_request["spectra"]),
                error_type=type(error).__name__,
                message=str(error),
            )
        terminal_report = report
        progression.append(
            {
                "position": int(tier["position"]),
                "id": str(tier["id"]),
                "numerical_overrides": _jsonable(tier["numerical_overrides"]),
                "refine_wave_number_grid": bool(
                    tier["refine_wave_number_grid"]
                ),
                "completion_state": "completed",
                "decision": _baseline_decision(report),
                "work_estimate": _baseline_work_estimate(tier_request),
                "typed_failure": _jsonable(report.failure),
                "diagnostic_sha256": _canonical_sha256(report.to_dict()),
            }
        )
    if terminal_report is None:  # pragma: no cover - tier validation guards it
        raise ValueError("USMF2 baseline progression did not execute a tier")
    if _completed_usmf2_progression(tiers, required_tiers):
        return _build_corpus_baseline_row(
            terminal_report,
            baseline_request=baseline_request,
            contract_audit=contract_audit,
            source_graph_audit=source_graph_audit,
            progression=progression,
            decision_context={
                "progression_complete": True,
                "remaining_tiers": (),
            },
        )
    remaining_tiers = tuple(
        {
            "position": int(tier["position"]),
            "id": str(tier["id"]),
            "numerical_overrides": _jsonable(tier["numerical_overrides"]),
            "refine_wave_number_grid": bool(tier["refine_wave_number_grid"]),
            "work_estimate": _baseline_work_estimate(
                _tier_baseline_request(baseline_request, tier)
            ),
        }
        for tier in required_tiers[len(tiers) :]
    )
    return _build_corpus_baseline_row(
        terminal_report,
        baseline_request=baseline_request,
        contract_audit=contract_audit,
        source_graph_audit=source_graph_audit,
        decision="unclassified",
        completion_state="incomplete",
        progression=progression,
        decision_context={
            "progression_complete": False,
            "incomplete_reason": (
                "The supplied USMF2 progression stopped before the named "
                "corpus baseline request."
            ),
            "remaining_tiers": remaining_tiers,
        },
    )


def run_bundled_cmb_corpus_baseline(
    *,
    model_directory: str | Path | None = None,
    baseline_request: Mapping[str, Any] | None = None,
    usmf2_progression: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the direct pre-repair baseline once for every bundled CMB model.

    This is intentionally separate from final matrix certification.  It runs
    the common fixed request, attaches declaration and source-graph audits,
    and records rejected or unfinished evidence without inventing a passing
    scientific result.  USMF2 only becomes classified after the supplied
    finite progression reaches the exact named baseline request.
    """

    request = _normalize_corpus_baseline_request(baseline_request)
    tiers = _normalize_usmf2_baseline_tiers(
        usmf2_progression,
        baseline_request=request,
    )
    required_usmf2_tiers = _normalize_usmf2_baseline_tiers(
        None,
        baseline_request=request,
    )
    plugins = discover_bundled_cmb_plugins(model_directory)
    discovered_filenames = tuple(
        str(plugin.MODEL_FILENAME) for plugin in plugins
    )
    expected = (
        BUNDLED_CMB_MODEL_FILENAMES
        if model_directory is None
        else tuple(sorted(discovered_filenames))
    )
    contract_audits = {
        audit.model_filename: audit.to_dict()
        for audit in audit_bundled_cmb_contracts(model_directory)
    }
    source_graph_audits = {
        audit.model_filename: audit.to_dict()
        for audit in audit_bundled_cmb_source_graphs(model_directory)
    }
    plugin_by_filename = {
        str(plugin.MODEL_FILENAME): plugin for plugin in plugins
    }
    rows: list[CMBCorpusBaselineRow] = []
    for filename in expected:
        plugin = plugin_by_filename.get(filename)
        if plugin is None:
            report = _missing_corpus_baseline_report(
                filename,
                baseline_request=request,
            )
            rows.append(
                _build_corpus_baseline_row(
                    report,
                    baseline_request=request,
                    contract_audit=contract_audits.get(filename),
                    source_graph_audit=source_graph_audits.get(filename),
                )
            )
            continue
        if filename == "model_usmf2.yml":
            rows.append(
                _run_usmf2_corpus_baseline(
                    plugin,
                    baseline_request=request,
                    tiers=tiers,
                    required_tiers=required_usmf2_tiers,
                    contract_audit=contract_audits.get(filename),
                    source_graph_audit=source_graph_audits.get(filename),
                )
            )
            continue
        try:
            report = run_cmb_model_diagnostic(
                plugin,
                ells=request["ells"],
                spectra=request["spectra"],
                numerical_overrides=request["numerical_overrides"],
                refine_wave_number_grid=bool(
                    request["refinement"]["required"]
                ),
                matrix_fast_path=True,
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
            report = _unavailable_report(
                plugin,
                requested_ells=tuple(request["ells"]),
                requested_spectra=tuple(request["spectra"]),
                error_type=type(error).__name__,
                message=str(error),
            )
        rows.append(
            _build_corpus_baseline_row(
                report,
                baseline_request=request,
                contract_audit=contract_audits.get(filename),
                source_graph_audit=source_graph_audits.get(filename),
            )
        )
    return build_cmb_corpus_baseline_report(
        rows,
        baseline_request=request,
        required_model_filenames=expected,
        discovered_model_filenames=discovered_filenames,
    )


__all__ = [
    "BUNDLED_CMB_MODEL_FILENAMES",
    "CMB_CERTIFICATION_TIER",
    "CMB_CORPUS_BASELINE_REQUEST",
    "CMB_USMF2_BASELINE_TIERS",
    "CMBCorpusBaselineRow",
    "CMBModelDiagnostic",
    "assess_scalar_batch_cache_evidence",
    "assess_acoustic_structure",
    "assess_physical_spectrum_shape",
    "audit_source_history_residuals",
    "build_bundled_cmb_matrix_report",
    "build_bundled_cmb_full_matrix_report",
    "build_cmb_corpus_baseline_report",
    "build_cmb_certification_report",
    "build_cmb_parity_report",
    "build_final_cmb_certification_report",
    "compare_cmb_spectra_to_reference",
    "compare_full_cmb_observable_parity",
    "discover_bundled_cmb_plugins",
    "run_bundled_cmb_matrix",
    "run_bundled_cmb_full_matrix",
    "run_bundled_cmb_corpus_baseline",
    "run_bundled_cmb_diagnostics",
    "run_cmb_model_diagnostic",
    "declared_cmb_spectrum_names",
    "write_bundled_cmb_full_matrix_report",
    "write_bundled_cmb_matrix_report",
    "write_cmb_corpus_baseline_report",
    "write_cmb_certification_report",
    "write_final_cmb_certification_report",
]
