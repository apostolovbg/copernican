"""Deterministic validation and evidence for declared CMB surfaces.

Post-processing is the boundary between projected transfer products and the
public CMB observables.  This module keeps that boundary model-independent:
it records which products feed each requested surface, validates finite and
positive auto spectra, and hashes the raw intermediates without changing
their numerical values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy

from ....cmb_output import (
    canonical_cmb_spectrum_name,
    describe_cmb_spectrum,
    split_cmb_spectrum_name,
)

POST_PROCESSING_SCHEMA_VERSION = 1
_AUTO_SPECTRA = frozenset({"TT", "EE", "BB", "PP"})
_LENSING_INPUTS = ("TT", "TE", "EE", "BB", "PP")


def _canonical_json(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if isinstance(value, numpy.ndarray):
        return _canonical_json(value.tolist())
    if isinstance(value, numpy.generic):
        return _canonical_json(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _canonical_json(item)
            for item in sorted(value, key=lambda item: str(item))
        ]
    return value


def _array_digest(values: Any) -> str:
    """Hash one numeric array in a portable, shape-aware representation."""

    array = numpy.asarray(values)
    if not numpy.all(numpy.isfinite(array)):
        raise ValueError("Post-processing inputs must be finite")
    normalized = numpy.ascontiguousarray(array)
    header = json.dumps(
        {
            "dtype": str(normalized.dtype),
            "shape": tuple(int(v) for v in array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + normalized.tobytes()).hexdigest()


def _surface_dependencies(name: str) -> tuple[str, ...]:
    """Return the physical products required by one public surface."""

    canonical = canonical_cmb_spectrum_name(name)
    lensed, _component, base = split_cmb_spectrum_name(canonical)
    if lensed:
        return _LENSING_INPUTS
    return (base,)


def _surface_record(
    name: str,
    values: Any,
    *,
    expected_length: int,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Validate and summarize one public surface."""

    array = numpy.asarray(values)
    finite = bool(array.ndim == 1 and array.size == expected_length)
    finite = bool(finite and numpy.all(numpy.isfinite(array)))
    _lensed, _component, base = split_cmb_spectrum_name(name)
    auto = base in _AUTO_SPECTRA
    maximum_ld = (
        numpy.max(numpy.abs(array)) if array.size else numpy.longdouble(0.0)
    )
    allowed_negative = numpy.longdouble(tolerance) * max(
        maximum_ld,
        numpy.longdouble(1.0),
    )
    minimum_ld = numpy.min(array) if array.size else numpy.longdouble(0.0)
    positive = bool(not auto or minimum_ld >= -allowed_negative)
    signs = numpy.sign(array)
    nonzero_signs = tuple(sorted({int(value) for value in signs if value}))
    sign_changes = bool(len(nonzero_signs) > 1)

    def _safe_float(value: Any) -> float:
        """Summarize extended-precision values without overflow warnings."""

        value_ld = numpy.longdouble(value)
        limit = numpy.longdouble(numpy.finfo(float).max)
        if value_ld > limit:
            return float("inf")
        if value_ld < -limit:
            return float("-inf")
        return float(value_ld)

    return {
        "name": canonical_cmb_spectrum_name(name),
        "units": describe_cmb_spectrum(name).units,
        "shape": tuple(int(value) for value in array.shape),
        "finite": finite,
        "auto": auto,
        "minimum": _safe_float(minimum_ld),
        "maximum_absolute": _safe_float(maximum_ld),
        "auto_nonnegative": positive,
        "signs": nonzero_signs,
        "sign_changes": sign_changes,
        "sha256": _array_digest(array) if finite else None,
        "accepted": bool(finite and positive),
    }


def build_postprocessing_evidence(
    *,
    transfer_components: Mapping[str, Any],
    unlensed_spectra: Mapping[str, Any],
    output_spectra: Mapping[str, Any],
    requested_spectra: Iterable[str],
    spectrum_availability: Mapping[str, str] | None,
    ell_grid: Any,
    k_grid: Any,
    lensed: bool,
    lensing_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build complete, hash-addressed post-processing evidence.

    The returned mapping contains no model-specific assumptions.  A surface
    is complete when its requested output exists (or is explicitly marked a
    physical zero by the declaration) and every numerical output is finite;
    positive auto-spectrum checks use a scale-aware roundoff allowance.
    """

    availability = {
        canonical_cmb_spectrum_name(name): str(status)
        for name, status in (spectrum_availability or {}).items()
    }
    requested = tuple(
        dict.fromkeys(
            canonical_cmb_spectrum_name(name) for name in requested_spectra
        )
    )
    outputs = {
        canonical_cmb_spectrum_name(name): numpy.asarray(values)
        for name, values in output_spectra.items()
    }
    base_outputs = {
        canonical_cmb_spectrum_name(name): numpy.asarray(values)
        for name, values in unlensed_spectra.items()
    }
    ell_values = numpy.asarray(ell_grid, dtype=int)
    k_values = numpy.asarray(k_grid, dtype=float)
    if ell_values.ndim != 1 or k_values.ndim != 1:
        raise ValueError("Post-processing grids must be one-dimensional")
    if not numpy.all(numpy.isfinite(k_values)):
        raise ValueError("Post-processing k grid must be finite")

    surface_records = {
        name: _surface_record(
            name,
            values,
            expected_length=int(ell_values.size),
        )
        for name, values in sorted(outputs.items())
    }
    missing: list[str] = []
    resolved_requested: dict[str, str] = {}
    for name in requested:
        if name in outputs:
            resolved_requested[name] = name
            continue
        lensed_name, _component, base_name = split_cmb_spectrum_name(name)
        fallback = ("lensed_" if lensed_name else "") + base_name
        if fallback in outputs:
            resolved_requested[name] = fallback
            continue
        if availability.get(name) == "physical_zero":
            continue
        missing.append(name)

    transfer_digests = {
        str(name): _array_digest(values)
        for name, values in sorted(transfer_components.items())
    }
    unlensed_digests = {
        str(name): _array_digest(values)
        for name, values in sorted(base_outputs.items())
    }
    output_digests = {
        name: record["sha256"] for name, record in surface_records.items()
    }
    surface_units = {
        name: record["units"] for name, record in surface_records.items()
    }
    intermediates = {
        "ell_grid_sha256": _array_digest(ell_values),
        "k_grid_sha256": _array_digest(k_values),
        "transfer_components": transfer_digests,
        "unlensed_spectra": unlensed_digests,
        "public_spectra": output_digests,
    }
    dependencies = {
        name: _surface_dependencies(resolved_requested.get(name, name))
        for name in requested
    }
    dependencies.update(
        {
            name: _surface_dependencies(name)
            for name in outputs
            if name not in dependencies
        }
    )
    issues = [
        f"missing requested surface(s): {', '.join(missing)}"
        for missing in (missing,)
        if missing
    ]
    invalid = [
        name
        for name, record in surface_records.items()
        if not bool(record["accepted"])
    ]
    if invalid:
        issues.append(
            "invalid post-processed surface(s): " + ", ".join(invalid)
        )
    physical_zero = tuple(
        sorted(
            name
            for name, status in availability.items()
            if status == "physical_zero"
        )
    )
    payload: dict[str, Any] = {
        "schema_version": POST_PROCESSING_SCHEMA_VERSION,
        "requested_surfaces": requested,
        "computed_surfaces": tuple(sorted(outputs)),
        "surface_dependencies": dependencies,
        "surface_records": surface_records,
        "surface_units": surface_units,
        "physical_zero_surfaces": physical_zero,
        "lensing": {
            "applied": bool(lensed),
            "input_surfaces": _LENSING_INPUTS if lensed else (),
            "cache": dict(lensing_cache or {}),
        },
        "intermediates": intermediates,
        "raw_transfer_components": transfer_digests,
        "raw_unlensed_spectra": unlensed_digests,
        "raw_public_spectra": output_digests,
        "ell_count": int(ell_values.size),
        "k_count": int(k_values.size),
        "issues": tuple(issues),
        "complete": not missing,
        "accepted": not issues,
    }
    digest_payload = _canonical_json(payload)
    digest_bytes = json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(digest_bytes).hexdigest()
    return payload


__all__ = [
    "POST_PROCESSING_SCHEMA_VERSION",
    "build_postprocessing_evidence",
]
