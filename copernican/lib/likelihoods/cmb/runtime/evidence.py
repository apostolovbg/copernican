"""Stable diagnostics and evidence identities for CCMBS execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy

from .background import CustomCMBSpectrumData


def _canonical_source_history_value(value: Any) -> Any:
    """Normalize source evidence before deterministic JSON hashing."""

    if isinstance(value, numpy.ndarray):
        return _canonical_source_history_value(value.tolist())
    if isinstance(value, numpy.generic):
        return _canonical_source_history_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_source_history_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_source_history_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _canonical_source_history_value(item)
            for item in sorted(value, key=str)
        ]
    return value


def _build_source_history_bundle_digest(
    *,
    source_eta_signature: str,
    source_history_refinement: Mapping[str, Any],
    source_history_residual_samples_by_k: Mapping[str, Any],
    hierarchy_equation_residuals_by_k: Mapping[str, Any],
    initial_state_diagnostics_by_k: Mapping[str, Any],
    metric_history_gradient_residual_by_k: Mapping[str, Any],
    runtime_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the complete generated-source evidence before projection output."""

    evidence_fields = (
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
    )
    payload = {
        "schema_version": 1,
        "source_eta_sha256": source_eta_signature,
        "source_history_residual_sample_schema": int(
            runtime_envelope.get("source_history_residual_sample_schema", 1)
        ),
        "source_history_residual_samples_by_k": (
            source_history_residual_samples_by_k
        ),
        "hierarchy_equation_residuals_by_k": (
            hierarchy_equation_residuals_by_k
        ),
        "initial_state_diagnostics_by_k": initial_state_diagnostics_by_k,
        "metric_history_gradient_residual_by_k": (
            metric_history_gradient_residual_by_k
        ),
        "source_history_refinement": source_history_refinement,
        "declared_source_history_convergence": runtime_envelope.get(
            "declared_source_history_convergence", {}
        ),
        "source_history_derivative_provenance": runtime_envelope.get(
            "source_history_derivative_provenance", {}
        ),
        "source_residual_audit_controls": runtime_envelope.get(
            "source_residual_audit_controls", {}
        ),
        "independent_source_residual_audit": runtime_envelope.get(
            "independent_source_residual_audit", {}
        ),
        "generated_scalar_source_closure": runtime_envelope.get(
            "generated_scalar_source_closure", {}
        ),
    }
    canonical = json.dumps(
        _canonical_source_history_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    sample_count = sum(
        int(values.get("sample_count", 0))
        for values in source_history_residual_samples_by_k.values()
        if isinstance(values, Mapping)
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "source_eta_sha256": source_eta_signature,
        "mode_count": int(len(source_history_residual_samples_by_k)),
        "sample_count": int(sample_count),
        "included_fields": evidence_fields,
    }


def _projection_array_digest(values: Any) -> str:
    """Return a shape-aware digest for one finite projection array."""

    array = numpy.ascontiguousarray(numpy.asarray(values, dtype=numpy.float64))
    if (
        array.ndim != 1
        or array.size == 0
        or not numpy.all(numpy.isfinite(array))
    ):
        raise ValueError(
            "Projection grid evidence must be finite and nonempty"
        )
    header = json.dumps(
        {
            "dtype": str(array.dtype),
            "shape": tuple(int(value) for value in array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _runtime_telemetry_context(
    result: CustomCMBSpectrumData,
) -> dict[str, Any]:
    """Extract compact grid metadata for per-proposal run telemetry."""

    envelope = dict(result.runtime_envelope)
    effective = dict(envelope.get("effective_numerical_controls", {}) or {})
    ell_grid = numpy.asarray(result.ell_grid, dtype=int)
    k_grid = numpy.asarray(result.k_grid, dtype=float)
    return {
        "ell_min": int(ell_grid.min()) if ell_grid.size else None,
        "ell_max": int(ell_grid.max()) if ell_grid.size else None,
        "ell_count": int(ell_grid.size),
        "k_sample_count": int(k_grid.size),
        "eta_sample_count": int(effective.get("eta_sample_count", 0)),
        "accuracy_tier": envelope.get("accuracy_tier"),
        "phase_aware_k_enabled": bool(
            envelope.get("phase_aware_k_enabled", False)
        ),
        "cache_state": envelope.get("cache_state"),
        "resolution_axis_evidence": dict(
            envelope.get("resolution_axis_evidence", {}) or {}
        ),
        "adaptive_errors": {
            name: float(envelope.get(name, 0.0))
            for name in (
                "adaptive_transfer_relative_error",
                "adaptive_source_relative_error",
                "adaptive_projection_relative_error",
                "adaptive_evolution_relative_error",
            )
        },
        "production_scalar_k_convergence": dict(
            envelope.get("production_scalar_k_convergence", {}) or {}
        ),
    }
