"""Physical conservation and scalar-constraint validation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy

from ....perturbation_contract import _evaluate_compiled_expression_noerr
from ..errors import ConstraintViolationError, NonFiniteEvolutionError
from .background import _coerce_numeric_scalar
from .evolution import _scalar_einstein_constraint_metrics

_SCALAR_CONSTRAINT_RESIDUALS = (
    "einstein_energy_residual",
    "einstein_momentum_residual",
    "einstein_shear_residual",
)
_DEFAULT_SCALAR_CONSTRAINT_ANCHORS = {
    "early": 0.05,
    "recombination": 0.50,
    "late": 0.95,
}
_DEFAULT_SCALAR_CONSTRAINT_TOLERANCES = {
    "einstein_energy_residual": 1.0e-3,
    "einstein_momentum_residual": 1.0e-6,
    "einstein_shear_residual": 1.0e-6,
}


def _validate_declared_conservation_rules(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    k_value: float,
    rule_names: Iterable[str] | None = None,
) -> None:
    """Raise when selected declared conservation rules exceed tolerance."""

    def _resolve_rule_dependency(
        dependency_name: str,
        *,
        local_context: dict[str, Any],
        visiting: set[str],
    ) -> bool:
        """Resolve one declared value dependency into ``local_context``."""

        if dependency_name in local_context:
            return True
        if dependency_name in visiting:
            return False
        visiting.add(dependency_name)
        relation_entries = {
            entry.target: entry
            for entry in perturbation_data.constraints.values()
        }
        relation_entries.update(
            {
                entry.target: entry
                for entry in perturbation_data.closures.values()
            }
        )
        candidate_entry = perturbation_data.derived.get(dependency_name)
        if candidate_entry is None:
            candidate_entry = getattr(
                perturbation_data,
                "interactions",
                {},
            ).get(dependency_name)
        if candidate_entry is None:
            candidate_entry = getattr(
                perturbation_data,
                "collision_operators",
                {},
            ).get(dependency_name)
        if candidate_entry is None:
            candidate_entry = relation_entries.get(dependency_name)
        compiled_expression = getattr(
            candidate_entry,
            "compiled_expression",
            None,
        )
        if compiled_expression is None:
            visiting.discard(dependency_name)
            return False
        dependencies = tuple(
            getattr(candidate_entry, "dependencies", ()) or ()
        )
        for child_name in dependencies:
            if child_name in local_context:
                continue
            if not _resolve_rule_dependency(
                str(child_name),
                local_context=local_context,
                visiting=visiting,
            ):
                visiting.discard(dependency_name)
                return False
        local_context[dependency_name] = _evaluate_compiled_expression_noerr(
            compiled_expression,
            local_context,
        )
        visiting.discard(dependency_name)
        return True

    rule_entries = getattr(perturbation_data, "conservation_rules", {}) or {}
    if not rule_entries:
        return
    resolved_context = dict(context)
    selected_rule_names = (
        None if rule_names is None else {str(name) for name in rule_names}
    )
    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for rule_name, rule_entry in rule_entries.items():
            if (
                selected_rule_names is not None
                and str(rule_name) not in selected_rule_names
            ):
                continue
            rule_kind = str(rule_entry.kind or "absolute_max")
            if rule_kind != "absolute_max":
                raise ValueError(
                    "Declared conservation rule uses unsupported kind "
                    f"'{rule_kind}': {rule_name}"
                )
            for dependency_name in tuple(rule_entry.dependencies or ()):
                if dependency_name in resolved_context:
                    continue
                _resolve_rule_dependency(
                    str(dependency_name),
                    local_context=resolved_context,
                    visiting=set(),
                )
            residual = numpy.asarray(
                _evaluate_compiled_expression_noerr(
                    rule_entry.compiled_expression,
                    resolved_context,
                ),
                dtype=float,
            )
            if not numpy.all(numpy.isfinite(residual)):
                raise ValueError(
                    "Declared conservation rule produced non-finite values: "
                    f"{rule_name} at k={k_value}"
                )
            max_abs_residual = float(numpy.max(numpy.abs(residual)))
            tolerance = float(rule_entry.tolerance)
            if max_abs_residual > tolerance:
                raise ValueError(
                    "Declared conservation rule exceeded tolerance: "
                    f"{rule_name} at k={k_value} "
                    f"({max_abs_residual} > {tolerance})"
                )


def _scalar_constraint_physical_regime(
    *,
    context: Mapping[str, Any],
    eta_values: numpy.ndarray,
    index: int,
    anchors: Mapping[str, float],
) -> tuple[str, float]:
    """Classify one residual maximum by background regime and grid fraction."""

    eta_span = max(float(eta_values[-1] - eta_values[0]), 1.0e-30)
    grid_fraction = float((eta_values[index] - eta_values[0]) / eta_span)
    visibility = context.get("visibility")
    if visibility is not None:
        visibility_values = numpy.asarray(visibility, dtype=float)
        if visibility_values.shape == eta_values.shape:
            visibility_peak = float(
                numpy.max(numpy.abs(visibility_values), initial=0.0)
            )
            if (
                visibility_peak > 0.0
                and abs(float(visibility_values[index]))
                >= 0.1 * visibility_peak
            ):
                return "recombination", grid_fraction
    scale_factor = context.get("a")
    if scale_factor is not None:
        scale_values = numpy.asarray(scale_factor, dtype=float)
        if scale_values.shape == eta_values.shape:
            value = float(scale_values[index])
            if numpy.isfinite(value) and value <= 3.0e-4:
                return "radiation", grid_fraction
            if numpy.isfinite(value) and value < 0.75:
                return "matter", grid_fraction
            if numpy.isfinite(value):
                return "late", grid_fraction
    anchor_name = min(
        anchors,
        key=lambda name: abs(float(anchors[name]) - grid_fraction),
    )
    return str(anchor_name), grid_fraction


def _validate_scalar_constraint_histories(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    eta_grid: numpy.ndarray,
    accuracy_controls: Mapping[str, Any],
    k_value: float,
) -> dict[str, dict[str, Any]]:
    """Validate normalized scalar residuals with convergence provenance."""

    residual_names = tuple(
        name for name in _SCALAR_CONSTRAINT_RESIDUALS if name in context
    )
    if not residual_names:
        return {}
    eta_values = numpy.asarray(eta_grid, dtype=float)
    if eta_values.ndim != 1 or eta_values.size == 0:
        raise ValueError("Scalar constraint validation requires an eta grid")
    declared_normalization = accuracy_controls.get(
        "scalar_constraint_normalization",
        "sum_abs_declared_einstein_terms",
    )
    if declared_normalization != "sum_abs_declared_einstein_terms":
        raise ValueError(
            "Scalar constraint normalization must be "
            "'sum_abs_declared_einstein_terms'"
        )
    raw_reference_count = accuracy_controls.get(
        "scalar_constraint_reference_eta_samples"
    )
    if raw_reference_count is None:
        reference_count = int(eta_values.size)
    else:
        reference_count = int(
            _coerce_numeric_scalar(
                raw_reference_count,
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "scalar_constraint_reference_eta_samples"
                ),
            )
        )
        if reference_count < 1:
            raise ValueError(
                "Scalar constraint reference eta samples must be positive"
            )
    reference_resolution_met = eta_values.size >= reference_count

    raw_anchors = accuracy_controls.get("scalar_constraint_anchors")
    if raw_anchors is None:
        anchors = dict(_DEFAULT_SCALAR_CONSTRAINT_ANCHORS)
    elif isinstance(raw_anchors, Mapping):
        anchors = {}
        for anchor_name, raw_fraction in raw_anchors.items():
            fraction = _coerce_numeric_scalar(
                raw_fraction,
                name=(
                    "cmb.perturbations.accuracy_controls."
                    f"scalar_constraint_anchors.{anchor_name}"
                ),
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    "Scalar constraint anchor fractions must lie in [0, 1]"
                )
            anchors[str(anchor_name)] = float(fraction)
    else:
        raise ValueError(
            "cmb.perturbations.accuracy_controls."
            "scalar_constraint_anchors must be a mapping"
        )
    if not anchors:
        raise ValueError("Scalar constraint anchors must not be empty")

    default_tolerances = dict(_DEFAULT_SCALAR_CONSTRAINT_TOLERANCES)
    rule_tolerances: dict[str, float] = {}
    for _rule_name, rule_entry in (
        getattr(perturbation_data, "conservation_rules", {}) or {}
    ).items():
        expression = str(getattr(rule_entry, "expression", ""))
        if expression in default_tolerances:
            rule_tolerances[expression] = float(rule_entry.tolerance)
    accuracy_tolerances: dict[str, float] = {}
    raw_tolerances = accuracy_controls.get("scalar_constraint_tolerances")
    if raw_tolerances is not None:
        if not isinstance(raw_tolerances, Mapping):
            raise ValueError(
                "cmb.perturbations.accuracy_controls."
                "scalar_constraint_tolerances must be a mapping"
            )
        for residual_name, raw_tolerance in raw_tolerances.items():
            residual_key = str(residual_name)
            if residual_key not in default_tolerances:
                raise ValueError(
                    "Unknown scalar constraint tolerance: " f"{residual_key}"
                )
            tolerance = _coerce_numeric_scalar(
                raw_tolerance,
                name=(
                    "cmb.perturbations.accuracy_controls."
                    f"scalar_constraint_tolerances.{residual_key}"
                ),
            )
            if tolerance <= 0.0:
                raise ValueError(
                    "Scalar constraint tolerances must be positive"
                )
            accuracy_tolerances[residual_key] = float(tolerance)

    diagnostics: dict[str, dict[str, Any]] = {}
    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    strict_generated_graph = bool(
        (
            manifest_summary.get("generated_scalar_source_closure", {}) or {}
        ).get("status")
        == "validated"
    )
    for residual_name in residual_names:
        metrics = _scalar_einstein_constraint_metrics(
            context,
            residual_name,
            strict=strict_generated_graph,
        )
        values = numpy.asarray(metrics["residual_values"], dtype=float)
        normalized_values = numpy.asarray(
            metrics["normalized_values"],
            dtype=float,
        )
        normalization_scale = numpy.asarray(
            metrics["normalization_scale"],
            dtype=float,
        )
        if values.ndim == 0:
            values = numpy.full_like(eta_values, float(values), dtype=float)
            normalized_values = numpy.full_like(
                eta_values,
                float(normalized_values),
                dtype=float,
            )
            normalization_scale = numpy.full_like(
                eta_values,
                float(normalization_scale),
                dtype=float,
            )
        if values.shape != eta_values.shape:
            raise ValueError(
                "Scalar Einstein residual has an invalid eta-grid shape: "
                f"{residual_name} at k={k_value}"
            )
        if not (
            numpy.all(numpy.isfinite(values))
            and numpy.all(numpy.isfinite(normalized_values))
            and numpy.all(numpy.isfinite(normalization_scale))
        ):
            raise NonFiniteEvolutionError(
                "Scalar Einstein residual is non-finite: "
                f"{residual_name} at k={k_value}",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(k_value),
                    "residual": residual_name,
                    "normalization_source": str(
                        metrics["normalization_source"]
                    ),
                },
            )
        absolute_values = numpy.abs(values)
        max_abs = float(numpy.max(absolute_values))
        maximum_absolute_index = int(numpy.argmax(absolute_values))
        maximum_normalized_index = int(numpy.argmax(normalized_values))
        maximum_normalized = float(normalized_values[maximum_normalized_index])
        maximum_regime, maximum_grid_fraction = (
            _scalar_constraint_physical_regime(
                context=context,
                eta_values=eta_values,
                index=maximum_normalized_index,
                anchors=anchors,
            )
        )
        term_values = {
            name: numpy.asarray(value, dtype=float)
            for name, value in metrics["term_values"].items()
        }
        term_values_at_maximum = {
            name: float(
                values if values.ndim == 0 else value[maximum_normalized_index]
            )
            for name, value in term_values.items()
        }
        if residual_name in rule_tolerances:
            tolerance = float(rule_tolerances[residual_name])
            tolerance_source = "conservation_rule"
            tolerance_kind = "absolute"
            enforcement_active = True
            enforcement_value = max_abs
        elif residual_name in accuracy_tolerances:
            tolerance = float(accuracy_tolerances[residual_name])
            tolerance_source = "accuracy_controls.scalar_constraint_tolerances"
            tolerance_kind = "normalized"
            enforcement_active = bool(reference_resolution_met)
            enforcement_value = maximum_normalized
        else:
            tolerance = float(default_tolerances[residual_name])
            tolerance_source = "declared_default_unenforced"
            tolerance_kind = "normalized"
            enforcement_active = False
            enforcement_value = maximum_normalized
        anchor_values = {
            anchor_name: float(
                absolute_values[
                    min(
                        int(round(fraction * (eta_values.size - 1))),
                        eta_values.size - 1,
                    )
                ]
            )
            for anchor_name, fraction in anchors.items()
        }
        normalized_anchor_values = {
            anchor_name: float(
                normalized_values[
                    min(
                        int(round(fraction * (eta_values.size - 1))),
                        eta_values.size - 1,
                    )
                ]
            )
            for anchor_name, fraction in anchors.items()
        }
        resolution_status = (
            "reference" if reference_resolution_met else "under_resolved"
        )
        refinement_evidence = {
            "source": "scalar_constraint_reference_eta_samples",
            "reference_eta_samples": int(reference_count),
            "evaluated_eta_samples": int(values.size),
            "reference_resolution_met": bool(reference_resolution_met),
            "resolution_status": resolution_status,
        }
        if enforcement_active and enforcement_value > tolerance:
            raise ConstraintViolationError(
                "Scalar Einstein constraint exceeded tolerance: "
                f"{residual_name} at k={k_value} "
                f"({enforcement_value} > {tolerance})",
                context={
                    "eta": float(eta_values[maximum_normalized_index]),
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(k_value),
                    "maximum_absolute": max_abs,
                    "maximum_normalized": maximum_normalized,
                    "normalization_scale": float(
                        normalization_scale[maximum_normalized_index]
                    ),
                    "normalization_terms": term_values_at_maximum,
                    "normalization_source": str(
                        metrics["normalization_source"]
                    ),
                    "physical_regime": maximum_regime,
                    "residual": residual_name,
                    "tolerance": float(tolerance),
                    "tolerance_kind": tolerance_kind,
                    "tolerance_provenance": tolerance_source,
                    "tolerance_source": tolerance_source,
                    "resolution_status": resolution_status,
                    "refinement_evidence": refinement_evidence,
                },
            )
        diagnostics[residual_name] = {
            "maximum_absolute": max_abs,
            "maximum_absolute_eta": float(eta_values[maximum_absolute_index]),
            "maximum_normalized": maximum_normalized,
            "maximum_eta": float(eta_values[maximum_normalized_index]),
            "maximum_grid_fraction": maximum_grid_fraction,
            "physical_regime": maximum_regime,
            "normalization_scale": float(
                normalization_scale[maximum_normalized_index]
            ),
            "normalization_terms": term_values_at_maximum,
            "normalization_source": str(metrics["normalization_source"]),
            "tolerance": float(tolerance),
            "tolerance_kind": tolerance_kind,
            "tolerance_provenance": tolerance_source,
            "tolerance_source": tolerance_source,
            "enforced": enforcement_active,
            "reference_eta_samples": int(reference_count),
            "reference_resolution_met": bool(reference_resolution_met),
            "resolution_status": resolution_status,
            "physical_judgement": (
                "evaluated" if enforcement_active else "deferred"
            ),
            "refinement_evidence": refinement_evidence,
            "anchors": anchor_values,
            "normalized_anchors": normalized_anchor_values,
            "sample_count": int(values.size),
        }
    return diagnostics
