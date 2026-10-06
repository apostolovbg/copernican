"""Numerical-grid and runtime-work controls for CCMBS execution."""

from __future__ import annotations

from typing import Any, Mapping

import numpy

from .background import (
    _accuracy_control_value,
    _coerce_numeric_scalar,
    _resolve_declared_accuracy_controls,
)
from .convergence import RUNTIME_WORK_LIMIT_NAMES

_EVOLUTION_WORK_CELL_BUDGET = 16_000_000
_WORK_ESTIMATE_VERSION = 1


def _refine_eta_grid(
    eta_grid: numpy.ndarray,
    *,
    refinement: int,
) -> numpy.ndarray:
    """Return ``eta_grid`` refined with midpoint-preserving subdivisions."""

    if refinement <= 1 or eta_grid.size < 2:
        return numpy.asarray(eta_grid, dtype=float)
    subdivisions = max(1, int(refinement))
    left_edges = eta_grid[:-1, numpy.newaxis]
    step_sizes = numpy.diff(eta_grid)[:, numpy.newaxis] / float(subdivisions)
    offsets = numpy.arange(subdivisions, dtype=float)[numpy.newaxis, :]
    refined = (left_edges + step_sizes * offsets).reshape(-1)
    return numpy.unique(
        numpy.concatenate(
            (numpy.asarray(refined, dtype=float), eta_grid[-1:]),
        )
    )


def _densify_eta_grid(
    eta_grid: numpy.ndarray,
    *,
    minimum_samples: int,
) -> numpy.ndarray:
    """Return ``eta_grid`` densified by midpoint insertion up to a minimum."""

    refined = numpy.asarray(eta_grid, dtype=float)
    target_size = max(int(minimum_samples), int(refined.size))
    while refined.size < target_size and refined.size >= 2:
        step_sizes = numpy.diff(refined)
        midpoint_budget = min(
            target_size - refined.size,
            step_sizes.size,
        )
        midpoint_indices = numpy.argsort(step_sizes)[-midpoint_budget:]
        midpoint_values = 0.5 * (
            refined[midpoint_indices] + refined[midpoint_indices + 1]
        )
        refined = numpy.unique(
            numpy.concatenate(
                (
                    refined,
                    numpy.asarray(midpoint_values, dtype=float),
                )
            )
        )
    return numpy.asarray(refined, dtype=float)


def _limit_eta_grid(
    eta_grid: numpy.ndarray,
    maximum_samples: int,
) -> numpy.ndarray:
    """Limit a source grid while retaining its nonuniform spacing."""

    target_size = max(int(maximum_samples), 16)
    if eta_grid.size <= target_size:
        return numpy.asarray(eta_grid, dtype=float)
    source_indices = numpy.linspace(
        0,
        eta_grid.size - 1,
        target_size,
        dtype=int,
    )
    return numpy.asarray(eta_grid[source_indices], dtype=float)


def _validate_runtime_envelope_controls(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return optional runtime-envelope controls without preset ceilings."""

    accuracy_controls = _resolve_declared_accuracy_controls(contract)
    runtime_envelope = accuracy_controls.get("runtime_envelope")
    if runtime_envelope is None:
        return {}
    if runtime_envelope == "bounded":
        return {}
    if not isinstance(runtime_envelope, Mapping):
        raise ValueError(
            "cmb.perturbations.accuracy_controls.runtime_envelope must be "
            "a mapping or the preset 'bounded'"
        )
    return runtime_envelope


def _estimate_runtime_work_units(
    *,
    ell_count: int,
    k_count: int,
    eta_count: int,
    state_slot_count: int,
    transfer_component_count: int,
    momentum_point_count: int,
    evolution_multiplier: int = 1,
) -> dict[str, int]:
    """Estimate deterministic declared work before allocating arrays.

    The estimate is accounting metadata only. Large requests are split into
    ordered chunks rather than rejected by a machine-local budget.
    """

    evolution_work_units = int(
        max(int(evolution_multiplier), 1)
        * max(int(k_count), 0)
        * max(int(eta_count), 0)
        * max(int(state_slot_count), 1)
    )
    projection_work_units = int(
        max(int(ell_count), 0)
        * max(int(k_count), 0)
        * max(int(eta_count), 0)
        * max(int(transfer_component_count), 1)
    )
    momentum_work_units = int(
        max(int(momentum_point_count), 0) * max(int(eta_count), 0)
    )
    return {
        "evolution_work_units": evolution_work_units,
        "projection_work_units": projection_work_units,
        "momentum_work_units": momentum_work_units,
        "total_work_units": int(
            evolution_work_units + projection_work_units + momentum_work_units
        ),
    }


def _resolve_evolution_chunk_size(
    *,
    k_count: int,
    eta_count: int,
    state_slot_count: int,
) -> int:
    """Resolve a deterministic mode chunk that bounds batched state memory."""

    cells_per_mode = max(int(eta_count), 1) * max(int(state_slot_count), 1)
    by_cells = max(_EVOLUTION_WORK_CELL_BUDGET // cells_per_mode, 1)
    return max(1, min(max(int(k_count), 1), by_cells))


def _enforce_runtime_envelope(
    contract: Mapping[str, Any],
    *,
    ell_count: int,
    k_count: int,
    eta_count: int,
    state_slot_count: int,
    transfer_component_count: int,
    momentum_point_count: int,
    evolution_multiplier: int = 1,
) -> dict[str, Any]:
    """Return accounted runtime work and validate malformed controls only."""

    work_units = _estimate_runtime_work_units(
        ell_count=ell_count,
        k_count=k_count,
        eta_count=eta_count,
        state_slot_count=state_slot_count,
        transfer_component_count=transfer_component_count,
        momentum_point_count=momentum_point_count,
        evolution_multiplier=evolution_multiplier,
    )
    envelope = {
        "work_estimate_version": _WORK_ESTIMATE_VERSION,
        "ell_count": int(ell_count),
        "k_sample_count": int(k_count),
        "eta_sample_count": int(eta_count),
        "state_slot_count": int(state_slot_count),
        "transfer_component_count": int(transfer_component_count),
        "momentum_point_count": int(momentum_point_count),
        **work_units,
    }
    runtime_envelope = _validate_runtime_envelope_controls(contract)
    controls = _resolve_declared_accuracy_controls(contract)
    explicit_limits: dict[str, int] = {}
    for limit_name in RUNTIME_WORK_LIMIT_NAMES:
        raw_limit = runtime_envelope.get(limit_name)
        if raw_limit is None:
            raw_limit = _accuracy_control_value(
                controls,
                limit_name,
            )
        if raw_limit is None:
            continue
        limit_value = int(
            _coerce_numeric_scalar(
                raw_limit,
                name=(
                    "cmb.perturbations.accuracy_controls.runtime_envelope."
                    f"{limit_name}"
                ),
            )
        )
        if limit_value < 1:
            raise ValueError(
                "cmb.perturbations.accuracy_controls.runtime_envelope."
                f"{limit_name} must be positive"
            )
        explicit_limits[limit_name] = limit_value
    envelope["work_accounting_mode"] = (
        "explicit_limits" if explicit_limits else "accounted"
    )
    envelope["work_limits"] = explicit_limits
    envelope["work_limits_enforced"] = False
    return envelope
