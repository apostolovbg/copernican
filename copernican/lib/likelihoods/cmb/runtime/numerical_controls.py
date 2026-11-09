"""Numerical-grid and runtime-work controls for CCMBS execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy

from .background import (
    _accuracy_control_positive_int,
    _accuracy_control_value,
    _coerce_numeric_scalar,
    _resolve_custom_cmb_numerics,
    _resolve_declared_accuracy_controls,
)
from .convergence import RUNTIME_WORK_LIMIT_NAMES

_EVOLUTION_WORK_CELL_BUDGET = 16_000_000
_WORK_ESTIMATE_VERSION = 1


def group_mode_schedules(rows, *, maximum_modes=256, work_multiplier=2.0):
    """Batch compatible meshes while bounding total extra integration work.

    Every interval uses the largest required schedule among its modes.
    Bound both total work and each interval's increase. The latter keeps
    batched collision splitting close to independently scheduled scalar
    trajectories; a total bound alone can conceal a large local change.
    """

    if (
        int(maximum_modes) < 1
        or not numpy.isfinite(work_multiplier)
        or work_multiplier < 1
    ):
        raise ValueError(
            "Mode grouping needs a positive width and work factor"
        )
    groups = []
    for row in rows:
        schedule = numpy.asarray(row["schedule"], dtype=numpy.int64)
        eta = numpy.asarray(row["eta"], dtype=numpy.float64)
        if (
            schedule.ndim != 1
            or schedule.size != eta.size - 1
            or numpy.any(schedule < 1)
        ):
            raise ValueError(
                "Mode schedules must match their positive mesh intervals"
            )
        signature = eta.tobytes()
        native_work = int(numpy.sum(schedule))
        selected = None
        for group in groups:
            if (
                len(group["indices"]) >= maximum_modes
                or group["eta_signature"] != signature
            ):
                continue
            combined = numpy.maximum(group["schedule_max"], schedule)
            if numpy.any(combined > work_multiplier * schedule):
                continue
            if numpy.any(combined > work_multiplier * group["schedule_min"]):
                continue
            actual_work = (len(group["indices"]) + 1) * int(
                numpy.sum(combined)
            )
            if actual_work <= work_multiplier * (
                group["native_work"] + native_work
            ):
                selected = group
                break
        if selected is None:
            selected = {
                "indices": [],
                "k_values": [],
                "grids": row["grids"],
                "eta_signature": signature,
                "schedule_max": schedule.copy(),
                "schedule_min": schedule.copy(),
                "native_work": 0,
            }
            groups.append(selected)
        selected["indices"].append(int(row["index"]))
        selected["k_values"].append(float(row["k"]))
        selected["schedule_max"] = numpy.maximum(
            selected["schedule_max"], schedule
        )
        selected["native_work"] += native_work
        selected["schedule_min"] = numpy.minimum(
            selected["schedule_min"], schedule
        )
    return groups


def resolve_background_numerics(contract, numerics):
    """Resolve the physical background seed independently of LOS sampling."""

    minimum = _accuracy_control_positive_int(
        _resolve_declared_accuracy_controls(contract),
        "background_minimum_nodes",
    )
    if minimum is None:
        return numerics
    return replace(
        numerics, eta_sample_count=max(int(numerics.eta_sample_count), minimum)
    )


def promote_failed_resolution(
    contract: Mapping[str, Any], error: Any, *, attempts=()
):
    """Promote only the measured failing engine axis, preserving tolerances."""

    labels = {
        "scalar evolution history": "evolution",
        "source-history": "source",
        "line-of-sight projection": "projection",
        "transfer k-grid": "k",
        "transfer-k-grid": "k",
        "source-k interpolation": "k",
        "angular k-grid": "angular_k",
        "k-domain": "physical_limits",
        "momentum-q": "momentum_q",
        "hierarchy-depth": "hierarchy_depth",
        "background": "background",
    }
    evidence = dict(error.evidence)
    axis = labels.get(error.label)
    if error.label == "observable-error-budget":
        contributors = {}
        for product in evidence["products"].values():
            for name, bound in product["axis_relative_bounds"].items():
                contributors[name] = max(
                    contributors.get(name, 0.0),
                    float(bound) / float(product["relative_tolerance"]),
                )
        # A larger contribution is not necessarily a useful next trial.
        # Observe the effect of earlier axis changes before spending another
        # full solve on an axis whose measured bound did not improve.
        trials = [
            row for row in attempts if "observable_axis_contributions" in row
        ]
        unproductive = set()
        following = [
            *trials[1:],
            {"observable_axis_contributions": contributors},
        ]
        for trial, after in zip(trials, following):
            name = trial["axis"]
            before = trial["observable_axis_contributions"][name]
            measured = after["observable_axis_contributions"][name]
            if measured >= 0.9 * before:
                unproductive.add(name)
            else:
                unproductive.discard(name)
        # Stagnation compares the actual joint-budget violation, not one
        # selected contribution that may stay flat while other errors fall.
        severity = max(
            max(
                float(product["relative_error_bound"]),
                float(product.get("normalized_cross_bound", 0.0)),
            )
            / float(product["relative_tolerance"])
            for product in evidence["products"].values()
        )
        # Cross normalization is nonlinear: its deficit is not in the
        # same scale as these per-product peak contributions. Rank material
        # alternatives relative to the measured contributors themselves.
        material = max(contributors.values()) / len(contributors)
        candidates = {
            name: value
            for name, value in contributors.items()
            if value > 0 and value >= material and name not in unproductive
        } or contributors
        axis = max(candidates, key=candidates.get)
        evidence["observable_axis_contributions"] = contributors
        evidence["unproductive_budget_axes"] = sorted(unproductive)
        evidence["relative_error"] = severity
    if axis is None:
        return None
    evidence["failure_label"] = error.label
    return refine_numerical_axis(contract, axis, evidence=evidence)


def refine_numerical_axis(contract, axis, *, evidence=None):
    """Construct a genuinely finer request for one numerical axis."""

    evidence = dict(evidence or {})
    updated = dict(contract)
    plan = dict(updated["_engine_numerical_plan"])
    overrides = dict(updated.get("_numerical_overrides", {}))
    controls = dict(_resolve_declared_accuracy_controls(updated))
    actual = _resolve_custom_cmb_numerics(updated)
    if axis == "hierarchy_depth":
        for name in (
            "photon_hierarchy_l_max",
            "photon_polarization_hierarchy_l_max",
            "neutrino_hierarchy_l_max",
            "massive_neutrino_hierarchy_l_max",
        ):
            previous = int(getattr(actual, name))
            plan[name] = overrides[name] = max(
                previous + 4, (3 * previous + 1) // 2
            )
    elif axis == "momentum_q":
        plan["momentum_grids"] = {
            name: {
                **dict(grid),
                "count": 2 * int(grid["count"]) + 1,
                "q_min": float(grid["q_min"]) / 4,
                "q_max": max(float(grid["q_max"]) + 8, 32.0),
            }
            for name, grid in plan["momentum_grids"].items()
        }
    elif axis == "angular_k":
        previous = int(controls.get("angular_k_refinement_factor", 1))
        controls["angular_k_refinement_factor"] = 2 * previous
        evidence["previous_angular_factor"] = previous
        evidence["next_angular_factor"] = 2 * previous
    elif axis == "physical_limits":
        previous_minimum = float(actual.k_min)
        previous_maximum = float(actual.k_max)
        plan["k_min"] = overrides["k_min"] = previous_minimum / 2
        plan["k_max"] = overrides["k_max"] = 2 * previous_maximum
        evidence["previous_domain"] = [previous_minimum, previous_maximum]
        evidence["next_domain"] = [previous_minimum / 2, 2 * previous_maximum]
    elif axis == "background":
        previous = max(
            int(actual.eta_sample_count),
            int(controls.get("background_minimum_nodes", 0)),
            int(evidence.get("background_resolution_seed", 0)),
        )
        controls["background_minimum_nodes"] = 2 * previous
        evidence["background_minimum_nodes"] = previous
        evidence["next_background_minimum_nodes"] = 2 * previous
    elif axis in {"source", "projection"}:
        transfer = controls.get("adaptive_transfer")
        independent_grids = isinstance(transfer, Mapping) and (
            transfer.get("algorithm") == "independent_source_interpolation"
        )
        names = tuple(
            name
            for name in ("adaptive_source", "adaptive_projection")
            if isinstance(controls.get(name), Mapping)
            and (not independent_grids or name == "adaptive_" + axis)
        ) or ("adaptive_" + axis,)
        previous = max(
            int(actual.eta_sample_count),
            *(
                int(
                    (
                        controls[name]
                        if isinstance(controls.get(name), Mapping)
                        else {}
                    ).get("minimum_nodes", 0)
                )
                for name in names
            ),
        )
        # A shared LOS ladder doubles its coarse floor. Independent source
        # representation instead doubles the actual measured sample count;
        # its radial quadrature and physical background stay independent.
        observed = int(
            evidence.get(
                (
                    "projection_eta_sample_count"
                    if axis == "projection" and independent_grids
                    else "source_eta_sample_count"
                ),
                0,
            )
        )
        next_count = max(
            2 * previous,
            (
                2 * observed
                if independent_grids and axis == "source"
                else observed
            ),
        )
        for name in names:
            section = controls.get(name)
            settings = dict(section) if isinstance(section, Mapping) else {}
            settings["minimum_nodes"] = next_count
            settings["maximum_nodes"] = max(
                int(settings.get("maximum_nodes", 0)), 2 * next_count
            )
            controls[name] = settings
        evidence["source_minimum_nodes"] = previous
        evidence["next_source_minimum_nodes"] = next_count
    else:
        field = {
            "evolution": "evolution_eta_sample_count",
            "k": "k_sample_count",
        }[axis]
        previous = int(getattr(actual, field))
        next_count = 2 * previous
        if axis == "k":
            next_count = max(
                next_count, 2 * int(evidence.get("base_count", 0))
            )
            if evidence.get("k_grid"):
                updated["_k_grid_refinement_anchors"] = tuple(
                    evidence["k_grid"]
                )
        plan[field] = overrides[field] = next_count
        for name in {
            "evolution": ("adaptive_evolution",),
            "k": ("adaptive_transfer",),
        }[axis]:
            settings = dict(controls.get(name, {}))
            settings["maximum_nodes"] = max(
                int(settings.get("maximum_nodes", 0)), 2 * next_count
            )
            if axis != "evolution":
                settings["minimum_nodes"] = max(
                    int(settings.get("minimum_nodes", 0)), next_count
                )
            controls[name] = settings
        evidence[field] = previous
        evidence["next_" + field] = next_count
    updated["_engine_numerical_plan"] = plan
    updated["_numerical_overrides"] = overrides
    updated["_engine_accuracy_controls"] = controls
    evidence["axis"] = axis
    evidence.setdefault(
        "relative_error",
        max(
            (
                float(value.get("relative_error", 0.0))
                for value in evidence.values()
                if isinstance(value, Mapping)
            ),
            default=0.0,
        ),
    )
    evidence.setdefault("absolute_error", 0.0)
    return updated, evidence


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
