"""Adaptive grids and convergence diagnostics for native CMB projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy


@dataclass(frozen=True, slots=True)
class NativeAdaptiveControls:
    """Validated physical refinement controls for one native CMB request."""

    transfer_enabled: bool = False
    transfer_relative_tolerance: float = 5.0e-2
    transfer_absolute_tolerance: float = 1.0e-12
    transfer_minimum_nodes: int = 0
    transfer_maximum_nodes: int = 0
    transfer_maximum_refinements: int = 1
    source_enabled: bool = False
    source_relative_tolerance: float = 5.0e-2
    source_absolute_tolerance: float = 1.0e-12
    source_minimum_nodes: int = 0
    source_maximum_nodes: int = 0
    source_maximum_refinements: int = 1
    projection_enabled: bool = False
    projection_relative_tolerance: float = 5.0e-2
    projection_absolute_tolerance: float = 1.0e-12
    projection_minimum_nodes: int = 0
    projection_maximum_nodes: int = 0
    projection_maximum_refinements: int = 1
    evolution_enabled: bool = False
    evolution_relative_tolerance: float = 1.0e-2
    evolution_absolute_tolerance: float = 1.0e-12
    evolution_minimum_nodes: int = 0
    evolution_maximum_nodes: int = 0
    evolution_maximum_refinements: int = 1
    phase_points_per_cycle: float = 8.0
    fail_on_nonconvergence: bool = True


@dataclass(frozen=True, slots=True)
class NativeConvergenceEstimate:
    """Maximum absolute and relative difference between two approximations."""

    absolute_error: float
    relative_error: float
    converged: bool


@dataclass(frozen=True, slots=True)
class NativeHistoryConvergence:
    """Convergence errors for state or source histories at physical anchors."""

    absolute_error: float
    relative_error: float
    anchor_absolute_errors: Mapping[str, float]
    anchor_relative_errors: Mapping[str, float]
    converged: bool


def _positive_float(value: Any, *, name: str) -> float:
    """Return one finite positive control value."""

    numeric = float(numpy.asarray(value, dtype=float))
    if not numpy.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    """Return one integer control value no smaller than ``minimum``."""

    numeric = float(numpy.asarray(value, dtype=float))
    if not numpy.isfinite(numeric) or int(numeric) != numeric:
        raise ValueError(f"{name} must be a finite integer")
    result = int(numeric)
    if result < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _section(
    controls: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Mapping[str, Any] | None:
    """Return one adaptive subsection, accepting its explicit aliases."""

    value = controls.get(name)
    if value is None:
        for alias in aliases:
            value = controls.get(alias)
            if value is not None:
                break
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"cmb.perturbations.accuracy_controls.{name} must be a mapping"
        )
    return value


def _read_section_values(
    section: Mapping[str, Any],
    *,
    name: str,
    base_nodes: int,
    default_maximum_nodes: int,
) -> tuple[bool, float, float, int, int, int]:
    """Resolve one adaptive section into validated scalar controls."""

    enabled = bool(section.get("enabled", True))
    relative_tolerance = _positive_float(
        section.get("relative_tolerance", 5.0e-2),
        name=f"{name}.relative_tolerance",
    )
    absolute_tolerance = _positive_float(
        section.get("absolute_tolerance", 1.0e-12),
        name=f"{name}.absolute_tolerance",
    )
    minimum_nodes = _positive_int(
        section.get("minimum_nodes", max(4, int(base_nodes))),
        name=f"{name}.minimum_nodes",
        minimum=4,
    )
    maximum_nodes = _positive_int(
        section.get(
            "maximum_nodes",
            max(int(minimum_nodes), int(default_maximum_nodes)),
        ),
        name=f"{name}.maximum_nodes",
        minimum=minimum_nodes,
    )
    maximum_refinements = _positive_int(
        section.get("maximum_refinements", 1),
        name=f"{name}.maximum_refinements",
    )
    return (
        enabled,
        relative_tolerance,
        absolute_tolerance,
        minimum_nodes,
        maximum_nodes,
        maximum_refinements,
    )


def resolve_native_adaptive_controls(
    accuracy_controls: Mapping[str, Any],
    *,
    base_k_nodes: int,
    base_eta_nodes: int,
    base_evolution_nodes: int | None = None,
) -> NativeAdaptiveControls:
    """Validate the adaptive accuracy sections of a native contract.

    ``adaptive_k_quadrature`` remains an accepted spelling for the transfer
    section so contracts written before the unified controls can migrate
    without changing their numerical meaning.
    """

    controls = accuracy_controls or {}
    transfer = _section(
        controls,
        "adaptive_transfer",
        aliases=("adaptive_k_quadrature",),
    )
    source = _section(
        controls,
        "adaptive_source",
        aliases=("adaptive_source_grid",),
    )
    projection = _section(controls, "adaptive_projection")
    transfer_values = (False, 5.0e-2, 1.0e-12, 0, 0, 1)
    source_values = (False, 5.0e-2, 1.0e-12, 0, 0, 1)
    projection_values = (False, 5.0e-2, 1.0e-12, 0, 0, 1)
    evolution_values = (False, 1.0e-2, 1.0e-12, 0, 0, 1)
    if transfer is not None:
        transfer_values = _read_section_values(
            transfer,
            name="cmb.perturbations.accuracy_controls.adaptive_transfer",
            base_nodes=int(base_k_nodes),
            default_maximum_nodes=max(2 * int(base_k_nodes), 64),
        )
    if source is not None:
        source_values = _read_section_values(
            source,
            name="cmb.perturbations.accuracy_controls.adaptive_source",
            base_nodes=int(base_eta_nodes),
            default_maximum_nodes=max(2 * int(base_eta_nodes), 256),
        )
    if projection is not None:
        projection_enabled = bool(projection.get("enabled", True))
        projection_minimum_nodes = _positive_int(
            projection.get("minimum_nodes", max(4, int(base_eta_nodes))),
            name=(
                "cmb.perturbations.accuracy_controls."
                "adaptive_projection.minimum_nodes"
            ),
            minimum=4,
        )
        projection_values = (
            projection_enabled,
            _positive_float(
                projection.get("relative_tolerance", 5.0e-2),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_projection.relative_tolerance"
                ),
            ),
            _positive_float(
                projection.get("absolute_tolerance", 1.0e-12),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_projection.absolute_tolerance"
                ),
            ),
            projection_minimum_nodes,
            _positive_int(
                projection.get(
                    "maximum_nodes",
                    max(2 * projection_minimum_nodes, 256),
                ),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_projection.maximum_nodes"
                ),
                minimum=projection_minimum_nodes,
            ),
            _positive_int(
                projection.get("maximum_refinements", 1),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_projection.maximum_refinements"
                ),
            ),
        )
    evolution = _section(
        controls,
        "adaptive_evolution",
        aliases=("scalar_evolution_convergence",),
    )
    if evolution is not None:
        evolution_values = _read_section_values(
            evolution,
            name=("cmb.perturbations.accuracy_controls." "adaptive_evolution"),
            base_nodes=int(
                base_eta_nodes
                if base_evolution_nodes is None
                else base_evolution_nodes
            ),
            default_maximum_nodes=max(
                2
                * int(
                    base_eta_nodes
                    if base_evolution_nodes is None
                    else base_evolution_nodes
                ),
                256,
            ),
        )
    phase_points = _positive_float(
        controls.get("phase_points_per_cycle", 8.0),
        name=("cmb.perturbations.accuracy_controls.phase_points_per_cycle"),
    )
    fail_on_nonconvergence = bool(controls.get("fail_on_nonconvergence", True))
    return NativeAdaptiveControls(
        transfer_enabled=bool(transfer_values[0]),
        transfer_relative_tolerance=float(transfer_values[1]),
        transfer_absolute_tolerance=float(transfer_values[2]),
        transfer_minimum_nodes=int(transfer_values[3]),
        transfer_maximum_nodes=int(transfer_values[4]),
        transfer_maximum_refinements=int(transfer_values[5]),
        source_enabled=bool(source_values[0]),
        source_relative_tolerance=float(source_values[1]),
        source_absolute_tolerance=float(source_values[2]),
        source_minimum_nodes=int(source_values[3]),
        source_maximum_nodes=int(source_values[4]),
        source_maximum_refinements=int(source_values[5]),
        projection_enabled=bool(projection_values[0]),
        projection_relative_tolerance=float(projection_values[1]),
        projection_absolute_tolerance=float(projection_values[2]),
        projection_minimum_nodes=int(projection_values[3]),
        projection_maximum_nodes=int(projection_values[4]),
        projection_maximum_refinements=int(projection_values[5]),
        evolution_enabled=bool(evolution_values[0]),
        evolution_relative_tolerance=float(evolution_values[1]),
        evolution_absolute_tolerance=float(evolution_values[2]),
        evolution_minimum_nodes=int(evolution_values[3]),
        evolution_maximum_nodes=int(evolution_values[4]),
        evolution_maximum_refinements=int(evolution_values[5]),
        phase_points_per_cycle=phase_points,
        fail_on_nonconvergence=fail_on_nonconvergence,
    )


def phase_aware_k_grid(
    k_min: float,
    k_max: float,
    *,
    minimum_nodes: int,
    maximum_nodes: int,
    phase_points_per_cycle: float,
    eta_distance: float,
    sound_horizon: float,
    anchors: tuple[float, ...] = (),
) -> numpy.ndarray:
    """Build a bounded logarithmic k grid with physical phase anchors."""

    lower = _positive_float(k_min, name="k_min")
    upper = _positive_float(k_max, name="k_max")
    if upper <= lower:
        raise ValueError("k_max must be greater than k_min")
    minimum = _positive_int(minimum_nodes, name="minimum_nodes", minimum=4)
    maximum = _positive_int(
        maximum_nodes,
        name="maximum_nodes",
        minimum=minimum,
    )
    phase_points = _positive_float(
        phase_points_per_cycle,
        name="phase_points_per_cycle",
    )
    distance = _positive_float(eta_distance, name="eta_distance")
    acoustic_distance = _positive_float(
        sound_horizon,
        name="sound_horizon",
    )
    nodes = list(numpy.geomspace(lower, upper, minimum))
    nodes.extend(float(value) for value in anchors)
    acoustic_phase_step = numpy.pi / phase_points
    for phase_distance in (distance, acoustic_distance):
        phase_count = int(
            numpy.ceil((upper - lower) * phase_distance / acoustic_phase_step)
        )
        phase_count = min(maximum, max(2, phase_count))
        phase_nodes = numpy.linspace(lower, upper, phase_count)
        nodes.extend(float(value) for value in phase_nodes)
    clipped = numpy.clip(numpy.asarray(nodes, dtype=float), lower, upper)
    result = numpy.unique(clipped)
    while result.size < minimum:
        log_values = numpy.log(result)
        gap_index = int(numpy.argmax(numpy.diff(log_values)))
        midpoint = numpy.exp(
            0.5 * (log_values[gap_index] + log_values[gap_index + 1])
        )
        result = numpy.insert(result, gap_index + 1, midpoint)
    if result.size <= maximum:
        return numpy.asarray(result, dtype=float)
    required = {float(result[0]), float(result[-1])}
    required.update(float(value) for value in anchors)
    required = {value for value in required if lower <= value <= upper}
    optional = [
        float(value) for value in result if float(value) not in required
    ]
    budget = max(0, maximum - len(required))
    if len(optional) > budget:
        indices = numpy.linspace(0, len(optional) - 1, budget, dtype=int)
        optional = [optional[int(index)] for index in sorted(set(indices))]
    return numpy.asarray(sorted(required | set(optional)), dtype=float)


def phase_aware_eta_grid(
    eta_grid: numpy.ndarray,
    *,
    visibility: numpy.ndarray,
    k_max: float,
    minimum_nodes: int,
    maximum_nodes: int,
    phase_points_per_cycle: float,
) -> numpy.ndarray:
    """Refine eta around visibility structure and rapid Fourier phase."""

    eta = numpy.asarray(eta_grid, dtype=float)
    visibility_values = numpy.asarray(visibility, dtype=float)
    if eta.ndim != 1 or visibility_values.shape != eta.shape:
        raise ValueError("eta and visibility grids must have matching shapes")
    if eta.size < 2 or not numpy.all(numpy.isfinite(eta)):
        raise ValueError("eta grid must contain finite ordered samples")
    if numpy.any(numpy.diff(eta) <= 0.0):
        raise ValueError("eta grid must be strictly increasing")
    if not numpy.all(numpy.isfinite(visibility_values)):
        raise ValueError("visibility grid must contain finite samples")
    minimum = _positive_int(minimum_nodes, name="minimum_nodes", minimum=4)
    maximum = _positive_int(
        maximum_nodes,
        name="maximum_nodes",
        minimum=minimum,
    )
    phase_step = numpy.pi / (
        _positive_float(phase_points_per_cycle, name="phase_points_per_cycle")
        * _positive_float(k_max, name="k_max")
    )
    peak = max(float(numpy.max(visibility_values)), 1.0e-30)
    visibility_scale = max(peak * 1.0e-4, 1.0e-30)
    result = eta.copy()
    for _ in range(32):
        if result.size >= maximum:
            break
        visibility_on_result = numpy.interp(
            result,
            eta,
            visibility_values,
        )
        steps = numpy.diff(result)
        phase_count = numpy.maximum(
            1, numpy.ceil(steps / phase_step).astype(int)
        )
        visibility_mask = (
            numpy.maximum(visibility_on_result[:-1], visibility_on_result[1:])
            > visibility_scale
        )
        split_count = numpy.where(
            visibility_mask, numpy.maximum(phase_count, 2), phase_count
        )
        remaining = maximum - result.size
        if remaining <= 0:
            break
        split_count = numpy.minimum(split_count, remaining + 1)
        additions: list[numpy.ndarray] = []
        for index, count in enumerate(split_count):
            if int(count) <= 1:
                continue
            additions.append(
                numpy.linspace(
                    result[index],
                    result[index + 1],
                    int(count) + 1,
                    dtype=float,
                )[1:-1]
            )
        if not additions:
            break
        result = numpy.unique(numpy.concatenate((result, *additions)))
    if result.size < minimum:
        indices = numpy.linspace(0, result.size - 1, minimum, dtype=int)
        result = numpy.unique(numpy.concatenate((result, result[indices])))
    if result.size > maximum:
        indices = numpy.linspace(0, result.size - 1, maximum, dtype=int)
        result = result[sorted(set(int(index) for index in indices))]
    return numpy.asarray(result, dtype=float)


def estimate_convergence(
    coarse: numpy.ndarray,
    fine: numpy.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NativeConvergenceEstimate:
    """Compare two finite approximations with absolute and relative floors."""

    coarse_values = numpy.asarray(coarse, dtype=float)
    fine_values = numpy.asarray(fine, dtype=float)
    if coarse_values.shape != fine_values.shape:
        raise ValueError("Convergence approximations must have equal shapes")
    if not numpy.all(numpy.isfinite(coarse_values)) or not numpy.all(
        numpy.isfinite(fine_values)
    ):
        raise ValueError("Convergence approximations must be finite")
    absolute_error = float(
        numpy.max(numpy.abs(fine_values - coarse_values), initial=0.0)
    )
    relative_error = float(
        numpy.max(
            numpy.abs(fine_values - coarse_values)
            / numpy.maximum(numpy.abs(fine_values), absolute_tolerance),
            initial=0.0,
        )
    )
    converged = bool(
        absolute_error <= float(absolute_tolerance)
        or relative_error <= float(relative_tolerance)
    )
    return NativeConvergenceEstimate(
        absolute_error=absolute_error,
        relative_error=relative_error,
        converged=converged,
    )


def estimate_history_convergence(
    coarse_eta: numpy.ndarray,
    coarse_histories: Mapping[str, numpy.ndarray],
    fine_eta: numpy.ndarray,
    fine_histories: Mapping[str, numpy.ndarray],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    anchors: Mapping[str, float] | None = None,
) -> NativeHistoryConvergence:
    """Compare history values at early, recombination, and late anchors."""

    coarse_grid = numpy.asarray(coarse_eta, dtype=float)
    fine_grid = numpy.asarray(fine_eta, dtype=float)
    for label, grid in (("coarse", coarse_grid), ("fine", fine_grid)):
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(f"{label} eta grid must contain two samples")
        if not numpy.all(numpy.isfinite(grid)) or numpy.any(
            numpy.diff(grid) <= 0.0
        ):
            raise ValueError(f"{label} eta grid must be finite and increasing")
    if not coarse_histories or set(coarse_histories) != set(fine_histories):
        raise ValueError("History comparisons require matching named states")
    anchor_positions = anchors or {
        "early": 0.05,
        "recombination": 0.50,
        "late": 0.95,
    }
    absolute_errors: dict[str, float] = {}
    relative_errors: dict[str, float] = {}
    for anchor_name, position in anchor_positions.items():
        fraction = float(position)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"History anchor '{anchor_name}' must be in [0, 1]"
            )
        coarse_eta_value = float(
            coarse_grid[0] + fraction * (coarse_grid[-1] - coarse_grid[0])
        )
        fine_eta_value = float(
            fine_grid[0] + fraction * (fine_grid[-1] - fine_grid[0])
        )
        anchor_absolute = 0.0
        anchor_relative = 0.0
        for name in coarse_histories:
            coarse_values = numpy.asarray(coarse_histories[name], dtype=float)
            fine_values = numpy.asarray(fine_histories[name], dtype=float)
            if coarse_values.shape != coarse_grid.shape or (
                fine_values.shape != fine_grid.shape
            ):
                raise ValueError(
                    f"History '{name}' does not match its eta grid"
                )
            if not numpy.all(numpy.isfinite(coarse_values)) or not numpy.all(
                numpy.isfinite(fine_values)
            ):
                raise ValueError(
                    f"History '{name}' contains non-finite values"
                )
            coarse_value = float(
                numpy.interp(coarse_eta_value, coarse_grid, coarse_values)
            )
            fine_value = float(
                numpy.interp(fine_eta_value, fine_grid, fine_values)
            )
            difference = abs(fine_value - coarse_value)
            anchor_absolute = max(anchor_absolute, difference)
            history_scale = max(
                float(numpy.max(numpy.abs(fine_values), initial=0.0)),
                float(absolute_tolerance),
            )
            anchor_relative = max(
                anchor_relative,
                difference / history_scale,
            )
        absolute_errors[str(anchor_name)] = anchor_absolute
        relative_errors[str(anchor_name)] = anchor_relative
    absolute_error = max(absolute_errors.values(), default=0.0)
    relative_error = max(relative_errors.values(), default=0.0)
    converged = all(
        absolute_errors[name] <= float(absolute_tolerance)
        or relative_errors[name] <= float(relative_tolerance)
        for name in absolute_errors
    )
    return NativeHistoryConvergence(
        absolute_error=float(absolute_error),
        relative_error=float(relative_error),
        anchor_absolute_errors=absolute_errors,
        anchor_relative_errors=relative_errors,
        converged=bool(converged),
    )


def require_convergence(
    estimate: NativeConvergenceEstimate,
    *,
    label: str,
    fail_on_nonconvergence: bool,
) -> None:
    """Raise a named under-resolution error when convergence is required."""

    if estimate.converged or not fail_on_nonconvergence:
        return
    raise ValueError(
        f"Native {label} refinement did not converge: "
        f"relative_error={estimate.relative_error:.6g}, "
        f"absolute_error={estimate.absolute_error:.6g}"
    )
