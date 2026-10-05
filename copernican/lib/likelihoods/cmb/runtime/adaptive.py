"""Adaptive grids and convergence diagnostics for declared CMB projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy

from ..errors import ConvergenceError


@dataclass(frozen=True, slots=True)
class AdaptiveControls:
    """Validated physical refinement controls for one declared CMB request."""

    transfer_enabled: bool = False
    transfer_relative_tolerance: float = 5.0e-2
    transfer_absolute_tolerance: float = 1.0e-12
    transfer_minimum_nodes: int = 0
    transfer_maximum_nodes: int = 0
    transfer_maximum_refinements: int = 1
    transfer_algorithm: str = "nested_independent_source_projection"
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
    evolution_validation_mode_count: int = 3
    phase_points_per_cycle: float = 8.0
    fail_on_nonconvergence: bool = True


@dataclass(frozen=True, slots=True)
class LOSQuadratureControls:
    """Validated phase-resolution controls for line-of-sight integration."""

    enabled: bool = False
    minimum_nodes: int = 0
    maximum_nodes: int = 0
    phase_points_per_cycle: float = 8.0
    configured_maximum_nodes: int = 0


@dataclass(frozen=True, slots=True)
class ConvergenceEstimate:
    """Maximum absolute and relative difference between two approximations."""

    absolute_error: float
    relative_error: float
    converged: bool


@dataclass(frozen=True, slots=True)
class HistoryConvergence:
    """Convergence errors for histories on a dense common physical surface."""

    absolute_error: float
    relative_error: float
    anchor_absolute_errors: Mapping[str, float]
    anchor_relative_errors: Mapping[str, float]
    sample_count: int
    converged: bool


class AdaptiveNonConvergenceError(ConvergenceError):
    """Typed failure carrying the products that missed an adaptive budget."""

    def __init__(
        self,
        message: str,
        *,
        label: str,
        failed_products: Sequence[str] = (),
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        """Retain machine-readable failure context beside the error text."""

        self.label = str(label)
        self.failed_products = tuple(str(name) for name in failed_products)
        self.evidence = dict(evidence or {})
        super().__init__(
            message,
            context={
                "refinement_label": self.label,
                "failed_products": self.failed_products,
                "refinement_evidence": self.evidence,
            },
        )


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


def resolve_adaptive_controls(
    accuracy_controls: Mapping[str, Any],
    *,
    base_k_nodes: int,
    base_eta_nodes: int,
    base_evolution_nodes: int | None = None,
) -> AdaptiveControls:
    """Validate the adaptive accuracy sections of a declared contract.

    ``adaptive_k_quadrature`` remains an accepted spelling for the transfer
    section when its declared mode is ``transfer``.  Source-mode contracts
    stay on the source path instead of being silently reinterpreted as
    transfer refinement.
    """

    controls = accuracy_controls or {}
    legacy_k_quadrature = _section(controls, "adaptive_k_quadrature")
    legacy_k_mode = (
        "transfer"
        if legacy_k_quadrature is None
        else str(legacy_k_quadrature.get("mode", "transfer")).strip().lower()
    )
    transfer_aliases = (
        ("adaptive_k_quadrature",) if legacy_k_mode == "transfer" else ()
    )
    transfer = _section(
        controls,
        "adaptive_transfer",
        aliases=transfer_aliases,
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
        transfer_algorithm = str(
            transfer.get(
                "algorithm",
                "nested_independent_source_projection",
            )
        ).strip()
        if transfer_algorithm not in {
            "nested_independent_source_projection",
            "diagnostic_transfer_interpolation",
        }:
            raise ValueError(
                "adaptive_transfer.algorithm must select a supported "
                "refinement algorithm"
            )
    else:
        transfer_algorithm = "nested_independent_source_projection"
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
        evolution_validation_mode_count = _positive_int(
            evolution.get("validation_mode_count", 3),
            name=(
                "cmb.perturbations.accuracy_controls."
                "adaptive_evolution.validation_mode_count"
            ),
        )
    else:
        evolution_validation_mode_count = 0
    phase_points = _positive_float(
        controls.get("phase_points_per_cycle", 8.0),
        name=("cmb.perturbations.accuracy_controls.phase_points_per_cycle"),
    )
    fail_on_nonconvergence = bool(controls.get("fail_on_nonconvergence", True))
    return AdaptiveControls(
        transfer_enabled=bool(transfer_values[0]),
        transfer_relative_tolerance=float(transfer_values[1]),
        transfer_absolute_tolerance=float(transfer_values[2]),
        transfer_minimum_nodes=int(transfer_values[3]),
        transfer_maximum_nodes=int(transfer_values[4]),
        transfer_maximum_refinements=int(transfer_values[5]),
        transfer_algorithm=transfer_algorithm,
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
        evolution_validation_mode_count=int(evolution_validation_mode_count),
        phase_points_per_cycle=phase_points,
        fail_on_nonconvergence=fail_on_nonconvergence,
    )


def resolve_los_quadrature_controls(
    accuracy_controls: Mapping[str, Any],
    *,
    base_eta_nodes: int,
) -> LOSQuadratureControls:
    """Resolve the explicit phase-aware line-of-sight grid controls.

    The line-of-sight grid is intentionally opt-in.  A declared production
    contract can request a bounded phase grid without changing low-resolution
    fixtures or silently multiplying their runtime envelope.
    """

    section = _section(accuracy_controls or {}, "los_phase_quadrature")
    if section is None:
        return LOSQuadratureControls()
    enabled = bool(section.get("enabled", True))
    base_nodes = max(4, int(base_eta_nodes))
    minimum_nodes = _positive_int(
        section.get("minimum_nodes", max(base_nodes, 512)),
        name=(
            "cmb.perturbations.accuracy_controls."
            "los_phase_quadrature.minimum_nodes"
        ),
        minimum=4,
    )
    configured_maximum_nodes = _positive_int(
        section.get("maximum_nodes", max(2 * minimum_nodes, 2048)),
        name=(
            "cmb.perturbations.accuracy_controls."
            "los_phase_quadrature.maximum_nodes"
        ),
        minimum=minimum_nodes,
    )
    # The phase-aware routine can coarsen a dense background grid before
    # refinement.  Keep the configured cap authoritative so bounded requests
    # remain bounded; a denser native history must not silently override it.
    maximum_nodes = configured_maximum_nodes
    phase_points = _positive_float(
        section.get(
            "phase_points_per_cycle",
            (accuracy_controls or {}).get("phase_points_per_cycle", 8.0),
        ),
        name=(
            "cmb.perturbations.accuracy_controls."
            "los_phase_quadrature.phase_points_per_cycle"
        ),
    )
    return LOSQuadratureControls(
        enabled=enabled,
        minimum_nodes=minimum_nodes,
        maximum_nodes=maximum_nodes,
        phase_points_per_cycle=phase_points,
        configured_maximum_nodes=configured_maximum_nodes,
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
    require_phase_resolution: bool = False,
) -> numpy.ndarray:
    """Build a bounded logarithmic k grid with physical phase anchors.

    The node budget is deliberately explicit.  Callers that are making a
    production acceptance decision may set ``require_phase_resolution`` so a
    capped smoke-test ladder cannot be mistaken for a resolved quadrature.
    """

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

    # If the explicit budget can hold the physical radial phase ladder,
    # preserve that ladder as an indivisible set.  The previous construction
    # merged logarithmic, radial, and acoustic ladders and then uniformly
    # thinned the union when it exceeded ``maximum_nodes``; that thinning
    # could remove every other phase node and report a false gap twice the
    # declared step.  Optional logarithmic nodes may be thinned, but required
    # phase nodes are never discarded.
    requirements = phase_aware_k_grid_requirements(
        lower,
        upper,
        phase_points_per_cycle=phase_points,
        eta_distance=distance,
        sound_horizon=acoustic_distance,
    )
    required_count = int(requirements["required_nodes"])
    if required_count <= maximum:
        phase_nodes = numpy.linspace(lower, upper, required_count)
        required_nodes = set(float(value) for value in phase_nodes)
        required_nodes.update(
            float(value) for value in anchors if lower <= float(value) <= upper
        )
        if len(required_nodes) > maximum:
            # Anchors are supplementary in this branch; the endpoint-
            # inclusive phase ladder is the non-negotiable evidence set.
            required_nodes = set(float(value) for value in phase_nodes)
        optional_budget = max(0, maximum - len(required_nodes))
        if optional_budget:
            optional_nodes = numpy.geomspace(
                lower,
                upper,
                optional_budget + 2,
            )[1:-1]
            required_nodes.update(float(value) for value in optional_nodes)
        resolved = numpy.asarray(sorted(required_nodes), dtype=float)
        while resolved.size < minimum:
            log_values = numpy.log(resolved)
            gap_index = int(numpy.argmax(numpy.diff(log_values)))
            midpoint = numpy.exp(
                0.5 * (log_values[gap_index] + log_values[gap_index + 1])
            )
            if not numpy.isfinite(midpoint) or midpoint <= resolved[gap_index]:
                midpoint = 0.5 * (
                    resolved[gap_index] + resolved[gap_index + 1]
                )
            if (
                midpoint <= resolved[gap_index]
                or midpoint >= resolved[gap_index + 1]
            ):
                raise ValueError(
                    "Phase-aware k quadrature could not satisfy its minimum "
                    "node budget"
                )
            resolved = numpy.insert(resolved, gap_index + 1, midpoint)
        if require_phase_resolution:
            status = phase_aware_k_grid_status(
                resolved,
                phase_points_per_cycle=phase_points,
                eta_distance=distance,
                sound_horizon=acoustic_distance,
            )
            if not bool(status["resolved"]):
                raise ValueError(
                    "Phase-aware k quadrature is under-resolved: "
                    f"actual_nodes={resolved.size}, "
                    f"required_nodes={status['required_nodes']}"
                )
        return resolved
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
        resolved = numpy.asarray(result, dtype=float)
        if require_phase_resolution:
            status = phase_aware_k_grid_status(
                resolved,
                phase_points_per_cycle=phase_points,
                eta_distance=distance,
                sound_horizon=acoustic_distance,
            )
            if not bool(status["resolved"]):
                raise ValueError(
                    "Phase-aware k quadrature is under-resolved: "
                    f"actual_nodes={resolved.size}, "
                    f"required_nodes={status['required_nodes']}"
                )
        return resolved
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
    resolved = numpy.asarray(sorted(required | set(optional)), dtype=float)
    if require_phase_resolution:
        status = phase_aware_k_grid_status(
            resolved,
            phase_points_per_cycle=phase_points,
            eta_distance=distance,
            sound_horizon=acoustic_distance,
        )
        if not bool(status["resolved"]):
            raise ValueError(
                "Phase-aware k quadrature is under-resolved: "
                f"actual_nodes={resolved.size}, "
                f"required_nodes={status['required_nodes']}"
            )
    return resolved


def nested_phase_aware_k_grid(
    base_k_values: numpy.ndarray,
    *,
    maximum_nodes: int,
    phase_points_per_cycle: float,
    eta_distance: float,
    sound_horizon: float,
    require_phase_resolution: bool = False,
) -> numpy.ndarray:
    """Add phase nodes while preserving every node in a base k ladder.

    This is the projection-only refinement surface.  Its returned ladder is
    nested by construction, so source histories and transfer products at the
    base nodes remain reusable while new nodes are explicitly attributable to
    interpolation work.  A caller that cannot afford the required physical
    node count receives an unresolved ladder or a hard error when resolution
    is required; it never receives a silently thinned replacement ladder.
    """

    base = numpy.asarray(base_k_values, dtype=float)
    if (
        base.ndim != 1
        or base.size < 2
        or not numpy.all(numpy.isfinite(base))
        or numpy.any(numpy.diff(base) <= 0.0)
        or numpy.any(base <= 0.0)
    ):
        raise ValueError("base_k_values must be finite, positive, and ordered")
    maximum = _positive_int(
        maximum_nodes,
        name="maximum_nodes",
        minimum=int(base.size),
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
    requirements = phase_aware_k_grid_requirements(
        float(base[0]),
        float(base[-1]),
        phase_points_per_cycle=phase_points,
        eta_distance=distance,
        sound_horizon=acoustic_distance,
    )
    required_nodes = int(requirements["required_nodes"])
    if require_phase_resolution and required_nodes > maximum:
        raise ValueError(
            "Nested phase-aware k quadrature cannot satisfy its node cap: "
            f"required_nodes={required_nodes}, maximum_nodes={maximum}"
        )
    target = min(
        maximum,
        max(
            int(base.size) + int(maximum > int(base.size)),
            required_nodes,
        ),
    )
    result = base.copy()

    def _insert_largest_phase_gap() -> None:
        """Bisect the currently worst-resolved physical phase interval."""

        nonlocal result
        phase_gaps = numpy.maximum(
            numpy.diff(result) * distance,
            numpy.diff(result) * acoustic_distance,
        )
        gap_index = int(numpy.argmax(phase_gaps))
        midpoint = 0.5 * (result[gap_index] + result[gap_index + 1])
        if not numpy.isfinite(midpoint) or midpoint <= result[gap_index]:
            raise ValueError("Nested phase-aware k quadrature stalled")
        result = numpy.insert(result, gap_index + 1, midpoint)

    while result.size < target:
        _insert_largest_phase_gap()

    # The endpoint-derived count is only a lower bound.  A nonuniform base
    # ladder can contain a large unresolved gap even after it reaches that
    # count, so continue measuring the actual spacing while budget remains.
    # This preserves every base node and avoids rejecting a feasible ladder
    # merely because its initial distribution was clustered.
    status = phase_aware_k_grid_status(
        result,
        phase_points_per_cycle=phase_points,
        eta_distance=distance,
        sound_horizon=acoustic_distance,
    )
    while not bool(status["spacing_resolved"]) and result.size < maximum:
        _insert_largest_phase_gap()
        status = phase_aware_k_grid_status(
            result,
            phase_points_per_cycle=phase_points,
            eta_distance=distance,
            sound_horizon=acoustic_distance,
        )
    if require_phase_resolution:
        if not bool(status["resolved"]):
            raise ValueError(
                "Nested phase-aware k quadrature is under-resolved: "
                f"actual_nodes={result.size}, required_nodes={required_nodes}"
            )
    return result


def phase_aware_k_grid_status(
    k_values: numpy.ndarray,
    *,
    phase_points_per_cycle: float,
    eta_distance: float,
    sound_horizon: float,
) -> dict[str, int | float | bool]:
    """Describe whether a bounded phase-aware grid meets its phase budget."""

    values = numpy.asarray(k_values, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not numpy.all(numpy.isfinite(values))
    ):
        raise ValueError(
            "Phase-aware status requires a finite one-dimensional k grid"
        )
    requirements = phase_aware_k_grid_requirements(
        float(values[0]),
        float(values[-1]),
        phase_points_per_cycle=phase_points_per_cycle,
        eta_distance=eta_distance,
        sound_horizon=sound_horizon,
    )
    actual = int(values.size)
    required = int(requirements["required_nodes"])
    phase_step = float(requirements["phase_step"])
    gaps = numpy.diff(values)
    maximum_radial_phase_step = float(
        numpy.max(gaps, initial=0.0) * float(eta_distance)
    )
    maximum_acoustic_phase_step = float(
        numpy.max(gaps, initial=0.0) * float(sound_horizon)
    )
    count_resolved = actual >= required
    spacing_resolved = (
        maximum_radial_phase_step <= phase_step
        and maximum_acoustic_phase_step <= phase_step
    )
    return {
        "actual_nodes": actual,
        "required_nodes": required,
        "radial_required_nodes": int(requirements["radial_required_nodes"]),
        "acoustic_required_nodes": int(
            requirements["acoustic_required_nodes"]
        ),
        "phase_step": phase_step,
        "maximum_radial_phase_step": maximum_radial_phase_step,
        "maximum_acoustic_phase_step": maximum_acoustic_phase_step,
        "count_resolved": count_resolved,
        "spacing_resolved": spacing_resolved,
        "resolved": bool(count_resolved and spacing_resolved),
    }


def phase_aware_k_grid_requirements(
    k_min: float,
    k_max: float,
    *,
    phase_points_per_cycle: float,
    eta_distance: float,
    sound_horizon: float,
) -> dict[str, int | float]:
    """Report the physical node count required by a phase-aware k ladder.

    ``phase_aware_k_grid`` accepts an explicit node budget because callers
    may need a bounded smoke-test grid.  This companion calculation keeps
    the physical requirement visible in runtime evidence instead of letting
    a capped ladder appear converged merely because it returned successfully.
    The count is the largest of the endpoint-inclusive phase ladders for the
    radial and acoustic distances.
    """

    lower = _positive_float(k_min, name="k_min")
    upper = _positive_float(k_max, name="k_max")
    if upper <= lower:
        raise ValueError("k_max must be greater than k_min")
    phase_points = _positive_float(
        phase_points_per_cycle,
        name="phase_points_per_cycle",
    )
    distance = _positive_float(eta_distance, name="eta_distance")
    acoustic_distance = _positive_float(
        sound_horizon,
        name="sound_horizon",
    )
    phase_step = numpy.pi / phase_points
    radial_count = max(
        2,
        int(numpy.ceil((upper - lower) * distance / phase_step)) + 1,
    )
    acoustic_count = max(
        2,
        int(numpy.ceil((upper - lower) * acoustic_distance / phase_step)) + 1,
    )
    return {
        "radial_required_nodes": radial_count,
        "acoustic_required_nodes": acoustic_count,
        "required_nodes": max(radial_count, acoustic_count),
        "phase_step": float(phase_step),
    }


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
    # A dense background history is an interpolation source, not a resolved
    # line-of-sight plan.  Start from the requested minimum so the remaining
    # budget can actually resolve visibility and Fourier phase instead of
    # filling the cap with uniformly selected background nodes.
    if eta.size > maximum:
        seed_count = min(minimum, maximum)
        uniform_seed_indices = numpy.unique(
            numpy.asarray(
                numpy.linspace(
                    0,
                    eta.size - 1,
                    seed_count,
                    dtype=int,
                ),
                dtype=int,
            )
        )
        feature_indices = [
            0,
            int(eta.size - 1),
            int(numpy.argmax(numpy.abs(visibility_values))),
        ]
        active_visibility = numpy.flatnonzero(
            numpy.abs(visibility_values) >= visibility_scale
        )
        if active_visibility.size:
            feature_indices.extend(
                (int(active_visibility[0]), int(active_visibility[-1]))
            )
        required_indices: list[int] = []
        for index in feature_indices:
            has_capacity = len(required_indices) < maximum
            if index not in required_indices and has_capacity:
                required_indices.append(index)
        optional_indices = numpy.setdiff1d(
            uniform_seed_indices,
            numpy.asarray(required_indices, dtype=int),
            assume_unique=False,
        )
        optional_budget = maximum - len(required_indices)
        if optional_indices.size > optional_budget:
            optional_indices = optional_indices[
                numpy.unique(
                    numpy.linspace(
                        0,
                        optional_indices.size - 1,
                        optional_budget,
                        dtype=int,
                    )
                )
            ]
        seed_indices = set(
            int(index)
            for index in numpy.concatenate(
                (
                    numpy.asarray(required_indices, dtype=int),
                    optional_indices,
                )
            )
        )
        selected = numpy.asarray(sorted(seed_indices), dtype=int)
        eta = eta[selected]
        visibility_values = visibility_values[selected]
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
        candidate_values = numpy.setdiff1d(
            numpy.unique(numpy.concatenate(additions)),
            result,
            assume_unique=True,
        )
        if candidate_values.size > remaining:
            selected_positions = numpy.unique(
                numpy.linspace(
                    0,
                    candidate_values.size - 1,
                    remaining,
                    dtype=int,
                )
            )
            candidate_values = candidate_values[selected_positions]
        result = numpy.unique(numpy.concatenate((result, candidate_values)))
    if result.size < minimum:
        while result.size < minimum:
            steps = numpy.diff(result)
            addition_count = min(minimum - result.size, steps.size)
            widest = numpy.argsort(steps, kind="stable")[-addition_count:]
            result = numpy.unique(
                numpy.concatenate(
                    (
                        result,
                        0.5 * (result[widest] + result[widest + 1]),
                    )
                )
            )
    return numpy.asarray(result, dtype=float)


def physical_history_anchors(
    eta_grid: numpy.ndarray,
    *,
    visibility: numpy.ndarray,
    interaction_rate: numpy.ndarray,
) -> dict[str, float]:
    """Locate informative history features on an actual physical eta grid.

    The returned coordinates cover both endpoints, visibility onset/peak/tail,
    and the sharpest interaction transition.  They are physical coordinates,
    not fixed fractions of a request-dependent interval.
    """

    eta = numpy.asarray(eta_grid, dtype=float)
    visibility_values = numpy.asarray(visibility, dtype=float)
    interaction_values = numpy.asarray(interaction_rate, dtype=float)
    if (
        eta.ndim != 1
        or eta.size < 3
        or visibility_values.shape != eta.shape
        or interaction_values.shape != eta.shape
        or not numpy.all(numpy.isfinite(eta))
        or not numpy.all(numpy.isfinite(visibility_values))
        or not numpy.all(numpy.isfinite(interaction_values))
        or numpy.any(numpy.diff(eta) <= 0.0)
    ):
        raise ValueError(
            "Physical history anchors require finite matching eta histories"
        )
    visibility_abs = numpy.abs(visibility_values)
    peak = float(numpy.max(visibility_abs, initial=0.0))
    active = numpy.flatnonzero(
        visibility_abs >= max(peak * 1.0e-3, numpy.finfo(float).tiny)
    )
    peak_index = int(numpy.argmax(visibility_abs))
    if active.size:
        onset_index = int(active[0])
        tail_index = int(active[-1])
    else:
        onset_index = peak_index
        tail_index = peak_index
    log_rate = numpy.log(
        numpy.maximum(numpy.abs(interaction_values), numpy.finfo(float).tiny)
    )
    transition_index = int(
        numpy.argmax(numpy.abs(numpy.gradient(log_rate, eta, edge_order=1)))
    )
    return {
        "integration_start": float(eta[0]),
        "visibility_onset": float(eta[onset_index]),
        "visibility_peak": float(eta[peak_index]),
        "visibility_tail": float(eta[tail_index]),
        "interaction_transition": float(eta[transition_index]),
        "integration_end": float(eta[-1]),
    }


def informative_k_indices(
    k_values: numpy.ndarray,
    *,
    count: int,
    feature_k: Sequence[float] = (),
) -> tuple[int, ...]:
    """Select endpoint, logarithmic-interior, and physical-feature modes."""

    values = numpy.asarray(k_values, dtype=float)
    if (
        values.ndim != 1
        or values.size < 1
        or not numpy.all(numpy.isfinite(values))
        or numpy.any(values <= 0.0)
        or numpy.any(numpy.diff(values) <= 0.0)
    ):
        raise ValueError("Informative k selection requires an ordered grid")
    requested = _positive_int(count, name="count")
    target = min(int(values.size), requested)
    candidates = {0, int(values.size) - 1}
    if target > 2:
        log_values = numpy.log(values)
        targets = numpy.linspace(log_values[0], log_values[-1], target)
        candidates.update(
            int(numpy.argmin(numpy.abs(log_values - target_value)))
            for target_value in targets
        )
    for raw_feature in feature_k:
        feature = float(raw_feature)
        if numpy.isfinite(feature) and values[0] <= feature <= values[-1]:
            candidates.add(int(numpy.argmin(numpy.abs(values - feature))))
    if len(candidates) > target:
        ordered = sorted(candidates)
        positions = numpy.linspace(0, len(ordered) - 1, target, dtype=int)
        candidates = {ordered[int(position)] for position in positions}
    while len(candidates) < target:
        ordered = sorted(candidates)
        widest = max(
            zip(ordered[:-1], ordered[1:]),
            key=lambda pair: float(values[pair[1]] / values[pair[0]]),
        )
        candidates.add((widest[0] + widest[1]) // 2)
    return tuple(sorted(candidates))


def nested_coarse_indices(
    eta_grid: numpy.ndarray,
    *,
    feature_eta: Mapping[str, float] | None = None,
) -> numpy.ndarray:
    """Return a genuinely coarser nested eta ladder retaining key features."""

    eta = numpy.asarray(eta_grid, dtype=float)
    if (
        eta.ndim != 1
        or eta.size < 4
        or not numpy.all(numpy.isfinite(eta))
        or numpy.any(numpy.diff(eta) <= 0.0)
    ):
        raise ValueError("Nested eta refinement requires four ordered nodes")
    indices = set(range(0, int(eta.size), 2))
    indices.update((0, int(eta.size) - 1))
    for coordinate in (feature_eta or {}).values():
        value = float(coordinate)
        if numpy.isfinite(value) and eta[0] <= value <= eta[-1]:
            indices.add(int(numpy.argmin(numpy.abs(eta - value))))
    resolved = numpy.asarray(sorted(indices), dtype=int)
    if resolved.size >= eta.size:
        raise ValueError(
            "Nested eta refinement produced an identical effective grid"
        )
    return resolved


def estimate_convergence(
    coarse: numpy.ndarray,
    fine: numpy.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> ConvergenceEstimate:
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
    return ConvergenceEstimate(
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
    feature_eta: Mapping[str, float] | None = None,
) -> HistoryConvergence:
    """Compare histories densely, including features and between-grid nodes.

    ``anchors`` retains the historical fractional spelling for callers that
    need it.  Production callers pass absolute ``feature_eta`` coordinates
    obtained from recombination, visibility, and interaction histories.
    Every node and interval midpoint in the overlapping grids is sampled, so
    a spike or sign reversal between named anchors cannot be certified away.
    """

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
    overlap_start = max(float(coarse_grid[0]), float(fine_grid[0]))
    overlap_end = min(float(coarse_grid[-1]), float(fine_grid[-1]))
    if overlap_end <= overlap_start:
        raise ValueError("History comparisons require overlapping eta grids")
    if feature_eta is not None:
        anchor_coordinates = {
            str(name): float(value) for name, value in feature_eta.items()
        }
    else:
        anchor_positions = anchors or {
            "early": 0.05,
            "recombination": 0.50,
            "late": 0.95,
        }
        anchor_coordinates = {}
        for anchor_name, position in anchor_positions.items():
            fraction = float(position)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"History anchor '{anchor_name}' must be in [0, 1]"
                )
            anchor_coordinates[str(anchor_name)] = float(
                overlap_start + fraction * (overlap_end - overlap_start)
            )
    for anchor_name, coordinate in anchor_coordinates.items():
        if not overlap_start <= coordinate <= overlap_end:
            raise ValueError(
                f"History feature '{anchor_name}' lies outside the common "
                "eta domain"
            )
    common_nodes = numpy.unique(
        numpy.concatenate(
            (
                coarse_grid[
                    (coarse_grid >= overlap_start)
                    & (coarse_grid <= overlap_end)
                ],
                fine_grid[
                    (fine_grid >= overlap_start) & (fine_grid <= overlap_end)
                ],
                numpy.asarray(tuple(anchor_coordinates.values()), dtype=float),
            )
        )
    )
    midpoint_nodes = 0.5 * (common_nodes[:-1] + common_nodes[1:])
    sample_eta = numpy.unique(
        numpy.concatenate((common_nodes, midpoint_nodes))
    )
    absolute_errors: dict[str, float] = {}
    relative_errors: dict[str, float] = {}
    maximum_absolute = 0.0
    maximum_relative = 0.0
    history_samples: dict[str, tuple[numpy.ndarray, numpy.ndarray, float]] = {}
    for name in coarse_histories:
        coarse_values = numpy.asarray(coarse_histories[name], dtype=float)
        fine_values = numpy.asarray(fine_histories[name], dtype=float)
        if coarse_values.shape != coarse_grid.shape or (
            fine_values.shape != fine_grid.shape
        ):
            raise ValueError(f"History '{name}' does not match its eta grid")
        if not numpy.all(numpy.isfinite(coarse_values)) or not numpy.all(
            numpy.isfinite(fine_values)
        ):
            raise ValueError(f"History '{name}' contains non-finite values")
        coarse_sample = numpy.interp(sample_eta, coarse_grid, coarse_values)
        fine_sample = numpy.interp(sample_eta, fine_grid, fine_values)
        scale = max(
            float(numpy.max(numpy.abs(coarse_sample), initial=0.0)),
            float(numpy.max(numpy.abs(fine_sample), initial=0.0)),
            float(absolute_tolerance),
        )
        difference = numpy.abs(fine_sample - coarse_sample)
        local_floor = max(scale * 1.0e-8, float(absolute_tolerance))
        normalized = difference / numpy.maximum(
            numpy.maximum(numpy.abs(fine_sample), numpy.abs(coarse_sample)),
            local_floor,
        )
        maximum_absolute = max(
            maximum_absolute,
            float(numpy.max(difference, initial=0.0)),
        )
        maximum_relative = max(
            maximum_relative,
            float(numpy.max(normalized, initial=0.0)),
        )
        history_samples[str(name)] = (coarse_sample, fine_sample, scale)
    for anchor_name, coordinate in anchor_coordinates.items():
        anchor_absolute = 0.0
        anchor_relative = 0.0
        for name in coarse_histories:
            coarse_values = numpy.asarray(coarse_histories[name], dtype=float)
            fine_values = numpy.asarray(fine_histories[name], dtype=float)
            coarse_value = float(
                numpy.interp(coordinate, coarse_grid, coarse_values)
            )
            fine_value = float(
                numpy.interp(coordinate, fine_grid, fine_values)
            )
            difference = abs(fine_value - coarse_value)
            anchor_absolute = max(anchor_absolute, difference)
            history_scale = history_samples[str(name)][2]
            anchor_relative = max(
                anchor_relative,
                difference
                / max(
                    abs(coarse_value),
                    abs(fine_value),
                    history_scale * 1.0e-8,
                    float(absolute_tolerance),
                ),
            )
        absolute_errors[str(anchor_name)] = anchor_absolute
        relative_errors[str(anchor_name)] = anchor_relative
    converged = bool(
        maximum_absolute <= float(absolute_tolerance)
        or maximum_relative <= float(relative_tolerance)
    )
    return HistoryConvergence(
        absolute_error=float(maximum_absolute),
        relative_error=float(maximum_relative),
        anchor_absolute_errors=absolute_errors,
        anchor_relative_errors=relative_errors,
        sample_count=int(sample_eta.size),
        converged=bool(converged),
    )


def require_convergence(
    estimate: ConvergenceEstimate,
    *,
    label: str,
    fail_on_nonconvergence: bool,
) -> None:
    """Raise a named under-resolution error when convergence is required."""

    if estimate.converged or not fail_on_nonconvergence:
        return
    raise AdaptiveNonConvergenceError(
        f"Declared {label} refinement did not converge: "
        f"relative_error={estimate.relative_error:.6g}, "
        f"absolute_error={estimate.absolute_error:.6g}",
        label=label,
        failed_products=(label,),
        evidence={
            "relative_error": float(estimate.relative_error),
            "absolute_error": float(estimate.absolute_error),
        },
    )
