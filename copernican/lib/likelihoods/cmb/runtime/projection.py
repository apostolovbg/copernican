r"""Declared transfer projection and spectrum integration helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gammaln, spherical_jn

from ....cmb_output import canonical_cmb_spectrum_name
from ....cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_kernel_spec,
    resolve_declared_source_kernel,
    validate_declared_projection_sector,
)
from ....model_adapter import FrozenMapping
from ....perturbation_contract import (
    PerturbationCollisionTargetSelectorData,
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)
from ..errors import (
    ConstraintViolationError,
    ConvergenceError,
    NonFiniteEvolutionError,
    classify_exception,
    failure_context,
)
from . import cache
from .adaptive import (
    ConvergenceEstimate,
    estimate_convergence,
    estimate_history_convergence,
    phase_aware_eta_grid,
    phase_aware_k_grid,
    phase_aware_k_grid_requirements,
    phase_aware_k_grid_status,
    require_convergence,
    resolve_adaptive_controls,
    resolve_los_quadrature_controls,
)
from .background import (
    _C_LIGHT_KM_S,
    _LEGACY_DECLARED_EVOLUTION_COORDINATES,
    CustomCMBSpectrumData,
    CustomCMBTransferData,
    _accuracy_control_value,
    _build_custom_cmb_background,
    _coerce_numeric_scalar,
    _compute_spherical_bessel_batch,
    _compute_spherical_bessel_mode_batch,
    _custom_cmb_spectrum_cache_key,
    _custom_cmb_transfer_cache_key,
    _CustomCMBPhysicalParameters,
    _DeclaredProjectionKernelBatch,
    _get_cached_custom_cmb_spectrum_data,
    _get_cached_declared_projection_kernel_batch,
    _physical_runtime_scalars,
    _resolve_custom_cmb_numerics,
    _resolve_custom_cmb_physical_parameters,
    _resolve_declared_accuracy_controls,
    _resolve_declared_background_context,
)
from .convergence import (
    RUNTIME_WORK_LIMIT_NAMES,
    evaluate_spectrum_refinement,
    resolve_declared_numerical_envelope,
    resolve_production_scalar_convergence,
)
from .evolution import (
    _build_declared_base_context,
    _compile_batched_row_equation_program,
    _compile_declared_perturbation_contract,
    _compile_equation_program,
    _compile_expression_tuple_program,
    _compile_ordered_context_program,
    _compute_tight_coupling_drag,
    _declared_momentum_grid_context,
    _declared_runtime_seed,
    _evaluate_declared_initial_state,
    _integrate_batched_rk4,
    _nonuniform_gradient,
    _resolve_declared_graph_context,
    _resolve_declared_graph_context_ordered,
    _resolve_declared_momentum_grid_runtimes,
    _scalar_einstein_constraint_metrics,
    _tight_coupling_is_active,
    _validate_generated_scalar_initial_constraints,
    _validate_generated_tensor_initial_constraints,
    _validate_generated_vector_initial_constraints,
    prepare_runtime_assets,
)
from .performance import PhaseTimer

_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}
_SCALAR_SUPERHORIZON_PREFIX_KETA = 5.0e-3
_SCALAR_INITIAL_SOLVE_NUMERICAL_TOLERANCE = 1.0e-9
_BESSEL_WORK_CELL_BUDGET = 8_000_000
_BESSEL_MAX_MODE_BATCH = 16
_EVOLUTION_WORK_CELL_BUDGET = 16_000_000
_WORK_ESTIMATE_VERSION = 1


def _can_batch_declared_evolution(
    *,
    generated_scalar_hierarchy: bool,
    shared_mode_grids_enabled: bool,
    mode_count: int,
    has_momentum_runtimes: bool,
    has_end_boundaries: bool,
    adaptive_evolution_enabled: bool,
    adaptive_source_enabled: bool,
    adaptive_transfer_enabled: bool,
    adaptive_projection_enabled: bool,
    adaptive_k_enabled: bool,
    continuous_collision_solver: bool,
    has_declared_collision_operators: bool,
    state_slots: Iterable[Any],
    collision_runtimes: Iterable[Any],
) -> bool:
    """Return whether declared modes share the vectorized RK capability.

    The batch path preserves the scalar executor for contracts with adaptive
    histories, non-shared grids, transformed coordinates, or conditional
    collision activation.  Those contracts require independently staged
    scalar control rather than a common explicit schedule.
    """

    if not (
        generated_scalar_hierarchy
        and shared_mode_grids_enabled
        and int(mode_count) > 1
    ):
        return False
    if (
        has_end_boundaries
        or adaptive_evolution_enabled
        or adaptive_source_enabled
        or adaptive_transfer_enabled
        or adaptive_projection_enabled
        or continuous_collision_solver
    ):
        return False
    if any(
        str(getattr(slot, "wrt", ""))
        not in _LEGACY_DECLARED_EVOLUTION_COORDINATES
        for slot in state_slots
    ):
        return False
    return all(
        str(getattr(runtime, "activation_strategy", "")) == "always"
        for runtime in collision_runtimes
    )


@dataclass(frozen=True, slots=True)
class _CompiledCollisionOperatorRuntime:
    """Resolved runtime metadata for one split collision operator."""

    name: str
    integration_strategy: str
    activation_strategy: str
    counterpart: str | None
    rate_expression: Any
    target_variables: tuple[str, ...]
    target_slot_indices: tuple[int, ...]
    matrix: tuple[tuple[Any, ...], ...]
    damping_slot_indices: tuple[int, ...] = ()
    damping_coefficient: Any | None = None
    fast_manifold: bool = False
    conservation_rule_names: tuple[str, ...] = ()


def _solve_declared_fast_collision_target(
    matrix: numpy.ndarray,
    forcing: numpy.ndarray,
    current_state: numpy.ndarray,
    collision_rate: float,
    *,
    solver_cache: dict[str, Any] | None = None,
) -> numpy.ndarray:
    """Return the declaration-defined first-order fast collision state.

    The collision matrix is fixed during one Fourier-mode evolution.  Cache
    its factorization so repeated fast-manifold projections do not redo an
    SVD for every integration stage.
    """

    operator = numpy.asarray(matrix, dtype=float)
    source = numpy.asarray(forcing, dtype=float)
    current = numpy.asarray(current_state, dtype=float)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError(
            "Declared fast collision operator must have a square matrix"
        )
    if operator.shape[0] != source.size or source.size != current.size:
        raise ValueError(
            "Declared fast collision operator dimensions do not match its "
            "target state"
        )
    if not numpy.isfinite(collision_rate) or abs(collision_rate) <= 1.0e-12:
        return current.copy()

    fast_result = _solve_small_declared_collision_target(
        operator,
        source,
        current,
        collision_rate,
    )
    if fast_result is not None:
        return fast_result

    cache_key = operator.tobytes()
    cached_solver = None
    if solver_cache is not None:
        cached_solver = solver_cache.get(cache_key)
    if cached_solver is None:
        singular_values = numpy.linalg.svd(operator, compute_uv=False)
        scale = max(float(numpy.max(singular_values, initial=0.0)), 1.0)
        tolerance = max(operator.shape) * numpy.finfo(float).eps * scale
        rank = int(numpy.count_nonzero(singular_values > tolerance))
        if rank == operator.shape[0]:
            cached_solver = (
                "full",
                numpy.linalg.inv(operator),
            )
        else:
            left_vectors, _, right_vectors_transposed = numpy.linalg.svd(
                operator
            )
            left_null = left_vectors[:, rank:]
            right_null = right_vectors_transposed[rank:, :].T
            projected_inverse = numpy.linalg.pinv(
                operator,
                rcond=tolerance,
            )
            invariant_map = left_null.T @ right_null
            invariant_solver = numpy.linalg.pinv(invariant_map)
            cached_solver = (
                "rank_deficient",
                left_null,
                right_null,
                projected_inverse,
                invariant_solver,
            )
        if solver_cache is not None:
            solver_cache[cache_key] = cached_solver
    solver_kind = cached_solver[0]
    if solver_kind == "full":
        return cached_solver[1] @ (-source / float(collision_rate))

    _, left_null, right_null, projected_inverse, invariant_solver = (
        cached_solver
    )
    projected_source = source - left_null @ (left_null.T @ source)
    particular = projected_inverse @ (
        -projected_source / float(collision_rate)
    )
    invariant_target = left_null.T @ (current - particular)
    coefficients = invariant_solver @ invariant_target
    return particular + right_null @ coefficients


def _solve_small_declared_collision_target(
    operator: numpy.ndarray,
    source: numpy.ndarray,
    current: numpy.ndarray,
    collision_rate: float,
) -> numpy.ndarray | None:
    """Solve the small block manifolds without a per-stage SVD.

    Thomson drag uses a four-state block diagonal operator containing one
    rank-one photon--baryon block and one full-rank polarization block.  The
    generic path below is deliberately retained for arbitrary declarations,
    but factoring this fixed small topology with scalar algebra avoids a
    costly SVD at every RK stage and Fourier mode.
    """

    size = int(operator.shape[0])
    if size == 2:
        blocks = ((0, 2),)
    elif size == 4:
        off_diagonal = numpy.concatenate(
            (operator[:2, 2:].ravel(), operator[2:, :2].ravel())
        )
        scale = max(float(numpy.max(numpy.abs(operator), initial=0.0)), 1.0)
        if float(numpy.max(numpy.abs(off_diagonal), initial=0.0)) > (
            32.0 * numpy.finfo(float).eps * scale
        ):
            return None
        blocks = ((0, 2), (2, 4))
    else:
        return None

    result = current.copy()
    for start, stop in blocks:
        block = operator[start:stop, start:stop]
        block_source = source[start:stop]
        block_current = current[start:stop]
        block_result = _solve_two_state_collision_block(
            block,
            block_source,
            block_current,
            collision_rate,
        )
        if block_result is None:
            return None
        result[start:stop] = block_result
    return result


def _solve_two_state_collision_block(
    operator: numpy.ndarray,
    source: numpy.ndarray,
    current: numpy.ndarray,
    collision_rate: float,
) -> numpy.ndarray | None:
    """Solve one two-state collision block using determinant algebra."""

    if operator.shape != (2, 2):
        return None
    scale = max(float(numpy.max(numpy.abs(operator), initial=0.0)), 1.0)
    tolerance = 32.0 * numpy.finfo(float).eps * scale
    operator_00, operator_01 = (float(value) for value in operator[0])
    operator_10, operator_11 = (float(value) for value in operator[1])
    determinant = operator_00 * operator_11 - operator_01 * operator_10
    if abs(determinant) > tolerance * scale:
        inverse = numpy.asarray(
            ((operator_11, -operator_01), (-operator_10, operator_00)),
            dtype=float,
        )
        inverse /= determinant
        return inverse @ (-source / float(collision_rate))

    row_norms = numpy.asarray(
        (
            math.hypot(operator_00, operator_01),
            math.hypot(operator_10, operator_11),
        ),
        dtype=float,
    )
    if float(numpy.max(row_norms, initial=0.0)) <= tolerance:
        return current.copy()
    if row_norms[0] >= row_norms[1]:
        row = numpy.asarray((operator_00, operator_01), dtype=float)
    else:
        row = numpy.asarray((operator_10, operator_11), dtype=float)
    right_null = numpy.asarray((-row[1], row[0]), dtype=float)
    column_norms = numpy.asarray(
        (
            math.hypot(operator_00, operator_10),
            math.hypot(operator_01, operator_11),
        ),
        dtype=float,
    )
    if column_norms[0] >= column_norms[1]:
        left_null = numpy.asarray((-operator_10, operator_00), dtype=float)
    else:
        left_null = numpy.asarray((-operator_11, operator_01), dtype=float)
    left_residual = left_null @ operator
    null_scale = max(float(numpy.max(numpy.abs(operator), initial=0.0)), 1.0)
    if float(numpy.max(numpy.abs(left_residual), initial=0.0)) > (
        128.0 * numpy.finfo(float).eps * null_scale
    ):
        return None
    left_norm_sq = float(left_null @ left_null)
    right_norm_sq = float(right_null @ right_null)
    operator_norm_sq = float(numpy.sum(operator * operator))
    if min(left_norm_sq, right_norm_sq, operator_norm_sq) <= 0.0:
        return current.copy()
    projected_source = source - left_null * (
        float(left_null @ source) / left_norm_sq
    )
    particular = operator.T @ (-projected_source / float(collision_rate))
    particular /= operator_norm_sq
    invariant_map = float(left_null @ right_null)
    if abs(invariant_map) <= tolerance:
        return particular
    coefficient = float(left_null @ (current - particular)) / invariant_map
    return particular + right_null * coefficient


def _solve_batched_small_declared_collision_target(
    matrices: numpy.ndarray,
    sources: numpy.ndarray,
    currents: numpy.ndarray,
    collision_rates: numpy.ndarray,
) -> numpy.ndarray | None:
    """Solve declared two-state collision blocks for all active modes.

    The generated Thomson operator is block diagonal with two two-state
    blocks.  Keeping this algebra batched avoids entering Python once per
    Fourier mode and Runge--Kutta stage while retaining the generic scalar
    solver for declarations with a different topology.
    """

    operators = numpy.asarray(matrices, dtype=float)
    source_rows = numpy.asarray(sources, dtype=float)
    current_rows = numpy.asarray(currents, dtype=float)
    rates = numpy.asarray(collision_rates, dtype=float)
    if operators.ndim != 3 or operators.shape[1:] not in {(2, 2), (4, 4)}:
        return None
    if source_rows.shape != (operators.shape[0], operators.shape[1]):
        return None
    if current_rows.shape != source_rows.shape or rates.shape != (
        operators.shape[0],
    ):
        return None
    if (
        not numpy.all(numpy.isfinite(operators))
        or not numpy.all(numpy.isfinite(source_rows))
        or not numpy.all(numpy.isfinite(current_rows))
        or not numpy.all(numpy.isfinite(rates))
    ):
        return None
    if numpy.any(numpy.abs(rates) <= 1.0e-12):
        return None
    if operators.shape[1] == 4:
        scale = numpy.maximum(
            numpy.max(numpy.abs(operators), axis=(1, 2)),
            1.0,
        )
        off_diagonal = numpy.maximum(
            numpy.max(numpy.abs(operators[:, :2, 2:]), axis=(1, 2)),
            numpy.max(numpy.abs(operators[:, 2:, :2]), axis=(1, 2)),
        )
        if numpy.any(off_diagonal > 32.0 * numpy.finfo(float).eps * scale):
            return None
        first = _solve_batched_two_state_collision_block(
            operators[:, :2, :2],
            source_rows[:, :2],
            current_rows[:, :2],
            rates,
        )
        second = _solve_batched_two_state_collision_block(
            operators[:, 2:, 2:],
            source_rows[:, 2:],
            current_rows[:, 2:],
            rates,
        )
        if first is None or second is None:
            return None
        return numpy.concatenate((first, second), axis=1)
    return _solve_batched_two_state_collision_block(
        operators,
        source_rows,
        current_rows,
        rates,
    )


def _solve_batched_two_state_collision_block(
    operators: numpy.ndarray,
    sources: numpy.ndarray,
    currents: numpy.ndarray,
    collision_rates: numpy.ndarray,
) -> numpy.ndarray | None:
    """Solve a batch of two-state full-rank or rank-one blocks."""

    if operators.ndim != 3 or operators.shape[1:] != (2, 2):
        return None
    operator_00 = operators[:, 0, 0]
    operator_01 = operators[:, 0, 1]
    operator_10 = operators[:, 1, 0]
    operator_11 = operators[:, 1, 1]
    scale = numpy.maximum(numpy.max(numpy.abs(operators), axis=(1, 2)), 1.0)
    tolerance = 32.0 * numpy.finfo(float).eps * scale
    determinant = operator_00 * operator_11 - operator_01 * operator_10
    result = numpy.empty_like(sources, dtype=float)
    full_rank = numpy.abs(determinant) > tolerance * scale
    if numpy.any(full_rank):
        inverse_source = numpy.stack(
            (
                operator_11 * (-sources[:, 0] / collision_rates)
                - operator_01 * (-sources[:, 1] / collision_rates),
                -operator_10 * (-sources[:, 0] / collision_rates)
                + operator_00 * (-sources[:, 1] / collision_rates),
            ),
            axis=1,
        )
        result[full_rank] = (
            inverse_source[full_rank] / determinant[full_rank, numpy.newaxis]
        )
    rank_one = ~full_rank
    if numpy.any(rank_one):
        row_norms = numpy.stack(
            (
                numpy.hypot(operator_00, operator_01),
                numpy.hypot(operator_10, operator_11),
            ),
            axis=1,
        )
        use_first_row = row_norms[:, 0] >= row_norms[:, 1]
        selected_row_first = numpy.where(
            use_first_row,
            operator_00,
            operator_10,
        )
        selected_row_second = numpy.where(
            use_first_row,
            operator_01,
            operator_11,
        )
        right_null = numpy.stack(
            (-selected_row_second, selected_row_first),
            axis=1,
        )
        column_norms = numpy.stack(
            (
                numpy.hypot(operator_00, operator_10),
                numpy.hypot(operator_01, operator_11),
            ),
            axis=1,
        )
        use_first_column = column_norms[:, 0] >= column_norms[:, 1]
        left_null = numpy.stack(
            (
                numpy.where(use_first_column, -operator_10, -operator_11),
                numpy.where(use_first_column, operator_00, operator_01),
            ),
            axis=1,
        )
        row_zero = numpy.max(row_norms, axis=1) <= tolerance
        left_residual = numpy.einsum("ni,nij->nj", left_null, operators)
        residual_ok = numpy.max(numpy.abs(left_residual), axis=1) <= (
            128.0 * numpy.finfo(float).eps * scale
        )
        if numpy.any(rank_one & ~residual_ok):
            return None
        left_norm_sq = numpy.einsum("ni,ni->n", left_null, left_null)
        right_norm_sq = numpy.einsum("ni,ni->n", right_null, right_null)
        operator_norm_sq = numpy.sum(operators * operators, axis=(1, 2))
        safe = (
            rank_one
            & ~row_zero
            & (
                (left_norm_sq > 0.0)
                & (right_norm_sq > 0.0)
                & (operator_norm_sq > 0.0)
            )
        )
        result[rank_one & row_zero] = currents[rank_one & row_zero]
        if numpy.any(safe):
            projected_source = (
                sources
                - left_null
                * (
                    numpy.einsum("ni,ni->n", left_null, sources)
                    / numpy.maximum(left_norm_sq, numpy.finfo(float).tiny)
                )[:, numpy.newaxis]
            )
            particular = numpy.einsum(
                "nji,nj->ni",
                operators,
                -projected_source / collision_rates[:, numpy.newaxis],
            ) / numpy.maximum(
                operator_norm_sq[:, numpy.newaxis], numpy.finfo(float).tiny
            )
            invariant_map = numpy.einsum("ni,ni->n", left_null, right_null)
            nonzero_invariant = safe & (numpy.abs(invariant_map) > tolerance)
            result[safe] = particular[safe]
            if numpy.any(nonzero_invariant):
                coefficient = (
                    numpy.einsum("ni,ni->n", left_null, currents - particular)
                    / invariant_map
                )
                result[nonzero_invariant] = (
                    particular[nonzero_invariant]
                    + right_null[nonzero_invariant]
                    * coefficient[nonzero_invariant, numpy.newaxis]
                )
        if numpy.any(rank_one & ~row_zero & ~safe):
            return None
    return result


def _resolve_collision_target_selector_slots(
    selector: PerturbationCollisionTargetSelectorData,
    *,
    perturbation_data: Any,
    state_index_by_key: Mapping[tuple[str, str, int], int],
    allow_multiple: bool,
    label: str,
) -> tuple[tuple[str, int], ...]:
    """Return the declared state slots selected by one collision selector."""

    matches: list[tuple[str, int]] = []
    if selector.variable is not None:
        slot_index = state_index_by_key.get((selector.variable, "tau", 0))
        if slot_index is None:
            raise ValueError(
                f"{label} references non-state variable '{selector.variable}'"
            )
        return ((selector.variable, int(slot_index)),)
    for variable_name, variable_entry in perturbation_data.variables.items():
        if str(getattr(variable_entry, "kind", "")) != str(selector.kind):
            continue
        slot_index = state_index_by_key.get((variable_name, "tau", 0))
        if slot_index is None:
            continue
        matches.append((str(variable_name), int(slot_index)))
    if not matches:
        raise ValueError(
            f"{label} did not resolve any state slot for kind "
            f"'{selector.kind}'"
        )
    if not allow_multiple and len(matches) != 1:
        raise ValueError(
            f"{label} resolved {len(matches)} state slots for kind "
            f"'{selector.kind}' where exactly one was required"
        )
    return tuple(matches)


def _compile_split_collision_operator_runtimes(
    *,
    perturbation_data: Any,
    runtime_spec: Any,
) -> tuple[_CompiledCollisionOperatorRuntime, ...]:
    """Return the resolved split-operator runtimes for one graph."""

    collision_operators = (
        getattr(
            perturbation_data,
            "collision_operators",
            {},
        )
        or {}
    )
    conservation_rules = (
        getattr(
            perturbation_data,
            "conservation_rules",
            {},
        )
        or {}
    )
    state_index_by_key = runtime_spec.state_index_by_key
    compiled_runtimes: list[_CompiledCollisionOperatorRuntime] = []
    for operator_name in sorted(collision_operators):
        operator_entry = collision_operators[operator_name]
        strategy = str(
            getattr(operator_entry, "integration_strategy", "explicit")
            or "explicit"
        )
        if strategy == "explicit":
            continue
        if strategy == "exact":
            linear_form = getattr(operator_entry, "exact_form", None)
        elif strategy == "implicit":
            linear_form = getattr(operator_entry, "linear_block", None)
        else:
            raise ValueError(
                "Declared collision operator uses unsupported integration "
                f"strategy '{strategy}': {operator_name}"
            )
        if linear_form is None:
            raise ValueError(
                "Declared collision operator requires a compiled "
                f"{'exact_form' if strategy == 'exact' else 'linear_block'}: "
                f"{operator_name}"
            )
        rate_expression = getattr(
            operator_entry,
            "compiled_rate_expression",
            None,
        )
        if rate_expression is None:
            raise ValueError(
                "Declared collision operator requires a rate_expression "
                f"before evolution: {operator_name}"
            )
        target_variables: list[str] = []
        target_slot_indices: list[int] = []
        seen_variables: set[str] = set()
        for selector_index, selector in enumerate(linear_form.targets):
            matches = _resolve_collision_target_selector_slots(
                selector,
                perturbation_data=perturbation_data,
                state_index_by_key=state_index_by_key,
                allow_multiple=False,
                label=(
                    f"collision operator '{operator_name}' "
                    f"target[{selector_index}]"
                ),
            )
            variable_name, slot_index = matches[0]
            if variable_name in seen_variables:
                raise ValueError(
                    "Declared collision operator targets the same state more "
                    f"than once: {operator_name} -> {variable_name}"
                )
            seen_variables.add(variable_name)
            target_variables.append(variable_name)
            target_slot_indices.append(slot_index)
        damping_slot_indices: list[int] = []
        if linear_form.damping_targets:
            for selector_index, selector in enumerate(
                linear_form.damping_targets
            ):
                matches = _resolve_collision_target_selector_slots(
                    selector,
                    perturbation_data=perturbation_data,
                    state_index_by_key=state_index_by_key,
                    allow_multiple=True,
                    label=(
                        f"collision operator '{operator_name}' "
                        f"damping_target[{selector_index}]"
                    ),
                )
                for _, slot_index in matches:
                    if slot_index not in damping_slot_indices:
                        damping_slot_indices.append(slot_index)
        conservation_rule_names = tuple(
            sorted(
                str(rule_name)
                for rule_name, rule_entry in conservation_rules.items()
                if operator_name in getattr(rule_entry, "dependencies", ())
                or (
                    getattr(operator_entry, "counterpart", None) is not None
                    and getattr(operator_entry, "counterpart", None)
                    in getattr(rule_entry, "dependencies", ())
                )
            )
        )
        compiled_runtimes.append(
            _CompiledCollisionOperatorRuntime(
                name=str(operator_name),
                integration_strategy=strategy,
                activation_strategy=(
                    str(
                        getattr(
                            operator_entry, "activation_strategy", "always"
                        )
                    )
                    if linear_form.activation_strategy == "always"
                    else str(linear_form.activation_strategy)
                ),
                counterpart=getattr(operator_entry, "counterpart", None),
                rate_expression=rate_expression,
                target_variables=tuple(target_variables),
                target_slot_indices=tuple(target_slot_indices),
                matrix=linear_form.compiled_matrix,
                damping_slot_indices=tuple(damping_slot_indices),
                damping_coefficient=linear_form.compiled_damping_coefficient,
                fast_manifold=bool(linear_form.fast_manifold),
                conservation_rule_names=conservation_rule_names,
            )
        )
    return tuple(compiled_runtimes)


def _integrate_power_spectrum(
    primordial_grid: numpy.ndarray,
    log_k_values: numpy.ndarray,
    primary: numpy.ndarray,
    secondary: numpy.ndarray,
    *,
    auto_spectrum: bool = False,
) -> numpy.ndarray:
    """Return one finite power-spectrum quadrature in extended precision.

    Adaptive projection paths may merge logarithmic anchors with a dense
    phase ladder.  The merged nodes are sorted and deduplicated here before
    applying Simpson integration on a uniform log grid or the positive
    composite trapezoid rule on an irregular grid.  Auto spectra reject
    material negative power instead of hiding a quadrature failure.
    """

    primordial_ld = numpy.asarray(primordial_grid, dtype=numpy.longdouble)
    log_k_ld = numpy.asarray(log_k_values, dtype=numpy.longdouble)
    primary_ld = numpy.asarray(primary, dtype=numpy.longdouble)
    secondary_ld = numpy.asarray(secondary, dtype=numpy.longdouble)
    if primary_ld.ndim == 1:
        primary_ld = primary_ld[numpy.newaxis, :]
    if secondary_ld.ndim == 1:
        secondary_ld = secondary_ld[numpy.newaxis, :]
    if log_k_ld.ndim != 1 or primordial_ld.ndim != 1:
        raise ValueError("log-k quadrature nodes must be one-dimensional")
    if (
        log_k_ld.size != primordial_ld.size
        or primary_ld.shape[-1] != log_k_ld.size
        or secondary_ld.shape[-1] != log_k_ld.size
    ):
        raise ValueError(
            "log-k quadrature arrays must have matching node counts"
        )
    if not (
        numpy.all(numpy.isfinite(log_k_ld))
        and numpy.all(numpy.isfinite(primordial_ld))
    ):
        raise ValueError("log-k quadrature inputs must be finite")
    # Adaptive source/transfer paths merge a dense local ladder with the
    # declared scaffold.  Sort that union and collapse nodes which round to
    # the same long-double logarithm before constructing quadrature weights.
    # This preserves every distinct physical node while making the numerical
    # integration contract independent of how the local ladder was assembled.
    order = numpy.argsort(log_k_ld, kind="stable")
    log_k_ld = log_k_ld[order]
    primordial_ld = primordial_ld[order]
    primary_ld = primary_ld[..., order]
    secondary_ld = secondary_ld[..., order]
    if log_k_ld.size > 1:
        keep = numpy.concatenate(
            (
                numpy.asarray((True,), dtype=bool),
                numpy.diff(log_k_ld) > 0.0,
            )
        )
        log_k_ld = log_k_ld[keep]
        primordial_ld = primordial_ld[keep]
        primary_ld = primary_ld[..., keep]
        secondary_ld = secondary_ld[..., keep]
    weighted = primordial_ld[numpy.newaxis, :] * (primary_ld * secondary_ld)
    log_k_steps = numpy.diff(log_k_ld)

    # The phase-aware projection grid is intentionally non-uniform: it mixes
    # a logarithmic super-horizon scaffold with linear k phase nodes.  Simpson
    # integration is useful on the small declared grids used by correctness
    # contracts, but production-sized sparse phase ladders must retain the
    # positive rule to avoid negative lobes between unresolved oscillations.
    uniform_log_grid = bool(
        log_k_steps.size == 0
        or numpy.allclose(
            log_k_steps,
            log_k_steps[0],
            rtol=1.0e-10,
            atol=1.0e-14,
        )
    )
    if uniform_log_grid or log_k_ld.size <= 128:
        # Small declared grids retain physical anchors and benefit materially
        # from generalized Simpson integration on their smooth transfer
        # functions.  The positive fallback below still protects negative
        # auto-spectrum rows.
        integral = simpson(weighted, x=log_k_ld, axis=1)
    else:
        # A positive composite trapezoid is slightly lower order, but it is
        # stable on every irregular phase grid and cannot invent negative
        # lobes between sparsely sampled Bessel oscillations.
        integral = numpy.sum(
            0.5
            * (weighted[:, :-1] + weighted[:, 1:])
            * log_k_steps[numpy.newaxis, :],
            axis=1,
        )
    if auto_spectrum and numpy.any(integral < 0.0):
        # Generalized Simpson weights can become negative on a sparse
        # anchor grid.  Re-evaluate only those rows with the positive
        # composite trapezoid rule; replacing every row makes a spectrum
        # depend on whether an unrelated multipole happens to be negative.
        positive_integral = numpy.sum(
            0.5
            * (weighted[:, :-1] + weighted[:, 1:])
            * log_k_steps[numpy.newaxis, :],
            axis=1,
        )
        negative_rows = integral < 0.0
        integral = numpy.where(negative_rows, positive_integral, integral)
    if auto_spectrum:
        # Auto spectra are positive by construction.  Clamp only tiny
        # negative roundoff after the stable positive quadrature; a material
        # negative value is an invariant failure, not a numerical fallback.
        scale = numpy.maximum(numpy.max(numpy.abs(weighted), axis=1), 1.0)
        roundoff = numpy.finfo(float).eps * scale
        if numpy.any(integral < -roundoff):
            raise ValueError(
                "Auto-spectrum quadrature produced a negative power"
            )
        integral = numpy.maximum(integral, 0.0)
    integrated = 4.0 * numpy.longdouble(math.pi) * integral
    # Keep the raw spectrum in extended precision until the public solver
    # applies its final float conversion.
    return numpy.asarray(integrated, dtype=numpy.longdouble)


def _configured_reference_ells(
    perturbation_data: Any,
    *,
    maximum_ell: int | None = None,
) -> tuple[int, ...]:
    """Return all declared reference multipoles for the declared run."""

    controls = getattr(perturbation_data, "accuracy_controls", {}) or {}
    anchor_ells: list[int] = []
    for control_name in (
        "scalar_reference_ells",
        "vector_reference_ells",
        "tensor_reference_ells",
    ):
        raw_values = controls.get(control_name)
        if not isinstance(raw_values, (tuple, list)):
            continue
        for index, raw_value in enumerate(raw_values):
            ell_value = int(
                _coerce_numeric_scalar(
                    raw_value,
                    name=(
                        "cmb.perturbations.accuracy_controls."
                        f"{control_name}[{index}]"
                    ),
                )
            )
            if maximum_ell is None or ell_value <= int(maximum_ell):
                anchor_ells.append(ell_value)
    return tuple(sorted(set(anchor_ells)))


def _projection_anchor_ells(
    ell_arr: numpy.ndarray,
    *,
    perturbation_data: Any,
    node_budget: int,
) -> tuple[int, ...]:
    """Return ell anchors that steer the declared projection k-grid."""

    ell_values = numpy.asarray(ell_arr, dtype=int)
    ell_min = int(ell_values.min())
    ell_max = int(ell_values.max())
    required_ells = {
        ell_min,
        ell_max,
        *_configured_reference_ells(
            perturbation_data,
            maximum_ell=ell_max,
        ),
    }
    if node_budget <= len(required_ells):
        return tuple(sorted(required_ells))
    sample_count = min(int(node_budget), int(ell_values.size))
    sampled_indices = numpy.linspace(
        0,
        ell_values.size - 1,
        num=sample_count,
        dtype=int,
    )
    sampled_ells = {int(ell_values[index]) for index in sampled_indices}
    optional_ells = sorted(sampled_ells - required_ells)
    optional_budget = max(0, int(node_budget) - len(required_ells))
    if len(optional_ells) > optional_budget:
        selected_optional_indices = numpy.linspace(
            0,
            len(optional_ells) - 1,
            num=optional_budget,
            dtype=int,
        )
        optional_ells = [
            optional_ells[index]
            for index in sorted(
                set(int(index) for index in selected_optional_indices)
            )
        ]
    return tuple(sorted(required_ells | set(optional_ells)))


def _build_projection_k_grid(
    *,
    ell_arr: numpy.ndarray,
    background: Any,
    numerics: Any,
    perturbation_data: Any,
    allow_final_production_floor: bool = True,
    surface_ell_max_override: int | None = None,
) -> numpy.ndarray:
    """Return a projection k-grid that satisfies declared numerical bounds.

    Fixed-point diagnostics may request a bounded multipole surface.  That
    surface is explicit at the call site and never changes ordinary
    production requests, which continue to use the contract's full ell
    ceiling.
    """

    ell_values = numpy.asarray(ell_arr, dtype=int)
    sample_count = max(8, int(numerics.k_sample_count))
    declared_ell_max = int(getattr(numerics, "ell_max", int(ell_values.max())))
    declared_k_min = float(numerics.k_min)
    declared_k_max = float(numerics.k_max)
    if not numpy.isfinite(declared_k_min) or not numpy.isfinite(
        declared_k_max
    ):
        raise ValueError("Declared numerical k limits must be finite")
    if declared_k_min <= 0.0 or declared_k_max < declared_k_min:
        raise ValueError(
            "Declared numerical k limits must satisfy " "0 < k_min <= k_max"
        )
    eta0_floor = max(float(background.eta0), 1.0e-6)
    eta_rec_distance = max(
        float(background.eta0) - float(background.eta_rec),
        1.0,
    )
    declared_required_k_max = 1.5 * (
        (float(declared_ell_max) + 16.0) / eta_rec_distance
    )
    if surface_ell_max_override is not None:
        requested_surface_ell_max = int(surface_ell_max_override)
        if requested_surface_ell_max < int(ell_values.max()):
            raise ValueError(
                "Projection surface ell ceiling cannot exclude a requested "
                "multipole"
            )
        surface_ell_max = min(declared_ell_max, requested_surface_ell_max)
    elif declared_required_k_max <= declared_k_max:
        surface_ell_max = declared_ell_max
    else:
        # A low-ell request can still be evaluated when a model declares an
        # ell ceiling whose k ceiling is too small to support the full range.
        # Keep the request-local surface in that inconsistent case, and let
        # the preflight below reject only requests that actually exceed k_max.
        surface_ell_max = int(ell_values.max())
    configured_reference_ells = _configured_reference_ells(
        perturbation_data,
        maximum_ell=max(int(ell_values.max()), surface_ell_max),
    )
    # The declared numerical interval, rather than the selected dataset
    # rows, owns the projection surface.  A likelihood commonly requests a
    # sparse band beginning at ell=30; letting that band raise k_min would
    # remove legitimate low-k power and change the same multipoles when a
    # caller later requests the full spectrum.
    grid_ell_min = min(
        (
            int(numerics.ell_min),
            *configured_reference_ells,
        )
    )
    # The declared numerical ell ceiling defines the physical projection
    # surface.  Deriving this bound from each request makes the quadrature
    # nodes depend on which other multipoles happen to be requested, so the
    # same low-ell spectrum changes when a caller adds high-ell observations.
    # Keep the surface fixed; sparse likelihood rows only select values from
    # this shared quadrature rather than changing its physical boundaries.
    grid_ell_max = max(
        (
            int(ell_values.max()),
            surface_ell_max,
            *configured_reference_ells,
        )
    )
    k_min = max(
        declared_k_min,
        0.2 * max(float(grid_ell_min), 2.0) / eta0_floor,
    )
    required_k_max = 1.5 * ((float(grid_ell_max) + 16.0) / eta_rec_distance)
    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if manifest_summary.get("generated_tensor_hierarchy"):
        # Tensor spin-2 kernels retain an oscillatory high-k tail beyond the
        # scalar projection envelope.  Keep that tail in the fixed node
        # budget so the absolute tensor surfaces converge at the reference
        # multipoles instead of biasing EE and BB low.
        k_floor = max(12.0 * k_min, 5.0 * required_k_max)
    else:
        # Keep scalar quadrature nodes on the requested projection surface.
        # A fixed 0.08/Mpc floor spends the declared node budget on modes
        # that cannot project to the requested ell range and leaves the
        # visibility-scale Bessel oscillations under-resolved.
        k_floor = max(12.0 * k_min, required_k_max)
    if k_min > declared_k_max or required_k_max > declared_k_max:
        raise ValueError(
            "Requested projection k-grid exceeds declared numerical limits: "
            f"requested=[{k_min}, {required_k_max}], "
            f"declared=[{declared_k_min}, {declared_k_max}]"
        )
    k_max = max(required_k_max, min(declared_k_max, k_floor))
    if not numpy.isfinite(k_min) or not numpy.isfinite(k_max):
        raise ValueError("Declared projection k-grid requires finite bounds")
    if k_max <= k_min:
        return numpy.asarray((k_min,), dtype=float)

    accuracy_controls = (
        getattr(
            perturbation_data,
            "accuracy_controls",
            {},
        )
        or {}
    )
    generated_final_hierarchy = bool(
        manifest_summary.get("generated_scalar_hierarchy")
        and accuracy_controls.get("accuracy_tier") == "final"
    )
    if generated_final_hierarchy and allow_final_production_floor:
        # A 64-node ladder is adequate for contract smoke tests but cannot
        # resolve the rapidly oscillating spherical-Bessel projection at the
        # public ell ceiling.  Keep the declared value as the lower bound and
        # promote generated final spectra to a deterministic production grid.
        # Production refinement carries an explicit factor so the doubled
        # request actually doubles the physical grid rather than being hidden
        # by this floor.
        # Scale the declared ladder before applying the floor.  Otherwise a
        # 64-node base request and a 96-node refinement both collapse to the
        # same 512-node grid, so the convergence comparison does not actually
        # measure a refinement.
        refinement_factor = max(
            1,
            int(getattr(numerics, "k_grid_refinement_factor", 1)),
        )
        sample_count = max(sample_count * refinement_factor * 8, 512)
    phase_setting = accuracy_controls.get("phase_aware_k_quadrature")
    phase_aware_k_enabled = (
        bool(phase_setting)
        if phase_setting is not None
        else generated_final_hierarchy
    )
    projection_ell_values = numpy.linspace(
        grid_ell_min,
        grid_ell_max,
        num=max(2, min(sample_count, grid_ell_max - grid_ell_min + 1)),
        dtype=int,
    )
    # Use one stable physical-anchor budget throughout production-sized
    # refinements.  The remaining nodes are deterministic midpoint
    # subdivisions, so the 64-node final ladder is retained at 96 nodes.
    anchor_node_budget = min(48, max(2, sample_count - 2))
    if phase_aware_k_enabled:
        # Preserve a logarithmic low-k scaffold for the largest-scale modes.
        # A full 48-point ell anchor set consumes nearly the whole 64-node
        # budget and leaves only a handful of nodes below the first acoustic
        # projection scale, which aliases the low-ell spectrum.
        anchor_node_budget = min(16, max(4, sample_count // 4))
    anchor_ells = _projection_anchor_ells(
        projection_ell_values,
        perturbation_data=perturbation_data,
        node_budget=anchor_node_budget,
    )
    k_nodes = {
        float(k_min),
        float(k_max),
        *(
            float(
                numpy.clip(
                    (float(ell_value) + 0.5) / eta_rec_distance,
                    k_min,
                    k_max,
                )
            )
            for ell_value in anchor_ells
        ),
    }
    if not phase_aware_k_enabled:
        ordered_nodes = sorted(k_nodes)
        if len(ordered_nodes) > sample_count:
            interior_nodes = ordered_nodes[1:-1]
            interior_budget = max(0, sample_count - 2)
            if len(interior_nodes) > interior_budget:
                selected_indices = numpy.linspace(
                    0,
                    len(interior_nodes) - 1,
                    num=interior_budget,
                    dtype=int,
                )
                interior_nodes = [
                    interior_nodes[index]
                    for index in sorted(
                        set(int(index) for index in selected_indices)
                    )
                ]
            ordered_nodes = [
                ordered_nodes[0],
                *interior_nodes,
                ordered_nodes[-1],
            ]
        while len(ordered_nodes) < sample_count:
            linear_nodes = numpy.asarray(ordered_nodes, dtype=float)
            widest_gap_index = int(numpy.argmax(numpy.diff(linear_nodes)))
            midpoint = float(
                0.5
                * (
                    linear_nodes[widest_gap_index]
                    + linear_nodes[widest_gap_index + 1]
                )
            )
            if (
                not numpy.isfinite(midpoint)
                or midpoint <= ordered_nodes[widest_gap_index]
                or midpoint >= ordered_nodes[widest_gap_index + 1]
            ):
                break
            ordered_nodes.insert(widest_gap_index + 1, midpoint)
        result = numpy.asarray(ordered_nodes, dtype=float)
    else:
        anchor_nodes = tuple(sorted(float(value) for value in k_nodes))
        phase_points_per_cycle = float(
            _accuracy_control_value(
                accuracy_controls,
                "phase_points_per_cycle",
            )
            or 8.0
        )
        result = phase_aware_k_grid(
            k_min,
            k_max,
            minimum_nodes=sample_count,
            maximum_nodes=sample_count,
            phase_points_per_cycle=phase_points_per_cycle,
            eta_distance=eta_rec_distance,
            sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
            anchors=anchor_nodes,
            require_phase_resolution=bool(
                accuracy_controls.get("require_phase_resolution", False)
            ),
        )
    if (
        result.ndim != 1
        or result.size == 0
        or not numpy.all(numpy.isfinite(result))
        or numpy.any(result < declared_k_min)
        or numpy.any(result > declared_k_max)
        or (result.size > 1 and numpy.any(numpy.diff(result) <= 0.0))
    ):
        raise ValueError(
            "Requested projection k-grid does not satisfy declared numerical "
            "limits"
        )
    return result


def _projection_ell_limit_for_mode(
    *,
    ell_values: numpy.ndarray,
    x_values: numpy.ndarray,
) -> int:
    """Return the non-negligible radial-order limit for one Fourier mode.

    Spherical Bessel functions are exponentially suppressed above ``ell``
    larger than their largest radial argument.  Leaving those rows at zero
    avoids evaluating a large high-ell recurrence for low-k modes without
    changing the line-of-sight integral at floating-point precision relevant
    to the requested surface.
    """

    if ell_values.size == 0 or x_values.size == 0:
        return 0
    maximum_ell = int(numpy.max(ell_values))
    maximum_x = float(numpy.max(numpy.abs(x_values)))
    if not numpy.isfinite(maximum_x):
        raise ValueError("Projection radial arguments must be finite")
    radial_limit = int(
        math.ceil(maximum_x + 32.0 + 8.0 * math.sqrt(max(maximum_x, 0.0)))
    )
    return min(maximum_ell, max(int(numpy.min(ell_values)), radial_limit))


def _exact_linear_collision_step(
    *,
    operator_matrix: numpy.ndarray,
    dt: float,
    target_state: numpy.ndarray,
    eigendecomposition: (
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None
    ) = None,
    operator_scale: float = 1.0,
) -> numpy.ndarray:
    """Return one exact linear collision update from its matrix exponential."""

    matrix = numpy.asarray(operator_matrix, dtype=float)
    state = numpy.asarray(target_state, dtype=float)
    scaled_matrix = matrix * float(dt) * float(operator_scale)
    if scaled_matrix.size == 0 or float(dt) == 0.0:
        return numpy.asarray(state, dtype=float)
    if scaled_matrix.shape == (2, 2):
        trace_half = 0.5 * (scaled_matrix[0, 0] + scaled_matrix[1, 1])
        centered = scaled_matrix - trace_half * numpy.eye(2)
        discriminant = (
            0.25 * (scaled_matrix[0, 0] - scaled_matrix[1, 1]) ** 2
            + scaled_matrix[0, 1] * scaled_matrix[1, 0]
        )
        delta = numpy.sqrt(complex(discriminant))
        if abs(delta) <= 1.0e-14:
            evolved_state = numpy.exp(trace_half) * (state + centered @ state)
        else:
            plus = numpy.exp(trace_half + delta)
            minus = numpy.exp(trace_half - delta)
            evolved_state = 0.5 * (
                (plus + minus) * state
                + ((plus - minus) / delta) * (centered @ state)
            )
        real_state = numpy.real_if_close(evolved_state, tol=1000)
        if not numpy.iscomplexobj(real_state) and numpy.all(
            numpy.isfinite(real_state)
        ):
            return numpy.asarray(real_state, dtype=float)
    structured = _structured_collision_action(scaled_matrix, state)
    if structured is not None:
        return structured
    try:
        if eigendecomposition is None:
            eigenvalues, eigenvectors = numpy.linalg.eig(scaled_matrix)
            eigenvector_inverse = numpy.linalg.inv(eigenvectors)
        else:
            eigenvalues, eigenvectors, eigenvector_inverse = eigendecomposition
            eigenvalues = (
                numpy.asarray(eigenvalues, dtype=complex)
                * float(operator_scale)
                * float(dt)
            )
        evolved_state = eigenvectors @ (
            numpy.exp(eigenvalues) * (eigenvector_inverse @ state)
        )
        if numpy.all(numpy.isfinite(evolved_state)):
            real_state = numpy.real_if_close(evolved_state, tol=1000)
            if not numpy.iscomplexobj(real_state):
                return numpy.asarray(real_state, dtype=float)
    except (numpy.linalg.LinAlgError, FloatingPointError):
        pass
    return numpy.asarray(expm(scaled_matrix) @ state, dtype=float)


def _exact_batched_two_state_blocks(
    blocks: numpy.ndarray,
    block_states: numpy.ndarray,
) -> numpy.ndarray | None:
    """Apply exact two-state exponentials to a batch of mode rows."""

    leading_diagonal = blocks[:, 0, 0]
    trailing_diagonal = blocks[:, 1, 1]
    upper = blocks[:, 0, 1]
    lower = blocks[:, 1, 0]
    trace_half = 0.5 * (leading_diagonal + trailing_diagonal)
    centered_diagonal = 0.5 * (leading_diagonal - trailing_diagonal)
    discriminant = numpy.square(centered_diagonal) + upper * lower
    centered_action = numpy.empty_like(block_states, dtype=float)
    centered_action[:, 0] = (
        centered_diagonal * block_states[:, 0] + upper * block_states[:, 1]
    )
    centered_action[:, 1] = (
        lower * block_states[:, 0] - centered_diagonal * block_states[:, 1]
    )
    with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if numpy.all(discriminant >= 0.0):
            delta = numpy.sqrt(discriminant)
            sinh_over_delta = numpy.ones_like(delta)
            nonzero_delta = delta > 1.0e-14
            sinh_over_delta[nonzero_delta] = (
                numpy.sinh(delta[nonzero_delta]) / delta[nonzero_delta]
            )
            evolved_real = numpy.exp(trace_half)[:, None] * (
                numpy.cosh(delta)[:, None] * block_states
                + sinh_over_delta[:, None] * centered_action
            )
            if numpy.all(numpy.isfinite(evolved_real)):
                return numpy.asarray(evolved_real, dtype=float)

        delta = numpy.sqrt(numpy.asarray(discriminant, dtype=complex))
        evolved = numpy.empty(block_states.shape, dtype=complex)
        nearly_degenerate = numpy.abs(delta) <= 1.0e-14
        if numpy.any(nearly_degenerate):
            indices = numpy.flatnonzero(nearly_degenerate)
            evolved[indices] = numpy.exp(trace_half[indices])[:, None] * (
                block_states[indices] + centered_action[indices]
            )
        if numpy.any(~nearly_degenerate):
            indices = numpy.flatnonzero(~nearly_degenerate)
            plus = numpy.exp(trace_half[indices] + delta[indices])
            minus = numpy.exp(trace_half[indices] - delta[indices])
            evolved[indices] = 0.5 * (
                (plus + minus)[:, None] * block_states[indices]
                + ((plus - minus) / delta[indices])[:, None]
                * centered_action[indices]
            )
    real_evolved = numpy.real_if_close(evolved, tol=1000)
    if numpy.iscomplexobj(real_evolved) or not numpy.all(
        numpy.isfinite(real_evolved)
    ):
        return None
    return numpy.asarray(real_evolved, dtype=float)


def _exact_batched_linear_collision_step(
    *,
    operator_matrices: numpy.ndarray,
    dt: float,
    target_states: numpy.ndarray,
    operator_scales: numpy.ndarray,
    assume_block_diagonal: bool = False,
) -> numpy.ndarray:
    """Return exact collision updates for compatible mode-row matrices.

    The declared scalar hierarchy uses independent one- and two-state
    collision blocks.  Evaluate those blocks together so a shared Fourier
    batch does not repeat a small eigensystem decomposition for every row.
    Unstructured declarations retain the scalar exact operator per row.
    """

    matrices = numpy.asarray(operator_matrices, dtype=float)
    states = numpy.asarray(target_states, dtype=float)
    scales = numpy.asarray(operator_scales, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Batched collision matrices must be square mode rows")
    mode_count, state_count, _ = matrices.shape
    if states.shape != (mode_count, state_count):
        raise ValueError("Batched collision states do not match matrices")
    if scales.shape != (mode_count,):
        raise ValueError("Batched collision scales do not match matrices")
    if not (
        numpy.all(numpy.isfinite(matrices))
        and numpy.all(numpy.isfinite(states))
        and numpy.all(numpy.isfinite(scales))
    ):
        raise ValueError("Batched collision inputs must be finite")
    if float(dt) == 0.0 or state_count == 0:
        return states.copy()

    scaled_matrices = matrices * scales[:, numpy.newaxis, numpy.newaxis]
    scaled_matrices *= float(dt)

    if assume_block_diagonal and state_count == 4:
        if numpy.all(scaled_matrices[:, :2, 2:] == 0.0) and numpy.all(
            scaled_matrices[:, 2:, :2] == 0.0
        ):
            leading = _exact_batched_two_state_blocks(
                scaled_matrices[:, :2, :2],
                states[:, :2],
            )
            trailing = _exact_batched_two_state_blocks(
                scaled_matrices[:, 2:, 2:],
                states[:, 2:],
            )
            if leading is not None and trailing is not None:
                return numpy.concatenate((leading, trailing), axis=1)

    def _apply_two_state_blocks(
        blocks: numpy.ndarray,
        block_states: numpy.ndarray,
    ) -> numpy.ndarray | None:
        """Apply the scalar two-state exponential formula to every row."""

        leading_diagonal = blocks[:, 0, 0]
        trailing_diagonal = blocks[:, 1, 1]
        upper = blocks[:, 0, 1]
        lower = blocks[:, 1, 0]
        trace_half = 0.5 * (leading_diagonal + trailing_diagonal)
        centered_diagonal = 0.5 * (leading_diagonal - trailing_diagonal)
        discriminant = numpy.square(centered_diagonal) + upper * lower
        centered_action = numpy.empty_like(block_states, dtype=float)
        centered_action[:, 0] = (
            centered_diagonal * block_states[:, 0] + upper * block_states[:, 1]
        )
        centered_action[:, 1] = (
            lower * block_states[:, 0] - centered_diagonal * block_states[:, 1]
        )
        with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if numpy.all(discriminant >= 0.0):
                delta = numpy.sqrt(discriminant)
                sinh_over_delta = numpy.ones_like(delta)
                nonzero_delta = delta > 1.0e-14
                sinh_over_delta[nonzero_delta] = (
                    numpy.sinh(delta[nonzero_delta]) / delta[nonzero_delta]
                )
                evolved_real = numpy.exp(trace_half)[:, None] * (
                    numpy.cosh(delta)[:, None] * block_states
                    + sinh_over_delta[:, None] * centered_action
                )
                if numpy.all(numpy.isfinite(evolved_real)):
                    return numpy.asarray(evolved_real, dtype=float)

            delta = numpy.sqrt(numpy.asarray(discriminant, dtype=complex))
            evolved = numpy.empty(block_states.shape, dtype=complex)
            nearly_degenerate = numpy.abs(delta) <= 1.0e-14
            if numpy.any(nearly_degenerate):
                indices = numpy.flatnonzero(nearly_degenerate)
                evolved[indices] = numpy.exp(trace_half[indices])[:, None] * (
                    block_states[indices] + centered_action[indices]
                )
            if numpy.any(~nearly_degenerate):
                indices = numpy.flatnonzero(~nearly_degenerate)
                plus = numpy.exp(trace_half[indices] + delta[indices])
                minus = numpy.exp(trace_half[indices] - delta[indices])
                evolved[indices] = 0.5 * (
                    (plus + minus)[:, None] * block_states[indices]
                    + ((plus - minus) / delta[indices])[:, None]
                    * centered_action[indices]
                )
        real_evolved = numpy.real_if_close(evolved, tol=1000)
        if numpy.iscomplexobj(real_evolved) or not numpy.all(
            numpy.isfinite(real_evolved)
        ):
            return None
        return numpy.asarray(real_evolved, dtype=float)

    batched_result: numpy.ndarray | None = None
    if state_count == 1:
        batched_result = numpy.exp(scaled_matrices[:, 0, 0])[:, None] * states
    elif state_count == 2:
        batched_result = _apply_two_state_blocks(scaled_matrices, states)
    elif (
        state_count == 4
        and numpy.all(scaled_matrices[:, :2, 2:] == 0.0)
        and numpy.all(scaled_matrices[:, 2:, :2] == 0.0)
    ):
        leading = _apply_two_state_blocks(
            scaled_matrices[:, :2, :2],
            states[:, :2],
        )
        trailing = _apply_two_state_blocks(
            scaled_matrices[:, 2:, 2:],
            states[:, 2:],
        )
        if leading is not None and trailing is not None:
            batched_result = numpy.concatenate((leading, trailing), axis=1)
    if batched_result is not None and numpy.all(
        numpy.isfinite(batched_result)
    ):
        return numpy.asarray(batched_result, dtype=float)

    return numpy.asarray(
        [
            _exact_linear_collision_step(
                operator_matrix=matrix,
                dt=float(dt),
                target_state=state,
                operator_scale=float(scale),
            )
            for matrix, state, scale in zip(matrices, states, scales)
        ],
        dtype=float,
    )


def _structured_collision_action(
    scaled_matrix: numpy.ndarray,
    state: numpy.ndarray,
) -> numpy.ndarray | None:
    """Evaluate exact actions for scalar or two-state matrix blocks."""

    if scaled_matrix.ndim != 2 or scaled_matrix.shape[0] <= 2:
        return None
    if scaled_matrix.shape[0] != scaled_matrix.shape[1]:
        return None
    components = _structured_collision_components(scaled_matrix)
    if components is None:
        return None
    evolved = numpy.asarray(state, dtype=float).copy()
    for component in components:
        indices = numpy.asarray(component, dtype=int)
        block = scaled_matrix[numpy.ix_(indices, indices)]
        if indices.size == 1:
            evolved[indices[0]] = numpy.exp(block[0, 0]) * state[indices[0]]
            continue
        evolved[indices] = _exact_linear_collision_step(
            operator_matrix=block,
            dt=1.0,
            target_state=state[indices],
        )
    if not numpy.all(numpy.isfinite(evolved)):
        return None
    return evolved


def _structured_collision_components(
    matrix: numpy.ndarray,
) -> tuple[tuple[int, ...], ...] | None:
    """Return collision blocks that can use scalar or two-state actions."""

    normalized = numpy.asarray(matrix, dtype=float)
    if normalized.ndim != 2 or normalized.shape[0] <= 2:
        return None
    if normalized.shape[0] != normalized.shape[1]:
        return None
    if (
        normalized.shape == (4, 4)
        and numpy.all(normalized[:2, 2:] == 0.0)
        and numpy.all(normalized[2:, :2] == 0.0)
    ):
        return ((0, 1), (2, 3))
    adjacency = normalized != 0.0
    numpy.fill_diagonal(adjacency, False)
    components: list[tuple[int, ...]] = []
    unseen = set(range(normalized.shape[0]))
    while unseen:
        start = min(unseen)
        component = {start}
        frontier = [start]
        unseen.remove(start)
        while frontier:
            index = frontier.pop()
            neighbors = set(numpy.flatnonzero(adjacency[index]))
            neighbors.update(numpy.flatnonzero(adjacency[:, index]))
            for neighbor in neighbors & unseen:
                neighbor_index = int(neighbor)
                unseen.remove(neighbor_index)
                component.add(neighbor_index)
                frontier.append(neighbor_index)
        components.append(tuple(sorted(component)))
    if any(len(component) > 2 for component in components):
        return None
    return tuple(components)


def _cached_collision_eigendecomposition(
    matrix: numpy.ndarray,
    cache: dict[
        tuple[tuple[int, ...], bytes],
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None,
    ],
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None:
    """Return a reusable eigensystem for one static collision matrix."""

    normalized = numpy.asarray(matrix, dtype=float)
    cache_key = (
        tuple(int(size) for size in normalized.shape),
        normalized.tobytes(),
    )
    if cache_key in cache:
        return cache[cache_key]
    try:
        eigenvalues, eigenvectors = numpy.linalg.eig(normalized)
        condition = numpy.linalg.cond(eigenvectors)
        if not numpy.isfinite(condition) or condition >= 1.0e10:
            cache[cache_key] = None
            return None
        decomposition = (
            numpy.asarray(eigenvalues, dtype=complex),
            numpy.asarray(eigenvectors, dtype=complex),
            numpy.asarray(numpy.linalg.inv(eigenvectors), dtype=complex),
        )
    except (numpy.linalg.LinAlgError, FloatingPointError):
        cache[cache_key] = None
        return None
    cache[cache_key] = decomposition
    return decomposition


def _primordial_power_grid_for_observable(
    *,
    physical_params: _CustomCMBPhysicalParameters,
    perturbation_data: Any,
    observable_entry: Any,
    k_values: numpy.ndarray,
) -> numpy.ndarray:
    """Return the primordial power grid driving ``observable_entry``."""

    sector = str(getattr(observable_entry, "sector", "") or "")
    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if sector == "tensor" and bool(
        manifest_summary.get("generated_tensor_hierarchy")
    ):
        tensor_ratio = getattr(physical_params, "tensor_to_scalar_ratio", None)
        tensor_tilt = getattr(
            physical_params,
            "tensor_spectral_index",
            None,
        )
        amplitude = float(physical_params.primordial_amplitude) * float(
            0.0 if tensor_ratio is None else max(float(tensor_ratio), 0.0)
        )
        # The declared tensor metric seed is h=1. CAMB/CLASS tensor power
        # conventions put the compensating 1/6 in the primordial spectrum.
        amplitude /= 6.0
        exponent = 0.0 if tensor_tilt is None else float(tensor_tilt)
    else:
        amplitude = float(physical_params.primordial_amplitude)
        exponent = float(physical_params.primordial_spectral_index) - 1.0
    return amplitude * numpy.power(k_values / 0.05, exponent)


def _integrate_declared_spectra(
    *,
    physical_params: _CustomCMBPhysicalParameters,
    perturbation_data: Any,
    power_spectrum_observables: Mapping[str, Any],
    transfer_components: Mapping[str, numpy.ndarray],
    k_values: numpy.ndarray,
    log_k_values: numpy.ndarray,
) -> dict[str, numpy.ndarray]:
    """Integrate declared spectra from transfer products and current tilt."""

    spectra_results: dict[str, numpy.ndarray] = {}
    for (
        observable_name,
        observable_entry,
    ) in power_spectrum_observables.items():
        primordial_grid = _primordial_power_grid_for_observable(
            physical_params=physical_params,
            perturbation_data=perturbation_data,
            observable_entry=observable_entry,
            k_values=k_values,
        )
        primary = numpy.asarray(
            transfer_components[str(observable_entry.primary)],
            dtype=numpy.longdouble,
        )
        secondary = numpy.asarray(
            transfer_components[str(observable_entry.secondary)],
            dtype=numpy.longdouble,
        )
        spectra_results[observable_name] = _integrate_power_spectrum(
            primordial_grid=primordial_grid,
            log_k_values=log_k_values,
            primary=primary,
            secondary=secondary,
            auto_spectrum=(
                str(observable_entry.primary)
                == str(observable_entry.secondary)
            ),
        )
    return spectra_results


def _declared_graph_projection(
    *,
    projection: str,
    kernel: str | None,
    sector: str | None = None,
    kernel_batch: _DeclaredProjectionKernelBatch,
    k_value: float,
    eta_weights: numpy.ndarray,
    chi_grid: numpy.ndarray,
    source_chi: float,
    source_histories: Mapping[str, numpy.ndarray],
) -> numpy.ndarray:
    """Return projected transfer component values for every ell."""

    if not source_histories:
        raise ValueError(
            f"Declared projection '{projection}' has no available source "
            "histories"
        )

    j_l = kernel_batch.j_l
    j_l_derivative = kernel_batch.j_l_derivative
    j_l_second_derivative = kernel_batch.j_l_second_derivative
    e_kernel = kernel_batch.e_kernel
    b_kernel = kernel_batch.b_kernel
    sector_name = "" if sector is None else str(sector)
    validate_declared_projection_sector(
        projection,
        sector_name or None,
        observable_name=projection,
        kernel=kernel,
    )

    if sector_name == "vector":
        temperature_kernel = kernel_batch.vector_temperature_1
        e_projection_kernel = kernel_batch.vector_e
        b_projection_kernel = kernel_batch.vector_b
    elif sector_name == "tensor":
        temperature_kernel = kernel_batch.tensor_temperature
        e_projection_kernel = kernel_batch.tensor_e
        b_projection_kernel = kernel_batch.tensor_b
    else:
        temperature_kernel = j_l
        e_projection_kernel = e_kernel
        b_projection_kernel = b_kernel

    def _apply_kernel(kernel_name: str) -> numpy.ndarray:
        """Return the ell-batched kernel selected by ``kernel_name``."""

        kernel_spec = get_declared_projection_kernel_spec(kernel_name)
        if kernel_spec.kind == "temperature_mixed":
            raise ValueError(
                "Temperature mixed kernels must use the dedicated "
                "temperature projection dispatch."
            )
        if kernel_spec.kind == "spherical_bessel":
            if (
                sector_name == "vector"
                and projection == "line_of_sight_vector_polarization_e"
            ):
                return e_projection_kernel
            if (
                sector_name == "vector"
                and projection == "line_of_sight_vector_polarization_b"
            ):
                return b_projection_kernel
            return temperature_kernel
        if kernel_spec.kind == "spherical_bessel_derivative":
            return j_l_derivative
        if kernel_spec.kind == "spherical_bessel_second_derivative":
            return j_l_second_derivative
        if kernel_spec.kind == "spin2_e":
            return e_projection_kernel
        if kernel_spec.kind == "spin2_b":
            return b_projection_kernel
        if kernel_spec.kind == "lensing_potential":
            geometry = numpy.clip(source_chi - chi_grid, 0.0, None) / (
                max(float(source_chi), 1.0e-12)
                * numpy.maximum(chi_grid, 1.0e-12)
            )
            return -j_l * geometry[numpy.newaxis, :]
        raise ValueError(
            "Declared observable requests unsupported kernel "
            f"'{kernel_name}'"
        )

    def _project_history(
        kernel_values: numpy.ndarray,
        history: numpy.ndarray,
    ) -> numpy.ndarray:
        """Project one source history through one ell-batched kernel."""

        return numpy.asarray(
            kernel_values @ (eta_weights * history),
            dtype=float,
        )

    if projection in SUPPORTED_DECLARED_TRANSFER_PROJECTIONS:
        projected = numpy.zeros(j_l.shape[0], dtype=float)
        for role_name, history in source_histories.items():
            source_kernel = resolve_declared_source_kernel(
                projection,
                role_name,
                kernel=kernel,
            )
            validate_declared_projection_sector(
                projection,
                sector_name or None,
                observable_name=projection,
                kernel=source_kernel,
            )
            projected += _project_history(
                _apply_kernel(source_kernel),
                history,
            )
        return projected
    raise ValueError(
        "Declared observable requests unsupported projection "
        f"'{projection}'"
    )


def _bind_declared_source_histories(
    *,
    component_name: str,
    component_entry: Any,
    source_arrays: Mapping[str, numpy.ndarray],
) -> dict[str, numpy.ndarray]:
    """Resolve a component's declared roles without fabricating sources."""

    source_terms = {
        str(role_name): str(source_name)
        for role_name, source_name in component_entry.source_terms.items()
    }
    missing = sorted(
        source_name
        for source_name in source_terms.values()
        if source_name not in source_arrays
    )
    if missing:
        raise ValueError(
            f"Declared transfer component '{component_name}' source "
            "histories unavailable: " + ", ".join(missing)
        )
    if not source_terms:
        raise ValueError(
            f"Declared transfer component '{component_name}' has no "
            "declared source histories"
        )
    return {
        role_name: numpy.asarray(source_arrays[source_name], dtype=float)
        for role_name, source_name in source_terms.items()
    }


def _slice_projection_kernel_batch(
    kernel_batch: _DeclaredProjectionKernelBatch,
    indices: numpy.ndarray,
) -> _DeclaredProjectionKernelBatch:
    """Return one radial-kernel batch restricted to an eta subset."""

    selected = numpy.asarray(indices, dtype=int)
    return _DeclaredProjectionKernelBatch(
        j_l=kernel_batch.j_l[:, selected],
        j_l_derivative=kernel_batch.j_l_derivative[:, selected],
        j_l_second_derivative=kernel_batch.j_l_second_derivative[:, selected],
        e_kernel=kernel_batch.e_kernel[:, selected],
        b_kernel=kernel_batch.b_kernel[:, selected],
        vector_temperature_1=kernel_batch.vector_temperature_1[:, selected],
        vector_temperature_2=kernel_batch.vector_temperature_2[:, selected],
        vector_e=kernel_batch.vector_e[:, selected],
        vector_b=kernel_batch.vector_b[:, selected],
        tensor_temperature=kernel_batch.tensor_temperature[:, selected],
        tensor_e=kernel_batch.tensor_e[:, selected],
        tensor_b=kernel_batch.tensor_b[:, selected],
    )


def _trapezoid_weights(grid: numpy.ndarray) -> numpy.ndarray:
    """Return composite trapezoid weights for a strictly ordered grid."""

    coordinates = numpy.asarray(grid, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("A trapezoid grid requires at least two samples")
    steps = numpy.diff(coordinates)
    if not numpy.all(numpy.isfinite(coordinates)) or numpy.any(steps <= 0.0):
        raise ValueError("A trapezoid grid must be finite and increasing")
    weights = numpy.zeros_like(coordinates, dtype=float)
    weights[0] = 0.5 * steps[0]
    weights[-1] = 0.5 * steps[-1]
    if coordinates.size > 2:
        weights[1:-1] = 0.5 * (steps[:-1] + steps[1:])
    return weights


def _simpson_weights(grid: numpy.ndarray) -> numpy.ndarray:
    """Return linear weights for nonuniform composite Simpson quadrature."""

    eta_grid = numpy.asarray(grid, dtype=float)
    step_sizes = numpy.diff(eta_grid)
    if eta_grid.size < 2 or not numpy.all(numpy.isfinite(eta_grid)):
        raise ValueError("eta_los_grid must contain a finite grid")
    if not numpy.all(numpy.isfinite(step_sizes)) or numpy.any(
        step_sizes <= 0.0
    ):
        raise ValueError("eta_los_grid must be strictly increasing")

    weights = numpy.zeros_like(eta_grid, dtype=float)
    simpson_stop = eta_grid.size
    if eta_grid.size % 2 == 0:
        # Apply Simpson to the odd-sized prefix and retain a second-order
        # endpoint rule for the unavoidable final interval.
        simpson_stop -= 1
        weights[-2:] += 0.5 * step_sizes[-1]
    for start in range(0, simpson_stop - 2, 2):
        left_step = step_sizes[start]
        right_step = step_sizes[start + 1]
        ratio_limit = 2.0 * (1.0 + 1.0e-12)
        if (
            right_step > ratio_limit * left_step
            or left_step > ratio_limit * right_step
        ):
            # Generalized Simpson weights become negative when adjacent
            # intervals differ by more than two.  Merged physical grids can
            # contain near-coincident background anchors, so use the
            # positive trapezoid rule for only that unstable interval pair.
            weights[start] += 0.5 * left_step
            weights[start + 1] += 0.5 * (left_step + right_step)
            weights[start + 2] += 0.5 * right_step
            continue
        total_step = left_step + right_step
        weights[start] += (
            total_step * (2.0 * left_step - right_step) / (6.0 * left_step)
        )
        weights[start + 1] += total_step**3 / (6.0 * left_step * right_step)
        weights[start + 2] += (
            total_step * (2.0 * right_step - left_step) / (6.0 * right_step)
        )
    return weights


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


def _compute_custom_cmb_spectrum_data_impl(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
    requested_spectra: Iterable[str] | None = None,
    diagnostic_source_audit: bool = False,
    performance_timer: PhaseTimer,
) -> CustomCMBSpectrumData:
    """Return transfer functions and spectra for a declared CMB graph."""

    request_started = perf_counter()
    requested_spectrum_names = None
    if requested_spectra is not None:
        requested_spectrum_names = {
            canonical_cmb_spectrum_name(name) for name in requested_spectra
        }
    cache_key = _custom_cmb_spectrum_cache_key(
        contract_or_params,
        ells,
        background_provider,
        requested_spectra=requested_spectrum_names,
    )
    cached_spectrum = (
        None if diagnostic_source_audit else cache.get_cmb_spectrum(cache_key)
    )
    if cached_spectrum is not None:
        performance_timer.mark_cache_state("exact_cache_hit")
        return _get_cached_custom_cmb_spectrum_data(cache_key)

    cache_stats_before = cache.cmb_cache_stats()
    graph_cache_before = cache_stats_before["declared_graph_execution_plan"]
    runtime_asset_cache_before = cache_stats_before["runtime_assets"]
    with performance_timer.phase("compilation"):
        perturbation_data = _compile_declared_perturbation_contract(
            contract_or_params
        )
        runtime_assets = prepare_runtime_assets(
            str(contract_or_params.get("runtime_signature", "")),
            perturbation_data,
        )
        execution_plan = runtime_assets.execution_plan
        envelope_contract = dict(contract_or_params)
        envelope_contract["perturbation_data"] = perturbation_data
        numerical_envelope = resolve_declared_numerical_envelope(
            envelope_contract
        )
    value_steps_by_name = {
        str(step.output_name): step for step in execution_plan.value_steps
    }
    equation_direct_names: set[str] = {
        str(dependency)
        for slot_plan in execution_plan.equation_slot_plans
        if slot_plan.compiled_rhs is not None
        for dependency in slot_plan.compiled_rhs.dependencies
    }
    equation_required_names = set(equation_direct_names)
    pending_required_names = list(equation_required_names)
    while pending_required_names:
        dependency_name = pending_required_names.pop()
        value_step = value_steps_by_name.get(dependency_name)
        if value_step is None:
            continue
        for dependency in value_step.dependencies:
            dependency_name = str(dependency)
            if dependency_name in equation_required_names:
                continue
            equation_required_names.add(dependency_name)
            pending_required_names.append(dependency_name)

    stage_required_names = set(equation_required_names)
    stage_required_names.update(
        {
            "einstein_energy_residual",
            "einstein_momentum_residual",
            "einstein_shear_residual",
            "total_density_source",
            "matter_density_source",
            "radiation_density_source",
            "total_momentum_source",
            "matter_momentum_source",
            "radiation_momentum_source",
            "total_shear_source",
        }
    )
    stage_required_names.update(
        str(entry.target)
        for entry in (
            getattr(perturbation_data, "constraints", {}) or {}
        ).values()
        if str(getattr(entry, "role", "")) == "initial_series"
    )
    pending_required_names = list(stage_required_names)
    while pending_required_names:
        dependency_name = pending_required_names.pop()
        value_step = value_steps_by_name.get(dependency_name)
        if value_step is None:
            continue
        for dependency in value_step.dependencies:
            dependency_name = str(dependency)
            if dependency_name in stage_required_names:
                continue
            stage_required_names.add(dependency_name)
            pending_required_names.append(dependency_name)
    stage_derivative_steps = tuple(
        step
        for step in execution_plan.derivative_steps
        if step.output_name in stage_required_names
    )
    equation_stage_derivative_steps = tuple(
        step
        for step in execution_plan.derivative_steps
        if step.output_name in equation_required_names
    )
    stage_value_steps = tuple(
        step
        for step in execution_plan.value_steps
        if step.output_name in stage_required_names
    )
    runtime_spec = execution_plan.runtime_spec
    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    generated_scalar_hierarchy = bool(
        manifest_summary.get("generated_scalar_hierarchy")
    )
    background_cache_before = cache.cmb_cache_stats()["background"]
    with performance_timer.phase("background"):
        physical_params = _resolve_custom_cmb_physical_parameters(
            contract_or_params,
            background_provider,
        )
        numerics = _resolve_custom_cmb_numerics(contract_or_params)
        background = _build_custom_cmb_background(
            contract_or_params,
            physical_params,
            numerics,
            background_provider=background_provider,
        )

    ell_arr = numpy.asarray(list(ells), dtype=int)
    if ell_arr.size == 0:
        raise ValueError("ells must not be empty")

    a_initial = max(
        background.a_grid[0],
        1.0 / (max(numerics.initial_redshift, 1.0) + 1.0),
    )
    eta_start = float(background.eta_of_a(a_initial))
    eta_los_grid = numpy.asarray(
        background.eta_grid[background.eta_grid >= eta_start],
        dtype=float,
    )
    eta_los_refinement = max(1, int(numerics.source_grid_multiplier))
    eta_los_grid = _refine_eta_grid(
        eta_los_grid,
        refinement=eta_los_refinement,
    )
    minimum_eta_samples = max(
        16,
        int(numerics.eta_sample_count) * eta_los_refinement,
    )
    if eta_los_grid.size < minimum_eta_samples:
        eta_los_grid = _densify_eta_grid(
            eta_los_grid,
            minimum_samples=minimum_eta_samples,
        )
    declared_accuracy_controls = _resolve_declared_accuracy_controls(
        contract_or_params
    )
    if bool(
        declared_accuracy_controls.get(
            "require_physical_source_residuals", False
        )
    ):
        # A strict physical-residual contract must retain the raw terms used
        # by the independent audit, even for an ordinary production request.
        diagnostic_source_audit = True
    los_quadrature_controls = resolve_los_quadrature_controls(
        declared_accuracy_controls,
        base_eta_nodes=int(eta_los_grid.size),
    )
    generated_final_evolution_floor = None
    if (
        generated_scalar_hierarchy
        and str(declared_accuracy_controls.get("accuracy_tier", "")) == "final"
        and los_quadrature_controls.enabled
    ):
        # The hierarchy must retain enough history samples to represent the
        # same phase surface that the line-of-sight quadrature resolves.
        # Interpolating a sparse evolution history onto a dense LOS grid
        # aliases the acoustic source before projection begins.
        generated_final_evolution_floor = max(
            int(los_quadrature_controls.minimum_nodes),
            int(los_quadrature_controls.maximum_nodes),
        )
    adaptive_controls = resolve_adaptive_controls(
        declared_accuracy_controls,
        base_k_nodes=int(numerics.k_sample_count),
        base_eta_nodes=int(eta_los_grid.size),
        base_evolution_nodes=numerics.evolution_eta_sample_count,
    )
    if adaptive_controls.evolution_enabled:
        if numerics.evolution_eta_sample_count is None:
            raise ValueError(
                "adaptive_evolution requires declared "
                "evolution_eta_sample_count"
            )
        if int(numerics.evolution_eta_sample_count) < 64:
            raise ValueError(
                "adaptive_evolution requires evolution_eta_sample_count "
                "of at least 64"
            )
    los_phase_quadrature_applied = False
    if (
        los_quadrature_controls.enabled
        and not adaptive_controls.source_enabled
        and not adaptive_controls.projection_enabled
    ):
        eta_los_grid = phase_aware_eta_grid(
            eta_los_grid,
            visibility=numpy.asarray(
                background.visibility_of_eta(eta_los_grid),
                dtype=float,
            ),
            k_max=float(numerics.k_max),
            minimum_nodes=max(
                int(los_quadrature_controls.minimum_nodes),
                int(eta_los_grid.size),
            ),
            maximum_nodes=int(los_quadrature_controls.maximum_nodes),
            phase_points_per_cycle=(
                los_quadrature_controls.phase_points_per_cycle
            ),
        )
        los_phase_quadrature_applied = True
    if adaptive_controls.source_enabled:
        eta_los_grid = phase_aware_eta_grid(
            eta_los_grid,
            visibility=numpy.asarray(
                background.visibility_of_eta(eta_los_grid),
                dtype=float,
            ),
            k_max=float(numerics.k_max),
            minimum_nodes=max(
                int(adaptive_controls.source_minimum_nodes),
                int(eta_los_grid.size),
            ),
            maximum_nodes=int(adaptive_controls.source_maximum_nodes),
            phase_points_per_cycle=(adaptive_controls.phase_points_per_cycle),
        )
    if (
        adaptive_controls.projection_enabled
        and not adaptive_controls.source_enabled
    ):
        eta_los_grid = phase_aware_eta_grid(
            eta_los_grid,
            visibility=numpy.asarray(
                background.visibility_of_eta(eta_los_grid),
                dtype=float,
            ),
            k_max=float(numerics.k_max),
            minimum_nodes=max(
                int(adaptive_controls.projection_minimum_nodes),
                int(eta_los_grid.size),
            ),
            maximum_nodes=int(adaptive_controls.projection_maximum_nodes),
            phase_points_per_cycle=(adaptive_controls.phase_points_per_cycle),
        )

    def _sample_eta_background_grids(
        eta_grid: numpy.ndarray,
    ) -> tuple[
        dict[str, numpy.ndarray],
        dict[str, numpy.ndarray],
        dict[str, numpy.ndarray],
    ]:
        """Return sampled background histories and coordinate rates."""

        eta_background = background.sample(eta_grid)
        a_grid = numpy.asarray(eta_background["a"], dtype=float)
        z_grid = numpy.asarray(eta_background["z"], dtype=float)
        H_grid = numpy.asarray(eta_background["H"], dtype=float)
        tau_grid = numpy.asarray(eta_background["tau"], dtype=float)
        tau_dot_grid = numpy.asarray(
            eta_background["tau_dot"],
            dtype=float,
        )
        visibility_grid = numpy.asarray(
            eta_background["visibility"],
            dtype=float,
        )
        chi_grid = numpy.asarray(
            eta_background["chi"],
            dtype=float,
        )
        angular_diameter_distance_grid = numpy.asarray(
            eta_background["angular_diameter_distance"],
            dtype=float,
        )
        sound_speed_grid = numpy.asarray(
            eta_background["sound_speed"],
            dtype=float,
        )
        baryon_sound_speed_sq_grid = numpy.asarray(
            eta_background["baryon_sound_speed_sq"],
            dtype=float,
        )
        Hconf_grid = a_grid * H_grid / _C_LIGHT_KM_S
        Hconf_tau_grid = _nonuniform_gradient(Hconf_grid, eta_grid)
        baryon_loading_grid = (
            3.0
            * physical_params.Omega_b0
            * a_grid
            / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
        )
        collision_rate_grid = numpy.maximum(-tau_dot_grid, 0.0)
        free_streaming_grid = 1.0 / (
            1.0
            + collision_rate_grid / max(float(collision_rate_grid.max()), 1.0)
        )
        sound_speed_sq_grid = 1.0 / (3.0 * (1.0 + baryon_loading_grid))
        declared_background = _resolve_declared_background_context(
            contract_or_params,
            a_values=a_grid,
            z_values=z_grid,
        )
        declared_background_histories: dict[str, numpy.ndarray] = {}
        for name, raw_value in declared_background.items():
            if name in {"a", "z"}:
                continue
            history = numpy.asarray(raw_value, dtype=float)
            if history.ndim == 0:
                history = numpy.full_like(
                    eta_grid,
                    float(history),
                    dtype=float,
                )
            if history.shape != eta_grid.shape:
                raise ValueError(
                    "Declared background symbol did not match the "
                    f"line-of-sight grid: {name}"
                )
            if not numpy.all(numpy.isfinite(history)):
                raise ValueError(
                    "Declared background symbol produced non-finite values: "
                    f"{name}"
                )
            declared_background_histories[name] = history
        coordinate_histories = {
            "a": a_grid,
            "z": z_grid,
            "eta": eta_grid,
            "H": H_grid,
            "Hconf": Hconf_grid,
            "Hconf_tau": Hconf_tau_grid,
            "tau": tau_grid,
            "tau_dot": tau_dot_grid,
            "visibility": visibility_grid,
            "chi": chi_grid,
            "angular_diameter_distance": angular_diameter_distance_grid,
            "sound_speed": sound_speed_grid,
            "baryon_sound_speed_sq": baryon_sound_speed_sq_grid,
        }
        for name, history in declared_background_histories.items():
            coordinate_histories.setdefault(name, history)
        coordinate_rate_histories = {
            "eta": numpy.ones_like(eta_grid, dtype=float)
        }
        for name, history in coordinate_histories.items():
            if name == "eta":
                continue
            coordinate_rate_histories[name] = _nonuniform_gradient(
                history,
                eta_grid,
            )
        coordinate_rate_histories["a"] = numpy.asarray(
            a_grid * Hconf_grid,
            dtype=float,
        )
        coordinate_rate_histories["z"] = numpy.asarray(
            -(1.0 + z_grid) * Hconf_grid,
            dtype=float,
        )
        return (
            {
                "eta": numpy.asarray(eta_grid, dtype=float),
                "a": a_grid,
                "z": z_grid,
                "H": H_grid,
                "Hconf": Hconf_grid,
                "Hconf_tau": Hconf_tau_grid,
                "tau": tau_grid,
                "tau_dot": tau_dot_grid,
                "visibility": visibility_grid,
                "chi": chi_grid,
                "angular_diameter_distance": angular_diameter_distance_grid,
                "sound_speed": sound_speed_grid,
                "sound_speed_sq": sound_speed_sq_grid,
                "baryon_sound_speed_sq": baryon_sound_speed_sq_grid,
                "baryon_loading": baryon_loading_grid,
                "collision_rate": collision_rate_grid,
                "free_streaming": free_streaming_grid,
            },
            declared_background_histories,
            coordinate_rate_histories,
        )

    with performance_timer.phase("preparation"):
        (
            source_grids,
            source_declared_background_histories,
            source_coordinate_rate_histories,
        ) = _sample_eta_background_grids(eta_los_grid)
    active_grids = dict(source_grids)
    active_declared_background_histories = source_declared_background_histories
    active_coordinate_rate_histories = source_coordinate_rate_histories
    active_k_value = 0.0
    shared_generated_mode_grids: (
        tuple[
            dict[str, numpy.ndarray],
            dict[str, numpy.ndarray],
            dict[str, numpy.ndarray],
        ]
        | None
    ) = None
    # Generated scalar modes use one common evolution schedule.  When a
    # contract omits an explicit evolution sample count, deriving a separate
    # super-horizon prefix for every k disables batching and makes the fixed
    # corpus audit needlessly serial.  The shared schedule is anchored to the
    # largest requested k in ``_mode_grids_for_k`` so every mode still retains
    # the longest required early-time prefix.
    shared_generated_mode_grids_enabled = bool(generated_scalar_hierarchy)

    with performance_timer.phase("preparation"):
        k_values = _build_projection_k_grid(
            ell_arr=ell_arr,
            background=background,
            numerics=numerics,
            perturbation_data=perturbation_data,
            allow_final_production_floor=not bool(
                contract_or_params.get("_joint_mcmc_fast_path", False)
                or contract_or_params.get(
                    "_diagnostic_matrix_fast_path", False
                )
            ),
            surface_ell_max_override=(
                int(ell_arr.max())
                if contract_or_params.get(
                    "_diagnostic_matrix_fast_path", False
                )
                else None
            ),
        )
        if adaptive_controls.transfer_enabled:
            eta_rec_distance = max(
                float(background.eta0) - float(background.eta_rec),
                1.0,
            )
            adaptive_anchors = tuple(
                float(value)
                for value in k_values
                if float(k_values[0]) <= float(value) <= float(k_values[-1])
            )
            k_values = phase_aware_k_grid(
                float(k_values[0]),
                float(k_values[-1]),
                minimum_nodes=max(
                    int(adaptive_controls.transfer_minimum_nodes),
                    int(k_values.size),
                ),
                maximum_nodes=int(adaptive_controls.transfer_maximum_nodes),
                phase_points_per_cycle=(
                    adaptive_controls.phase_points_per_cycle
                ),
                eta_distance=eta_rec_distance,
                sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
                anchors=adaptive_anchors,
                require_phase_resolution=bool(
                    declared_accuracy_controls.get(
                        "require_phase_resolution", False
                    )
                ),
            )

    phase_setting = declared_accuracy_controls.get("phase_aware_k_quadrature")
    phase_aware_k_enabled = (
        bool(phase_setting)
        if phase_setting is not None
        else generated_scalar_hierarchy
        and declared_accuracy_controls.get("accuracy_tier") == "final"
    )
    if phase_aware_k_enabled:
        phase_requirements = phase_aware_k_grid_requirements(
            float(k_values[0]),
            float(k_values[-1]),
            phase_points_per_cycle=float(
                _accuracy_control_value(
                    declared_accuracy_controls,
                    "phase_points_per_cycle",
                )
                or 8.0
            ),
            eta_distance=max(
                float(background.eta0) - float(background.eta_rec),
                1.0,
            ),
            sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
        )
    else:
        phase_requirements = {
            "radial_required_nodes": 0,
            "acoustic_required_nodes": 0,
            "required_nodes": 0,
            "phase_step": 0.0,
        }
    if k_values.size >= 2:
        phase_status = phase_aware_k_grid_status(
            k_values,
            phase_points_per_cycle=float(
                _accuracy_control_value(
                    declared_accuracy_controls,
                    "phase_points_per_cycle",
                )
                or 8.0
            ),
            eta_distance=max(
                float(background.eta0) - float(background.eta_rec),
                1.0,
            ),
            sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
        )
    else:
        phase_status = {
            "actual_nodes": int(k_values.size),
            "required_nodes": 1,
            "radial_required_nodes": 1,
            "acoustic_required_nodes": 1,
            "phase_step": 0.0,
            "resolved": True,
        }

    eta0 = background.eta0
    source_chi = float(background.chi_of_eta(background.eta_rec))
    source_parameters: dict[str, float] = {}
    for source in (
        contract_or_params.get("param_map", {}) or {},
        contract_or_params.get("model_parameters", {}) or {},
    ):
        if not isinstance(source, Mapping):
            continue
        for name, value in source.items():
            if str(name) in source_parameters:
                continue
            try:
                source_parameters[str(name)] = _coerce_numeric_scalar(
                    value,
                    name=str(name),
                )
            except ValueError:
                continue
    physical_runtime_scalars = _physical_runtime_scalars(physical_params)

    all_power_spectrum_observables = {}
    for name, entry in perturbation_data.observables.items():
        if entry.kind != "angular_power_spectrum":
            continue
        canonical_name = canonical_cmb_spectrum_name(name)
        if canonical_name in all_power_spectrum_observables:
            raise ValueError(
                "Declared angular spectra must have unique canonical names: "
                f"{canonical_name}"
            )
        all_power_spectrum_observables[canonical_name] = entry
    physical_zero_spectra: set[str] = set()
    if requested_spectrum_names is not None:
        if not requested_spectrum_names:
            raise ValueError(
                "Requested declared spectra must contain at least one name"
            )
        unavailable_spectra = sorted(
            requested_spectrum_names - set(all_power_spectrum_observables)
        )
        if (
            unavailable_spectra == ["BB"]
            and {"TT", "TE", "EE", "BB", "PP"} <= requested_spectrum_names
        ):
            # Exact lensing remapping accepts an absent unlensed BB input as
            # the physical zero-parity baseline and generates lensed BB from
            # the declared E-mode and lensing-potential spectra.
            physical_zero_spectra.add("BB")
            unavailable_spectra = []
        if unavailable_spectra:
            raise ValueError(
                "Declared CMB graph does not provide requested spectra: "
                + ", ".join(unavailable_spectra)
            )
    if requested_spectrum_names is None:
        power_spectrum_observables = all_power_spectrum_observables
        required_transfer_components = {
            str(observable.primary)
            for observable in power_spectrum_observables.values()
        }
        required_transfer_components.update(
            str(observable.secondary)
            for observable in power_spectrum_observables.values()
        )
    else:
        power_spectrum_observables = {
            name: entry
            for name, entry in all_power_spectrum_observables.items()
            if name in requested_spectrum_names
        }
        required_transfer_components = {
            str(observable.primary)
            for observable in power_spectrum_observables.values()
        }
        required_transfer_components.update(
            str(observable.secondary)
            for observable in power_spectrum_observables.values()
        )
    spectrum_availability = {
        name: (
            "computed" if name in power_spectrum_observables else "unrequested"
        )
        for name in all_power_spectrum_observables
    }
    for name in physical_zero_spectra:
        spectrum_availability[name] = "physical_zero"
    transfer_component_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "transfer_component"
        and (
            requested_spectrum_names is None
            or name in required_transfer_components
        )
    }
    required_source_names = {
        str(source_name)
        for component_entry in transfer_component_observables.values()
        for source_name in component_entry.source_terms.values()
    }
    if diagnostic_source_audit:
        # Fixed-point diagnostics need every scalar source that participates in
        # the declared closure, even when the requested spectrum only consumes
        # one transfer component.  Production requests retain demand-driven
        # source evaluation and its lower cost.
        required_source_names.update(
            {
                "temperature_monopole",
                "temperature_quadrupole",
                "temperature_quadrupole_derivative",
                "temperature_doppler",
                "temperature_isw",
                "polarization_source",
            }
        )
    declared_source_history_roles = tuple(
        f"{component_name}:{role_name}"
        for component_name, component_entry in (
            transfer_component_observables.items()
        )
        for role_name in component_entry.source_terms
    )
    source_history_max_abs = {
        role_name: 0.0 for role_name in declared_source_history_roles
    }
    source_history_max_abs_by_k: dict[str, dict[str, float]] = {}
    state_history_max_abs_by_k: dict[str, dict[str, float]] = {}
    state_history_polarization_ratio_by_k: dict[str, dict[str, float]] = {}
    source_context_max_abs_by_k: dict[str, dict[str, float]] = {}
    source_context_pre_resolution_by_k: dict[str, dict[str, float]] = {}
    source_history_residual_samples_by_k: dict[str, dict[str, Any]] = {}
    hierarchy_equation_residuals_by_k: dict[str, dict[str, Any]] = {}
    initial_state_diagnostics_by_k: dict[str, dict[str, Any]] = {}
    metric_history_gradient_residual_by_k: dict[str, dict[str, float]] = {}
    source_history_mode_count = 0
    source_history_cache_hits = 0
    source_history_cache_misses = 0
    source_eta_signature = hashlib.sha256(
        numpy.asarray(source_grids["eta"], dtype=numpy.float64).tobytes()
    ).hexdigest()
    source_history_cache_prefix = (
        cache_key.contract_static,
        cache_key.model_static,
        cache_key.execution_solver,
        source_eta_signature,
        tuple(sorted(str(name) for name in required_source_names)),
    )

    def _source_history_cache_key(mode_k_value: float) -> tuple[Any, ...]:
        """Return an exact cache key for one parameter/grid source history."""

        return source_history_cache_prefix + (float(mode_k_value),)

    def _record_source_history_residual_samples(
        mode_k_value: float,
        context: Mapping[str, Any],
    ) -> None:
        """Capture compact raw source terms for independent auditing.

        The runtime validator owns enforcement, while scientific diagnostics
        recompute closure residuals from these raw terms independently.
        Deterministic eta anchors keep the evidence compact for production
        mode grids and preserve visibility-era structure.
        """

        eta_values = numpy.asarray(active_grids["eta"], dtype=float)
        if eta_values.ndim != 1 or eta_values.size == 0:
            return
        visibility = numpy.asarray(
            context.get("visibility", numpy.zeros_like(eta_values)),
            dtype=float,
        )
        anchor_indices = {
            0,
            max(int(eta_values.size // 4), 0),
            max(int(eta_values.size // 2), 0),
            max(int(3 * eta_values.size // 4), 0),
            int(eta_values.size - 1),
        }
        if visibility.shape == eta_values.shape and numpy.any(
            numpy.isfinite(visibility)
        ):
            anchor_indices.add(
                int(
                    numpy.nanargmax(
                        numpy.nan_to_num(visibility, nan=-numpy.inf)
                    )
                )
            )
        field_names = (
            "eta",
            "Phi",
            "Psi",
            "Phi_tau",
            "Psi_tau",
            "Phi_history_tau",
            "Hconf",
            "acoustic_k",
            "acoustic_k_sq",
            "einstein_gravity_strength",
            "metric_shear_correction",
            "total_density_source",
            "matter_density_source",
            "radiation_density_source",
            "total_momentum_source",
            "matter_momentum_source",
            "radiation_momentum_source",
            "total_shear_source",
            "visibility",
            "tau",
            "delta_b",
            "delta_c",
            "delta_nu",
            "theta_gamma0",
            "theta_gamma1",
            "theta_gamma2",
            "theta_b",
            "theta_c",
            "theta_nu",
            "sigma_nu",
            "observable_theta_gamma0",
            "observable_theta_b",
            "polarization_moment",
            "temperature_monopole",
            "temperature_quadrupole",
            "temperature_quadrupole_derivative",
            "temperature_doppler",
            "temperature_isw",
            "polarization_source",
            "visibility_polarization_moment",
        )
        samples = []
        for index in sorted(anchor_indices):
            sample: dict[str, float] = {}
            for name in field_names:
                if name == "eta":
                    value = eta_values[index]
                elif name not in context:
                    continue
                else:
                    values = numpy.asarray(context[name], dtype=float)
                    if values.ndim == 0:
                        value = values
                    elif values.shape == eta_values.shape:
                        value = values[index]
                    else:
                        continue
                scalar = float(value)
                if numpy.isfinite(scalar):
                    sample[name] = scalar
            samples.append(sample)
        if generated_scalar_hierarchy:
            required_history_fields = {
                "eta",
                "Phi",
                "Psi",
                "Phi_tau",
                "Psi_tau",
                "Phi_history_tau",
                "Hconf",
                "acoustic_k",
                "acoustic_k_sq",
                "einstein_gravity_strength",
                "metric_shear_correction",
                "total_density_source",
                "total_momentum_source",
                "total_shear_source",
                "visibility",
                "tau",
                "observable_theta_gamma0",
                "observable_theta_b",
                "polarization_moment",
                "temperature_monopole",
                "temperature_quadrupole",
                "temperature_quadrupole_derivative",
                "temperature_doppler",
                "temperature_isw",
                "polarization_source",
            }
            missing_by_sample = {
                int(index): tuple(
                    sorted(required_history_fields - set(sample))
                )
                for index, sample in enumerate(samples)
                if required_history_fields - set(sample)
            }
            if missing_by_sample:
                raise ConstraintViolationError(
                    "Generated scalar source-history audit omitted declared "
                    "terms",
                    context={
                        "k": float(mode_k_value),
                        "missing_by_sample": missing_by_sample,
                    },
                )
        source_history_residual_samples_by_k[f"{float(mode_k_value):.12g}"] = {
            "k": float(mode_k_value),
            "sample_count": int(len(samples)),
            "samples": tuple(samples),
        }

    def _record_hierarchy_equation_residuals(
        mode_k_value: float,
        raw_histories: Mapping[str, numpy.ndarray],
    ) -> None:
        """Compare raw history derivatives with the compiled hierarchy RHS.

        The comparison is intentionally made before any constraint-history
        reconstruction.  A finite-difference derivative of the emitted
        state history is independent evidence about the actual integration
        result; it cannot be made to pass by replacing a history with a
        post-processed Einstein constraint solution.
        """

        if not generated_scalar_hierarchy:
            return
        eta_values = numpy.asarray(active_grids["eta"], dtype=float)
        if eta_values.size < 3:
            return
        state_slots_by_index = {
            int(slot.index): slot
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
        }
        if not state_slots_by_index:
            return
        histories = {
            str(name): numpy.asarray(values, dtype=float)
            for name, values in raw_histories.items()
        }
        if any(
            values.shape != eta_values.shape
            or not numpy.all(numpy.isfinite(values))
            for values in histories.values()
        ):
            raise NonFiniteEvolutionError(
                "Generated scalar hierarchy audit histories must be finite "
                "and aligned with the source eta grid",
                context={
                    "k": float(mode_k_value),
                    "eta_size": int(eta_values.size),
                    "history_shapes": {
                        str(name): tuple(values.shape)
                        for name, values in histories.items()
                    },
                },
            )
        derivatives = {
            name: _nonuniform_gradient(values, eta_values)
            for name, values in histories.items()
        }
        anchor_indices = tuple(
            index
            for index in sorted(
                {
                    1,
                    int(eta_values.size // 4),
                    int(eta_values.size // 2),
                    int(3 * eta_values.size // 4),
                    int(eta_values.size - 2),
                }
            )
            if 1 <= index < eta_values.size - 1
        )
        equation_metrics: dict[str, dict[str, float]] = {}
        anchor_residuals: dict[str, dict[str, dict[str, float]]] = {}
        collision_active_anchor_indices: list[int] = []
        audited_anchor_indices: list[int] = []
        for index in anchor_indices:
            state_size = max(state_slots_by_index) + 1
            state_vector = numpy.zeros(state_size, dtype=float)
            for state_index, slot in state_slots_by_index.items():
                if slot.variable not in histories:
                    continue
                state_vector[state_index] = histories[slot.variable][index]
            collision_active = _tight_coupling_is_active(
                active=False,
                collision_rate=float(active_grids["collision_rate"][index]),
                k_value=float(mode_k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
                exit_ratio=float(numerics.tight_coupling_exit_ratio),
            )
            if collision_active:
                # The split collision integrator advances this interval with
                # the declaration's exact/implicit collision map after the
                # explicit hierarchy RHS.  Comparing the finite-difference
                # history with the unsplit RHS at a tight-coupling anchor
                # would therefore report the omitted operator as an equation
                # error.  Record the anchor explicitly and audit the free
                # streaming intervals below instead.
                collision_active_anchor_indices.append(int(index))
                continue
            audited_anchor_indices.append(int(index))
            rhs = _mode_rhs(
                state_vector,
                step_index=index,
                blend=0.0,
                k_value=float(mode_k_value),
                tight_coupling_active=collision_active,
                include_split_collision_outputs=False,
            )
            rhs_context = _build_scalar_state_context(
                state_vector,
                k_value=float(mode_k_value),
                eta_value=float(eta_values[index]),
                background_scalars=_scalar_background_context(index, 0.0)[1],
                cache_token=(int(index), 0.0),
            )
            for state_index, slot in state_slots_by_index.items():
                if slot.variable not in derivatives:
                    continue
                expected = float(derivatives[slot.variable][index])
                actual = float(rhs[state_index])
                absolute = abs(expected - actual)
                characteristic_rate = max(
                    abs(float(active_grids["Hconf"][index])),
                    abs(float(mode_k_value)),
                    1.0e-8,
                )
                state_scale = (
                    max(
                        abs(float(histories[slot.variable][index])),
                        1.0e-6,
                    )
                    * characteristic_rate
                )
                scale = max(abs(expected), abs(actual), state_scale)
                metric = equation_metrics.setdefault(
                    str(slot.variable),
                    {
                        "maximum_absolute": 0.0,
                        "maximum_normalized": 0.0,
                        "maximum_expected": 0.0,
                        "maximum_actual": 0.0,
                    },
                )
                metric["maximum_absolute"] = max(
                    float(metric["maximum_absolute"]), absolute
                )
                metric["maximum_normalized"] = max(
                    float(metric["maximum_normalized"]), absolute / scale
                )
                metric["maximum_expected"] = max(
                    float(metric["maximum_expected"]), abs(expected)
                )
                metric["maximum_actual"] = max(
                    float(metric["maximum_actual"]), abs(actual)
                )
                anchor_residuals.setdefault(str(slot.variable), {})[
                    str(index)
                ] = {
                    "eta": float(eta_values[index]),
                    "expected": expected,
                    "actual": actual,
                    "absolute": absolute,
                    "normalized": absolute / scale,
                }
                if slot.variable == "Phi":
                    context_anchor = {
                        "eta": float(eta_values[index]),
                        "Phi": float(histories["Phi"][index]),
                        "Phi_tau": float(
                            rhs_context.get("Phi_tau", numpy.nan)
                        ),
                        "Psi": float(rhs_context.get("Psi", numpy.nan)),
                        "Hconf": float(rhs_context.get("Hconf", numpy.nan)),
                        "total_momentum_source": float(
                            rhs_context.get("total_momentum_source", numpy.nan)
                        ),
                        "rhs_phi": float(rhs[state_index]),
                    }
                    anchor_residuals.setdefault("__context__", {})[
                        str(index)
                    ] = context_anchor
        hierarchy_equation_residuals_by_k[f"{float(mode_k_value):.12g}"] = {
            "k": float(mode_k_value),
            "sample_count": int(len(audited_anchor_indices)),
            "candidate_sample_count": int(len(anchor_indices)),
            "collision_active_anchor_indices": tuple(
                collision_active_anchor_indices
            ),
            "audited_anchor_indices": tuple(audited_anchor_indices),
            "equations": equation_metrics,
            "anchors": anchor_residuals,
        }

    def _validate_metric_history_derivatives(
        mode_k_value: float,
        context: Mapping[str, Any],
    ) -> None:
        """Require explicit, finite, and aligned generated metric histories.

        ``Phi_tau`` is the compiled Einstein-system derivative used while
        evolving the hierarchy.  ``Psi_tau`` and ``Phi_history_tau`` are
        runtime-bound history gradients used by source terms.  Keeping the
        three checks at this boundary prevents a missing derivative from
        being replaced by a zero or by an unrelated stage value.
        """

        if not generated_scalar_hierarchy:
            return
        eta_values = numpy.asarray(active_grids["eta"], dtype=float)
        residuals: dict[str, float] = {}
        phi_tau = numpy.asarray(context.get("Phi_tau", ()), dtype=float)
        phi_history = numpy.asarray(context.get("Phi", ()), dtype=float)
        if (
            phi_tau.shape != eta_values.shape
            or phi_history.shape != eta_values.shape
            or not numpy.all(numpy.isfinite(phi_tau))
            or not numpy.all(numpy.isfinite(phi_history))
        ):
            raise NonFiniteEvolutionError(
                "Generated scalar Phi_tau history must be finite and "
                "aligned with the source eta grid",
                context={
                    "k": float(mode_k_value),
                    "derivative": "Phi_tau",
                },
            )
        expected_phi_tau = None
        if {
            "metric_momentum_source_drive",
            "Hconf",
            "Psi",
        }.issubset(context):
            expected_phi_tau = numpy.asarray(
                context["metric_momentum_source_drive"], dtype=float
            ) - numpy.asarray(context["Hconf"], dtype=float) * numpy.asarray(
                context["Psi"], dtype=float
            )
            if expected_phi_tau.shape != eta_values.shape or not numpy.all(
                numpy.isfinite(expected_phi_tau)
            ):
                raise NonFiniteEvolutionError(
                    "Generated scalar Phi_tau dependencies must be finite "
                    "and aligned with the source eta grid",
                    context={
                        "k": float(mode_k_value),
                        "derivative": "Phi_tau",
                    },
                )
            scale = numpy.maximum(
                numpy.maximum(numpy.abs(expected_phi_tau), numpy.abs(phi_tau)),
                1.0e-30,
            )
            phi_tau_residual = float(
                numpy.max(
                    numpy.abs(phi_tau - expected_phi_tau) / scale,
                    initial=0.0,
                )
            )
            if phi_tau_residual > 1.0e-8:
                raise ConstraintViolationError(
                    "Generated scalar Phi_tau does not match its declared "
                    "Einstein-system expression",
                    context={
                        "k": float(mode_k_value),
                        "derivative": "Phi_tau",
                        "maximum_normalized": phi_tau_residual,
                    },
                )
            residuals["Phi_tau"] = phi_tau_residual
        for derivative_name, history_name in (
            ("Psi_tau", "Psi"),
            ("Phi_history_tau", "Phi"),
        ):
            if derivative_name not in context or history_name not in context:
                raise ConstraintViolationError(
                    "Generated scalar source graph omitted the explicit "
                    f"{derivative_name} history derivative",
                    context={
                        "k": float(mode_k_value),
                        "derivative": derivative_name,
                        "history": history_name,
                    },
                )
            derivative = numpy.asarray(context[derivative_name], dtype=float)
            history = numpy.asarray(context[history_name], dtype=float)
            if (
                derivative.shape != eta_values.shape
                or history.shape != eta_values.shape
                or not numpy.all(numpy.isfinite(derivative))
                or not numpy.all(numpy.isfinite(history))
            ):
                raise NonFiniteEvolutionError(
                    "Generated scalar metric history derivatives must be "
                    "finite and aligned with the source eta grid",
                    context={
                        "k": float(mode_k_value),
                        "derivative": derivative_name,
                    },
                )
            expected = _nonuniform_gradient(history, eta_values)
            scale = numpy.maximum(
                numpy.maximum(numpy.abs(expected), numpy.abs(derivative)),
                1.0e-30,
            )
            residuals[derivative_name] = float(
                numpy.max(
                    numpy.abs(derivative - expected) / scale,
                    initial=0.0,
                )
            )
        metric_history_gradient_residual_by_k[
            f"{float(mode_k_value):.12g}"
        ] = residuals

    def _record_source_history_diagnostics(
        source_arrays: Mapping[str, numpy.ndarray],
        *,
        mode_k_value: float | None = None,
    ) -> None:
        """Record finite declared source histories without copying them."""

        nonlocal source_history_mode_count
        mode_maxima: dict[str, float] = {}
        for (
            component_name,
            component_entry,
        ) in transfer_component_observables.items():
            histories = _bind_declared_source_histories(
                component_name=str(component_name),
                component_entry=component_entry,
                source_arrays=source_arrays,
            )
            for role_name, history in histories.items():
                if not numpy.all(numpy.isfinite(history)):
                    raise ValueError(
                        f"Declared source history '{component_name}:"
                        f"{role_name}' is non-finite"
                    )
                role_key = f"{component_name}:{role_name}"
                role_maximum = float(
                    numpy.max(numpy.abs(history), initial=0.0)
                )
                source_history_max_abs[role_key] = max(
                    source_history_max_abs[role_key],
                    role_maximum,
                )
                mode_maxima[role_key] = role_maximum
        if mode_k_value is not None:
            source_history_max_abs_by_k[f"{float(mode_k_value):.12g}"] = (
                mode_maxima
            )
        source_history_mode_count += 1

    declared_projection_sectors = {
        str(getattr(entry, "sector", "") or "scalar")
        for entry in transfer_component_observables.values()
    }
    streaming_projection_sectors = (
        ("scalar",) if declared_projection_sectors <= {"scalar"} else None
    )
    momentum_runtimes = _resolve_declared_momentum_grid_runtimes(
        perturbation_data,
        model_parameters=source_parameters,
        physical_params=physical_params,
    )
    runtime_envelope = _enforce_runtime_envelope(
        contract_or_params,
        ell_count=int(ell_arr.size),
        k_count=int(k_values.size),
        eta_count=int(source_grids["eta"].size),
        state_slot_count=int(len(runtime_spec.state_slots)),
        transfer_component_count=int(len(transfer_component_observables)),
        momentum_point_count=int(
            sum(runtime.points.size for runtime in momentum_runtimes)
        ),
        evolution_multiplier=(3 if adaptive_controls.evolution_enabled else 1),
    )
    evolution_chunk_size = _resolve_evolution_chunk_size(
        k_count=int(k_values.size),
        eta_count=int(source_grids["eta"].size),
        state_slot_count=int(len(runtime_spec.state_slots)),
    )
    runtime_envelope["evolution_chunk_size"] = int(evolution_chunk_size)
    runtime_envelope["evolution_chunk_count"] = int(
        (int(k_values.size) + evolution_chunk_size - 1) // evolution_chunk_size
    )
    runtime_envelope["evolution_chunk_accumulation_order"] = "k_index"
    runtime_envelope["evolution_peak_state_cells"] = int(
        evolution_chunk_size
        * max(int(source_grids["eta"].size), 1)
        * max(int(len(runtime_spec.state_slots)), 1)
    )
    runtime_envelope["configured_numerical_controls"] = dict(
        numerical_envelope.numerical_controls
    )
    runtime_envelope["generated_scalar_source_closure"] = dict(
        (manifest_summary.get("generated_scalar_source_closure", {}) or {})
    )
    if generated_scalar_hierarchy:
        runtime_envelope["source_history_derivative_provenance"] = {
            "Phi_tau": {
                "kind": "algebraic_einstein_derivative",
                "variable": "Phi",
                "wrt": "tau",
                "order": 1,
                "independent_from_history_gradient": True,
            },
            "Psi_tau": {
                "kind": "evolved_history_gradient",
                "variable": "Psi",
                "wrt": "tau",
                "order": 1,
                "independent_from_algebraic_closure": True,
            },
            "Phi_history_tau": {
                "kind": "evolved_history_gradient",
                "variable": "Phi",
                "wrt": "tau",
                "order": 1,
                "independent_from_algebraic_closure": True,
            },
        }
    else:
        runtime_envelope["source_history_derivative_provenance"] = {
            "status": "not_applicable",
            "reason": "explicit_model_graph",
        }
    runtime_envelope["effective_numerical_controls"] = {
        **dict(numerical_envelope.numerical_controls),
        "k_sample_count": int(k_values.size),
        "eta_sample_count": int(source_grids["eta"].size),
        "ell_count": int(ell_arr.size),
    }
    runtime_envelope["resolution_reduction"] = False
    runtime_envelope["numerical_envelope"] = numerical_envelope.to_dict()
    runtime_envelope["accuracy_tier"] = numerical_envelope.accuracy_tier
    runtime_envelope["lensing_sampling_factor"] = float(
        numerical_envelope.numerical_controls["lensing_sampling_factor"]
    )
    runtime_envelope["spectrum_availability"] = FrozenMapping(
        dict(sorted(spectrum_availability.items()))
    )
    runtime_asset_cache_after = cache.cmb_cache_stats()["runtime_assets"]
    structural_cache_hit = bool(
        runtime_asset_cache_after["hits"] > runtime_asset_cache_before["hits"]
    )
    runtime_envelope["static_graph_preparations"] = int(
        not structural_cache_hit
    )
    runtime_envelope["contract_static_preparations"] = int(
        not structural_cache_hit
    )
    runtime_envelope["model_static_preparations"] = 1
    runtime_envelope["request_specific_preparations"] = 1
    runtime_envelope["dynamic_mode_count"] = int(k_values.size)
    runtime_envelope["declared_k_sample_count"] = int(numerics.k_sample_count)
    runtime_envelope["k_grid_actual_count"] = int(k_values.size)
    runtime_envelope["phase_aware_k_enabled"] = bool(phase_aware_k_enabled)
    runtime_envelope["phase_required_nodes"] = int(
        phase_requirements["required_nodes"]
    )
    runtime_envelope["phase_radial_required_nodes"] = int(
        phase_requirements["radial_required_nodes"]
    )
    runtime_envelope["phase_acoustic_required_nodes"] = int(
        phase_requirements["acoustic_required_nodes"]
    )
    runtime_envelope["phase_resolution_limited"] = bool(
        phase_aware_k_enabled and not bool(phase_status["resolved"])
    )
    runtime_envelope["phase_resolution_status"] = (
        "resolved"
        if not phase_aware_k_enabled or bool(phase_status["resolved"])
        else "under_resolved"
    )
    runtime_envelope["phase_grid_status"] = dict(phase_status)
    runtime_envelope["k_quadrature_rule"] = (
        "simpson_uniform_log_k"
        if k_values.size < 2
        or numpy.allclose(
            numpy.diff(numpy.log(numpy.asarray(k_values, dtype=float))),
            numpy.diff(numpy.log(numpy.asarray(k_values, dtype=float)))[0],
            rtol=1.0e-10,
            atol=1.0e-14,
        )
        else "positive_trapezoid_irregular_phase_grid"
    )
    runtime_envelope["batch_count"] = 0
    runtime_envelope["batch_mode_count"] = 0
    runtime_envelope["batched_rk_stage_count"] = 0
    runtime_envelope["batched_max_substeps"] = 0
    runtime_envelope["batched_schedule_correction_mode_count"] = 0
    graph_cache_after = cache.cmb_cache_stats()[
        "declared_graph_execution_plan"
    ]
    background_cache_after = cache.cmb_cache_stats()["background"]
    runtime_envelope["graph_plan_cache_hit"] = bool(
        structural_cache_hit
        or graph_cache_after["hits"] > graph_cache_before["hits"]
    )
    runtime_envelope["runtime_asset_cache_hit"] = structural_cache_hit
    runtime_envelope["background_cache_hit"] = bool(
        background_cache_after["misses"] == background_cache_before["misses"]
    )
    runtime_envelope["model_static_preparations"] = int(
        not runtime_envelope["background_cache_hit"]
    )
    previous_request_identity = cache.latest_cmb_request_identity()
    same_request_shape = bool(
        previous_request_identity is not None
        and previous_request_identity.contract_static
        == cache_key.contract_static
        and previous_request_identity.request_specific
        == cache_key.request_specific
    )
    # A structural graph hit alone does not make a request a warm parameter
    # rebound: a new background or graph changes the numerical work. Classify
    # warm only when the compiled structure, parameter-dependent background,
    # and request shape are reusable; otherwise the full-spectrum budget owns
    # the request.
    performance_timer.mark_cache_state(
        "warm"
        if (
            runtime_envelope["graph_plan_cache_hit"]
            and runtime_envelope["background_cache_hit"]
            and same_request_shape
        )
        else "cold"
    )
    performance_timer.set_work_units(
        {
            name: int(value)
            for name, value in runtime_envelope.items()
            if str(name).endswith("work_units")
        }
    )
    runtime_envelope["adaptive_transfer_enabled"] = bool(
        adaptive_controls.transfer_enabled
    )
    runtime_envelope["adaptive_source_enabled"] = bool(
        adaptive_controls.source_enabled
    )
    runtime_envelope["adaptive_projection_enabled"] = bool(
        adaptive_controls.projection_enabled
    )
    runtime_envelope["adaptive_evolution_enabled"] = bool(
        adaptive_controls.evolution_enabled
    )
    runtime_envelope["adaptive_phase_points_per_cycle"] = float(
        adaptive_controls.phase_points_per_cycle
    )
    runtime_envelope["los_phase_quadrature_enabled"] = bool(
        los_quadrature_controls.enabled
    )
    runtime_envelope["los_phase_quadrature_applied"] = bool(
        los_phase_quadrature_applied
    )
    runtime_envelope["los_phase_points_per_cycle"] = float(
        los_quadrature_controls.phase_points_per_cycle
    )
    runtime_envelope["los_phase_minimum_nodes"] = int(
        los_quadrature_controls.minimum_nodes
    )
    runtime_envelope["los_phase_maximum_nodes"] = int(
        los_quadrature_controls.maximum_nodes
    )
    runtime_envelope["los_phase_configured_maximum_nodes"] = int(
        los_quadrature_controls.configured_maximum_nodes
    )
    runtime_envelope["los_phase_eta_sample_count"] = int(
        source_grids["eta"].size
    )
    runtime_envelope["generated_final_evolution_floor"] = int(
        generated_final_evolution_floor or 0
    )
    runtime_envelope["los_phase_eta_min_step"] = float(
        numpy.min(numpy.diff(source_grids["eta"]))
    )
    runtime_envelope["los_phase_eta_max_step"] = float(
        numpy.max(numpy.diff(source_grids["eta"]))
    )
    runtime_envelope["adaptive_transfer_refinement_levels"] = 0
    runtime_envelope["adaptive_source_refinement_levels"] = 0
    runtime_envelope["adaptive_projection_refinement_levels"] = 0
    runtime_envelope["adaptive_evolution_refinement_levels"] = 0
    runtime_envelope["adaptive_transfer_relative_error"] = 0.0
    runtime_envelope["adaptive_source_relative_error"] = 0.0
    runtime_envelope["adaptive_projection_relative_error"] = 0.0
    runtime_envelope["adaptive_evolution_relative_error"] = 0.0
    runtime_envelope["adaptive_evolution_absolute_error"] = 0.0
    runtime_envelope["declared_source_history_roles"] = (
        declared_source_history_roles
    )
    runtime_envelope["declared_source_history_sample_count"] = int(
        source_grids["eta"].size
    )
    runtime_envelope["declared_source_history_mode_count"] = 0
    runtime_envelope["declared_source_history_finite"] = True
    runtime_envelope["generated_scalar_hierarchy"] = bool(
        generated_scalar_hierarchy
    )
    transfer_components = {
        name: numpy.zeros((ell_arr.size, k_values.size), dtype=float)
        for name in transfer_component_observables
    }
    declared_accuracy_controls = _resolve_declared_accuracy_controls(
        contract_or_params
    )
    scalar_constraint_diagnostics: dict[str, dict[str, Any]] = {}
    scalar_constraint_projection_count = 0
    scalar_constraint_diagnostic_projection_count = 0
    scalar_constraint_projection_max_relative_correction = 0.0
    adaptive_k_controls = declared_accuracy_controls.get(
        "adaptive_k_quadrature"
    )
    reconstruction_control = declared_accuracy_controls.get(
        "source_history_reconstruction"
    )
    source_history_reconstruction_enabled = (
        bool(reconstruction_control)
        if reconstruction_control is not None
        else not generated_scalar_hierarchy
    )
    runtime_envelope["source_history_reconstruction_enabled"] = bool(
        source_history_reconstruction_enabled
    )
    runtime_envelope["source_history_reconstruction_diagnostic_only"] = bool(
        generated_scalar_hierarchy
        and not source_history_reconstruction_enabled
    )
    if isinstance(
        declared_accuracy_controls.get("adaptive_transfer"), Mapping
    ):
        adaptive_k_controls = None
    adaptive_k_enabled = isinstance(adaptive_k_controls, Mapping)
    adaptive_k_min_ell = 0
    adaptive_k_node_count = 0
    adaptive_k_window_fraction = 0.0
    adaptive_k_ell_stride = 1
    adaptive_k_eta_stride = 1
    adaptive_k_mode = "transfer"
    direct_source_quadrature = False
    if adaptive_k_enabled:
        adaptive_k_min_ell = int(
            _coerce_numeric_scalar(
                adaptive_k_controls.get("ell_min", 100),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_k_quadrature.ell_min"
                ),
            )
        )
        adaptive_k_node_count = int(
            _coerce_numeric_scalar(
                adaptive_k_controls.get("node_count", 24),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_k_quadrature.node_count"
                ),
            )
        )
        adaptive_k_window_fraction = float(
            _coerce_numeric_scalar(
                adaptive_k_controls.get("window_fraction", 0.2),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_k_quadrature.window_fraction"
                ),
            )
        )
        adaptive_k_ell_stride = int(
            _coerce_numeric_scalar(
                adaptive_k_controls.get("ell_stride", 4),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_k_quadrature.ell_stride"
                ),
            )
        )
        adaptive_k_eta_stride = int(
            _coerce_numeric_scalar(
                adaptive_k_controls.get("eta_stride", 4),
                name=(
                    "cmb.perturbations.accuracy_controls."
                    "adaptive_k_quadrature.eta_stride"
                ),
            )
        )
        adaptive_k_mode = (
            str(adaptive_k_controls.get("mode", "transfer")).strip().lower()
        )
        direct_source_quadrature = bool(
            adaptive_k_controls.get("direct_source_quadrature", False)
        )
        if adaptive_k_min_ell < 2:
            raise ValueError(
                "adaptive_k_quadrature.ell_min must be at least 2"
            )
        if adaptive_k_node_count < 4:
            raise ValueError(
                "adaptive_k_quadrature.node_count must be at least 4"
            )
        if adaptive_k_ell_stride < 1:
            raise ValueError(
                "adaptive_k_quadrature.ell_stride must be positive"
            )
        if adaptive_k_eta_stride < 1:
            raise ValueError(
                "adaptive_k_quadrature.eta_stride must be positive"
            )
        if adaptive_k_mode not in {"source", "transfer"}:
            raise ValueError(
                "adaptive_k_quadrature.mode must be 'source' or 'transfer'"
            )
        if not 0.0 < adaptive_k_window_fraction <= 1.0:
            raise ValueError(
                "adaptive_k_quadrature.window_fraction must be in (0, 1]"
            )
    use_streaming_projection = (
        int(numpy.max(ell_arr)) >= 500
        and not adaptive_controls.transfer_enabled
        and not adaptive_controls.source_enabled
        and not adaptive_controls.projection_enabled
        and not adaptive_k_enabled
    )
    adaptive_source_history_rows: dict[
        tuple[str, str], list[numpy.ndarray]
    ] = {}
    eta_integration_weights = _simpson_weights(source_grids["eta"])
    split_collision_runtimes = _compile_split_collision_operator_runtimes(
        perturbation_data=perturbation_data,
        runtime_spec=runtime_spec,
    )
    equation_program_specs = tuple(
        (
            int(slot_plan.state_index),
            str(slot_plan.wrt),
            (
                None
                if slot_plan.compiled_rhs is None
                else str(slot_plan.compiled_rhs.expression)
            ),
            (
                None
                if slot_plan.promote_from_index is None
                else int(slot_plan.promote_from_index)
            ),
        )
        for slot_plan in execution_plan.equation_slot_plans
    )
    equation_program = _compile_equation_program(equation_program_specs)
    scalar_base_context_cache: dict[
        tuple[float, tuple[tuple[str, float], ...]],
        dict[str, Any],
    ] = {}
    scalar_background_context_cache: dict[
        tuple[int, float, float], tuple[float, dict[str, float]]
    ] = {}
    momentum_grid_context_cache: dict[float, dict[str, Any]] = {}

    generated_final_phase_step = float(numerics.evolution_phase_step)
    if (
        generated_scalar_hierarchy
        and declared_accuracy_controls.get("accuracy_tier") == "final"
    ):
        # The generated photon and neutrino hierarchies are phase-sensitive
        # before last scattering.  The historical default of two radians per
        # RK stage is stable but under-resolves the acoustic transfer function
        # by the time it reaches the visibility surface.  Keep late-time ISW
        # evolution on the declared step and use a quarter-cycle stage only
        # through the recombination neighbourhood where the high-ell signal
        # is formed.
        generated_final_phase_step = min(generated_final_phase_step, 0.25)

    def _phase_step_for_interval(
        *,
        step_index: int,
        blend: float = 0.5,
    ) -> float:
        """Return the phase step required by one generated-mode interval."""

        if generated_final_phase_step >= float(numerics.evolution_phase_step):
            return float(numerics.evolution_phase_step)
        eta_value = _blend_history(
            active_grids["eta"],
            step_index=step_index,
            blend=blend,
        )
        recombination_window = max(
            float(background.eta_rec)
            + 2.0 * float(background.sound_horizon_mpc),
            float(background.eta_rec) + 64.0,
        )
        if eta_value <= recombination_window:
            return generated_final_phase_step
        return float(numerics.evolution_phase_step)

    def _blend_history(
        history: numpy.ndarray,
        *,
        step_index: int,
        blend: float,
    ) -> float:
        """Return one linearly interpolated history value."""

        next_index = min(step_index + 1, active_grids["eta"].size - 1)
        weight_next = float(blend)
        weight_current = 1.0 - weight_next
        return float(
            weight_current * history[step_index]
            + weight_next * history[next_index]
        )

    def _scalar_background_context(
        step_index: int,
        blend: float,
        *,
        k_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Return one interpolated scalar background context."""

        context_key = (
            int(step_index),
            float(blend),
            float(active_k_value if k_value is None else k_value),
        )
        cached_context = scalar_background_context_cache.get(context_key)
        if cached_context is not None:
            return cached_context

        eta_value = _blend_history(
            active_grids["eta"],
            step_index=step_index,
            blend=blend,
        )
        scalar_context = {
            "a": _blend_history(
                active_grids["a"],
                step_index=step_index,
                blend=blend,
            ),
            "z": _blend_history(
                active_grids["z"],
                step_index=step_index,
                blend=blend,
            ),
            "eta": float(eta_value),
            "H": _blend_history(
                active_grids["H"],
                step_index=step_index,
                blend=blend,
            ),
            "Hconf": _blend_history(
                active_grids["Hconf"],
                step_index=step_index,
                blend=blend,
            ),
            "Hconf_tau": _blend_history(
                active_grids["Hconf_tau"],
                step_index=step_index,
                blend=blend,
            ),
            "tau": _blend_history(
                active_grids["tau"],
                step_index=step_index,
                blend=blend,
            ),
            "tau_dot": _blend_history(
                active_grids["tau_dot"],
                step_index=step_index,
                blend=blend,
            ),
            "visibility": _blend_history(
                active_grids["visibility"],
                step_index=step_index,
                blend=blend,
            ),
            "chi": _blend_history(
                active_grids["chi"],
                step_index=step_index,
                blend=blend,
            ),
            "angular_diameter_distance": _blend_history(
                active_grids["angular_diameter_distance"],
                step_index=step_index,
                blend=blend,
            ),
            "sound_speed": _blend_history(
                active_grids["sound_speed"],
                step_index=step_index,
                blend=blend,
            ),
            "baryon_sound_speed_sq": _blend_history(
                active_grids["baryon_sound_speed_sq"],
                step_index=step_index,
                blend=blend,
            ),
            "sound_speed_sq": _blend_history(
                active_grids["sound_speed_sq"],
                step_index=step_index,
                blend=blend,
            ),
            "baryon_loading": _blend_history(
                active_grids["baryon_loading"],
                step_index=step_index,
                blend=blend,
            ),
            "free_streaming": _blend_history(
                active_grids["free_streaming"],
                step_index=step_index,
                blend=blend,
            ),
            "sound_horizon": float(background.sound_horizon_mpc),
        }
        collision_rate = _blend_history(
            active_grids["collision_rate"],
            step_index=step_index,
            blend=blend,
        )
        tight_coupling_drag = _compute_tight_coupling_drag(
            collision_rate=collision_rate,
            k_value=float(active_k_value if k_value is None else k_value),
            tight_coupling_ratio=float(numerics.tight_coupling_ratio),
        )
        scalar_context["collision_rate"] = float(collision_rate)
        scalar_context["tight_coupling_drag"] = float(tight_coupling_drag)
        for name, history in active_declared_background_histories.items():
            scalar_context[name] = _blend_history(
                history,
                step_index=step_index,
                blend=blend,
            )
        cached_context = (float(eta_value), scalar_context)
        scalar_background_context_cache[context_key] = cached_context
        return cached_context

    def _resolve_coordinate_rate(
        *,
        wrt_name: str,
        scalar_context: Mapping[str, float],
        step_index: int,
        blend: float,
        k_value: float,
    ) -> float:
        """Return ``dwrt/deta`` for one declared runtime coordinate."""

        if wrt_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
            return 1.0
        if wrt_name == "a":
            rate = float(scalar_context["a"]) * float(scalar_context["Hconf"])
        elif wrt_name == "z":
            rate = -(1.0 + float(scalar_context["z"])) * float(
                scalar_context["Hconf"]
            )
        else:
            rate = None
        for legacy_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
            derivative_symbol = f"__d1_{wrt_name}_{legacy_name}"
            if derivative_symbol not in scalar_context:
                continue
            rate = float(scalar_context[derivative_symbol])
            break
        else:
            if rate is not None:
                pass
            elif wrt_name not in active_coordinate_rate_histories:
                raise ValueError(
                    "Declared CMB coordinate transform does not support "
                    f"wrt '{wrt_name}'."
                )
            else:
                rate = _blend_history(
                    active_coordinate_rate_histories[wrt_name],
                    step_index=step_index,
                    blend=blend,
                )
        if not numpy.isfinite(rate) or abs(rate) <= 1.0e-12:
            eta_value = _blend_history(
                active_grids["eta"],
                step_index=step_index,
                blend=blend,
            )
            raise ValueError(
                "Declared CMB coordinate transform is singular for "
                f"wrt '{wrt_name}' at eta={eta_value}, k={k_value}"
            )
        return rate

    def _mode_grids_for_k(
        k_value: float,
        *,
        evolution_sample_count_override: int | None = None,
    ) -> tuple[
        dict[str, numpy.ndarray],
        dict[str, numpy.ndarray],
        dict[str, numpy.ndarray],
    ]:
        """Return the evolution grids used for one Fourier mode."""

        if not generated_scalar_hierarchy:
            if evolution_sample_count_override is not None:
                requested_samples = int(evolution_sample_count_override)
                eta_grid = numpy.asarray(source_grids["eta"], dtype=float)
                if eta_grid.size <= requested_samples:
                    eta_grid = _densify_eta_grid(
                        eta_grid,
                        minimum_samples=requested_samples,
                    )
                else:
                    eta_grid = _limit_eta_grid(
                        eta_grid,
                        maximum_samples=requested_samples,
                    )
                return _sample_eta_background_grids(eta_grid)
            return (
                source_grids,
                source_declared_background_histories,
                source_coordinate_rate_histories,
            )
        initial_families = tuple(
            str(name)
            for name in manifest_summary.get(
                "initial_condition_family_names",
                (),
            )
        )
        if "adiabatic_scalar" not in initial_families:
            return (
                source_grids,
                source_declared_background_histories,
                source_coordinate_rate_histories,
            )
        abs_k = abs(float(k_value))
        if not numpy.isfinite(abs_k) or abs_k <= 1.0e-12:
            return (
                source_grids,
                source_declared_background_histories,
                source_coordinate_rate_histories,
            )
        source_eta_start = float(source_grids["eta"][0])

        nonlocal shared_generated_mode_grids
        if (
            evolution_sample_count_override is None
            and shared_generated_mode_grids_enabled
            and shared_generated_mode_grids is not None
        ):
            return shared_generated_mode_grids

        def _evolution_eta_grid(
            eta_floor: float,
            *,
            sample_count_override: int | None = None,
        ) -> numpy.ndarray:
            """Build a controlled hierarchy grid without coupling it to LOS.

            The source-grid multiplier controls line-of-sight quadrature only.
            Generated hierarchy evolution uses its explicit resolution when
            declared, while the absent control retains the bounded legacy
            resolution for ordinary runtime requests.
            """

            base_grid = numpy.asarray(
                background.eta_grid[background.eta_grid >= float(eta_floor)],
                dtype=float,
            )
            requested_samples = (
                numerics.evolution_eta_sample_count
                if sample_count_override is None
                else int(sample_count_override)
            )
            if (
                sample_count_override is None
                and generated_final_evolution_floor is not None
            ):
                requested_samples = max(
                    int(requested_samples or 0),
                    int(generated_final_evolution_floor),
                )
            if requested_samples is None:
                if int(numerics.source_grid_multiplier) <= 1:
                    return base_grid
                requested_samples = max(
                    192,
                    min(256, int(numerics.eta_sample_count)),
                )
            if (
                sample_count_override is None
                and str(declared_accuracy_controls.get("accuracy_tier", ""))
                == "final"
            ):
                # Production hierarchy histories retain the declared
                # background phase grid.  A hidden stride/maximum cap here
                # can erase the acoustic phase before line-of-sight sampling.
                return base_grid
            if base_grid.size <= int(requested_samples):
                if sample_count_override is None:
                    return base_grid
                return _densify_eta_grid(
                    base_grid,
                    minimum_samples=int(requested_samples),
                )
            base_indices = numpy.flatnonzero(
                background.eta_grid >= float(eta_floor)
            )
            stride = 8
            keep = (numpy.arange(base_indices.size) % stride) == 0
            visibility = numpy.asarray(
                background.visibility_grid[base_indices],
                dtype=float,
            )
            visibility_peak = float(
                numpy.max(background.visibility_grid, initial=0.0)
            )
            keep |= visibility > max(visibility_peak * 1.0e-3, 1.0e-14)
            keep[0] = True
            keep[-1] = True
            return _limit_eta_grid(
                numpy.asarray(base_grid[keep], dtype=float),
                maximum_samples=int(requested_samples),
            )

        maximum_mode_k = (
            max(abs_k, float(numpy.max(numpy.abs(k_values), initial=abs_k)))
            if shared_generated_mode_grids_enabled
            else abs_k
        )
        eta_target = min(
            source_eta_start,
            _SCALAR_SUPERHORIZON_PREFIX_KETA / maximum_mode_k,
        )
        eta_target = max(eta_target, float(background.eta_grid[0]))
        eta_prefix = numpy.asarray(
            background.eta_grid[
                (background.eta_grid >= eta_target)
                & (background.eta_grid < source_eta_start)
            ],
            dtype=float,
        )
        requested_evolution_samples = (
            numerics.evolution_eta_sample_count
            if evolution_sample_count_override is None
            else int(evolution_sample_count_override)
        )
        if (
            evolution_sample_count_override is None
            and generated_final_evolution_floor is not None
        ):
            requested_evolution_samples = max(
                int(requested_evolution_samples or 0),
                int(generated_final_evolution_floor),
            )
        post_source_sample_count = None
        if requested_evolution_samples is not None:
            post_source_sample_count = max(
                16,
                int(requested_evolution_samples) - int(eta_prefix.size),
            )
        eta_mode_grid = numpy.unique(
            numpy.concatenate(
                (
                    numpy.asarray((eta_target,), dtype=float),
                    eta_prefix,
                    _evolution_eta_grid(
                        source_eta_start,
                        sample_count_override=post_source_sample_count,
                    ),
                )
            )
        )
        sampled_mode_grids = _sample_eta_background_grids(eta_mode_grid)
        if shared_generated_mode_grids_enabled:
            shared_generated_mode_grids = sampled_mode_grids
        return sampled_mode_grids

    def _build_scalar_base_context(
        *,
        k_value: float,
        eta_value: float,
        background_scalars: Mapping[str, float],
        cache_token: tuple[int, float] | None = None,
        resolve_graph: bool = False,
        graph_value_steps: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        """Return the cached scalar expression environment for backgrounds."""

        if cache_token is None:
            base_context_key = (
                float(k_value),
                tuple(
                    sorted(
                        (str(name), float(value))
                        for name, value in background_scalars.items()
                    )
                ),
                bool(resolve_graph),
            )
        else:
            base_context_key = (
                float(k_value),
                cache_token,
                bool(resolve_graph),
            )
        base_context = scalar_base_context_cache.get(base_context_key)
        if base_context is None:
            scale_factor = float(background_scalars["a"])
            momentum_grid_context = momentum_grid_context_cache.get(
                scale_factor
            )
            if momentum_grid_context is None:
                momentum_grid_context = _declared_momentum_grid_context(
                    perturbation_data,
                    model_parameters=source_parameters,
                    physical_params=physical_params,
                    scale_factor=scale_factor,
                )
                momentum_grid_context_cache[scale_factor] = (
                    momentum_grid_context
                )
            base_context = _build_declared_base_context(
                perturbation_data=perturbation_data,
                model_parameters=source_parameters,
                physical_params=physical_params,
                numerics=numerics,
                k_value=float(k_value),
                eta_value=float(eta_value),
                background_scalars=background_scalars,
                momentum_grid_context=momentum_grid_context,
            )
            if resolve_graph:
                base_context = _resolve_declared_graph_context_ordered(
                    base_context,
                    perturbation_data,
                    allow_partial=True,
                    eta_grid=None,
                    execution_plan=execution_plan,
                    value_steps=(
                        execution_plan.value_steps
                        if graph_value_steps is None
                        else graph_value_steps
                    ),
                    compiled_value_program=(
                        full_context_program
                        if graph_value_steps is None
                        else state_independent_context_program
                    ),
                )
            scalar_base_context_cache[base_context_key] = base_context
        return base_context

    def _build_scalar_state_context(
        state_vector: numpy.ndarray,
        *,
        k_value: float,
        eta_value: float,
        background_scalars: Mapping[str, float],
        suppressed_collision_outputs: Mapping[str, float] | None = None,
        cache_token: tuple[int, float] | None = None,
    ) -> dict[str, Any]:
        """Return the scalar expression environment for one solver stage."""

        context = dict(
            _build_scalar_base_context(
                k_value=float(k_value),
                eta_value=float(eta_value),
                background_scalars=background_scalars,
                cache_token=cache_token,
                resolve_graph=bool(generated_scalar_hierarchy),
                graph_value_steps=(
                    state_independent_value_steps
                    if generated_scalar_hierarchy
                    else None
                ),
            )
        )
        for slot in runtime_spec.state_slots:
            value = float(state_vector[slot.index])
            if slot.order == 0:
                context[slot.variable] = value
            else:
                context[f"__d{slot.order}_{slot.variable}_{slot.wrt}"] = value
        return _resolve_declared_graph_context_ordered(
            context,
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
            derivative_steps=stage_derivative_steps,
            value_steps=(
                state_dependent_value_steps
                if generated_scalar_hierarchy
                else stage_value_steps
            ),
            suppressed_outputs=suppressed_collision_outputs,
            use_compiled_program=True,
            compiled_value_program=(
                state_dependent_context_program
                if generated_scalar_hierarchy
                else stage_context_program
            ),
        )

    def _build_array_context(
        histories: Mapping[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> dict[str, Any]:
        """Return the array-valued expression environment for one mode."""

        context: dict[str, Any] = {
            **{
                name: float(value) for name, value in source_parameters.items()
            },
            **{
                name: float(value)
                for name, value in physical_runtime_scalars.items()
            },
            "a": active_grids["a"],
            "a_initial": float(active_grids["a"][0]),
            "z": active_grids["z"],
            "eta": active_grids["eta"],
            "eta_initial": float(active_grids["eta"][0]),
            "H": active_grids["H"],
            "Hconf": active_grids["Hconf"],
            "Hconf_tau": active_grids["Hconf_tau"],
            "tau": active_grids["tau"],
            "tau_dot": active_grids["tau_dot"],
            "visibility": active_grids["visibility"],
            "chi": active_grids["chi"],
            "angular_diameter_distance": numpy.asarray(
                active_grids["angular_diameter_distance"],
                dtype=float,
            ),
            "sound_speed": active_grids["sound_speed"],
            "sound_speed_sq": active_grids["sound_speed_sq"],
            "baryon_sound_speed_sq": active_grids["baryon_sound_speed_sq"],
            "collision_rate": active_grids["collision_rate"],
            "free_streaming": active_grids["free_streaming"],
            "tight_coupling_drag": _compute_tight_coupling_drag(
                collision_rate=active_grids["collision_rate"],
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            ),
            "sound_horizon": float(background.sound_horizon_mpc),
            "k": float(k_value),
            "seed": _declared_runtime_seed(
                k_value=float(k_value),
                physical_params=physical_params,
                model_parameters=source_parameters,
            ),
        }
        for name, history in active_declared_background_histories.items():
            context.setdefault(name, numpy.asarray(history, dtype=float))
        context.update(
            _declared_momentum_grid_context(
                perturbation_data,
                model_parameters=source_parameters,
                physical_params=physical_params,
                scale_factor=active_grids["a"],
            )
        )
        for slot in runtime_spec.state_slots:
            if slot.order != 0:
                continue
            context[slot.variable] = numpy.asarray(
                histories[slot.variable],
                dtype=float,
            )
        if "Phi" in histories:
            # The evolution graph needs the algebraic Einstein relation
            # ``Phi_tau`` while it advances each mode.  The integrated
            # Sachs-Wolfe source, however, must use the derivative of the
            # evolved potential history; reusing that stage relation leaves
            # a spurious early-time ``-Hconf*Psi`` contribution.  Bind the
            # source-only symbol explicitly at the projection boundary.
            context["Phi_history_tau"] = _nonuniform_gradient(
                numpy.asarray(histories["Phi"], dtype=float),
                numpy.asarray(active_grids["eta"], dtype=float),
            )
        return _resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=False,
            eta_grid=active_grids["eta"],
            execution_plan=execution_plan,
        )

    def _evaluate_declared_sources(
        context: Mapping[str, Any],
        *,
        k_value: float,
        required_source_names: set[str] | None = None,
    ) -> dict[str, numpy.ndarray]:
        """Return source arrays keyed by source-term name."""

        source_arrays: dict[str, numpy.ndarray] = {}
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for source_step in execution_plan.source_steps:
                if (
                    required_source_names is not None
                    and source_step.output_name not in required_source_names
                ):
                    continue
                value = numpy.asarray(
                    _evaluate_compiled_expression_noerr(
                        source_step.compiled_expression,
                        context,
                    ),
                    dtype=float,
                )
                if value.ndim == 0:
                    value = numpy.full_like(
                        active_grids["eta"],
                        float(value),
                        dtype=float,
                    )
                if value.shape != active_grids["eta"].shape:
                    raise ValueError(
                        "Source term "
                        f"'{source_step.output_name}' did not evaluate to "
                        "an eta-grid history."
                    )
                if not numpy.all(numpy.isfinite(value)):
                    raise ValueError(
                        "Declared source term produced non-finite values: "
                        f"{source_step.output_name} at k={k_value}"
                    )
                source_arrays[source_step.output_name] = value
        return source_arrays

    def _mode_rhs(
        state_vector: numpy.ndarray,
        *,
        step_index: int,
        blend: float,
        k_value: float,
        tight_coupling_active: bool,
        include_split_collision_outputs: bool = False,
    ) -> numpy.ndarray:
        """Return the state derivative for one RK stage."""

        effective_state_vector = numpy.asarray(state_vector, dtype=float)
        eta_value, background_scalars = _scalar_background_context(
            step_index,
            blend,
        )
        if include_split_collision_outputs:
            suppressed_collision_outputs = {
                output_name: 0.0
                for runtime in split_collision_runtimes
                for output_name in (
                    runtime.name,
                    runtime.counterpart,
                )
                if output_name is not None
                and runtime.target_slot_indices
                and (
                    runtime.activation_strategy != "tight_coupling"
                    or tight_coupling_active
                )
            }
        else:
            suppressed_collision_outputs = {
                output_name: 0.0
                for runtime in split_collision_runtimes
                for output_name in (
                    runtime.name,
                    runtime.counterpart,
                )
                if output_name is not None
                and (
                    runtime.activation_strategy == "always"
                    or (
                        runtime.activation_strategy == "tight_coupling"
                        and tight_coupling_active
                    )
                )
            }
        scalar_context = _build_scalar_state_context(
            effective_state_vector,
            k_value=float(k_value),
            eta_value=float(eta_value),
            background_scalars=background_scalars,
            suppressed_collision_outputs=suppressed_collision_outputs,
            cache_token=(int(step_index), float(blend)),
        )
        derivative = numpy.zeros_like(state_vector, dtype=float)
        coordinate_rates: dict[str, float] = {}
        for slot_plan in execution_plan.equation_slot_plans:
            if slot_plan.wrt in coordinate_rates:
                continue
            coordinate_rates[slot_plan.wrt] = _resolve_coordinate_rate(
                wrt_name=slot_plan.wrt,
                scalar_context=scalar_context,
                step_index=step_index,
                blend=blend,
                k_value=float(k_value),
            )
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            try:
                equation_program(
                    scalar_context,
                    effective_state_vector,
                    derivative,
                    coordinate_rates,
                )
            except (KeyError, NameError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Declared CMB equation program failed at "
                    f"eta={eta_value}, k={k_value}"
                ) from exc
            except ArithmeticError as exc:
                raise ValueError(
                    "Declared CMB equation result must be finite; "
                    f"evaluation failed at eta={eta_value}, k={k_value}"
                ) from exc
        if include_split_collision_outputs:
            # Continuous evolution must include the complete declared exact
            # collision block.  Its scalar expression is suppressed above
            # for the same reason that split evolution suppresses it: the
            # matrix is the authoritative operator for every selected state.
            for runtime in split_collision_runtimes:
                if runtime.activation_strategy == "tight_coupling":
                    if not tight_coupling_active:
                        continue
                if not runtime.target_slot_indices:
                    continue
                collision_rate = _coerce_numeric_scalar(
                    _evaluate_compiled_expression_noerr(
                        runtime.rate_expression,
                        scalar_context,
                    ),
                    name=f"collision operator '{runtime.name}' rate",
                )
                matrix = numpy.asarray(
                    [
                        [
                            _coerce_numeric_scalar(
                                _evaluate_compiled_expression_noerr(
                                    entry,
                                    scalar_context,
                                ),
                                name=(
                                    f"collision operator '{runtime.name}' "
                                    "matrix entry"
                                ),
                            )
                            for entry in row
                        ]
                        for row in runtime.matrix
                    ],
                    dtype=float,
                )
                damping_coefficient = None
                if runtime.damping_coefficient is not None:
                    damping_coefficient = _coerce_numeric_scalar(
                        _evaluate_compiled_expression_noerr(
                            runtime.damping_coefficient,
                            scalar_context,
                        ),
                        name=(
                            f"collision operator '{runtime.name}' "
                            "damping coefficient"
                        ),
                    )
                if not numpy.isfinite(collision_rate):
                    raise ValueError(
                        "Declared collision operator produced a non-finite "
                        f"rate during continuous evolution: {runtime.name}"
                    )
                target_indices = tuple(runtime.target_slot_indices)
                target_state = effective_state_vector[list(target_indices)]
                derivative[list(target_indices)] += float(collision_rate) * (
                    numpy.asarray(matrix, dtype=float) @ target_state
                )
                if runtime.damping_slot_indices:
                    if damping_coefficient is None:
                        raise ValueError(
                            "Declared exact collision operator omitted a "
                            f"damping coefficient: {runtime.name}"
                        )
                    damping_indices = tuple(runtime.damping_slot_indices)
                    derivative[list(damping_indices)] += (
                        float(collision_rate)
                        * float(damping_coefficient)
                        * effective_state_vector[list(damping_indices)]
                    )
        if not numpy.all(numpy.isfinite(derivative)):
            bad_indices = numpy.flatnonzero(~numpy.isfinite(derivative))
            bad_index = int(bad_indices[0]) if bad_indices.size else -1
            raise ValueError(
                "Declared CMB evolution produced non-finite derivatives at "
                f"eta={eta_value}, k={k_value}, state_index={bad_index}"
            )
        return derivative

    # These dependency classifications depend on the declared graph and the
    # stable background-history names, not on the Fourier mode being evolved.
    state_variable_names = {
        str(slot.variable)
        for slot in runtime_spec.state_slots
        if int(slot.order) == 0
    }
    dynamic_context_names = {
        *active_grids,
        *active_declared_background_histories,
    }
    derived_entries = getattr(perturbation_data, "derived", {}) or {}
    changed = True
    while changed:
        changed = False
        for name, entry in derived_entries.items():
            if str(name) in dynamic_context_names:
                continue
            dependencies = set(getattr(entry, "dependencies", ()))
            if dependencies & dynamic_context_names:
                dynamic_context_names.add(str(name))
                changed = True
    state_dependent_names = set(state_variable_names)
    state_dependency_entries: list[tuple[str, set[str]]] = [
        (
            str(name),
            set(getattr(entry, "dependencies", ()) or ()),
        )
        for name, entry in derived_entries.items()
    ]
    for relation_entries in (
        getattr(perturbation_data, "constraints", {}).values(),
        getattr(perturbation_data, "closures", {}).values(),
        getattr(perturbation_data, "interactions", {}).values(),
        getattr(perturbation_data, "collision_operators", {}).values(),
    ):
        for entry in relation_entries:
            target_name = getattr(entry, "target", None)
            if target_name is None:
                target_name = getattr(entry, "name", None)
            dependencies = getattr(entry, "dependencies", ()) or ()
            if target_name is not None:
                state_dependency_entries.append(
                    (str(target_name), set(dependencies))
                )
    changed = True
    while changed:
        changed = False
        for name, dependencies in state_dependency_entries:
            if name in state_dependent_names:
                continue
            if dependencies & state_dependent_names:
                state_dependent_names.add(name)
                changed = True
    state_dependent_value_steps = tuple(
        step
        for step in execution_plan.value_steps
        if str(step.output_name) in state_dependent_names
        or bool(set(step.dependencies) & state_dependent_names)
    )
    state_independent_value_steps = tuple(
        step
        for step in execution_plan.value_steps
        if step not in state_dependent_value_steps
    )
    batched_rhs_value_steps = tuple(
        step
        for step in state_dependent_value_steps
        if str(step.output_name) in equation_required_names
    )

    def _compile_value_program(
        value_steps: tuple[Any, ...],
        *,
        overwrite_outputs: tuple[str, ...] = (),
    ) -> Any | None:
        """Compile one reusable direct-assignment context program."""

        if not value_steps:
            return None
        value_names = tuple(str(step.output_name) for step in value_steps)
        return _compile_ordered_context_program(
            tuple(
                (
                    str(step.output_name),
                    str(step.compiled_expression.expression),
                )
                for step in value_steps
            ),
            tuple(
                output_name
                for output_name in (
                    overwrite_outputs or execution_plan.relation_target_names
                )
                if output_name in value_names
            ),
        )

    full_context_program = _compile_value_program(execution_plan.value_steps)
    state_independent_context_program = _compile_value_program(
        state_independent_value_steps
    )
    state_dependent_context_program = _compile_value_program(
        state_dependent_value_steps,
        overwrite_outputs=tuple(
            str(step.output_name) for step in state_dependent_value_steps
        ),
    )
    batched_rhs_context_program = _compile_value_program(
        batched_rhs_value_steps
    )
    stage_context_program = _compile_value_program(stage_value_steps)
    runtime_envelope["batched_rhs_value_step_count"] = int(
        len(batched_rhs_value_steps)
    )
    runtime_envelope["batched_diagnostic_value_step_count"] = int(
        len(state_dependent_value_steps)
    )
    static_collision_runtimes = {
        runtime.name: not (
            set(runtime.rate_expression.dependencies)
            | {
                dependency
                for row in runtime.matrix
                for entry in row
                for dependency in entry.dependencies
            }
            | (
                set(runtime.damping_coefficient.dependencies)
                if runtime.damping_coefficient is not None
                else set()
            )
        )
        & (state_variable_names | dynamic_context_names)
        for runtime in split_collision_runtimes
    }
    state_independent_collision_runtimes = {
        runtime.name: not (
            set(runtime.rate_expression.dependencies)
            | {
                dependency
                for row in runtime.matrix
                for entry in row
                for dependency in entry.dependencies
            }
            | (
                set(runtime.damping_coefficient.dependencies)
                if runtime.damping_coefficient is not None
                else set()
            )
        )
        & state_dependent_names
        for runtime in split_collision_runtimes
    }

    metric_constraint_state_key = next(
        (
            key
            for key in (
                ("Phi", "tau", 0),
                ("Phi_gi", "tau", 0),
            )
            if key in runtime_spec.state_index_by_key
        ),
        None,
    )

    def _prepare_mode_initial_state(
        mode_k_value: float,
    ) -> tuple[
        numpy.ndarray,
        tuple[tuple[str, str, int], ...],
        dict[str, dict[str, Any]],
    ]:
        """Prepare a declared regular initial state and validate constraints.

        Generated scalar contracts provide a regular superhorizon metric seed.
        The Einstein energy equation is nearly singular on that surface, so
        validation must not replace the declared seed with an algebraic solve
        that amplifies its small residual into a spurious zero potential.
        """

        initial_eta, initial_background = _scalar_background_context(
            0,
            0.0,
            k_value=float(mode_k_value),
        )
        initial_context = _build_declared_base_context(
            perturbation_data=perturbation_data,
            model_parameters=source_parameters,
            physical_params=physical_params,
            numerics=numerics,
            k_value=float(mode_k_value),
            eta_value=float(initial_eta),
            background_scalars=initial_background,
        )
        initial_state, assigned_targets = _evaluate_declared_initial_state(
            perturbation_data=perturbation_data,
            execution_plan=execution_plan,
            base_context=initial_context,
        )
        state = numpy.asarray(initial_state, dtype=float)
        if not numpy.all(numpy.isfinite(state)):
            raise ValueError(
                "Declared initial state is non-finite before evolution: "
                f"k={float(mode_k_value)}"
            )
        initial_state_context = _build_scalar_state_context(
            state,
            k_value=float(mode_k_value),
            eta_value=float(initial_eta),
            background_scalars=initial_background,
        )
        if generated_scalar_hierarchy and metric_constraint_state_key is None:
            raise ConstraintViolationError(
                "Generated scalar initial data do not expose a metric state",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                },
            )
        if (
            generated_scalar_hierarchy
            and ("theta_gamma0", "tau", 0) in runtime_spec.state_index_by_key
        ):
            # The leading regular series cancels the k=0 Einstein surface.
            # At finite k its omitted O((k eta)^2) density term must be put
            # into the photon monopole, not absorbed by changing the metric
            # seed.  This is the unique local radiation-density correction
            # that closes the energy constraint while preserving the declared
            # primordial potential.
            photon_index = runtime_spec.state_index_by_key[
                ("theta_gamma0", "tau", 0)
            ]
            energy_residual = float(
                initial_state_context.get("einstein_energy_residual", 0.0)
            )
            gravity = float(
                initial_state_context.get("einstein_gravity_strength", 0.0)
            )
            omega_gamma = float(initial_state_context.get("Omega_gamma0", 0.0))
            scale_factor = float(initial_background["a"])
            coefficient = 1.5 * gravity * 4.0 * omega_gamma / (scale_factor**2)
            if (
                numpy.isfinite(energy_residual)
                and numpy.isfinite(coefficient)
                and abs(coefficient) > 1.0e-30
            ):
                state[photon_index] -= energy_residual / coefficient
                initial_state_context = _build_scalar_state_context(
                    state,
                    k_value=float(mode_k_value),
                    eta_value=float(initial_eta),
                    background_scalars=initial_background,
                )
        initial_diagnostics = _validate_generated_scalar_initial_constraints(
            perturbation_data=perturbation_data,
            context=initial_state_context,
            k_value=float(mode_k_value),
        )
        if generated_scalar_hierarchy:
            initial_state_diagnostics_by_k[f"{float(mode_k_value):.12g}"] = {
                "k": float(mode_k_value),
                "eta": float(initial_eta),
                "state": {
                    str(slot.variable): float(state[slot.index])
                    for slot in runtime_spec.state_slots
                    if int(slot.order) == 0
                },
                "constraint_diagnostics": dict(initial_diagnostics),
            }
        if generated_scalar_hierarchy:
            _validate_declared_conservation_rules(
                perturbation_data=perturbation_data,
                context=initial_state_context,
                k_value=float(mode_k_value),
            )
        _validate_generated_vector_initial_constraints(
            perturbation_data=perturbation_data,
            context=initial_state_context,
            k_value=float(mode_k_value),
        )
        _validate_generated_tensor_initial_constraints(
            perturbation_data=perturbation_data,
            context=initial_state_context,
            k_value=float(mode_k_value),
        )
        return state, assigned_targets, initial_diagnostics

    scalar_initial_constraint_preflight: dict[str, Any] = {
        "performed": False,
        "failure_order": "ascending_k",
        "k_values": (),
        "mode_count": 0,
        "residuals": {},
    }

    def _preflight_generated_scalar_initial_conditions() -> None:
        """Validate every requested scalar mode before any ODE evolution."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal active_k_value
        nonlocal scalar_base_context_cache
        nonlocal scalar_background_context_cache
        if not generated_scalar_hierarchy:
            return
        ordered_k_values = numpy.sort(
            numpy.unique(numpy.asarray(k_values, dtype=float))
        )
        if (
            ordered_k_values.ndim != 1
            or ordered_k_values.size == 0
            or not numpy.all(numpy.isfinite(ordered_k_values))
        ):
            raise ValueError(
                "Generated scalar initial-condition preflight requires a "
                "finite requested k grid"
            )
        residuals: dict[str, dict[str, Any]] = {}
        for mode_k_value in ordered_k_values:
            scalar_base_context_cache = {}
            scalar_background_context_cache = {}
            active_k_value = float(mode_k_value)
            (
                active_grids,
                active_declared_background_histories,
                active_coordinate_rate_histories,
            ) = _mode_grids_for_k(float(mode_k_value))
            _, _, mode_diagnostics = _prepare_mode_initial_state(
                float(mode_k_value)
            )
            for residual_name, metrics in mode_diagnostics.items():
                aggregate = residuals.setdefault(
                    residual_name,
                    {
                        "maximum_normalized": -numpy.inf,
                        "maximum_absolute": 0.0,
                        "normalization_scale": 0.0,
                        "normalization_terms": {},
                        "normalization_source": "",
                        "k": 0.0,
                        "tolerance": float(metrics["tolerance"]),
                        "tolerance_provenance": str(
                            metrics["tolerance_provenance"]
                        ),
                    },
                )
                if float(metrics["normalized_residual"]) > float(
                    aggregate["maximum_normalized"]
                ):
                    aggregate.update(
                        {
                            "maximum_normalized": float(
                                metrics["normalized_residual"]
                            ),
                            "maximum_absolute": float(
                                metrics["absolute_residual"]
                            ),
                            "normalization_scale": float(
                                metrics["normalization_scale"]
                            ),
                            "normalization_terms": dict(
                                metrics["normalization_terms"]
                            ),
                            "normalization_source": str(
                                metrics["normalization_source"]
                            ),
                            "k": float(mode_k_value),
                        }
                    )
        scalar_initial_constraint_preflight.update(
            {
                "performed": True,
                "k_values": tuple(float(value) for value in ordered_k_values),
                "mode_count": int(ordered_k_values.size),
                "residuals": residuals,
            }
        )
        scalar_base_context_cache = {}
        scalar_background_context_cache = {}
        active_grids = dict(source_grids)
        active_declared_background_histories = (
            source_declared_background_histories
        )
        active_coordinate_rate_histories = source_coordinate_rate_histories

    with performance_timer.phase("initial_data"):
        _preflight_generated_scalar_initial_conditions()

    def _reconstruct_scalar_constraint_source_histories(
        source_histories: Mapping[str, numpy.ndarray],
        *,
        mode_k_value: float,
        apply_reconstruction: bool | None = None,
    ) -> dict[str, numpy.ndarray]:
        """Optionally solve the coupled scalar Einstein surface.

        Generated scalar evolution is advanced as a declared differential
        system.  Algebraic reconstruction is therefore opt-in: dividing the
        density constraint by ``k**2`` can amplify ordinary early-time
        truncation error into an unphysical metric, especially on
        super-horizon modes.  Production line-of-sight integration keeps the
        evolved histories unless a contract explicitly requests this
        reconstruction for a diagnostic comparison.
        """

        nonlocal scalar_constraint_projection_count
        nonlocal scalar_constraint_diagnostic_projection_count
        nonlocal scalar_constraint_projection_max_relative_correction
        if not generated_scalar_hierarchy:
            return {
                name: numpy.asarray(values, dtype=float)
                for name, values in source_histories.items()
            }
        strict_generated_graph = bool(
            (
                (
                    getattr(
                        perturbation_data,
                        "manifest_summary",
                        {},
                    )
                    or {}
                ).get("generated_scalar_source_closure", {})
                or {}
            ).get("status")
            == "validated"
        )
        should_reconstruct = (
            source_history_reconstruction_enabled
            if apply_reconstruction is None
            else bool(apply_reconstruction)
        )
        if not should_reconstruct:
            return {
                name: numpy.asarray(values, dtype=float).copy()
                for name, values in source_histories.items()
            }
        if metric_constraint_state_key is None:
            raise ConstraintViolationError(
                "Generated scalar source histories do not expose a metric "
                "state for the Einstein reconstruction",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                },
            )
        projected_histories = {
            name: numpy.asarray(values, dtype=float).copy()
            for name, values in source_histories.items()
        }
        state_name = str(metric_constraint_state_key[0])
        if state_name not in projected_histories:
            raise ConstraintViolationError(
                "Generated scalar source histories omit the metric state "
                "required by the Einstein reconstruction",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                    "state": state_name,
                },
            )

        def _bind_constraint_metric(
            histories: dict[str, numpy.ndarray],
            phi_values: numpy.ndarray,
            context: Mapping[str, Any],
        ) -> None:
            """Bind the reconstructed metric into one history set."""

            histories[state_name] = numpy.asarray(phi_values, dtype=float)
            if str(getattr(perturbation_data, "gauge", "")) != "synchronous":
                return
            if not {
                "eta_sync_metric",
                "gauge_shift_alpha",
            }.issubset(histories):
                return
            histories["eta_sync_metric"] = numpy.asarray(
                phi_values, dtype=float
            ) + numpy.asarray(context["Hconf"], dtype=float) * numpy.asarray(
                histories["gauge_shift_alpha"], dtype=float
            )

        source_context = _build_array_context(
            projected_histories,
            k_value=float(mode_k_value),
        )
        required_names = (
            "acoustic_k_sq",
            "Hconf",
            "metric_momentum_source_drive",
            "einstein_gravity_strength",
            "total_density_source",
        )
        missing_names = tuple(
            name for name in required_names if name not in source_context
        )
        if missing_names:
            raise ConstraintViolationError(
                "Generated scalar source histories cannot reconstruct "
                "the Einstein energy surface",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                    "missing_terms": missing_names,
                },
            )
        acoustic_k_sq = numpy.asarray(
            source_context["acoustic_k_sq"],
            dtype=float,
        )
        if not numpy.all(
            numpy.isfinite(acoustic_k_sq) & (acoustic_k_sq > 0.0)
        ):
            raise ConstraintViolationError(
                "Generated scalar source histories require a positive finite "
                "Einstein k^2 scale",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                },
            )
        previous_phi = numpy.asarray(
            projected_histories[state_name],
            dtype=float,
        )
        energy_source = 3.0 * numpy.asarray(
            source_context["Hconf"], dtype=float
        ) * numpy.asarray(
            source_context["metric_momentum_source_drive"],
            dtype=float,
        ) + 1.5 * numpy.asarray(
            source_context["einstein_gravity_strength"],
            dtype=float,
        ) * numpy.asarray(
            source_context["total_density_source"],
            dtype=float,
        )
        reconstructed_phi = -energy_source / acoustic_k_sq
        if not numpy.all(numpy.isfinite(reconstructed_phi)):
            raise NonFiniteEvolutionError(
                "Generated scalar Einstein reconstruction produced "
                f"non-finite metric values at k={mode_k_value}",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                },
            )
        _bind_constraint_metric(
            projected_histories,
            reconstructed_phi,
            source_context,
        )
        reconstructed_context = _build_array_context(
            projected_histories,
            k_value=float(mode_k_value),
        )
        energy_metrics = _scalar_einstein_constraint_metrics(
            reconstructed_context,
            "einstein_energy_residual",
            strict=strict_generated_graph,
        )
        maximum_normalized = float(
            numpy.max(
                numpy.asarray(
                    energy_metrics["normalized_values"],
                    dtype=float,
                ),
                initial=0.0,
            )
        )
        if maximum_normalized > 1.0e-10:
            raise ConstraintViolationError(
                "Generated scalar source-history Einstein reconstruction did "
                "not converge",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(mode_k_value),
                    "iterations": 1,
                    "maximum_normalized": maximum_normalized,
                },
            )
        correction = numpy.abs(
            reconstructed_phi - previous_phi
        ) / numpy.maximum(
            numpy.maximum(
                numpy.abs(reconstructed_phi),
                numpy.abs(previous_phi),
            ),
            numpy.finfo(float).tiny,
        )
        mode_max_relative_correction = float(
            numpy.max(correction, initial=0.0)
        )
        if source_history_reconstruction_enabled:
            scalar_constraint_projection_count += 1
        else:
            scalar_constraint_diagnostic_projection_count += 1
        scalar_constraint_projection_max_relative_correction = max(
            scalar_constraint_projection_max_relative_correction,
            mode_max_relative_correction,
        )
        return projected_histories

    def _evaluate_source_histories(
        mode_k_value: float,
        source_histories: Mapping[str, numpy.ndarray],
        *,
        collect_diagnostics: bool = True,
        source_grid_indices: numpy.ndarray | None = None,
        required_source_names: set[str] | None = None,
    ) -> dict[str, numpy.ndarray]:
        """Evaluate declared sources and conservation on source-grid rows."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal scalar_constraint_diagnostics
        if source_grid_indices is None:
            active_grids = dict(source_grids)
            active_declared_background_histories = (
                source_declared_background_histories
            )
            active_coordinate_rate_histories = source_coordinate_rate_histories
            evaluation_histories = source_histories
        else:
            indices = numpy.asarray(source_grid_indices, dtype=int)
            active_grids = {
                name: numpy.asarray(values)[indices]
                for name, values in source_grids.items()
            }
            active_declared_background_histories = {
                name: numpy.asarray(values)[indices]
                for (
                    name,
                    values,
                ) in source_declared_background_histories.items()
            }
            active_coordinate_rate_histories = {
                name: numpy.asarray(values)[indices]
                for name, values in source_coordinate_rate_histories.items()
            }
            evaluation_histories = {
                name: numpy.asarray(history, dtype=float)[indices]
                for name, history in source_histories.items()
            }
        raw_evaluation_histories = {
            name: numpy.asarray(history, dtype=float)
            for name, history in evaluation_histories.items()
        }
        raw_conservation_context = None
        if (
            diagnostic_source_audit
            and collect_diagnostics
            and source_grid_indices is None
        ):
            raw_array_context = _build_array_context(
                raw_evaluation_histories,
                k_value=float(mode_k_value),
            )
            raw_source_arrays = _evaluate_declared_sources(
                raw_array_context,
                k_value=float(mode_k_value),
                required_source_names=required_source_names,
            )
            raw_conservation_context = dict(raw_array_context)
            raw_conservation_context.update(raw_source_arrays)
            raw_conservation_context = _resolve_declared_graph_context(
                raw_conservation_context,
                perturbation_data,
                allow_partial=True,
                eta_grid=active_grids["eta"],
                execution_plan=execution_plan,
            )
        evaluation_histories = _reconstruct_scalar_constraint_source_histories(
            evaluation_histories,
            mode_k_value=float(mode_k_value),
        )
        array_context = _build_array_context(
            evaluation_histories,
            k_value=float(mode_k_value),
        )
        source_context_pre_resolution_by_k[f"{float(mode_k_value):.12g}"] = {
            name: float(
                numpy.max(
                    numpy.abs(numpy.asarray(array_context[name], dtype=float)),
                    initial=0.0,
                )
            )
            for name in ("Phi", "Psi", "metric_shear_correction")
            if name in array_context
        }
        source_arrays = _evaluate_declared_sources(
            array_context,
            k_value=float(mode_k_value),
            required_source_names=required_source_names,
        )
        conservation_context = dict(array_context)
        conservation_context.update(source_arrays)
        conservation_context = _resolve_declared_graph_context(
            conservation_context,
            perturbation_data,
            allow_partial=True,
            eta_grid=active_grids["eta"],
            execution_plan=execution_plan,
        )
        if source_grid_indices is None:
            _validate_metric_history_derivatives(
                float(mode_k_value),
                conservation_context,
            )
        if (
            diagnostic_source_audit
            and collect_diagnostics
            and source_grid_indices is None
        ):
            _record_source_history_residual_samples(
                float(mode_k_value),
                (
                    conservation_context
                    if raw_conservation_context is None
                    else raw_conservation_context
                ),
            )
        source_context_max_abs_by_k[f"{float(mode_k_value):.12g}"] = {
            name: float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(conservation_context[name], dtype=float)
                    ),
                    initial=0.0,
                )
            )
            for name in (
                "visibility",
                "metric_shear_correction",
                "Psi",
                "Phi_tau",
                "Psi_tau",
                "total_shear_source",
                "polarization_moment",
                "temperature_quadrupole",
                "polarization_source",
            )
            if name in conservation_context
        }
        diagnostic_context = conservation_context
        if (
            generated_scalar_hierarchy
            and not source_history_reconstruction_enabled
        ):
            diagnostic_histories = (
                _reconstruct_scalar_constraint_source_histories(
                    evaluation_histories,
                    mode_k_value=float(mode_k_value),
                    apply_reconstruction=True,
                )
            )
            diagnostic_context = _build_array_context(
                diagnostic_histories,
                k_value=float(mode_k_value),
            )
            diagnostic_source_arrays = _evaluate_declared_sources(
                diagnostic_context,
                k_value=float(mode_k_value),
                required_source_names=required_source_names,
            )
            diagnostic_context = dict(diagnostic_context)
            diagnostic_context.update(diagnostic_source_arrays)
            diagnostic_context = _resolve_declared_graph_context(
                diagnostic_context,
                perturbation_data,
                allow_partial=True,
                eta_grid=active_grids["eta"],
                execution_plan=execution_plan,
            )
        mode_constraint_diagnostics = _validate_scalar_constraint_histories(
            perturbation_data=perturbation_data,
            context=diagnostic_context,
            eta_grid=active_grids["eta"],
            accuracy_controls=declared_accuracy_controls,
            k_value=float(mode_k_value),
        )
        if collect_diagnostics:
            for (
                residual_name,
                mode_metrics,
            ) in mode_constraint_diagnostics.items():
                aggregate = scalar_constraint_diagnostics.setdefault(
                    residual_name,
                    {
                        "maximum_absolute": 0.0,
                        "maximum_absolute_eta": 0.0,
                        "maximum_normalized": -numpy.inf,
                        "maximum_eta": 0.0,
                        "maximum_grid_fraction": 0.0,
                        "physical_regime": "",
                        "normalization_scale": 0.0,
                        "normalization_terms": {},
                        "normalization_source": "",
                        "tolerance": float(mode_metrics["tolerance"]),
                        "tolerance_kind": str(mode_metrics["tolerance_kind"]),
                        "tolerance_provenance": str(
                            mode_metrics["tolerance_provenance"]
                        ),
                        "tolerance_source": str(
                            mode_metrics["tolerance_source"]
                        ),
                        "enforced": bool(mode_metrics["enforced"]),
                        "reference_eta_samples": int(
                            mode_metrics["reference_eta_samples"]
                        ),
                        "reference_resolution_met": True,
                        "resolution_status": "reference",
                        "physical_judgement": "evaluated",
                        "refinement_evidence": {},
                        "anchors": {},
                        "normalized_anchors": {},
                        "mode_count": 0,
                        "sample_count": 0,
                    },
                )
                aggregate["maximum_absolute"] = max(
                    float(aggregate["maximum_absolute"]),
                    float(mode_metrics["maximum_absolute"]),
                )
                if float(mode_metrics["maximum_normalized"]) > float(
                    aggregate["maximum_normalized"]
                ):
                    aggregate.update(
                        {
                            "maximum_normalized": float(
                                mode_metrics["maximum_normalized"]
                            ),
                            "maximum_eta": float(mode_metrics["maximum_eta"]),
                            "maximum_grid_fraction": float(
                                mode_metrics["maximum_grid_fraction"]
                            ),
                            "physical_regime": str(
                                mode_metrics["physical_regime"]
                            ),
                            "normalization_scale": float(
                                mode_metrics["normalization_scale"]
                            ),
                            "normalization_terms": dict(
                                mode_metrics["normalization_terms"]
                            ),
                            "normalization_source": str(
                                mode_metrics["normalization_source"]
                            ),
                            "tolerance": float(mode_metrics["tolerance"]),
                            "tolerance_kind": str(
                                mode_metrics["tolerance_kind"]
                            ),
                            "tolerance_provenance": str(
                                mode_metrics["tolerance_provenance"]
                            ),
                            "tolerance_source": str(
                                mode_metrics["tolerance_source"]
                            ),
                            "enforced": bool(mode_metrics["enforced"]),
                            "refinement_evidence": dict(
                                mode_metrics["refinement_evidence"]
                            ),
                        }
                    )
                if float(mode_metrics["maximum_absolute"]) >= float(
                    aggregate["maximum_absolute"]
                ):
                    aggregate["maximum_absolute_eta"] = float(
                        mode_metrics["maximum_absolute_eta"]
                    )
                aggregate["reference_resolution_met"] = bool(
                    aggregate["reference_resolution_met"]
                    and mode_metrics["reference_resolution_met"]
                )
                if not mode_metrics["reference_resolution_met"]:
                    aggregate["resolution_status"] = "under_resolved"
                    aggregate["physical_judgement"] = "deferred"
                aggregate["mode_count"] = int(aggregate["mode_count"]) + 1
                aggregate["sample_count"] = int(
                    aggregate["sample_count"]
                ) + int(mode_metrics["sample_count"])
                for anchor_name, anchor_value in mode_metrics[
                    "anchors"
                ].items():
                    aggregate["anchors"][anchor_name] = max(
                        float(aggregate["anchors"].get(anchor_name, 0.0)),
                        float(anchor_value),
                    )
                for anchor_name, anchor_value in mode_metrics[
                    "normalized_anchors"
                ].items():
                    aggregate["normalized_anchors"][anchor_name] = max(
                        float(
                            aggregate["normalized_anchors"].get(
                                anchor_name,
                                0.0,
                            )
                        ),
                        float(anchor_value),
                    )
        _validate_declared_conservation_rules(
            perturbation_data=perturbation_data,
            context=conservation_context,
            k_value=float(mode_k_value),
        )
        return source_arrays

    def _evolve_declared_mode(
        k_value: float,
        *,
        evolution_sample_count_override: int | None = None,
        history_sink: dict[str, Any] | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
        """Integrate one Fourier mode through the declared graph."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal scalar_base_context_cache
        nonlocal scalar_background_context_cache
        nonlocal active_k_value

        scalar_base_context_cache = {}
        scalar_background_context_cache = {}
        active_k_value = float(k_value)

        end_boundary_entries = execution_plan.end_condition_entries
        (
            active_grids,
            active_declared_background_histories,
            active_coordinate_rate_histories,
        ) = _mode_grids_for_k(
            float(k_value),
            evolution_sample_count_override=evolution_sample_count_override,
        )
        initial_eta, initial_background = _scalar_background_context(0, 0.0)

        collision_metadata_cache: dict[
            tuple[str, int, float], tuple[float, numpy.ndarray, float | None]
        ] = {}
        state_independent_collision_metadata_cache: dict[
            tuple[str, int, float], tuple[float, numpy.ndarray, float | None]
        ] = {}
        collision_eigendecomposition_cache: dict[
            tuple[tuple[int, ...], bytes],
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None,
        ] = {}
        fast_collision_solver_cache: dict[str, tuple[Any, ...]] = {}

        def _describe_nonfinite_state(
            state_vector: numpy.ndarray,
        ) -> str:
            """Return the names of the first few non-finite state slots."""

            bad_indices = numpy.flatnonzero(~numpy.isfinite(state_vector))
            if bad_indices.size == 0:
                return ""
            bad_names = [
                runtime_spec.state_slots[int(index)].variable
                for index in bad_indices[:5]
            ]
            return ", ".join(bad_names)

        def _collision_metadata_for_state(
            state_vector: numpy.ndarray,
            *,
            runtime: _CompiledCollisionOperatorRuntime,
            step_index: int,
            blend: float,
            k_value: float,
        ) -> tuple[float, numpy.ndarray, float | None]:
            """Resolve one declared collision operator at one RK stage."""

            metadata_key = (runtime.name, int(step_index), float(blend))
            metadata = None
            if static_collision_runtimes.get(runtime.name, False):
                metadata = collision_metadata_cache.get(metadata_key)
            elif state_independent_collision_runtimes.get(runtime.name, False):
                metadata = state_independent_collision_metadata_cache.get(
                    metadata_key
                )
            if metadata is not None:
                return metadata

            eta_value, background_scalars = _scalar_background_context(
                step_index,
                blend,
            )
            if state_independent_collision_runtimes.get(runtime.name, False):
                scalar_context = _build_scalar_base_context(
                    k_value=float(k_value),
                    eta_value=float(eta_value),
                    background_scalars=background_scalars,
                    cache_token=(int(step_index), float(blend)),
                    resolve_graph=True,
                )
            else:
                scalar_context = _build_scalar_state_context(
                    state_vector,
                    k_value=float(k_value),
                    eta_value=float(eta_value),
                    background_scalars=background_scalars,
                    cache_token=(int(step_index), float(blend)),
                )
            collision_rate = _coerce_numeric_scalar(
                _evaluate_compiled_expression_noerr(
                    runtime.rate_expression,
                    scalar_context,
                ),
                name=f"collision operator '{runtime.name}' rate",
            )
            matrix = numpy.asarray(
                [
                    [
                        _coerce_numeric_scalar(
                            _evaluate_compiled_expression_noerr(
                                entry,
                                scalar_context,
                            ),
                            name=(
                                f"collision operator '{runtime.name}' "
                                "matrix entry"
                            ),
                        )
                        for entry in row
                    ]
                    for row in runtime.matrix
                ],
                dtype=float,
            )
            damping_coefficient = None
            if runtime.damping_coefficient is not None:
                damping_coefficient = _coerce_numeric_scalar(
                    _evaluate_compiled_expression_noerr(
                        runtime.damping_coefficient,
                        scalar_context,
                    ),
                    name=(
                        f"collision operator '{runtime.name}' "
                        "damping coefficient"
                    ),
                )
            metadata = (
                float(collision_rate),
                matrix,
                damping_coefficient,
            )
            if static_collision_runtimes.get(runtime.name, False):
                collision_metadata_cache[metadata_key] = metadata
            elif state_independent_collision_runtimes.get(runtime.name, False):
                state_independent_collision_metadata_cache[metadata_key] = (
                    metadata
                )
            return metadata

        def _validate_collision_invariants(
            state_vector: numpy.ndarray,
            *,
            runtime: _CompiledCollisionOperatorRuntime,
            step_index: int,
            blend: float,
            k_value: float,
        ) -> None:
            """Validate one operator's conservation rules after its update."""

            if not runtime.conservation_rule_names:
                return
            eta_value, background_scalars = _scalar_background_context(
                step_index,
                blend,
            )
            context = _build_scalar_state_context(
                state_vector,
                k_value=float(k_value),
                eta_value=float(eta_value),
                background_scalars=background_scalars,
            )
            _validate_declared_conservation_rules(
                perturbation_data=perturbation_data,
                context=context,
                k_value=float(k_value),
                rule_names=runtime.conservation_rule_names,
            )

        def _project_declared_fast_collision_state(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            k_value: float,
            tight_coupling_active: bool,
        ) -> numpy.ndarray:
            """Apply declared first-order fast-manifold constraints."""

            if not tight_coupling_active or not split_collision_runtimes:
                return numpy.asarray(state_vector, dtype=float)
            projected = numpy.asarray(state_vector, dtype=float).copy()
            for runtime in split_collision_runtimes:
                if not runtime.fast_manifold:
                    continue
                if runtime.activation_strategy == "tight_coupling":
                    if not tight_coupling_active:
                        continue
                if runtime.integration_strategy != "exact":
                    continue
                collision_rate, matrix, damping_coefficient = (
                    _collision_metadata_for_state(
                        projected,
                        runtime=runtime,
                        step_index=step_index,
                        blend=blend,
                        k_value=float(k_value),
                    )
                )
                if (
                    not numpy.isfinite(collision_rate)
                    or collision_rate <= 1.0e-12
                ):
                    continue
                if not numpy.all(numpy.isfinite(matrix)):
                    raise ValueError(
                        "Declared collision operator produced a non-finite "
                        "matrix before fast-manifold projection: "
                        f"{runtime.name}"
                    )
                target_indices = tuple(runtime.target_slot_indices)
                damping_indices = tuple(
                    index
                    for index in runtime.damping_slot_indices
                    if index not in target_indices
                )
                forcing = None
                if target_indices or damping_indices:
                    forcing = _mode_rhs(
                        projected,
                        step_index=step_index,
                        blend=blend,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                if target_indices:
                    target_state = _solve_declared_fast_collision_target(
                        matrix,
                        forcing[list(target_indices)],  # type: ignore[index]
                        projected[list(target_indices)],
                        float(collision_rate),
                        solver_cache=fast_collision_solver_cache,
                    )
                    for slot_index, value in zip(target_indices, target_state):
                        projected[slot_index] = float(value)
                if damping_indices:
                    if damping_coefficient is None:
                        raise ValueError(
                            "Declared exact collision operator omitted a "
                            f"damping coefficient: {runtime.name}"
                        )
                    if not numpy.isfinite(damping_coefficient):
                        raise ValueError(
                            "Declared exact collision operator produced a "
                            f"non-finite damping coefficient: {runtime.name}"
                        )
                    if abs(float(damping_coefficient)) <= 1.0e-12:
                        raise ValueError(
                            "Declared exact collision operator has a zero "
                            f"damping coefficient: {runtime.name}"
                        )
                    damping_state = -forcing[list(damping_indices)] / (
                        float(collision_rate) * float(damping_coefficient)
                    )
                    for slot_index, value in zip(
                        damping_indices,
                        damping_state,
                    ):
                        projected[slot_index] = float(value)
                _validate_collision_invariants(
                    projected,
                    runtime=runtime,
                    step_index=step_index,
                    blend=blend,
                    k_value=float(k_value),
                )
            if not numpy.all(numpy.isfinite(projected)):
                raise ValueError(
                    "Declared fast collision projection produced non-finite "
                    "state values"
                )
            return projected

        def _constrained_mode_rhs(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            k_value: float,
            tight_coupling_active: bool,
        ) -> numpy.ndarray:
            """Evaluate the graph after the interval-boundary projection."""

            # The split collision half-steps and interval boundaries project
            # the state onto the declared fast manifold.  Re-projecting all
            # four RK stages would evaluate the full graph recursively and
            # duplicate the same expensive collision solve without improving
            # the declared operator-splitting order.
            return _mode_rhs(
                state_vector,
                step_index=step_index,
                blend=blend,
                k_value=float(k_value),
                tight_coupling_active=tight_coupling_active,
            )

        def _apply_split_collision_steps(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            dt: float,
            k_value: float,
            tight_coupling_active: bool,
            validate_invariants: bool = False,
        ) -> numpy.ndarray:
            """Return one state vector after the split collision sub-step.

            Conservation rules are evaluated at accepted interval boundaries,
            where the evolved history is retained, rather than at every
            provisional split sub-step.
            """

            if dt == 0.0:
                return numpy.asarray(state_vector, dtype=float)
            if not split_collision_runtimes:
                return numpy.asarray(state_vector, dtype=float)
            relaxed = numpy.asarray(state_vector, dtype=float).copy()
            for runtime in split_collision_runtimes:
                if tight_coupling_active and runtime.fast_manifold:
                    continue
                if runtime.activation_strategy == "tight_coupling":
                    if not tight_coupling_active:
                        continue
                metadata = _collision_metadata_for_state(
                    relaxed,
                    runtime=runtime,
                    step_index=step_index,
                    blend=blend,
                    k_value=float(k_value),
                )
                collision_rate, matrix, damping_coefficient = metadata
                if (
                    not numpy.isfinite(collision_rate)
                    or abs(collision_rate) <= 1.0e-12
                ):
                    continue
                if not numpy.all(numpy.isfinite(matrix)):
                    raise ValueError(
                        "Declared collision operator produced a non-finite "
                        f"matrix before evolution: {runtime.name}"
                    )
                collision_target_indices = runtime.target_slot_indices
                collision_matrix = matrix
                target_state = numpy.asarray(
                    [
                        float(relaxed[slot_index])
                        for slot_index in collision_target_indices
                    ],
                    dtype=float,
                )
                operator_matrix = collision_matrix
                if runtime.integration_strategy == "exact":
                    eigendecomposition = None
                    if (
                        state_independent_collision_runtimes.get(
                            runtime.name, False
                        )
                        and _structured_collision_components(matrix) is None
                    ):
                        eigendecomposition = (
                            _cached_collision_eigendecomposition(
                                matrix,
                                collision_eigendecomposition_cache,
                            )
                        )
                    evolved_state = _exact_linear_collision_step(
                        operator_matrix=operator_matrix,
                        dt=float(dt),
                        target_state=target_state,
                        eigendecomposition=eigendecomposition,
                        operator_scale=float(collision_rate),
                    )
                elif runtime.integration_strategy == "implicit":
                    evolved_state = numpy.linalg.solve(
                        numpy.eye(operator_matrix.shape[0], dtype=float)
                        - float(dt) * operator_matrix,
                        target_state,
                    )
                else:
                    raise ValueError(
                        "Declared collision operator reached an unsupported "
                        f"split strategy: {runtime.name}"
                    )
                if not numpy.all(numpy.isfinite(evolved_state)):
                    raise ValueError(
                        "Declared collision operator produced non-finite "
                        f"state updates: {runtime.name}"
                    )
                for slot_index, value in zip(
                    collision_target_indices,
                    evolved_state,
                ):
                    relaxed[slot_index] = float(value)
                if runtime.damping_slot_indices:
                    if runtime.damping_coefficient is None:
                        raise ValueError(
                            "Declared exact collision operator omitted a "
                            f"damping coefficient: {runtime.name}"
                        )
                    if damping_coefficient is None:
                        raise ValueError(
                            "Declared exact collision operator omitted a "
                            f"damping coefficient: {runtime.name}"
                        )
                    damping = math.exp(
                        float(collision_rate)
                        * float(damping_coefficient)
                        * float(dt)
                    )
                    for slot_index in runtime.damping_slot_indices:
                        relaxed[slot_index] *= damping
                if validate_invariants:
                    _validate_collision_invariants(
                        relaxed,
                        runtime=runtime,
                        step_index=step_index,
                        blend=blend,
                        k_value=float(k_value),
                    )
            return relaxed

        def _advance_declared_interval(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            dt: float,
            k_value: float,
            tight_coupling_active: bool,
        ) -> numpy.ndarray:
            """Advance one LOS interval with split streaming and collisions."""

            stiffness_scale = max(
                abs(float(k_value)),
                1.0e-12,
            )
            target_stage_scale = _phase_step_for_interval(
                step_index=step_index,
            )
            required_substeps = max(
                1,
                int(
                    math.ceil(
                        abs(float(dt)) * stiffness_scale / target_stage_scale
                    )
                ),
                1,
            )
            # Exact symmetric collision half-steps absorb the collision
            # stiffness.  Their magnitude must not force the explicit
            # streaming RK schedule into redundant microsteps after the
            # declared tight-coupling transition has ended.
            substep_count = 1
            while substep_count < required_substeps:
                substep_count *= 2
            max_substep_count = 65536
            failure_detail = "unspecified"
            while substep_count <= max_substep_count:
                trial_state = numpy.asarray(state_vector, dtype=float).copy()
                sub_dt = dt / float(substep_count)
                failed = False
                for substep_index in range(substep_count):
                    blend_start = substep_index / substep_count
                    blend_mid = (substep_index + 0.5) / substep_count
                    blend_end = (substep_index + 1.0) / substep_count
                    trial_state = _apply_split_collision_steps(
                        trial_state,
                        step_index=step_index,
                        blend=blend_start,
                        dt=0.5 * sub_dt,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    if not numpy.all(numpy.isfinite(trial_state)):
                        failure_detail = (
                            "exact collision sub-step start: "
                            f"{_describe_nonfinite_state(trial_state)}"
                        )
                        failed = True
                        break
                    stage_rhs_initial = _constrained_mode_rhs(
                        trial_state,
                        step_index=step_index,
                        blend=blend_start,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    stage_rhs_mid_a = _constrained_mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_initial,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    stage_rhs_mid_b = _constrained_mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_mid_a,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    stage_rhs_final = _constrained_mode_rhs(
                        trial_state + sub_dt * stage_rhs_mid_b,
                        step_index=step_index,
                        blend=blend_end,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    candidate_state = trial_state + (sub_dt / 6.0) * (
                        stage_rhs_initial
                        + 2.0 * stage_rhs_mid_a
                        + 2.0 * stage_rhs_mid_b
                        + stage_rhs_final
                    )
                    if not numpy.all(numpy.isfinite(candidate_state)):
                        failure_detail = (
                            "explicit sub-step: "
                            f"{_describe_nonfinite_state(candidate_state)}"
                        )
                        failed = True
                        break
                    candidate_state = _apply_split_collision_steps(
                        candidate_state,
                        step_index=step_index,
                        blend=blend_end,
                        dt=0.5 * sub_dt,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                        validate_invariants=(
                            substep_index == substep_count - 1
                        ),
                    )
                    candidate_state = _project_declared_fast_collision_state(
                        candidate_state,
                        step_index=step_index,
                        blend=blend_end,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    if not numpy.all(numpy.isfinite(candidate_state)):
                        failure_detail = (
                            "exact collision sub-step end: "
                            f"{_describe_nonfinite_state(candidate_state)}"
                        )
                        failed = True
                        break
                    trial_state = candidate_state
                if not failed:
                    return trial_state
                substep_count *= 2
            raise ValueError(
                "Declared CMB evolution produced non-finite state values "
                f"at k={k_value}, step_index={step_index}: "
                f"{failure_detail} "
                f"(required_substeps={required_substeps}, "
                f"last_substep_count={substep_count}, dt={dt}, "
                f"stiffness_scale={stiffness_scale})"
            )

        def _integrate_declared_state_history(
            initial_state: numpy.ndarray,
        ) -> tuple[dict[str, numpy.ndarray], numpy.ndarray]:
            """Return mode histories and the final state vector."""

            histories = {
                slot.variable: numpy.empty_like(
                    active_grids["eta"],
                    dtype=float,
                )
                for slot in runtime_spec.state_slots
                if slot.order == 0
            }
            state = numpy.asarray(initial_state, dtype=float).copy()
            continuous_collision_control = declared_accuracy_controls.get(
                "continuous_collision_solver"
            )
            if continuous_collision_control is not None and not isinstance(
                continuous_collision_control, bool
            ):
                raise ValueError(
                    "cmb.perturbations.accuracy_controls."
                    "continuous_collision_solver must be a boolean"
                )
            continuous_collision_solver = bool(
                (
                    continuous_collision_control
                    or (
                        diagnostic_source_audit
                        and not contract_or_params.get(
                            "_diagnostic_matrix_fast_path", False
                        )
                    )
                )
                and split_collision_runtimes
                and "massive_neutrino"
                not in set(manifest_summary.get("hierarchy_family_names", ()))
                and all(
                    runtime.integration_strategy in {"exact", "implicit"}
                    for runtime in split_collision_runtimes
                )
            )
            if not split_collision_runtimes or continuous_collision_solver:
                eta_values = numpy.asarray(active_grids["eta"], dtype=float)

                def _continuous_rhs(
                    eta_value: float,
                    state_vector: numpy.ndarray,
                ) -> numpy.ndarray:
                    """Evaluate the compiled graph on the continuous grid."""

                    right_index = int(
                        numpy.searchsorted(
                            eta_values,
                            float(eta_value),
                            side="right",
                        )
                    )
                    step_index = min(
                        max(right_index - 1, 0), eta_values.size - 2
                    )
                    left_eta = float(eta_values[step_index])
                    interval = float(eta_values[step_index + 1] - left_eta)
                    blend = numpy.clip(
                        (float(eta_value) - left_eta) / interval,
                        0.0,
                        1.0,
                    )
                    return _mode_rhs(
                        state_vector,
                        step_index=step_index,
                        blend=float(blend),
                        k_value=float(k_value),
                        tight_coupling_active=False,
                        include_split_collision_outputs=(
                            continuous_collision_solver
                        ),
                    )

                solution = solve_ivp(
                    _continuous_rhs,
                    (float(eta_values[0]), float(eta_values[-1])),
                    state,
                    method="BDF",
                    t_eval=eta_values,
                    rtol=float(numerics.ode_rtol),
                    atol=float(numerics.ode_atol),
                )
                if not solution.success:
                    raise ValueError(
                        "Declared CMB continuous evolution failed: "
                        f"{solution.message}"
                    )
                if solution.y.shape[1] != eta_values.size:
                    raise ValueError(
                        "Declared CMB continuous evolution returned an "
                        "incomplete state history"
                    )
                histories = {
                    slot.variable: numpy.asarray(
                        solution.y[slot.index],
                        dtype=float,
                    )
                    for slot in runtime_spec.state_slots
                    if slot.order == 0
                }
                final_state = numpy.asarray(solution.y[:, -1], dtype=float)
                if not numpy.all(numpy.isfinite(final_state)):
                    raise ValueError(
                        "Declared CMB continuous evolution produced "
                        "non-finite state values"
                    )
                return histories, final_state
            tight_coupling_active = _tight_coupling_is_active(
                active=False,
                collision_rate=float(active_grids["collision_rate"][0]),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
                exit_ratio=float(numerics.tight_coupling_exit_ratio),
            )
            for step_index, eta_value in enumerate(active_grids["eta"]):
                state = _project_declared_fast_collision_state(
                    state,
                    step_index=step_index,
                    blend=0.0,
                    k_value=float(k_value),
                    tight_coupling_active=tight_coupling_active,
                )
                for slot in runtime_spec.state_slots:
                    if slot.order != 0:
                        continue
                    histories[slot.variable][step_index] = state[slot.index]
                if step_index == active_grids["eta"].size - 1:
                    break
                dt = float(active_grids["eta"][step_index + 1] - eta_value)
                state = _advance_declared_interval(
                    state,
                    step_index=step_index,
                    dt=dt,
                    k_value=float(k_value),
                    tight_coupling_active=tight_coupling_active,
                )
                end_collision_rate = float(
                    active_grids["collision_rate"][step_index + 1]
                )
                tight_coupling_active = _tight_coupling_is_active(
                    active=tight_coupling_active,
                    collision_rate=end_collision_rate,
                    k_value=float(k_value),
                    tight_coupling_ratio=float(numerics.tight_coupling_ratio),
                    exit_ratio=float(numerics.tight_coupling_exit_ratio),
                )
            return histories, state

        def _evaluate_end_boundary_residuals(
            final_state: numpy.ndarray,
        ) -> numpy.ndarray:
            """Return end-boundary residuals for one integrated mode."""

            if not end_boundary_entries:
                return numpy.zeros(0, dtype=float)
            final_eta, final_background = _scalar_background_context(
                active_grids["eta"].size - 1,
                0.0,
            )
            final_context = _build_scalar_state_context(
                final_state,
                k_value=float(k_value),
                eta_value=float(final_eta),
                background_scalars=final_background,
            )
            residuals = []
            for entry in end_boundary_entries:
                state_index = runtime_spec.state_index_by_key[
                    (
                        str(entry.target.variable),
                        str(entry.target.wrt),
                        int(entry.target.order),
                    )
                ]
                expected_value = _coerce_numeric_scalar(
                    evaluate_compiled_expression(
                        entry.compiled_expression,
                        final_context,
                    ),
                    name=f"end boundary '{entry.name}'",
                )
                residuals.append(
                    float(final_state[state_index]) - float(expected_value)
                )
            return numpy.asarray(residuals, dtype=float)

        with performance_timer.phase("initial_data"):
            state, assigned_targets, _ = _prepare_mode_initial_state(
                float(k_value)
            )
        if end_boundary_entries:
            assigned_target_set = set(assigned_targets)
            free_target_keys = tuple(
                sorted(
                    (
                        slot.variable,
                        slot.wrt,
                        slot.order,
                    )
                    for slot in runtime_spec.state_slots
                    if (
                        slot.variable,
                        slot.wrt,
                        slot.order,
                    )
                    not in assigned_target_set
                )
            )
            end_target_keys = tuple(
                sorted(
                    (
                        str(entry.target.variable),
                        str(entry.target.wrt),
                        int(entry.target.order),
                    )
                    for entry in end_boundary_entries
                )
            )
            if free_target_keys != end_target_keys:
                raise ValueError(
                    "Declared end boundary solver requires end anchors to "
                    "replace exactly the missing start-state slots."
                )
            free_indices = numpy.asarray(
                [
                    runtime_spec.state_index_by_key[target_key]
                    for target_key in free_target_keys
                ],
                dtype=int,
            )
            initial_guess_context = _build_scalar_state_context(
                state,
                k_value=float(k_value),
                eta_value=float(initial_eta),
                background_scalars=initial_background,
            )
            boundary_guess = []
            for entry in end_boundary_entries:
                try:
                    boundary_guess.append(
                        _coerce_numeric_scalar(
                            evaluate_compiled_expression(
                                entry.compiled_expression,
                                initial_guess_context,
                            ),
                            name=f"end boundary '{entry.name}' guess",
                        )
                    )
                except ValueError:
                    boundary_guess.append(
                        float(
                            state[
                                runtime_spec.state_index_by_key[
                                    (
                                        str(entry.target.variable),
                                        str(entry.target.wrt),
                                        int(entry.target.order),
                                    )
                                ]
                            ]
                        )
                    )

            def _boundary_objective(
                unknown_values: numpy.ndarray,
            ) -> numpy.ndarray:
                """Return end-boundary residuals for one shooting guess."""

                trial_state = numpy.asarray(state, dtype=float).copy()
                trial_state[free_indices] = numpy.asarray(
                    unknown_values,
                    dtype=float,
                )
                _, final_state = _integrate_declared_state_history(trial_state)
                return _evaluate_end_boundary_residuals(final_state)

            boundary_solution = least_squares(
                _boundary_objective,
                numpy.asarray(boundary_guess, dtype=float),
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
            residual_scale = max(float(numerics.ode_atol) * 50.0, 1.0e-8)
            final_residuals = numpy.asarray(
                boundary_solution.fun,
                dtype=float,
            )
            if (
                not boundary_solution.success
                or not numpy.all(numpy.isfinite(boundary_solution.x))
                or not numpy.all(numpy.isfinite(final_residuals))
                or numpy.max(numpy.abs(final_residuals), initial=0.0)
                > residual_scale
            ):
                message = str(getattr(boundary_solution, "message", "unknown"))
                raise ValueError(
                    "Declared end boundary solver failed to converge: "
                    f"{message}"
                )
            state[free_indices] = numpy.asarray(
                boundary_solution.x,
                dtype=float,
            )
        histories, final_state = _integrate_declared_state_history(state)
        if diagnostic_source_audit and collect_diagnostics:
            # Audit the solver's native evolution grid before any source-grid
            # interpolation or optional constraint reconstruction.
            _record_hierarchy_equation_residuals(
                float(k_value),
                histories,
            )
        final_residuals = _evaluate_end_boundary_residuals(final_state)
        if final_residuals.size and numpy.max(
            numpy.abs(final_residuals), initial=0.0
        ) > max(float(numerics.ode_atol) * 50.0, 1.0e-8):
            raise ValueError(
                "Declared end boundary conditions remained unsatisfied "
                "after integration."
            )
        source_histories = histories
        if history_sink is not None:
            history_sink["evolution_eta"] = numpy.asarray(
                active_grids["eta"],
                dtype=float,
            ).copy()
            history_sink["evolution_histories"] = {
                name: numpy.asarray(history, dtype=float).copy()
                for name, history in histories.items()
            }
        if active_grids["eta"].shape != source_grids[
            "eta"
        ].shape or not numpy.array_equal(
            active_grids["eta"],
            source_grids["eta"],
        ):
            source_histories = {
                name: numpy.asarray(
                    numpy.interp(
                        source_grids["eta"],
                        active_grids["eta"],
                        history,
                    ),
                    dtype=float,
                )
                for name, history in histories.items()
            }
        source_arrays = _evaluate_source_histories(
            float(k_value),
            source_histories,
            collect_diagnostics=collect_diagnostics,
            required_source_names=required_source_names,
        )
        if history_sink is not None:
            history_sink["source_eta"] = numpy.asarray(
                source_grids["eta"],
                dtype=float,
            ).copy()
            history_sink["source_histories"] = {
                name: numpy.asarray(history, dtype=float).copy()
                for name, history in source_histories.items()
            }
        state_history_max_abs_by_k[f"{float(k_value):.12g}"] = {
            name: float(numpy.max(numpy.abs(history), initial=0.0))
            for name, history in source_histories.items()
            if name
            in {
                "theta_gamma0",
                "theta_gamma1",
                "theta_gamma2",
                "e_gamma2",
                "e_gamma3",
                "theta_b",
                "delta_b",
                "delta_c",
                "delta_nu",
                "sigma_nu",
                "Phi",
                "Psi",
            }
        }
        if {"theta_gamma2", "e_gamma2"}.issubset(source_histories):
            visibility = numpy.asarray(source_grids["visibility"], dtype=float)
            active_visibility = visibility >= 0.1 * float(
                numpy.max(visibility, initial=0.0)
            )
            if numpy.any(active_visibility):
                theta_values = numpy.asarray(
                    source_histories["theta_gamma2"], dtype=float
                )[active_visibility]
                e_values = numpy.asarray(
                    source_histories["e_gamma2"], dtype=float
                )[active_visibility]
                state_history_polarization_ratio_by_k[
                    f"{float(k_value):.12g}"
                ] = {
                    "maximum_abs_e_over_theta": float(
                        numpy.max(
                            numpy.abs(e_values)
                            / numpy.maximum(numpy.abs(theta_values), 1.0e-30),
                            initial=0.0,
                        )
                    ),
                    "maximum_abs_theta": float(
                        numpy.max(numpy.abs(theta_values), initial=0.0)
                    ),
                    "maximum_abs_e": float(
                        numpy.max(numpy.abs(e_values), initial=0.0)
                    ),
                }
        return source_histories, source_arrays

    def _evolve_declared_modes_batched(
        mode_k_values: numpy.ndarray,
    ) -> dict[int, dict[str, numpy.ndarray]]:
        """Evolve compatible declared modes through one shared RK schedule.

        The compiled declaration remains the numerical authority: the batch
        transposes the state layout so the existing equation program evaluates
        every independent Fourier mode in one NumPy operation.  Contracts
        outside that capability retain the scalar evolution path.
        """

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal active_k_value
        nonlocal scalar_base_context_cache
        nonlocal scalar_background_context_cache

        continuous_collision_control = declared_accuracy_controls.get(
            "continuous_collision_solver"
        )
        if continuous_collision_control is not None and not isinstance(
            continuous_collision_control,
            bool,
        ):
            raise ValueError(
                "cmb.perturbations.accuracy_controls."
                "continuous_collision_solver must be a boolean"
            )
        continuous_collision_solver = bool(
            continuous_collision_control
            and split_collision_runtimes
            and "massive_neutrino"
            not in set(manifest_summary.get("hierarchy_family_names", ()))
            and all(
                runtime.integration_strategy in {"exact", "implicit"}
                for runtime in split_collision_runtimes
            )
        )
        k_values_batch = numpy.asarray(mode_k_values, dtype=float)
        if k_values_batch.ndim != 1 or not numpy.all(
            numpy.isfinite(k_values_batch)
        ):
            raise ValueError("Batched CMB evolution requires finite k modes")
        if not _can_batch_declared_evolution(
            generated_scalar_hierarchy=generated_scalar_hierarchy,
            shared_mode_grids_enabled=shared_generated_mode_grids_enabled,
            mode_count=int(k_values_batch.size),
            has_momentum_runtimes=bool(momentum_runtimes),
            has_end_boundaries=bool(execution_plan.end_condition_entries),
            adaptive_evolution_enabled=adaptive_controls.evolution_enabled,
            adaptive_source_enabled=adaptive_controls.source_enabled,
            adaptive_transfer_enabled=adaptive_controls.transfer_enabled,
            adaptive_projection_enabled=adaptive_controls.projection_enabled,
            adaptive_k_enabled=adaptive_k_enabled,
            continuous_collision_solver=continuous_collision_solver,
            has_declared_collision_operators=bool(
                getattr(perturbation_data, "collision_operators", {})
            ),
            state_slots=runtime_spec.state_slots,
            collision_runtimes=split_collision_runtimes,
        ):
            return {}
        if not all(
            state_independent_collision_runtimes.get(runtime.name, False)
            for runtime in split_collision_runtimes
        ):
            return {}

        grouped_modes: dict[bytes, dict[str, Any]] = {}
        for mode_index, mode_k_value in enumerate(k_values_batch):
            mode_grids = _mode_grids_for_k(float(mode_k_value))
            eta_values = numpy.asarray(mode_grids[0]["eta"], dtype=float)
            group = grouped_modes.setdefault(
                eta_values.tobytes(),
                {
                    "indices": [],
                    "k_values": [],
                    "grids": mode_grids,
                },
            )
            group["indices"].append(int(mode_index))
            group["k_values"].append(float(mode_k_value))

        results: dict[int, dict[str, numpy.ndarray]] = {}
        for group in grouped_modes.values():
            if len(group["indices"]) < 2:
                continue
            (
                active_grids,
                active_declared_background_histories,
                active_coordinate_rate_histories,
            ) = group["grids"]
            active_k_value = float(group["k_values"][0])
            scalar_base_context_cache = {}
            scalar_background_context_cache = {}
            local_k_values = numpy.asarray(group["k_values"], dtype=float)
            mode_count = int(local_k_values.size)
            initial_states = []
            for mode_k_value in local_k_values:
                initial_state, _, _ = _prepare_mode_initial_state(
                    float(mode_k_value)
                )
                initial_states.append(initial_state)
            states = numpy.asarray(initial_states, dtype=float)
            if states.ndim != 2 or states.shape[0] != mode_count:
                raise ValueError("Batched CMB initial states have wrong shape")
            validate_batch_collision_invariants = bool(
                diagnostic_source_audit or mode_count <= 128
            )
            last_record_active: numpy.ndarray | None = None

            base_context_cache: dict[tuple[int, float], dict[str, Any]] = {}
            momentum_grid_context_cache: dict[float, dict[str, Any]] = {}
            collision_metadata_cache: dict[
                tuple[str, int, float],
                tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None],
            ] = {}
            fast_collision_solver_cache: dict[str, tuple[Any, ...]] = {}
            batched_suppressed_collision_outputs = {
                output_name: numpy.zeros(mode_count, dtype=float)
                for runtime in split_collision_runtimes
                for output_name in (runtime.name, runtime.counterpart)
                if output_name is not None
            }
            collision_expression_programs = {
                runtime.name: _compile_expression_tuple_program(
                    (
                        str(runtime.rate_expression.expression),
                        *(
                            str(entry.expression)
                            for matrix_row in runtime.matrix
                            for entry in matrix_row
                        ),
                        *(
                            ()
                            if runtime.damping_coefficient is None
                            else (str(runtime.damping_coefficient.expression),)
                        ),
                    )
                )
                for runtime in split_collision_runtimes
            }
            batch_static_context: dict[str, Any] = dict(source_parameters)
            for name, value in _physical_runtime_scalars(
                physical_params
            ).items():
                batch_static_context.setdefault(name, float(value))
            batch_static_context["tight_coupling_ratio"] = float(
                numerics.tight_coupling_ratio
            )
            batch_seed_values = numpy.asarray(
                [
                    _declared_runtime_seed(
                        k_value=float(mode_k_value),
                        physical_params=physical_params,
                        model_parameters=source_parameters,
                    )
                    for mode_k_value in local_k_values
                ],
                dtype=float,
            )
            batch_coordinate_rates = {
                str(slot_plan.wrt): 1.0
                for slot_plan in execution_plan.equation_slot_plans
            }
            momentum_context_by_stage: dict[
                tuple[int, float], dict[str, Any]
            ] = {}
            batched_row_program: Any | None = None
            batched_row_vector_names: tuple[str, ...] | None = None

            def _batch_base_context(
                step_index: int,
                blend: float,
            ) -> dict[str, Any]:
                """Return the vectorized state-independent graph context."""

                cache_key = (int(step_index), float(blend))
                cached = base_context_cache.get(cache_key)
                if cached is not None:
                    return cached
                eta_value, background_scalars = _scalar_background_context(
                    int(step_index),
                    float(blend),
                    k_value=float(local_k_values[0]),
                )
                context = dict(batch_static_context)
                context.update(background_scalars)
                context["k"] = local_k_values
                context["seed"] = batch_seed_values
                context["a_initial"] = float(background_scalars["a"])
                context["eta_initial"] = float(eta_value)
                context["sound_horizon"] = float(
                    background_scalars["sound_horizon"]
                )
                context["sound_speed_sq"] = float(
                    background_scalars["sound_speed_sq"]
                )
                context["collision_rate"] = float(
                    background_scalars["collision_rate"]
                )
                context["free_streaming"] = float(
                    background_scalars["free_streaming"]
                )
                if momentum_runtimes:
                    momentum_context = momentum_context_by_stage.get(cache_key)
                    if momentum_context is None:
                        scale_factor = float(background_scalars["a"])
                        momentum_context = momentum_grid_context_cache.get(
                            scale_factor
                        )
                        if momentum_context is None:
                            momentum_context = _declared_momentum_grid_context(
                                perturbation_data,
                                model_parameters=source_parameters,
                                physical_params=physical_params,
                                scale_factor=scale_factor,
                            )
                            momentum_grid_context_cache[scale_factor] = (
                                momentum_context
                            )
                    context.update(momentum_context)
                collision_rate = float(context["collision_rate"])
                coupling_cap = numpy.maximum(
                    local_k_values * float(numerics.tight_coupling_ratio),
                    1.0e-12,
                )
                context["tight_coupling_drag"] = collision_rate / (
                    1.0 + collision_rate / coupling_cap
                )
                cached = _resolve_declared_graph_context_ordered(
                    context,
                    perturbation_data,
                    allow_partial=True,
                    eta_grid=None,
                    execution_plan=execution_plan,
                    value_steps=state_independent_value_steps,
                    use_compiled_program=True,
                    compiled_value_program=state_independent_context_program,
                )
                base_context_cache[cache_key] = cached
                return cached

            def _batched_state_context(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                suppress_split_collision: bool = True,
                include_diagnostics: bool = False,
            ) -> dict[str, Any]:
                """Bind all state rows into one vector-valued graph context."""

                context = dict(_batch_base_context(step_index, blend))
                state_columns = numpy.asarray(state_rows, dtype=float).T
                for slot in runtime_spec.state_slots:
                    if slot.order == 0:
                        name = slot.variable
                    else:
                        name = f"__d{slot.order}_{slot.variable}_{slot.wrt}"
                    context[name] = state_columns[slot.index]
                suppressed_outputs = (
                    batched_suppressed_collision_outputs
                    if suppress_split_collision
                    else {}
                )
                return _resolve_declared_graph_context_ordered(
                    context,
                    perturbation_data,
                    allow_partial=True,
                    eta_grid=None,
                    execution_plan=execution_plan,
                    derivative_steps=(
                        stage_derivative_steps
                        if include_diagnostics
                        else equation_stage_derivative_steps
                    ),
                    value_steps=(
                        state_dependent_value_steps
                        if include_diagnostics
                        else batched_rhs_value_steps
                    ),
                    suppressed_outputs=suppressed_outputs,
                    use_compiled_program=True,
                    compiled_value_program=(
                        state_dependent_context_program
                        if include_diagnostics
                        else batched_rhs_context_program
                    ),
                )

            def _batch_rhs(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                active: numpy.ndarray,
            ) -> numpy.ndarray:
                """Evaluate declared mode derivatives across all batch rows."""

                del active
                context = _batched_state_context(
                    state_rows,
                    step_index=step_index,
                    blend=blend,
                )
                state_columns = numpy.asarray(state_rows, dtype=float).T
                derivative_columns = numpy.zeros_like(
                    state_columns,
                    dtype=float,
                )
                equation_executor = equation_program
                if mode_count <= 5:
                    vector_names = tuple(
                        name
                        for name in sorted(equation_direct_names)
                        if numpy.asarray(context[name]).shape == (mode_count,)
                    )
                    nonlocal batched_row_program
                    nonlocal batched_row_vector_names
                    if batched_row_vector_names is None:
                        batched_row_vector_names = vector_names
                        batched_row_program = (
                            _compile_batched_row_equation_program(
                                equation_program_specs,
                                vector_names,
                            )
                        )
                    if vector_names == batched_row_vector_names:
                        equation_executor = batched_row_program
                with numpy.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                ):
                    try:
                        equation_executor(
                            context,
                            state_columns,
                            derivative_columns,
                            batch_coordinate_rates,
                        )
                    except (KeyError, NameError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "Declared batched CMB equation program failed "
                            f"at step_index={step_index}"
                        ) from exc
                derivative = derivative_columns.T
                if not numpy.all(numpy.isfinite(derivative)):
                    bad = numpy.argwhere(~numpy.isfinite(derivative))[0]
                    raise ValueError(
                        "Declared batched CMB evolution produced a "
                        "non-finite derivative at "
                        f"mode_index={int(bad[0])}, "
                        f"state_index={int(bad[1])}"
                    )
                return derivative

            def _batch_collision_metadata(
                *,
                runtime: _CompiledCollisionOperatorRuntime,
                step_index: int,
                blend: float,
            ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None]:
                """Resolve one declared collision matrix for every mode row."""

                metadata_key = (
                    runtime.name,
                    int(step_index),
                    float(blend),
                )
                cached = collision_metadata_cache.get(metadata_key)
                if cached is not None:
                    return cached
                context = _batch_base_context(step_index, blend)
                expression_values = collision_expression_programs[
                    runtime.name
                ](context)

                def _mode_values(value: Any, *, name: str) -> numpy.ndarray:
                    """Return one scalar declaration value per mode row."""

                    values = numpy.asarray(value, dtype=float)
                    if values.ndim == 0:
                        values = numpy.full(
                            mode_count,
                            float(values),
                            dtype=float,
                        )
                    if values.shape != (mode_count,):
                        raise ValueError(
                            "Declared batched collision value has the "
                            f"wrong shape for {name}: {values.shape}"
                        )
                    return values

                collision_rate = _mode_values(
                    expression_values[0],
                    name=f"collision operator '{runtime.name}' rate",
                )
                matrix = numpy.empty(
                    (
                        mode_count,
                        len(runtime.matrix),
                        len(runtime.matrix[0]),
                    ),
                    dtype=float,
                )
                value_index = 1
                for row_index, matrix_row in enumerate(runtime.matrix):
                    for column_index, _entry in enumerate(matrix_row):
                        matrix[:, row_index, column_index] = _mode_values(
                            expression_values[value_index],
                            name=(
                                "collision operator "
                                f"'{runtime.name}' matrix entry"
                            ),
                        )
                        value_index += 1
                damping_coefficient = None
                if runtime.damping_coefficient is not None:
                    damping_coefficient = _mode_values(
                        expression_values[value_index],
                        name=(
                            "collision operator "
                            f"'{runtime.name}' damping coefficient"
                        ),
                    )
                if not numpy.all(numpy.isfinite(collision_rate)):
                    raise ValueError(
                        "Declared batched collision value is non-finite for "
                        f"collision operator '{runtime.name}' rate"
                    )
                if not numpy.all(numpy.isfinite(matrix)):
                    raise ValueError(
                        "Declared batched collision value is non-finite for "
                        f"collision operator '{runtime.name}' matrix"
                    )
                if damping_coefficient is not None and not numpy.all(
                    numpy.isfinite(damping_coefficient)
                ):
                    raise ValueError(
                        "Declared batched collision value is non-finite for "
                        f"collision operator '{runtime.name}' damping "
                        "coefficient"
                    )
                metadata = (collision_rate, matrix, damping_coefficient)
                collision_metadata_cache[metadata_key] = metadata
                return metadata

            def _validate_batch_collision_invariants(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
            ) -> None:
                """Validate all declared collision rules in one graph pass."""

                rule_names = tuple(
                    sorted(
                        {
                            rule_name
                            for runtime in split_collision_runtimes
                            for rule_name in runtime.conservation_rule_names
                        }
                    )
                )
                if not rule_names:
                    return
                context = _batched_state_context(
                    state_rows,
                    step_index=step_index,
                    blend=blend,
                    suppress_split_collision=False,
                    include_diagnostics=True,
                )
                _validate_declared_conservation_rules(
                    perturbation_data=perturbation_data,
                    context=context,
                    k_value=float(local_k_values[0]),
                    rule_names=rule_names,
                )

            def _apply_split_collision_steps(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                dt: float,
                active: numpy.ndarray,
            ) -> numpy.ndarray:
                """Apply the exact declared collision half-step per row."""

                if dt == 0.0 or not split_collision_runtimes:
                    return numpy.asarray(state_rows, dtype=float)
                relaxed = numpy.asarray(state_rows, dtype=float).copy()
                for runtime in split_collision_runtimes:
                    collision_rates, matrices, damping_coefficients = (
                        _batch_collision_metadata(
                            runtime=runtime,
                            step_index=step_index,
                            blend=blend,
                        )
                    )
                    active_rows = numpy.asarray(active, dtype=bool)
                    apply_rows = numpy.flatnonzero(
                        ~(active_rows & bool(runtime.fast_manifold))
                    )
                    if apply_rows.size == 0:
                        continue
                    nonzero_rows = apply_rows[
                        numpy.abs(collision_rates[apply_rows]) > 1.0e-12
                    ]
                    if nonzero_rows.size == 0:
                        continue
                    target_indices = numpy.asarray(
                        runtime.target_slot_indices,
                        dtype=int,
                    )
                    target_states = relaxed[
                        numpy.ix_(nonzero_rows, target_indices)
                    ]
                    if runtime.integration_strategy == "exact":
                        evolved_states = _exact_batched_linear_collision_step(
                            operator_matrices=matrices[nonzero_rows],
                            dt=float(dt),
                            target_states=target_states,
                            operator_scales=collision_rates[nonzero_rows],
                            assume_block_diagonal=(
                                target_indices.size == 4
                                and numpy.all(
                                    matrices[nonzero_rows, :2, 2:] == 0.0
                                )
                                and numpy.all(
                                    matrices[nonzero_rows, 2:, :2] == 0.0
                                )
                            ),
                        )
                    elif runtime.integration_strategy == "implicit":
                        operator = (
                            numpy.eye(target_indices.size, dtype=float)[
                                numpy.newaxis, :, :
                            ]
                            - float(dt) * matrices[nonzero_rows]
                        )
                        evolved_states = numpy.linalg.solve(
                            operator,
                            target_states[:, :, numpy.newaxis],
                        )[:, :, 0]
                    else:
                        raise ValueError(
                            "Declared collision operator reached an "
                            "unsupported split strategy: "
                            f"{runtime.name}"
                        )
                    if not numpy.all(numpy.isfinite(evolved_states)):
                        raise ValueError(
                            "Declared collision operator produced non-finite "
                            f"batched state updates: {runtime.name}"
                        )
                    relaxed[numpy.ix_(nonzero_rows, target_indices)] = (
                        evolved_states
                    )
                    if runtime.damping_slot_indices:
                        if damping_coefficients is None:
                            raise ValueError(
                                "Declared exact collision operator omitted a "
                                f"damping coefficient: {runtime.name}"
                            )
                        damping_indices = numpy.asarray(
                            runtime.damping_slot_indices,
                            dtype=int,
                        )
                        damping = numpy.exp(
                            collision_rates[nonzero_rows]
                            * damping_coefficients[nonzero_rows]
                            * float(dt)
                        )
                        damping_target = relaxed[
                            numpy.ix_(nonzero_rows, damping_indices)
                        ]
                        damping_target *= damping[:, numpy.newaxis]
                return relaxed

            def _project_fast_collision_state(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                active: numpy.ndarray,
            ) -> numpy.ndarray:
                """Project active rows onto declared fast collision manifolds.

                The projection restores each active row's declared constraint.
                """

                if not numpy.any(active) or not split_collision_runtimes:
                    return numpy.asarray(state_rows, dtype=float)
                projected = numpy.asarray(state_rows, dtype=float).copy()
                for runtime in split_collision_runtimes:
                    if (
                        not runtime.fast_manifold
                        or runtime.integration_strategy != "exact"
                    ):
                        continue
                    collision_rates, matrices, damping_coefficients = (
                        _batch_collision_metadata(
                            runtime=runtime,
                            step_index=step_index,
                            blend=blend,
                        )
                    )
                    forcing = _batch_rhs(
                        projected,
                        step_index=step_index,
                        blend=blend,
                        active=active,
                    )
                    target_indices = tuple(runtime.target_slot_indices)
                    damping_indices = tuple(
                        index
                        for index in runtime.damping_slot_indices
                        if index not in target_indices
                    )
                    active_rows = numpy.flatnonzero(active)
                    valid_rows = active_rows[
                        numpy.isfinite(collision_rates[active_rows])
                        & (collision_rates[active_rows] > 1.0e-12)
                    ]
                    if valid_rows.size == 0:
                        continue
                    if not numpy.all(numpy.isfinite(matrices[valid_rows])):
                        raise ValueError(
                            "Declared collision operator produced a "
                            "non-finite matrix before batched fast "
                            f"projection: {runtime.name}"
                        )
                    if target_indices:
                        target_states = projected[
                            numpy.ix_(valid_rows, target_indices)
                        ]
                        target_forcing = forcing[
                            numpy.ix_(valid_rows, target_indices)
                        ]
                        target_matrices = matrices[valid_rows]
                        target_rates = collision_rates[valid_rows]
                        target_state = (
                            _solve_batched_small_declared_collision_target(
                                target_matrices,
                                target_forcing,
                                target_states,
                                target_rates,
                            )
                        )
                        if target_state is None:
                            target_state = numpy.vstack(
                                [
                                    _solve_declared_fast_collision_target(
                                        target_matrices[row_index],
                                        target_forcing[row_index],
                                        target_states[row_index],
                                        float(target_rates[row_index]),
                                        solver_cache=(
                                            fast_collision_solver_cache
                                        ),
                                    )
                                    for row_index in range(valid_rows.size)
                                ]
                            )
                        projected[numpy.ix_(valid_rows, target_indices)] = (
                            target_state
                        )
                    if damping_indices:
                        if damping_coefficients is None:
                            raise ValueError(
                                "Declared exact collision operator omitted a "
                                f"damping coefficient: {runtime.name}"
                            )
                        damping_values = damping_coefficients[valid_rows]
                        if numpy.any(
                            ~numpy.isfinite(damping_values)
                            | (numpy.abs(damping_values) <= 1.0e-12)
                        ):
                            raise ValueError(
                                "Declared exact collision operator has an "
                                f"invalid damping coefficient: {runtime.name}"
                            )
                        damping_selector = numpy.ix_(
                            valid_rows,
                            damping_indices,
                        )
                        damping_forcing = forcing[damping_selector]
                        damping_rates = collision_rates[
                            valid_rows,
                            numpy.newaxis,
                        ]
                        projected[damping_selector] = -damping_forcing / (
                            damping_rates * damping_values[:, numpy.newaxis]
                        )
                if validate_batch_collision_invariants:
                    _validate_batch_collision_invariants(
                        projected,
                        step_index=step_index,
                        blend=blend,
                    )
                if not numpy.all(numpy.isfinite(projected)):
                    raise ValueError(
                        "Declared batched fast collision projection produced "
                        "non-finite state values"
                    )
                return projected

            def _batch_pre_step(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                dt: float,
                active: numpy.ndarray,
            ) -> numpy.ndarray:
                """Apply one initial Strang-split collision half-step."""

                return _apply_split_collision_steps(
                    state_rows,
                    step_index=step_index,
                    blend=blend,
                    dt=dt,
                    active=active,
                )

            def _batch_post_step(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                dt: float,
                active: numpy.ndarray,
            ) -> numpy.ndarray:
                """Finish a split step and restore fast collision constraints.

                The post-step path returns the state to the fast manifold.
                """

                relaxed = _apply_split_collision_steps(
                    state_rows,
                    step_index=step_index,
                    blend=blend,
                    dt=dt,
                    active=active,
                )
                return _project_fast_collision_state(
                    relaxed,
                    step_index=step_index,
                    blend=blend,
                    active=active,
                )

            def _batch_record_step(
                state_rows: numpy.ndarray,
                *,
                step_index: int,
                blend: float,
                active: numpy.ndarray,
            ) -> numpy.ndarray:
                """Record a finite grid state after its fast projection."""

                nonlocal last_record_active
                active_array = numpy.asarray(active, dtype=bool)
                active_changed = (
                    last_record_active is None
                    or not numpy.array_equal(active_array, last_record_active)
                )
                if active_changed:
                    projected = _project_fast_collision_state(
                        state_rows,
                        step_index=step_index,
                        blend=blend,
                        active=active_array,
                    )
                else:
                    projected = numpy.asarray(state_rows, dtype=float)
                last_record_active = active_array.copy()
                if validate_batch_collision_invariants:
                    _validate_batch_collision_invariants(
                        projected,
                        step_index=step_index,
                        blend=blend,
                    )
                return projected

            eta_values = numpy.asarray(active_grids["eta"], dtype=float)
            interval_count = max(int(eta_values.size) - 1, 0)
            active_intervals = numpy.zeros(
                (mode_count, interval_count),
                dtype=bool,
            )
            for row_index, mode_k_value in enumerate(local_k_values):
                active_mode = _tight_coupling_is_active(
                    active=False,
                    collision_rate=float(active_grids["collision_rate"][0]),
                    k_value=float(mode_k_value),
                    tight_coupling_ratio=float(numerics.tight_coupling_ratio),
                    exit_ratio=float(numerics.tight_coupling_exit_ratio),
                )
                for step_index in range(interval_count):
                    active_intervals[row_index, step_index] = active_mode
                    active_mode = _tight_coupling_is_active(
                        active=active_mode,
                        collision_rate=float(
                            active_grids["collision_rate"][step_index + 1]
                        ),
                        k_value=float(mode_k_value),
                        tight_coupling_ratio=float(
                            numerics.tight_coupling_ratio
                        ),
                        exit_ratio=float(numerics.tight_coupling_exit_ratio),
                    )
            dt_values = numpy.diff(eta_values)
            phase_step_values = numpy.asarray(
                [
                    _phase_step_for_interval(step_index=step_index)
                    for step_index in range(interval_count)
                ],
                dtype=float,
            )
            required_substeps = numpy.maximum(
                1,
                numpy.ceil(
                    numpy.abs(dt_values)[numpy.newaxis, :]
                    * numpy.abs(local_k_values)[:, numpy.newaxis]
                    / phase_step_values[numpy.newaxis, :]
                ).astype(int),
            )
            scalar_substeps = numpy.ones_like(required_substeps)
            for row_index in range(mode_count):
                for step_index in range(interval_count):
                    requested = max(
                        int(required_substeps[row_index, step_index]),
                        1,
                    )
                    substep_count = 1
                    while substep_count < requested:
                        substep_count *= 2
                    scalar_substeps[row_index, step_index] = substep_count
            batch_substeps = numpy.ones(interval_count, dtype=int)
            for step_index in range(interval_count):
                requested = max(
                    int(numpy.max(required_substeps[:, step_index])),
                    1,
                )
                substep_count = 1
                while substep_count < requested:
                    substep_count *= 2
                batch_substeps[step_index] = substep_count
            if momentum_runtimes:
                stage_keys: list[tuple[int, float]] = [(0, 0.0)]
                seen_stage_keys = {(0, 0.0)}
                for step_index in range(interval_count):
                    substep_count = int(batch_substeps[step_index])
                    for substep_index in range(substep_count):
                        for blend in (
                            substep_index / float(substep_count),
                            (substep_index + 0.5) / float(substep_count),
                            (substep_index + 1.0) / float(substep_count),
                        ):
                            stage_key = (int(step_index), float(blend))
                            if stage_key not in seen_stage_keys:
                                seen_stage_keys.add(stage_key)
                                stage_keys.append(stage_key)
                    record_key = (int(step_index + 1), 0.0)
                    if record_key not in seen_stage_keys:
                        seen_stage_keys.add(record_key)
                        stage_keys.append(record_key)
                stage_scales = numpy.asarray(
                    [
                        float(
                            _scalar_background_context(
                                step_index,
                                blend,
                                k_value=float(local_k_values[0]),
                            )[1]["a"]
                        )
                        for step_index, blend in stage_keys
                    ],
                    dtype=float,
                )
                batch_momentum_context = _declared_momentum_grid_context(
                    perturbation_data,
                    model_parameters=source_parameters,
                    physical_params=physical_params,
                    scale_factor=stage_scales,
                )
                stage_count = len(stage_keys)
                for stage_index, stage_key in enumerate(stage_keys):
                    stage_context: dict[str, Any] = {}
                    for name, value in batch_momentum_context.items():
                        array_value = numpy.asarray(value)
                        if (
                            array_value.ndim > 0
                            and array_value.shape[0] == stage_count
                        ):
                            stage_context[name] = array_value[stage_index]
                        else:
                            stage_context[name] = value
                    momentum_context_by_stage[stage_key] = stage_context
            schedule_matches = numpy.all(
                scalar_substeps == batch_substeps[numpy.newaxis, :],
                axis=1,
            )
            common_schedule_allowed = bool(
                momentum_runtimes
                or generated_scalar_hierarchy
                or declared_accuracy_controls.get("accuracy_tier") == "final"
            )
            if (
                int(numpy.count_nonzero(schedule_matches)) < 2
                and not common_schedule_allowed
            ):
                for mode_index, mode_k_value in zip(
                    group["indices"],
                    local_k_values,
                    strict=True,
                ):
                    _, scalar_source_arrays = _evolve_declared_mode(
                        float(mode_k_value),
                        collect_diagnostics=False,
                    )
                    results[int(mode_index)] = scalar_source_arrays
                continue
            histories, _, batch_stats = _integrate_batched_rk4(
                states,
                eta_values,
                required_substeps=required_substeps,
                active_intervals=active_intervals,
                rhs=_batch_rhs,
                pre_step=_batch_pre_step,
                post_step=_batch_post_step,
                record_step=_batch_record_step,
            )
            runtime_envelope["batch_count"] = (
                int(runtime_envelope["batch_count"]) + 1
            )
            runtime_envelope["batch_mode_count"] = int(
                runtime_envelope["batch_mode_count"]
            ) + int(batch_stats.mode_count)
            runtime_envelope["batched_rk_stage_count"] = int(
                runtime_envelope["batched_rk_stage_count"]
            ) + int(batch_stats.rk_stage_count)
            runtime_envelope["batched_max_substeps"] = max(
                int(runtime_envelope["batched_max_substeps"]),
                int(batch_stats.maximum_substeps),
            )
            schedule_correction_required = bool(
                generated_scalar_hierarchy
                and declared_accuracy_controls.get("accuracy_tier") != "final"
                and not contract_or_params.get(
                    "_diagnostic_matrix_fast_path", False
                )
            )
            history_names = tuple(
                slot.variable
                for slot in runtime_spec.state_slots
                if slot.order == 0
            )
            source_eta = numpy.asarray(source_grids["eta"], dtype=float)
            for row_index, mode_index in enumerate(group["indices"]):
                # Source evaluation binds the nonlocal runtime grids to the
                # line-of-sight grid.  Restore this batch's native evolution
                # context before auditing the next row so the finite-
                # difference history and compiled RHS use the same eta grid.
                (
                    active_grids,
                    active_declared_background_histories,
                    active_coordinate_rate_histories,
                ) = group["grids"]
                active_k_value = float(local_k_values[row_index])
                scalar_base_context_cache = {}
                scalar_background_context_cache = {}
                source_histories = {
                    name: numpy.asarray(
                        histories[row_index, :, slot.index],
                        dtype=float,
                    )
                    for name in history_names
                    for slot in runtime_spec.state_slots
                    if slot.variable == name and slot.order == 0
                }
                mode_k_value = float(local_k_values[row_index])
                if diagnostic_source_audit:
                    # Audit native hierarchy states before interpolation onto
                    # the line-of-sight source grid.
                    _record_hierarchy_equation_residuals(
                        mode_k_value,
                        source_histories,
                    )
                if not numpy.array_equal(eta_values, source_eta):
                    source_histories = {
                        name: numpy.asarray(
                            numpy.interp(source_eta, eta_values, values),
                            dtype=float,
                        )
                        for name, values in source_histories.items()
                    }
                state_history_max_abs_by_k[f"{mode_k_value:.12g}"] = {
                    name: float(numpy.max(numpy.abs(history), initial=0.0))
                    for name, history in source_histories.items()
                    if name
                    in {
                        "theta_gamma0",
                        "theta_gamma1",
                        "theta_gamma2",
                        "e_gamma2",
                        "e_gamma3",
                        "theta_b",
                        "delta_b",
                        "delta_c",
                        "delta_nu",
                        "sigma_nu",
                        "Phi",
                        "Psi",
                    }
                }
                if {"theta_gamma2", "e_gamma2"}.issubset(source_histories):
                    visibility = numpy.asarray(
                        source_grids["visibility"], dtype=float
                    )
                    active_visibility = visibility >= 0.1 * float(
                        numpy.max(visibility, initial=0.0)
                    )
                    if numpy.any(active_visibility):
                        theta_values = numpy.asarray(
                            source_histories["theta_gamma2"], dtype=float
                        )[active_visibility]
                        e_values = numpy.asarray(
                            source_histories["e_gamma2"], dtype=float
                        )[active_visibility]
                        state_history_polarization_ratio_by_k[
                            f"{mode_k_value:.12g}"
                        ] = {
                            "maximum_abs_e_over_theta": float(
                                numpy.max(
                                    numpy.abs(e_values)
                                    / numpy.maximum(
                                        numpy.abs(theta_values), 1.0e-30
                                    ),
                                    initial=0.0,
                                )
                            ),
                            "maximum_abs_theta": float(
                                numpy.max(numpy.abs(theta_values), initial=0.0)
                            ),
                            "maximum_abs_e": float(
                                numpy.max(numpy.abs(e_values), initial=0.0)
                            ),
                        }
                if schedule_correction_required and not numpy.array_equal(
                    scalar_substeps[row_index],
                    batch_substeps,
                ):
                    runtime_envelope[
                        "batched_schedule_correction_mode_count"
                    ] = (
                        int(
                            runtime_envelope[
                                "batched_schedule_correction_mode_count"
                            ]
                        )
                        + 1
                    )
                    _, scalar_source_arrays = _evolve_declared_mode(
                        mode_k_value,
                        collect_diagnostics=False,
                    )
                    results[int(mode_index)] = scalar_source_arrays
                    continue
                results[int(mode_index)] = _evaluate_source_histories(
                    mode_k_value,
                    source_histories,
                    required_source_names=required_source_names,
                )

        scalar_base_context_cache = {}
        scalar_background_context_cache = {}
        active_grids = dict(source_grids)
        active_declared_background_histories = (
            source_declared_background_histories
        )
        active_coordinate_rate_histories = source_coordinate_rate_histories
        return results

    transfer_cache_reuse_allowed = not (
        diagnostic_source_audit
        or adaptive_controls.transfer_enabled
        or adaptive_controls.source_enabled
        or adaptive_controls.projection_enabled
        or adaptive_controls.evolution_enabled
        or adaptive_k_enabled
        or k_values.size < 2
    )
    transfer_cache_key = _custom_cmb_transfer_cache_key(
        contract_or_params,
        ell_arr,
        background_provider,
        requested_spectra=requested_spectrum_names,
    )
    cached_transfer = (
        cache.get_cmb_transfer(transfer_cache_key)
        if transfer_cache_reuse_allowed
        else None
    )
    if cached_transfer is not None and (
        not numpy.array_equal(cached_transfer.ell_grid, ell_arr)
        or not numpy.array_equal(cached_transfer.k_grid, k_values)
        or set(cached_transfer.transfer_components)
        != set(transfer_component_observables)
    ):
        cached_transfer = None
    if cached_transfer is not None:
        transfer_components = {
            name: numpy.asarray(cached_transfer.transfer_components[name])
            for name in transfer_component_observables
        }
        for key in (
            "scalar_initial_constraint_preflight",
            "scalar_constraint_projection",
            "scalar_constraint_diagnostics",
            "declared_source_history_roles",
            "declared_source_history_sample_count",
            "declared_source_history_finite",
            "generated_scalar_hierarchy",
            "declared_source_history_convergence",
            "declared_source_history_max_abs",
            "declared_source_history_max_abs_by_k",
            "state_history_max_abs_by_k",
            "state_history_polarization_ratio_by_k",
            "source_context_max_abs_by_k",
            "source_context_pre_resolution_max_abs_by_k",
            "metric_history_gradient_residual_by_k",
            "metric_history_derivative_validation",
            "source_history_residual_samples_by_k",
            "source_history_residual_sample_schema",
            "source_history_cache_hit_count",
            "source_history_cache_miss_count",
            "source_history_cache_reused",
            "source_history_reconstruction_enabled",
            "source_history_reconstruction_diagnostic_only",
            "generated_scalar_source_closure",
            "source_history_derivative_provenance",
            "numerical_envelope",
            "accuracy_tier",
            "lensing_sampling_factor",
            "declared_k_sample_count",
            "k_grid_actual_count",
            "phase_aware_k_enabled",
            "phase_required_nodes",
            "phase_radial_required_nodes",
            "phase_acoustic_required_nodes",
            "phase_resolution_limited",
            "phase_resolution_status",
            "phase_grid_status",
            "projection_kernel_cache_keys",
        ):
            if key in cached_transfer.runtime_envelope:
                runtime_envelope[key] = cached_transfer.runtime_envelope[key]
        runtime_envelope["transfer_cache_hit"] = True
        runtime_envelope["transfer_cache_preparations"] = 0
        runtime_envelope["projection_kernel_cache_hits"] = 0
        runtime_envelope["projection_bessel_batch_count"] = 0
        runtime_envelope["projection_bessel_mode_count"] = 0
        runtime_envelope["declared_source_history_mode_count"] = 0
        projection_kernel_cache_hits = 0
        for kernel_cache_key in runtime_envelope.get(
            "projection_kernel_cache_keys", ()
        ):
            if (
                cache.get_declared_projection_kernel_batch(kernel_cache_key)
                is not None
            ):
                projection_kernel_cache_hits += 1
        runtime_envelope["projection_kernel_cache_hits"] = (
            projection_kernel_cache_hits
        )
        log_k_values = numpy.log(k_values)
        with performance_timer.phase("power_spectrum"):
            spectra_results = _integrate_declared_spectra(
                physical_params=physical_params,
                perturbation_data=perturbation_data,
                power_spectrum_observables=power_spectrum_observables,
                transfer_components=transfer_components,
                k_values=k_values,
                log_k_values=log_k_values,
            )
        elapsed_seconds = perf_counter() - request_started
        timing_snapshot = performance_timer.snapshot(
            total_seconds=elapsed_seconds,
        )
        runtime_envelope.update(timing_snapshot)
        spectrum_data = CustomCMBSpectrumData(
            ell_grid=ell_arr,
            k_grid=k_values,
            transfer_components=FrozenMapping(transfer_components),
            spectra=FrozenMapping(spectra_results),
            runtime_envelope=FrozenMapping(runtime_envelope),
            spectrum_availability=FrozenMapping(spectrum_availability),
        )
        cache.set_cmb_spectrum(cache_key, spectrum_data)
        return _get_cached_custom_cmb_spectrum_data(cache_key)

    log_k_values = numpy.log(k_values)
    projection_ell_batch_size = 512 if use_streaming_projection else 128
    kernel_cache_before = cache.cmb_cache_stats()[
        "declared_projection_kernel_batch"
    ]
    source_error = 0.0
    source_absolute_error = 0.0
    source_history_error = 0.0
    source_history_absolute_error = 0.0
    source_history_refinement_mode_count = 0
    projection_error = 0.0
    projection_absolute_error = 0.0
    evolution_anchor_errors: dict[str, float] = {
        "early": 0.0,
        "recombination": 0.0,
        "late": 0.0,
    }
    evolution_anchor_absolute_errors: dict[str, float] = {
        "early": 0.0,
        "recombination": 0.0,
        "late": 0.0,
    }
    evolution_error = 0.0
    evolution_absolute_error = 0.0
    evolution_coarse_to_intermediate_error = 0.0
    evolution_coarse_to_intermediate_absolute_error = 0.0
    evolution_intermediate_to_reference_error = 0.0
    evolution_intermediate_to_reference_absolute_error = 0.0
    evolution_coarse_to_intermediate_anchor_errors: dict[str, float] = {
        "early": 0.0,
        "recombination": 0.0,
        "late": 0.0,
    }
    evolution_intermediate_to_reference_anchor_errors: dict[str, float] = {
        "early": 0.0,
        "recombination": 0.0,
        "late": 0.0,
    }
    evolution_mode_count = 0
    evolution_fine_sample_count = 0
    evolution_intermediate_sample_count = 0
    evolution_coarse_sample_count = 0
    source_eta_indices = numpy.arange(
        0,
        int(source_grids["eta"].size),
        2,
        dtype=int,
    )
    if source_eta_indices[-1] != source_grids["eta"].size - 1:
        source_eta_indices = numpy.append(
            source_eta_indices,
            int(source_grids["eta"].size - 1),
        )
    source_eta_indices = numpy.unique(source_eta_indices)
    source_coarse_weights = None
    projection_coarse_weights = None
    if adaptive_controls.source_enabled and source_eta_indices.size >= 3:
        source_coarse_weights = _simpson_weights(
            source_grids["eta"][source_eta_indices]
        )
    if adaptive_controls.projection_enabled and source_eta_indices.size >= 3:
        projection_coarse_weights = _simpson_weights(
            source_grids["eta"][source_eta_indices]
        )

    with performance_timer.phase("evolution"):
        cached_mode_source_arrays: dict[int, dict[str, numpy.ndarray]] = {}
        if not diagnostic_source_audit:
            for k_index, k_value in enumerate(k_values):
                cached = cache.get_cmb_source_history(
                    _source_history_cache_key(float(k_value))
                )
                if cached is None:
                    source_history_cache_misses += 1
                    continue
                source_history_cache_hits += 1
                cached_mode_source_arrays[int(k_index)] = {
                    str(name): numpy.asarray(values, dtype=float).copy()
                    for name, values in cached.items()
                }
        if cached_mode_source_arrays and len(cached_mode_source_arrays) == int(
            k_values.size
        ):
            batched_mode_source_arrays = cached_mode_source_arrays
            runtime_envelope["evolution_chunks_completed"] = int(
                runtime_envelope["evolution_chunk_count"]
            )
            runtime_envelope["evolution_modes_completed"] = int(k_values.size)
        else:
            batched_mode_source_arrays = {}
            for chunk_index, chunk_start in enumerate(
                range(
                    0,
                    int(k_values.size),
                    int(evolution_chunk_size),
                ),
                start=1,
            ):
                chunk_stop = min(
                    chunk_start + int(evolution_chunk_size),
                    int(k_values.size),
                )
                chunk_results = _evolve_declared_modes_batched(
                    k_values[chunk_start:chunk_stop]
                )
                for local_index, source_arrays in chunk_results.items():
                    batched_mode_source_arrays[
                        int(chunk_start + local_index)
                    ] = source_arrays
                runtime_envelope["evolution_chunks_completed"] = int(
                    chunk_index
                )
                runtime_envelope["evolution_modes_completed"] = int(chunk_stop)
            runtime_envelope["evolution_chunks_completed"] = int(
                runtime_envelope["evolution_chunk_count"]
            )
            runtime_envelope["evolution_modes_completed"] = int(k_values.size)
            for k_index, source_arrays in batched_mode_source_arrays.items():
                if not diagnostic_source_audit:
                    cache.set_cmb_source_history(
                        _source_history_cache_key(float(k_values[k_index])),
                        {
                            str(name): numpy.asarray(
                                values, dtype=float
                            ).copy()
                            for name, values in source_arrays.items()
                        },
                    )
            if cached_mode_source_arrays:
                batched_mode_source_arrays = {
                    **batched_mode_source_arrays,
                    **cached_mode_source_arrays,
                }
        runtime_envelope["source_history_cache_hit_count"] = int(
            source_history_cache_hits
        )
        runtime_envelope["source_history_cache_miss_count"] = int(
            source_history_cache_misses
        )
        runtime_envelope["source_history_cache_reused"] = bool(
            source_history_cache_hits > 0
        )

    mode_projection_metadata: dict[
        int,
        tuple[numpy.ndarray, str, numpy.ndarray, tuple[int, ...]],
    ] = {}
    mode_kernel_batches: dict[
        int,
        dict[tuple[int, ...], _DeclaredProjectionKernelBatch],
    ] = {}
    bessel_work_groups: dict[
        int,
        list[tuple[int, numpy.ndarray, str, tuple[int, ...]]],
    ] = {}
    with performance_timer.phase("projection"):
        bessel_batch_count = 0
        bessel_mode_count = 0
        for k_index, k_value in enumerate(k_values):
            x_values = numpy.asarray(
                k_value * (eta0 - source_grids["eta"]),
                dtype=float,
            )
            x_signature = hashlib.sha256(x_values.tobytes()).hexdigest()
            cache.store_bessel_inputs(
                x_signature,
                x_values.copy(),
            )
            mode_ell_limit = _projection_ell_limit_for_mode(
                ell_values=ell_arr,
                x_values=x_values,
            )
            mode_ell_indices = numpy.flatnonzero(ell_arr <= mode_ell_limit)
            if mode_ell_indices.size == 0:
                continue
            mode_ell_signature = tuple(
                int(ell_value) for ell_value in ell_arr[mode_ell_indices]
            )
            mode_projection_metadata[int(k_index)] = (
                x_values,
                x_signature,
                mode_ell_indices,
                mode_ell_signature,
            )
            if use_streaming_projection:
                bessel_work_groups.setdefault(
                    ((len(mode_ell_signature) + 511) // 512) * 512,
                    [],
                ).append(
                    (
                        int(k_index),
                        x_values,
                        x_signature,
                        mode_ell_signature,
                    )
                )
                continue
            cached_batches: dict[
                tuple[int, ...], _DeclaredProjectionKernelBatch
            ] = {}
            missing_kernel = False
            for ell_start in range(0, ell_arr.size, projection_ell_batch_size):
                ell_stop = min(
                    ell_start + projection_ell_batch_size,
                    ell_arr.size,
                )
                batch_indices = mode_ell_indices[
                    (mode_ell_indices >= ell_start)
                    & (mode_ell_indices < ell_stop)
                ]
                if batch_indices.size == 0:
                    continue
                ell_signature = tuple(
                    int(ell_value) for ell_value in ell_arr[batch_indices]
                )
                sector_key = (
                    ("all",)
                    if streaming_projection_sectors is None
                    else tuple(sorted(streaming_projection_sectors))
                )
                cached = cache.get_declared_projection_kernel_batch(
                    (ell_signature, x_signature, sector_key)
                )
                if cached is None:
                    missing_kernel = True
                else:
                    cached_batches[ell_signature] = cached
            mode_kernel_batches[int(k_index)] = cached_batches
            if missing_kernel:
                mode_ell_bucket = (
                    (len(mode_ell_signature) + 511) // 512
                ) * 512
                bessel_work_groups.setdefault(mode_ell_bucket, []).append(
                    (
                        int(k_index),
                        x_values,
                        x_signature,
                        mode_ell_signature,
                    )
                )

        if use_streaming_projection:
            mode_source_arrays = dict(batched_mode_source_arrays)
            for k_index, k_value in enumerate(k_values):
                source_arrays = mode_source_arrays.get(int(k_index))
                if source_arrays is None:
                    with performance_timer.phase("evolution"):
                        _, source_arrays = _evolve_declared_mode(
                            float(k_value)
                        )
                _record_source_history_diagnostics(
                    source_arrays,
                    mode_k_value=float(k_value),
                )
                mode_source_arrays[int(k_index)] = source_arrays

            for work_group in bessel_work_groups.values():
                mode_ell_signature = max(
                    (entry[3] for entry in work_group),
                    key=len,
                )
                maximum_bessel_order = max(
                    1,
                    int(max(mode_ell_signature)),
                )
                eta_count = max(1, int(work_group[0][1].size))
                mode_batch_size = max(
                    1,
                    min(
                        _BESSEL_MAX_MODE_BATCH,
                        len(work_group),
                        max(
                            1,
                            32
                            * _BESSEL_WORK_CELL_BUDGET
                            // max(
                                (maximum_bessel_order + 65) * eta_count,
                                1,
                            ),
                        ),
                    ),
                )
                for group_start in range(0, len(work_group), mode_batch_size):
                    mode_group = work_group[
                        group_start : group_start + mode_batch_size
                    ]
                    mode_ell_signature = max(
                        (entry[3] for entry in mode_group),
                        key=len,
                    )
                    grouped_x_values = numpy.stack(
                        [entry[1] for entry in mode_group],
                        axis=0,
                    )
                    grouped_bessel, grouped_derivatives = (
                        _compute_spherical_bessel_mode_batch(
                            mode_ell_signature,
                            grouped_x_values,
                        )
                    )
                    bessel_batch_count += 1
                    bessel_mode_count += len(mode_group)
                    for group_index, (
                        k_index,
                        mode_x_values,
                        x_signature,
                        _,
                    ) in enumerate(mode_group):
                        mode_ell_indices = mode_projection_metadata[k_index][2]
                        source_arrays = mode_source_arrays[k_index]
                        precomputed_projection_bessel = (
                            mode_ell_signature,
                            grouped_bessel[:, group_index, :],
                            grouped_derivatives[:, group_index, :],
                        )
                        for ell_start in range(
                            0,
                            ell_arr.size,
                            projection_ell_batch_size,
                        ):
                            ell_stop = min(
                                ell_start + projection_ell_batch_size,
                                ell_arr.size,
                            )
                            batch_indices = mode_ell_indices[
                                (mode_ell_indices >= ell_start)
                                & (mode_ell_indices < ell_stop)
                            ]
                            if batch_indices.size == 0:
                                continue
                            ell_signature = tuple(
                                int(ell_value)
                                for ell_value in ell_arr[batch_indices]
                            )
                            kernel_batch = (
                                _get_cached_declared_projection_kernel_batch(
                                    ell_signature,
                                    x_signature,
                                    x_values=mode_x_values,
                                    precomputed_bessel=(
                                        precomputed_projection_bessel
                                    ),
                                    required_sectors=(
                                        streaming_projection_sectors
                                    ),
                                )
                            )
                            for (
                                component_name,
                                component_entry,
                            ) in transfer_component_observables.items():
                                source_histories = (
                                    _bind_declared_source_histories(
                                        component_name=str(component_name),
                                        component_entry=component_entry,
                                        source_arrays=source_arrays,
                                    )
                                )
                                transfer_components[component_name][
                                    batch_indices, k_index
                                ] = _declared_graph_projection(
                                    projection=str(
                                        component_entry.projection or ""
                                    ),
                                    kernel=(
                                        None
                                        if component_entry.kernel is None
                                        else str(component_entry.kernel)
                                    ),
                                    sector=(
                                        None
                                        if component_entry.sector is None
                                        else str(component_entry.sector)
                                    ),
                                    kernel_batch=kernel_batch,
                                    k_value=float(k_values[k_index]),
                                    eta_weights=eta_integration_weights,
                                    chi_grid=source_grids["chi"],
                                    source_chi=source_chi,
                                    source_histories=source_histories,
                                )
            bessel_work_groups.clear()

        for work_group in bessel_work_groups.values():
            mode_ell_signature = max(
                (entry[3] for entry in work_group),
                key=len,
            )
            maximum_bessel_order = max(
                1,
                int(max(mode_ell_signature)),
            )
            eta_count = max(1, int(work_group[0][1].size))
            mode_batch_size = max(
                1,
                min(
                    _BESSEL_MAX_MODE_BATCH,
                    len(work_group),
                    max(
                        1,
                        32
                        * _BESSEL_WORK_CELL_BUDGET
                        // max(
                            (maximum_bessel_order + 65) * eta_count,
                            1,
                        ),
                    ),
                ),
            )
            for group_start in range(0, len(work_group), mode_batch_size):
                mode_group = work_group[
                    group_start : group_start + mode_batch_size
                ]
                mode_ell_signature = max(
                    (entry[3] for entry in mode_group),
                    key=len,
                )
                grouped_x_values = numpy.stack(
                    [entry[1] for entry in mode_group],
                    axis=0,
                )
                grouped_bessel, grouped_derivatives = (
                    _compute_spherical_bessel_mode_batch(
                        mode_ell_signature,
                        grouped_x_values,
                    )
                )
                bessel_batch_count += 1
                bessel_mode_count += len(mode_group)
                for group_index, (k_index, _, x_signature, _) in enumerate(
                    mode_group
                ):
                    precomputed_projection_bessel = (
                        mode_ell_signature,
                        numpy.asarray(
                            grouped_bessel[:, group_index, :],
                            dtype=float,
                        ).copy(),
                        numpy.asarray(
                            grouped_derivatives[:, group_index, :],
                            dtype=float,
                        ).copy(),
                    )
                    mode_ell_indices = mode_projection_metadata[k_index][2]
                    cached_batches = mode_kernel_batches[k_index]
                    for ell_start in range(
                        0,
                        ell_arr.size,
                        projection_ell_batch_size,
                    ):
                        ell_stop = min(
                            ell_start + projection_ell_batch_size,
                            ell_arr.size,
                        )
                        batch_indices = mode_ell_indices[
                            (mode_ell_indices >= ell_start)
                            & (mode_ell_indices < ell_stop)
                        ]
                        if batch_indices.size == 0:
                            continue
                        ell_signature = tuple(
                            int(ell_value)
                            for ell_value in ell_arr[batch_indices]
                        )
                        if ell_signature in cached_batches:
                            continue
                        cached_batches[ell_signature] = (
                            _get_cached_declared_projection_kernel_batch(
                                ell_signature,
                                x_signature,
                                x_values=grouped_x_values[group_index],
                                precomputed_bessel=(
                                    precomputed_projection_bessel
                                ),
                                required_sectors=streaming_projection_sectors,
                            )
                        )

    for k_index, k_value in enumerate(k_values):
        if use_streaming_projection:
            continue
        base_history_sink = (
            {}
            if (
                adaptive_controls.evolution_enabled
                or adaptive_controls.source_enabled
            )
            else None
        )
        source_arrays = batched_mode_source_arrays.get(int(k_index))
        if source_arrays is None:
            with performance_timer.phase("evolution"):
                _, source_arrays = _evolve_declared_mode(
                    float(k_value),
                    history_sink=base_history_sink,
                )
            if not diagnostic_source_audit:
                cache.set_cmb_source_history(
                    _source_history_cache_key(float(k_value)),
                    {
                        str(name): numpy.asarray(values, dtype=float).copy()
                        for name, values in source_arrays.items()
                    },
                )
        with performance_timer.phase("evolution"):
            _record_source_history_diagnostics(
                source_arrays,
                mode_k_value=float(k_value),
            )
            if adaptive_controls.source_enabled:
                if base_history_sink is None:
                    raise RuntimeError(
                        "Source refinement requires a source-history sink"
                    )
                coarse_source_arrays = _evaluate_source_histories(
                    float(k_value),
                    {
                        name: numpy.asarray(history, dtype=float)
                        for name, history in base_history_sink[
                            "source_histories"
                        ].items()
                    },
                    collect_diagnostics=False,
                    source_grid_indices=source_eta_indices,
                    required_source_names=required_source_names,
                )
                coarse_eta = source_grids["eta"][source_eta_indices]
                for source_name, fine_values in source_arrays.items():
                    coarse_values = coarse_source_arrays[source_name]
                    interpolated_values = numpy.interp(
                        source_grids["eta"],
                        coarse_eta,
                        coarse_values,
                    )
                    estimate = estimate_convergence(
                        interpolated_values,
                        fine_values,
                        relative_tolerance=(
                            adaptive_controls.source_relative_tolerance
                        ),
                        absolute_tolerance=(
                            adaptive_controls.source_absolute_tolerance
                        ),
                    )
                    source_history_error = max(
                        source_history_error,
                        estimate.relative_error,
                    )
                    source_history_absolute_error = max(
                        source_history_absolute_error,
                        estimate.absolute_error,
                    )
                source_history_refinement_mode_count += 1
            if adaptive_controls.evolution_enabled:
                fine_sample_count = int(numerics.evolution_eta_sample_count)
                if not (
                    adaptive_controls.evolution_minimum_nodes
                    <= fine_sample_count
                    <= adaptive_controls.evolution_maximum_nodes
                ):
                    raise ValueError(
                        "evolution_eta_sample_count must be within the "
                        "adaptive_evolution node bounds"
                    )
                coarse_sample_count = max(32, fine_sample_count // 4)
                intermediate_sample_count = max(
                    coarse_sample_count + 1,
                    (coarse_sample_count + fine_sample_count) // 2,
                )
                if intermediate_sample_count >= fine_sample_count:
                    raise ValueError(
                        "adaptive_evolution requires a refinable "
                        "evolution_eta_sample_count"
                    )
                coarse_history_sink: dict[str, Any] = {}
                intermediate_history_sink: dict[str, Any] = {}
                _evolve_declared_mode(
                    float(k_value),
                    evolution_sample_count_override=coarse_sample_count,
                    history_sink=coarse_history_sink,
                    collect_diagnostics=False,
                )
                _evolve_declared_mode(
                    float(k_value),
                    evolution_sample_count_override=intermediate_sample_count,
                    history_sink=intermediate_history_sink,
                    collect_diagnostics=False,
                )

                def _estimate_evolution_pair(
                    lower_history_sink: Mapping[str, Any],
                    higher_history_sink: Mapping[str, Any],
                ) -> tuple[Any, Any]:
                    """Compare adjacent deterministic evolution tiers."""

                    state_estimate = estimate_history_convergence(
                        lower_history_sink["evolution_eta"],
                        lower_history_sink["evolution_histories"],
                        higher_history_sink["evolution_eta"],
                        higher_history_sink["evolution_histories"],
                        relative_tolerance=(
                            adaptive_controls.evolution_relative_tolerance
                        ),
                        absolute_tolerance=(
                            adaptive_controls.evolution_absolute_tolerance
                        ),
                    )
                    source_estimate = estimate_history_convergence(
                        lower_history_sink["source_eta"],
                        lower_history_sink["source_histories"],
                        higher_history_sink["source_eta"],
                        higher_history_sink["source_histories"],
                        relative_tolerance=(
                            adaptive_controls.evolution_relative_tolerance
                        ),
                        absolute_tolerance=(
                            adaptive_controls.evolution_absolute_tolerance
                        ),
                    )
                    return state_estimate, source_estimate

                (
                    coarse_state_estimate,
                    coarse_source_estimate,
                ) = _estimate_evolution_pair(
                    coarse_history_sink,
                    intermediate_history_sink,
                )
                (
                    state_estimate,
                    source_estimate,
                ) = _estimate_evolution_pair(
                    intermediate_history_sink,
                    base_history_sink,
                )
                for anchor_name in evolution_anchor_errors:
                    evolution_coarse_to_intermediate_anchor_errors[
                        anchor_name
                    ] = max(
                        evolution_coarse_to_intermediate_anchor_errors[
                            anchor_name
                        ],
                        float(
                            coarse_state_estimate.anchor_relative_errors[
                                anchor_name
                            ]
                        ),
                        float(
                            coarse_source_estimate.anchor_relative_errors[
                                anchor_name
                            ]
                        ),
                    )
                    evolution_intermediate_to_reference_anchor_errors[
                        anchor_name
                    ] = max(
                        evolution_intermediate_to_reference_anchor_errors[
                            anchor_name
                        ],
                        float(
                            state_estimate.anchor_relative_errors[anchor_name]
                        ),
                        float(
                            source_estimate.anchor_relative_errors[anchor_name]
                        ),
                    )
                for anchor_name in evolution_anchor_errors:
                    evolution_anchor_errors[anchor_name] = max(
                        evolution_anchor_errors[anchor_name],
                        float(
                            state_estimate.anchor_relative_errors[anchor_name]
                        ),
                        float(
                            source_estimate.anchor_relative_errors[anchor_name]
                        ),
                    )
                    evolution_anchor_absolute_errors[anchor_name] = max(
                        evolution_anchor_absolute_errors[anchor_name],
                        float(
                            state_estimate.anchor_absolute_errors[anchor_name]
                        ),
                        float(
                            source_estimate.anchor_absolute_errors[anchor_name]
                        ),
                    )
                evolution_coarse_to_intermediate_error = max(
                    evolution_coarse_to_intermediate_error,
                    coarse_state_estimate.relative_error,
                    coarse_source_estimate.relative_error,
                )
                evolution_coarse_to_intermediate_absolute_error = max(
                    evolution_coarse_to_intermediate_absolute_error,
                    coarse_state_estimate.absolute_error,
                    coarse_source_estimate.absolute_error,
                )
                evolution_intermediate_to_reference_error = max(
                    evolution_intermediate_to_reference_error,
                    state_estimate.relative_error,
                    source_estimate.relative_error,
                )
                evolution_intermediate_to_reference_absolute_error = max(
                    evolution_intermediate_to_reference_absolute_error,
                    state_estimate.absolute_error,
                    source_estimate.absolute_error,
                )
                evolution_error = max(
                    evolution_error,
                    state_estimate.relative_error,
                    source_estimate.relative_error,
                )
                evolution_absolute_error = max(
                    evolution_absolute_error,
                    state_estimate.absolute_error,
                    source_estimate.absolute_error,
                )
                evolution_mode_count += 1
                evolution_fine_sample_count = max(
                    evolution_fine_sample_count,
                    int(base_history_sink["evolution_eta"].size),
                )
                evolution_intermediate_sample_count = max(
                    evolution_intermediate_sample_count,
                    int(intermediate_history_sink["evolution_eta"].size),
                )
                evolution_coarse_sample_count = max(
                    evolution_coarse_sample_count,
                    int(coarse_history_sink["evolution_eta"].size),
                )
        if adaptive_k_enabled:
            for (
                component_name,
                component_entry,
            ) in transfer_component_observables.items():
                for (
                    role_name,
                    source_name,
                ) in component_entry.source_terms.items():
                    adaptive_source_history_rows.setdefault(
                        (str(component_name), str(role_name)),
                        [],
                    ).append(
                        numpy.asarray(
                            source_arrays[str(source_name)], dtype=float
                        )
                    )
        mode_projection = mode_projection_metadata.get(int(k_index))
        if mode_projection is None:
            continue
        (
            _,
            _,
            mode_ell_indices,
            _,
        ) = mode_projection
        projection_started = perf_counter()
        cached_batches = mode_kernel_batches[int(k_index)]
        if not cached_batches:
            raise RuntimeError(
                "Declared projection did not prepare any radial kernel batches"
            )
        for ell_start in range(0, ell_arr.size, projection_ell_batch_size):
            ell_stop = min(
                ell_start + projection_ell_batch_size,
                ell_arr.size,
            )
            batch_indices = mode_ell_indices[
                (mode_ell_indices >= ell_start) & (mode_ell_indices < ell_stop)
            ]
            if batch_indices.size == 0:
                continue
            ell_signature = tuple(
                int(ell_value) for ell_value in ell_arr[batch_indices]
            )
            kernel_batch = cached_batches.get(ell_signature)
            if kernel_batch is None:
                raise RuntimeError(
                    "Declared projection radial kernel batch was not cached"
                )
            for (
                component_name,
                component_entry,
            ) in transfer_component_observables.items():
                source_histories = _bind_declared_source_histories(
                    component_name=str(component_name),
                    component_entry=component_entry,
                    source_arrays=source_arrays,
                )
                transfer_components[component_name][batch_indices, k_index] = (
                    _declared_graph_projection(
                        projection=str(component_entry.projection or ""),
                        kernel=(
                            None
                            if component_entry.kernel is None
                            else str(component_entry.kernel)
                        ),
                        sector=(
                            None
                            if component_entry.sector is None
                            else str(component_entry.sector)
                        ),
                        kernel_batch=kernel_batch,
                        k_value=float(k_value),
                        eta_weights=eta_integration_weights,
                        chi_grid=source_grids["chi"],
                        source_chi=source_chi,
                        source_histories=source_histories,
                    )
                )
                projected_values = transfer_components[component_name][
                    batch_indices,
                    k_index,
                ]
                coarse_values = None
                if (
                    source_coarse_weights is not None
                    or projection_coarse_weights is not None
                ):
                    coarse_kernel_batch = _slice_projection_kernel_batch(
                        kernel_batch,
                        source_eta_indices,
                    )
                    coarse_histories = {
                        role_name: history[source_eta_indices]
                        for role_name, history in source_histories.items()
                    }
                    coarse_weights = (
                        source_coarse_weights
                        if source_coarse_weights is not None
                        else projection_coarse_weights
                    )
                    coarse_values = _declared_graph_projection(
                        projection=str(component_entry.projection or ""),
                        kernel=(
                            None
                            if component_entry.kernel is None
                            else str(component_entry.kernel)
                        ),
                        sector=(
                            None
                            if component_entry.sector is None
                            else str(component_entry.sector)
                        ),
                        kernel_batch=coarse_kernel_batch,
                        k_value=float(k_value),
                        eta_weights=coarse_weights,
                        chi_grid=source_grids["chi"][source_eta_indices],
                        source_chi=source_chi,
                        source_histories=coarse_histories,
                    )
                    estimate = estimate_convergence(
                        coarse_values,
                        projected_values,
                        relative_tolerance=(
                            adaptive_controls.source_relative_tolerance
                        ),
                        absolute_tolerance=(
                            adaptive_controls.source_absolute_tolerance
                        ),
                    )
                    source_error = max(source_error, estimate.relative_error)
                    source_absolute_error = max(
                        source_absolute_error,
                        estimate.absolute_error,
                    )
                if projection_coarse_weights is not None:
                    if coarse_values is None:
                        raise RuntimeError(
                            "Projection convergence surface was not built"
                        )
                    estimate = estimate_convergence(
                        coarse_values,
                        projected_values,
                        relative_tolerance=(
                            adaptive_controls.projection_relative_tolerance
                        ),
                        absolute_tolerance=(
                            adaptive_controls.projection_absolute_tolerance
                        ),
                    )
                    projection_error = max(
                        projection_error,
                        estimate.relative_error,
                    )
                    projection_absolute_error = max(
                        projection_absolute_error,
                        estimate.absolute_error,
                    )
        performance_timer.add(
            "projection",
            perf_counter() - projection_started,
        )
    for component_name, component_matrix in transfer_components.items():
        if not numpy.all(numpy.isfinite(component_matrix)):
            raise ValueError(
                "Declared transfer component produced non-finite values: "
                f"{component_name}"
            )

    spectra_results: dict[str, numpy.ndarray] = {}
    with performance_timer.phase("power_spectrum"):
        spectra_results = _integrate_declared_spectra(
            physical_params=physical_params,
            perturbation_data=perturbation_data,
            power_spectrum_observables=power_spectrum_observables,
            transfer_components=transfer_components,
            k_values=k_values,
            log_k_values=log_k_values,
        )

    if adaptive_controls.transfer_enabled and k_values.size >= 5:
        coarse_k_indices = numpy.arange(0, int(k_values.size), 2, dtype=int)
        if coarse_k_indices[-1] != k_values.size - 1:
            coarse_k_indices = numpy.append(
                coarse_k_indices,
                int(k_values.size - 1),
            )
        coarse_k_indices = numpy.unique(coarse_k_indices)
        transfer_estimates = []
        for (
            observable_name,
            observable_entry,
        ) in power_spectrum_observables.items():
            primary = numpy.asarray(
                transfer_components[str(observable_entry.primary)][
                    :, coarse_k_indices
                ],
                dtype=numpy.longdouble,
            )
            secondary = numpy.asarray(
                transfer_components[str(observable_entry.secondary)][
                    :, coarse_k_indices
                ],
                dtype=numpy.longdouble,
            )
            coarse_spectrum = _integrate_power_spectrum(
                primordial_grid=_primordial_power_grid_for_observable(
                    physical_params=physical_params,
                    perturbation_data=perturbation_data,
                    observable_entry=observable_entry,
                    k_values=k_values[coarse_k_indices],
                ),
                log_k_values=log_k_values[coarse_k_indices],
                primary=primary,
                secondary=secondary,
                auto_spectrum=(
                    str(observable_entry.primary)
                    == str(observable_entry.secondary)
                ),
            )
            full_spectrum = numpy.asarray(
                spectra_results[str(observable_name)],
                dtype=float,
            )
            transfer_estimates.append(
                estimate_convergence(
                    coarse_spectrum,
                    full_spectrum,
                    relative_tolerance=(
                        adaptive_controls.transfer_relative_tolerance
                    ),
                    absolute_tolerance=(
                        adaptive_controls.transfer_absolute_tolerance
                    ),
                )
            )
        if transfer_estimates:
            transfer_estimate = max(
                transfer_estimates,
                key=lambda estimate: estimate.relative_error,
            )
            runtime_envelope["adaptive_transfer_relative_error"] = float(
                transfer_estimate.relative_error
            )
            runtime_envelope["adaptive_transfer_absolute_error"] = float(
                transfer_estimate.absolute_error
            )
            runtime_envelope["adaptive_transfer_refinement_levels"] = 1
            require_convergence(
                transfer_estimate,
                label="transfer",
                fail_on_nonconvergence=(
                    adaptive_controls.fail_on_nonconvergence
                ),
            )
    if adaptive_controls.source_enabled:
        runtime_envelope["adaptive_source_relative_error"] = float(
            source_error
        )
        runtime_envelope["adaptive_source_absolute_error"] = float(
            source_absolute_error
        )
        runtime_envelope["adaptive_source_refinement_levels"] = 1
        source_estimate = ConvergenceEstimate(
            absolute_error=source_absolute_error,
            relative_error=source_error,
            converged=(
                source_absolute_error
                <= adaptive_controls.source_absolute_tolerance
                or source_error <= adaptive_controls.source_relative_tolerance
            ),
        )
        require_convergence(
            source_estimate,
            label="source-history",
            fail_on_nonconvergence=adaptive_controls.fail_on_nonconvergence,
        )
    if adaptive_controls.projection_enabled:
        runtime_envelope["adaptive_projection_relative_error"] = float(
            projection_error
        )
        runtime_envelope["adaptive_projection_absolute_error"] = float(
            projection_absolute_error
        )
        runtime_envelope["adaptive_projection_refinement_levels"] = 1
        projection_estimate = ConvergenceEstimate(
            absolute_error=projection_absolute_error,
            relative_error=projection_error,
            converged=(
                projection_absolute_error
                <= adaptive_controls.projection_absolute_tolerance
                or projection_error
                <= adaptive_controls.projection_relative_tolerance
            ),
        )
        require_convergence(
            projection_estimate,
            label="line-of-sight projection",
            fail_on_nonconvergence=adaptive_controls.fail_on_nonconvergence,
        )

    if adaptive_controls.evolution_enabled:
        runtime_envelope["adaptive_evolution_relative_error"] = float(
            evolution_error
        )
        runtime_envelope["adaptive_evolution_absolute_error"] = float(
            evolution_absolute_error
        )
        runtime_envelope["adaptive_evolution_refinement_levels"] = 2
        evolution_refinement_evidence = {
            "same_model": True,
            "tiers": {
                "coarse": {
                    "sample_count": int(evolution_coarse_sample_count),
                },
                "intermediate": {
                    "sample_count": int(evolution_intermediate_sample_count),
                },
                "reference": {
                    "sample_count": int(evolution_fine_sample_count),
                },
            },
            "coarse_to_intermediate": {
                "relative_error": float(
                    evolution_coarse_to_intermediate_error
                ),
                "absolute_error": float(
                    evolution_coarse_to_intermediate_absolute_error
                ),
                "anchor_relative_errors": dict(
                    evolution_coarse_to_intermediate_anchor_errors
                ),
            },
            "intermediate_to_reference": {
                "relative_error": float(
                    evolution_intermediate_to_reference_error
                ),
                "absolute_error": float(
                    evolution_intermediate_to_reference_absolute_error
                ),
                "anchor_relative_errors": dict(
                    evolution_intermediate_to_reference_anchor_errors
                ),
                "anchor_absolute_errors": dict(
                    evolution_anchor_absolute_errors
                ),
            },
            "relative_tolerance": float(
                adaptive_controls.evolution_relative_tolerance
            ),
            "absolute_tolerance": float(
                adaptive_controls.evolution_absolute_tolerance
            ),
        }
        runtime_envelope["scalar_evolution_convergence"] = {
            "tier_order": ("coarse", "intermediate", "reference"),
            "relative_error": float(evolution_error),
            "absolute_error": float(evolution_absolute_error),
            "anchor_relative_errors": dict(evolution_anchor_errors),
            "anchor_absolute_errors": dict(evolution_anchor_absolute_errors),
            "mode_count": int(evolution_mode_count),
            "reference_sample_count": int(evolution_fine_sample_count),
            "fine_sample_count": int(evolution_fine_sample_count),
            "intermediate_sample_count": int(
                evolution_intermediate_sample_count
            ),
            "coarse_sample_count": int(evolution_coarse_sample_count),
            "refinement_evidence": evolution_refinement_evidence,
            "relative_tolerance": float(
                adaptive_controls.evolution_relative_tolerance
            ),
            "absolute_tolerance": float(
                adaptive_controls.evolution_absolute_tolerance
            ),
        }
        for metrics in scalar_constraint_diagnostics.values():
            refinement_evidence = dict(metrics["refinement_evidence"])
            refinement_evidence["evolution"] = evolution_refinement_evidence
            metrics["refinement_evidence"] = refinement_evidence
        evolution_estimate = ConvergenceEstimate(
            absolute_error=float(evolution_absolute_error),
            relative_error=float(evolution_error),
            converged=all(
                evolution_anchor_absolute_errors[name]
                <= adaptive_controls.evolution_absolute_tolerance
                or evolution_anchor_errors[name]
                <= adaptive_controls.evolution_relative_tolerance
                for name in evolution_anchor_errors
            ),
        )
        require_convergence(
            evolution_estimate,
            label="scalar evolution history",
            fail_on_nonconvergence=adaptive_controls.fail_on_nonconvergence,
        )

    if (
        adaptive_k_enabled
        and adaptive_k_mode == "source"
        and direct_source_quadrature
    ):
        """Re-evolve declared modes on the source quadrature grid.

        Interpolating a sparse set of source histories cannot preserve the
        acoustic oscillations that the line-of-sight kernels resolve.  The
        source mode therefore uses the declared node budget for actual mode
        evolution and reserves interpolation for the separate transfer mode.
        """

        direct_ell_indices = numpy.flatnonzero(
            ell_arr >= int(adaptive_k_min_ell)
        )[::adaptive_k_ell_stride]
        direct_k = numpy.geomspace(
            float(k_values[0]),
            float(k_values[-1]),
            max(32, int(adaptive_k_node_count)),
            dtype=float,
        )
        direct_transfer_components = {
            name: numpy.zeros(
                (direct_ell_indices.size, direct_k.size),
                dtype=float,
            )
            for name in transfer_component_observables
        }
        direct_envelope = _enforce_runtime_envelope(
            contract_or_params,
            ell_count=int(direct_ell_indices.size),
            k_count=int(direct_k.size),
            eta_count=int(source_grids["eta"].size),
            state_slot_count=int(len(runtime_spec.state_slots)),
            transfer_component_count=int(len(transfer_component_observables)),
            momentum_point_count=int(
                sum(runtime.points.size for runtime in momentum_runtimes)
            ),
            evolution_multiplier=(
                3 if adaptive_controls.evolution_enabled else 1
            ),
        )
        direct_envelope["static_graph_preparations"] = runtime_envelope[
            "static_graph_preparations"
        ]
        direct_envelope["contract_static_preparations"] = runtime_envelope[
            "contract_static_preparations"
        ]
        direct_envelope["model_static_preparations"] = runtime_envelope[
            "model_static_preparations"
        ]
        direct_envelope["request_specific_preparations"] = 1
        direct_envelope["dynamic_mode_count"] = int(direct_k.size)
        direct_envelope["batch_count"] = 0
        direct_envelope["batch_mode_count"] = 0
        direct_envelope["batched_rk_stage_count"] = 0
        direct_envelope["batched_max_substeps"] = 0
        for direct_k_index, direct_k_value in enumerate(direct_k):
            _, direct_source_arrays = _evolve_declared_mode(
                float(direct_k_value)
            )
            _record_source_history_diagnostics(
                direct_source_arrays,
                mode_k_value=float(direct_k_value),
            )
            x_values = float(direct_k_value) * (eta0 - source_grids["eta"])
            x_signature = hashlib.sha256(
                numpy.asarray(x_values, dtype=float).tobytes()
            ).hexdigest()
            cache.store_bessel_inputs(
                x_signature,
                numpy.asarray(x_values, dtype=float).copy(),
            )
            mode_ell_values = numpy.asarray(
                ell_arr[direct_ell_indices],
                dtype=int,
            )
            mode_ell_limit = _projection_ell_limit_for_mode(
                ell_values=mode_ell_values,
                x_values=numpy.asarray(x_values, dtype=float),
            )
            mode_indices = numpy.flatnonzero(mode_ell_values <= mode_ell_limit)
            if mode_indices.size == 0:
                continue
            mode_signature = tuple(
                int(value) for value in mode_ell_values[mode_indices]
            )
            precomputed_bessel = _compute_spherical_bessel_batch(
                mode_signature,
                numpy.asarray(x_values, dtype=float),
            )
            for batch_start in range(0, mode_indices.size, 128):
                batch_stop = min(batch_start + 128, mode_indices.size)
                batch_indices = mode_indices[batch_start:batch_stop]
                batch_signature = tuple(
                    int(value) for value in mode_ell_values[batch_indices]
                )
                kernel_batch = _get_cached_declared_projection_kernel_batch(
                    batch_signature,
                    x_signature,
                    x_values=numpy.asarray(x_values, dtype=float),
                    precomputed_bessel=(
                        mode_signature,
                        precomputed_bessel[0],
                        precomputed_bessel[1],
                    ),
                )
                for (
                    component_name,
                    component_entry,
                ) in transfer_component_observables.items():
                    source_histories = _bind_declared_source_histories(
                        component_name=str(component_name),
                        component_entry=component_entry,
                        source_arrays=direct_source_arrays,
                    )
                    projected = _declared_graph_projection(
                        projection=str(component_entry.projection or ""),
                        kernel=(
                            None
                            if component_entry.kernel is None
                            else str(component_entry.kernel)
                        ),
                        sector=(
                            None
                            if component_entry.sector is None
                            else str(component_entry.sector)
                        ),
                        kernel_batch=kernel_batch,
                        k_value=float(direct_k_value),
                        eta_weights=eta_integration_weights,
                        chi_grid=source_grids["chi"],
                        source_chi=source_chi,
                        source_histories=source_histories,
                    )
                    direct_transfer_components[component_name][
                        batch_indices,
                        direct_k_index,
                    ] = projected
        direct_spectra = {
            name: numpy.asarray(values, dtype=numpy.longdouble).copy()
            for name, values in spectra_results.items()
        }
        for (
            observable_name,
            observable_entry,
        ) in power_spectrum_observables.items():
            primary_name = str(observable_entry.primary)
            secondary_name = str(observable_entry.secondary)
            if (
                primary_name not in direct_transfer_components
                or secondary_name not in direct_transfer_components
            ):
                continue
            primordial_grid = _primordial_power_grid_for_observable(
                physical_params=physical_params,
                perturbation_data=perturbation_data,
                observable_entry=observable_entry,
                k_values=direct_k,
            )
            for row_index, ell_index in enumerate(direct_ell_indices):
                direct_spectra[observable_name][ell_index] = (
                    _integrate_power_spectrum(
                        primordial_grid=primordial_grid,
                        log_k_values=numpy.log(direct_k),
                        primary=direct_transfer_components[primary_name][
                            row_index
                        ],
                        secondary=direct_transfer_components[secondary_name][
                            row_index
                        ],
                        auto_spectrum=primary_name == secondary_name,
                    )[0]
                )
        spectra_results = direct_spectra
        adaptive_k_enabled = False

    if (
        adaptive_k_enabled
        and adaptive_k_mode == "source"
        and adaptive_source_history_rows
    ):
        adaptive_eta_indices = numpy.arange(
            0,
            int(source_grids["eta"].size),
            adaptive_k_eta_stride,
            dtype=int,
        )
        adaptive_eta_grid = numpy.asarray(
            source_grids["eta"][adaptive_eta_indices],
            dtype=float,
        )
        adaptive_eta_integration_weights = _simpson_weights(adaptive_eta_grid)
        adaptive_source_histories = {
            key: numpy.asarray(rows, dtype=float)[:, adaptive_eta_indices]
            for key, rows in adaptive_source_history_rows.items()
        }
        adaptive_source_interpolators = [
            (
                history,
                CubicSpline(
                    k_values,
                    history,
                    axis=0,
                    bc_type="natural",
                    extrapolate=False,
                ),
            )
            for history in adaptive_source_histories.values()
            if k_values.size >= 4
        ]
        adaptive_ell_indices = numpy.flatnonzero(
            ell_arr >= int(adaptive_k_min_ell)
        )
        adaptive_ell_indices = adaptive_ell_indices[::adaptive_k_ell_stride]
        scalar_components = {
            name
            for name, entry in transfer_component_observables.items()
            if str(entry.sector or "scalar") == "scalar"
        }
        scalar_components.intersection_update(
            {
                "temperature",
                "polarization_e",
                "lensing_potential",
            }
        )

        def _adaptive_scalar_kernel(
            component_name: str,
            role_name: str,
            *,
            ell_value: int,
            j_values: numpy.ndarray,
            j_derivatives: numpy.ndarray,
            inverse_x: numpy.ndarray,
        ) -> numpy.ndarray:
            """Return the canonical kernel for one adaptive source role."""

            component_entry = transfer_component_observables[component_name]
            projection_name = str(component_entry.projection or "")
            kernel_name = resolve_declared_source_kernel(
                projection_name,
                role_name,
                kernel=(
                    None
                    if component_entry.kernel is None
                    else str(component_entry.kernel)
                ),
            )
            kernel_kind = get_declared_projection_kernel_spec(kernel_name).kind
            if kernel_kind == "spherical_bessel":
                return j_values
            if kernel_kind == "spherical_bessel_derivative":
                return j_derivatives
            if kernel_kind == "spherical_bessel_second_derivative":
                return (
                    float(ell_value * (ell_value + 1)) * inverse_x * inverse_x
                    - 1.0
                ) * j_values - 2.0 * inverse_x * j_derivatives
            if kernel_kind == "spin2_e":
                prefactor = math.exp(
                    0.5
                    * (
                        math.lgamma(int(ell_value) + 3)
                        - math.lgamma(int(ell_value) - 1)
                    )
                )
                return prefactor * j_values * inverse_x * inverse_x
            if kernel_kind == "spin2_b":
                prefactor = math.exp(
                    0.5
                    * (
                        math.lgamma(int(ell_value) + 3)
                        - math.lgamma(int(ell_value) - 1)
                    )
                )
                return prefactor * j_values * inverse_x * inverse_x
            if kernel_kind == "lensing_potential":
                geometry = numpy.clip(
                    source_chi - source_grids["chi"],
                    0.0,
                    None,
                ) / (
                    max(float(source_chi), 1.0e-12)
                    * numpy.maximum(source_grids["chi"], 1.0e-12)
                )
                return -j_values * geometry[numpy.newaxis, :]
            raise ValueError(
                f"Adaptive scalar projection does not support kernel "
                f"'{kernel_name}'"
            )

        def _interpolate_mode_histories(
            histories: numpy.ndarray,
            local_k: numpy.ndarray,
        ) -> numpy.ndarray:
            """Interpolate mode histories onto one local quadrature grid."""

            right_indices = numpy.searchsorted(k_values, local_k, side="left")
            right_indices = numpy.clip(
                right_indices,
                1,
                int(k_values.size) - 1,
            )
            left_indices = right_indices - 1
            left_k = k_values[left_indices]
            right_k = k_values[right_indices]
            fraction = (local_k - left_k) / numpy.maximum(
                right_k - left_k,
                1.0e-30,
            )
            return (1.0 - fraction[:, numpy.newaxis]) * histories[
                left_indices
            ] + fraction[:, numpy.newaxis] * histories[right_indices]

        def _interpolate_mode_history_batch(
            histories: numpy.ndarray,
            local_k: numpy.ndarray,
        ) -> numpy.ndarray:
            """Interpolate several local quadrature windows at once."""

            for cached_history, interpolator in adaptive_source_interpolators:
                if histories is cached_history:
                    return numpy.asarray(interpolator(local_k), dtype=float)

            flat_k = numpy.asarray(local_k, dtype=float).reshape(-1)
            right_indices = numpy.searchsorted(
                k_values,
                flat_k,
                side="left",
            )
            right_indices = numpy.clip(
                right_indices,
                1,
                int(k_values.size) - 1,
            )
            left_indices = right_indices - 1
            left_k = k_values[left_indices]
            right_k = k_values[right_indices]
            fraction = (flat_k - left_k) / numpy.maximum(
                right_k - left_k,
                1.0e-30,
            )
            interpolated = (1.0 - fraction[:, numpy.newaxis]) * histories[
                left_indices
            ] + fraction[:, numpy.newaxis] * histories[right_indices]
            return interpolated.reshape(
                (*local_k.shape, int(histories.shape[-1]))
            )

        def _adaptive_component_transfer(
            component_name: str,
            ell_value: int,
            local_k: numpy.ndarray,
        ) -> numpy.ndarray:
            """Project interpolated source histories for one scalar ell."""

            x_values = local_k[:, numpy.newaxis] * (
                eta0 - source_grids["eta"][numpy.newaxis, :]
            )
            inverse_x = 1.0 / numpy.maximum(numpy.abs(x_values), 1.0e-12)
            j_values = spherical_jn(int(ell_value), x_values)
            j_derivatives = spherical_jn(
                int(ell_value),
                x_values,
                derivative=True,
            )
            projected = numpy.zeros(local_k.size, dtype=float)
            for component, role_name in adaptive_source_histories:
                if component != component_name:
                    continue
                history = adaptive_source_histories[
                    (component_name, role_name)
                ]
                kernel = _adaptive_scalar_kernel(
                    component_name,
                    role_name,
                    ell_value=ell_value,
                    j_values=j_values,
                    j_derivatives=j_derivatives,
                    inverse_x=inverse_x,
                )
                projected += numpy.sum(
                    kernel
                    * _interpolate_mode_histories(history, local_k)
                    * eta_integration_weights[numpy.newaxis, :],
                    axis=1,
                )
            return projected

        def _adaptive_component_transfer_batch(
            component_name: str,
            ell_values: numpy.ndarray,
            local_k: numpy.ndarray,
        ) -> numpy.ndarray:
            """Project a batch of local scalar windows in one Bessel pass."""

            ell_grid = numpy.asarray(ell_values, dtype=int)
            x_values = local_k[:, :, numpy.newaxis] * (
                eta0 - adaptive_eta_grid[numpy.newaxis, numpy.newaxis, :]
            )
            inverse_x = 1.0 / numpy.maximum(numpy.abs(x_values), 1.0e-12)
            bessel_order = ell_grid[:, numpy.newaxis, numpy.newaxis]
            j_values = spherical_jn(bessel_order, x_values)
            j_derivatives = spherical_jn(
                bessel_order,
                x_values,
                derivative=True,
            )
            projected = numpy.zeros(local_k.shape, dtype=float)
            for component, role_name in adaptive_source_histories:
                if component != component_name:
                    continue
                history = adaptive_source_histories.get(
                    (component_name, role_name)
                )
                if history is None:
                    continue
                component_entry = transfer_component_observables[
                    component_name
                ]
                projection_name = str(component_entry.projection or "")
                kernel_name = resolve_declared_source_kernel(
                    projection_name,
                    role_name,
                    kernel=(
                        None
                        if component_entry.kernel is None
                        else str(component_entry.kernel)
                    ),
                )
                kernel_kind = get_declared_projection_kernel_spec(
                    kernel_name
                ).kind
                if kernel_kind == "spherical_bessel":
                    kernel = j_values
                elif kernel_kind == "spherical_bessel_derivative":
                    kernel = j_derivatives
                elif kernel_kind == "spherical_bessel_second_derivative":
                    kernel = (
                        bessel_order
                        * (bessel_order + 1)
                        * inverse_x
                        * inverse_x
                        - 1.0
                    ) * j_values - 2.0 * inverse_x * j_derivatives
                elif kernel_kind in {"spin2_e", "spin2_b"}:
                    prefactor = numpy.exp(
                        0.5
                        * (gammaln(ell_grid + 3.0) - gammaln(ell_grid - 1.0))
                    )
                    kernel = (
                        prefactor[:, numpy.newaxis, numpy.newaxis]
                        * j_values
                        * inverse_x
                        * inverse_x
                    )
                elif kernel_kind == "lensing_potential":
                    geometry = numpy.clip(
                        source_chi - source_grids["chi"][adaptive_eta_indices],
                        0.0,
                        None,
                    ) / (
                        max(float(source_chi), 1.0e-12)
                        * numpy.maximum(
                            source_grids["chi"][adaptive_eta_indices],
                            1.0e-12,
                        )
                    )
                    kernel = -j_values * geometry[None, None, :]
                else:
                    raise ValueError(
                        f"Adaptive scalar projection does not support kernel "
                        f"'{kernel_name}'"
                    )
                projected += numpy.sum(
                    kernel
                    * _interpolate_mode_history_batch(history, local_k)
                    * adaptive_eta_integration_weights[None, None, :],
                    axis=2,
                )
            return projected

        adaptive_spectra = {
            name: numpy.asarray(values, dtype=numpy.longdouble).copy()
            for name, values in spectra_results.items()
        }
        adaptive_batch_size = 8
        dense_k_count = max(32, int(adaptive_k_node_count))
        dense_log_k = numpy.linspace(
            float(numpy.log(k_values[0])),
            float(numpy.log(k_values[-1])),
            dense_k_count,
            dtype=float,
        )
        dense_k = numpy.unique(
            numpy.concatenate(
                (
                    numpy.asarray(k_values, dtype=float),
                    numpy.clip(
                        numpy.exp(dense_log_k),
                        float(k_values[0]),
                        float(k_values[-1]),
                    ),
                )
            )
        )
        for batch_start in range(
            0,
            int(adaptive_ell_indices.size),
            adaptive_batch_size,
        ):
            batch_indices = adaptive_ell_indices[
                batch_start : batch_start + adaptive_batch_size
            ]
            ell_values = numpy.asarray(ell_arr[batch_indices], dtype=int)
            local_k = numpy.broadcast_to(
                dense_k[numpy.newaxis, :],
                (ell_values.size, dense_k.size),
            )
            local_transfers = {
                name: _adaptive_component_transfer_batch(
                    name,
                    ell_values,
                    local_k,
                )
                for name in scalar_components
            }
            for (
                observable_name,
                observable_entry,
            ) in power_spectrum_observables.items():
                primary_name = str(observable_entry.primary)
                secondary_name = str(observable_entry.secondary)
                if (
                    primary_name not in local_transfers
                    or secondary_name not in local_transfers
                ):
                    continue
                primordial_grid = _primordial_power_grid_for_observable(
                    physical_params=physical_params,
                    perturbation_data=perturbation_data,
                    observable_entry=observable_entry,
                    k_values=local_k,
                )
                primary = local_transfers[primary_name]
                secondary = local_transfers[secondary_name]
                for row_index, ell_index in enumerate(batch_indices):
                    adaptive_spectra[observable_name][ell_index] = (
                        _integrate_power_spectrum(
                            primordial_grid=primordial_grid[row_index],
                            log_k_values=numpy.log(local_k[row_index]),
                            primary=primary[row_index],
                            secondary=secondary[row_index],
                            auto_spectrum=primary_name == secondary_name,
                        )
                    )[0]
        if adaptive_ell_indices.size >= 2:
            dense_indices = numpy.flatnonzero(
                ell_arr >= int(ell_arr[adaptive_ell_indices[0]])
            )
            for observable_name, values in adaptive_spectra.items():
                sampled_values = values[adaptive_ell_indices]
                values[dense_indices] = numpy.interp(
                    numpy.asarray(dense_indices, dtype=float),
                    numpy.asarray(adaptive_ell_indices, dtype=float),
                    numpy.asarray(sampled_values, dtype=float),
                )
        spectra_results = adaptive_spectra

    if adaptive_k_enabled and adaptive_k_mode == "transfer":
        """Refine the k quadrature from the evolved transfer functions."""

        adaptive_ell_indices = numpy.flatnonzero(
            ell_arr >= int(adaptive_k_min_ell)
        )[::adaptive_k_ell_stride]
        adaptive_spectra = {
            name: numpy.asarray(values, dtype=numpy.longdouble).copy()
            for name, values in spectra_results.items()
        }

        def _interpolate_transfer_batch(
            component_name: str,
            ell_indices: numpy.ndarray,
            local_k: numpy.ndarray,
        ) -> numpy.ndarray:
            """Evaluate cubic local k interpolants for one component batch."""

            matrix = numpy.asarray(
                transfer_components[component_name][ell_indices],
                dtype=float,
            )
            right_indices = numpy.searchsorted(
                k_values,
                local_k,
                side="left",
            )
            right_indices = numpy.clip(
                right_indices,
                2,
                int(k_values.size) - 2,
            )
            first_indices = right_indices - 2
            node_indices = first_indices[:, :, numpy.newaxis] + numpy.arange(
                4,
                dtype=int,
            )
            node_values = k_values[node_indices]
            query_values = local_k[:, :, numpy.newaxis]
            weights = numpy.ones_like(node_values, dtype=float)
            for node_index in range(4):
                other_indices = [
                    index for index in range(4) if index != node_index
                ]
                weights[:, :, node_index] = numpy.prod(
                    (query_values - node_values[:, :, other_indices])
                    / (
                        node_values[:, :, node_index, numpy.newaxis]
                        - node_values[:, :, other_indices]
                    ),
                    axis=2,
                )
            row_indices = numpy.arange(matrix.shape[0])[:, None, None]
            values = matrix[row_indices, node_indices]
            return numpy.sum(values * weights, axis=2)

        adaptive_batch_size = 64
        for batch_start in range(
            0,
            int(adaptive_ell_indices.size),
            adaptive_batch_size,
        ):
            batch_indices = adaptive_ell_indices[
                batch_start : batch_start + adaptive_batch_size
            ]
            ell_values = numpy.asarray(ell_arr[batch_indices], dtype=int)
            dense_k = numpy.geomspace(
                float(k_values[0]),
                float(k_values[-1]),
                max(256, adaptive_k_node_count),
                dtype=float,
            )
            local_k = numpy.broadcast_to(
                dense_k[numpy.newaxis, :],
                (ell_values.size, dense_k.size),
            )
            component_names = set(transfer_components)
            local_transfers = {
                name: _interpolate_transfer_batch(
                    name,
                    batch_indices,
                    local_k,
                )
                for name in component_names
            }
            for (
                observable_name,
                observable_entry,
            ) in power_spectrum_observables.items():
                primary_name = str(observable_entry.primary)
                secondary_name = str(observable_entry.secondary)
                if (
                    primary_name not in local_transfers
                    or secondary_name not in local_transfers
                ):
                    continue
                primordial_grid = _primordial_power_grid_for_observable(
                    physical_params=physical_params,
                    perturbation_data=perturbation_data,
                    observable_entry=observable_entry,
                    k_values=local_k,
                )
                for row_index, ell_index in enumerate(batch_indices):
                    adaptive_spectra[observable_name][ell_index] = (
                        _integrate_power_spectrum(
                            primordial_grid=primordial_grid[row_index],
                            log_k_values=numpy.log(local_k[row_index]),
                            primary=local_transfers[primary_name][row_index],
                            secondary=local_transfers[secondary_name][
                                row_index
                            ],
                            auto_spectrum=primary_name == secondary_name,
                        )
                    )[0]
        if adaptive_ell_indices.size >= 2:
            dense_indices = numpy.flatnonzero(
                ell_arr >= int(ell_arr[adaptive_ell_indices[0]])
            )
            for values in adaptive_spectra.values():
                values[dense_indices] = numpy.interp(
                    numpy.asarray(dense_indices, dtype=float),
                    numpy.asarray(adaptive_ell_indices, dtype=float),
                    numpy.asarray(values[adaptive_ell_indices], dtype=float),
                )
        spectra_results = adaptive_spectra

    elapsed_seconds = perf_counter() - request_started
    runtime_envelope["scalar_initial_constraint_preflight"] = (
        scalar_initial_constraint_preflight
    )
    runtime_envelope["scalar_constraint_projection"] = {
        "method": "source_history_coupled_einstein_reconstruction",
        "mode_count": int(scalar_constraint_projection_count),
        "diagnostic_mode_count": int(
            scalar_constraint_diagnostic_projection_count
        ),
        "maximum_relative_metric_correction": float(
            scalar_constraint_projection_max_relative_correction
        ),
    }
    runtime_envelope["scalar_constraint_diagnostics"] = (
        scalar_constraint_diagnostics
    )
    runtime_envelope["declared_source_history_mode_count"] = int(
        source_history_mode_count
    )
    runtime_envelope["declared_source_history_max_abs"] = dict(
        source_history_max_abs
    )
    runtime_envelope["declared_source_history_max_abs_by_k"] = {
        key: dict(value) for key, value in source_history_max_abs_by_k.items()
    }
    runtime_envelope["state_history_max_abs_by_k"] = {
        key: dict(value) for key, value in state_history_max_abs_by_k.items()
    }
    runtime_envelope["state_history_polarization_ratio_by_k"] = {
        key: dict(value)
        for key, value in state_history_polarization_ratio_by_k.items()
    }
    runtime_envelope["source_context_max_abs_by_k"] = {
        key: dict(value) for key, value in source_context_max_abs_by_k.items()
    }
    runtime_envelope["source_context_pre_resolution_max_abs_by_k"] = {
        key: dict(value)
        for key, value in source_context_pre_resolution_by_k.items()
    }
    runtime_envelope["metric_history_gradient_residual_by_k"] = {
        key: dict(value)
        for key, value in metric_history_gradient_residual_by_k.items()
    }
    derivative_validation = {
        name: max(
            (
                float(values.get(name, 0.0))
                for values in metric_history_gradient_residual_by_k.values()
            ),
            default=0.0,
        )
        for name in ("Phi_tau", "Psi_tau", "Phi_history_tau")
    }
    derivative_validation_finite = bool(
        all(
            numpy.isfinite(float(value))
            for residuals in metric_history_gradient_residual_by_k.values()
            for value in residuals.values()
        )
    )
    runtime_envelope["metric_history_derivative_validation"] = {
        "required": ("Phi_tau", "Psi_tau", "Phi_history_tau"),
        "mode_count": int(len(metric_history_gradient_residual_by_k)),
        "finite": derivative_validation_finite,
        "coordinate": "tau",
        "independent_history_gradients": True,
        "maximum_normalized_residual": derivative_validation,
    }
    runtime_envelope["source_history_residual_samples_by_k"] = {
        key: dict(value)
        for key, value in source_history_residual_samples_by_k.items()
    }
    runtime_envelope["hierarchy_equation_residuals_by_k"] = {
        key: dict(value)
        for key, value in hierarchy_equation_residuals_by_k.items()
    }
    runtime_envelope["initial_state_diagnostics_by_k"] = {
        key: dict(value)
        for key, value in initial_state_diagnostics_by_k.items()
    }
    runtime_envelope["source_history_residual_sample_schema"] = 1
    runtime_envelope["declared_source_history_convergence"] = {
        "sample_count": int(source_grids["eta"].size),
        "coarse_sample_count": int(source_eta_indices.size),
        "mode_count": int(source_history_mode_count),
        "refinement_mode_count": int(source_history_refinement_mode_count),
        "roles": declared_source_history_roles,
        "finite": True,
        "relative_error": float(source_history_error),
        "absolute_error": float(source_history_absolute_error),
        "tolerance_relative": float(
            adaptive_controls.source_relative_tolerance
        ),
        "tolerance_absolute": float(
            adaptive_controls.source_absolute_tolerance
        ),
    }
    if generated_scalar_hierarchy and source_history_residual_samples_by_k:
        # Keep the independent audit in the raw runtime envelope as well as
        # in the fixed-point diagnostic harness.  The import is local to
        # avoid coupling the projection module's import graph to diagnostics.
        from ..diagnostics import (
            audit_source_history_residuals,
            resolve_source_residual_audit_controls,
        )

        source_residual_audit_controls = (
            resolve_source_residual_audit_controls(declared_accuracy_controls)
        )
        runtime_envelope["source_residual_audit_controls"] = (
            source_residual_audit_controls
        )

        independent_source_audit = audit_source_history_residuals(
            runtime_envelope
        )
        runtime_envelope["independent_source_residual_audit"] = (
            independent_source_audit
        )
        if bool(
            declared_accuracy_controls.get(
                "require_physical_source_residuals", False
            )
        ) and not bool(independent_source_audit.get("converged", False)):
            raise ConvergenceError(
                "Generated CCMBS source histories failed the independent "
                "physical residual audit",
                context={
                    "audit": independent_source_audit,
                    "mode_count": int(source_history_mode_count),
                },
            )
    kernel_cache_after = cache.cmb_cache_stats()[
        "declared_projection_kernel_batch"
    ]
    runtime_envelope["projection_kernel_cache_hits"] = int(
        kernel_cache_after["hits"] - kernel_cache_before["hits"]
    )
    projection_sector_key = (
        ("all",)
        if streaming_projection_sectors is None
        else tuple(sorted(streaming_projection_sectors))
    )
    runtime_envelope["projection_kernel_cache_keys"] = tuple(
        (
            ell_signature,
            mode_projection_metadata[int(k_index)][1],
            projection_sector_key,
        )
        for k_index, kernel_batches in mode_kernel_batches.items()
        for ell_signature in kernel_batches
        if int(k_index) in mode_projection_metadata
    )
    runtime_envelope["projection_bessel_batch_count"] = int(bessel_batch_count)
    runtime_envelope["projection_bessel_mode_count"] = int(bessel_mode_count)
    runtime_envelope["projection_chunk_count"] = int(bessel_batch_count)
    runtime_envelope["projection_chunk_size"] = int(_BESSEL_MAX_MODE_BATCH)
    runtime_envelope["projection_chunk_accumulation_order"] = "k_index"
    runtime_envelope["projection_peak_bessel_cells"] = int(
        _BESSEL_WORK_CELL_BUDGET
    )
    timing_snapshot = performance_timer.snapshot(
        total_seconds=elapsed_seconds,
    )
    runtime_envelope.update(timing_snapshot)
    if transfer_cache_reuse_allowed:
        cache.set_cmb_transfer(
            transfer_cache_key,
            CustomCMBTransferData(
                ell_grid=ell_arr,
                k_grid=k_values,
                transfer_components=transfer_components,
                runtime_envelope=runtime_envelope,
            ),
        )
    spectrum_data = CustomCMBSpectrumData(
        ell_grid=ell_arr,
        k_grid=k_values,
        transfer_components=FrozenMapping(
            {name: matrix for name, matrix in transfer_components.items()}
        ),
        spectra=FrozenMapping(spectra_results),
        runtime_envelope=FrozenMapping(runtime_envelope),
        spectrum_availability=FrozenMapping(spectrum_availability),
    )
    cache.set_cmb_spectrum(cache_key, spectrum_data)
    return _get_cached_custom_cmb_spectrum_data(cache_key)


def _compute_custom_cmb_spectrum_data(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
    requested_spectra: Iterable[str] | None = None,
    workload: str = "full_spectrum",
) -> CustomCMBSpectrumData:
    """Execute one request and enforce its declared production rule."""

    timer = PhaseTimer()
    started = perf_counter()
    requested = (
        None
        if requested_spectra is None
        else tuple(str(name) for name in requested_spectra)
    )
    production_controls = None
    effective_requested_spectra = requested
    context = failure_context(
        contract_or_params,
        workload=workload,
        spectra=requested or (),
    )
    try:
        production_controls = resolve_production_scalar_convergence(
            contract_or_params
        )
        production_enforced = bool(
            production_controls.enabled
            and workload != "joint_mcmc"
            and not contract_or_params.get(
                "_diagnostic_matrix_fast_path", False
            )
        )
        if production_enforced and requested is not None:
            effective_requested_spectra = tuple(
                dict.fromkeys(requested + production_controls.required_spectra)
            )
        result = _compute_custom_cmb_spectrum_data_impl(
            contract_or_params,
            ells,
            background_provider=background_provider,
            requested_spectra=effective_requested_spectra,
            diagnostic_source_audit=workload.startswith(
                "fixed_parameter_diagnostic"
            ),
            performance_timer=timer,
        )
        production_record = result.runtime_envelope.get(
            "production_scalar_k_convergence"
        )
        if production_enforced and production_record is None:
            base_numerical = dict(
                contract_or_params.get("numerical", {}) or {}
            )
            base_k_count = int(base_numerical.get("k_sample_count", 0))
            if base_k_count < 1:
                raise ValueError(
                    "Production scalar convergence requires a positive "
                    "k_sample_count"
                )
            refined_contract = dict(contract_or_params)
            refined_contract["_k_grid_refinement_factor"] = int(
                production_controls.k_refinement_factor
            )
            refined_contract["_numerical_overrides"] = {
                "k_sample_count": (
                    base_k_count * production_controls.k_refinement_factor
                )
            }
            refined_timer = PhaseTimer()
            refinement_started = perf_counter()
            refined = _compute_custom_cmb_spectrum_data_impl(
                refined_contract,
                ells,
                background_provider=background_provider,
                requested_spectra=effective_requested_spectra,
                diagnostic_source_audit=workload.startswith(
                    "fixed_parameter_diagnostic"
                ),
                performance_timer=refined_timer,
            )
            report = evaluate_spectrum_refinement(
                result.spectra,
                refined.spectra,
                required_spectra=production_controls.required_spectra,
                relative_tolerances=(production_controls.relative_tolerances),
            )
            production_record = {
                "axis": "k_sample_count",
                "base_count": int(result.k_grid.size),
                "refined_count": int(refined.k_grid.size),
                "declared_base_count": base_k_count,
                "declared_refined_count": (
                    base_k_count * production_controls.k_refinement_factor
                ),
                "refinement_factor": production_controls.k_refinement_factor,
                "required_spectra": production_controls.required_spectra,
                "metrics": report.to_dict()["metrics"],
                "converged": report.converged,
                "fail_on_nonconvergence": (
                    production_controls.fail_on_nonconvergence
                ),
                "elapsed_seconds": perf_counter() - refinement_started,
                "refined_cache_state": refined_timer.cache_state,
            }
            enriched_envelope = dict(result.runtime_envelope)
            enriched_envelope["production_scalar_k_convergence"] = (
                production_record
            )
            result = CustomCMBSpectrumData(
                ell_grid=result.ell_grid,
                k_grid=result.k_grid,
                transfer_components=result.transfer_components,
                spectra=result.spectra,
                runtime_envelope=enriched_envelope,
                spectrum_availability=result.spectrum_availability,
            )
            cache_key = _custom_cmb_spectrum_cache_key(
                contract_or_params,
                ells,
                background_provider,
                requested_spectra=effective_requested_spectra,
            )
            cache.set_cmb_spectrum(cache_key, result)
        if (
            production_enforced
            and production_record is not None
            and not bool(production_record.get("converged", False))
            and production_controls.fail_on_nonconvergence
        ):
            raise ConvergenceError(
                "Production scalar CCMBS spectrum did not converge under "
                "the declared doubled k-grid",
                context={
                    "axis": "k_sample_count",
                    "base_count": production_record.get("base_count"),
                    "refined_count": production_record.get("refined_count"),
                    "metrics": production_record.get("metrics", {}),
                },
            )
        if production_enforced and requested is not None:
            requested_names = {
                canonical_cmb_spectrum_name(name) for name in requested
            }
            result = CustomCMBSpectrumData(
                ell_grid=result.ell_grid,
                k_grid=result.k_grid,
                transfer_components=result.transfer_components,
                spectra={
                    name: values
                    for name, values in result.spectra.items()
                    if canonical_cmb_spectrum_name(name) in requested_names
                },
                runtime_envelope=result.runtime_envelope,
                spectrum_availability=result.spectrum_availability,
            )
        elif production_controls.enabled and production_record is None:
            deferred_envelope = dict(result.runtime_envelope)
            deferred_envelope["production_scalar_k_convergence"] = {
                "status": "deferred",
                "workload": workload,
                "reason": (
                    "joint_mcmc uses the declared base grid; full-spectrum "
                    "and diagnostic workloads enforce doubled-grid closure"
                ),
                "required_spectra": production_controls.required_spectra,
                "fail_on_nonconvergence": (
                    production_controls.fail_on_nonconvergence
                ),
            }
            result = CustomCMBSpectrumData(
                ell_grid=result.ell_grid,
                k_grid=result.k_grid,
                transfer_components=result.transfer_components,
                spectra=result.spectra,
                runtime_envelope=deferred_envelope,
                spectrum_availability=result.spectrum_availability,
            )
        elapsed = perf_counter() - started
    # DEVCOV_ALLOW_BROAD_ONCE declared projection normalization boundary.
    except Exception as exc:
        elapsed = perf_counter() - started
        typed_error = classify_exception(exc, context=context)
        timing = timer.snapshot(total_seconds=elapsed)
        record = cache.record_cmb_performance(
            timing,
            cache_hit=timer.cache_state == "exact_cache_hit",
            workload=workload,
            cache_state=timer.cache_state,
            outcome="failure",
            stop_phase=timer.failed_phase,
            work_units=timer.work_units,
            failure=typed_error.diagnostic(),
            context=context,
        )
        typed_error.add_context(
            stop_phase=timer.failed_phase,
            performance_record=record,
        )
        if typed_error is exc:
            raise
        raise typed_error from exc

    timing = timer.snapshot(total_seconds=elapsed)
    cache.record_cmb_performance(
        timing,
        cache_hit=timer.cache_state == "exact_cache_hit",
        workload=workload,
        cache_state=timer.cache_state,
        work_units=timer.work_units,
        context=context,
    )
    return result
