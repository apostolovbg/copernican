"""Declared collision operators for CCMBS mode evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy
from scipy.linalg import expm

from ....perturbation_contract import PerturbationCollisionTargetSelectorData


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
    eigendecomposition_cache: (
        dict[
            tuple[tuple[int, ...], bytes],
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None,
        ]
        | None
    ) = None,
    kernel_metrics: dict[str, Any] | None = None,
) -> numpy.ndarray:
    """Return exact collision updates for compatible mode-row matrices.

    The declared scalar hierarchy uses independent one- and two-state
    collision blocks.  Evaluate those blocks together so a shared Fourier
    batch does not repeat a small eigensystem decomposition for every row.
    Unstructured declarations use grouped matrix actions and a reusable
    eigensystem cache before retaining the scalar exact operator as a
    numerically safe fallback.
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

    kernel_started = perf_counter()

    def _update_digest(hasher: Any, values: numpy.ndarray) -> None:
        """Append one shape-aware raw array payload to a kernel digest."""

        array = numpy.ascontiguousarray(numpy.asarray(values, dtype=float))
        hasher.update(repr(array.shape).encode("ascii"))
        hasher.update(array.tobytes())

    def _finish(result: numpy.ndarray, *, vectorized: bool) -> numpy.ndarray:
        """Record bounded kernel evidence and return a finite array."""

        normalized = numpy.asarray(result, dtype=float)
        if kernel_metrics is not None:
            kernel_metrics["exact_vectorized_calls"] += int(vectorized)
            kernel_metrics["exact_scalar_fallback_calls"] += int(
                not vectorized
            )
            kernel_metrics["result_array_allocations"] += 1
            kernel_metrics["elapsed_seconds"] += (
                perf_counter() - kernel_started
            )
            input_digest = kernel_metrics.get("input_digest")
            output_digest = kernel_metrics.get("output_digest")
            digest_sample_count = int(
                kernel_metrics.get("digest_sample_count", 0)
            )
            digest_sample_limit = int(
                kernel_metrics.get("digest_sample_limit", 0)
            )
            if digest_sample_count < digest_sample_limit:
                if input_digest is not None:
                    _update_digest(input_digest, matrices)
                    _update_digest(input_digest, states)
                    _update_digest(input_digest, scales)
                if output_digest is not None:
                    _update_digest(output_digest, normalized)
                kernel_metrics["digest_sample_count"] = digest_sample_count + 1
        return normalized

    if kernel_metrics is not None:
        kernel_metrics["exact_batch_calls"] += 1
        kernel_metrics["exact_batch_mode_rows"] += int(mode_count)
    if float(dt) == 0.0 or state_count == 0:
        return _finish(states.copy(), vectorized=True)

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
                return _finish(
                    numpy.concatenate((leading, trailing), axis=1),
                    vectorized=True,
                )

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
        return _finish(batched_result, vectorized=True)

    fallback_result = numpy.empty_like(states, dtype=float)
    matrix_groups: dict[tuple[tuple[int, ...], bytes], list[int]] = {}
    for row_index, matrix in enumerate(matrices):
        matrix_key = (
            tuple(int(size) for size in matrix.shape),
            matrix.tobytes(),
        )
        matrix_groups.setdefault(matrix_key, []).append(int(row_index))
    if kernel_metrics is not None:
        kernel_metrics["fallback_matrix_groups"] += int(len(matrix_groups))

    scalar_fallback_rows = 0
    for row_indices in matrix_groups.values():
        indices = numpy.asarray(row_indices, dtype=int)
        matrix = matrices[int(indices[0])]
        components = _structured_collision_components(matrix)
        decomposition = None
        if components is None:
            if kernel_metrics is not None:
                kernel_metrics["eigendecomposition_lookups"] += 1
            cache_key = (
                tuple(int(size) for size in matrix.shape),
                matrix.tobytes(),
            )
            cache_was_populated = bool(
                eigendecomposition_cache is not None
                and cache_key in eigendecomposition_cache
            )
            if eigendecomposition_cache is not None:
                decomposition = _cached_collision_eigendecomposition(
                    matrix,
                    eigendecomposition_cache,
                )
                if kernel_metrics is not None:
                    kernel_metrics["eigendecomposition_cache_entries"] = len(
                        eigendecomposition_cache
                    )
                    kernel_metrics["eigendecomposition_cache_hits"] += int(
                        cache_was_populated
                    )
            if decomposition is not None:
                eigenvalues, eigenvectors, eigenvector_inverse = decomposition
                scaled_eigenvalues = (
                    eigenvalues[numpy.newaxis, :]
                    * scales[indices, numpy.newaxis]
                    * float(dt)
                )
                modal_states = states[indices] @ eigenvector_inverse.T
                evolved = (modal_states * numpy.exp(scaled_eigenvalues)) @ (
                    eigenvectors.T
                )
                real_evolved = numpy.real_if_close(evolved, tol=1000)
                if not numpy.iscomplexobj(real_evolved) and numpy.all(
                    numpy.isfinite(real_evolved)
                ):
                    fallback_result[indices] = numpy.asarray(
                        real_evolved,
                        dtype=float,
                    )
                    continue
            scalar_fallback_rows += len(row_indices)
        else:
            scalar_fallback_rows += len(row_indices)

        for row_index in row_indices:
            fallback_result[int(row_index)] = _exact_linear_collision_step(
                operator_matrix=matrices[int(row_index)],
                dt=float(dt),
                target_state=states[int(row_index)],
                eigendecomposition=decomposition,
                operator_scale=float(scales[int(row_index)]),
            )
    if kernel_metrics is not None:
        kernel_metrics["exact_scalar_fallback_rows"] += int(
            scalar_fallback_rows
        )
    return _finish(fallback_result, vectorized=False)


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
