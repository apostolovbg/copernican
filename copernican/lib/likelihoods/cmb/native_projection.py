r"""Declared native transfer projection and spectrum integration helpers."""

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

from ...cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_kernel_spec,
    resolve_declared_source_kernel,
    validate_declared_projection_sector,
)
from ...engine_adapter import FrozenMapping
from ...perturbation_contract import (
    PerturbationCollisionTargetSelectorData,
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)
from . import native_cache
from .native_adaptive import (
    NativeConvergenceEstimate,
    estimate_convergence,
    estimate_history_convergence,
    phase_aware_eta_grid,
    phase_aware_k_grid,
    require_convergence,
    resolve_native_adaptive_controls,
)
from .native_background import (
    _C_LIGHT_KM_S,
    _LEGACY_DECLARED_EVOLUTION_COORDINATES,
    CustomCMBSpectrumData,
    _accuracy_control_value,
    _build_custom_cmb_background,
    _coerce_numeric_scalar,
    _compute_spherical_bessel_batch,
    _compute_spherical_bessel_mode_batch,
    _custom_cmb_spectrum_cache_key,
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
from .native_evolution import (
    _COMPILED_CONTEXT_GLOBALS,
    _build_declared_base_context,
    _compile_declared_graph_execution_plan,
    _compile_declared_perturbation_contract,
    _compile_equation_program,
    _compile_ordered_context_program,
    _compute_tight_coupling_drag,
    _declared_momentum_grid_context,
    _declared_runtime_seed,
    _evaluate_declared_initial_state,
    _nonuniform_gradient,
    _resolve_declared_graph_context,
    _resolve_declared_graph_context_ordered,
    _resolve_declared_momentum_grid_runtimes,
    _tight_coupling_is_active,
    _validate_generated_scalar_initial_constraints,
    _validate_generated_vector_initial_constraints,
)
from .native_performance import (
    NativePhaseTimer,
    enforce_native_performance_budget,
    resolve_native_performance_budget,
)

_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}
_SCALAR_SUPERHORIZON_PREFIX_KETA = 5.0e-3
_BESSEL_WORK_CELL_BUDGET = 8_000_000
_BESSEL_MAX_MODE_BATCH = 16


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

    Auto spectra use a positive trapezoid fallback only when irregular-grid
    Simpson weights produce a negative roundoff artifact.
    """

    primordial_ld = numpy.asarray(primordial_grid, dtype=numpy.longdouble)
    log_k_ld = numpy.asarray(log_k_values, dtype=numpy.longdouble)
    primary_ld = numpy.asarray(primary, dtype=numpy.longdouble)
    secondary_ld = numpy.asarray(secondary, dtype=numpy.longdouble)
    if primary_ld.ndim == 1:
        primary_ld = primary_ld[numpy.newaxis, :]
    if secondary_ld.ndim == 1:
        secondary_ld = secondary_ld[numpy.newaxis, :]
    weighted = primordial_ld[numpy.newaxis, :] * (primary_ld * secondary_ld)
    simpson_integral = simpson(weighted, x=log_k_ld, axis=1)
    if auto_spectrum and numpy.any(simpson_integral < 0.0):
        trapezoid_integral = numpy.sum(
            0.5
            * (weighted[:, :-1] + weighted[:, 1:])
            * numpy.diff(log_k_ld)[numpy.newaxis, :],
            axis=1,
        )
        simpson_integral = numpy.where(
            simpson_integral < 0.0,
            trapezoid_integral,
            simpson_integral,
        )
    integrated = 4.0 * numpy.longdouble(math.pi) * simpson_integral
    # Keep the raw spectrum in extended precision until the public solver
    # applies its final float conversion. Simpson quadrature reduces the
    # leading log-k integration error on the nonuniform anchor grid.
    return numpy.asarray(integrated, dtype=numpy.longdouble)


def _configured_reference_ells(
    perturbation_data: Any,
    *,
    maximum_ell: int | None = None,
) -> tuple[int, ...]:
    """Return all declared reference multipoles for the native run."""

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
    """Return ell anchors that steer the native projection k-grid."""

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
) -> numpy.ndarray:
    """Return the fixed-count native projection k-grid for one run."""

    ell_values = numpy.asarray(ell_arr, dtype=int)
    sample_count = max(8, int(numerics.k_sample_count))
    configured_reference_ells = _configured_reference_ells(
        perturbation_data,
        maximum_ell=int(ell_values.max()),
    )
    grid_ell_min = min(
        int(ell_values.min()),
        int(numerics.ell_min),
        *configured_reference_ells,
    )
    grid_ell_max = max(
        (
            int(ell_values.max()),
            *configured_reference_ells,
        )
    )
    eta0_floor = max(float(background.eta0), 1.0e-6)
    k_min = max(
        float(numerics.k_min),
        0.2 * max(float(grid_ell_min), 2.0) / eta0_floor,
    )
    eta_rec_distance = max(
        float(background.eta0) - float(background.eta_rec),
        1.0,
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
    k_max = max(
        required_k_max,
        min(float(numerics.k_max), k_floor),
    )
    if not numpy.isfinite(k_min) or not numpy.isfinite(k_max):
        raise ValueError("Declared projection k-grid requires finite bounds")
    if k_max <= k_min:
        return numpy.linspace(k_min, k_min, sample_count, dtype=float)

    projection_ell_values = numpy.linspace(
        grid_ell_min,
        grid_ell_max,
        num=max(2, min(sample_count, grid_ell_max - grid_ell_min + 1)),
        dtype=int,
    )
    anchor_node_budget = max(
        2,
        sample_count // (4 if sample_count >= 256 else 2),
    )
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
    if sample_count >= 256 and k_max > k_min:
        low_k_max = min(
            k_max,
            max(
                k_min * 64.0,
                0.02,
            ),
        )
        low_node_count = max(16, sample_count // 8)
        k_nodes.update(
            float(value)
            for value in numpy.geomspace(
                k_min,
                low_k_max,
                num=low_node_count,
                dtype=float,
            )
        )
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
        ordered_nodes = [ordered_nodes[0], *interior_nodes, ordered_nodes[-1]]
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
        ordered_nodes.insert(widest_gap_index + 1, float(midpoint))
    return numpy.asarray(ordered_nodes, dtype=float)


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
        # The native tensor metric seed is h=1. CAMB/CLASS tensor power
        # conventions put the compensating 1/6 in the primordial spectrum.
        amplitude /= 6.0
        exponent = 0.0 if tensor_tilt is None else float(tensor_tilt)
    else:
        amplitude = float(physical_params.primordial_amplitude)
        exponent = float(physical_params.primordial_spectral_index) - 1.0
    return amplitude * numpy.power(k_values / 0.05, exponent)


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
    return numpy.concatenate(
        (numpy.asarray(refined, dtype=float), eta_grid[-1:]),
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
    """Return the validated runtime-envelope control mapping."""

    accuracy_controls = _resolve_declared_accuracy_controls(contract)
    runtime_envelope = accuracy_controls.get("runtime_envelope")
    if runtime_envelope is None:
        return {}
    if runtime_envelope == "bounded":
        return {
            "maximum_evolution_work_units": 100_000_000,
            "maximum_momentum_work_units": 10_000_000,
            "maximum_projection_work_units": 5_000_000_000,
            "maximum_total_work_units": 5_200_000_000,
        }
    if not isinstance(runtime_envelope, Mapping):
        raise ValueError(
            "cmb.perturbations.accuracy_controls.runtime_envelope must be "
            "a mapping or the preset 'bounded'"
        )
    return runtime_envelope


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
) -> dict[str, int]:
    """Return and validate the declared runtime envelope for one run."""

    evolution_work_units = int(
        evolution_multiplier * k_count * eta_count * max(state_slot_count, 1)
    )
    projection_work_units = int(
        ell_count * k_count * eta_count * max(transfer_component_count, 1)
    )
    momentum_work_units = int(max(momentum_point_count, 0) * eta_count)
    total_work_units = int(
        evolution_work_units + projection_work_units + momentum_work_units
    )
    envelope = {
        "ell_count": int(ell_count),
        "k_sample_count": int(k_count),
        "eta_sample_count": int(eta_count),
        "state_slot_count": int(state_slot_count),
        "transfer_component_count": int(transfer_component_count),
        "momentum_point_count": int(momentum_point_count),
        "evolution_work_units": evolution_work_units,
        "projection_work_units": projection_work_units,
        "momentum_work_units": momentum_work_units,
        "total_work_units": total_work_units,
    }
    runtime_envelope = _validate_runtime_envelope_controls(contract)
    for limit_name, work_name in (
        ("maximum_evolution_work_units", "evolution_work_units"),
        ("maximum_momentum_work_units", "momentum_work_units"),
        ("maximum_projection_work_units", "projection_work_units"),
        ("maximum_total_work_units", "total_work_units"),
    ):
        raw_limit = runtime_envelope.get(limit_name)
        if raw_limit is None:
            raw_limit = _accuracy_control_value(
                _resolve_declared_accuracy_controls(contract),
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
        if envelope[work_name] > limit_value:
            raise ValueError(
                "Declared runtime_envelope exceeded "
                f"{limit_name}: {envelope[work_name]} > {limit_value}"
            )
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


def _validate_scalar_constraint_histories(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    eta_grid: numpy.ndarray,
    accuracy_controls: Mapping[str, Any],
    k_value: float,
) -> dict[str, dict[str, Any]]:
    """Validate generated Einstein residuals and return anchor metrics."""

    residual_names = tuple(
        name for name in _SCALAR_CONSTRAINT_RESIDUALS if name in context
    )
    if not residual_names:
        return {}
    eta_values = numpy.asarray(eta_grid, dtype=float)
    if eta_values.ndim != 1 or eta_values.size == 0:
        raise ValueError("Scalar constraint validation requires an eta grid")
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

    tolerances = dict(_DEFAULT_SCALAR_CONSTRAINT_TOLERANCES)
    rule_enforced_residual_names: set[str] = set()
    for _rule_name, rule_entry in (
        getattr(perturbation_data, "conservation_rules", {}) or {}
    ).items():
        expression = str(getattr(rule_entry, "expression", ""))
        if expression in tolerances:
            tolerances[expression] = float(rule_entry.tolerance)
            rule_enforced_residual_names.add(expression)
    accuracy_enforced_residual_names: set[str] = set()
    raw_tolerances = accuracy_controls.get("scalar_constraint_tolerances")
    if raw_tolerances is not None:
        if not isinstance(raw_tolerances, Mapping):
            raise ValueError(
                "cmb.perturbations.accuracy_controls."
                "scalar_constraint_tolerances must be a mapping"
            )
        for residual_name, raw_tolerance in raw_tolerances.items():
            residual_key = str(residual_name)
            if residual_key not in tolerances:
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
            tolerances[residual_key] = float(tolerance)
            accuracy_enforced_residual_names.add(residual_key)

    diagnostics: dict[str, dict[str, Any]] = {}
    for residual_name in residual_names:
        values = numpy.asarray(context[residual_name], dtype=float)
        if values.ndim == 0:
            values = numpy.full_like(eta_values, float(values), dtype=float)
        if values.shape != eta_values.shape:
            raise ValueError(
                "Scalar Einstein residual has an invalid eta-grid shape: "
                f"{residual_name} at k={k_value}"
            )
        if not numpy.all(numpy.isfinite(values)):
            raise ValueError(
                "Scalar Einstein residual is non-finite: "
                f"{residual_name} at k={k_value}"
            )
        absolute_values = numpy.abs(values)
        tolerance = tolerances[residual_name]
        max_abs = float(numpy.max(absolute_values))
        enforcement_active = residual_name in rule_enforced_residual_names or (
            residual_name in accuracy_enforced_residual_names
            and reference_resolution_met
        )
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
        if enforcement_active and max_abs > tolerance:
            raise ValueError(
                "Scalar Einstein constraint exceeded tolerance: "
                f"{residual_name} at k={k_value} "
                f"({max_abs} > {tolerance})"
            )
        diagnostics[residual_name] = {
            "maximum_absolute": max_abs,
            "tolerance": float(tolerance),
            "enforced": enforcement_active,
            "reference_eta_samples": int(reference_count),
            "reference_resolution_met": bool(reference_resolution_met),
            "anchors": anchor_values,
            "sample_count": int(values.size),
        }
    return diagnostics


def _compute_custom_cmb_spectrum_data(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
    requested_spectra: Iterable[str] | None = None,
) -> CustomCMBSpectrumData:
    """Return transfer functions and spectra for a declared CMB graph."""

    performance_timer = NativePhaseTimer()
    request_started = perf_counter()
    requested_spectrum_names = None
    if requested_spectra is not None:
        requested_spectrum_names = {
            str(name).upper() for name in requested_spectra
        }
    cache_key = _custom_cmb_spectrum_cache_key(
        contract_or_params,
        ells,
        background_provider,
        requested_spectra=requested_spectrum_names,
    )
    cached_spectrum = native_cache.get_custom_cmb_spectrum(cache_key)
    if cached_spectrum is not None:
        native_cache.record_native_cmb_performance(
            {"total_seconds": 0.0},
            cache_hit=True,
        )
        return _get_cached_custom_cmb_spectrum_data(cache_key)

    graph_cache_before = native_cache.native_cmb_cache_stats()[
        "declared_graph_execution_plan"
    ]
    with performance_timer.phase("compilation"):
        perturbation_data = _compile_declared_perturbation_contract(
            contract_or_params
        )
        if perturbation_data.standard:
            raise ValueError("Standard perturbation contracts must use CAMB.")
        execution_plan = _compile_declared_graph_execution_plan(
            perturbation_data
        )
    value_steps_by_name = {
        str(step.output_name): step for step in execution_plan.value_steps
    }
    stage_required_names: set[str] = {
        str(dependency)
        for slot_plan in execution_plan.equation_slot_plans
        if slot_plan.compiled_rhs is not None
        for dependency in slot_plan.compiled_rhs.dependencies
    }
    stage_required_names.update(
        {
            "einstein_energy_residual",
            "einstein_momentum_residual",
            "einstein_shear_residual",
            "total_density_source",
            "total_momentum_source",
            "total_shear_source",
        }
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
    background_cache_before = native_cache.native_cmb_cache_stats()[
        "custom_background"
    ]
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
    adaptive_controls = resolve_native_adaptive_controls(
        _resolve_declared_accuracy_controls(contract_or_params),
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
    if (
        generated_scalar_hierarchy
        and eta_los_refinement > 1
        and not adaptive_controls.source_enabled
    ):
        eta_los_grid = _limit_eta_grid(
            eta_los_grid,
            maximum_samples=minimum_eta_samples,
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
    shared_generated_mode_grids_enabled = (
        numerics.evolution_eta_sample_count is not None
    )

    with performance_timer.phase("preparation"):
        k_values = _build_projection_k_grid(
            ell_arr=ell_arr,
            background=background,
            numerics=numerics,
            perturbation_data=perturbation_data,
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
            )

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

    all_power_spectrum_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "angular_power_spectrum"
    }
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
    transfer_component_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "transfer_component"
        and (
            requested_spectrum_names is None
            or name in required_transfer_components
        )
    }
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
    source_history_mode_count = 0

    def _record_source_history_diagnostics(
        source_arrays: Mapping[str, numpy.ndarray],
    ) -> None:
        """Record finite declared source histories without copying them."""

        nonlocal source_history_mode_count
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
                source_history_max_abs[f"{component_name}:{role_name}"] = max(
                    source_history_max_abs[f"{component_name}:{role_name}"],
                    float(numpy.max(numpy.abs(history), initial=0.0)),
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
        evolution_multiplier=(2 if adaptive_controls.evolution_enabled else 1),
    )
    runtime_envelope["static_graph_preparations"] = 1
    runtime_envelope["contract_static_preparations"] = 1
    runtime_envelope["cosmology_static_preparations"] = 1
    runtime_envelope["request_specific_preparations"] = 1
    runtime_envelope["dynamic_mode_count"] = int(k_values.size)
    runtime_envelope["batch_count"] = 0
    runtime_envelope["batch_mode_count"] = 0
    runtime_envelope["batched_rk_stage_count"] = 0
    runtime_envelope["batched_max_substeps"] = 0
    graph_cache_after = native_cache.native_cmb_cache_stats()[
        "declared_graph_execution_plan"
    ]
    background_cache_after = native_cache.native_cmb_cache_stats()[
        "custom_background"
    ]
    runtime_envelope["graph_plan_cache_hit"] = bool(
        graph_cache_after["hits"] > graph_cache_before["hits"]
    )
    runtime_envelope["background_cache_hit"] = bool(
        background_cache_after["misses"] == background_cache_before["misses"]
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
    transfer_components = {
        name: numpy.zeros((ell_arr.size, k_values.size), dtype=float)
        for name in transfer_component_observables
    }
    declared_accuracy_controls = _resolve_declared_accuracy_controls(
        contract_or_params
    )
    scalar_constraint_diagnostics: dict[str, dict[str, Any]] = {}
    adaptive_k_controls = declared_accuracy_controls.get(
        "adaptive_k_quadrature"
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
        tuple[int, float], tuple[float, dict[str, float]]
    ] = {}
    momentum_grid_context_cache: dict[float, dict[str, Any]] = {}

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

        context_key = (int(step_index), float(blend))
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
            if requested_samples is None:
                if int(numerics.source_grid_multiplier) <= 1:
                    return base_grid
                requested_samples = max(
                    192,
                    min(256, int(numerics.eta_sample_count)),
                )
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
    ) -> dict[str, numpy.ndarray]:
        """Return source arrays keyed by source-term name."""

        source_arrays: dict[str, numpy.ndarray] = {}
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for source_step in execution_plan.source_steps:
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
            suppressed_collision_outputs = None
        else:
            suppressed_collision_outputs = {
                runtime.name: 0.0
                for runtime in split_collision_runtimes
                if (
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
                # security-scanner: allow validated declared program execution.
                exec(  # nosec B102 - expressions passed AST validation.
                    equation_program,
                    _COMPILED_CONTEXT_GLOBALS,
                    {
                        "context": scalar_context,
                        "state_vector": effective_state_vector,
                        "derivative": derivative,
                        "coordinate_rates": coordinate_rates,
                    },
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

    def _compile_value_program(value_steps: tuple[Any, ...]) -> Any | None:
        """Compile one reusable direct-assignment context program."""

        if not value_steps:
            return None
        return _compile_ordered_context_program(
            tuple(
                (
                    str(step.output_name),
                    str(step.compiled_expression.expression),
                )
                for step in value_steps
            )
        )

    full_context_program = _compile_value_program(execution_plan.value_steps)
    state_independent_context_program = _compile_value_program(
        state_independent_value_steps
    )
    state_dependent_context_program = _compile_value_program(
        state_dependent_value_steps
    )
    stage_context_program = _compile_value_program(stage_value_steps)
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

    def _prepare_mode_initial_state(
        mode_k_value: float,
    ) -> tuple[numpy.ndarray, set[tuple[str, str, int]]]:
        """Prepare one mode state without entering its evolution path."""

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
        _validate_generated_scalar_initial_constraints(
            perturbation_data=perturbation_data,
            context=initial_state_context,
            k_value=float(mode_k_value),
        )
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
        return state, assigned_targets

    def _evaluate_source_histories(
        mode_k_value: float,
        source_histories: Mapping[str, numpy.ndarray],
        *,
        collect_diagnostics: bool = True,
        source_grid_indices: numpy.ndarray | None = None,
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
        array_context = _build_array_context(
            evaluation_histories,
            k_value=float(mode_k_value),
        )
        source_arrays = _evaluate_declared_sources(
            array_context,
            k_value=float(mode_k_value),
        )
        conservation_context = dict(array_context)
        conservation_context.update(source_arrays)
        conservation_context = _resolve_declared_graph_context(
            conservation_context,
            perturbation_data,
            allow_partial=True,
            eta_grid=source_grids["eta"],
            execution_plan=execution_plan,
        )
        mode_constraint_diagnostics = _validate_scalar_constraint_histories(
            perturbation_data=perturbation_data,
            context=conservation_context,
            eta_grid=source_grids["eta"],
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
                        "tolerance": float(mode_metrics["tolerance"]),
                        "anchors": {},
                        "mode_count": 0,
                        "sample_count": 0,
                    },
                )
                aggregate["maximum_absolute"] = max(
                    float(aggregate["maximum_absolute"]),
                    float(mode_metrics["maximum_absolute"]),
                )
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
        ) -> numpy.ndarray:
            """Return one state vector after the split collision sub-step."""

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
            target_stage_scale = float(numerics.evolution_phase_step)
            required_substeps = max(
                1,
                int(
                    math.ceil(
                        abs(float(dt)) * stiffness_scale / target_stage_scale
                    )
                ),
            )
            if not (tight_coupling_active and split_collision_runtimes):
                start_collision_rate = float(
                    active_grids["collision_rate"][step_index]
                )
                end_collision_rate = float(
                    active_grids["collision_rate"][step_index + 1]
                )
                collision_scale = max(
                    start_collision_rate,
                    end_collision_rate,
                    0.0,
                )
                required_substeps = max(
                    required_substeps,
                    int(math.ceil(abs(float(dt)) * collision_scale / 0.25)),
                )
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
                continuous_collision_control
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

        state, assigned_targets = _prepare_mode_initial_state(float(k_value))
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
        return source_histories, source_arrays

    log_k_values = numpy.log(k_values)
    projection_ell_batch_size = 512 if use_streaming_projection else 128
    kernel_cache_before = native_cache.native_cmb_cache_stats()[
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
    evolution_mode_count = 0
    evolution_fine_sample_count = 0
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
    projection_trapezoid_weights = None
    if adaptive_controls.source_enabled and source_eta_indices.size >= 3:
        source_coarse_weights = _simpson_weights(
            source_grids["eta"][source_eta_indices]
        )
    if adaptive_controls.projection_enabled:
        projection_trapezoid_weights = _trapezoid_weights(source_grids["eta"])

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
            native_cache.store_bessel_inputs(
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
                cached = native_cache.get_declared_projection_kernel_batch(
                    (ell_signature, x_signature)
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
            mode_source_arrays: dict[int, dict[str, numpy.ndarray]] = {}
            for k_index, k_value in enumerate(k_values):
                with performance_timer.phase("evolution"):
                    _, source_arrays = _evolve_declared_mode(float(k_value))
                _record_source_history_diagnostics(source_arrays)
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
                        _,
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
                                precomputed_bessel=(
                                    precomputed_projection_bessel
                                ),
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
        with performance_timer.phase("evolution"):
            _, source_arrays = _evolve_declared_mode(
                float(k_value),
                history_sink=base_history_sink,
            )
            _record_source_history_diagnostics(source_arrays)
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
                coarse_sample_count = max(32, fine_sample_count // 2)
                if coarse_sample_count >= fine_sample_count:
                    raise ValueError(
                        "adaptive_evolution requires a refinable "
                        "evolution_eta_sample_count"
                    )
                coarse_history_sink: dict[str, Any] = {}
                _evolve_declared_mode(
                    float(k_value),
                    evolution_sample_count_override=coarse_sample_count,
                    history_sink=coarse_history_sink,
                    collect_diagnostics=False,
                )
                state_estimate = estimate_history_convergence(
                    coarse_history_sink["evolution_eta"],
                    coarse_history_sink["evolution_histories"],
                    base_history_sink["evolution_eta"],
                    base_history_sink["evolution_histories"],
                    relative_tolerance=(
                        adaptive_controls.evolution_relative_tolerance
                    ),
                    absolute_tolerance=(
                        adaptive_controls.evolution_absolute_tolerance
                    ),
                )
                source_estimate = estimate_history_convergence(
                    coarse_history_sink["source_eta"],
                    coarse_history_sink["source_histories"],
                    base_history_sink["source_eta"],
                    base_history_sink["source_histories"],
                    relative_tolerance=(
                        adaptive_controls.evolution_relative_tolerance
                    ),
                    absolute_tolerance=(
                        adaptive_controls.evolution_absolute_tolerance
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
                "Native projection did not prepare any radial kernel batches"
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
                    "Native projection radial kernel batch was not cached"
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
                if source_coarse_weights is not None:
                    coarse_kernel_batch = _slice_projection_kernel_batch(
                        kernel_batch,
                        source_eta_indices,
                    )
                    coarse_histories = {
                        role_name: history[source_eta_indices]
                        for role_name, history in source_histories.items()
                    }
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
                        eta_weights=source_coarse_weights,
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
                if projection_trapezoid_weights is not None:
                    trapezoid_values = _declared_graph_projection(
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
                        eta_weights=projection_trapezoid_weights,
                        chi_grid=source_grids["chi"],
                        source_chi=source_chi,
                        source_histories=source_histories,
                    )
                    estimate = estimate_convergence(
                        trapezoid_values,
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
        source_estimate = NativeConvergenceEstimate(
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
        projection_estimate = NativeConvergenceEstimate(
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
        runtime_envelope["adaptive_evolution_refinement_levels"] = 1
        runtime_envelope["scalar_evolution_convergence"] = {
            "relative_error": float(evolution_error),
            "absolute_error": float(evolution_absolute_error),
            "anchor_relative_errors": dict(evolution_anchor_errors),
            "anchor_absolute_errors": dict(evolution_anchor_absolute_errors),
            "mode_count": int(evolution_mode_count),
            "fine_sample_count": int(evolution_fine_sample_count),
            "coarse_sample_count": int(evolution_coarse_sample_count),
            "relative_tolerance": float(
                adaptive_controls.evolution_relative_tolerance
            ),
            "absolute_tolerance": float(
                adaptive_controls.evolution_absolute_tolerance
            ),
        }
        evolution_estimate = NativeConvergenceEstimate(
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

    if adaptive_k_enabled and adaptive_k_mode == "source":
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
                2 if adaptive_controls.evolution_enabled else 1
            ),
        )
        direct_envelope["static_graph_preparations"] = 1
        direct_envelope["contract_static_preparations"] = 1
        direct_envelope["cosmology_static_preparations"] = 1
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
            _record_source_history_diagnostics(direct_source_arrays)
            x_values = float(direct_k_value) * (eta0 - source_grids["eta"])
            x_signature = hashlib.sha256(
                numpy.asarray(x_values, dtype=float).tobytes()
            ).hexdigest()
            native_cache.store_bessel_inputs(
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
    runtime_envelope["scalar_constraint_diagnostics"] = (
        scalar_constraint_diagnostics
    )
    runtime_envelope["declared_source_history_mode_count"] = int(
        source_history_mode_count
    )
    runtime_envelope["declared_source_history_max_abs"] = dict(
        source_history_max_abs
    )
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
    kernel_cache_after = native_cache.native_cmb_cache_stats()[
        "declared_projection_kernel_batch"
    ]
    runtime_envelope["projection_kernel_cache_hits"] = int(
        kernel_cache_after["hits"] - kernel_cache_before["hits"]
    )
    runtime_envelope["projection_bessel_batch_count"] = int(bessel_batch_count)
    runtime_envelope["projection_bessel_mode_count"] = int(bessel_mode_count)
    performance_budget = resolve_native_performance_budget(
        declared_accuracy_controls
    )
    enforce_native_performance_budget(
        elapsed_seconds,
        workload="full_spectrum",
        budget=performance_budget,
    )
    timing_snapshot = performance_timer.snapshot(
        total_seconds=elapsed_seconds,
    )
    runtime_envelope.update(timing_snapshot)
    if performance_budget is not None:
        runtime_envelope.update(
            {
                "performance_budget_full_spectrum_seconds": float(
                    performance_budget.full_spectrum_seconds
                ),
                "performance_budget_joint_mcmc_seconds": float(
                    performance_budget.joint_mcmc_seconds
                ),
            }
        )
    native_cache.record_native_cmb_performance(timing_snapshot)

    spectrum_data = CustomCMBSpectrumData(
        ell_grid=ell_arr,
        k_grid=k_values,
        transfer_components=FrozenMapping(
            {name: matrix for name, matrix in transfer_components.items()}
        ),
        spectra=FrozenMapping(spectra_results),
        runtime_envelope=FrozenMapping(runtime_envelope),
    )
    native_cache.set_custom_cmb_spectrum(cache_key, spectrum_data)
    return _get_cached_custom_cmb_spectrum_data(cache_key)
