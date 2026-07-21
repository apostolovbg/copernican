r"""Declared native transfer projection and spectrum integration helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gammaln, spherical_jn

from ...cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_kernel_spec,
)
from ...engine_adapter import FrozenMapping
from ...perturbation_contract import (
    PerturbationCollisionTargetSelectorData,
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)
from . import native_cache
from .native_background import (
    _C_LIGHT_KM_S,
    _LEGACY_DECLARED_EVOLUTION_COORDINATES,
    CustomCMBSpectrumData,
    _accuracy_control_value,
    _build_custom_cmb_background,
    _coerce_numeric_scalar,
    _compute_spherical_bessel_batch,
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
    _compute_tight_coupling_drag,
    _declared_momentum_grid_context,
    _declared_runtime_seed,
    _evaluate_declared_initial_state,
    _integrate_batched_rk4,
    _nonuniform_gradient,
    _resolve_declared_graph_context,
    _resolve_declared_graph_context_ordered,
    _resolve_declared_momentum_grid_runtimes,
    _tight_coupling_is_active,
    _validate_generated_scalar_initial_constraints,
    _validate_generated_vector_initial_constraints,
)

_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}
_SCALAR_SUPERHORIZON_PREFIX_KETA = 5.0e-3


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
    conservation_rule_names: tuple[str, ...] = ()


def _generated_scalar_tight_coupling_multipoles(
    *,
    photon_dipole: float,
    baryon_velocity_divergence: float,
    baryon_loading: float,
    k_value: float,
    collision_rate: float,
) -> tuple[float, float, float, float, float]:
    """Return the first-order scalar Thomson tight-coupling moments.

    The returned values are the common photon-baryon dipole, photon
    temperature quadrupole and octopole, and their E-polarization partners.
    The quadrupole follows the generated scalar hierarchy and its exact
    Thomson block: the leading collision balance is
    ``(2/5) k Theta_1 - (3/4) opacity Theta_2 = 0``.
    """

    if not numpy.all(
        numpy.isfinite(
            (
                photon_dipole,
                baryon_velocity_divergence,
                baryon_loading,
                k_value,
                collision_rate,
            )
        )
    ):
        raise ValueError("Scalar tight-coupling inputs must be finite")
    if collision_rate <= 1.0e-12 or abs(float(k_value)) <= 1.0e-12:
        return float(photon_dipole), 0.0, 0.0, 0.0, 0.0
    common_dipole = (
        float(photon_dipole)
        + max(float(baryon_loading), 1.0e-12)
        * float(baryon_velocity_divergence)
        / (3.0 * float(k_value))
    ) / (1.0 + max(float(baryon_loading), 1.0e-12))
    temperature_quadrupole = (
        8.0 / 15.0 * float(k_value) * common_dipole / float(collision_rate)
    )
    temperature_octopole = (
        3.0
        / 7.0
        * float(k_value)
        / float(collision_rate)
        * temperature_quadrupole
    )
    return (
        common_dipole,
        temperature_quadrupole,
        temperature_octopole,
        temperature_quadrupole / 4.0,
        temperature_octopole / 4.0,
    )


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
                conservation_rule_names=conservation_rule_names,
            )
        )
    return tuple(compiled_runtimes)


def _integrate_power_spectrum(
    primordial_grid: numpy.ndarray,
    log_k_values: numpy.ndarray,
    primary: numpy.ndarray,
    secondary: numpy.ndarray,
) -> numpy.ndarray:
    """Return one finite power-spectrum quadrature in extended precision."""

    primordial_ld = numpy.asarray(primordial_grid, dtype=numpy.longdouble)
    log_k_ld = numpy.asarray(log_k_values, dtype=numpy.longdouble)
    primary_ld = numpy.asarray(primary, dtype=numpy.longdouble)
    secondary_ld = numpy.asarray(secondary, dtype=numpy.longdouble)
    if primary_ld.ndim == 1:
        primary_ld = primary_ld[numpy.newaxis, :]
    if secondary_ld.ndim == 1:
        secondary_ld = secondary_ld[numpy.newaxis, :]
    weighted = primordial_ld[numpy.newaxis, :] * (primary_ld * secondary_ld)
    integrated = (
        4.0 * numpy.longdouble(math.pi) * simpson(weighted, x=log_k_ld, axis=1)
    )
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
        k_floor = max(12.0 * k_min, 2.5 * required_k_max)
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
    anchor_ells = _projection_anchor_ells(
        projection_ell_values,
        perturbation_data=perturbation_data,
        node_budget=max(2, sample_count // 2),
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
        log_nodes = numpy.log(numpy.asarray(ordered_nodes, dtype=float))
        widest_gap_index = int(numpy.argmax(numpy.diff(log_nodes)))
        midpoint = float(
            math.exp(
                0.5
                * (
                    log_nodes[widest_gap_index]
                    + log_nodes[widest_gap_index + 1]
                )
            )
        )
        if (
            not numpy.isfinite(midpoint)
            or midpoint <= ordered_nodes[widest_gap_index]
            or midpoint >= ordered_nodes[widest_gap_index + 1]
        ):
            midpoint = 0.5 * (
                ordered_nodes[widest_gap_index]
                + ordered_nodes[widest_gap_index + 1]
            )
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
    if (
        scaled_matrix.shape == (4, 4)
        and numpy.all(scaled_matrix[:2, 2:] == 0.0)
        and numpy.all(scaled_matrix[2:, :2] == 0.0)
    ):
        first = _exact_linear_collision_step(
            operator_matrix=scaled_matrix[:2, :2],
            dt=1.0,
            target_state=state[:2],
        )
        second = _exact_linear_collision_step(
            operator_matrix=scaled_matrix[2:, 2:],
            dt=1.0,
            target_state=state[2:],
        )
        result = numpy.concatenate((first, second))
        if numpy.all(numpy.isfinite(result)):
            return result
        return None
    adjacency = scaled_matrix != 0.0
    numpy.fill_diagonal(adjacency, False)
    components: list[tuple[int, ...]] = []
    unseen = set(range(scaled_matrix.shape[0]))
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

    j_l = kernel_batch.j_l
    j_l_derivative = kernel_batch.j_l_derivative
    j_l_second_derivative = kernel_batch.j_l_second_derivative
    e_kernel = kernel_batch.e_kernel
    b_kernel = kernel_batch.b_kernel
    sector_name = "" if sector is None else str(sector)

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
        if kernel_spec.kind == "spin2_e":
            return e_projection_kernel
        if kernel_spec.kind == "spin2_b":
            return b_projection_kernel
        if kernel_spec.kind == "lensing_potential":
            geometry = numpy.clip(source_chi - chi_grid, 0.0, None) / (
                max(float(source_chi), 1.0e-12)
                * numpy.maximum(chi_grid, 1.0e-12)
            )
            return 2.0 * j_l * geometry[numpy.newaxis, :]
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

    def _sum_projected_sources(kernel_name: str) -> numpy.ndarray:
        """Project every declared source through one shared kernel."""

        kernel_values = _apply_kernel(kernel_name)
        source = numpy.zeros_like(eta_weights, dtype=float)
        for history in source_histories.values():
            source += history
        return _project_history(kernel_values, source)

    if projection == "line_of_sight_temperature":
        projected = numpy.zeros(j_l.shape[0], dtype=float)
        if "monopole" in source_histories:
            projected += _project_history(
                j_l,
                source_histories["monopole"],
            )
        if "doppler" in source_histories:
            projected += _project_history(
                j_l_derivative,
                source_histories["doppler"],
            )
        if "isw" in source_histories:
            projected += _project_history(
                j_l,
                source_histories["isw"],
            )
        if "additive" in source_histories:
            projected += _project_history(
                j_l,
                source_histories["additive"],
            )
        if "additive_derivative" in source_histories:
            projected += _project_history(
                j_l_second_derivative,
                source_histories["additive_derivative"],
            )
        return projected
    if projection in {
        "line_of_sight_polarization_e",
        "line_of_sight_signal",
        "line_of_sight_signal_derivative",
        "line_of_sight_vector_polarization_b",
        "line_of_sight_vector_polarization_e",
        "line_of_sight_vector_temperature",
        "spin2_e_mode",
        "spin2_b_mode",
        "line_of_sight_potential",
        "line_of_sight_lensing_potential",
        "custom_line_of_sight",
    }:
        if kernel is None:
            raise ValueError(
                f"Declared observable projection '{projection}' did not "
                "resolve a kernel."
            )
        return _sum_projected_sources(kernel)
    if projection in SUPPORTED_DECLARED_TRANSFER_PROJECTIONS:
        raise ValueError(
            "Declared observable projection dispatch is incomplete for "
            f"'{projection}'"
        )
    raise ValueError(
        "Declared observable requests unsupported projection "
        f"'{projection}'"
    )


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
) -> dict[str, int]:
    """Return and validate the declared runtime envelope for one run."""

    evolution_work_units = int(k_count * eta_count * max(state_slot_count, 1))
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
) -> None:
    """Raise when one declared conservation rule exceeds its tolerance."""

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
    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for rule_name, rule_entry in rule_entries.items():
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


def _compute_custom_cmb_spectrum_data(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
    requested_spectra: Iterable[str] | None = None,
) -> CustomCMBSpectrumData:
    """Return transfer functions and spectra for a declared CMB graph."""

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
        return _get_cached_custom_cmb_spectrum_data(cache_key)

    perturbation_data = _compile_declared_perturbation_contract(
        contract_or_params
    )
    if perturbation_data.standard:
        raise ValueError("Standard perturbation contracts must use CAMB.")

    execution_plan = _compile_declared_graph_execution_plan(perturbation_data)
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
    if generated_scalar_hierarchy and eta_los_refinement > 1:
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

    (
        source_grids,
        source_declared_background_histories,
        source_coordinate_rate_histories,
    ) = _sample_eta_background_grids(eta_los_grid)
    active_grids = dict(source_grids)
    active_declared_background_histories = source_declared_background_histories
    active_coordinate_rate_histories = source_coordinate_rate_histories
    active_k_value = 0.0

    k_values = _build_projection_k_grid(
        ell_arr=ell_arr,
        background=background,
        numerics=numerics,
        perturbation_data=perturbation_data,
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
    transfer_components = {
        name: numpy.zeros((ell_arr.size, k_values.size), dtype=float)
        for name in transfer_component_observables
    }
    adaptive_k_controls = _resolve_declared_accuracy_controls(
        contract_or_params
    ).get("adaptive_k_quadrature")
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
    scalar_temperature_slot_indices: dict[int, int] = {}
    scalar_polarization_slot_indices: dict[int, int] = {}
    for slot in runtime_spec.state_slots:
        if slot.order != 0:
            continue
        if slot.variable.startswith("theta_gamma"):
            suffix = slot.variable[len("theta_gamma") :]
            if suffix.isdigit():
                scalar_temperature_slot_indices[int(suffix)] = int(slot.index)
        if slot.variable.startswith("e_gamma"):
            suffix = slot.variable[len("e_gamma") :]
            if suffix.isdigit():
                scalar_polarization_slot_indices[int(suffix)] = int(slot.index)
    scalar_tight_coupling_closure_indices = tuple(
        int(slot_index)
        for moment, slot_index in sorted(
            {
                **{
                    moment: slot_index
                    for moment, slot_index in (
                        scalar_temperature_slot_indices.items()
                    )
                    if moment >= 2
                },
                **{
                    1000 + moment: slot_index
                    for moment, slot_index in (
                        scalar_polarization_slot_indices.items()
                    )
                },
            }.items()
        )
    )
    scalar_base_context_cache: dict[
        tuple[float, tuple[tuple[str, float], ...]],
        dict[str, Any],
    ] = {}
    theta_gamma1_index = scalar_temperature_slot_indices.get(1)
    theta_gamma2_index = scalar_temperature_slot_indices.get(2)
    theta_b_index = runtime_spec.state_index_by_key.get(("theta_b", "tau", 0))
    e_gamma2_index = scalar_polarization_slot_indices.get(2)

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

    def _apply_generated_scalar_tight_coupling_closure(
        state_vector: numpy.ndarray,
        *,
        step_index: int,
        blend: float,
        k_value: float,
    ) -> numpy.ndarray:
        """Return ``state_vector`` with scalar TCA multipoles enforced."""

        if not generated_scalar_hierarchy:
            return numpy.asarray(state_vector, dtype=float)
        if theta_gamma1_index is None:
            return numpy.asarray(state_vector, dtype=float)
        _, background_scalars = _scalar_background_context(
            step_index,
            blend,
        )
        collision_rate = float(background_scalars["collision_rate"])
        if not numpy.isfinite(collision_rate) or collision_rate <= 1.0e-12:
            return numpy.asarray(state_vector, dtype=float)
        closed_state = numpy.asarray(state_vector, dtype=float).copy()
        theta_b = (
            float(closed_state[int(theta_b_index)])
            if theta_b_index is not None
            else 3.0 * float(k_value) * float(closed_state[theta_gamma1_index])
        )
        (
            common_theta_gamma1,
            theta_gamma2,
            theta_gamma3,
            e_gamma2,
            e_gamma3,
        ) = _generated_scalar_tight_coupling_multipoles(
            photon_dipole=float(closed_state[theta_gamma1_index]),
            baryon_velocity_divergence=theta_b,
            baryon_loading=float(background_scalars["baryon_loading"]),
            k_value=float(k_value),
            collision_rate=collision_rate,
        )
        closed_state[theta_gamma1_index] = common_theta_gamma1
        if theta_b_index is not None:
            closed_state[int(theta_b_index)] = (
                3.0 * float(k_value) * common_theta_gamma1
            )
        if theta_gamma2_index is not None:
            closed_state[theta_gamma2_index] = theta_gamma2
        if e_gamma2_index is not None:
            closed_state[e_gamma2_index] = e_gamma2
        for moment, slot_index in scalar_temperature_slot_indices.items():
            if moment >= 3:
                closed_state[slot_index] = theta_gamma3 if moment == 3 else 0.0
        for moment, slot_index in scalar_polarization_slot_indices.items():
            if moment == 3:
                closed_state[slot_index] = e_gamma3
            elif moment != 2:
                closed_state[slot_index] = 0.0
        return closed_state

    def _scalar_background_context(
        step_index: int,
        blend: float,
        *,
        k_value: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Return one interpolated scalar background context."""

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
        return float(eta_value), scalar_context

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
    ) -> tuple[
        dict[str, numpy.ndarray],
        dict[str, numpy.ndarray],
        dict[str, numpy.ndarray],
    ]:
        """Return the evolution grids used for one Fourier mode."""

        if not generated_scalar_hierarchy:
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

        def _evolution_eta_grid(
            eta_floor: float,
        ) -> numpy.ndarray:
            """Keep background resolution near last scattering and decimate
            smooth free-streaming intervals for the generated hierarchy.
            """

            base_grid = numpy.asarray(
                background.eta_grid[background.eta_grid >= float(eta_floor)],
                dtype=float,
            )
            if int(numerics.source_grid_multiplier) <= 1:
                return base_grid
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
                maximum_samples=max(
                    192,
                    min(256, int(numerics.eta_sample_count)),
                ),
            )

        eta_target = min(
            source_eta_start,
            _SCALAR_SUPERHORIZON_PREFIX_KETA / abs_k,
        )
        eta_target = max(eta_target, float(background.eta_grid[0]))
        if eta_target >= source_eta_start - 1.0e-12:
            eta_mode_grid = _evolution_eta_grid(source_eta_start)
            return _sample_eta_background_grids(eta_mode_grid)
        eta_prefix = numpy.asarray(
            background.eta_grid[
                (background.eta_grid >= eta_target)
                & (background.eta_grid < source_eta_start)
            ],
            dtype=float,
        )
        eta_mode_grid = numpy.unique(
            numpy.concatenate(
                (
                    numpy.asarray((eta_target,), dtype=float),
                    eta_prefix,
                    _evolution_eta_grid(source_eta_start),
                )
            )
        )
        return _sample_eta_background_grids(eta_mode_grid)

    def _build_scalar_base_context(
        *,
        k_value: float,
        eta_value: float,
        background_scalars: Mapping[str, float],
        cache_token: tuple[int, float] | None = None,
        resolve_graph: bool = False,
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
            base_context = _build_declared_base_context(
                perturbation_data=perturbation_data,
                model_parameters=source_parameters,
                physical_params=physical_params,
                numerics=numerics,
                k_value=float(k_value),
                eta_value=float(eta_value),
                background_scalars=background_scalars,
            )
            if resolve_graph:
                base_context = _resolve_declared_graph_context_ordered(
                    base_context,
                    perturbation_data,
                    allow_partial=True,
                    eta_grid=None,
                    execution_plan=execution_plan,
                    value_steps=execution_plan.value_steps,
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
                resolve_graph=False,
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
            value_steps=stage_value_steps,
            suppressed_outputs=suppressed_collision_outputs,
            use_compiled_program=True,
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
    ) -> numpy.ndarray:
        """Return the state derivative for one RK stage."""

        effective_state_vector = numpy.asarray(state_vector, dtype=float)
        if generated_scalar_hierarchy and tight_coupling_active:
            effective_state_vector = (
                _apply_generated_scalar_tight_coupling_closure(
                    effective_state_vector,
                    step_index=step_index,
                    blend=blend,
                    k_value=float(k_value),
                )
            )
        eta_value, background_scalars = _scalar_background_context(
            step_index,
            blend,
        )
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
        compiled_equations_succeeded = True
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
            except (KeyError, NameError, TypeError, ValueError):
                compiled_equations_succeeded = False
            except ArithmeticError as exc:
                raise ValueError(
                    "Declared CMB equation result must be finite; "
                    f"evaluation failed at eta={eta_value}, k={k_value}"
                ) from exc
            if not compiled_equations_succeeded:
                for slot_plan in execution_plan.equation_slot_plans:
                    coordinate_rate = coordinate_rates[slot_plan.wrt]
                    if slot_plan.promote_from_index is not None:
                        derivative[slot_plan.state_index] = (
                            float(
                                effective_state_vector[
                                    slot_plan.promote_from_index
                                ]
                            )
                            * coordinate_rate
                        )
                        continue
                    derivative[slot_plan.state_index] = (
                        _coerce_numeric_scalar(
                            _evaluate_compiled_expression_noerr(
                                slot_plan.compiled_rhs,
                                scalar_context,
                            ),
                            name=f"equation '{slot_plan.equation_name}'",
                        )
                        * coordinate_rate
                    )
        if generated_scalar_hierarchy and tight_coupling_active:
            for slot_index in scalar_tight_coupling_closure_indices:
                derivative[slot_index] = 0.0
        if not numpy.all(numpy.isfinite(derivative)):
            bad_indices = numpy.flatnonzero(~numpy.isfinite(derivative))
            bad_index = int(bad_indices[0]) if bad_indices.size else -1
            raise ValueError(
                "Declared CMB evolution produced non-finite derivatives at "
                f"eta={eta_value}, k={k_value}, state_index={bad_index}"
            )
        return derivative

    def _integrate_generated_scalar_history_fast(
        initial_state: numpy.ndarray,
        *,
        k_value: float,
    ) -> tuple[dict[str, numpy.ndarray], numpy.ndarray]:
        """Integrate the generated scalar graph with array-free physics code.

        Generated scalar contracts are already validated declarations.  This
        path evaluates that fixed physical hierarchy directly so reference
        grids do not spend most of their time reinterpreting the same
        compiled expression graph at every Runge-Kutta stage.  User-declared
        graphs continue through the fully generic executor below.
        """

        state_indices = {
            str(slot.variable): int(slot.index)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
        }
        temperature_indices = {
            int(moment): int(index)
            for moment, index in scalar_temperature_slot_indices.items()
        }
        polarization_indices = {
            int(moment): int(index)
            for moment, index in scalar_polarization_slot_indices.items()
        }
        neutrino_indices = {
            int(slot.variable[len("nu_l") :]): int(slot.index)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
            and str(slot.variable).startswith("nu_l")
            and str(slot.variable[len("nu_l") :]).isdigit()
        }
        history_names = tuple(
            str(slot.variable)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
        )
        theta_gamma0_index = state_indices.get("theta_gamma0")
        theta_gamma1_index = state_indices.get("theta_gamma1")
        theta_gamma2_index = state_indices.get("theta_gamma2")
        theta_gamma3_index = state_indices.get("theta_gamma3")
        e_gamma2_index = state_indices.get("e_gamma2")
        e_gamma3_index = state_indices.get("e_gamma3")
        theta_b_index = state_indices.get("theta_b")
        theta_c_index = state_indices.get("theta_c")
        theta_nu_index = state_indices.get("theta_nu")
        sigma_nu_index = state_indices.get("sigma_nu")
        delta_b_index = state_indices.get("delta_b")
        delta_c_index = state_indices.get("delta_c")
        delta_nu_index = state_indices.get("delta_nu")
        nu_l3_index = state_indices.get("nu_l3")
        temperature_hierarchy = tuple(
            (
                int(moment),
                int(index),
                int(
                    theta_gamma2_index
                    if int(moment) == 3
                    else state_indices[f"theta_gamma{int(moment) - 1}"]
                ),
                (
                    None
                    if int(moment) == max(temperature_indices)
                    else int(state_indices[f"theta_gamma{int(moment) + 1}"])
                ),
            )
            for moment, index in sorted(temperature_indices.items())
            if int(moment) >= 3
        )
        polarization_hierarchy = tuple(
            (
                int(moment),
                int(index),
                int(
                    e_gamma2_index
                    if int(moment) == 3
                    else state_indices[f"e_gamma{int(moment) - 1}"]
                ),
                (
                    None
                    if int(moment) == max(polarization_indices)
                    else int(state_indices[f"e_gamma{int(moment) + 1}"])
                ),
            )
            for moment, index in sorted(polarization_indices.items())
            if int(moment) >= 3
        )
        neutrino_hierarchy = tuple(
            (
                int(moment),
                int(index),
                int(
                    sigma_nu_index
                    if int(moment) == 3
                    else state_indices[f"nu_l{int(moment) - 1}"]
                ),
                (
                    None
                    if int(moment) == max(neutrino_indices)
                    else int(state_indices[f"nu_l{int(moment) + 1}"])
                ),
            )
            for moment, index in sorted(neutrino_indices.items())
            if int(moment) >= 3
        )
        histories = {
            name: numpy.empty(active_grids["eta"].size, dtype=float)
            for name in history_names
        }
        metric_phi_name = "Phi"
        phi_index = state_indices.get(metric_phi_name)
        if phi_index is None:
            metric_phi_name = "Phi_gi"
            phi_index = state_indices.get(metric_phi_name)
        sync_gauge = str(getattr(perturbation_data, "gauge", "")) == (
            "synchronous"
        )
        sync_h_index = state_indices.get("h_sync_metric")
        sync_eta_index = state_indices.get("eta_sync_metric")
        sync_alpha_index = state_indices.get("gauge_shift_alpha")
        if phi_index is None and not (
            sync_gauge
            and sync_h_index is not None
            and sync_eta_index is not None
            and sync_alpha_index is not None
        ):
            raise ValueError(
                "Generated scalar fast path requires a declared metric basis"
            )
        H0c_sq = float(physical_params.H0_over_c_Mpc_inv) ** 2
        omega_nu = float(physical_params.Omega_nu0 or 0.0)
        omega_c = float(physical_params.Omega_c0 or 0.0)

        def _fast_background(
            step_index: int,
            blend: float,
        ) -> tuple[float, float, float, float, float, float, float]:
            """Return only the background scalars used by the fast path."""

            next_index = min(
                int(step_index) + 1,
                active_grids["eta"].size - 1,
            )
            weight_next = float(blend)
            weight_current = 1.0 - weight_next

            def _interpolate(name: str) -> float:
                """Interpolate one cached background history."""

                history = active_grids[name]
                return float(
                    weight_current * history[step_index]
                    + weight_next * history[next_index]
                )

            a_value = _interpolate("a")
            collision_rate = _interpolate("collision_rate")
            baryon_loading = _interpolate("baryon_loading")
            return (
                _interpolate("eta"),
                a_value,
                _interpolate("Hconf"),
                _interpolate("Hconf_tau"),
                _interpolate("baryon_sound_speed_sq"),
                collision_rate,
                baryon_loading,
            )

        def _fast_tight_coupling_closure(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
        ) -> numpy.ndarray:
            """Apply the generated scalar tight-coupling moments cheaply."""

            if theta_gamma1_index is None:
                return numpy.asarray(state_vector, dtype=float)
            (_, _, _, _, _, collision_rate, baryon_loading) = _fast_background(
                step_index, blend
            )
            if not numpy.isfinite(collision_rate) or collision_rate <= 1.0e-12:
                return numpy.asarray(state_vector, dtype=float)
            closed_state = numpy.asarray(state_vector, dtype=float).copy()
            theta_b = (
                float(closed_state[int(theta_b_index)])
                if theta_b_index is not None
                else 3.0
                * float(k_value)
                * float(closed_state[theta_gamma1_index])
            )
            (
                common_theta_gamma1,
                theta_gamma2,
                theta_gamma3,
                e_gamma2,
                e_gamma3,
            ) = _generated_scalar_tight_coupling_multipoles(
                photon_dipole=float(closed_state[theta_gamma1_index]),
                baryon_velocity_divergence=theta_b,
                baryon_loading=baryon_loading,
                k_value=float(k_value),
                collision_rate=collision_rate,
            )
            closed_state[theta_gamma1_index] = common_theta_gamma1
            if theta_b_index is not None:
                closed_state[int(theta_b_index)] = (
                    3.0 * float(k_value) * common_theta_gamma1
                )
            if theta_gamma2_index is not None:
                closed_state[theta_gamma2_index] = theta_gamma2
            if e_gamma2_index is not None:
                closed_state[e_gamma2_index] = e_gamma2
            for moment, index in temperature_indices.items():
                if moment >= 3:
                    closed_state[index] = theta_gamma3 if moment == 3 else 0.0
            for moment, index in polarization_indices.items():
                if moment == 3:
                    closed_state[index] = e_gamma3
                elif moment != 2:
                    closed_state[index] = 0.0
            return closed_state

        def _rhs(
            vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            tight_coupling_active: bool,
            phi_override: float | None = None,
        ) -> tuple[numpy.ndarray, float]:
            """Return direct generated-hierarchy derivatives."""

            working = numpy.asarray(vector, dtype=float)
            if tight_coupling_active:
                working = _fast_tight_coupling_closure(
                    working,
                    step_index=step_index,
                    blend=blend,
                )
            (
                eta_value,
                a_value,
                Hconf,
                Hconf_tau,
                baryon_sound_speed_sq,
                _,
                _,
            ) = _fast_background(step_index, blend)
            theta0 = (
                0.0
                if theta_gamma0_index is None
                else float(working[theta_gamma0_index])
            )
            theta1 = (
                0.0
                if theta_gamma1_index is None
                else float(working[theta_gamma1_index])
            )
            theta2 = (
                0.0
                if theta_gamma2_index is None
                else float(working[theta_gamma2_index])
            )
            theta_b = (
                0.0 if theta_b_index is None else float(working[theta_b_index])
            )
            theta_c = (
                0.0 if theta_c_index is None else float(working[theta_c_index])
            )
            theta_nu = (
                0.0
                if theta_nu_index is None
                else float(working[theta_nu_index])
            )
            sigma_nu = (
                0.0
                if sigma_nu_index is None
                else float(working[sigma_nu_index])
            )
            delta_b = (
                0.0 if delta_b_index is None else float(working[delta_b_index])
            )
            delta_nu = (
                0.0
                if delta_nu_index is None
                else float(working[delta_nu_index])
            )
            radiation_momentum = (
                (4.0 / 3.0)
                * float(physical_params.Omega_gamma0)
                * (3.0 * float(k_value) * theta1)
                + (4.0 / 3.0) * omega_nu * theta_nu
            ) / (a_value * a_value)
            total_momentum = (
                float(physical_params.Omega_b0) * theta_b + omega_c * theta_c
            ) / a_value + radiation_momentum
            total_shear = (
                4.0 * float(physical_params.Omega_gamma0) * theta2
                + 2.0 * omega_nu * sigma_nu
            ) / (a_value * a_value)
            if phi_override is not None:
                phi = float(phi_override)
            elif phi_index is not None:
                phi = float(working[phi_index])
            else:
                phi = float(working[sync_eta_index]) - Hconf * float(
                    working[sync_alpha_index]
                )
            shear_correction = (
                3.0 * H0c_sq * total_shear / (float(k_value) ** 2)
            )
            psi = phi - shear_correction
            phi_tau = (
                1.5 * H0c_sq * total_momentum / (float(k_value) ** 2)
                - Hconf * psi
            )
            derivative = numpy.zeros_like(working, dtype=float)

            k_squared = float(k_value) ** 2
            if theta_gamma0_index is not None:
                derivative[theta_gamma0_index] = (
                    -float(k_value) * theta1 + phi_tau
                )
            if theta_gamma1_index is not None:
                derivative[theta_gamma1_index] = (
                    float(k_value) * (theta0 + psi - 2.0 * theta2) / 3.0
                )
            theta3 = (
                0.0
                if theta_gamma3_index is None
                else float(working[theta_gamma3_index])
            )
            if theta_gamma2_index is not None:
                derivative[theta_gamma2_index] = (
                    2.0 * float(k_value) * theta1 / 5.0
                    - 3.0 * float(k_value) * theta3 / 5.0
                )
            polarization_third_moment = (
                0.0
                if e_gamma3_index is None
                else float(working[e_gamma3_index])
            )
            for name in ("e_gamma0", "e_gamma1"):
                index = state_indices.get(name)
                if index is not None:
                    derivative[index] = 0.0
            if e_gamma2_index is not None:
                derivative[e_gamma2_index] = (
                    -float(k_value) * polarization_third_moment / 3.0
                )
            if delta_b_index is not None:
                derivative[delta_b_index] = -theta_b + 3.0 * phi_tau
            if theta_b_index is not None:
                derivative[theta_b_index] = (
                    -Hconf * theta_b
                    + baryon_sound_speed_sq * k_squared * delta_b
                    + k_squared * psi
                )
            if delta_c_index is not None:
                derivative[delta_c_index] = -theta_c + 3.0 * phi_tau
            if theta_c_index is not None:
                derivative[theta_c_index] = -Hconf * theta_c + k_squared * psi
            if delta_nu_index is not None:
                derivative[delta_nu_index] = (
                    -(4.0 / 3.0) * theta_nu + 4.0 * phi_tau
                )
            if theta_nu_index is not None:
                derivative[theta_nu_index] = k_squared * (
                    0.25 * delta_nu + psi - sigma_nu
                )
            if sigma_nu_index is not None:
                nu_l3 = (
                    0.0 if nu_l3_index is None else float(working[nu_l3_index])
                )
                derivative[sigma_nu_index] = (4.0 / 15.0) * theta_nu - (
                    3.0 / 5.0
                ) * float(k_value) * nu_l3
            if phi_index is not None:
                derivative[phi_index] = phi_tau
            else:
                alpha = float(working[sync_alpha_index])
                alpha_tau = psi - Hconf * alpha
                eta_tau = phi_tau + Hconf_tau * alpha + Hconf * alpha_tau
                derivative[sync_alpha_index] = alpha_tau
                derivative[sync_eta_index] = eta_tau
                derivative[sync_h_index] = (
                    2.0 * k_squared * alpha - 6.0 * eta_tau
                )

            for (
                moment,
                index,
                previous_index,
                next_index,
            ) in temperature_hierarchy:
                previous = float(working[previous_index])
                current = float(working[index])
                if next_index is None:
                    denominator = math.sqrt(
                        (float(k_value) * eta_value) ** 2
                        + (float(moment) + 1.0) ** 2
                    )
                    derivative[index] = (
                        float(k_value) * previous
                        - float(k_value)
                        * (float(moment) + 1.0)
                        * current
                        / denominator
                    )
                else:
                    coupling = (
                        float(moment)
                        / (2.0 * float(moment) + 1.0)
                        * float(k_value)
                        * previous
                    )
                    decay = (
                        (float(moment) + 1.0)
                        / (2.0 * float(moment) + 1.0)
                        * float(k_value)
                        * float(working[next_index])
                    )
                    derivative[index] = coupling - decay
            for (
                moment,
                index,
                previous_index,
                next_index,
            ) in polarization_hierarchy:
                previous = float(working[previous_index])
                current = float(working[index])
                if next_index is None:
                    denominator = math.sqrt(
                        (float(k_value) * eta_value) ** 2
                        + (float(moment) + 3.0) ** 2
                    )
                    derivative[index] = (
                        float(moment)
                        / float(moment - 2)
                        * float(k_value)
                        * previous
                        - float(k_value)
                        * (float(moment) + 3.0)
                        * current
                        / denominator
                    )
                else:
                    next_coefficient = (
                        (float(moment) + 3.0)
                        * (float(moment) - 1.0)
                        / ((2.0 * float(moment) + 1.0) * (float(moment) + 1.0))
                    )
                    coupling = (
                        float(moment)
                        / (2.0 * float(moment) + 1.0)
                        * float(k_value)
                        * previous
                    )
                    decay = (
                        next_coefficient
                        * float(k_value)
                        * float(working[next_index])
                    )
                    derivative[index] = coupling - decay
            for (
                moment,
                index,
                previous_index,
                next_index,
            ) in neutrino_hierarchy:
                previous = float(working[previous_index])
                current = float(working[index])
                if next_index is None:
                    denominator = math.sqrt(
                        (float(k_value) * eta_value) ** 2
                        + (float(moment) + 1.0) ** 2
                    )
                    derivative[index] = (
                        float(k_value) * previous
                        - float(k_value)
                        * (float(moment) + 1.0)
                        * current
                        / denominator
                    )
                else:
                    coupling = (
                        float(moment)
                        / (2.0 * float(moment) + 1.0)
                        * float(k_value)
                        * previous
                    )
                    decay = (
                        (float(moment) + 1.0)
                        / (2.0 * float(moment) + 1.0)
                        * float(k_value)
                        * float(working[next_index])
                    )
                    derivative[index] = coupling - decay
            if tight_coupling_active:
                for moment, index in temperature_indices.items():
                    if moment >= 2:
                        derivative[index] = 0.0
                for moment, index in polarization_indices.items():
                    if moment >= 2:
                        derivative[index] = 0.0
            return derivative, float(phi_tau)

        polarization_matrix = numpy.asarray(
            ((-0.9, 0.6), (0.1, -0.4)),
            dtype=float,
        )
        polarization_eigensystem = _cached_collision_eigendecomposition(
            polarization_matrix,
            {},
        )

        def _collision_step(
            vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            dt: float,
            tight_coupling_active: bool,
        ) -> numpy.ndarray:
            """Apply the declared Thomson block and hierarchy damping."""

            if dt == 0.0:
                return numpy.asarray(vector, dtype=float)
            if tight_coupling_active:
                return numpy.asarray(vector, dtype=float)
            (
                _,
                a_value,
                _,
                _,
                _,
                collision_rate,
                _,
            ) = _fast_background(step_index, blend)
            rate = max(collision_rate, 0.0)
            if rate <= 1.0e-12:
                return numpy.asarray(vector, dtype=float)
            result = numpy.asarray(vector, dtype=float).copy()
            dipole_indices = (
                temperature_indices.get(1),
                state_indices.get("theta_b"),
            )
            if all(index is not None for index in dipole_indices):
                dipole_matrix = numpy.asarray(
                    (
                        (-1.0, 1.0 / (3.0 * float(k_value))),
                        (
                            3.0
                            * float(k_value)
                            * (
                                4.0
                                * float(physical_params.Omega_gamma0)
                                / (
                                    3.0
                                    * float(physical_params.Omega_b0)
                                    * a_value
                                )
                            ),
                            -4.0
                            * float(physical_params.Omega_gamma0)
                            / (
                                3.0 * float(physical_params.Omega_b0) * a_value
                            ),
                        ),
                    ),
                    dtype=float,
                )
                indices = tuple(int(index) for index in dipole_indices)
                result[list(indices)] = _exact_linear_collision_step(
                    operator_matrix=dipole_matrix,
                    dt=dt,
                    target_state=result[list(indices)],
                    operator_scale=rate,
                )
            if not tight_coupling_active:
                quadrupole_indices = tuple(
                    int(index)
                    for index in (
                        temperature_indices.get(2),
                        polarization_indices.get(2),
                    )
                    if index is not None
                )
                if len(quadrupole_indices) == 2:
                    result[list(quadrupole_indices)] = (
                        _exact_linear_collision_step(
                            operator_matrix=polarization_matrix,
                            dt=dt,
                            target_state=result[list(quadrupole_indices)],
                            eigendecomposition=polarization_eigensystem,
                            operator_scale=rate,
                        )
                    )
            damping = math.exp(-rate * float(dt))
            for moment, index in temperature_indices.items():
                if moment >= 3:
                    result[index] *= damping
            for moment, index in polarization_indices.items():
                if moment >= 3:
                    result[index] *= damping
            return result

        state = numpy.asarray(initial_state, dtype=float).copy()
        physical_phi = None
        if phi_index is None:
            (
                _,
                _,
                initial_Hconf,
                _,
                _,
                _,
                _,
            ) = _fast_background(0, 0.0)
            physical_phi = float(state[sync_eta_index]) - float(
                initial_Hconf
            ) * float(state[sync_alpha_index])
        tight_coupling_active = _tight_coupling_is_active(
            active=False,
            collision_rate=float(active_grids["collision_rate"][0]),
            k_value=float(k_value),
            tight_coupling_ratio=float(numerics.tight_coupling_ratio),
        )
        for step_index, eta_value in enumerate(active_grids["eta"]):
            if tight_coupling_active:
                state = _fast_tight_coupling_closure(
                    state,
                    step_index=step_index,
                    blend=0.0,
                )
            for name, index in state_indices.items():
                histories[name][step_index] = state[index]
            if step_index == active_grids["eta"].size - 1:
                break
            # Use the same explicit stepping for every generated scalar gauge.
            # Gauge-equivalent contracts must not diverge because one route
            # selected a different adaptive step sequence for its state basis.
            dt = float(active_grids["eta"][step_index + 1] - eta_value)
            phase_step = 0.5
            required_substeps = int(
                math.ceil(abs(float(dt)) * abs(float(k_value)) / phase_step)
            )
            if not tight_coupling_active:
                collision_step = 0.25
                start_collision_rate = float(
                    active_grids["collision_rate"][step_index]
                )
                end_collision_rate = float(
                    active_grids["collision_rate"][step_index + 1]
                )
                required_substeps = max(
                    required_substeps,
                    int(
                        math.ceil(
                            abs(float(dt))
                            * max(start_collision_rate, end_collision_rate)
                            / collision_step
                        )
                    ),
                )
            required_substeps = max(1, required_substeps)
            substep_count = 1
            while substep_count < required_substeps:
                substep_count *= 2
            sub_dt = dt / float(substep_count)
            for substep_index in range(substep_count):
                blend_start = substep_index / substep_count
                blend_mid = (substep_index + 0.5) / substep_count
                blend_end = (substep_index + 1.0) / substep_count
                state = _collision_step(
                    state,
                    step_index=step_index,
                    blend=blend_start,
                    dt=0.5 * sub_dt,
                    tight_coupling_active=tight_coupling_active,
                )
                rhs_a, phi_tau_a = _rhs(
                    state,
                    step_index=step_index,
                    blend=blend_start,
                    tight_coupling_active=tight_coupling_active,
                    phi_override=physical_phi,
                )
                rhs_b, phi_tau_b = _rhs(
                    state + 0.5 * sub_dt * rhs_a,
                    step_index=step_index,
                    blend=blend_mid,
                    tight_coupling_active=tight_coupling_active,
                    phi_override=(
                        None
                        if physical_phi is None
                        else physical_phi + 0.5 * sub_dt * phi_tau_a
                    ),
                )
                rhs_c, phi_tau_c = _rhs(
                    state + 0.5 * sub_dt * rhs_b,
                    step_index=step_index,
                    blend=blend_mid,
                    tight_coupling_active=tight_coupling_active,
                    phi_override=(
                        None
                        if physical_phi is None
                        else physical_phi + 0.5 * sub_dt * phi_tau_b
                    ),
                )
                rhs_d, phi_tau_d = _rhs(
                    state + sub_dt * rhs_c,
                    step_index=step_index,
                    blend=blend_end,
                    tight_coupling_active=tight_coupling_active,
                    phi_override=(
                        None
                        if physical_phi is None
                        else physical_phi + sub_dt * phi_tau_c
                    ),
                )
                state = state + (sub_dt / 6.0) * (
                    rhs_a + 2.0 * rhs_b + 2.0 * rhs_c + rhs_d
                )
                if physical_phi is not None:
                    physical_phi += (sub_dt / 6.0) * (
                        phi_tau_a
                        + 2.0 * phi_tau_b
                        + 2.0 * phi_tau_c
                        + phi_tau_d
                    )
                    endpoint_background = _fast_background(
                        step_index,
                        blend_end,
                    )
                    state[sync_eta_index] = physical_phi + float(
                        endpoint_background[2]
                    ) * float(state[sync_alpha_index])
                state = _collision_step(
                    state,
                    step_index=step_index,
                    blend=blend_end,
                    dt=0.5 * sub_dt,
                    tight_coupling_active=tight_coupling_active,
                )
                if tight_coupling_active:
                    state = _fast_tight_coupling_closure(
                        state,
                        step_index=step_index,
                        blend=blend_end,
                    )
            tight_coupling_active = _tight_coupling_is_active(
                active=tight_coupling_active,
                collision_rate=float(
                    active_grids["collision_rate"][step_index + 1]
                ),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
        return histories, state

    def _integrate_generated_scalar_history_batch(
        initial_states: numpy.ndarray,
        *,
        k_values_batch: numpy.ndarray,
    ) -> tuple[
        dict[str, numpy.ndarray],
        numpy.ndarray,
        Any,
    ]:
        """Integrate generated scalar modes in one vectorized hierarchy."""

        state_indices = {
            str(slot.variable): int(slot.index)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
        }
        temperature_indices = {
            int(moment): int(index)
            for moment, index in scalar_temperature_slot_indices.items()
        }
        polarization_indices = {
            int(moment): int(index)
            for moment, index in scalar_polarization_slot_indices.items()
        }
        neutrino_indices = {
            int(slot.variable[len("nu_l") :]): int(slot.index)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
            and str(slot.variable).startswith("nu_l")
            and str(slot.variable[len("nu_l") :]).isdigit()
        }
        history_names = tuple(
            str(slot.variable)
            for slot in runtime_spec.state_slots
            if int(slot.order) == 0
        )
        theta_gamma0_index = state_indices.get("theta_gamma0")
        theta_gamma1_index = state_indices.get("theta_gamma1")
        theta_gamma2_index = state_indices.get("theta_gamma2")
        theta_gamma3_index = state_indices.get("theta_gamma3")
        e_gamma2_index = state_indices.get("e_gamma2")
        e_gamma3_index = state_indices.get("e_gamma3")
        theta_b_index = state_indices.get("theta_b")
        theta_c_index = state_indices.get("theta_c")
        theta_nu_index = state_indices.get("theta_nu")
        sigma_nu_index = state_indices.get("sigma_nu")
        delta_b_index = state_indices.get("delta_b")
        delta_c_index = state_indices.get("delta_c")
        delta_nu_index = state_indices.get("delta_nu")
        nu_l3_index = state_indices.get("nu_l3")
        temperature_hierarchy = tuple(
            (
                int(moment),
                int(index),
                int(
                    theta_gamma2_index
                    if int(moment) == 3
                    else state_indices[f"theta_gamma{int(moment) - 1}"]
                ),
                (
                    None
                    if int(moment) == max(temperature_indices)
                    else int(state_indices[f"theta_gamma{int(moment) + 1}"])
                ),
            )
            for moment, index in sorted(temperature_indices.items())
            if int(moment) >= 3
        )
        polarization_hierarchy = tuple(
            (
                int(moment),
                int(index),
                int(
                    e_gamma2_index
                    if int(moment) == 3
                    else state_indices[f"e_gamma{int(moment) - 1}"]
                ),
                (
                    None
                    if int(moment) == max(polarization_indices)
                    else int(state_indices[f"e_gamma{int(moment) + 1}"])
                ),
            )
            for moment, index in sorted(polarization_indices.items())
            if int(moment) >= 3
        )
        neutrino_hierarchy = tuple(
            (
                int(moment),
                int(index),
                int(
                    sigma_nu_index
                    if int(moment) == 3
                    else state_indices[f"nu_l{int(moment) - 1}"]
                ),
                (
                    None
                    if int(moment) == max(neutrino_indices)
                    else int(state_indices[f"nu_l{int(moment) + 1}"])
                ),
            )
            for moment, index in sorted(neutrino_indices.items())
            if int(moment) >= 3
        )
        state_count = int(initial_states.shape[1])
        mode_count = int(initial_states.shape[0])
        k_batch = numpy.asarray(k_values_batch, dtype=float)
        states = numpy.asarray(initial_states, dtype=float)
        if states.shape != (mode_count, state_count):
            raise ValueError("Generated batch initial states have wrong shape")
        metric_phi_name = "Phi"
        phi_index = state_indices.get(metric_phi_name)
        if phi_index is None:
            metric_phi_name = "Phi_gi"
            phi_index = state_indices.get(metric_phi_name)
        sync_gauge = str(getattr(perturbation_data, "gauge", "")) == (
            "synchronous"
        )
        sync_h_index = state_indices.get("h_sync_metric")
        sync_eta_index = state_indices.get("eta_sync_metric")
        sync_alpha_index = state_indices.get("gauge_shift_alpha")
        if phi_index is None and not (
            sync_gauge
            and sync_h_index is not None
            and sync_eta_index is not None
            and sync_alpha_index is not None
        ):
            raise ValueError(
                "Generated scalar batch requires a declared metric basis"
            )
        H0c_sq = float(physical_params.H0_over_c_Mpc_inv) ** 2
        omega_nu = float(physical_params.Omega_nu0 or 0.0)
        omega_c = float(physical_params.Omega_c0 or 0.0)
        omega_gamma = float(physical_params.Omega_gamma0)
        omega_b = float(physical_params.Omega_b0)

        def _fast_background() -> tuple[float, ...]:
            """Return common background values for one RK stage."""

            return (
                float(active_grids["eta"][0]),
                float(active_grids["a"][0]),
                float(active_grids["Hconf"][0]),
                float(active_grids["Hconf_tau"][0]),
                float(active_grids["baryon_sound_speed_sq"][0]),
                float(active_grids["collision_rate"][0]),
                float(active_grids["baryon_loading"][0]),
            )

        def _background_values(
            step_index: int,
            blend: float,
        ) -> tuple[float, float, float, float, float, float, float]:
            """Interpolate common background histories once per stage."""

            next_index = min(
                int(step_index) + 1,
                active_grids["eta"].size - 1,
            )
            current_weight = 1.0 - float(blend)
            next_weight = float(blend)

            def _value(name: str) -> float:
                """Interpolate one common background scalar."""

                history = active_grids[name]
                return float(
                    current_weight * history[step_index]
                    + next_weight * history[next_index]
                )

            return (
                _value("eta"),
                _value("a"),
                _value("Hconf"),
                _value("Hconf_tau"),
                _value("baryon_sound_speed_sq"),
                _value("collision_rate"),
                _value("baryon_loading"),
            )

        def _tight_coupling_closure_batch(
            vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            active: numpy.ndarray,
        ) -> numpy.ndarray:
            """Apply tight-coupling moment closure to active mode rows."""

            if theta_gamma1_index is None or not numpy.any(active):
                return numpy.asarray(vector, dtype=float)
            (_, _, _, _, _, collision_rate, baryon_loading) = (
                _background_values(step_index, blend)
            )
            if not numpy.isfinite(collision_rate) or collision_rate <= 1.0e-12:
                return numpy.asarray(vector, dtype=float)
            closed = numpy.asarray(vector, dtype=float).copy()
            photon_dipole = closed[:, theta_gamma1_index]
            if theta_b_index is None:
                baryon_velocity = 3.0 * k_batch * photon_dipole
            else:
                baryon_velocity = closed[:, theta_b_index]
            loading = max(float(baryon_loading), 1.0e-12)
            common_dipole = (
                photon_dipole + loading * baryon_velocity / (3.0 * k_batch)
            ) / (1.0 + loading)
            quadrupole = (
                8.0 / 15.0 * k_batch * common_dipole / float(collision_rate)
            )
            octopole = 3.0 / 7.0 * k_batch / float(collision_rate) * quadrupole
            closed[active, theta_gamma1_index] = common_dipole[active]
            if theta_b_index is not None:
                closed[active, theta_b_index] = (
                    3.0 * k_batch[active] * common_dipole[active]
                )
            if theta_gamma2_index is not None:
                closed[active, theta_gamma2_index] = quadrupole[active]
            if e_gamma2_index is not None:
                closed[active, e_gamma2_index] = quadrupole[active] / 4.0
            for moment, index in temperature_indices.items():
                if moment >= 3:
                    closed[active, index] = numpy.where(
                        int(moment) == 3,
                        octopole[active],
                        0.0,
                    )
            for moment, index in polarization_indices.items():
                if moment >= 3:
                    closed[active, index] = numpy.where(
                        int(moment) == 3,
                        octopole[active] / 4.0,
                        0.0,
                    )
            return closed

        physical_phi = None
        if phi_index is None:
            _, _, initial_hconf, _, _, _, _ = _fast_background()
            physical_phi = (
                states[:, sync_eta_index]
                - initial_hconf * states[:, sync_alpha_index]
            )
            states = numpy.column_stack((states, physical_phi))

        def _rhs_batch(
            vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            active: numpy.ndarray,
        ) -> numpy.ndarray:
            """Return generated hierarchy derivatives for all mode rows."""

            working = _tight_coupling_closure_batch(
                vector,
                step_index=step_index,
                blend=blend,
                active=active,
            )
            (
                eta_value,
                a_value,
                Hconf,
                Hconf_tau,
                sound_speed_sq,
                _,
                _,
            ) = _background_values(step_index, blend)

            def _column(index: int | None) -> numpy.ndarray:
                """Return one state column or a zero mode vector."""

                if index is None:
                    return numpy.zeros(mode_count, dtype=float)
                return working[:, int(index)]

            theta0 = _column(theta_gamma0_index)
            theta1 = _column(theta_gamma1_index)
            theta2 = _column(theta_gamma2_index)
            theta3 = _column(theta_gamma3_index)
            theta_b = _column(theta_b_index)
            theta_c = _column(theta_c_index)
            theta_nu = _column(theta_nu_index)
            sigma_nu = _column(sigma_nu_index)
            delta_b = _column(delta_b_index)
            delta_nu = _column(delta_nu_index)
            radiation_momentum = (
                (4.0 / 3.0) * omega_gamma * (3.0 * k_batch * theta1)
                + (4.0 / 3.0) * omega_nu * theta_nu
            ) / (a_value * a_value)
            total_momentum = (
                omega_b * theta_b + omega_c * theta_c
            ) / a_value + radiation_momentum
            total_shear = (
                4.0 * omega_gamma * theta2 + 2.0 * omega_nu * sigma_nu
            ) / (a_value * a_value)
            if phi_index is not None:
                phi = working[:, phi_index]
            else:
                phi = vector[:, -1]
            shear_correction = 3.0 * H0c_sq * total_shear / (k_batch**2)
            psi = phi - shear_correction
            phi_tau = (
                1.5 * H0c_sq * total_momentum / (k_batch**2) - Hconf * psi
            )
            derivative = numpy.zeros_like(vector, dtype=float)
            k_squared = k_batch**2
            if theta_gamma0_index is not None:
                derivative[:, theta_gamma0_index] = -k_batch * theta1 + phi_tau
            if theta_gamma1_index is not None:
                derivative[:, theta_gamma1_index] = (
                    k_batch * (theta0 + psi - 2.0 * theta2) / 3.0
                )
            if theta_gamma2_index is not None:
                derivative[:, theta_gamma2_index] = (
                    2.0 * k_batch * theta1 / 5.0 - 3.0 * k_batch * theta3 / 5.0
                )
            polarization_third_moment = _column(e_gamma3_index)
            for name in ("e_gamma0", "e_gamma1"):
                index = state_indices.get(name)
                if index is not None:
                    derivative[:, index] = 0.0
            if e_gamma2_index is not None:
                derivative[:, e_gamma2_index] = (
                    -k_batch * polarization_third_moment / 3.0
                )
            if delta_b_index is not None:
                derivative[:, delta_b_index] = -theta_b + 3.0 * phi_tau
            if theta_b_index is not None:
                derivative[:, theta_b_index] = (
                    -Hconf * theta_b
                    + sound_speed_sq * k_squared * delta_b
                    + k_squared * psi
                )
            if delta_c_index is not None:
                derivative[:, delta_c_index] = -theta_c + 3.0 * phi_tau
            if theta_c_index is not None:
                derivative[:, theta_c_index] = (
                    -Hconf * theta_c + k_squared * psi
                )
            if delta_nu_index is not None:
                derivative[:, delta_nu_index] = (
                    -(4.0 / 3.0) * theta_nu + 4.0 * phi_tau
                )
            if theta_nu_index is not None:
                derivative[:, theta_nu_index] = k_squared * (
                    0.25 * delta_nu + psi - sigma_nu
                )
            if sigma_nu_index is not None:
                nu_l3 = _column(nu_l3_index)
                derivative[:, sigma_nu_index] = (4.0 / 15.0) * theta_nu - (
                    3.0 / 5.0
                ) * k_batch * nu_l3
            if phi_index is not None:
                derivative[:, phi_index] = phi_tau
            else:
                alpha = working[:, sync_alpha_index]
                alpha_tau = psi - Hconf * alpha
                eta_tau = phi_tau + Hconf_tau * alpha + Hconf * alpha_tau
                derivative[:, sync_alpha_index] = alpha_tau
                derivative[:, sync_eta_index] = eta_tau
                derivative[:, sync_h_index] = (
                    2.0 * k_squared * alpha - 6.0 * eta_tau
                )
                derivative[:, -1] = phi_tau
            for (
                moment,
                index,
                previous_index,
                next_index,
            ) in temperature_hierarchy:
                previous = working[:, previous_index]
                current = working[:, index]
                if next_index is None:
                    denominator = numpy.sqrt(
                        (k_batch * eta_value) ** 2 + (float(moment) + 1.0) ** 2
                    )
                    derivative[:, index] = (
                        k_batch * previous
                        - k_batch
                        * (float(moment) + 1.0)
                        * current
                        / denominator
                    )
                else:
                    derivative[:, index] = (
                        float(moment)
                        / (2.0 * float(moment) + 1.0)
                        * k_batch
                        * previous
                        - (float(moment) + 1.0)
                        / (2.0 * float(moment) + 1.0)
                        * k_batch
                        * working[:, next_index]
                    )
            for (
                moment,
                index,
                previous_index,
                next_index,
            ) in polarization_hierarchy:
                previous = working[:, previous_index]
                current = working[:, index]
                if next_index is None:
                    denominator = numpy.sqrt(
                        (k_batch * eta_value) ** 2 + (float(moment) + 3.0) ** 2
                    )
                    derivative[:, index] = (
                        float(moment) / float(moment - 2) * k_batch * previous
                        - k_batch
                        * (float(moment) + 3.0)
                        * current
                        / denominator
                    )
                else:
                    next_coefficient = (
                        (float(moment) + 3.0)
                        * (float(moment) - 1.0)
                        / ((2.0 * float(moment) + 1.0) * (float(moment) + 1.0))
                    )
                    derivative[:, index] = (
                        float(moment)
                        / (2.0 * float(moment) + 1.0)
                        * k_batch
                        * previous
                        - next_coefficient * k_batch * working[:, next_index]
                    )
            for (
                moment,
                index,
                previous_index,
                next_index,
            ) in neutrino_hierarchy:
                previous = working[:, previous_index]
                current = working[:, index]
                if next_index is None:
                    denominator = numpy.sqrt(
                        (k_batch * eta_value) ** 2 + (float(moment) + 1.0) ** 2
                    )
                    derivative[:, index] = (
                        k_batch * previous
                        - k_batch
                        * (float(moment) + 1.0)
                        * current
                        / denominator
                    )
                else:
                    derivative[:, index] = (
                        float(moment)
                        / (2.0 * float(moment) + 1.0)
                        * k_batch
                        * previous
                        - (float(moment) + 1.0)
                        / (2.0 * float(moment) + 1.0)
                        * k_batch
                        * working[:, next_index]
                    )
            if numpy.any(active):
                active_rows = numpy.flatnonzero(active)
                for moment, index in temperature_indices.items():
                    if moment >= 2:
                        derivative[active_rows, index] = 0.0
                for moment, index in polarization_indices.items():
                    if moment >= 2:
                        derivative[active_rows, index] = 0.0
            return derivative

        def _exact_two_state_batch(
            matrix_00: numpy.ndarray,
            matrix_01: numpy.ndarray,
            matrix_10: numpy.ndarray,
            matrix_11: numpy.ndarray,
            target: numpy.ndarray,
            *,
            scale: float,
        ) -> numpy.ndarray:
            """Apply a real two-by-two exponential to every mode row."""

            a00 = matrix_00 * float(scale)
            a01 = matrix_01 * float(scale)
            a10 = matrix_10 * float(scale)
            a11 = matrix_11 * float(scale)
            trace_half = 0.5 * (a00 + a11)
            centered00 = a00 - trace_half
            centered11 = a11 - trace_half
            discriminant = 0.25 * (a00 - a11) ** 2 + a01 * a10
            delta = numpy.sqrt(discriminant.astype(complex))
            exp_trace = numpy.exp(trace_half)
            with numpy.errstate(divide="ignore", invalid="ignore"):
                coefficient = numpy.where(
                    numpy.abs(delta) <= 1.0e-14,
                    exp_trace,
                    (
                        numpy.exp(trace_half + delta)
                        - numpy.exp(trace_half - delta)
                    )
                    / delta,
                )
            result_0 = 0.5 * (
                (numpy.exp(trace_half + delta) + numpy.exp(trace_half - delta))
                * target[:, 0]
                + coefficient
                * (centered00 * target[:, 0] + a01 * target[:, 1])
            )
            result_1 = 0.5 * (
                (numpy.exp(trace_half + delta) + numpy.exp(trace_half - delta))
                * target[:, 1]
                + coefficient
                * (a10 * target[:, 0] + centered11 * target[:, 1])
            )
            near_zero = numpy.abs(delta) <= 1.0e-14
            if numpy.any(near_zero):
                result_0[near_zero] = exp_trace[near_zero] * (
                    target[near_zero, 0]
                    + centered00[near_zero] * target[near_zero, 0]
                    + a01[near_zero] * target[near_zero, 1]
                )
                result_1[near_zero] = exp_trace[near_zero] * (
                    target[near_zero, 1]
                    + a10[near_zero] * target[near_zero, 0]
                    + centered11[near_zero] * target[near_zero, 1]
                )
            return numpy.asarray(
                numpy.real_if_close(
                    numpy.column_stack((result_0, result_1)),
                    tol=1000,
                ),
                dtype=float,
            )

        polarization_matrix = numpy.asarray(
            ((-0.9, 0.6), (0.1, -0.4)),
            dtype=float,
        )
        polarization_eigensystem = _cached_collision_eigendecomposition(
            polarization_matrix,
            {},
        )

        def _collision_batch(
            vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            dt: float,
            active: numpy.ndarray,
        ) -> numpy.ndarray:
            """Apply generated collision blocks to all non-tight rows."""

            if dt == 0.0:
                return numpy.asarray(vector, dtype=float)
            if numpy.all(active):
                return _tight_coupling_closure_batch(
                    numpy.asarray(vector, dtype=float),
                    step_index=step_index,
                    blend=blend,
                    active=active,
                )
            (_, a_value, _, _, _, collision_rate, _) = _background_values(
                step_index, blend
            )
            rate = max(collision_rate, 0.0)
            if rate <= 1.0e-12:
                return _tight_coupling_closure_batch(
                    numpy.asarray(vector, dtype=float),
                    step_index=step_index,
                    blend=blend,
                    active=active,
                )
            result = numpy.asarray(vector, dtype=float).copy()
            inactive = ~active
            dipole_temperature = temperature_indices.get(1)
            if dipole_temperature is not None and theta_b_index is not None:
                target = result[:, (dipole_temperature, theta_b_index)]
                gamma = 4.0 * omega_gamma / (3.0 * omega_b * a_value)
                updated = _exact_two_state_batch(
                    numpy.full(mode_count, -1.0),
                    1.0 / (3.0 * k_batch),
                    3.0 * k_batch * gamma,
                    numpy.full(mode_count, -gamma),
                    target,
                    scale=rate * dt,
                )
                result[inactive, dipole_temperature] = updated[inactive, 0]
                result[inactive, theta_b_index] = updated[inactive, 1]
            temperature_quadrupole = temperature_indices.get(2)
            polarization_quadrupole = polarization_indices.get(2)
            if (
                temperature_quadrupole is not None
                and polarization_quadrupole is not None
            ):
                target = result[
                    :, (temperature_quadrupole, polarization_quadrupole)
                ]
                eigenvalues, eigenvectors, inverse = polarization_eigensystem
                transformed = (
                    eigenvectors
                    @ (
                        numpy.exp(
                            numpy.asarray(eigenvalues, dtype=complex)
                            * rate
                            * float(dt)
                        )[:, numpy.newaxis]
                        * (inverse @ target.T)
                    )
                ).T
                updated = numpy.asarray(
                    numpy.real_if_close(transformed, tol=1000),
                    dtype=float,
                )
                result[inactive, temperature_quadrupole] = updated[inactive, 0]
                result[inactive, polarization_quadrupole] = updated[
                    inactive, 1
                ]
            damping = math.exp(-rate * float(dt))
            for moment, index in temperature_indices.items():
                if moment >= 3:
                    result[inactive, index] *= damping
            for moment, index in polarization_indices.items():
                if moment >= 3:
                    result[inactive, index] *= damping
            if phi_index is None:
                Hconf = _background_values(step_index, blend)[2]
                result[:, sync_eta_index] = (
                    result[:, -1] + Hconf * result[:, sync_alpha_index]
                )
            result = _tight_coupling_closure_batch(
                result,
                step_index=step_index,
                blend=blend,
                active=active,
            )
            return result

        def _record_batch(
            vector: numpy.ndarray,
            *,
            step_index: int,
            blend: float,
            active: numpy.ndarray,
        ) -> numpy.ndarray:
            """Apply the declared closure before storing a grid node."""

            return _tight_coupling_closure_batch(
                vector,
                step_index=step_index,
                blend=blend,
                active=active,
            )

        eta_values = numpy.asarray(active_grids["eta"], dtype=float)
        interval_count = max(int(eta_values.size) - 1, 0)
        active_intervals = numpy.zeros(
            (mode_count, interval_count),
            dtype=bool,
        )
        for row_index, mode_k in enumerate(k_batch):
            active_mode = _tight_coupling_is_active(
                active=False,
                collision_rate=float(active_grids["collision_rate"][0]),
                k_value=float(mode_k),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
            for step_index in range(interval_count):
                active_intervals[row_index, step_index] = active_mode
                active_mode = _tight_coupling_is_active(
                    active=active_mode,
                    collision_rate=float(
                        active_grids["collision_rate"][step_index + 1]
                    ),
                    k_value=float(mode_k),
                    tight_coupling_ratio=float(numerics.tight_coupling_ratio),
                )
        dt_values = numpy.diff(eta_values)
        phase_substeps = numpy.maximum(
            1,
            numpy.ceil(
                numpy.abs(dt_values)[numpy.newaxis, :] * k_batch[:, None] / 0.5
            ).astype(int),
        )
        collision_rates = numpy.maximum(
            numpy.asarray(active_grids["collision_rate"][:-1], dtype=float),
            numpy.asarray(active_grids["collision_rate"][1:], dtype=float),
        )
        collision_substeps = numpy.ceil(
            numpy.abs(dt_values)[numpy.newaxis, :] * collision_rates / 0.25
        ).astype(int)
        required_substeps = numpy.maximum(phase_substeps, collision_substeps)
        required_substeps = numpy.where(
            active_intervals,
            phase_substeps,
            required_substeps,
        )
        histories, final_states, stats = _integrate_batched_rk4(
            states,
            eta_values,
            required_substeps=required_substeps,
            active_intervals=active_intervals,
            rhs=_rhs_batch,
            pre_step=_collision_batch,
            post_step=_collision_batch,
            record_step=_record_batch,
        )
        source_histories = {
            name: histories[:, :, index]
            for name, index in state_indices.items()
            if name in history_names
        }
        if phi_index is None:
            final_states = final_states[:, :state_count]
        return source_histories, final_states, stats

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
        initial_state_context = _build_scalar_state_context(
            state,
            k_value=float(mode_k_value),
            eta_value=float(initial_eta),
            background_scalars=initial_background,
        )
        if generated_scalar_hierarchy:
            velocity_state_indices = tuple(
                int(slot.index)
                for slot in runtime_spec.state_slots
                if slot.order == 0
                and (
                    str(slot.variable)
                    in {
                        "theta_b",
                        "theta_c",
                        "theta_gamma1",
                        "theta_nu",
                    }
                    or str(slot.variable).startswith("theta_nu_massive_q")
                )
            )
            if velocity_state_indices:
                zero_velocity_state = state.copy()
                zero_velocity_state[list(velocity_state_indices)] = 0.0
                zero_velocity_context = _build_scalar_state_context(
                    zero_velocity_state,
                    k_value=float(mode_k_value),
                    eta_value=float(initial_eta),
                    background_scalars=initial_background,
                )
                zero_velocity_residual = float(
                    zero_velocity_context.get(
                        "einstein_energy_residual",
                        0.0,
                    )
                )
                current_velocity_residual = float(
                    initial_state_context.get(
                        "einstein_energy_residual",
                        zero_velocity_residual,
                    )
                )
                residual_slope = (
                    current_velocity_residual - zero_velocity_residual
                )
                if abs(residual_slope) > 1.0e-30:
                    velocity_factor = -zero_velocity_residual / residual_slope
                    if (
                        numpy.isfinite(velocity_factor)
                        and 0.25 <= float(velocity_factor) <= 4.0
                    ):
                        state[list(velocity_state_indices)] *= float(
                            velocity_factor
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
        _validate_generated_vector_initial_constraints(
            perturbation_data=perturbation_data,
            context=initial_state_context,
            k_value=float(mode_k_value),
        )
        return state, assigned_targets

    def _evaluate_source_histories(
        mode_k_value: float,
        source_histories: Mapping[str, numpy.ndarray],
    ) -> dict[str, numpy.ndarray]:
        """Evaluate declared sources and conservation on source-grid rows."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        active_grids = dict(source_grids)
        active_declared_background_histories = (
            source_declared_background_histories
        )
        active_coordinate_rate_histories = source_coordinate_rate_histories
        array_context = _build_array_context(
            source_histories,
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
        _validate_declared_conservation_rules(
            perturbation_data=perturbation_data,
            context=conservation_context,
            k_value=float(mode_k_value),
        )
        return source_arrays

    def _evolve_declared_mode(
        k_value: float,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
        """Integrate one Fourier mode through the declared graph."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal scalar_base_context_cache
        nonlocal active_k_value

        scalar_base_context_cache = {}
        active_k_value = float(k_value)

        end_boundary_entries = execution_plan.end_condition_entries
        (
            active_grids,
            active_declared_background_histories,
            active_coordinate_rate_histories,
        ) = _mode_grids_for_k(float(k_value))
        initial_eta, initial_background = _scalar_background_context(0, 0.0)

        collision_metadata_cache: dict[
            tuple[str, int, float], tuple[float, numpy.ndarray, float | None]
        ] = {}
        collision_eigendecomposition_cache: dict[
            tuple[tuple[int, ...], bytes],
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None,
        ] = {}

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
            _, background_scalars = _scalar_background_context(
                step_index,
                blend,
            )
            for runtime in split_collision_runtimes:
                if runtime.activation_strategy == "tight_coupling":
                    if not tight_coupling_active:
                        continue
                metadata_key = (runtime.name, int(step_index), float(blend))
                metadata = None
                if static_collision_runtimes.get(runtime.name, False):
                    metadata = collision_metadata_cache.get(metadata_key)
                if metadata is None:
                    eta_value = _blend_history(
                        active_grids["eta"],
                        step_index=step_index,
                        blend=blend,
                    )
                    if state_independent_collision_runtimes.get(
                        runtime.name, False
                    ):
                        scalar_context = _build_scalar_base_context(
                            k_value=float(k_value),
                            eta_value=float(eta_value),
                            background_scalars=background_scalars,
                            cache_token=(int(step_index), float(blend)),
                            resolve_graph=True,
                        )
                    else:
                        scalar_context = _build_scalar_state_context(
                            relaxed,
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
                    if static_collision_runtimes.get(runtime.name, False):
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
            target_stage_scale = 0.5
            required_substeps = max(
                1,
                int(
                    math.ceil(
                        abs(float(dt)) * stiffness_scale / target_stage_scale
                    )
                ),
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
                    stage_rhs_initial = _mode_rhs(
                        trial_state,
                        step_index=step_index,
                        blend=blend_start,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    stage_rhs_mid_a = _mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_initial,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    stage_rhs_mid_b = _mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_mid_a,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                        tight_coupling_active=tight_coupling_active,
                    )
                    stage_rhs_final = _mode_rhs(
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
                    if not numpy.all(numpy.isfinite(candidate_state)):
                        failure_detail = (
                            "exact collision sub-step end: "
                            f"{_describe_nonfinite_state(candidate_state)}"
                        )
                        failed = True
                        break
                    trial_state = candidate_state
                if not failed:
                    if generated_scalar_hierarchy and tight_coupling_active:
                        trial_state = (
                            _apply_generated_scalar_tight_coupling_closure(
                                trial_state,
                                step_index=step_index,
                                blend=1.0,
                                k_value=float(k_value),
                            )
                        )
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
            tight_coupling_active = _tight_coupling_is_active(
                active=False,
                collision_rate=float(active_grids["collision_rate"][0]),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
            for step_index, eta_value in enumerate(active_grids["eta"]):
                if generated_scalar_hierarchy and tight_coupling_active:
                    state = _apply_generated_scalar_tight_coupling_closure(
                        state,
                        step_index=step_index,
                        blend=0.0,
                        k_value=float(k_value),
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
        fast_generated_history = None
        if (
            generated_scalar_hierarchy
            and not momentum_runtimes
            and not end_boundary_entries
            and str(getattr(perturbation_data, "gauge", ""))
            in {"conformal_newtonian", "synchronous", "gauge_invariant"}
        ):
            fast_generated_history = _integrate_generated_scalar_history_fast(
                state,
                k_value=float(k_value),
            )
        if fast_generated_history is None and end_boundary_entries:
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
        if fast_generated_history is None:
            histories, final_state = _integrate_declared_state_history(state)
        else:
            histories, final_state = fast_generated_history
        final_residuals = _evaluate_end_boundary_residuals(final_state)
        if final_residuals.size and numpy.max(
            numpy.abs(final_residuals), initial=0.0
        ) > max(float(numerics.ode_atol) * 50.0, 1.0e-8):
            raise ValueError(
                "Declared end boundary conditions remained unsatisfied "
                "after integration."
            )
        source_histories = histories
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
        )
        return source_histories, source_arrays

    def _batch_generated_source_histories(
        mode_k_values: numpy.ndarray,
        *,
        envelope: dict[str, Any] | None = None,
    ) -> dict[int, tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]]:
        """Batch generated scalar modes that share one evolution grid."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        if envelope is None:
            envelope = runtime_envelope

        if (
            not generated_scalar_hierarchy
            or momentum_runtimes
            or execution_plan.end_condition_entries
            or str(getattr(perturbation_data, "gauge", ""))
            not in {"conformal_newtonian", "synchronous", "gauge_invariant"}
        ):
            return {}
        groups: dict[bytes, dict[str, Any]] = {}
        for mode_index, mode_k_value in enumerate(
            numpy.asarray(mode_k_values, dtype=float)
        ):
            mode_grids = _mode_grids_for_k(float(mode_k_value))
            eta_mode = numpy.asarray(mode_grids[0]["eta"], dtype=float)
            group_key = eta_mode.tobytes()
            group = groups.setdefault(
                group_key,
                {
                    "indices": [],
                    "k_values": [],
                    "grids": mode_grids,
                },
            )
            group["indices"].append(int(mode_index))
            group["k_values"].append(float(mode_k_value))
        results: dict[
            int,
            tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]],
        ] = {}
        for group in groups.values():
            if len(group["indices"]) < 4:
                continue
            (
                active_grids,
                active_declared_background_histories,
                active_coordinate_rate_histories,
            ) = group["grids"]
            initial_states = []
            for mode_k_value in group["k_values"]:
                state, _ = _prepare_mode_initial_state(mode_k_value)
                initial_states.append(state)
            source_histories_batch, _, batch_stats = (
                _integrate_generated_scalar_history_batch(
                    numpy.asarray(initial_states, dtype=float),
                    k_values_batch=numpy.asarray(
                        group["k_values"],
                        dtype=float,
                    ),
                )
            )
            envelope["batch_count"] = int(envelope.get("batch_count", 0)) + 1
            envelope["batch_mode_count"] = int(
                envelope.get("batch_mode_count", 0)
            ) + int(batch_stats.mode_count)
            envelope["batched_rk_stage_count"] = int(
                envelope.get("batched_rk_stage_count", 0)
            ) + int(batch_stats.rk_stage_count)
            envelope["batched_max_substeps"] = max(
                int(envelope.get("batched_max_substeps", 0)),
                int(batch_stats.maximum_substeps),
            )
            eta_mode = numpy.asarray(active_grids["eta"], dtype=float)
            for row_index, mode_index in enumerate(group["indices"]):
                mode_histories = {
                    name: numpy.asarray(values[row_index], dtype=float)
                    for name, values in source_histories_batch.items()
                }
                if not numpy.array_equal(eta_mode, source_grids["eta"]):
                    mode_histories = {
                        name: numpy.asarray(
                            numpy.interp(
                                source_grids["eta"],
                                eta_mode,
                                history,
                            ),
                            dtype=float,
                        )
                        for name, history in mode_histories.items()
                    }
                mode_k_value = float(group["k_values"][row_index])
                mode_sources = _evaluate_source_histories(
                    mode_k_value,
                    mode_histories,
                )
                results[mode_index] = (mode_histories, mode_sources)
        return results

    log_k_values = numpy.log(k_values)
    projection_ell_batch_size = 128
    batched_source_histories = _batch_generated_source_histories(k_values)

    for k_index, k_value in enumerate(k_values):
        batched_mode = batched_source_histories.get(int(k_index))
        if batched_mode is None:
            _, source_arrays = _evolve_declared_mode(float(k_value))
        else:
            _, source_arrays = batched_mode
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
        x_values = k_value * (eta0 - source_grids["eta"])
        x_signature = hashlib.sha256(
            numpy.asarray(x_values, dtype=float).tobytes()
        ).hexdigest()
        native_cache.store_bessel_inputs(
            x_signature,
            numpy.asarray(x_values, dtype=float).copy(),
        )
        mode_ell_limit = _projection_ell_limit_for_mode(
            ell_values=ell_arr,
            x_values=numpy.asarray(x_values, dtype=float),
        )
        mode_ell_indices = numpy.flatnonzero(ell_arr <= mode_ell_limit)
        if mode_ell_indices.size == 0:
            continue
        mode_ell_signature = tuple(
            int(ell_value) for ell_value in ell_arr[mode_ell_indices]
        )
        projection_bessel_values = _compute_spherical_bessel_batch(
            mode_ell_signature,
            numpy.asarray(x_values, dtype=float),
        )
        precomputed_projection_bessel = (
            mode_ell_signature,
            projection_bessel_values[0],
            projection_bessel_values[1],
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
            kernel_batch = _get_cached_declared_projection_kernel_batch(
                ell_signature,
                x_signature,
                precomputed_bessel=precomputed_projection_bessel,
            )
            for (
                component_name,
                component_entry,
            ) in transfer_component_observables.items():
                component_source_terms = component_entry.source_terms.items()
                source_histories = {
                    role_name: source_arrays[source_name]
                    for role_name, source_name in component_source_terms
                }
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
    for component_name, component_matrix in transfer_components.items():
        if not numpy.all(numpy.isfinite(component_matrix)):
            raise ValueError(
                "Declared transfer component produced non-finite values: "
                f"{component_name}"
            )

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
        direct_batched_source_histories = _batch_generated_source_histories(
            direct_k,
            envelope=direct_envelope,
        )
        for direct_k_index, direct_k_value in enumerate(direct_k):
            direct_batched_mode = direct_batched_source_histories.get(
                int(direct_k_index)
            )
            if direct_batched_mode is None:
                _, direct_source_arrays = _evolve_declared_mode(
                    float(direct_k_value)
                )
            else:
                _, direct_source_arrays = direct_batched_mode
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
                    source_histories = {
                        role_name: direct_source_arrays[source_name]
                        for role_name, source_name in (
                            component_entry.source_terms.items()
                        )
                    }
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
            if component_name == "temperature":
                j_second = (
                    float(ell_value * (ell_value + 1)) * inverse_x * inverse_x
                    - 1.0
                ) * j_values - 2.0 * inverse_x * j_derivatives
                projected = numpy.zeros(local_k.size, dtype=float)
                for role_name, kernel in (
                    ("monopole", j_values),
                    ("isw", j_values),
                    ("additive", j_values),
                    ("doppler", j_derivatives),
                    ("additive_derivative", j_second),
                ):
                    history = adaptive_source_histories.get(
                        (component_name, role_name)
                    )
                    if history is not None:
                        projected += numpy.sum(
                            kernel
                            * _interpolate_mode_histories(history, local_k)
                            * eta_integration_weights[numpy.newaxis, :],
                            axis=1,
                        )
                return projected
            if component_name == "polarization_e":
                prefactor = math.exp(
                    0.5
                    * (
                        math.lgamma(int(ell_value) + 3)
                        - math.lgamma(int(ell_value) - 1)
                    )
                )
                kernel = prefactor * j_values * inverse_x * inverse_x
            elif component_name == "lensing_potential":
                geometry = numpy.clip(
                    source_chi - source_grids["chi"],
                    0.0,
                    None,
                ) / (
                    max(float(source_chi), 1.0e-12)
                    * numpy.maximum(source_grids["chi"], 1.0e-12)
                )
                kernel = 2.0 * j_values * geometry[numpy.newaxis, :]
            else:
                raise ValueError(
                    "Adaptive scalar projection received unsupported "
                    f"component '{component_name}'"
                )
            source_roles = tuple(
                role_name
                for (component, role_name) in adaptive_source_histories
                if component == component_name
            )
            projected = numpy.zeros(local_k.size, dtype=float)
            for role_name in source_roles:
                history = adaptive_source_histories[
                    (component_name, role_name)
                ]
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
            if component_name == "temperature":
                j_second = (
                    bessel_order * (bessel_order + 1) * inverse_x * inverse_x
                    - 1.0
                ) * j_values - 2.0 * inverse_x * j_derivatives
                kernels = {
                    "monopole": j_values,
                    "isw": j_values,
                    "additive": j_values,
                    "doppler": j_derivatives,
                    "additive_derivative": j_second,
                }
            elif component_name == "polarization_e":
                prefactor = numpy.exp(
                    0.5 * (gammaln(ell_grid + 3.0) - gammaln(ell_grid - 1.0))
                )
                kernels = {
                    role_name: prefactor[:, numpy.newaxis, numpy.newaxis]
                    * j_values
                    * inverse_x
                    * inverse_x
                    for (component, role_name) in adaptive_source_histories
                    if component == component_name
                }
            elif component_name == "lensing_potential":
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
                kernels = {
                    role_name: 2.0 * j_values * geometry[None, None, :]
                    for (component, role_name) in adaptive_source_histories
                    if component == component_name
                }
            else:
                raise ValueError(
                    "Adaptive scalar projection received unsupported "
                    f"component '{component_name}'"
                )
            projected = numpy.zeros(local_k.shape, dtype=float)
            for role_name, kernel in kernels.items():
                history = adaptive_source_histories.get(
                    (component_name, role_name)
                )
                if history is None:
                    continue
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
            dense_k = numpy.linspace(
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
