r"""Declared scalar-mode evolution and runtime trace capture."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix

from ....perturbation_contract import (
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)
from ..errors import ConstraintViolationError, NonFiniteEvolutionError
from . import cache
from .adaptive import rate_transition_eta_nodes, resolve_feature_eta_grid
from .background import (
    _C_LIGHT_KM_S,
    _LEGACY_DECLARED_EVOLUTION_COORDINATES,
    _coerce_numeric_scalar,
    _physical_runtime_scalars,
    _resolve_declared_background_context,
)
from .collisions import (
    _cached_collision_eigendecomposition,
    _CompiledCollisionOperatorRuntime,
    _exact_batched_linear_collision_step,
    _exact_linear_collision_step,
    _solve_batched_small_declared_collision_target,
    _solve_declared_fast_collision_target,
    _structured_collision_components,
    declared_collision_phase_rates,
)
from .constraint_validation import (
    _validate_declared_conservation_rules,
    _validate_scalar_constraint_histories,
)
from .equation_kernel import (
    compile_grouped_equation_program,
    evaluate_scalar_equations,
    is_affine_state_system,
)
from .evolution import (
    _build_declared_base_context,
    _compile_batched_row_equation_program,
    _compile_expression_tuple_program,
    _compile_ordered_context_program,
    _compute_tight_coupling_drag,
    _declared_momentum_grid_context,
    _declared_runtime_seed,
    _evaluate_declared_initial_state,
    _integrate_batched_rk4,
    _interpolate_declared_histories,
    _nonuniform_gradient,
    _resolve_declared_graph_context,
    _resolve_declared_graph_context_ordered,
    _scalar_einstein_constraint_metrics,
    _split_collision_phase_scale,
    _tight_coupling_is_active,
    _validate_generated_scalar_initial_constraints,
    _validate_generated_tensor_initial_constraints,
    _validate_generated_vector_initial_constraints,
)
from .history_store import (
    StageContextColumns,
    allocate_native_history,
    with_native_history_storage,
)
from .line_of_sight import sample_eta_background_grids
from .numerical_controls import (
    _densify_eta_grid,
    _limit_eta_grid,
    group_mode_schedules,
)
from .spectrum_projection import _bind_declared_source_histories
from .stage_background import sample_declared_stage_contexts

_LOGGER = logging.getLogger(__name__)

# Keep the physical initial-condition threshold with the hierarchy owner.
_SCALAR_SUPERHORIZON_PREFIX_KETA = 5.5e-3
_SCALAR_INITIAL_SOLVE_NUMERICAL_TOLERANCE = 1.0e-9


def _apply_batched_collision_damping(
    states: numpy.ndarray,
    row_indices: numpy.ndarray,
    slot_indices: numpy.ndarray,
    damping: numpy.ndarray,
) -> None:
    """Retain exact moment damping in the selected accepted state rows."""

    selected = numpy.ix_(row_indices, slot_indices)
    # Advanced indexing returns a copy; write the damped moments back.
    moments = states[selected]
    moments *= damping[:, numpy.newaxis]
    states[selected] = moments


def _integrate_scaled_continuous_history(
    rhs: Callable[..., numpy.ndarray],
    initial_state: numpy.ndarray,
    eta_values: numpy.ndarray,
    *,
    initial_time_scale: float,
    rtol: float,
    atol: float,
    affine_system: bool = False,
    vectorized_rhs=None,
    method: str = "BDF",
) -> numpy.ndarray:
    """Integrate declared states with component-scaled stiff error control.

    Regular initial data can mix order-one metric potentials with shear
    and velocity moments many orders of magnitude smaller. An absolute
    error budget in the raw coordinates can erase those moments even when
    their contribution to the metric is large. Normalize each coordinate
    by its initial amplitude or its initial drive over a physical time
    scale, then propagate scales through coupled initially zero states.
    Only unseeded coordinates with no detected initial coupling retain
    unit scale. This avoids unbalanced Jacobians in higher-moment chains.
    """

    initial = numpy.asarray(initial_state, dtype=float)
    eta = numpy.asarray(eta_values, dtype=float)
    initial_drive = numpy.asarray(rhs(float(eta[0]), initial), dtype=float)
    scales = numpy.maximum(
        numpy.abs(initial),
        float(initial_time_scale) * numpy.abs(initial_drive),
    )
    pending = list(numpy.flatnonzero(scales > 0.0))
    while pending and numpy.any(scales == 0.0):
        unresolved = scales == 0.0
        driven_scales = numpy.zeros_like(scales)
        if affine_system and vectorized_rhs is not None:
            probes = numpy.repeat(initial[:, None], len(pending), axis=1)
            for column, index in enumerate(pending):
                probes[index, column] += math.copysign(
                    scales[index], initial[index]
                )
            responses = numpy.asarray(
                vectorized_rhs(float(eta[0]), probes), dtype=float
            )
            driven_scales = numpy.max(
                float(initial_time_scale)
                * abs(responses - initial_drive[:, None]),
                axis=1,
            )
        else:
            for index in pending:
                probe = initial.copy()
                probe[index] += math.copysign(scales[index], initial[index])
                response = numpy.asarray(
                    rhs(float(eta[0]), probe), dtype=float
                )
                driven_scales = numpy.maximum(
                    driven_scales,
                    float(initial_time_scale) * abs(response - initial_drive),
                )
        discovered = unresolved & (driven_scales > 0.0)
        scales[discovered] = driven_scales[discovered]
        pending = list(numpy.flatnonzero(discovered))
    if not numpy.all(numpy.isfinite(scales)):
        raise ValueError("Declared continuous initial scales are non-finite")
    scales = numpy.where(scales > 0.0, scales, 1.0)

    def scaled_rhs(
        eta_value: float,
        scaled_state: numpy.ndarray,
    ) -> numpy.ndarray:
        """Evaluate the same declared equations in normalized coordinates."""

        return numpy.asarray(rhs(eta_value, scaled_state * scales)) / scales

    def scaled_jacobian(
        eta_value: float,
        scaled_state: numpy.ndarray,
    ) -> numpy.ndarray:
        """Resolve derivatives without perturbing tiny raw states by atol.

        Nonlinear systems use centered differences with the cube-root
        roundoff step in normalized coordinates. Initially zero or
        identically constant moments do not trigger vanishing probes.
        """

        if affine_system and vectorized_rhs is not None:
            probes = numpy.column_stack(
                (numpy.zeros(initial.size), numpy.eye(initial.size))
            )
            derivatives = (
                numpy.asarray(
                    vectorized_rhs(eta_value, probes * scales[:, None])
                )
                / scales[:, None]
            )
            return derivatives[:, 1:] - derivatives[:, :1]
        if affine_system:
            # For a compiler-proven affine system, coefficient probes at
            # zero avoid subtracting tiny perturbations of a large current
            # trajectory. No sampled linearity assumption is used.
            origin = scaled_rhs(eta_value, numpy.zeros(initial.size))
            matrix = numpy.empty((initial.size, initial.size))
            for index in range(initial.size):
                probe = numpy.zeros(initial.size)
                probe[index] = 1.0
                matrix[:, index] = scaled_rhs(eta_value, probe) - origin
            return matrix
        steps = numpy.cbrt(numpy.finfo(float).eps) * numpy.maximum(
            numpy.abs(scaled_state), 1.0
        )
        jacobian = numpy.empty((initial.size, initial.size), dtype=float)
        for index, step in enumerate(steps):
            above = scaled_state.copy()
            below = scaled_state.copy()
            above[index] += step
            below[index] -= step
            jacobian[:, index] = (
                scaled_rhs(eta_value, above) - scaled_rhs(eta_value, below)
            ) / (2.0 * step)
        return jacobian

    use_sparse_jacobian = None

    def solver_jacobian(eta_value, scaled_state):
        """Choose LU storage once, retaining every nonzero coefficient.

        Hierarchy Jacobians couple few neighboring moments. Sparse LU
        avoids factoring their large zero regions. The storage decision
        stays fixed because SciPy binds its dense/sparse backend at startup;
        coefficients and the nonzero pattern may still change with time.
        """

        nonlocal use_sparse_jacobian
        matrix = scaled_jacobian(eta_value, scaled_state)
        if use_sparse_jacobian is None:
            use_sparse_jacobian = numpy.count_nonzero(matrix) * 4 < matrix.size
        return csc_matrix(matrix) if use_sparse_jacobian else matrix

    solution = solve_ivp(
        scaled_rhs,
        (float(eta[0]), float(eta[-1])),
        initial / scales,
        method=method,
        jac=solver_jacobian,
        t_eval=eta,
        rtol=float(rtol),
        atol=float(atol),
    )
    if not solution.success:
        raise ValueError(
            "Declared CMB continuous evolution failed: " f"{solution.message}"
        )
    if solution.y.shape != (initial.size, eta.size):
        raise ValueError(
            "Declared CMB continuous evolution returned an "
            "incomplete state history"
        )
    history = numpy.asarray(solution.y * scales[:, numpy.newaxis], dtype=float)
    if not numpy.all(numpy.isfinite(history)):
        raise ValueError(
            "Declared CMB continuous evolution produced non-finite states"
        )
    return history


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
    scalar control rather than a common explicit schedule. One pending row
    keeps its compatible schedule when cache reuse supplies the other rows.
    """

    if not (
        generated_scalar_hierarchy
        and shared_mode_grids_enabled
        and int(mode_count) > 0
    ):
        return False
    if has_end_boundaries or continuous_collision_solver:
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


@dataclass(frozen=True)
class DeclaredModeEvolutionInputs:
    """Explicit dependencies for one declared hierarchy evolution runtime."""

    active_coordinate_rate_histories: Any
    active_declared_background_histories: Any
    active_grids: Any
    active_k_value: Any
    adaptive_controls: Any
    adaptive_k_enabled: Any
    background: Any
    collision_kernel_metrics: Any
    contract_or_params: Any
    declared_accuracy_controls: Any
    diagnostic_source_audit: Any
    equation_direct_names: Any
    equation_program: Any
    equation_program_specs: Any
    equation_required_names: Any
    equation_stage_derivative_steps: Any
    exact_collision_phase_cap: Any
    execution_plan: Any
    generated_final_evolution_floor: Any
    generated_final_phase_step: Any
    generated_scalar_hierarchy: Any
    hierarchy_equation_residuals_by_k: Any
    hierarchy_schedule_evidence: Any
    initial_state_cache_hits: Any
    initial_state_cache_misses: Any
    initial_state_diagnostics_by_k: Any
    k_values: Any
    manifest_summary: Any
    metric_history_gradient_residual_by_k: Any
    model_label: Any
    momentum_grid_context_cache: Any
    momentum_runtimes: Any
    numerics: Any
    performance_timer: Any
    perturbation_data: Any
    physical_params: Any
    physical_runtime_scalars: Any
    required_source_names: Any
    runtime_envelope: Any
    runtime_spec: Any
    scalar_background_context_cache: Any
    scalar_base_context_cache: Any
    scalar_constraint_diagnostic_projection_count: Any
    scalar_constraint_diagnostics: Any
    scalar_constraint_projection_count: Any
    scalar_constraint_projection_max_relative_correction: Any
    shared_generated_mode_grids: Any
    shared_generated_mode_grids_enabled: Any
    source_context_max_abs_by_k: Any
    source_context_pre_resolution_by_k: Any
    source_coordinate_rate_histories: Any
    source_declared_background_histories: Any
    source_grids: Any
    source_history_cache_prefix: Any
    source_history_max_abs: Any
    source_history_max_abs_by_k: Any
    source_history_mode_count: Any
    source_history_reconstruction_enabled: Any
    source_history_residual_samples_by_k: Any
    source_parameters: Any
    split_collision_runtimes: Any
    stage_derivative_steps: Any
    stage_diagnostic_fields: Any
    stage_diagnostic_histories_by_k: Any
    stage_diagnostic_k_values: Any
    stage_value_steps: Any
    state_history_max_abs_by_k: Any
    state_history_polarization_ratio_by_k: Any
    transfer_component_observables: Any


@dataclass(frozen=True)
class DeclaredModeEvolutionRuntime:
    """Callbacks and mutable-state snapshot emitted by the evolution owner."""

    evolve_declared_mode: Callable[..., Any]
    evolve_declared_modes_batched: Callable[..., Any]
    record_source_history_diagnostics: Callable[..., Any]
    record_stage_diagnostic_histories: Callable[..., Any]
    scalar_initial_constraint_preflight: Any
    snapshot: Callable[[], dict[str, Any]]
    mode_grids_for_k: Callable[..., Any]
    history_capture_k_values: set[float]
    resample_mode_history: Callable[..., Any]


def build_declared_mode_evolution(
    inputs: DeclaredModeEvolutionInputs,
) -> DeclaredModeEvolutionRuntime:
    """Bind declared physics to numerical runtime state for one request."""

    active_coordinate_rate_histories = inputs.active_coordinate_rate_histories
    active_declared_background_histories = (
        inputs.active_declared_background_histories
    )
    active_grids = inputs.active_grids
    active_k_value = inputs.active_k_value
    adaptive_controls = inputs.adaptive_controls
    adaptive_k_enabled = inputs.adaptive_k_enabled
    background = inputs.background
    collision_kernel_metrics = inputs.collision_kernel_metrics
    contract_or_params = inputs.contract_or_params
    declared_accuracy_controls = inputs.declared_accuracy_controls
    diagnostic_source_audit = inputs.diagnostic_source_audit
    equation_direct_names = inputs.equation_direct_names
    equation_program = inputs.equation_program
    equation_program_specs = inputs.equation_program_specs
    equation_required_names = inputs.equation_required_names
    equation_stage_derivative_steps = inputs.equation_stage_derivative_steps
    exact_collision_phase_cap = inputs.exact_collision_phase_cap
    execution_plan = inputs.execution_plan
    generated_final_evolution_floor = inputs.generated_final_evolution_floor
    generated_final_phase_step = inputs.generated_final_phase_step
    generated_scalar_hierarchy = inputs.generated_scalar_hierarchy
    hierarchy_equation_residuals_by_k = (
        inputs.hierarchy_equation_residuals_by_k
    )
    hierarchy_schedule_evidence = inputs.hierarchy_schedule_evidence
    initial_state_cache_hits = inputs.initial_state_cache_hits
    initial_state_cache_misses = inputs.initial_state_cache_misses
    initial_state_diagnostics_by_k = inputs.initial_state_diagnostics_by_k
    k_values = inputs.k_values
    manifest_summary = inputs.manifest_summary
    metric_history_gradient_residual_by_k = (
        inputs.metric_history_gradient_residual_by_k
    )
    model_label = inputs.model_label
    momentum_grid_context_cache = inputs.momentum_grid_context_cache
    momentum_runtimes = inputs.momentum_runtimes
    numerics = inputs.numerics
    performance_timer = inputs.performance_timer
    perturbation_data = inputs.perturbation_data
    physical_params = inputs.physical_params
    physical_runtime_scalars = inputs.physical_runtime_scalars
    required_source_names = inputs.required_source_names
    runtime_envelope = inputs.runtime_envelope
    runtime_spec = inputs.runtime_spec
    state_context_names = tuple(
        (
            slot.variable
            if slot.order == 0
            else f"__d{slot.order}_{slot.variable}_{slot.wrt}"
        )
        for slot in runtime_spec.state_slots
    )
    scalar_rhs_calls = 0
    scalar_work_scope = "scalar_preflight"
    history_capture_k_values: set[float] = set()
    mode_grid_cache: dict[tuple[float, int | None], Any] = {}
    coupling_transition_nodes: numpy.ndarray | None = None
    scalar_background_context_cache = inputs.scalar_background_context_cache
    scheduled_background_contexts = None
    scalar_base_context_cache = inputs.scalar_base_context_cache
    scalar_constraint_diagnostic_projection_count = (
        inputs.scalar_constraint_diagnostic_projection_count
    )
    scalar_constraint_diagnostics = inputs.scalar_constraint_diagnostics
    scalar_constraint_projection_count = (
        inputs.scalar_constraint_projection_count
    )
    scalar_constraint_projection_max_relative_correction = (
        inputs.scalar_constraint_projection_max_relative_correction
    )
    shared_generated_mode_grids = inputs.shared_generated_mode_grids
    shared_generated_mode_grids_enabled = (
        inputs.shared_generated_mode_grids_enabled
    )
    source_context_max_abs_by_k = inputs.source_context_max_abs_by_k
    source_context_pre_resolution_by_k = (
        inputs.source_context_pre_resolution_by_k
    )
    source_coordinate_rate_histories = inputs.source_coordinate_rate_histories
    source_declared_background_histories = (
        inputs.source_declared_background_histories
    )
    source_grids = inputs.source_grids
    source_history_cache_prefix = inputs.source_history_cache_prefix
    source_history_max_abs = inputs.source_history_max_abs
    source_history_max_abs_by_k = inputs.source_history_max_abs_by_k
    source_history_mode_count = inputs.source_history_mode_count
    source_history_reconstruction_enabled = (
        inputs.source_history_reconstruction_enabled
    )
    source_history_residual_samples_by_k = (
        inputs.source_history_residual_samples_by_k
    )
    source_parameters = inputs.source_parameters
    split_collision_runtimes = inputs.split_collision_runtimes
    stage_derivative_steps = inputs.stage_derivative_steps
    stage_diagnostic_fields = inputs.stage_diagnostic_fields
    stage_diagnostic_histories_by_k = inputs.stage_diagnostic_histories_by_k
    stage_diagnostic_k_values = inputs.stage_diagnostic_k_values
    stage_value_steps = inputs.stage_value_steps
    state_history_max_abs_by_k = inputs.state_history_max_abs_by_k
    state_history_polarization_ratio_by_k = (
        inputs.state_history_polarization_ratio_by_k
    )
    transfer_component_observables = inputs.transfer_component_observables

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
    # Auditing a history must not select a different differential solver.
    # Continuous collision integration is an explicit declaration shared by
    # the scalar and batch capability decisions.
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
    requires_continuous_collisions = continuous_collision_solver
    has_ppf = any(
        getattr(item, "scalar_closure", {}).get("kind") == "ppf"
        for item in perturbation_data.species.values()
    )
    if has_ppf:
        # PPF has a physical relaxation rate proportional to k^2/Hconf.
        # Scalar fallbacks integrate the coupled graph implicitly. A later
        # batch capability check can select exponential relaxation when the
        # declared rate is independent of perturbation states.
        continuous_collision_solver = True
    runtime_envelope["continuous_collision_solver"] = (
        continuous_collision_solver
    )

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
                expansion_rate=float(active_grids["Hconf"][index]),
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
            "acoustic_k_sq",
            "Phi",
            "Hconf",
            "Psi",
            "einstein_gravity_strength",
            "total_density_source",
            "total_momentum_source",
        }.issubset(context):
            hubble = numpy.asarray(context["Hconf"], dtype=float)
            wavenumber_sq = numpy.asarray(
                context["acoustic_k_sq"], dtype=float
            )
            gravity_strength = float(context["einstein_gravity_strength"])
            density_source = numpy.asarray(
                context["total_density_source"], dtype=float
            )
            psi_history = numpy.asarray(context["Psi"], dtype=float)
            energy_source = (
                wavenumber_sq * phi_history
                + 1.5 * gravity_strength * density_source
            )
            momentum_source = numpy.asarray(
                context["total_momentum_source"], dtype=float
            )
            expected_phi_tau = -hubble * psi_history + (
                1.5 * gravity_strength * momentum_source
                - hubble * energy_source / 3.0
            ) / (hubble * hubble + wavenumber_sq)
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

    def _phase_step_for_interval(
        *,
        step_index: int,
        blend: float = 0.5,
    ) -> float:
        """Return the phase step required by one generated-mode interval."""

        if exact_collision_phase_cap:
            return generated_final_phase_step
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

        if scheduled_background_contexts is not None:
            scheduled_grids, scheduled_wave, scheduled_columns = (
                scheduled_background_contexts
            )
            if (
                active_grids is scheduled_grids
                and context_key[2] == scheduled_wave
            ):
                context = scheduled_columns.get(context_key[:2])
                if context is not None:
                    epoch = float(context.pop("__scheduled_epoch"))
                    cached_context = (epoch, context)
                    scalar_background_context_cache[context_key] = (
                        cached_context
                    )
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
        if generated_scalar_hierarchy:
            # The generated hierarchy can start at eta values many orders
            # below recombination.  Linear interpolation of Hconf across a
            # sparse evolution interval turns the radiation-era relation
            # Hconf ~ 1/eta into an O(1) stage error, which then breaks the
            # regular Einstein momentum cancellation.  Use the background's
            # monotone physical interpolants at RK stage coordinates instead
            # of interpolating the sampled history values a second time.
            sampled_background = background.sample(float(eta_value))

            def _sampled_scalar(name: str) -> float:
                """Return one finite scalar from a sampled background."""

                values = numpy.asarray(sampled_background[name], dtype=float)
                return float(values.reshape(-1)[-1])

            sampled_a = _sampled_scalar("a")
            sampled_z = _sampled_scalar("z")
            try:
                a_tau = float(
                    background.a_of_eta.derivative()(float(eta_value))
                )
                a_tau_tau = float(
                    background.a_of_eta.derivative(2)(float(eta_value))
                )
                sampled_hconf = a_tau / max(sampled_a, 1.0e-30)
                sampled_h = (
                    sampled_hconf * _C_LIGHT_KM_S / max(sampled_a, 1.0e-30)
                )
                sampled_hconf_tau = (
                    a_tau_tau / max(sampled_a, 1.0e-30)
                    - sampled_hconf * sampled_hconf
                )
            except (AttributeError, TypeError, ValueError):
                sampled_h = _sampled_scalar("H")
                sampled_hconf = sampled_a * sampled_h / _C_LIGHT_KM_S
                sampled_hconf_tau = _blend_history(
                    active_grids["Hconf_tau"],
                    step_index=step_index,
                    blend=blend,
                )
            sampled_collision_rate = max(
                -_sampled_scalar("tau_dot"),
                0.0,
            )
            scalar_context.update(
                {
                    "a": sampled_a,
                    "z": sampled_z,
                    "H": sampled_h,
                    "Hconf": sampled_hconf,
                    "Hconf_tau": sampled_hconf_tau,
                    "tau": _sampled_scalar("tau"),
                    "tau_dot": _sampled_scalar("tau_dot"),
                    "visibility": _sampled_scalar("visibility"),
                    "chi": _sampled_scalar("chi"),
                    "angular_diameter_distance": _sampled_scalar(
                        "angular_diameter_distance"
                    ),
                    "sound_speed": _sampled_scalar("sound_speed"),
                    "baryon_sound_speed_sq": _sampled_scalar(
                        "baryon_sound_speed_sq"
                    ),
                    "collision_rate": sampled_collision_rate,
                    "sound_speed_sq": 1.0
                    / (
                        3.0
                        * (
                            1.0
                            + (
                                3.0
                                * physical_params.Omega_b0
                                * sampled_a
                                / max(
                                    4.0 * physical_params.Omega_gamma0,
                                    1.0e-12,
                                )
                            )
                        )
                    ),
                }
            )
            scalar_context["baryon_loading"] = (
                3.0
                * physical_params.Omega_b0
                * sampled_a
                / max(4.0 * physical_params.Omega_gamma0, 1.0e-12)
            )
        collision_rate = _blend_history(
            active_grids["collision_rate"],
            step_index=step_index,
            blend=blend,
        )
        if generated_scalar_hierarchy:
            collision_rate = float(scalar_context["collision_rate"])
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
        if generated_scalar_hierarchy:
            declared_stage = _resolve_declared_background_context(
                contract_or_params,
                a_values=float(scalar_context["a"]),
                z_values=float(scalar_context["z"]),
            )
            for name, value in declared_stage.items():
                if name in {"a", "z"}:
                    continue
                values = numpy.asarray(value, dtype=float)
                scalar_context[name] = float(values.reshape(-1)[-1])
            if "Hconf" not in declared_stage:
                # H(a) has just been evaluated from the declaration. The
                # derivative of an independently interpolated a(eta) is a
                # different numerical approximation; mixing the two drives
                # an artificial superhorizon Einstein-constraint mode.
                scalar_context["Hconf"] = (
                    float(scalar_context["a"])
                    * float(scalar_context["H"])
                    / _C_LIGHT_KM_S
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
            # Evaluate the chain-rule rate from the same stage values that
            # a declared equation sees. A separately sampled ``Hconf`` can
            # introduce a mesh-dependent cross-term when paired with ``a``.
            # The relation da/deta = a^2 H/c keeps equivalent coordinate
            # declarations invariant at every stage.
            scale_factor = float(scalar_context["a"])
            rate = (
                scale_factor
                * scale_factor
                * float(scalar_context["H"])
                / _C_LIGHT_KM_S
            )
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
                return sample_eta_background_grids(
                    eta_grid,
                    background=background,
                    physical_params=physical_params,
                    contract_or_params=contract_or_params,
                )
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

        # Share meshes within deterministic factor-two wave cohorts. Using
        # the entire request's largest wave here forced every low-k native
        # reference to resolve unrelated high-k oscillations, and changed a
        # mode's trajectory when an unrelated wave was added to a request.
        # Cohorts require independent native-error acceptance. Without that
        # check, retain the shared request mesh rather than coarsening a
        # diagnostic trajectory without measuring its interpolation error.
        maximum_mode_k = (
            (
                math.exp2(math.ceil(math.log2(abs_k)))
                if adaptive_controls.evolution_enabled
                else max(abs_k, float(numpy.max(numpy.abs(k_values))))
            )
            if shared_generated_mode_grids_enabled
            else abs_k
        )
        mode_grid_key = (maximum_mode_k, evolution_sample_count_override)

        nonlocal shared_generated_mode_grids
        nonlocal coupling_transition_nodes
        if (
            shared_generated_mode_grids_enabled
            and mode_grid_key in mode_grid_cache
        ):
            return mode_grid_cache[mode_grid_key]
        if (
            adaptive_controls.evolution_enabled
            and coupling_transition_nodes is None
        ):

            def _evaluate_transition_rates(
                probe_eta: numpy.ndarray,
            ) -> tuple[numpy.ndarray, numpy.ndarray]:
                """Evaluate the declared background at shared event probes."""

                unique_eta, inverse = numpy.unique(
                    numpy.concatenate(
                        (source_grids["eta"][[0, -1]], probe_eta)
                    ),
                    return_inverse=True,
                )
                inverse = inverse[2:]
                sampled = sample_eta_background_grids(
                    unique_eta,
                    background=background,
                    physical_params=physical_params,
                    contract_or_params=contract_or_params,
                )[0]
                return (
                    sampled["collision_rate"][inverse],
                    sampled["Hconf"][inverse],
                )

            # A source sampling change is not a different physical event.
            # Seed event searches on the accepted physical background so
            # source-only refinements retain the same native evolution grid.
            physical_grids = sample_eta_background_grids(
                numpy.asarray(background.eta_grid),
                background=background,
                physical_params=physical_params,
                contract_or_params=contract_or_params,
            )[0]
            positive_waves = numpy.abs(numpy.asarray(k_values, dtype=float))
            positive_waves = positive_waves[positive_waves > 0]
            # Resolve the physical transition band at deterministic cohort
            # boundaries. Adding quadrature anchors inside the same domain
            # cannot change every existing mode's native mesh/cache identity.
            transition_waves = numpy.exp2(
                numpy.arange(
                    math.floor(math.log2(float(positive_waves.min()))),
                    math.ceil(math.log2(float(positive_waves.max()))) + 1,
                )
            )
            coupling_transition_nodes = rate_transition_eta_nodes(
                physical_grids["eta"],
                physical_grids["collision_rate"],
                physical_grids["Hconf"],
                transition_waves,
                ratios=(
                    float(numerics.tight_coupling_ratio),
                    float(numerics.tight_coupling_ratio)
                    * float(numerics.tight_coupling_exit_ratio),
                ),
                evaluate_rates=_evaluate_transition_rates,
            )

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
            limited = _limit_eta_grid(
                numpy.unique(
                    numpy.concatenate(
                        (
                            numpy.asarray((float(eta_floor),)),
                            base_grid[keep],
                        )
                    )
                ),
                maximum_samples=int(requested_samples),
            )
            # Diagnostic refinement controls are required to change the
            # actual hierarchy grid.  Visibility-aware thinning can leave
            # fewer retained points than requested; densify that result so
            # coarse/intermediate/fine runs cannot silently collapse onto
            # one identical grid.
            if (
                sample_count_override is not None
                or adaptive_controls.evolution_enabled
            ):
                limited = _densify_eta_grid(
                    limited,
                    minimum_samples=int(requested_samples),
                )
            if adaptive_controls.evolution_enabled:
                # Linear interpolation of a unit oscillation has error at
                # most phase_step**2/8. Reserve one eighth of the history
                # amplitude budget for that error, using the actual largest
                # evolved k. Coarse controls enlarge this phase step, so an
                # independent refinement cannot collapse onto one mesh.
                # Use the stable engine minimum as the refinement scale.
                # Doubling a nominal control must tighten the feature and
                # phase criteria; normalizing to the current fine control
                # would keep its criterion unchanged after every promotion.
                # Ordinary final requests apply these same criteria instead
                # of bypassing them with the unrelated background mesh.
                reference_count = max(
                    int(adaptive_controls.evolution_minimum_nodes), 1
                )
                dense_eta = numpy.unique(
                    numpy.concatenate(
                        (
                            numpy.asarray((float(eta_floor),)),
                            base_grid,
                        )
                    )
                )
                # Evolution planning reads the physical background directly.
                # Its rates/features must not inherit the independently
                # selected source/projection representation error.
                physical_history = background.sample(dense_eta)
                profiles = {
                    "collision_rate": numpy.maximum(
                        -numpy.asarray(physical_history["tau_dot"]), 0.0
                    ),
                    "Hconf": numpy.asarray(physical_history["a"])
                    * numpy.asarray(physical_history["H"])
                    / _C_LIGHT_KM_S,
                    "visibility": numpy.asarray(
                        physical_history["visibility"]
                    ),
                }
                rate_floor = numpy.maximum(
                    numpy.abs(profiles["Hconf"]), maximum_mode_k
                )
                feature_tolerance = (
                    adaptive_controls.evolution_relative_tolerance
                    * (reference_count / int(requested_samples)) ** 2
                    / 8.0
                )
                visibility_floor = max(
                    float(numpy.max(numpy.abs(profiles["visibility"])))
                    * adaptive_controls.evolution_relative_tolerance,
                    1e-30,
                )
                limited = resolve_feature_eta_grid(
                    dense_eta,
                    limited,
                    profiles=profiles,
                    scales={
                        "collision_rate": numpy.maximum(
                            numpy.abs(profiles["collision_rate"]), rate_floor
                        ),
                        "Hconf": numpy.maximum(
                            numpy.abs(profiles["Hconf"]), 1e-30
                        ),
                        "visibility": numpy.maximum(
                            numpy.abs(profiles["visibility"]),
                            visibility_floor,
                        ),
                    },
                    relative_tolerance=feature_tolerance,
                )
                phase_step = (
                    numpy.sqrt(adaptive_controls.evolution_relative_tolerance)
                    * reference_count
                    / int(requested_samples)
                )
                phase_intervals = max(
                    1,
                    int(
                        numpy.ceil(
                            maximum_mode_k
                            * (limited[-1] - limited[0])
                            / phase_step
                        )
                    ),
                )
                limited = numpy.unique(
                    numpy.concatenate(
                        (
                            limited,
                            numpy.linspace(
                                limited[0], limited[-1], phase_intervals + 1
                            ),
                            coupling_transition_nodes[
                                (coupling_transition_nodes >= limited[0])
                                & (coupling_transition_nodes <= limited[-1])
                            ],
                        )
                    )
                )
            return limited

        # The hidden prefix must start on the earliest physical background
        # point.  Starting at the first sparse background point above the
        # super-horizon target can create an order-unity eta jump and excite
        # the regular mode before the visible line-of-sight grid begins.
        eta_background_start = float(background.eta_grid[0])
        eta_target = min(
            source_eta_start,
            _SCALAR_SUPERHORIZON_PREFIX_KETA / maximum_mode_k,
        )
        eta_target = min(eta_target, source_eta_start)
        if eta_background_start < source_eta_start:
            # Use the native background grid for the complete prefix.  This
            # makes an early LOS request and a later LOS request integrate the
            # same regular solution up to the later request's start instead
            # of comparing trajectories produced by different meshes.
            eta_prefix = numpy.asarray(
                background.eta_grid[
                    (background.eta_grid >= eta_background_start)
                    & (background.eta_grid < source_eta_start)
                ],
                dtype=float,
            )
            eta_target = float(eta_prefix[0])
        else:
            eta_prefix = numpy.asarray((), dtype=float)
        if adaptive_controls.evolution_enabled and eta_prefix.size:
            prefix_phase_step = (
                numpy.sqrt(adaptive_controls.evolution_relative_tolerance)
                * max(int(adaptive_controls.evolution_minimum_nodes), 1)
                / max(int(numerics.evolution_eta_sample_count or 1), 1)
            )
            prefix_intervals = max(
                1,
                int(
                    numpy.ceil(
                        maximum_mode_k
                        * (source_eta_start - eta_target)
                        / prefix_phase_step
                    )
                ),
            )
            eta_prefix = numpy.unique(
                numpy.concatenate(
                    (
                        eta_prefix,
                        numpy.linspace(
                            eta_target, source_eta_start, prefix_intervals + 1
                        ),
                    )
                )
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
        # A generated scalar hierarchy must evolve every request on one
        # physical eta mesh.  Starting the post-prefix mesh at the caller's
        # LOS start makes an early-start and a late-start request follow
        # different discretised trajectories, so their common visible
        # history is not a valid observable invariant.  Use a bounded
        # engine-owned mesh from the earliest background point; production
        # final controls may still request the full native grid below.
        common_scalar_evolution = bool(
            generated_scalar_hierarchy
            and evolution_sample_count_override is None
            and not adaptive_controls.evolution_enabled
            and not stage_diagnostic_k_values
        )
        if common_scalar_evolution:
            requested_evolution_samples = max(
                int(requested_evolution_samples or 0),
                512,
            )
            # The hierarchy history is a physical product, not a
            # line-of-sight product.  Building it from ``source_eta_start``
            # makes an early-start and a late-start request use different
            # meshes over their shared interval, so the later request can
            # drift even though both requests have the same declaration.
            # Evolve the common scalar history from the earliest background
            # surface on one engine-owned mesh and sample it separately for
            # each request below.
            eta_mode_grid = _evolution_eta_grid(
                eta_background_start,
                sample_count_override=int(requested_evolution_samples),
            )
            sampled_mode_grids = sample_eta_background_grids(
                eta_mode_grid,
                background=background,
                physical_params=physical_params,
                contract_or_params=contract_or_params,
            )
            if shared_generated_mode_grids_enabled:
                shared_generated_mode_grids = sampled_mode_grids
                mode_grid_cache[mode_grid_key] = sampled_mode_grids
            return sampled_mode_grids
        post_source_sample_count = None
        if requested_evolution_samples is not None:
            # The mandatory early prefix is additional physical work. It
            # must not consume the controlled late-time grid or make three
            # independent resolution tiers identical.
            post_source_sample_count = max(
                16,
                int(requested_evolution_samples),
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
        sampled_mode_grids = sample_eta_background_grids(
            eta_mode_grid,
            background=background,
            physical_params=physical_params,
            contract_or_params=contract_or_params,
        )
        if (
            shared_generated_mode_grids_enabled
            and evolution_sample_count_override is None
        ):
            shared_generated_mode_grids = sampled_mode_grids
        if shared_generated_mode_grids_enabled:
            mode_grid_cache[mode_grid_key] = sampled_mode_grids
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
        # Slots are contiguous in compiler order. Keep scalar conversion
        # identical while avoiding repeated metadata lookup per RK stage.
        state_values = (
            map(float, state_vector)
            if state_vector.ndim == 1
            else state_vector
        )
        context.update(zip(state_context_names, state_values))
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
        background_grids=None,
        include_states: bool = True,
    ) -> dict[str, Any]:
        """Return the array-valued expression environment for one mode."""

        context_grids = (
            active_grids if background_grids is None else background_grids[0]
        )
        context_backgrounds = (
            active_declared_background_histories
            if background_grids is None
            else background_grids[1]
        )

        context: dict[str, Any] = {
            **{
                name: float(value) for name, value in source_parameters.items()
            },
            **{
                name: float(value)
                for name, value in physical_runtime_scalars.items()
            },
            "a": context_grids["a"],
            "a_initial": float(context_grids["a"][0]),
            "z": context_grids["z"],
            "eta": context_grids["eta"],
            "eta_initial": float(context_grids["eta"][0]),
            "H": context_grids["H"],
            "Hconf": context_grids["Hconf"],
            "Hconf_tau": context_grids["Hconf_tau"],
            "tau": context_grids["tau"],
            "tau_dot": context_grids["tau_dot"],
            "visibility": context_grids["visibility"],
            "chi": context_grids["chi"],
            "angular_diameter_distance": numpy.asarray(
                context_grids["angular_diameter_distance"],
                dtype=float,
            ),
            "sound_speed": context_grids["sound_speed"],
            "sound_speed_sq": context_grids["sound_speed_sq"],
            "baryon_sound_speed_sq": context_grids["baryon_sound_speed_sq"],
            "collision_rate": context_grids["collision_rate"],
            "free_streaming": context_grids["free_streaming"],
            "tight_coupling_drag": _compute_tight_coupling_drag(
                collision_rate=context_grids["collision_rate"],
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
        for name, history in context_backgrounds.items():
            context.setdefault(name, numpy.asarray(history, dtype=float))
        context.update(
            _declared_momentum_grid_context(
                perturbation_data,
                model_parameters=source_parameters,
                physical_params=physical_params,
                scale_factor=context_grids["a"],
            )
        )
        if not include_states:
            return _resolve_declared_graph_context_ordered(
                context,
                perturbation_data,
                allow_partial=True,
                eta_grid=None,
                execution_plan=execution_plan,
                value_steps=state_independent_value_steps,
                use_compiled_program=True,
                compiled_value_program=state_independent_context_program,
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

        nonlocal scalar_rhs_calls
        scalar_rhs_calls += 1
        if scalar_rhs_calls % 256 == 0:
            performance_timer.heartbeat(
                "evolution",
                scope=scalar_work_scope,
                details={
                    "rhs_evaluations": scalar_rhs_calls,
                    "mode_k": float(k_value),
                    "remaining_work": "adaptive_unknown",
                },
            )

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
                evaluate_scalar_equations(
                    grouped_equation_program,
                    equation_program,
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
                                    entry, scalar_context
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
                # Retain the scalar dot-product reduction order for each
                # Jacobian probe. A GEMM reduction can change cancellation
                # in conservation rows relative to the scalar GEMV owner.
                collision_drive = (
                    matrix @ target_state
                    if target_state.ndim == 1
                    else numpy.column_stack(
                        [matrix @ column for column in target_state.T]
                    )
                )
                derivative[list(target_indices)] += (
                    float(collision_rate) * collision_drive
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
    rebound_names = {
        str(step.output_name) for step in execution_plan.value_steps
    } | set(execution_plan.relation_target_names)
    grouped_equation_program = compile_grouped_equation_program(
        equation_program_specs,
        tuple(
            (name, slot.index)
            for name, slot in zip(
                state_context_names, runtime_spec.state_slots
            )
            if name not in rebound_names
        ),
    )
    coordinate_names = tuple(
        dict.fromkeys(
            str(slot.wrt) for slot in execution_plan.equation_slot_plans
        )
    )
    state_context_bindings = tuple(
        (name, slot.index)
        for name, slot in zip(state_context_names, runtime_spec.state_slots)
    )
    rhs_context_names = set(grouped_equation_program.context_names)
    rhs_context_names.update(
        name for step in stage_value_steps for name in step.dependencies
    )
    rhs_context_names.update(step.slot_name for step in stage_derivative_steps)
    rhs_state_bindings = tuple(
        (name, index)
        for name, index in state_context_bindings
        if name in rhs_context_names
    )
    can_prune_rhs_context = set(coordinate_names).issubset(
        {*_LEGACY_DECLARED_EVOLUTION_COORDINATES, "a", "z"}
    )
    runtime_envelope["equation_context_required_state_count"] = (
        len(rhs_state_bindings)
        if can_prune_rhs_context
        else len(state_context_bindings)
    )
    runtime_envelope["declared_context_state_count"] = len(
        state_context_bindings
    )
    runtime_envelope["equation_arithmetic_group_count"] = len(
        grouped_equation_program.groups
    )
    continuous_affine_system = is_affine_state_system(
        equation_program_specs,
        state_context_names,
        tuple(
            (step.output_name, step.compiled_expression.expression)
            for step in stage_value_steps
        ),
        tuple(
            (step.output_name, step.slot_name)
            for step in stage_derivative_steps
        ),
        tuple(
            expression.expression
            for runtime in split_collision_runtimes
            for expression in (
                runtime.rate_expression,
                *(entry for row in runtime.matrix for entry in row),
                *(
                    (runtime.damping_coefficient,)
                    if runtime.damping_coefficient is not None
                    else ()
                ),
            )
        ),
    )
    runtime_envelope["continuous_affine_system"] = continuous_affine_system
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
    if any(
        not value for value in state_independent_collision_runtimes.values()
    ):
        # A state-dependent rate cannot have a fixed future split schedule.
        # Integrate its actual declared operator with local error control.
        continuous_collision_solver = True
        requires_continuous_collisions = True
        runtime_envelope["continuous_collision_solver"] = True
    ppf_batch_enabled = bool(
        has_ppf
        and not requires_continuous_collisions
        and "ppf_transition_factor" not in state_dependent_names
    )
    collision_phase_history_cache = {}
    collision_k_names = {"k"}
    for step in state_independent_value_steps:
        if set(step.dependencies) & collision_k_names:
            collision_k_names.add(str(step.output_name))
    collision_depends_on_k = any(
        collision_k_names
        & (
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
        for runtime in split_collision_runtimes
    )

    def _collision_phase_history(mode_k_value, mode_grids, active_mask):
        """Measure operator scales once per compatible physical grid."""

        key = (
            numpy.asarray(mode_grids[0]["eta"]).tobytes(),
            float(mode_k_value) if collision_depends_on_k else None,
            numpy.asarray(active_mask, dtype=bool).tobytes(),
        )
        if key not in collision_phase_history_cache:
            context = _build_array_context(
                {},
                k_value=float(mode_k_value),
                background_grids=mode_grids,
                include_states=False,
            )
            mask = numpy.concatenate((active_mask, active_mask[-1:]))
            collision_phase_history_cache[key] = (
                declared_collision_phase_rates(
                    split_collision_runtimes, context, mask
                )[:-1]
            )
        return collision_phase_history_cache[key]

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

        nonlocal initial_state_cache_hits
        nonlocal initial_state_cache_misses
        initial_eta, initial_background = _scalar_background_context(
            0,
            0.0,
            k_value=float(mode_k_value),
        )
        initial_cache_key = source_history_cache_prefix + (
            "initial_state",
            float(mode_k_value),
            float(initial_eta),
            int(len(runtime_spec.state_slots)),
        )
        if not diagnostic_source_audit and not stage_diagnostic_k_values:
            cached_initial = cache.get_cmb_initial_state(initial_cache_key)
            if cached_initial is not None:
                initial_state_cache_hits += 1
                cached_state, cached_targets, cached_diagnostics = (
                    cached_initial
                )
                if generated_scalar_hierarchy:
                    initial_state_diagnostics_by_k[
                        f"{float(mode_k_value):.12g}"
                    ] = {
                        "k": float(mode_k_value),
                        "eta": float(initial_eta),
                        "state": {
                            str(slot.variable): float(cached_state[slot.index])
                            for slot in runtime_spec.state_slots
                            if int(slot.order) == 0
                        },
                        "constraint_diagnostics": {
                            str(name): dict(values)
                            for name, values in cached_diagnostics.items()
                        },
                    }
                return (
                    numpy.asarray(cached_state, dtype=float).copy(),
                    tuple(cached_targets),
                    {
                        str(name): dict(values)
                        for name, values in cached_diagnostics.items()
                    },
                )
            initial_state_cache_misses += 1
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
            gravity = float(
                initial_state_context.get("einstein_gravity_strength", 0.0)
            )
            # Phi' now uses the 00 equation. Its algebraic energy residual
            # cannot decide this initial projection: form 00 with the
            # independent 0i source so both constraints are closed.
            energy_residual = (
                float(initial_state_context["acoustic_k_sq"])
                * float(initial_state_context["Phi"])
                + 3.0
                * float(initial_state_context["Hconf"])
                * float(initial_state_context["metric_momentum_source_drive"])
                + 1.5
                * gravity
                * float(initial_state_context["total_density_source"])
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
        if not diagnostic_source_audit and not stage_diagnostic_k_values:
            cache.set_cmb_initial_state(
                initial_cache_key,
                (
                    state.copy(),
                    tuple(assigned_targets),
                    {
                        str(name): dict(values)
                        for name, values in initial_diagnostics.items()
                    },
                ),
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
        performance_timer.start_work(
            "initial_data", "scalar_initial_constraints"
        )
        for mode_index, mode_k_value in enumerate(ordered_k_values):
            performance_timer.heartbeat(
                "initial_data",
                scope="scalar_initial_constraints",
                completed=mode_index,
                total=int(ordered_k_values.size),
                details={"mode_k": float(mode_k_value)},
            )
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
        performance_timer.heartbeat(
            "initial_data",
            scope="scalar_initial_constraints",
            completed=int(ordered_k_values.size),
            total=int(ordered_k_values.size),
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

    _LOGGER.info(
        "CCMBS phase completed: model=%s phase=preparation "
        "k=%d source_eta_samples=%d",
        model_label,
        int(k_values.size),
        int(source_grids["eta"].size),
    )
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
        # The reconstructed energy surface is evaluated in float64 after
        # subtracting radiation-era terms that can be many orders of
        # magnitude larger than their residual.  A 2e-7 normalized residual
        # is the round-off envelope of that cancellation across the bundled
        # backgrounds; the published scalar constraint tolerance remains the
        # stricter 1e-3 acceptance bound below.
        if maximum_normalized > 2.0e-7:
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
        evaluation_grids=None,
    ) -> dict[str, numpy.ndarray]:
        """Evaluate declared sources and conservation on the supplied rows."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal scalar_constraint_diagnostics
        if evaluation_grids is not None:
            (
                active_grids,
                active_declared_background_histories,
                active_coordinate_rate_histories,
            ) = evaluation_grids
            evaluation_histories = source_histories
        elif source_grid_indices is None:
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
            apply_reconstruction=source_history_reconstruction_enabled,
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
            # Keep the evolved histories as the production source product,
            # but evaluate the declared Einstein diagnostics on the
            # algebraically projected comparison surface.  This separates a
            # model's differential evolution from the independent closure
            # audit without counting the comparison as a production
            # reconstruction.
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

    def _record_resampled_state_diagnostics(mode_k_value, source_histories):
        """Retain current state diagnostics, including resampled raw work."""

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
                    f"{mode_k_value:.12g}"
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

    def _resample_mode_history(
        k_value,
        native_eta,
        native_histories,
        *,
        collect_diagnostics=True,
        target_eta=None,
    ):
        """Reevaluate sources on an independent physical time grid."""

        nonlocal active_k_value, active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal scalar_base_context_cache
        nonlocal scalar_background_context_cache
        retained_k_value = active_k_value
        mode_k_value = float(k_value)
        target = (
            source_grids["eta"]
            if target_eta is None
            else numpy.asarray(target_eta)
        )
        histories = _interpolate_declared_histories(
            native_eta, native_histories, target
        )
        retained_grids = (
            active_grids,
            active_declared_background_histories,
            active_coordinate_rate_histories,
            scalar_base_context_cache,
            scalar_background_context_cache,
        )
        evaluation_grids = None
        if target_eta is not None:
            evaluation_grids = sample_eta_background_grids(
                target,
                background=background,
                physical_params=physical_params,
                contract_or_params=contract_or_params,
            )
            scalar_base_context_cache = {}
            scalar_background_context_cache = {}
        active_k_value = mode_k_value
        try:
            sources = _evaluate_source_histories(
                float(k_value),
                histories,
                collect_diagnostics=collect_diagnostics,
                required_source_names=required_source_names,
                evaluation_grids=evaluation_grids,
            )
            if collect_diagnostics:
                _record_resampled_state_diagnostics(float(k_value), histories)
        finally:
            if target_eta is not None:
                (
                    active_grids,
                    active_declared_background_histories,
                    active_coordinate_rate_histories,
                    scalar_base_context_cache,
                    scalar_background_context_cache,
                ) = retained_grids
                active_k_value = retained_k_value
        return histories, sources

    def _evolve_declared_mode(
        k_value: float,
        *,
        evolution_sample_count_override: int | None = None,
        history_sink: dict[str, Any] | None = None,
        collect_diagnostics: bool = True,
        count_as_primary_evolution: bool = True,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
        """Integrate one Fourier mode through the declared graph."""

        nonlocal active_grids
        nonlocal active_declared_background_histories
        nonlocal active_coordinate_rate_histories
        nonlocal scalar_base_context_cache
        nonlocal scalar_background_context_cache
        nonlocal active_k_value

        nonlocal scalar_rhs_calls
        nonlocal scalar_work_scope
        scalar_rhs_calls = 0
        scalar_work_scope = f"scalar_mode_{float(k_value):.12g}"
        performance_timer.start_work("evolution", scalar_work_scope)
        if (
            count_as_primary_evolution
            and "evolution_modes_evolved" in runtime_envelope
        ):
            runtime_envelope["evolution_modes_evolved"] = (
                int(runtime_envelope["evolution_modes_evolved"]) + 1
            )
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

            # The declared hierarchy contains expansion-rate terms as well
            # as Fourier streaming.  Resolving only ``k*dt`` is unstable on
            # the first logarithmic background intervals, where
            # ``Hconf*dt`` can be orders of magnitude larger while the
            # collision block is handled exactly.  Include the local
            # conformal-Hubble rate in the explicit RK stability budget.
            _, interval_background = _scalar_background_context(
                step_index,
                0.0,
                k_value=float(k_value),
            )
            stiffness_scale = max(
                abs(float(k_value)),
                abs(float(interval_background.get("Hconf", 0.0))),
                1.0e-12,
            )
            streaming_scale = stiffness_scale
            continuous_block_scale = 0.0
            if exact_collision_phase_cap:
                operator_scale = 0.0
                for operator in split_collision_runtimes:
                    if operator.fast_manifold and tight_coupling_active:
                        continue
                    if (
                        operator.activation_strategy == "tight_coupling"
                        and not tight_coupling_active
                    ):
                        continue
                    rate, matrix, damping = _collision_metadata_for_state(
                        state_vector,
                        runtime=operator,
                        step_index=step_index,
                        blend=0.0,
                        k_value=k_value,
                    )
                    norm = (
                        float(
                            numpy.max(abs(numpy.linalg.eigvals(abs(matrix))))
                        )
                        if matrix.size
                        else 0.0
                    )
                    if damping is not None:
                        norm = max(norm, abs(float(damping)))
                    operator_scale = max(operator_scale, abs(rate) * norm)
                    if not operator.fast_manifold:
                        continuous_block_scale = max(
                            continuous_block_scale, abs(rate) * norm
                        )
                stiffness_scale = float(
                    _split_collision_phase_scale(
                        k_value,
                        interval_background.get("Hconf", 0.0),
                        operator_scale,
                        False,
                    )
                )
            target_stage_scale = _phase_step_for_interval(
                step_index=step_index,
            )
            if continuous_block_scale > streaming_scale:
                # A declared stiff block without a fast manifold needs
                # controlled forcing accuracy, not hundreds of thousands
                # of split microsteps. Solve the complete interval locally;
                # the caller still owns all boundary projections/events.
                left_eta = float(active_grids["eta"][step_index])

                def interval_rhs(eta_value, states):
                    """Retain the actual activation and authoritative block."""

                    return _mode_rhs(
                        states,
                        step_index=step_index,
                        blend=float(
                            numpy.clip((eta_value - left_eta) / dt, 0.0, 1.0)
                        ),
                        k_value=k_value,
                        tight_coupling_active=tight_coupling_active,
                        include_split_collision_outputs=True,
                    )

                local = _integrate_scaled_continuous_history(
                    interval_rhs,
                    state_vector,
                    numpy.asarray((left_eta, left_eta + dt)),
                    initial_time_scale=min(abs(dt), 1.0 / stiffness_scale),
                    rtol=float(numerics.ode_rtol),
                    atol=float(numerics.ode_atol),
                    affine_system=continuous_affine_system,
                    vectorized_rhs=(
                        interval_rhs if continuous_affine_system else None
                    ),
                    # Each projected interval restarts integration. A
                    # one-step L-stable collocation method avoids rebuilding
                    # multistep BDF history at every output boundary.
                    method="Radau",
                )[:, -1]
                runtime_envelope["stiff_collision_intervals"] = (
                    int(runtime_envelope.get("stiff_collision_intervals", 0))
                    + 1
                )
                for operator in split_collision_runtimes:
                    _validate_collision_invariants(
                        local,
                        runtime=operator,
                        step_index=step_index,
                        blend=1.0,
                        k_value=k_value,
                    )
                return _project_declared_fast_collision_state(
                    local,
                    step_index=step_index,
                    blend=1.0,
                    k_value=k_value,
                    tight_coupling_active=tight_coupling_active,
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
            substep_count = 1
            while substep_count < required_substeps:
                substep_count *= 2
            failure_detail = "unspecified"
            while True:
                trial_state = numpy.asarray(state_vector, dtype=float).copy()
                sub_dt = dt / float(substep_count)
                left_eta = float(active_grids["eta"][step_index])
                if sub_dt == 0 or left_eta + sub_dt == left_eta:
                    break
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
            runtime_envelope["evolution_peak_state_cells"] = max(
                int(runtime_envelope.get("evolution_peak_state_cells", 0)),
                int(active_grids["eta"].size) * len(runtime_spec.state_slots),
            )
            state = numpy.asarray(initial_state, dtype=float).copy()
            if not split_collision_runtimes or continuous_collision_solver:
                if has_ppf:
                    runtime_envelope["ppf_scalar_evolution"] = True
                    runtime_envelope["continuous_collision_solver"] = True
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

                _, initial_background = _scalar_background_context(0, 0.0)
                inverse_time_scale = max(
                    abs(float(k_value)),
                    abs(float(initial_background.get("Hconf", 0.0))),
                    1.0 / float(eta_values[-1] - eta_values[0]),
                )
                continuous_history = _integrate_scaled_continuous_history(
                    _continuous_rhs,
                    state,
                    eta_values,
                    initial_time_scale=1.0 / inverse_time_scale,
                    rtol=float(numerics.ode_rtol),
                    atol=float(numerics.ode_atol),
                    affine_system=continuous_affine_system,
                    vectorized_rhs=(
                        _continuous_rhs if continuous_affine_system else None
                    ),
                )
                histories = {
                    slot.variable: numpy.asarray(
                        continuous_history[slot.index],
                        dtype=float,
                    )
                    for slot in runtime_spec.state_slots
                    if slot.order == 0
                }
                final_state = continuous_history[:, -1].copy()
                return histories, final_state
            tight_coupling_active = _tight_coupling_is_active(
                active=False,
                collision_rate=float(active_grids["collision_rate"][0]),
                k_value=float(k_value),
                expansion_rate=float(active_grids["Hconf"][0]),
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
                    expansion_rate=float(
                        active_grids["Hconf"][step_index + 1]
                    ),
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
            # Retain derived quantities on the evolved metric itself. This
            # diagnostic product must not inherit the optional algebraic
            # reconstruction used by a separate source comparison.
            native_context = _build_array_context(
                histories, k_value=float(k_value)
            )
            history_sink["native_context"] = {
                name: numpy.asarray(values, dtype=float).copy()
                for name, values in native_context.items()
                if isinstance(values, numpy.ndarray)
                and values.shape == active_grids["eta"].shape
            }
            history_sink["native_sources"] = _evaluate_declared_sources(
                native_context, k_value=float(k_value)
            )
        if active_grids["eta"].shape != source_grids[
            "eta"
        ].shape or not numpy.array_equal(
            active_grids["eta"],
            source_grids["eta"],
        ):
            source_histories = _interpolate_declared_histories(
                active_grids["eta"], histories, source_grids["eta"]
            )
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
            history_sink["source_arrays"] = {
                name: numpy.asarray(values, dtype=float).copy()
                for name, values in source_arrays.items()
            }
        if collect_diagnostics:
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
                        f"{float(k_value):.12g}"
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
        return source_histories, source_arrays

    def _record_stage_diagnostic_histories(
        mode_index: int,
        mode_k_value: float,
        *,
        evolution_eta: numpy.ndarray,
        evolution_histories: Mapping[str, numpy.ndarray],
        source_eta: numpy.ndarray,
        source_histories: Mapping[str, numpy.ndarray],
        source_arrays: Mapping[str, numpy.ndarray],
    ) -> None:
        """Retain native and source-grid histories for selected modes."""

        if float(mode_k_value) in history_capture_k_values:
            cache.set_cmb_evolution_history(
                source_history_cache_prefix + (float(mode_k_value),),
                evolution_eta,
                evolution_histories,
            )
        if not stage_diagnostic_k_values:
            return
        for requested_k_value in stage_diagnostic_k_values:
            selected_index = int(
                numpy.argmin(numpy.abs(k_values - requested_k_value))
            )
            if selected_index != int(mode_index):
                continue
            evolution_payload = {
                name: numpy.asarray(values, dtype=float).copy()
                for name, values in evolution_histories.items()
                if name in stage_diagnostic_fields
            }
            source_payload = {
                name: numpy.asarray(values, dtype=float).copy()
                for name, values in source_histories.items()
                if name in stage_diagnostic_fields
            }
            source_payload.update(
                {
                    name: numpy.asarray(values, dtype=float).copy()
                    for name, values in source_arrays.items()
                    if name in stage_diagnostic_fields
                }
            )
            stage_diagnostic_histories_by_k[f"{requested_k_value:.12g}"] = {
                "requested_k": float(requested_k_value),
                "selected_k": float(mode_k_value),
                "evolution_eta": numpy.asarray(
                    evolution_eta,
                    dtype=float,
                ).copy(),
                "evolution_histories": evolution_payload,
                "source_eta": numpy.asarray(source_eta, dtype=float).copy(),
                "source_histories": source_payload,
            }

    @with_native_history_storage
    def _evolve_declared_modes_batched(
        mode_k_values: numpy.ndarray,
        *,
        evolution_sample_count_override: int | None = None,
        history_sinks: dict[float, dict[str, Any]] | None = None,
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
        nonlocal scheduled_background_contexts

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
            continuous_collision_solver=(
                continuous_collision_solver and not ppf_batch_enabled
            ),
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

        recombination_window = max(
            float(background.eta_rec)
            + 2.0 * float(background.sound_horizon_mpc),
            float(background.eta_rec) + 64.0,
        )
        schedule_cache_prefix = (
            str(hierarchy_schedule_evidence["signature"]),
            float(generated_final_phase_step),
            float(numerics.evolution_phase_step),
            float(recombination_window),
            bool(exact_collision_phase_cap),
            float(numerics.tight_coupling_ratio),
            float(numerics.tight_coupling_exit_ratio),
        )

        def _tight_interval_mask(
            mode_k_value: float,
            collision_values: numpy.ndarray,
            expansion_values: numpy.ndarray,
        ) -> numpy.ndarray:
            """Retain the declared hysteresis at each interval's left node."""

            active = False
            mask = numpy.empty(collision_values.size - 1, dtype=bool)
            for index in range(mask.size):
                active = _tight_coupling_is_active(
                    active=active,
                    collision_rate=float(collision_values[index]),
                    k_value=mode_k_value,
                    expansion_rate=float(expansion_values[index]),
                    tight_coupling_ratio=float(numerics.tight_coupling_ratio),
                    exit_ratio=float(numerics.tight_coupling_exit_ratio),
                )
                mask[index] = active
            return mask

        def _mode_schedule_values(
            mode_k_value: float,
            eta_values: numpy.ndarray,
            hconf_values: numpy.ndarray,
            collision_values: numpy.ndarray,
            mode_grids,
        ) -> numpy.ndarray:
            """Return the phase schedule required by one mode."""

            operator_phase = (
                _collision_phase_history(
                    mode_k_value,
                    mode_grids,
                    _tight_interval_mask(
                        mode_k_value, collision_values, hconf_values
                    ),
                )
                if exact_collision_phase_cap
                else numpy.zeros(hconf_values.shape)
            )

            grid_hasher = hashlib.sha256(
                numpy.asarray(eta_values, dtype=numpy.float64).tobytes()
            )
            grid_hasher.update(
                numpy.asarray(hconf_values, dtype=numpy.float64).tobytes()
            )
            grid_hasher.update(
                numpy.asarray(collision_values, dtype=numpy.float64).tobytes()
            )
            grid_hasher.update(operator_phase.tobytes())
            eta_signature = grid_hasher.hexdigest()
            schedule_cache_key = schedule_cache_prefix + (
                eta_signature,
                float(mode_k_value),
            )
            cached_schedule = cache.get_cmb_hierarchy_schedule(
                schedule_cache_key
            )
            if cached_schedule is not None:
                return numpy.asarray(cached_schedule, dtype=int).copy()

            dt_values = numpy.diff(eta_values)
            if exact_collision_phase_cap:
                phase_steps = numpy.full(
                    dt_values.shape,
                    float(generated_final_phase_step),
                    dtype=float,
                )
            elif generated_final_phase_step >= float(
                numerics.evolution_phase_step
            ):
                phase_steps = numpy.full(
                    dt_values.shape,
                    float(numerics.evolution_phase_step),
                    dtype=float,
                )
            else:
                eta_mid = 0.5 * (eta_values[:-1] + eta_values[1:])
                phase_steps = numpy.where(
                    eta_mid <= recombination_window,
                    float(generated_final_phase_step),
                    float(numerics.evolution_phase_step),
                )
            phase_scale = numpy.maximum(
                abs(float(mode_k_value)), numpy.abs(hconf_values)
            )
            if exact_collision_phase_cap:
                phase_scale = _split_collision_phase_scale(
                    mode_k_value,
                    hconf_values,
                    operator_phase,
                    False,
                )
            required = numpy.maximum(
                1,
                numpy.ceil(
                    numpy.abs(dt_values)
                    * phase_scale
                    / numpy.maximum(phase_steps, 1.0e-12)
                ).astype(int),
            )
            schedule = numpy.ones(required.shape, dtype=int)
            for index, requested in enumerate(required):
                substep_count = 1
                while substep_count < int(requested):
                    substep_count *= 2
                schedule[index] = substep_count
            cache.set_cmb_hierarchy_schedule(
                schedule_cache_key,
                schedule.copy(),
            )
            return schedule

        # Modes with adjacent k values have schedules that differ only at
        # isolated phase thresholds.  Requiring byte-identical schedules
        # would turn almost every production mode into a scalar evolution.
        # Bound extra work to one refinement level for every request size.
        # Tiny diagnostic requests need the same accuracy contract as large
        # ones; byte-identical schedules needlessly serialize their modes.
        scheduled_modes: list[dict[str, Any]] = []
        batched_collision_eigendecomposition_cache: dict[
            tuple[tuple[int, ...], bytes],
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None,
        ] = {}
        schedule_group_width = 256
        for mode_index, mode_k_value in enumerate(k_values_batch):
            mode_grids = _mode_grids_for_k(
                float(mode_k_value),
                evolution_sample_count_override=(
                    evolution_sample_count_override
                ),
            )
            eta_values = numpy.asarray(mode_grids[0]["eta"], dtype=float)
            schedule_values = _mode_schedule_values(
                float(mode_k_value),
                eta_values,
                numpy.asarray(mode_grids[0]["Hconf"][:-1], dtype=float),
                numpy.asarray(mode_grids[0]["collision_rate"], dtype=float),
                mode_grids,
            )
            scheduled_modes.append(
                {
                    "index": mode_index,
                    "k": mode_k_value,
                    "grids": mode_grids,
                    "eta": eta_values,
                    "schedule": schedule_values,
                }
            )
        grouped_modes = group_mode_schedules(
            scheduled_modes,
            maximum_modes=schedule_group_width,
        )

        results: dict[int, dict[str, numpy.ndarray]] = {}
        for group in grouped_modes:
            # Capability was checked for the whole call. A singleton
            # partition or independent reference keeps that same solver.
            (
                active_grids,
                active_declared_background_histories,
                active_coordinate_rate_histories,
            ) = group["grids"]
            active_k_value = float(group["k_values"][0])
            scheduled_background_contexts = None
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
            momentum_context_by_stage = None
            batched_row_program: Any | None = None
            batched_row_vector_names: tuple[str, ...] | None = None

            def _batch_base_context(
                step_index: int,
                blend: float,
                *,
                scalar: bool = False,
            ) -> dict[str, Any]:
                """Return the vectorized state-independent graph context."""

                cache_key = (int(step_index), float(blend), scalar)
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
                context["k"] = (
                    float(local_k_values[0]) if scalar else local_k_values
                )
                context["seed"] = (
                    float(batch_seed_values[0])
                    if scalar
                    else batch_seed_values
                )
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
                    momentum_context = (
                        momentum_context_by_stage.get(cache_key[:2])
                        if momentum_context_by_stage is not None
                        else None
                    )
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
                if scalar:
                    coupling_cap = float(coupling_cap[0])
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
                equations_only: bool = False,
            ) -> dict[str, Any]:
                """Bind required states into a vector-valued graph context."""

                scalar_rhs = mode_count == 1 and equations_only
                context = dict(
                    _batch_base_context(step_index, blend, scalar=scalar_rhs)
                )
                state_columns = numpy.asarray(state_rows, dtype=float).T
                bindings = state_context_bindings
                if (
                    equations_only
                    and not include_diagnostics
                    and can_prune_rhs_context
                ):
                    bindings = rhs_state_bindings
                for name, index in bindings:
                    context[name] = (
                        float(state_rows[0, index])
                        if scalar_rhs
                        else state_columns[index]
                    )
                suppressed_outputs = (
                    batched_suppressed_collision_outputs
                    if suppress_split_collision
                    else {}
                )
                if scalar_rhs:
                    suppressed_outputs = {
                        name: float(values[0])
                        for name, values in suppressed_outputs.items()
                    }
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
                    equations_only=True,
                )
                # Coordinate rates are background quantities and therefore
                # common to the batch, but they are not all unity.  In
                # particular, equations declared with ``wrt: a`` or
                # ``wrt: z`` must be converted from their declared coordinate
                # to conformal time before the vectorized equation program
                # runs.  The former hard-coded unity map silently changed
                # those theories' dynamics.
                batch_coordinate_rates = {
                    wrt: _resolve_coordinate_rate(
                        wrt_name=wrt,
                        scalar_context=context,
                        step_index=step_index,
                        blend=blend,
                        k_value=float(local_k_values[0]),
                    )
                    for wrt in coordinate_names
                }
                state_columns = numpy.asarray(state_rows, dtype=float).T
                derivative_columns = numpy.zeros_like(
                    state_columns,
                    dtype=float,
                )
                equation_executor = grouped_equation_program
                if mode_count == 1:
                    # The row executor avoids allocating hierarchy-shaped
                    # NumPy operands for a single mode. Bind only direct
                    # state names that the pruned graph context omitted;
                    # preserve graph-rebound values already in that context.
                    for name, index in state_context_bindings:
                        if (
                            name in equation_direct_names
                            and name not in context
                        ):
                            context[name] = float(state_rows[0, index])
                    vector_names = tuple(
                        name
                        for name in sorted(equation_direct_names)
                        if getattr(context[name], "shape", ()) == (mode_count,)
                        or (
                            isinstance(context[name], (tuple, list))
                            and numpy.asarray(context[name]).shape
                            == (mode_count,)
                        )
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
                    failed_row, failed_slot = int(bad[0]), int(bad[1])
                    raise NonFiniteEvolutionError(
                        "Declared batched CMB evolution produced a "
                        "non-finite derivative at "
                        f"mode_index={int(bad[0])}, "
                        f"state_index={int(bad[1])}",
                        context={
                            "k": float(local_k_values[failed_row]),
                            "eta": float(
                                active_grids["eta"][step_index]
                                + blend
                                * (
                                    active_grids["eta"][step_index + 1]
                                    - active_grids["eta"][step_index]
                                )
                            ),
                            "step_index": int(step_index),
                            "variable": str(
                                runtime_spec.state_slots[failed_slot].variable
                            ),
                            "state_snapshot": {
                                str(slot.variable): (
                                    float(state_rows[failed_row, slot.index])
                                    if numpy.isfinite(
                                        state_rows[failed_row, slot.index]
                                    )
                                    else str(
                                        state_rows[failed_row, slot.index]
                                    )
                                )
                                for slot in runtime_spec.state_slots
                            },
                        },
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
                            eigendecomposition_cache=(
                                batched_collision_eigendecomposition_cache
                            ),
                            kernel_metrics=collision_kernel_metrics,
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
                        _apply_batched_collision_damping(
                            relaxed, nonzero_rows, damping_indices, damping
                        )
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

                active_array = numpy.asarray(active, dtype=bool)
                # Match the scalar interval boundary, including its
                # declaration-defined fast projection. An unchanged active
                # mask does not make a state-dependent projection redundant.
                projected = _project_fast_collision_state(
                    state_rows,
                    step_index=step_index,
                    blend=blend,
                    active=active_array,
                )
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
                    expansion_rate=float(active_grids["Hconf"][0]),
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
                        expansion_rate=float(
                            active_grids["Hconf"][step_index + 1]
                        ),
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
            hconf_interval_values = numpy.asarray(
                active_grids["Hconf"],
                dtype=float,
            )[:-1]
            phase_scale = numpy.maximum(
                numpy.abs(local_k_values)[:, numpy.newaxis],
                numpy.abs(hconf_interval_values[numpy.newaxis, :]),
            )
            if exact_collision_phase_cap:
                phase_scale = _split_collision_phase_scale(
                    local_k_values[:, numpy.newaxis],
                    hconf_interval_values[numpy.newaxis, :],
                    numpy.stack(
                        [
                            _collision_phase_history(
                                wave, group["grids"], mask
                            )
                            for wave, mask in zip(
                                local_k_values, active_intervals
                            )
                        ]
                    ),
                    False,
                )
            required_substeps = numpy.maximum(
                1,
                numpy.ceil(
                    numpy.abs(dt_values)[numpy.newaxis, :]
                    * phase_scale
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
            if momentum_runtimes or generated_scalar_hierarchy:
                stage_keys: list[tuple[int, float]] = [(0, 0.0)]
                for step_index in range(interval_count):
                    substep_count = int(batch_substeps[step_index])
                    for substep_index in range(substep_count):
                        # The left endpoint is the preceding stage's right
                        # endpoint (or the already saved record coordinate).
                        # Dyadic subdivisions make these keys exact, so a
                        # second full-size deduplication set is unnecessary.
                        for blend in (
                            (substep_index + 0.5) / float(substep_count),
                            (substep_index + 1.0) / float(substep_count),
                        ):
                            stage_keys.append((int(step_index), float(blend)))
                    stage_keys.append((int(step_index + 1), 0.0))
                if generated_scalar_hierarchy:
                    # These coordinates are already fixed by the actual RK
                    # schedule. Evaluate the same physical graph in bounded
                    # vector blocks, never on an interpolated substitute.
                    stage_wave = float(local_k_values[0])
                    performance_timer.start_work(
                        "background", "scheduled_rk_background"
                    )
                    with performance_timer.phase("background"):
                        scheduled_columns = {}
                        for start in range(0, len(stage_keys), 4096):
                            selected_keys = stage_keys[start : start + 4096]
                            epochs, columns = sample_declared_stage_contexts(
                                background=background,
                                grids=active_grids,
                                declared_histories=(
                                    active_declared_background_histories
                                ),
                                contract=contract_or_params,
                                physical=physical_params,
                                numerics=numerics,
                                indices=[key[0] for key in selected_keys],
                                blends=[key[1] for key in selected_keys],
                                wave=stage_wave,
                            )
                            completed = start + len(selected_keys)
                            block = {"__scheduled_epoch": epochs, **columns}
                            if not scheduled_columns:
                                storage = allocate_native_history(
                                    (len(stage_keys), len(block))
                                )
                                scheduled_columns = {
                                    name: storage[:, index]
                                    for index, name in enumerate(block)
                                }
                            for name, values in block.items():
                                column = scheduled_columns[name]
                                column[start:completed] = values
                            performance_timer.heartbeat(
                                "background",
                                scope="scheduled_rk_background",
                                completed=completed,
                                total=len(stage_keys),
                                weighted_completed=float(completed),
                                weighted_total=float(len(stage_keys)),
                            )
                        scheduled_background_contexts = (
                            active_grids,
                            stage_wave,
                            StageContextColumns(stage_keys, scheduled_columns),
                        )
            if momentum_runtimes:
                stage_scales = (
                    scheduled_columns["a"]
                    if generated_scalar_hierarchy
                    else numpy.asarray(
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
                )
                batch_momentum_context = _declared_momentum_grid_context(
                    perturbation_data,
                    model_parameters=source_parameters,
                    physical_params=physical_params,
                    scale_factor=stage_scales,
                )
                stage_count = len(stage_keys)
                static_context = {}
                stage_columns = []
                for name, value in batch_momentum_context.items():
                    array_value = numpy.asarray(value)
                    if (
                        array_value.ndim > 0
                        and array_value.shape[0] == stage_count
                    ):
                        stage_columns.append((name, array_value))
                    else:
                        static_context[name] = value
                momentum_context_by_stage = StageContextColumns(
                    stage_keys, stage_columns, static=static_context
                )
            schedule_matches = numpy.all(
                scalar_substeps == batch_substeps[numpy.newaxis, :],
                axis=1,
            )
            runtime_envelope["schedule_group_count"] = (
                int(runtime_envelope.get("schedule_group_count", 0)) + 1
            )
            runtime_envelope["schedule_max_group_size"] = max(
                int(runtime_envelope.get("schedule_max_group_size", 0)),
                int(mode_count),
            )
            runtime_envelope["schedule_scalar_substeps"] = int(
                runtime_envelope.get("schedule_scalar_substeps", 0)
            ) + int(numpy.sum(scalar_substeps))
            runtime_envelope["schedule_batched_substeps"] = int(
                runtime_envelope.get("schedule_batched_substeps", 0)
            ) + int(mode_count * numpy.sum(batch_substeps))
            runtime_envelope["schedule_total_overintegration_ratio"] = (
                runtime_envelope["schedule_batched_substeps"]
                / max(runtime_envelope["schedule_scalar_substeps"], 1)
            )
            runtime_envelope["schedule_max_overintegration_ratio"] = max(
                float(
                    runtime_envelope.get(
                        "schedule_max_overintegration_ratio", 1.0
                    )
                ),
                float(
                    numpy.max(
                        batch_substeps[numpy.newaxis, :]
                        / numpy.maximum(scalar_substeps, 1)
                    )
                ),
            )
            runtime_envelope["schedule_forced_overintegration"] = int(
                runtime_envelope.get("schedule_forced_overintegration", 0)
            ) + int(
                numpy.count_nonzero(
                    scalar_substeps != batch_substeps[numpy.newaxis, :]
                )
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
            scope = "rk_group_" + str(runtime_envelope["schedule_group_count"])
            performance_timer.start_work("evolution", scope)

            def _report_batch_work(
                completed_work: int,
                scheduled_work: int,
                *,
                interval: int,
                intervals: int,
            ) -> None:
                """Estimate only this known schedule, weighted by mode rows."""

                performance_timer.heartbeat(
                    "evolution",
                    scope=scope,
                    completed=completed_work,
                    total=scheduled_work,
                    weighted_completed=float(completed_work),
                    weighted_total=float(scheduled_work),
                    details={
                        "rk_interval": f"{interval}/{intervals}",
                        "schedule_group_modes": mode_count,
                    },
                )
                # Stage coordinates advance monotonically. Past background
                # and collision contexts are never needed by later RK work;
                # retaining them made memory grow with every microstep.
                # Clearing memoized intermediates changes no states, grids,
                # coefficients, or independent acceptance calculations.
                working_caches = (
                    base_context_cache,
                    scalar_background_context_cache,
                    scalar_base_context_cache,
                    momentum_grid_context_cache,
                    collision_metadata_cache,
                    batched_collision_eigendecomposition_cache,
                )
                runtime_envelope["stage_cache_peak_entries"] = max(
                    int(runtime_envelope.get("stage_cache_peak_entries", 0)),
                    sum(len(store) for store in working_caches),
                )
                for store in working_caches:
                    store.clear()

            relaxation_slot = None
            relaxation_rate = None
            if ppf_batch_enabled:
                runtime_envelope["ppf_batched_relaxation"] = True
                if not runtime_envelope.get("ppf_scalar_evolution", False):
                    runtime_envelope["continuous_collision_solver"] = False
                ppf_variables = [
                    name
                    for name, metadata in perturbation_data.variables.items()
                    if metadata.kind == "ppf_metric_correction"
                ]
                if len(ppf_variables) != 1:
                    raise ValueError(
                        "PPF requires exactly one relaxation state"
                    )
                relaxation_slot = runtime_spec.state_index_by_key[
                    (ppf_variables[0], "tau", 0)
                ]

                def _ppf_relaxation_rate(
                    state_rows, *, step_index, blend, active
                ):
                    """Resolve the graph's physical PPF damping rate."""
                    del active
                    context = _batched_state_context(
                        state_rows,
                        step_index=step_index,
                        blend=blend,
                        equations_only=True,
                    )
                    return context["Hconf"] * context["ppf_transition_factor"]

                relaxation_rate = _ppf_relaxation_rate

            histories, _, batch_stats = _integrate_batched_rk4(
                states,
                eta_values,
                required_substeps=required_substeps,
                active_intervals=active_intervals,
                rhs=_batch_rhs,
                pre_step=_batch_pre_step,
                post_step=_batch_post_step,
                record_step=_batch_record_step,
                progress=_report_batch_work,
                relaxation_slot=relaxation_slot,
                relaxation_rate=relaxation_rate,
            )
            scheduled_background_contexts = None
            momentum_context_by_stage = None
            storage_key = (
                "evolution_peak_mapped_history_cells"
                if isinstance(histories, numpy.memmap)
                else "evolution_peak_state_cells"
            )
            runtime_envelope[storage_key] = max(
                int(runtime_envelope.get(storage_key, 0)), int(histories.size)
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
            # A shared schedule can refine a row beyond its scalar budget.
            # Keep that resolved history rather than duplicating evolution
            # and replacing it with a different scalar trajectory.
            history_slots = tuple(
                slot for slot in runtime_spec.state_slots if slot.order == 0
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
                    slot.variable: numpy.asarray(
                        histories[row_index, :, slot.index],
                        dtype=float,
                    )
                    for slot in history_slots
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
                    source_histories = _interpolate_declared_histories(
                        eta_values, source_histories, source_eta
                    )
                if evolution_sample_count_override is None:
                    _record_resampled_state_diagnostics(
                        mode_k_value, source_histories
                    )
                if not common_schedule_allowed and not numpy.array_equal(
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
                    scalar_history_sink: dict[str, Any] = {}
                    _, scalar_source_arrays = _evolve_declared_mode(
                        mode_k_value,
                        history_sink=scalar_history_sink,
                        # The corrective scalar evolution is the accepted
                        # source history for this mode.  Always collect the
                        # scalar constraint diagnostics so a generated
                        # transfer cache cannot hide an incomplete audit;
                        # the expensive raw residual bundle remains gated by
                        # ``diagnostic_source_audit`` inside the evaluator.
                        collect_diagnostics=True,
                    )
                    _record_stage_diagnostic_histories(
                        int(mode_index),
                        float(mode_k_value),
                        evolution_eta=scalar_history_sink["evolution_eta"],
                        evolution_histories=scalar_history_sink[
                            "evolution_histories"
                        ],
                        source_eta=scalar_history_sink["source_eta"],
                        source_histories=scalar_history_sink[
                            "source_histories"
                        ],
                        source_arrays=scalar_source_arrays,
                    )
                    results[int(mode_index)] = scalar_source_arrays
                    continue
                evaluated_source_arrays = _evaluate_source_histories(
                    mode_k_value,
                    source_histories,
                    required_source_names=required_source_names,
                    collect_diagnostics=evolution_sample_count_override
                    is None,
                )
                native_histories = {
                    slot.variable: numpy.asarray(
                        histories[row_index, :, slot.index],
                        dtype=float,
                    )
                    for slot in history_slots
                }
                if history_sinks is not None:
                    history_sinks[mode_k_value] = {
                        "evolution_eta": eta_values.copy(),
                        "evolution_histories": {
                            name: values.copy()
                            for name, values in native_histories.items()
                        },
                        "source_eta": source_eta.copy(),
                        "source_arrays": {
                            name: numpy.array(values, copy=True)
                            for name, values in evaluated_source_arrays.items()
                        },
                    }
                if evolution_sample_count_override is None:
                    _record_stage_diagnostic_histories(
                        int(mode_index),
                        mode_k_value,
                        evolution_eta=eta_values,
                        evolution_histories=native_histories,
                        source_eta=source_eta,
                        source_histories=source_histories,
                        source_arrays=evaluated_source_arrays,
                    )
                results[int(mode_index)] = {
                    name: numpy.array(values, copy=True)
                    for name, values in evaluated_source_arrays.items()
                }

        scheduled_background_contexts = None
        scalar_base_context_cache = {}
        scalar_background_context_cache = {}
        active_grids = dict(source_grids)
        active_declared_background_histories = (
            source_declared_background_histories
        )
        active_coordinate_rate_histories = source_coordinate_rate_histories
        return results

    def _snapshot() -> dict[str, Any]:
        """Return runtime state whose scalar updates outlive callbacks."""

        return {
            "active_coordinate_rate_histories": (
                active_coordinate_rate_histories
            ),
            "active_declared_background_histories": (
                active_declared_background_histories
            ),
            "active_grids": active_grids,
            "active_k_value": active_k_value,
            "initial_state_cache_hits": initial_state_cache_hits,
            "initial_state_cache_misses": initial_state_cache_misses,
            "scalar_background_context_cache": scalar_background_context_cache,
            "scalar_base_context_cache": scalar_base_context_cache,
            "scalar_constraint_diagnostic_projection_count": (
                scalar_constraint_diagnostic_projection_count
            ),
            "scalar_constraint_diagnostics": scalar_constraint_diagnostics,
            "scalar_constraint_projection_count": (
                scalar_constraint_projection_count
            ),
            "scalar_constraint_projection_max_relative_correction": (
                scalar_constraint_projection_max_relative_correction
            ),
            "shared_generated_mode_grids": shared_generated_mode_grids,
            "source_history_mode_count": source_history_mode_count,
        }

    return DeclaredModeEvolutionRuntime(
        evolve_declared_mode=_evolve_declared_mode,
        evolve_declared_modes_batched=_evolve_declared_modes_batched,
        record_source_history_diagnostics=(_record_source_history_diagnostics),
        record_stage_diagnostic_histories=(_record_stage_diagnostic_histories),
        scalar_initial_constraint_preflight=(
            scalar_initial_constraint_preflight
        ),
        snapshot=_snapshot,
        mode_grids_for_k=_mode_grids_for_k,
        history_capture_k_values=history_capture_k_values,
        resample_mode_history=_resample_mode_history,
    )
