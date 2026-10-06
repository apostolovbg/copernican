r"""Declared transfer projection and spectrum integration helpers."""

from __future__ import annotations

import hashlib
import logging
import math
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy
from scipy.interpolate import CubicSpline
from scipy.special import gammaln, spherical_jn

from ....cmb_output import canonical_cmb_spectrum_name
from ....cmb_projection_contract import (
    get_declared_projection_kernel_spec,
    resolve_declared_source_kernel,
)
from ....model_adapter import FrozenMapping, _freeze_for_cache
from ..errors import ConvergenceError, classify_exception, failure_context
from . import cache
from .adaptive import (
    AdaptiveNonConvergenceError,
    ConvergenceEstimate,
    estimate_convergence,
    estimate_history_convergence,
    informative_k_indices,
    nested_coarse_indices,
    nested_phase_aware_k_grid,
    phase_aware_eta_grid,
    phase_aware_k_grid_requirements,
    phase_aware_k_grid_status,
    physical_history_anchors,
    require_convergence,
    resolve_adaptive_controls,
    resolve_los_quadrature_controls,
)
from .background import (
    CustomCMBSpectrumData,
    CustomCMBTransferData,
    _accuracy_control_value,
    _build_custom_cmb_background,
    _coerce_numeric_scalar,
    _compute_spherical_bessel_batch,
    _compute_spherical_bessel_mode_batch,
    _contract_structural_cache_view,
    _custom_cmb_spectrum_cache_key,
    _custom_cmb_transfer_cache_key,
    _DeclaredProjectionKernelBatch,
    _get_cached_custom_cmb_spectrum_data,
    _get_cached_declared_projection_kernel_batch,
    _physical_runtime_scalars,
    _resolve_custom_cmb_numerics,
    _resolve_custom_cmb_physical_parameters,
    _resolve_declared_accuracy_controls,
)
from .collisions import _compile_split_collision_operator_runtimes
from .convergence import (
    evaluate_hierarchy_truncation_bounds,
    evaluate_momentum_grid_refinement_bounds,
    evaluate_spectrum_refinement,
    resolve_declared_numerical_envelope,
    resolve_production_scalar_convergence,
)
from .evidence import (
    _build_source_history_bundle_digest,
    _projection_array_digest,
    _runtime_telemetry_context,
)
from .evolution import (
    _NEUTRINO_TEMPERATURE_EV_PER_K,
    _compile_declared_perturbation_contract,
    _compile_equation_program,
    _resolve_declared_momentum_grid_runtimes,
    describe_declared_execution_schedule,
    prepare_runtime_assets,
)
from .line_of_sight import (
    build_phase_aware_projection_ladder,
    sample_eta_background_grids,
)
from .mode_evolution import (
    DeclaredModeEvolutionInputs,
    build_declared_mode_evolution,
)
from .numerical_controls import (
    _densify_eta_grid,
    _enforce_runtime_envelope,
    _limit_eta_grid,
    _refine_eta_grid,
    _resolve_evolution_chunk_size,
)
from .performance import PhaseTimer
from .postprocessing import build_postprocessing_evidence
from .source_graph import compile_declared_source_graph
from .spectrum_projection import (
    _bind_declared_source_histories,
    _build_projection_k_grid,
    _configured_reference_ells,
    _declared_graph_projection,
    _integrate_declared_spectra,
    _integrate_power_spectrum,
    _primordial_power_grid_for_observable,
    _projection_ell_limit_for_mode,
    _slice_projection_kernel_batch,
    _trapezoid_weights,
)

_LOGGER = logging.getLogger(__name__)


_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}
# Keep a genuinely pre-visibility prefix for every declared scalar mode.  The
# prefix is part of the physical initial-condition evolution, not a plotting
# convenience, so it must remain long enough to leave a measurable
# super-horizon history in diagnostics.
_SCALAR_SUPERHORIZON_PREFIX_KETA = 5.5e-3
_SCALAR_INITIAL_SOLVE_NUMERICAL_TOLERANCE = 1.0e-9
_BESSEL_WORK_CELL_BUDGET = 8_000_000
_BESSEL_MAX_MODE_BATCH = 16


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
    raw_stage_diagnostic = contract_or_params.get("_stage_diagnostic")
    stage_diagnostic_requested = bool(
        isinstance(raw_stage_diagnostic, Mapping)
        and raw_stage_diagnostic.get("k_values") is not None
        and len(raw_stage_diagnostic.get("k_values")) > 0
    )
    diagnostic_request = bool(
        diagnostic_source_audit or stage_diagnostic_requested
    )
    cache_key = _custom_cmb_spectrum_cache_key(
        contract_or_params,
        ells,
        background_provider,
        requested_spectra=requested_spectrum_names,
    )
    # Diagnostic requests deliberately bypass result storage so the report
    # owns the base/refined comparison.  Publish their semantic request
    # identity nevertheless, allowing retained evidence to identify both
    # products without pretending that either product was cached.  Normal
    # requests must publish only through cache get/set so a changed request
    # cannot overwrite the previous identity before warm-state classification.
    if diagnostic_request:
        cache.remember_cmb_request_identity(cache_key)
    cached_spectrum = (
        None if diagnostic_request else cache.get_cmb_spectrum(cache_key)
    )
    if cached_spectrum is not None:
        performance_timer.mark_cache_state("exact_cache_hit")
        return _get_cached_custom_cmb_spectrum_data(cache_key)

    cache_stats_before = cache.cmb_cache_stats()
    graph_cache_before = cache_stats_before["declared_graph_execution_plan"]
    runtime_asset_cache_before = cache_stats_before["runtime_assets"]
    hierarchy_schedule_cache_before = cache_stats_before["hierarchy_schedule"]
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
        hierarchy_schedule_evidence = describe_declared_execution_schedule(
            perturbation_data,
            execution_plan,
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
    model_label = str(contract_or_params.get("model_name", "declared"))
    _LOGGER.info(
        "CCMBS phase started: model=%s phase=background",
        model_label,
    )
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
    _LOGGER.info(
        "CCMBS phase completed: model=%s phase=background eta=%d",
        model_label,
        int(background.eta_grid.size),
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
    eta_los_grid = numpy.unique(
        numpy.concatenate(
            (
                numpy.asarray((eta_start,), dtype=float),
                eta_los_grid,
                numpy.asarray((float(background.eta0),), dtype=float),
            )
        )
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
    if not generated_scalar_hierarchy:
        # Explicit transfer graphs use the declared LOS resolution as their
        # base surface.  The background builder retains a dense physical
        # grid for interpolation, but feeding that grid directly here would
        # make a bounded request accidentally finer than its adaptive
        # refinement target and would prevent the refinement from being
        # observable.  Preserve endpoints and the nonuniform physical
        # spacing while coarsening only to the requested base count.
        eta_los_grid = _limit_eta_grid(
            eta_los_grid,
            maximum_samples=minimum_eta_samples,
        )
    declared_accuracy_controls = dict(
        _resolve_declared_accuracy_controls(contract_or_params)
    )
    # Generated scalar histories are the physical evolution product.  The
    # algebraic Einstein reconstruction is retained as a diagnostic-only
    # comparison; applying it to production source histories would replace
    # the evolved metric with a density-constraint solve and change the
    # declared differential system.
    if generated_scalar_hierarchy:
        declared_accuracy_controls["source_history_reconstruction"] = False
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
    declared_accuracy_tier = str(
        declared_accuracy_controls.get("accuracy_tier", "")
    )
    explicit_diagnostic_request = (
        bool(
            contract_or_params.get("numerical")
            or contract_or_params.get("perturbations", {}).get("numerics")
        )
        and declared_accuracy_tier != "final"
    )
    diagnostic_matrix_fast_path = bool(
        contract_or_params.get("_diagnostic_matrix_fast_path", False)
        or explicit_diagnostic_request
    )
    k_values = _build_projection_k_grid(
        ell_arr=ell_arr,
        background=background,
        numerics=numerics,
        perturbation_data=perturbation_data,
        # The sampler may request the joint-MCMC workload, but that workload
        # must not opt out of the declared final phase grid.
        allow_final_production_floor=not diagnostic_matrix_fast_path,
        diagnostic_matrix_fast_path=diagnostic_matrix_fast_path,
        surface_ell_max_override=(
            max(int(numerics.ell_max), int(ell_arr.max()))
            if diagnostic_matrix_fast_path
            else None
        ),
        retain_declared_surface=(
            generated_scalar_hierarchy
            and declared_accuracy_controls.get("accuracy_tier") == "final"
        ),
        refinement_anchors=contract_or_params.get(
            "_k_grid_refinement_anchors"
        ),
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
    projection_coarse_eta_plan: numpy.ndarray | None = None

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
            k_max=float(k_values[-1]),
            minimum_nodes=int(los_quadrature_controls.minimum_nodes),
            maximum_nodes=int(los_quadrature_controls.maximum_nodes),
            phase_points_per_cycle=(
                los_quadrature_controls.phase_points_per_cycle
            ),
        )
        los_phase_quadrature_applied = True
    if adaptive_controls.source_enabled:
        source_minimum_nodes = int(adaptive_controls.source_minimum_nodes)
        source_maximum_nodes = int(adaptive_controls.source_maximum_nodes)
        if adaptive_controls.projection_enabled:
            source_minimum_nodes = max(
                source_minimum_nodes,
                int(adaptive_controls.projection_minimum_nodes),
            )
            source_maximum_nodes = max(
                source_maximum_nodes,
                int(adaptive_controls.projection_maximum_nodes),
            )
        if adaptive_controls.projection_enabled:
            eta_los_grid, projection_coarse_eta_plan = (
                build_phase_aware_projection_ladder(
                    eta_los_grid,
                    background=background,
                    k_max=float(k_values[-1]),
                    phase_points_per_cycle=(
                        adaptive_controls.phase_points_per_cycle
                    ),
                    minimum_nodes=source_minimum_nodes,
                    maximum_nodes=source_maximum_nodes,
                )
            )
        else:
            eta_los_grid = phase_aware_eta_grid(
                eta_los_grid,
                visibility=numpy.asarray(
                    background.visibility_of_eta(eta_los_grid),
                    dtype=float,
                ),
                k_max=float(k_values[-1]),
                minimum_nodes=source_minimum_nodes,
                maximum_nodes=source_maximum_nodes,
                phase_points_per_cycle=(
                    adaptive_controls.phase_points_per_cycle
                ),
            )
    if (
        adaptive_controls.projection_enabled
        and not adaptive_controls.source_enabled
    ):
        eta_los_grid, projection_coarse_eta_plan = (
            build_phase_aware_projection_ladder(
                eta_los_grid,
                background=background,
                k_max=float(k_values[-1]),
                phase_points_per_cycle=(
                    adaptive_controls.phase_points_per_cycle
                ),
                minimum_nodes=int(adaptive_controls.projection_minimum_nodes),
                maximum_nodes=int(adaptive_controls.projection_maximum_nodes),
            )
        )

    with performance_timer.phase("preparation"):
        (
            source_grids,
            source_declared_background_histories,
            source_coordinate_rate_histories,
        ) = sample_eta_background_grids(
            eta_los_grid,
            background=background,
            physical_params=physical_params,
            contract_or_params=contract_or_params,
        )
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
        # A contract carrying explicit numerical values without the engine's
        # final accuracy tier is a bounded diagnostic request.  Keep those
        # requests on their declared ladder: production-only phase floors
        # would both hide requested refinements and multiply test/runtime
        # work without improving the diagnostic surface.  Bundled production
        # contracts receive ``accuracy_tier=final`` from the planner and are
        # therefore unaffected.
        adaptive_transfer_k_values: numpy.ndarray | None = None
        adaptive_transfer_phase_status: dict[str, Any] | None = None
        if adaptive_controls.transfer_enabled:
            eta_rec_distance = max(
                float(background.eta0) - float(background.eta_rec),
                1.0,
            )
            configured_transfer_maximum = int(
                adaptive_controls.transfer_maximum_nodes
            )
            transfer_maximum_nodes = configured_transfer_maximum
            if (
                configured_transfer_maximum >= 64
                and adaptive_controls.transfer_relative_tolerance <= 1.0e-2
            ):
                # Strict production transfer surfaces need enough phase nodes
                # to resolve lensing/polarization oscillations.  The declared
                # minimum remains authoritative; this engine-owned floor is
                # applied only to the strict high-resolution tier.
                transfer_maximum_nodes = max(configured_transfer_maximum, 256)
            base_evolution_maximum = min(
                configured_transfer_maximum,
                max(
                    int(k_values.size),
                    2 * int(adaptive_controls.transfer_minimum_nodes),
                ),
            )
            if base_evolution_maximum > int(k_values.size):
                k_values = nested_phase_aware_k_grid(
                    k_values,
                    maximum_nodes=base_evolution_maximum,
                    phase_points_per_cycle=(
                        adaptive_controls.phase_points_per_cycle
                    ),
                    eta_distance=eta_rec_distance,
                    sound_horizon=max(
                        float(background.sound_horizon_mpc),
                        1.0,
                    ),
                )
            adaptive_transfer_k_values = nested_phase_aware_k_grid(
                k_values,
                maximum_nodes=transfer_maximum_nodes,
                phase_points_per_cycle=(
                    adaptive_controls.phase_points_per_cycle
                ),
                eta_distance=eta_rec_distance,
                sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
                require_phase_resolution=bool(
                    declared_accuracy_controls.get(
                        "require_phase_resolution", False
                    )
                ),
            )
            adaptive_transfer_phase_status = phase_aware_k_grid_status(
                adaptive_transfer_k_values,
                phase_points_per_cycle=(
                    adaptive_controls.phase_points_per_cycle
                ),
                eta_distance=eta_rec_distance,
                sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
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
    declared_source_graph = compile_declared_source_graph(
        perturbation_data,
        requested_spectra=requested_spectrum_names,
    )
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
    initial_state_cache_hits = 0
    initial_state_cache_misses = 0
    stage_diagnostic = contract_or_params.get("_stage_diagnostic")
    if stage_diagnostic is None:
        stage_diagnostic = {}
    if not isinstance(stage_diagnostic, Mapping):
        raise ValueError("_stage_diagnostic must be a mapping")
    stage_diagnostic_k_values = tuple(
        sorted(
            {float(value) for value in stage_diagnostic.get("k_values", ())}
        )
    )
    if any(
        not numpy.isfinite(value) or value <= 0.0
        for value in stage_diagnostic_k_values
    ):
        raise ValueError(
            "_stage_diagnostic.k_values must be finite and positive"
        )
    stage_diagnostic_fields = tuple(
        str(value)
        for value in stage_diagnostic.get(
            "fields",
            (
                "Phi",
                "Psi",
                "theta_gamma0",
                "theta_gamma1",
                "theta_gamma2",
                "e_gamma2",
                "theta_b",
                "delta_b",
                "delta_c",
                "delta_nu",
                "sigma_nu",
                "temperature_monopole",
                "temperature_quadrupole",
                "temperature_doppler",
                "polarization_source",
                "lensing_potential",
            ),
        )
    )
    stage_diagnostic_histories_by_k: dict[str, dict[str, Any]] = {}
    source_eta_signature = hashlib.sha256(
        numpy.asarray(source_grids["eta"], dtype=numpy.float64).tobytes()
    ).hexdigest()
    source_history_cache_prefix = (
        _freeze_for_cache(
            {
                key: value
                for key, value in _contract_structural_cache_view(
                    contract_or_params
                ).items()
                if key
                not in {
                    "_engine_numerical_plan",
                    "_numerical_overrides",
                    "_k_grid_refinement_factor",
                    "_k_grid_refinement_anchors",
                    "numerical",
                }
            }
        ),
        cache_key.model_static,
        cache_key.execution_solver,
        declared_source_graph.digest,
        source_eta_signature,
        tuple(sorted(str(name) for name in required_source_names)),
    )

    def _source_history_cache_key(mode_k_value: float) -> tuple[Any, ...]:
        """Return an exact cache key for one parameter/grid source history."""

        return source_history_cache_prefix + (float(mode_k_value),)

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
    collision_kernel_metrics: dict[str, Any] = {
        "schema_version": 1,
        "exact_batch_calls": 0,
        "exact_batch_mode_rows": 0,
        "exact_vectorized_calls": 0,
        "exact_scalar_fallback_calls": 0,
        "exact_scalar_fallback_rows": 0,
        "fallback_matrix_groups": 0,
        "eigendecomposition_lookups": 0,
        "eigendecomposition_cache_hits": 0,
        "eigendecomposition_cache_entries": 0,
        "result_array_allocations": 0,
        "elapsed_seconds": 0.0,
        "digest_sample_count": 0,
        "digest_sample_limit": 8,
        "input_digest": hashlib.sha256(),
        "output_digest": hashlib.sha256(),
    }
    planner_evidence = contract_or_params.get("_engine_planner_evidence")
    if isinstance(planner_evidence, Mapping):
        runtime_envelope["numerical_planner"] = dict(planner_evidence)
    neutrino_temperature_eV = max(
        float(physical_params.Tcmb_K) * _NEUTRINO_TEMPERATURE_EV_PER_K,
        1.0e-12,
    )
    maximum_mass_ratio = max(
        (
            float(runtime.mass_eV) / neutrino_temperature_eV
            for runtime in momentum_runtimes
        ),
        default=0.0,
    )
    momentum_refinement_evidence = (
        evaluate_momentum_grid_refinement_bounds(
            numerical_envelope.momentum_grid_controls,
            mass_ratio_max=maximum_mass_ratio,
            tolerance=numerical_envelope.q_grid_relative_tolerance,
        )
        if momentum_runtimes
        else {}
    )
    failed_momentum_grids = tuple(
        name
        for name, evidence in momentum_refinement_evidence.items()
        if not bool(evidence.get("converged", False))
    )
    if numerical_envelope.accuracy_tier == "final" and failed_momentum_grids:
        raise AdaptiveNonConvergenceError(
            "Declared momentum-q refinement did not converge: "
            + ", ".join(failed_momentum_grids),
            label="momentum-q",
            failed_products=failed_momentum_grids,
            evidence=momentum_refinement_evidence,
        )
    raw_hierarchy_bounds: Mapping[str, Mapping[str, Any]] = {}
    if isinstance(planner_evidence, Mapping):
        hierarchy_resolution = planner_evidence.get("hierarchy_resolution", {})
        if isinstance(hierarchy_resolution, Mapping):
            candidate_bounds = hierarchy_resolution.get(
                "truncation_bounds", {}
            )
            if isinstance(candidate_bounds, Mapping):
                raw_hierarchy_bounds = candidate_bounds
    hierarchy_truncation_evidence = evaluate_hierarchy_truncation_bounds(
        raw_hierarchy_bounds,
        tolerance=numerical_envelope.hierarchy_relative_tolerance,
    )
    declared_hierarchy_families = tuple(
        str(row.get("name", ""))
        for row in hierarchy_schedule_evidence.get("hierarchy_families", ())
        if isinstance(row, Mapping) and str(row.get("name", ""))
    )
    missing_hierarchy_families = tuple(
        name
        for name in declared_hierarchy_families
        if name not in hierarchy_truncation_evidence
    )
    failed_hierarchy_families = tuple(
        name
        for name, evidence in hierarchy_truncation_evidence.items()
        if not bool(evidence.get("converged", False))
    )
    unresolved_hierarchy_families = tuple(
        dict.fromkeys(
            (*missing_hierarchy_families, *failed_hierarchy_families)
        )
    )
    if (
        numerical_envelope.accuracy_tier == "final"
        and unresolved_hierarchy_families
    ):
        raise AdaptiveNonConvergenceError(
            "Declared hierarchy truncation bound failed: "
            + ", ".join(unresolved_hierarchy_families),
            label="hierarchy-depth",
            failed_products=unresolved_hierarchy_families,
            evidence={
                "declared_families": declared_hierarchy_families,
                "missing_families": missing_hierarchy_families,
                "bounds": hierarchy_truncation_evidence,
            },
        )
    runtime_envelope["hierarchy_schedule_evidence"] = {
        str(key): value for key, value in hierarchy_schedule_evidence.items()
    }
    runtime_envelope["hierarchy_schedule_signature"] = str(
        hierarchy_schedule_evidence["signature"]
    )
    runtime_envelope["hierarchy_schedule_cache_identity"] = (
        "compiled_graph_and_declared_physics"
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
    runtime_envelope["resolved_physical_parameters"] = {
        str(name): float(value)
        for name, value in physical_runtime_scalars.items()
        if numpy.isfinite(float(value))
    }
    runtime_envelope["background_resolution_evidence"] = dict(
        getattr(background, "resolution_evidence", {}) or {}
    )
    runtime_envelope["drag_sound_horizon_mpc"] = float(
        background.drag_sound_horizon_mpc
    )
    runtime_envelope["drag_redshift"] = float(background.drag_redshift)
    if background.massive_neutrino_density_grid is not None:
        runtime_envelope["massive_neutrino_density_grid"] = numpy.asarray(
            background.massive_neutrino_density_grid,
            dtype=float,
        )
    if background.massive_neutrino_pressure_grid is not None:
        runtime_envelope["massive_neutrino_pressure_grid"] = numpy.asarray(
            background.massive_neutrino_pressure_grid,
            dtype=float,
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
    full_visibility = numpy.asarray(
        background.visibility_grid,
        dtype=float,
    )
    full_eta = numpy.asarray(background.eta_grid, dtype=float)
    total_visibility = float(numpy.trapz(numpy.abs(full_visibility), full_eta))
    integration_start = float(source_grids["eta"][0])
    omitted_visibility_mask = full_eta < integration_start
    if numpy.any(omitted_visibility_mask):
        omitted_eta = numpy.concatenate(
            (
                full_eta[omitted_visibility_mask],
                numpy.asarray((integration_start,), dtype=float),
            )
        )
        omitted_values = numpy.asarray(
            background.visibility_of_eta(omitted_eta),
            dtype=float,
        )
        omitted_visibility = float(
            numpy.trapz(numpy.abs(omitted_values), omitted_eta)
        )
    else:
        omitted_visibility = 0.0
    omitted_visibility_fraction = omitted_visibility / max(
        total_visibility,
        numpy.finfo(float).tiny,
    )
    configured_limit_ells = _configured_reference_ells(
        perturbation_data,
        maximum_ell=max(int(ell_arr.max()), int(numerics.ell_max)),
    )
    required_grid_ell_min = min(
        (int(numerics.ell_min), *configured_limit_ells)
    )
    required_grid_ell_max = max((int(ell_arr.max()), *configured_limit_ells))
    required_k_start = max(
        float(numerics.k_min),
        0.2
        * max(float(required_grid_ell_min), 2.0)
        / max(float(background.eta0), 1.0e-6),
    )
    required_k_end = 1.5 * (
        (float(required_grid_ell_max) + 16.0)
        / max(float(background.eta0) - float(background.eta_rec), 1.0)
    )
    physical_limit_checks = {
        "k_lower_endpoint": bool(
            numpy.isclose(
                float(k_values[0]),
                required_k_start,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        ),
        "k_upper_endpoint": bool(
            float(k_values[-1]) >= required_k_end * (1.0 - 1.0e-12)
        ),
        "k_declared_domain": bool(
            float(k_values[0]) >= float(numerics.k_min) * (1.0 - 1.0e-12)
            and float(k_values[-1]) <= float(numerics.k_max) * (1.0 + 1.0e-12)
        ),
        "eta_integration_start": bool(
            numpy.isclose(
                float(source_grids["eta"][0]),
                eta_start,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "eta_today_endpoint": bool(
            numpy.isclose(
                float(source_grids["eta"][-1]),
                float(background.eta0),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "visibility_early_tail": bool(omitted_visibility_fraction <= 1.0e-6),
        "thermal_background_start": bool(
            float(background.a_grid[0])
            <= float(numerics.a_min) * (1.0 + 1.0e-12)
        ),
        "thermal_background_end": bool(
            numpy.isclose(
                float(background.a_grid[-1]),
                1.0,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "lensing_ell_support": bool(
            "PP" not in (requested_spectrum_names or set())
            or int(ell_arr.max()) <= int(numerics.ell_max)
        ),
    }
    physical_limit_evidence = {
        "status": (
            "measured" if all(physical_limit_checks.values()) else "unresolved"
        ),
        "checks": physical_limit_checks,
        "k_domain": [float(k_values[0]), float(k_values[-1])],
        "required_k_domain": [required_k_start, required_k_end],
        "declared_k_domain": [
            float(numerics.k_min),
            float(numerics.k_max),
        ],
        "eta_domain": [
            float(source_grids["eta"][0]),
            float(source_grids["eta"][-1]),
        ],
        "background_a_domain": [
            float(background.a_grid[0]),
            float(background.a_grid[-1]),
        ],
        "omitted_visibility_fraction": omitted_visibility_fraction,
        "lensing_sampling_factor": float(numerics.lensing_sampling_factor),
    }
    failed_physical_limits = tuple(
        name for name, passed in physical_limit_checks.items() if not passed
    )
    if numerical_envelope.accuracy_tier == "final" and failed_physical_limits:
        raise AdaptiveNonConvergenceError(
            "Declared physical integration limits are under-resolved: "
            + ", ".join(failed_physical_limits),
            label="physical-limits",
            failed_products=failed_physical_limits,
            evidence=physical_limit_evidence,
        )
    runtime_envelope["physical_limit_evidence"] = physical_limit_evidence
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
    runtime_envelope["schedule_group_count"] = 0
    runtime_envelope["schedule_max_group_size"] = 0
    runtime_envelope["schedule_forced_overintegration"] = 0
    runtime_envelope["hierarchy_schedule_cache_hit_count"] = 0
    runtime_envelope["hierarchy_schedule_cache_miss_count"] = 0
    runtime_envelope["hierarchy_schedule_cache_reused"] = False
    runtime_envelope["evolution_cached_mode_count"] = 0
    runtime_envelope["evolution_missing_mode_count"] = int(k_values.size)
    runtime_envelope["evolution_partial_cache_hit"] = False
    runtime_envelope["evolution_modes_evolved"] = 0
    runtime_envelope["evolution_heartbeat_count"] = 0
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
    runtime_envelope["adaptive_transfer_interpolation_seconds"] = 0.0
    runtime_envelope["adaptive_transfer_refinement_seconds"] = 0.0
    runtime_envelope["adaptive_transfer_base_node_count"] = 0
    runtime_envelope["adaptive_transfer_refined_node_count"] = 0
    runtime_envelope["adaptive_transfer_new_node_count"] = 0
    runtime_envelope["adaptive_transfer_new_node_work_units"] = 0
    runtime_envelope["adaptive_transfer_new_kernel_work_units"] = 0
    runtime_envelope["adaptive_transfer_nested"] = False
    runtime_envelope["adaptive_transfer_source_history_reused"] = False
    runtime_envelope["adaptive_transfer_background_reused"] = False
    runtime_envelope["adaptive_transfer_kernel_reused"] = False
    runtime_envelope["adaptive_transfer_phase_status"] = None
    runtime_envelope["adaptive_source_relative_error"] = 0.0
    runtime_envelope["adaptive_projection_relative_error"] = 0.0
    runtime_envelope["adaptive_evolution_relative_error"] = 0.0
    runtime_envelope["adaptive_evolution_absolute_error"] = 0.0
    runtime_envelope["adaptive_evolution_validation_mode_count"] = 0
    runtime_envelope["adaptive_evolution_validation_mode_indices"] = ()
    runtime_envelope["adaptive_source_independent_mode_count"] = 0
    runtime_envelope["adaptive_source_independent_mode_indices"] = ()
    runtime_envelope["declared_source_history_roles"] = (
        declared_source_history_roles
    )
    runtime_envelope["declared_source_graph"] = declared_source_graph.to_dict()
    runtime_envelope["declared_source_graph_digest"] = (
        declared_source_graph.digest
    )
    runtime_envelope["declared_source_graph_schema"] = int(
        declared_source_graph.schema_version
    )
    runtime_envelope["declared_projection_route_count"] = int(
        len(declared_source_graph.transfer_routes)
    )
    runtime_envelope["declared_projection_active_route_count"] = int(
        sum(
            bool(row.get("active", False))
            for row in declared_source_graph.transfer_routes
        )
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
    declared_accuracy_controls = dict(
        _resolve_declared_accuracy_controls(contract_or_params)
    )
    if generated_scalar_hierarchy:
        declared_accuracy_controls["source_history_reconstruction"] = False
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
    eta_integration_weights = _trapezoid_weights(source_grids["eta"])
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
    exact_collision_phase_cap = any(
        runtime.integration_strategy == "exact"
        for runtime in split_collision_runtimes
    )
    if exact_collision_phase_cap:
        # Strang splitting is second order in the streaming/collision phase.
        # Keep its explicit RK phase budget at or below half a radian even for
        # bounded diagnostic contracts; otherwise changing a legacy phase
        # request changes TE/EE at the percent level despite exact collision
        # exponentials.  This is an engine stability bound, not a model knob.
        generated_final_phase_step = min(generated_final_phase_step, 0.5)
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

    mode_evolution_runtime = build_declared_mode_evolution(
        DeclaredModeEvolutionInputs(
            active_coordinate_rate_histories=active_coordinate_rate_histories,
            active_declared_background_histories=(
                active_declared_background_histories
            ),
            active_grids=active_grids,
            active_k_value=active_k_value,
            adaptive_controls=adaptive_controls,
            adaptive_k_enabled=adaptive_k_enabled,
            background=background,
            collision_kernel_metrics=collision_kernel_metrics,
            contract_or_params=contract_or_params,
            declared_accuracy_controls=declared_accuracy_controls,
            diagnostic_source_audit=diagnostic_source_audit,
            equation_direct_names=equation_direct_names,
            equation_program=equation_program,
            equation_program_specs=equation_program_specs,
            equation_required_names=equation_required_names,
            equation_stage_derivative_steps=equation_stage_derivative_steps,
            exact_collision_phase_cap=exact_collision_phase_cap,
            execution_plan=execution_plan,
            generated_final_evolution_floor=generated_final_evolution_floor,
            generated_final_phase_step=generated_final_phase_step,
            generated_scalar_hierarchy=generated_scalar_hierarchy,
            hierarchy_equation_residuals_by_k=(
                hierarchy_equation_residuals_by_k
            ),
            hierarchy_schedule_evidence=hierarchy_schedule_evidence,
            initial_state_cache_hits=initial_state_cache_hits,
            initial_state_cache_misses=initial_state_cache_misses,
            initial_state_diagnostics_by_k=initial_state_diagnostics_by_k,
            k_values=k_values,
            manifest_summary=manifest_summary,
            metric_history_gradient_residual_by_k=(
                metric_history_gradient_residual_by_k
            ),
            model_label=model_label,
            momentum_grid_context_cache=momentum_grid_context_cache,
            momentum_runtimes=momentum_runtimes,
            numerics=numerics,
            performance_timer=performance_timer,
            perturbation_data=perturbation_data,
            physical_params=physical_params,
            physical_runtime_scalars=physical_runtime_scalars,
            required_source_names=required_source_names,
            runtime_envelope=runtime_envelope,
            runtime_spec=runtime_spec,
            scalar_background_context_cache=scalar_background_context_cache,
            scalar_base_context_cache=scalar_base_context_cache,
            scalar_constraint_diagnostic_projection_count=(
                scalar_constraint_diagnostic_projection_count
            ),
            scalar_constraint_diagnostics=scalar_constraint_diagnostics,
            scalar_constraint_projection_count=(
                scalar_constraint_projection_count
            ),
            scalar_constraint_projection_max_relative_correction=(
                scalar_constraint_projection_max_relative_correction
            ),
            shared_generated_mode_grids=shared_generated_mode_grids,
            shared_generated_mode_grids_enabled=(
                shared_generated_mode_grids_enabled
            ),
            source_context_max_abs_by_k=source_context_max_abs_by_k,
            source_context_pre_resolution_by_k=(
                source_context_pre_resolution_by_k
            ),
            source_coordinate_rate_histories=source_coordinate_rate_histories,
            source_declared_background_histories=(
                source_declared_background_histories
            ),
            source_grids=source_grids,
            source_history_cache_prefix=source_history_cache_prefix,
            source_history_max_abs=source_history_max_abs,
            source_history_max_abs_by_k=source_history_max_abs_by_k,
            source_history_mode_count=source_history_mode_count,
            source_history_reconstruction_enabled=(
                source_history_reconstruction_enabled
            ),
            source_history_residual_samples_by_k=(
                source_history_residual_samples_by_k
            ),
            source_parameters=source_parameters,
            split_collision_runtimes=split_collision_runtimes,
            stage_derivative_steps=stage_derivative_steps,
            stage_diagnostic_fields=stage_diagnostic_fields,
            stage_diagnostic_histories_by_k=stage_diagnostic_histories_by_k,
            stage_diagnostic_k_values=stage_diagnostic_k_values,
            stage_value_steps=stage_value_steps,
            state_history_max_abs_by_k=state_history_max_abs_by_k,
            state_history_polarization_ratio_by_k=(
                state_history_polarization_ratio_by_k
            ),
            transfer_component_observables=transfer_component_observables,
        )
    )
    _evolve_declared_mode = mode_evolution_runtime.evolve_declared_mode
    _evolve_declared_modes_batched = (
        mode_evolution_runtime.evolve_declared_modes_batched
    )
    _record_source_history_diagnostics = (
        mode_evolution_runtime.record_source_history_diagnostics
    )
    _record_stage_diagnostic_histories = (
        mode_evolution_runtime.record_stage_diagnostic_histories
    )
    scalar_initial_constraint_preflight = (
        mode_evolution_runtime.scalar_initial_constraint_preflight
    )

    def _synchronize_mode_evolution_state() -> None:
        """Publish callback-updated evolution state to orchestration."""

        nonlocal active_coordinate_rate_histories
        nonlocal active_declared_background_histories
        nonlocal active_grids
        nonlocal active_k_value
        nonlocal initial_state_cache_hits
        nonlocal initial_state_cache_misses
        nonlocal scalar_background_context_cache
        nonlocal scalar_base_context_cache
        nonlocal scalar_constraint_diagnostic_projection_count
        nonlocal scalar_constraint_diagnostics
        nonlocal scalar_constraint_projection_count
        nonlocal scalar_constraint_projection_max_relative_correction
        nonlocal shared_generated_mode_grids
        nonlocal source_history_mode_count
        state = mode_evolution_runtime.snapshot()
        active_coordinate_rate_histories = state[
            "active_coordinate_rate_histories"
        ]
        active_declared_background_histories = state[
            "active_declared_background_histories"
        ]
        active_grids = state["active_grids"]
        active_k_value = state["active_k_value"]
        initial_state_cache_hits = state["initial_state_cache_hits"]
        initial_state_cache_misses = state["initial_state_cache_misses"]
        scalar_background_context_cache = state[
            "scalar_background_context_cache"
        ]
        scalar_base_context_cache = state["scalar_base_context_cache"]
        scalar_constraint_diagnostic_projection_count = state[
            "scalar_constraint_diagnostic_projection_count"
        ]
        scalar_constraint_diagnostics = state["scalar_constraint_diagnostics"]
        scalar_constraint_projection_count = state[
            "scalar_constraint_projection_count"
        ]
        scalar_constraint_projection_max_relative_correction = state[
            "scalar_constraint_projection_max_relative_correction"
        ]
        shared_generated_mode_grids = state["shared_generated_mode_grids"]
        source_history_mode_count = state["source_history_mode_count"]

    transfer_cache_reuse_allowed = not (
        stage_diagnostic_k_values
        or diagnostic_source_audit
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
    if (
        cached_transfer is not None
        and generated_scalar_hierarchy
        and not cached_transfer.runtime_envelope.get(
            "scalar_constraint_diagnostics"
        )
    ):
        # Older/fast transfer entries may predate the generated scalar audit.
        # Do not reuse one as if it were complete: rebuild once and retain
        # the full constraint evidence with the reusable transfer product.
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
            "source_eta_signature",
            "source_history_residual_samples_by_k",
            "source_history_residual_sample_schema",
            "source_history_refinement",
            "source_history_refinement_mode_count",
            "source_history_bundle_digest",
            "source_history_cache_hit_count",
            "source_history_cache_miss_count",
            "source_history_cache_reused",
            "initial_state_cache_hit_count",
            "initial_state_cache_miss_count",
            "initial_state_cache_reused",
            "source_history_reconstruction_enabled",
            "source_history_reconstruction_diagnostic_only",
            "generated_scalar_source_closure",
            "source_history_derivative_provenance",
            "hierarchy_schedule_evidence",
            "hierarchy_schedule_signature",
            "hierarchy_schedule_cache_identity",
            "hierarchy_schedule_cache_hit_count",
            "hierarchy_schedule_cache_miss_count",
            "hierarchy_schedule_cache_reused",
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
            "collision_kernel_metrics",
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
        if not diagnostic_request:
            cache.set_cmb_spectrum(cache_key, spectrum_data)
            return _get_cached_custom_cmb_spectrum_data(cache_key)
        return spectrum_data

    log_k_values = numpy.log(k_values)
    projection_ell_batch_size = 512 if use_streaming_projection else 128
    kernel_cache_before = cache.cmb_cache_stats()[
        "declared_projection_kernel_batch"
    ]
    source_history_error = 0.0
    source_history_local_relative_error = 0.0
    source_history_absolute_error = 0.0
    source_history_product_errors: dict[str, dict[str, float | int]] = {}
    source_history_refinement_attempts_by_k: dict[
        str, tuple[dict[str, Any], ...]
    ] = {}
    source_history_mode_grids: dict[str, dict[str, Any]] = {}
    source_history_refinement_mode_count = 0
    source_history_refinement_levels = 0
    source_history_coarse_evolution_sample_count = 0
    source_history_fine_evolution_sample_count = 0
    projection_error = 0.0
    projection_absolute_error = 0.0
    projection_product_errors: dict[str, dict[str, float]] = {}
    coarse_projection_spectra: dict[str, numpy.ndarray] = {}
    history_feature_eta = physical_history_anchors(
        source_grids["eta"],
        visibility=source_grids["visibility"],
        interaction_rate=source_grids["collision_rate"],
    )
    evolution_anchor_errors: dict[str, float] = {
        name: 0.0 for name in history_feature_eta
    }
    evolution_anchor_absolute_errors: dict[str, float] = {
        name: 0.0 for name in history_feature_eta
    }
    evolution_error = 0.0
    evolution_absolute_error = 0.0
    evolution_coarse_to_intermediate_error = 0.0
    evolution_coarse_to_intermediate_absolute_error = 0.0
    evolution_intermediate_to_reference_error = 0.0
    evolution_intermediate_to_reference_absolute_error = 0.0
    evolution_coarse_to_intermediate_anchor_errors: dict[str, float] = {
        name: 0.0 for name in history_feature_eta
    }
    evolution_intermediate_to_reference_anchor_errors: dict[str, float] = {
        name: 0.0 for name in history_feature_eta
    }
    evolution_mode_count = 0
    evolution_fine_sample_count = 0
    evolution_intermediate_sample_count = 0
    evolution_coarse_sample_count = 0
    validation_mode_count = min(
        int(k_values.size),
        max(1, int(adaptive_controls.evolution_validation_mode_count)),
    )
    validation_mode_indices = set(
        informative_k_indices(
            k_values,
            count=validation_mode_count,
            feature_k=(
                1.0 / max(float(background.sound_horizon_mpc), 1.0e-12),
                1.0 / max(float(background.eta_rec), 1.0e-12),
                float(ell_arr.max())
                / max(float(background.eta0 - background.eta_rec), 1.0),
            ),
        )
    )
    source_validation_mode_indices = (
        validation_mode_indices if adaptive_controls.source_enabled else set()
    )
    source_eta_indices = nested_coarse_indices(
        source_grids["eta"],
        feature_eta=history_feature_eta,
    )
    source_history_representative_indices = source_eta_indices

    def _refine_source_history_grid(
        source_arrays: Mapping[str, numpy.ndarray],
    ) -> tuple[
        numpy.ndarray,
        Any,
        dict[str, dict[str, float]],
        tuple[dict[str, Any], ...],
    ]:
        """Refine a nested source grid while holding evolution fixed."""

        eta_values = numpy.asarray(source_grids["eta"], dtype=float)
        selected = set(int(index) for index in source_eta_indices)
        maximum_count = min(
            max(len(selected) + 1, int(numpy.floor(0.875 * eta_values.size))),
            int(eta_values.size) - 1,
        )
        attempts: list[dict[str, Any]] = []
        while True:
            indices = numpy.asarray(sorted(selected), dtype=int)
            coarse_eta = eta_values[indices]
            coarse_arrays = {
                name: numpy.asarray(values, dtype=float)[indices]
                for name, values in source_arrays.items()
            }
            aggregate_estimate = estimate_history_convergence(
                coarse_eta,
                coarse_arrays,
                eta_values,
                source_arrays,
                relative_tolerance=(
                    adaptive_controls.source_relative_tolerance
                ),
                absolute_tolerance=max(
                    adaptive_controls.source_absolute_tolerance,
                    1.0e-10,
                ),
                feature_eta=history_feature_eta,
            )
            product_errors: dict[str, dict[str, float]] = {}
            error_surface = numpy.zeros(eta_values.size, dtype=float)
            for source_name, fine_values_raw in source_arrays.items():
                fine_values = numpy.asarray(fine_values_raw, dtype=float)
                product_estimate = estimate_history_convergence(
                    coarse_eta,
                    {source_name: coarse_arrays[source_name]},
                    eta_values,
                    {source_name: fine_values},
                    relative_tolerance=(
                        adaptive_controls.source_relative_tolerance
                    ),
                    absolute_tolerance=max(
                        adaptive_controls.source_absolute_tolerance,
                        1.0e-10,
                    ),
                    feature_eta=history_feature_eta,
                )
                product_scale = max(
                    float(numpy.max(numpy.abs(fine_values), initial=0.0)),
                    adaptive_controls.source_absolute_tolerance,
                )
                product_errors[source_name] = {
                    "relative_error": float(
                        product_estimate.absolute_error / product_scale
                    ),
                    "local_relative_error": float(
                        product_estimate.relative_error
                    ),
                    "absolute_error": float(product_estimate.absolute_error),
                }
                interpolated = numpy.interp(
                    eta_values,
                    coarse_eta,
                    coarse_arrays[source_name],
                )
                difference = numpy.abs(fine_values - interpolated)
                candidate_error = difference / product_scale
                candidate_error[
                    difference <= adaptive_controls.source_absolute_tolerance
                ] = 0.0
                error_surface = numpy.maximum(
                    error_surface,
                    candidate_error,
                )
            failed_products = tuple(
                name
                for name, values in product_errors.items()
                if values["absolute_error"]
                > adaptive_controls.source_absolute_tolerance
                and values["relative_error"]
                > adaptive_controls.source_relative_tolerance
            )
            attempts.append(
                {
                    "source_sample_count": int(indices.size),
                    "relative_error": max(
                        (
                            values["relative_error"]
                            for values in product_errors.values()
                        ),
                        default=0.0,
                    ),
                    "local_relative_error": float(
                        aggregate_estimate.relative_error
                    ),
                    "absolute_error": float(aggregate_estimate.absolute_error),
                    "failed_products": failed_products,
                    "converged": not failed_products,
                    "product_errors": product_errors,
                }
            )
            if (
                not failed_products
                or indices.size >= maximum_count
                or len(attempts) > adaptive_controls.source_maximum_refinements
            ):
                return (
                    indices,
                    aggregate_estimate,
                    product_errors,
                    tuple(attempts),
                )
            error_surface[indices] = 0.0
            failing_indices = numpy.flatnonzero(
                error_surface > adaptive_controls.source_relative_tolerance
            )
            if failing_indices.size == 0:
                failing_indices = numpy.asarray(
                    (int(numpy.argmax(error_surface)),),
                    dtype=int,
                )
            remaining = maximum_count - len(selected)
            if failing_indices.size > remaining:
                ranked = failing_indices[
                    numpy.argsort(
                        error_surface[failing_indices],
                        kind="stable",
                    )
                ]
                failing_indices = ranked[-remaining:]
            selected.update(int(index) for index in failing_indices)

    projection_eta_indices: numpy.ndarray | None = None
    projection_coarse_weights: numpy.ndarray | None = None
    coarse_projection_components: dict[str, numpy.ndarray] = {}
    if adaptive_controls.projection_enabled and source_eta_indices.size >= 3:
        if projection_coarse_eta_plan is None:
            raise RuntimeError("Projection refinement has no coarse eta plan")
        projection_eta_indices = numpy.searchsorted(
            source_grids["eta"],
            projection_coarse_eta_plan,
        )
        if (
            projection_eta_indices.size >= source_grids["eta"].size
            or numpy.any(projection_eta_indices >= source_grids["eta"].size)
            or not numpy.allclose(
                source_grids["eta"][projection_eta_indices],
                projection_coarse_eta_plan,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise RuntimeError(
                "Projection refinement grids are not materially nested"
            )
        projection_coarse_weights = _trapezoid_weights(
            source_grids["eta"][projection_eta_indices]
        )
        coarse_projection_components = {
            name: numpy.zeros_like(values)
            for name, values in transfer_components.items()
        }

    evolution_started = perf_counter()
    _LOGGER.info(
        "CCMBS phase started: model=%s phase=evolution modes=%d",
        model_label,
        int(k_values.size),
    )
    with performance_timer.phase("evolution"):
        cached_mode_source_arrays: dict[int, dict[str, numpy.ndarray]] = {}
        if not diagnostic_source_audit and not stage_diagnostic_k_values:
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
        missing_mode_indices = numpy.asarray(
            [
                index
                for index in range(int(k_values.size))
                if index not in cached_mode_source_arrays
            ],
            dtype=int,
        )
        runtime_envelope["evolution_cached_mode_count"] = int(
            len(cached_mode_source_arrays)
        )
        runtime_envelope["evolution_missing_mode_count"] = int(
            missing_mode_indices.size
        )
        runtime_envelope["evolution_partial_cache_hit"] = bool(
            cached_mode_source_arrays and missing_mode_indices.size
        )
        runtime_envelope["evolution_modes_evolved"] = 0
        if not missing_mode_indices.size:
            batched_mode_source_arrays = cached_mode_source_arrays
            runtime_envelope["evolution_chunks_completed"] = int(
                runtime_envelope["evolution_chunk_count"]
            )
            runtime_envelope["evolution_modes_completed"] = int(k_values.size)
        else:
            batched_mode_source_arrays = {}
            missing_count = int(missing_mode_indices.size)
            missing_chunk_count = (
                missing_count + int(evolution_chunk_size) - 1
            ) // int(evolution_chunk_size)
            for chunk_index, chunk_start in enumerate(
                range(0, missing_count, int(evolution_chunk_size)),
                start=1,
            ):
                missing_chunk = missing_mode_indices[
                    chunk_start : min(
                        chunk_start + int(evolution_chunk_size),
                        missing_count,
                    )
                ]
                chunk_results = _evolve_declared_modes_batched(
                    k_values[missing_chunk]
                )
                for local_index, source_arrays in chunk_results.items():
                    global_index = int(missing_chunk[int(local_index)])
                    batched_mode_source_arrays[global_index] = source_arrays
                runtime_envelope["evolution_modes_evolved"] = int(
                    runtime_envelope["evolution_modes_evolved"]
                ) + len(chunk_results)
                runtime_envelope["evolution_chunks_completed"] = int(
                    chunk_index
                )
                runtime_envelope["evolution_modes_completed"] = int(
                    len(cached_mode_source_arrays)
                    + sum(
                        1
                        for index in batched_mode_source_arrays
                        if index not in cached_mode_source_arrays
                    )
                )
                performance_timer.heartbeat(
                    "evolution",
                    completed=int(
                        runtime_envelope["evolution_modes_completed"]
                    ),
                    total=int(k_values.size),
                )
            runtime_envelope["evolution_chunk_count"] = int(
                missing_chunk_count
            )
            runtime_envelope["evolution_chunks_completed"] = int(
                missing_chunk_count
            )
            runtime_envelope["evolution_modes_completed"] = int(k_values.size)
            performance_timer.heartbeat(
                "evolution",
                completed=int(k_values.size),
                total=int(k_values.size),
            )
            for k_index, source_arrays in batched_mode_source_arrays.items():
                if (
                    not diagnostic_source_audit
                    and not stage_diagnostic_k_values
                ):
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
        _synchronize_mode_evolution_state()
        runtime_envelope["source_history_cache_hit_count"] = int(
            source_history_cache_hits
        )
    runtime_envelope["source_history_cache_miss_count"] = int(
        source_history_cache_misses
    )
    runtime_envelope["source_history_cache_reused"] = bool(
        source_history_cache_hits > 0
    )
    runtime_envelope["initial_state_cache_hit_count"] = int(
        initial_state_cache_hits
    )
    runtime_envelope["initial_state_cache_miss_count"] = int(
        initial_state_cache_misses
    )
    runtime_envelope["initial_state_cache_reused"] = bool(
        initial_state_cache_hits > 0
    )
    _LOGGER.info(
        "CCMBS phase completed: model=%s phase=evolution elapsed=%.3fs",
        model_label,
        perf_counter() - evolution_started,
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
    projection_phase_started = perf_counter()
    _LOGGER.info(
        "CCMBS phase started: model=%s phase=projection modes=%d ells=%d",
        model_label,
        int(k_values.size),
        int(ell_arr.size),
    )
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

            performance_timer.heartbeat(
                "projection",
                completed=int(k_index + 1),
                total=int(k_values.size),
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
        if (
            base_history_sink is not None
            and (
                adaptive_controls.source_enabled
                or adaptive_controls.evolution_enabled
            )
            and k_index in validation_mode_indices
            and "source_histories" not in base_history_sink
        ):
            _evolve_declared_mode(
                float(k_value),
                history_sink=base_history_sink,
                collect_diagnostics=False,
                count_as_primary_evolution=False,
            )
        batched_mode_source_arrays[int(k_index)] = source_arrays
        bound_source_histories = {
            str(component_name): _bind_declared_source_histories(
                component_name=str(component_name),
                component_entry=component_entry,
                source_arrays=source_arrays,
            )
            for component_name, component_entry in (
                transfer_component_observables.items()
            )
        }
        performance_timer.heartbeat(
            "projection",
            completed=int(k_index + 1),
            total=int(k_values.size),
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
                if k_index in source_validation_mode_indices:
                    (
                        source_mode_indices,
                        source_estimate,
                        mode_product_errors,
                        mode_attempts,
                    ) = _refine_source_history_grid(source_arrays)
                    mode_key = f"{float(k_value):.12g}"
                    source_history_refinement_attempts_by_k[mode_key] = (
                        mode_attempts
                    )
                    source_history_mode_grids[mode_key] = {
                        "coarse_sample_count": int(source_mode_indices.size),
                        "fine_sample_count": int(source_grids["eta"].size),
                        "maximum_fraction": 0.875,
                        "coarse_eta_sha256": hashlib.sha256(
                            numpy.asarray(
                                source_grids["eta"][source_mode_indices],
                                dtype=numpy.float64,
                            ).tobytes()
                        ).hexdigest(),
                    }
                    source_history_refinement_levels = max(
                        source_history_refinement_levels,
                        len(mode_attempts),
                    )
                    if (
                        source_mode_indices.size
                        > source_history_representative_indices.size
                    ):
                        source_history_representative_indices = (
                            source_mode_indices
                        )
                    source_history_local_relative_error = max(
                        source_history_local_relative_error,
                        float(source_estimate.relative_error),
                    )
                    source_history_absolute_error = max(
                        source_history_absolute_error,
                        float(source_estimate.absolute_error),
                    )
                    for (
                        source_name,
                        mode_values,
                    ) in mode_product_errors.items():
                        product_row = source_history_product_errors.setdefault(
                            source_name,
                            {
                                "relative_error": 0.0,
                                "local_relative_error": 0.0,
                                "absolute_error": 0.0,
                                "mode_count": 0,
                            },
                        )
                        product_row["relative_error"] = max(
                            float(product_row["relative_error"]),
                            mode_values["relative_error"],
                        )
                        product_row["local_relative_error"] = max(
                            float(product_row["local_relative_error"]),
                            mode_values["local_relative_error"],
                        )
                        product_row["absolute_error"] = max(
                            float(product_row["absolute_error"]),
                            mode_values["absolute_error"],
                        )
                        product_row["mode_count"] = (
                            int(product_row["mode_count"]) + 1
                        )
                        source_history_error = max(
                            source_history_error,
                            mode_values["relative_error"],
                        )
                    source_history_refinement_mode_count += 1
                    source_history_coarse_evolution_sample_count = max(
                        source_history_coarse_evolution_sample_count,
                        int(base_history_sink["evolution_eta"].size),
                    )
                    source_history_fine_evolution_sample_count = max(
                        source_history_fine_evolution_sample_count,
                        int(base_history_sink["evolution_eta"].size),
                    )
            if (
                adaptive_controls.evolution_enabled
                and k_index in validation_mode_indices
            ):
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
                    count_as_primary_evolution=False,
                )
                _evolve_declared_mode(
                    float(k_value),
                    evolution_sample_count_override=intermediate_sample_count,
                    history_sink=intermediate_history_sink,
                    collect_diagnostics=False,
                    count_as_primary_evolution=False,
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
                        feature_eta=history_feature_eta,
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
                        feature_eta=history_feature_eta,
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
                source_histories = bound_source_histories[str(component_name)]
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
                if (
                    projection_eta_indices is not None
                    and projection_coarse_weights is not None
                ):
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
                        kernel_batch=_slice_projection_kernel_batch(
                            kernel_batch,
                            projection_eta_indices,
                        ),
                        k_value=float(k_value),
                        eta_weights=projection_coarse_weights,
                        chi_grid=source_grids["chi"][projection_eta_indices],
                        source_chi=source_chi,
                        source_histories={
                            role_name: history[projection_eta_indices]
                            for role_name, history in source_histories.items()
                        },
                    )
                    coarse_projection_components[component_name][
                        batch_indices, k_index
                    ] = coarse_values
        performance_timer.add(
            "projection",
            perf_counter() - projection_phase_started,
        )
    _LOGGER.info(
        "CCMBS phase completed: model=%s phase=projection elapsed=%.3fs",
        model_label,
        perf_counter() - projection_phase_started,
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
        if coarse_projection_components:
            coarse_projection_spectra = _integrate_declared_spectra(
                physical_params=physical_params,
                perturbation_data=perturbation_data,
                power_spectrum_observables=power_spectrum_observables,
                transfer_components=coarse_projection_components,
                k_values=k_values,
                log_k_values=log_k_values,
            )
    spectrum_k_values = k_values
    spectrum_transfer_components = transfer_components

    if (
        adaptive_controls.transfer_enabled
        and adaptive_transfer_k_values is not None
    ):
        """Refine projection quadrature from reusable base products."""

        interpolation_started = perf_counter()
        base_k_values = numpy.asarray(k_values, dtype=float)
        refined_k_values = numpy.asarray(
            adaptive_transfer_k_values,
            dtype=float,
        )
        base_log_k_values = numpy.log(base_k_values)
        refined_log_k_values = numpy.log(refined_k_values)
        source_interpolation_started = perf_counter()
        source_names = tuple(
            sorted(
                {
                    str(name)
                    for source_arrays in batched_mode_source_arrays.values()
                    for name in source_arrays
                }
            )
        )
        if not source_names or len(batched_mode_source_arrays) != int(
            base_k_values.size
        ):
            raise RuntimeError(
                "Adaptive transfer refinement requires every base source "
                "history for nested projection refinement"
            )
        refined_source_arrays: dict[str, numpy.ndarray] = {}
        refined_new_mode_count = 0
        refined_warm_mode_count = 0
        if (
            adaptive_controls.transfer_algorithm
            == "diagnostic_transfer_interpolation"
        ):
            for source_name in source_names:
                source_matrix = numpy.vstack(
                    [
                        numpy.asarray(
                            batched_mode_source_arrays[index][source_name],
                            dtype=float,
                        )
                        for index in range(int(base_k_values.size))
                    ]
                )
                if base_k_values.size >= 4:
                    interpolated = CubicSpline(
                        base_log_k_values,
                        source_matrix,
                        axis=0,
                        extrapolate=False,
                    )(refined_log_k_values)
                else:
                    interpolated = numpy.vstack(
                        [
                            numpy.interp(
                                refined_log_k_values,
                                base_log_k_values,
                                source_matrix[:, eta_index],
                            )
                            for eta_index in range(source_matrix.shape[1])
                        ]
                    ).T
                if not numpy.all(numpy.isfinite(interpolated)):
                    raise ValueError(
                        "Adaptive source-history interpolation produced "
                        f"non-finite values: {source_name}"
                    )
                refined_source_arrays[source_name] = numpy.asarray(
                    interpolated,
                    dtype=float,
                )
        else:
            base_index_by_k = {
                float(value): int(index)
                for index, value in enumerate(base_k_values)
            }
            refined_rows: list[dict[str, numpy.ndarray]] = []
            for refined_k_value in refined_k_values:
                base_index = base_index_by_k.get(float(refined_k_value))
                if base_index is not None:
                    refined_rows.append(batched_mode_source_arrays[base_index])
                    continue
                source_cache_key = _source_history_cache_key(
                    float(refined_k_value)
                )
                cached_source_arrays = cache.get_cmb_source_history(
                    source_cache_key
                )
                if cached_source_arrays is None:
                    _, actual_source_arrays = _evolve_declared_mode(
                        float(refined_k_value),
                        collect_diagnostics=False,
                        count_as_primary_evolution=False,
                    )
                    cache.set_cmb_source_history(
                        source_cache_key,
                        {
                            str(name): numpy.asarray(
                                values,
                                dtype=float,
                            ).copy()
                            for name, values in actual_source_arrays.items()
                        },
                    )
                    refined_new_mode_count += 1
                else:
                    actual_source_arrays = {
                        str(name): numpy.asarray(values, dtype=float).copy()
                        for name, values in cached_source_arrays.items()
                    }
                    refined_warm_mode_count += 1
                refined_rows.append(actual_source_arrays)
            for source_name in source_names:
                resolved_rows = numpy.vstack(
                    [
                        numpy.asarray(row[source_name], dtype=float)
                        for row in refined_rows
                    ]
                )
                if not numpy.all(numpy.isfinite(resolved_rows)):
                    raise ValueError(
                        "Adaptive source-history refinement produced "
                        f"non-finite values: {source_name}"
                    )
                refined_source_arrays[source_name] = resolved_rows
        source_interpolation_elapsed = perf_counter() - (
            source_interpolation_started
        )

        refined_transfer_components = {
            str(component_name): numpy.zeros(
                (int(ell_arr.size), int(refined_k_values.size)),
                dtype=float,
            )
            for component_name in transfer_component_observables
        }
        refined_kernel_work_units = 0
        refined_projection_started = perf_counter()
        for refined_index, refined_k_value in enumerate(refined_k_values):
            x_values = numpy.asarray(
                refined_k_value * (eta0 - source_grids["eta"]),
                dtype=float,
            )
            x_signature = hashlib.sha256(x_values.tobytes()).hexdigest()
            cache.store_bessel_inputs(x_signature, x_values.copy())
            mode_ell_limit = _projection_ell_limit_for_mode(
                ell_values=ell_arr,
                x_values=x_values,
            )
            mode_ell_indices = numpy.flatnonzero(ell_arr <= mode_ell_limit)
            if mode_ell_indices.size == 0:
                continue
            source_arrays = {
                source_name: refined_source_values[refined_index]
                for source_name, refined_source_values in (
                    refined_source_arrays.items()
                )
            }
            for ell_start in range(
                0,
                int(ell_arr.size),
                projection_ell_batch_size,
            ):
                ell_stop = min(
                    ell_start + projection_ell_batch_size,
                    int(ell_arr.size),
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
                kernel_batch = _get_cached_declared_projection_kernel_batch(
                    ell_signature,
                    x_signature,
                    x_values=x_values,
                    required_sectors=streaming_projection_sectors,
                )
                refined_kernel_work_units += int(batch_indices.size)
                for (
                    component_name,
                    component_entry,
                ) in transfer_component_observables.items():
                    source_histories = _bind_declared_source_histories(
                        component_name=str(component_name),
                        component_entry=component_entry,
                        source_arrays=source_arrays,
                    )
                    refined_transfer_components[component_name][
                        batch_indices, refined_index
                    ] = _declared_graph_projection(
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
                        k_value=float(refined_k_value),
                        eta_weights=eta_integration_weights,
                        chi_grid=source_grids["chi"],
                        source_chi=source_chi,
                        source_histories=source_histories,
                    )
        refined_projection_elapsed = (
            perf_counter() - refined_projection_started
        )

        if (
            adaptive_controls.transfer_algorithm
            == "diagnostic_transfer_interpolation"
        ):
            # This explicitly named diagnostic comparison remains available
            # to bounded fixtures.  Production selection is independent of
            # tolerance values and always validates source histories before
            # projection.
            refined_transfer_components = {}
            for (
                component_name,
                component_values,
            ) in transfer_components.items():
                matrix = numpy.asarray(component_values, dtype=float)
                if base_k_values.size >= 4:
                    interpolated = CubicSpline(
                        base_log_k_values,
                        matrix,
                        axis=1,
                        extrapolate=False,
                    )(refined_log_k_values)
                else:
                    interpolated = numpy.vstack(
                        [
                            numpy.interp(
                                refined_log_k_values,
                                base_log_k_values,
                                row,
                            )
                            for row in matrix
                        ]
                    )
                if not numpy.all(numpy.isfinite(interpolated)):
                    raise ValueError(
                        "Adaptive transfer interpolation produced non-finite "
                        f"values: {component_name}"
                    )
                refined_transfer_components[component_name] = numpy.asarray(
                    interpolated,
                    dtype=float,
                )
            refined_kernel_work_units = 0

        refined_spectrum_started = perf_counter()
        refined_spectra: dict[str, numpy.ndarray] = {}
        for (
            observable_name,
            observable_entry,
        ) in power_spectrum_observables.items():
            primordial_grid = _primordial_power_grid_for_observable(
                physical_params=physical_params,
                perturbation_data=perturbation_data,
                observable_entry=observable_entry,
                k_values=refined_k_values,
            )
            refined_spectra[str(observable_name)] = _integrate_power_spectrum(
                primordial_grid=primordial_grid,
                log_k_values=refined_log_k_values,
                primary=refined_transfer_components[
                    str(observable_entry.primary)
                ],
                secondary=refined_transfer_components[
                    str(observable_entry.secondary)
                ],
                auto_spectrum=(
                    str(observable_entry.primary)
                    == str(observable_entry.secondary)
                ),
            )
        refined_spectrum_elapsed = perf_counter() - refined_spectrum_started

        transfer_estimates = [
            estimate_convergence(
                numpy.asarray(spectra_results[name], dtype=float),
                numpy.asarray(refined_spectra[name], dtype=float),
                relative_tolerance=(
                    adaptive_controls.transfer_relative_tolerance
                ),
                absolute_tolerance=(
                    adaptive_controls.transfer_absolute_tolerance
                ),
            )
            for name in refined_spectra
        ]
        transfer_estimate = max(
            transfer_estimates,
            key=lambda estimate: estimate.relative_error,
            default=ConvergenceEstimate(0.0, 0.0, True),
        )
        spectra_results = refined_spectra
        spectrum_k_values = refined_k_values
        spectrum_transfer_components = refined_transfer_components
        runtime_envelope["k_grid_actual_count"] = int(refined_k_values.size)
        interpolation_elapsed = perf_counter() - interpolation_started
        performance_timer.add("projection", refined_projection_elapsed)
        performance_timer.add("power_spectrum", refined_spectrum_elapsed)
        runtime_envelope["adaptive_transfer_relative_error"] = float(
            transfer_estimate.relative_error
        )
        runtime_envelope["adaptive_transfer_absolute_error"] = float(
            transfer_estimate.absolute_error
        )
        runtime_envelope["adaptive_transfer_refinement_levels"] = 1
        runtime_envelope["adaptive_transfer_source_refinement_seconds"] = (
            float(source_interpolation_elapsed)
        )
        runtime_envelope["adaptive_transfer_interpolation_seconds"] = (
            float(source_interpolation_elapsed)
            if adaptive_controls.transfer_algorithm
            == "diagnostic_transfer_interpolation"
            else 0.0
        )
        runtime_envelope["adaptive_transfer_refinement_seconds"] = float(
            interpolation_elapsed
        )
        runtime_envelope["adaptive_transfer_base_node_count"] = int(
            base_k_values.size
        )
        runtime_envelope["adaptive_transfer_refined_node_count"] = int(
            refined_k_values.size
        )
        runtime_envelope["adaptive_transfer_new_node_count"] = int(
            max(0, refined_k_values.size - base_k_values.size)
        )
        runtime_envelope["adaptive_transfer_new_node_work_units"] = int(
            refined_new_mode_count
        )
        runtime_envelope["adaptive_transfer_warm_node_count"] = int(
            refined_warm_mode_count
        )
        runtime_envelope["adaptive_transfer_algorithm"] = str(
            adaptive_controls.transfer_algorithm
        )
        runtime_envelope["adaptive_transfer_new_kernel_work_units"] = int(
            refined_kernel_work_units * max(1, len(transfer_components))
        )
        runtime_envelope["adaptive_transfer_nested"] = bool(
            numpy.all(numpy.isin(base_k_values, refined_k_values))
        )
        runtime_envelope["adaptive_transfer_source_history_reused"] = True
        runtime_envelope["adaptive_transfer_background_reused"] = True
        runtime_envelope["adaptive_transfer_kernel_reused"] = True
        runtime_envelope["adaptive_transfer_phase_status"] = (
            None
            if adaptive_transfer_phase_status is None
            else dict(adaptive_transfer_phase_status)
        )
        actual_new_node_count = int(
            max(0, refined_k_values.size - base_k_values.size)
        )
        if (
            adaptive_controls.transfer_algorithm
            == "nested_independent_source_projection"
            and actual_new_node_count > 0
            and refined_new_mode_count + refined_warm_mode_count
            != actual_new_node_count
        ):
            raise RuntimeError(
                "Adaptive transfer refinement did not account for every "
                "new nested k mode"
            )
        if (
            adaptive_controls.transfer_algorithm
            == "nested_independent_source_projection"
            and actual_new_node_count == 0
        ):
            raise AdaptiveNonConvergenceError(
                "Adaptive transfer refinement produced an identical "
                "effective k grid",
                label="transfer-k-grid",
                failed_products=("k_grid",),
                evidence={
                    "base_nodes": int(base_k_values.size),
                    "refined_nodes": int(refined_k_values.size),
                },
            )
        require_convergence(
            transfer_estimate,
            label="transfer k-grid",
            fail_on_nonconvergence=adaptive_controls.fail_on_nonconvergence,
        )
    if adaptive_controls.source_enabled:
        runtime_envelope["adaptive_source_relative_error"] = float(
            source_history_error
        )
        runtime_envelope["adaptive_source_absolute_error"] = float(
            source_history_absolute_error
        )
        runtime_envelope["adaptive_source_refinement_levels"] = int(
            source_history_refinement_levels
        )
        runtime_envelope["adaptive_source_validation_mode_count"] = int(
            source_history_refinement_mode_count
        )
        runtime_envelope["adaptive_source_validation_mode_indices"] = tuple(
            sorted(source_validation_mode_indices)
        )
        failed_source_products = tuple(
            name
            for name, evidence in source_history_product_errors.items()
            if float(evidence["absolute_error"])
            > adaptive_controls.source_absolute_tolerance
            and float(evidence["relative_error"])
            > adaptive_controls.source_relative_tolerance
        )
        if adaptive_controls.fail_on_nonconvergence and failed_source_products:
            raise AdaptiveNonConvergenceError(
                "Declared source-history refinement did not converge: "
                + ", ".join(
                    "{}={:.6g}".format(
                        name,
                        float(
                            source_history_product_errors[name][
                                "relative_error"
                            ]
                        ),
                    )
                    for name in failed_source_products
                ),
                label="source-history",
                failed_products=failed_source_products,
                evidence={
                    "product_errors": source_history_product_errors,
                    "refinement_attempts_by_k": (
                        source_history_refinement_attempts_by_k
                    ),
                },
            )
    if adaptive_controls.projection_enabled:
        if projection_eta_indices is None:
            raise RuntimeError("Projection refinement built no nested grid")
        for product_name, projected_values in spectra_results.items():
            coarse_values = coarse_projection_spectra[product_name]
            absolute_error = float(
                numpy.max(
                    numpy.abs(coarse_values - projected_values),
                    initial=0.0,
                )
            )
            product_scale = max(
                float(numpy.max(numpy.abs(coarse_values), initial=0.0)),
                float(numpy.max(numpy.abs(projected_values), initial=0.0)),
                adaptive_controls.projection_absolute_tolerance,
            )
            refinement_ratio = (source_grids["eta"].size - 1) / max(
                projection_eta_indices.size - 1, 1
            )
            richardson_factor = 1.0 / max(
                refinement_ratio**2 - 1.0,
                numpy.finfo(float).eps,
            )
            estimated_remaining_error = absolute_error * richardson_factor
            relative_error = absolute_error / product_scale
            projection_product_errors[product_name] = {
                "relative_error": relative_error,
                "absolute_error": absolute_error,
                "measured_difference": absolute_error,
                "richardson_estimated_remaining_error": (
                    estimated_remaining_error
                ),
                "scale": product_scale,
                "refinement_ratio": refinement_ratio,
                "richardson_order": 2.0,
            }
            projection_error = max(projection_error, relative_error)
            projection_absolute_error = max(
                projection_absolute_error,
                absolute_error,
            )
        runtime_envelope["adaptive_projection_relative_error"] = float(
            projection_error
        )
        runtime_envelope["adaptive_projection_absolute_error"] = float(
            projection_absolute_error
        )
        runtime_envelope["adaptive_projection_refinement_levels"] = 1
        runtime_envelope["adaptive_projection_refinement_attempts"] = (
            {
                "coarse_sample_count": int(projection_eta_indices.size),
                "fine_sample_count": int(source_grids["eta"].size),
                "relative_error": float(projection_error),
                "absolute_error": float(projection_absolute_error),
                "product_errors": {
                    name: dict(values)
                    for name, values in sorted(
                        projection_product_errors.items()
                    )
                },
            },
        )
        failed_projection_products = tuple(
            name
            for name, evidence in projection_product_errors.items()
            if evidence["absolute_error"]
            > adaptive_controls.projection_absolute_tolerance
            and evidence["relative_error"]
            > adaptive_controls.projection_relative_tolerance
        )
        if (
            adaptive_controls.fail_on_nonconvergence
            and failed_projection_products
        ):
            raise AdaptiveNonConvergenceError(
                "Declared line-of-sight projection refinement did not "
                "converge: "
                + ", ".join(
                    f"{name}="
                    f"{projection_product_errors[name]['relative_error']:.6g}"
                    for name in failed_projection_products
                ),
                label="line-of-sight projection",
                failed_products=failed_projection_products,
                evidence={
                    "product_errors": projection_product_errors,
                    "refinement_attempts": runtime_envelope[
                        "adaptive_projection_refinement_attempts"
                    ],
                },
            )

    if adaptive_controls.evolution_enabled:
        runtime_envelope["adaptive_evolution_relative_error"] = float(
            evolution_error
        )
        runtime_envelope["adaptive_evolution_absolute_error"] = float(
            evolution_absolute_error
        )
        runtime_envelope["adaptive_evolution_refinement_levels"] = 2
        runtime_envelope["adaptive_evolution_validation_mode_count"] = int(
            evolution_mode_count
        )
        runtime_envelope["adaptive_evolution_validation_mode_indices"] = tuple(
            sorted(validation_mode_indices)
        )
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
            converged=bool(
                evolution_absolute_error
                <= adaptive_controls.evolution_absolute_tolerance
                or evolution_error
                <= adaptive_controls.evolution_relative_tolerance
            ),
        )
        require_convergence(
            evolution_estimate,
            label="scalar evolution history",
            fail_on_nonconvergence=adaptive_controls.fail_on_nonconvergence,
        )

    background_refinement = dict(
        runtime_envelope.get("background_resolution_evidence", {}).get(
            "refinement", {}
        )
        or {}
    )
    hierarchy_controls = dict(numerical_envelope.hierarchy_controls)
    hierarchy_is_applicable = bool(declared_hierarchy_families)
    momentum_controls = {
        str(name): dict(values)
        for name, values in numerical_envelope.momentum_grid_controls.items()
    }
    runtime_envelope["resolution_axis_evidence"] = {
        "background": {
            "method": "measured_background_refinement",
            "status": (
                "measured"
                if bool(background_refinement.get("converged", False))
                else "unresolved"
            ),
            "evidence": background_refinement,
        },
        "momentum_q": {
            "method": "doubled_count_extended_support_thermal_moments",
            "status": (
                "validated_bound"
                if momentum_refinement_evidence
                and all(
                    bool(evidence.get("converged", False))
                    for evidence in momentum_refinement_evidence.values()
                )
                else (
                    "not_applicable" if not momentum_controls else "unresolved"
                )
            ),
            "reason": (
                "thermal-tail support and quadrature order are enforced by "
                "the final numerical envelope"
                if momentum_controls
                else "no massive-neutrino momentum hierarchy is declared"
            ),
            "controls": momentum_controls,
            "evidence": momentum_refinement_evidence,
            "relative_tolerance": float(
                numerical_envelope.q_grid_relative_tolerance
            ),
        },
        "hierarchy_depth": {
            "method": "free_streaming_spherical_tail_bound",
            "status": (
                "validated_bound"
                if hierarchy_is_applicable
                and hierarchy_truncation_evidence
                and all(
                    bool(evidence.get("converged", False))
                    for evidence in hierarchy_truncation_evidence.values()
                )
                else (
                    "not_applicable"
                    if not hierarchy_is_applicable
                    else "unresolved"
                )
            ),
            "reason": (
                "active hierarchy families meet the engine final-depth floor"
                if hierarchy_is_applicable
                else "no declared hierarchy family is active"
            ),
            "controls": hierarchy_controls,
            "evidence": hierarchy_truncation_evidence,
            "relative_tolerance": float(
                numerical_envelope.hierarchy_relative_tolerance
            ),
        },
        "evolution": {
            "method": "independent_dense_feature_history_refinement",
            "status": (
                "measured"
                if adaptive_controls.evolution_enabled
                else "not_applicable"
            ),
            "mode_count": int(evolution_mode_count),
            "mode_indices": tuple(sorted(validation_mode_indices)),
            "relative_error": float(evolution_error),
            "absolute_error": float(evolution_absolute_error),
        },
        "source": {
            "method": "nested_source_history_interpolation_refinement",
            "status": (
                "measured"
                if adaptive_controls.source_enabled
                else "not_applicable"
            ),
            "mode_count": int(source_history_refinement_mode_count),
            "mode_indices": tuple(sorted(source_validation_mode_indices)),
            "relative_error": float(source_history_error),
            "absolute_error": float(source_history_absolute_error),
        },
        "projection": {
            "method": "nested_line_of_sight_quadrature_refinement",
            "status": (
                "measured"
                if adaptive_controls.projection_enabled
                else "not_applicable"
            ),
            "relative_error": float(projection_error),
            "absolute_error": float(projection_absolute_error),
            "product_errors": {
                name: dict(values)
                for name, values in sorted(projection_product_errors.items())
            },
            "refinement_attempts": tuple(
                runtime_envelope.get(
                    "adaptive_projection_refinement_attempts", ()
                )
            ),
        },
        "physical_limits": {
            "method": "measured_k_eta_tail_and_lensing_support",
            **physical_limit_evidence,
        },
    }

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
                        use_positive_trapezoid=True,
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
        adaptive_eta_integration_weights = _trapezoid_weights(
            adaptive_eta_grid
        )
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
                            use_positive_trapezoid=True,
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
                            use_positive_trapezoid=True,
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

    base_postprocessing_evidence = build_postprocessing_evidence(
        transfer_components=transfer_components,
        unlensed_spectra=spectra_results,
        output_spectra=spectra_results,
        requested_spectra=(
            tuple(sorted(power_spectrum_observables))
            + tuple(sorted(physical_zero_spectra))
        ),
        spectrum_availability=spectrum_availability,
        ell_grid=ell_arr,
        k_grid=k_values,
        lensed=False,
    )
    runtime_envelope["postprocessing_evidence"] = base_postprocessing_evidence
    if not bool(base_postprocessing_evidence["accepted"]):
        raise ValueError(
            "Declared CMB surface validation failed: "
            + "; ".join(base_postprocessing_evidence["issues"])
        )

    _synchronize_mode_evolution_state()
    elapsed_seconds = perf_counter() - request_started
    runtime_envelope["scalar_initial_constraint_preflight"] = (
        scalar_initial_constraint_preflight
    )
    if background.dark_energy_audit:
        runtime_envelope["dark_energy_background"] = dict(
            background.dark_energy_audit
        )
    if background.modified_background_audit:
        runtime_envelope["modified_model_background"] = dict(
            background.modified_background_audit
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
        "coarse_sample_count": int(source_history_representative_indices.size),
        "mode_count": int(source_history_mode_count),
        "refinement_mode_count": int(source_history_refinement_mode_count),
        "coarse_evolution_sample_count": int(
            source_history_coarse_evolution_sample_count
        ),
        "fine_evolution_sample_count": int(
            source_history_fine_evolution_sample_count
        ),
        "independently_evolved": False,
        "evolution_held_fixed": bool(adaptive_controls.source_enabled),
        "validation_mode_indices": tuple(
            sorted(source_validation_mode_indices)
        ),
        "roles": declared_source_history_roles,
        "finite": True,
        "relative_error": float(source_history_error),
        "local_relative_error": float(source_history_local_relative_error),
        "absolute_error": float(source_history_absolute_error),
        "product_errors": {
            name: dict(values)
            for name, values in sorted(source_history_product_errors.items())
        },
        "mode_grids": {
            name: dict(values)
            for name, values in sorted(source_history_mode_grids.items())
        },
        "refinement_attempts_by_k": {
            name: tuple(dict(attempt) for attempt in attempts)
            for name, attempts in sorted(
                source_history_refinement_attempts_by_k.items()
            )
        },
        "tolerance_relative": float(
            adaptive_controls.source_relative_tolerance
        ),
        "tolerance_absolute": float(
            adaptive_controls.source_absolute_tolerance
        ),
    }
    coarse_eta_values = numpy.asarray(
        source_grids["eta"][source_history_representative_indices],
        dtype=numpy.float64,
    )
    coarse_eta_signature = hashlib.sha256(
        coarse_eta_values.tobytes()
    ).hexdigest()
    source_history_refinement = {
        "axis": "eta",
        "independently_evolved": False,
        "evolution_held_fixed": bool(adaptive_controls.source_enabled),
        "validation_mode_indices": tuple(
            sorted(source_validation_mode_indices)
        ),
        "coarse_indices": tuple(
            int(index) for index in source_history_representative_indices
        ),
        "coarse_eta": tuple(float(value) for value in coarse_eta_values),
        "fine_eta": tuple(
            float(value)
            for value in numpy.asarray(source_grids["eta"], dtype=float)
        ),
        "coarse_eta_sha256": coarse_eta_signature,
        "fine_eta_sha256": source_eta_signature,
        "coarse_sample_count": int(coarse_eta_values.size),
        "fine_sample_count": int(source_grids["eta"].size),
        "coarse_evolution_sample_count": int(
            source_history_coarse_evolution_sample_count
        ),
        "fine_evolution_sample_count": int(
            source_history_fine_evolution_sample_count
        ),
        "mode_count": int(source_history_mode_count),
        "refinement_mode_count": int(source_history_refinement_mode_count),
        "relative_error": float(source_history_error),
        "local_relative_error": float(source_history_local_relative_error),
        "absolute_error": float(source_history_absolute_error),
        "product_errors": {
            name: dict(values)
            for name, values in sorted(source_history_product_errors.items())
        },
        "mode_grids": {
            name: dict(values)
            for name, values in sorted(source_history_mode_grids.items())
        },
        "refinement_attempts_by_k": {
            name: tuple(dict(attempt) for attempt in attempts)
            for name, attempts in sorted(
                source_history_refinement_attempts_by_k.items()
            )
        },
    }
    runtime_envelope["source_eta_signature"] = source_eta_signature
    runtime_envelope["source_history_refinement"] = source_history_refinement
    runtime_envelope["source_history_refinement_mode_count"] = int(
        source_history_refinement_mode_count
    )
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
        runtime_envelope["source_history_bundle_digest"] = (
            _build_source_history_bundle_digest(
                source_eta_signature=source_eta_signature,
                source_history_refinement=source_history_refinement,
                source_history_residual_samples_by_k=(
                    source_history_residual_samples_by_k
                ),
                hierarchy_equation_residuals_by_k=(
                    hierarchy_equation_residuals_by_k
                ),
                initial_state_diagnostics_by_k=initial_state_diagnostics_by_k,
                metric_history_gradient_residual_by_k=(
                    metric_history_gradient_residual_by_k
                ),
                runtime_envelope=runtime_envelope,
            )
        )
    else:
        bundle_status = (
            "unavailable" if generated_scalar_hierarchy else "not_applicable"
        )
        bundle_reason = (
            "diagnostic source-history capture disabled"
            if generated_scalar_hierarchy
            else "generated source-history samples unavailable"
        )
        runtime_envelope["source_history_bundle_digest"] = {
            "schema_version": 1,
            "status": bundle_status,
            "reason": bundle_reason,
            "sha256": None,
            "source_eta_sha256": source_eta_signature,
            "mode_count": int(source_history_mode_count),
            "sample_count": 0,
            "included_fields": (),
        }
    if stage_diagnostic_k_values:
        for requested_k_value in stage_diagnostic_k_values:
            diagnostic_key = f"{requested_k_value:.12g}"
            record = stage_diagnostic_histories_by_k.get(diagnostic_key)
            if record is None:
                selected_index = int(
                    numpy.argmin(numpy.abs(k_values - requested_k_value))
                )
                selected_k_value = float(k_values[selected_index])
                history_sink: dict[str, Any] = {}
                state_histories, source_arrays = _evolve_declared_mode(
                    selected_k_value,
                    history_sink=history_sink,
                    collect_diagnostics=False,
                )
                _record_stage_diagnostic_histories(
                    selected_index,
                    selected_k_value,
                    evolution_eta=history_sink["evolution_eta"],
                    evolution_histories=history_sink["evolution_histories"],
                    source_eta=history_sink["source_eta"],
                    source_histories=state_histories,
                    source_arrays=source_arrays,
                )
                record = stage_diagnostic_histories_by_k[diagnostic_key]
            stage_diagnostic_histories_by_k[diagnostic_key] = {
                "requested_k": float(record["requested_k"]),
                "selected_k": float(record["selected_k"]),
                "evolution_eta": numpy.asarray(
                    record["evolution_eta"],
                    dtype=float,
                ).tolist(),
                "evolution_histories": {
                    name: numpy.asarray(values, dtype=float).tolist()
                    for name, values in record["evolution_histories"].items()
                },
                "source_eta": numpy.asarray(
                    record["source_eta"],
                    dtype=float,
                ).tolist(),
                "source_histories": {
                    name: numpy.asarray(values, dtype=float).tolist()
                    for name, values in record["source_histories"].items()
                },
            }
        runtime_envelope["stage_diagnostic"] = {
            "schema_version": 1,
            "requested_k_values": stage_diagnostic_k_values,
            "fields": stage_diagnostic_fields,
            "histories_by_k": stage_diagnostic_histories_by_k,
        }
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
    runtime_envelope["collision_kernel_metrics"] = {
        key: (
            value.hexdigest()
            if key in {"input_digest", "output_digest"}
            else value
        )
        for key, value in collision_kernel_metrics.items()
        if key not in {"input_digest", "output_digest"}
        or hasattr(value, "hexdigest")
    }
    runtime_envelope["collision_kernel_metrics"]["input_sha256"] = (
        collision_kernel_metrics["input_digest"].hexdigest()
    )
    runtime_envelope["collision_kernel_metrics"]["output_sha256"] = (
        collision_kernel_metrics["output_digest"].hexdigest()
    )
    runtime_envelope["collision_kernel_metrics"].pop("input_digest", None)
    runtime_envelope["collision_kernel_metrics"].pop("output_digest", None)
    runtime_envelope["projection_bessel_batch_count"] = int(bessel_batch_count)
    runtime_envelope["projection_bessel_mode_count"] = int(bessel_mode_count)
    runtime_envelope["projection_chunk_count"] = int(bessel_batch_count)
    runtime_envelope["projection_chunk_size"] = int(_BESSEL_MAX_MODE_BATCH)
    runtime_envelope["projection_chunk_accumulation_order"] = "k_index"
    runtime_envelope["projection_peak_bessel_cells"] = int(
        _BESSEL_WORK_CELL_BUDGET
    )
    hierarchy_schedule_cache_after = cache.cmb_cache_stats()[
        "hierarchy_schedule"
    ]
    runtime_envelope["hierarchy_schedule_cache_hit_count"] = int(
        hierarchy_schedule_cache_after["hits"]
        - hierarchy_schedule_cache_before["hits"]
    )
    runtime_envelope["hierarchy_schedule_cache_miss_count"] = int(
        hierarchy_schedule_cache_after["misses"]
        - hierarchy_schedule_cache_before["misses"]
    )
    runtime_envelope["hierarchy_schedule_cache_reused"] = bool(
        runtime_envelope["hierarchy_schedule_cache_hit_count"] > 0
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
                k_grid=spectrum_k_values,
                transfer_components=spectrum_transfer_components,
                runtime_envelope=runtime_envelope,
            ),
        )
    spectrum_data = CustomCMBSpectrumData(
        ell_grid=ell_arr,
        k_grid=spectrum_k_values,
        transfer_components=FrozenMapping(
            {
                name: matrix
                for name, matrix in spectrum_transfer_components.items()
            }
        ),
        spectra=FrozenMapping(spectra_results),
        runtime_envelope=FrozenMapping(runtime_envelope),
        spectrum_availability=FrozenMapping(spectrum_availability),
    )
    if not diagnostic_request:
        cache.set_cmb_spectrum(cache_key, spectrum_data)
        return _get_cached_custom_cmb_spectrum_data(cache_key)
    return spectrum_data


def _refined_adaptive_transfer_controls(
    contract: Mapping[str, Any],
    *,
    factor: int,
) -> dict[str, Any] | None:
    """Scale the engine transfer ladder for an independent k refinement."""

    raw_controls = contract.get("_engine_accuracy_controls")
    if not isinstance(raw_controls, Mapping):
        return None
    controls = dict(raw_controls)
    raw_transfer = controls.get("adaptive_transfer")
    if not isinstance(raw_transfer, Mapping):
        return controls
    transfer = dict(raw_transfer)
    for name in ("minimum_nodes", "maximum_nodes"):
        if name in transfer:
            transfer[name] = int(transfer[name]) * int(factor)
    controls["adaptive_transfer"] = transfer
    return controls


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
        raw_stage_diagnostic = contract_or_params.get("_stage_diagnostic")
        stage_diagnostic_requested = bool(
            isinstance(raw_stage_diagnostic, Mapping)
            and raw_stage_diagnostic.get("k_values") is not None
            and len(raw_stage_diagnostic.get("k_values")) > 0
        )
        diagnostic_request = bool(
            workload.startswith("fixed_parameter_diagnostic")
            or stage_diagnostic_requested
        )
        production_controls = resolve_production_scalar_convergence(
            contract_or_params
        )
        request_ells = tuple(int(value) for value in ells)
        production_enforced = bool(
            production_controls.enabled
            and workload != "joint_mcmc"
            and not contract_or_params.get(
                "_diagnostic_matrix_fast_path", False
            )
            and not contract_or_params.get(
                "_defer_production_scalar_convergence", False
            )
        )
        if production_enforced and requested is not None:
            effective_requested_spectra = tuple(
                dict.fromkeys(requested + production_controls.required_spectra)
            )
        result = _compute_custom_cmb_spectrum_data_impl(
            contract_or_params,
            request_ells,
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
                contract_or_params.get(
                    "_engine_numerical_plan",
                    contract_or_params.get("numerical", {}) or {},
                )
                or {}
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
            refined_accuracy_controls = _refined_adaptive_transfer_controls(
                refined_contract,
                factor=production_controls.k_refinement_factor,
            )
            if refined_accuracy_controls is not None:
                refined_contract["_engine_accuracy_controls"] = (
                    refined_accuracy_controls
                )
            refined_contract["_numerical_overrides"] = {
                # The grid builder applies the declared refinement factor.
                # Keep the base count here so the safety floor and the
                # refinement factor cannot silently multiply one another.
                "k_sample_count": base_k_count,
            }
            refined_contract["_k_grid_refinement_anchors"] = tuple(
                float(value) for value in numpy.asarray(result.k_grid)
            )
            refined_timer = PhaseTimer()
            refinement_started = perf_counter()
            refined = _compute_custom_cmb_spectrum_data_impl(
                refined_contract,
                request_ells,
                background_provider=background_provider,
                requested_spectra=effective_requested_spectra,
                diagnostic_source_audit=workload.startswith(
                    "fixed_parameter_diagnostic"
                ),
                performance_timer=refined_timer,
            )
            base_k_grid = numpy.asarray(result.k_grid, dtype=numpy.float64)
            refined_k_grid = numpy.asarray(refined.k_grid, dtype=numpy.float64)
            base_grid_digest = _projection_array_digest(base_k_grid)
            refined_grid_digest = _projection_array_digest(refined_k_grid)
            nested = bool(numpy.all(numpy.isin(base_k_grid, refined_k_grid)))
            distinct = bool(
                base_grid_digest != refined_grid_digest
                and not numpy.array_equal(base_k_grid, refined_k_grid)
            )
            new_node_count = int(
                max(0, refined_k_grid.size - base_k_grid.size)
            )
            if not distinct or not nested or new_node_count <= 0:
                raise ValueError(
                    "Production scalar convergence produced invalid k-grid "
                    "evidence: grids must be distinct, nested, and add nodes"
                )
            required_for_report = tuple(
                dict.fromkeys(
                    tuple(production_controls.required_spectra)
                    + tuple(requested or ())
                )
            )
            report = evaluate_spectrum_refinement(
                result.spectra,
                refined.spectra,
                required_spectra=required_for_report,
                relative_tolerances=(production_controls.relative_tolerances),
            )
            refined_runtime_envelope = dict(refined.runtime_envelope)
            measured_new_work = int(
                refined_runtime_envelope.get("evolution_modes_evolved", 0)
            ) + int(
                refined_runtime_envelope.get(
                    "adaptive_transfer_new_node_work_units",
                    0,
                )
            )
            exact_refinement_reuse = bool(
                refined_timer.cache_state == "exact_cache_hit"
            )
            if not exact_refinement_reuse and measured_new_work <= 0:
                raise ValueError(
                    "Cold production scalar refinement recorded no measured "
                    "new work"
                )
            production_record = {
                "axis": "k_sample_count",
                "base_count": int(base_k_grid.size),
                "refined_count": int(refined_k_grid.size),
                "declared_base_count": base_k_count,
                "declared_refined_count": (
                    base_k_count * production_controls.k_refinement_factor
                ),
                "refinement_factor": production_controls.k_refinement_factor,
                "required_spectra": required_for_report,
                "base_grid_sha256": base_grid_digest,
                "refined_grid_sha256": refined_grid_digest,
                "nested": nested,
                "distinct": distinct,
                "new_node_count": new_node_count,
                "new_node_work_units": measured_new_work,
                "refined_cache_state": refined_timer.cache_state,
                "cold_refinement": bool(not exact_refinement_reuse),
                "warm_reuse": exact_refinement_reuse,
                "matching_accepted_finer_calculation": bool(
                    exact_refinement_reuse
                    and distinct
                    and nested
                    and new_node_count > 0
                ),
                "refinement_identity": hashlib.sha256(
                    (
                        f"{base_grid_digest}:{refined_grid_digest}:"
                        f"{production_controls.k_refinement_factor}"
                    ).encode("utf-8")
                ).hexdigest(),
                "metrics": report.to_dict()["metrics"],
                "converged": report.converged,
                "fail_on_nonconvergence": (
                    production_controls.fail_on_nonconvergence
                ),
                "elapsed_seconds": perf_counter() - refinement_started,
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
            if not diagnostic_request:
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
    telemetry_context = dict(context)
    telemetry_context["runtime"] = _runtime_telemetry_context(result)
    cache.record_cmb_performance(
        timing,
        cache_hit=timer.cache_state == "exact_cache_hit",
        workload=workload,
        cache_state=timer.cache_state,
        work_units=timer.work_units,
        context=telemetry_context,
    )
    return result
