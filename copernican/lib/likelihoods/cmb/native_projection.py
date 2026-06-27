r"""Declared native transfer projection and spectrum integration helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

import numpy
from scipy.optimize import least_squares

from ...cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_kernel_spec,
)
from ...engine_adapter import FrozenMapping
from ...perturbation_contract import (
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
    _custom_cmb_spectrum_cache_key,
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
    _build_declared_base_context,
    _compile_declared_graph_execution_plan,
    _compile_declared_perturbation_contract,
    _compute_tight_coupling_drag,
    _declared_momentum_grid_context,
    _declared_runtime_seed,
    _evaluate_declared_initial_state,
    _resolve_declared_graph_context,
    _resolve_declared_momentum_grid_runtimes,
)

_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}


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
    weighted = primordial_ld[numpy.newaxis, :] * (primary_ld * secondary_ld)
    integrated = (
        4.0
        * numpy.longdouble(math.pi)
        * numpy.trapz(weighted, log_k_ld, axis=1)
    )
    # Keep the raw spectrum in extended precision until the public solver
    # applies its output scaling and final float conversion.
    return numpy.asarray(integrated, dtype=numpy.longdouble)


def _declared_graph_projection(
    *,
    projection: str,
    kernel: str | None,
    kernel_batch: _DeclaredProjectionKernelBatch,
    eta_weights: numpy.ndarray,
    chi_grid: numpy.ndarray,
    source_chi: float,
    source_histories: Mapping[str, numpy.ndarray],
) -> numpy.ndarray:
    """Return projected transfer component values for every ell."""

    j_l = kernel_batch.j_l
    j_l_derivative = kernel_batch.j_l_derivative
    e_kernel = kernel_batch.e_kernel
    b_kernel = kernel_batch.b_kernel

    def _apply_kernel(kernel_name: str) -> numpy.ndarray:
        """Return the ell-batched kernel selected by ``kernel_name``."""

        kernel_spec = get_declared_projection_kernel_spec(kernel_name)
        if kernel_spec.kind == "temperature_mixed":
            raise ValueError(
                "Temperature mixed kernels must use the dedicated "
                "temperature projection dispatch."
            )
        if kernel_spec.kind == "spherical_bessel":
            return j_l
        if kernel_spec.kind == "spherical_bessel_derivative":
            return j_l_derivative
        if kernel_spec.kind == "spin2_e":
            return e_kernel
        if kernel_spec.kind == "spin2_b":
            return b_kernel
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
        return projected
    if projection in {
        "line_of_sight_polarization_e",
        "line_of_sight_signal",
        "line_of_sight_signal_derivative",
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


def _trapezoid_weights(grid: numpy.ndarray) -> numpy.ndarray:
    """Return the integration weights for one strictly increasing grid."""

    step_sizes = numpy.diff(grid)
    if step_sizes.size == 0 or not numpy.all(numpy.isfinite(step_sizes)):
        raise ValueError("eta_los_grid must be a finite grid")
    if numpy.any(step_sizes <= 0.0):
        raise ValueError("eta_los_grid must be strictly increasing")
    weights = numpy.empty_like(grid, dtype=float)
    weights[0] = 0.5 * step_sizes[0]
    weights[-1] = 0.5 * step_sizes[-1]
    if grid.size > 2:
        weights[1:-1] = 0.5 * (step_sizes[:-1] + step_sizes[1:])
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


def _validate_runtime_envelope_controls(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the validated runtime-envelope control mapping."""

    accuracy_controls = _resolve_declared_accuracy_controls(contract)
    runtime_envelope = accuracy_controls.get("runtime_envelope")
    if runtime_envelope in (None, "bounded"):
        return {}
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

    rule_entries = getattr(perturbation_data, "conservation_rules", {}) or {}
    if not rule_entries:
        return
    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for rule_name, rule_entry in rule_entries.items():
            rule_kind = str(rule_entry.kind or "absolute_max")
            if rule_kind != "absolute_max":
                raise ValueError(
                    "Declared conservation rule uses unsupported kind "
                    f"'{rule_kind}': {rule_name}"
                )
            residual = numpy.asarray(
                _evaluate_compiled_expression_noerr(
                    rule_entry.compiled_expression,
                    context,
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
) -> CustomCMBSpectrumData:
    """Return transfer functions and spectra for a declared CMB graph."""

    cache_key = _custom_cmb_spectrum_cache_key(
        contract_or_params,
        ells,
        background_provider,
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
    runtime_spec = execution_plan.runtime_spec
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
    minimum_eta_samples = max(128, 128 * eta_los_refinement)
    if eta_los_grid.size < minimum_eta_samples:
        eta_los_grid = numpy.linspace(
            eta_start,
            float(background.eta_grid[-1]),
            minimum_eta_samples,
        )
    eta_los_background = background.sample(eta_los_grid)
    a_los_grid = numpy.asarray(eta_los_background["a"], dtype=float)
    z_los_grid = numpy.asarray(eta_los_background["z"], dtype=float)
    H_los_grid = numpy.asarray(eta_los_background["H"], dtype=float)
    tau_los_grid = numpy.asarray(eta_los_background["tau"], dtype=float)
    tau_dot_los_grid = numpy.asarray(
        eta_los_background["tau_dot"],
        dtype=float,
    )
    visibility_los_grid = numpy.asarray(
        eta_los_background["visibility"],
        dtype=float,
    )
    chi_los_grid = numpy.asarray(
        eta_los_background["chi"],
        dtype=float,
    )
    angular_diameter_distance_grid = numpy.asarray(
        eta_los_background["angular_diameter_distance"],
        dtype=float,
    )
    sound_speed_los_grid = numpy.asarray(
        eta_los_background["sound_speed"],
        dtype=float,
    )
    Hconf_los_grid = a_los_grid * H_los_grid / _C_LIGHT_KM_S
    baryon_loading_grid = (
        3.0
        * physical_params.Omega_b0
        * a_los_grid
        / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
    )
    collision_rate_grid = numpy.maximum(-tau_dot_los_grid, 0.0)
    free_streaming_grid = 1.0 / (
        1.0 + collision_rate_grid / max(float(collision_rate_grid.max()), 1.0)
    )
    sound_speed_sq_grid = 1.0 / (3.0 * (1.0 + baryon_loading_grid))
    declared_background_los = _resolve_declared_background_context(
        contract_or_params,
        a_values=a_los_grid,
        z_values=z_los_grid,
    )
    declared_background_histories: dict[str, numpy.ndarray] = {}
    for name, raw_value in declared_background_los.items():
        if name in {"a", "z"}:
            continue
        history = numpy.asarray(raw_value, dtype=float)
        if history.ndim == 0:
            history = numpy.full_like(
                eta_los_grid,
                float(history),
                dtype=float,
            )
        if history.shape != eta_los_grid.shape:
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
        "a": a_los_grid,
        "z": z_los_grid,
        "eta": eta_los_grid,
        "H": H_los_grid,
        "Hconf": Hconf_los_grid,
        "tau": tau_los_grid,
        "tau_dot": tau_dot_los_grid,
        "visibility": visibility_los_grid,
        "chi": chi_los_grid,
        "angular_diameter_distance": angular_diameter_distance_grid,
        "sound_speed": sound_speed_los_grid,
    }
    for name, history in declared_background_histories.items():
        coordinate_histories.setdefault(name, history)
    coordinate_rate_histories = {
        "eta": numpy.ones_like(eta_los_grid, dtype=float)
    }
    for name, history in coordinate_histories.items():
        if name == "eta":
            continue
        coordinate_rate_histories[name] = numpy.asarray(
            numpy.gradient(history, eta_los_grid, edge_order=1),
            dtype=float,
        )

    eta0_floor = max(background.eta0, 1.0e-6)
    k_min = max(
        numerics.k_min,
        0.2 * max(float(ell_arr.min()), 2.0) / eta0_floor,
    )
    eta_rec_distance = max(background.eta0 - background.eta_rec, 1.0)
    required_k_max = 1.5 * ((float(ell_arr.max()) + 16.0) / eta_rec_distance)
    k_max = max(
        required_k_max,
        min(numerics.k_max, max(12.0 * k_min, 0.08)),
    )
    k_values = numpy.logspace(
        math.log10(k_min),
        math.log10(k_max),
        max(16, int(numerics.k_sample_count)),
    )
    k_values = numpy.asarray(k_values, dtype=float)

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

    transfer_component_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "transfer_component"
    }
    power_spectrum_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "angular_power_spectrum"
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
        eta_count=int(eta_los_grid.size),
        state_slot_count=int(len(runtime_spec.state_slots)),
        transfer_component_count=int(len(transfer_component_observables)),
        momentum_point_count=int(
            sum(runtime.points.size for runtime in momentum_runtimes)
        ),
    )
    transfer_components = {
        name: numpy.zeros((ell_arr.size, k_values.size), dtype=float)
        for name in transfer_component_observables
    }
    eta_integration_weights = _trapezoid_weights(eta_los_grid)

    def _blend_history(
        history: numpy.ndarray,
        *,
        step_index: int,
        blend: float,
    ) -> float:
        """Return one linearly interpolated history value."""

        next_index = min(step_index + 1, eta_los_grid.size - 1)
        weight_next = float(blend)
        weight_current = 1.0 - weight_next
        return float(
            weight_current * history[step_index]
            + weight_next * history[next_index]
        )

    def _scalar_background_context(
        step_index: int,
        blend: float,
    ) -> tuple[float, dict[str, float]]:
        """Return one interpolated scalar background context."""

        eta_value = _blend_history(
            eta_los_grid,
            step_index=step_index,
            blend=blend,
        )
        scalar_context = {
            "a": _blend_history(
                a_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "z": _blend_history(
                z_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "eta": float(eta_value),
            "H": _blend_history(
                H_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "Hconf": _blend_history(
                Hconf_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "tau": _blend_history(
                tau_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "tau_dot": _blend_history(
                tau_dot_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "visibility": _blend_history(
                visibility_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "chi": _blend_history(
                chi_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "angular_diameter_distance": _blend_history(
                angular_diameter_distance_grid,
                step_index=step_index,
                blend=blend,
            ),
            "sound_speed": _blend_history(
                sound_speed_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "sound_speed_sq": _blend_history(
                sound_speed_sq_grid,
                step_index=step_index,
                blend=blend,
            ),
            "collision_rate": _blend_history(
                collision_rate_grid,
                step_index=step_index,
                blend=blend,
            ),
            "free_streaming": _blend_history(
                free_streaming_grid,
                step_index=step_index,
                blend=blend,
            ),
            "sound_horizon": float(background.sound_horizon_mpc),
        }
        for name, history in declared_background_histories.items():
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
        for legacy_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
            derivative_symbol = f"__d1_{wrt_name}_{legacy_name}"
            if derivative_symbol not in scalar_context:
                continue
            rate = float(scalar_context[derivative_symbol])
            break
        else:
            if wrt_name not in coordinate_rate_histories:
                raise ValueError(
                    "Declared CMB coordinate transform does not support "
                    f"wrt '{wrt_name}'."
                )
            rate = _blend_history(
                coordinate_rate_histories[wrt_name],
                step_index=step_index,
                blend=blend,
            )
        if not numpy.isfinite(rate) or abs(rate) <= 1.0e-12:
            eta_value = _blend_history(
                eta_los_grid,
                step_index=step_index,
                blend=blend,
            )
            raise ValueError(
                "Declared CMB coordinate transform is singular for "
                f"wrt '{wrt_name}' at eta={eta_value}, k={k_value}"
            )
        return rate

    def _build_scalar_state_context(
        state_vector: numpy.ndarray,
        *,
        k_value: float,
        eta_value: float,
        background_scalars: Mapping[str, float],
    ) -> dict[str, Any]:
        """Return the scalar expression environment for one solver stage."""

        context = _build_declared_base_context(
            perturbation_data=perturbation_data,
            model_parameters=source_parameters,
            physical_params=physical_params,
            numerics=numerics,
            k_value=float(k_value),
            eta_value=float(eta_value),
            background_scalars=background_scalars,
        )
        for slot in runtime_spec.state_slots:
            value = float(state_vector[slot.index])
            if slot.order == 0:
                context[slot.variable] = value
            else:
                context[f"__d{slot.order}_{slot.variable}_{slot.wrt}"] = value
        return _resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
        )

    def _build_array_context(
        histories: Mapping[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> dict[str, Any]:
        """Return the array-valued expression environment for one mode."""

        context = {
            "a": a_los_grid,
            "z": z_los_grid,
            "eta": eta_los_grid,
            "H": H_los_grid,
            "Hconf": Hconf_los_grid,
            "tau": tau_los_grid,
            "tau_dot": tau_dot_los_grid,
            "visibility": visibility_los_grid,
            "chi": chi_los_grid,
            "angular_diameter_distance": numpy.asarray(
                angular_diameter_distance_grid,
                dtype=float,
            ),
            "sound_speed": sound_speed_los_grid,
            "sound_speed_sq": sound_speed_sq_grid,
            "collision_rate": collision_rate_grid,
            "free_streaming": free_streaming_grid,
            "tight_coupling_drag": _compute_tight_coupling_drag(
                collision_rate=collision_rate_grid,
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
        for name, value in physical_runtime_scalars.items():
            context[name] = float(value)
        for name, history in declared_background_histories.items():
            context[name] = numpy.asarray(history, dtype=float)
        for name, value in source_parameters.items():
            context[name] = float(value)
        context.update(
            _declared_momentum_grid_context(
                perturbation_data,
                model_parameters=source_parameters,
                physical_params=physical_params,
                scale_factor=a_los_grid,
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
            eta_grid=eta_los_grid,
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
                        eta_los_grid,
                        float(value),
                        dtype=float,
                    )
                if value.shape != eta_los_grid.shape:
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
    ) -> numpy.ndarray:
        """Return the state derivative for one RK stage."""

        eta_value, background_scalars = _scalar_background_context(
            step_index,
            blend,
        )
        scalar_context = _build_scalar_state_context(
            state_vector,
            k_value=float(k_value),
            eta_value=float(eta_value),
            background_scalars=background_scalars,
        )
        derivative = numpy.zeros_like(state_vector, dtype=float)
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for slot_plan in execution_plan.equation_slot_plans:
                coordinate_rate = _resolve_coordinate_rate(
                    wrt_name=slot_plan.wrt,
                    scalar_context=scalar_context,
                    step_index=step_index,
                    blend=blend,
                    k_value=float(k_value),
                )
                if slot_plan.promote_from_index is not None:
                    derivative[slot_plan.state_index] = (
                        float(state_vector[slot_plan.promote_from_index])
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
        if not numpy.all(numpy.isfinite(derivative)):
            bad_indices = numpy.flatnonzero(~numpy.isfinite(derivative))
            bad_index = int(bad_indices[0]) if bad_indices.size else -1
            raise ValueError(
                "Declared CMB evolution produced non-finite derivatives at "
                f"eta={eta_value}, k={k_value}, state_index={bad_index}"
            )
        return derivative

    def _evolve_declared_mode(
        k_value: float,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
        """Integrate one Fourier mode through the declared graph."""

        end_boundary_entries = execution_plan.end_condition_entries

        def _advance_declared_interval(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            dt: float,
            k_value: float,
        ) -> numpy.ndarray:
            """Advance one LOS interval with adaptive RK4 sub-stepping."""

            _, start_scalars = _scalar_background_context(step_index, 0.0)
            _, end_scalars = _scalar_background_context(step_index, 1.0)
            start_drag = _compute_tight_coupling_drag(
                collision_rate=float(start_scalars["collision_rate"]),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
            end_drag = _compute_tight_coupling_drag(
                collision_rate=float(end_scalars["collision_rate"]),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
            stiffness_scale = max(
                abs(float(k_value)),
                abs(float(start_scalars["Hconf"])),
                abs(float(end_scalars["Hconf"])),
                abs(float(start_drag)),
                abs(float(end_drag)),
                1.0e-12,
            )
            target_stage_scale = 0.25
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
            max_substep_count = 512
            while substep_count <= max_substep_count:
                trial_state = numpy.asarray(state_vector, dtype=float).copy()
                sub_dt = dt / float(substep_count)
                failed = False
                for substep_index in range(substep_count):
                    blend_start = substep_index / substep_count
                    blend_mid = (substep_index + 0.5) / substep_count
                    blend_end = (substep_index + 1.0) / substep_count
                    stage_rhs_initial = _mode_rhs(
                        trial_state,
                        step_index=step_index,
                        blend=blend_start,
                        k_value=float(k_value),
                    )
                    stage_rhs_mid_a = _mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_initial,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                    )
                    stage_rhs_mid_b = _mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_mid_a,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                    )
                    stage_rhs_final = _mode_rhs(
                        trial_state + sub_dt * stage_rhs_mid_b,
                        step_index=step_index,
                        blend=blend_end,
                        k_value=float(k_value),
                    )
                    candidate_state = trial_state + (sub_dt / 6.0) * (
                        stage_rhs_initial
                        + 2.0 * stage_rhs_mid_a
                        + 2.0 * stage_rhs_mid_b
                        + stage_rhs_final
                    )
                    if not numpy.all(numpy.isfinite(candidate_state)):
                        failed = True
                        break
                    trial_state = candidate_state
                if not failed:
                    return trial_state
                substep_count *= 2
            raise ValueError(
                "Declared CMB evolution produced non-finite state values "
                f"at k={k_value}, step_index={step_index}"
            )

        def _integrate_declared_state_history(
            initial_state: numpy.ndarray,
        ) -> tuple[dict[str, numpy.ndarray], numpy.ndarray]:
            """Return mode histories and the final state vector."""

            histories = {
                slot.variable: numpy.empty_like(eta_los_grid, dtype=float)
                for slot in runtime_spec.state_slots
                if slot.order == 0
            }
            state = numpy.asarray(initial_state, dtype=float).copy()
            for step_index, eta_value in enumerate(eta_los_grid):
                for slot in runtime_spec.state_slots:
                    if slot.order != 0:
                        continue
                    histories[slot.variable][step_index] = state[slot.index]
                if step_index == eta_los_grid.size - 1:
                    break
                dt = float(eta_los_grid[step_index + 1] - eta_value)
                state = _advance_declared_interval(
                    state,
                    step_index=step_index,
                    dt=dt,
                    k_value=float(k_value),
                )
            return histories, state

        def _evaluate_end_boundary_residuals(
            final_state: numpy.ndarray,
        ) -> numpy.ndarray:
            """Return end-boundary residuals for one integrated mode."""

            if not end_boundary_entries:
                return numpy.zeros(0, dtype=float)
            final_eta, final_background = _scalar_background_context(
                eta_los_grid.size - 1,
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

        initial_eta, initial_background = _scalar_background_context(0, 0.0)
        initial_context = _build_declared_base_context(
            perturbation_data=perturbation_data,
            model_parameters=source_parameters,
            physical_params=physical_params,
            numerics=numerics,
            k_value=float(k_value),
            eta_value=float(initial_eta),
            background_scalars=initial_background,
        )
        initial_state, assigned_targets = _evaluate_declared_initial_state(
            perturbation_data=perturbation_data,
            execution_plan=execution_plan,
            base_context=initial_context,
        )
        state = numpy.asarray(initial_state, dtype=float)
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
        array_context = _build_array_context(histories, k_value=float(k_value))
        source_arrays = _evaluate_declared_sources(
            array_context,
            k_value=float(k_value),
        )
        conservation_context = dict(array_context)
        conservation_context.update(source_arrays)
        _validate_declared_conservation_rules(
            perturbation_data=perturbation_data,
            context=conservation_context,
            k_value=float(k_value),
        )
        return histories, source_arrays

    log_k_values = numpy.log(k_values)
    primordial_grid = physical_params.primordial_amplitude * numpy.power(
        k_values / 0.05,
        physical_params.primordial_spectral_index - 1.0,
    )
    ell_signature = tuple(int(ell_value) for ell_value in ell_arr)

    for k_index, k_value in enumerate(k_values):
        _, source_arrays = _evolve_declared_mode(float(k_value))
        x_values = k_value * (eta0 - eta_los_grid)
        x_signature = hashlib.sha256(
            numpy.asarray(x_values, dtype=float).tobytes()
        ).hexdigest()
        native_cache.store_bessel_inputs(
            x_signature,
            numpy.asarray(x_values, dtype=float).copy(),
        )
        kernel_batch = _get_cached_declared_projection_kernel_batch(
            ell_signature,
            x_signature,
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
            transfer_components[component_name][:, k_index] = (
                _declared_graph_projection(
                    projection=str(component_entry.projection or ""),
                    kernel=(
                        None
                        if component_entry.kernel is None
                        else str(component_entry.kernel)
                    ),
                    kernel_batch=kernel_batch,
                    eta_weights=eta_integration_weights,
                    chi_grid=chi_los_grid,
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
