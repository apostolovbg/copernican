r"""Declared native perturbation compilation and graph-evolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy

from ...engine_adapter import FrozenMapping
from ...perturbation_contract import (
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)
from . import native_cache
from .native_background import (
    _LEGACY_DECLARED_EVOLUTION_COORDINATES,
    _coerce_numeric_scalar,
    _CustomCMBNumerics,
    _CustomCMBPhysicalParameters,
    _physical_runtime_scalars,
    _resolve_declared_accuracy_controls,
)


def _compile_declared_perturbation_contract(
    contract: Mapping[str, Any],
):
    """Return the precompiled perturbation contract for generic execution."""

    precompiled = contract.get("perturbation_data")
    if precompiled is not None:
        return precompiled
    raise ValueError(
        "Native CMB execution requires precompiled perturbation_data. "
        "Prepare the runtime through model_coder before likelihood "
        "evaluation."
    )


@dataclass(frozen=True, slots=True)
class _DeclaredStateSlot:
    """Describe one state-vector slot for the declared graph solver."""

    variable: str
    wrt: str
    order: int
    index: int


@dataclass(frozen=True, slots=True)
class _DeclaredGraphRuntimeSpec:
    """Prepared runtime metadata for the declared graph solver."""

    evolution_variable: str
    state_slots: tuple[_DeclaredStateSlot, ...]
    state_index_by_key: FrozenMapping
    equation_by_variable: FrozenMapping
    equation_orders: FrozenMapping
    equation_wrt_by_variable: FrozenMapping


@dataclass(frozen=True, slots=True)
class _DeclaredDerivativeStep:
    """Prepared derivative-symbol resolution metadata."""

    output_name: str
    variable: str
    wrt: str
    order: int
    slot_name: str


@dataclass(frozen=True, slots=True)
class _DeclaredValueStep:
    """Prepared expression or algebraic relation evaluation step."""

    output_name: str
    compiled_expression: Any
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DeclaredEquationSlotPlan:
    """Prepared derivative update rule for one state-vector slot."""

    state_index: int
    wrt: str
    promote_from_index: int | None
    compiled_rhs: Any | None
    equation_name: str | None


@dataclass(frozen=True, slots=True)
class _DeclaredGraphExecutionPlan:
    """Immutable compiled execution plan for the declared graph solver."""

    runtime_spec: _DeclaredGraphRuntimeSpec
    derivative_steps: tuple[_DeclaredDerivativeStep, ...]
    value_steps: tuple[_DeclaredValueStep, ...]
    source_steps: tuple[_DeclaredValueStep, ...]
    start_condition_entries: tuple[Any, ...]
    end_condition_entries: tuple[Any, ...]
    equation_slot_plans: tuple[_DeclaredEquationSlotPlan, ...]


@dataclass(frozen=True, slots=True)
class _DeclaredMomentumGridRuntime:
    """Prepared quadrature metadata for one declared momentum grid."""

    name: str
    points: numpy.ndarray
    weights: numpy.ndarray
    mass_eV: float
    family_names: tuple[str, ...]


def _prepare_declared_graph_runtime_spec(
    perturbation_data: Any,
) -> _DeclaredGraphRuntimeSpec:
    """Return state-vector metadata for the declared graph contract."""

    equation_by_variable: dict[str, Any] = {}
    equation_orders: dict[str, int] = {}
    equation_wrt_by_variable: dict[str, str] = {}
    for equation_name, equation_entry in perturbation_data.equations.items():
        variable_name = str(equation_entry.lhs.variable)
        if variable_name in equation_by_variable:
            previous_name = equation_by_variable[variable_name].name
            raise ValueError(
                "Declared CMB graph defines more than one differential "
                f"equation for variable '{variable_name}' via "
                f"'{previous_name}' and '{equation_name}'"
            )
        equation_by_variable[variable_name] = equation_entry
        equation_orders[variable_name] = int(equation_entry.lhs.order)
        equation_wrt_by_variable[variable_name] = str(equation_entry.lhs.wrt)

    state_slots: list[_DeclaredStateSlot] = []
    state_index_by_key: dict[tuple[str, str, int], int] = {}
    for variable_name in sorted(equation_by_variable):
        order = equation_orders[variable_name]
        variable_wrt = equation_wrt_by_variable[variable_name]
        for derivative_order in range(order):
            index = len(state_slots)
            slot = _DeclaredStateSlot(
                variable=variable_name,
                wrt=variable_wrt,
                order=derivative_order,
                index=index,
            )
            state_slots.append(slot)
            state_index_by_key[
                (variable_name, variable_wrt, derivative_order)
            ] = index

    return _DeclaredGraphRuntimeSpec(
        evolution_variable="eta",
        state_slots=tuple(state_slots),
        state_index_by_key=FrozenMapping(state_index_by_key),
        equation_by_variable=FrozenMapping(equation_by_variable),
        equation_orders=FrozenMapping(equation_orders),
        equation_wrt_by_variable=FrozenMapping(equation_wrt_by_variable),
    )


def _compile_declared_graph_execution_plan(
    perturbation_data: Any,
) -> _DeclaredGraphExecutionPlan:
    """Return the compiled execution plan for one declared graph."""

    cache_token = object.__hash__(perturbation_data)
    cached = native_cache.get_declared_graph_execution_plan(cache_token)
    if cached is not None:
        return cached

    runtime_spec = _prepare_declared_graph_runtime_spec(perturbation_data)
    derivative_steps = tuple(
        _DeclaredDerivativeStep(
            output_name=entry.name,
            variable=str(entry.variable or ""),
            wrt=str(entry.wrt or ""),
            order=int(entry.order or 1),
            slot_name=(
                f"__d{int(entry.order or 1)}_"
                f"{entry.variable}_"
                f"{entry.wrt}"
            ),
        )
        for entry in perturbation_data.derived.values()
        if entry.expression is None
    )
    relation_entries = {
        entry.target: entry for entry in perturbation_data.constraints.values()
    }
    relation_entries.update(
        {entry.target: entry for entry in perturbation_data.closures.values()}
    )
    interaction_entries = getattr(perturbation_data, "interactions", {})
    collision_operator_entries = getattr(
        perturbation_data,
        "collision_operators",
        {},
    )
    value_steps: list[_DeclaredValueStep] = []
    for (
        node_name
    ) in perturbation_data.dependency_graph_summary.evaluation_order:
        derived_entry = perturbation_data.derived.get(node_name)
        if derived_entry is not None and derived_entry.expression is not None:
            value_steps.append(
                _DeclaredValueStep(
                    output_name=node_name,
                    compiled_expression=derived_entry.compiled_expression,
                    dependencies=tuple(derived_entry.dependencies),
                )
            )
            continue
        interaction_entry = interaction_entries.get(node_name)
        if (
            interaction_entry is not None
            and interaction_entry.compiled_expression is not None
        ):
            value_steps.append(
                _DeclaredValueStep(
                    output_name=node_name,
                    compiled_expression=(
                        interaction_entry.compiled_expression
                    ),
                    dependencies=tuple(interaction_entry.dependencies),
                )
            )
            continue
        collision_entry = collision_operator_entries.get(node_name)
        if (
            collision_entry is not None
            and collision_entry.compiled_expression is not None
        ):
            value_steps.append(
                _DeclaredValueStep(
                    output_name=node_name,
                    compiled_expression=collision_entry.compiled_expression,
                    dependencies=tuple(collision_entry.dependencies),
                )
            )
            continue
        relation_entry = relation_entries.get(node_name)
        if relation_entry is None:
            continue
        value_steps.append(
            _DeclaredValueStep(
                output_name=node_name,
                compiled_expression=relation_entry.compiled_expression,
                dependencies=tuple(relation_entry.dependencies),
            )
        )
    source_steps = tuple(
        _DeclaredValueStep(
            output_name=entry.name,
            compiled_expression=entry.compiled_expression,
            dependencies=tuple(entry.dependencies),
        )
        for entry in perturbation_data.sources.values()
    )
    start_condition_entries = tuple(
        sorted(
            tuple(perturbation_data.initial_conditions.values())
            + tuple(
                entry
                for entry in perturbation_data.boundary_conditions.values()
                if str(getattr(entry, "anchor", "start")) == "start"
            ),
            key=lambda entry: (
                str(entry.target.variable),
                str(entry.target.wrt),
                int(entry.target.order),
                str(entry.name),
            ),
        )
    )
    end_condition_entries = tuple(
        sorted(
            (
                entry
                for entry in perturbation_data.boundary_conditions.values()
                if str(getattr(entry, "anchor", "start")) == "end"
            ),
            key=lambda entry: (
                str(entry.target.variable),
                str(entry.target.wrt),
                int(entry.target.order),
                str(entry.name),
            ),
        )
    )
    equation_slot_plans: list[_DeclaredEquationSlotPlan] = []
    for slot in runtime_spec.state_slots:
        promote_from_index = None
        compiled_rhs = None
        equation_name = None
        if slot.order + 1 < runtime_spec.equation_orders[slot.variable]:
            promote_from_index = runtime_spec.state_index_by_key[
                (
                    slot.variable,
                    slot.wrt,
                    slot.order + 1,
                )
            ]
        else:
            equation_entry = runtime_spec.equation_by_variable[slot.variable]
            compiled_rhs = equation_entry.compiled_rhs
            equation_name = equation_entry.name
        equation_slot_plans.append(
            _DeclaredEquationSlotPlan(
                state_index=int(slot.index),
                wrt=str(slot.wrt),
                promote_from_index=promote_from_index,
                compiled_rhs=compiled_rhs,
                equation_name=equation_name,
            )
        )
    compiled_plan = _DeclaredGraphExecutionPlan(
        runtime_spec=runtime_spec,
        derivative_steps=derivative_steps,
        value_steps=tuple(value_steps),
        source_steps=source_steps,
        start_condition_entries=start_condition_entries,
        end_condition_entries=end_condition_entries,
        equation_slot_plans=tuple(equation_slot_plans),
    )
    native_cache.set_declared_graph_execution_plan(cache_token, compiled_plan)
    return compiled_plan


def _resolve_declared_momentum_grid_runtimes(
    perturbation_data: Any,
    *,
    model_parameters: Mapping[str, float],
    physical_params: _CustomCMBPhysicalParameters,
) -> tuple[_DeclaredMomentumGridRuntime, ...]:
    """Return cached quadrature metadata for declared momentum grids."""

    numerics_mapping = getattr(perturbation_data, "numerics", {})
    momentum_grid_defs = numerics_mapping.get("momentum_grids", {})
    if momentum_grid_defs in (None, {}):
        momentum_grid_defs = {}
    if not isinstance(momentum_grid_defs, Mapping):
        raise ValueError(
            "cmb.perturbations.numerics.momentum_grids must be a mapping "
            "when declared momentum-grid families are used."
        )
    accuracy_controls = _resolve_declared_accuracy_controls(
        {"perturbation_data": perturbation_data}
    )
    minimum_momentum_counts = accuracy_controls.get(
        "minimum_momentum_grid_count"
    )
    if not isinstance(minimum_momentum_counts, Mapping):
        minimum_momentum_counts = {}

    family_groups: dict[str, list[str]] = {}
    for (
        family_name,
        family_entry,
    ) in perturbation_data.hierarchy_families.items():
        grid_name = str(family_entry.momentum_grid or "").strip()
        if not grid_name:
            continue
        family_groups.setdefault(grid_name, []).append(str(family_name))
    if not family_groups:
        return ()

    relevant_parameter_names = {
        "num_massive_neutrinos",
        "sum_mnu",
        "mnu",
        "omnuh2",
    }
    for grid_name in family_groups:
        grid_def = momentum_grid_defs.get(grid_name, {})
        if isinstance(grid_def, Mapping):
            parameter_name = grid_def.get("mass_parameter")
            if isinstance(parameter_name, str):
                relevant_parameter_names.add(parameter_name)
    cache_key = (
        tuple(
            sorted(
                (
                    str(name),
                    repr(momentum_grid_defs.get(name, {})),
                    tuple(sorted(family_names)),
                )
                for name, family_names in family_groups.items()
            )
        ),
        tuple(
            sorted(
                (
                    str(name),
                    float(model_parameters[name]),
                )
                for name in relevant_parameter_names
                if name in model_parameters
            )
        ),
        float(physical_params.hubble_ratio),
        float(physical_params.Omega_nu0 or 0.0),
    )
    cached = native_cache.get_declared_momentum_grid(cache_key)
    if cached is not None:
        return cached

    def _grid_mass_eV(grid_name: str, grid_def: Mapping[str, Any]) -> float:
        """Resolve one representative massive-neutrino mass in eV."""

        explicit = grid_def.get("mass_eV")
        if explicit is not None:
            return max(float(explicit), 0.0)
        parameter_name = grid_def.get("mass_parameter")
        if (
            isinstance(parameter_name, str)
            and parameter_name in model_parameters
        ):
            return max(float(model_parameters[parameter_name]), 0.0)
        for candidate in ("sum_mnu", "mnu"):
            if candidate in model_parameters:
                total_mass = max(float(model_parameters[candidate]), 0.0)
                count = max(
                    int(
                        round(
                            float(
                                model_parameters.get(
                                    "num_massive_neutrinos", 1.0
                                )
                            )
                        )
                    ),
                    1,
                )
                return total_mass / float(count)
        if "omnuh2" in model_parameters:
            total_mass = max(float(model_parameters["omnuh2"]), 0.0) * 93.14
            count = max(
                int(
                    round(
                        float(
                            model_parameters.get("num_massive_neutrinos", 1.0)
                        )
                    )
                ),
                1,
            )
            return total_mass / float(count)
        omega_nu0 = physical_params.Omega_nu0
        if omega_nu0 is None:
            return 0.0
        total_mass = (
            max(float(omega_nu0), 0.0)
            * (float(physical_params.hubble_ratio) ** 2)
            * 93.14
        )
        count = max(
            int(
                round(
                    float(model_parameters.get("num_massive_neutrinos", 1.0))
                )
            ),
            1,
        )
        return total_mass / float(count)

    runtimes: list[_DeclaredMomentumGridRuntime] = []
    for grid_name, family_names in sorted(family_groups.items()):
        grid_def = momentum_grid_defs.get(grid_name, {})
        if grid_def in (None, {}):
            grid_def = {}
        if not isinstance(grid_def, Mapping):
            raise ValueError(
                "cmb.perturbations.numerics.momentum_grids."
                f"{grid_name} must be a mapping"
            )
        count = max(int(grid_def.get("count", 8)), 4)
        minimum_count = minimum_momentum_counts.get(grid_name)
        if minimum_count is not None:
            required_count = int(
                _coerce_numeric_scalar(
                    minimum_count,
                    name=(
                        "cmb.perturbations.accuracy_controls."
                        f"minimum_momentum_grid_count.{grid_name}"
                    ),
                )
            )
            if required_count < 1:
                raise ValueError(
                    "Declared accuracy_controls require positive momentum "
                    f"grid counts for '{grid_name}'"
                )
            if count < required_count:
                raise ValueError(
                    "Declared accuracy_controls require "
                    "cmb.perturbations.numerics.momentum_grids."
                    f"{grid_name}.count >= {required_count}"
                )
        q_min = max(float(grid_def.get("q_min", 0.05)), 1.0e-4)
        q_max = max(float(grid_def.get("q_max", 15.0)), q_min * 1.01)
        points = numpy.geomspace(q_min, q_max, count, dtype=float)
        log_points = numpy.log(points)
        weights = numpy.empty_like(points)
        if points.size == 1:
            weights[0] = 1.0
        else:
            deltas = numpy.diff(log_points)
            weights[0] = 0.5 * deltas[0]
            weights[-1] = 0.5 * deltas[-1]
            if points.size > 2:
                weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
        weights = numpy.asarray(weights, dtype=float)
        weights /= max(float(numpy.sum(weights)), 1.0e-12)
        runtimes.append(
            _DeclaredMomentumGridRuntime(
                name=str(grid_name),
                points=points,
                weights=weights,
                mass_eV=_grid_mass_eV(str(grid_name), grid_def),
                family_names=tuple(sorted(str(name) for name in family_names)),
            )
        )
    runtime_tuple = tuple(runtimes)
    native_cache.set_declared_momentum_grid(cache_key, runtime_tuple)
    return runtime_tuple


def _declared_momentum_grid_context(
    perturbation_data: Any,
    *,
    model_parameters: Mapping[str, float],
    physical_params: _CustomCMBPhysicalParameters,
    scale_factor: float | numpy.ndarray,
) -> dict[str, Any]:
    """Return momentum-grid quadrature scalars for one runtime context."""

    runtimes = _resolve_declared_momentum_grid_runtimes(
        perturbation_data,
        model_parameters=model_parameters,
        physical_params=physical_params,
    )
    if not runtimes:
        return {}

    a_values = numpy.asarray(scale_factor, dtype=float)
    context: dict[str, Any] = {}
    for runtime in runtimes:
        mass_term = float(runtime.mass_eV) * a_values
        epsilon = numpy.sqrt(
            numpy.square(runtime.points) + numpy.square(mass_term[..., None])
        )
        q_velocity_ratio = runtime.points / epsilon
        q_pressure_ratio = numpy.square(q_velocity_ratio) / 3.0
        q_mass_fraction = mass_term[..., None] / epsilon
        q_streaming_speed = numpy.asarray(q_velocity_ratio, dtype=float)
        velocity_ratio = numpy.sum(
            runtime.weights * q_velocity_ratio,
            axis=-1,
        )
        pressure_ratio = (
            numpy.sum(
                runtime.weights * numpy.square(q_velocity_ratio), axis=-1
            )
            / 3.0
        )
        mass_fraction = numpy.sum(
            runtime.weights * q_mass_fraction,
            axis=-1,
        )
        prefix = f"momentum_grid_{runtime.name}"
        context[f"{prefix}_points"] = numpy.asarray(
            runtime.points, dtype=float
        )
        context[f"{prefix}_weights"] = numpy.asarray(
            runtime.weights,
            dtype=float,
        )
        context[f"{prefix}_mass_eV"] = float(runtime.mass_eV)
        for name, value in (
            ("velocity_ratio", velocity_ratio),
            ("streaming_speed", velocity_ratio),
            ("pressure_ratio", pressure_ratio),
            ("mass_fraction", mass_fraction),
        ):
            normalized = numpy.asarray(value, dtype=float)
            if normalized.ndim == 0:
                context[f"{prefix}_{name}"] = float(normalized)
            else:
                context[f"{prefix}_{name}"] = normalized
        for index, point in enumerate(runtime.points):
            context[f"{prefix}_q{index}_point"] = float(point)
            context[f"{prefix}_q{index}_weight"] = float(
                runtime.weights[index]
            )
            q_velocity_value = numpy.asarray(
                q_velocity_ratio[..., index],
                dtype=float,
            )
            q_pressure_value = numpy.asarray(
                q_pressure_ratio[..., index],
                dtype=float,
            )
            q_mass_value = numpy.asarray(
                q_mass_fraction[..., index],
                dtype=float,
            )
            q_streaming_value = numpy.asarray(
                q_streaming_speed[..., index],
                dtype=float,
            )
            context[f"{prefix}_q{index}_velocity_ratio"] = (
                float(q_velocity_value)
                if q_velocity_value.ndim == 0
                else q_velocity_value
            )
            context[f"{prefix}_q{index}_streaming_speed"] = (
                float(q_streaming_value)
                if q_streaming_value.ndim == 0
                else q_streaming_value
            )
            context[f"{prefix}_q{index}_pressure_ratio"] = (
                float(q_pressure_value)
                if q_pressure_value.ndim == 0
                else q_pressure_value
            )
            context[f"{prefix}_q{index}_mass_fraction"] = (
                float(q_mass_value) if q_mass_value.ndim == 0 else q_mass_value
            )
        if any(
            "massive_neutrino" in family_name
            for family_name in runtime.family_names
        ):
            for name in (
                "mass_eV",
                "velocity_ratio",
                "streaming_speed",
                "pressure_ratio",
                "mass_fraction",
            ):
                context[f"massive_neutrino_{name}"] = context[
                    f"{prefix}_{name}"
                ]
            for index in range(runtime.points.size):
                for name in (
                    "point",
                    "weight",
                    "velocity_ratio",
                    "streaming_speed",
                    "pressure_ratio",
                    "mass_fraction",
                ):
                    context[f"massive_neutrino_q{index}_{name}"] = context[
                        f"{prefix}_q{index}_{name}"
                    ]
    return context


def _declared_runtime_seed(
    *,
    k_value: float,
    physical_params: _CustomCMBPhysicalParameters,
    model_parameters: Mapping[str, float],
) -> float:
    """Return the declared-graph initial-condition normalization."""

    del k_value
    del physical_params
    for parameter_name in ("seed", "primordial_seed", "transfer_seed"):
        if parameter_name not in model_parameters:
            continue
        return _coerce_numeric_scalar(
            model_parameters[parameter_name],
            name=parameter_name,
        )
    # Keep transfer functions unit-normalized unless the contract declares
    # an explicit seed for its initial conditions.
    return 1.0


def _build_declared_base_context(
    *,
    perturbation_data: Any,
    model_parameters: Mapping[str, float],
    physical_params: _CustomCMBPhysicalParameters,
    numerics: _CustomCMBNumerics,
    k_value: float,
    eta_value: float,
    background_scalars: Mapping[str, float],
) -> dict[str, Any]:
    """Return scalar runtime values shared by equations and conditions."""

    tight_coupling_drag = _compute_tight_coupling_drag(
        collision_rate=float(background_scalars["collision_rate"]),
        k_value=float(k_value),
        tight_coupling_ratio=float(numerics.tight_coupling_ratio),
    )
    context: dict[str, Any] = dict(model_parameters)
    context.update(background_scalars)
    context["k"] = float(k_value)
    context["seed"] = _declared_runtime_seed(
        k_value=float(k_value),
        physical_params=physical_params,
        model_parameters=model_parameters,
    )
    context["a_initial"] = float(background_scalars["a"])
    context["eta_initial"] = float(eta_value)
    for name, value in _physical_runtime_scalars(physical_params).items():
        context.setdefault(name, float(value))
    context["sound_horizon"] = float(background_scalars["sound_horizon"])
    context["sound_speed_sq"] = float(background_scalars["sound_speed_sq"])
    context["collision_rate"] = float(background_scalars["collision_rate"])
    context["free_streaming"] = float(background_scalars["free_streaming"])
    context["tight_coupling_drag"] = float(tight_coupling_drag)
    context["tight_coupling_ratio"] = float(numerics.tight_coupling_ratio)
    context.update(
        _declared_momentum_grid_context(
            perturbation_data,
            model_parameters=model_parameters,
            physical_params=physical_params,
            scale_factor=float(background_scalars["a"]),
        )
    )
    return context


def _compute_tight_coupling_drag(
    *,
    collision_rate: float | numpy.ndarray,
    k_value: float,
    tight_coupling_ratio: float,
) -> float | numpy.ndarray:
    """Return the diagnostic tight-coupling rate for native contexts."""

    tight_coupling_cap = _tight_coupling_entry_rate(
        k_value=float(k_value),
        tight_coupling_ratio=float(tight_coupling_ratio),
    )
    collision_rate_array = numpy.asarray(collision_rate, dtype=float)
    drag = collision_rate_array / (
        1.0 + collision_rate_array / tight_coupling_cap
    )
    if drag.ndim == 0:
        return float(drag)
    return drag


def _tight_coupling_entry_rate(
    *,
    k_value: float,
    tight_coupling_ratio: float,
) -> float:
    """Return the collision rate that activates tight coupling."""

    return max(
        float(k_value) * float(tight_coupling_ratio),
        1.0e-12,
    )


def _tight_coupling_exit_rate(
    *,
    k_value: float,
    tight_coupling_ratio: float,
) -> float:
    """Return the collision rate below which tight coupling is disabled."""

    return 0.1 * _tight_coupling_entry_rate(
        k_value=k_value,
        tight_coupling_ratio=tight_coupling_ratio,
    )


def _tight_coupling_is_active(
    *,
    active: bool,
    collision_rate: float,
    k_value: float,
    tight_coupling_ratio: float,
) -> bool:
    """Return the updated tight-coupling regime with hysteresis."""

    if not numpy.isfinite(collision_rate) or collision_rate <= 0.0:
        return False
    if active:
        return collision_rate > _tight_coupling_exit_rate(
            k_value=k_value,
            tight_coupling_ratio=tight_coupling_ratio,
        )
    return collision_rate >= _tight_coupling_entry_rate(
        k_value=k_value,
        tight_coupling_ratio=tight_coupling_ratio,
    )


def _exact_thomson_relaxation_step(
    *,
    theta_gamma1: float,
    theta_b: float,
    theta_gamma2: float,
    e_gamma2: float,
    collision_rate: float,
    baryon_loading: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Return the exact collision-only Thomson update for one sub-step."""

    if dt == 0.0 or collision_rate <= 0.0:
        return theta_gamma1, theta_b, theta_gamma2, e_gamma2
    photon_baryon_ratio = 1.0 / max(float(baryon_loading), 1.0e-12)
    collision_strength = float(collision_rate) * float(dt)

    dipole_mode = float(theta_gamma1) - float(theta_b) / 3.0
    momentum_mode = float(theta_b) + photon_baryon_ratio * float(theta_gamma1)
    dipole_decay = numpy.exp(
        -collision_strength * (1.0 + photon_baryon_ratio / 3.0)
    )
    relaxed_theta_gamma1 = (
        momentum_mode + 3.0 * dipole_mode * dipole_decay
    ) / (3.0 + photon_baryon_ratio)
    relaxed_theta_b = momentum_mode - (
        photon_baryon_ratio * relaxed_theta_gamma1
    )

    fast_mode = float(theta_gamma2) - float(e_gamma2)
    slow_mode = float(theta_gamma2) + 6.0 * float(e_gamma2)
    fast_decay = numpy.exp(-collision_strength)
    slow_decay = numpy.exp(-0.3 * collision_strength)
    relaxed_theta_gamma2 = (
        slow_mode * slow_decay + 6.0 * fast_mode * fast_decay
    ) / 7.0
    relaxed_e_gamma2 = (slow_mode * slow_decay - fast_mode * fast_decay) / 7.0
    return (
        float(relaxed_theta_gamma1),
        float(relaxed_theta_b),
        float(relaxed_theta_gamma2),
        float(relaxed_e_gamma2),
    )


def _resolve_declared_graph_context(
    context: dict[str, Any],
    perturbation_data: Any,
    *,
    allow_partial: bool = False,
    eta_grid: numpy.ndarray | None,
    execution_plan: _DeclaredGraphExecutionPlan | None,
) -> dict[str, Any]:
    """Resolve derivative symbols, derived expressions, and relations."""

    if execution_plan is None:
        execution_plan = _compile_declared_graph_execution_plan(
            perturbation_data
        )
    runtime_spec = execution_plan.runtime_spec

    pending_derivatives = list(execution_plan.derivative_steps)
    pending_values = list(execution_plan.value_steps)
    while pending_derivatives or pending_values:
        progress = False
        next_derivatives: list[_DeclaredDerivativeStep] = []
        for step in pending_derivatives:
            target_name = step.variable
            if target_name not in context:
                next_derivatives.append(step)
                continue
            target_value = context[target_name]
            derivative_order = int(step.order)
            if eta_grid is None:
                slot_index = runtime_spec.state_index_by_key.get(
                    (target_name, step.wrt, derivative_order)
                )
                if slot_index is None or step.slot_name not in context:
                    next_derivatives.append(step)
                    continue
                context[step.output_name] = context[step.slot_name]
                progress = True
                continue
            coordinate_name = str(step.wrt or runtime_spec.evolution_variable)
            derivative_value = numpy.asarray(target_value, dtype=float)
            if coordinate_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
                coordinate_history = numpy.asarray(eta_grid, dtype=float)
            else:
                if coordinate_name not in context:
                    next_derivatives.append(step)
                    continue
                coordinate_history = numpy.asarray(
                    context[coordinate_name],
                    dtype=float,
                )
                if coordinate_history.ndim == 0:
                    coordinate_history = numpy.full_like(
                        eta_grid,
                        float(coordinate_history),
                        dtype=float,
                    )
                if coordinate_history.shape != eta_grid.shape:
                    raise ValueError(
                        "Declared coordinate history must match the eta "
                        f"grid for derivative symbol '{step.output_name}'."
                    )
            for _ in range(derivative_order):
                derivative_eta = numpy.asarray(
                    numpy.gradient(
                        derivative_value,
                        eta_grid,
                        edge_order=1,
                    ),
                    dtype=float,
                )
                if coordinate_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
                    derivative_value = derivative_eta
                    continue
                coordinate_rate = numpy.asarray(
                    numpy.gradient(
                        coordinate_history,
                        eta_grid,
                        edge_order=1,
                    ),
                    dtype=float,
                )
                if not numpy.all(numpy.isfinite(coordinate_rate)):
                    raise ValueError(
                        "Declared coordinate history produced non-finite "
                        f"rates for derivative symbol '{step.output_name}'."
                    )
                if numpy.any(numpy.abs(coordinate_rate) <= 1.0e-12):
                    raise ValueError(
                        "Declared coordinate history is singular for "
                        f"derivative symbol '{step.output_name}'."
                    )
                derivative_value = derivative_eta / coordinate_rate
            context[step.output_name] = derivative_value
            progress = True

        next_values: list[_DeclaredValueStep] = []
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for step in pending_values:
                missing = [
                    dependency
                    for dependency in step.dependencies
                    if dependency not in context
                ]
                if missing:
                    next_values.append(step)
                    continue
                context[step.output_name] = (
                    _evaluate_compiled_expression_noerr(
                        step.compiled_expression,
                        context,
                    )
                )
                progress = True

        if progress:
            pending_derivatives = next_derivatives
            pending_values = next_values
            continue
        if allow_partial:
            return context
        pending_names = sorted(
            [step.output_name for step in next_derivatives]
            + [step.output_name for step in next_values]
        )
        pending_str = ", ".join(pending_names)
        raise ValueError(
            "Declared CMB graph references unresolved symbol(s): "
            f"{pending_str}"
        )
    return context


def _evaluate_declared_initial_state(
    *,
    perturbation_data: Any,
    execution_plan: _DeclaredGraphExecutionPlan,
    base_context: Mapping[str, Any],
) -> tuple[numpy.ndarray, tuple[tuple[str, str, int], ...]]:
    """Return the initial state vector for one Fourier mode."""

    runtime_spec = execution_plan.runtime_spec
    state_vector = numpy.zeros(len(runtime_spec.state_slots), dtype=float)
    assigned_targets: list[tuple[str, str, int]] = []
    context = dict(base_context)
    condition_entries = execution_plan.start_condition_entries
    pending = list(condition_entries)
    while pending:
        context = _resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
        )
        progress = False
        next_round: list[Any] = []
        for entry in pending:
            missing = [
                dependency
                for dependency in entry.dependencies
                if dependency not in context
            ]
            if missing:
                next_round.append(entry)
                continue
            value = _coerce_numeric_scalar(
                evaluate_compiled_expression(
                    entry.compiled_expression,
                    context,
                ),
                name=f"condition '{entry.name}'",
            )
            state_index = runtime_spec.state_index_by_key[
                (
                    str(entry.target.variable),
                    str(entry.target.wrt),
                    int(entry.target.order),
                )
            ]
            state_vector[state_index] = value
            assigned_targets.append(
                (
                    str(entry.target.variable),
                    str(entry.target.wrt),
                    int(entry.target.order),
                )
            )
            if int(entry.target.order) == 0:
                context[str(entry.target.variable)] = value
            else:
                context[
                    "__d"
                    f"{int(entry.target.order)}_"
                    f"{entry.target.variable}_"
                    f"{entry.target.wrt}"
                ] = value
            progress = True
        if not progress and next_round:
            pending_names = ", ".join(entry.name for entry in next_round)
            raise ValueError(
                "Declared CMB start conditions could not be resolved: "
                f"{pending_names}"
            )
        pending = next_round
    _resolve_declared_graph_context(
        context,
        perturbation_data,
        allow_partial=True,
        eta_grid=None,
        execution_plan=execution_plan,
    )
    return state_vector, tuple(assigned_targets)
