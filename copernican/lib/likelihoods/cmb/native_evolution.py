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
    return context


def _compute_tight_coupling_drag(
    *,
    collision_rate: float | numpy.ndarray,
    k_value: float,
    tight_coupling_ratio: float,
) -> float | numpy.ndarray:
    """Return the capped collision rate used by declared CMB graphs."""

    tight_coupling_cap = max(
        float(k_value) * float(tight_coupling_ratio),
        1.0e-12,
    )
    collision_rate_array = numpy.asarray(collision_rate, dtype=float)
    drag = collision_rate_array / (
        1.0 + collision_rate_array / tight_coupling_cap
    )
    if drag.ndim == 0:
        return float(drag)
    return drag


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
