r"""Declared native perturbation compilation and graph-evolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy

from ...engine_adapter import FrozenMapping, _freeze_for_cache
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

_NEUTRINO_TEMPERATURE_EV_PER_K = (4.0 / 11.0) ** (
    1.0 / 3.0
) * 8.617_333_262_145e-5
_NEUTRINO_DENSITY_EV_H2 = 93.14


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


def _thermal_fermi_dirac_distribution(
    q_points: numpy.ndarray,
) -> numpy.ndarray:
    """Return the thermal Fermi-Dirac occupation for one q grid."""

    exp_neg_q = numpy.exp(-numpy.asarray(q_points, dtype=float))
    return numpy.asarray(exp_neg_q / (1.0 + exp_neg_q), dtype=float)


def _normalize_declared_momentum_weights(
    raw_weights: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return normalized weights and their summed physical moments."""

    raw = numpy.asarray(raw_weights, dtype=float)
    totals = numpy.sum(raw, axis=-1, keepdims=True)
    safe_totals = numpy.where(
        numpy.abs(totals) > 1.0e-300,
        totals,
        1.0e-300,
    )
    normalized = numpy.asarray(raw / safe_totals, dtype=float)
    return normalized, numpy.asarray(
        numpy.squeeze(totals, axis=-1),
        dtype=float,
    )


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


def _declared_graph_execution_plan_cache_token(
    perturbation_data: Any,
) -> Any:
    """Return one structural cache token for a declared execution plan."""

    equation_entries = getattr(perturbation_data, "equations", {}) or {}
    derived_entries = getattr(perturbation_data, "derived", {}) or {}
    constraint_entries = getattr(perturbation_data, "constraints", {}) or {}
    closure_entries = getattr(perturbation_data, "closures", {}) or {}
    interaction_entries = getattr(perturbation_data, "interactions", {}) or {}
    source_entries = getattr(perturbation_data, "sources", {}) or {}
    initial_condition_entries = (
        getattr(perturbation_data, "initial_conditions", {}) or {}
    )
    boundary_condition_entries = (
        getattr(perturbation_data, "boundary_conditions", {}) or {}
    )
    collision_entries = (
        getattr(perturbation_data, "collision_operators", {}) or {}
    )
    dependency_graph = getattr(
        perturbation_data,
        "dependency_graph_summary",
        None,
    )
    structural_view = {
        "backend": getattr(perturbation_data, "backend", ""),
        "closures": {
            name: (
                getattr(entry, "target", None),
                getattr(entry, "expression", None),
            )
            for name, entry in closure_entries.items()
        },
        "collisions": {
            name: (
                getattr(entry, "expression", None),
                getattr(entry, "counterpart", None),
                getattr(entry, "integration_strategy", None),
                getattr(entry, "rate_expression", None),
            )
            for name, entry in collision_entries.items()
        },
        "constraints": {
            name: (
                getattr(entry, "target", None),
                getattr(entry, "expression", None),
            )
            for name, entry in constraint_entries.items()
        },
        "derived": {
            name: (
                getattr(entry, "expression", None),
                getattr(entry, "variable", None),
                getattr(entry, "wrt", None),
                getattr(entry, "order", None),
            )
            for name, entry in derived_entries.items()
        },
        "equations": {
            name: (
                getattr(entry.lhs, "variable", None),
                getattr(entry.lhs, "wrt", None),
                getattr(entry.lhs, "order", None),
                getattr(entry, "rhs", None),
            )
            for name, entry in equation_entries.items()
        },
        "evaluation_order": (
            ()
            if dependency_graph is None
            else tuple(dependency_graph.evaluation_order)
        ),
        "gauge": getattr(perturbation_data, "gauge", ""),
        "initial_conditions": {
            name: (
                getattr(entry.target, "variable", None),
                getattr(entry.target, "wrt", None),
                getattr(entry.target, "order", None),
                getattr(entry, "expression", None),
                getattr(entry, "anchor", None),
            )
            for name, entry in initial_condition_entries.items()
        },
        "interactions": {
            name: (
                getattr(entry, "expression", None),
                getattr(entry, "counterpart", None),
            )
            for name, entry in interaction_entries.items()
        },
        "model_name": getattr(perturbation_data, "model_name", ""),
        "boundary_conditions": {
            name: (
                getattr(entry.target, "variable", None),
                getattr(entry.target, "wrt", None),
                getattr(entry.target, "order", None),
                getattr(entry, "expression", None),
                getattr(entry, "anchor", None),
            )
            for name, entry in boundary_condition_entries.items()
        },
        "sources": {
            name: (
                getattr(entry, "expression", None),
                getattr(entry, "role", None),
            )
            for name, entry in source_entries.items()
        },
    }
    return _freeze_for_cache(structural_view)


def _compile_declared_graph_execution_plan(
    perturbation_data: Any,
) -> _DeclaredGraphExecutionPlan:
    """Return the compiled execution plan for one declared graph."""

    cache_token = _declared_graph_execution_plan_cache_token(perturbation_data)
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
            total_mass = max(float(model_parameters[parameter_name]), 0.0)
            if parameter_name in {"sum_mnu", "mnu"}:
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
            return total_mass
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

    def _context_value(value: Any) -> Any:
        """Return ``value`` as a scalar or float array for the context."""

        array_value = numpy.asarray(value, dtype=float)
        if array_value.ndim == 0:
            return float(array_value)
        return array_value

    neutrino_temperature_eV = (
        float(physical_params.Tcmb_K) * _NEUTRINO_TEMPERATURE_EV_PER_K
    )
    neutrino_temperature_eV = max(neutrino_temperature_eV, 1.0e-12)
    context["neutrino_temperature_eV"] = neutrino_temperature_eV

    def _declared_total_mass_eV(
        runtime: _DeclaredMomentumGridRuntime,
    ) -> float:
        """Return the total thermal mass represented by one q family."""

        for parameter_name in ("sum_mnu", "mnu"):
            if parameter_name in model_parameters:
                return max(float(model_parameters[parameter_name]), 0.0)
        count = max(
            int(
                round(float(model_parameters.get("num_massive_neutrinos", 1)))
            ),
            1,
        )
        return max(float(runtime.mass_eV) * count, 0.0)

    def _massive_neutrino_omega0(
        runtime: _DeclaredMomentumGridRuntime,
        total_mass_eV: float,
    ) -> float:
        """Return the present massive-neutrino density fraction."""

        del runtime

        if total_mass_eV > 0.0:
            return total_mass_eV / (
                _NEUTRINO_DENSITY_EV_H2
                * max(float(physical_params.hubble_ratio) ** 2, 1.0e-30)
            )
        neutrino_count = max(
            int(
                round(float(model_parameters.get("num_massive_neutrinos", 1)))
            ),
            0,
        )
        effective_neff = max(float(physical_params.Neff or 0.0), 1.0e-30)
        return max(float(physical_params.Omega_nu0 or 0.0), 0.0) * (
            float(neutrino_count) / effective_neff
        )

    for runtime in runtimes:
        thermal_distribution = _thermal_fermi_dirac_distribution(
            runtime.points
        )
        quadrature_weights = numpy.asarray(
            runtime.weights * thermal_distribution,
            dtype=float,
        )
        mass_ratio_today = float(runtime.mass_eV) / neutrino_temperature_eV
        mass_term = mass_ratio_today * a_values
        epsilon = numpy.sqrt(
            numpy.square(runtime.points) + numpy.square(mass_term[..., None])
        )
        q_velocity_ratio = runtime.points / epsilon
        q_pressure_ratio = numpy.square(q_velocity_ratio) / 3.0
        q_mass_fraction = mass_term[..., None] / epsilon
        q_streaming_speed = numpy.asarray(q_velocity_ratio, dtype=float)
        density_weight_raw = (
            quadrature_weights * numpy.power(runtime.points, 3.0) * epsilon
        )
        pressure_weight_raw = (
            quadrature_weights
            * numpy.power(runtime.points, 5.0)
            / (3.0 * epsilon)
        )
        momentum_weight_raw = quadrature_weights * numpy.power(
            runtime.points, 4.0
        )
        shear_weight_raw = (
            quadrature_weights * numpy.power(runtime.points, 5.0) / epsilon
        )
        density_weights, background_density_moment = (
            _normalize_declared_momentum_weights(density_weight_raw)
        )
        pressure_weights, background_pressure_moment = (
            _normalize_declared_momentum_weights(pressure_weight_raw)
        )
        momentum_weights, background_momentum_moment = (
            _normalize_declared_momentum_weights(momentum_weight_raw)
        )
        shear_weights, background_shear_moment = (
            _normalize_declared_momentum_weights(shear_weight_raw)
        )
        total_mass_eV = _declared_total_mass_eV(runtime)
        massive_omega0 = _massive_neutrino_omega0(
            runtime,
            total_mass_eV,
        )
        epsilon_today = numpy.sqrt(
            numpy.square(runtime.points) + mass_ratio_today * mass_ratio_today
        )
        density_moment_today = numpy.sum(
            quadrature_weights
            * numpy.power(runtime.points, 3.0)
            * epsilon_today
        )
        density_moment_today = max(float(density_moment_today), 1.0e-300)
        scale_factor_array = numpy.maximum(a_values, 1.0e-30)
        density_fraction = (
            massive_omega0
            * numpy.power(scale_factor_array, -4.0)
            * background_density_moment
            / density_moment_today
        )
        pressure_fraction = (
            massive_omega0
            * numpy.power(scale_factor_array, -4.0)
            * background_pressure_moment
            / density_moment_today
        )
        momentum_fraction = (
            massive_omega0
            * numpy.power(scale_factor_array, -4.0)
            * background_momentum_moment
            / density_moment_today
        )
        shear_fraction = (
            massive_omega0
            * numpy.power(scale_factor_array, -4.0)
            * background_shear_moment
            / density_moment_today
        )
        velocity_ratio = numpy.sum(
            momentum_weights * q_velocity_ratio,
            axis=-1,
        )
        pressure_ratio = numpy.divide(
            background_pressure_moment,
            numpy.maximum(background_density_moment, 1.0e-300),
        )
        mass_fraction = numpy.sum(
            density_weights * q_mass_fraction,
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
        context[f"{prefix}_distribution_weights"] = quadrature_weights
        context[f"{prefix}_mass_eV"] = float(runtime.mass_eV)
        context[f"{prefix}_background_density_moment"] = _context_value(
            background_density_moment
        )
        context[f"{prefix}_background_pressure_moment"] = _context_value(
            background_pressure_moment
        )
        context[f"{prefix}_background_momentum_moment"] = _context_value(
            background_momentum_moment
        )
        context[f"{prefix}_background_shear_moment"] = _context_value(
            background_shear_moment
        )
        for name, value in (
            ("velocity_ratio", velocity_ratio),
            ("streaming_speed", velocity_ratio),
            ("pressure_ratio", pressure_ratio),
            ("mass_fraction", mass_fraction),
            ("density_fraction", density_fraction),
            ("pressure_fraction", pressure_fraction),
            ("momentum_fraction", momentum_fraction),
            ("shear_fraction", shear_fraction),
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
            context[f"{prefix}_q{index}_distribution_weight"] = float(
                quadrature_weights[index]
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
            q_density_weight = numpy.asarray(
                density_weights[..., index],
                dtype=float,
            )
            q_pressure_weight = numpy.asarray(
                pressure_weights[..., index],
                dtype=float,
            )
            q_momentum_weight = numpy.asarray(
                momentum_weights[..., index],
                dtype=float,
            )
            q_shear_weight = numpy.asarray(
                shear_weights[..., index],
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
            context[f"{prefix}_q{index}_density_weight"] = (
                float(q_density_weight)
                if q_density_weight.ndim == 0
                else q_density_weight
            )
            context[f"{prefix}_q{index}_momentum_weight"] = (
                float(q_momentum_weight)
                if q_momentum_weight.ndim == 0
                else q_momentum_weight
            )
            context[f"{prefix}_q{index}_pressure_weight"] = (
                float(q_pressure_weight)
                if q_pressure_weight.ndim == 0
                else q_pressure_weight
            )
            context[f"{prefix}_q{index}_shear_weight"] = (
                float(q_shear_weight)
                if q_shear_weight.ndim == 0
                else q_shear_weight
            )
        if any(
            "massive_neutrino" in family_name
            for family_name in runtime.family_names
        ):
            for name in (
                "mass_eV",
                "background_density_moment",
                "background_pressure_moment",
                "background_momentum_moment",
                "background_shear_moment",
                "velocity_ratio",
                "streaming_speed",
                "pressure_ratio",
                "mass_fraction",
                "density_fraction",
                "pressure_fraction",
                "momentum_fraction",
                "shear_fraction",
            ):
                context[f"massive_neutrino_{name}"] = context[
                    f"{prefix}_{name}"
                ]
            for index in range(runtime.points.size):
                for name in (
                    "point",
                    "weight",
                    "distribution_weight",
                    "velocity_ratio",
                    "streaming_speed",
                    "pressure_ratio",
                    "mass_fraction",
                    "density_weight",
                    "momentum_weight",
                    "pressure_weight",
                    "shear_weight",
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


def _nonuniform_gradient(
    values: numpy.ndarray,
    grid: numpy.ndarray,
) -> numpy.ndarray:
    """Return a three-point derivative on a strictly increasing grid."""

    samples = numpy.asarray(values, dtype=float)
    coordinates = numpy.asarray(grid, dtype=float)
    if samples.ndim != 1 or coordinates.ndim != 1:
        raise ValueError("Nonuniform derivatives require one-dimensional data")
    if samples.size != coordinates.size or samples.size < 2:
        raise ValueError("Nonuniform derivative data must have two samples")
    steps = numpy.diff(coordinates)
    if (
        not numpy.all(numpy.isfinite(samples))
        or not numpy.all(numpy.isfinite(coordinates))
        or numpy.any(steps <= 0.0)
    ):
        raise ValueError(
            "Nonuniform derivative grid must be finite and ordered"
        )
    if samples.size == 2:
        slope = (samples[1] - samples[0]) / steps[0]
        return numpy.asarray((slope, slope), dtype=float)

    derivative = numpy.empty_like(samples, dtype=float)
    left_step = float(steps[0])
    right_step = float(steps[1])
    left_span = left_step + right_step
    derivative[0] = (
        -(2.0 * left_step + right_step) * samples[0] / (left_step * left_span)
        + left_span * samples[1] / (left_step * right_step)
        - left_step * samples[2] / (right_step * left_span)
    )
    for index in range(1, samples.size - 1):
        left_step = float(steps[index - 1])
        right_step = float(steps[index])
        span = left_step + right_step
        derivative[index] = (
            -right_step * samples[index - 1] / (left_step * span)
            + (right_step - left_step)
            * samples[index]
            / (left_step * right_step)
            + left_step * samples[index + 1] / (right_step * span)
        )
    left_step = float(steps[-2])
    right_step = float(steps[-1])
    right_span = left_step + right_step
    derivative[-1] = (
        right_step * samples[-3] / (left_step * right_span)
        - right_span * samples[-2] / (left_step * right_step)
        + (left_step + 2.0 * right_step)
        * samples[-1]
        / (right_step * right_span)
    )
    return derivative


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


def _resolve_declared_graph_context(
    context: dict[str, Any],
    perturbation_data: Any,
    *,
    allow_partial: bool = False,
    eta_grid: numpy.ndarray | None,
    execution_plan: _DeclaredGraphExecutionPlan | None,
    suppressed_outputs: Mapping[str, Any] | None = None,
    required_names: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve derivative symbols, derived expressions, and relations."""

    if execution_plan is None:
        execution_plan = _compile_declared_graph_execution_plan(
            perturbation_data
        )
    runtime_spec = execution_plan.runtime_spec

    if required_names is not None and eta_grid is None:
        unresolved = False
        for step in execution_plan.derivative_steps:
            if step.output_name not in required_names:
                continue
            if step.output_name in context:
                continue
            slot_index = runtime_spec.state_index_by_key.get(
                (step.variable, step.wrt, int(step.order))
            )
            if slot_index is None or step.slot_name not in context:
                unresolved = True
                continue
            context[step.output_name] = context[step.slot_name]
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for step in execution_plan.value_steps:
                if step.output_name not in required_names:
                    continue
                if (
                    suppressed_outputs is not None
                    and step.output_name in suppressed_outputs
                ):
                    context[step.output_name] = suppressed_outputs[
                        step.output_name
                    ]
                    continue
                if step.output_name in context:
                    continue
                if any(
                    dependency not in context
                    for dependency in step.dependencies
                ):
                    unresolved = True
                    break
                context[step.output_name] = (
                    _evaluate_compiled_expression_noerr(
                        step.compiled_expression,
                        context,
                    )
                )
        if not unresolved:
            return context

    pending_derivatives = list(execution_plan.derivative_steps)
    pending_values = list(execution_plan.value_steps)
    while pending_derivatives or pending_values:
        progress = False
        next_derivatives: list[_DeclaredDerivativeStep] = []
        for step in pending_derivatives:
            if (
                required_names is not None
                and step.output_name not in required_names
            ):
                continue
            target_name = step.variable
            if step.output_name in context:
                continue
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
                derivative_eta = _nonuniform_gradient(
                    derivative_value,
                    eta_grid,
                )
                if coordinate_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
                    derivative_value = derivative_eta
                    continue
                coordinate_rate = _nonuniform_gradient(
                    coordinate_history,
                    eta_grid,
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
                if (
                    required_names is not None
                    and step.output_name not in required_names
                ):
                    continue
                if (
                    suppressed_outputs is not None
                    and step.output_name in suppressed_outputs
                ):
                    context[step.output_name] = suppressed_outputs[
                        step.output_name
                    ]
                    progress = True
                    continue
                if step.output_name in context:
                    continue
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


def _validate_generated_scalar_initial_constraints(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    k_value: float,
) -> None:
    """Raise when generated scalar data violate Einstein constraints."""

    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if not manifest_summary.get("generated_scalar_hierarchy"):
        return

    residual_specs = (
        (
            "einstein_energy_residual",
            max(
                abs(float(context.get("acoustic_k_sq", 0.0) * context["Phi"])),
                abs(
                    float(
                        1.5
                        * context["einstein_gravity_strength"]
                        * context["total_density_source"]
                    )
                ),
                1.0,
            ),
        ),
        (
            "einstein_momentum_residual",
            max(
                abs(
                    float(
                        context["acoustic_k_sq"]
                        * context["metric_momentum_constraint"]
                    )
                ),
                abs(
                    float(
                        1.5
                        * context["einstein_gravity_strength"]
                        * context["total_momentum_source"]
                    )
                ),
                1.0,
            ),
        ),
        (
            "einstein_shear_residual",
            max(
                abs(
                    float(
                        context["acoustic_k_sq"]
                        * (context["Phi"] - context["Psi"])
                    )
                ),
                abs(
                    float(
                        4.5
                        * context["einstein_gravity_strength"]
                        * context["total_shear_source"]
                    )
                ),
                1.0,
            ),
        ),
    )
    tolerance = (
        5.0e-2
        if str(getattr(perturbation_data, "gauge", "")) == "synchronous"
        else 1.0e-2
    )
    for residual_name, scale in residual_specs:
        if residual_name not in context:
            continue
        normalized_residual = abs(float(context[residual_name])) / float(scale)
        if not numpy.isfinite(normalized_residual):
            raise ValueError(
                "Generated scalar initial data produced non-finite Einstein "
                f"diagnostics for {residual_name} at k={k_value}"
            )
        if normalized_residual > tolerance:
            raise ValueError(
                "Generated scalar initial data violate the Einstein "
                f"constraints for {residual_name} at k={k_value} "
                f"({normalized_residual} > {tolerance})"
            )


def _validate_generated_vector_initial_constraints(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    k_value: float,
) -> None:
    """Raise when generated vector initial data violate their constraint.

    The vector momentum constraint is numerically cancellation-dominated at
    early times because the regular photon and neutrino heat fluxes nearly
    cancel. The robust diagnostic is therefore the residual on the underlying
    momentum-source surface rather than the ratio between two nearly cancelled
    sigma amplitudes.
    """

    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if not manifest_summary.get("generated_vector_hierarchy"):
        return
    residual_name = "vector_einstein_momentum_residual"
    required_names = {
        "a",
        "acoustic_k_sq",
        "einstein_gravity_strength",
        "q_gamma_vector",
        "q_nu_vector",
        "sigma_vector",
        "v_b_vector",
        "vector_neutrino_density",
        "vector_total_momentum_source",
        "Omega_b0",
        "Omega_gamma0",
    }
    if residual_name not in context or not required_names.issubset(context):
        return

    scale_factor = max(abs(float(context["a"])), 1.0e-30)
    radiation_scale_factor = scale_factor * scale_factor
    source_scale = (
        abs(float(context["Omega_b0"]) * float(context["v_b_vector"]))
        / scale_factor
        + abs(
            float(context["Omega_gamma0"]) * float(context["q_gamma_vector"])
        )
        / radiation_scale_factor
        + abs(
            float(context["vector_neutrino_density"])
            * float(context["q_nu_vector"])
        )
        / radiation_scale_factor
    )
    if "Omega_c0" in context and "v_c_vector" in context:
        source_scale += (
            abs(float(context["Omega_c0"]) * float(context["v_c_vector"]))
            / scale_factor
        )

    gravity_strength = max(
        abs(float(context["einstein_gravity_strength"])),
        1.0e-30,
    )
    constrained_momentum_source = (
        float(context["acoustic_k_sq"]) * float(context["sigma_vector"])
    ) / (6.0 * gravity_strength)
    residual_source = abs(
        float(context["vector_total_momentum_source"])
        - constrained_momentum_source
    )
    normalized_residual = residual_source / max(source_scale, 1.0)
    if not numpy.isfinite(normalized_residual):
        raise ValueError(
            "Generated vector initial data produced non-finite Einstein "
            f"diagnostics for {residual_name} at k={k_value}"
        )
    tolerance = 2.0e-2
    if normalized_residual > tolerance:
        raise ValueError(
            "Generated vector initial data violate the Einstein "
            f"constraint for {residual_name} at k={k_value} "
            f"({normalized_residual} > {tolerance})"
        )
