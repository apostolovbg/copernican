r"""Declared native perturbation compilation and graph-evolution helpers."""

from __future__ import annotations

import ast
import copy
import keyword
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping

import numpy

from ...model_adapter import FrozenMapping, _freeze_for_cache
from ...perturbation_contract import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _evaluate_compiled_expression_noerr,
    _parse_safe_expression,
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
from .native_errors import (
    NativeConstraintViolationError,
    NativeNonFiniteEvolutionError,
)

_NEUTRINO_TEMPERATURE_EV_PER_K = (4.0 / 11.0) ** (
    1.0 / 3.0
) * 8.617_333_262_145e-5
_NEUTRINO_DENSITY_EV_H2 = 93.14
_COMPILED_CONTEXT_GLOBALS = {
    "__builtins__": {},
    **_ALLOWED_CONSTANTS,
    **_ALLOWED_MATH_FUNCS,
}


class _ContextNameRewriter(ast.NodeTransformer):
    """Rewrite declared symbols to direct context lookups."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Keep approved globals and map runtime names to context."""

        if node.id in _ALLOWED_CONSTANTS or node.id in _ALLOWED_MATH_FUNCS:
            return node
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id="context", ctx=ast.Load()),
                slice=ast.Constant(node.id),
                ctx=ast.Load(),
            ),
            node,
        )


class _ContextAliasRewriter(ast.NodeTransformer):
    """Rewrite declared symbols to preloaded private local names."""

    def __init__(self, aliases: Mapping[str, str]) -> None:
        """Record the local alias assigned to each declared input."""

        self._aliases = dict(aliases)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Keep approved globals and replace context names with locals."""

        if node.id in _ALLOWED_CONSTANTS or node.id in _ALLOWED_MATH_FUNCS:
            return node
        return ast.copy_location(
            ast.Name(id=self._aliases[node.id], ctx=node.ctx),
            node,
        )


def _private_local_aliases(
    names: set[str],
    *,
    prefix: str,
) -> dict[str, str]:
    """Return collision-free local aliases for declared runtime names."""

    occupied = {
        *names,
        "context",
        "state_vector",
        "derivative",
        "coordinate_rates",
        "row_index",
    }
    aliases: dict[str, str] = {}
    alias_index = 0
    for name in sorted(names):
        alias = f"{prefix}{alias_index}"
        while alias in occupied:
            alias_index += 1
            alias = f"{prefix}{alias_index}"
        aliases[name] = alias
        occupied.add(alias)
        alias_index += 1
    return aliases


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
class _DeclaredRuntimeAssets:
    """Process-local structural graph assets for one compiled model."""

    runtime_signature: str
    perturbation_data: Any
    execution_plan: _DeclaredGraphExecutionPlan
    owner_pid: int


@dataclass(frozen=True, slots=True)
class NativeBatchedEvolutionStats:
    """Account for one shared explicit evolution over several modes."""

    mode_count: int
    interval_count: int
    rk_stage_count: int
    substep_count: int
    maximum_substeps: int


def _integrate_batched_rk4(
    initial_states: numpy.ndarray,
    eta_grid: numpy.ndarray,
    *,
    required_substeps: numpy.ndarray,
    active_intervals: numpy.ndarray,
    rhs: Callable[..., numpy.ndarray],
    pre_step: Callable[..., numpy.ndarray] | None = None,
    post_step: Callable[..., numpy.ndarray] | None = None,
    record_step: Callable[..., numpy.ndarray] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, NativeBatchedEvolutionStats]:
    """Integrate mode rows on a shared grid with a common RK4 schedule.

    ``required_substeps`` may vary by mode, but each interval uses the next
    power of two above the largest requested mode count.  This removes the
    per-mode adaptive control path while retaining the declared stiffness
    budget for every row.  Callbacks operate on all rows at once.
    """

    states = numpy.asarray(initial_states, dtype=float).copy()
    if states.ndim != 2:
        raise ValueError("Batched evolution states must be a two-dimensional")
    eta_values = numpy.asarray(eta_grid, dtype=float)
    if eta_values.ndim != 1 or eta_values.size < 1:
        raise ValueError("Batched evolution requires a one-dimensional grid")
    mode_count = int(states.shape[0])
    interval_count = max(int(eta_values.size) - 1, 0)
    required = numpy.asarray(required_substeps, dtype=int)
    if required.ndim == 0:
        required = numpy.full(
            (mode_count, interval_count),
            max(int(required), 1),
            dtype=int,
        )
    elif required.ndim == 1:
        if required.size != interval_count:
            raise ValueError("Batched substep schedule has the wrong length")
        required = numpy.broadcast_to(
            required[numpy.newaxis, :],
            (mode_count, interval_count),
        )
    elif required.shape != (mode_count, interval_count):
        raise ValueError("Batched substep schedule has the wrong shape")
    active = numpy.asarray(active_intervals, dtype=bool)
    if active.shape != (mode_count, interval_count):
        raise ValueError("Batched active-interval mask has the wrong shape")
    if not numpy.all(numpy.isfinite(states)):
        raise ValueError(
            "Batched evolution received non-finite initial states"
        )
    histories = numpy.empty(
        (mode_count, eta_values.size, states.shape[1]),
        dtype=float,
    )
    if record_step is not None and interval_count:
        states = record_step(
            states,
            step_index=0,
            blend=0.0,
            active=active[:, 0],
        )
    histories[:, 0, :] = states
    rk_stage_count = 0
    total_substep_count = 0
    maximum_substeps = 0
    for step_index in range(interval_count):
        dt = float(eta_values[step_index + 1] - eta_values[step_index])
        requested = max(int(numpy.max(required[:, step_index])), 1)
        substep_count = 1
        while substep_count < requested:
            substep_count *= 2
        maximum_substeps = max(maximum_substeps, substep_count)
        total_substep_count += substep_count
        interval_active = active[:, step_index]
        sub_dt = dt / float(substep_count)
        for substep_index in range(substep_count):
            blend_start = substep_index / float(substep_count)
            blend_mid = (substep_index + 0.5) / float(substep_count)
            blend_end = (substep_index + 1.0) / float(substep_count)
            if pre_step is not None:
                states = pre_step(
                    states,
                    step_index=step_index,
                    blend=blend_start,
                    dt=0.5 * sub_dt,
                    active=interval_active,
                )
            rhs_a = rhs(
                states,
                step_index=step_index,
                blend=blend_start,
                active=interval_active,
            )
            rhs_b = rhs(
                states + 0.5 * sub_dt * rhs_a,
                step_index=step_index,
                blend=blend_mid,
                active=interval_active,
            )
            rhs_c = rhs(
                states + 0.5 * sub_dt * rhs_b,
                step_index=step_index,
                blend=blend_mid,
                active=interval_active,
            )
            rhs_d = rhs(
                states + sub_dt * rhs_c,
                step_index=step_index,
                blend=blend_end,
                active=interval_active,
            )
            rk_stage_count += 4
            states = states + (sub_dt / 6.0) * (
                rhs_a + 2.0 * rhs_b + 2.0 * rhs_c + rhs_d
            )
            if post_step is not None:
                states = post_step(
                    states,
                    step_index=step_index,
                    blend=blend_end,
                    dt=0.5 * sub_dt,
                    active=interval_active,
                )
            if not numpy.all(numpy.isfinite(states)):
                bad_rows = numpy.flatnonzero(
                    ~numpy.all(numpy.isfinite(states), axis=1)
                )
                bad_row = int(bad_rows[0]) if bad_rows.size else -1
                raise ValueError(
                    "Batched CMB evolution produced non-finite state values "
                    f"at mode_index={bad_row}, step_index={step_index}"
                )
        if record_step is not None:
            if step_index + 1 < interval_count:
                record_active = active[:, step_index + 1]
            else:
                record_active = interval_active
            states = record_step(
                states,
                step_index=step_index + 1,
                blend=0.0,
                active=record_active,
            )
        histories[:, step_index + 1, :] = states
    return (
        histories,
        states,
        NativeBatchedEvolutionStats(
            mode_count=mode_count,
            interval_count=interval_count,
            rk_stage_count=rk_stage_count,
            substep_count=total_substep_count,
            maximum_substeps=maximum_substeps,
        ),
    )


@lru_cache(maxsize=256)
def _compile_ordered_context_program(
    value_specs: tuple[tuple[str, str], ...],
) -> Any:
    """Compile one direct-assignment program for a prepared value order."""

    statements: list[ast.stmt] = []
    for output_name, expression in value_specs:
        if not output_name.isidentifier() or keyword.iskeyword(output_name):
            raise ValueError(
                "Declared value names must be identifiers for the native "
                f"compiled context path: {output_name}"
            )
        expression_node = copy.deepcopy(_parse_safe_expression(expression))
        expression_node = _ContextNameRewriter().visit(expression_node)
        expression_node = ast.fix_missing_locations(expression_node)
        condition = ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=ast.Constant(output_name),
                    ops=[ast.NotIn()],
                    comparators=[
                        ast.Name(
                            id="context",
                            ctx=ast.Load(),
                        )
                    ],
                ),
                ast.Compare(
                    left=ast.Constant(output_name),
                    ops=[ast.NotIn()],
                    comparators=[
                        ast.Name(
                            id="suppressed_outputs",
                            ctx=ast.Load(),
                        )
                    ],
                ),
            ],
        )
        statements.append(
            ast.If(
                test=condition,
                body=[
                    ast.Assign(
                        targets=[
                            ast.Subscript(
                                value=ast.Name(
                                    id="context",
                                    ctx=ast.Load(),
                                ),
                                slice=ast.Constant(output_name),
                                ctx=ast.Store(),
                            )
                        ],
                        value=expression_node.body,
                    )
                ],
                orelse=[],
            )
        )
    executor = ast.FunctionDef(
        name="_execute_declared_context",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="context"),
                ast.arg(arg="suppressed_outputs"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=statements or [ast.Pass()],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[executor], type_ignores=[])
    )
    namespace = dict(_COMPILED_CONTEXT_GLOBALS)
    # security-scanner: allow generated code execution
    exec(  # nosec B102 - expressions passed AST validation.
        compile(module, "<declared-cmb-context>", "exec"),
        namespace,
    )
    return namespace["_execute_declared_context"]


@lru_cache(maxsize=256)
def _compile_expression_tuple_program(
    expressions: tuple[str, ...],
) -> Any:
    """Compile several validated expressions into one context executor."""

    values: list[ast.expr] = []
    for expression in expressions:
        expression_node = copy.deepcopy(_parse_safe_expression(expression))
        expression_node = _ContextNameRewriter().visit(expression_node)
        values.append(ast.fix_missing_locations(expression_node).body)
    executor = ast.FunctionDef(
        name="_execute_declared_expression_tuple",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="context")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=ast.Tuple(elts=values, ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[executor], type_ignores=[])
    )
    namespace = dict(_COMPILED_CONTEXT_GLOBALS)
    # security-scanner: allow generated code execution
    exec(  # nosec B102 - expressions passed AST validation.
        compile(module, "<declared-cmb-expression-tuple>", "exec"),
        namespace,
    )
    return namespace["_execute_declared_expression_tuple"]


@lru_cache(maxsize=128)
def _compile_equation_program(
    slot_specs: tuple[tuple[int, str, str | None, int | None], ...],
) -> Any:
    """Compile direct scalar assignments for one declared state layout."""

    parsed_expressions: list[ast.Expression | None] = []
    context_names: set[str] = set()
    for _, _, expression, _ in slot_specs:
        if expression is None:
            parsed_expressions.append(None)
            continue
        expression_node = copy.deepcopy(_parse_safe_expression(expression))
        parsed_expressions.append(expression_node)
        context_names.update(
            node.id
            for node in ast.walk(expression_node)
            if isinstance(node, ast.Name)
            and node.id not in _ALLOWED_CONSTANTS
            and node.id not in _ALLOWED_MATH_FUNCS
        )
    context_aliases = _private_local_aliases(
        context_names,
        prefix="_context_value_",
    )
    rate_aliases = _private_local_aliases(
        {str(wrt_name) for _, wrt_name, _, _ in slot_specs},
        prefix="_coordinate_rate_",
    )
    statements: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id=alias, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Name(id="context", ctx=ast.Load()),
                slice=ast.Constant(name),
                ctx=ast.Load(),
            ),
        )
        for name, alias in context_aliases.items()
    ]
    statements.extend(
        ast.Assign(
            targets=[ast.Name(id=alias, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Name(id="coordinate_rates", ctx=ast.Load()),
                slice=ast.Constant(name),
                ctx=ast.Load(),
            ),
        )
        for name, alias in rate_aliases.items()
    )
    for (
        state_index,
        wrt_name,
        expression,
        promote_from_index,
    ), expression_node in zip(slot_specs, parsed_expressions, strict=True):
        if expression is None:
            if promote_from_index is None:
                raise ValueError(
                    "Declared state slot lacks an equation or promotion "
                    f"source: {state_index}"
                )
            value_node: ast.expr = ast.Subscript(
                value=ast.Name(id="state_vector", ctx=ast.Load()),
                slice=ast.Constant(int(promote_from_index)),
                ctx=ast.Load(),
            )
        else:
            if expression_node is None:  # pragma: no cover - internal guard.
                raise ValueError("Declared equation expression is missing")
            expression_node = _ContextAliasRewriter(context_aliases).visit(
                expression_node
            )
            expression_node = ast.fix_missing_locations(expression_node)
            value_node = expression_node.body
        value_node = ast.BinOp(
            left=value_node,
            op=ast.Mult(),
            right=ast.Name(
                id=rate_aliases[str(wrt_name)],
                ctx=ast.Load(),
            ),
        )
        statements.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id="derivative", ctx=ast.Load()),
                        slice=ast.Constant(int(state_index)),
                        ctx=ast.Store(),
                    )
                ],
                value=value_node,
            )
        )
    executor = ast.FunctionDef(
        name="_execute_equations",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="context"),
                ast.arg(arg="state_vector"),
                ast.arg(arg="derivative"),
                ast.arg(arg="coordinate_rates"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=statements or [ast.Pass()],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[executor], type_ignores=[])
    )
    namespace = dict(_COMPILED_CONTEXT_GLOBALS)
    # security-scanner: allow generated code execution
    exec(  # nosec B102 - expressions passed AST validation.
        compile(module, "<declared-cmb-equations>", "exec"),
        namespace,
    )
    return namespace["_execute_equations"]


class _BatchedRowContextAliasRewriter(ast.NodeTransformer):
    """Bind vector-valued declaration inputs to one batch-row scalar."""

    def __init__(
        self,
        aliases: Mapping[str, str],
        vector_names: frozenset[str],
    ) -> None:
        """Record the preloaded aliases and their vector-valued names."""

        self._aliases = dict(aliases)
        self._vector_names = frozenset(vector_names)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Select one row for vector inputs and retain scalar inputs."""

        if node.id in _ALLOWED_CONSTANTS or node.id in _ALLOWED_MATH_FUNCS:
            return node
        alias = ast.Name(id=self._aliases[node.id], ctx=node.ctx)
        if node.id not in self._vector_names:
            return ast.copy_location(alias, node)
        return ast.copy_location(
            ast.Subscript(
                value=alias,
                slice=ast.Name(id="row_index", ctx=ast.Load()),
                ctx=node.ctx,
            ),
            node,
        )


@lru_cache(maxsize=128)
def _compile_batched_row_equation_program(
    slot_specs: tuple[tuple[int, str, str | None, int | None], ...],
    vector_context_names: tuple[str, ...],
) -> Any:
    """Compile a scalar-row executor for small compatible mode batches."""

    parsed_expressions: list[ast.Expression | None] = []
    context_names: set[str] = set()
    for _, _, expression, _ in slot_specs:
        if expression is None:
            parsed_expressions.append(None)
            continue
        expression_node = copy.deepcopy(_parse_safe_expression(expression))
        parsed_expressions.append(expression_node)
        context_names.update(
            node.id
            for node in ast.walk(expression_node)
            if isinstance(node, ast.Name)
            and node.id not in _ALLOWED_CONSTANTS
            and node.id not in _ALLOWED_MATH_FUNCS
        )
    vector_names = frozenset(vector_context_names)
    if not vector_names <= context_names:
        raise ValueError(
            "Batched row equation inputs must be declared equation names"
        )
    context_aliases = _private_local_aliases(
        context_names,
        prefix="_context_value_",
    )
    rate_aliases = _private_local_aliases(
        {str(wrt_name) for _, wrt_name, _, _ in slot_specs},
        prefix="_coordinate_rate_",
    )
    statements: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id=alias, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Name(id="context", ctx=ast.Load()),
                slice=ast.Constant(name),
                ctx=ast.Load(),
            ),
        )
        for name, alias in context_aliases.items()
    ]
    statements.extend(
        ast.Assign(
            targets=[ast.Name(id=alias, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Name(id="coordinate_rates", ctx=ast.Load()),
                slice=ast.Constant(name),
                ctx=ast.Load(),
            ),
        )
        for name, alias in rate_aliases.items()
    )
    row_rewriter = _BatchedRowContextAliasRewriter(
        context_aliases,
        vector_names,
    )
    row_statements: list[ast.stmt] = []
    for (
        state_index,
        wrt_name,
        expression,
        promote_from_index,
    ), expression_node in zip(slot_specs, parsed_expressions, strict=True):
        if expression is None:
            if promote_from_index is None:
                raise ValueError(
                    "Declared state slot lacks an equation or promotion "
                    f"source: {state_index}"
                )
            value_node: ast.expr = ast.Subscript(
                value=ast.Subscript(
                    value=ast.Name(id="state_vector", ctx=ast.Load()),
                    slice=ast.Constant(int(promote_from_index)),
                    ctx=ast.Load(),
                ),
                slice=ast.Name(id="row_index", ctx=ast.Load()),
                ctx=ast.Load(),
            )
        else:
            if expression_node is None:  # pragma: no cover - internal guard.
                raise ValueError("Declared equation expression is missing")
            expression_node = row_rewriter.visit(expression_node)
            expression_node = ast.fix_missing_locations(expression_node)
            value_node = expression_node.body
        row_statements.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id="derivative", ctx=ast.Load()),
                        slice=ast.Tuple(
                            elts=[
                                ast.Constant(int(state_index)),
                                ast.Name(
                                    id="row_index",
                                    ctx=ast.Load(),
                                ),
                            ],
                            ctx=ast.Load(),
                        ),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.BinOp(
                    left=value_node,
                    op=ast.Mult(),
                    right=ast.Name(
                        id=rate_aliases[str(wrt_name)],
                        ctx=ast.Load(),
                    ),
                ),
            )
        )
    statements.append(
        ast.For(
            target=ast.Name(id="row_index", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(
                                id="state_vector",
                                ctx=ast.Load(),
                            ),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(1),
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
            body=row_statements,
            orelse=[],
        )
    )
    executor = ast.FunctionDef(
        name="_execute_batched_row_equations",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="context"),
                ast.arg(arg="state_vector"),
                ast.arg(arg="derivative"),
                ast.arg(arg="coordinate_rates"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=statements or [ast.Pass()],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[executor], type_ignores=[])
    )
    namespace = {**_COMPILED_CONTEXT_GLOBALS, "range": range}
    # security-scanner: allow generated code execution
    exec(  # nosec B102 - expressions passed AST validation.
        compile(module, "<declared-cmb-batched-row-equations>", "exec"),
        namespace,
    )
    return namespace["_execute_batched_row_equations"]


@dataclass(frozen=True, slots=True)
class _DeclaredMomentumGridTopology:
    """Parameter-independent nodes and weights for one momentum grid."""

    name: str
    points: numpy.ndarray
    weights: numpy.ndarray
    quadrature_order: int
    family_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DeclaredMomentumGridRuntime:
    """Prepared quadrature metadata for one declared momentum grid."""

    name: str
    points: numpy.ndarray
    weights: numpy.ndarray
    quadrature_order: int
    mass_eV: float
    family_names: tuple[str, ...]


def _thermal_fermi_dirac_distribution(
    q_points: numpy.ndarray,
) -> numpy.ndarray:
    """Return the thermal Fermi-Dirac occupation for one q grid."""

    points = numpy.asarray(q_points, dtype=float)
    if points.ndim != 1 or points.size == 0:
        raise ValueError("Massive-neutrino q nodes must be a non-empty vector")
    if not numpy.all(numpy.isfinite(points)) or numpy.any(points <= 0.0):
        raise ValueError(
            "Massive-neutrino q nodes must be finite and strictly positive"
        )
    exp_neg_q = numpy.exp(-points)
    return numpy.asarray(exp_neg_q / (1.0 + exp_neg_q), dtype=float)


def _normalize_declared_momentum_weights(
    raw_weights: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return normalized weights and their summed physical moments."""

    raw = numpy.asarray(raw_weights, dtype=float)
    if raw.ndim == 0 or not numpy.all(numpy.isfinite(raw)):
        raise ValueError("Massive-neutrino quadrature weights must be finite")
    if numpy.any(raw < 0.0):
        raise ValueError(
            "Massive-neutrino quadrature weights must be non-negative"
        )
    totals = numpy.sum(raw, axis=-1, keepdims=True)
    if numpy.any(totals <= 0.0):
        raise ValueError(
            "Massive-neutrino quadrature moments must be positive"
        )
    normalized = numpy.asarray(raw / totals, dtype=float)
    return normalized, numpy.asarray(
        numpy.squeeze(totals, axis=-1),
        dtype=float,
    )


def _validate_declared_momentum_grid_definition(
    grid_name: str,
    grid_def: Mapping[str, Any],
) -> tuple[int, float, float, int]:
    """Validate one logarithmic q-grid definition before materialization."""

    count_value = _coerce_numeric_scalar(
        grid_def.get("count", 8),
        name=f"momentum grid '{grid_name}' count",
    )
    count = int(count_value)
    if count != count_value or count < 2:
        raise ValueError(
            f"momentum grid '{grid_name}' count must be an integer >= 2"
        )
    q_min = _coerce_numeric_scalar(
        grid_def.get("q_min", 0.05),
        name=f"momentum grid '{grid_name}' q_min",
    )
    q_max = _coerce_numeric_scalar(
        grid_def.get("q_max", 15.0),
        name=f"momentum grid '{grid_name}' q_max",
    )
    if not numpy.isfinite(q_min) or not numpy.isfinite(q_max):
        raise ValueError(
            f"momentum grid '{grid_name}' q bounds must be finite"
        )
    if q_min <= 0.0 or q_max <= q_min:
        raise ValueError(
            f"momentum grid '{grid_name}' requires 0 < q_min < q_max"
        )
    quadrature_order_value = _coerce_numeric_scalar(
        grid_def.get("quadrature_order", 2),
        name=f"momentum grid '{grid_name}' quadrature_order",
    )
    quadrature_order = int(quadrature_order_value)
    if quadrature_order != quadrature_order_value or quadrature_order != 2:
        raise ValueError(
            f"momentum grid '{grid_name}' quadrature_order must be 2"
        )
    return count, q_min, q_max, quadrature_order


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


def prepare_native_runtime_assets(
    runtime_signature: str,
    perturbation_data: Any,
) -> _DeclaredRuntimeAssets:
    """Prepare one model's structural graph assets once in this process."""

    signature = str(runtime_signature or "").strip()
    if not signature:
        signature = repr(
            _declared_graph_execution_plan_cache_token(perturbation_data)
        )
    owner_pid = os.getpid()
    cache_key = (owner_pid, signature)
    cached = native_cache.get_native_runtime_assets(cache_key)
    if cached is not None:
        return cached
    assets = _DeclaredRuntimeAssets(
        runtime_signature=signature,
        perturbation_data=perturbation_data,
        execution_plan=_compile_declared_graph_execution_plan(
            perturbation_data
        ),
        owner_pid=owner_pid,
    )
    native_cache.set_native_runtime_assets(cache_key, assets)
    return assets


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
        family_species = set(getattr(family_entry, "species", ()))
        declared_species = getattr(perturbation_data, "species", {})
        if (
            "massive_neutrino" not in family_species
            or "massive_neutrino" not in declared_species
        ):
            continue
        grid_name = str(family_entry.momentum_grid or "").strip()
        if not grid_name:
            continue
        family_groups.setdefault(grid_name, []).append(str(family_name))
    if not family_groups:
        return ()

    topology_key = tuple(
        sorted(
            (
                str(name),
                repr(momentum_grid_defs.get(name, {})),
                tuple(sorted(family_names)),
                repr(minimum_momentum_counts.get(name)),
            )
            for name, family_names in family_groups.items()
        )
    )
    topologies = native_cache.get_declared_momentum_topology(topology_key)
    if topologies is None:
        prepared_topologies: list[_DeclaredMomentumGridTopology] = []
        for grid_name, family_names in sorted(family_groups.items()):
            grid_def = momentum_grid_defs.get(grid_name, {})
            if grid_def in (None, {}):
                grid_def = {}
            if not isinstance(grid_def, Mapping):
                raise ValueError(
                    "cmb.perturbations.numerics.momentum_grids."
                    f"{grid_name} must be a mapping"
                )
            count, q_min, q_max, quadrature_order = (
                _validate_declared_momentum_grid_definition(
                    grid_name,
                    grid_def,
                )
            )
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
                        "Declared accuracy_controls require positive "
                        f"momentum grid counts for '{grid_name}'"
                    )
                if count < required_count:
                    raise ValueError(
                        "Declared accuracy_controls require "
                        "cmb.perturbations.numerics.momentum_grids."
                        f"{grid_name}.count >= {required_count}"
                    )
            points = numpy.geomspace(q_min, q_max, count, dtype=float)
            if not numpy.all(numpy.isfinite(points)) or not numpy.all(
                numpy.diff(points) > 0.0
            ):
                raise ValueError(
                    f"momentum grid '{grid_name}' produced invalid q nodes"
                )
            log_points = numpy.log(points)
            weights = numpy.empty_like(points)
            deltas = numpy.diff(log_points)
            weights[0] = 0.5 * deltas[0]
            weights[-1] = 0.5 * deltas[-1]
            if points.size > 2:
                weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
            weights = numpy.asarray(weights, dtype=float)
            if not numpy.all(numpy.isfinite(weights)) or numpy.any(
                weights <= 0.0
            ):
                raise ValueError(
                    f"momentum grid '{grid_name}' produced invalid q weights"
                )
            points.flags.writeable = False
            weights.flags.writeable = False
            prepared_topologies.append(
                _DeclaredMomentumGridTopology(
                    name=str(grid_name),
                    points=points,
                    weights=weights,
                    quadrature_order=quadrature_order,
                    family_names=tuple(
                        sorted(str(name) for name in family_names)
                    ),
                )
            )
        topologies = tuple(prepared_topologies)
        native_cache.set_declared_momentum_topology(
            topology_key,
            topologies,
        )

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
        topology_key,
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
    for topology in topologies:
        grid_def = momentum_grid_defs.get(topology.name, {}) or {}
        runtimes.append(
            _DeclaredMomentumGridRuntime(
                name=topology.name,
                points=topology.points,
                weights=topology.weights,
                quadrature_order=topology.quadrature_order,
                mass_eV=_grid_mass_eV(topology.name, grid_def),
                family_names=topology.family_names,
            )
        )
    runtime_tuple = tuple(runtimes)
    native_cache.set_declared_momentum_grid(cache_key, runtime_tuple)
    return runtime_tuple


@lru_cache(maxsize=128)
def _prepare_declared_momentum_static_terms(
    points_key: tuple[float, ...],
    weights_key: tuple[float, ...],
    mass_ratio_today: float,
) -> tuple[Any, ...]:
    """Cache q-grid algebra that is independent of the scale factor."""

    points = numpy.asarray(points_key, dtype=float)
    weights = numpy.asarray(weights_key, dtype=float)
    thermal_distribution = _thermal_fermi_dirac_distribution(points)
    quadrature_weights = numpy.asarray(
        weights * thermal_distribution,
        dtype=float,
    )
    points_squared = numpy.square(points)
    density_weight_base = quadrature_weights * numpy.power(points, 3.0)
    pressure_weight_base = quadrature_weights * numpy.power(points, 5.0)
    momentum_weight_raw = quadrature_weights * numpy.power(points, 4.0)
    base_momentum_weights, base_momentum_moment = (
        _normalize_declared_momentum_weights(momentum_weight_raw)
    )
    epsilon_today = numpy.sqrt(
        points_squared + float(mass_ratio_today) * float(mass_ratio_today)
    )
    density_moment_today = numpy.sum(density_weight_base * epsilon_today)
    if not numpy.isfinite(density_moment_today) or density_moment_today <= 0.0:
        raise ValueError(
            "Massive-neutrino density quadrature must produce a positive "
            "finite moment"
        )
    return (
        quadrature_weights,
        density_weight_base,
        pressure_weight_base,
        base_momentum_weights,
        float(base_momentum_moment),
        epsilon_today,
        max(float(density_moment_today), 1.0e-300),
    )


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
        mass_ratio_today = float(runtime.mass_eV) / neutrino_temperature_eV
        (
            quadrature_weights,
            density_weight_base,
            pressure_weight_base,
            base_momentum_weights,
            background_momentum_moment,
            epsilon_today,
            density_moment_today,
        ) = _prepare_declared_momentum_static_terms(
            tuple(float(value) for value in runtime.points),
            tuple(float(value) for value in runtime.weights),
            mass_ratio_today,
        )
        mass_term = mass_ratio_today * a_values
        epsilon = numpy.sqrt(
            numpy.square(runtime.points) + numpy.square(mass_term[..., None])
        )
        q_velocity_ratio = runtime.points / epsilon
        q_pressure_ratio = numpy.square(q_velocity_ratio) / 3.0
        q_mass_fraction = mass_term[..., None] / epsilon
        q_streaming_speed = numpy.asarray(q_velocity_ratio, dtype=float)
        density_weight_raw = density_weight_base * epsilon
        pressure_weight_raw = pressure_weight_base / (3.0 * epsilon)
        shear_weight_raw = pressure_weight_base / epsilon
        density_weights, background_density_moment = (
            _normalize_declared_momentum_weights(density_weight_raw)
        )
        pressure_weights, background_pressure_moment = (
            _normalize_declared_momentum_weights(pressure_weight_raw)
        )
        shear_weights, background_shear_moment = (
            _normalize_declared_momentum_weights(shear_weight_raw)
        )
        total_mass_eV = _declared_total_mass_eV(runtime)
        massive_omega0 = _massive_neutrino_omega0(
            runtime,
            total_mass_eV,
        )
        scale_factor_array = numpy.maximum(a_values, 1.0e-30)
        density_fraction = (
            massive_omega0
            * numpy.power(scale_factor_array, -4.0)
            * background_density_moment
            / density_moment_today
        )
        # The evolved dipole is v(q,a) * Psi_1.  Its metric moment therefore
        # uses q^4 f_0 / v, which remains finite as a bin becomes
        # non-relativistic while avoiding a singular dipole equation.
        momentum_weight_raw = (
            quadrature_weights
            * numpy.power(runtime.points, 4.0)
            / numpy.maximum(q_velocity_ratio, 1.0e-30)
        )
        momentum_weights, background_momentum_moment = (
            _normalize_declared_momentum_weights(momentum_weight_raw)
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
            base_momentum_weights * q_velocity_ratio,
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
        context[f"{prefix}_quadrature_order"] = int(runtime.quadrature_order)
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
    momentum_grid_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return scalar runtime values shared by equations and conditions.

    A caller evolving several Fourier modes may provide the cached
    scale-factor-only momentum context instead of rebuilding it for every
    equation stage.
    """

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
    if momentum_grid_context is None:
        momentum_grid_context = _declared_momentum_grid_context(
            perturbation_data,
            model_parameters=model_parameters,
            physical_params=physical_params,
            scale_factor=float(background_scalars["a"]),
        )
    context.update(momentum_grid_context)
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
    exit_ratio: float = 0.1,
) -> float:
    """Return the collision rate below which tight coupling is disabled."""

    if not 0.0 < float(exit_ratio) < 1.0:
        raise ValueError("tight-coupling exit ratio must be in (0, 1)")
    return float(exit_ratio) * _tight_coupling_entry_rate(
        k_value=k_value,
        tight_coupling_ratio=tight_coupling_ratio,
    )


def _tight_coupling_is_active(
    *,
    active: bool,
    collision_rate: float,
    k_value: float,
    tight_coupling_ratio: float,
    exit_ratio: float = 0.1,
) -> bool:
    """Return the updated tight-coupling regime with hysteresis."""

    if not numpy.isfinite(collision_rate) or collision_rate <= 0.0:
        return False
    if active:
        return collision_rate > _tight_coupling_exit_rate(
            k_value=k_value,
            tight_coupling_ratio=tight_coupling_ratio,
            exit_ratio=exit_ratio,
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


def _resolve_declared_graph_context_ordered(
    context: dict[str, Any],
    perturbation_data: Any,
    *,
    allow_partial: bool,
    eta_grid: numpy.ndarray | None,
    execution_plan: _DeclaredGraphExecutionPlan,
    derivative_steps: tuple[_DeclaredDerivativeStep, ...] = (),
    value_steps: tuple[_DeclaredValueStep, ...] = (),
    suppressed_outputs: Mapping[str, Any] | None = None,
    use_compiled_program: bool = False,
    compiled_value_program: Any | None = None,
) -> dict[str, Any]:
    """Resolve a prepared dependency order without pending-round scans."""

    runtime_spec = execution_plan.runtime_spec
    unresolved = False
    for step in derivative_steps:
        if step.output_name in context:
            continue
        slot_index = runtime_spec.state_index_by_key.get(
            (step.variable, step.wrt, int(step.order))
        )
        if eta_grid is None:
            if slot_index is None or step.slot_name not in context:
                unresolved = True
                continue
            context[step.output_name] = context[step.slot_name]
            continue

        if step.variable not in context:
            unresolved = True
            continue
        coordinate_name = str(step.wrt or runtime_spec.evolution_variable)
        derivative_value = numpy.asarray(context[step.variable], dtype=float)
        if coordinate_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
            coordinate_history = numpy.asarray(eta_grid, dtype=float)
        else:
            if coordinate_name not in context:
                unresolved = True
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
                    "Declared coordinate history must match the eta grid "
                    f"for derivative symbol '{step.output_name}'."
                )
        for _ in range(int(step.order)):
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
                    "Declared coordinate history produced non-finite rates "
                    f"for derivative symbol '{step.output_name}'."
                )
            if numpy.any(numpy.abs(coordinate_rate) <= 1.0e-12):
                raise ValueError(
                    "Declared coordinate history is singular for derivative "
                    f"symbol '{step.output_name}'."
                )
            derivative_value = derivative_eta / coordinate_rate
        context[step.output_name] = derivative_value

    if use_compiled_program and value_steps:
        suppressed = dict(suppressed_outputs or {})
        context.update(suppressed)
        if compiled_value_program is None:
            value_specs = tuple(
                (
                    str(step.output_name),
                    str(step.compiled_expression.expression),
                )
                for step in value_steps
            )
            compiled_value_program = _compile_ordered_context_program(
                value_specs
            )
        try:
            compiled_value_program(context, suppressed)
        except (NameError, ValueError):
            return _resolve_declared_graph_context(
                context,
                perturbation_data,
                allow_partial=allow_partial,
                eta_grid=eta_grid,
                execution_plan=execution_plan,
                suppressed_outputs=suppressed_outputs,
            )
        return context

    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for step in value_steps:
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
                dependency not in context for dependency in step.dependencies
            ):
                unresolved = True
                continue
            context[step.output_name] = _evaluate_compiled_expression_noerr(
                step.compiled_expression,
                context,
            )

    if not unresolved:
        return context
    return _resolve_declared_graph_context(
        context,
        perturbation_data,
        allow_partial=allow_partial,
        eta_grid=eta_grid,
        execution_plan=execution_plan,
        suppressed_outputs=suppressed_outputs,
    )


def _evaluate_declared_initial_state(
    *,
    perturbation_data: Any,
    execution_plan: _DeclaredGraphExecutionPlan,
    base_context: Mapping[str, Any],
    fixed_state_values: Mapping[tuple[str, str, int], float] | None = None,
) -> tuple[numpy.ndarray, tuple[tuple[str, str, int], ...]]:
    """Return the initial state vector for one Fourier mode.

    ``fixed_state_values`` supplies an algebraic initial-surface solution for
    selected state slots.  The remaining declared conditions are still
    evaluated in dependency order, so conditions coupled to a solved metric
    value observe the same state that enters evolution.
    """

    runtime_spec = execution_plan.runtime_spec
    state_vector = numpy.zeros(len(runtime_spec.state_slots), dtype=float)
    assigned_targets: list[tuple[str, str, int]] = []
    context = dict(base_context)
    condition_entries = execution_plan.start_condition_entries
    declared_target_keys = {
        (
            str(entry.target.variable),
            str(entry.target.wrt),
            int(entry.target.order),
        )
        for entry in condition_entries
    }
    fixed_values = {
        (str(variable), str(wrt), int(order)): float(value)
        for (variable, wrt, order), value in (fixed_state_values or {}).items()
    }
    unknown_fixed_targets = sorted(set(fixed_values) - declared_target_keys)
    if unknown_fixed_targets:
        raise ValueError(
            "Declared initial-state constraint solve targeted an undeclared "
            f"slot: {unknown_fixed_targets[0]}"
        )
    if not numpy.all(numpy.isfinite(tuple(fixed_values.values()))):
        raise ValueError(
            "Declared initial-state constraint solve produced non-finite "
            "fixed values"
        )
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
            target_key = (
                str(entry.target.variable),
                str(entry.target.wrt),
                int(entry.target.order),
            )
            if target_key in fixed_values:
                value = fixed_values[target_key]
            else:
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
            state_index = runtime_spec.state_index_by_key[target_key]
            state_vector[state_index] = value
            assigned_targets.append(target_key)
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


_GENERATED_SCALAR_INITIAL_CONSTRAINT_TOLERANCE = 1.0e-2
_SCALAR_EINSTEIN_RESIDUAL_NAMES = (
    "einstein_energy_residual",
    "einstein_momentum_residual",
    "einstein_shear_residual",
)


def _scalar_einstein_constraint_metrics(
    context: Mapping[str, Any],
    residual_name: str,
) -> dict[str, Any]:
    """Return dimensionally matched terms and normalized residual metrics.

    Each constraint is normalized by the sum of the magnitudes of every term
    in its declared equation.  This retains the units of the residual while
    avoiding an arbitrary dimensionless floor in a physical comparison.
    """

    if residual_name not in _SCALAR_EINSTEIN_RESIDUAL_NAMES:
        raise ValueError(
            "Unknown scalar Einstein residual: " f"{residual_name}"
        )
    if residual_name not in context:
        raise ValueError(
            "Scalar Einstein residual is missing from its context: "
            f"{residual_name}"
        )

    stable_shear_closure = False
    try:
        acoustic_k_sq = numpy.asarray(context["acoustic_k_sq"], dtype=float)
        gravity = numpy.asarray(
            context["einstein_gravity_strength"],
            dtype=float,
        )
        if residual_name == "einstein_energy_residual":
            term_values = {
                "metric_laplacian": acoustic_k_sq
                * numpy.asarray(context["Phi"], dtype=float),
                "metric_expansion": 3.0
                * numpy.asarray(context["Hconf"], dtype=float)
                * numpy.asarray(
                    context["metric_momentum_constraint"],
                    dtype=float,
                ),
                "density_source": 1.5
                * gravity
                * numpy.asarray(context["total_density_source"], dtype=float),
            }
        elif residual_name == "einstein_momentum_residual":
            term_values = {
                "metric_momentum": acoustic_k_sq
                * numpy.asarray(
                    context["metric_momentum_constraint"],
                    dtype=float,
                ),
                "momentum_source": -1.5
                * gravity
                * numpy.asarray(
                    context["total_momentum_source"],
                    dtype=float,
                ),
            }
        else:
            stable_shear_closure = {
                "metric_constraint_scale",
                "metric_shear_correction",
            }.issubset(context)
            if stable_shear_closure:
                metric_shear = numpy.asarray(
                    context["metric_constraint_scale"],
                    dtype=float,
                ) * numpy.asarray(
                    context["metric_shear_correction"],
                    dtype=float,
                )
            else:
                metric_shear = acoustic_k_sq * (
                    numpy.asarray(context["Phi"], dtype=float)
                    - numpy.asarray(context["Psi"], dtype=float)
                )
            term_values = {
                "metric_shear": metric_shear,
                "shear_source": -3.0
                * gravity
                * numpy.asarray(context["total_shear_source"], dtype=float),
            }
    except KeyError:
        # A hand-authored diagnostic can still be validated, but generated
        # scalar contracts must provide the complete term set above.
        residual_values = numpy.asarray(context[residual_name], dtype=float)
        scale = numpy.maximum(
            numpy.abs(residual_values),
            numpy.finfo(float).tiny,
        )
        return {
            "residual_values": residual_values,
            "normalized_values": numpy.abs(residual_values) / scale,
            "normalization_scale": scale,
            "term_values": {"declared_residual": residual_values},
            "normalization_source": "residual_magnitude_fallback",
        }

    residual_values = numpy.asarray(context[residual_name], dtype=float)
    arrays = numpy.broadcast_arrays(residual_values, *term_values.values())
    residual_values = arrays[0]
    term_values = {
        name: arrays[index + 1] for index, name in enumerate(term_values)
    }
    if residual_name == "einstein_shear_residual" and stable_shear_closure:
        residual_values = sum(term_values.values())
    normalization_scale = numpy.maximum(
        sum(numpy.abs(values) for values in term_values.values()),
        numpy.finfo(float).tiny,
    )
    return {
        "residual_values": residual_values,
        "normalized_values": numpy.abs(residual_values) / normalization_scale,
        "normalization_scale": normalization_scale,
        "term_values": term_values,
        "normalization_source": "sum_abs_declared_einstein_terms",
    }


def _solve_generated_scalar_initial_einstein_surface(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    k_value: float,
) -> dict[str, Any]:
    """Solve the coupled scalar Einstein surface for one initial context.

    The three algebraic unknowns are the observable curvature potential,
    metric momentum combination, and lapse potential.  Generated closures
    represent the latter two in the state graph; the caller binds the solved
    curvature value back to its gauge-specific state slot and re-evaluates
    the coupled start conditions.
    """

    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if not manifest_summary.get("generated_scalar_hierarchy"):
        return {}
    required_names = (
        "acoustic_k_sq",
        "Hconf",
        "einstein_gravity_strength",
        "total_density_source",
        "total_momentum_source",
        "total_shear_source",
    )
    missing_names = tuple(
        name for name in required_names if name not in context
    )
    if missing_names:
        raise NativeConstraintViolationError(
            "Generated scalar initial data cannot form the coupled Einstein "
            "constraint system",
            context={
                "gauge": str(getattr(perturbation_data, "gauge", "")),
                "k": float(k_value),
                "missing_terms": missing_names,
            },
        )
    acoustic_k_sq = float(context["acoustic_k_sq"])
    if not numpy.isfinite(acoustic_k_sq) or acoustic_k_sq <= 0.0:
        raise NativeConstraintViolationError(
            "Generated scalar initial data require a positive finite k^2 "
            "constraint scale",
            context={
                "gauge": str(getattr(perturbation_data, "gauge", "")),
                "k": float(k_value),
                "acoustic_k_sq": acoustic_k_sq,
            },
        )
    hconf = float(context["Hconf"])
    gravity = float(context["einstein_gravity_strength"])
    density = float(context["total_density_source"])
    momentum = float(context["total_momentum_source"])
    shear = float(context["total_shear_source"])
    coefficients = numpy.asarray(
        (
            (acoustic_k_sq, 3.0 * hconf, 0.0),
            (0.0, acoustic_k_sq, 0.0),
            (acoustic_k_sq, 0.0, -acoustic_k_sq),
        ),
        dtype=float,
    )
    source_terms = numpy.asarray(
        (
            -1.5 * gravity * density,
            1.5 * gravity * momentum,
            3.0 * gravity * shear,
        ),
        dtype=float,
    )
    if not numpy.all(numpy.isfinite(coefficients)) or not numpy.all(
        numpy.isfinite(source_terms)
    ):
        raise NativeNonFiniteEvolutionError(
            "Generated scalar initial Einstein system contains non-finite "
            f"terms at k={k_value}",
            context={
                "gauge": str(getattr(perturbation_data, "gauge", "")),
                "k": float(k_value),
            },
        )
    try:
        solution = numpy.linalg.solve(coefficients, source_terms)
    except numpy.linalg.LinAlgError as error:
        raise NativeConstraintViolationError(
            "Generated scalar initial Einstein system is singular",
            context={
                "gauge": str(getattr(perturbation_data, "gauge", "")),
                "k": float(k_value),
            },
        ) from error
    if not numpy.all(numpy.isfinite(solution)):
        raise NativeNonFiniteEvolutionError(
            "Generated scalar initial Einstein system produced non-finite "
            f"metric values at k={k_value}",
            context={
                "gauge": str(getattr(perturbation_data, "gauge", "")),
                "k": float(k_value),
            },
        )
    return {
        "Phi": float(solution[0]),
        "metric_momentum_constraint": float(solution[1]),
        "Psi": float(solution[2]),
        "coefficient_matrix": coefficients,
        "source_terms": source_terms,
    }


def _validate_generated_scalar_initial_constraints(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    k_value: float,
) -> dict[str, dict[str, Any]]:
    """Validate and record generated scalar Einstein initial conditions."""

    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if not manifest_summary.get("generated_scalar_hierarchy"):
        return {}

    tolerance = _GENERATED_SCALAR_INITIAL_CONSTRAINT_TOLERANCE
    diagnostics: dict[str, dict[str, Any]] = {}
    for residual_name in _SCALAR_EINSTEIN_RESIDUAL_NAMES:
        if residual_name not in context:
            continue
        metrics = _scalar_einstein_constraint_metrics(context, residual_name)
        residual_values = numpy.asarray(
            metrics["residual_values"],
            dtype=float,
        )
        normalized_values = numpy.asarray(
            metrics["normalized_values"],
            dtype=float,
        )
        normalization_scale = numpy.asarray(
            metrics["normalization_scale"],
            dtype=float,
        )
        if residual_values.ndim != 0:
            raise ValueError(
                "Generated scalar initial Einstein diagnostics must be "
                f"scalar values: {residual_name} at k={k_value}"
            )
        normalized_residual = float(normalized_values)
        term_values = {
            name: float(numpy.asarray(value, dtype=float))
            for name, value in metrics["term_values"].items()
        }
        diagnostic = {
            "absolute_residual": abs(float(residual_values)),
            "normalized_residual": normalized_residual,
            "normalization_scale": float(normalization_scale),
            "normalization_terms": term_values,
            "normalization_source": str(metrics["normalization_source"]),
            "tolerance": float(tolerance),
            "tolerance_provenance": "generated_initial_fixed_normalized",
        }
        if not numpy.isfinite(normalized_residual):
            raise NativeNonFiniteEvolutionError(
                "Generated scalar initial data produced non-finite Einstein "
                f"diagnostics for {residual_name} at k={k_value}",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(k_value),
                    "residual": residual_name,
                    **diagnostic,
                },
            )
        if normalized_residual > tolerance:
            raise NativeConstraintViolationError(
                "Generated scalar initial data violate the Einstein "
                f"constraints for {residual_name} at k={k_value} "
                f"({normalized_residual} > {tolerance})",
                context={
                    "gauge": str(getattr(perturbation_data, "gauge", "")),
                    "k": float(k_value),
                    "residual": residual_name,
                    **diagnostic,
                },
            )
        diagnostics[residual_name] = diagnostic

    for operator_name, operator_entry in (
        getattr(perturbation_data, "collision_operators", {}) or {}
    ).items():
        exact_form = getattr(operator_entry, "exact_form", None)
        if exact_form is None or not bool(exact_form.fast_manifold):
            continue
        compiled_expression = getattr(
            operator_entry,
            "compiled_expression",
            None,
        )
        if compiled_expression is None:
            continue
        collision_value = float(
            _evaluate_compiled_expression_noerr(
                compiled_expression,
                context,
            )
        )
        if not numpy.isfinite(collision_value):
            raise ValueError(
                "Generated scalar initial collision constraint produced "
                f"non-finite values for {operator_name} at k={k_value}"
            )
        collision_rate = abs(float(context.get("collision_rate", 0.0)))
        if collision_rate <= 1.0e-12:
            continue
        collision_tolerance = 1.0e-8 * max(1.0, collision_rate)
        if abs(collision_value) > collision_tolerance:
            raise ValueError(
                "Generated scalar initial collision constraint exceeded "
                f"tolerance for {operator_name} at k={k_value} "
                f"({abs(collision_value)} > {collision_tolerance})"
            )
    return diagnostics


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


def _validate_generated_tensor_initial_constraints(
    *,
    perturbation_data: Any,
    context: Mapping[str, Any],
    k_value: float,
) -> None:
    """Raise when generated tensor data violate their regular series."""

    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if not manifest_summary.get("generated_tensor_hierarchy"):
        return

    denominator = float(context["tensor_initial_series_denominator"])
    metric_series = (
        5.0
        * float(context["acoustic_k_sq"])
        * float(context["eta"])
        * float(context["h_tensor"])
        / denominator
    )
    neutrino_series = (
        4.0
        * float(context["acoustic_k_sq"])
        * float(context["eta"])
        * float(context["eta"])
        * float(context["h_tensor"])
        / (3.0 * denominator)
    )
    collision_series = (32.0 / 45.0) * float(context["h_tensor_tau"])
    residual_specs = (
        (
            "tensor_initial_metric_residual",
            max(abs(float(context["h_tensor_tau"])), abs(metric_series)),
        ),
        (
            "tensor_initial_neutrino_residual",
            max(abs(float(context["pi_nu_tensor"])), abs(neutrino_series)),
        ),
        (
            "tensor_initial_collision_residual",
            max(
                abs(
                    float(context["collision_rate"])
                    * float(context["pi_gamma_tensor"])
                ),
                abs(collision_series),
            ),
        ),
    )
    tolerance = 1.0e-8
    for residual_name, scale in residual_specs:
        if residual_name not in context:
            raise ValueError(
                "Generated tensor initial data omitted declared constraint "
                f"{residual_name} at k={k_value}"
            )
        normalized_residual = abs(float(context[residual_name])) / max(
            float(scale),
            1.0e-30,
        )
        if not numpy.isfinite(normalized_residual):
            raise ValueError(
                "Generated tensor initial data produced non-finite "
                f"diagnostics for {residual_name} at k={k_value}"
            )
        if normalized_residual > tolerance:
            raise ValueError(
                "Generated tensor initial data violate the regular-series "
                f"constraint for {residual_name} at k={k_value} "
                f"({normalized_residual} > {tolerance})"
            )
