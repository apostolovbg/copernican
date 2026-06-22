"""Compile declared CMB graph contracts into immutable runtime data.

`standard: false` contracts now describe one declared-math graph rather than
selecting a hard-coded solver family. The compiler validates symbols,
dependencies, observables, and runtime requirements before the numerical CMB
engine tries to evolve the system.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy

from .cmb_projection_contract import (
    get_declared_projection_spec,
    resolve_declared_projection_kernel,
    validate_declared_projection_source_roles,
)
from .engine_adapter import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES,
    FrozenMapping,
    _build_parameter_replacements,
    _freeze_for_cache,
    _parse_safe_expression,
    _replace_latex_tokens,
    _validate_safe_expression,
)

_COMPILED_BINARY_OPCODE_NAMES = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
}
_COMPILED_UNARY_OPCODE_NAMES = {
    ast.UAdd: "uadd",
    ast.USub: "usub",
}
_COMPILED_BINARY_OPERATORS = {
    "add": numpy.add,
    "sub": numpy.subtract,
    "mul": numpy.multiply,
    "div": numpy.divide,
    "pow": numpy.power,
}
_COMPILED_UNARY_OPERATORS = {
    "uadd": lambda value: value,
    "usub": numpy.negative,
}

_RUNTIME_REFERENCE_NAMES = {
    "Omega_b0",
    "Omega_c0",
    "Omega_de0",
    "Omega_gamma0",
    "Omega_k0",
    "Omega_m0",
    "Omega_nu0",
    "Omega_r0",
    "a_initial",
    "angular_diameter_distance",
    "chi",
    "collision_rate",
    "eta_initial",
    "free_streaming",
    "seed",
    "sound_horizon",
    "sound_speed",
    "sound_speed_sq",
    "tight_coupling_drag",
    "tight_coupling_ratio",
}

_SUPPORTED_PERTURBATION_KEYS = {
    "backend_mapping",
    "boundary_conditions",
    "closures",
    "constraints",
    "contract_version",
    "derived",
    "equations",
    "gauge",
    "initial_conditions",
    "notes",
    "numerics",
    "observables",
    "sources",
    "standard",
    "validity",
    "variables",
}
_SUPPORTED_VARIABLE_KEYS = {
    "description",
    "domain",
    "gauge_role",
    "kind",
    "notes",
    "parity",
    "projection_role",
    "rank",
    "source_role",
    "spin",
    "tensor_character",
    "units",
}
_SUPPORTED_DERIVED_KEYS = {
    "description",
    "domain",
    "expression",
    "kind",
    "notes",
    "order",
    "variable",
    "wrt",
}
_SUPPORTED_EQUATION_KEYS = {
    "dependencies",
    "description",
    "domain",
    "lhs",
    "notes",
    "rhs",
    "role",
}
_SUPPORTED_LHS_KEYS = {"kind", "order", "variable", "wrt"}
_SUPPORTED_RELATION_KEYS = {
    "dependencies",
    "description",
    "domain",
    "expression",
    "notes",
    "role",
    "target",
}
_SUPPORTED_SOURCE_KEYS = {
    "dependencies",
    "description",
    "domain",
    "expression",
    "notes",
    "role",
}
_SUPPORTED_OBSERVABLE_KEYS = {
    "dependencies",
    "description",
    "domain",
    "kernel",
    "kind",
    "notes",
    "primary",
    "projection",
    "required_projection_roles",
    "secondary",
    "source_terms",
}
_SUPPORTED_CONDITION_KEYS = {
    "anchor",
    "dependencies",
    "description",
    "domain",
    "expression",
    "notes",
    "target",
}
_SUPPORTED_CONDITION_TARGET_KEYS = {"order", "variable", "wrt"}
_SUPPORTED_VALIDITY_KEYS = {"notes", "regimes"}
_SUPPORTED_BACKEND_KEYS = {"camb"}
_STANDARD_BACKEND_KEYS = {"uses_standard_perturbations"}
_NONSTANDARD_BACKEND_KEYS = {
    "implemented",
    "native_solver_required",
}
_SUPPORTED_GAUGES = {
    "conformal_newtonian",
    "gauge_invariant",
    "synchronous",
    "unspecified",
}
_SUPPORTED_CONDITION_ANCHORS = {"end", "start"}
_SUPPORTED_OBSERVABLE_KINDS = {
    "angular_power_spectrum",
    "transfer_component",
}
_COMPILED_CONTRACT_RESULTS: dict[
    tuple[Any, ...], "PerturbationContractData"
] = {}


@lru_cache(maxsize=256)
def _get_cached_perturbation_contract(
    cache_key: tuple[Any, ...],
) -> "PerturbationContractData":
    """Return a cached contract for ``cache_key``."""

    return _COMPILED_CONTRACT_RESULTS[cache_key]


@dataclass(frozen=True, slots=True)
class PerturbationVariableData:
    """Immutable metadata for one declared graph variable."""

    name: str
    kind: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    units: str | None = None
    gauge_role: str | None = None
    source_role: str | None = None
    projection_role: str | None = None
    tensor_character: str | None = None
    parity: str | None = None
    rank: int | None = None
    spin: float | None = None


@dataclass(frozen=True, slots=True)
class PerturbationCompiledExpressionData:
    """Picklable stack program for one validated declared expression."""

    expression: str
    dependencies: tuple[str, ...]
    program: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PerturbationDerivedData:
    """Immutable metadata for one declared derived graph symbol."""

    name: str
    kind: str
    expression: str | None = None
    variable: str | None = None
    wrt: str | None = None
    order: int | None = None
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationDerivativeLhsData:
    """Immutable typed left-hand side for one differential equation."""

    kind: str
    variable: str
    wrt: str
    order: int


@dataclass(frozen=True, slots=True)
class PerturbationEquationData:
    """Immutable metadata for one declared differential equation."""

    name: str
    lhs: PerturbationDerivativeLhsData
    rhs: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_rhs: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationClosureData:
    """Immutable metadata for one declared algebraic closure relation."""

    name: str
    target: str
    expression: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationConstraintData:
    """Immutable metadata for one declared algebraic constraint."""

    name: str
    target: str
    expression: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationSourceData:
    """Immutable metadata for one declared observable source term."""

    name: str
    expression: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationObservableData:
    """Immutable metadata for one declared observable mapping."""

    name: str
    kind: str
    projection: str | None = None
    kernel: str | None = None
    primary: str | None = None
    secondary: str | None = None
    source_terms: FrozenMapping = field(default_factory=FrozenMapping)
    required_projection_roles: tuple[str, ...] = ()
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationConditionTargetData:
    """Immutable target descriptor for an initial/boundary condition."""

    variable: str
    wrt: str
    order: int


@dataclass(frozen=True, slots=True)
class PerturbationConditionData:
    """Immutable metadata for an initial or boundary condition."""

    name: str
    target: PerturbationConditionTargetData
    expression: str
    anchor: str = "start"
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationValidityData:
    """Immutable validity declaration for the declared graph."""

    regimes: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationBackendMappingData:
    """Immutable backend execution metadata."""

    backend: str
    uses_standard_perturbations: bool | None = None
    native_solver_required: bool | None = None
    implemented: bool | None = None


@dataclass(frozen=True, slots=True)
class PerturbationDependencyGraphSummaryData:
    """Immutable summary of declared graph dependencies."""

    variable_names: tuple[str, ...]
    derived_names: tuple[str, ...]
    equation_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    closure_names: tuple[str, ...]
    source_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    initial_condition_names: tuple[str, ...]
    boundary_condition_names: tuple[str, ...]
    independent_variables_used: tuple[str, ...]
    model_parameters_used: tuple[str, ...]
    background_references_used: tuple[str, ...]
    derived_dependencies: FrozenMapping
    equation_dependencies: FrozenMapping
    constraint_dependencies: FrozenMapping
    closure_dependencies: FrozenMapping
    source_dependencies: FrozenMapping
    observable_dependencies: FrozenMapping
    initial_condition_dependencies: FrozenMapping
    boundary_condition_dependencies: FrozenMapping
    evaluation_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerturbationContractData:
    """Immutable internal representation of a declared CMB graph."""

    model_name: str
    backend: str
    contract_version: int
    standard: bool
    gauge: str
    variables: FrozenMapping
    derived: FrozenMapping
    equations: FrozenMapping
    constraints: FrozenMapping
    closures: FrozenMapping
    sources: FrozenMapping
    observables: FrozenMapping
    initial_conditions: FrozenMapping
    boundary_conditions: FrozenMapping
    numerics: FrozenMapping
    validity: PerturbationValidityData
    backend_mapping: FrozenMapping
    dependency_graph_summary: PerturbationDependencyGraphSummaryData
    manifest_summary: FrozenMapping


@lru_cache(maxsize=4096)
def _collect_expression_names(expr: str) -> tuple[str, ...]:
    """Return the symbol names referenced by ``expr``."""

    node = _parse_safe_expression(expr)
    names: list[str] = []
    seen: set[str] = set()
    for current in ast.walk(node):
        if not isinstance(current, ast.Name):
            continue
        if current.id in _ALLOWED_MATH_FUNCS:
            continue
        if current.id in _ALLOWED_CONSTANTS:
            continue
        if current.id in seen:
            continue
        seen.add(current.id)
        names.append(current.id)
    return tuple(names)


@lru_cache(maxsize=4096)
def _compile_expression_program(
    expr: str,
) -> tuple[tuple[str, Any], ...]:
    """Return a picklable stack program for one validated expression."""

    node = _parse_safe_expression(expr)
    program: list[tuple[str, Any]] = []

    def _visit(current: ast.AST) -> None:
        """Append stack-machine instructions for ``current``."""

        if isinstance(current, ast.Expression):
            _visit(current.body)
            return
        if isinstance(current, ast.Constant):
            if not isinstance(current.value, (int, float)):
                raise ValueError("non-numeric literal")
            program.append(("const", float(current.value)))
            return
        if isinstance(current, ast.Name):
            program.append(("name", current.id))
            return
        if isinstance(current, ast.BinOp):
            opcode = _COMPILED_BINARY_OPCODE_NAMES.get(type(current.op))
            if opcode is None:
                raise ValueError("operator not allowed")
            _visit(current.left)
            _visit(current.right)
            program.append(("binary", opcode))
            return
        if isinstance(current, ast.UnaryOp):
            opcode = _COMPILED_UNARY_OPCODE_NAMES.get(type(current.op))
            if opcode is None:
                raise ValueError("operator not allowed")
            _visit(current.operand)
            program.append(("unary", opcode))
            return
        if isinstance(current, ast.Call):
            if not isinstance(current.func, ast.Name):
                raise ValueError("invalid function call")
            if current.keywords:
                raise ValueError("keyword arguments not supported")
            for argument in current.args:
                _visit(argument)
            program.append(
                ("call", (current.func.id, len(tuple(current.args))))
            )
            return
        raise ValueError("expression not allowed")

    _visit(node)
    return tuple(program)


@lru_cache(maxsize=4096)
def _compile_expression_plan(
    expr: str,
    *,
    dependencies: tuple[str, ...] | None = None,
) -> PerturbationCompiledExpressionData:
    """Return picklable evaluator metadata for one validated expression."""

    dependency_names = (
        _collect_expression_names(expr)
        if dependencies is None
        else tuple(dependencies)
    )
    return PerturbationCompiledExpressionData(
        expression=expr,
        dependencies=dependency_names,
        program=_compile_expression_program(expr),
    )


def _evaluate_compiled_expression_noerr(
    expression_data: PerturbationCompiledExpressionData,
    env: Mapping[str, Any],
) -> Any:
    """Evaluate one compiled expression against ``env`` without errstate."""

    stack: list[Any] = []
    for opcode, payload in expression_data.program:
        if opcode == "const":
            stack.append(payload)
            continue
        if opcode == "name":
            if payload in env:
                stack.append(env[payload])
                continue
            if payload in _ALLOWED_CONSTANTS:
                stack.append(_ALLOWED_CONSTANTS[payload])
                continue
            raise ValueError(f"name '{payload}' not allowed")
        if opcode == "binary":
            right = stack.pop()
            left = stack.pop()
            stack.append(_COMPILED_BINARY_OPERATORS[payload](left, right))
            continue
        if opcode == "unary":
            stack.append(_COMPILED_UNARY_OPERATORS[payload](stack.pop()))
            continue
        if opcode == "call":
            func_name, arg_count = payload
            func = _ALLOWED_MATH_FUNCS.get(func_name)
            if func is None:
                raise ValueError(f"function '{func_name}' not allowed")
            args = [stack.pop() for _ in range(int(arg_count))]
            args.reverse()
            stack.append(func(*args))
            continue
        raise ValueError("expression not allowed")
    if len(stack) != 1:
        raise ValueError(
            "Compiled expression evaluation did not produce one result"
        )
    return stack[0]


def evaluate_compiled_expression(
    expression_data: PerturbationCompiledExpressionData,
    env: Mapping[str, Any],
) -> Any:
    """Evaluate one compiled declared expression against ``env``."""

    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _evaluate_compiled_expression_noerr(expression_data, env)


def _validate_entry_keys(
    *,
    entry: Mapping[str, Any],
    allowed_keys: set[str],
    label: str,
) -> None:
    """Reject unknown keys inside one contract entry."""

    entry_keys = {str(key) for key in entry.keys()}
    invalid_keys = entry_keys - allowed_keys
    if invalid_keys:
        invalid_str = ", ".join(sorted(invalid_keys))
        raise ValueError(f"Unknown key(s) in {label}: {invalid_str}")


def _validate_string(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    """Return ``value`` as a non-empty string unless ``allow_empty``."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ValueError(f"{label} must not be empty")
    return cleaned


def _validate_optional_string(
    value: Any,
    *,
    label: str,
) -> str | None:
    """Return ``value`` as ``str`` when present."""

    if value is None:
        return None
    return _validate_string(value, label=label)


def _validate_optional_int(
    value: Any,
    *,
    label: str,
) -> int | None:
    """Return ``value`` as an integer when present."""

    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _validate_optional_float(
    value: Any,
    *,
    label: str,
) -> float | None:
    """Return ``value`` as a float when present."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _validate_regimes(value: Any) -> tuple[str, ...]:
    """Return a validated tuple of declared validity regimes."""

    if not isinstance(value, list):
        raise ValueError("cmb.perturbations.validity.regimes must be a list")
    cleaned: list[str] = []
    for item in value:
        cleaned.append(
            _validate_string(
                item,
                label="cmb.perturbations.validity.regimes entry",
            )
        )
    if not cleaned:
        raise ValueError(
            "cmb.perturbations.validity.regimes must not be empty"
        )
    return tuple(cleaned)


def _validate_optional_string_list(
    value: Any,
    *,
    label: str,
) -> tuple[str, ...]:
    """Return ``value`` as a deduplicated string tuple when present."""

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    cleaned: list[str] = []
    for item in value:
        cleaned.append(_validate_string(item, label=f"{label} entry"))
    return _dedupe_names(cleaned)


def _replace_and_validate_expression(
    expression: Any,
    *,
    label: str,
    replacements: Mapping[str, str],
    allowed_names: set[str],
) -> tuple[str, tuple[str, ...]]:
    """Return a cleaned, validated expression and its dependencies."""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"{label} must be a non-empty string expression")
    clean_expression = _replace_latex_tokens(expression, replacements)
    names = _collect_expression_names(clean_expression)
    unknown = sorted(set(names) - allowed_names)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(
            f"{label} references unknown symbol(s): {unknown_str}"
        )
    _validate_safe_expression(clean_expression, allowed_names)
    return clean_expression, names


def _dedupe_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return ``names`` in the order of first appearance without duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _relation_target_nodes(
    constraints: Mapping[str, PerturbationConstraintData],
    closures: Mapping[str, PerturbationClosureData],
) -> dict[str, tuple[str, str]]:
    """Return graph nodes for algebraic relation targets."""

    nodes: dict[str, tuple[str, str]] = {}
    for name, entry in constraints.items():
        nodes[entry.target] = ("constraint", name)
    for name, entry in closures.items():
        if entry.target in nodes:
            previous_kind, previous_name = nodes[entry.target]
            raise ValueError(
                "Declared graph defines algebraic target "
                f"'{entry.target}' more than once via "
                f"{previous_kind} '{previous_name}' and closure '{name}'"
            )
        nodes[entry.target] = ("closure", name)
    return nodes


def _relation_target_entries(
    constraints: Mapping[str, PerturbationConstraintData],
    closures: Mapping[str, PerturbationClosureData],
) -> dict[str, PerturbationConstraintData | PerturbationClosureData]:
    """Return algebraic relation entries keyed by target variable name."""

    relation_entries: dict[
        str, PerturbationConstraintData | PerturbationClosureData
    ] = {}
    for entry in constraints.values():
        relation_entries[entry.target] = entry
    for entry in closures.values():
        relation_entries[entry.target] = entry
    return relation_entries


def _topological_evaluation_order(
    *,
    derived: Mapping[str, PerturbationDerivedData],
    constraints: Mapping[str, PerturbationConstraintData],
    closures: Mapping[str, PerturbationClosureData],
) -> tuple[str, ...]:
    """Return a topological order for expression and algebraic nodes."""

    relation_nodes = _relation_target_nodes(constraints, closures)
    graph: dict[str, tuple[str, ...]] = {}
    expression_names = {
        name for name, entry in derived.items() if entry.expression is not None
    }
    node_names = expression_names | set(relation_nodes)
    for name, entry in derived.items():
        if entry.expression is None:
            continue
        graph[name] = tuple(
            dependency
            for dependency in entry.dependencies
            if dependency in node_names
        )
    for target_name, (kind, entry_name) in relation_nodes.items():
        if kind == "constraint":
            entry = constraints[entry_name]
        else:
            entry = closures[entry_name]
        graph[target_name] = tuple(
            dependency
            for dependency in entry.dependencies
            if dependency in node_names
        )

    active: set[str] = set()
    completed: set[str] = set()
    visiting: list[str] = []
    order: list[str] = []

    def _visit(node: str) -> None:
        """Depth-first walk that detects declared-graph dependency cycles."""

        if node in completed:
            return
        if node in active:
            cycle_start = visiting.index(node)
            cycle = visiting[cycle_start:] + [node]
            raise ValueError(
                "Declared graph contains a cycle: " + " -> ".join(cycle)
            )
        active.add(node)
        visiting.append(node)
        for dependency in graph.get(node, ()):
            _visit(dependency)
        visiting.pop()
        active.remove(node)
        completed.add(node)
        order.append(node)

    for node in sorted(graph):
        _visit(node)
    return tuple(order)


def _build_manifest_summary(
    *,
    model_name: str,
    backend: str,
    contract_version: int,
    standard: bool,
    gauge: str,
    variables: tuple[str, ...],
    derived: tuple[str, ...],
    equations: tuple[str, ...],
    constraints: tuple[str, ...],
    closures: tuple[str, ...],
    sources: tuple[str, ...],
    observables: tuple[str, ...],
    initial_conditions: tuple[str, ...],
    boundary_conditions: tuple[str, ...],
    validity: PerturbationValidityData,
    numerics: Mapping[str, Any],
    backend_mapping: PerturbationBackendMappingData,
    dependency_summary: PerturbationDependencyGraphSummaryData,
    equation_wrt_by_variable: Mapping[str, str],
    boundary_condition_anchors: Mapping[str, str],
    transfer_component_contracts: Mapping[str, Mapping[str, Any]],
    angular_power_spectrum_targets: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Return a manifest-friendly summary of the compiled graph."""

    execution_route = _build_execution_route_summary(
        backend=backend,
        standard=standard,
        backend_mapping=backend_mapping,
    )
    return {
        "model_name": model_name,
        "backend": backend,
        "contract_version": contract_version,
        "standard": standard,
        "gauge": gauge,
        "variable_names": variables,
        "derived_names": derived,
        "equation_names": equations,
        "constraint_names": constraints,
        "closure_names": closures,
        "source_names": sources,
        "observable_names": observables,
        "initial_condition_names": initial_conditions,
        "boundary_condition_names": boundary_conditions,
        "validity_regimes": validity.regimes,
        "validity_notes": validity.notes,
        "numerics_keys": tuple(sorted(str(key) for key in numerics)),
        "backend_implemented": backend_mapping.implemented,
        "backend_native_solver_required": (
            backend_mapping.native_solver_required
        ),
        "backend_uses_standard_perturbations": (
            backend_mapping.uses_standard_perturbations
        ),
        "independent_variables_used": (
            dependency_summary.independent_variables_used
        ),
        "model_parameters_used": dependency_summary.model_parameters_used,
        "background_references_used": (
            dependency_summary.background_references_used
        ),
        "evaluation_order": dependency_summary.evaluation_order,
        "equation_wrt_by_variable": {
            str(name): str(wrt_name)
            for name, wrt_name in equation_wrt_by_variable.items()
        },
        "boundary_condition_anchors": {
            str(name): str(anchor_name)
            for name, anchor_name in boundary_condition_anchors.items()
        },
        "execution_route": execution_route,
        "transfer_component_contracts": {
            str(name): {
                str(key): value for key, value in contract_data.items()
            }
            for name, contract_data in transfer_component_contracts.items()
        },
        "angular_power_spectrum_targets": {
            str(name): {
                str(key): str(value) for key, value in target_data.items()
            }
            for name, target_data in angular_power_spectrum_targets.items()
        },
    }


def _build_execution_route_summary(
    *,
    backend: str,
    standard: bool,
    backend_mapping: PerturbationBackendMappingData,
) -> dict[str, Any]:
    """Return manifest-friendly execution-route metadata."""

    if standard:
        route_id = "backend_standard_perturbations"
        prediction_engine = backend
        transfer_function_path = f"{backend}.standard"
        solver = f"{backend}_standard"
    else:
        route_id = "native_declared_graph"
        prediction_engine = "copernican_native_declared_graph"
        transfer_function_path = (
            "copernican.lib.likelihoods.cmb.copernican_cmb_solver"
        )
        solver = "declared_math_graph"
    uses_camb_prediction = bool(
        standard and str(backend).strip().lower() == "camb"
    )
    return {
        "route_id": route_id,
        "prediction_engine": prediction_engine,
        "transfer_function_path": transfer_function_path,
        "solver": solver,
        "route_ready_for_execution": bool(
            standard
            or (
                backend_mapping.native_solver_required is True
                and backend_mapping.implemented is True
            )
        ),
        "uses_backend_standard_perturbations": bool(standard),
        "uses_native_declared_graph": bool(not standard),
        "uses_camb_prediction": uses_camb_prediction,
        "uses_camb_standard_perturbations": uses_camb_prediction,
        "backend_mapping_implemented": backend_mapping.implemented,
        "backend_mapping_native_solver_required": (
            backend_mapping.native_solver_required
        ),
        "backend_mapping_uses_standard_perturbations": (
            backend_mapping.uses_standard_perturbations
        ),
    }


def compile_perturbation_contract(
    contract: Mapping[str, Any],
    *,
    model_name: str,
    backend: str,
    parameter_names: Sequence[str],
    latex_names: Sequence[str],
    background_reference_names: Sequence[str],
) -> PerturbationContractData:
    """Validate and compile a declared CMB graph contract."""

    if not isinstance(contract, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")

    cache_key = (
        _freeze_for_cache(contract),
        str(model_name),
        str(backend),
        tuple(str(name) for name in parameter_names),
        tuple(str(name) for name in latex_names),
        tuple(str(name) for name in background_reference_names),
    )
    cached_result = _COMPILED_CONTRACT_RESULTS.get(cache_key)
    if cached_result is not None:
        return cached_result

    contract_keys = {str(key) for key in contract.keys()}
    required_sections = {
        "backend_mapping",
        "contract_version",
        "gauge",
        "standard",
        "validity",
    }
    missing_keys = required_sections - contract_keys
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Missing perturbation contract key(s): {missing_str}"
        )
    invalid_keys = contract_keys - _SUPPORTED_PERTURBATION_KEYS
    if invalid_keys:
        invalid_str = ", ".join(sorted(invalid_keys))
        raise ValueError(
            f"Unknown perturbation contract key(s): {invalid_str}"
        )

    contract_version = contract.get("contract_version")
    if isinstance(contract_version, bool) or not isinstance(
        contract_version, int
    ):
        raise ValueError("cmb.perturbations.contract_version must be an int")
    standard = contract.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")
    if standard:
        if contract_version not in {1, 2}:
            raise ValueError(
                "Standard perturbations must declare contract_version 1 or 2"
            )
    elif contract_version != 2:
        raise ValueError(
            "Non-standard perturbations must declare contract_version: 2"
        )

    gauge = _validate_string(
        contract.get("gauge"),
        label="cmb.perturbations.gauge",
    )
    if gauge not in _SUPPORTED_GAUGES:
        raise ValueError("cmb.perturbations.gauge is invalid")
    _validate_optional_string(
        contract.get("notes"),
        label="cmb.perturbations.notes",
    )

    sections = {
        "variables": contract.get("variables", {}),
        "derived": contract.get("derived", {}),
        "equations": contract.get("equations", {}),
        "constraints": contract.get("constraints", {}),
        "closures": contract.get("closures", {}),
        "sources": contract.get("sources", {}),
        "observables": contract.get("observables", {}),
        "initial_conditions": contract.get("initial_conditions", {}),
        "boundary_conditions": contract.get("boundary_conditions", {}),
        "validity": contract.get("validity", {}),
        "backend_mapping": contract.get("backend_mapping"),
        "numerics": contract.get("numerics", {}),
    }
    for section_name, section_value in sections.items():
        if not isinstance(section_value, Mapping):
            raise ValueError(
                f"cmb.perturbations.{section_name} must be a mapping"
            )

    parameter_name_set = {str(name) for name in parameter_names}
    background_reference_set = {
        str(name) for name in background_reference_names
    } | _RUNTIME_REFERENCE_NAMES
    replacements = _build_parameter_replacements(
        parameter_names,
        latex_names,
    )

    backend_keys = {str(key) for key in sections["backend_mapping"].keys()}
    invalid_backend_keys = backend_keys - _SUPPORTED_BACKEND_KEYS
    if invalid_backend_keys:
        invalid_str = ", ".join(sorted(invalid_backend_keys))
        raise ValueError(f"Unknown perturbation backend(s): {invalid_str}")
    backend_contract = sections["backend_mapping"].get(backend)
    if not isinstance(backend_contract, Mapping):
        raise ValueError(
            f"cmb.perturbations.backend_mapping must include {backend}"
        )
    backend_contract_keys = {str(key) for key in backend_contract.keys()}
    if standard:
        invalid_standard_keys = backend_contract_keys - _STANDARD_BACKEND_KEYS
        if invalid_standard_keys:
            invalid_str = ", ".join(sorted(invalid_standard_keys))
            raise ValueError(
                "Standard perturbation mappings may only declare "
                f"uses_standard_perturbations: {invalid_str}"
            )
        if backend_contract.get("uses_standard_perturbations") is not True:
            raise ValueError(
                "cmb.perturbations.backend_mapping.camb must declare "
                "uses_standard_perturbations: true"
            )
    else:
        invalid_nonstandard_keys = (
            backend_contract_keys - _NONSTANDARD_BACKEND_KEYS
        )
        if invalid_nonstandard_keys:
            invalid_str = ", ".join(sorted(invalid_nonstandard_keys))
            raise ValueError(
                "Non-standard perturbation mappings may only declare "
                f"native_solver_required, implemented: {invalid_str}"
            )
        if backend_contract.get("native_solver_required") is not True:
            raise ValueError(
                "cmb.perturbations.backend_mapping.camb must declare "
                "native_solver_required: true"
            )
        implemented = backend_contract.get("implemented")
        if not isinstance(implemented, bool):
            raise ValueError(
                "cmb.perturbations.backend_mapping.camb.implemented must be "
                "boolean"
            )

    variable_entries: dict[str, PerturbationVariableData] = {}
    for variable_name, variable_def in sections["variables"].items():
        name = _validate_string(
            variable_name,
            label="Perturbation variable name",
        )
        if not isinstance(variable_def, Mapping):
            raise ValueError(
                f"Perturbation variable '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=variable_def,
            allowed_keys=_SUPPORTED_VARIABLE_KEYS,
            label=f"cmb.perturbations.variables.{name}",
        )
        if name in parameter_name_set or name in background_reference_set:
            raise ValueError(
                f"Perturbation variable '{name}' collides with an "
                "existing background symbol"
            )
        if name in variable_entries:
            raise ValueError(f"Perturbation variable '{name}' is duplicated")
        variable_entries[name] = PerturbationVariableData(
            name=name,
            kind=_validate_string(
                variable_def.get("kind"),
                label=f"cmb.perturbations.variables.{name}.kind",
            ),
            description=_validate_optional_string(
                variable_def.get("description"),
                label=f"cmb.perturbations.variables.{name}.description",
            ),
            notes=_validate_optional_string(
                variable_def.get("notes"),
                label=f"cmb.perturbations.variables.{name}.notes",
            ),
            domain=_validate_optional_string(
                variable_def.get("domain"),
                label=f"cmb.perturbations.variables.{name}.domain",
            ),
            units=_validate_optional_string(
                variable_def.get("units"),
                label=f"cmb.perturbations.variables.{name}.units",
            ),
            gauge_role=_validate_optional_string(
                variable_def.get("gauge_role"),
                label=f"cmb.perturbations.variables.{name}.gauge_role",
            ),
            source_role=_validate_optional_string(
                variable_def.get("source_role"),
                label=f"cmb.perturbations.variables.{name}.source_role",
            ),
            projection_role=_validate_optional_string(
                variable_def.get("projection_role"),
                label=(f"cmb.perturbations.variables.{name}.projection_role"),
            ),
            tensor_character=_validate_optional_string(
                variable_def.get("tensor_character"),
                label=(f"cmb.perturbations.variables.{name}.tensor_character"),
            ),
            parity=_validate_optional_string(
                variable_def.get("parity"),
                label=f"cmb.perturbations.variables.{name}.parity",
            ),
            rank=_validate_optional_int(
                variable_def.get("rank"),
                label=f"cmb.perturbations.variables.{name}.rank",
            ),
            spin=_validate_optional_float(
                variable_def.get("spin"),
                label=f"cmb.perturbations.variables.{name}.spin",
            ),
        )

    allowed_name_pool: set[str] = set(parameter_name_set)
    allowed_name_pool.update(background_reference_set)
    allowed_name_pool.update(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
    allowed_name_pool.update(variable_entries)

    derived_entries: dict[str, PerturbationDerivedData] = {}
    declared_derived_names = {str(name) for name in sections["derived"]}
    expression_derived_names: list[str] = []
    for derived_name, derived_def in sections["derived"].items():
        name = _validate_string(
            derived_name,
            label="Derived perturbation name",
        )
        if not isinstance(derived_def, Mapping):
            raise ValueError(
                f"Perturbation derived '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=derived_def,
            allowed_keys=_SUPPORTED_DERIVED_KEYS,
            label=f"cmb.perturbations.derived.{name}",
        )
        if (
            name in parameter_name_set
            or name in background_reference_set
            or name in variable_entries
            or name in derived_entries
        ):
            raise ValueError(
                f"Perturbation derived '{name}' collides with an "
                "existing symbol"
            )
        expression = derived_def.get("expression")
        derivative_variable = derived_def.get("variable")
        derivative_wrt = derived_def.get("wrt")
        derivative_order = derived_def.get("order")
        if expression is None:
            if derivative_variable is None:
                raise ValueError(
                    f"Perturbation derived '{name}' must declare either "
                    "expression or variable"
                )
            variable_name = _validate_string(
                derivative_variable,
                label=f"cmb.perturbations.derived.{name}.variable",
            )
            if (
                variable_name not in variable_entries
                and variable_name not in background_reference_set
            ):
                raise ValueError(
                    f"Derivative symbol '{name}' references unknown "
                    f"variable '{variable_name}'"
                )
            wrt_name = _validate_string(
                derivative_wrt,
                label=f"cmb.perturbations.derived.{name}.wrt",
            )
            if wrt_name not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
                raise ValueError(
                    f"Derivative symbol '{name}' uses unsupported wrt "
                    f"'{wrt_name}'"
                )
            order_value = _validate_optional_int(
                derivative_order,
                label=f"cmb.perturbations.derived.{name}.order",
            )
            if order_value is None or order_value < 1:
                raise ValueError(
                    f"Derivative symbol '{name}' order must be a positive "
                    "integer"
                )
            derived_entries[name] = PerturbationDerivedData(
                name=name,
                kind=_validate_string(
                    derived_def.get("kind") or "derivative_symbol",
                    label=f"cmb.perturbations.derived.{name}.kind",
                ),
                variable=variable_name,
                wrt=wrt_name,
                order=order_value,
                description=_validate_optional_string(
                    derived_def.get("description"),
                    label=f"cmb.perturbations.derived.{name}.description",
                ),
                notes=_validate_optional_string(
                    derived_def.get("notes"),
                    label=f"cmb.perturbations.derived.{name}.notes",
                ),
                domain=_validate_optional_string(
                    derived_def.get("domain"),
                    label=f"cmb.perturbations.derived.{name}.domain",
                ),
                dependencies=(variable_name,),
            )
            continue
        if derivative_variable is not None:
            raise ValueError(
                f"Perturbation derived '{name}' cannot declare both "
                "expression and variable"
            )
        clean_expression, dependencies = _replace_and_validate_expression(
            expression,
            label=f"cmb.perturbations.derived.{name}.expression",
            replacements=replacements,
            allowed_names=allowed_name_pool
            | declared_derived_names
            | set(derived_entries)
            | set(expression_derived_names),
        )
        derived_entries[name] = PerturbationDerivedData(
            name=name,
            kind=_validate_string(
                derived_def.get("kind") or "expression",
                label=f"cmb.perturbations.derived.{name}.kind",
            ),
            expression=clean_expression,
            description=_validate_optional_string(
                derived_def.get("description"),
                label=f"cmb.perturbations.derived.{name}.description",
            ),
            notes=_validate_optional_string(
                derived_def.get("notes"),
                label=f"cmb.perturbations.derived.{name}.notes",
            ),
            domain=_validate_optional_string(
                derived_def.get("domain"),
                label=f"cmb.perturbations.derived.{name}.domain",
            ),
            dependencies=dependencies,
            compiled_expression=_compile_expression_plan(
                clean_expression,
                dependencies=dependencies,
            ),
        )
        expression_derived_names.append(name)

    expression_names = {
        name
        for name, entry in derived_entries.items()
        if entry.expression is not None
    }
    all_expression_names = (
        allowed_name_pool | set(derived_entries) | expression_names
    )

    equation_entries: dict[str, PerturbationEquationData] = {}
    equation_targets: set[tuple[str, int]] = set()
    for equation_name, equation_def in sections["equations"].items():
        name = _validate_string(
            equation_name,
            label="Differential equation name",
        )
        if not isinstance(equation_def, Mapping):
            raise ValueError(
                f"Perturbation equation '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=equation_def,
            allowed_keys=_SUPPORTED_EQUATION_KEYS,
            label=f"cmb.perturbations.equations.{name}",
        )
        lhs = equation_def.get("lhs")
        if not isinstance(lhs, Mapping):
            raise ValueError(
                f"Perturbation equation '{name}' must declare typed lhs"
            )
        _validate_entry_keys(
            entry=lhs,
            allowed_keys=_SUPPORTED_LHS_KEYS,
            label=f"cmb.perturbations.equations.{name}.lhs",
        )
        lhs_kind = _validate_string(
            lhs.get("kind"),
            label=f"cmb.perturbations.equations.{name}.lhs.kind",
        )
        if lhs_kind != "derivative":
            raise ValueError(
                f"Perturbation equation '{name}' lhs kind must be "
                "derivative"
            )
        lhs_variable = _validate_string(
            lhs.get("variable"),
            label=f"cmb.perturbations.equations.{name}.lhs.variable",
        )
        if lhs_variable not in variable_entries:
            raise ValueError(
                f"Perturbation equation '{name}' references unknown "
                f"variable '{lhs_variable}'"
            )
        lhs_wrt = _validate_string(
            lhs.get("wrt"),
            label=f"cmb.perturbations.equations.{name}.lhs.wrt",
        )
        if lhs_wrt not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
            raise ValueError(
                f"Perturbation equation '{name}' uses unsupported wrt "
                f"'{lhs_wrt}'"
            )
        lhs_order = _validate_optional_int(
            lhs.get("order"),
            label=f"cmb.perturbations.equations.{name}.lhs.order",
        )
        if lhs_order is None or lhs_order < 1:
            raise ValueError(
                f"Perturbation equation '{name}' lhs order must be a "
                "positive integer"
            )
        target_key = (lhs_variable, lhs_order)
        if target_key in equation_targets:
            raise ValueError(
                f"Perturbation equation '{name}' duplicates the derivative "
                f"target for variable '{lhs_variable}' order {lhs_order}"
            )
        rhs_expression, dependencies = _replace_and_validate_expression(
            equation_def.get("rhs"),
            label=f"cmb.perturbations.equations.{name}.rhs",
            replacements=replacements,
            allowed_names=all_expression_names,
        )
        equation_entries[name] = PerturbationEquationData(
            name=name,
            lhs=PerturbationDerivativeLhsData(
                kind="derivative",
                variable=lhs_variable,
                wrt=lhs_wrt,
                order=lhs_order,
            ),
            rhs=rhs_expression,
            role=_validate_string(
                equation_def.get("role") or "differential",
                label=f"cmb.perturbations.equations.{name}.role",
            ),
            description=_validate_optional_string(
                equation_def.get("description"),
                label=f"cmb.perturbations.equations.{name}.description",
            ),
            notes=_validate_optional_string(
                equation_def.get("notes"),
                label=f"cmb.perturbations.equations.{name}.notes",
            ),
            domain=_validate_optional_string(
                equation_def.get("domain"),
                label=f"cmb.perturbations.equations.{name}.domain",
            ),
            dependencies=dependencies,
            compiled_rhs=_compile_expression_plan(
                rhs_expression,
                dependencies=dependencies,
            ),
        )
        equation_targets.add(target_key)

    evolved_variable_names = {
        entry.lhs.variable for entry in equation_entries.values()
    }

    def _compile_relations(
        relation_defs: Mapping[str, Any],
        *,
        label_prefix: str,
        relation_kind: str,
    ) -> dict[str, Any]:
        """Compile constraint or closure mappings into typed relation data."""

        compiled: dict[str, Any] = {}
        seen_targets: set[str] = set()
        for relation_name, relation_def in relation_defs.items():
            name = _validate_string(
                relation_name,
                label=f"{relation_kind.title()} relation name",
            )
            if not isinstance(relation_def, Mapping):
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' must be a mapping"
                )
            _validate_entry_keys(
                entry=relation_def,
                allowed_keys=_SUPPORTED_RELATION_KEYS,
                label=f"{label_prefix}.{name}",
            )
            target_name = _validate_string(
                relation_def.get("target"),
                label=f"{label_prefix}.{name}.target",
            )
            if target_name in parameter_name_set:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' cannot target "
                    f"parameter '{target_name}'"
                )
            if target_name in background_reference_set:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' cannot target "
                    f"background symbol '{target_name}'"
                )
            if target_name in seen_targets:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' duplicates "
                    f"target '{target_name}'"
                )
            if target_name in evolved_variable_names:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' cannot target "
                    f"evolved variable '{target_name}'"
                )
            expression_text, dependencies = _replace_and_validate_expression(
                relation_def.get("expression"),
                label=f"{label_prefix}.{name}.expression",
                replacements=replacements,
                allowed_names=all_expression_names
                | {target_name}
                | seen_targets,
            )
            entry_kwargs = {
                "name": name,
                "target": target_name,
                "expression": expression_text,
                "role": _validate_string(
                    relation_def.get("role") or relation_kind,
                    label=f"{label_prefix}.{name}.role",
                ),
                "description": _validate_optional_string(
                    relation_def.get("description"),
                    label=f"{label_prefix}.{name}.description",
                ),
                "notes": _validate_optional_string(
                    relation_def.get("notes"),
                    label=f"{label_prefix}.{name}.notes",
                ),
                "domain": _validate_optional_string(
                    relation_def.get("domain"),
                    label=f"{label_prefix}.{name}.domain",
                ),
                "dependencies": dependencies,
                "compiled_expression": _compile_expression_plan(
                    expression_text,
                    dependencies=dependencies,
                ),
            }
            if relation_kind == "constraint":
                compiled[name] = PerturbationConstraintData(**entry_kwargs)
            else:
                compiled[name] = PerturbationClosureData(**entry_kwargs)
            seen_targets.add(target_name)
        return compiled

    constraint_entries = _compile_relations(
        sections["constraints"],
        label_prefix="cmb.perturbations.constraints",
        relation_kind="constraint",
    )
    closure_entries = _compile_relations(
        sections["closures"],
        label_prefix="cmb.perturbations.closures",
        relation_kind="closure",
    )

    source_entries: dict[str, PerturbationSourceData] = {}
    for source_name, source_def in sections["sources"].items():
        name = _validate_string(
            source_name,
            label="Source-term name",
        )
        if not isinstance(source_def, Mapping):
            raise ValueError(f"Perturbation source '{name}' must be a mapping")
        _validate_entry_keys(
            entry=source_def,
            allowed_keys=_SUPPORTED_SOURCE_KEYS,
            label=f"cmb.perturbations.sources.{name}",
        )
        expression_text, dependencies = _replace_and_validate_expression(
            source_def.get("expression"),
            label=f"cmb.perturbations.sources.{name}.expression",
            replacements=replacements,
            allowed_names=all_expression_names
            | set(_relation_target_nodes(constraint_entries, closure_entries)),
        )
        source_entries[name] = PerturbationSourceData(
            name=name,
            expression=expression_text,
            role=_validate_string(
                source_def.get("role"),
                label=f"cmb.perturbations.sources.{name}.role",
            ),
            description=_validate_optional_string(
                source_def.get("description"),
                label=f"cmb.perturbations.sources.{name}.description",
            ),
            notes=_validate_optional_string(
                source_def.get("notes"),
                label=f"cmb.perturbations.sources.{name}.notes",
            ),
            domain=_validate_optional_string(
                source_def.get("domain"),
                label=f"cmb.perturbations.sources.{name}.domain",
            ),
            dependencies=dependencies,
            compiled_expression=_compile_expression_plan(
                expression_text,
                dependencies=dependencies,
            ),
        )

    relation_entries = _relation_target_entries(
        constraint_entries,
        closure_entries,
    )

    def _reachable_variable_names(
        dependencies: Sequence[str],
        *,
        seen: set[str] | None = None,
    ) -> set[str]:
        """Return transitive variable ancestry for ``dependencies``."""

        if seen is None:
            seen = set()
        reachable: set[str] = set()
        for dependency in dependencies:
            if dependency in seen:
                continue
            seen.add(dependency)
            if dependency in variable_entries:
                reachable.add(dependency)
                continue
            if dependency in derived_entries:
                reachable.update(
                    _reachable_variable_names(
                        derived_entries[dependency].dependencies,
                        seen=seen,
                    )
                )
                continue
            if dependency in relation_entries:
                reachable.update(
                    _reachable_variable_names(
                        relation_entries[dependency].dependencies,
                        seen=seen,
                    )
                )
        return reachable

    def _supports_odd_parity_projection(
        variable_name: str,
    ) -> bool:
        """Return whether ``variable_name`` can feed a B-like projection."""

        variable_entry = variable_entries[variable_name]
        explicit_projection_role = (
            variable_entry.source_role == "polarization_b"
            or variable_entry.projection_role == "b_mode"
        )
        odd_parity = variable_entry.parity == "odd"
        has_non_scalar_character = (
            (
                variable_entry.spin is not None
                and abs(float(variable_entry.spin)) >= 1.0
            )
            or (
                variable_entry.rank is not None
                and int(variable_entry.rank) >= 1
            )
            or variable_entry.tensor_character
            in {"vector_like", "tensor_like"}
        )
        return explicit_projection_role or (
            odd_parity and has_non_scalar_character
        )

    def _supports_projection_role(
        variable_name: str,
        projection_role: str,
    ) -> bool:
        """Return whether ``variable_name`` satisfies ``projection_role``."""

        if projection_role == "b_mode":
            return _supports_odd_parity_projection(variable_name)
        variable_entry = variable_entries[variable_name]
        return variable_entry.projection_role == projection_role

    def _source_ancestry_supports_projection_roles(
        source_name: str,
        required_roles: Sequence[str],
    ) -> bool:
        """Return whether ``source_name`` ancestry satisfies all roles."""

        reachable_variables = sorted(
            _reachable_variable_names(source_entries[source_name].dependencies)
        )
        if not reachable_variables:
            return False
        return all(
            any(
                _supports_projection_role(variable_name, role_name)
                for variable_name in reachable_variables
            )
            for role_name in required_roles
        )

    observable_entries: dict[str, PerturbationObservableData] = {}
    observable_names: set[str] = set()
    transfer_component_names: set[str] = set()
    for observable_name, observable_def in sections["observables"].items():
        name = _validate_string(
            observable_name,
            label="Observable name",
        )
        if not isinstance(observable_def, Mapping):
            raise ValueError(
                f"Perturbation observable '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=observable_def,
            allowed_keys=_SUPPORTED_OBSERVABLE_KEYS,
            label=f"cmb.perturbations.observables.{name}",
        )
        observable_kind = _validate_string(
            observable_def.get("kind"),
            label=f"cmb.perturbations.observables.{name}.kind",
        )
        if observable_kind not in _SUPPORTED_OBSERVABLE_KINDS:
            raise ValueError(
                f"Perturbation observable '{name}' uses unsupported kind "
                f"'{observable_kind}'"
            )
        projection = _validate_optional_string(
            observable_def.get("projection"),
            label=f"cmb.perturbations.observables.{name}.projection",
        )
        primary = _validate_optional_string(
            observable_def.get("primary"),
            label=f"cmb.perturbations.observables.{name}.primary",
        )
        secondary = _validate_optional_string(
            observable_def.get("secondary"),
            label=f"cmb.perturbations.observables.{name}.secondary",
        )
        source_term_mapping = observable_def.get("source_terms", {})
        if source_term_mapping is None:
            source_term_mapping = {}
        if not isinstance(source_term_mapping, Mapping):
            raise ValueError(
                f"Perturbation observable '{name}' source_terms must be a "
                "mapping"
            )
        source_term_refs: dict[str, str] = {}
        for role_name, source_name in source_term_mapping.items():
            role = _validate_string(
                role_name,
                label=f"cmb.perturbations.observables.{name}.source_terms key",
            )
            source_ref = _validate_string(
                source_name,
                label=(
                    f"cmb.perturbations.observables.{name}.source_terms."
                    f"{role}"
                ),
            )
            if source_ref not in source_entries:
                raise ValueError(
                    f"Perturbation observable '{name}' references unknown "
                    f"source term '{source_ref}'"
                )
            source_term_refs[role] = source_ref
        kernel = _validate_optional_string(
            observable_def.get("kernel"),
            label=f"cmb.perturbations.observables.{name}.kernel",
        )
        required_projection_roles = _validate_optional_string_list(
            observable_def.get("required_projection_roles"),
            label=(
                f"cmb.perturbations.observables.{name}"
                ".required_projection_roles"
            ),
        )
        effective_projection_roles = required_projection_roles
        if observable_kind == "transfer_component":
            if projection is None:
                raise ValueError(
                    f"Perturbation observable '{name}' must declare "
                    "projection"
                )
            if primary is not None or secondary is not None:
                raise ValueError(
                    f"Perturbation observable '{name}' kind "
                    "'transfer_component' must not declare primary or "
                    "secondary"
                )
            declared_source_roles = {
                role_name: source_entries[source_name].role
                for role_name, source_name in source_term_refs.items()
            }
            validate_declared_projection_source_roles(
                str(projection),
                observable_name=name,
                source_roles=declared_source_roles,
            )
            for role_name, source_role in declared_source_roles.items():
                if source_role is None:
                    continue
                if source_role != role_name:
                    source_name = source_term_refs[role_name]
                    raise ValueError(
                        f"Perturbation observable '{name}' binds source term "
                        f"role '{role_name}' to source '{source_name}' "
                        f"with declared role '{source_role}'"
                    )
            projection_spec = get_declared_projection_spec(str(projection))
            kernel = resolve_declared_projection_kernel(
                str(projection),
                observable_name=name,
                kernel=kernel,
            )
            effective_projection_roles = _dedupe_names(
                projection_spec.required_projection_roles
                + required_projection_roles
            )
            if projection_spec.requires_odd_parity_source and (
                "b_mode" not in effective_projection_roles
            ):
                effective_projection_roles = effective_projection_roles + (
                    "b_mode",
                )
            if effective_projection_roles:
                for source_name in source_term_refs.values():
                    if _source_ancestry_supports_projection_roles(
                        source_name,
                        effective_projection_roles,
                    ):
                        continue
                    if (
                        projection_spec.requires_odd_parity_source
                        and effective_projection_roles == ("b_mode",)
                    ):
                        raise ValueError(
                            f"Perturbation observable '{name}' projection "
                            f"'{projection}' requires an odd-parity "
                            "declared source ancestry"
                        )
                    raise ValueError(
                        f"Perturbation observable '{name}' projection "
                        f"'{projection}' requires source '{source_name}' "
                        "to provide declared projection roles: "
                        + ", ".join(effective_projection_roles)
                    )
            transfer_component_names.add(name)
        else:
            if primary is None or secondary is None:
                raise ValueError(
                    f"Perturbation observable '{name}' must declare "
                    "primary and secondary"
                )
            if projection is not None or source_term_refs:
                raise ValueError(
                    f"Perturbation observable '{name}' kind "
                    "'angular_power_spectrum' must not declare projection "
                    "or source_terms"
                )
            if kernel is not None or required_projection_roles:
                raise ValueError(
                    f"Perturbation observable '{name}' kind "
                    "'angular_power_spectrum' must not declare kernel or "
                    "required_projection_roles"
                )
        dependencies = _dedupe_names(
            tuple(source_term_refs.values())
            + (() if primary is None else (primary,))
            + (() if secondary is None else (secondary,))
        )
        observable_entries[name] = PerturbationObservableData(
            name=name,
            kind=observable_kind,
            projection=projection,
            kernel=kernel,
            primary=primary,
            secondary=secondary,
            source_terms=FrozenMapping(source_term_refs),
            required_projection_roles=effective_projection_roles,
            description=_validate_optional_string(
                observable_def.get("description"),
                label=f"cmb.perturbations.observables.{name}.description",
            ),
            notes=_validate_optional_string(
                observable_def.get("notes"),
                label=f"cmb.perturbations.observables.{name}.notes",
            ),
            domain=_validate_optional_string(
                observable_def.get("domain"),
                label=f"cmb.perturbations.observables.{name}.domain",
            ),
            dependencies=dependencies,
        )
        observable_names.add(name)
    for observable_name, observable_entry in observable_entries.items():
        if observable_entry.kind != "angular_power_spectrum":
            continue
        if observable_entry.primary not in transfer_component_names:
            raise ValueError(
                f"Perturbation observable '{observable_name}' references "
                f"unknown transfer component '{observable_entry.primary}'"
            )
        if observable_entry.secondary not in transfer_component_names:
            raise ValueError(
                f"Perturbation observable '{observable_name}' references "
                f"unknown transfer component '{observable_entry.secondary}'"
            )

    def _compile_conditions(
        condition_defs: Mapping[str, Any],
        *,
        label_prefix: str,
        default_anchor: str,
    ) -> dict[str, PerturbationConditionData]:
        """Compile initial or boundary-condition mappings into typed data."""

        compiled: dict[str, PerturbationConditionData] = {}
        seen_targets: set[tuple[str, str, int]] = set()
        for condition_name, condition_def in condition_defs.items():
            name = _validate_string(
                condition_name,
                label="Condition name",
            )
            if not isinstance(condition_def, Mapping):
                raise ValueError(
                    f"Perturbation condition '{name}' must be a mapping"
                )
            _validate_entry_keys(
                entry=condition_def,
                allowed_keys=_SUPPORTED_CONDITION_KEYS,
                label=f"{label_prefix}.{name}",
            )
            target = condition_def.get("target")
            if not isinstance(target, Mapping):
                raise ValueError(
                    f"{label_prefix}.{name}.target must be a mapping"
                )
            _validate_entry_keys(
                entry=target,
                allowed_keys=_SUPPORTED_CONDITION_TARGET_KEYS,
                label=f"{label_prefix}.{name}.target",
            )
            variable_name = _validate_string(
                target.get("variable"),
                label=f"{label_prefix}.{name}.target.variable",
            )
            if variable_name not in variable_entries:
                raise ValueError(
                    f"Perturbation condition '{name}' references unknown "
                    f"variable '{variable_name}'"
                )
            wrt_name = _validate_string(
                target.get("wrt"),
                label=f"{label_prefix}.{name}.target.wrt",
            )
            if wrt_name not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
                raise ValueError(
                    f"Perturbation condition '{name}' uses unsupported wrt "
                    f"'{wrt_name}'"
                )
            order_value = _validate_optional_int(
                target.get("order"),
                label=f"{label_prefix}.{name}.target.order",
            )
            if order_value is None or order_value < 0:
                raise ValueError(
                    f"Perturbation condition '{name}' order must be a "
                    "non-negative integer"
                )
            target_key = (variable_name, wrt_name, order_value)
            if target_key in seen_targets:
                raise ValueError(
                    f"Perturbation condition '{name}' duplicates target "
                    f"{target_key}"
                )
            expression_text, dependencies = _replace_and_validate_expression(
                condition_def.get("expression"),
                label=f"{label_prefix}.{name}.expression",
                replacements=replacements,
                allowed_names=all_expression_names
                | set(
                    _relation_target_nodes(constraint_entries, closure_entries)
                )
                | {entry_name for entry_name in variable_entries},
            )
            anchor_name = str(
                _validate_optional_string(
                    condition_def.get("anchor"),
                    label=f"{label_prefix}.{name}.anchor",
                )
                or default_anchor
            )
            if anchor_name not in _SUPPORTED_CONDITION_ANCHORS:
                raise ValueError(
                    f"Perturbation condition '{name}' uses unsupported "
                    f"anchor '{anchor_name}'"
                )
            compiled[name] = PerturbationConditionData(
                name=name,
                target=PerturbationConditionTargetData(
                    variable=variable_name,
                    wrt=wrt_name,
                    order=order_value,
                ),
                expression=expression_text,
                anchor=anchor_name,
                description=_validate_optional_string(
                    condition_def.get("description"),
                    label=f"{label_prefix}.{name}.description",
                ),
                notes=_validate_optional_string(
                    condition_def.get("notes"),
                    label=f"{label_prefix}.{name}.notes",
                ),
                domain=_validate_optional_string(
                    condition_def.get("domain"),
                    label=f"{label_prefix}.{name}.domain",
                ),
                dependencies=dependencies,
                compiled_expression=_compile_expression_plan(
                    expression_text,
                    dependencies=dependencies,
                ),
            )
            seen_targets.add(target_key)
        return compiled

    initial_condition_entries = _compile_conditions(
        sections["initial_conditions"],
        label_prefix="cmb.perturbations.initial_conditions",
        default_anchor="start",
    )
    boundary_condition_entries = _compile_conditions(
        sections["boundary_conditions"],
        label_prefix="cmb.perturbations.boundary_conditions",
        default_anchor="start",
    )

    if standard:
        for section_name in (
            "variables",
            "derived",
            "equations",
            "constraints",
            "closures",
            "sources",
            "observables",
            "initial_conditions",
            "boundary_conditions",
        ):
            if sections[section_name]:
                raise ValueError(
                    f"Standard perturbations require {section_name}: {{}}"
                )
    else:
        if not variable_entries:
            raise ValueError(
                "Non-standard perturbations must declare variables"
            )
        if not equation_entries:
            raise ValueError(
                "Non-standard perturbations must declare equations"
            )
        if not initial_condition_entries and not boundary_condition_entries:
            raise ValueError(
                "Non-standard perturbations must declare initial_conditions "
                "or boundary_conditions"
            )
        if not observable_entries:
            raise ValueError(
                "Non-standard perturbations must declare observables"
            )
        if not sections["validity"]:
            raise ValueError(
                "Non-standard perturbations must declare validity"
            )

    validity_notes = _validate_optional_string(
        sections["validity"].get("notes"),
        label="cmb.perturbations.validity.notes",
    )
    validity_regimes = sections["validity"].get("regimes")
    regimes = ()
    if validity_regimes is not None:
        regimes = _validate_regimes(validity_regimes)
    elif not standard:
        raise ValueError(
            "Non-standard perturbations must declare validity.regimes"
        )
    validity_data = PerturbationValidityData(
        regimes=regimes,
        notes=validity_notes,
    )

    numerics_mapping = FrozenMapping(
        {str(key): value for key, value in sections["numerics"].items()}
    )

    derivative_symbol_orders: dict[tuple[str, str], int] = {}
    for entry in derived_entries.values():
        if entry.expression is not None:
            continue
        key = (str(entry.variable), str(entry.wrt))
        derivative_symbol_orders[key] = max(
            derivative_symbol_orders.get(key, 0),
            int(entry.order or 0),
        )
    equation_orders: dict[tuple[str, str], int] = {}
    for entry in equation_entries.values():
        key = (entry.lhs.variable, entry.lhs.wrt)
        equation_orders[key] = max(
            equation_orders.get(key, 0),
            entry.lhs.order,
        )
    evolved_variable_names = {
        entry.lhs.variable for entry in equation_entries.values()
    }
    relation_target_names = set(
        _relation_target_nodes(constraint_entries, closure_entries)
    )
    for key, required_order in derivative_symbol_orders.items():
        if key not in equation_orders:
            variable_name, wrt_name = key
            if variable_name in relation_target_names:
                continue
            raise ValueError(
                "Derivative symbol requires an evolved variable: "
                f"{variable_name} wrt {wrt_name}"
            )
        if required_order >= equation_orders[key]:
            variable_name, wrt_name = key
            raise ValueError(
                "Derivative symbol order exceeds the declared differential "
                f"state for variable '{variable_name}' wrt '{wrt_name}'"
            )

    required_initial_targets = {
        (
            equation_entry.lhs.variable,
            equation_entry.lhs.wrt,
            derivative_order,
        )
        for equation_entry in equation_entries.values()
        for derivative_order in range(equation_entry.lhs.order)
    }
    declared_condition_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in initial_condition_entries.values()
    } | {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in boundary_condition_entries.values()
    }
    unsupported_condition_targets = sorted(
        declared_condition_targets - required_initial_targets
    )
    if unsupported_condition_targets:
        readable = ", ".join(
            f"{variable} wrt {wrt} order {order}"
            for variable, wrt, order in unsupported_condition_targets
        )
        raise ValueError(
            "Perturbation conditions may only target declared differential "
            f"state slots: {readable}"
        )
    declared_initial_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in initial_condition_entries.values()
    }
    declared_start_boundary_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in boundary_condition_entries.values()
        if condition_entry.anchor == "start"
    }
    declared_boundary_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in boundary_condition_entries.values()
    }
    duplicate_start_targets = sorted(
        declared_initial_targets & declared_start_boundary_targets
    )
    if duplicate_start_targets:
        readable = ", ".join(
            f"{variable} wrt {wrt} order {order}"
            for variable, wrt, order in duplicate_start_targets
        )
        raise ValueError(
            "Initial conditions and start-anchored boundary conditions "
            f"duplicate targets: {readable}"
        )
    missing_initial_targets = sorted(
        required_initial_targets
        - declared_initial_targets
        - declared_boundary_targets
    )
    if missing_initial_targets:
        readable = ", ".join(
            f"{variable} wrt {wrt} order {order}"
            for variable, wrt, order in missing_initial_targets
        )
        raise ValueError(
            "Non-standard perturbations are missing required initial "
            f"conditions: {readable}"
        )

    solved_variable_names = evolved_variable_names | relation_target_names
    referenced_unsolved_variables = sorted(
        {
            dependency
            for entry in (
                list(derived_entries.values())
                + list(equation_entries.values())
                + list(constraint_entries.values())
                + list(closure_entries.values())
                + list(source_entries.values())
                + list(initial_condition_entries.values())
                + list(boundary_condition_entries.values())
            )
            for dependency in entry.dependencies
            if (
                dependency in variable_entries
                and dependency not in solved_variable_names
            )
        }
    )
    if referenced_unsolved_variables:
        readable = ", ".join(referenced_unsolved_variables)
        raise ValueError(
            "Declared graph references variable(s) without evolution or "
            f"algebraic definitions: {readable}"
        )

    evaluation_order = _topological_evaluation_order(
        derived=derived_entries,
        constraints=constraint_entries,
        closures=closure_entries,
    )
    dependency_summary = PerturbationDependencyGraphSummaryData(
        variable_names=tuple(sorted(variable_entries)),
        derived_names=tuple(sorted(derived_entries)),
        equation_names=tuple(sorted(equation_entries)),
        constraint_names=tuple(sorted(constraint_entries)),
        closure_names=tuple(sorted(closure_entries)),
        source_names=tuple(sorted(source_entries)),
        observable_names=tuple(sorted(observable_entries)),
        initial_condition_names=tuple(sorted(initial_condition_entries)),
        boundary_condition_names=tuple(sorted(boundary_condition_entries)),
        independent_variables_used=tuple(
            sorted(
                set().union(
                    *(
                        set(entry.dependencies)
                        for entry in (
                            list(derived_entries.values())
                            + list(equation_entries.values())
                            + list(constraint_entries.values())
                            + list(closure_entries.values())
                            + list(source_entries.values())
                            + list(observable_entries.values())
                            + list(initial_condition_entries.values())
                            + list(boundary_condition_entries.values())
                        )
                    )
                )
                & set(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
            )
        ),
        model_parameters_used=tuple(
            sorted(
                set().union(
                    *(
                        set(entry.dependencies)
                        for entry in (
                            list(derived_entries.values())
                            + list(equation_entries.values())
                            + list(constraint_entries.values())
                            + list(closure_entries.values())
                            + list(source_entries.values())
                            + list(initial_condition_entries.values())
                            + list(boundary_condition_entries.values())
                        )
                    )
                )
                & parameter_name_set
            )
        ),
        background_references_used=tuple(
            sorted(
                set().union(
                    *(
                        set(entry.dependencies)
                        for entry in (
                            list(derived_entries.values())
                            + list(equation_entries.values())
                            + list(constraint_entries.values())
                            + list(closure_entries.values())
                            + list(source_entries.values())
                            + list(initial_condition_entries.values())
                            + list(boundary_condition_entries.values())
                        )
                    )
                )
                & background_reference_set
            )
        ),
        derived_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in derived_entries.items()
            }
        ),
        equation_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in equation_entries.items()
            }
        ),
        constraint_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in constraint_entries.items()
            }
        ),
        closure_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in closure_entries.items()
            }
        ),
        source_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in source_entries.items()
            }
        ),
        observable_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in observable_entries.items()
            }
        ),
        initial_condition_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in initial_condition_entries.items()
            }
        ),
        boundary_condition_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in boundary_condition_entries.items()
            }
        ),
        evaluation_order=evaluation_order,
    )

    backend_data = PerturbationBackendMappingData(
        backend=backend,
        uses_standard_perturbations=backend_contract.get(
            "uses_standard_perturbations"
        ),
        native_solver_required=backend_contract.get("native_solver_required"),
        implemented=backend_contract.get("implemented"),
    )
    transfer_component_contracts = {
        name: {
            "projection": str(entry.projection or ""),
            "kernel": (None if entry.kernel is None else str(entry.kernel)),
            "source_term_roles": tuple(
                str(role) for role in entry.source_terms
            ),
            "source_term_names": {
                str(role): str(source_name)
                for role, source_name in entry.source_terms.items()
            },
            "required_projection_roles": tuple(
                str(role) for role in entry.required_projection_roles
            ),
        }
        for name, entry in observable_entries.items()
        if entry.kind == "transfer_component"
    }
    angular_power_spectrum_targets = {
        name: {
            "primary": str(entry.primary or ""),
            "secondary": str(entry.secondary or ""),
        }
        for name, entry in observable_entries.items()
        if entry.kind == "angular_power_spectrum"
    }
    manifest_summary = FrozenMapping(
        _build_manifest_summary(
            model_name=model_name,
            backend=backend,
            contract_version=contract_version,
            standard=standard,
            gauge=gauge,
            variables=dependency_summary.variable_names,
            derived=dependency_summary.derived_names,
            equations=dependency_summary.equation_names,
            constraints=dependency_summary.constraint_names,
            closures=dependency_summary.closure_names,
            sources=dependency_summary.source_names,
            observables=dependency_summary.observable_names,
            initial_conditions=dependency_summary.initial_condition_names,
            boundary_conditions=(dependency_summary.boundary_condition_names),
            validity=validity_data,
            numerics=numerics_mapping,
            backend_mapping=backend_data,
            dependency_summary=dependency_summary,
            equation_wrt_by_variable={
                entry.lhs.variable: entry.lhs.wrt
                for entry in equation_entries.values()
            },
            boundary_condition_anchors={
                name: entry.anchor
                for name, entry in boundary_condition_entries.items()
            },
            transfer_component_contracts=transfer_component_contracts,
            angular_power_spectrum_targets=angular_power_spectrum_targets,
        )
    )

    compiled = PerturbationContractData(
        model_name=model_name,
        backend=backend,
        contract_version=contract_version,
        standard=standard,
        gauge=gauge,
        variables=FrozenMapping(variable_entries),
        derived=FrozenMapping(derived_entries),
        equations=FrozenMapping(equation_entries),
        constraints=FrozenMapping(constraint_entries),
        closures=FrozenMapping(closure_entries),
        sources=FrozenMapping(source_entries),
        observables=FrozenMapping(observable_entries),
        initial_conditions=FrozenMapping(initial_condition_entries),
        boundary_conditions=FrozenMapping(boundary_condition_entries),
        numerics=numerics_mapping,
        validity=validity_data,
        backend_mapping=FrozenMapping({backend: backend_data}),
        dependency_graph_summary=dependency_summary,
        manifest_summary=manifest_summary,
    )
    _COMPILED_CONTRACT_RESULTS[cache_key] = compiled
    return _get_cached_perturbation_contract(cache_key)


__all__ = [
    "PerturbationBackendMappingData",
    "PerturbationClosureData",
    "PerturbationCompiledExpressionData",
    "PerturbationConditionData",
    "PerturbationConditionTargetData",
    "PerturbationConstraintData",
    "PerturbationContractData",
    "PerturbationDependencyGraphSummaryData",
    "PerturbationDerivedData",
    "PerturbationDerivativeLhsData",
    "PerturbationEquationData",
    "PerturbationObservableData",
    "PerturbationSourceData",
    "PerturbationValidityData",
    "PerturbationVariableData",
    "compile_perturbation_contract",
    "evaluate_compiled_expression",
]
