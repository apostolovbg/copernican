"""Typed perturbation-contract validation and compilation helpers.

The compiler keeps perturbation declarations declarative, validates them
against the model metadata, and produces a picklable internal
representation that downstream code can inspect without re-parsing YAML.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

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

_SUPPORTED_PERTURBATION_KEYS = {
    "backend_mapping",
    "closures",
    "contract_version",
    "derived",
    "equations",
    "equation_mode",
    "gauge",
    "notes",
    "sources",
    "standard",
    "validity",
    "variables",
}
_SUPPORTED_EQUATION_MODES = {
    "declared_equations",
    "mapped_sector",
}
_SUPPORTED_VARIABLE_KEYS = {"description", "kind", "notes"}
_SUPPORTED_DERIVED_KEYS = {
    "description",
    "expression",
    "kind",
    "notes",
    "order",
    "variable",
    "wrt",
}
_SUPPORTED_EQUATION_KEYS = {"description", "lhs", "notes", "rhs"}
_SUPPORTED_LHS_KEYS = {"kind", "order", "variable", "wrt"}
_SUPPORTED_CLOSURE_KEYS = {
    "description",
    "equals",
    "expression",
    "notes",
}
_SUPPORTED_SOURCE_KEYS = {"channel", "description", "expression", "notes"}
_SUPPORTED_SOURCE_CHANNELS = {
    "polarization",
    "temperature_additive",
    "temperature_doppler",
    "temperature_isw",
    "temperature_monopole",
}
_SUPPORTED_VALIDITY_KEYS = {"notes", "regimes"}
_SUPPORTED_BACKEND_KEYS = {"camb"}
_STANDARD_BACKEND_KEYS = {"uses_standard_perturbations"}
_NONSTANDARD_BACKEND_KEYS = {
    "implemented",
    "native_solver_required",
}
_COMPILED_CONTRACT_RESULTS: dict[
    tuple[Any, ...], "PerturbationContractData"
] = {}


@lru_cache(maxsize=256)
def _get_cached_perturbation_contract(
    cache_key: tuple[Any, ...],
) -> "PerturbationContractData":
    """Return a cached perturbation contract for ``cache_key``."""

    return _COMPILED_CONTRACT_RESULTS[cache_key]


@dataclass(frozen=True, slots=True)
class PerturbationVariableData:
    """Typed representation of a declared perturbation variable."""

    name: str
    kind: str
    description: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationDerivedData:
    """Typed representation of a declared derived perturbation symbol."""

    name: str
    kind: str
    expression: str | None = None
    variable: str | None = None
    wrt: str | None = None
    order: int | None = None
    description: str | None = None
    notes: str | None = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationDerivativeLhsData:
    """Typed representation of a derivative equation left-hand side."""

    kind: str
    variable: str
    wrt: str
    order: int


@dataclass(frozen=True, slots=True)
class PerturbationEquationData:
    """Typed representation of a perturbation evolution equation."""

    name: str
    lhs: PerturbationDerivativeLhsData
    rhs: str
    description: str | None = None
    notes: str | None = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationClosureData:
    """Typed representation of a perturbation closure relation."""

    name: str
    expression: str
    equals: str
    description: str | None = None
    notes: str | None = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationSourceData:
    """Typed representation of a perturbation source term."""

    name: str
    expression: str
    channel: str
    description: str | None = None
    notes: str | None = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationValidityData:
    """Typed representation of a perturbation validity declaration."""

    regimes: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationBackendMappingData:
    """Typed representation of the backend execution mapping."""

    backend: str
    uses_standard_perturbations: bool | None = None
    native_solver_required: bool | None = None
    implemented: bool | None = None


@dataclass(frozen=True, slots=True)
class PerturbationDependencyGraphSummaryData:
    """Summary of perturbation symbol dependencies."""

    variable_names: tuple[str, ...]
    derived_expression_names: tuple[str, ...]
    derivative_symbol_names: tuple[str, ...]
    equation_names: tuple[str, ...]
    closure_names: tuple[str, ...]
    source_names: tuple[str, ...]
    independent_variables_used: tuple[str, ...]
    model_parameters_used: tuple[str, ...]
    background_references_used: tuple[str, ...]
    derived_expression_dependencies: FrozenMapping
    equation_dependencies: FrozenMapping
    closure_dependencies: FrozenMapping
    source_dependencies: FrozenMapping


@dataclass(frozen=True, slots=True)
class PerturbationContractData:
    """Immutable internal representation of a perturbation contract."""

    model_name: str
    backend: str
    contract_version: int
    standard: bool
    equation_mode: str
    gauge: str
    variables: FrozenMapping
    derived: FrozenMapping
    equations: FrozenMapping
    closures: FrozenMapping
    sources: FrozenMapping
    validity: PerturbationValidityData
    backend_mapping: FrozenMapping
    dependency_graph_summary: PerturbationDependencyGraphSummaryData
    manifest_summary: FrozenMapping


@lru_cache(maxsize=2048)
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


def _validate_entry_keys(
    *,
    entry: Mapping[str, Any],
    allowed_keys: set[str],
    label: str,
) -> None:
    """Reject unknown keys inside a contract entry."""

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
    """Return ``value`` as a validated string."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return cleaned


def _validate_regimes(value: Any) -> tuple[str, ...]:
    """Return a validated sequence of regimes."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("cmb.perturbations.validity.regimes must be a list")
    regimes: list[str] = []
    for regime in value:
        regimes.append(
            _validate_string(
                regime,
                label="cmb.perturbations.validity.regimes entry",
            )
        )
    if not regimes:
        raise ValueError(
            "cmb.perturbations.validity.regimes must not be empty"
        )
    return tuple(regimes)


def _categorise_names(
    names: Iterable[str],
    *,
    variable_names: set[str],
    derived_expression_names: set[str],
    derivative_symbol_names: set[str],
    parameter_names: set[str],
    background_reference_names: set[str],
) -> tuple[
    set[str],
    set[str],
    set[str],
    set[str],
    set[str],
]:
    """Split dependency names into the tracked symbol categories."""

    independent_vars: set[str] = set()
    model_params: set[str] = set()
    background_refs: set[str] = set()
    derived_refs: set[str] = set()
    variable_refs: set[str] = set()

    allowed_independent = set(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
    for name in names:
        if name in allowed_independent:
            independent_vars.add(name)
        elif name in parameter_names:
            model_params.add(name)
        elif name in background_reference_names:
            background_refs.add(name)
        elif name in derived_expression_names:
            derived_refs.add(name)
        elif name in variable_names or name in derivative_symbol_names:
            variable_refs.add(name)

    return (
        independent_vars,
        model_params,
        background_refs,
        derived_refs,
        variable_refs,
    )


def _build_manifest_summary(
    *,
    model_name: str,
    backend: str,
    contract_version: int,
    standard: bool,
    equation_mode: str,
    gauge: str,
    variables: tuple[str, ...],
    derived: tuple[str, ...],
    equations: tuple[str, ...],
    closures: tuple[str, ...],
    sources: tuple[str, ...],
    validity: PerturbationValidityData,
    backend_mapping: PerturbationBackendMappingData,
    dependency_summary: PerturbationDependencyGraphSummaryData,
) -> dict[str, Any]:
    """Return a manifest-safe summary of the perturbation contract."""

    return {
        "model_name": model_name,
        "backend": backend,
        "contract_version": contract_version,
        "standard": standard,
        "equation_mode": equation_mode,
        "gauge": gauge,
        "variable_names": variables,
        "derived_names": derived,
        "equation_names": equations,
        "closure_names": closures,
        "source_names": sources,
        "validity_regimes": validity.regimes,
        "backend_implemented": backend_mapping.implemented,
        (
            "backend_native_solver_required"
        ): backend_mapping.native_solver_required,
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
    """Validate and compile a perturbation contract into typed data."""

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
    missing_keys = {
        "backend_mapping",
        "contract_version",
        "derived",
        "gauge",
        "standard",
        "validity",
        "variables",
        "closures",
    } - contract_keys
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
    if contract_version != 1 or isinstance(contract_version, bool):
        raise ValueError("cmb.perturbations.contract_version must be 1")

    standard = contract.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")

    equation_mode = contract.get("equation_mode", "mapped_sector")
    if not isinstance(equation_mode, str):
        raise ValueError("cmb.perturbations.equation_mode must be a string")
    equation_mode = equation_mode.strip()
    if not equation_mode:
        raise ValueError("cmb.perturbations.equation_mode must not be empty")
    if equation_mode not in _SUPPORTED_EQUATION_MODES:
        allowed_modes = ", ".join(sorted(_SUPPORTED_EQUATION_MODES))
        raise ValueError(
            "cmb.perturbations.equation_mode must be one of "
            f"{allowed_modes}"
        )
    if standard and equation_mode != "mapped_sector":
        raise ValueError(
            "Standard perturbations must use equation_mode: mapped_sector"
        )

    gauge = contract.get("gauge")
    if gauge not in {
        "conformal_newtonian",
        "gauge_invariant",
        "synchronous",
        "unspecified",
    }:
        raise ValueError("cmb.perturbations.gauge is invalid")
    gauge = str(gauge)

    notes = contract.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("cmb.perturbations.notes must be a string")

    variables = contract.get("variables")
    derived = contract.get("derived")
    equations = contract.get("equations")
    closures = contract.get("closures")
    sources = contract.get("sources")
    validity = contract.get("validity")
    backend_mapping = contract.get("backend_mapping")

    for section_name, section_value in (
        ("variables", variables),
        ("derived", derived),
        ("equations", equations),
        ("closures", closures),
        ("sources", sources),
        ("validity", validity),
        ("backend_mapping", backend_mapping),
    ):
        if not isinstance(section_value, Mapping):
            raise ValueError(
                f"cmb.perturbations.{section_name} must be a mapping"
            )

    parameter_name_set = {str(name) for name in parameter_names}
    background_reference_set = {
        str(name) for name in background_reference_names
    }
    replacements = _build_parameter_replacements(
        parameter_names,
        latex_names,
    )

    backend_keys = {str(key) for key in backend_mapping.keys()}
    invalid_backend_keys = backend_keys - _SUPPORTED_BACKEND_KEYS
    if invalid_backend_keys:
        invalid_str = ", ".join(sorted(invalid_backend_keys))
        raise ValueError(f"Unknown perturbation backend(s): {invalid_str}")
    backend_contract = backend_mapping.get(backend)
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
    derivative_symbol_entries: dict[str, PerturbationDerivedData] = {}
    derived_expression_entries: dict[str, PerturbationDerivedData] = {}
    equation_entries: dict[str, PerturbationEquationData] = {}
    closure_entries: dict[str, PerturbationClosureData] = {}
    source_entries: dict[str, PerturbationSourceData] = {}

    if standard:
        for section_name, section_value in (
            ("variables", variables),
            ("derived", derived),
            ("equations", equations),
            ("closures", closures),
            ("sources", sources),
        ):
            if section_value:
                raise ValueError(
                    f"Standard perturbations require {section_name}: {{}}"
                )
    else:
        if not variables:
            raise ValueError(
                "Non-standard perturbations must declare variables"
            )
        if not validity:
            raise ValueError(
                "Non-standard perturbations must declare validity"
            )
        if not backend_mapping:
            raise ValueError(
                "Non-standard perturbations must declare backend mapping"
            )
        if equation_mode == "declared_equations" and not equations:
            raise ValueError(
                "Non-standard perturbations in declared_equations mode "
                "must declare equations"
            )

    variable_names: set[str] = set()
    derived_symbol_names: set[str] = set()
    derived_expression_names: set[str] = set()
    equation_names: set[str] = set()
    closure_names: set[str] = set()
    source_names: set[str] = set()

    for variable_name, variable_def in variables.items():
        if not isinstance(variable_name, str) or not variable_name.strip():
            raise ValueError("Perturbation variable names must be strings")
        if not isinstance(variable_def, Mapping):
            raise ValueError(
                f"Perturbation variable '{variable_name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=variable_def,
            allowed_keys=_SUPPORTED_VARIABLE_KEYS,
            label=f"cmb.perturbations.variables.{variable_name}",
        )
        if (
            variable_name in parameter_name_set
            or variable_name in background_reference_set
        ):
            raise ValueError(
                f"Perturbation variable '{variable_name}' collides with an "
                "existing background symbol"
            )
        if variable_name in variable_names:
            raise ValueError(
                f"Perturbation variable '{variable_name}' is duplicated"
            )
        kind = _validate_string(
            variable_def.get("kind"),
            label=f"cmb.perturbations.variables.{variable_name}.kind",
        )
        description = variable_def.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Perturbation variable '{variable_name}' description must "
                "be a string"
            )
        notes = variable_def.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(
                f"Perturbation variable '{variable_name}' notes must be a "
                "string"
            )
        variable_entries[variable_name] = PerturbationVariableData(
            name=variable_name,
            kind=kind,
            description=description,
            notes=notes,
        )
        variable_names.add(variable_name)

    allowed_name_pool: set[str] = set(parameter_name_set)
    allowed_name_pool.update(background_reference_set)
    allowed_name_pool.update(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
    allowed_name_pool.update(variable_names)

    for derived_name, derived_def in derived.items():
        if not isinstance(derived_name, str) or not derived_name.strip():
            raise ValueError("Derived perturbation names must be strings")
        if not isinstance(derived_def, Mapping):
            raise ValueError(
                f"Perturbation derived '{derived_name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=derived_def,
            allowed_keys=_SUPPORTED_DERIVED_KEYS,
            label=f"cmb.perturbations.derived.{derived_name}",
        )
        if (
            derived_name in parameter_name_set
            or derived_name in background_reference_set
            or derived_name in variable_names
        ):
            raise ValueError(
                f"Perturbation derived '{derived_name}' collides with an "
                "existing symbol"
            )
        if (
            derived_name in derived_expression_names
            or derived_name in derived_symbol_names
        ):
            raise ValueError(
                f"Perturbation derived '{derived_name}' is duplicated"
            )

        expression = derived_def.get("expression")
        kind = derived_def.get("kind")
        variable = derived_def.get("variable")
        wrt = derived_def.get("wrt")
        order = derived_def.get("order")
        description = derived_def.get("description")
        notes = derived_def.get("notes")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Perturbation derived '{derived_name}' description must be a "
                "string"
            )
        if notes is not None and not isinstance(notes, str):
            raise ValueError(
                f"Perturbation derived '{derived_name}' notes must be a "
                "string"
            )
        if expression is not None and kind is not None:
            raise ValueError(
                f"Perturbation derived '{derived_name}' cannot declare both "
                "expression and kind"
            )
        if expression is None and kind is None:
            raise ValueError(
                f"Perturbation derived '{derived_name}' must declare either "
                "expression or kind"
            )

        if kind is None:
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError(
                    f"Perturbation derived '{derived_name}' must define a "
                    "string expression"
                )
            clean_expr = _replace_latex_tokens(expression, replacements)
            _validate_safe_expression(
                clean_expr,
                allowed_name_pool
                | derived_expression_names
                | derived_symbol_names,
            )
            dependencies = _collect_expression_names(clean_expr)
            derived_expression_entries[derived_name] = PerturbationDerivedData(
                name=derived_name,
                kind="expression",
                expression=clean_expr,
                description=description,
                notes=notes,
                dependencies=dependencies,
            )
            derived_expression_names.add(derived_name)
            allowed_name_pool.add(derived_name)
            continue

        if kind != "derivative_symbol":
            raise ValueError(
                f"Perturbation derived '{derived_name}' has unsupported "
                "kind"
            )
        if expression is not None:
            raise ValueError(
                f"Perturbation derived '{derived_name}' cannot combine "
                "derivative_symbol with expression"
            )
        variable_name = _validate_string(
            variable,
            label=f"cmb.perturbations.derived.{derived_name}.variable",
        )
        wrt_name = _validate_string(
            wrt,
            label=f"cmb.perturbations.derived.{derived_name}.wrt",
        )
        if wrt_name not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
            raise ValueError(
                f"Perturbation derived '{derived_name}' uses unsupported "
                f"independent variable '{wrt_name}'"
            )
        if variable_name not in allowed_name_pool:
            raise ValueError(
                f"Perturbation derived '{derived_name}' references unknown "
                f"variable '{variable_name}'"
            )
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ValueError(
                f"Perturbation derived '{derived_name}' order must be a "
                "positive integer"
            )
        dependencies = (variable_name, wrt_name)
        derivative_symbol_entries[derived_name] = PerturbationDerivedData(
            name=derived_name,
            kind="derivative_symbol",
            variable=variable_name,
            wrt=wrt_name,
            order=order,
            description=description,
            notes=notes,
            dependencies=dependencies,
        )
        derived_symbol_names.add(derived_name)
        allowed_name_pool.add(derived_name)

    allowed_expression_names = (
        set(parameter_name_set)
        | set(background_reference_set)
        | set(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
        | set(variable_names)
        | set(derived_expression_names)
        | set(derived_symbol_names)
    )

    derived_expression_dependencies: dict[str, tuple[str, ...]] = {}
    for derived_name, derived_data in derived_expression_entries.items():
        names = tuple(
            name
            for name in derived_data.dependencies
            if name in allowed_expression_names
        )
        unknown_names = set(derived_data.dependencies) - set(names)
        if unknown_names:
            unknown_str = ", ".join(sorted(unknown_names))
            raise ValueError(
                f"Perturbation derived '{derived_name}' references unknown "
                f"symbol(s): {unknown_str}"
            )
        _validate_safe_expression(
            derived_data.expression or "",
            allowed_expression_names,
        )
        derived_expression_dependencies[derived_name] = names

    for equation_name, equation_def in equations.items():
        if not isinstance(equation_name, str) or not equation_name.strip():
            raise ValueError("Equation names must be strings")
        if not isinstance(equation_def, Mapping):
            raise ValueError(
                f"Perturbation equation '{equation_name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=equation_def,
            allowed_keys=_SUPPORTED_EQUATION_KEYS,
            label=f"cmb.perturbations.equations.{equation_name}",
        )
        if equation_name in equation_names:
            raise ValueError(
                f"Perturbation equation '{equation_name}' is duplicated"
            )
        lhs = equation_def.get("lhs")
        rhs = equation_def.get("rhs")
        if not isinstance(lhs, Mapping):
            raise ValueError(
                f"Perturbation equation '{equation_name}' needs typed lhs"
            )
        if not isinstance(rhs, str) or not rhs.strip():
            raise ValueError(
                f"Perturbation equation '{equation_name}' needs rhs"
            )
        _validate_entry_keys(
            entry=lhs,
            allowed_keys=_SUPPORTED_LHS_KEYS,
            label=f"cmb.perturbations.equations.{equation_name}.lhs",
        )
        if lhs.get("kind") != "derivative":
            raise ValueError(
                f"Perturbation equation '{equation_name}' lhs must declare "
                "kind: derivative"
            )
        lhs_variable = _validate_string(
            lhs.get("variable"),
            label=(
                f"cmb.perturbations.equations.{equation_name}.lhs.variable"
            ),
        )
        if lhs_variable not in variable_names:
            raise ValueError(
                f"Perturbation equation '{equation_name}' lhs variable "
                f"'{lhs_variable}' must name a declared perturbation variable"
            )
        lhs_wrt = _validate_string(
            lhs.get("wrt"),
            label=f"cmb.perturbations.equations.{equation_name}.lhs.wrt",
        )
        if lhs_wrt not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
            raise ValueError(
                f"Perturbation equation '{equation_name}' lhs wrt '{lhs_wrt}' "
                "is not allowed"
            )
        lhs_order = lhs.get("order")
        if not isinstance(lhs_order, int) or isinstance(lhs_order, bool):
            raise ValueError(
                f"Perturbation equation '{equation_name}' lhs order must be "
                "a positive integer"
            )
        if lhs_order < 1:
            raise ValueError(
                f"Perturbation equation '{equation_name}' lhs order must be "
                "a positive integer"
            )
        description = equation_def.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Perturbation equation '{equation_name}' description must "
                "be a string"
            )
        notes = equation_def.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(
                f"Perturbation equation '{equation_name}' notes must be a "
                "string"
            )
        clean_rhs = _replace_latex_tokens(rhs, replacements)
        equation_dependencies = tuple(
            name
            for name in _collect_expression_names(clean_rhs)
            if name in allowed_expression_names
        )
        unknown_rhs = set(_collect_expression_names(clean_rhs)) - set(
            equation_dependencies
        )
        if unknown_rhs:
            unknown_str = ", ".join(sorted(unknown_rhs))
            raise ValueError(
                f"Perturbation equation '{equation_name}' references unknown "
                f"symbol(s): {unknown_str}"
            )
        _validate_safe_expression(clean_rhs, allowed_expression_names)
        equation_entries[equation_name] = PerturbationEquationData(
            name=equation_name,
            lhs=PerturbationDerivativeLhsData(
                kind="derivative",
                variable=lhs_variable,
                wrt=lhs_wrt,
                order=lhs_order,
            ),
            rhs=clean_rhs,
            description=description,
            notes=notes,
            dependencies=equation_dependencies,
        )
        equation_names.add(equation_name)

    for closure_name, closure_def in closures.items():
        if not isinstance(closure_name, str) or not closure_name.strip():
            raise ValueError("Closure names must be strings")
        if not isinstance(closure_def, Mapping):
            raise ValueError(
                f"Perturbation closure '{closure_name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=closure_def,
            allowed_keys=_SUPPORTED_CLOSURE_KEYS,
            label=f"cmb.perturbations.closures.{closure_name}",
        )
        if closure_name in closure_names:
            raise ValueError(
                f"Perturbation closure '{closure_name}' is duplicated"
            )
        expression = closure_def.get("expression")
        equals = closure_def.get("equals")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(
                f"Perturbation closure '{closure_name}' needs expression"
            )
        if not isinstance(equals, str) or not equals.strip():
            raise ValueError(
                f"Perturbation closure '{closure_name}' needs equals"
            )
        description = closure_def.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Perturbation closure '{closure_name}' description must be "
                "a string"
            )
        notes = closure_def.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(
                f"Perturbation closure '{closure_name}' notes must be a "
                "string"
            )
        clean_expression = _replace_latex_tokens(expression, replacements)
        clean_equals = _replace_latex_tokens(equals, replacements)
        dependencies = tuple(
            name
            for name in _collect_expression_names(clean_expression)
            if name in allowed_expression_names
        )
        unknown_expression = set(
            _collect_expression_names(clean_expression)
        ) - set(dependencies)
        if unknown_expression:
            unknown_str = ", ".join(sorted(unknown_expression))
            raise ValueError(
                f"Perturbation closure '{closure_name}' references unknown "
                f"symbol(s): {unknown_str}"
            )
        unknown_equals = set(_collect_expression_names(clean_equals)) - set(
            name
            for name in _collect_expression_names(clean_equals)
            if name in allowed_expression_names
        )
        if unknown_equals:
            unknown_str = ", ".join(sorted(unknown_equals))
            raise ValueError(
                f"Perturbation closure '{closure_name}' references unknown "
                f"symbol(s): {unknown_str}"
            )
        _validate_safe_expression(clean_expression, allowed_expression_names)
        _validate_safe_expression(clean_equals, allowed_expression_names)
        closure_entries[closure_name] = PerturbationClosureData(
            name=closure_name,
            expression=clean_expression,
            equals=clean_equals,
            description=description,
            notes=notes,
            dependencies=dependencies,
        )
        closure_names.add(closure_name)

    for source_name, source_def in sources.items():
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("Source names must be strings")
        if not isinstance(source_def, Mapping):
            raise ValueError(
                f"Perturbation source '{source_name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=source_def,
            allowed_keys=_SUPPORTED_SOURCE_KEYS,
            label=f"cmb.perturbations.sources.{source_name}",
        )
        if source_name in source_names:
            raise ValueError(
                f"Perturbation source '{source_name}' is duplicated"
            )
        expression = source_def.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(
                f"Perturbation source '{source_name}' needs expression"
            )
        channel = source_def.get("channel")
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError(
                f"Perturbation source '{source_name}' needs channel"
            )
        channel = channel.strip()
        if channel not in _SUPPORTED_SOURCE_CHANNELS:
            supported = ", ".join(sorted(_SUPPORTED_SOURCE_CHANNELS))
            raise ValueError(
                f"Perturbation source '{source_name}' declares unsupported "
                f"channel '{channel}'. Supported channels are: {supported}"
            )
        description = source_def.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Perturbation source '{source_name}' description must be a "
                "string"
            )
        notes = source_def.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(
                f"Perturbation source '{source_name}' notes must be a string"
            )
        clean_expression = _replace_latex_tokens(expression, replacements)
        dependencies = tuple(
            name
            for name in _collect_expression_names(clean_expression)
            if name in allowed_expression_names
        )
        unknown_expression = set(
            _collect_expression_names(clean_expression)
        ) - set(dependencies)
        if unknown_expression:
            unknown_str = ", ".join(sorted(unknown_expression))
            raise ValueError(
                f"Perturbation source '{source_name}' references unknown "
                f"symbol(s): {unknown_str}"
            )
        _validate_safe_expression(clean_expression, allowed_expression_names)
        source_entries[source_name] = PerturbationSourceData(
            name=source_name,
            expression=clean_expression,
            channel=channel,
            description=description,
            notes=notes,
            dependencies=dependencies,
        )
        source_names.add(source_name)

    validity_notes = validity.get("notes")
    if validity_notes is not None and not isinstance(validity_notes, str):
        raise ValueError("cmb.perturbations.validity.notes must be a string")
    validity_regimes = validity.get("regimes")
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
    if standard and not validity_data.regimes and validity_data.notes is None:
        validity_data = PerturbationValidityData()

    backend_data = PerturbationBackendMappingData(
        backend=backend,
        uses_standard_perturbations=backend_contract.get(
            "uses_standard_perturbations"
        ),
        native_solver_required=backend_contract.get("native_solver_required"),
        implemented=backend_contract.get("implemented"),
    )

    dependency_graph = PerturbationDependencyGraphSummaryData(
        variable_names=tuple(sorted(variable_names)),
        derived_expression_names=tuple(sorted(derived_expression_names)),
        derivative_symbol_names=tuple(sorted(derived_symbol_names)),
        equation_names=tuple(sorted(equation_names)),
        closure_names=tuple(sorted(closure_names)),
        source_names=tuple(sorted(source_names)),
        independent_variables_used=tuple(
            sorted(
                (
                    set().union(
                        *(
                            set(entry.dependencies)
                            for entry in derived_expression_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in derivative_symbol_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in equation_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in closure_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in source_entries.values()
                        )
                    )
                )
                & set(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
            )
        ),
        model_parameters_used=tuple(
            sorted(
                (
                    set().union(
                        *(
                            set(entry.dependencies)
                            for entry in derived_expression_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in derivative_symbol_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in equation_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in closure_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in source_entries.values()
                        )
                    )
                )
                & parameter_name_set
            )
        ),
        background_references_used=tuple(
            sorted(
                (
                    set().union(
                        *(
                            set(entry.dependencies)
                            for entry in derived_expression_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in derivative_symbol_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in equation_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in closure_entries.values()
                        )
                    )
                    | set().union(
                        *(
                            set(entry.dependencies)
                            for entry in source_entries.values()
                        )
                    )
                )
                & background_reference_set
            )
        ),
        derived_expression_dependencies=FrozenMapping(
            derived_expression_dependencies
        ),
        equation_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in equation_entries.items()
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
    )

    # Detect cycles among derived expressions only.
    graph = {
        name: tuple(
            dep for dep in dependencies if dep in derived_expression_names
        )
        for name, dependencies in derived_expression_dependencies.items()
    }
    visiting: list[str] = []
    active: set[str] = set()
    completed: set[str] = set()

    def _visit(node: str) -> None:
        """Walk derived-expression dependencies and reject cycles."""

        if node in completed:
            return
        if node in active:
            cycle_start = visiting.index(node)
            cycle = visiting[cycle_start:] + [node]
            raise ValueError(
                "Derived perturbation expressions contain a cycle: "
                + " -> ".join(cycle)
            )
        active.add(node)
        visiting.append(node)
        for dependency in graph.get(node, ()):
            _visit(dependency)
        visiting.pop()
        active.remove(node)
        completed.add(node)

    for node in graph:
        _visit(node)

    manifest_summary = FrozenMapping(
        _build_manifest_summary(
            model_name=model_name,
            backend=backend,
            contract_version=contract_version,
            standard=standard,
            equation_mode=equation_mode,
            gauge=gauge,
            variables=tuple(sorted(variable_names)),
            derived=tuple(
                sorted(
                    set(derived_expression_names) | set(derived_symbol_names)
                )
            ),
            equations=tuple(sorted(equation_names)),
            closures=tuple(sorted(closure_names)),
            sources=tuple(sorted(source_names)),
            validity=validity_data,
            backend_mapping=backend_data,
            dependency_summary=dependency_graph,
        )
    )

    compiled = PerturbationContractData(
        model_name=model_name,
        backend=backend,
        contract_version=contract_version,
        standard=standard,
        equation_mode=equation_mode,
        gauge=gauge,
        variables=FrozenMapping(variable_entries),
        derived=FrozenMapping(
            {
                **derived_expression_entries,
                **derivative_symbol_entries,
            }
        ),
        equations=FrozenMapping(equation_entries),
        closures=FrozenMapping(closure_entries),
        sources=FrozenMapping(source_entries),
        validity=validity_data,
        backend_mapping=FrozenMapping({backend: backend_data}),
        dependency_graph_summary=dependency_graph,
        manifest_summary=manifest_summary,
    )
    _COMPILED_CONTRACT_RESULTS[cache_key] = compiled
    return _get_cached_perturbation_contract(cache_key)


__all__ = [
    "PerturbationBackendMappingData",
    "PerturbationClosureData",
    "PerturbationContractData",
    "PerturbationDependencyGraphSummaryData",
    "PerturbationDerivedData",
    "PerturbationDerivativeLhsData",
    "PerturbationEquationData",
    "PerturbationSourceData",
    "PerturbationValidityData",
    "PerturbationVariableData",
    "compile_perturbation_contract",
]
