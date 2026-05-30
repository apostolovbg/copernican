# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Runtime adapter utilities for engine integrations.

This module builds :class:`EnginePlugin` instances from parsed model
metadata, validates the required callables, and evaluates structured CAMB
adapter contracts. The adapter keeps model metadata, priors, distance
helpers, and CAMB contract state in a picklable dataclass so engines and run
manifest code can share the same object without a package-level plugin layer.
It also validates the declared CMB perturbation contract and exposes the
backend capability registry used by the likelihood layer to reject unsupported
non-standard perturbation declarations.

The module exposes the main adapter entry points:

``build_engine_plugin``
    Normalises parsed YAML metadata and generated callables into an
    :class:`EnginePlugin`. The builder eagerly converts lists into tuples to
    encourage immutability and caches a picklable CAMB expression evaluator for
    models that expose ``cmb.param_map`` definitions.

``build_plugin``
    Validates the assembled plugin before returning it, matching the public
    entry point used by the older runtime wrapper.

``validate_plugin``
    Replaces the ad-hoc validator from the legacy interface. Validation rules
    are centralised and shared across all engines so plugin compatibility stays
    consistent, regardless of the sampling backend.

``REQUIRED_FUNCTIONS`` / ``REQUIRED_ATTRIBUTES``
    Canonical lists used by tests and runtime checks. Importers can reference
    them directly without touching the builder implementation.
"""

from __future__ import annotations

import ast
import copy
import inspect
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

import numpy

from . import priors as prior_lib
from .model_coder import CMB_BACKEND_CAPABILITIES
from .posterior import PosteriorEvaluator, make_logposterior

LOGGER = logging.getLogger(__name__)

REQUIRED_FUNCTIONS: list[str] = [
    "distance_modulus_model",
    "get_comoving_distance_Mpc",
    "get_luminosity_distance_Mpc",
    "get_angular_diameter_distance_Mpc",
    "get_Hz_per_Mpc",
    "get_DV_Mpc",
    "get_sound_horizon_rs_Mpc",
]

REQUIRED_ATTRIBUTES: list[str] = [
    "MODEL_NAME",
    "MODEL_DESCRIPTION",
    "MODEL_ABSTRACT",
    "PARAMETER_NAMES",
    "PARAMETER_LATEX_NAMES",
    "PARAMETER_UNITS",
    "INITIAL_GUESSES",
    "PARAMETER_BOUNDS",
    "FIXED_PARAMS",
    "PARAMETER_PRIORS",
    "CMB_CONTRACT",
    "CMB_PARAM_MAP",
    "CMB_PERTURBATION_CONTRACT",
    "CMB_PERTURBATION_STANDARD",
    "CMB_PERTURBATION_DATA",
]

_OPTIONAL_FUNCTIONS: tuple[str, ...] = (
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_from_dict",
)

_ALLOWED_MATH_FUNCS = {
    "abs": numpy.abs,
    "acos": numpy.arccos,
    "asin": numpy.arcsin,
    "atan": numpy.arctan,
    "atan2": numpy.arctan2,
    "ceil": numpy.ceil,
    "cos": numpy.cos,
    "cosh": numpy.cosh,
    "exp": numpy.exp,
    "floor": numpy.floor,
    "log": numpy.log,
    "log10": numpy.log10,
    "pow": numpy.power,
    "sin": numpy.sin,
    "sinh": numpy.sinh,
    "sqrt": numpy.sqrt,
    "tan": numpy.tan,
    "tanh": numpy.tanh,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}
_BIN_OPS = {
    ast.Add: numpy.add,
    ast.Sub: numpy.subtract,
    ast.Mult: numpy.multiply,
    ast.Div: numpy.divide,
    ast.Pow: numpy.power,
}
_UNARY_OPS = {ast.UAdd: lambda x: x, ast.USub: numpy.negative}
_SUPPORTED_CMB_BACKEND = "camb"
_SUPPORTED_CMB_CALL_METHODS = {
    "set_dark_energy",
    "set_dark_energy_w_a",
}
_SUPPORTED_CMB_PARAM_KEYS = {
    "AccuracyBoost",
    "Alens",
    "As",
    "H0",
    "Neff",
    "YHe",
    "kAccuracyBoost",
    "lAccuracyBoost",
    "mnu",
    "nrun",
    "nrunrun",
    "ns",
    "neutrino_hierarchy",
    "num_massive_neutrinos",
    "omch2",
    "ombh2",
    "omk",
    "omnuh2",
    "r",
    "standard_neutrino_neff",
    "sum_mnu",
    "tau",
    "theta_H0_range",
}
_REQUIRED_CMB_CONTRACT_KEYS = {
    "backend",
    "calls",
    "grids",
    "param_map",
    "perturbations",
    "values",
}
_SUPPORTED_CMB_CONTRACT_KEYS = {
    *_REQUIRED_CMB_CONTRACT_KEYS,
    "model_parameters",
    "value_definitions",
}
_SUPPORTED_CMB_GRID_KEYS = {
    "lower",
    "points",
    "spacing",
    "symbol",
    "upper",
}
_SUPPORTED_CMB_VALUE_KEYS = {"expression", "grid"}
_SUPPORTED_CMB_CALL_KEYS = {"args", "kwargs", "method"}
_SUPPORTED_CMB_PERTURBATION_KEYS = {
    "backend_mapping",
    "closures",
    "contract_version",
    "derived",
    "equations",
    "gauge",
    "notes",
    "sources",
    "standard",
    "validity",
    "variables",
}
_SUPPORTED_CMB_PERTURBATION_VALUE_KEYS = {
    "description",
    "equals",
    "expression",
    "kind",
    "notes",
    "rhs",
    "lhs",
}
_SUPPORTED_CMB_GRID_SPACING = {"linear"}
_SUPPORTED_CMB_PERTURBATION_GAUGES = {
    "conformal_newtonian",
    "gauge_invariant",
    "synchronous",
    "unspecified",
}
_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES = {
    "a",
    "z",
    "k",
    "tau",
    "eta",
    "H",
    "Hconf",
    "Phi",
    "Psi",
}
_CMB_REFERENCE_PATTERN = re.compile(
    r"^@(grid|value)\.([A-Za-z_][A-Za-z0-9_]*)$"
)
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")


def _coerce_numeric_scalar(value: Any) -> float:
    """Return ``value`` as a finite ``float`` when it is scalar-like."""

    if isinstance(value, (int, float, numpy.integer, numpy.floating)):
        return float(value)
    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 0:
        raise ValueError("expression did not evaluate to a scalar")
    return float(array_value.item())


def _coerce_numeric_array(value: Any, *, name: str) -> numpy.ndarray:
    """Return ``value`` as a finite one-dimensional ``ndarray``."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 1:
        raise ValueError(f"{name} must evaluate to a one-dimensional array")
    if array_value.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not numpy.all(numpy.isfinite(array_value)):
        raise ValueError(f"{name} must contain only finite values")
    return array_value


def _safe_expression_names() -> set[str]:
    """Return the names accepted by the restricted expression parser."""

    return set(_ALLOWED_CONSTANTS).union(_ALLOWED_MATH_FUNCS)


def _parse_safe_expression(expr: str) -> ast.Expression:
    """Parse ``expr`` with the restricted expression grammar."""

    if not isinstance(expr, str):
        raise ValueError("expression must be a string")
    if "__" in expr:
        raise ValueError(
            "Double underscores are not permitted in expressions."
        )
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression '{expr}'") from exc
    if sum(1 for _ in ast.walk(node)) > 100:
        raise ValueError("expression too complex")
    return node


def _validate_safe_expression(
    expr: str,
    allowed_names: set[str],
) -> None:
    """Ensure ``expr`` only references names in ``allowed_names``."""

    node = _parse_safe_expression(expr)

    # Recursive validator over the restricted AST nodes.
    def _validate_node(
        current: ast.AST, *, in_call_func: bool = False, depth: int = 0
    ) -> None:
        if depth > 20:
            raise ValueError("expression too complex")
        if isinstance(current, ast.Expression):
            _validate_node(current.body, depth=depth + 1)
            return
        if isinstance(current, ast.Constant):
            if not isinstance(current.value, (int, float)):
                raise ValueError("non-numeric literal")
            return
        if isinstance(current, ast.Name):
            if in_call_func:
                if current.id not in _ALLOWED_MATH_FUNCS:
                    raise ValueError(f"function '{current.id}' not allowed")
                return
            if current.id in allowed_names or current.id in _ALLOWED_CONSTANTS:
                return
            raise ValueError(f"name '{current.id}' not allowed")
        if isinstance(current, ast.BinOp):
            if type(current.op) not in _BIN_OPS:
                raise ValueError("operator not allowed")
            _validate_node(current.left, depth=depth + 1)
            _validate_node(current.right, depth=depth + 1)
            return
        if isinstance(current, ast.UnaryOp):
            if type(current.op) not in _UNARY_OPS:
                raise ValueError("operator not allowed")
            _validate_node(current.operand, depth=depth + 1)
            return
        if isinstance(current, ast.Call):
            if not isinstance(current.func, ast.Name):
                raise ValueError("invalid function call")
            if current.func.id not in _ALLOWED_MATH_FUNCS:
                raise ValueError(f"function '{current.func.id}' not allowed")
            if current.keywords:
                raise ValueError("keyword arguments not supported")
            for arg in current.args:
                _validate_node(arg, depth=depth + 1)
            return
        raise ValueError("expression not allowed")

    _validate_node(node)


def _evaluate_safe_expression(expr: str, env: Mapping[str, Any]) -> Any:
    """Evaluate ``expr`` against ``env`` using the restricted grammar."""

    node = _parse_safe_expression(expr)

    # Recursive evaluator that mirrors the same AST restrictions.
    def _evaluate_node(current: ast.AST, *, depth: int = 0) -> Any:
        if depth > 20:
            raise ValueError("expression too complex")
        if isinstance(current, ast.Expression):
            return _evaluate_node(current.body, depth=depth + 1)
        if isinstance(current, ast.Constant):
            if isinstance(current.value, (int, float)):
                return float(current.value)
            raise ValueError("non-numeric literal")
        if isinstance(current, ast.Name):
            if current.id in env:
                return env[current.id]
            if current.id in _ALLOWED_CONSTANTS:
                return _ALLOWED_CONSTANTS[current.id]
            raise ValueError(f"name '{current.id}' not allowed")
        if isinstance(current, ast.BinOp):
            operator_func = _BIN_OPS.get(type(current.op))
            if operator_func is None:
                raise ValueError("operator not allowed")
            left = _evaluate_node(current.left, depth=depth + 1)
            right = _evaluate_node(current.right, depth=depth + 1)
            return operator_func(left, right)
        if isinstance(current, ast.UnaryOp):
            operator_func = _UNARY_OPS.get(type(current.op))
            if operator_func is None:
                raise ValueError("operator not allowed")
            operand = _evaluate_node(current.operand, depth=depth + 1)
            return operator_func(operand)
        if isinstance(current, ast.Call):
            if not isinstance(current.func, ast.Name):
                raise ValueError("invalid function call")
            func = _ALLOWED_MATH_FUNCS.get(current.func.id)
            if func is None:
                raise ValueError(f"function '{current.func.id}' not allowed")
            if current.keywords:
                raise ValueError("keyword arguments not supported")
            args = [
                _evaluate_node(argument, depth=depth + 1)
                for argument in current.args
            ]
            return func(*args)
        raise ValueError("expression not allowed")

    return _evaluate_node(node)


def _replace_latex_tokens(expr: str, replacements: Mapping[str, str]) -> str:
    """Replace LaTeX-style parameter placeholders with Python names."""

    cleaned = expr
    for latex, name in replacements.items():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(latex)}(?![A-Za-z0-9_])"
        )
        cleaned = pattern.sub(name, cleaned)
    return cleaned


def _reference_target(reference: Any) -> tuple[str, str] | None:
    """Return the kind and name embedded in a contract reference token."""

    if not isinstance(reference, str):
        return None
    match = _CMB_REFERENCE_PATTERN.match(reference)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _freeze_for_cache(value: Any) -> Any:
    """Return a deterministic, hashable representation of ``value``."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_for_cache(value[key]))
            for key in sorted(value, key=str)
        )
    if isinstance(value, numpy.ndarray):
        array_value = numpy.asarray(value, dtype=float)
        return (
            "ndarray",
            tuple(array_value.shape),
            tuple(array_value.ravel()),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, (int, float, numpy.integer, numpy.floating)):
        return float(value)
    return str(value)


class PluginValidationError(RuntimeError):
    """Raised when an engine plugin fails validation."""


def _build_parameter_replacements(
    parameter_names: Sequence[str], latex_names: Sequence[str]
) -> dict[str, str]:
    """Return a mapping from LaTeX placeholders to Python identifiers."""

    replacements: dict[str, str] = {}
    for latex_name, parameter_name in zip(
        latex_names, parameter_names, strict=False
    ):
        if latex_name:
            replacements[str(latex_name).strip("$")] = str(parameter_name)
    return replacements


def _validate_camb_contract_definition(
    contract: Mapping[str, Any],
    parameter_names: Sequence[str],
    latex_names: Sequence[str],
) -> None:
    """Validate the declared CAMB adapter contract."""

    if not isinstance(contract, Mapping):
        raise ValueError("CMB_CONTRACT must be a mapping")

    contract_keys = {str(key) for key in contract.keys()}
    missing_contract_keys = _REQUIRED_CMB_CONTRACT_KEYS - contract_keys
    if missing_contract_keys:
        missing_str = ", ".join(sorted(missing_contract_keys))
        raise ValueError(f"Missing CMB contract key(s): {missing_str}")
    invalid_contract_keys = contract_keys - _SUPPORTED_CMB_CONTRACT_KEYS
    if invalid_contract_keys:
        invalid_str = ", ".join(sorted(invalid_contract_keys))
        raise ValueError(f"Unknown CMB contract key(s): {invalid_str}")

    if contract.get("backend") != _SUPPORTED_CMB_BACKEND:
        raise ValueError("cmb.backend must be 'camb'")

    param_map = contract.get("param_map")
    if not isinstance(param_map, Mapping):
        raise ValueError("cmb.param_map must be a mapping")

    param_map_keys = {str(key) for key in param_map.keys()}
    dynamic_mass_keys = {
        key for key in param_map_keys if _MNU_PATTERN.match(key) is not None
    }
    invalid_param_keys = (
        param_map_keys - _SUPPORTED_CMB_PARAM_KEYS - dynamic_mass_keys
    )
    if invalid_param_keys:
        invalid_str = ", ".join(sorted(invalid_param_keys))
        raise ValueError(
            f"Unsupported CAMB parameter(s) in cmb.param_map: {invalid_str}"
        )
    if "mnu" in param_map_keys and "sum_mnu" in param_map_keys:
        raise ValueError("'mnu' and 'sum_mnu' are mutually exclusive")
    if dynamic_mass_keys and (
        "mnu" in param_map_keys or "sum_mnu" in param_map_keys
    ):
        raise ValueError(
            "individual 'mnuN' entries cannot be combined with 'mnu' or "
            "'sum_mnu'"
        )

    replacements = _build_parameter_replacements(
        parameter_names,
        latex_names,
    )
    model_parameter_names = {str(name) for name in parameter_names}

    grids = contract.get("grids", {}) or {}
    if not isinstance(grids, Mapping):
        raise ValueError("cmb.grids must be a mapping")

    grid_names: set[str] = set()
    grid_symbols: dict[str, str] = {}
    for grid_name, grid_def in grids.items():
        if not isinstance(grid_def, Mapping):
            raise ValueError(f"Grid '{grid_name}' must be a mapping")
        grid_keys = {str(key) for key in grid_def.keys()}
        invalid_grid_keys = grid_keys - _SUPPORTED_CMB_GRID_KEYS
        if invalid_grid_keys:
            invalid_str = ", ".join(sorted(invalid_grid_keys))
            raise ValueError(
                f"Unknown key(s) in cmb.grids.{grid_name}: {invalid_str}"
            )
        symbol = grid_def.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Grid '{grid_name}' must define a symbol")
        symbol = symbol.strip()
        lower = grid_def.get("lower")
        upper = grid_def.get("upper")
        points = grid_def.get("points")
        spacing = grid_def.get("spacing")
        if spacing not in _SUPPORTED_CMB_GRID_SPACING:
            raise ValueError(f"Grid '{grid_name}' must use linear spacing")
        if not isinstance(lower, (int, float, numpy.integer, numpy.floating)):
            raise ValueError(f"Grid '{grid_name}' lower bound must be numeric")
        if not isinstance(upper, (int, float, numpy.integer, numpy.floating)):
            raise ValueError(f"Grid '{grid_name}' upper bound must be numeric")
        if isinstance(points, (int, numpy.integer)):
            point_count = int(points)
        elif isinstance(points, float) and float(points).is_integer():
            point_count = int(points)
        else:
            raise ValueError(f"Grid '{grid_name}' points must be an integer")
        if point_count < 2:
            raise ValueError(
                f"Grid '{grid_name}' must contain at least two points"
            )
        if not (
            numpy.isfinite(float(lower))
            and numpy.isfinite(float(upper))
            and float(upper) > float(lower)
        ):
            raise ValueError(
                f"Grid '{grid_name}' must produce a strictly increasing grid"
            )
        if symbol in model_parameter_names:
            raise ValueError(
                f"Grid symbol '{symbol}' collides with a model parameter"
            )
        if symbol in param_map_keys:
            raise ValueError(
                f"Grid symbol '{symbol}' collides with a CAMB key"
            )
        if symbol in grid_symbols.values():
            raise ValueError(
                f"Grid symbol '{symbol}' is declared more than once"
            )
        grid_names.add(str(grid_name))
        grid_symbols[str(grid_name)] = symbol

    values = contract.get("values", {}) or {}
    if not isinstance(values, Mapping):
        raise ValueError("cmb.values must be a mapping")

    value_names: set[str] = set()
    allowed_value_names = set(model_parameter_names)
    allowed_value_names.update(grid_symbols.values())
    for value_name, value_def in values.items():
        if not isinstance(value_def, Mapping):
            raise ValueError(f"Value '{value_name}' must be a mapping")
        value_keys = {str(key) for key in value_def.keys()}
        invalid_value_keys = value_keys - _SUPPORTED_CMB_VALUE_KEYS
        if invalid_value_keys:
            invalid_str = ", ".join(sorted(invalid_value_keys))
            raise ValueError(
                f"Unknown key(s) in cmb.values.{value_name}: {invalid_str}"
            )
        if (
            value_name in model_parameter_names
            or value_name in param_map_keys
            or value_name in grid_names
            or value_name in grid_symbols.values()
        ):
            raise ValueError(
                f"Value name '{value_name}' collides with a model symbol"
            )
        if value_name in value_names:
            raise ValueError(
                f"Value name '{value_name}' is declared more than once"
            )
        expression = value_def.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Value '{value_name}' must define an expression")
        grid_name = value_def.get("grid")
        if grid_name is not None:
            if not isinstance(grid_name, str):
                raise ValueError(
                    f"Value '{value_name}' grid reference must be a string"
                )
            if grid_name not in grid_names:
                raise ValueError(
                    f"Value '{value_name}' references unknown grid "
                    f"'{grid_name}'"
                )
            allowed_value_names.add(grid_symbols[grid_name])
        clean_expr = _replace_latex_tokens(expression, replacements)
        _validate_safe_expression(clean_expr, allowed_value_names)
        value_names.add(str(value_name))
        allowed_value_names.add(str(value_name))

    calls = contract.get("calls", [])
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise ValueError("cmb.calls must be a list")

    def _iter_references(value: Any) -> Iterator[tuple[str, str]]:
        """Yield reference tokens embedded in ``value`` recursively."""

        reference = _reference_target(value)
        if reference is not None:
            yield reference
            return
        if isinstance(value, Mapping):
            for nested_value in value.values():
                yield from _iter_references(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                yield from _iter_references(nested_value)

    for index, call_def in enumerate(calls):
        if not isinstance(call_def, Mapping):
            raise ValueError(f"Call #{index} must be a mapping")
        call_keys = {str(key) for key in call_def.keys()}
        invalid_call_keys = call_keys - _SUPPORTED_CMB_CALL_KEYS
        if invalid_call_keys:
            invalid_str = ", ".join(sorted(invalid_call_keys))
            raise ValueError(
                f"Unknown key(s) in cmb.calls[{index}]: {invalid_str}"
            )
        method = call_def.get("method")
        if method not in _SUPPORTED_CMB_CALL_METHODS:
            raise ValueError(f"Unsupported CAMB call method: {method!r}")

        args = call_def.get("args", {}) or {}
        kwargs = call_def.get("kwargs", {}) or {}
        if not isinstance(args, Mapping):
            raise ValueError(f"cmb.calls[{index}].args must be a mapping")
        if not isinstance(kwargs, Mapping):
            raise ValueError(f"cmb.calls[{index}].kwargs must be a mapping")

        for reference_kind, reference_name in _iter_references(args):
            if reference_kind == "grid":
                if reference_name not in grid_names:
                    raise ValueError(
                        f"Call '{method}' references unknown grid "
                        f"'{reference_name}'"
                    )
            elif reference_name not in value_names:
                raise ValueError(
                    f"Call '{method}' references unknown value "
                    f"'{reference_name}'"
                )

        for reference_kind, reference_name in _iter_references(kwargs):
            if reference_kind == "grid":
                if reference_name not in grid_names:
                    raise ValueError(
                        f"Call '{method}' references unknown grid "
                        f"'{reference_name}'"
                    )
            elif reference_name not in value_names:
                raise ValueError(
                    f"Call '{method}' references unknown value "
                    f"'{reference_name}'"
                )

        if method == "set_dark_energy":
            allowed_kwargs = {"w", "w0", "wa", "cs2", "dark_energy_model"}
            invalid_kwargs = set(kwargs) - allowed_kwargs
            if invalid_kwargs:
                invalid_str = ", ".join(sorted(invalid_kwargs))
                raise ValueError(
                    f"set_dark_energy does not accept: {invalid_str}"
                )
            if args:
                raise ValueError("set_dark_energy does not accept args")
            if "w" not in kwargs and "w0" not in kwargs:
                raise ValueError("set_dark_energy requires w or w0")
            if "w" in kwargs and "w0" in kwargs:
                raise ValueError("set_dark_energy cannot accept both w and w0")
        elif method == "set_dark_energy_w_a":
            allowed_kwargs = {"dark_energy_model"}
            invalid_kwargs = set(kwargs) - allowed_kwargs
            if invalid_kwargs:
                invalid_str = ", ".join(sorted(invalid_kwargs))
                raise ValueError(
                    "set_dark_energy_w_a does not accept: " f"{invalid_str}"
                )
            required_args = {"a", "w"}
            invalid_args = set(args) - required_args
            if invalid_args:
                invalid_str = ", ".join(sorted(invalid_args))
                raise ValueError(
                    "set_dark_energy_w_a does not accept args: "
                    f"{invalid_str}"
                )
            if not required_args.issubset(args):
                raise ValueError(
                    "set_dark_energy_w_a requires args 'a' and 'w'"
                )

    perturbations = contract.get("perturbations")
    if not isinstance(perturbations, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")
    background_reference_names = set(param_map_keys)
    background_reference_names.update(grid_symbols.values())
    background_reference_names.update(value_names)
    from .perturbation_contract import compile_perturbation_contract

    compile_perturbation_contract(
        perturbations,
        model_name=str(contract.get("model_name", "unknown model")),
        backend=_SUPPORTED_CMB_BACKEND,
        parameter_names=parameter_names,
        latex_names=latex_names,
        background_reference_names=tuple(background_reference_names),
    )


def _validate_cmb_perturbation_definition(
    perturbations: Mapping[str, Any],
    *,
    parameter_names: Sequence[str],
    latex_names: Sequence[str],
    background_reference_names: set[str],
) -> None:
    """Validate the declared CMB perturbation contract."""

    from .perturbation_contract import compile_perturbation_contract

    compile_perturbation_contract(
        perturbations,
        model_name="unknown model",
        backend=_SUPPORTED_CMB_BACKEND,
        parameter_names=parameter_names,
        latex_names=latex_names,
        background_reference_names=tuple(background_reference_names),
    )


def _evaluate_contract_reference(value: Any, env: Mapping[str, Any]) -> Any:
    """Resolve reference tokens while preserving literal strings."""

    reference = _reference_target(value)
    if reference is None:
        return value
    _, name = reference
    if name not in env:
        raise ValueError(f"reference '{value}' not found")
    return env[name]


def _evaluate_contract_payload(
    payload: Mapping[str, Any], env: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate a mapping of contract values against ``env``."""

    evaluated: dict[str, Any] = {}
    for key, raw_value in payload.items():
        if isinstance(raw_value, Mapping):
            evaluated[key] = _evaluate_contract_payload(raw_value, env)
        elif isinstance(raw_value, list):
            evaluated[key] = [
                _evaluate_contract_reference(item, env) for item in raw_value
            ]
        elif isinstance(raw_value, tuple):
            evaluated[key] = tuple(
                _evaluate_contract_reference(item, env) for item in raw_value
            )
        else:
            evaluated[key] = _evaluate_contract_reference(raw_value, env)
    return evaluated


@dataclass(slots=True)
class CAMBParameterEvaluator:
    """Safe evaluator for ``cmb.param_map`` expressions."""

    parameter_names: tuple[str, ...]
    latex_names: tuple[str, ...]
    param_map: Mapping[str, Any]
    logger_name: str = field(default="copernican_lib.engine_adapter")
    _logger: logging.Logger = field(init=False, repr=False)
    _replacements: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Prepare internal helpers after dataclass creation."""

        object.__setattr__(
            self, "_logger", logging.getLogger(self.logger_name)
        )
        object.__setattr__(
            self,
            "_replacements",
            _build_parameter_replacements(
                self.parameter_names,
                self.latex_names,
            ),
        )

    def __call__(self, values: Sequence[float]) -> dict[str, float]:
        """Evaluate the parameter map using the supplied parameter values."""

        env = {
            name: float(value)
            for name, value in zip(self.parameter_names, values, strict=False)
        }
        results: dict[str, float] = {}
        for key, expr in self.param_map.items():
            if isinstance(expr, str):
                clean_expr = _replace_latex_tokens(expr, self._replacements)
                results[key] = _coerce_numeric_scalar(
                    _evaluate_safe_expression(clean_expr, env)
                )
            else:
                results[key] = _coerce_numeric_scalar(expr)
        return results


@dataclass(slots=True)
class CAMBContractEvaluator:
    """Evaluate a full CAMB adapter contract for a plugin."""

    parameter_names: tuple[str, ...]
    latex_names: tuple[str, ...]
    contract: Mapping[str, Any]
    logger_name: str = field(default="copernican_lib.engine_adapter")
    _logger: logging.Logger = field(init=False, repr=False)
    _replacements: dict[str, str] = field(init=False, repr=False)
    _param_evaluator: CAMBParameterEvaluator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Prepare helpers and validate the contract structure."""

        object.__setattr__(
            self, "_logger", logging.getLogger(self.logger_name)
        )
        object.__setattr__(
            self,
            "_replacements",
            _build_parameter_replacements(
                self.parameter_names,
                self.latex_names,
            ),
        )
        _validate_camb_contract_definition(
            self.contract,
            self.parameter_names,
            self.latex_names,
        )
        object.__setattr__(
            self,
            "_param_evaluator",
            CAMBParameterEvaluator(
                self.parameter_names,
                self.latex_names,
                self.contract.get("param_map", {}),
                logger_name=self.logger_name,
            ),
        )

    def _evaluate_grid(
        self, grid_name: str, grid_def: Mapping[str, Any]
    ) -> numpy.ndarray:
        """Return a finite, strictly increasing grid array."""

        lower = float(grid_def["lower"])
        upper = float(grid_def["upper"])
        points = int(grid_def["points"])
        grid = numpy.linspace(lower, upper, points, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(f"Grid '{grid_name}' must be one-dimensional")
        if not numpy.all(numpy.isfinite(grid)):
            raise ValueError(f"Grid '{grid_name}' must be finite")
        if not numpy.all(numpy.diff(grid) > 0.0):
            raise ValueError(f"Grid '{grid_name}' must be strictly increasing")
        return grid

    def _evaluate_value(
        self,
        value_name: str,
        value_def: Mapping[str, Any],
        env: Mapping[str, Any],
    ) -> Any:
        """Return a scalar or array value evaluated from the contract."""

        expression = value_def["expression"]
        clean_expr = _replace_latex_tokens(expression, self._replacements)
        result = _evaluate_safe_expression(clean_expr, env)
        grid_name = value_def.get("grid")
        if grid_name is None:
            return _coerce_numeric_scalar(result)

        grid_def = self.contract["grids"][grid_name]
        grid_symbol = grid_def["symbol"]
        grid_array = numpy.asarray(env[grid_symbol], dtype=float)
        if numpy.isscalar(result) or numpy.asarray(result).ndim == 0:
            value_array = numpy.full_like(
                grid_array, float(result), dtype=float
            )
        else:
            value_array = _coerce_numeric_array(result, name=value_name)
        if value_array.shape != grid_array.shape:
            raise ValueError(
                f"Value '{value_name}' must match grid '{grid_name}'"
            )
        if not numpy.all(numpy.isfinite(value_array)):
            raise ValueError(f"Value '{value_name}' must be finite")
        return value_array

    def _evaluate_call(
        self,
        method_name: str,
        call_def: Mapping[str, Any],
        env: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the evaluated call payload for the structured contract."""

        args = _evaluate_contract_payload(call_def.get("args", {}) or {}, env)
        kwargs = _evaluate_contract_payload(
            call_def.get("kwargs", {}) or {},
            env,
        )
        if method_name == "set_dark_energy":
            return {
                "method": method_name,
                "args": {},
                "kwargs": kwargs,
            }
        return {
            "method": method_name,
            "args": args,
            "kwargs": kwargs,
        }

    def evaluate_param_map(self, values: Sequence[float]) -> dict[str, float]:
        """Return the evaluated scalar CAMB parameter map."""

        return self._param_evaluator(values)

    def __call__(self, values: Sequence[float]) -> dict[str, Any]:
        """Return the fully evaluated CAMB adapter contract."""

        param_map = self.evaluate_param_map(values)
        model_env = {
            name: float(value)
            for name, value in zip(self.parameter_names, values, strict=False)
        }
        evaluated: dict[str, Any] = {
            "model_name": self.contract.get("model_name"),
            "backend": self.contract.get("backend", _SUPPORTED_CMB_BACKEND),
            "param_map": param_map,
            "grids": {},
            "values": {},
            "calls": [],
        }
        env: dict[str, Any] = dict(model_env)
        env.update(param_map)

        grid_defs = self.contract.get("grids", {}) or {}
        for grid_name, grid_def in grid_defs.items():
            grid_array = self._evaluate_grid(grid_name, grid_def)
            evaluated["grids"][grid_name] = grid_array
            env[grid_name] = grid_array
            env[grid_def["symbol"]] = grid_array

        value_defs = self.contract.get("values", {}) or {}
        for value_name, value_def in value_defs.items():
            value = self._evaluate_value(value_name, value_def, env)
            evaluated["values"][value_name] = value
            env[value_name] = value

        for call_def in self.contract.get("calls", []) or []:
            method_name = call_def["method"]
            evaluated_call = self._evaluate_call(
                method_name,
                call_def,
                env,
            )
            evaluated["calls"].append(evaluated_call)

        return evaluated


class FrozenMapping(Mapping[str, Any]):
    """Picklable read-only mapping used to freeze plugin metadata."""

    __slots__ = ("_data",)

    def __init__(self, source: Mapping[str, Any] | None = None) -> None:
        """Store a shallow copy of ``source`` for later access.

        ``types.MappingProxyType`` offered similar immutability but refused to
        pickle on Python 3.11, triggering ``TypeError: cannot pickle
        'mappingproxy' object`` exceptions inside ``multiprocessing`` spawn
        pools.  Engines serialise :class:`EnginePlugin` instances whenever they
        hand work to worker processes, so the read-only wrapper must cooperate
        with ``pickle``.
        """

        self._data = dict(source or {})

    def __getitem__(self, key: str) -> Any:
        """Return the stored value for *key* (supports pickling)."""

        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Yield the recorded keys for mapping iteration."""

        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of stored entries."""

        return len(self._data)

    def __repr__(self) -> str:
        """Show a repr that reveals the wrapped data."""

        return f"FrozenMapping({self._data!r})"

    def __getstate__(self) -> dict[str, Any]:
        """Return a picklable snapshot for ``pickle`` round-trips."""

        return self._data

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`__getstate__`."""

        object.__setattr__(self, "_data", dict(state))

    def to_dict(self) -> dict[str, Any]:
        """Expose a shallow copy for callers needing a mutable mapping."""

        return dict(self._data)


@dataclass(slots=True)
class EnginePlugin:
    """Container describing a generated model and CAMB contracts."""

    MODEL_NAME: str
    MODEL_DESCRIPTION: str
    MODEL_ABSTRACT: str
    PARAMETER_NAMES: tuple[str, ...]
    PARAMETER_LATEX_NAMES: tuple[str, ...]
    PARAMETER_UNITS: tuple[str, ...]
    INITIAL_GUESSES: tuple[float, ...]
    PARAMETER_BOUNDS: tuple[tuple[float | None, float | None], ...]
    FIXED_PARAMS: Mapping[str, float]
    PARAMETER_PRIORS: tuple[Mapping[str, Any], ...]
    PARAMETER_PRIOR_OBJECTS: tuple[prior_lib.BasePrior | None, ...]
    PARAMETER_TRANSFORMS: tuple[Callable[[float], Any] | None, ...] | None
    valid_for_distance_metrics: bool
    valid_for_bao: bool
    valid_for_cmb: bool
    CMB_CONTRACT: Mapping[str, Any]
    CMB_PARAM_MAP: Mapping[str, Any]
    CMB_PERTURBATION_CONTRACT: Mapping[str, Any]
    CMB_PERTURBATION_STANDARD: bool
    CMB_PERTURBATION_DATA: Any
    LIKELIHOOD_CONFIG: Mapping[str, Any]
    MODEL_EQUATIONS_LATEX_SN: tuple[str, ...]
    MODEL_EQUATIONS_LATEX_BAO: tuple[str, ...]
    MODEL_FILENAME: str | None
    distance_modulus_model: Callable[..., Any] | None
    get_comoving_distance_Mpc: Callable[..., Any] | None
    get_luminosity_distance_Mpc: Callable[..., Any] | None
    get_angular_diameter_distance_Mpc: Callable[..., Any] | None
    get_Hz_per_Mpc: Callable[..., Any] | None
    get_DV_Mpc: Callable[..., Any] | None
    get_sound_horizon_rs_Mpc: Callable[..., Any] | None
    compute_cmb_spectrum: Callable[..., Any] | None
    compute_cmb_spectrum_from_dict: Callable[..., Any] | None
    extras: Mapping[str, Any] = field(default_factory=dict)
    _camb_evaluator: CAMBContractEvaluator | None = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Normalise extras and prepare the CAMB evaluators."""

        self.extras = FrozenMapping(self.extras)
        self.CMB_CONTRACT = copy.deepcopy(self.CMB_CONTRACT or {})
        self.CMB_PARAM_MAP = copy.deepcopy(self.CMB_PARAM_MAP or {})
        self.CMB_PERTURBATION_CONTRACT = copy.deepcopy(
            self.CMB_PERTURBATION_CONTRACT or {}
        )
        self.CMB_PERTURBATION_STANDARD = self.CMB_PERTURBATION_CONTRACT.get(
            "standard",
            False,
        )
        if self.valid_for_cmb and self.CMB_CONTRACT:
            evaluator = CAMBContractEvaluator(
                self.PARAMETER_NAMES,
                self.PARAMETER_LATEX_NAMES,
                self.CMB_CONTRACT,
            )
            object.__setattr__(self, "_camb_evaluator", evaluator)
        else:
            object.__setattr__(self, "_camb_evaluator", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookups to the extras mapping when missing."""

        if name == "extras":
            raise AttributeError(name)

        try:
            extras = object.__getattribute__(self, "extras")
        except AttributeError as exc:
            raise AttributeError(name) from exc

        try:
            return extras[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self) -> list[str]:
        """Include extras keys alongside the normal attribute list."""

        default = set(super().__dir__())
        default.update(self.extras.keys())
        return sorted(default)

    def get_camb_params(self, values: Sequence[float]) -> dict[str, float]:
        """Return CAMB parameters derived from ``values``."""

        evaluator = getattr(self, "_camb_evaluator", None)
        if evaluator is None:
            return {}
        return evaluator.evaluate_param_map(values)

    def get_camb_contract(self, values: Sequence[float]) -> dict[str, Any]:
        """Return the fully evaluated CAMB adapter contract."""

        evaluator = getattr(self, "_camb_evaluator", None)
        if evaluator is None:
            raise ValueError("Model does not declare a CAMB contract")
        evaluated = evaluator(values)
        evaluated["model_parameters"] = {
            name: float(value)
            for name, value in zip(self.PARAMETER_NAMES, values, strict=False)
        }
        evaluated["value_definitions"] = copy.deepcopy(
            self.CMB_CONTRACT.get("values", {}) or {}
        )
        evaluated["model_name"] = self.MODEL_NAME
        evaluated["backend"] = self.CMB_CONTRACT.get(
            "backend", _SUPPORTED_CMB_BACKEND
        )
        return evaluated

    def get_cmb_perturbation_contract(
        self, values: Sequence[float]
    ) -> dict[str, Any]:
        """Return the declared CMB perturbation contract."""

        del values
        if not self.CMB_PERTURBATION_CONTRACT:
            raise ValueError(
                "Model does not declare a CMB perturbation contract"
            )
        contract = copy.deepcopy(self.CMB_PERTURBATION_CONTRACT)
        contract["model_name"] = self.MODEL_NAME
        contract["backend"] = self.CMB_CONTRACT.get(
            "backend", _SUPPORTED_CMB_BACKEND
        )
        return contract

    def get_cmb_perturbation_data(self, values: Sequence[float]) -> Any:
        """Return the compiled CMB perturbation data."""

        del values
        perturbation_data = getattr(self, "CMB_PERTURBATION_DATA", None)
        if perturbation_data is None:
            raise ValueError("Model does not declare CMB perturbation data")
        return perturbation_data


def sanitize_equation(equation_line: str) -> str:
    """Return a Matplotlib-friendly LaTeX string."""

    if not isinstance(equation_line, str):
        return ""
    equation = equation_line.strip()
    equation = re.sub(r"^\$+", "", equation)
    equation = re.sub(r"\$+$", "", equation)
    return f"${equation.strip()}$" if equation else ""


def _prepare_priors(
    params: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[prior_lib.BasePrior | None, ...],
    tuple[Callable[[float], Any] | None, ...] | None,
    Mapping[str, float],
]:
    """Build prior metadata, transform callables and fixed constants."""
    prior_mappings: list[Mapping[str, Any]] = []
    prior_objects: list[prior_lib.BasePrior | None] = []
    transforms: list[Callable[[float], Any] | None] = []
    fixed_params: dict[str, float] = {}

    for param in params:
        raw_prior = param.get("prior") or {}
        prior_obj: prior_lib.BasePrior | None = None
        if raw_prior:
            try:
                prior_obj = prior_lib.prior_from_mapping(raw_prior)
            except prior_lib.PriorError as exc:
                raise ValueError(str(exc)) from exc
            prior_mappings.append(prior_obj.to_mapping())
            transforms.append(prior_obj.create_transform())
        else:
            prior_mappings.append({})
            transforms.append(None)
        prior_objects.append(prior_obj)
        if isinstance(prior_obj, prior_lib.FixedPrior):
            prior_value = prior_obj.fixed_value
            python_var = param.get("python_var") or param.get("name")
            if python_var:
                fixed_params[python_var] = prior_value
                fixed_params[python_var.upper()] = prior_value
            latex_name = param.get("latex_name")
            if isinstance(latex_name, str) and latex_name:
                fixed_params[latex_name.strip("$")] = prior_value

    if any(transform is not None for transform in transforms):
        transform_tuple: tuple[Callable[[float], Any] | None, ...] | None = (
            tuple(transforms)
        )
    else:
        transform_tuple = None

    return (
        tuple(prior_mappings),
        tuple(prior_objects),
        transform_tuple,
        FrozenMapping(fixed_params),
    )


def build_engine_plugin(
    model_data: Mapping[str, Any],
    func_dict: Mapping[str, Callable[..., Any]],
) -> EnginePlugin:
    """Return an :class:`EnginePlugin` for the provided model."""

    params: Sequence[Mapping[str, Any]] = model_data.get("parameters", [])
    names = tuple(
        param.get("python_var", param.get("name")) for param in params
    )
    latex_names = tuple(param.get("latex_name", "") for param in params)
    units = tuple(param.get("unit", "") for param in params)
    guesses = tuple(
        sum(param.get("bounds", (0.0, 0.0))) / 2.0 for param in params
    )
    bounds = tuple(
        tuple(param.get("bounds", (None, None))) for param in params
    )

    (
        prior_mappings,
        prior_objects,
        transforms,
        fixed_params,
    ) = _prepare_priors(params)

    likelihood_config = model_data.get("likelihood", {}) or {}
    cmb_contract = model_data.get("cmb", {}) or {}
    perturbation_contract = cmb_contract.get("perturbations", {}) or {}
    background_reference_names = {
        str(key) for key in (cmb_contract.get("param_map", {}) or {})
    }
    for grid_def in (cmb_contract.get("grids", {}) or {}).values():
        if isinstance(grid_def, Mapping):
            symbol = grid_def.get("symbol")
            if isinstance(symbol, str) and symbol.strip():
                background_reference_names.add(symbol.strip())
    background_reference_names.update(
        str(key) for key in (cmb_contract.get("values", {}) or {})
    )

    perturbation_data = None
    if model_data.get("valid_for_cmb", True):
        from .perturbation_contract import compile_perturbation_contract

        perturbation_data = compile_perturbation_contract(
            perturbation_contract,
            model_name=model_data.get("model_name", "GeneratedModel"),
            backend=cmb_contract.get("backend", _SUPPORTED_CMB_BACKEND),
            parameter_names=names,
            latex_names=latex_names,
            background_reference_names=tuple(background_reference_names),
        )

    extras: MutableMapping[str, Any] = {}
    known_names = set(REQUIRED_FUNCTIONS).union(_OPTIONAL_FUNCTIONS)
    functions = {name: func_dict.get(name) for name in known_names}
    for name, func in func_dict.items():
        if name not in known_names:
            extras[name] = func

    equations = model_data.get("equations", {})
    sne_eqs = tuple(
        sanitize_equation(equation) for equation in equations.get("sne", [])
    )
    bao_eqs = tuple(
        sanitize_equation(equation) for equation in equations.get("bao", [])
    )

    plugin = EnginePlugin(
        MODEL_NAME=model_data.get("model_name", "GeneratedModel"),
        MODEL_DESCRIPTION=model_data.get("description", ""),
        MODEL_ABSTRACT=model_data.get("abstract", ""),
        PARAMETER_NAMES=names,
        PARAMETER_LATEX_NAMES=latex_names,
        PARAMETER_UNITS=units,
        INITIAL_GUESSES=guesses,
        PARAMETER_BOUNDS=bounds,
        FIXED_PARAMS=fixed_params,
        PARAMETER_PRIORS=prior_mappings,
        PARAMETER_PRIOR_OBJECTS=prior_objects,
        PARAMETER_TRANSFORMS=transforms,
        valid_for_distance_metrics=model_data.get(
            "valid_for_distance_metrics", True
        ),
        valid_for_bao=model_data.get("valid_for_bao", True),
        valid_for_cmb=model_data.get("valid_for_cmb", True),
        CMB_CONTRACT=cmb_contract,
        CMB_PARAM_MAP=cmb_contract.get("param_map", {}),
        CMB_PERTURBATION_CONTRACT=perturbation_contract,
        CMB_PERTURBATION_STANDARD=perturbation_contract.get("standard", False),
        CMB_PERTURBATION_DATA=perturbation_data,
        LIKELIHOOD_CONFIG=likelihood_config,
        MODEL_EQUATIONS_LATEX_SN=sne_eqs,
        MODEL_EQUATIONS_LATEX_BAO=bao_eqs,
        MODEL_FILENAME=model_data.get("filename"),
        distance_modulus_model=functions.get("distance_modulus_model"),
        get_comoving_distance_Mpc=functions.get("get_comoving_distance_Mpc"),
        get_luminosity_distance_Mpc=functions.get(
            "get_luminosity_distance_Mpc"
        ),
        get_angular_diameter_distance_Mpc=functions.get(
            "get_angular_diameter_distance_Mpc"
        ),
        get_Hz_per_Mpc=functions.get("get_Hz_per_Mpc"),
        get_DV_Mpc=functions.get("get_DV_Mpc"),
        get_sound_horizon_rs_Mpc=functions.get("get_sound_horizon_rs_Mpc"),
        compute_cmb_spectrum=functions.get("compute_cmb_spectrum"),
        compute_cmb_spectrum_from_dict=functions.get(
            "compute_cmb_spectrum_from_dict"
        ),
        extras=extras,
    )

    return plugin


def build_plugin(
    model_data: Mapping[str, Any],
    func_dict: Mapping[str, Callable[..., Any]],
) -> EnginePlugin:
    """Return a validated :class:`EnginePlugin` for the provided model."""

    plugin = build_engine_plugin(model_data, func_dict)
    validate_plugin(plugin)
    return plugin


def _validate_plugin_cmb_contract(plugin: EnginePlugin) -> None:
    """Validate the plugin's declared CAMB adapter contract."""

    if not getattr(plugin, "valid_for_cmb", True):
        return
    contract = getattr(plugin, "CMB_CONTRACT", {}) or {}
    if not isinstance(contract, Mapping):
        raise ValueError("CMB_CONTRACT must be a mapping")
    _validate_camb_contract_definition(
        contract,
        getattr(plugin, "PARAMETER_NAMES", ()),
        getattr(plugin, "PARAMETER_LATEX_NAMES", ()),
    )


def validate_plugin(plugin: EnginePlugin) -> bool:
    """Validate that ``plugin`` exposes required attributes and callables."""

    errors: list[str] = []
    missing_attrs = [
        attr for attr in REQUIRED_ATTRIBUTES if not hasattr(plugin, attr)
    ]
    if missing_attrs:
        missing_list = ", ".join(sorted(missing_attrs))
        errors.append(f"Missing attributes: {missing_list}")

    required_funcs: list[str] = []
    if getattr(plugin, "valid_for_distance_metrics", True):
        required_funcs.extend(
            [
                "distance_modulus_model",
                "get_comoving_distance_Mpc",
                "get_luminosity_distance_Mpc",
                "get_angular_diameter_distance_Mpc",
                "get_Hz_per_Mpc",
            ]
        )
    if getattr(plugin, "valid_for_bao", True):
        required_funcs.extend(["get_DV_Mpc", "get_sound_horizon_rs_Mpc"])
    if not required_funcs:
        required_funcs = ["distance_modulus_model"]

    if getattr(plugin, "valid_for_cmb", True):
        required_funcs.extend(
            [
                "get_camb_params",
                "get_camb_contract",
                "get_cmb_perturbation_contract",
                "get_cmb_perturbation_data",
            ]
        )

    for fname in required_funcs:
        func = getattr(plugin, fname, None)
        if not callable(func):
            errors.append(f"Missing callable '{fname}'")
            continue
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            errors.append(f"Unable to inspect callable '{fname}'")
            continue
        if not sig.parameters:
            errors.append(
                f"Callable '{fname}' must accept at least one parameter"
            )

    if getattr(plugin, "valid_for_cmb", True):
        perturbation_contract = getattr(
            plugin, "CMB_PERTURBATION_CONTRACT", {}
        )
        if not isinstance(perturbation_contract, Mapping):
            errors.append("CMB_PERTURBATION_CONTRACT must be a mapping")
        if not isinstance(
            getattr(plugin, "CMB_PERTURBATION_STANDARD", None),
            bool,
        ):
            errors.append("CMB_PERTURBATION_STANDARD must be boolean")
        if getattr(plugin, "CMB_PERTURBATION_DATA", None) is None:
            errors.append("CMB_PERTURBATION_DATA must be present")

    try:
        _validate_plugin_cmb_contract(plugin)
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        model_name = getattr(plugin, "MODEL_NAME", "engine plugin")
        joined = "; ".join(errors)
        for entry in errors:
            LOGGER.error(
                "Plugin validation issue for %s: %s", model_name, entry
            )
        raise PluginValidationError(
            f"Validation failed for {model_name}: {joined}"
        )

    return True


__all__ = [
    "CMB_BACKEND_CAPABILITIES",
    "EnginePlugin",
    "CAMBParameterEvaluator",
    "CAMBContractEvaluator",
    "FrozenMapping",
    "PluginValidationError",
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_FUNCTIONS",
    "PosteriorEvaluator",
    "build_engine_plugin",
    "build_plugin",
    "make_logposterior",
    "sanitize_equation",
    "validate_plugin",
]
