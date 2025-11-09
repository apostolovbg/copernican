"""Runtime plugin assembly utilities for engine integrations.

**Last Updated:** 2025-11-09

The legacy :mod:`copernican_lib.engine_interface` module combined plugin
construction, validation, CAMB parameter synthesis and posterior helpers in a
single 500+ line file. That structure complicated multiprocessing support and
made it difficult to reason about picklability when new features were added.
The refreshed layout promotes plugin assembly into a dedicated package so the
API surface is explicit, picklable and thoroughly documented. Engines now
operate on the :class:`EnginePlugin` dataclass, which stores metadata, dataset
compatibility toggles and distance functions in a predictable, serialisable
form. Optional helpers such as ``compute_cmb_spectrum`` are captured in an
``extras`` mapping so future extensions remain backwards compatible without
silently mutating ``__dict__`` structures.

The module exposes three primary entry points:

``build_engine_plugin``
    Normalises parsed YAML metadata and generated callables into an
    :class:`EnginePlugin`. The builder eagerly converts lists into tuples to
    encourage immutability and caches a picklable CAMB expression evaluator for
    models that expose ``cmb.param_map`` definitions.

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
import inspect
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

from .. import priors as prior_lib

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
]

_OPTIONAL_FUNCTIONS: tuple[str, ...] = (
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_from_dict",
)

_ALLOWED_MATH_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}
_BIN_OPS = {
    ast.Add: math.fsum,
    ast.Sub: lambda x, y: x - y,
    ast.Mult: lambda x, y: x * y,
    ast.Div: lambda x, y: x / y,
    ast.Pow: lambda x, y: x**y,
}
_UNARY_OPS = {ast.UAdd: lambda x: x, ast.USub: lambda x: -x}


class PluginValidationError(RuntimeError):
    """Raised when an engine plugin fails validation."""


@dataclass(slots=True)
class CAMBParameterEvaluator:
    """Safe evaluator for ``cmb.param_map`` expressions."""

    parameter_names: tuple[str, ...]
    latex_names: tuple[str, ...]
    param_map: Mapping[str, Any]
    logger_name: str = field(default="copernican_lib.plugins")
    _logger: logging.Logger = field(init=False, repr=False)
    _replacements: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_logger", logging.getLogger(self.logger_name)
        )
        object.__setattr__(
            self,
            "_replacements",
            {
                latex.strip("$"): name
                for latex, name in zip(self.latex_names, self.parameter_names)
            },
        )

    def __call__(self, values: Sequence[float]) -> dict[str, float]:
        env = {
            name: float(val) for name, val in zip(self.parameter_names, values)
        }
        results: dict[str, float] = {}
        for key, expr in self.param_map.items():
            if isinstance(expr, str):
                clean_expr = self._replace_latex(expr)
                results[key] = float(self._eval_expression(clean_expr, env))
            else:
                results[key] = float(expr)
        return results

    def _replace_latex(self, expr: str) -> str:
        cleaned = expr
        for latex, name in self._replacements.items():
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(latex)}(?![A-Za-z0-9_])"
            )
            cleaned = pattern.sub(name, cleaned)
        return cleaned

    def _eval_expression(self, expr: str, env: Mapping[str, float]) -> float:
        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - guarded by validation
            message = f"invalid expression '{expr}'"
            self._logger.error("(CAMBParameterEvaluator): %s", message)
            raise ValueError(message) from exc

        if sum(1 for _ in ast.walk(node)) > 100:
            raise ValueError("expression too complex")

        return self._eval_node(node.body, env, depth=0)

    def _eval_node(
        self,
        node: ast.AST,
        env: Mapping[str, float],
        *,
        depth: int,
    ) -> float:
        if depth > 20:
            raise ValueError("expression too complex")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("non-numeric literal")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in _ALLOWED_CONSTANTS:
                return _ALLOWED_CONSTANTS[node.id]
            raise ValueError(f"name '{node.id}' not allowed")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError("operator not allowed")
            left = self._eval_node(node.left, env, depth=depth + 1)
            right = self._eval_node(node.right, env, depth=depth + 1)
            if op is math.fsum:
                return math.fsum((left, right))
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("operator not allowed")
            operand = self._eval_node(node.operand, env, depth=depth + 1)
            return op(operand)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("invalid function call")
            func = _ALLOWED_MATH_FUNCS.get(node.func.id)
            if func is None:
                raise ValueError(f"function '{node.func.id}' not allowed")
            if node.keywords:
                raise ValueError("keyword arguments not supported")
            args = [
                self._eval_node(arg, env, depth=depth + 1) for arg in node.args
            ]
            return float(func(*args))
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, env, depth=depth + 1)
        raise ValueError("expression not allowed")


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
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
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
    """Container describing a generated cosmological model."""

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
    CMB_PARAM_MAP: Mapping[str, Any]
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
    _camb_evaluator: CAMBParameterEvaluator | None = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.extras = FrozenMapping(self.extras)
        if self.valid_for_cmb and not self.CMB_PARAM_MAP:
            LOGGER.warning(
                "Model marked valid_for_cmb but no cmb.param_map provided. "
                "Disabling CMB support.",
            )
            object.__setattr__(self, "valid_for_cmb", False)
        if self.valid_for_cmb:
            evaluator = CAMBParameterEvaluator(
                self.PARAMETER_NAMES,
                self.PARAMETER_LATEX_NAMES,
                self.CMB_PARAM_MAP,
            )
            object.__setattr__(self, "_camb_evaluator", evaluator)
        else:
            object.__setattr__(self, "_camb_evaluator", None)

    def __getattr__(self, name: str) -> Any:
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
        default = set(super().__dir__())
        default.update(self.extras.keys())
        return sorted(default)

    def get_camb_params(self, values: Sequence[float]) -> dict[str, float]:
        """Return CAMB parameters derived from ``values``."""

        evaluator = getattr(self, "_camb_evaluator", None)
        if evaluator is None:
            return {}
        return evaluator(values)


def sanitize_equation(eq_line: str) -> str:
    """Return a Matplotlib-friendly LaTeX string."""

    if not isinstance(eq_line, str):
        return ""
    eq = eq_line.strip()
    eq = re.sub(r"^\$+", "", eq)
    eq = re.sub(r"\$+$", "", eq)
    return f"${eq.strip()}$" if eq else ""


def _prepare_priors(
    params: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[prior_lib.BasePrior | None, ...],
    tuple[Callable[[float], Any] | None, ...] | None,
    Mapping[str, float],
]:
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
            value = prior_obj.value
            python_var = param.get("python_var") or param.get("name")
            if python_var:
                fixed_params[python_var] = value
                fixed_params[python_var.upper()] = value
            latex_name = param.get("latex_name")
            if isinstance(latex_name, str) and latex_name:
                fixed_params[latex_name.strip("$")] = value

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

    extras: MutableMapping[str, Any] = {}
    known_names = set(REQUIRED_FUNCTIONS).union(_OPTIONAL_FUNCTIONS)
    functions = {name: func_dict.get(name) for name in known_names}
    for name, func in func_dict.items():
        if name not in known_names:
            extras[name] = func

    equations = model_data.get("equations", {})
    sne_eqs = tuple(sanitize_equation(eq) for eq in equations.get("sne", []))
    bao_eqs = tuple(sanitize_equation(eq) for eq in equations.get("bao", []))

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
        CMB_PARAM_MAP=model_data.get("cmb", {}).get("param_map", {}),
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
    "EnginePlugin",
    "CAMBParameterEvaluator",
    "PluginValidationError",
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_FUNCTIONS",
    "build_engine_plugin",
    "sanitize_equation",
    "validate_plugin",
]
