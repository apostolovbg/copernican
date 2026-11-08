"""Physical phase-space declarations shared by planning and evolution.

Momentum is dimensionless, q=p/T(today). Distribution expressions use the
same restricted arithmetic grammar as other model declarations. Numerical
support and quadrature density remain decisions of the solver.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping

import numpy

from .model_adapter import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _evaluate_safe_expression,
    _parse_safe_expression,
)

_DISTRIBUTIONS = {
    "fermi_dirac": ("exp(-q) / (1 + exp(-q))", "-q / (1 + exp(-q))"),
    "bose_einstein": ("exp(-q) / (1 - exp(-q))", "-q / (1 - exp(-q))"),
    "maxwell_boltzmann": ("exp(-q)", "-q"),
}


def validate_phase_space(
    declaration: Mapping[str, Any], *, available_parameters=None
) -> dict[str, Any]:
    """Validate physical metadata without supplying an undeclared law."""

    if not isinstance(declaration, Mapping):
        raise ValueError("Phase-space metadata must be a mapping")
    allowed = {
        "distribution",
        "occupation",
        "log_derivative",
        "mass_parameter",
        "multiplicity_parameter",
        "temperature_ratio",
        "density_parameter",
    }
    unknown = set(declaration) - allowed
    if unknown:
        raise ValueError(f"Unknown phase-space fields: {sorted(unknown)}")
    result = dict(declaration)
    distribution = str(result.get("distribution", "expression"))
    if distribution == "expression":
        if not all(
            name in result for name in ("occupation", "log_derivative")
        ):
            raise ValueError(
                "An expression distribution needs occupation and "
                "log_derivative"
            )
    elif distribution not in _DISTRIBUTIONS:
        raise ValueError(f"Unknown phase-space distribution: {distribution}")
    elif "occupation" in result or "log_derivative" in result:
        raise ValueError("Named distributions cannot override their laws")
    result["distribution"] = distribution
    for name in ("occupation", "log_derivative", "temperature_ratio"):
        if name in result:
            _parse_safe_expression(str(result[name]))
    if "temperature_ratio" not in result:
        raise ValueError("Phase space must declare temperature_ratio")
    if not result.get("mass_parameter"):
        raise ValueError("Phase space must declare mass_parameter")
    if available_parameters is not None:
        available = set(available_parameters)
        for field_name in (
            "mass_parameter",
            "multiplicity_parameter",
            "density_parameter",
        ):
            if result.get(field_name) and result[field_name] not in available:
                raise ValueError(
                    f"Phase-space {field_name} is undeclared: "
                    f"{result[field_name]}"
                )
        for field_name in (
            "occupation",
            "log_derivative",
            "temperature_ratio",
        ):
            if field_name not in result:
                continue
            expression = _parse_safe_expression(str(result[field_name]))
            names = {
                node.id
                for node in ast.walk(expression)
                if isinstance(node, ast.Name)
            }
            allowed = available | set(_ALLOWED_MATH_FUNCS)
            allowed |= set(_ALLOWED_CONSTANTS)
            if field_name != "temperature_ratio":
                allowed.add("q")
            missing = names - allowed
            if missing:
                raise ValueError(
                    f"Phase-space {field_name} has undeclared names: "
                    + ", ".join(sorted(missing))
                )
    return result


def sample_distribution(
    q_values: Any,
    declaration: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    *,
    allow_zero: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Evaluate occupation and d ln(f)/d ln(q) on actual positive nodes."""

    points = numpy.asarray(q_values, dtype=float)
    if (
        points.ndim != 1
        or not points.size
        or not numpy.all(numpy.isfinite(points) & (points > 0))
    ):
        raise ValueError("Distribution nodes must be finite positive vectors")
    kind = str(declaration.get("distribution", "expression"))
    if kind == "expression":
        expressions = (
            str(declaration["occupation"]),
            str(declaration["log_derivative"]),
        )
    else:
        try:
            expressions = _DISTRIBUTIONS[kind]
        except KeyError as error:
            raise ValueError(f"Unknown distribution: {kind}") from error
    context = {**dict(parameters or {}), "q": points}
    values = tuple(
        numpy.broadcast_to(
            numpy.asarray(
                _evaluate_safe_expression(expr, context), dtype=float
            ),
            points.shape,
        ).copy()
        for expr in expressions
    )
    if not all(numpy.all(numpy.isfinite(value)) for value in values):
        raise ValueError("Distribution produced non-finite values")
    if numpy.any(values[0] < 0) or (
        not allow_zero and not numpy.any(values[0] > 0)
    ):
        raise ValueError("Occupation must be nonnegative and nonzero")
    return values


def phase_space_temperature(
    declaration: Mapping[str, Any], parameters: Mapping[str, Any], tcmb: float
) -> float:
    """Resolve the declared present temperature in eV, including heating."""

    ratio = float(
        _evaluate_safe_expression(
            str(declaration["temperature_ratio"]), parameters
        )
    )
    temperature = ratio * float(tcmb) * 8.617333262145e-5
    if not numpy.isfinite(temperature) or temperature <= 0:
        raise ValueError("Phase-space temperature must be finite and positive")
    return temperature


def phase_space_grid_name(name: str, entry: Any) -> str:
    """Return a declared topology identifier, independent of species labels."""

    values = (
        entry
        if isinstance(entry, Mapping)
        else {
            key: getattr(entry, key, None)
            for key in ("momentum_grid", "physical_role")
        }
    )
    explicit = values.get("momentum_grid")
    if explicit:
        return str(explicit)
    if values.get("physical_role") == "massive_neutrino":
        return "massive_neutrino_default"
    return str(name) + "_q"


def plan_distribution_grid(
    declaration: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-4,
    allow_unbound: bool = False,
) -> dict[str, Any]:
    """Resolve support and log-q sampling from independent physical moments.

    Expanded support and independently denser quadrature are tested
    separately. The relativistic and nonrelativistic moment limits bracket
    the mass transition. Nonintegrable or stagnant tails fail explicitly.
    """

    dependencies = set()
    for field in ("occupation", "log_derivative"):
        if field in declaration:
            dependencies.update(
                node.id
                for node in ast.walk(
                    _parse_safe_expression(str(declaration[field]))
                )
                if isinstance(node, ast.Name)
            )
    missing = (
        dependencies
        - set(parameters)
        - {"q"}
        - set(_ALLOWED_CONSTANTS)
        - set(_ALLOWED_MATH_FUNCS)
    )
    if missing:
        if not allow_unbound:
            raise ValueError(
                f"Unbound distribution parameters: {sorted(missing)}"
            )
        return {
            "count": 32,
            "q_min": 0.02,
            "q_max": 24.0,
            "planning_evidence": {
                "status": "deferred_until_parameter_binding",
                "unbound_parameters": sorted(missing),
                "converged": False,
            },
        }

    def moments(count, lower, upper):
        """Measure positive moments on an independently sampled log grid."""

        q_values = numpy.geomspace(lower, upper, count)
        occupation = sample_distribution(
            q_values, declaration, parameters, allow_zero=True
        )[0]
        integrand = (
            occupation[None, :]
            * q_values[None, :] ** numpy.arange(3, 7)[:, None]
        )
        return numpy.sum(
            (integrand[:, :-1] + integrand[:, 1:])
            * numpy.diff(numpy.log(q_values))[None, :]
            / 2,
            axis=1,
        )

    count, lower, upper = 32, 0.02, 24.0
    attempts = []
    previous = []
    while True:
        coarse = moments(count, lower, upper)
        dense = moments(2 * count + 1, lower, upper)
        expanded = moments(4 * count + 1, lower / 4, upper * 2)
        scale = numpy.maximum(abs(expanded), numpy.finfo(float).tiny)
        density_error = float(numpy.max(abs(dense - coarse) / scale))
        support_error = float(numpy.max(abs(expanded - dense) / scale))
        measured = max(density_error, support_error)
        attempts.append(
            {
                "count": count,
                "q_min": lower,
                "q_max": upper,
                "quadrature_error": density_error,
                "support_error": support_error,
            }
        )
        if numpy.all(expanded > 0) and measured <= tolerance:
            return {
                "count": count,
                "q_min": lower,
                "q_max": upper,
                "planning_evidence": {
                    "method": "independent_distribution_moments_and_tails",
                    "attempts": attempts,
                    "tolerance": tolerance,
                    "converged": True,
                },
            }
        if numpy.any(expanded > 0) and density_error <= tolerance:
            if len(previous) >= 3 and all(
                measured >= error * 0.99 for error in previous[-3:]
            ):
                raise ValueError("Phase-space support refinement stagnated")
            previous.append(measured)
        if density_error > tolerance:
            count = 2 * count + 1
        if (density_error <= tolerance and support_error > tolerance) or (
            not numpy.all(expanded > 0)
        ):
            lower /= 4
            upper *= 2
            count = max(count, 32)
        if not numpy.isfinite(upper) or lower <= 0:
            raise ValueError("Phase-space support is not representable")
