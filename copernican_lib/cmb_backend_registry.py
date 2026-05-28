"""CMB backend capability and solver registry.

The registry keeps backend support explicit so perturbation contracts can be
validated against declared capabilities before any scientific execution runs.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

_CMB_BACKEND_CAPABILITIES_BY_NAME: dict[str, dict[str, bool]] = {
    "camb": {
        "scalar_param_map": True,
        "grids_values_calls": True,
        "standard_perturbations": True,
        "native_nonstandard_perturbations": False,
    }
}

CMB_BACKEND_CAPABILITIES = _CMB_BACKEND_CAPABILITIES_BY_NAME

REGISTERED_NATIVE_PERTURBATION_SOLVERS: dict[
    str,
    dict[str, Callable[..., Any]],
] = {
    "camb": {},
}


def get_backend_capabilities(backend: str) -> Mapping[str, bool]:
    """Return the declared capability mapping for ``backend``."""

    return CMB_BACKEND_CAPABILITIES.get(backend, {})


def backend_supports_standard_perturbations(backend: str) -> bool:
    """Return ``True`` when ``backend`` supports standard perturbations."""

    return bool(
        get_backend_capabilities(backend).get("standard_perturbations")
    )


def backend_supports_native_nonstandard_perturbations(backend: str) -> bool:
    """Return ``True`` when ``backend`` supports native non-standard modes."""

    return bool(
        get_backend_capabilities(backend).get(
            "native_nonstandard_perturbations"
        )
    )


def get_registered_native_solver(
    backend: str,
    solver: str,
) -> Callable[..., Any] | None:
    """Return the registered native solver for ``backend`` if available."""

    backend_solvers = REGISTERED_NATIVE_PERTURBATION_SOLVERS.get(backend, {})
    return backend_solvers.get(solver)


def native_solver_is_registered(backend: str, solver: str) -> bool:
    """Return ``True`` when ``solver`` is registered for ``backend``."""

    return get_registered_native_solver(backend, solver) is not None


def validate_native_perturbation_execution(
    *,
    model_name: str,
    backend: str,
    standard: bool,
    solver: str | None,
    implemented: bool | None,
) -> None:
    """Raise ``ValueError`` when non-standard execution is unsupported."""

    if standard:
        return

    if not backend_supports_native_nonstandard_perturbations(backend):
        raise ValueError(
            "Model "
            f"'{model_name}' declares non-standard perturbations for "
            f"backend '{backend}' (standard={standard}, solver={solver!r}), "
            "but the backend capability registry does not support native "
            "non-standard perturbations. A native backend implementation is "
            "required."
        )

    if implemented is not True:
        raise ValueError(
            "Model "
            f"'{model_name}' declares non-standard perturbations for "
            f"backend '{backend}' (standard={standard}, solver={solver!r}), "
            "but the backend mapping does not mark a native implementation "
            "as available. A native backend implementation is required."
        )

    if not solver:
        raise ValueError(
            "Model "
            f"'{model_name}' declares non-standard perturbations for "
            f"backend '{backend}' (standard={standard}), but no native "
            "solver name was declared."
        )

    if not native_solver_is_registered(backend, solver):
        raise ValueError(
            "Model "
            f"'{model_name}' declares non-standard perturbations for "
            f"backend '{backend}' (standard={standard}, solver={solver!r}), "
            "but the named native solver is not registered. A native "
            "backend implementation is required."
        )


__all__ = [
    "CMB_BACKEND_CAPABILITIES",
    "REGISTERED_NATIVE_PERTURBATION_SOLVERS",
    "backend_supports_native_nonstandard_perturbations",
    "backend_supports_standard_perturbations",
    "get_backend_capabilities",
    "get_registered_native_solver",
    "native_solver_is_registered",
    "validate_native_perturbation_execution",
]
