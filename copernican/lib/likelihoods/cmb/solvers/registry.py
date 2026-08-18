"""Registry and selection helpers for pluggable CMB solver backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import CMBSolverProtocol
from ..errors import UnsupportedCapabilityError

CMBSolverFactory = type[CMBSolverProtocol] | CMBSolverProtocol
CMB_SOLVER_REGISTRY: dict[str, CMBSolverFactory] = {}


def register_cmb_solver(
    solver: CMBSolverFactory,
    *,
    replace: bool = False,
) -> CMBSolverFactory:
    """Register a solver object or zero-argument solver class by identity."""

    solver_id = str(getattr(solver, "solver_id", "")).strip()
    solver_label = str(getattr(solver, "solver_label", "")).strip()
    if not solver_id or not solver_label:
        raise ValueError("CMB solvers require solver_id and solver_label")
    if solver_id in CMB_SOLVER_REGISTRY and not replace:
        raise ValueError(f"CMB solver '{solver_id}' is already registered")
    CMB_SOLVER_REGISTRY[solver_id] = solver
    return solver


def _ensure_defaults() -> None:
    """Register the reference CCMBS backend lazily to avoid import cycles."""

    if CMB_SOLVER_REGISTRY:
        return
    from .ccmbs_numpy import CCMBSNumpySolver

    register_cmb_solver(CCMBSNumpySolver())


def available_cmb_solvers() -> tuple[str, ...]:
    """Return registered solver identities in deterministic order."""

    _ensure_defaults()
    return tuple(sorted(CMB_SOLVER_REGISTRY))


def resolve_cmb_solver(
    selection: str | Mapping[str, Any] | CMBSolverProtocol | None = None,
) -> CMBSolverProtocol:
    """Resolve a solver object, manifest selection, or the CCMBS default."""

    _ensure_defaults()
    if selection is not None and hasattr(selection, "evaluate"):
        solver = selection
        if not isinstance(solver, CMBSolverProtocol):
            raise TypeError(
                "Selected CMB solver does not satisfy its protocol"
            )
        return solver
    if isinstance(selection, Mapping):
        solver_id = selection.get("id") or selection.get("solver_id")
    elif selection is None:
        solver_id = None
    else:
        solver_id = selection
    normalized_id = str(solver_id or "ccmbs_numpy").strip()
    registered = CMB_SOLVER_REGISTRY.get(normalized_id)
    if registered is None:
        available = ", ".join(available_cmb_solvers()) or "none"
        raise UnsupportedCapabilityError(
            f"Unknown CMB solver '{normalized_id}'; available: {available}",
            context={"solver_id": normalized_id},
        )
    if isinstance(registered, type):
        return registered()
    return registered


def get_cmb_solver(
    selection: str | Mapping[str, Any] | CMBSolverProtocol | None = None,
) -> CMBSolverProtocol:
    """Alias for :func:`resolve_cmb_solver` used by runtime callers."""

    return resolve_cmb_solver(selection)


def solver_provenance(
    solver: CMBSolverProtocol,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable identity and capability metadata for run provenance."""

    capabilities = solver.capabilities()
    if not isinstance(capabilities, Mapping):
        raise TypeError("CMB solver capabilities must be a mapping")
    if contract is not None:
        capabilities = solver.capabilities(contract)
    return {
        "solver_id": str(solver.solver_id),
        "solver_label": str(solver.solver_label),
        "capabilities": dict(capabilities),
    }


__all__ = [
    "CMB_SOLVER_REGISTRY",
    "available_cmb_solvers",
    "get_cmb_solver",
    "register_cmb_solver",
    "resolve_cmb_solver",
    "solver_provenance",
]
