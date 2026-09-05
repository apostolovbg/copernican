"""Likelihood components for Copernican Suite datasets.

The modules in this package expose small, stateful helpers that evaluate
log-likelihoods for individual observational datasets. Each helper implements
:class:`LikelihoodProtocol`, guaranteeing the presence of a :meth:`loglike`
method returning a floating-point log-likelihood together with an
introspectable :pyattr:`state` mapping capturing diagnostic values such as χ²
totals. Samplers combine these helpers to assemble complete likelihoods
without duplicating validation logic. Shared interfaces and the joint
aggregator live in ``copernican.lib.likelihoods.likelihoods`` so dataset
helpers can depend on them without creating circular imports during test
collection.
"""

from __future__ import annotations

from .bao import BAOLike
from .cmb import (
    CMB_SOLVER_REGISTRY,
    CMBError,
    CMBLike,
    CMBResult,
    CMBSolverCapabilities,
    CMBSolverProtocol,
    ConstraintViolationError,
    ContractError,
    ConvergenceError,
    EngineCapabilityError,
    ImplementationError,
    InitialPointError,
    ModelDeclarationError,
    NonFiniteEvolutionError,
    ParameterDomainError,
    UnsupportedCapabilityError,
    available_cmb_solvers,
    compute_cmb_spectrum,
    compute_cmb_spectrum_batch,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_contract,
    get_cmb_solver,
    register_cmb_solver,
    resolve_cmb_solver,
    solver_provenance,
)
from .likelihoods import JointLike, LikelihoodProtocol, LikelihoodState
from .sne import SNeLike

__all__ = [
    "BAOLike",
    "CMBResult",
    "CMBLike",
    "CMBSolverCapabilities",
    "CMBSolverProtocol",
    "CMB_SOLVER_REGISTRY",
    "JointLike",
    "LikelihoodProtocol",
    "LikelihoodState",
    "CMBError",
    "ConstraintViolationError",
    "ContractError",
    "ConvergenceError",
    "EngineCapabilityError",
    "ImplementationError",
    "InitialPointError",
    "NonFiniteEvolutionError",
    "ParameterDomainError",
    "ModelDeclarationError",
    "UnsupportedCapabilityError",
    "SNeLike",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_batch",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_contract",
    "available_cmb_solvers",
    "get_cmb_solver",
    "register_cmb_solver",
    "resolve_cmb_solver",
    "solver_provenance",
]
