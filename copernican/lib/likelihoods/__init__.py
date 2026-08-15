"""Likelihood components for Copernican Suite datasets.

The modules in this package expose small, stateful helpers that evaluate
log-likelihoods for individual observational datasets. Each helper implements
:class:`LikelihoodProtocol`, guaranteeing the presence of a :meth:`loglike`
method returning a floating-point log-likelihood together with an
introspectable :pyattr:`state` mapping capturing diagnostic values such as χ²
totals. Engines combine these helpers to assemble complete likelihoods
without duplicating validation logic. Shared interfaces and the joint
aggregator live in ``copernican.lib.likelihoods.likelihoods`` so dataset
helpers can depend on them without creating circular imports during test
collection.
"""

from __future__ import annotations

from .bao import BAOLike
from .cmb import (
    CMBLike,
    NativeCMBError,
    NativeConstraintViolationError,
    NativeContractError,
    NativeConvergenceError,
    NativeImplementationError,
    NativeInitialPointError,
    NativeNonFiniteEvolutionError,
    NativeParameterDomainError,
    NativePerformanceBudgetError,
    NativeUnsupportedCapabilityError,
    compute_cmb_spectrum,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_contract,
)
from .likelihoods import JointLike, LikelihoodProtocol, LikelihoodState
from .sne import SNeLike

__all__ = [
    "BAOLike",
    "CMBLike",
    "JointLike",
    "LikelihoodProtocol",
    "LikelihoodState",
    "NativeCMBError",
    "NativeConstraintViolationError",
    "NativeContractError",
    "NativeConvergenceError",
    "NativeImplementationError",
    "NativeInitialPointError",
    "NativeNonFiniteEvolutionError",
    "NativeParameterDomainError",
    "NativePerformanceBudgetError",
    "NativeUnsupportedCapabilityError",
    "SNeLike",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_contract",
]
