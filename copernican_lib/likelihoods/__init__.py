"""Likelihood components for Copernican Suite datasets.

The modules in this package expose small, stateful helpers that evaluate
log-likelihoods for individual observational datasets.  Each helper implements
:class:`LikelihoodProtocol`, guaranteeing the presence of a :meth:`loglike`
method returning a floating-point log-likelihood together with an
introspectable :pyattr:`state` mapping capturing diagnostic values such as χ²
totals.  Engines combine these helpers to assemble complete likelihoods
without duplicating validation logic.  Shared interfaces live in
``copernican_lib.likelihoods._protocol`` so dataset helpers can depend on them
without creating circular imports during test collection, keeping module
imports predictable even when optional science dependencies are missing.
"""

from __future__ import annotations

from ._protocol import LikelihoodProtocol, LikelihoodState
from .bao import BAOLike
from .cmb import (
    CMBLike,
    compute_camb_background_observables,
    compute_cmb_spectrum,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_dict,
    describe_camb_configuration,
)
from .joint import JointLike
from .sne import SNeLike

__all__ = [
    "BAOLike",
    "CMBLike",
    "JointLike",
    "LikelihoodProtocol",
    "LikelihoodState",
    "SNeLike",
    "compute_camb_background_observables",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_dict",
    "describe_camb_configuration",
]
