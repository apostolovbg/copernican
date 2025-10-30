"""Likelihood components for Copernican Suite datasets.

**Last Updated:** 2025-10-30

The modules in this package expose small, stateful helpers that evaluate
log-likelihoods for individual observational datasets.  Each helper implements
:class:`LikelihoodProtocol` which guarantees the presence of a
:meth:`loglike` method returning a floating-point log-likelihood together with
an introspectable :pyattr:`state` mapping capturing diagnostic values such as
χ² totals.  Engines combine these helpers to assemble complete likelihoods
without duplicating validation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol, Sequence


class LikelihoodProtocol(Protocol):
    """Runtime contract implemented by all likelihood helpers."""

    enabled: bool

    def loglike(self, params: Sequence[float]) -> float:
        """Return the natural logarithm of the likelihood for ``params``."""

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostic information captured during the last call."""


@dataclass(slots=True)
class LikelihoodState:
    """Mutable container storing likelihood diagnostics."""

    chi2: float = float("inf")
    loglike: float = float("-inf")
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> Mapping[str, Any]:
        """Return an immutable representation of the stored state."""

        return {
            "chi2": self.chi2,
            "loglike": self.loglike,
            "metadata": dict(self.metadata),
        }


from .bao import BAOLike
from .cmb import (
    CMBLike,
    compute_cmb_spectrum,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_dict,
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
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_dict",
]
