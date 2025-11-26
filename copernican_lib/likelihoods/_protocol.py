# Rationale: Protocols live in this lightweight module because keeping the
# interface importable without heavy dependencies prevents circular imports and
# keeps early CLI actions fast.
"""Shared protocol definitions for likelihood helpers.

The :mod:`copernican_lib.likelihoods` package exposes multiple dataset
likelihood helpers that all share a common interface.  To avoid circular
imports each helper depends on the protocol and state container defined here
instead of importing them via :mod:`copernican_lib.likelihoods.__init__`.  The
module lives on its own so lightweight protocol types remain importable even
when optional scientific dependencies such as CAMB are absent, keeping CLI
tools responsive during early startup and in constrained test environments.
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


__all__ = ["LikelihoodProtocol", "LikelihoodState"]
