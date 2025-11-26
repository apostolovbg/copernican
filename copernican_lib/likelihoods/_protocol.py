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
Keeping the protocols isolated also means engines can perform static checks
without pulling heavier likelihood implementations into memory prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol, Sequence


class LikelihoodProtocol(Protocol):
    """Runtime contract implemented by all likelihood helpers.

    Engines rely on this minimal protocol to swap in different dataset
    likelihoods without caring about their concrete implementations.  Stating
    the interface explicitly prevents regressions when individual likelihoods
    gain new diagnostics because engines continue to depend only on the
    required surface.
    """

    enabled: bool

    def loglike(self, params: Sequence[float]) -> float:
        """Return the natural log likelihood for ``params``.

        Engines call this method rather than rolling their own loops so every
        dataset helper can maintain its own internal caches and diagnostics.
        """

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostic information captured during the last call.

        Providing structured diagnostics alongside the likelihood value helps
        orchestration code emit richer progress logs without guessing at
        helper internals.
        """


@dataclass(slots=True)
class LikelihoodState:
    """Mutable container storing likelihood diagnostics.

    Storing state in a dedicated dataclass keeps diagnostics grouped and makes
    it obvious which attributes callers may rely on when emitting logs or
    debugging convergence issues.
    """

    chi2: float = float("inf")
    loglike: float = float("-inf")
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> Mapping[str, Any]:
        """Return an immutable representation of the stored state.

        Converting to plain dictionaries ensures JSON logging and drift checks
        can capture the diagnostics without tripping over mutable mapping
        implementations.
        """

        return {
            "chi2": self.chi2,
            "loglike": self.loglike,
            "metadata": dict(self.metadata),
        }


__all__ = ["LikelihoodProtocol", "LikelihoodState"]
