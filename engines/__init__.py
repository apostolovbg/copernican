# See LICENSE.md in the repository root for details.

"""Engine package exposing available backends."""

# The suite now ships exclusively with the MCMC backend.  Engines remain
# discoverable via ``cosmo_engine_*.py`` modules so future contributors can add
# deterministic or stochastic solvers without touching the import surface.

from . import cosmo_engine_mcmc

__all__ = [
    "cosmo_engine_mcmc",
]
