# See LICENSE.md in the repository root for details.

"""Engine package exposing available backends."""

# The suite now ships exclusively with the MCMC backend. Engines remain
# discoverable via ``copernican.engines.engine_*.py`` modules so future
# contributors can add deterministic or stochastic solvers without touching the
# import surface.

from . import engine_mcmc

__all__ = [
    "engine_mcmc",
]
