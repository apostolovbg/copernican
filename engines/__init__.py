# See LICENSE.md in the repository root for details.

"""Engine package exposing available backends.

The package exports available inference engines through a stable surface so
callers can request backends without hard-coding module paths.  Keeping the
registry minimal avoids importing heavy numerical dependencies until an engine
is explicitly selected, which keeps startup quick for CLI users exploring
their options.
"""

# The suite now ships exclusively with the MCMC backend.  Engines remain
# discoverable via ``cosmo_engine_*.py`` modules so future contributors can add
# deterministic or stochastic solvers without touching the import surface,
# because the package-level imports stay stable between releases.

from . import cosmo_engine_mcmc

__all__ = [
    "cosmo_engine_mcmc",
]
