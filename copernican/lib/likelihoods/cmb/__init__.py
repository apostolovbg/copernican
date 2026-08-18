"""Public CMB likelihood facade and selectable solver entrypoints.

The package root keeps likelihood callers independent of solver subpackages,
while exposing the typed result, failure, capability, and registry contracts
needed by the NumPy reference backend and future device-specific solvers.
"""

from .cmb import (
    CMBLike,
    compute_cmb_spectrum,
    compute_cmb_spectrum_batch,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_contract,
)
from .contracts import (  # noqa: F401
    CMBResult,
    CMBSolverCapabilities,
    CMBSolverProtocol,
)
from .errors import (
    CMBError,
    ConstraintViolationError,
    ContractError,
    ConvergenceError,
    ImplementationError,
    InitialPointError,
    NonFiniteEvolutionError,
    ParameterDomainError,
    PerformanceBudgetError,
    UnsupportedCapabilityError,
)
from .results import CMBBatchResult
from .runtime import cache, projection  # noqa: F401
from .solvers.registry import CMB_SOLVER_REGISTRY  # noqa: F401
from .solvers.registry import available_cmb_solvers  # noqa: F401
from .solvers.registry import get_cmb_solver  # noqa: F401
from .solvers.registry import register_cmb_solver  # noqa: F401
from .solvers.registry import resolve_cmb_solver  # noqa: F401
from .solvers.registry import solver_provenance  # noqa: F401

__all__ = [
    "CMBLike",
    "CMBBatchResult",
    "CMBError",
    "ConstraintViolationError",
    "ContractError",
    "ConvergenceError",
    "ImplementationError",
    "InitialPointError",
    "NonFiniteEvolutionError",
    "ParameterDomainError",
    "PerformanceBudgetError",
    "UnsupportedCapabilityError",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_batch",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_contract",
]
