"""Numerical runtime components shared by selectable CMB solvers.

The runtime owns reusable grids, evolution assets, projections, lensing,
convergence checks, performance accounting, and bounded process-local caches.
"""

from .evolution import describe_declared_execution_schedule
from .planner import (
    CMBNumericalPlan,
    build_cmb_planner_manifest,
    plan_cmb_numerics,
    planner_accuracy_controls,
)

__all__ = [
    "CMBNumericalPlan",
    "build_cmb_planner_manifest",
    "plan_cmb_numerics",
    "planner_accuracy_controls",
    "describe_declared_execution_schedule",
]
