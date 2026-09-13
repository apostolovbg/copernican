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
from .postprocessing import (
    POST_PROCESSING_SCHEMA_VERSION,
    build_postprocessing_evidence,
)
from .source_graph import DeclaredSourceGraph, compile_declared_source_graph

__all__ = [
    "CMBNumericalPlan",
    "build_cmb_planner_manifest",
    "plan_cmb_numerics",
    "planner_accuracy_controls",
    "describe_declared_execution_schedule",
    "DeclaredSourceGraph",
    "compile_declared_source_graph",
    "POST_PROCESSING_SCHEMA_VERSION",
    "build_postprocessing_evidence",
]
