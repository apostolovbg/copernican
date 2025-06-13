# cosmo_engine_new1.py
"""
DEV NOTE: Initial implementation of the plugin-based engine for CosmoDSL.
This engine provides helper utilities for engine discovery and a minimal
numerical API. The true cosmology calculations are placeholders but
illustrate how CosmoDSL equations could be compiled with Numba.
"""

import os
import importlib.util
from numba import jit
import numpy as np

# --- Engine Discovery -------------------------------------------------------

def discover_engines(engine_dir):
    """Return paths to all available cosmo_engine_*.py files."""
    # DEV NOTE (v2.0): Previous versions excluded this file from discovery,
    # which left only legacy engines selectable in the CLI. The new CLI relies
    # on the modern engine defined here, so we include every matching file.
    engines = []
    for fname in os.listdir(engine_dir):
        if fname.startswith("cosmo_engine_") and fname.endswith(".py"):
            engines.append(os.path.join(engine_dir, fname))
    return sorted(engines)

# --- Equation Compilation ---------------------------------------------------

def _compile_equation(expr):
    """Compile a single expression string into a numba JIT function."""
    # We create a function that evaluates the expression with numpy available.
    @jit(nopython=True)
    def func(**kwargs):
        return eval(expr, {"np": np}, kwargs)
    return func

def compile_equations(model_ast):
    """Compile all equations from the model AST into callable functions."""
    compiled = {}
    eq_blocks = model_ast.get("equations", "")
    if not eq_blocks:
        return compiled
    lines = [line.strip() for line in eq_blocks.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        compiled[f"eq_{idx}"] = _compile_equation(line)
    return compiled

# --- Engine API -------------------------------------------------------------

class Engine:
    """Minimal engine that executes compiled equation functions."""

    def __init__(self, model_ast):
        self.functions = compile_equations(model_ast)

    def run(self, model, data_files, cov_matrix=None):
        """Run the model. Returns a dictionary of evaluated equations."""
        results = {}
        for name, func in self.functions.items():
            # Example evaluation using model parameters only
            results[name] = func(**model.get("parameters", {}))
        return results
