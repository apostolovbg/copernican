# Copernican Suite Plotter
"""Wrapper module exposing plotting utilities."""
# DEV NOTE (v1.5e): Thin wrappers calling `output_manager` functions for reuse across engines.

from typing import Any
import output_manager as _om


def format_model_summary_text(*args: Any, **kwargs: Any) -> str:
    """Proxy to output_manager.format_model_summary_text."""
    return _om.format_model_summary_text(*args, **kwargs)


def plot_hubble_diagram(*args: Any, **kwargs: Any) -> None:
    """Proxy to output_manager.plot_hubble_diagram."""
    return _om.plot_hubble_diagram(*args, **kwargs)


def plot_bao_observables(*args: Any, **kwargs: Any) -> None:
    """Proxy to output_manager.plot_bao_observables."""
    return _om.plot_bao_observables(*args, **kwargs)
