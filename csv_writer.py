# Copernican Suite CSV Writer
"""Wrapper module exposing CSV writing utilities."""
# DEV NOTE (v1.5e): Thin wrappers calling `output_manager` CSV helpers.

from typing import Any
import output_manager as _om


def save_sne_results_detailed_csv(*args: Any, **kwargs: Any) -> None:
    """Proxy to output_manager.save_sne_results_detailed_csv."""
    return _om.save_sne_results_detailed_csv(*args, **kwargs)


def save_bao_results_csv(*args: Any, **kwargs: Any) -> None:
    """Proxy to output_manager.save_bao_results_csv."""
    return _om.save_bao_results_csv(*args, **kwargs)
