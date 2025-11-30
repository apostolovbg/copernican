# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Utilities for saving parameter fit summaries.

The Copernican Suite evaluates cosmological models and often produces
parameter estimates alongside their uncertainties.  This module serialises
those results to both JSON and YAML so that external tools can ingest the
numbers without depending on the full code base.  The writer accepts a
mapping of model names to engine result dictionaries.  Each entry should
contain ``fitted_cosmological_params`` with best-fit values, optional
``parameter_errors`` describing 1σ uncertainties and an optional
``covariance_matrix``.  Starting with version 7.1.0 the summary also embeds
the sampler configuration—production steps, burn-in length, walker count and
worker pool size—so output files mirror the interactive configuration menu.
Version 7.7.1 polishes this metadata by reflowing nested-sampling helpers for
lint compliance, documenting the adjustments and retaining the live-point and
evidence tolerance fields so both backends emit comparable records. NumPy
arrays are converted to plain Python lists to keep the output fully
serialisable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import yaml

from .utils import get_timestamp

def _to_serialisable(obj: Any) -> Any:
    """Return ``obj`` converted to JSON/YAML friendly types.

    NumPy arrays are cast to nested lists while scalars become plain ``float``
    objects.  This keeps the writer lightweight and avoids introducing a
    heavier dependency such as ``pandas`` for simple transformations.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

def save_summary(
    results: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Write parameter summaries for one or more models.

    Parameters
    ----------
    results : mapping
        Mapping of model name to dictionaries containing fit information.  At
        minimum each entry should provide ``fitted_cosmological_params``.
    output_dir : path-like
        Directory where the summary files will be written.
    timestamp : str, optional
        Timestamp appended to the filename.  When ``None`` a current timestamp
        is generated via :func:`copernican_lib.utils.get_timestamp`.

    Returns
    -------
    tuple of :class:`~pathlib.Path`
        Paths to the JSON and YAML files, respectively.  Returning the paths
        simplifies testing and allows callers to log the exact locations.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = timestamp or get_timestamp()

    summary: Dict[str, Dict[str, Any]] = {}
    for model_name, res in results.items():
        params = res.get("fitted_cosmological_params") or {}
        errors = res.get("parameter_errors") or {}
        cov = res.get("covariance_matrix")
        param_names = list(params.keys())
        cov_entry = None
        if cov is not None:
            cov_entry = {
                "param_names": param_names,
                "matrix": _to_serialisable(cov),
            }
        sampling_entry = None
        sampling_fields = (
            ("production_steps", "production_steps"),
            ("burn_in_steps", "burn_in_steps"),
            ("n_walkers", "n_walkers"),
            ("pool_workers", "pool_workers"),
            ("n_live_points", "n_live_points"),
            ("max_iterations", "max_iterations"),
            ("evidence_tolerance", "evidence_tolerance"),
            (
                "enlargement_fraction",
                "enlargement_fraction",
            ),
            ("iterations_completed", "iterations_completed"),
        )
        for out_key, res_key in sampling_fields:
            value = res.get(res_key)
            if value is None:
                continue
            if sampling_entry is None:
                sampling_entry = {}
            sampling_entry[out_key] = _to_serialisable(value)
        summary[model_name] = {
            "parameters": {k: _to_serialisable(v) for k, v in params.items()},
            "errors_1sigma": (
                {k: _to_serialisable(v) for k, v in errors.items()}
                if errors
                else None
            ),
            "covariance_matrix": cov_entry,
            "sampling": sampling_entry,
        }

    base_name = f"parameter-summary_{ts}"
    json_path = out_path / f"{base_name}.json"
    yaml_path = out_path / f"{base_name}.yml"
    with open(json_path, "w", encoding="utf-8") as jh:
        json.dump(summary, jh, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as yh:
        yaml.safe_dump(summary, yh, sort_keys=False)
    return json_path, yaml_path

__all__ = ["save_summary"]
