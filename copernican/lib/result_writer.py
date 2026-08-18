# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Utilities for saving parameter fit summaries.

The Copernican Suite evaluates cosmological models and often produces
parameter estimates alongside their uncertainties. This module serialises
the control and test results to both JSON and YAML so external tools can
ingest the numbers without depending on the full code base. Each role entry
contains its selected model identity, ``fitted_model_params`` with
best-fit values, optional
``parameter_errors`` describing 1σ uncertainties and an optional
``covariance_matrix``.  Starting with version 7.1.0 the summary also embeds
the sampler configuration—production steps, burn-in length, walker count and
worker pool size—so output files mirror the interactive configuration menu.
NumPy arrays are converted to plain Python lists to keep the output fully
serialisable. Role keys preserve both records when the control and test model
names are identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy
import yaml

from .model_selection import ComparisonRequest, comparison_slug
from .utils import get_timestamp


def _to_serialisable(payload: Any) -> Any:
    """Return ``payload`` converted to JSON/YAML friendly types.

    NumPy arrays are cast to nested lists while scalars become plain ``float``
    objects.  This keeps the writer lightweight and avoids introducing a
    heavier dependency such as ``pandas`` for simple transformations.
    """

    if isinstance(payload, numpy.ndarray):
        return payload.tolist()
    if isinstance(payload, (numpy.floating, numpy.integer)):
        return payload.item()
    return payload


def save_summary(
    control_results: Mapping[str, Any],
    test_results: Mapping[str, Any],
    output_dir: str | Path,
    *,
    comparison: ComparisonRequest,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Write role-preserving parameter summaries for one comparison.

    Parameters
    ----------
    control_results, test_results : mapping
        Fit information for the selected control and test models. At minimum,
        each entry should provide ``fitted_model_params``.
    output_dir : path-like
        Directory where the summary files will be written.
    timestamp : str, optional
        Timestamp appended to the filename.  When ``None`` a current timestamp
        is generated via :func:`copernican.lib.utils.get_timestamp`.

    Returns
    -------
    tuple of :class:`~pathlib.Path`
        Paths to the JSON and YAML files, respectively.  Returning the paths
        simplifies testing and allows callers to log the exact locations.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_timestamp = timestamp or get_timestamp()

    summary: Dict[str, Dict[str, Any]] = {}
    role_results = (
        ("control", comparison.control_model.name, control_results),
        ("test", comparison.test_model.name, test_results),
    )
    for role, model_name, res in role_results:
        params = res.get("fitted_model_params") or {}
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
            sampling_value = res.get(res_key)
            if sampling_value is None:
                continue
            if sampling_entry is None:
                sampling_entry = {}
            sampling_entry[out_key] = _to_serialisable(sampling_value)
        ensemble_performance = res.get("ensemble_performance")
        if ensemble_performance is not None:
            if sampling_entry is None:
                sampling_entry = {}
            sampling_entry["ensemble_performance"] = _to_serialisable(
                ensemble_performance
            )
        cmb_solver = res.get("cmb_solver")
        if cmb_solver is not None:
            if sampling_entry is None:
                sampling_entry = {}
            sampling_entry["cmb_solver"] = _to_serialisable(cmb_solver)
        summary[role] = {
            "model": model_name,
            "parameters": {k: _to_serialisable(v) for k, v in params.items()},
            "errors_1sigma": (
                {k: _to_serialisable(v) for k, v in errors.items()}
                if errors
                else None
            ),
            "covariance_matrix": cov_entry,
            "sampling": sampling_entry,
        }

    pair_token = f"_{comparison_slug(comparison)}"
    base_name = f"parameter-summary{pair_token}_{summary_timestamp}"
    json_path = out_path / f"{base_name}.json"
    yaml_path = out_path / f"{base_name}.yml"
    with open(json_path, "w", encoding="utf-8") as json_file_handle:
        json.dump(summary, json_file_handle, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as yaml_file_handle:
        yaml.safe_dump(summary, yaml_file_handle, sort_keys=False)
    return json_path, yaml_path


__all__ = ["save_summary"]
