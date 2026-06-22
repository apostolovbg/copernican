"""Run manifest generator for Copernican.

The manifest records critical information required to reproduce a run. It
captures the Copernican version, model and engine details, parameter
priors, dataset hashes provided by the data loaders and the Git state.  Each
run directory stores the resulting YAML file so that analyses can be traced
back unambiguously. CMB entries now include both the background adapter
summary and the perturbation-contract metadata declared on each model.
"""

from __future__ import annotations

import os
import subprocess  # nosec
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from copernican import version as version_module

from . import utils
from .likelihoods import cmb as cmb_module
from .likelihoods.cmb import copernican_cmb_solver as native_cmb_module


def _copernican_version() -> str:
    """Return the Copernican version while tolerating missing helpers.

    Some macOS installations reported ``ImportError`` when
    ``copernican.version.get_version`` was unavailable even though the
    module itself existed. Importing the attribute lazily keeps the
    ``run_manifest`` module importable in that scenario so
    ``python -m copernican`` can still launch and emit a manifest.
    Falling back to ``"0+unknown"``
    mirrors the final stage inside :func:`copernican.version.get_version`
    and ensures the manifest always carries a deterministic placeholder.
    """

    getter = getattr(version_module, "get_version", None)
    if callable(getter):
        return getter()
    return "0+unknown"


def _git_info() -> dict:
    """Return the current commit hash and dirty state.

    The function falls back to ``"unknown"`` if Git is unavailable.  A
    ``dirty`` flag indicates whether uncommitted changes were present
    during execution.
    """

    try:
        commit = (
            subprocess.check_output(  # nosec
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    try:
        subprocess.check_output(  # nosec
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
        dirty = False
    except subprocess.CalledProcessError:
        dirty = True
    except OSError:
        dirty = True
    return {"commit": commit, "dirty": dirty}


def _camb_info(models: Iterable[tuple[object, str]]) -> dict | None:
    """Return CAMB metadata for models that supply a CMB mapping."""

    camb_models: list[object] = []
    for plugin, _ in models:
        if getattr(plugin, "valid_for_cmb", True) is False:
            continue
        contract = getattr(plugin, "CMB_CONTRACT", {}) or {}
        if contract:
            camb_models.append(plugin)
    if not camb_models:
        return None

    try:  # pragma: no cover - graceful when CAMB absent in minimal envs
        import camb  # type: ignore

        version = getattr(camb, "__version__", "unknown")
    except ImportError:
        version = "unavailable"

    configuration = cmb_module.describe_camb_configuration()
    models_meta: list[dict[str, Any]] = []
    for plugin in camb_models:
        contract = getattr(plugin, "CMB_CONTRACT", {}) or {}
        perturbations = getattr(plugin, "CMB_PERTURBATION_CONTRACT", {}) or {}
        perturbation_data = getattr(plugin, "CMB_PERTURBATION_DATA", None)
        dependency_summary = getattr(
            perturbation_data, "dependency_graph_summary", None
        )
        manifest_summary = getattr(perturbation_data, "manifest_summary", {})
        manifest_summary_data = (
            manifest_summary.to_dict()
            if hasattr(manifest_summary, "to_dict")
            else (
                dict(manifest_summary)
                if isinstance(manifest_summary, dict)
                else {}
            )
        )
        backend_mapping_data = getattr(
            perturbation_data, "backend_mapping", {}
        )
        backend_mapping_camb = (
            backend_mapping_data.get("camb", {})
            if hasattr(backend_mapping_data, "get")
            else {}
        )
        param_map = contract.get("param_map", {}) or {}
        grids = contract.get("grids", {}) or {}
        values = contract.get("values", {}) or {}
        calls = contract.get("calls", []) or []
        background_manifest_summary = (
            native_cmb_module._summarize_declared_background_manifest_summary(
                contract
            )
            if isinstance(contract, dict) and contract.get("background")
            else {}
        )
        execution_route = manifest_summary_data.get("execution_route", {})
        perturbation_sources = getattr(perturbation_data, "sources", {}) or {}
        numerical_settings = contract.get("numerical", {}) or {}
        numerical_settings = (
            dict(numerical_settings)
            if isinstance(numerical_settings, dict)
            else numerical_settings
        )
        grid_meta = {
            str(grid_name): {
                "lower": grid_def.get("lower"),
                "upper": grid_def.get("upper"),
                "points": grid_def.get("points"),
                "spacing": grid_def.get("spacing"),
            }
            for grid_name, grid_def in grids.items()
        }
        models_meta.append(
            {
                "model": getattr(plugin, "MODEL_NAME", "unknown"),
                "backend": contract.get("backend", "unknown"),
                "param_map_keys": sorted(str(key) for key in param_map),
                "call_methods": [
                    str(call.get("method"))
                    for call in calls
                    if call.get("method") is not None
                ],
                "grids": grid_meta,
                "value_names": [str(key) for key in values],
                "perturbation_contract_version": getattr(
                    perturbation_data, "contract_version", None
                ),
                "perturbation_standard": getattr(
                    perturbation_data,
                    "standard",
                    perturbations.get("standard"),
                ),
                "perturbation_gauge": getattr(
                    perturbation_data, "gauge", perturbations.get("gauge")
                ),
                "perturbation_variable_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "variables", {}) or {}
                    )
                ),
                "perturbation_derived_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "derived", {}) or {}
                    )
                ),
                "perturbation_equation_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "equations", {}) or {}
                    )
                ),
                "perturbation_constraint_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "constraints", {}) or {}
                    )
                ),
                "perturbation_closure_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "closures", {}) or {}
                    )
                ),
                "perturbation_source_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "sources", {}) or {}
                    )
                ),
                "perturbation_observable_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "observables", {}) or {}
                    )
                ),
                "perturbation_initial_condition_names": sorted(
                    str(key)
                    for key in (
                        getattr(
                            perturbation_data,
                            "initial_conditions",
                            {},
                        )
                        or {}
                    )
                ),
                "perturbation_boundary_condition_names": sorted(
                    str(key)
                    for key in (
                        getattr(
                            perturbation_data,
                            "boundary_conditions",
                            {},
                        )
                        or {}
                    )
                ),
                "perturbation_equation_count": len(
                    getattr(perturbation_data, "equations", {}) or {}
                ),
                "perturbation_constraint_count": len(
                    getattr(perturbation_data, "constraints", {}) or {}
                ),
                "perturbation_closure_count": len(
                    getattr(perturbation_data, "closures", {}) or {}
                ),
                "perturbation_source_count": len(perturbation_sources),
                "perturbation_observable_count": len(
                    getattr(perturbation_data, "observables", {}) or {}
                ),
                "perturbation_initial_condition_count": len(
                    getattr(perturbation_data, "initial_conditions", {}) or {}
                ),
                "perturbation_boundary_condition_count": len(
                    getattr(perturbation_data, "boundary_conditions", {}) or {}
                ),
                "perturbation_numerical_settings": numerical_settings,
                "perturbation_independent_variables_used": sorted(
                    str(key)
                    for key in getattr(
                        dependency_summary, "independent_variables_used", ()
                    )
                ),
                "perturbation_model_parameters_used": sorted(
                    str(key)
                    for key in getattr(
                        dependency_summary, "model_parameters_used", ()
                    )
                ),
                "perturbation_background_references_used": sorted(
                    str(key)
                    for key in getattr(
                        dependency_summary, "background_references_used", ()
                    )
                ),
                "perturbation_evaluation_order": [
                    str(key)
                    for key in getattr(
                        dependency_summary, "evaluation_order", ()
                    )
                ],
                "perturbation_backend": getattr(
                    perturbation_data, "backend", contract.get("backend")
                ),
                "perturbation_backend_implemented": getattr(
                    backend_mapping_camb, "implemented", None
                ),
                "perturbation_backend_uses_standard_perturbations": getattr(
                    backend_mapping_camb,
                    "uses_standard_perturbations",
                    None,
                ),
                "perturbation_backend_native_solver_required": getattr(
                    backend_mapping_camb, "native_solver_required", None
                ),
                "perturbation_backend_mapping_summary": {
                    str(backend_name): (
                        {
                            "keys": sorted(
                                str(key) for key in backend_mapping.keys()
                            ),
                            "implemented": backend_mapping.get("implemented"),
                            "native_solver_required": backend_mapping.get(
                                "native_solver_required"
                            ),
                            "uses_standard_perturbations": backend_mapping.get(
                                "uses_standard_perturbations"
                            ),
                        }
                        if isinstance(backend_mapping, dict)
                        else backend_mapping
                    )
                    for backend_name, backend_mapping in (
                        perturbations.get("backend_mapping", {}) or {}
                    ).items()
                },
                "custom_cmb_execution_route": {
                    str(key): value for key, value in execution_route.items()
                },
                "custom_cmb_equation_count": len(
                    getattr(perturbation_data, "equations", {}) or {}
                ),
                "custom_cmb_constraint_count": len(
                    getattr(perturbation_data, "constraints", {}) or {}
                ),
                "custom_cmb_closure_count": len(
                    getattr(perturbation_data, "closures", {}) or {}
                ),
                "custom_cmb_source_count": len(perturbation_sources),
                "custom_cmb_observable_count": len(
                    getattr(perturbation_data, "observables", {}) or {}
                ),
                "custom_cmb_observable_names": [
                    str(key)
                    for key in manifest_summary_data.get(
                        "observable_names", ()
                    )
                ],
                "custom_cmb_initial_condition_count": len(
                    getattr(perturbation_data, "initial_conditions", {}) or {}
                ),
                "custom_cmb_boundary_condition_count": len(
                    getattr(perturbation_data, "boundary_conditions", {}) or {}
                ),
                "custom_cmb_numerical_settings": numerical_settings,
                "custom_cmb_graph_manifest_summary": manifest_summary_data,
                "custom_cmb_background_manifest_summary": (
                    background_manifest_summary
                ),
                "custom_cmb_runtime_manifest_summary": {
                    "execution_route": {
                        str(key): value
                        for key, value in execution_route.items()
                    },
                    "numerical_settings": numerical_settings,
                    "recombination_runtime": (
                        background_manifest_summary.get(
                            "recombination_runtime",
                            {},
                        )
                    ),
                    "reionization_calibration": (
                        background_manifest_summary.get(
                            "reionization_calibration",
                            {},
                        )
                    ),
                },
                "custom_cmb_reference_validation_status": (
                    configuration.get("reference_validation_status")
                    if isinstance(configuration, dict)
                    else None
                ),
                "custom_cmb_validation_status": (
                    configuration.get("reference_validation_status")
                    if isinstance(configuration, dict)
                    else None
                ),
            }
        )

    return {
        "version": version,
        "configuration": configuration,
        "models": models_meta,
    }


def build_manifest(
    models: Iterable[tuple[object, str]],
    engine_module: object,
    datasets: Iterable[Dict[str, Any]],
    *,
    state: str = "pending",
    output_policy: str = "unprepared",
    configuration: Optional[dict[str, Any]] = None,
) -> dict:
    """Collect manifest information for the current run.

    Parameters
    ----------
    models:
        Iterable of ``(plugin, version)`` pairs where ``plugin`` exposes
        ``MODEL_NAME``, ``MODEL_FILENAME``, ``PARAMETER_NAMES`` and
        ``PARAMETER_PRIORS`` attributes.
    engine_module:
        Selected engine module object.  ``ENGINE_VERSION`` is queried when
        available.
    datasets:
        Iterable of dictionaries describing each dataset.  Expected keys are
        ``id``, ``name``, ``version``, ``path``, ``hashes`` and
        ``independence``.  The manifest builder ignores missing keys so
        callers may provide partial information when necessary.
    state:
        Lifecycle status recorded at the time the manifest is built.  The
        default ``"pending"`` mirrors the GUI's start confirmation stage.
    output_policy:
        Textual description of whether output directories and logs have been
        created.  ``"unprepared"`` signals that confirmation is still
        required.
    configuration:
        Optional configuration snapshot that captures the human-facing
        selections that drove the manifest.  When omitted the snapshot is
        derived from the collected model and dataset metadata.
    """

    manifest = {
        "copernican": {"version": _copernican_version()},
        "models": [],
        "engine": {
            "name": getattr(engine_module, "__name__", "unknown"),
            "version": getattr(engine_module, "ENGINE_VERSION", "unknown"),
        },
        "seed": utils.get_random_seed(),
        "datasets": {},
        "git": _git_info(),
        "status": {"state": state, "outputs": output_policy},
        "selection": {"models": [], "engine": {}, "datasets": []},
    }

    for plugin, version in models:
        priors = {
            name: prior
            for name, prior in zip(
                getattr(plugin, "PARAMETER_NAMES", []),
                getattr(plugin, "PARAMETER_PRIORS", []),
            )
            if prior
        }
        model_entry = {
            "name": getattr(plugin, "MODEL_NAME", "unknown"),
            "version": version,
            "filename": getattr(plugin, "MODEL_FILENAME", ""),
            "priors": priors,
        }
        manifest["models"].append(model_entry)
        manifest["selection"]["models"].append(model_entry["name"])

    for dataset in datasets:
        dataset_id = dataset.get("id")
        if not dataset_id:
            continue
        independence = dataset.get("independence", [])
        if isinstance(independence, str):
            independence = [independence]
        manifest["datasets"][dataset_id] = {
            "name": dataset.get("name", dataset_id),
            "version": dataset.get("version", "unknown"),
            "path": dataset.get("path", ""),
            "hashes": dataset.get("hashes", {}),
            "independence": independence,
            "condition_number": dataset.get("condition_number"),
            "type": dataset.get("type", "unknown"),
        }
        manifest["selection"]["datasets"].append(dataset_id)

    manifest["selection"]["engine"] = manifest["engine"].copy()

    if configuration:
        manifest["configuration"] = configuration
    else:
        manifest["configuration"] = {
            "notes": "Derived from GUI selections; update when importing.",
            "engine": manifest["selection"]["engine"],
            "models": manifest["selection"]["models"],
            "datasets": manifest["selection"]["datasets"],
        }

    camb_details = _camb_info(models)
    if camb_details is not None:
        manifest["camb"] = camb_details

    return manifest


def save_manifest(
    manifest: dict,
    output_dir: str,
    *,
    target_path: str | os.PathLike | None = None,
) -> str:
    """Persist ``manifest`` to a deterministic path.

    When ``target_path`` is provided the manifest is written to that exact
    location; otherwise it is saved as ``run_manifest_<timestamp>.yml`` under
    ``output_dir``.  The full path to the saved file is returned so callers can
    log reproducible locations for CI archives.
    """

    utils.ensure_dir_exists(output_dir)
    if target_path is not None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = utils.get_timestamp()
        target = Path(output_dir) / f"run_manifest_{timestamp}.yml"
    with open(target, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(manifest, file_handle, sort_keys=False)
    return str(target)


def load_manifest(path: str) -> dict:
    """Load a manifest from disk for reuse in a new run."""

    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def annotate_outcome(
    manifest: dict,
    *,
    state: str,
    outputs: Optional[str] = None,
    reason: str = "",
) -> dict:
    """Return ``manifest`` with an updated status block.

    The helper avoids in-place mutation surprises during GUI flows and ensures
    every saved manifest records whether a run completed, was cancelled or was
    aborted via a hard stop.  ``outputs`` captures the user's retention choice
    (for example ``"kept"``, ``"deleted"`` or ``"archived"``) so post-run
    housekeeping decisions remain visible in downstream analyses.
    """

    updated = dict(manifest)
    status = dict(updated.get("status", {}))
    status["state"] = state
    if outputs is not None:
        status["outputs"] = outputs
    status["reason"] = reason
    updated["status"] = status
    return updated


__all__ = [
    "build_manifest",
    "save_manifest",
    "load_manifest",
    "annotate_outcome",
]
