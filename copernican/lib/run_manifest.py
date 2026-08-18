"""Run manifest generator for Copernican.

The manifest records critical information required to reproduce a run. It
captures the Copernican version, model and sampler details, parameter
priors, dataset hashes provided by the data loaders and the Git state.  Each
run directory stores the resulting YAML file so that analyses can be traced
back unambiguously. CMB entries include the native background and compiled
perturbation-contract metadata declared on each model.
"""

from __future__ import annotations

import os
import subprocess  # nosec
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from copernican import version as version_module

from . import utils
from .cmb_identity import CCMBS_ID, CCMBS_LABEL
from .likelihoods.cmb.native_background import (
    _summarize_declared_background_manifest_summary,
)
from .likelihoods.cmb.native_convergence import (
    resolve_native_numerical_envelope,
)
from .model_selection import (
    ComparisonRequest,
    build_comparison_request,
    validate_comparison_compatibility,
)


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


def _cmb_info(models: Iterable[tuple[object, str]]) -> dict | None:
    """Return native runtime metadata for CMB-capable models."""

    cmb_models: list[object] = []
    for plugin, _ in models:
        if getattr(plugin, "valid_for_cmb", True) is False:
            continue
        contract = getattr(plugin, "CMB_CONTRACT", {}) or {}
        if contract:
            cmb_models.append(plugin)
    if not cmb_models:
        return None

    models_meta: list[dict[str, Any]] = []
    for plugin in cmb_models:
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
        param_map = contract.get("param_map", {}) or {}
        grids = contract.get("grids", {}) or {}
        values = contract.get("values", {}) or {}
        calls = contract.get("calls", []) or []
        if isinstance(contract, dict) and contract.get("background"):
            background_manifest_summary = (
                _summarize_declared_background_manifest_summary(contract)
            )
        else:
            background_manifest_summary = {}
        execution_route = manifest_summary_data.get("execution_route", {})
        perturbation_sources = getattr(perturbation_data, "sources", {}) or {}
        numerical_settings = contract.get("numerical", {}) or {}
        numerical_settings = (
            dict(numerical_settings)
            if isinstance(numerical_settings, dict)
            else numerical_settings
        )
        accuracy_controls = (
            getattr(perturbation_data, "accuracy_controls", {}) or {}
        )
        native_runtime = getattr(plugin, "CMB_NATIVE_RUNTIME", None)
        compile_diagnostics = getattr(
            native_runtime, "compile_diagnostics", None
        )
        envelope_contract = dict(contract)
        if perturbation_data is not None:
            envelope_contract["perturbation_data"] = perturbation_data
        numerical_envelope = resolve_native_numerical_envelope(
            envelope_contract
        ).to_dict()
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
                "execution_solver": CCMBS_ID,
                "execution_solver_label": CCMBS_LABEL,
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
                "perturbation_interaction_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "interactions", {}) or {}
                    )
                ),
                "perturbation_conservation_rule_names": sorted(
                    str(key)
                    for key in (
                        getattr(
                            perturbation_data,
                            "conservation_rules",
                            {},
                        )
                        or {}
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
                "perturbation_sector_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "sectors", {}) or {}
                    )
                ),
                "perturbation_species_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "species", {}) or {}
                    )
                ),
                "perturbation_hierarchy_family_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "hierarchy_families", {})
                        or {}
                    )
                ),
                "perturbation_collision_operator_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "collision_operators", {})
                        or {}
                    )
                ),
                "perturbation_projection_extension_names": sorted(
                    str(key)
                    for key in (
                        getattr(
                            perturbation_data,
                            "projection_extensions",
                            {},
                        )
                        or {}
                    )
                ),
                "perturbation_initial_condition_family_names": sorted(
                    str(key)
                    for key in (
                        getattr(
                            perturbation_data,
                            "initial_condition_families",
                            {},
                        )
                        or {}
                    )
                ),
                "perturbation_projection_typing_names": sorted(
                    str(key)
                    for key in (
                        getattr(perturbation_data, "projection_typing", {})
                        or {}
                    )
                ),
                "perturbation_numerical_settings": numerical_settings,
                "perturbation_accuracy_controls": (
                    dict(accuracy_controls)
                    if hasattr(accuracy_controls, "items")
                    else accuracy_controls
                ),
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
                "native_cmb_execution": {
                    str(key): value for key, value in execution_route.items()
                },
                "native_cmb_numerical_settings": numerical_settings,
                "native_cmb_numerical_envelope": numerical_envelope,
                "native_cmb_graph_manifest_summary": manifest_summary_data,
                "native_cmb_background_manifest_summary": (
                    background_manifest_summary
                ),
                "native_cmb_runtime_manifest_summary": {
                    "execution_route": {
                        str(key): value
                        for key, value in execution_route.items()
                    },
                    "numerical_settings": numerical_settings,
                    "numerical_envelope": numerical_envelope,
                    "accuracy_controls": (
                        dict(accuracy_controls)
                        if hasattr(accuracy_controls, "items")
                        else accuracy_controls
                    ),
                    "runtime_signature": getattr(
                        native_runtime,
                        "runtime_signature",
                        None,
                    ),
                    "compile_diagnostics": (
                        {
                            "runtime_signature": getattr(
                                compile_diagnostics,
                                "runtime_signature",
                                None,
                            ),
                            "compiler": getattr(
                                compile_diagnostics,
                                "compiler",
                                None,
                            ),
                            "compiled_upstream": getattr(
                                compile_diagnostics,
                                "compiled_upstream",
                                None,
                            ),
                            "hot_path_recompilation_allowed": getattr(
                                compile_diagnostics,
                                "hot_path_recompilation_allowed",
                                None,
                            ),
                            "parameter_names": list(
                                getattr(
                                    compile_diagnostics,
                                    "parameter_names",
                                    (),
                                )
                            ),
                            "background_reference_names": list(
                                getattr(
                                    compile_diagnostics,
                                    "background_reference_names",
                                    (),
                                )
                            ),
                        }
                        if compile_diagnostics is not None
                        else {}
                    ),
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
            }
        )

    return {
        "execution_solver": CCMBS_ID,
        "execution_solver_label": CCMBS_LABEL,
        "models": models_meta,
    }


def build_manifest(
    models: Iterable[tuple[object, str]],
    sampler_module: object,
    datasets: Iterable[Dict[str, Any]],
    *,
    state: str = "pending",
    output_policy: str = "unprepared",
    configuration: Optional[dict[str, Any]] = None,
    comparison: ComparisonRequest | None = None,
) -> dict:
    """Collect manifest information for the current run.

    Parameters
    ----------
    models:
        Iterable of ``(plugin, version)`` pairs where ``plugin`` exposes
        ``MODEL_NAME``, ``MODEL_FILENAME``, ``PARAMETER_NAMES`` and
        ``PARAMETER_PRIORS`` attributes.
    sampler_module:
        Selected sampler module object.  ``SAMPLER_VERSION`` is queried when
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
    comparison:
        Required role identity when caller metadata differs from plugin
        display names. When omitted, the two ordered model records define the
        control and test roles.
    """

    model_records = list(models)
    if len(model_records) != 2:
        raise ValueError(
            "Run manifests require exactly one control model and one test "
            "model."
        )
    if comparison is None:
        comparison = build_comparison_request(
            getattr(model_records[0][0], "MODEL_NAME", "control"),
            getattr(model_records[1][0], "MODEL_NAME", "test"),
            control_filename=getattr(
                model_records[0][0], "MODEL_FILENAME", ""
            ),
            test_filename=getattr(model_records[1][0], "MODEL_FILENAME", ""),
        )

    record_names = tuple(
        str(getattr(plugin, "MODEL_NAME", "")) for plugin, _ in model_records
    )
    if comparison.model_names != record_names:
        raise ValueError(
            "Manifest model records must be ordered as the declared control "
            "and test comparison."
        )
    validate_comparison_compatibility(
        comparison,
        control_metadata=getattr(model_records[0][0], "CMB_CONTRACT", {}),
        test_metadata=getattr(model_records[1][0], "CMB_CONTRACT", {}),
    )

    manifest = {
        "copernican": {"version": _copernican_version()},
        "models": [],
        "sampler": {
            "name": getattr(sampler_module, "__name__", "unknown"),
            "version": getattr(sampler_module, "SAMPLER_VERSION", "unknown"),
        },
        "seed": utils.get_random_seed(),
        "datasets": {},
        "git": _git_info(),
        "status": {"state": state, "outputs": output_policy},
        "selection": {
            "models": [],
            "sampler": {},
            "datasets": [],
            "comparison": comparison.as_manifest(),
            "control_model": comparison.control_model.name,
            "test_model": comparison.test_model.name,
        },
    }

    for plugin, version in model_records:
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

    manifest["selection"]["sampler"] = manifest["sampler"].copy()

    if configuration:
        manifest["configuration"] = dict(configuration)
        manifest_configuration = manifest["configuration"]
        manifest_configuration["models"] = list(comparison.model_names)
        manifest_configuration["comparison"] = comparison.as_manifest()
        manifest_configuration["control_model"] = comparison.control_model.name
        manifest_configuration["test_model"] = comparison.test_model.name
    else:
        manifest["configuration"] = {
            "notes": "Derived from GUI selections; update when importing.",
            "sampler": manifest["selection"]["sampler"],
            "models": manifest["selection"]["models"],
            "datasets": manifest["selection"]["datasets"],
            "comparison": comparison.as_manifest(),
            "control_model": comparison.control_model.name,
            "test_model": comparison.test_model.name,
        }

    cmb_details = _cmb_info(model_records)
    if cmb_details is not None:
        manifest["cmb"] = cmb_details

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
