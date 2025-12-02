"""Run manifest generator for the Copernican Suite.

The manifest records critical information required to reproduce a run. It
captures the Copernican Suite version, model and engine details, parameter
priors, dataset hashes provided by the data loaders and the Git state.  Each
run directory stores the resulting YAML file so that analyses can be traced
back unambiguously.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from . import utils
from . import version as version_module
from .likelihoods import cmb as cmb_module


def _copernican_version() -> str:
    """Return the suite version while tolerating missing helpers.

    Some macOS installations reported ``ImportError`` when
    ``copernican_lib.version.get_version`` was unavailable even though the
    module itself existed. Importing the attribute lazily keeps the
    ``run_manifest`` module importable in that scenario so ``start.command``
    can still launch and emit a manifest. Falling back to ``"0+unknown"``
    mirrors the final stage inside :func:`copernican_lib.version.get_version`
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
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"
    try:
        subprocess.check_output(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
        dirty = False
    except subprocess.CalledProcessError:
        dirty = True
    except Exception:
        dirty = True
    return {"commit": commit, "dirty": dirty}


def _camb_info(models: Iterable[tuple[object, str]]) -> dict | None:
    """Return CAMB metadata for models that supply a CMB mapping."""

    camb_models: list[object] = []
    for plugin, _ in models:
        if getattr(plugin, "valid_for_cmb", True) is False:
            continue
        param_map = getattr(plugin, "CMB_PARAM_MAP", {}) or {}
        if param_map:
            camb_models.append(plugin)
    if not camb_models:
        return None

    try:  # pragma: no cover - graceful when CAMB absent in minimal envs
        import camb  # type: ignore

        version = getattr(camb, "__version__", "unknown")
    except Exception:
        version = "unavailable"

    configuration = cmb_module.describe_camb_configuration()
    models_meta: list[dict[str, Any]] = []
    for plugin in camb_models:
        keys = sorted(str(key) for key in getattr(plugin, "CMB_PARAM_MAP", {}))
        models_meta.append(
            {
                "model": getattr(plugin, "MODEL_NAME", "unknown"),
                "param_map_keys": keys,
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
        ts = utils.get_timestamp()
        target = Path(output_dir) / f"run_manifest_{ts}.yml"
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return str(target)


def load_manifest(path: str) -> dict:
    """Load a manifest from disk for reuse in a new run."""

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
