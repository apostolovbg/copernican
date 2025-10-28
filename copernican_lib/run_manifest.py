"""Run manifest generator for the Copernican Suite.

The manifest records critical information required to reproduce a run.
It captures the Copernican Suite version, model and engine details,
parameter priors, dataset hashes provided by the data loaders and the Git
state.  Each run directory stores the resulting YAML file so that analyses can
be traced back unambiguously.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Iterable, Tuple

import yaml

from . import utils
from .version import get_version


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


def build_manifest(
    models: Iterable[Tuple[object, str]],
    engine_module: object,
    datasets: Iterable[Tuple[str, str, Dict[str, str]]],
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
        Iterable of ``(dataset_id, data_dir, file_hashes)`` tuples.  The
        ``file_hashes`` mapping mirrors the ``file_hashes`` attribute attached
        to the :class:`pandas.DataFrame` produced by the dataset loader.
    """

    manifest = {
        "copernican": {"version": get_version()},
        "models": [],
        "engine": {
            "name": getattr(engine_module, "__name__", "unknown"),
            "version": getattr(engine_module, "ENGINE_VERSION", "unknown"),
        },
        "seed": utils.get_random_seed(),
        "datasets": {},
        "git": _git_info(),
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
        manifest["models"].append(
            {
                "name": getattr(plugin, "MODEL_NAME", "unknown"),
                "version": version,
                "filename": getattr(plugin, "MODEL_FILENAME", ""),
                "priors": priors,
            }
        )

    for dataset_id, data_dir, file_hashes in datasets:
        manifest["datasets"][dataset_id] = {
            "path": data_dir,
            "hashes": file_hashes,
        }

    return manifest


def save_manifest(manifest: dict, output_dir: str) -> str:
    """Persist ``manifest`` as ``run_manifest_<timestamp>.yml``.

    The filename includes a timestamp so repeated runs do not clobber
    earlier manifests.  The full path to the saved file is returned.
    """

    utils.ensure_dir_exists(output_dir)
    ts = utils.get_timestamp()
    path = os.path.join(output_dir, f"run_manifest_{ts}.yml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return path


__all__ = ["build_manifest", "save_manifest"]
