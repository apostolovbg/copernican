"""Run manifest generator for the Copernican Suite.

The manifest records critical information required to reproduce a run.
It captures model and engine details, parameter priors, dataset hashes
and the Git state.  Each run directory stores the resulting YAML file so
that analyses can be traced back unambiguously.
"""

from __future__ import annotations

import os
import subprocess
from typing import Iterable, Tuple

import yaml

from . import utils


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
    datasets: Iterable[Tuple[str, str]],
    seed: int,
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
        Iterable of ``(dataset_id, data_dir)`` tuples.
    seed:
        RNG seed applied to the run.
    """

    manifest = {
        "models": [],
        "engine": {
            "name": getattr(engine_module, "__name__", "unknown"),
            "version": getattr(engine_module, "ENGINE_VERSION", "unknown"),
        },
        "seed": seed,
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

    for dataset_id, data_dir in datasets:
        file_hashes = {}
        for root, _, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, data_dir)
                file_hashes[rel] = utils.compute_sha256(path)
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
