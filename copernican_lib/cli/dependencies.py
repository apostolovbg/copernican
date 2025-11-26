"""Startup dependency helpers for the Copernican Suite CLI.

These routines isolate dependency discovery, caching and optional test
execution so ``copernican.py`` can defer heavy imports until the runtime
configuration is known. Consolidating the logic here keeps the launcher thin
and ensures the same caching rules apply across interactive and automated
invocations. Capturing these steps in one place also reduces regressions when
the managed start scripts evolve because the CLI follows the same rules.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from copernican_lib import console_output as console

DEPENDENCY_CACHE_ENV_VAR = "COPERNICAN_DEP_CACHE_DIR"
DEPENDENCY_CACHE_FILENAME = "dependency_scan.json"
DEPENDENCY_CACHE_SCHEMA = 1


@dataclass
class RuntimeOptions:
    """Runtime configuration derived from environment variables.

    Centralising this metadata avoids scattering environment lookups and keeps
    related defaults aligned across the CLI helpers.
    """

    run_tests: bool = False
    strict_warnings: bool = False


def get_runtime_options() -> RuntimeOptions:
    """Return options from ``COPERNICAN_*`` environment variables.

    This wrapper keeps environment parsing consistent so future options need
    only be added in one location.
    """

    return RuntimeOptions(
        run_tests=os.environ.get("COPERNICAN_RUN_TESTS") == "1",
        strict_warnings=os.environ.get("COPERNICAN_STRICT_WARNINGS") == "1",
    )


def run_startup_tests() -> bool:
    """Execute the project's unit tests via ``python -m unittest discover``.

    Running these tests early provides a lightweight confidence check before
    the CLI proceeds to expensive data preparation steps.
    """

    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        console.write(f"Error running startup tests: {exc}")
        return False
    return result.returncode == 0


def _resolve_dependency_cache_paths() -> tuple[Path, Path]:
    """Return the cache directory and file for dependency scans.

    Keeping this calculation in one helper ensures overrides stay in sync
    wherever the cache is read or written.
    """

    override = os.environ.get(DEPENDENCY_CACHE_ENV_VAR)
    if override:
        cache_dir = Path(override).expanduser()
    else:
        cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_file = cache_dir / DEPENDENCY_CACHE_FILENAME
    return cache_dir, cache_file


def _scan_python_sources(
    search_dirs: list[str], ignore_dirs: set[str]
) -> tuple[list[Path], dict[str, dict[str, int]]]:
    """Return Python source paths and a metadata snapshot for caching.

    The snapshot provides a cheap drift detector so dependency lists can be
    reused when files are untouched.
    """

    python_files: list[Path] = []
    snapshot: dict[str, dict[str, int]] = {}
    for base in search_dirs:
        base_path = Path(base).resolve()
        if not base_path.is_dir():
            continue
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [
                d
                for d in dirs
                if d not in ignore_dirs
                and not d.startswith(".")
                and "site-packages" not in d
            ]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = Path(root, fname).resolve()
                python_files.append(path)
                stat_info = path.stat()
                mtime_ns = getattr(stat_info, "st_mtime_ns", None)
                if mtime_ns is None:
                    mtime_ns = int(stat_info.st_mtime * 1_000_000_000)
                snapshot[str(path)] = {
                    "mtime_ns": int(mtime_ns),
                    "size": int(stat_info.st_size),
                }
    python_files.sort()
    return python_files, snapshot


def _load_cached_dependencies(
    snapshot: dict[str, dict[str, int]], search_dirs: list[str]
) -> set[str] | None:
    """Return cached dependency names when the snapshot is unchanged.

    Avoiding unnecessary rescans keeps startup latency low on repeated runs.
    """

    _, cache_file = _resolve_dependency_cache_paths()
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("schema") != DEPENDENCY_CACHE_SCHEMA:
        return None
    if cached.get("search_dirs") != search_dirs:
        return None
    # Abort on snapshot drift because any file touched since the last run could
    # hide a new import that the managed environment must install.
    if cached.get("snapshot") != snapshot:
        return None
    cached_packages = cached.get("packages", [])
    if not isinstance(cached_packages, list):
        return None
    return set(cached_packages)


def _store_dependency_cache(
    snapshot: dict[str, dict[str, int]],
    search_dirs: list[str],
    pkg_names: Iterable[str],
) -> None:
    """Persist the dependency cache snapshot for subsequent runs.

    Writing a normalised record allows later runs to detect drift quickly and
    reuse the previous dependency list.
    """

    cache_dir, cache_file = _resolve_dependency_cache_paths()
    cache_dir.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "schema": DEPENDENCY_CACHE_SCHEMA,
        "snapshot": snapshot,
        "packages": sorted(pkg_names),
        "search_dirs": search_dirs,
    }
    try:
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2, sort_keys=True)
    except OSError:
        console.write("Warning: Unable to update dependency cache.")


def _gather_required_packages(
    search_dirs: list[str] | None = None,
) -> set[str]:
    """Inspect the source tree to derive external dependencies.

    Parameters
    ----------
    search_dirs
        Optional set of directories to scan. When omitted the helper walks the
        installed Copernican library and bundled ``engines`` tree.
    """

    # Reasoning: deriving dependencies programmatically ensures custom
    # engines and datasets pulled into the tree are reflected in the managed
    # environment without maintaining a hand-written list.

    if search_dirs is None:
        search_dirs = [
            str(Path(__file__).resolve().parents[2] / "copernican_lib"),
            str(Path(__file__).resolve().parents[2] / "engines"),
        ]
    ignore_dirs = {"__pycache__", "tests", "output", "logs"}
    python_files, snapshot = _scan_python_sources(search_dirs, ignore_dirs)
    cached = _load_cached_dependencies(snapshot, search_dirs)
    if cached is not None:
        return cached

    pkg_names: set[str] = set()
    for file in python_files:
        try:
            with file.open("r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=str(file))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pkg_names.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Relative imports always target modules inside the
                    # Copernican tree, so they should never be treated as
                    # external dependencies.
                    continue
                if node.module is None:
                    continue
                pkg_names.add(node.module.split(".")[0])
    ignore = {
        "__future__",
        "sys",
        "os",
        "math",
        "random",
        "logging",
        "time",
        "argparse",
        "json",
        "datetime",
        "pathlib",
        "typing",
        "traceback",
        "importlib",
        "inspect",
        "types",
        "platform",
        "builtins",
        "dataset_registry",
        "engine_plugin_validation",
        "model_spec_validator",
        "csv_writer",
        "plotter",
        "logger",
        "utils",
    }
    filtered = {
        pkg
        for pkg in pkg_names
        if pkg not in ignore
        and not pkg.startswith(("copernican_lib", "engines"))
    }
    _store_dependency_cache(snapshot, search_dirs, filtered)
    return filtered


def check_dependencies() -> None:
    """Ensure required packages exist inside the local ``.venv``.

    The helper no longer attempts to install dependencies on the user's
    behalf. Failing fast when packages are missing keeps the CLI aligned with
    the managed launcher scripts, which already provision the pinned
    ``requirements.lock`` set before handing control to ``copernican.py``.
    Operators encountering missing wheels must re-run the appropriate
    ``start.*`` helper to rebuild the environment rather than invoking ``pip``
    from inside the program.

    Performing the verification inside the CLI catches misconfigured
    environments early and mirrors the expectations baked into the launch
    scripts.
    """

    console.write("--- Running System Dependency Check ---")

    if Path(sys.prefix).resolve().name != ".venv":
        console.write(
            (
                "ERROR: The Copernican Suite must run inside the local "
                "'.venv'. Launch the appropriate start script for your OS."
            ),
            error=True,
        )
        sys.exit(1)

    required = sorted(_gather_required_packages())
    missing: list[str] = []
    for pkg in required:
        try:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        except ValueError:
            try:
                importlib.import_module(pkg)
            except Exception:
                missing.append(pkg)

    if missing:
        console.write(
            (
                "Missing packages detected: "
                f"{', '.join(sorted(missing))}. Please rerun the "
                "appropriate start script to rebuild the managed environment."
            ),
            error=True,
        )
        sys.exit(1)

    console.write("✅ System Dependency Check Passed. Continuing...\n")


def load_third_party_modules():
    """Import heavy optional dependencies lazily for CLI use.

    Delaying these imports keeps the interactive launcher responsive until the
    operator opts into actions that genuinely need the heavy libraries.
    """

    import multiprocessing as mp

    import matplotlib.pyplot as plt
    import numpy as np  # local import to defer heavy wheels

    return np, plt, mp
