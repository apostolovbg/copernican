"""Startup dependency helpers for the Copernican CLI.

These routines isolate dependency discovery, caching and optional test
execution so the package entrypoint can defer heavy imports until the runtime
configuration is known. Consolidating the logic here keeps the launcher thin
and ensures the same caching rules apply across interactive and automated
invocations.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy

from copernican_lib import console_output as console

DEPENDENCY_CACHE_ENV_VAR = "COPERNICAN_DEP_CACHE_DIR"
DEPENDENCY_CACHE_FILENAME = "dependency_scan.json"
DEPENDENCY_CACHE_SCHEMA = 1


@dataclass
class RuntimeOptions:
    """Runtime configuration derived from environment variables."""

    run_tests: bool = False
    strict_warnings: bool = False


def get_runtime_options() -> RuntimeOptions:
    """Return options from ``COPERNICAN_*`` environment variables."""

    return RuntimeOptions(
        run_tests=os.environ.get("COPERNICAN_RUN_TESTS") == "1",
        strict_warnings=os.environ.get("COPERNICAN_STRICT_WARNINGS") == "1",
    )


def run_startup_tests() -> bool:
    """Execute the project's unit tests in-process via unittest discovery."""

    try:
        repo_root = Path(__file__).resolve().parents[2]
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(repo_root),
            pattern="test*.py",
            top_level_dir=str(repo_root),
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover
        console.write(f"Error running startup tests: {exc}")
        return False
    return result.wasSuccessful()


def _resolve_dependency_cache_paths() -> tuple[Path, Path]:
    """Return the cache directory and file for dependency scans."""

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
    """Return Python source paths and a metadata snapshot for caching."""

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
    """Return cached dependency names when the snapshot is unchanged."""

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
    if cached.get("snapshot") != snapshot:
        return None
    cached_packages = cached.get("packages", [])
    if not isinstance(cached_packages, list):
        return None
    console.write("Dependency scan: cache hit, using cached package list.")
    return set(cached_packages)


def _store_dependency_cache(
    snapshot: dict[str, dict[str, int]],
    search_dirs: list[str],
    pkg_names: Iterable[str],
) -> None:
    """Persist the dependency cache snapshot for subsequent runs."""

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
        installed Copernican package, bundled datasets, and ``engines``
        tree.
    """

    if search_dirs is None:
        search_dirs = [
            str(Path(__file__).resolve().parents[2] / "copernican"),
            str(Path(__file__).resolve().parents[2] / "copernican_lib"),
            str(Path(__file__).resolve().parents[2] / "engines"),
        ]
    ignore_dirs = {"__pycache__", "tests", "output", "logs"}
    console.write(
        "Dependency scan: scanning Python sources for dependency imports..."
    )
    python_files, snapshot = _scan_python_sources(search_dirs, ignore_dirs)
    console.write(
        f"Dependency scan: inspected {len(python_files)} Python files."
    )
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
        "engine_adapter",
        "model_spec_validator",
        "csv_writer",
        "plotter",
        "logger",
        "utils",
    }
    console.write(
        "Dependency scan: cache missing or stale; analysing parsed imports..."
    )
    filtered = {
        pkg
        for pkg in pkg_names
        if pkg not in ignore
        and not pkg.startswith(("copernican", "copernican_lib", "engines"))
    }
    _store_dependency_cache(snapshot, search_dirs, filtered)
    console.write(
        f"Dependency scan: resolved {len(filtered)} external package(s)."
    )
    return filtered


def check_dependencies() -> None:
    """Ensure required packages exist inside the local ``.venv``.

    The helper no longer attempts to install dependencies on the user's
    behalf. Failing fast when packages are missing keeps the CLI aligned with
    the managed environment, which provisions the pinned ``requirements.lock``
    set before handing control to ``python -m copernican``.
    Operators encountering missing wheels must repair the local ``.venv`` and
    relaunch the package entrypoint rather than invoking ``pip`` from inside
    the program.
    """

    console.write("--- Running System Dependency Check ---")

    if Path(sys.prefix).resolve().name != ".venv":
        console.write(
            (
                "ERROR: Copernican must run inside the local '.venv'. "
                "Launch it with `python -m copernican`."
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
            except (ImportError, ModuleNotFoundError):
                missing.append(pkg)

    if missing:
        console.write(
            (
                "Missing packages detected: "
                f"{', '.join(sorted(missing))}. Please repair the local "
                "'.venv' and relaunch `python -m copernican`."
            ),
            error=True,
        )
        sys.exit(1)

    console.write("✅ System Dependency Check Passed. Continuing...\n")


def load_third_party_modules():
    """Import heavy optional dependencies lazily for CLI use."""

    import multiprocessing as multiprocessing_module

    import matplotlib.pyplot as plt

    return numpy, plt, multiprocessing_module
