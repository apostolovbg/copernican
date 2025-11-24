# Last Updated: 2025-11-24
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# copernican_suite/copernican.py
# flake8: noqa
# isort: skip_file
# fmt: off
"""Copernican Suite - Main Orchestrator.

Last Updated: 2025-11-24

This script ties together model selection, dataset loading, dependency
checks and result generation.  Runtime behaviour is configured through
environment variables set by the cross-platform ``start`` launchers, so
no raw command line flags are exposed to end users.  The module also
houses the optional test runner and automated package installer so that
a fresh checkout can execute with minimal setup.
"""


import ast
import copy
import datetime
import faulthandler
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from copernican_lib import console_output as console
from copernican_lib import run_manifest
from copernican_lib import result_writer
from copernican_lib.diagnostics import (
    bao_residual_diagnostics,
    cmb_residual_diagnostics,
)
import copernican_lib.version as version_module
from copernican_lib.plugins import PluginValidationError

# Verify interpreter version early so users see clear feedback
MIN_PYTHON = (3, 11)


def exit_clean(code: int = 0) -> None:
    """Exit the program after printing a newline."""
    console.write("")
    sys.exit(code)


if sys.version_info < MIN_PYTHON:
    console.write(
        (
            f"ERROR: Copernican Suite requires Python {MIN_PYTHON[0]}"
            f".{MIN_PYTHON[1]} or later."
        ),
        error=True,
    )
    exit_clean(1)

# Require execution from the repository's virtual environment so that global
# site-packages are ignored.
EXPECTED_VENV = (Path(__file__).resolve().parent / ".venv").resolve()
current_venv = os.environ.get("VIRTUAL_ENV")
if current_venv is None or Path(current_venv).resolve() != EXPECTED_VENV:
    console.write(
        (
            "ERROR: Run Copernican Suite via start.sh, start.command or "
            "start.bat inside its managed .venv."
        ),
        error=True,
    )
    exit_clean(1)

# Enable low-level stack tracing so crashes reveal their origin.
faulthandler.enable()

# Delay heavy third-party imports until after the dependency check.
# Doing so keeps startup quick and lets ``check_dependencies`` provide a
# clean error message before the interpreter tries to import missing
# modules.
np = None
plt = None
mp = None

model_spec_validator = None
model_coder = None
engine_plugin_validation = None
plotter = None
csv_writer = None
log_mod = None
logger = None
dataset_registry = None

# Retrieve the runtime version from installed package metadata. When the
# distribution is not installed, the helper below returns ``"0+unknown"`` so
# logs and plot footers still carry a version-like identifier.  Importing the
# attribute lazily avoids the ``ImportError`` seen on some macOS systems where
# ``copernican_lib.version`` was importable but ``get_version`` was not
# exported.


def _copernican_version() -> str:
    """Return the Copernican Suite version while tolerating missing helpers.

    The launcher crashed on macOS when ``start.command`` re-imported this
    module and ``copernican_lib.version.get_version`` was absent even though
    the version module itself was available.  Looking up the attribute at
    runtime allows the menu to load successfully and matches the fallbacks
    inside :func:`copernican_lib.version.get_version`.
    """

    getter = getattr(version_module, "get_version", None)
    if callable(getter):
        return getter()
    return "0+unknown"


COPERNICAN_VERSION = _copernican_version()
CURRENT_LOG_FILE = None

DEPENDENCY_CACHE_ENV_VAR = "COPERNICAN_DEP_CACHE_DIR"
DEPENDENCY_CACHE_FILENAME = "dependency_scan.json"
DEPENDENCY_CACHE_SCHEMA = 1


def _handle_fatal_signal(signum: int, _frame: object) -> None:
    """Dump a stack trace to the log and console then exit cleanly.

    Critical signals such as ``SIGSEGV`` may indicate a corrupted process
    state, so the handler writes a traceback for debugging before terminating
    immediately using :func:`os._exit`.
    """

    sig_name = signal.Signals(signum).name
    msg = f"Fatal signal {sig_name} received"
    console.write(msg, error=True)
    if CURRENT_LOG_FILE:
        try:
            with open(CURRENT_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception as exc:
            if logger:
                # Preserve the failure details in the central log.
                logger.exception(
                    "copernican.py: failed to append fatal trace to %s",
                    CURRENT_LOG_FILE,
                )
            else:
                # Fallback so the issue is still visible to the user.
                console.write(
                    f"copernican.py: failed to write to {CURRENT_LOG_FILE}:"
                    f" {exc}",
                    error=True,
                )
    faulthandler.dump_traceback(all_threads=True)
    console.write("Exiting due to fatal signal.", error=True)
    os._exit(1)


for _name in ("SIGILL", "SIGSEGV", "SIGFPE"):
    _sig = getattr(signal, _name, None)
    if _sig is not None:
        signal.signal(_sig, _handle_fatal_signal)


def _delete_log_file(path: str) -> None:
    """Remove the given log file if it exists."""
    if path and os.path.isfile(path):
        try:
            os.remove(path)
            console.write(f"Removed log file {path}")
        except OSError as exc:
            if logger:
                # Deletion failures are non-critical but worth recording.
                logger.warning(
                    "copernican.py: could not remove log file %s: %s",
                    path,
                    exc,
                )
            else:
                # Ensure the user sees the failure even without the logger.
                console.write(
                    f"copernican.py: unable to remove log file {path}:"
                    f" {exc}",
                    error=True,
                )


def _remove_run_dir(path: str) -> None:
    """Delete the run output directory and its contents."""
    if path and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            console.write(f"Removed run directory {path}")
        except OSError as exc:
            if logger:
                logger.warning(
                    "copernican.py: could not remove run dir %s: %s",
                    path,
                    exc,
                )
            else:
                console.write(
                    f"copernican.py: could not remove run dir {path}: {exc}",
                    error=True,
                )


def _get_cpu_info() -> tuple[str, str]:
    """Return CPU model and current clock speed."""
    cpu = platform.processor() or platform.uname().processor or "Unknown CPU"
    freq = None
    try:
        import psutil  # type: ignore

        freq_info = psutil.cpu_freq()
        if freq_info:
            freq = freq_info.current / 1000.0
    except Exception as exc:
        if logger:
            # ``psutil`` is optional; log and continue with unknown frequency.
            logger.warning(
                "copernican.py: psutil unavailable for CPU freq: %s",
                exc,
            )
        else:
            # Without a logger, surface the issue via the console.
            console.write(
                f"copernican.py: psutil import failed: {exc}",
                error=True,
            )
    if freq is None and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    if line.startswith("model name") and cpu == "Unknown CPU":
                        cpu = line.split(":", 1)[1].strip()
                    if line.startswith("cpu MHz") and freq is None:
                        freq = float(line.split(":", 1)[1]) / 1000.0
        except Exception as exc:
            if logger:
                # Reading ``/proc/cpuinfo`` can fail in restricted
                # environments; log and fall back to placeholders.
                logger.warning(
                    "copernican.py: could not read /proc/cpuinfo: %s",
                    exc,
                )
            else:
                # Fallback console message when the logger is unavailable.
                console.write(
                    f"copernican.py: cannot read /proc/cpuinfo: {exc}",
                    error=True,
                )
    freq_str = f"{freq:.2f} GHz" if freq else "Unknown GHz"
    return cpu, freq_str


# The high-level workflow is broken into small helper functions below. Each
# helper is documented in plain language so non-programmers can follow the
# logic of the program.


def run_startup_tests():
    """Execute the project's unit tests via ``python -m unittest discover``.

    The helper delegates to Python's standard discovery mechanism so the
    start script's *Run tests* option behaves identically to invoking
    ``python -m unittest discover`` from the command line. A ``True`` return
    value indicates that all tests passed.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            check=False,
        )
    except Exception as exc:
        console.write(f"Error running startup tests: {exc}")
        return False
    return result.returncode == 0


@dataclass
class RuntimeOptions:
    """Runtime configuration derived from environment variables."""

    run_tests: bool = False
    strict_warnings: bool = False


@dataclass
class DashboardState:
    """Track dashboard selections across configuration cycles."""

    seed: int | None = None
    selected_model: str = ""
    alt_model_plugin: Any | None = None
    alt_model_parsed: dict[str, Any] | None = None
    engine_module: Any | None = None
    sampling_plan: dict[str, Any] | None = None
    use_bao: bool = True
    use_cmb: bool = True
    display_progress: bool = True
    last_output_dir: str | None = None
    last_log_file: str | None = None
    last_run_started: datetime.datetime | None = None
    last_run_finished: datetime.datetime | None = None


@dataclass
class DashboardState:
    """Track dashboard selections across configuration cycles."""

    seed: int | None = None
    selected_model: str = ""
    alt_model_plugin: Any | None = None
    alt_model_parsed: dict[str, Any] | None = None
    engine_module: Any | None = None
    sampling_plan: dict[str, Any] | None = None
    use_bao: bool = True
    use_cmb: bool = True
    display_progress: bool = True
    last_output_dir: str | None = None
    last_log_file: str | None = None
    last_run_started: datetime.datetime | None = None
    last_run_finished: datetime.datetime | None = None


def get_runtime_options() -> RuntimeOptions:
    """Return options from ``COPERNICAN_*`` environment variables."""

    return RuntimeOptions(
        run_tests=os.environ.get("COPERNICAN_RUN_TESTS") == "1",
        strict_warnings=os.environ.get("COPERNICAN_STRICT_WARNINGS") == "1",
    )


def select_seed() -> int:
    """Prompt the operator to choose a reproducible random seed.

    The dialog mirrors other Copernican menus with structured spacing so the
    seed selection feels intentional inside the dashboard. When
    ``COPERNICAN_SEED`` is provided the helper honours it immediately while
    documenting the choice for interactive users.
    """

    from copernican_lib import utils as _utils

    console.write("")
    console.write("Random Seed Selection")
    console.write("---------------------")
    console.write(
        "This seed initialises every random number generator used by "
        "Copernican so runs can be repeated exactly."
    )
    console.write("")

    env_seed = os.environ.get("COPERNICAN_SEED")
    if env_seed is not None:
        try:
            seed = int(env_seed)
        except ValueError:
            console.write(
                "COPERNICAN_SEED is not an integer; falling back to the menu.",
                error=True,
            )
        else:
            console.write(
                "Using environment-provided seed: "
                f"{seed}"
            )
            _utils.set_random_seed(seed)
            return seed

    console.write("Please choose how to seed the sampler:")
    console.write("  1) Accept the default seed (0)")
    console.write("  2) Enter a custom integer seed")
    console.write(
        "  3) Generate a random seed (uniform in [0, 2^32 - 1])"
    )
    console.write("")

    seed = 0
    while True:
        choice = _wait_for_menu_choice(
            {"1", "2", "3", "", "default", "custom", "random", "\r", "\n"},
            prompt="Press 1, 2 or 3: ",
        )
        if choice in {"1", "", "default", "\r", "\n"}:
            seed = 0
            console.write("Default seed 0 selected.")
            break
        if choice in {"2", "custom"}:
            while True:
                entry = console.ask("Enter integer seed: ").strip()
                try:
                    seed = int(entry)
                    console.write(f"Custom seed {seed} selected.")
                    break
                except ValueError:
                    console.write(
                        "Seeds must be whole numbers. Please try again.",
                        error=True,
                    )
            break
        if choice in {"3", "random"}:
            seed = random.randint(0, 2**32 - 1)
            console.write(f"Generated random seed {seed}.")
            break
        console.write("Please choose 1, 2 or 3.", error=True)

    _utils.set_random_seed(seed)
    return seed


def show_splash_screen():
    """Displays the startup banner once at launch."""
    banner = [
        "=" * 70,
        "\n",
        "C O P E R N I C A N   S U I T E".center(70),
        "\n",
        "=" * 70,
        "\n",
        (
            "A tool for rapid development, prototyping and testing of\n"
        ).center(70),
        (
            "alternative cosmological frameworks against observational data\n"
        ).center(70),
        "-" * 70,
        f"build {COPERNICAN_VERSION}".center(70),
        "=" * 70,
        "\n",
    ]
    for line in banner:
        console.write(line)
    # ``time.sleep`` pauses briefly so operators can read the banner.
    # Importing ``time`` at module scope keeps the helper available even when
    # tests stub timing utilities.
    time.sleep(1)
    # The runtime banner now concludes with a single spacer so subsequent
    # prompts sit on a clean line without repeating explanatory text that the
    # documentation already covers.
    console.write("")


# --- System Dependency and Sanity Checker ---


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
    if cached.get("schema") != DEPENDENCY_CACHE_SCHEMA:
        return None
    canonical_dirs = sorted(str(Path(d).resolve()) for d in search_dirs)
    if cached.get("search_dirs") != canonical_dirs:
        return None
    if cached.get("files") != snapshot:
        return None
    deps = cached.get("dependencies")
    if not isinstance(deps, list):
        return None
    return set(deps)


def _store_dependency_cache(
    snapshot: dict[str, dict[str, int]],
    search_dirs: list[str],
    dependencies: set[str],
) -> None:
    """Persist the dependency cache snapshot for subsequent runs."""

    cache_dir, cache_file = _resolve_dependency_cache_paths()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(".tmp")
        payload = {
            "schema": DEPENDENCY_CACHE_SCHEMA,
            "search_dirs": sorted(str(Path(d).resolve()) for d in search_dirs),
            "files": snapshot,
            "dependencies": sorted(dependencies),
        }
        with tmp_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_file.replace(cache_file)
    except OSError as exc:
        console.write(
            "Warning: Unable to update dependency cache."
            f" {exc}",
            error=True,
        )


def _gather_required_packages(
    search_dirs: list[str] | None = None,
) -> set[str]:
    """Return external packages imported across project modules."""
    # Rather than rely on ``pip freeze`` or manual lists this function
    # walks through the source tree and parses each ``import`` statement
    # with :mod:`ast`.  This keeps the dependency check accurate even
    # when new optional modules are added.
    pkg_names = set()
    if search_dirs is None:
        search_dirs = ["copernican_lib", "engines", "tests", "."]
    ignore_dirs = {
        "venv",
        ".venv",
        "env",
        "build",
        "dist",
        "__pycache__",
        "copernican_suite.egg-info",
    }
    py_files, snapshot = _scan_python_sources(search_dirs, ignore_dirs)
    cached = _load_cached_dependencies(snapshot, search_dirs)
    if cached is not None:
        console.write(
            "Dependency scan cache is current; skipping source parsing."
        )
        return cached
    for path in py_files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    pkg_names.add(node.module.split(".")[0])
    ignore = {
        # Standard library modules or local packages that should not trigger
        # the dependency installer
        "os",
        "sys",
        "time",
        "logging",
        "subprocess",
        "importlib",
        "multiprocessing",
        "glob",
        "shutil",
        "platform",
        "inspect",
        "types",
        "pathlib",
        "builtins",
        "traceback",
        "typing",
        "msvcrt",
        # Local modules within this repository (under ``copernican_lib``)
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
    """Verify required packages exist inside the managed ``.venv``.

    The suite bundles a virtual environment under ``.venv`` that is activated
    by the ``start.*`` launchers. This check confirms the interpreter is
    running from that environment and reports any missing packages so the
    caller can rebuild the environment via the launchers. Dependency
    installation is intentionally delegated to the start scripts to keep the
    runtime thin and predictable.
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
        exit_clean(1)

    required = sorted(_gather_required_packages())
    missing: list[str] = []
    for pkg in required:
        try:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        except ValueError:
            # Python 3.13 may raise ValueError when __main__.__spec__ is None.
            # Fallback to a simple import attempt in that case.
            try:
                importlib.import_module(pkg)
            except Exception:
                missing.append(pkg)

    if missing:
        console.write(
            f"❌ Missing packages detected: {', '.join(missing)}",
            error=True,
        )
        console.write(
            (
                "Please rerun start.sh, start.command or start.bat to refresh "
                "the managed environment. Dependency installation now lives in "
                "the launchers."
            ),
            error=True,
        )
        exit_clean(1)

    console.write("✅ System Dependency Check Passed. Continuing...\n")


# Modules that rely on optional packages will be imported in ``main_workflow``

lcdm = None


def load_alternative_model_plugin(model_filepath):
    """Dynamically loads an alternative cosmological model plugin."""
    logger = log_mod.get_logger()
    if not model_filepath.endswith(".py"):
        model_filepath += ".py"
    if not os.path.isfile(model_filepath):
        logger.error(
            f"Alternative model plugin file '{model_filepath}' not found."
        )
        return None
    try:
        module_name = os.path.splitext(os.path.basename(model_filepath))[0]
        spec = importlib.util.spec_from_file_location(
            module_name, model_filepath
        )
        alt_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alt_model_module)
        try:
            engine_plugin_validation.validate_plugin(alt_model_module)
        except PluginValidationError as exc:
            logger.error(
                (
                    f"Model plugin '{os.path.basename(model_filepath)}' "
                    f"failed validation: {exc}"
                )
            )
            return None
        logger.info(
            f"Successfully loaded alternative model: "
            f"{alt_model_module.MODEL_NAME}"
        )
        return alt_model_module
    except Exception as e:
        logger.error(
            f"Error loading model plugin "
            f"'{os.path.basename(model_filepath)}': {e}",
            exc_info=True,
        )
        return None


def select_from_list(options, prompt):
    """Display ``options`` and return the item chosen by the user."""

    # The caller supplies a short prompt ("Select model").  This helper prints
    # each option with a number so the user can respond with just an integer.
    # Returning ``None`` signals that the user cancelled the operation.
    if not options:
        return None
    header = prompt.replace("Select ", "").strip()
    if not header.endswith("s"):
        header += "s"
    console.write(f"\nAvailable {header}:")
    for i, opt in enumerate(options, 1):
        console.write(f"  {i}. {opt}")
    console.write(
        "Write the number of your preferred choice or 'c' to cancel:"
    )
    while True:
        choice = console.ask("> ").strip()
        if choice.lower() == "c":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        console.write("Invalid selection. Try again.")


def _normalise_failure_reasons(details: Iterable[str] | str) -> list[str]:
    """Return a list of human-readable reasons extracted from ``details``."""

    if isinstance(details, str):
        text = details.split(":", 1)[-1] if ":" in details else details
        raw_parts = text.replace(";", "\n").splitlines()
    else:
        raw_parts = []
        for item in details:
            raw_parts.extend(str(item).splitlines())

    reasons: list[str] = []
    for part in raw_parts:
        cleaned = part.strip()
        if cleaned:
            reasons.append(cleaned)
    return reasons or ["An unspecified error occurred during model setup."]


def _prompt_configuration_retry(reasons: Iterable[str]) -> bool:
    """Return ``True`` to retry configuration, ``False`` to stop."""

    console.write("")
    console.write("Configuration cannot continue because:")
    for entry in reasons:
        console.write(f"  - {entry}")
    console.write("")
    console.write("How would you like to proceed?")
    console.write("  1) Retry the configuration flow")
    console.write("  C) Return to the dashboard")
    console.write("")

    while True:
        decision = _wait_for_menu_choice(
            {"1", "c", "restart", "retry", "cancel", "exit", "dashboard"},
            prompt="Press 1 to retry or C to return: ",
        )
        if decision in {"", "1", "restart", "retry"}:
            console.write("")
            console.write("Restarting the configuration prompts.")
            return True
        if decision in {"c", "cancel", "exit", "dashboard"}:
            return False
        console.write("Please choose 1 to retry or C to return.", error=True)


def _count_active_parameters(plugin, *, engine_module) -> int:
    """Return the number of free parameters declared by ``plugin``.

    ``emcee`` mandates at least ``2 * ndim`` walkers.  The helper mirrors the
    engine's fixed-parameter detection so the interactive prompt can recommend
    an ensemble size that respects any effectively locked coordinates.
    """

    bounds = list(getattr(plugin, "PARAMETER_BOUNDS", []))
    names = list(getattr(plugin, "PARAMETER_NAMES", []))
    if not bounds or len(bounds) != len(names):
        return max(len(names), 1)

    rtol = getattr(engine_module, "_FIXED_BOUNDS_RTOL", 1e-9)
    atol = getattr(engine_module, "_FIXED_BOUNDS_ATOL", 1e-12)

    active = 0
    for low, high in bounds:
        lower = -math.inf if low is None else float(low)
        upper = math.inf if high is None else float(high)
        if math.isfinite(lower) and math.isfinite(upper):
            width = upper - lower
            centre = (upper + lower) / 2.0
            scale = max(abs(centre), 1.0)
            threshold = scale * rtol + atol
            if math.isfinite(width) and width <= threshold:
                continue
        active += 1
    return max(active, 1)


def _resolve_fit_function(engine_module):
    """Return the engine's cosmology fitting callable and its name."""

    fit_fn = getattr(engine_module, "fit_cosmology_parameters", None)
    if fit_fn is not None:
        return fit_fn, "fit_cosmology_parameters"

    legacy_fn = getattr(engine_module, "fit_sne_parameters", None)
    if legacy_fn is not None:
        warning_logger = log_mod.get_logger() if log_mod else None
        if warning_logger:
            warning_logger.warning(
                (
                    "Engine %s exposes the legacy fit_sne_parameters; "
                    "prefer fit_cosmology_parameters to match the "
                    "multi-probe workflow."
                ),
                getattr(engine_module, "__name__", engine_module),
            )
        return legacy_fn, "fit_sne_parameters"

    raise AttributeError(
        "Engine lacks fit_cosmology_parameters and fit_sne_parameters"
    )


def _prompt_nested_configuration(
    engine_module,
    lcdm_plugin,
    alt_model_plugin,
) -> dict[str, int | float] | str | None:
    """Collect configuration parameters for the nested sampler backend."""

    fit_fn, _ = _resolve_fit_function(engine_module)
    fit_sig = inspect.signature(fit_fn)

    def _default_int(param: str, fallback: int) -> int:
        try:
            value = fit_sig.parameters[param].default
        except (KeyError, AttributeError):
            return fallback
        if value is inspect._empty:
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _default_float(param: str, fallback: float) -> float:
        try:
            value = fit_sig.parameters[param].default
        except (KeyError, AttributeError):
            return fallback
        if value is inspect._empty:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    lcdm_active = _count_active_parameters(
        lcdm_plugin,
        engine_module=engine_module,
    )
    alt_active = _count_active_parameters(
        alt_model_plugin,
        engine_module=engine_module,
    )
    max_active = max(lcdm_active, alt_active)
    min_live = max(20, 4 * max_active)

    recommended_live = max(_default_int("n_live_points", min_live), min_live)
    recommended_max_iter = max(
        _default_int("max_iterations", recommended_live * 50),
        recommended_live * 25,
    )
    recommended_tol = max(
        _default_float("evidence_tolerance", 1e-3),
        1e-12,
    )
    recommended_enlargement = max(
        _default_float("enlargement_fraction", 1.5),
        1.0,
    )

    def _collect_custom_plan() -> dict[str, int | float] | str | None:
        while True:
            console.write("")
            console.write(
                "Live points control the resolution of nested contours."
            )
            console.write(f"  Minimum required: {min_live}")
            console.write(f"  Recommended default: {recommended_live}")
            entry = console.ask(
                f"Number of live points [{recommended_live}]: "
            ).strip()
            if not entry:
                live_points = recommended_live
            else:
                try:
                    live_points = int(entry)
                except ValueError:
                    console.write(
                        "Live points must be an integer.",
                        error=True,
                    )
                    continue
                if live_points < min_live:
                    console.write(
                        "Live points below the minimum risk under-sampling.",
                        error=True,
                    )
                    continue

            console.write("")
            console.write(
                "Maximum iterations set the hard cap on nested replacements."
            )
            console.write(f"  Recommended default: {recommended_max_iter}")
            entry = console.ask(
                f"Maximum iterations [{recommended_max_iter}]: "
            ).strip()
            if not entry:
                max_iter = recommended_max_iter
            else:
                try:
                    max_iter = int(entry)
                except ValueError:
                    console.write(
                        "Maximum iterations must be an integer.",
                        error=True,
                    )
                    continue
                if max_iter <= live_points:
                    console.write(
                        "Iterations must exceed the live-point count.",
                        error=True,
                    )
                    continue

            console.write("")
            console.write(
                "Evidence tolerance stops sampling once remaining weight is"
            )
            console.write(
                "  sufficiently small. Smaller values increase convergence"
            )
            console.write("  certainty at the cost of runtime.")
            entry = console.ask(
                f"Evidence tolerance [{recommended_tol:g}]: "
            ).strip()
            if not entry:
                tol = recommended_tol
            else:
                try:
                    tol = float(entry)
                except ValueError:
                    console.write(
                        "Tolerance must be a floating-point number.",
                        error=True,
                    )
                    continue
                if tol <= 0:
                    console.write(
                        "Tolerance must be strictly positive.",
                        error=True,
                    )
                    continue

            console.write("")
            console.write(
                "Enlargement fraction widens proposal clouds around live "
                "points."
            )
            console.write(
                "  Values near 1.0 follow the tightest ellipsoid, larger "
                "values"
            )
            console.write("  trade efficiency for robustness.")
            entry = console.ask(
                f"Enlargement fraction [{recommended_enlargement:g}]: "
            ).strip()
            if not entry:
                enlarge = recommended_enlargement
            else:
                try:
                    enlarge = float(entry)
                except ValueError:
                    console.write(
                        "Enlargement must be a floating-point number.",
                        error=True,
                    )
                    continue
                if enlarge < 1.0:
                    console.write(
                        "Enlargement cannot be below 1.0.",
                        error=True,
                    )
                    continue

            console.write("")
            console.write("Nested sampler plan summary:")
            console.write(f"  Live points: {live_points}")
            console.write(f"  Max iterations: {max_iter}")
            console.write(f"  Evidence tolerance: {tol:g}")
            console.write(f"  Enlargement fraction: {enlarge:g}")
            console.write("")
            console.write("How should we proceed?")
            console.write("  1) Accept this nested plan and continue")
            console.write("  2) Revisit the questionnaire from the beginning")
            console.write("  B) Back to the nested defaults summary")
            console.write("  C) Cancel sampler configuration")
            console.write("")

            confirm = _wait_for_menu_choice(
                {
                    "1",
                    "2",
                    "b",
                    "c",
                    "y",
                    "yes",
                    "r",
                    "restart",
                    "n",
                    "no",
                    "cancel",
                    "back",
                },
                prompt="Press a key to accept, restart or exit: ",
            )
            current_plan = {
                "engine_kind": "nested",
                "n_live_points": live_points,
                "max_iterations": max_iter,
                "evidence_tolerance": tol,
                "enlargement_fraction": enlarge,
            }
            if confirm in {"", "1", "y", "yes"}:
                return current_plan
            if confirm in {"c", "cancel"}:
                return None
            if confirm in {"b", "back"}:
                return "back"
            if confirm in {"2", "r", "restart", "n", "no"}:
                console.write("")
                console.write(
                    "Restarting the nested questionnaire from step one."
                )
                continue
            console.write("Please choose 1, 2, B or C.", error=True)

    while True:
        console.write("")
        console.write("Nested sampler defaults summary:")
        console.write(f"  ΛCDM active parameters: {lcdm_active}")
        console.write(
            f"  {alt_model_plugin.MODEL_NAME} active parameters: {alt_active}"
        )
        console.write(f"  Minimum live points: {min_live}")
        console.write(f"  Recommended live points: {recommended_live}")
        console.write(f"  Recommended max iterations: {recommended_max_iter}")
        console.write(f"  Recommended evidence tolerance: {recommended_tol:g}")
        console.write(
            f"  Recommended enlargement fraction: {recommended_enlargement:g}"
        )
        console.write("")
        console.write("1) Run with default settings")
        console.write("2) Change settings")
        console.write("C) Cancel configuration")
        console.write("")

        choice = console.ask("Write the number of choice: ").strip().lower()
        if choice in {"", "1"}:
            return {
                "engine_kind": "nested",
                "n_live_points": recommended_live,
                "max_iterations": recommended_max_iter,
                "evidence_tolerance": recommended_tol,
                "enlargement_fraction": recommended_enlargement,
            }
        if choice == "2":
            custom_plan = _collect_custom_plan()
            if custom_plan is None:
                return None
            if custom_plan == "back":
                console.write("")
                console.write("Returning to nested defaults summary.")
                continue
            return custom_plan
        if choice in {"c", "cancel"}:
            return None
        console.write("Please choose 1, 2 or C.", error=True)


def prompt_sampling_configuration(
    engine_module,
    lcdm_plugin,
    alt_model_plugin,
    sne_data_df,
    bao_data_df,
    cmb_data_df,
):
    """Return user-selected sampler settings or ``None`` if cancelled."""

    engine_kind = str(getattr(engine_module, "ENGINE_KIND", "mcmc")).lower()
    if engine_kind == "nested":
        return _prompt_nested_configuration(
            engine_module,
            lcdm_plugin,
            alt_model_plugin,
        )

    fit_fn, _ = _resolve_fit_function(engine_module)
    fit_sig = inspect.signature(fit_fn)
    try:
        default_steps = int(fit_sig.parameters["n_steps"].default)
    except (KeyError, ValueError, TypeError):
        default_steps = 200
    try:
        default_walkers = int(fit_sig.parameters["n_walkers"].default)
    except (KeyError, ValueError, TypeError):
        default_walkers = 32

    lcdm_active = _count_active_parameters(
        lcdm_plugin,
        engine_module=engine_module,
    )
    alt_active = _count_active_parameters(
        alt_model_plugin,
        engine_module=engine_module,
    )
    minimum_walkers = max(2 * max(lcdm_active, alt_active), 2)

    cpu_total = os.cpu_count() or 0
    default_pool = cpu_total if cpu_total > 0 else minimum_walkers
    default_pool = max(default_pool, 1)

    cpu_display = cpu_total if cpu_total and cpu_total > 0 else "unknown"
    recommended_steps = default_steps
    recommended_burn = max(100, recommended_steps // 5)
    recommended_walkers = max(default_walkers, minimum_walkers)
    recommended_pool = default_pool

    def _format_pool(value: int | None) -> str:
        return str(value) if value is not None else "auto"

    def _collect_custom_plan() -> dict[str, int | None] | str | None:
        """Gather a customised sampler configuration from the operator."""

        while True:
            console.write("")
            console.write(
                "Production steps control the total sampler iterations."
            )
            console.write(f"  Recommended default: {recommended_steps}")
            entry = console.ask(
                f"Production steps [{recommended_steps}]: "
            ).strip()
            if not entry:
                n_steps = recommended_steps
            else:
                try:
                    n_steps = int(entry)
                except ValueError:
                    console.write("Steps must be an integer.", error=True)
                    continue
                if n_steps <= 0:
                    console.write("Steps must be positive.", error=True)
                    continue

            default_burn = max(100, n_steps // 5)
            quick_burn = max(1, n_steps // 5)
            console.write("")
            console.write(
                "Burn-in steps discard the early samples so the chain can "
                "stabilise."
            )
            console.write(f"  Recommended warm-up: {default_burn}")
            console.write(
                f"  A shorter option such as {quick_burn} trades certainty "
                "for speed."
            )
            entry = console.ask(
                f"Burn-in steps [{default_burn}]: "
            ).strip()
            if not entry:
                burn_in = default_burn
            else:
                try:
                    burn_in = int(entry)
                except ValueError:
                    console.write("Burn-in must be an integer.", error=True)
                    continue
                if burn_in <= 0:
                    console.write("Burn-in must be positive.", error=True)
                    continue

            walker_default = max(default_walkers, minimum_walkers)
            console.write("")
            console.write(
                "Walkers sample the posterior in parallel; more walkers "
                "increase convergence confidence."
            )
            console.write(f"  Required minimum: {minimum_walkers}")
            console.write(f"  Recommended default: {walker_default}")
            entry = console.ask(
                f"Number of walkers [{walker_default}]: "
            ).strip()
            if not entry:
                n_walkers = walker_default
            else:
                try:
                    n_walkers = int(entry)
                except ValueError:
                    console.write(
                        "Walker count must be an integer.",
                        error=True,
                    )
                    continue
                if n_walkers < minimum_walkers:
                    console.write(
                        "Walker count is below the required minimum; the "
                        "ensemble would stagnate.",
                        error=True,
                    )
                    continue

            console.write("")
            console.write(
                "Worker pools accelerate sampling by spreading walkers across "
                "processes."
            )
            console.write(
                "  Recommended pool size: "
                f"{recommended_pool} (detected CPUs: {cpu_display})"
            )
            console.write("  Enter 0 to disable multiprocessing entirely.")
            entry = console.ask(
                f"Pool workers [{_format_pool(recommended_pool)}]: "
            ).strip().lower()
            if not entry:
                pool_size: int | None = recommended_pool
            elif entry in {"0", "none", "disable"}:
                pool_size = None
            else:
                try:
                    pool_value = int(entry)
                except ValueError:
                    console.write("Pool size must be an integer.", error=True)
                    continue
                if pool_value < 0:
                    console.write("Pool size cannot be negative.", error=True)
                    continue
                pool_size = pool_value if pool_value > 0 else None

            effective_pool = pool_size if pool_size is not None else None
            adjusted_walkers = max(
                n_walkers,
                minimum_walkers,
                effective_pool or 0,
            )
            if adjusted_walkers != n_walkers:
                console.write("")
                console.write(
                    f"Walker count increased to {adjusted_walkers} "
                    "to match the worker pool."
                )
                n_walkers = adjusted_walkers

            console.write("")
            console.write("Sampling plan summary:")
            console.write(f"  Production steps: {n_steps}")
            console.write(f"  Burn-in steps: {burn_in}")
            console.write(f"  Walkers: {n_walkers}")
            console.write(f"  Pool workers: {_format_pool(effective_pool)}")
            console.write("")
            console.write("How should we proceed?")
            console.write("  1) Accept this sampler plan and continue")
            console.write(
                "  2) Revisit the questionnaire from the beginning"
            )
            console.write("  B) Back to the sampler defaults summary")
            console.write("  C) Cancel sampler configuration")
            console.write("")

            confirm = _wait_for_menu_choice(
                {
                    "1",
                    "2",
                    "b",
                    "c",
                    "y",
                    "yes",
                    "r",
                    "restart",
                    "n",
                    "no",
                    "cancel",
                    "back",
                },
                prompt="Press a key to accept, restart or exit: ",
            )
            current_plan = {
                "engine_kind": "mcmc",
                "n_steps": n_steps,
                "burn_in_steps": burn_in,
                "n_walkers": n_walkers,
                "pool_size": effective_pool,
            }
            if confirm in {"", "1", "y", "yes"}:
                return current_plan
            if confirm in {"c", "cancel"}:
                return None
            if confirm in {"b", "back"}:
                return "back"
            if confirm in {"2", "r", "restart", "n", "no"}:
                console.write("")
                console.write(
                    "Restarting the sampler questionnaire from step one."
                )
                continue
            console.write("Please choose 1, 2, B or C.", error=True)

    while True:
        console.write("")
        console.write("Sampler defaults summary:")
        console.write(f"  ΛCDM active parameters: {lcdm_active}")
        console.write(
            f"  {alt_model_plugin.MODEL_NAME} active parameters: {alt_active}"
        )
        console.write(
            f"  Minimum walkers per emcee rule: {minimum_walkers}"
        )
        console.write(
            f"  Recommended production steps: {recommended_steps}"
        )
        console.write(f"  Recommended burn-in steps: {recommended_burn}")
        console.write(f"  Recommended walkers: {recommended_walkers}")
        console.write(
            f"  Recommended pool workers: {_format_pool(recommended_pool)}"
        )
        console.write("")
        console.write("1) Run with default settings")
        console.write("2) Change settings")
        console.write("C) Cancel configuration")
        console.write("")

        choice = console.ask("Write the number of choice: ").strip().lower()
        if choice in {"", "1"}:
            return {
                "engine_kind": "mcmc",
                "n_steps": recommended_steps,
                "burn_in_steps": recommended_burn,
                "n_walkers": recommended_walkers,
                "pool_size": recommended_pool,
            }
        if choice == "2":
            custom_plan = _collect_custom_plan()
            if custom_plan is None:
                return None
            if custom_plan == "back":
                console.write("")
                console.write("Returning to default sampler summary.")
                continue
            return custom_plan
        if choice in {"c", "cancel"}:
            return None
        console.write("Please choose 1, 2 or C.", error=True)


def cleanup_cache(base_dir):
    """Remove temporary files left behind by previous runs."""

    # Python leaves ``__pycache__`` folders behind when modules are imported.
    # Removing them ensures that stale bytecode doesn't interfere with
    # subsequent executions, especially when models are re-generated.
    logger = log_mod.get_logger()
    logger.info("--- Cleaning up cache files ---")
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                logger.info(f"Removed cache directory: {pycache_path}")
            except OSError as e:
                logger.error(
                    f"Error removing cache directory {pycache_path}: {e}"
                )
    cache_dir = os.path.join(base_dir, "models", "cache")
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith("cache_") and fname.endswith(".yml"):
                path = os.path.join(cache_dir, fname)
                try:
                    os.remove(path)
                    logger.info(f"Removed cache file: {path}")
                except OSError as e:
                    logger.error(f"Error removing cache file {path}: {e}")


def extract_cosmological_param_vector(
    fit_results,
    model_plugin,
    *,
    logger=None,
):
    """Return fitted cosmological parameters ordered for ``model_plugin``.

    The helper hides the boilerplate required to guard against partial engine
    failures.  Engines return ``success=False`` and omit
    ``fitted_cosmological_params`` when the sampler cannot initialise (for
    example if ``emcee`` rejects the walker ensemble).  Higher level workflow
    code can call this function and receive either a list of parameter values
    in the plugin's declared order or ``None`` when the data is unavailable.
    ``logger`` is optional so unit tests can remain silent while the runtime
    continues to emit descriptive warnings for users.
    """

    if not isinstance(fit_results, Mapping):
        return None
    if not fit_results.get("success"):
        return None
    params = fit_results.get("fitted_cosmological_params")
    if not isinstance(params, Mapping):
        if logger is not None:
            model_name = getattr(model_plugin, "MODEL_NAME", "model")
            logger.warning(
                "%s fit results did not expose 'fitted_cosmological_params'; "
                "skipping dependent analyses.",
                model_name,
            )
        return None

    names = list(getattr(model_plugin, "PARAMETER_NAMES", []))
    if not names:
        return list(params.values())

    missing = [name for name in names if name not in params]
    if missing:
        if logger is not None:
            model_name = getattr(model_plugin, "MODEL_NAME", "model")
            joined = ", ".join(missing)
            logger.warning(
                "%s fit is missing values for %s; skipping dependent "
                "analyses.",
                model_name,
                joined,
            )
        return None

    return [params[name] for name in names]


def _sanity_check_numpy_scipy(log):
    """Run a tiny NumPy/SciPy calculation to verify binary compatibility.

    Mismatched CPU features or corrupted installations can cause crashes when
    heavy computations begin. A trivial dot product and determinant expose such
    issues early so the program can advise reinstalling with suitable wheels.
    """

    try:
        import numpy as _np
        from scipy import linalg as _linalg

        _np.dot(_np.ones(1), _np.ones(1))
        _linalg.det([[1.0]])
    except Exception as exc:  # pragma: no cover - depends on local install
        log.error(
            "Basic NumPy/SciPy check failed. This often points to "
            "CPU feature mismatches or a corrupted install. "
            "Reinstall NumPy and SciPy with wheels built for your machine.",
            exc_info=exc,
        )
        raise


def _summarise_sampling_plan(plan: Mapping[str, Any] | None) -> str:
    """Return a human-readable description of the sampler plan."""

    if not plan:
        return "No sampler configured"
    engine_kind = str(plan.get("engine_kind", "mcmc")).lower()
    if engine_kind == "nested":
        return (
            "Nested sampler: "
            f"live {plan.get('n_live_points', '??')}, "
            f"max iter {plan.get('max_iterations', '??')}"
        )
    return (
        "MCMC sampler: "
        f"steps {plan.get('n_steps', '??')}, "
        f"burn-in {plan.get('burn_in_steps', '??')}, "
        f"walkers {plan.get('n_walkers', '??')}"
    )


def _wait_for_menu_choice(valid_options: Iterable[str], prompt: str) -> str:
    """Return a single-key menu selection while leaving the menu on screen."""

    return console.read_keypress(
        {option.lower() for option in valid_options},
        prompt=prompt,
    )


def _display_dashboard_menu(state: DashboardState) -> str:
    """Render the dashboard overview and return the chosen section key."""

    console.write("")
    console.write("Copernican Dashboard")
    console.write("---------------------")
    console.write(
        f"Seed: {state.seed if state.seed is not None else 'not set'}"
    )
    console.write(
        "Model: "
        + (
            getattr(state.alt_model_plugin, "MODEL_NAME", "not selected")
            if state.alt_model_plugin
            else "not selected"
        )
    )
    engine_label = (
        getattr(state.engine_module, "ENGINE_LABEL", None)
        or getattr(state.engine_module, "__name__", "not selected")
        if state.engine_module
        else "not selected"
    )
    console.write(f"Engine: {engine_label}")
    console.write(
        f"Datasets: SNe on | BAO {'on' if state.use_bao else 'off'} | "
        f"CMB {'on' if state.use_cmb else 'off'}"
    )
    console.write(
        f"Sampler: {_summarise_sampling_plan(state.sampling_plan)}"
    )
    if state.last_output_dir:
        console.write(f"Last output: {state.last_output_dir}")
    console.write("")
    console.write("Sections:")
    console.write("  1) Configuration (seed and sampler)")
    console.write("  2) Engine and model selection")
    console.write("  3) Dataset toggles")
    console.write("  4) Run control")
    console.write("  5) Outputs")
    console.write("  6) Settings")
    console.write("  C) Close the Copernican Suite")
    console.write("")
    return _wait_for_menu_choice(
        {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "c",
            "close",
            "q",
            "quit",
            "exit",
            "config",
            "configuration",
            "engine",
            "model",
            "dataset",
            "datasets",
            "run",
            "output",
            "outputs",
            "settings",
        },
        prompt="Press a key to open a section (no Enter needed): ",
    )
    if state.last_output_dir:
        console.write(f"Last output: {state.last_output_dir}")
    console.write("")
    console.write("Sections:")
    console.write("  1) Configuration (seed and sampler)")
    console.write("  2) Engine and model selection")
    console.write("  3) Dataset toggles")
    console.write("  4) Run control")
    console.write("  5) Outputs")
    console.write("  6) Settings")
    console.write("  C) Close the Copernican Suite")
    console.write("")
    return _wait_for_menu_choice(
        {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "c",
            "close",
            "q",
            "quit",
            "exit",
            "config",
            "configuration",
            "engine",
            "model",
            "dataset",
            "datasets",
            "run",
            "output",
            "outputs",
            "settings",
        },
        prompt="Press a key to open a section (no Enter needed): ",
    )
    console.write(f"Engine: {engine_label}")
    console.write(f"Datasets: SNe on | BAO {'on' if state.use_bao else 'off'} | "
                  f"CMB {'on' if state.use_cmb else 'off'}")
    console.write(f"Sampler: {_summarise_sampling_plan(state.sampling_plan)}")
    if state.last_output_dir:
        console.write(f"Last output: {state.last_output_dir}")
    console.write("")
    console.write("Sections:")
    console.write("  1) Configuration (seed and sampler)")
    console.write("  2) Engine and model selection")
    console.write("  3) Dataset toggles")
    console.write("  4) Run control")
    console.write("  5) Outputs")
    console.write("  6) Settings")
    console.write("  C) Close the Copernican Suite")
    console.write("")
    return console.ask("Select a section: ").strip().lower()


def prompt_post_run_action() -> bool:
    """Ask whether to launch another evaluation from the dashboard."""

    console.write("What next?")
    console.write("  1) Run another evaluation")
    console.write("  C) Return to the dashboard")
    console.write("")

    while True:
        choice = _wait_for_menu_choice(
            {"1", "c", "cancel", ""},
            prompt="Press 1 to run again or C to return: ",
        )
        if choice in {"", "1"}:
            return True
        if choice in {"c", "cancel"}:
            return False
        console.write("Please choose 1 or C.", error=True)


def _configuration_section(state: DashboardState, lcdm_plugin) -> None:
    """Handle seed and sampler configuration from the dashboard."""

    while True:
        console.write("")
        console.write("Configuration")
        console.write("--------------")
        seed_label = (
            "Current seed: "
            f"{state.seed if state.seed is not None else 'not set'}"
        )
        console.write(seed_label)
        console.write(
            f"Sampler plan: {_summarise_sampling_plan(state.sampling_plan)}"
        )
        console.write("")
        console.write("  1) Set or change the random seed")
        console.write("  2) Configure sampler settings")
        console.write("  3) Clear sampler configuration")
        console.write("  B) Back to dashboard")
        console.write("")

        choice = _wait_for_menu_choice(
            {"1", "2", "3", "b", "back"},
            prompt="Press a key to configure or return: ",
        )
        if choice in {"", "b", "back"}:
            return
        if choice == "1":
            state.seed = select_seed()
            continue
        if choice == "3":
            state.sampling_plan = None
            console.write("Sampler configuration cleared.")
            continue
        if choice == "2":
            if not state.engine_module or not state.alt_model_plugin:
                console.write(
                    "Select both an engine and an alternative model before "
                    "configuring the sampler.",
                    error=True,
                )
                continue
            plan = prompt_sampling_configuration(
                state.engine_module,
                lcdm_plugin,
                state.alt_model_plugin,
                None,
                None,
                None,
            )
            if plan is None:
                console.write("Sampler configuration cancelled.")
                continue
            state.sampling_plan = plan
            console.write("Sampler configuration saved.")
            continue
        console.write("Please choose a listed option.", error=True)


def _engine_model_section(state: DashboardState, script_dir: str) -> None:
    """Manage model and engine selection without leaving the dashboard."""

    while True:
        console.write("")
        console.write("Engine and Model Selection")
        console.write("---------------------------")
        console.write(
            "Model: "
            + (
                getattr(state.alt_model_plugin, "MODEL_NAME", "not selected")
                if state.alt_model_plugin
                else "not selected"
            )
        )
        engine_label = (
            getattr(state.engine_module, "ENGINE_LABEL", None)
            or getattr(state.engine_module, "__name__", "not selected")
            if state.engine_module
            else "not selected"
        )
        console.write(f"Engine: {engine_label}")
        console.write("")
        console.write("  1) Choose cosmological model")
        console.write("  2) Choose computation engine")
        console.write("  3) Clear selections")
        console.write("  B) Back to dashboard")
        console.write("")

        choice = _wait_for_menu_choice(
            {"1", "2", "3", "b", "back"},
            prompt="Press a key to select or return: ",
        )
        if choice in {"", "b", "back"}:
            return
        if choice == "3":
            state.alt_model_plugin = None
            state.alt_model_parsed = None
            state.selected_model = ""
            state.engine_module = None
            console.write("Model and engine selections cleared.")
            continue
        if choice not in {"1", "2"}:
            console.write(
                "Please choose one of the listed options.",
                error=True,
            )
            continue

        if choice == "1":
            models_dir = os.path.join(script_dir, "models")
            model_files = sorted(
                [
                    f
                    for f in os.listdir(models_dir)
                    if f.startswith("cosmo_model_") and f.endswith(".yml")
                ]
            )
            selected_model = select_from_list(
                model_files, "Select cosmological model"
            )
            if not selected_model:
                console.write("Model selection cancelled.")
                continue
            yaml_path = os.path.join(models_dir, selected_model)
            cache_dir = os.path.join(models_dir, "cache")
            while True:
                try:
                    cache_path = model_spec_validator.validate_and_cache_model(
                        yaml_path, cache_dir
                    )
                    func_dict, parsed = model_coder.generate_callables(
                        cache_path
                    )
                    plugin = engine_plugin_validation.build_plugin(
                        parsed,
                        func_dict,
                    )
                    plugin.MODEL_FILENAME = os.path.basename(yaml_path)
                    state.alt_model_plugin = plugin
                    state.alt_model_parsed = parsed
                    state.selected_model = selected_model
                    console.write(
                        "Loaded YAML model: "
                        f"{parsed.get('model_name', selected_model)}"
                    )
                    break
                except PluginValidationError as exc:
                    reasons = _normalise_failure_reasons(str(exc))
                    if _prompt_configuration_retry(reasons):
                        continue
                    break
                except Exception as exc:
                    # pragma: no cover - defensive log path
                    reasons = _normalise_failure_reasons(str(exc))
                    if _prompt_configuration_retry(reasons):
                        continue
                    break
            continue

        engines_dir = os.path.join(script_dir, "engines")
        engine_files = sorted(
            [
                f
                for f in os.listdir(engines_dir)
                if f.startswith("cosmo_engine_") and f.endswith(".py")
            ]
        )
        engine_choice = select_from_list(
            engine_files, "Select computation engine"
        )
        if not engine_choice:
            console.write("Engine selection cancelled.")
            continue
        while True:
            try:
                engine_module = importlib.import_module(
                    f"engines.{engine_choice[:-3]}"
                )
                state.engine_module = engine_module
                console.write(
                    "Selected engine: "
                    f"{getattr(engine_module, 'ENGINE_LABEL', engine_choice)}"
                )
                break
            except Exception as exc:  # pragma: no cover - runtime import guard
                reasons = _normalise_failure_reasons(str(exc))
                if _prompt_configuration_retry(reasons):
                    continue
                break


def _dataset_toggle_section(state: DashboardState) -> None:
    """Allow operators to enable or disable optional datasets."""

    while True:
        console.write("")
        console.write("Dataset Toggles")
        console.write("----------------")
        console.write("SNe Ia: on (required)")
        console.write(f"BAO: {'on' if state.use_bao else 'off'}")
        console.write(f"CMB: {'on' if state.use_cmb else 'off'}")
        console.write("")
        console.write("  1) Toggle BAO usage")
        console.write("  2) Toggle CMB usage")
        console.write("  3) Restore default dataset mix")
        console.write("  B) Back to dashboard")
        console.write("")

        choice = _wait_for_menu_choice(
            {"1", "2", "3", "b", "back"},
            prompt="Press a key to toggle or return: ",
        )
        if choice in {"", "b", "back"}:
            return
        if choice == "1":
            state.use_bao = not state.use_bao
            console.write(
                f"BAO dataset {'enabled' if state.use_bao else 'disabled'}."
            )
            continue
        if choice == "2":
            state.use_cmb = not state.use_cmb
            console.write(
                f"CMB dataset {'enabled' if state.use_cmb else 'disabled'}."
            )
            continue
        if choice == "3":
            state.use_bao = True
            state.use_cmb = True
            console.write("Dataset selection reset to defaults.")
            continue
        console.write("Please choose a listed option.", error=True)


def _settings_section(state: DashboardState, opts: RuntimeOptions) -> None:
    """Expose miscellaneous runtime settings from the dashboard."""

    while True:
        console.write("")
        console.write("Settings")
        console.write("--------")
        console.write(
            f"Progress display: {'on' if state.display_progress else 'off'}"
        )
        console.write(
            f"Strict warnings: {'on' if opts.strict_warnings else 'off'}"
        )
        console.write("")
        console.write("  1) Toggle progress display")
        console.write("  B) Back to dashboard")
        console.write("")

        choice = _wait_for_menu_choice(
            {"1", "b", "back"},
            prompt="Press a key to adjust or return: ",
        )
        if choice in {"", "b", "back"}:
            return
        if choice == "1":
            state.display_progress = not state.display_progress
            console.write(
                "Progress display "
                f"{'enabled' if state.display_progress else 'disabled'}."
            )
            continue
        console.write("Please choose a listed option.", error=True)


def _outputs_section(state: DashboardState) -> None:
    """Summarise the latest run outputs for quick navigation."""

    console.write("")
    console.write("Outputs")
    console.write("-------")
    if not state.last_output_dir:
        console.write("No runs have been completed in this session.")
        console.write("")
        return
    console.write(f"Most recent output directory: {state.last_output_dir}")
    if state.last_log_file:
        console.write(f"Most recent log file: {state.last_log_file}")
    if state.last_run_started and state.last_run_finished:
        console.write(
            "Run window: "
            f"{state.last_run_started.strftime('%Y-%m-%d %H:%M:%S')} UTC "
            f"to {state.last_run_finished.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    console.write("")
    console.ask("Press Enter to return to the dashboard.")


def _run_dashboard_evaluation(
    state: DashboardState,
    lcdm_plugin,
    lcdm_parsed,
    opts: RuntimeOptions,
    script_dir: str,
    output_base_dir: str,
    program_logger,
) -> None:
    """Execute a single evaluation using the dashboard configuration."""

    alt_model_plugin = state.alt_model_plugin
    if alt_model_plugin is None or state.engine_module is None:
        console.write(
            "Select both a model and an engine before launching a run.",
            error=True,
        )
        return

    if state.seed is None:
        state.seed = select_seed()

    run_start_ts = utils.get_timestamp()
    output_dir = os.path.join(
        output_base_dir, f"copernican-run_{run_start_ts}"
    )
    utils.ensure_dir_exists(output_dir)
    program_logger.info(
        "Run %s initialised with output directory %s",
        run_start_ts,
        output_dir,
    )
    log_file = log_mod.setup_logging(log_dir=output_dir, base_dir=script_dir)
    console.write("")

    global CURRENT_LOG_FILE, logger
    CURRENT_LOG_FILE = log_file
    logger = log_mod.get_logger()
    error_handler.configure_warnings(strict=opts.strict_warnings)
    if opts.strict_warnings:
        logger.info(
            "Strict warnings mode enabled; treating warnings as errors"
        )
    else:
        logger.info("Warnings will be logged but not treated as errors")

    log_mod.log_environment_info()
    try:
        _sanity_check_numpy_scipy(logger)
    except Exception:
        _delete_log_file(log_file)
        _remove_run_dir(output_dir)
        cleanup_cache(script_dir)
        console.write("")
        return

    run_start_dt = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Using RNG seed %s", utils.get_random_seed())
    logger.info("")
    logger.info(
        "Using standard CPU (SciPy) computational backend with "
        "multiprocessing."
    )
    logger.info(f"Running from base directory: {script_dir}")
    logger.info(f"All outputs will be saved to: {output_dir}")

    def _abort_run() -> None:
        _delete_log_file(log_file)
        _remove_run_dir(output_dir)
        cleanup_cache(script_dir)
        console.write("")

    sne_data_df = dataset_registry.load_sne_data()
    if sne_data_df is None:
        _abort_run()
        return

    bao_data_df = dataset_registry.load_bao_data() if state.use_bao else None
    if state.use_bao and bao_data_df is None:
        _abort_run()
        return

    cmb_data_df = dataset_registry.load_cmb_data() if state.use_cmb else None
    if state.use_cmb and cmb_data_df is None:
        _abort_run()
        return
    dataset_info = []
    for df in (sne_data_df, bao_data_df, cmb_data_df):
        if df is None:
            continue
        ds_id = df.attrs.get("dataset_id")
        data_dir = df.attrs.get("data_path")
        if not ds_id or not data_dir:
            continue
        hashes = df.attrs.get("file_hashes", {})
        dataset_info.append(
            {
                "id": ds_id,
                "name": df.attrs.get("dataset_name", ds_id),
                "version": df.attrs.get("dataset_version", "unknown"),
                "path": data_dir,
                "hashes": hashes,
                "independence": df.attrs.get(
                    "independence_assumptions",
                    [],
                ),
                "condition_number": df.attrs.get(
                    "covariance_condition_number"
                ),
            }
        )

    sampling_plan = state.sampling_plan
    if sampling_plan is None:
        sampling_plan = prompt_sampling_configuration(
            state.engine_module,
            lcdm_plugin,
            alt_model_plugin,
            sne_data_df,
            bao_data_df,
            cmb_data_df,
        )
        if sampling_plan is None:
            logger.info("User cancelled sampling configuration; aborting run.")
            _abort_run()
            return
        state.sampling_plan = sampling_plan

    plan_kind = sampling_plan.get("engine_kind", "mcmc").lower()
    display_progress = bool(
        sampling_plan.get("display_progress", state.display_progress)
    )

    if plan_kind == "nested":
        sampling_live = int(sampling_plan["n_live_points"])
        sampling_max_iter = int(sampling_plan["max_iterations"])
        sampling_tol = float(sampling_plan["evidence_tolerance"])
        sampling_enlarge = float(sampling_plan["enlargement_fraction"])
        logger.info(
            "Nested sampler configuration: "
            "live=%d, max_iter=%d, tol=%g, enlarge=%.2f",
            sampling_live,
            sampling_max_iter,
            sampling_tol,
            sampling_enlarge,
        )
        console.write(
            "Configured nested sampler: "
            f"live points {sampling_live}, "
            f"max iterations {sampling_max_iter}."
        )
        console.write(
            f"Evidence tolerance {sampling_tol:g}, "
            f"enlargement fraction {sampling_enlarge:g}."
        )
    else:
        sampling_steps = int(sampling_plan["n_steps"])
        sampling_burn_in = int(sampling_plan["burn_in_steps"])
        sampling_walkers = int(sampling_plan["n_walkers"])
        sampling_pool = sampling_plan["pool_size"]

        pool_label = sampling_pool if sampling_pool is not None else "auto"
        logger.info(
            (
                "Sampler configuration: steps=%d, burn-in=%d, "
                "walkers=%d, pool=%s"
            ),
            sampling_steps,
            sampling_burn_in,
            sampling_walkers,
            pool_label,
        )
        console.write(
            "Configured sampler: "
            f"steps {sampling_steps}, burn-in {sampling_burn_in}."
        )
        console.write(
            f"Walker ensemble {sampling_walkers}, pool {pool_label}."
        )

    alt_parsed = state.alt_model_parsed or {}
    manifest = run_manifest.build_manifest(
        models=[
            (lcdm_plugin, lcdm_parsed.get("version", "unknown")),
            (alt_model_plugin, alt_parsed.get("version", "unknown")),
        ],
        engine_module=state.engine_module,
        datasets=dataset_info,
    )
    run_manifest.save_manifest(manifest, output_dir)

    lcdm_name = getattr(lcdm_plugin, "MODEL_NAME", "ΛCDM")
    alt_name = getattr(alt_model_plugin, "MODEL_NAME", "Alternative")
    same_name = lcdm_name.casefold() == alt_name.casefold()
    lcdm_file = getattr(lcdm_plugin, "MODEL_FILENAME", "")
    alt_file = getattr(alt_model_plugin, "MODEL_FILENAME", "")
    reuse_alt = same_name and lcdm_file == alt_file and (
        type(lcdm_plugin) is type(alt_model_plugin)
    )
    engine_label = getattr(
        state.engine_module,
        "ENGINE_LABEL",
        getattr(state.engine_module, "__name__", "Engine"),
    )
    logger.info("\n--- Sampling with %s ---\n", engine_label)
    console.write("")
    try:
        fit_fn, _ = _resolve_fit_function(state.engine_module)
    except AttributeError:
        logger.error(
            (
                "Selected engine %s lacks a cosmology fitting entry point; "
                "aborting run."
            ),
            getattr(state.engine_module, "__name__", "unknown"),
        )
        _abort_run()
        return
    console.write("ΛCDM reference chain")
    if plan_kind == "nested":
        console.write(f"  Live points: {sampling_live}")
        console.write(f"  Max iterations: {sampling_max_iter}")
        console.write(f"  Evidence tolerance: {sampling_tol:g}")
        console.write(f"  Enlargement fraction: {sampling_enlarge:g}")
        console.write("  Starting ΛCDM sampler...")
        console.write("")
        lcdm_fit_results = fit_fn(
            sne_data_df,
            lcdm_plugin,
            bao_data_df=bao_data_df,
            cmb_data_df=cmb_data_df,
            n_live_points=sampling_live,
            max_iterations=sampling_max_iter,
            evidence_tolerance=sampling_tol,
            enlargement_fraction=sampling_enlarge,
            display_progress=display_progress,
        )
    else:
        console.write(f"  Burn-in steps: {sampling_burn_in}")
        console.write(f"  Production steps: {sampling_steps}")
        console.write(f"  Walkers: {sampling_walkers}")
        console.write(f"  Worker pool: {pool_label}")
        console.write("  Starting ΛCDM sampler...")
        console.write("")
        lcdm_fit_results = fit_fn(
            sne_data_df,
            lcdm_plugin,
            bao_data_df=bao_data_df,
            cmb_data_df=cmb_data_df,
            n_walkers=sampling_walkers,
            n_steps=sampling_steps,
            pool_size=sampling_pool,
            burn_in_steps=sampling_burn_in,
            display_progress=display_progress,
        )
    if reuse_alt:
        logger.info(
            "Alternative model matches ΛCDM; reusing SNe chain from %s.",
            engine_label,
        )
        console.write(
            "Alternative model matches ΛCDM; reusing the completed ΛCDM "
            "chain for further analysis."
        )
        console.write("")
        alt_model_fit_results = copy.deepcopy(lcdm_fit_results)
    else:
        console.write("")
        console.write(f"Alternative model: {alt_model_plugin.MODEL_NAME}")
        if plan_kind == "nested":
            console.write(f"  Live points: {sampling_live}")
            console.write(f"  Max iterations: {sampling_max_iter}")
            console.write(f"  Evidence tolerance: {sampling_tol:g}")
            console.write(f"  Enlargement fraction: {sampling_enlarge:g}")
            console.write("  Starting alternative sampler...")
            console.write("")
            alt_model_fit_results = fit_fn(
                sne_data_df,
                alt_model_plugin,
                bao_data_df=bao_data_df,
                cmb_data_df=cmb_data_df,
                n_live_points=sampling_live,
                max_iterations=sampling_max_iter,
                evidence_tolerance=sampling_tol,
                enlargement_fraction=sampling_enlarge,
                display_progress=display_progress,
            )
        else:
            console.write(f"  Burn-in steps: {sampling_burn_in}")
            console.write(f"  Production steps: {sampling_steps}")
            console.write(f"  Walkers: {sampling_walkers}")
            console.write(f"  Worker pool: {pool_label}")
        console.write("  Starting alternative sampler...")
        console.write("")
        alt_model_fit_results = fit_fn(
            sne_data_df,
            alt_model_plugin,
            bao_data_df=bao_data_df,
            cmb_data_df=cmb_data_df,
            n_walkers=sampling_walkers,
            n_steps=sampling_steps,
            pool_size=sampling_pool,
            burn_in_steps=sampling_burn_in,
            display_progress=display_progress,
        )
    console.write(
        "Completed alternative sampling for "
        f"{alt_model_plugin.MODEL_NAME}."
    )
    console.write("")

    console.write("Sampling complete.")
    console.write("")

    result_writer.save_summary(
        {
            lcdm_plugin.MODEL_NAME: lcdm_fit_results,
            alt_model_plugin.MODEL_NAME: alt_model_fit_results,
        },
        output_dir,
    )

    lcdm_bao_summary: dict[str, Any] = {}
    alt_bao_summary: dict[str, Any] = {}
    if bao_data_df is not None:
        logger.info("\n--- BAO Analysis ---\n")

        def _component_enabled(fit_results, component):
            state_map = (
                fit_results.get("likelihood_state", {}) if fit_results else {}
            )
            metadata = state_map.get("metadata", {})
            components = metadata.get("components", {})
            entry = components.get(component, {})
            enabled_flag = entry.get("metadata", {}).get("enabled")
            if enabled_flag is not None:
                return bool(enabled_flag)
            enabled_components = metadata.get("enabled_components", ())
            return component in enabled_components

        min_z, max_z = (
            bao_data_df["redshift"].min(),
            bao_data_df["redshift"].max(),
        )
        z_plot_smooth = np.geomspace(
            max(min_z * 0.8, 0.01), max_z * 1.2, 100
        )

        def run_bao_analysis(model_plugin, fit_results, z_smooth_arr):
            """Return BAO diagnostics and predictions for ``model_plugin``."""

            summary = {
                "sne_fit_results": fit_results,
                "pred_df": None,
                "rs_Mpc": np.nan,
                "chi2_bao": float(
                    (fit_results or {}).get("chi2_bao", float("inf"))
                ),
                "smooth_predictions": None,
            }

            if not (fit_results and fit_results.get("success")):
                logger.warning(
                    (
                        f"{model_plugin.MODEL_NAME} fit failed; skipping BAO analysis."
                    )
                )
                return summary

            if not _component_enabled(fit_results, "bao"):
                logger.info(
                    (
                        f"{model_plugin.MODEL_NAME} BAO likelihood disabled; skipping predictions."
                    )
                )
                summary["chi2_bao"] = float("inf")
                return summary

            fitted_cosmo_p = extract_cosmological_param_vector(
                fit_results,
                model_plugin,
                logger=logger,
            )
            if fitted_cosmo_p is None:
                logger.warning(
                    (
                        f"{model_plugin.MODEL_NAME} fit does not expose cosmological parameters; skipping BAO analysis."
                    )
                )
                summary["chi2_bao"] = float("inf")
                return summary

            pred_df, rs_Mpc, smooth_preds = (
                state.engine_module.calculate_bao_observables(
                    bao_data_df,
                    model_plugin,
                    fitted_cosmo_p,
                    z_smooth=z_smooth_arr,
                )
            )
            summary.update(
                {
                    "pred_df": pred_df,
                    "rs_Mpc": rs_Mpc,
                    "smooth_predictions": smooth_preds,
                }
            )

            for line in bao_residual_diagnostics(
                bao_data_df,
                pred_df,
                model_name=model_plugin.MODEL_NAME,
            ):
                logger.info(line)

            chi2_bao = summary["chi2_bao"]
            if pred_df is not None and np.isfinite(rs_Mpc):
                if np.isfinite(chi2_bao):
                    logger.info(
                        (
                            f"{model_plugin.MODEL_NAME} BAO: r_s = "
                            f"{rs_Mpc:.2f} Mpc, "
                            f"χ²_BAO = {chi2_bao:.2f}"
                        )
                    )
                else:
                    logger.warning(
                        (
                            f"{model_plugin.MODEL_NAME} BAO predictions available but χ² is non-finite."
                        )
                    )
            else:
                logger.warning(
                    f"{model_plugin.MODEL_NAME} BAO calculation failed or produced invalid r_s."
                )

            return summary

        lcdm_bao_summary = run_bao_analysis(
            lcdm_plugin,
            lcdm_fit_results,
            z_plot_smooth,
        )
        alt_bao_summary = run_bao_analysis(
            alt_model_plugin,
            alt_model_fit_results,
            z_plot_smooth,
        )

        if (
            np.isfinite(lcdm_bao_summary.get("rs_Mpc", np.nan))
            and np.isfinite(alt_bao_summary.get("rs_Mpc", np.nan))
        ):
            delta_rs = alt_bao_summary["rs_Mpc"] - lcdm_bao_summary["rs_Mpc"]
            logger.info(
                (
                    f"{alt_model_plugin.MODEL_NAME} r_s offset relative to "
                    f"{lcdm_plugin.MODEL_NAME}: {delta_rs:+.3f} Mpc"
                )
            )
    else:
        logger.info("BAO analysis skipped; dataset disabled in the dashboard.")

    lcdm_cmb_summary: dict[str, Any] = {}
    alt_cmb_summary: dict[str, Any] = {}
    if cmb_data_df is not None:
        logger.info("\n--- CMB Analysis ---\n")

        def _component_enabled(fit_results, component):
            state_map = (
                fit_results.get("likelihood_state", {}) if fit_results else {}
            )
            metadata = state_map.get("metadata", {})
            components = metadata.get("components", {})
            entry = components.get(component, {})
            enabled_flag = entry.get("metadata", {}).get("enabled")
            if enabled_flag is not None:
                return bool(enabled_flag)
            enabled_components = metadata.get("enabled_components", ())
            return component in enabled_components

        def run_cmb_analysis(model_plugin, fit_results):
            """Return CMB diagnostics and spectra for ``model_plugin``."""

            summary = {
                "chi2_cmb": float(
                    (fit_results or {}).get("chi2_cmb", float("inf"))
                ),
                "theory_spectrum": None,
            }

            if not _component_enabled(fit_results, "cmb"):
                return summary

            if getattr(model_plugin, "valid_for_cmb", True) is False:
                logger.info(
                    (
                        f"{model_plugin.MODEL_NAME} does not support CMB; skipping analysis."
                    )
                )
                summary["chi2_cmb"] = float("inf")
                return summary

            cosmo_params = extract_cosmological_param_vector(
                fit_results,
                model_plugin,
                logger=logger,
            )
            if cosmo_params is None:
                logger.warning(
                    (
                        f"{model_plugin.MODEL_NAME} fit does not provide cosmological parameters; skipping CMB predictions."
                    )
                )
                summary["chi2_cmb"] = float("inf")
                return summary

            try:
                camb_params = model_plugin.get_camb_params(cosmo_params)
            except Exception as exc:
                logger.warning(
                    (
                        f"{model_plugin.MODEL_NAME} failed to build CAMB parameters: {exc}"
                    )
                )
                summary["chi2_cmb"] = float("inf")
                return summary

            components = ["TT"]
            if "Dl_te_obs" in cmb_data_df.columns:
                components.append("TE")
            if "Dl_ee_obs" in cmb_data_df.columns:
                components.append("EE")

            theory = state.engine_module.compute_cmb_spectrum(
                camb_params,
                cmb_data_df["ell"].values,
                spectra=tuple(components),
            )
            summary["theory_spectrum"] = theory

            for line in cmb_residual_diagnostics(
                cmb_data_df,
                theory,
                model_name=model_plugin.MODEL_NAME,
            ):
                logger.info(line)

            chi2_cmb = summary["chi2_cmb"]
            if np.isfinite(chi2_cmb):
                logger.info(
                    f"{model_plugin.MODEL_NAME} CMB χ² = {chi2_cmb:.2f}"
                )
            else:
                logger.info(
                    (
                        f"{model_plugin.MODEL_NAME} CMB likelihood disabled or returned a non-finite χ²."
                    )
                )

            return summary

        lcdm_cmb_summary = run_cmb_analysis(lcdm_plugin, lcdm_fit_results)
        alt_cmb_summary = run_cmb_analysis(
            alt_model_plugin,
            alt_model_fit_results,
        )
    else:
        logger.info("CMB analysis skipped; dataset disabled in the dashboard.")

    logger.info("\n--- Generating Outputs ---\n")
    if lcdm_cmb_summary:
        logger.info(
            f"{lcdm_plugin.MODEL_NAME} CMB χ² = "
            f"{lcdm_cmb_summary.get('chi2_cmb', float('nan')):.2f}"
        )
    if alt_cmb_summary:
        logger.info(
            f"{alt_model_plugin.MODEL_NAME} CMB χ² = "
            f"{alt_cmb_summary.get('chi2_cmb', float('nan')):.2f}"
        )

    run_end_dt = datetime.datetime.now(datetime.timezone.utc)
    end_ts = run_end_dt.strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(
        output_base_dir, f"copernican-run_{end_ts}"
    )
    if output_dir != new_dir:
        try:
            os.rename(output_dir, new_dir)
            output_dir = new_dir
            log_file = os.path.join(output_dir, os.path.basename(log_file))
        except OSError as e_dir:
            logger.error(f"Failed renaming output directory: {e_dir}")
    new_log = os.path.join(output_dir, f"copernican-run_{end_ts}.txt")
    if log_file != new_log:
        try:
            os.rename(log_file, new_log)
            CURRENT_LOG_FILE = new_log
            logger.info(
                f"Log file renamed to {os.path.basename(new_log)}"
            )
            log_file = new_log
        except OSError as e_ren:
            logger.error(f"Failed renaming log file: {e_ren}")

    plotter.plot_hubble_diagram(
        sne_data_df,
        lcdm_fit_results,
        alt_model_fit_results,
        lcdm_plugin,
        alt_model_plugin,
        plot_dir=output_dir,
        timestamp=end_ts,
    )
    if bao_data_df is not None:
        plotter.plot_bao_observables(
            bao_data_df,
            lcdm_bao_summary,
            alt_bao_summary,
            lcdm_plugin,
            alt_model_plugin,
            sne_data_df,
            plot_dir=output_dir,
            timestamp=end_ts,
        )
    if cmb_data_df is not None:
        plotter.plot_cmb_spectrum(
            cmb_data_df,
            lcdm_cmb_summary,
            alt_cmb_summary,
            lcdm_fit_results,
            alt_model_fit_results,
            lcdm_plugin,
            alt_model_plugin,
            plot_dir=output_dir,
            timestamp=end_ts,
        )

    posterior_attrs = {
        "dataset_id": (
            f"{sne_data_df.attrs.get('dataset_id', 'joint')}-posterior"
        ),
        "dataset_name": (
            f"{sne_data_df.attrs.get('dataset_name', 'Joint dataset')} "
            "Posterior Samples"
        ),
        "description": (
            "Corner plot summarising the joint posterior derived from the "
            "configured likelihood evaluation."
        ),
        "citation": sne_data_df.attrs.get("citation", ""),
        "notes": sne_data_df.attrs.get("notes", ""),
    }

    def _maybe_plot_corner(
        fit_results: Mapping[str, Any],
        plugin: Any,
        label: str,
    ) -> None:
        """Render a corner plot for ``fit_results`` when samples exist."""

        samples = fit_results.get("samples") if fit_results else None
        if samples is None:
            return
        param_names = fit_results.get("param_names") if fit_results else None
        try:
            plotter.plot_corner(
                samples,
                plugin,
                posterior_attrs,
                plot_dir=output_dir,
                parameter_names=param_names,
                timestamp=end_ts,
            )
        except Exception as exc:  # pragma: no cover - log path only
            logger.error(
                "Failed to generate %s corner plot: %s",
                label,
                exc,
            )

    _maybe_plot_corner(
        alt_model_fit_results,
        alt_model_plugin,
        alt_model_plugin.MODEL_NAME,
    )

    console.write("\n--- Theory Abstracts ---\n")
    console.write(f"ΛCDM Abstract:\n{lcdm_plugin.MODEL_ABSTRACT}\n")
    console.write(
        f"{alt_model_plugin.MODEL_NAME} Abstract:\n"
        f"{alt_model_plugin.MODEL_ABSTRACT}\n"
    )

    def _print_fit(label, fit_res, bao_res, cmb_res, plugin):
        """Pretty-print χ² stats and fitted parameters for a model."""

        console.write(f"--- {label} Fit Report ---\n")
        if fit_res:
            from copernican_lib import latex_utils

            p_names = getattr(plugin, "PARAMETER_NAMES", [])
            p_latex = getattr(plugin, "PARAMETER_LATEX_NAMES", [])
            for name, latex_name in zip(p_names, p_latex):
                val = fit_res.get(
                    "fitted_cosmological_params", {}
                ).get(name)
                if val is not None:
                    disp = latex_utils.latex_to_unicode(latex_name)
                    console.write(f"  {disp} = {val:.5g}")
        chi2_sne = fit_res.get(
            "chi2_sne", fit_res.get("chi2_min", float("nan"))
        )
        chi2_total = fit_res.get("chi2_total", float("nan"))
        console.write(f"  χ²_Total = {chi2_total:.2f}")
        console.write(f"  χ²_SNe = {chi2_sne:.2f}")
        if bao_res:
            console.write(
                f"  χ²_BAO = {bao_res.get('chi2_bao', float('nan')):.2f}"
            )
        if cmb_res:
            console.write(
                f"  χ²_CMB = {cmb_res.get('chi2_cmb', float('nan')):.2f}"
            )
        console.write("")

    _print_fit(
        "ΛCDM",
        lcdm_fit_results,
        lcdm_bao_summary,
        lcdm_cmb_summary,
        lcdm_plugin,
    )
    _print_fit(
        alt_model_plugin.MODEL_NAME,
        alt_model_fit_results,
        alt_bao_summary,
        alt_cmb_summary,
        alt_model_plugin,
    )

    csv_writer.save_sne_results_detailed_csv(
        sne_data_df,
        lcdm_fit_results,
        alt_model_fit_results,
        lcdm_plugin,
        alt_model_plugin,
        csv_dir=output_dir,
        timestamp=end_ts,
    )

    if bao_data_df is not None:
        csv_writer.save_bao_results_csv(
            bao_data_df,
            lcdm_bao_summary,
            alt_bao_summary,
            alt_model_name=alt_model_plugin.MODEL_NAME,
            csv_dir=output_dir,
            timestamp=end_ts,
        )
    if cmb_data_df is not None:
        csv_writer.save_cmb_results_csv(
            cmb_data_df,
            lcdm_cmb_summary,
            alt_cmb_summary,
            alt_model_name=alt_model_plugin.MODEL_NAME,
            csv_dir=output_dir,
            timestamp=end_ts,
        )

    if lcdm_fit_results.get("samples") is not None:
        fname = utils.generate_filename(
            "posterior",
            sne_data_df.attrs.get("dataset_id", "sne_data"),
            "nc",
            model_name=lcdm_plugin.MODEL_NAME.replace(" ", "_"),
            timestamp=end_ts,
        )
        chain_io.save_posterior(
            lcdm_fit_results["samples"],
            lcdm_fit_results.get(
                "param_names", lcdm_plugin.PARAMETER_NAMES
            ),
            os.path.join(output_dir, fname),
            metadata={
                "model": lcdm_plugin.MODEL_NAME,
                "dataset": sne_data_df.attrs.get("dataset_id", ""),
            },
        )
    if alt_model_fit_results.get("samples") is not None:
        fname = utils.generate_filename(
            "posterior",
            sne_data_df.attrs.get("dataset_id", "sne_data"),
            "nc",
            model_name=alt_model_plugin.MODEL_NAME.replace(" ", "_"),
            timestamp=end_ts,
        )
        chain_io.save_posterior(
            alt_model_fit_results["samples"],
            alt_model_fit_results.get(
                "param_names", alt_model_plugin.PARAMETER_NAMES
            ),
            os.path.join(output_dir, fname),
            metadata={
                "model": alt_model_plugin.MODEL_NAME,
                "dataset": sne_data_df.attrs.get("dataset_id", ""),
            },
        )

    console.write("\n" + "=" * 50)
    console.write(
        "Evaluation complete. All files saved to the 'output' directory."
    )
    console.write("=" * 50 + "\n")

    cpu_model, cpu_freq = _get_cpu_info()
    os_info = platform.platform()

    logger.info(f"Run completed at {end_ts} UTC.")

    console.write(
        f"Run started on {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    console.write(
        f"Run ended on {run_end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    console.write(
        "System summary: "
        f"{cpu_model} {cpu_freq} running {os_info}"
    )

    state.last_output_dir = output_dir
    state.last_log_file = log_file
    state.last_run_started = run_start_dt
    state.last_run_finished = run_end_dt
    program_logger.info(
        "Run %s saved to %s",
        run_start_ts,
        output_dir,
    )
    cleanup_cache(script_dir)
    console.write("Returning to dashboard.")
    console.write("")


def main_workflow():
    """Main workflow for the Copernican Suite."""

    opts = get_runtime_options()
    check_dependencies()
    if opts.run_tests:
        success = run_startup_tests()
        exit_clean(0 if success else 1)

    global np, plt, mp, model_spec_validator, model_coder
    global engine_plugin_validation
    global dataset_registry, plotter, csv_writer, log_mod, logger
    global error_handler
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    from copernican_lib import (
        model_spec_validator,
        model_coder,
        engine_plugin_validation,
        dataset_registry,
        plotter,
        csv_writer,
        logger as log_mod,
        utils,
        error_handler,
        chain_io,
    )

    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()

    OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

    program_log_file = log_mod.setup_program_logging(
        log_dir=LOGS_DIR,
        base_dir=SCRIPT_DIR,
        rollover_mb=10.0,
    )
    program_logger = log_mod.get_program_logger()
    logger = log_mod.get_logger()
    program_logger.info(
        "Diagnostics logging active at %s; outputs live under %s",
        program_log_file,
        LOGS_DIR,
    )

    show_splash_screen()

    def _load_lcdm_model():
        """Load and validate the reference ΛCDM model from its YAML file."""

        models_dir = os.path.join(SCRIPT_DIR, "models")
        yaml_path = os.path.join(models_dir, "cosmo_model_lcdm.yml")
        cache_dir = os.path.join(models_dir, "cache")
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        func_dict, parsed = model_coder.generate_callables(cache_path)
        plugin = engine_plugin_validation.build_plugin(parsed, func_dict)
        plugin.MODEL_FILENAME = os.path.basename(yaml_path)
        return plugin, parsed

    global lcdm, lcdm_parsed
    lcdm, lcdm_parsed = _load_lcdm_model()
    engine_plugin_validation.validate_plugin(lcdm)

    state = DashboardState()

    while True:
        choice = _display_dashboard_menu(state)
        if choice in {"1", "config", "configuration"}:
            _configuration_section(state, lcdm)
        elif choice in {"2", "engine", "model"}:
            _engine_model_section(state, SCRIPT_DIR)
        elif choice in {"3", "dataset", "datasets"}:
            _dataset_toggle_section(state)
        elif choice in {"4", "run"}:
            _run_dashboard_evaluation(
                state,
                lcdm,
                lcdm_parsed,
                opts,
                SCRIPT_DIR,
                OUTPUT_BASE_DIR,
                program_logger,
            )
        elif choice in {"5", "output", "outputs"}:
            _outputs_section(state)
        elif choice in {"6", "settings", "s"}:
            _settings_section(state, opts)
        elif choice in {"c", "close", "q", "quit", "exit"}:
            cleanup_cache(SCRIPT_DIR)
            program_logger.info("Session closed at operator request.")
            console.write("")
            return
        else:
            console.write(
                "Please choose one of the dashboard sections.",
                error=True,
            )


if __name__ == "__main__":
    # Multiprocessing start method must be 'spawn' so that each child process
    # inherits a pristine interpreter state. This avoids subtle issues when
    # worker processes import project modules that expect to run only once.
    import multiprocessing as _mp

    _mp.freeze_support()
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # The start method was already set (e.g. by another library). Using
        # 'force=True' above normally prevents this, but wrap in try/except
        # for absolute safety.
        pass
    try:
        main_workflow()
    except Exception:
        logger_obj = log_mod.get_logger() if log_mod else None
        if logger_obj and logger_obj.hasHandlers():
            logger_obj.critical(
                "Unhandled exception in main_workflow!",
                exc_info=True,
            )
        else:
            console.write("CRITICAL UNHANDLED EXCEPTION IN MAIN WORKFLOW:")
            import traceback

            traceback.print_exc()
    finally:
        # Ensure that any generated plot windows are displayed at the very end
        if (
            plt is not None
            and hasattr(plt, "get_fignums")
            and plt.get_fignums()
        ):
            console.write(
                "\nDisplaying plot(s). Close plot window(s) to exit script "
                "fully."
            )
            try:
                plt.show(block=True)
            except Exception as e_show:
                console.write(f"Error during final plt.show(): {e_show}")
        console.write("")

# fmt: on
