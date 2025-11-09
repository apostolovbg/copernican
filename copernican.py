# Last Updated: 2025-11-09
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# copernican_suite/copernican.py
# flake8: noqa
# isort: skip_file
# fmt: off
"""Copernican Suite - Main Orchestrator.

This script ties together model selection, dataset loading, dependency
checks and result generation.  Runtime behaviour is configured through
environment variables set by the cross-platform ``start`` launchers, so
no raw command line flags are exposed to end users.  The module also
houses the optional test runner and automated package installer so that
a fresh checkout can execute with minimal setup.
"""


import ast
import copy
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import platform
import shutil
import time
import datetime
import subprocess
import faulthandler
import signal
from collections.abc import Mapping
from dataclasses import dataclass
import random
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
from copernican_lib.stage2_runtime import StageTwoRuntimeTracker

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

model_parser = None
model_coder = None
engine_interface = None
plotter = None
csv_writer = None
log_mod = None
logger = None
data_loaders = None

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
    auto_confirm: bool = False


def get_runtime_options() -> RuntimeOptions:
    """Return options from ``COPERNICAN_*`` environment variables."""

    return RuntimeOptions(
        run_tests=os.environ.get("COPERNICAN_RUN_TESTS") == "1",
        strict_warnings=os.environ.get("COPERNICAN_STRICT_WARNINGS") == "1",
        auto_confirm=os.environ.get("COPERNICAN_AUTO_INSTALL") == "1",
    )


def select_seed() -> int:
    """Prompt the operator to choose a reproducible random seed.

    The dialog mirrors other Copernican menus with structured spacing so the
    seed selection feels like a deliberate Stage 1 action. When
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
        choice = console.ask("Select an option: ").strip().lower()
        if choice in {"1", "", "default"}:
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
    time.sleep(1)
    console.write(
        "Follow the prompts to configure a run. Results are saved in the "
        "'output' directory.\n\n"
    )


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
        # Local modules within this repository (under ``copernican_lib``)
        "data_loaders",
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


def check_dependencies(auto_confirm: bool = False) -> None:
    """Ensure required packages exist inside the local ``.venv``.

    Parameters
    ----------
    auto_confirm : bool, optional
        When ``True`` any missing packages are installed without prompting
        the user.  This is intended for non-interactive environments such as
        continuous integration systems.  When ``False`` the user is asked to
        confirm installation before ``pip`` is invoked.

    The suite bundles a virtual environment under ``.venv`` that is activated
    by the ``start.*`` launchers.  This check confirms the interpreter is
    running from that environment before installing any missing packages.
    Required packages are installed automatically from ``requirements.lock``
    and re-imported to verify success so the workflow
    can proceed without manual steps.
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
            f"Missing packages detected: {', '.join(missing)}"
        )
        if not auto_confirm:
            reply = console.ask("Install missing packages? [y/N] ")
            if reply.lower() not in {"y", "yes"}:
                console.write(
                    "Dependency installation aborted by user.",
                    error=True,
                )
                exit_clean(1)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.lock",
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            console.write(
                (
                    "Automatic installation failed. Please check the log and "
                    "install the required packages manually."
                ),
                error=True,
            )
            exit_clean(1)

        failed = []
        for pkg in missing:
            try:
                importlib.import_module(pkg)
            except Exception:
                failed.append(pkg)
        if failed:
            console.write(
                (
                    "Still missing packages after installation: "
                    f"{', '.join(failed)}"
                ),
                error=True,
            )
            exit_clean(1)
        console.write("✅ Packages installed successfully. Continuing...\n")
    else:
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
            engine_interface.validate_plugin(alt_model_module)
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


def _prompt_stage1_retry(reasons: Iterable[str]) -> bool:
    """Return ``True`` to restart Stage 1, ``False`` to exit the workflow."""

    console.write("")
    console.write("Stage 1 cannot continue because:")
    for entry in reasons:
        console.write(f"  - {entry}")
    console.write("")
    console.write("How would you like to proceed?")
    console.write("  1) Restart Stage 1 configuration from the beginning")
    console.write("  C) Exit the Copernican Suite")
    console.write("")

    while True:
        decision = console.ask("Select an option: ").strip().lower()
        if decision in {"", "1", "restart"}:
            console.write("")
            console.write("Restarting Stage 1 configuration.")
            return True
        if decision in {"c", "cancel", "exit"}:
            return False
        console.write("Please choose 1 to restart or C to exit.", error=True)


def _estimate_sampler_timing(
    engine_module,
    lcdm_plugin,
    alt_model_plugin,
    plan: Mapping[str, int | None],
    *,
    sne_df,
    bao_df,
    cmb_df,
) -> dict[str, Any]:
    """Return timing estimates for the provided sampler ``plan``."""

    burn_in_target = max(int(plan["burn_in_steps"]), 0)
    production_target = max(int(plan["n_steps"]), 0)
    n_walkers = int(plan["n_walkers"])
    pool_size = plan.get("pool_size")

    # Probe the sampler with a single iteration for each requested phase.  The
    # previous heuristic scaled trial lengths with the requested totals, but
    # that often overshot and distorted estimates when operators experimented
    # with long production chains.  Running precisely one step for burn-in and
    # one for production captures the ensemble and pool overhead while keeping
    # the measurement lightweight.
    trial_burn = 1 if burn_in_target > 0 else 0
    trial_prod = 1 if production_target > 0 else 0

    def _run_trial(plugin) -> float:
        start = time.perf_counter()
        result = engine_module.fit_sne_parameters(
            sne_df,
            plugin,
            bao_data_df=bao_df,
            cmb_data_df=cmb_df,
            n_walkers=n_walkers,
            n_steps=trial_prod,
            pool_size=pool_size,
            burn_in_steps=trial_burn,
            progress_granularity=max(1, min(trial_prod, 5)),
            display_progress=False,
        )
        duration = time.perf_counter() - start
        if not result.get("success", False):
            raise RuntimeError(
                f"Trial sampling for {getattr(plugin, 'MODEL_NAME', 'model')} "
                "did not succeed; unable to project runtime."
            )
        return duration

    lcdm_duration = _run_trial(lcdm_plugin) if (trial_burn or trial_prod) else 0.0

    same_name = (
        getattr(lcdm_plugin, "MODEL_NAME", "").casefold()
        == getattr(alt_model_plugin, "MODEL_NAME", "").casefold()
    )
    same_file = (
        getattr(lcdm_plugin, "MODEL_FILENAME", "")
        == getattr(alt_model_plugin, "MODEL_FILENAME", "")
    )
    alt_duration = 0.0
    if not (same_name and same_file and type(lcdm_plugin) is type(alt_model_plugin)):
        alt_duration = _run_trial(alt_model_plugin)

    steps_in_trial = trial_burn + trial_prod
    # When at least one phase was sampled, extrapolate totals using the average
    # per-step duration.  Otherwise, retain zero-filled estimates so callers can
    # present a consistent structure even when the operator skips Stage 2.
    if steps_in_trial:
        lcdm_per_step = lcdm_duration / steps_in_trial
        alt_per_step = (
            alt_duration / steps_in_trial if alt_duration else lcdm_per_step
        )
    else:
        lcdm_per_step = 0.0
        alt_per_step = 0.0

    def _split_duration(per_step: float) -> tuple[float, float, float]:
        burn = per_step * burn_in_target
        prod = per_step * production_target
        return burn, prod, burn + prod

    lcdm_components = _split_duration(lcdm_per_step)
    alt_components = _split_duration(alt_per_step)

    return {
        "lcdm": {
            "burn_in": lcdm_components[0],
            "production": lcdm_components[1],
            "total": lcdm_components[2],
        },
        "alt": {
            "burn_in": alt_components[0],
            "production": alt_components[1],
            "total": alt_components[2],
        },
        "combined": lcdm_components[2] + alt_components[2],
        "trial": {
            "burn_in": trial_burn,
            "production": trial_prod,
        },
    }


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


def prompt_sampling_configuration(
    engine_module,
    lcdm_plugin,
    alt_model_plugin,
    sne_data_df,
    bao_data_df,
    cmb_data_df,
):
    """Return user-selected sampler settings or ``None`` if cancelled."""

    fit_sig = inspect.signature(engine_module.fit_sne_parameters)
    try:
        default_steps = int(fit_sig.parameters["n_steps"].default)
    except (KeyError, ValueError, TypeError):
        default_steps = 200
    try:
        default_walkers = int(fit_sig.parameters["n_walkers"].default)
    except (KeyError, ValueError, TypeError):
        default_walkers = 32

    lcdm_active = _count_active_parameters(lcdm_plugin, engine_module=engine_module)
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

    def _handle_time_estimate(plan: Mapping[str, int | None]) -> str:
        """Return next action after displaying runtime estimates."""

        console.write("")
        console.write("Calculating live runtime estimate...")
        try:
            estimate = _estimate_sampler_timing(
                engine_module,
                lcdm_plugin,
                alt_model_plugin,
                plan,
                sne_df=sne_data_df,
                bao_df=bao_data_df,
                cmb_df=cmb_data_df,
            )
        except Exception as exc:
            console.write(f"Unable to estimate runtime: {exc}", error=True)
            console.write("")
            return "back"

        def _format_duration(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            hours, remainder = divmod(seconds, 3600.0)
            minutes, secs = divmod(remainder, 60.0)
            parts: list[str] = []
            if hours >= 1.0:
                parts.append(f"{int(hours)}h")
            if minutes >= 1.0 or hours >= 1.0:
                parts.append(f"{int(minutes)}m")
            parts.append(f"{secs:.1f}s")
            return " ".join(parts)

        console.write("")
        console.write("Runtime estimate")
        console.write("----------------")
        console.write(
            "Trial sample sizes: "
            f"burn-in {estimate['trial']['burn_in']} step(s), production "
            f"{estimate['trial']['production']} step(s)."
        )
        console.write("")
        console.write(
            f"ΛCDM burn-in ≈ {_format_duration(estimate['lcdm']['burn_in'])}"
        )
        console.write(
            f"ΛCDM production ≈ {_format_duration(estimate['lcdm']['production'])}"
        )
        console.write(
            f"ΛCDM total ≈ {_format_duration(estimate['lcdm']['total'])}"
        )
        if estimate["alt"]["total"] <= 1e-6:
            console.write(
                "Alternative model reuses the ΛCDM chain; no extra sampling "
                "time expected."
            )
        else:
            console.write(
                f"Alternative burn-in ≈ {_format_duration(estimate['alt']['burn_in'])}"
            )
            console.write(
                "Alternative production ≈ "
                f"{_format_duration(estimate['alt']['production'])}"
            )
            console.write(
                f"Alternative total ≈ {_format_duration(estimate['alt']['total'])}"
            )
        console.write(
            f"Combined sampler time ≈ {_format_duration(estimate['combined'])}"
        )
        console.write("")
        console.write("1) Continue with these settings")
        console.write("B) Return to the sampler menu")
        console.write("C) Exit the Copernican Suite")
        console.write("")

        while True:
            choice = console.ask("Select an option: ").strip().lower()
            if choice in {"", "1", "continue"}:
                return "continue"
            if choice in {"b", "back"}:
                return "back"
            if choice in {"c", "cancel", "exit"}:
                return "exit"
            console.write("Please choose 1, B or C.", error=True)

    def _collect_custom_plan() -> dict[str, int | None] | str | None:
        """Gather a customised sampler configuration from the operator."""

        while True:
            console.write("")
            console.write("Production steps control the total sampler iterations.")
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
                f"  A shorter option such as {quick_burn} trades certainty for "
                "speed."
            )
            entry = console.ask(f"Burn-in steps [{default_burn}]: ").strip()
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
            entry = console.ask(f"Number of walkers [{walker_default}]: ").strip()
            if not entry:
                n_walkers = walker_default
            else:
                try:
                    n_walkers = int(entry)
                except ValueError:
                    console.write("Walker count must be an integer.", error=True)
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
                    f"Walker count increased to {adjusted_walkers} to match the "
                    "worker pool."
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
            console.write(
                "  3) Estimate runtime with these settings"
            )
            console.write("  B) Back to the sampler defaults summary")
            console.write("  C) Cancel sampler configuration")
            console.write("")

            confirm = console.ask("Select an option: ").strip().lower()
            current_plan = {
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
            if confirm in {"3", "estimate", "e"}:
                outcome = _handle_time_estimate(current_plan)
                if outcome == "continue":
                    return current_plan
                if outcome == "exit":
                    return None
                console.write("")
                console.write("Returning to the sampler confirmation menu.")
                continue
            if confirm in {"2", "r", "restart", "n", "no"}:
                console.write("")
                console.write("Restarting the sampler questionnaire from step one.")
                continue
            console.write("Please choose 1, 2, 3, B or C.", error=True)

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
        console.write("3) Estimate runtime with default settings")
        console.write("C) Cancel configuration")
        console.write("")

        choice = console.ask("Write the number of choice: ").strip().lower()
        if choice in {"", "1"}:
            return {
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
        if choice == "3":
            defaults = {
                "n_steps": recommended_steps,
                "burn_in_steps": recommended_burn,
                "n_walkers": recommended_walkers,
                "pool_size": recommended_pool,
            }
            outcome = _handle_time_estimate(defaults)
            if outcome == "continue":
                return defaults
            if outcome == "exit":
                return None
            console.write("")
            console.write("Returning to the sampler defaults summary.")
            continue
        if choice in {"c", "cancel"}:
            return None
        console.write("Please choose 1, 2, 3 or C.", error=True)


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
            "Basic NumPy/SciPy check failed. This often points to CPU feature "
            "mismatches or a corrupted install. Reinstall NumPy and SciPy with "
            "wheels built for your machine.",
            exc_info=exc,
        )
        raise


def prompt_post_run_action():
    """Return ``True`` to launch another evaluation, ``False`` to exit."""

    while True:
        console.write("")
        console.write("Next actions:")
        console.write("  1) Start another evaluation run")
        console.write("  C) Close the Copernican Suite")
        console.write("")
        choice = console.ask("Select an option: ").strip().lower()
        if choice in {"", "1", "y", "yes"}:
            console.write("")
            return True
        if choice in {"c", "cancel", "n", "no", "2"}:
            return False
        console.write("Please choose 1 or C.", error=True)


def main_workflow():
    """Main workflow for the Copernican Suite."""
    # This routine coordinates the entire user interaction:
    #  * read environment-controlled runtime options
    #  * verify Python dependencies
    #  * perform a NumPy/SciPy sanity check
    #  * load the reference ΛCDM model
    #  * repeatedly ask the user for models, data sources and engines
    #  * produce plots and CSV files with the results
    opts = get_runtime_options()
    check_dependencies(auto_confirm=opts.auto_confirm)
    if opts.run_tests:
        success = run_startup_tests()
        exit_clean(0 if success else 1)

    # Import optional third-party packages after confirming they are installed
    global np, plt, mp, model_parser, model_coder, engine_interface, \
        data_loaders, plotter, csv_writer, log_mod, logger, error_handler
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    from copernican_lib import model_parser, model_coder, engine_interface
    from copernican_lib import (
        data_loaders,
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

    show_splash_screen()

    # Load the baseline LCDM model from YAML and validate it
    def _load_lcdm_model():
        """Load and validate the reference ΛCDM model from its YAML file."""
        models_dir = os.path.join(SCRIPT_DIR, "models")
        yaml_path = os.path.join(models_dir, "cosmo_model_lcdm.yml")
        cache_dir = os.path.join(models_dir, "cache")
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        func_dict, parsed = model_coder.generate_callables(cache_path)
        plugin = engine_interface.build_plugin(parsed, func_dict)
        plugin.MODEL_FILENAME = os.path.basename(yaml_path)
        return plugin, parsed

    global lcdm, lcdm_parsed
    lcdm, lcdm_parsed = _load_lcdm_model()
    engine_interface.validate_plugin(lcdm)

    while True:
        run_start_ts = utils.get_timestamp()
        OUTPUT_DIR = os.path.join(
            OUTPUT_BASE_DIR, f"copernican-run_{run_start_ts}"
        )
        utils.ensure_dir_exists(OUTPUT_DIR)
        global CURRENT_LOG_FILE
        log_file = log_mod.setup_logging(
            log_dir=OUTPUT_DIR, base_dir=SCRIPT_DIR
        )
        # The historical Gemini-era banner declared that Copernican had
        # initialised.  The suite now conveys that status implicitly through
        # the surrounding menu flow, but the console still expects a blank
        # spacer where the message once lived so spacing remains familiar.
        console.write("")
        CURRENT_LOG_FILE = log_file
        logger = log_mod.get_logger()
        error_handler.configure_warnings(strict=opts.strict_warnings)
        if opts.strict_warnings:
            logger.info(
                "Strict warnings mode enabled; treating warnings as errors"
            )
        else:
            logger.info(
                "Warnings will be logged but not treated as errors"
            )
        # Record interpreter and package details for reproducibility
        log_mod.log_environment_info()
        try:
            _sanity_check_numpy_scipy(logger)
        except Exception:
            exit_clean(1)
        run_start_dt = datetime.datetime.now(datetime.timezone.utc)
        run_start_pc = time.perf_counter()
        logger.info("")
        logger.info(
            "Using standard CPU (SciPy) computational backend with "
            "multiprocessing."
        )
        logger.info(f"Running from base directory: {SCRIPT_DIR}")
        logger.info(f"All outputs will be saved to: {OUTPUT_DIR}")
        alt_model_plugin = None
        cosmo_engine_selected = None
        selected_model = ""

        # Stage 1 collects all configuration inputs. Wrapping the sequence in a
        # loop keeps the workflow responsive when validation fails: operators can
        # review the reported reasons, restart immediately and continue without
        # relaunching the suite.
        while True:
            logger.info("\n--- Stage 1: Configuration ---\n")
            console.write("")
            console.write("Stage 1 – Configuration")
            console.write("-----------------------")
            console.write(
                "We will set the random seed, pick an alternative model "
                "and select the computation engine before loading data."
            )
            console.write("")

            # Seed selection appears directly after the banner so the
            # reproducibility contract is front and centre before model or
            # engine choices begin.
            select_seed()
            logger.info("Using RNG seed %s", utils.get_random_seed())

            models_dir = os.path.join(SCRIPT_DIR, "models")
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
                _delete_log_file(log_file)
                _remove_run_dir(OUTPUT_DIR)
                cleanup_cache(SCRIPT_DIR)
                console.write("")
                return
            yaml_path = os.path.join(models_dir, selected_model)
            cache_dir = os.path.join(models_dir, "cache")
            try:
                cache_path = model_parser.parse_model(yaml_path, cache_dir)
                func_dict, parsed = model_coder.generate_callables(cache_path)
                alt_model_plugin = engine_interface.build_plugin(
                    parsed, func_dict
                )
                alt_model_plugin.MODEL_FILENAME = os.path.basename(yaml_path)
                logger.info(f"Loaded YAML model: {parsed.get('model_name')}")
            except PluginValidationError as exc:
                logger.error("Model validation failed: %s", exc)
                reasons = _normalise_failure_reasons(str(exc))
                if _prompt_stage1_retry(reasons):
                    logger.info(
                        "Restarting Stage 1 after alternative model validation failure."
                    )
                    continue
                _delete_log_file(log_file)
                _remove_run_dir(OUTPUT_DIR)
                cleanup_cache(SCRIPT_DIR)
                console.write("")
                return
            except Exception as exc:
                logger.error(
                    "Error generating model from YAML: %s",
                    exc,
                    exc_info=True,
                )
                reasons = _normalise_failure_reasons(str(exc))
                if _prompt_stage1_retry(reasons):
                    logger.info(
                        "Restarting Stage 1 after model parsing error."
                    )
                    continue
                _delete_log_file(log_file)
                _remove_run_dir(OUTPUT_DIR)
                cleanup_cache(SCRIPT_DIR)
                console.write("")
                return

            engines_dir = os.path.join(SCRIPT_DIR, "engines")
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
                _delete_log_file(log_file)
                _remove_run_dir(OUTPUT_DIR)
                cleanup_cache(SCRIPT_DIR)
                console.write("")
                return
            try:
                engine_module = importlib.import_module(
                    f"engines.{engine_choice[:-3]}"
                )
            except Exception as exc:
                logger.error(
                    "Engine import failed: %s",
                    exc,
                    exc_info=True,
                )
                reasons = _normalise_failure_reasons(str(exc))
                if _prompt_stage1_retry(reasons):
                    logger.info(
                        "Restarting Stage 1 after engine import failure."
                    )
                    continue
                _delete_log_file(log_file)
                _remove_run_dir(OUTPUT_DIR)
                cleanup_cache(SCRIPT_DIR)
                console.write("")
                return
            cosmo_engine_selected = engine_module
            break

        sne_data_df = data_loaders.load_sne_data()
        if sne_data_df is None:
            _delete_log_file(log_file)
            _remove_run_dir(OUTPUT_DIR)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            continue

        bao_data_df = data_loaders.load_bao_data()
        if bao_data_df is None:
            _delete_log_file(log_file)
            _remove_run_dir(OUTPUT_DIR)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            continue

        cmb_data_df = data_loaders.load_cmb_data()
        if cmb_data_df is None:
            _delete_log_file(log_file)
            _remove_run_dir(OUTPUT_DIR)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            continue
        dataset_info = []
        for df in (sne_data_df, bao_data_df, cmb_data_df):
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

        sampling_plan = prompt_sampling_configuration(
            cosmo_engine_selected,
            lcdm,
            alt_model_plugin,
            sne_data_df,
            bao_data_df,
            cmb_data_df,
        )
        if sampling_plan is None:
            logger.info("User cancelled sampling configuration; aborting run.")
            _delete_log_file(log_file)
            _remove_run_dir(OUTPUT_DIR)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            return

        sampling_steps = int(sampling_plan["n_steps"])
        sampling_burn_in = int(sampling_plan["burn_in_steps"])
        sampling_walkers = int(sampling_plan["n_walkers"])
        sampling_pool = sampling_plan["pool_size"]

        pool_label = sampling_pool if sampling_pool is not None else "auto"
        logger.info(
            (
                "Sampler configuration: steps=%d, burn-in=%d, walkers=%d, "
                "pool=%s"
            ),
            sampling_steps,
            sampling_burn_in,
            sampling_walkers,
            pool_label,
        )
        console.write(
            f"Configured sampler: steps {sampling_steps}, burn-in "
            f"{sampling_burn_in}."
        )
        console.write(
            f"Walker ensemble {sampling_walkers}, pool {pool_label}."
        )

        manifest = run_manifest.build_manifest(
            models=[
                (lcdm, lcdm_parsed.get("version", "unknown")),
                (alt_model_plugin, parsed.get("version", "unknown")),
            ],
            engine_module=cosmo_engine_selected,
            datasets=dataset_info,
        )
        run_manifest.save_manifest(manifest, OUTPUT_DIR)

        lcdm_name = getattr(lcdm, "MODEL_NAME", "ΛCDM")
        alt_name = getattr(alt_model_plugin, "MODEL_NAME", "Alternative")
        same_name = lcdm_name.casefold() == alt_name.casefold()
        lcdm_file = getattr(lcdm, "MODEL_FILENAME", "")
        alt_file = getattr(alt_model_plugin, "MODEL_FILENAME", "")
        reuse_alt = same_name and lcdm_file == alt_file and (
            type(lcdm) is type(alt_model_plugin)
        )
        stage_plan: list[tuple[str, str, int]] = []
        if sampling_burn_in > 0:
            stage_plan.append((lcdm_name, "burn-in", sampling_burn_in))
        if sampling_steps > 0:
            stage_plan.append((lcdm_name, "production", sampling_steps))
        if not reuse_alt:
            if sampling_burn_in > 0:
                stage_plan.append((alt_name, "burn-in", sampling_burn_in))
            if sampling_steps > 0:
                stage_plan.append((alt_name, "production", sampling_steps))
        runtime_tracker = (
            StageTwoRuntimeTracker(stage_plan)
            if stage_plan
            else None
        )

        def _dispatch_runtime_update(event: Mapping[str, Any]) -> str | None:
            """Forward sampler metrics to the combined runtime tracker."""

            if runtime_tracker is None:
                return None
            return runtime_tracker.update(event)

        lcdm_time = 0.0
        alt_time = 0.0
        engine_label = getattr(
            cosmo_engine_selected,
            "ENGINE_LABEL",
            getattr(cosmo_engine_selected, "__name__", "Engine"),
        )
        logger.info("\n--- Stage 2: %s ---\n", engine_label)
        console.write("")
        console.write(f"Stage 2 – {engine_label}")
        console.write("------------------------------")
        console.write(
            "Stage 2 estimates parameters for the ΛCDM reference model and "
            "then the selected alternative theory."
        )
        console.write(
            "Burn-in prepares the chains; production accumulates posterior "
            "samples for analysis."
        )
        console.write(
            "Each sampler phase announces its start and progress bars fill "
            "gradually for every batch so lengthy runs remain easy to follow."
        )
        console.write("")
        if not hasattr(cosmo_engine_selected, "fit_sne_parameters"):
            logger.error(
                "Selected engine %s does not expose fit_sne_parameters;"
                " aborting run.",
                getattr(cosmo_engine_selected, "__name__", "unknown"),
            )
            _delete_log_file(log_file)
            _remove_run_dir(OUTPUT_DIR)
            cleanup_cache(SCRIPT_DIR)
            console.write("")
            return
        console.write("ΛCDM reference chain")
        console.write(f"  Burn-in steps: {sampling_burn_in}")
        console.write(f"  Production steps: {sampling_steps}")
        console.write(f"  Walkers: {sampling_walkers}")
        console.write(f"  Worker pool: {pool_label}")
        console.write("  Starting ΛCDM sampler...")
        console.write("")
        t0 = time.perf_counter()
        lcdm_fit_results = cosmo_engine_selected.fit_sne_parameters(
            sne_data_df,
            lcdm,
            bao_data_df=bao_data_df,
            cmb_data_df=cmb_data_df,
            n_walkers=sampling_walkers,
            n_steps=sampling_steps,
            pool_size=sampling_pool,
            burn_in_steps=sampling_burn_in,
            progress_listener=_dispatch_runtime_update,
        )
        lcdm_time += time.perf_counter() - t0
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
            console.write(f"  Burn-in steps: {sampling_burn_in}")
            console.write(f"  Production steps: {sampling_steps}")
            console.write(f"  Walkers: {sampling_walkers}")
            console.write(f"  Worker pool: {pool_label}")
            console.write("  Starting alternative sampler...")
            console.write("")
            t0 = time.perf_counter()
            alt_model_fit_results = cosmo_engine_selected.fit_sne_parameters(
                sne_data_df,
                alt_model_plugin,
                bao_data_df=bao_data_df,
                cmb_data_df=cmb_data_df,
                n_walkers=sampling_walkers,
                n_steps=sampling_steps,
                pool_size=sampling_pool,
                burn_in_steps=sampling_burn_in,
                progress_listener=_dispatch_runtime_update,
            )
            alt_time += time.perf_counter() - t0
            console.write(
                f"Completed alternative sampling for {alt_model_plugin.MODEL_NAME}."
            )
            console.write("")

        console.write("Stage 2 sampling complete.")
        console.write("")

        # Persist parameter estimates so external tools can inspect the
        # numerical results without parsing logs.  The summary includes fitted
        # values, 1σ errors and the covariance matrix for each model.
        result_writer.save_summary(
            {
                lcdm.MODEL_NAME: lcdm_fit_results,
                alt_model_plugin.MODEL_NAME: alt_model_fit_results,
            },
            OUTPUT_DIR,
        )

        logger.info("\n--- Stage 3: BAO Analysis ---\n")

        def _component_enabled(fit_results, component):
            state = fit_results.get("likelihood_state", {}) if fit_results else {}
            metadata = state.get("metadata", {})
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
        z_plot_smooth = np.geomspace(max(min_z * 0.8, 0.01), max_z * 1.2, 100)

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
                        f"{model_plugin.MODEL_NAME} fit failed; "
                        "skipping BAO analysis."
                    )
                )
                return summary

            if not _component_enabled(fit_results, "bao"):
                logger.info(
                    (
                        f"{model_plugin.MODEL_NAME} BAO likelihood disabled; "
                        "skipping predictions."
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
                        f"{model_plugin.MODEL_NAME} fit does not expose "
                        "cosmological parameters; skipping BAO analysis."
                    )
                )
                summary["chi2_bao"] = float("inf")
                return summary

            pred_df, rs_Mpc, smooth_preds = (
                cosmo_engine_selected.calculate_bao_observables(
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
                            f"{model_plugin.MODEL_NAME} BAO predictions "
                            "available but χ² is non-finite."
                        )
                    )
            else:
                logger.warning(
                    f"{model_plugin.MODEL_NAME} BAO calculation failed or "
                    "produced invalid r_s."
                )

            return summary

        t0 = time.perf_counter()
        lcdm_bao_summary = run_bao_analysis(
            lcdm,
            lcdm_fit_results,
            z_plot_smooth,
        )
        lcdm_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        alt_bao_summary = run_bao_analysis(
            alt_model_plugin,
            alt_model_fit_results,
            z_plot_smooth,
        )
        alt_time += time.perf_counter() - t0

        if (
            np.isfinite(lcdm_bao_summary.get("rs_Mpc", np.nan))
            and np.isfinite(alt_bao_summary.get("rs_Mpc", np.nan))
        ):
            delta_rs = alt_bao_summary["rs_Mpc"] - lcdm_bao_summary["rs_Mpc"]
            logger.info(
                (
                    f"{alt_model_plugin.MODEL_NAME} r_s offset relative to "
                    f"{lcdm.MODEL_NAME}: {delta_rs:+.3f} Mpc"
                )
            )

        logger.info("\n--- Stage 4: CMB Analysis ---\n")

        def run_cmb_analysis(model_plugin, fit_results):
            """Return CMB diagnostics and theory spectra for ``model_plugin``."""

            summary = {
                "chi2_cmb": float(
                    (fit_results or {}).get("chi2_cmb", float("inf"))
                ),
                "theory_spectrum": None,
            }

            if not _component_enabled(fit_results, "cmb"):
                return summary

            if cmb_data_df is None or getattr(cmb_data_df, "empty", True):
                return summary

            if getattr(model_plugin, "valid_for_cmb", True) is False:
                logger.info(
                    (
                        f"{model_plugin.MODEL_NAME} does not support CMB; "
                        "skipping analysis."
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
                        f"{model_plugin.MODEL_NAME} fit does not provide "
                        "cosmological parameters; skipping CMB predictions."
                    )
                )
                summary["chi2_cmb"] = float("inf")
                return summary

            try:
                camb_params = model_plugin.get_camb_params(cosmo_params)
            except Exception as exc:
                logger.warning(
                    (
                        f"{model_plugin.MODEL_NAME} failed to build CAMB "
                        f"parameters: {exc}"
                    )
                )
                summary["chi2_cmb"] = float("inf")
                return summary

            components = ["TT"]
            if "Dl_te_obs" in cmb_data_df.columns:
                components.append("TE")
            if "Dl_ee_obs" in cmb_data_df.columns:
                components.append("EE")

            theory = cosmo_engine_selected.compute_cmb_spectrum(
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
                        f"{model_plugin.MODEL_NAME} CMB likelihood disabled "
                        "or returned a non-finite χ²."
                    )
                )

            return summary

        t0 = time.perf_counter()
        lcdm_cmb_summary = run_cmb_analysis(lcdm, lcdm_fit_results)
        lcdm_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        alt_cmb_summary = run_cmb_analysis(
            alt_model_plugin,
            alt_model_fit_results,
        )
        alt_time += time.perf_counter() - t0

        logger.info("\n--- Stage 5: Generating Outputs ---\n")
        logger.info(
            f"{lcdm.MODEL_NAME} CMB χ² = {lcdm_cmb_summary['chi2_cmb']:.2f}"
        )
        logger.info(
            f"{alt_model_plugin.MODEL_NAME} CMB χ² = "
            f"{alt_cmb_summary['chi2_cmb']:.2f}"
        )

        run_end_dt = datetime.datetime.now(datetime.timezone.utc)
        end_ts = run_end_dt.strftime("%Y%m%d_%H%M%S")
        new_dir = os.path.join(
            OUTPUT_BASE_DIR, f"copernican-run_{end_ts}"
        )
        if OUTPUT_DIR != new_dir:
            try:
                os.rename(OUTPUT_DIR, new_dir)
                OUTPUT_DIR = new_dir
                log_file = os.path.join(
                    OUTPUT_DIR, os.path.basename(log_file)
                )
            except OSError as e_dir:
                logger.error(f"Failed renaming output directory: {e_dir}")
        new_log = os.path.join(OUTPUT_DIR, f"copernican-run_{end_ts}.txt")
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
            lcdm,
            alt_model_plugin,
            plot_dir=OUTPUT_DIR,
            timestamp=end_ts,
        )
        if bao_data_df is not None:
            plotter.plot_bao_observables(
                bao_data_df,
                lcdm_bao_summary,
                alt_bao_summary,
                lcdm,
                alt_model_plugin,
                sne_data_df,
                plot_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )
        if cmb_data_df is not None:
            plotter.plot_cmb_spectrum(
                cmb_data_df,
                lcdm_cmb_summary,
                alt_cmb_summary,
                lcdm_fit_results,
                alt_model_fit_results,
                lcdm,
                alt_model_plugin,
                plot_dir=OUTPUT_DIR,
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
                "Stage 2 corner plot summarising the joint posterior derived "
                "from the combined SNe, BAO and CMB likelihood evaluation."
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
                    plot_dir=OUTPUT_DIR,
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
        console.write(f"ΛCDM Abstract:\n{lcdm.MODEL_ABSTRACT}\n")
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
            "ΛCDM", lcdm_fit_results, lcdm_bao_summary, lcdm_cmb_summary, lcdm
        )
        _print_fit(
            alt_model_plugin.MODEL_NAME,
            alt_model_fit_results,
            alt_bao_summary,
            alt_cmb_summary,
            alt_model_plugin,
        )

        # The call to the redundant summary CSV has been removed.
        # csv_writer.save_sne_fit_results_csv(...)

        # Save the detailed point-by-point SNe results CSV
        csv_writer.save_sne_results_detailed_csv(
            sne_data_df,
            lcdm_fit_results,
            alt_model_fit_results,
            lcdm,
            alt_model_plugin,
            csv_dir=OUTPUT_DIR,
            timestamp=end_ts,
        )

        if bao_data_df is not None:
            csv_writer.save_bao_results_csv(
                bao_data_df,
                lcdm_bao_summary,
                alt_bao_summary,
                alt_model_name=alt_model_plugin.MODEL_NAME,
                csv_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )
        if cmb_data_df is not None:
            csv_writer.save_cmb_results_csv(
                cmb_data_df,
                lcdm_cmb_summary,
                alt_cmb_summary,
                alt_model_name=alt_model_plugin.MODEL_NAME,
                csv_dir=OUTPUT_DIR,
                timestamp=end_ts,
            )

        if lcdm_fit_results.get("samples") is not None:
            fname = utils.generate_filename(
                "posterior",
                sne_data_df.attrs.get("dataset_id", "sne_data"),
                "nc",
                model_name=lcdm.MODEL_NAME.replace(" ", "_"),
                timestamp=end_ts,
            )
            chain_io.save_posterior(
                lcdm_fit_results["samples"],
                lcdm_fit_results.get(
                    "param_names", lcdm.PARAMETER_NAMES
                ),
                os.path.join(OUTPUT_DIR, fname),
                metadata={
                    "model": lcdm.MODEL_NAME,
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
                os.path.join(OUTPUT_DIR, fname),
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

        total_time = time.perf_counter() - run_start_pc
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
            f"Run took {lcdm_time:.2f}s for LCDM and {alt_time:.2f}s for "
            f"{alt_model_plugin.MODEL_NAME}, "
            f"or {total_time:.2f}s in total, on a system with a {cpu_model} "
            f"{cpu_freq}, "
            f"under {os_info}"
        )

        run_again = prompt_post_run_action()
        cleanup_cache(SCRIPT_DIR)
        if not run_again:
            logger.info("Exiting Copernican Suite. Goodbye!")
            console.write("")
            return


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
