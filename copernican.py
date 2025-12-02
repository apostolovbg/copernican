# Last Updated: 2025-11-29
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# copernican_suite/copernican.py
# flake8: noqa
# isort: skip_file
# fmt: off
"""Copernican Suite - Main Orchestrator.

Last Updated: 2025-11-29

This script ties together model selection, dataset loading and result
generation while delegating dependency checks and menu rendering to
``copernican_lib.cli`` helpers. Runtime behaviour is configured through
environment variables set by the cross-platform ``start`` launchers, while
the ``--cli`` and ``--gui`` flags provide a thin shim for callers that want
to bypass the interactive menus and request GUI-safe orchestration services.
The module retains the optional test runner entrypoint so a fresh checkout
can execute with minimal setup before heavier imports occur.
"""


from __future__ import annotations

import argparse
import copy
import datetime
import faulthandler
import importlib
import importlib.util
import inspect
import os
import platform
import shutil
import signal
import subprocess
import sys
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from copernican_lib.cli import dependencies as cli_dependencies
from copernican_lib import console_output as console
from copernican_lib import logger as log_mod
from copernican_lib import orchestration
from copernican_lib import progress_state
from copernican_lib import run_manifest
from copernican_lib import utils
from copernican_lib.gui import CopernicanGUI
import copernican_lib.version as version_module
from copernican_lib.plugins import PluginValidationError

# Verify interpreter version early so users see clear feedback
MIN_PYTHON = (3, 11)

# Initialise optional heavy imports so cleanup code can reference them safely
# even when the dependency guard exits early.
np = None
plt = None
mp = None


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
allow_direct = os.environ.get("COPERNICAN_ALLOW_DIRECT") == "1"
if (
    not allow_direct
    and (current_venv is None or Path(current_venv).resolve() != EXPECTED_VENV)
):
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

# Heavy imports are deferred to ``copernican_lib.cli.dependencies`` so startup
# stays quick and dependency checks run before NumPy or Matplotlib load.

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


try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CURRENT_LOG_FILE = None
PROGRAM_LOG_FILE: str | None = None
PROGRAM_LOGGER: logging.Logger | None = None
_legacy_stage_menu_override = False
_launch_args: LaunchRequest | None = None


def _ensure_program_logging() -> logging.Logger:
    """
    Lazily configure program-level logging so both CLI and GUI paths write to
    ``logs/copernican-program_*.txt``.
    """

    global PROGRAM_LOGGER, PROGRAM_LOG_FILE
    if PROGRAM_LOGGER is not None:
        return PROGRAM_LOGGER

    logs_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    PROGRAM_LOG_FILE = log_mod.setup_program_logging(
        log_dir=logs_dir,
        base_dir=SCRIPT_DIR,
        rollover_mb=10.0,
    )
    PROGRAM_LOGGER = log_mod.get_program_logger()
    PROGRAM_LOGGER.info(
        "Diagnostics logging active at %s; outputs live under %s",
        PROGRAM_LOG_FILE,
        logs_dir,
    )
    return PROGRAM_LOGGER


def _build_gui_progress_callback(
) -> Callable[[dict[str, object]], None] | None:
    """Return a callable that records GUI progress updates when requested."""

    path_value = os.environ.get("COPERNICAN_GUI_PROGRESS_PATH")
    if not path_value:
        return None
    target = Path(path_value)

    def _callback(record: dict[str, object]) -> None:
        payload = dict(record)
        payload.setdefault("timestamp", utils.get_timestamp())
        try:
            progress_state.record_progress(target, payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger = log_mod.get_program_logger()
            logger.debug("Failed to update GUI progress state: %s", exc)

    return _callback


COPERNICAN_VERSION = _copernican_version()
CURRENT_LOG_FILE = None
_legacy_stage_menu_override = False
_launch_args: LaunchRequest | None = None


@dataclass
class LaunchRequest:
    """Parsed launcher arguments shared across CLI and GUI flows."""

    mode: orchestration.LaunchMode
    legacy_stage_menu: bool
    detach_gui: bool
    manifest_path: Path | None
    output_dir: Path | None


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


def legacy_stage_menu_enabled() -> bool:
    """Return ``True`` when the staged menu is explicitly requested."""

    return _legacy_stage_menu_override or (
        os.environ.get("COPERNICAN_ENABLE_STAGED_MENU") == "1"
    )


def _parse_launch_args(argv: Iterable[str] | None = None) -> LaunchRequest:
    """Return launch settings covering GUI, CLI and manifest routing."""

    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cli",
        action="store_true",
        help=(
            "Force the classic CLI launcher even when GUI tooling calls us "
            "so CI stays headless."
        ),
    )
    group.add_argument(
        "--gui",
        action="store_true",
        help="Expose orchestration services without starting the CLI menus.",
    )
    group.add_argument(
        "--no-gui",
        action="store_true",
        help=(
            "Skip GUI bootstrap even if wrappers request it. This keeps "
            "headless runs deterministic."
        ),
    )
    parser.add_argument(
        "--manifest",
        help=(
            "Save the generated manifest to this path instead of the default "
            "timestamped location under the run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Store run outputs beneath this directory instead of the "
            "repository's output folder."
        ),
    )
    parser.add_argument(
        "--enable-legacy-stage-menu",
        action="store_true",
        help=(
            "Re-enable the retired staged menu for CI-only coverage runs. "
            "Production launches stay forward-only."
        ),
    )
    argv_list = list(argv) if argv is not None else None
    parsed, _ = parser.parse_known_args(argv_list)
    legacy_menu = parsed.enable_legacy_stage_menu or (
        os.environ.get("COPERNICAN_ENABLE_STAGED_MENU") == "1"
    )
    detach_gui = os.environ.get("COPERNICAN_DETACH_GUI", "1") != "0"
    manifest_path = (
        Path(parsed.manifest).expanduser().resolve()
        if parsed.manifest
        else None
    )
    output_dir = (
        Path(parsed.output_dir).expanduser().resolve()
        if parsed.output_dir
        else None
    )
    mode = (
        orchestration.LaunchMode.GUI
        if parsed.gui
        else orchestration.LaunchMode.CLI
    )
    if parsed.no_gui or parsed.cli:
        mode = orchestration.LaunchMode.CLI
    return LaunchRequest(
        mode=mode,
        legacy_stage_menu=legacy_menu,
        detach_gui=detach_gui,
        manifest_path=manifest_path,
        output_dir=output_dir,
    )


def _gui_executable_candidates() -> list[Path]:
    """Return possible Python executables suitable for GUI launchers.

    GUI shells should prefer ``pythonw`` on platforms that supply it so the
    calling terminal can close without warning about a lingering console.  The
    managed virtual environment ships a ``pythonw`` binary on Windows, so the
    launcher checks for the variant before falling back to the active
    interpreter.
    """

    exe_path = Path(sys.executable).resolve()
    candidates = [exe_path]
    for suffix in ("pythonw.exe", "pythonw"):
        alt = exe_path.with_name(suffix)
        if alt.exists():
            candidates.insert(0, alt)
    return candidates


def _launch_detached_process(
    command: list[str], env: Mapping[str, str]
) -> None:
    """Launch ``command`` in the background without tying up the console."""

    devnull = subprocess.DEVNULL
    kwargs: dict[str, Any] = {
        "stdin": devnull,
        "stdout": devnull,
        "stderr": devnull,
        "env": env,
        "cwd": Path(__file__).resolve().parent,
    }
    if os.name == "nt":
        creation_flags = 0
        creation_flags |= getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        creation_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = creation_flags
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def _spawn_detached_gui(argv: list[str], launch: LaunchRequest) -> bool:
    """Attempt to hand GUI launch to a detached interpreter.

    Returning ``True`` signals that the GUI is already running in a new
    process, allowing the bootstrapper to exit so terminals close cleanly on
    macOS, Windows and Linux alike.
    """

    if not launch.detach_gui:
        return False

    program_logger = _ensure_program_logging()
    program_logger.info(
        "Attempting to detach GUI: detach flag=%s, argv=%s",
        launch.detach_gui,
        argv,
    )

    env = os.environ.copy()
    env["COPERNICAN_DETACH_GUI"] = "0"
    command_tail = list(argv)
    failures: list[str] = []
    for candidate in _gui_executable_candidates():
        program_logger.debug("Trying GUI detachment candidate: %s", candidate)
        cmd = [str(candidate), str(Path(__file__).resolve()), *command_tail]
        try:
            _launch_detached_process(cmd, env)
            console.write(
                "Handed GUI startup to a detached Copernican process; "
                "closing the launcher terminal."
            )
            program_logger.info(
                "Detached GUI launched with %s", candidate
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive guard
            failures.append(f"{candidate}: {exc}")
            program_logger.warning(
                "Failed to detach GUI with %s: %s", candidate, exc
            )
    if failures:
        console.write(
            "Falling back to inline GUI startup because detaching failed: "
            + "; ".join(failures),
            error=True,
        )
        program_logger.warning(
            "All GUI detachment attempts failed: %s", "; ".join(failures)
        )
    return False


def launch_gui() -> None:
    """Start the GUI scaffold and log the shared orchestration services.

    Legacy behaviour printed the orchestration descriptor list so GUI wrappers
    could discover the available entry points without triggering the CLI.  The
    Tkinter scaffold keeps that behaviour while providing a navigation rail,
    Run Builder and monitoring shells.  When Tk is unavailable the scaffold
    continues in headless mode so CI can validate navigation logic without a
    display server.
    """

    program_logger = _ensure_program_logging()
    program_logger.info(
        "GUI mode requested inline; detach flag=%s, Tcl=%s, Tk=%s",
        _launch_args.detach_gui if _launch_args else None,
        os.getenv("TCL_LIBRARY"),
        os.getenv("TK_LIBRARY"),
    )
    log_mod.log_environment_info()

    service_map = orchestration.describe_orchestration_services()
    console.write("GUI mode requested. Shared services available:")
    program_logger.info("Describing orchestration services for GUI mode.")
    for descriptor in (
        service_map.config_validation,
        service_map.manifest_generation,
        service_map.run_control,
    ):
        entrypoints = ", ".join(descriptor.entrypoints)
        console.write(
            f"- {descriptor.name}: {descriptor.module} ({entrypoints})"
        )
        program_logger.info(
            "Service available: %s (%s) entries %s",
            descriptor.name,
            descriptor.module,
            descriptor.entrypoints,
        )
        console.write(f"  {descriptor.rationale}")
    console.write(
        "Use copernican_lib.orchestration.RunController implementations to "
        "request runs, pause or resume sampling and stream status updates "
        "from the existing orchestration pipeline."
    )
    gui = CopernicanGUI(render=True)
    gui.show_home()
    gui.run()
    program_logger.info("GUI run loop completed; exiting inline GUI mode.")


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



def main_workflow(manifest_path: Path | None = None):
    opts = cli_dependencies.get_runtime_options()
    cli_dependencies.check_dependencies()
    if opts.run_tests:
        success = cli_dependencies.run_startup_tests()
        exit_clean(0 if success else 1)

    global np, plt, mp, model_spec_validator, model_coder
    global engine_plugin_validation, dataset_registry
    global utils, error_handler, log_mod, logger
    np, plt, mp = cli_dependencies.load_third_party_modules()
    from copernican_lib import (
        engine_plugin_validation,
        dataset_registry,
        error_handler,
        model_coder,
        model_spec_validator,
        run_manifest,
        run_executor,
        utils,
    )

    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()

    output_root = (
        _launch_args.output_dir
        if _launch_args and _launch_args.output_dir is not None
        else Path(SCRIPT_DIR) / "output"
    )
    OUTPUT_BASE_DIR = str(Path(output_root))
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    program_logger = _ensure_program_logging()
    program_logger.info(
        "CLI output base directory initialised at %s", OUTPUT_BASE_DIR
    )
    progress_callback = _build_gui_progress_callback()

    if manifest_path is None:
        console.write(
            "Manifest-driven configuration required. Use --manifest.",
            error=True,
        )
        return

    manifest = run_manifest.load_manifest(str(manifest_path))
    run_executor.execute_run_from_manifest(
        manifest,
        script_dir=Path(SCRIPT_DIR),
        output_root=Path(OUTPUT_BASE_DIR),
        progress_callback=progress_callback,
        strict_warnings=opts.strict_warnings,
    )
def _handle_cli_exception(exc: Exception) -> None:
    logger_obj = log_mod.get_logger() if log_mod else None
    if logger_obj and logger_obj.hasHandlers():
        logger_obj.critical(
            "Unhandled exception in main_workflow!",
            exc_info=exc,
        )
    else:
        console.write("CRITICAL UNHANDLED EXCEPTION IN MAIN WORKFLOW:")
        import traceback

        traceback.print_exc()


def _finalize_cli_run() -> None:
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


def _run_cli_launch(
    launch_request: LaunchRequest,
    argv: Iterable[str] | None,
) -> int:
    if not launch_request.manifest_path:
        missing_manifest_message = (
            "Copernican CLI requires a manifest file. Use --manifest to point"
            " at a saved run configuration."
        )
        console.write(missing_manifest_message, error=True)
        return 1
    try:
        main_workflow(manifest_path=launch_request.manifest_path)
        return 0
    except Exception as exc:  # pragma: no cover - deferred logging
        _handle_cli_exception(exc)
        return 1
    finally:
        _finalize_cli_run()


def main(argv: Iterable[str] | None = None) -> int:
    import multiprocessing as _mp

    _mp.freeze_support()
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    launch_request = _parse_launch_args(
        list(argv) if argv is not None else None
    )
    global _launch_args, _legacy_stage_menu_override
    _launch_args = launch_request
    _legacy_stage_menu_override = launch_request.legacy_stage_menu
    if launch_request.mode is orchestration.LaunchMode.GUI:
        if _spawn_detached_gui(
            list(argv) if argv is not None else [], launch_request
        ):
            return 0
        launch_gui()
        return 0
    return _run_cli_launch(launch_request, argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# fmt: on
