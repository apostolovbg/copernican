# Copyright (c) 2025 Copernican developers.
# See LICENSE.md in the repository root for details.

"""Copernican package workflow orchestration.

This module ties together model selection, dataset loading and result
generation while delegating dependency checks and menu rendering to
``copernican.lib.cli`` helpers. The package entrypoint now lives here so the
distribution can run through ``python -m copernican`` and the console script
without a root-level wrapper file.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import importlib
import importlib.util
import logging
import os
import platform
import shutil
import signal
import subprocess  # nosec
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

import copernican.version as version_module
from copernican.lib import analysis as analysis
from copernican.lib import console_output as console
from copernican.lib import logger as log_mod
from copernican.lib import orchestration, progress_state, run_manifest
from copernican.lib import settings as settings_mod
from copernican.lib import utils
from copernican.lib.cli import dependencies as cli_dependencies

# Verify interpreter version early so users see clear feedback
MIN_PYTHON = (3, 11)

# Initialise optional heavy imports so cleanup code can reference them safely
# even when the dependency guard exits early.
numpy_module = None
plt = None
multiprocessing_module = None


def exit_clean(code: int = 0) -> None:
    """Exit the program after printing a newline."""
    console.write("")
    sys.exit(code)


if sys.version_info < MIN_PYTHON:
    console.write(
        (
            f"ERROR: Copernican requires Python {MIN_PYTHON[0]}"
            f".{MIN_PYTHON[1]} or later."
        ),
        error=True,
    )
    exit_clean(1)

# Require execution from the repository's virtual environment so that global
# site-packages are ignored.
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VENV = (REPO_ROOT / ".venv").resolve()
current_venv = os.environ.get("VIRTUAL_ENV")
allow_direct = os.environ.get("COPERNICAN_ALLOW_DIRECT") == "1"
if not allow_direct and (
    current_venv is None or Path(current_venv).resolve() != EXPECTED_VENV
):
    console.write(
        (
            "ERROR: Run Copernican via `python -m copernican` inside the "
            "managed `.venv`."
        ),
        error=True,
    )
    exit_clean(1)

# Enable low-level stack tracing so crashes reveal their origin.
faulthandler.enable()

# Heavy imports are deferred to ``copernican.lib.cli.dependencies`` so startup
# stays quick and dependency checks run before NumPy or Matplotlib load.

# Retrieve the runtime version from installed package metadata. When the
# distribution is not installed, the helper below returns ``"0+unknown"`` so
# logs and plot footers still carry a version-like identifier.  Importing the
# attribute lazily avoids the ``ImportError`` seen on some macOS systems where
# ``copernican.version`` was importable but ``get_version`` was not
# exported.


def _copernican_version() -> str:
    """Return the Copernican version while tolerating missing helpers.

    The launcher crashed on macOS when the package entrypoint re-imported
    this module and ``copernican.version.get_version`` was absent even
    though the version module itself was available. Looking up the
    attribute at
    runtime allows the menu to load successfully and matches the fallbacks
    inside :func:`copernican.version.get_version`.
    """

    getter = getattr(version_module, "get_version", None)
    if callable(getter):
        return getter()
    return "0+unknown"


SCRIPT_DIR = str(REPO_ROOT)

MPL_CONFIG_DIR = Path(SCRIPT_DIR) / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_LOG_FILE = None
_dataset_registry: Any | None = None
_launch_args: LaunchRequest | None = None
_MODEL_SUFFIXES = (".yml", ".yaml")


def _get_dataset_registry():
    """Lazily import the dataset registry after logging is ready."""

    global _dataset_registry
    if _dataset_registry is None:
        from copernican.lib import dataset_registry as registry_module

        _dataset_registry = registry_module
    return _dataset_registry


def _build_gui_progress_callback() -> (
    Callable[[dict[str, object]], None] | None
):
    """Return a callable that records GUI progress updates when requested."""

    path_value = os.environ.get("COPERNICAN_GUI_PROGRESS_PATH")
    if not path_value:
        return None
    target = Path(path_value)

    def _callback(record: dict[str, object]) -> None:
        """Persist GUI progress updates to the configured file."""
        payload = dict(record)
        payload.setdefault("timestamp", utils.get_timestamp())
        try:
            progress_state.record_progress(target, payload)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive logging
            logger = log_mod.get_logger()
            logger.debug("Failed to update GUI progress state: %s", exc)

    return _callback


COPERNICAN_VERSION = _copernican_version()
CURRENT_LOG_FILE = None
_launch_args: LaunchRequest | None = None


@dataclass
class LaunchRequest:
    """Parsed launcher arguments shared across CLI and GUI flows."""

    mode: orchestration.LaunchMode
    detach_gui: bool
    manifest_path: Path | None
    output_dir: Path | None
    catalogue_summary: bool = False
    revalidate_dataset: str | None = None
    list_manifests: bool = False
    show_manifest_path: Path | None = None
    run_validation: bool = False
    analysis_summary_dir: Path | None = None
    analysis_summary_output: Path | None = None
    analysis_summary_formats: tuple[str, ...] = field(
        default_factory=lambda: ("yml", "json")
    )
    analysis_compare_dirs: tuple[Path, Path] | None = None
    analysis_compare_output: Path | None = None
    analysis_posterior_dir: Path | None = None
    analysis_posterior_file: Path | None = None
    analysis_posterior_output: Path | None = None


def _data_root() -> Path:
    """Return the repository's canonical data directory."""

    return Path(SCRIPT_DIR) / "copernican" / "datasets"


def _models_root() -> Path:
    """Return the path where YAML models are stored."""

    return Path(SCRIPT_DIR) / "copernican" / "models"


def _engines_root() -> Path:
    """Return the path containing computational engine modules."""

    return Path(SCRIPT_DIR) / "copernican" / "engines"


def _output_root(override: Path | None = None) -> Path:
    """Return the output directory, optionally using an override."""

    if override is not None:
        return override
    return Path.home() / "copernican_output"


def _parser_path_for_dir(data_dir: Path) -> Path | None:
    """Locate the parser module belonging to a dataset directory."""

    candidates = sorted(data_dir.glob("cosmo_parser_*.py"))
    return candidates[0] if candidates else None


def _relative_parser_key(
    parser_path: Path | None,
    data_root: Path,
) -> str | None:
    """Normalize the parser path relative to the shared data root."""

    if parser_path is None:
        return None
    try:
        rel = parser_path.relative_to(data_root)
    except ValueError:
        rel = Path(os.path.relpath(parser_path, data_root))
    return str(rel).replace("\\", "/")


def _collect_dataset_entries(
    data_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return every dataset entry along with parser trust validation notes."""

    root = data_root or _data_root()
    registry_module = _get_dataset_registry()
    registry_module.discover_trusted_parsers(str(root))
    registries = registry_module.get_parser_registries()
    entries: list[dict[str, Any]] = []
    notes: list[str] = []
    for dtype, registry in registries.items():
        for dataset_id, entry in registry.items():
            data_dir = entry.get("data_dir")
            if not data_dir:
                continue
            data_dir_path = Path(data_dir)
            parser_path = _parser_path_for_dir(data_dir_path)
            rel_key = _relative_parser_key(parser_path, root)
            expected_digest = (
                registry_module.TRUSTED_PARSER_DIGESTS.get(rel_key)
                if rel_key
                else None
            )
            parser_digest = (
                registry_module._file_sha256(str(parser_path))
                if parser_path
                else ""
            )
            parser_trusted = bool(
                expected_digest and parser_digest == expected_digest
            )
            if not parser_trusted:
                descriptor = rel_key or (
                    str(parser_path) if parser_path else "parser"
                )
                notes.append(
                    (
                        f"Parser {descriptor} failed trust validation; "
                        "verify digests."
                    )
                )
            entries.append(
                {
                    "id": dataset_id,
                    "type": dtype,
                    "name": entry.get("dataset_name", dataset_id),
                    "parser_trusted": parser_trusted,
                    "parser_digest": parser_digest,
                    "expected_digest": expected_digest,
                }
            )
    return entries, notes


def _gather_catalogue_summary(
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Summarize discovered datasets and parser health for CLI output."""

    catalogue, notes = _collect_dataset_entries(data_root)
    type_counter = Counter(entry["type"].upper() for entry in catalogue)
    untrusted = [entry for entry in catalogue if not entry["parser_trusted"]]
    return {
        "dataset_count": len(catalogue),
        "type_counter": type_counter,
        "untrusted": untrusted,
        "notes": notes,
        "entries": catalogue,
    }


def _read_model_file(path: Path) -> dict[str, Any]:
    """Safely parse a YAML model definition and warn on syntax failures."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except yaml.YAMLError:
        logger = log_mod.get_logger()
        logger.warning("Model metadata in %s is malformed", path)
        return {}


def _collect_model_index(
    models_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a metadata index for every YAML model in the models folder."""

    root = models_root or _models_root()
    models: dict[str, dict[str, Any]] = {}
    for pattern in ("*.yml", "*.yaml"):
        for path in sorted(Path(root).glob(pattern)):
            if path.name.startswith("__"):
                continue
            meta = _read_model_file(path)
            parameters = meta.get("parameters") or []
            compatibility = {
                "sne": True,
                "bao": meta.get("valid_for_bao", True),
                "cmb": meta.get("valid_for_cmb", True),
            }
            badges = [
                name.upper() for name, valid in compatibility.items() if valid
            ]
            models[path.stem] = {
                "id": meta.get("model_name", path.stem),
                "filename": path.name,
                "path": str(path),
                "citation": meta.get("citation", ""),
                "license": meta.get(
                    "license",
                    "Copernican default license; add model notes",
                ),
                "version": meta.get("version", "unknown"),
                "badges": badges,
                "hash": utils.compute_sha256(str(path)),
                "parameter_count": len(parameters),
            }
    return models


def _collect_engine_index(
    engines_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build metadata records for each engine module."""

    root = engines_root or _engines_root()
    engines: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).glob("*.py")):
        if path.name.startswith("__"):
            continue
        module_name = f"copernican.engines.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            label = getattr(module, "ENGINE_LABEL", path.stem)
            version_label = getattr(module, "ENGINE_VERSION", "unknown")
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
        ):
            module = None
            label = path.stem
            version_label = "unavailable"
            log_mod.get_logger().warning(
                "Engine metadata import failed for %s", module_name
            )
        engines[module_name] = {
            "id": module_name,
            "filename": path.name,
            "path": str(path),
            "stem": path.stem,
            "citation": getattr(module, "__doc__", ""),
            "license": "Copernican default license; verify engines",
            "version": version_label,
            "label": label,
            "badges": ["SNE", "BAO", "CMB"],
            "hash": utils.compute_sha256(str(path)),
        }
    return engines


def _gather_model_engine_summary(
    models_root: Path | None = None,
    engines_root: Path | None = None,
) -> dict[str, Any]:
    """Summarize model and engine compatibility badges plus missing data."""

    model_index = _collect_model_index(models_root)
    engine_index = _collect_engine_index(engines_root)
    model_badges = Counter()
    stale_models: list[tuple[str, str]] = []
    for entry in model_index.values():
        for badge in entry.get("badges", []):
            model_badges[badge.upper()] += 1
        version_label = (entry.get("version") or "").lower()
        if not version_label or version_label in {"unknown", "unavailable"}:
            stale_models.append(
                (
                    entry.get("id") or entry.get("filename", "model"),
                    "missing",
                )
            )
    stale_engines: list[tuple[str, str]] = []
    for entry in engine_index.values():
        version_label = (entry.get("version") or "").lower()
        if not version_label or version_label in {"unknown", "unavailable"}:
            stale_engines.append(
                (
                    entry.get("label") or entry.get("id", "engine"),
                    "missing",
                )
            )
    return {
        "model_count": len(model_index),
        "engine_count": len(engine_index),
        "model_badges": model_badges,
        "stale_models": stale_models,
        "stale_engines": stale_engines,
    }


def _print_catalogue_summary_cli(
    data_root: Path | None = None,
    models_root: Path | None = None,
    engines_root: Path | None = None,
) -> None:
    """Emit the catalogue summary details to the CLI."""

    catalogue = _gather_catalogue_summary(data_root)
    model_engine = _gather_model_engine_summary(models_root, engines_root)
    console.write("")
    console.write("Catalogue summary")
    console.write("------------------")
    console.write(f"Datasets discovered: {catalogue['dataset_count']}")
    if catalogue["type_counter"]:
        console.write("By type:")
        for dtype, count in sorted(catalogue["type_counter"].items()):
            console.write(f"  {dtype}: {count}")
    if catalogue["untrusted"]:
        console.write("Untrusted datasets:")
        for entry in catalogue["untrusted"]:
            console.write(
                (
                    f"  - {entry['id']} ({entry['type']}) requires "
                    "parser hash validation"
                ),
                error=True,
            )
    if not catalogue["untrusted"]:
        console.write("All registered parsers match their trusted digests.")
    if catalogue["notes"]:
        for note in catalogue["notes"][:3]:
            console.write(f"Note: {note}")
    console.write("")
    console.write(
        (
            f"Models discovered: {model_engine['model_count']} | "
            f"Engines: {model_engine['engine_count']}"
        )
    )
    if model_engine["model_badges"]:
        badge_parts = [
            f"{badge}: {count}"
            for badge, count in sorted(model_engine["model_badges"].items())
        ]
        console.write("Model compatibility: " + ", ".join(badge_parts))
    if model_engine["stale_models"]:
        console.write("Models missing version metadata:")
        for name, _reason in model_engine["stale_models"][:5]:
            console.write(f"  - {name}")
    if model_engine["stale_engines"]:
        console.write("Engines missing version metadata:")
        for name, _reason in model_engine["stale_engines"][:5]:
            console.write(f"  - {name}")


def _cli_revalidate_dataset(
    dataset_id: str,
    data_root: Path | None = None,
) -> bool:
    """Recompute and report parser trust for a single dataset identifier."""

    root = data_root or _data_root()
    registry_module = _get_dataset_registry()
    registry_module.discover_trusted_parsers(str(root), force=True)
    registries = registry_module.get_parser_registries()
    target: dict[str, Any] | None = None
    dtype = ""
    for dtype_name, registry in registries.items():
        if dataset_id in registry:
            target = registry[dataset_id]
            dtype = dtype_name
            break
    if target is None:
        console.write(f"Dataset '{dataset_id}' is not registered.", error=True)
        return False
    data_dir = target.get("data_dir")
    if not data_dir:
        console.write(
            f"Dataset '{dataset_id}' has no associated data directory.",
            error=True,
        )
        return False
    parser_path = _parser_path_for_dir(Path(data_dir))
    rel_key = _relative_parser_key(parser_path, root)
    expected_digest = (
        registry_module.TRUSTED_PARSER_DIGESTS.get(rel_key)
        if rel_key
        else None
    )
    parser_digest = (
        registry_module._file_sha256(str(parser_path)) if parser_path else ""
    )
    if expected_digest and parser_digest == expected_digest:
        console.write(
            (
                f"{dataset_id} ({dtype}) parser matches trusted digest "
                f"({parser_digest})."
            )
        )
        return True
    if not expected_digest:
        console.write(
            f"Dataset '{dataset_id}' is missing a trusted digest entry.",
            error=True,
        )
    else:
        console.write(
            (
                f"Digest mismatch for {dataset_id}: expected "
                f"{expected_digest} but observed {parser_digest}."
            ),
            error=True,
        )
    return False


def _discover_manifest_records(
    output_root: Path,
) -> list[tuple[Path, Path]]:
    """Find recent run manifest files inside the output directory."""

    if not output_root.exists():
        return []
    records: list[tuple[Path, Path]] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        manifest_files = sorted(child.glob("run_manifest_*.yml"))
        if not manifest_files:
            continue
        manifest_files.sort(
            key=lambda manifest_file: manifest_file.stat().st_mtime,
            reverse=True,
        )
        records.append((child, manifest_files[0]))
    records.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
    return records


def _print_manifest_listing(output_root: Path) -> None:
    """Present the most recent manifests under the configured output path."""

    records = _discover_manifest_records(output_root)
    console.write("")
    console.write(f"Run manifests under {output_root}:")
    if not records:
        console.write("  (no manifest directories found)")
        return
    for directory, manifest_path in records[:20]:
        timestamp = datetime.datetime.fromtimestamp(
            manifest_path.stat().st_mtime,
            datetime.timezone.utc,
        ).isoformat()
        console.write(
            f"  - {directory.name}: {manifest_path.name} "
            f"(mtime {timestamp})"
        )
    if len(records) > 20:
        console.write(
            (
                f"  … {len(records) - 20} additional run folder(s) hidden; "
                "clean up old runs if needed."
            )
        )


def _print_manifest_file(path: Path) -> bool:
    """Load and pretty-print a manifest file, returning success."""

    if not path.exists():
        console.write(f"Manifest not found at {path}", error=True)
        return False
    try:
        manifest_data = run_manifest.load_manifest(str(path))
    except (
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        console.write(f"Failed to load manifest {path}: {exc}", error=True)
        return False
    console.write("")
    console.write(f"Manifest {path}:")
    console.write(yaml.safe_dump(manifest_data, sort_keys=False))
    return True


def _run_validation_cli() -> bool:
    """Execute the lightweight validation suite and summarize results."""

    from copernican import validation as validation_utils

    repo_root = REPO_ROOT
    try:
        code, summary = validation_utils.run_validation_suite(
            script_dir=repo_root
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        code = 1
        summary = f"Validation runner could not start: {exc}"
        console.write(summary, error=True)
    console.write("")
    console.write("Validation summary")
    console.write("------------------")
    if summary:
        for line in summary.splitlines():
            console.write(line)
    validation_utils.write_validation_summary(summary, code == 0)
    console.write(
        ("Saved validation report to " f"{validation_utils.VALIDATION_FILE}")
    )
    return code == 0


def _run_analysis_summary_cli(
    run_dir: Path,
    output_dir: Path | None,
    formats: tuple[str, ...],
) -> bool:
    """Print a run summary and optionally write the structured files."""

    if not run_dir.exists():
        console.write(f"Run directory not found: {run_dir}", error=True)
        return False
    if not run_dir.is_dir():
        console.write(f"Expected a directory but found: {run_dir}", error=True)
        return False
    try:
        result = analysis.analyze_run(run_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        console.write(f"Failed to analyse run {run_dir}: {exc}", error=True)
        return False

    console.write("")
    console.write("Run analysis summary")
    console.write("--------------------")
    console.write(analysis.format_run_summary_text(result))

    if output_dir:
        try:
            saved = analysis.save_run_summary(
                run_dir,
                output_dir,
                formats=formats,
                result=result,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            console.write(f"Failed to export run summary: {exc}", error=True)
            return False
        console.write(
            "Summary exports:\n"
            + "\n".join(f"  - {path}" for path in saved.values())
        )
    return True


def _run_analysis_compare_cli(
    base_dir: Path,
    alt_dir: Path,
    output_dir: Path | None,
    formats: tuple[str, ...],
) -> bool:
    """Compare two runs and print the delta summary."""

    if not base_dir.is_dir() or not alt_dir.is_dir():
        console.write(
            "Both base and alternative run directories must exist.",
            error=True,
        )
        return False
    try:
        base_result = analysis.analyze_run(base_dir)
        alt_result = analysis.analyze_run(alt_dir)
        comparison = analysis.compare_runs(base_result, alt_result)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        console.write(f"Comparison failed: {exc}", error=True)
        return False

    console.write("")
    console.write("Analysis comparison")
    console.write("-------------------")
    console.write(yaml.safe_dump(comparison, sort_keys=False))

    if output_dir:
        try:
            saved = analysis.save_comparison_summary(
                base_result,
                alt_result,
                output_dir,
                formats=formats,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            console.write(
                f"Failed to export comparison summary: {exc}", error=True
            )
            return False
        console.write(
            "Comparison exports:\n"
            + "\n".join(f"  - {path}" for path in saved.values())
        )
    return True


def _run_analysis_posterior_cli(
    run_dir: Path,
    posterior_file: Path | str | None,
    output_file: Path | str | None,
) -> bool:
    """Generate a posterior overview plot for a run directory."""

    if not run_dir.is_dir():
        console.write(
            "Posterior analysis requires an existing run folder.", error=True
        )
        return False

    try:
        result = analysis.analyze_run(run_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        console.write(
            f"Failed to analyse run for posterior: {exc}", error=True
        )
        return False

    output_dest = None
    output_directory = run_dir
    if output_file:
        provided = Path(output_file)
        if provided.suffix.lower() == ".png":
            output_dest = provided
            output_directory = provided.parent
        else:
            output_directory = provided
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        saved = analysis.plot_posterior(
            run_dir,
            output_dir=output_directory,
            posterior_file=posterior_file,
            kinds=("overview", "corner", "histograms"),
            result=result,
            overview_path=output_dest,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        console.write(f"Failed to render posterior: {exc}", error=True)
        return False

    for label, path in saved.items():
        console.write(f"{label.title()} plot saved to {path}")
    return True


def _handle_auxiliary_requests(
    launch_request: LaunchRequest,
) -> tuple[bool, int]:
    """Process auxiliary CLI options such as analysis helpers or validation."""

    handled = False
    exit_code = 0
    data_root = _data_root()
    models_root = _models_root()
    engines_root = _engines_root()
    output_root = _output_root(launch_request.output_dir)
    if launch_request.analysis_summary_dir is not None:
        success = _run_analysis_summary_cli(
            launch_request.analysis_summary_dir,
            launch_request.analysis_summary_output,
            launch_request.analysis_summary_formats,
        )
        handled = True
        if not success:
            exit_code = 1
        return handled, exit_code
    if launch_request.analysis_compare_dirs is not None:
        base_dir, alt_dir = launch_request.analysis_compare_dirs
        success = _run_analysis_compare_cli(
            base_dir,
            alt_dir,
            launch_request.analysis_compare_output,
            launch_request.analysis_summary_formats,
        )
        handled = True
        if not success:
            exit_code = 1
        return handled, exit_code
    if launch_request.analysis_posterior_dir is not None:
        success = _run_analysis_posterior_cli(
            launch_request.analysis_posterior_dir,
            launch_request.analysis_posterior_file,
            launch_request.analysis_posterior_output,
        )
        handled = True
        if not success:
            exit_code = 1
        return handled, exit_code
    if launch_request.run_validation:
        success = _run_validation_cli()
        handled = True
        if not success:
            exit_code = 1
        return handled, exit_code
    if launch_request.catalogue_summary:
        _print_catalogue_summary_cli(data_root, models_root, engines_root)
        handled = True
    if launch_request.revalidate_dataset:
        dataset_id = launch_request.revalidate_dataset.strip()
        if dataset_id:
            success = _cli_revalidate_dataset(dataset_id, data_root)
            handled = True
            if not success:
                exit_code = 1
        else:
            console.write(
                "--revalidate-dataset requires a non-empty dataset id.",
                error=True,
            )
            handled = True
            exit_code = 1
    if launch_request.list_manifests:
        _print_manifest_listing(output_root)
        handled = True
    if launch_request.show_manifest_path is not None:
        success = _print_manifest_file(launch_request.show_manifest_path)
        handled = True
        if not success:
            exit_code = 1
    return handled, exit_code


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
            with open(CURRENT_LOG_FILE, "a", encoding="utf-8") as file_handle:
                file_handle.write(msg + "\n")
                faulthandler.dump_traceback(file=file_handle, all_threads=True)
        except (OSError, ValueError) as exc:
            if logger:
                # Preserve the failure details in the central log.
                logger.exception(
                    "copernican: failed to append fatal trace to %s",
                    CURRENT_LOG_FILE,
                )
            else:
                # Fallback so the issue is still visible to the user.
                console.write(
                    f"copernican: failed to write to {CURRENT_LOG_FILE}:"
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
            "per-user ~/copernican_output folder."
        ),
    )
    parser.add_argument(
        "--catalogue-summary",
        action="store_true",
        help=(
            "Print dataset, model and engine inventory health information "
            "then exit."
        ),
    )
    parser.add_argument(
        "--revalidate-dataset",
        metavar="DATASET_ID",
        help=(
            "Re-run the parser trust check for the specified dataset id "
            "and report the digest status."
        ),
    )
    parser.add_argument(
        "--list-manifests",
        action="store_true",
        help=(
            "List run directories under ~/copernican_output along with "
            "their most recent manifest files."
        ),
    )
    parser.add_argument(
        "--show-manifest",
        metavar="MANIFEST_PATH",
        help="Pretty-print the specified manifest file and exit.",
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help=(
            "Execute the lightweight validation suite "
            "and record the summary."
        ),
    )
    parser.add_argument(
        "--analysis-summary",
        metavar="RUN_DIR",
        help=(
            "Inspect a run directory, print its summary, "
            "and optionally export it."
        ),
    )
    parser.add_argument(
        "--analysis-summary-output",
        metavar="OUTPUT_DIR",
        help="Directory where structured summary files will be written.",
    )
    parser.add_argument(
        "--analysis-summary-formats",
        metavar="FORMATS",
        default="yml,json",
        help=(
            "Comma-separated formats (yml,json) for exported run summaries."
        ),
    )
    parser.add_argument(
        "--analysis-compare",
        nargs=2,
        metavar=("BASE_DIR", "ALT_DIR"),
        help="Compare two run directories and report χ²/parameter deltas.",
    )
    parser.add_argument(
        "--analysis-compare-output",
        metavar="OUTPUT_DIR",
        help="Directory for analysis comparison summary files.",
    )
    parser.add_argument(
        "--analysis-posterior",
        metavar="RUN_DIR",
        help=(
            "Render a posterior overview plot for the selected run directory."
        ),
    )
    parser.add_argument(
        "--analysis-posterior-file",
        metavar="POSTERIOR_FILE",
        help=("Posterior NetCDF file (relative to the run) to visualise."),
    )
    parser.add_argument(
        "--analysis-posterior-output",
        metavar="OUTPUT_FILE",
        help=(
            "Destination PNG for the posterior overview "
            "(defaults inside run)."
        ),
    )
    argv_list = list(argv) if argv is not None else None
    parsed, _ = parser.parse_known_args(argv_list)
    settings_data = settings_mod.get_settings()
    gui_defaults = settings_data.get("gui", {})
    default_detach = bool(gui_defaults.get("detach_gui", True))
    detach_env = os.environ.get("COPERNICAN_DETACH_GUI")
    detach_gui = (
        detach_env != "0" if detach_env is not None else default_detach
    )
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
    manifest_display = (
        Path(parsed.show_manifest).expanduser().resolve()
        if parsed.show_manifest
        else None
    )
    analysis_summary_dir = (
        Path(parsed.analysis_summary).expanduser().resolve()
        if parsed.analysis_summary
        else None
    )
    analysis_summary_output = (
        Path(parsed.analysis_summary_output).expanduser().resolve()
        if parsed.analysis_summary_output
        else None
    )
    analysis_summary_formatted = tuple(
        fmt.strip().lower()
        for fmt in (parsed.analysis_summary_formats or "yml,json").split(",")
        if fmt.strip()
    )
    if not analysis_summary_formatted:
        analysis_summary_formatted = ("yml", "json")
    analysis_compare_dirs = None
    if parsed.analysis_compare:
        base_path = Path(parsed.analysis_compare[0]).expanduser().resolve()
        alt_path = Path(parsed.analysis_compare[1]).expanduser().resolve()
        analysis_compare_dirs = (base_path, alt_path)
    analysis_compare_output = (
        Path(parsed.analysis_compare_output).expanduser().resolve()
        if parsed.analysis_compare_output
        else None
    )
    analysis_posterior_dir = (
        Path(parsed.analysis_posterior).expanduser().resolve()
        if parsed.analysis_posterior
        else None
    )
    analysis_posterior_file = (
        Path(parsed.analysis_posterior_file).expanduser()
        if parsed.analysis_posterior_file
        else None
    )
    analysis_posterior_output = (
        Path(parsed.analysis_posterior_output).expanduser()
        if parsed.analysis_posterior_output
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
        detach_gui=detach_gui,
        manifest_path=manifest_path,
        output_dir=output_dir,
        catalogue_summary=parsed.catalogue_summary,
        revalidate_dataset=parsed.revalidate_dataset,
        list_manifests=parsed.list_manifests,
        show_manifest_path=manifest_display,
        run_validation=parsed.run_validation,
        analysis_summary_dir=analysis_summary_dir,
        analysis_summary_output=analysis_summary_output,
        analysis_summary_formats=analysis_summary_formatted,
        analysis_compare_dirs=analysis_compare_dirs,
        analysis_compare_output=analysis_compare_output,
        analysis_posterior_dir=analysis_posterior_dir,
        analysis_posterior_file=analysis_posterior_file,
        analysis_posterior_output=analysis_posterior_output,
    )


def _gui_executable_candidates() -> list[Path]:
    """Return possible Python executables suitable for GUI launchers.

    GUI shells should prefer ``pythonw`` on platforms that supply it so the
    calling terminal can close without warning about a lingering console.  The
    managed virtual environment ships a ``pythonw`` binary on Windows, so the
    launcher checks for the variant before falling back to the active
    interpreter.
    """

    # Keep the venv wrapper path so the child preserves the managed env.
    exe_path = Path(sys.executable)
    candidates = [exe_path]
    for suffix in ("pythonw.exe", "pythonw"):
        alt = exe_path.with_name(suffix)
        if alt.exists():
            candidates.insert(0, alt)
    return candidates


def _launch_detached_process(
    command: list[str], env: Mapping[str, str]
) -> subprocess.Popen[Any]:
    """Launch ``command`` in the background without tying up the console."""

    devnull = subprocess.DEVNULL
    kwargs: dict[str, Any] = {
        "stdin": devnull,
        "stdout": devnull,
        "stderr": devnull,
        "env": env,
        "cwd": REPO_ROOT,
    }
    if os.name == "nt":
        creation_flags = 0
        creation_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creation_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = creation_flags
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)  # nosec


def _activate_detached_gui_on_macos(pid: int) -> None:
    """Ask macOS to bring the detached Copernican GUI to the front."""

    if sys.platform != "darwin":
        return
    script = (
        "delay 2\n"
        'tell application "System Events"\n'
        f"  set frontmost of first process whose unix id is {pid} to true\n"
        "end tell"
    )
    try:
        osascript = "/usr/bin/osascript"
        subprocess.Popen(
            [osascript, "-e", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # nosec
    except (OSError, RuntimeError, ValueError):
        pass


def _detached_gui_env() -> dict[str, str]:
    """Return the GUI child environment with bundled Tcl/Tk paths."""

    env = os.environ.copy()
    bundle_lib = REPO_ROOT / ".python" / "lib"
    tcl_dir = bundle_lib / "tcl8.6"
    tk_dir = bundle_lib / "tk8.6"
    if tcl_dir.exists():
        env["TCL_LIBRARY"] = str(tcl_dir)
    if tk_dir.exists():
        env["TK_LIBRARY"] = str(tk_dir)
    env["COPERNICAN_DETACH_GUI"] = "0"
    return env


def _spawn_detached_gui(argv: list[str], launch: LaunchRequest) -> bool:
    """Attempt to hand GUI launch to a detached interpreter.

    Returning ``True`` signals that the GUI is already running in a new
    process, allowing the bootstrapper to exit so terminals close cleanly on
    macOS, Windows and Linux alike.
    """

    if not launch.detach_gui:
        return False

    app_logger = log_mod.get_logger()
    app_logger.info(
        "Attempting to detach GUI: detach flag=%s, argv=%s",
        launch.detach_gui,
        argv,
    )

    env = _detached_gui_env()
    command_tail = list(argv)
    failures: list[str] = []
    for candidate in _gui_executable_candidates():
        app_logger.debug("Trying GUI detachment candidate: %s", candidate)
        # Re-enter through the package entrypoint so imports resolve in the
        # detached child the same way they do for the normal launcher.
        cmd = [str(candidate), "-m", "copernican", *command_tail]
        try:
            detached_proc = _launch_detached_process(cmd, env)
            _activate_detached_gui_on_macos(detached_proc.pid)
            app_logger.info("Detached GUI launched with %s", candidate)
            console.write(
                "Handed GUI startup to a detached Copernican process; "
                "closing the launcher terminal."
            )
            return True
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:  # pragma: no cover - defensive guard
            failures.append(f"{candidate}: {exc}")
            app_logger.warning(
                "Failed to detach GUI with %s: %s", candidate, exc
            )
    if failures:
        console.write(
            "Falling back to inline GUI startup because detaching failed: "
            + "; ".join(failures),
            error=True,
        )
        app_logger.warning(
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

    app_logger = log_mod.get_logger()
    from copernican.lib.gui import CopernicanGUI

    app_logger.info(
        "GUI mode requested inline; detach flag=%s, Tcl=%s, Tk=%s",
        _launch_args.detach_gui if _launch_args else None,
        os.getenv("TCL_LIBRARY"),
        os.getenv("TK_LIBRARY"),
    )

    service_map = orchestration.describe_orchestration_services()
    console.write("GUI mode requested. Shared services available:")
    app_logger.info("Describing orchestration services for GUI mode.")
    for descriptor in (
        service_map.config_validation,
        service_map.manifest_generation,
        service_map.run_control,
    ):
        entrypoints = ", ".join(descriptor.entrypoints)
        console.write(
            f"- {descriptor.name}: {descriptor.module} ({entrypoints})"
        )
        app_logger.info(
            "Service available: %s (%s) entries %s",
            descriptor.name,
            descriptor.module,
            descriptor.entrypoints,
        )
        console.write(f"  {descriptor.rationale}")
    console.write(
        "Use copernican.lib.orchestration.RunController implementations to "
        "request runs, pause or resume sampling and stream status updates "
        "from the existing orchestration pipeline."
    )
    gui = CopernicanGUI(render=True)
    gui.show_home()
    gui.run()
    app_logger.info("GUI run loop completed; exiting inline GUI mode.")


def _remove_run_dir(path: str) -> None:
    """Delete the run output directory and its contents."""
    if path and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            console.write(f"Removed run directory {path}")
        except OSError as exc:
            if logger:
                logger.warning(
                    "copernican: could not remove run dir %s: %s",
                    path,
                    exc,
                )
            else:
                console.write(
                    f"copernican: could not remove run dir {path}: {exc}",
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
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if logger:
            # ``psutil`` is optional; log and continue with unknown frequency.
            logger.warning(
                "copernican: psutil unavailable for CPU freq: %s",
                exc,
            )
        else:
            # Without a logger, surface the issue via the console.
            console.write(
                f"copernican: psutil import failed: {exc}",
                error=True,
            )
    if freq is None and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as file_handle:
                for line in file_handle:
                    if line.startswith("model name") and cpu == "Unknown CPU":
                        cpu = line.split(":", 1)[1].strip()
                    if line.startswith("cpu MHz") and freq is None:
                        freq = float(line.split(":", 1)[1]) / 1000.0
        except (OSError, UnicodeError, ValueError) as exc:
            if logger:
                # Reading ``/proc/cpuinfo`` can fail in restricted
                # environments; log and fall back to placeholders.
                logger.warning(
                    "copernican: could not read /proc/cpuinfo: %s",
                    exc,
                )
            else:
                # Fallback console message when the logger is unavailable.
                console.write(
                    f"copernican: cannot read /proc/cpuinfo: {exc}",
                    error=True,
                )
    freq_str = f"{freq:.2f} GHz" if freq else "Unknown GHz"
    return cpu, freq_str


# The high-level workflow is broken into small helper functions below. Each
# helper is documented in plain language so non-programmers can follow the
# logic of the program.


def main_workflow(manifest_path: Path | None = None):
    """Execute the manifest-driven CLI workflow after dependency checks."""
    opts = cli_dependencies.get_runtime_options()
    if opts.run_tests:
        success = cli_dependencies.run_startup_tests()
        exit_clean(0 if success else 1)

    global numpy_module, plt, multiprocessing_module, model_spec_validator
    global model_coder
    global engine_plugin_validation
    global utils, error_handler, log_mod, logger
    numpy_module, plt, multiprocessing_module = (
        cli_dependencies.load_third_party_modules()
    )
    import copernican.lib.engine_adapter as engine_plugin_validation
    from copernican.lib import (
        error_handler,
        model_coder,
        model_spec_validator,
        run_executor,
        run_manifest,
        utils,
    )

    try:
        SCRIPT_DIR = str(REPO_ROOT)
    except NameError:
        SCRIPT_DIR = os.getcwd()

    launch_output_dir = (
        getattr(_launch_args, "output_dir", None) if _launch_args else None
    )
    output_root = _output_root(launch_output_dir)
    OUTPUT_BASE_DIR = str(Path(output_root))
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    app_logger = log_mod.get_logger()
    app_logger.info(
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
    """Log and display unexpected exceptions from the CLI."""
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
    """Display any remaining Matplotlib figures and print a separator."""

    if plt is not None and hasattr(plt, "get_fignums") and plt.get_fignums():
        console.write(
            "\nDisplaying plot(s). Close plot window(s) to exit script "
            "fully."
        )
        try:
            plt.show(block=True)
        except (OSError, RuntimeError, TypeError, ValueError) as e_show:
            console.write(f"Error during final plt.show(): {e_show}")
    console.write("")


def _run_cli_launch(
    launch_request: LaunchRequest,
    argv: Iterable[str] | None,
) -> int:
    """Run the CLI path when a manifest is provided, logging failures."""

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
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - deferred logging
        _handle_cli_exception(exc)
        return 1
    finally:
        _finalize_cli_run()


def _ensure_managed_environment() -> None:
    """Abort if the interpreter is not the managed Copernican `.venv`."""

    if Path(sys.prefix).resolve().name != ".venv":
        console.write(
            (
                "ERROR: Run Copernican via `python -m copernican` so the "
                "managed `.venv` is prepared automatically."
            ),
            error=True,
        )
        exit_clean(1)


def _announce_program_start(
    launch_request: LaunchRequest, logger: logging.Logger
) -> None:
    """Print the launcher banner, environment metadata and sanity check."""

    _ensure_managed_environment()
    console.write("")
    console.write("Copernican has initialised.")
    console.write(f"Version: {COPERNICAN_VERSION}")
    mode_label = getattr(launch_request.mode, "name", str(launch_request.mode))
    console.write(f"Launch mode: {mode_label}")
    console.write(f"Python interpreter: {sys.executable}")
    console.write(f"Working directory: {Path.cwd()}")
    cpu_model, freq_str = _get_cpu_info()
    console.write(f"Hardware: {cpu_model} @ {freq_str}")
    log_mod.log_environment_info(console=True)
    console.write(
        "Sanity check: dependency provisioning is handled by the managed "
        "environment. Rebuild `.venv` if packages go missing."
    )
    logger.info(
        "Sanity check noted: the managed environment handles dependencies "
        "without an in-process check."
    )


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point that decides whether to run GUI or CLI workflows."""
    import multiprocessing as multiprocessing_module

    multiprocessing_module.freeze_support()
    try:
        multiprocessing_module.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    launch_request = _parse_launch_args(
        list(argv) if argv is not None else None
    )
    global _launch_args
    _launch_args = launch_request
    app_logger = log_mod.get_logger()
    aux_handled, aux_exit = _handle_auxiliary_requests(launch_request)
    if aux_handled:
        return aux_exit
    _announce_program_start(launch_request, app_logger)
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
