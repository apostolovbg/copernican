"""Tkinter-based GUI scaffold with headless fallbacks.

The GUI keeps a persistent navigation rail and modular content panes so the
Copernican Suite can expose future workflows without reworking the layout.
Screen-reader friendly labels and keyboard shortcuts mirror each navigation
item to keep the flow efficient even when the mouse is unavailable.  The
implementation is intentionally light on dependencies so users can experiment
with the GUI shell inside the managed virtual environment without installing
extra frameworks.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Mapping, Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - executed only when Tk is missing
    tk = None
    ttk = None
    filedialog = None
    messagebox = None

import yaml

from copernican_lib import (
    console_output,
    dataset_registry,
    logger,
    progress_state,
    run_manifest,
    utils,
    version,
)
from copernican_lib.engine_capabilities import (
    EngineCapabilities,
    EngineSetting,
    get_engine_capabilities,
)
from copernican_lib.run_lifecycle import (
    ManifestWorkspace,
    create_manifest_workspace,
    delete_manifest_workspace,
    finalize_run_workspace,
)

log_mod = logger

_PROGRESS_SPINNER_CHARS = frozenset("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
_NAV_PANE_WIDTH = 140
_LOGO_PADDING = 12
_LOGO_SIDE = _NAV_PANE_WIDTH // 4


def _is_progress_line(line: str) -> bool:
    """Return True for stdout lines that stream the CLI progress bar."""

    stripped = line.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if "progress:" in lower and "batch" in lower:
        return True
    if any(char in stripped for char in _PROGRESS_SPINNER_CHARS):
        return True
    return False


class RunStatus(Enum):
    """Enumerate the run lifecycle states shown in the status strip."""

    IDLE = "Idle"
    CONFIGURING = "Configuring"
    RUNNING = "Running"
    PAUSED = "Paused"
    CANCELLED = "Cancelled"
    ABORTED = "Aborted"


@dataclass
class RunDraft:
    """Persist partially completed Run Builder selections."""

    seed: str = ""
    model: str = ""
    data: str = ""
    engine: str = ""
    plan: str = ""
    notes: str = ""
    completed_step: int = 0
    walkers: str = ""
    burn_in: str = ""
    production_steps: str = ""
    pool_size: str = ""


@dataclass
class NavigationItem:
    """Describe a navigation rail entry and its callback."""

    name: str
    label: str
    action: Callable[["CopernicanGUI"], None]


@dataclass
class RunSummary:
    """Capture the artefacts exposed on the completion screen."""

    output_links: list[str] = field(default_factory=list)
    manifest_actions: list[str] = field(default_factory=list)
    manifest_metadata: list[str] = field(default_factory=list)


@dataclass
class LogEntry:
    """Structure log lines for filtering and UI navigation."""

    timestamp: str
    severity: str
    message: str
    anchor: str
    formatted: str


@dataclass
class UIMessage:
    """Track toast or inline alerts with log anchors."""

    message: str
    anchor: str
    severity: str
    context: str


class _MemoryLogHandler(logging.Handler):
    """Capture structured log lines for on-screen diagnostics."""

    def __init__(self, *, prefix: str) -> None:
        super().__init__(level=logging.INFO)
        self.prefix = prefix
        self.entries: list[LogEntry] = []
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        formatter.converter = time.gmtime
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Store the formatted log entry with a stable anchor."""

        anchor = f"{self.prefix}-{len(self.entries) + 1}"
        formatted = self.format(record)
        timestamp = self.formatter.formatTime(record, self.formatter.datefmt)
        entry = LogEntry(
            timestamp=timestamp,
            severity=record.levelname,
            message=record.getMessage(),
            anchor=anchor,
            formatted=formatted,
        )
        self.entries.append(entry)

    def last_anchor(self) -> Optional[str]:
        """Return the most recent anchor if one exists."""

        if not self.entries:
            return None
        return self.entries[-1].anchor


class CopernicanGUI:
    """Build and manage the GUI layout with optional rendering.

    The widget tree is created only when Tkinter is available and the caller
    requests rendering.  When Tk support is absent the controller still keeps
    full state so tests can exercise navigation, builder jumps and summary
    handling without a display server.
    """

    nav_items: list[NavigationItem]
    builder_steps: list[str] = [
        "Seed",
        "Models",
        "Data",
        "Engine",
        "Manifest",
        "Confirm and start",
    ]
    _TEMP_MANIFEST_FOLDER = "copernican_run_NEW_CONFIG"
    _TEMP_MANIFEST_FILE = "run_manifest_NEW_CONFIG.yml"
    _BUILDER_COMPLETION_MESSAGE = (
        "Set all required selections in Seed, Models, Data and Engine "
        "before saving the manifest."
    )
    _OVERWRITE_MANIFEST_MESSAGE = (
        "Are you sure you want to overwrite the manifest and run "
        "configuration with the new settings?"
    )
    _CLEAR_CONFIGURATION_MESSAGE = (
        "Are you sure you want to clear the configuration?"
    )
    _MANIFEST_REQUIRED_MESSAGE = (
        "Save the manifest before advancing to confirmation."
    )
    _MANIFEST_REMINDER_MESSAGE = (
        "To start a run, you need to save the manifest."
    )
    _PROGRESS_POLL_INTERVAL = 0.5
    severity_order: dict[str, int] = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4,
    }

    def __init__(self, render: bool = True) -> None:
        self.render = render and tk is not None
        self.root: Optional[tk.Tk] = None
        self.frames: Dict[str, tk.Frame] = {}
        self.content_area: Optional[tk.Frame] = None
        self.status: RunStatus = RunStatus.IDLE
        self.recent_runs: list[str] = []
        self.pinned_configs: list[str] = []
        self.quick_actions: list[tuple[str, Callable[[], None]]] = [
            ("New Run", self.show_run_builder),
            ("Import manifest...", self.prompt_manifest_import),
            ("Open Run Monitor", self.show_run_monitor),
            ("Open Output Folder", self.open_output_directory),
        ]
        self.current_step_index = 0
        self.draft = RunDraft()
        self.summary = RunSummary()
        self.progress = 0
        self.nav_items = []
        self.gui_version = version.get_version()
        self.current_phase = "Idle"
        self.selected_models: list[str] = []
        self.selected_engine: str = ""
        self._selected_model_entry: dict | None = None
        self._selected_engine_entry: dict | None = None
        self.engine_capabilities: EngineCapabilities | None = None
        self._engine_setting_vars: dict[str, tk.StringVar] = {}
        self._engine_setting_specs: dict[str, EngineSetting] = {}
        self._engine_knobs_frame: ttk.Frame | None = None
        self.selected_datasets: list[dict[str, str]] = []
        self.pending_manifest: Optional[dict] = None
        self.manifest_workspace: ManifestWorkspace | None = None
        self._staged_confirm_manifest: Optional[dict] = None
        self.catalogue_index: dict[str, dict] = {}
        self.model_index: dict[str, dict] = {}
        self.engine_index: dict[str, dict] = {}
        self._current_run_output_dir: str | None = None
        self.metadata_cache: dict[str, str] = {}
        self.validation_notes: list[str] = []
        self.last_filter_types: list[str] = []
        self.output_directory_prepared = False
        self.output_retention_decision: str | None = None
        self.application_log_path = ""
        self.application_log_handler: _MemoryLogHandler | None = None
        self.run_log_path: str | None = None
        self.run_log_handler: _MemoryLogHandler | None = None
        self.run_logger: logging.Logger | None = None
        self.diagnostics_filter_level = "INFO"
        self.monitor_filter_level = "INFO"
        self.diagnostics_clipboard = ""
        self.run_clipboard = ""
        self.alerts: list[UIMessage] = []
        self.inline_messages: list[UIMessage] = []
        self.last_log_jump: str | None = None
        self._run_process: subprocess.Popen[str] | None = None
        self._run_config_path: str | None = None
        self._status_label: ttk.Label | None = None
        self._progress_state_path: str | None = None
        self._progress_snapshot: dict | None = None
        self._progress_poll_thread: threading.Thread | None = None
        self._progress_poll_stop: threading.Event | None = None
        self._monitor_refresh_job: str | None = None
        self._progress_status_label: ttk.Label | None = None
        self._batch_progressbar: ttk.Progressbar | None = None
        self._walker_progressbar: ttk.Progressbar | None = None
        self._monitor_log_widget: tk.Text | None = None
        self._monitor_filter_label: ttk.Label | None = None
        self._monitor_log_view_button: ttk.Button | None = None
        self._monitor_log_open_button: ttk.Button | None = None
        self._monitor_control_buttons: list[ttk.Button] = []
        self._monitor_button_style_name = "Copernican.RunControl.TButton"
        self._monitor_button_style_ready = False
        self._diagnostics_log_widget: tk.Text | None = None
        self._diagnostics_filter_label: ttk.Label | None = None
        self._cancel_button: ttk.Button | None = None
        self._pause_button: ttk.Button | None = None
        self._hard_stop_button: ttk.Button | None = None
        self._run_output_button: ttk.Button | None = None
        self.logo_image: tk.PhotoImage | None = None
        self._bootstrap_logging()
        self._build_navigation()
        self._initialise_rendering()
        self.refresh_inventory()
        self.help_banner_image = None
        self._load_saved_manifest_workspace()
        self._builder_step_buttons: list[ttk.Button] = []

    def _bootstrap_logging(self) -> None:
        """Start the diagnostics log and capture environment details."""

        base_dir = os.getcwd()
        self.application_log_path = logger.setup_program_logging(
            log_dir="logs",
            base_dir=base_dir,
        )
        program_logger = logger.get_program_logger()
        self.application_log_handler = _MemoryLogHandler(prefix="app")
        self._attach_handler(program_logger, self.application_log_handler)
        logger.log_environment_info(target_logger=program_logger)
        self._log_program_event(
            "GUI launch completed; diagnostics stream active",
            logging.INFO,
        )

    def _ensure_monitor_button_styles(self) -> None:
        """Configure ttk styles so disabled run controls appear grey."""

        if self._monitor_button_style_ready:
            return
        if not self.render or ttk is None or self.root is None:
            return
        style = ttk.Style(self.root)
        style.configure(self._monitor_button_style_name)
        style.map(
            self._monitor_button_style_name,
            foreground=[
                ("disabled", "#7f7f7f"),
            ],
        )
        self._monitor_button_style_ready = True

    def _monitor_button_kwargs(self) -> dict[str, str]:
        """Return ttk configuration for buttons with the monitor style."""

        self._ensure_monitor_button_styles()
        if self._monitor_button_style_ready:
            return {"style": self._monitor_button_style_name}
        return {}

    def _attach_handler(
        self, logger_obj: logging.Logger, handler: _MemoryLogHandler
    ) -> None:
        """Attach a single in-memory handler per logger prefix."""

        for existing in list(logger_obj.handlers):
            if (
                isinstance(existing, _MemoryLogHandler)
                and getattr(existing, "prefix", None) == handler.prefix
            ):
                logger_obj.removeHandler(existing)
        logger_obj.addHandler(handler)

    def _filter_entries_by_severity(
        self, entries: list[LogEntry], threshold: str
    ) -> list[LogEntry]:
        """Return log entries at or above the chosen severity level."""

        target = self.severity_order.get(threshold, 0)
        return [
            entry
            for entry in entries
            if self.severity_order.get(entry.severity, 0) >= target
        ]

    def _log_program_event(self, message: str, level: int) -> Optional[str]:
        """Record an application-level message and return its anchor."""

        program_logger = logger.get_program_logger()
        handler = self.application_log_handler
        program_logger.log(level, message)
        if handler is None:
            return None
        return handler.last_anchor()

    def _level_from_name(self, severity: str) -> int:
        """Convert a string severity into a logging level."""

        name = severity.upper()
        return getattr(logging, name, logging.INFO)

    def _dispatch_and_anchor(
        self,
        logger_obj: logging.Logger,
        handler: _MemoryLogHandler | None,
        message: str,
        level: int,
    ) -> Optional[str]:
        """Log to ``logger_obj`` and return the handler's anchor."""

        logger_obj.log(level, message)
        if handler is None:
            return None
        return handler.last_anchor()

    def _start_run_logging(self, manifest: dict) -> None:
        """Initialise run-level logging after confirmation."""

        os.makedirs("logs/runs", exist_ok=True)
        log_tag = f"copernican-run_{utils.get_timestamp()}.txt"
        self.run_log_path = os.path.join("logs", "runs", log_tag)
        self._current_run_output_dir = os.path.join(
            "output", Path(log_tag).stem
        )
        self.run_logger = logging.getLogger("copernican.gui.run")
        self.run_logger.setLevel(logging.INFO)
        self.run_logger.propagate = False
        for handler in list(self.run_logger.handlers):
            self.run_logger.removeHandler(handler)
        file_handler = logging.FileHandler(self.run_log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)
        self.run_logger.addHandler(file_handler)
        self.run_log_handler = _MemoryLogHandler(prefix="run")
        self.run_log_handler.setFormatter(formatter)
        self._attach_handler(self.run_logger, self.run_log_handler)
        selection = manifest.get("selection", {})
        models = ", ".join(selection.get("models", [])) or "unspecified"
        engine_meta = selection.get("engine", {})
        engine_name = engine_meta.get("name", "engine")
        engine_ver = engine_meta.get("version", "unspecified")
        datasets = selection.get("datasets", [])
        dataset_label = ", ".join(datasets) or "none"
        confirmation = manifest.get("confirmation", {})
        seed_value = confirmation.get("seed", "unset")
        plan_desc = confirmation.get("plan", "unspecified")
        self._log_run_event(
            (
                "Run confirmed with manifest: models=%s; engine=%s v%s; "
                "datasets=%s; seed=%s; plan=%s"
            )
            % (
                models,
                engine_name,
                engine_ver,
                dataset_label,
                seed_value,
                plan_desc,
            ),
            logging.INFO,
        )

    def _log_run_event(
        self, message: str, level: int = logging.INFO
    ) -> Optional[str]:
        """Record a run-level event and return the anchor."""

        if self.run_logger is None:
            return self._log_program_event(message, level)
        return self._dispatch_and_anchor(
            self.run_logger, self.run_log_handler, message, level
        )

    def get_application_log_entries(self) -> list[LogEntry]:
        """Return filtered application log entries."""

        if self.application_log_handler is None:
            return []
        return self._filter_entries_by_severity(
            self.application_log_handler.entries,
            self.diagnostics_filter_level,
        )

    def get_run_log_entries(self) -> list[LogEntry]:
        """Return filtered run log entries."""

        if self.run_log_handler is None:
            return []
        return self._filter_entries_by_severity(
            self.run_log_handler.entries, self.monitor_filter_level
        )

    def set_diagnostics_filter(self, severity: str) -> None:
        """Update the severity filter applied in Diagnostics."""

        self.diagnostics_filter_level = severity.upper()
        if self._diagnostics_filter_label:
            self._diagnostics_filter_label.configure(
                text=f"Filter: {self.diagnostics_filter_level}+"
            )
        self._refresh_diagnostics_widget()

    def set_monitor_filter(self, severity: str) -> None:
        """Update the Run Monitor severity filter."""

        self.monitor_filter_level = severity.upper()
        if self._monitor_filter_label:
            self._monitor_filter_label.configure(
                text=f"Filter: {self.monitor_filter_level}+"
            )
        self._refresh_run_log_widget()

    def copy_application_logs(self) -> str:
        """Copy filtered diagnostics logs into a clipboard buffer."""

        entries = self.get_application_log_entries()
        self.diagnostics_clipboard = "\n".join(
            entry.formatted for entry in entries
        )
        return self.diagnostics_clipboard

    def copy_run_logs(self) -> str:
        """Copy filtered run logs into a clipboard buffer."""

        entries = self.get_run_log_entries()
        self.run_clipboard = "\n".join(entry.formatted for entry in entries)
        return self.run_clipboard

    def export_application_logs(self, output_dir: str) -> str:
        """Write filtered diagnostics logs to the requested directory."""

        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, "diagnostics_log.txt")
        with open(export_path, "w", encoding="utf-8") as handle:
            handle.write(self.copy_application_logs())
        return export_path

    def export_run_logs(self, output_dir: str) -> str:
        """Write filtered run logs to the requested directory."""

        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, "run_log.txt")
        with open(export_path, "w", encoding="utf-8") as handle:
            handle.write(self.copy_run_logs())
        return export_path

    def create_toast(
        self,
        message: str,
        *,
        severity: str = "ERROR",
        context: str = "run",
        anchor: str | None = None,
    ) -> UIMessage:
        """Register an alert and link it to a log anchor."""

        resolved_anchor = anchor
        if resolved_anchor is None:
            logger_obj = self.run_logger or logger.get_program_logger()
            handler = (
                self.run_log_handler
                if self.run_logger
                else self.application_log_handler
            )
            resolved_anchor = self._dispatch_and_anchor(
                logger_obj, handler, message, self._level_from_name(severity)
            )
        toast = UIMessage(
            message=message,
            anchor=resolved_anchor or "",
            severity=severity.upper(),
            context=context,
        )
        self.alerts.append(toast)
        self.inline_messages.append(toast)
        return toast

    def jump_to_log_anchor(self, anchor: str) -> Optional[str]:
        """Return the log line for the provided anchor."""

        for handler in (self.run_log_handler, self.application_log_handler):
            if handler is None:
                continue
            for entry in handler.entries:
                if entry.anchor == anchor:
                    self.last_log_jump = anchor
                    return entry.formatted
        return None

    def _initialise_rendering(self) -> None:
        """Create the Tk root window and layout when rendering is enabled."""

        if not self.render:
            console_output.write(
                (
                    "GUI rendering disabled; running in headless controller "
                    "mode."
                )
            )
            return
        try:
            self.root = tk.Tk()
            self.root.title(f"Copernican Suite {self.gui_version}")
            self.root.geometry("1200x800")
            self.root.configure(padx=12, pady=12)
            self._ensure_monitor_button_styles()
            self._build_layout()
        except Exception as exc:  # pragma: no cover - only hits Tk failures
            console_output.write(
                (
                    "Tkinter failed to initialise; continuing without "
                    "rendering."
                ),
                error=True,
            )
            console_output.write(str(exc), error=True)
            self.render = False
            try:
                gui_logger = log_mod.get_logger()
                gui_logger.warning(
                    (
                        "Tkinter initialisation failed [%s]; "
                        "TCL_LIBRARY=%s TK_LIBRARY=%s executable=%s"
                    ),
                    exc,
                    os.environ.get("TCL_LIBRARY"),
                    os.environ.get("TK_LIBRARY"),
                    sys.executable,
                    exc_info=True,
                )
            except Exception:
                pass

    def _data_root(self) -> str:
        """Return the absolute data directory used for catalogue scans.

        Centralising the discovery root keeps the GUI aligned with
        :func:`dataset_registry.discover_trusted_parsers`, preventing the
        headless controller from drifting away from the registry's own safety
        checks when the repository layout changes.
        """

        return str(Path(__file__).resolve().parents[2] / "data")

    def _models_root(self) -> str:
        """Return the absolute path to the packaged model definitions."""

        return str(Path(__file__).resolve().parents[2] / "models")

    def _engines_root(self) -> str:
        """Return the absolute path to the available engine modules."""

        return str(Path(__file__).resolve().parents[2] / "engines")

    def _metadata_path_for_dir(self, data_dir: str) -> str | None:
        """Return the first metadata YAML file beneath ``data_dir``.

        This mirrors :func:`utils.load_metadata_from_dir` but also returns the
        filename so compatibility badges and digest displays can cite the
        provenance of the metadata used during parser registration.
        """

        for pattern in ("metadata*.yml", "metadata*.yaml"):
            matches = sorted(Path(data_dir).glob(pattern))
            if matches:
                return str(matches[0])
        return None

    def _parser_path_for_dir(self, data_dir: str) -> str | None:
        """Return the registered parser path beneath ``data_dir`` if found."""

        candidates = sorted(Path(data_dir).glob("cosmo_parser_*.py"))
        if not candidates:
            return None
        return str(candidates[0])

    def _collect_dataset_hashes(self, data_dir: str) -> dict[str, str]:
        """Compute SHA256 digests for non-parser files under ``data_dir``.

        The manifest builder expects per-file hashes so duplicated runs can
        prove they consumed identical inputs.  The GUI mirrors
        :func:`dataset_registry._attach_file_hashes` here so the catalogue view
        can surface the same digests without forcing a full data load.
        """

        hashes: dict[str, str] = {}
        for root, _, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, data_dir)
                hashes[rel] = utils.compute_sha256(path)
        return hashes

    def _compatibility_badges(self, dataset_key: str, meta: dict) -> list[str]:
        """Return badges describing dataset compatibility and claims.

        Badges mirror the dataset type and flag whether independence notes or
        citations were provided so the UI can nudge users toward well-formed
        metadata.  The list remains intentionally short to keep the Tk labels
        legible.
        """

        badges = [dataset_key.upper()]
        if meta.get("citation"):
            badges.append("CITED")
        if meta.get("license"):
            badges.append("LICENSED")
        if dataset_key in ("bao", "cmb") and meta.get("version"):
            badges.append(str(meta.get("version")))
        return badges

    def _discover_dataset_catalogue(self) -> dict[str, dict]:
        """Return dataset metadata indexed by dataset identifier.

        Discovery defers to :func:`dataset_registry.discover_trusted_parsers`
        so the GUI honours the same hash validation rules as the CLI.  The
        resulting catalogue surfaces parser digests, metadata digests and
        compatibility badges for the detail panes.
        """

        dataset_registry.discover_trusted_parsers(self._data_root())
        catalogue: dict[str, dict] = {}
        registries = dataset_registry.get_parser_registries()
        for dtype, registry in registries.items():
            for dataset_id, entry in registry.items():
                data_dir = entry.get("data_dir")
                if not data_dir:
                    continue
                meta = utils.load_metadata_from_dir(data_dir) or {}
                metadata_path = self._metadata_path_for_dir(data_dir)
                parser_path = self._parser_path_for_dir(data_dir)
                parser_digest = (
                    dataset_registry._file_sha256(parser_path)
                    if parser_path
                    else ""
                )
                rel_parser = (
                    os.path.relpath(parser_path, self._data_root())
                    if parser_path
                    else None
                )
                if rel_parser is not None:
                    rel_parser = rel_parser.replace("\\", "/")
                expected_digest = (
                    dataset_registry.TRUSTED_PARSER_DIGESTS.get(rel_parser)
                    if rel_parser
                    else None
                )
                parser_trusted = bool(
                    expected_digest and parser_digest == expected_digest
                )
                if not parser_trusted:
                    note = (
                        f"Parser {rel_parser} failed trust validation; "
                        "skipping until hashes match"
                    )
                    self.validation_notes.append(note)
                independence_notes = (
                    dataset_registry.OBSERVATION_INDEPENDENCE_NOTES.get(
                        dtype, []
                    )
                )
                record = {
                    "id": dataset_id,
                    "type": dtype,
                    "name": meta.get("dataset_name", dataset_id),
                    "path": data_dir,
                    "citation": meta.get("citation", ""),
                    "license": meta.get("license", ""),
                    "version": meta.get("version", "unknown"),
                    "metadata_path": metadata_path,
                    "metadata_digest": (
                        utils.compute_sha256(metadata_path)
                        if metadata_path
                        else ""
                    ),
                    "parser_path": parser_path,
                    "parser_digest": parser_digest,
                    "expected_digest": expected_digest,
                    "parser_trusted": parser_trusted,
                    "hashes": self._collect_dataset_hashes(data_dir),
                    "badges": self._compatibility_badges(dtype, meta),
                    "independence": independence_notes,
                }
                catalogue[dataset_id] = record
        return catalogue

    def _read_model_file(self, path: Path) -> dict:
        """Return parsed YAML for ``path`` while handling empty files."""

        try:
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except yaml.YAMLError:
            logger.get_program_logger().warning(
                "Model metadata in %s is malformed; continuing with blanks",
                path,
            )
            return {}

    def _discover_model_library(self) -> dict[str, dict]:
        """Return model metadata keyed by filename stem."""

        models: dict[str, dict] = {}
        for path in sorted(Path(self._models_root()).glob("*.yml")):
            if path.name.startswith("__"):
                continue
            meta = self._read_model_file(path)
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
                    "Copernican Suite default license; add model notes",
                ),
                "version": meta.get("version", "unknown"),
                "badges": badges,
                "hash": utils.compute_sha256(str(path)),
                "parameter_count": len(parameters),
            }
        return models

    def _discover_engine_library(self) -> dict[str, dict]:
        """Return engine metadata keyed by module stem."""

        engines: dict[str, dict] = {}
        for path in sorted(Path(self._engines_root()).glob("*.py")):
            if path.name.startswith("__"):
                continue
            module_name = f"engines.{path.stem}"
            try:
                module = importlib.import_module(module_name)
                label = getattr(module, "ENGINE_LABEL", path.stem)
                version = getattr(module, "ENGINE_VERSION", "unknown")
            except Exception:
                module = None
                label = path.stem
                version = "unavailable"
                logger.get_program_logger().warning(
                    "Engine metadata import failed for %s; using fallbacks",
                    module_name,
                )
            engines[module_name] = {
                "id": module_name,
                "filename": path.name,
                "path": str(path),
                "stem": path.stem,
                "citation": getattr(module, "__doc__", ""),
                "license": "Copernican Suite default license; verify engines",
                "version": version,
                "label": label,
                "badges": ["SNE", "BAO", "CMB"],
                "hash": utils.compute_sha256(str(path)),
            }
        return engines

    def refresh_inventory(self) -> None:
        """Refresh catalogue, model and engine metadata for list views.

        The method runs at startup and when panels request a revalidation.  It
        guarantees that GUI detail panes reflect the parser registration rules
        enforced by :mod:`dataset_registry` and that Run Builder defaults pull
        from the newest manifests and compatibility declarations.
        """

        self.validation_notes = []
        self.catalogue_index = self._discover_dataset_catalogue()
        self.model_index = self._discover_model_library()
        self.engine_index = self._discover_engine_library()

    def filter_catalogue(self, types: list[str] | None = None) -> list[dict]:
        """Return datasets matching ``types`` while recording the filter."""

        if types is None:
            types = []
        resolved = [t.lower() for t in types if t]
        self.last_filter_types = resolved
        if not resolved:
            return list(self.catalogue_index.values())
        return [
            entry
            for entry in self.catalogue_index.values()
            if entry.get("type", "").lower() in resolved
        ]

    def open_folder(self, path: str) -> str:
        """Return ``path`` after confirming it exists for quick actions."""

        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        if self.render:
            self._launch_folder(path)
        return path

    def _open_folder_or_warn(
        self, path: str, *, context: str, subject: str
    ) -> None:
        """Open ``path`` if it exists, otherwise register a toast."""

        if not path:
            self.create_toast(
                f"No file path recorded for {subject}.",
                severity="WARNING",
                context=context,
            )
            return
        try:
            self.open_folder(path)
        except FileNotFoundError:
            self.create_toast(
                f"{subject} folder is unavailable at {path}.",
                severity="ERROR",
                context=context,
            )

    def _output_root(self) -> str:
        """Return the canonical output directory for GUI quick actions."""

        return os.path.abspath(os.path.join(os.getcwd(), "output"))

    def _launch_folder(self, path: str) -> None:
        """Open ``path`` in the native file manager when rendering."""

        if not self.render:
            return
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as exc:
            console_output.write(
                f"Unable to open folder {path}: {exc}", error=True
            )

    def open_output_directory(self) -> str:
        """Ensure the output directory exists and open it for the user."""

        output_dir = self._output_root()
        os.makedirs(output_dir, exist_ok=True)
        return self.open_folder(output_dir)

    def _open_path_with_system(self, path: str) -> None:
        """Open ``path`` using the operating system defaults."""

        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as exc:
            console_output.write(f"Unable to open {path}: {exc}", error=True)

    def _show_metadata_dialog(
        self, title: str, content: str, source_path: str | None = None
    ) -> None:
        """Show metadata content in a scrollable dialog when rendering."""

        if not self.render or self.root is None:
            console_output.write(f"{title}:\n{content}")
            return
        window = tk.Toplevel(self.root)
        window.title(title)
        window.transient(self.root)
        raw_lines = content.splitlines() or [""]
        longest = max((len(line) for line in raw_lines), default=80)
        char_units = max(longest, 80)
        initial_width = int(char_units * 7.2)
        min_lines = 15
        default_lines = 25
        line_count = max(len(raw_lines), 1)
        resizable_vertical = line_count > min_lines
        if not resizable_vertical:
            display_lines = line_count
        else:
            display_lines = min(default_lines, line_count)
            display_lines = max(display_lines, min_lines)
        line_height = 20
        chrome_height = 120
        initial_height = display_lines * line_height + chrome_height
        window.geometry(f"{initial_width}x{initial_height}")
        window.resizable(False, resizable_vertical)
        if resizable_vertical:
            min_height = min_lines * line_height + chrome_height
            window.minsize(width=initial_width, height=min_height)
        container = ttk.Frame(window, padding=(8, 6))
        container.pack(fill="both", expand=True)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        text = tk.Text(container, wrap="word", height=display_lines or 1)
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=text.yview
        )
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        buttons = ttk.Frame(container)
        buttons.grid(row=1, column=0, columnspan=2, pady=(8, 8))
        ttk.Button(buttons, text="Close", command=window.destroy).pack(
            side="left", padx=4
        )
        if source_path:
            ttk.Button(
                buttons,
                text="Open file…",
                command=lambda: self._open_path_with_system(source_path),
            ).pack(side="left", padx=4)

    def _create_scrollable_panel(
        self, parent: tk.Frame, *, height: int | None = None
    ) -> tk.Frame:
        """Return a frame that scrolls vertically with the given parent."""

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        canvas_kwargs: dict[str, int | str] = {
            "borderwidth": 0,
            "highlightthickness": 0,
        }
        if height is not None:
            canvas_kwargs["height"] = height
        canvas = tk.Canvas(container, **canvas_kwargs)
        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=canvas.yview
        )
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        inner_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all"),
                width=event.width,
            ),
        )
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return inner_frame

    def _resolve_asset_path(self, asset_id: str) -> str:
        """Return the on-disk path linked to ``asset_id``."""

        entry = self.catalogue_index.get(asset_id)
        if entry and entry.get("metadata_path"):
            return entry["metadata_path"]
        model_entry = self.model_index.get(asset_id)
        if model_entry:
            return model_entry["path"]
        for record in self.model_index.values():
            if record.get("id") == asset_id:
                return record["path"]
        engine_entry = self.engine_index.get(asset_id)
        if engine_entry:
            return engine_entry["path"]
        for record in self.engine_index.values():
            if record.get("id") == asset_id:
                return record["path"]
        raise KeyError(asset_id)

    def _read_asset_text(self, path: str) -> str:
        """Return cached text content for ``path``."""

        if path in self.metadata_cache:
            return self.metadata_cache[path]
        with open(path, "r", encoding="utf-8") as handle:
            contents = handle.read()
        self.metadata_cache[path] = contents
        return contents

    def _present_metadata(self, asset_id: str, title: str) -> None:
        """Fetch metadata text and present it to the operator."""

        try:
            path = self._resolve_asset_path(asset_id)
            content = self._read_asset_text(path)
        except (KeyError, OSError) as exc:
            self.create_toast(
                f"Metadata unavailable for {asset_id}: {exc}",
                severity="ERROR",
                context="data",
            )
            return
        self._show_metadata_dialog(title, content, path)

    def _revalidate_dataset_action(self, dataset_id: str) -> None:
        """Run parser trust validation and share the result."""

        try:
            record = self.revalidate_dataset(dataset_id)
        except KeyError as exc:
            self.create_toast(
                f"Revalidation failed for {dataset_id}: {exc}",
                severity="ERROR",
                context="data",
            )
            return
        parser_digest = record.get("parser_digest", "n/a")
        self.create_toast(
            f"Parser {dataset_id} revalidated with digest {parser_digest}",
            severity="INFO",
            context="data",
        )
        self.show_data_overview()

    def _load_help_banner(self) -> None:
        """Load the README banner image for the Help panel."""

        if self.help_banner_image is not None or not self.render or tk is None:
            return
        banner_path = (
            Path(__file__).resolve().parents[2] / "docs" / "banner_github.png"
        )
        if not banner_path.exists():
            return
        try:
            self.help_banner_image = tk.PhotoImage(file=str(banner_path))
        except Exception as exc:
            logger.get_program_logger().warning(
                "Failed to load help banner image: %s", exc
            )
            self.help_banner_image = None

    def _logo_image_path(self) -> Path:
        """Return the expected path to the resized navigation logo."""

        return Path(__file__).resolve().parents[2] / "img" / "logogui.png"

    def _load_logo_image(self) -> None:
        """Load the navigation logo when rendering is enabled."""

        if self.logo_image is not None or not self.render or tk is None:
            return
        logo_path = self._logo_image_path()
        if not logo_path.exists():
            return
        try:
            self.logo_image = tk.PhotoImage(file=str(logo_path))
        except Exception as exc:
            logger.get_program_logger().warning(
                "Failed to load navigation logo: %s", exc
            )

    def _build_navigation_logo(self, nav_frame: tk.Frame) -> None:
        """Add the padded logo square above the navigation buttons."""

        if not self.render or tk is None:
            return
        self._load_logo_image()
        logo_holder = ttk.Frame(nav_frame)
        logo_holder.pack(fill="x", pady=(0, _LOGO_PADDING))
        logo_holder.pack_propagate(False)
        base_side = _LOGO_SIDE
        image_width = self.logo_image.width() if self.logo_image else base_side
        image_height = (
            self.logo_image.height() if self.logo_image else base_side
        )
        holder_height = image_height + 2 * _LOGO_PADDING
        holder_width = image_width + 2 * _LOGO_PADDING
        logo_holder.configure(height=holder_height)
        square = ttk.Frame(
            logo_holder,
            width=holder_width,
            height=holder_height,
            padding=_LOGO_PADDING,
        )
        square.pack_propagate(False)
        square.pack(anchor="center")
        square.columnconfigure(0, weight=1)
        square.rowconfigure(0, weight=1)
        if self.logo_image:
            logo_widget = ttk.Label(
                square,
                image=self.logo_image,
            )
            logo_widget.image = self.logo_image
        else:
            logo_widget = ttk.Label(
                square,
                text="Copernican",
                anchor="center",
                justify="center",
            )
        logo_widget.grid(row=0, column=0, sticky="nsew")

    def _load_help_markdown(self) -> str:
        """Return the contents of README.md for the Help panel."""

        readme_path = Path(__file__).resolve().parents[2] / "README.md"
        try:
            return readme_path.read_text(encoding="utf-8")
        except Exception as exc:
            message = f"Unable to read README.md for Help panel: {exc}"
            logger.get_program_logger().warning(message)
            return message

    def _render_markdown_in_text_widget(
        self, widget: tk.Text, markdown: str
    ) -> None:
        """Render simplified Markdown into the provided text widget."""

        widget.tag_configure(
            "heading1", font=("Helvetica", 18, "bold"), spacing1=10
        )
        widget.tag_configure(
            "heading2", font=("Helvetica", 16, "bold"), spacing1=8
        )
        widget.tag_configure(
            "heading3", font=("Helvetica", 14, "bold"), spacing1=6
        )
        widget.tag_configure("bold", font=("Helvetica", 11, "bold"))
        widget.tag_configure("italic", font=("Helvetica", 11, "italic"))
        widget.tag_configure(
            "code",
            font=("TkFixedFont", 10),
            background="#f3f3f3",
            spacing1=2,
            spacing3=2,
        )
        widget.tag_configure("normal", font=("Helvetica", 11))

        inline_pattern = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
        in_code = False
        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                self._insert_inline_text(
                    widget, line + "\n", ("code",), inline_pattern
                )
                continue
            if line.startswith("!["):
                continue
            if not line:
                widget.insert("end", "\n")
                continue
            stripped = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", line)
            if stripped.startswith("# "):
                self._insert_inline_text(
                    widget,
                    stripped[2:].strip() + "\n",
                    ("heading1",),
                    inline_pattern,
                )
                continue
            if stripped.startswith("## "):
                self._insert_inline_text(
                    widget,
                    stripped[3:].strip() + "\n",
                    ("heading2",),
                    inline_pattern,
                )
                continue
            if stripped.startswith("### "):
                self._insert_inline_text(
                    widget,
                    stripped[4:].strip() + "\n",
                    ("heading3",),
                    inline_pattern,
                )
                continue
            if stripped.startswith(("- ", "* ")):
                bullet = "• " + stripped[2:]
                self._insert_inline_text(
                    widget, bullet + "\n", ("normal",), inline_pattern
                )
                continue
            widget.insert("end", "")
            self._insert_inline_text(
                widget, stripped + "\n", ("normal",), inline_pattern
            )

    def _insert_inline_text(
        self,
        widget: tk.Text,
        text: str,
        base_tags: tuple[str, ...],
        pattern: re.Pattern,
    ) -> None:
        """Insert text and honour inline Markdown markers."""

        last = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            widget.insert("end", text[last:start], base_tags)
            if match.group(1):
                widget.insert("end", match.group(1), base_tags + ("bold",))
            else:
                widget.insert("end", match.group(2), base_tags + ("italic",))
            last = end
        widget.insert("end", text[last:], base_tags)

    def view_metadata_file(self, asset_id: str) -> str:
        """Return cached metadata content for the requested asset."""

        path = self._resolve_asset_path(asset_id)
        return self._read_asset_text(path)

    def revalidate_dataset(self, dataset_id: str) -> dict[str, str]:
        """Re-run parser trust checks and return the refreshed record."""

        self.refresh_inventory()
        if dataset_id not in self.catalogue_index:
            raise KeyError(dataset_id)
        record = self.catalogue_index[dataset_id]
        if record.get("expected_digest") and record.get("parser_digest"):
            if record["expected_digest"] != record["parser_digest"]:
                note = (
                    f"Digest mismatch for {dataset_id}: expected "
                    f"{record['expected_digest']} but observed "
                    f"{record['parser_digest']}"
                )
                self.validation_notes.append(note)
        return record

    def _build_navigation(self) -> None:
        """Populate the navigation rail definitions and shortcuts."""

        self.nav_items = [
            NavigationItem("home", "Home", CopernicanGUI.show_home),
            NavigationItem(
                "run_builder",
                "Run Builder",
                CopernicanGUI.show_run_builder,
            ),
            NavigationItem(
                "run_monitor",
                "Run Monitor",
                CopernicanGUI.show_run_monitor,
            ),
            NavigationItem("data", "Data", CopernicanGUI.show_data_overview),
            NavigationItem("models", "Models", CopernicanGUI.show_models),
            NavigationItem("engines", "Engines", CopernicanGUI.show_engines),
            NavigationItem(
                "settings", "Settings", CopernicanGUI.show_settings
            ),
            NavigationItem("help", "Help", CopernicanGUI.show_help),
            NavigationItem("about", "About", CopernicanGUI.show_about),
            NavigationItem("exit", "Exit Suite", CopernicanGUI.exit_suite),
        ]

    def _build_layout(self) -> None:
        """Construct the navigation rail and main content area."""

        if not self.render or self.root is None:
            return
        nav_frame = ttk.Frame(self.root, padding=(12, 12, 24, 12))
        nav_frame.configure(width=_NAV_PANE_WIDTH)
        nav_frame.grid(row=0, column=0, sticky="nsw")
        nav_frame.grid_propagate(False)
        self.root.grid_columnconfigure(0, minsize=_NAV_PANE_WIDTH)
        separator = ttk.Separator(self.root, orient="vertical")
        separator.grid(row=0, column=1, sticky="ns")
        self.root.grid_columnconfigure(1, minsize=2)
        self.content_area = ttk.Frame(self.root, padding=(12, 12, 12, 12))
        self.content_area.grid(row=0, column=2, sticky="nsew")
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self._build_navigation_logo(nav_frame)
        for item in self.nav_items:
            button = ttk.Button(
                nav_frame,
                text=item.label,
                command=lambda i=item: i.action(self),
                takefocus=True,
            )
            button.pack(fill="x", pady=4)

        self.show_home()

    def _swap_content(self, frame_builder: Callable[[tk.Frame], None]) -> None:
        """Replace the right-hand content area with a new frame."""

        if not self.render or self.content_area is None:
            return
        for child in self.content_area.winfo_children():
            child.destroy()
        frame = ttk.Frame(self.content_area, padding=(8, 8))
        frame.pack(fill="both", expand=True)
        frame_builder(frame)

    def show_home(self) -> None:
        """Render the project home panel with recents and quick actions."""

        if not self.recent_runs:
            self.recent_runs = [
                "copernican-run_20251124_120000",
                "copernican-run_20251124_094500",
            ]
        if not self.pinned_configs:
            self.pinned_configs = [
                "LambdaCDM baseline",
                "Joint likelihood sandbox",
            ]

        def builder(frame: tk.Frame) -> None:
            header = ttk.Label(
                frame, text="Project Home", font=("Helvetica", 16)
            )
            header.pack(anchor="w", pady=(0, 8))
            ttk.Label(frame, text="Recent runs", takefocus=True).pack(
                anchor="w"
            )
            for run in self.recent_runs:
                ttk.Label(frame, text=run, takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text="Pinned configurations",
                takefocus=True,
            ).pack(anchor="w", pady=(12, 0))
            for config in self.pinned_configs:
                ttk.Label(frame, text=config, takefocus=True).pack(anchor="w")
            ttk.Label(frame, text="Quick actions", takefocus=True).pack(
                anchor="w", pady=(12, 0)
            )
            for label, callback in self.quick_actions:
                ttk.Button(
                    frame,
                    text=label,
                    command=callback,
                    takefocus=True,
                ).pack(anchor="w", pady=2)

        self._swap_content(builder)

    def _is_seed_step_complete(self) -> bool:
        """Return True once the seed input has some text."""

        return bool(self.draft.seed.strip())

    def _is_model_step_complete(self) -> bool:
        """Return True when the user has picked at least one model."""

        return bool(self.selected_models or self.draft.model.strip())

    def _is_data_step_complete(self) -> bool:
        """Return True once datasets are recorded."""

        return bool(self.selected_datasets)

    def _is_engine_step_complete(self) -> bool:
        """Return True when an engine is selected."""

        return bool(self.selected_engine)

    def _builder_ready(self) -> bool:
        """Indicate whether the first four builder pages are complete."""

        return (
            self._is_seed_step_complete()
            and self._is_model_step_complete()
            and self._is_data_step_complete()
            and self._is_engine_step_complete()
        )

    def _has_configuration(self) -> bool:
        """Return True when there is something saved or selected."""

        return bool(
            self.manifest_workspace
            or self.selected_models
            or self.selected_engine
            or self.selected_datasets
            or self.draft.seed.strip()
            or self.draft.data.strip()
        )

    def _can_enter_confirm(self) -> bool:
        """Return True when the confirm page should be unlocked."""

        return self.manifest_workspace is not None and self._builder_ready()

    def _notify_builder_completion_required(self) -> None:
        """Warn the operator that pages 1–4 must be filled before saving."""

        message = self._BUILDER_COMPLETION_MESSAGE
        if self.render and messagebox:
            messagebox.showwarning(
                "Complete Run Builder steps",
                message,
            )
        self.create_toast(
            message,
            severity="WARNING",
            context="builder",
        )

    def _notify_manifest_save_required(self) -> None:
        """Remind the user to save the manifest before confirming."""

        message = self._MANIFEST_REQUIRED_MESSAGE
        if self.render and messagebox:
            messagebox.showwarning(
                "Save manifest first",
                message,
            )
        self.create_toast(
            message,
            severity="WARNING",
            context="builder",
        )

    def _handle_builder_next(self) -> None:
        """Guard the transition from Engine to Manifest."""

        engine_index = self.builder_steps.index("Engine")
        if (
            self.current_step_index == engine_index
            and not self._builder_ready()
        ):
            self._notify_builder_completion_required()
            return
        manifest_index = self.builder_steps.index("Manifest")
        if (
            self.current_step_index == manifest_index
            and self.manifest_workspace is None
        ):
            self._notify_manifest_save_required()
            return
        self.next_step()

    def _confirm_overwrite_manifest(self) -> bool:
        """Ask before overwriting an existing saved manifest."""

        if self.render and messagebox:
            return messagebox.askyesno(
                "Overwrite manifest",
                self._OVERWRITE_MANIFEST_MESSAGE,
            )
        self.create_toast(
            self._OVERWRITE_MANIFEST_MESSAGE,
            severity="WARNING",
            context="run",
        )
        return True

    def _confirm_clear_configuration(self) -> bool:
        """Prompt the user before clearing the builder configuration."""

        if self.render and messagebox:
            return messagebox.askyesno(
                "Clear configuration",
                self._CLEAR_CONFIGURATION_MESSAGE,
            )
        self.create_toast(
            self._CLEAR_CONFIGURATION_MESSAGE,
            severity="WARNING",
            context="builder",
        )
        return True

    def _clear_builder_selections(self) -> None:
        """Reset models, engines and drafts without altering workspace."""

        self.draft = RunDraft()
        self.selected_models = []
        self.selected_engine = ""
        self._selected_model_entry = None
        self._selected_engine_entry = None
        self.selected_datasets = []
        self.engine_capabilities = None
        self._staged_confirm_manifest = None

    def _clear_manifest_configuration(self) -> None:
        """Delete the active manifest and reset the builder wizard."""

        self._reset_manifest_state()
        self._clear_builder_selections()
        self.current_step_index = 0
        self.summary.manifest_actions.append(
            "Run builder configuration cleared."
        )

    def _save_manifest_and_proceed(self) -> None:
        """Persist the manifest and jump straight to the confirmation step."""

        workspace = self._persist_manifest_workspace(notify=True)
        if workspace:
            confirm_index = self.builder_steps.index("Confirm and start")
            self.current_step_index = confirm_index
            self.show_run_builder()

    def _save_manifest_to_external_folder(
        self,
        *,
        output_path: str | None = None,
    ) -> None:
        """Save the current manifest to an external file."""

        if output_path is None:
            if filedialog is None:
                self.create_toast(
                    "File dialogs are unavailable in this environment.",
                    severity="ERROR",
                    context="run",
                )
                return
            output_path = filedialog.asksaveasfilename(
                title="Save manifest externally",
                initialdir=str(self._output_root()),
                defaultextension=".yml",
                filetypes=[
                    ("YAML files", "*.yml *.yaml"),
                    ("All files", "*.*"),
                ],
            )
            if not output_path:
                return
        manifest = self._ensure_manifest_snapshot()
        if manifest is None:
            return
        directory = os.path.dirname(output_path)
        path = run_manifest.save_manifest(
            manifest,
            directory or self._output_root(),
            target_path=Path(output_path),
        )
        self.create_toast(
            f"Manifest exported to {path}",
            severity="INFO",
            context="run",
        )

    def prompt_manifest_import(self) -> None:
        """Ask the user to select a manifest file and load it."""

        if filedialog is None:
            self.create_toast(
                "File dialogs are unavailable in the current environment.",
                severity="ERROR",
                context="home",
            )
            return
        initial_dir = str(self._repo_root() / "output")
        path = filedialog.askopenfilename(
            title="Import manifest",
            initialdir=initial_dir,
            filetypes=[
                ("YAML files", "*.yml *.yaml"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.import_manifest(path)
        except Exception as exc:
            self.create_toast(
                f"Failed to import manifest: {exc}",
                severity="ERROR",
                context="home",
            )
            return
        self.summary.manifest_actions.append(f"Imported manifest from {path}")
        self.create_toast(
            "Manifest imported. Review the Run Builder selections.",
            severity="INFO",
            context="home",
        )
        self.show_run_builder()

    def show_run_builder(self) -> None:
        """Render the Run Builder wizard with jump controls."""

        self.refresh_inventory()

        def builder(frame: tk.Frame) -> None:
            header = ttk.Label(
                frame, text="Run Builder", font=("Helvetica", 16)
            )
            header.pack(anchor="w", pady=(0, 8))
            step_frame = ttk.Frame(frame)
            step_frame.pack(fill="x", pady=(0, 12))
            self._builder_step_buttons = []
            for index, step in enumerate(self.builder_steps):
                indicator = ttk.Button(
                    step_frame,
                    text=f"{index + 1}. {step}",
                    command=lambda idx=index: self.jump_to_step(idx),
                    takefocus=True,
                )
                indicator.pack(side="left", padx=2)
                self._builder_step_buttons.append(indicator)
            body = ttk.Label(
                frame,
                text=(
                    "Work through the stages to assemble a reproducible run "
                    "manifest. The controls below let you move between steps, "
                    "save your progress or cancel entirely."
                ),
                wraplength=720,
                takefocus=True,
            )
            body.pack(anchor="w", pady=(8, 8))

            controls = ttk.Frame(frame)
            controls.pack(anchor="w")
            nav_button_style = self._monitor_button_kwargs()
            ttk.Button(
                controls,
                text="Previous",
                command=self.previous_step,
                state=(
                    tk.DISABLED if self.current_step_index == 0 else tk.NORMAL
                ),
                takefocus=True,
                **nav_button_style,
            ).pack(side="left", padx=4)
            manifest_index = self.builder_steps.index("Manifest")
            next_disabled = self.current_step_index >= len(
                self.builder_steps
            ) - 1 or (
                self.current_step_index == manifest_index
                and self.manifest_workspace is None
            )
            ttk.Button(
                controls,
                text="Next",
                command=self._handle_builder_next,
                state=tk.DISABLED if next_disabled else tk.NORMAL,
                takefocus=True,
                **nav_button_style,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Cancel",
                command=self.cancel_builder,
                state=(
                    tk.NORMAL if self._has_configuration() else tk.DISABLED
                ),
                takefocus=True,
                **nav_button_style,
            ).pack(side="left", padx=4)
            content_container = ttk.Frame(frame)
            content_container.pack(fill="both", expand=True)
            scroll_panel = self._create_scrollable_panel(content_container)
            self._build_run_builder_step(scroll_panel)
            self._refresh_builder_step_indicators()

        self._swap_content(builder)

    def _refresh_builder_step_indicators(self) -> None:
        """Update builder jump buttons so only the active step is bold."""

        if not self.render or not self._builder_step_buttons:
            return
        manifest_index = self.builder_steps.index("Manifest")
        confirm_index = self.builder_steps.index("Confirm and start")
        for index, button in enumerate(self._builder_step_buttons):
            font = (
                ("Helvetica", 11, "bold")
                if index == self.current_step_index
                else ("Helvetica", 11)
            )
            button.configure(font=font)
            if index == manifest_index:
                desired_state = (
                    tk.NORMAL if self._builder_ready() else tk.DISABLED
                )
            elif index == confirm_index:
                desired_state = (
                    tk.NORMAL if self._can_enter_confirm() else tk.DISABLED
                )
            else:
                desired_state = tk.NORMAL
            button.configure(state=desired_state)

    def _build_run_builder_step(self, container: tk.Frame) -> None:
        """Render the content for the current builder step."""

        handlers = [
            self._render_builder_step_seed,
            self._render_builder_step_models,
            self._render_builder_step_data,
            self._render_builder_step_engine,
            self._render_builder_step_manifest,
            self._render_builder_step_confirm,
        ]
        if 0 <= self.current_step_index < len(handlers):
            handlers[self.current_step_index](container)

    def _parameter_count_for_selection(self) -> int:
        """Return summed parameter count for selected models."""

        if not self.selected_models and self.draft.model:
            candidate = self.draft.model.strip()
            if candidate:
                self.selected_models = [candidate]
        counts = []
        for model_id in self.selected_models:
            entry = self.model_index.get(model_id)
            if entry:
                counts.append(entry.get("parameter_count", 0))
                continue
            for record in self.model_index.values():
                if record.get("id") == model_id:
                    counts.append(record.get("parameter_count", 0))
                    break
        return max(sum(counts), 0)

    def _engine_default_settings(self) -> tuple[int, int, int]:
        """Return defaults for steps, walkers and worker pools."""

        default_steps = 200
        default_walkers = 32
        default_pool = os.cpu_count() or 1
        engine_entry = None
        try:
            engine_entry = self._resolve_engine_entry()
        except RuntimeError:
            return default_steps, default_walkers, default_pool
        module_name = engine_entry.get("id", "")
        if not module_name:
            return default_steps, default_walkers, default_pool
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return default_steps, default_walkers, default_pool
        fit_fn = getattr(
            module,
            "fit_cosmology_parameters",
            getattr(module, "fit_sne_parameters", None),
        )
        if not callable(fit_fn):
            return default_steps, default_walkers, default_pool
        try:
            signature = inspect.signature(fit_fn)
        except Exception:
            return default_steps, default_walkers, default_pool

        def _default_value(name: str, fallback: int) -> int:
            param = signature.parameters.get(name)
            if param is None or param.default is inspect._empty:
                return fallback
            try:
                return int(param.default)
            except (TypeError, ValueError):
                return fallback

        steps = _default_value("n_steps", default_steps)
        walkers = _default_value("n_walkers", default_walkers)
        pool = default_pool
        pool_param = signature.parameters.get("pool_size")
        if pool_param and pool_param.default is not inspect._empty:
            try:
                pool_val = int(pool_param.default)
                if pool_val > 0:
                    pool = pool_val
            except (TypeError, ValueError):
                pool = default_pool
        return steps, walkers, pool

    def _compute_run_recommendations(self) -> dict[str, int | str]:
        """Return heuristic recommendations for run settings."""

        param_total = max(self._parameter_count_for_selection(), 1)
        minimum_walkers = max(2 * param_total, 2)
        default_steps, default_walkers, default_pool = (
            self._engine_default_settings()
        )
        recommended_steps = max(default_steps, 1)
        burn_in_recommended = max(100, recommended_steps // 5)
        quick_burn = max(1, recommended_steps // 5)
        recommended_walkers = max(default_walkers, minimum_walkers)
        cpu_detected = os.cpu_count() or 0
        cpu_label = cpu_detected if cpu_detected > 0 else "unknown"
        recommended_pool = (
            cpu_detected if cpu_detected > 0 else minimum_walkers
        )
        recommended_pool = max(recommended_pool, 1)
        production_min = max(recommended_steps, 500)
        return {
            "minimum_walkers": minimum_walkers,
            "recommended_walkers": recommended_walkers,
            "recommended_steps": recommended_steps,
            "burn_in_recommended": burn_in_recommended,
            "quick_burn": quick_burn,
            "production_min": production_min,
            "pool_max": recommended_pool,
            "cpu_label": cpu_label,
        }

    def _render_builder_step_seed(self, container: tk.Frame) -> None:
        ttk.Label(
            container,
            text="Seed selection",
            font=("Helvetica", 14, "bold"),
            takefocus=True,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            container,
            text=(
                "Choose a deterministic seed or let the system generate one. "
                "Hints are logged in the run manifest for reproducibility."
            ),
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        seed_var = tk.StringVar(value=self.draft.seed)

        def _update_seed(*_args: object) -> None:
            self.draft.seed = seed_var.get().strip()
            self._refresh_builder_step_indicators()

        seed_var.trace_add("write", _update_seed)
        entry = ttk.Entry(container, textvariable=seed_var, width=24)
        entry.pack(anchor="w", pady=(6, 8))
        button_row = ttk.Frame(container)
        button_row.pack(anchor="w")
        ttk.Button(
            button_row,
            text="Default (0)",
            command=lambda: seed_var.set("0"),
        ).pack(side="left", padx=2)
        ttk.Button(
            button_row,
            text="Random timestamp",
            command=lambda: seed_var.set(str(int(time.time()))),
        ).pack(side="left", padx=2)
        env_seed = os.environ.get("COPERNICAN_SEED")
        if env_seed:
            ttk.Button(
                button_row,
                text="Use COPERNICAN_SEED",
                command=lambda: seed_var.set(env_seed),
            ).pack(side="left", padx=2)

    def _render_builder_step_models(self, container: tk.Frame) -> None:
        ttk.Label(
            container,
            text="Model selection",
            font=("Helvetica", 14, "bold"),
            takefocus=True,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            container,
            text="Pick one YAML-defined model to include in the run.",
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        available = sorted(
            self.model_index.values(), key=lambda entry: entry["id"]
        )
        if not available:
            ttk.Label(
                container,
                text="No models available; refresh inventory and try again.",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            return
        list_container = ttk.Frame(container)
        list_container.pack(fill="both", expand=True, pady=(8, 4))
        list_frame = ttk.Frame(list_container)
        list_frame.pack(side="left", fill="both", expand=True)
        listbox = tk.Listbox(
            list_frame,
            height=8,
            selectmode="browse",
            exportselection=False,
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=listbox.yview
        )
        scrollbar.pack(side="right", fill="y")
        listbox.configure(yscrollcommand=scrollbar.set)
        button_frame = ttk.Frame(list_container)
        button_frame.pack(side="left", fill="y", padx=(8, 0), anchor="n")

        def _view_selected_model() -> None:
            entry = self._selected_model_entry
            if not entry:
                self.create_toast(
                    "Select a model before viewing its definition.",
                    severity="WARNING",
                    context="models",
                )
                return
            self._present_metadata(
                entry["id"], f"Model definition: {entry['id']}"
            )

        def _open_selected_model_file() -> None:
            entry = self._selected_model_entry
            if not entry:
                self.create_toast(
                    "Select a model before opening its YAML file.",
                    severity="WARNING",
                    context="models",
                )
                return
            self._open_path_with_system(entry["path"])

        ttk.Button(
            button_frame,
            text="View model",
            command=_view_selected_model,
        ).pack(fill="x", pady=(0, 4))
        ttk.Button(
            button_frame,
            text="Open model YML...",
            command=_open_selected_model_file,
        ).pack(fill="x")
        summary = ttk.Label(container, text="", wraplength=720, takefocus=True)
        summary.pack(anchor="w", pady=(4, 4))

        for index, model in enumerate(available):
            listbox.insert("end", f"{model['id']} ({model['filename']})")
            if model["id"] == (
                self.selected_models[0] if self.selected_models else None
            ):
                listbox.select_set(index)

        preview_frame = ttk.LabelFrame(
            container,
            text="Model preview",
            padding=(8, 6),
        )
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        preview_text = tk.Text(
            preview_frame,
            wrap="none",
            borderwidth=1,
            relief="solid",
        )
        preview_text.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(
            preview_frame, orient="vertical", command=preview_text.yview
        )
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll = ttk.Scrollbar(
            preview_frame, orient="horizontal", command=preview_text.xview
        )
        hscroll.grid(row=1, column=0, sticky="ew")
        preview_text.configure(
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )

        def _refresh_model_preview(entry: dict | None = None) -> None:
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            if entry:
                try:
                    content = self._read_asset_text(entry["path"])
                except Exception as exc:
                    content = f"Unable to load {entry['id']}: {exc}"
                preview_text.insert("1.0", content)
            else:
                preview_text.insert(
                    "1.0",
                    "Select a model to preview its YAML definition.",
                )
            preview_text.configure(state="disabled")
            preview_text.yview_moveto(0)

        def _refresh_model_selection(_: tk.Event | None = None) -> None:
            indices = listbox.curselection()
            if indices:
                entry = available[indices[0]]
                selected_model = entry["id"]
                self.selected_models = [selected_model]
                self.draft.model = selected_model
                self._selected_model_entry = entry
                summary.config(text=f"Selected model: {selected_model}")
                _refresh_model_preview(entry)
            else:
                self.selected_models = []
                self.draft.model = ""
                self._selected_model_entry = None
                summary.config(text="No model selected yet.")
                _refresh_model_preview(None)
            self._refresh_builder_step_indicators()

        listbox.bind("<<ListboxSelect>>", _refresh_model_selection)
        _refresh_model_selection()

    def _render_builder_step_data(self, container: tk.Frame) -> None:
        ttk.Label(
            container,
            text="Data selection",
            font=("Helvetica", 14, "bold"),
            takefocus=True,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            container,
            text=(
                "Pick one dataset per observation type "
                "(SNe, BAO, CMB, etc.). Each list is "
                "scoped to its data category so selections "
                "remain clear and auditable."
            ),
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        entries = sorted(
            self.catalogue_index.values(), key=lambda entry: entry["id"]
        )
        if not entries:
            ttk.Label(
                container,
                text="No datasets registered; run inventory refresh first.",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            return
        type_groups: dict[str, list[dict]] = {}
        for record in entries:
            dtype = record.get("type", "other").lower()
            type_groups.setdefault(dtype, []).append(record)
        ordered_types = [
            dtype for dtype in ("sne", "bao", "cmb") if dtype in type_groups
        ] + sorted(
            dtype
            for dtype in type_groups
            if dtype not in ("sne", "bao", "cmb")
        )
        catalogue_panel = self._create_scrollable_panel(container, height=360)
        listboxes: dict[str, tk.Listbox] = {}
        id_lookup: dict[str, tuple[str, int]] = {}

        def _add_dataset_section(
            parent: tk.Frame,
            dtype: str,
            records: list[dict],
            *,
            pack_kwargs: dict[str, object] | None = None,
            height: int | None = None,
        ) -> tk.Listbox:
            frame = ttk.LabelFrame(
                parent,
                text=f"{dtype.upper()} datasets",
                padding=(6, 4),
            )
            pack_kwargs = pack_kwargs or {"fill": "x", "pady": 4}
            frame.pack(**pack_kwargs)
            ttk.Label(
                frame,
                text=(
                    f"{len(records)} "
                    f"{'candidate' if len(records) == 1 else 'candidates'}"
                ),
                takefocus=True,
            ).pack(anchor="w")
            list_frame = ttk.Frame(frame)
            list_frame.pack(fill="both", expand=True, pady=(4, 0))
            listbox = tk.Listbox(
                list_frame,
                height=(
                    height
                    if height is not None
                    else min(8, max(4, len(records)))
                ),
                selectmode="browse",
                exportselection=False,
                width=32,
            )
            listbox.pack(side="left", fill="both", expand=True)
            scroll = ttk.Scrollbar(
                list_frame, orient="vertical", command=listbox.yview
            )
            scroll.pack(side="right", fill="y")
            listbox.configure(yscrollcommand=scroll.set)
            for index, record in enumerate(records):
                listbox.insert(
                    "end",
                    f"{record['id']} – {record.get('name', record['id'])}",
                )
                id_lookup[record["id"]] = (dtype, index)
            listboxes[dtype] = listbox
            return listbox

        primary_types = [
            dtype for dtype in ("sne", "bao", "cmb") if dtype in type_groups
        ]
        additional_types = [
            dtype for dtype in ordered_types if dtype not in primary_types
        ]
        if primary_types:
            primary_row = ttk.Frame(catalogue_panel)
            primary_row.pack(fill="x", pady=(0, 4))
            for dtype in primary_types:
                records = type_groups[dtype]
                _add_dataset_section(
                    primary_row,
                    dtype,
                    records,
                    pack_kwargs={
                        "side": "left",
                        "fill": "both",
                        "expand": True,
                        "padx": 4,
                    },
                    height=min(12, max(6, len(records))),
                )
        for dtype in additional_types:
            _add_dataset_section(
                catalogue_panel,
                dtype,
                type_groups[dtype],
            )
        detail_label = ttk.Label(
            container,
            text="Select datasets to preview details.",
            wraplength=720,
            takefocus=True,
        )
        detail_label.pack(anchor="w", pady=(6, 6))
        selection_map: dict[str, dict] = {}
        focus_state: dict[str, dict | None] = {"record": None}

        def _refresh_data_selection() -> None:
            selection_map.clear()
            selected_records: list[dict] = []
            for dtype in ordered_types:
                listbox = listboxes[dtype]
                indices = listbox.curselection()
                if not indices:
                    continue
                record = type_groups[dtype][indices[0]]
                selection_map[dtype] = record
                selected_records.append(record)
            self.selected_datasets = [
                self._dataset_manifest_record(record)
                for record in selected_records
            ]
            ids = [record["id"] for record in selected_records]
            self.draft.data = ", ".join(ids)
            if selected_records:
                focus_state["record"] = selected_records[0]
                first = selected_records[0]
                info_lines = [
                    (
                        f"{first['name']} ({first['id']}) "
                        f"[{first.get('type', '').upper()}]"
                    ),
                    "Badges: " + ", ".join(first.get("badges", [])),
                    f"License: {first.get('license', 'unspecified')}",
                    f"Version: {first.get('version', 'unknown')}",
                ]
                detail_label.config(text="\n".join(info_lines))
            else:
                detail_label.config(
                    text=(
                        "No datasets selected yet; highlight an entry to "
                        "inspect it."
                    )
                )
            self._refresh_builder_step_indicators()

        def _update_focus(dtype: str) -> None:
            listbox = listboxes[dtype]
            indices = listbox.curselection()
            if indices:
                focus_state["record"] = type_groups[dtype][indices[0]]

        for dtype, listbox in listboxes.items():
            listbox.bind(
                "<<ListboxSelect>>",
                lambda _evt, t=dtype: (
                    _update_focus(t),
                    _refresh_data_selection(),
                ),
            )
        for dataset in self.selected_datasets:
            lookup = id_lookup.get(dataset["id"])
            if lookup:
                dtype, index = lookup
                listboxes[dtype].select_set(index)
        _refresh_data_selection()
        action_row = ttk.Frame(container)
        action_row.pack(anchor="w")

        def _open_focused_folder() -> None:
            record = focus_state["record"]
            if record:
                self._open_folder_or_warn(
                    record.get("path", ""),
                    context="data",
                    subject=f"dataset {record.get('id', '')}",
                )
            else:
                self.create_toast(
                    "No dataset folder available to open.",
                    severity="WARNING",
                    context="data",
                )

        def _view_focused_metadata() -> None:
            record = focus_state["record"]
            if record:
                self._present_metadata(
                    record["id"], f"Dataset metadata: {record['id']}"
                )
            else:
                self.create_toast(
                    "Select a dataset before viewing metadata.",
                    severity="WARNING",
                    context="data",
                )

        def _revalidate_focused_parser() -> None:
            record = focus_state["record"]
            if record:
                self._revalidate_dataset_action(record["id"])
            else:
                self.create_toast(
                    "Choose a dataset to revalidate.",
                    severity="WARNING",
                    context="data",
                )

        ttk.Button(
            action_row,
            text="Open folder",
            command=_open_focused_folder,
        ).pack(side="left", padx=2)
        ttk.Button(
            action_row,
            text="View metadata",
            command=_view_focused_metadata,
        ).pack(side="left", padx=2)
        ttk.Button(
            action_row,
            text="Revalidate parser",
            command=_revalidate_focused_parser,
        ).pack(side="left", padx=2)

    def _render_builder_step_engine(self, container: tk.Frame) -> None:
        ttk.Label(
            container,
            text="Engine selection",
            font=("Helvetica", 14, "bold"),
            takefocus=True,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            container,
            text=("Choose the computational backend to run your models."),
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        options = sorted(
            self.engine_index.values(), key=lambda entry: entry["label"]
        )
        if not options:
            ttk.Label(
                container,
                text=(
                    "No engines discovered; ensure the engines folder is "
                    "populated."
                ),
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            return
        display_map: dict[str, dict] = {}
        choices: list[str] = []
        for entry in options:
            label = entry["label"] or entry["stem"]
            display = f"{label} (v{entry.get('version', 'unknown')})"
            display_map[display] = entry
            choices.append(display)
        initial_display = next(
            (
                key
                for key, value in display_map.items()
                if value["id"] == self.selected_engine
            ),
            choices[0],
        )
        combo_var = tk.StringVar(value=initial_display)
        combo = ttk.Combobox(
            container,
            textvariable=combo_var,
            values=choices,
            state="readonly",
            width=48,
        )
        combo.pack(anchor="w", pady=(6, 6))

        def _apply_engine_selection(_: tk.Event | None = None) -> None:
            selection = combo_var.get()
            record = display_map.get(selection)
            if record:
                self.selected_engine = record["id"]
                self.draft.engine = record["id"]
                self._selected_engine_entry = record
                self.engine_capabilities = self._load_engine_capabilities(
                    record["id"]
                )
                detail_label.config(
                    text=(
                        f"{record['label']} uses module {record['id']} "
                        f"with SHA256 {record.get('hash', '')}."
                    )
                )
                self._refresh_builder_step_indicators()

        combo.bind("<<ComboboxSelected>>", _apply_engine_selection)
        detail_label = ttk.Label(
            container,
            text="Select an engine to see details.",
            wraplength=720,
            takefocus=True,
        )
        detail_label.pack(anchor="w", pady=(4, 4))
        _apply_engine_selection()
        button_frame = ttk.Frame(container)
        button_frame.pack(anchor="w", pady=(4, 0))

        def _open_selected_engine_folder() -> None:
            selection = combo_var.get()
            record = display_map.get(selection)
            if record:
                self._open_folder_or_warn(
                    os.path.dirname(record["path"]),
                    context="engines",
                    subject=f"engine {record['label']}",
                )
            else:
                self.create_toast(
                    "Choose an engine before opening its folder.",
                    severity="WARNING",
                    context="engine",
                )

        def _view_selected_engine_module() -> None:
            selection = combo_var.get()
            record = display_map.get(selection)
            if record:
                self._present_metadata(
                    record["id"], f"Engine module: {record['label']}"
                )
            else:
                self.create_toast(
                    "Select an engine before viewing its module.",
                    severity="WARNING",
                    context="engine",
                )

        ttk.Button(
            button_frame,
            text="Open engine folder",
            command=_open_selected_engine_folder,
        ).pack(side="left", padx=2)
        ttk.Button(
            button_frame,
            text="View module",
            command=_view_selected_engine_module,
        ).pack(side="left", padx=2)
        _apply_engine_selection()
        self._render_engine_knobs(container)
        self._render_engine_run_settings(container)

    def _render_engine_knobs(self, container: tk.Frame) -> None:
        """Expose the knobs declared by each engine alongside descriptions."""

        if self._engine_knobs_frame is not None:
            self._engine_knobs_frame.destroy()
            self._engine_knobs_frame = None
        capabilities = self.engine_capabilities
        self._engine_setting_specs.clear()
        if not capabilities or not capabilities.settings:
            return
        knobs_frame = ttk.LabelFrame(
            container,
            text="Engine knobs",
            padding=(8, 6),
        )
        knobs_frame.pack(fill="x", pady=(8, 0))
        self._engine_knobs_frame = knobs_frame
        for setting in capabilities.settings:
            row = ttk.Frame(knobs_frame)
            row.pack(fill="x", pady=(4, 0))
            ttk.Label(
                row,
                text=f"{setting.label} ({setting.key})",
                width=26,
                takefocus=True,
            ).pack(side="left")
            var = self._engine_setting_vars.get(setting.key)
            if var is None:
                initial = (
                    str(setting.default) if setting.default is not None else ""
                )
                var = tk.StringVar(value=initial)
                self._engine_setting_vars[setting.key] = var
            ttk.Entry(
                row,
                textvariable=var,
                width=16,
            ).pack(side="left", padx=(6, 0))
            self._engine_setting_specs[setting.key] = setting
            metadata: list[str] = []
            if setting.description:
                metadata.append(setting.description)
            if setting.default is not None:
                metadata.append(f"Default: {setting.default}")
            if setting.hint:
                metadata.append(f"Hint: {setting.hint}")
            if metadata:
                ttk.Label(
                    knobs_frame,
                    text=" ".join(metadata),
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w", padx=(16, 0))
        if capabilities.progress_chunks:
            chunk_labels = ", ".join(
                chunk.label for chunk in capabilities.progress_chunks
            )
            ttk.Label(
                knobs_frame,
                text=f"Progress reports: {chunk_labels}",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))

    def _collect_engine_knob_settings(self) -> dict[str, object]:
        """Return the sanitized knob values entered by the operator."""

        settings: dict[str, object] = {}
        for key, var in self._engine_setting_vars.items():
            spec = self._engine_setting_specs.get(key)
            if spec is None:
                continue
            raw_value = var.get().strip()
            if not raw_value:
                continue
            parsed = self._parse_engine_setting_value(
                raw_value, spec.dtype or "str"
            )
            settings[key] = parsed
        return settings

    @staticmethod
    def _parse_engine_setting_value(raw_value: str, dtype: str) -> object:
        """Convert a string knob value into the declared dtype.

        Returns the raw input when conversion fails.
        """

        dtype_key = dtype.lower()
        if dtype_key == "int":
            try:
                return int(raw_value)
            except ValueError:
                return raw_value
        if dtype_key == "float":
            try:
                return float(raw_value)
            except ValueError:
                return raw_value
        if dtype_key == "bool":
            normalized = raw_value.lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return raw_value
        return raw_value

    def _render_engine_run_settings(self, container: tk.Frame) -> None:
        """Render engine-run tuning inputs next to the engine selector."""

        settings_frame = ttk.LabelFrame(
            container,
            text="Run settings",
        )
        settings_frame.pack(fill="x", pady=(8, 0))
        recommendations = self._compute_run_recommendations()
        hints = [
            (
                "Production steps control the total sampler iterations. "
                "Recommended default: "
                f"{recommendations['recommended_steps']}."
            ),
            (
                "Burn-in steps discard early samples so the chain can "
                "stabilise. Recommended warm-up: "
                f"{recommendations['burn_in_recommended']}. "
                f"Quicker option: {recommendations['quick_burn']}."
            ),
            (
                "Walkers sample the posterior in parallel; more walkers "
                f"boost convergence. Required minimum: "
                f"{recommendations['minimum_walkers']}."
            ),
            (
                "Worker pools accelerate sampling by spreading walkers across "
                "processes. Recommended pool size: "
                f"{recommendations['pool_max']} "
                f"(detected CPUs: {recommendations['cpu_label']}). "
                "Enter 0 to disable multiprocessing entirely."
            ),
        ]
        for tip in hints:
            ttk.Label(
                settings_frame,
                text=tip,
                wraplength=720,
                justify="left",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
        field_specs = [
            (
                "Walkers",
                "walkers",
                "Number of ensemble walkers",
                (
                    f"Minimum: {recommendations['minimum_walkers']}  "
                    f"Recommended: {recommendations['recommended_walkers']}"
                ),
            ),
            (
                "Burn-in steps",
                "burn_in",
                "Iterations used for burn-in",
                (
                    "Recommended warm-up: "
                    f"{recommendations['burn_in_recommended']}  "
                    f"Quicker option: {recommendations['quick_burn']}"
                ),
            ),
            (
                "Production steps",
                "production_steps",
                "Iterations used during production",
                (
                    "Recommended default: "
                    f"{recommendations['recommended_steps']}  "
                    f"Minimum suggested: {recommendations['production_min']}"
                ),
            ),
            (
                "Pool size",
                "pool_size",
                "Multiprocessing pool size",
                (
                    f"Recommended pool: {recommendations['pool_max']}  "
                    f"Detected CPUs: {recommendations['cpu_label']}"
                ),
            ),
        ]

        def _track(attribute: str, string_var: tk.StringVar) -> None:
            string_var.trace_add(
                "write",
                lambda *_: setattr(self.draft, attribute, string_var.get()),
            )

        for label_text, field_name, helper, recommendation in field_specs:
            var = tk.StringVar(value=getattr(self.draft, field_name))
            _track(field_name, var)
            field_row = ttk.Frame(settings_frame)
            field_row.pack(anchor="w", pady=(4, 0))
            ttk.Label(
                field_row,
                text=f"{label_text}:",
                width=16,
                takefocus=True,
            ).pack(side="left")
            ttk.Entry(
                field_row,
                textvariable=var,
                width=16,
            ).pack(side="left")
            ttk.Label(
                field_row,
                text=helper,
                wraplength=420,
                takefocus=True,
            ).pack(side="left", padx=(8, 0))
            ttk.Label(
                settings_frame,
                text=recommendation,
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", padx=(16, 0))

    def _render_builder_step_manifest(self, container: tk.Frame) -> None:
        ttk.Label(
            container,
            text="Run plan",
            font=("Helvetica", 14, "bold"),
            takefocus=True,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            container,
            text="Describe the production plan or testing notes for this run.",
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        plan_var = tk.StringVar(value=self.draft.plan)

        def _update_plan(*_args: object) -> None:
            self.draft.plan = plan_var.get()

        plan_var.trace_add("write", _update_plan)
        ttk.Entry(
            container,
            textvariable=plan_var,
            width=80,
        ).pack(anchor="w", pady=(6, 0))

        manifest_frame = ttk.LabelFrame(
            container,
            text="Manifest",
        )
        manifest_frame.pack(fill="both", expand=True, pady=(12, 0))
        status_text = (
            f"Saved: {self.manifest_workspace.manifest_path}"
            if self.manifest_workspace
            else "Manifest not saved yet."
        )
        ttk.Label(
            manifest_frame,
            text=status_text,
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w", pady=(0, 4))
        preview_panel = ttk.Frame(manifest_frame)
        preview_panel.pack(fill="both", expand=True)
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)
        manifest_text_widget = tk.Text(
            preview_panel,
            wrap="none",
            borderwidth=1,
            relief="solid",
            height=16,
        )
        manifest_text_widget.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(
            preview_panel,
            orient="vertical",
            command=manifest_text_widget.yview,
        )
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll = ttk.Scrollbar(
            preview_panel,
            orient="horizontal",
            command=manifest_text_widget.xview,
        )
        hscroll.grid(row=1, column=0, sticky="ew")
        manifest_text_widget.configure(
            yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
        )
        manifest_text_widget.insert("1.0", self._manifest_preview_text())
        manifest_text_widget.configure(state="disabled")

        reminder_text = (
            self._MANIFEST_REMINDER_MESSAGE
            if self.manifest_workspace is None
            else "Manifest saved; proceed to confirmation."
        )
        ttk.Label(
            container,
            text=reminder_text,
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w", pady=(8, 0))

        actions_frame = ttk.Frame(container)
        actions_frame.pack(anchor="w", pady=(12, 0))

        def _save_manifest_action() -> None:
            self._persist_manifest_workspace(notify=True)
            self.show_run_builder()

        save_state = tk.NORMAL if self._builder_ready() else tk.DISABLED
        clear_state = tk.NORMAL if self._has_configuration() else tk.DISABLED
        open_state = (
            tk.NORMAL if self.manifest_workspace is not None else tk.DISABLED
        )
        ttk.Button(
            actions_frame,
            text="Save manifest",
            command=_save_manifest_action,
            state=save_state,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            actions_frame,
            text="Save and confirm",
            command=self._save_manifest_and_proceed,
            state=save_state,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            actions_frame,
            text="Save to external folder...",
            command=self._save_manifest_to_external_folder,
            state=save_state,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            actions_frame,
            text="Clear manifest",
            command=lambda: (
                self._clear_manifest_configuration(),
                self.show_run_builder(),
            ),
            state=clear_state,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            actions_frame,
            text="Open manifest...",
            command=self._open_manifest_file,
            state=open_state,
        ).pack(side="left", padx=(0, 4))

    def _manifest_preview_text(self) -> str:
        """Return a textual snapshot for the manifest preview area."""

        manifest = self.pending_manifest
        if manifest is None:
            manifest = self._ensure_manifest_snapshot()
        if manifest is None:
            return (
                "Manifest preview unavailable; complete the builder to "
                "generate a snapshot."
            )
        try:
            return yaml.safe_dump(manifest, sort_keys=False)
        except Exception:
            return json.dumps(manifest, indent=2)

    def _open_manifest_file(self) -> None:
        """Open the currently saved manifest using the OS default handler."""

        if self.manifest_workspace is None:
            self.create_toast(
                "Save a manifest before opening it.",
                severity="WARNING",
                context="run",
            )
            return
        self._open_path_with_system(str(self.manifest_workspace.manifest_path))

    def _stage_confirm_manifest(self) -> None:
        """Capture builder selections as a manifest snapshot for later use."""

        manifest = self._ensure_manifest_snapshot()
        if manifest is None:
            self._staged_confirm_manifest = None
            return
        self._staged_confirm_manifest = copy.deepcopy(manifest)

    def _render_builder_step_confirm(self, container: tk.Frame) -> None:
        self._stage_confirm_manifest()
        ttk.Label(
            container,
            text="Confirm and start",
            font=("Helvetica", 14, "bold"),
            takefocus=True,
        ).pack(anchor="w", pady=(0, 6))
        summary_frame = ttk.Frame(container)
        summary_frame.pack(anchor="w")
        summary_entries = [
            ("Seed", self.draft.seed or "unset"),
            (
                "Models",
                ", ".join(self.selected_models) or "no models selected",
            ),
            (
                "Datasets",
                ", ".join(entry["id"] for entry in self.selected_datasets)
                or "no datasets selected",
            ),
            ("Engine", self.selected_engine or "unspecified"),
            ("Plan", self.draft.plan or "no plan provided"),
        ]
        summary_entries.extend(
            [
                ("Walkers", self.draft.walkers or "unset"),
                ("Burn-in", self.draft.burn_in or "unset"),
                ("Production", self.draft.production_steps or "unset"),
                ("Pool size", self.draft.pool_size or "unset"),
            ]
        )
        for label, value in summary_entries:
            ttk.Label(
                summary_frame,
                text=f"{label}: {value}",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w")
        button_frame = ttk.Frame(container)
        button_frame.pack(anchor="w", pady=(12, 0))
        start_state = (
            tk.NORMAL if self.manifest_workspace is not None else tk.DISABLED
        )
        ttk.Button(
            button_frame,
            text="Start run",
            command=self.confirm_start_run,
            state=start_state,
            takefocus=True,
            **self._monitor_button_kwargs(),
        ).pack(side="left")

    def _dataset_manifest_record(self, entry: dict) -> dict[str, object]:
        """Return the manifest-safe dict for a dataset catalogue entry."""

        independence = entry.get("independence", [])
        if isinstance(independence, str):
            independence = [independence]
        return {
            "id": entry.get("id", "dataset"),
            "path": entry.get("path", ""),
            "name": entry.get("name", entry.get("id", "dataset")),
            "hashes": entry.get("hashes", {}),
            "version": entry.get("version", "unversioned"),
            "independence": independence,
            "type": entry.get("type", ""),
            "license": entry.get("license", "unspecified"),
            "badges": entry.get("badges", []),
        }

    def show_data_overview(self) -> None:
        """Display dataset catalogue with metadata, hashes and filters."""

        self.refresh_inventory()

        def builder(frame: tk.Frame) -> None:
            ttk.Label(frame, text="Data catalogue", takefocus=True).pack(
                anchor="w"
            )
            ttk.Label(
                frame,
                text=(
                    "Datasets remain selectable from the Run Builder while "
                    "this catalogue surfaces parser digests, citations and "
                    "licensing to prove registration rules are honoured."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            filters = ttk.Frame(frame)
            filters.pack(anchor="w", pady=(0, 8))
            ttk.Label(filters, text="Filters:", takefocus=True).pack(
                side="left", padx=(0, 4)
            )
            ttk.Button(
                filters,
                text="All",
                command=lambda: (
                    self.filter_catalogue([]),
                    self.show_data_overview(),
                ),
                takefocus=True,
            ).pack(side="left", padx=2)
            for key in ("sne", "bao", "cmb"):
                ttk.Button(
                    filters,
                    text=key.upper(),
                    command=lambda k=key: (
                        self.filter_catalogue([k]),
                        self.show_data_overview(),
                    ),
                    takefocus=True,
                ).pack(side="left", padx=2)
            active = self.filter_catalogue(self.last_filter_types or [])
            ttk.Label(
                frame,
                text=(
                    f"Showing {len(active)} dataset(s); trust alerts: "
                    f"{len(self.validation_notes)}"
                ),
                takefocus=True,
            ).pack(anchor="w", pady=(0, 6))
            catalogue_panel = self._create_scrollable_panel(frame)
            for dataset in active:
                entry_frame = ttk.LabelFrame(
                    catalogue_panel,
                    text=f"{dataset['name']} ({dataset['id']})",
                    padding=(8, 6),
                )
                entry_frame.pack(fill="x", pady=4)
                ttk.Label(
                    entry_frame,
                    text=" | ".join(dataset.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                citation_value = dataset.get("citation", "missing")
                license_value = dataset.get("license", "unspecified")
                parser_digest = dataset.get("parser_digest", "n/a")
                metadata_digest = dataset.get("metadata_digest", "")
                ttk.Label(
                    entry_frame,
                    text=(
                        f"Citation: {citation_value}\n"
                        f"License: {license_value}\n"
                        f"Parser SHA256: {parser_digest}\n"
                        f"Metadata SHA256: {metadata_digest}"
                    ),
                    wraplength=720,
                    justify="left",
                    takefocus=True,
                ).pack(anchor="w", pady=(4, 4))
                ttk.Label(
                    entry_frame,
                    text="Independence notes: "
                    + "; ".join(dataset.get("independence", [])),
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w", pady=(0, 4))
                actions = ttk.Frame(entry_frame)
                actions.pack(anchor="w")
                dataset_id = dataset["id"]

                def _open_current_folder(
                    path: str = dataset["path"], ds: str = dataset_id
                ) -> None:
                    self._open_folder_or_warn(
                        path,
                        context="data",
                        subject=f"dataset {ds}",
                    )

                ttk.Button(
                    actions,
                    text="Open folder",
                    command=_open_current_folder,
                    takefocus=True,
                ).pack(side="left", padx=2)

                def _view_current_metadata() -> None:
                    self._present_metadata(
                        dataset_id, f"Dataset metadata: {dataset_id}"
                    )

                def _revalidate_current_parser() -> None:
                    self._revalidate_dataset_action(dataset_id)

                ttk.Button(
                    actions,
                    text="View metadata",
                    command=_view_current_metadata,
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="Revalidate parser",
                    command=_revalidate_current_parser,
                    takefocus=True,
                ).pack(side="left", padx=2)

        self._swap_content(builder)

    def show_models(self) -> None:
        """Display installed model definitions and digests."""

        self.refresh_inventory()

        def builder(frame: tk.Frame) -> None:
            ttk.Label(frame, text="Models", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "Model YAML files drive compatibility badges and priors. "
                    "This view lists their hashes so Run Builder choices can "
                    "be audited alongside dataset digests."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            ttk.Label(
                frame,
                text=f"Discovered {len(self.model_index)} model(s)",
                takefocus=True,
            ).pack(anchor="w", pady=(0, 6))
            catalogue_panel = self._create_scrollable_panel(frame)
            for model in sorted(
                self.model_index.values(), key=lambda entry: entry["id"]
            ):
                entry_frame = ttk.LabelFrame(
                    catalogue_panel,
                    text=f"{model['id']} ({model['filename']})",
                    padding=(8, 6),
                )
                entry_frame.pack(fill="x", pady=4)
                ttk.Label(
                    entry_frame,
                    text="Badges: " + ", ".join(model.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                ttk.Label(
                    entry_frame,
                    text=(
                        f"SHA256: {model.get('hash', '')}\n"
                        f"License: {model.get('license', 'unspecified')}\n"
                        f"Version: {model.get('version', 'unknown')}"
                    ),
                    wraplength=720,
                    takefocus=True,
                    justify="left",
                ).pack(anchor="w", pady=(4, 4))
                actions = ttk.Frame(entry_frame)
                actions.pack(anchor="w")
                model_folder = os.path.dirname(model["path"])
                model_id = model["id"]

                def _open_model_folder() -> None:
                    self._open_folder_or_warn(
                        model_folder,
                        context="models",
                        subject=f"model {model_id}",
                    )

                def _view_model_yaml() -> None:
                    self._present_metadata(
                        model_id, f"Model definition: {model_id}"
                    )

                ttk.Button(
                    actions,
                    text="Open model folder",
                    command=_open_model_folder,
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="View YAML",
                    command=_view_model_yaml,
                    takefocus=True,
                ).pack(side="left", padx=2)

        self._swap_content(builder)

    def show_engines(self) -> None:
        """Display engine overview panel with digests and health checks."""

        self.refresh_inventory()

        def builder(frame: tk.Frame) -> None:
            ttk.Label(frame, text="Engines", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "Engines expose dataset compatibility and sampler labels. "
                    "Hashes appear here so health checks can confirm which "
                    "module executed a run."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            ttk.Label(
                frame,
                text=f"Discovered {len(self.engine_index)} engine(s)",
                takefocus=True,
            ).pack(anchor="w", pady=(0, 6))
            catalogue_panel = self._create_scrollable_panel(frame)
            for engine in sorted(
                self.engine_index.values(), key=lambda entry: entry["label"]
            ):
                entry_frame = ttk.LabelFrame(
                    catalogue_panel,
                    text=f"{engine['label']} ({engine['filename']})",
                    padding=(8, 6),
                )
                entry_frame.pack(fill="x", pady=4)
                ttk.Label(
                    entry_frame,
                    text="Badges: " + ", ".join(engine.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                ttk.Label(
                    entry_frame,
                    text=(
                        f"Version: {engine.get('version', 'unknown')}\n"
                        f"SHA256: {engine.get('hash', '')}"
                    ),
                    wraplength=720,
                    takefocus=True,
                    justify="left",
                ).pack(anchor="w", pady=(4, 4))
                actions = ttk.Frame(entry_frame)
                actions.pack(anchor="w")
                engine_folder = os.path.dirname(engine["path"])
                engine_id = engine["id"]
                engine_label = engine["label"] or engine["stem"]

                def _open_engine_folder() -> None:
                    self._open_folder_or_warn(
                        engine_folder,
                        context="engines",
                        subject=f"engine {engine_label}",
                    )

                def _view_engine_module() -> None:
                    self._present_metadata(
                        engine_id, f"Engine module: {engine_label}"
                    )

                ttk.Button(
                    actions,
                    text="Open engine folder",
                    command=_open_engine_folder,
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="View module",
                    command=_view_engine_module,
                    takefocus=True,
                ).pack(side="left", padx=2)

        self._swap_content(builder)

    def show_settings(self) -> None:
        """Display settings placeholder panel."""

        def builder(frame: tk.Frame) -> None:
            ttk.Label(frame, text="Settings", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "Adjust notification and logging preferences before "
                    "launching runs. Diagnostics stream from GUI launch "
                    "throughout the session so early environment checks "
                    "remain available."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            diag_frame = ttk.LabelFrame(frame, text="Diagnostics")
            diag_frame.pack(fill="x", pady=(4, 4))
            ttk.Label(
                diag_frame,
                text=f"App log path: {self.application_log_path}",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w")
            self._diagnostics_filter_label = ttk.Label(
                diag_frame,
                text=f"Filter: {self.diagnostics_filter_level}+",
                takefocus=True,
            )
            self._diagnostics_filter_label.pack(anchor="w", pady=(2, 0))
            filter_frame = ttk.Frame(diag_frame)
            filter_frame.pack(anchor="w", pady=(4, 4))
            ttk.Button(
                filter_frame,
                text="Show all",
                command=lambda: self.set_diagnostics_filter("INFO"),
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                filter_frame,
                text="Errors only",
                command=lambda: self.set_diagnostics_filter("ERROR"),
                takefocus=True,
            ).pack(side="left", padx=2)
            text_panel = ttk.Frame(diag_frame)
            text_panel.pack(fill="both", expand=True)
            text_panel.columnconfigure(0, weight=1)
            text_panel.rowconfigure(0, weight=1)
            diag_text = tk.Text(
                text_panel,
                wrap="none",
                padx=8,
                pady=6,
                borderwidth=0,
                highlightthickness=0,
                height=10,
            )
            diag_text.grid(row=0, column=0, sticky="nsew")
            vscroll = ttk.Scrollbar(
                text_panel, orient="vertical", command=diag_text.yview
            )
            vscroll.grid(row=0, column=1, sticky="ns")
            hscroll = ttk.Scrollbar(
                text_panel, orient="horizontal", command=diag_text.xview
            )
            hscroll.grid(row=1, column=0, sticky="ew")
            diag_text.configure(
                yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
            )
            diag_text.configure(state="disabled")
            self._diagnostics_log_widget = diag_text
            actions = ttk.Frame(diag_frame)
            actions.pack(anchor="w", pady=(6, 0))
            ttk.Button(
                actions,
                text="View diagnostics log",
                command=self._view_diagnostics_log,
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                actions,
                text="Open diagnostics log…",
                command=self._open_diagnostics_log,
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                actions,
                text="Flush log",
                command=self._flush_application_log,
                takefocus=True,
            ).pack(side="left", padx=2)

            output_frame = ttk.LabelFrame(frame, text="Output directory")
            output_frame.pack(fill="x", pady=(6, 4))
            output_root = self._output_root()
            output_var = tk.StringVar(value=output_root)
            ttk.Entry(
                output_frame,
                textvariable=output_var,
                width=64,
            ).pack(anchor="w", pady=(4, 0))

            def _apply_output_path() -> None:
                target = output_var.get().strip() or output_root
                os.makedirs(target, exist_ok=True)
                self.create_toast(
                    f"Output directory ready at {target}",
                    severity="INFO",
                    context="settings",
                )

            output_buttons = ttk.Frame(output_frame)
            output_buttons.pack(anchor="w", pady=(4, 0))

            def _open_output_directory() -> None:
                target = output_var.get().strip() or output_root
                os.makedirs(target, exist_ok=True)
                try:
                    self.open_folder(target)
                except FileNotFoundError:
                    self.create_toast(
                        f"Output directory missing at {target}",
                        severity="ERROR",
                        context="settings",
                    )

            ttk.Button(
                output_buttons,
                text="Open directory",
                command=_open_output_directory,
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                output_buttons,
                text="Create/refresh",
                command=_apply_output_path,
                takefocus=True,
            ).pack(side="left", padx=2)

            env_frame = ttk.LabelFrame(frame, text="Environment hints")
            env_frame.pack(fill="x", pady=(6, 4))
            env_values = [
                (
                    "COPERNICAN_SEED",
                    os.environ.get("COPERNICAN_SEED", "unset"),
                ),
                (
                    "COPERNICAN_STRICT_WARNINGS",
                    os.environ.get("COPERNICAN_STRICT_WARNINGS", "0"),
                ),
                (
                    "COPERNICAN_ENABLE_STAGED_MENU",
                    os.environ.get("COPERNICAN_ENABLE_STAGED_MENU", "0"),
                ),
                (
                    "COPERNICAN_DETACH_GUI",
                    os.environ.get("COPERNICAN_DETACH_GUI", "0"),
                ),
            ]
            for name, value in env_values:
                ttk.Label(
                    env_frame,
                    text=f"{name}: {value}",
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w")

        self._swap_content(builder)

    def show_help(self) -> None:
        """Display contextual help panel."""

        def builder(frame: tk.Frame) -> None:
            ttk.Label(
                frame,
                text="Copernican Suite Documentation",
                font=("Helvetica", 16),
                takefocus=True,
            ).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "The repository README is rendered below so you can read "
                    "the same guidance that ships with the suite."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            content_frame = ttk.Frame(frame)
            content_frame.pack(fill="both", expand=True)
            if self.render and tk is not None:
                self._load_help_banner()
                if self.help_banner_image:
                    banner_label = ttk.Label(
                        content_frame, image=self.help_banner_image
                    )
                    banner_label.image = self.help_banner_image
                    banner_label.pack(pady=(0, 8))
                text_frame = ttk.Frame(content_frame)
                text_frame.pack(fill="both", expand=True)
                text_frame.columnconfigure(0, weight=1)
                text_frame.rowconfigure(0, weight=1)
                text_widget = tk.Text(
                    text_frame,
                    wrap="word",
                    padx=8,
                    pady=6,
                    borderwidth=0,
                    highlightthickness=0,
                )
                text_widget.grid(
                    row=0, column=0, sticky="nsew", padx=(0, 0), pady=(0, 0)
                )
                scrollbar = ttk.Scrollbar(
                    text_frame,
                    orient="vertical",
                    command=text_widget.yview,
                )
                scrollbar.grid(row=0, column=1, sticky="ns")
                text_widget.configure(yscrollcommand=scrollbar.set)
                markdown = self._load_help_markdown()
                self._render_markdown_in_text_widget(text_widget, markdown)
                text_widget.configure(state="disabled")
            else:
                ttk.Label(
                    content_frame,
                    text=(
                        "Help content is available from README.md in the "
                        "project root."
                    ),
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w")

        self._swap_content(builder)

    def show_about(self) -> None:
        """Display the ABOUT.md content inside a view window."""

        about_path = Path(__file__).resolve().parents[2] / "ABOUT.md"
        if not about_path.exists():
            self.create_toast(
                "About document is missing; check ABOUT.md",
                severity="ERROR",
                context="about",
            )
            return
        content = self._read_asset_text(str(about_path))
        self._show_metadata_dialog(
            "About the Copernican Suite",
            content,
            str(about_path),
        )

    def exit_suite(self) -> None:
        """Shut down the GUI session and delegate to the CLI exit helper."""

        self._stop_progress_poller()
        self._cancel_monitor_refresh()
        if self._progress_state_path:
            progress_state.clear_progress(self._progress_state_path)
            self._progress_state_path = None
        self._log_program_event("GUI exit requested", logging.INFO)
        try:
            import copernican

            copernican.exit_clean(0)
        except Exception:
            sys.exit(0)

    def show_run_monitor(self) -> None:
        """Display live run status controls."""

        def builder(frame: tk.Frame) -> None:
            self._status_label = None
            self._progress_status_label = None
            self._batch_progressbar = None
            self._walker_progressbar = None
            self._monitor_log_widget = None
            self._monitor_filter_label = None
            header = ttk.Label(
                frame, text="Run Monitor", font=("Helvetica", 16)
            )
            header.pack(anchor="w", pady=(0, 8))
            self._status_label = ttk.Label(
                frame,
                text=self._status_text(),
                takefocus=True,
            )
            self._status_label.pack(anchor="w")
            progress_frame = ttk.Frame(frame)
            progress_frame.pack(fill="x", pady=(8, 8))
            self._progress_status_label = ttk.Label(
                progress_frame,
                text="Stage: Idle",
                font=("Helvetica", 12, "bold"),
                takefocus=True,
            )
            self._progress_status_label.pack(anchor="w")
            ttk.Label(
                progress_frame,
                text="Overall batch progress",
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))
            self._batch_progressbar = ttk.Progressbar(
                progress_frame, maximum=100, length=360
            )
            self._batch_progressbar.pack(fill="x", pady=(2, 0))
            ttk.Label(
                progress_frame,
                text="Walker progress",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            self._walker_progressbar = ttk.Progressbar(
                progress_frame, maximum=100, length=360
            )
            self._walker_progressbar.pack(fill="x", pady=(2, 0))
            meta_frame = ttk.Frame(frame)
            meta_frame.pack(anchor="w", pady=(4, 4))
            ttk.Label(
                meta_frame,
                text="Active manifest metadata:",
                takefocus=True,
            ).pack(anchor="w")
            for line in self.summary.manifest_metadata:
                ttk.Label(meta_frame, text=line, takefocus=True).pack(
                    anchor="w"
                )
            log_frame = ttk.LabelFrame(frame, text="Run logs")
            log_frame.pack(fill="both", expand=True, pady=(8, 4))
            self._monitor_filter_label = ttk.Label(
                log_frame,
                text=f"Filter: {self.monitor_filter_level}+",
                takefocus=True,
            )
            self._monitor_filter_label.pack(anchor="w")
            log_filters = ttk.Frame(log_frame)
            log_filters.pack(anchor="w", pady=(2, 4))
            ttk.Button(
                log_filters,
                text="Info",
                command=lambda: self.set_monitor_filter("INFO"),
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                log_filters,
                text="Warnings",
                command=lambda: self.set_monitor_filter("WARNING"),
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                log_filters,
                text="Errors",
                command=lambda: self.set_monitor_filter("ERROR"),
                takefocus=True,
            ).pack(side="left", padx=2)
            text_panel = ttk.Frame(log_frame)
            text_panel.pack(fill="both", expand=True)
            text_panel.columnconfigure(0, weight=1)
            text_panel.rowconfigure(0, weight=1)
            text_widget = tk.Text(
                text_panel,
                wrap="none",
                padx=8,
                pady=6,
                borderwidth=0,
                highlightthickness=0,
                height=12,
            )
            text_widget.grid(row=0, column=0, sticky="nsew")
            vscroll = ttk.Scrollbar(
                text_panel,
                orient="vertical",
                command=text_widget.yview,
            )
            vscroll.grid(row=0, column=1, sticky="ns")
            hscroll = ttk.Scrollbar(
                text_panel,
                orient="horizontal",
                command=text_widget.xview,
            )
            hscroll.grid(row=1, column=0, sticky="ew")
            text_widget.configure(
                yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
            )
            text_widget.configure(state="disabled")
            self._monitor_log_widget = text_widget
            log_actions = ttk.Frame(log_frame)
            log_actions.pack(anchor="w", pady=(4, 0))
            button_style = self._monitor_button_kwargs()
            self._monitor_log_view_button = ttk.Button(
                log_actions,
                text="View log",
                command=self._view_run_log,
                takefocus=True,
                **button_style,
            )
            self._monitor_log_view_button.pack(side="left", padx=2)
            self._monitor_log_open_button = ttk.Button(
                log_actions,
                text="Open log…",
                command=self._open_run_log_file,
                takefocus=True,
                **button_style,
            )
            self._monitor_log_open_button.pack(side="left", padx=2)
            alerts = ttk.LabelFrame(frame, text="Active alerts")
            alerts.pack(fill="x", pady=(4, 8))
            for alert in self.alerts[-5:]:
                alert_row = ttk.Frame(alerts)
                alert_row.pack(anchor="w", fill="x", pady=(2, 0))
                ttk.Label(
                    alert_row,
                    text=(
                        f"{alert.severity}: {alert.message} "
                        f"(anchor {alert.anchor})"
                    ),
                    wraplength=720,
                    takefocus=True,
                ).pack(side="left", padx=2)
                jump_target = alert.anchor

                def _jump(anchor: str = jump_target) -> None:
                    self.jump_to_log_anchor(anchor)

                ttk.Button(
                    alert_row,
                    text="Jump to log",
                    command=_jump,
                    takefocus=True,
                ).pack(side="left", padx=2)
            controls = ttk.Frame(frame)
            controls.pack(anchor="w")
            button_style = self._monitor_button_kwargs()
            self._run_output_button = ttk.Button(
                controls,
                text="Open run output",
                command=self.open_current_run_output,
                takefocus=True,
                **button_style,
            )
            self._run_output_button.pack(side="left", padx=4)
            self._cancel_button = ttk.Button(
                controls,
                text="Cancel",
                command=self.cancel_run,
                takefocus=True,
                **button_style,
            )
            self._cancel_button.pack(side="left", padx=4)
            self._pause_button = ttk.Button(
                controls,
                text="Pause",
                command=self.pause_run,
                takefocus=True,
                **button_style,
            )
            self._pause_button.pack(side="left", padx=4)
            self._hard_stop_button = ttk.Button(
                controls,
                text="Hard Stop",
                command=self.stop_run,
                takefocus=True,
                **button_style,
            )
            self._hard_stop_button.pack(side="left", padx=4)
            self._monitor_control_buttons = [
                self._cancel_button,
                self._pause_button,
                self._hard_stop_button,
            ]
            self._update_monitor_controls_state()
            self._refresh_monitor_widgets()
            self._schedule_monitor_refresh()

        self._swap_content(builder)

    def _status_text(self) -> str:
        """Return the formatted status line for the run monitor."""

        suffix = (
            f" – {self.current_phase}"
            if getattr(self, "current_phase", "")
            else ""
        )
        return f"Status: {self.status.value}{suffix}"

    def _refresh_status_label(self) -> None:
        """Update the status label text if it is visible."""

        if self._status_label is not None:
            self._status_label.configure(text=self._status_text())

    def _update_monitor_controls_state(self) -> None:
        """Enable or disable monitor buttons based on current state."""

        run_active = self.status is RunStatus.RUNNING
        control_state = tk.NORMAL if run_active else tk.DISABLED
        for button in self._monitor_control_buttons:
            if button:
                button.configure(state=control_state)
        log_available = bool(self.run_log_path)
        log_state = tk.NORMAL if log_available else tk.DISABLED
        if self._monitor_log_view_button:
            self._monitor_log_view_button.configure(state=log_state)
        if self._monitor_log_open_button:
            self._monitor_log_open_button.configure(state=log_state)
        output_available = bool(
            self._current_run_output_dir
        ) and os.path.isdir(self._current_run_output_dir)
        if self._run_output_button:
            self._run_output_button.configure(
                state=tk.NORMAL if output_available else tk.DISABLED
            )

    def show_summary(self) -> None:
        """Display the completion summary with manifest reuse actions."""

        def builder(frame: tk.Frame) -> None:
            header = ttk.Label(
                frame, text="Run Summary", font=("Helvetica", 16)
            )
            header.pack(anchor="w", pady=(0, 8))
            ttk.Label(frame, text="Outputs", takefocus=True).pack(anchor="w")
            for link in self.summary.output_links:
                ttk.Label(frame, text=link, takefocus=True).pack(anchor="w")
            ttk.Label(frame, text="Manifest actions", takefocus=True).pack(
                anchor="w", pady=(12, 0)
            )
            for action in self.summary.manifest_actions:
                ttk.Button(
                    frame, text=action, command=self._noop, takefocus=True
                ).pack(anchor="w", pady=2)
            ttk.Label(frame, text="Manifest metadata", takefocus=True).pack(
                anchor="w", pady=(12, 0)
            )
            for line in self.summary.manifest_metadata:
                ttk.Label(frame, text=line, takefocus=True).pack(anchor="w")

        self._swap_content(builder)

    def next_step(self) -> None:
        """Advance the builder step index while clamping to the final stage."""

        if self.current_step_index < len(self.builder_steps) - 1:
            self.current_step_index += 1
        self.show_run_builder()

    def previous_step(self) -> None:
        """Move back one builder step when possible."""

        if self.current_step_index > 0:
            self.current_step_index -= 1
        self.show_run_builder()

    def jump_to_step(self, step_index: int) -> None:
        """Jump directly to any builder step."""

        confirm_index = self.builder_steps.index("Confirm and start")
        if step_index == confirm_index and not self._can_enter_confirm():
            self._notify_manifest_save_required()
            return
        if 0 <= step_index < len(self.builder_steps):
            self.current_step_index = step_index
        self.show_run_builder()

    def cancel_builder(self) -> None:
        """Abandon the builder flow and reset its state."""

        if not self._confirm_clear_configuration():
            return
        self._clear_manifest_configuration()
        self.show_home()

    def save_draft(self) -> RunDraft:
        """Record the current builder selections and return the draft."""

        self.draft.completed_step = self.current_step_index
        return self.draft

    def start_run(self) -> None:
        """Move into the monitoring view with a running status."""

        self.current_phase = "Initialising"
        self.output_directory_prepared = True
        if self.pending_manifest is not None:
            self.pending_manifest = run_manifest.annotate_outcome(
                self.pending_manifest,
                state="running",
                outputs="prepared",
                reason="Start confirmed",
            )
            self.summary.manifest_metadata = self._summarise_manifest()
        self._log_run_event("Run execution started; outputs prepared")
        self.status = RunStatus.RUNNING
        self.progress = 0
        self.show_run_monitor()
        self._refresh_status_label()

    def _resolve_model_entry(self) -> dict:
        """Return the currently selected model metadata record."""

        if self._selected_model_entry:
            return self._selected_model_entry
        candidate = ""
        if self.selected_models:
            candidate = self.selected_models[0]
        elif self.draft.model:
            candidate = self.draft.model.split(",")[0].strip()
        if candidate:
            for entry in self.model_index.values():
                if (
                    entry.get("id") == candidate
                    or entry["filename"] == candidate
                ):
                    self._selected_model_entry = entry
                    return entry
        raise RuntimeError("Select a model before starting the run.")

    def _resolve_engine_entry(self) -> dict:
        """Return the currently selected engine metadata record."""

        if self._selected_engine_entry:
            return self._selected_engine_entry
        candidate = self.selected_engine or self.draft.engine
        if candidate:
            for entry in self.engine_index.values():
                if (
                    entry.get("id") == candidate
                    or entry["filename"] == candidate
                ):
                    self._selected_engine_entry = entry
                    return entry
        raise RuntimeError("Select an engine before starting the run.")

    def _load_engine_capabilities(
        self, module_name: str
    ) -> EngineCapabilities | None:
        """Return engine metadata descriptors when available."""

        try:
            module = importlib.import_module(module_name)
            return get_engine_capabilities(module)
        except Exception as exc:
            logger.get_program_logger().warning(
                "Failed to load engine capabilities for %s: %s",
                module_name,
                exc,
            )
        return None

    def _prepare_progress_path(self) -> str:
        """Return the path where the CLI worker writes GUI progress."""

        directory = os.path.join("logs", "progress")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(
            directory,
            f"gui_progress_{utils.get_timestamp()}.json",
        )

    def _build_worker_config(self) -> dict:
        """Return the serialized configuration passed to the CLI worker."""

        if self.manifest_workspace is None:
            raise RuntimeError("No saved manifest is available.")
        progress_path = self._prepare_progress_path()
        self._progress_state_path = progress_path
        progress_state.clear_progress(progress_path)
        return {
            "manifest_path": str(self.manifest_workspace.manifest_path),
            "output_dir": str(self.manifest_workspace.folder),
            "progress_path": progress_path,
        }

    def _write_worker_config(self, config: dict) -> str:
        """Persist worker config to a temporary JSON file."""

        Path(tempfile.gettempdir()).mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            suffix=".json",
            dir=tempfile.gettempdir(),
            encoding="utf-8",
        )
        with handle:
            json.dump(config, handle, indent=2)
            return handle.name

    def _launch_worker_process(self, *, config: dict) -> None:
        """Start the CLI runner in a child process."""

        config_path = self._write_worker_config(config)
        command = [
            sys.executable,
            "-m",
            "copernican_lib.gui.run_worker",
            config_path,
        ]
        env = os.environ.copy()
        env.setdefault("COPERNICAN_DETACH_GUI", "0")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self._repo_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.create_toast(
                f"Failed to start run: {exc}",
                severity="ERROR",
                context="run",
            )
            self.status = RunStatus.ABORTED
            self._refresh_status_label()
            os.unlink(config_path)
            return
        self._progress_snapshot = None
        self._run_process = process
        self._run_config_path = config_path
        self._start_progress_poller()
        self._schedule_monitor_refresh()
        threading.Thread(
            target=self._stream_worker_output, args=(process,), daemon=True
        ).start()
        threading.Thread(
            target=self._wait_for_worker, args=(process,), daemon=True
        ).start()

    def _stream_worker_output(self, process: subprocess.Popen[str]) -> None:
        """Forward worker stdout into the GUI log."""

        if process.stdout is None:
            return
        for line in process.stdout:
            cleaned = line.rstrip()
            if not cleaned:
                continue
            if cleaned.startswith("\r"):
                continue
            if _is_progress_line(cleaned):
                continue
            self._log_run_event(cleaned, logging.INFO)

    def _wait_for_worker(self, process: subprocess.Popen[str]) -> None:
        """Update GUI state when the worker finishes."""

        return_code = process.wait()
        self._stop_progress_poller()
        self._cancel_monitor_refresh()
        if self._progress_state_path:
            progress_state.clear_progress(self._progress_state_path)
            self._progress_state_path = None
        if self._run_config_path and os.path.exists(self._run_config_path):
            os.unlink(self._run_config_path)
        self._run_config_path = None
        self._run_process = None
        if return_code == 0:
            self.update_progress(100)
            if self.status is RunStatus.RUNNING:
                self.status = RunStatus.IDLE
                self.current_phase = "Completed"
                self._refresh_status_label()
                self.create_toast(
                    "Run completed successfully.",
                    severity="INFO",
                    context="run",
                )
                self.show_summary()
        else:
            if self.status is RunStatus.RUNNING:
                self.status = RunStatus.ABORTED
                self.current_phase = "Failed"
                self._refresh_status_label()
                self.create_toast(
                    "Run aborted; review logs for details.",
                    severity="ERROR",
                    context="run",
                )

    def _start_progress_poller(self) -> None:
        """Kick off the progress file watcher when UI is active."""

        if not self.render or not self._progress_state_path:
            return
        if self._progress_poll_thread is not None:
            return
        self._progress_poll_stop = threading.Event()
        self._progress_poll_thread = threading.Thread(
            target=self._progress_poll_loop,
            daemon=True,
        )
        self._progress_poll_thread.start()

    def _stop_progress_poller(self) -> None:
        """Stop the file watcher thread if it is running."""

        if self._progress_poll_stop is not None:
            self._progress_poll_stop.set()
        if self._progress_poll_thread is not None:
            self._progress_poll_thread.join(timeout=0.5)
        self._progress_poll_thread = None
        self._progress_poll_stop = None

    def _progress_poll_loop(self) -> None:
        """Watch the progress JSON file and surface snapshots to the GUI."""

        path = self._progress_state_path
        if not path:
            return
        last_snapshot: dict | None = None
        while (
            self._progress_poll_stop is None
            or not self._progress_poll_stop.is_set()
        ):
            snapshot = progress_state.load_progress(path)
            if snapshot and snapshot != last_snapshot:
                last_snapshot = snapshot
                self._apply_progress_snapshot(snapshot)
            if (
                self._progress_poll_stop is not None
                and self._progress_poll_stop.wait(self._PROGRESS_POLL_INTERVAL)
            ):
                break

    def _apply_progress_snapshot(self, snapshot: dict) -> None:
        """Store the latest progress record and refresh the monitor."""

        self._progress_snapshot = snapshot
        if snapshot.get("stage_label"):
            self.current_phase = snapshot["stage_label"]
        if self.render and self.root is not None:
            self.root.after(0, self._refresh_monitor_widgets)
        else:
            self._refresh_monitor_widgets()

    def _refresh_monitor_widgets(self) -> None:
        """Update the progress bars, status label and log console."""

        snapshot = self._progress_snapshot
        stage_label = "Stage: Idle"
        if snapshot:
            label = snapshot.get("stage_label", "Stage")
            event = snapshot.get("event", "").replace("_", " ")
            stage_label = f"{label} – {event}".strip(" –")
            if snapshot.get("stage_label"):
                self.current_phase = snapshot["stage_label"]
        if self._progress_status_label:
            self._progress_status_label.configure(text=stage_label)
        if self._batch_progressbar:
            percent = snapshot.get("batch_percent", 0) if snapshot else 0
            self._batch_progressbar["value"] = min(max(percent, 0), 100)
        if self._walker_progressbar:
            walker_percent = (
                snapshot.get("walker_percent", 0) if snapshot else 0
            )
            self._walker_progressbar["value"] = min(
                max(walker_percent, 0), 100
            )
        self._refresh_status_label()
        self._refresh_run_log_widget()
        self._update_monitor_controls_state()

    def _refresh_run_log_widget(self) -> None:
        """Populate the run log text widget with the latest entries."""

        if self._monitor_log_widget is None:
            return
        entries = self.get_run_log_entries()
        prev_view = self._monitor_log_widget.yview()
        self._monitor_log_widget.configure(state="normal")
        self._monitor_log_widget.delete("1.0", "end")
        for entry in entries[-200:]:
            self._monitor_log_widget.insert(
                "end",
                f"[{entry.anchor}] {entry.formatted}\n",
            )
        self._monitor_log_widget.configure(state="disabled")
        if prev_view:
            try:
                self._monitor_log_widget.yview_moveto(
                    max(0.0, min(prev_view[0], 1.0))
                )
            except Exception:
                pass

    def _schedule_monitor_refresh(self) -> None:
        """Keep the monitor refreshing at regular intervals."""

        if not self.render or self.root is None:
            return
        if self._monitor_refresh_job is not None:
            return
        self._monitor_refresh_job = self.root.after(
            int(self._PROGRESS_POLL_INTERVAL * 1000),
            self._monitor_refresh_periodic,
        )

    def _monitor_refresh_periodic(self) -> None:
        """Periodic callback that refreshes monitor widgets."""

        self._monitor_refresh_job = None
        self._refresh_monitor_widgets()
        if (
            self.status is RunStatus.RUNNING
            and self.render
            and self.root is not None
        ):
            self._schedule_monitor_refresh()

    def _cancel_monitor_refresh(self) -> None:
        """Stop the pending monitor refresh job."""

        if self._monitor_refresh_job and self.render and self.root is not None:
            self.root.after_cancel(self._monitor_refresh_job)
        self._monitor_refresh_job = None

    def open_current_run_output(self) -> None:
        """Open the latest run output directory if it exists."""

        if not self._current_run_output_dir:
            self.create_toast(
                "Run output directory is not ready yet.",
                severity="WARNING",
                context="monitor",
            )
            return
        try:
            self.open_folder(self._current_run_output_dir)
        except FileNotFoundError:
            self.create_toast(
                "Run output directory is missing on disk.",
                severity="ERROR",
                context="monitor",
            )

    def _view_run_log(self) -> None:
        """Show the filtered run log entries in a read-only dialog."""

        if not self.run_log_path:
            self.create_toast(
                "Run log is not yet available.",
                severity="WARNING",
                context="monitor",
            )
            return
        entries = self.get_run_log_entries()
        content = (
            "\n".join(
                f"[{entry.anchor}] {entry.formatted}" for entry in entries
            )
            or "Run log is empty."
        )
        self._show_metadata_dialog(
            "Run log",
            content,
            self.run_log_path,
        )

    def _open_run_log_file(self) -> None:
        """Open the run log with the desktop default editor."""

        if not self.run_log_path:
            self.create_toast(
                "Run log is not yet available.",
                severity="WARNING",
                context="monitor",
            )
            return
        try:
            self._open_path_with_system(self.run_log_path)
        except Exception as exc:
            self.create_toast(
                f"Unable to open run log: {exc}",
                severity="ERROR",
                context="monitor",
            )

    def _refresh_diagnostics_widget(self) -> None:
        """Refresh the diagnostics text widget with the latest entries."""

        if self._diagnostics_log_widget is None:
            return
        prev_view = self._diagnostics_log_widget.yview()
        entries = self.get_application_log_entries()
        self._diagnostics_log_widget.configure(state="normal")
        self._diagnostics_log_widget.delete("1.0", "end")
        for entry in entries[-200:]:
            self._diagnostics_log_widget.insert(
                "end", f"[{entry.anchor}] {entry.formatted}\n"
            )
        self._diagnostics_log_widget.configure(state="disabled")
        if prev_view:
            try:
                self._diagnostics_log_widget.yview_moveto(
                    max(0.0, min(prev_view[0], 1.0))
                )
            except Exception:
                pass

    def _view_diagnostics_log(self) -> None:
        """Present the diagnostics log in a dedicated viewer."""

        if not self.application_log_path:
            self.create_toast(
                "Diagnostics log is not yet available.",
                severity="WARNING",
                context="settings",
            )
            return
        entries = self.get_application_log_entries()
        content = (
            "\n".join(
                f"[{entry.anchor}] {entry.formatted}" for entry in entries
            )
            or "Diagnostics log is empty."
        )
        self._show_metadata_dialog(
            "Diagnostics log",
            content,
            self.application_log_path,
        )

    def _open_diagnostics_log(self) -> None:
        """Open the diagnostics log file with the system default editor."""

        if not self.application_log_path:
            self.create_toast(
                "Diagnostics log is not yet available.",
                severity="WARNING",
                context="settings",
            )
            return
        try:
            self._open_path_with_system(self.application_log_path)
        except Exception as exc:
            self.create_toast(
                f"Unable to open diagnostics log: {exc}",
                severity="ERROR",
                context="settings",
            )

    def _flush_application_log(self) -> None:
        """Flush the diagnostics log buffer and clear the in-memory entries."""

        program_logger = logger.get_program_logger()
        for handler in list(program_logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
        if self.application_log_handler:
            self.application_log_handler.entries.clear()
        self._refresh_diagnostics_widget()
        self.create_toast(
            "Diagnostics log flushed to disk.",
            severity="INFO",
            context="settings",
        )

    def _terminate_run_process(self, *, force: bool) -> None:
        """Terminate the active worker process if it exists."""

        if self._run_process is None:
            return
        try:
            if force:
                self._run_process.kill()
            else:
                self._run_process.terminate()
        except Exception as exc:
            self._log_run_event(
                f"Failed to terminate worker: {exc}", logging.WARNING
            )

    def _repo_root(self) -> Path:
        """Return repository root path."""

        return Path(__file__).resolve().parents[2]

    def _safe_int(self, value: str, default: int | None) -> int | None:
        """Return ``int(value)`` or ``default`` when parsing fails."""

        try:
            stripped = value.strip()
        except AttributeError:
            return default
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            return default

    def _build_sampling_plan_values(self, engine_module: str) -> dict:
        """Return the sampling plan dict fed to the CLI."""

        module = importlib.import_module(engine_module)
        kind = getattr(module, "ENGINE_KIND", "mcmc").lower()
        if kind != "mcmc":
            raise RuntimeError(
                "GUI-triggered runs currently support MCMC engines only."
            )
        steps = self._safe_int(self.draft.production_steps, 500)
        burn_in = self._safe_int(self.draft.burn_in, max(steps // 5, 100))
        walkers = self._safe_int(self.draft.walkers, 32)
        pool_text = getattr(self.draft, "pool_size", "")
        pool_value = None
        if isinstance(pool_text, str):
            pool_text = pool_text.strip()
            if pool_text:
                pool_value = self._safe_int(pool_text, None)
        return {
            "engine_kind": "mcmc",
            "n_steps": max(steps, 1),
            "burn_in_steps": max(burn_in, 1),
            "n_walkers": max(walkers, 1),
            "pool_size": pool_value,
            "display_progress": True,
        }

    def update_progress(self, value: int) -> None:
        """Update the monitor progress meter."""

        self.progress = max(0, min(100, value))
        self._log_run_event(
            f"Run progress updated to {self.progress}%", logging.INFO
        )
        self._refresh_status_label()
        if self.progress >= 100:
            self.status = RunStatus.IDLE
            self._refresh_status_label()
            self.summary.output_links = ["/output/latest/chains.nc"]
            self.summary.manifest_actions = [
                "Clone manifest",
                "Open output directory",
            ]
            if self.pending_manifest is not None:
                self.pending_manifest = run_manifest.annotate_outcome(
                    self.pending_manifest,
                    state="completed",
                    outputs=self.output_retention_decision or "kept",
                    reason="Sampling finished",
                )
                self.summary.manifest_metadata = self._summarise_manifest()
            completion_anchor = self._log_run_event(
                "Run completed; artefacts ready for review", logging.INFO
            )
            if completion_anchor:
                self.create_toast(
                    "Run completed successfully.",
                    severity="INFO",
                    context="run",
                    anchor=completion_anchor,
                )
            self.show_summary()

    def cancel_run(self, disposition: str | None = None) -> None:
        """Mark the run as cancelled and reset the progress."""

        if self.status not in (RunStatus.RUNNING, RunStatus.CANCELLED):
            return
        self._terminate_run_process(force=False)
        self.status = RunStatus.CANCELLED
        self.current_phase = "Cancelled"
        self.progress = 0
        self._record_output_decision(disposition)
        if self.pending_manifest is not None:
            self.pending_manifest = run_manifest.annotate_outcome(
                self.pending_manifest,
                state="cancelled",
                outputs=self.output_retention_decision,
                reason="User requested cancellation",
            )
            self.summary.manifest_metadata = self._summarise_manifest()
        self._log_run_event("Run cancelled at user request", logging.WARNING)
        self._refresh_status_label()
        self.create_toast(
            "Cancellation requested; the worker will exit shortly.",
            severity="WARNING",
            context="run",
        )

    def pause_run(self) -> None:
        """Pause the run while keeping the monitor visible."""

        self.create_toast(
            (
                "Pause/resume is not available for CLI runs; use Cancel "
                "or Hard Stop."
            ),
            severity="WARNING",
            context="run",
        )

    def stop_run(self, disposition: str | None = None) -> None:
        """Stop the run while keeping the monitor visible."""

        if self.status not in (RunStatus.RUNNING, RunStatus.CANCELLED):
            return
        self._terminate_run_process(force=True)
        self.status = RunStatus.ABORTED
        self.current_phase = "Aborted"
        self.progress = 0
        self._record_output_decision(disposition)
        if self.pending_manifest is not None:
            self.pending_manifest = run_manifest.annotate_outcome(
                self.pending_manifest,
                state="aborted",
                outputs=self.output_retention_decision,
                reason="Hard stop triggered",
            )
            self.summary.manifest_metadata = self._summarise_manifest()
        anchor = self._log_run_event(
            "Run hard stop requested; monitoring halted",
            logging.ERROR,
        )
        if anchor:
            self.create_toast(
                "Run aborted. Review diagnostics for details.",
                severity="ERROR",
                context="monitor",
                anchor=anchor,
            )
        self._refresh_status_label()
        self._refresh_status_label()

    def _noop(self) -> None:
        """Placeholder callback for summary actions."""

        return None

    def _delete_manifest_workspace(self) -> None:
        """Remove the current manifest workspace if it exists."""

        if self.manifest_workspace is None:
            return
        delete_manifest_workspace(self.manifest_workspace)
        self.manifest_workspace = None

    def _load_saved_manifest_workspace(self) -> bool:
        """Load the temporary manifest workspace if the files still exist."""

        if self.manifest_workspace is not None:
            return False
        output_root = Path(self._output_root())
        workspace_folder = output_root / self._TEMP_MANIFEST_FOLDER
        manifest_path = workspace_folder / self._TEMP_MANIFEST_FILE
        if not manifest_path.exists():
            return False
        manifest = run_manifest.load_manifest(str(manifest_path))
        self.manifest_workspace = ManifestWorkspace(
            folder=workspace_folder,
            manifest_path=manifest_path,
            creation_timestamp=manifest_path.stem,
        )
        self.pending_manifest = manifest
        self.summary.manifest_metadata = self._summarise_manifest()
        self.summary.manifest_actions.append(
            f"Loaded saved manifest from {manifest_path}"
        )
        return True

    def _reset_manifest_state(self) -> None:
        """Clear manifest state after a cancellation or restart."""

        self._delete_manifest_workspace()
        self.pending_manifest = None
        self.summary.manifest_metadata = []
        self.summary.manifest_actions = []

    def _ensure_manifest_snapshot(self) -> dict | None:
        """Generate and cache the latest manifest snapshot."""

        try:
            manifest = self._generate_manifest_snapshot()
        except Exception as exc:
            self.create_toast(
                f"Manifest generation failed: {exc}",
                severity="ERROR",
                context="run",
            )
            return None
        self.pending_manifest = manifest
        return manifest

    def _persist_manifest_workspace(
        self,
        *,
        notify: bool = False,
    ) -> ManifestWorkspace | None:
        """Persist the current manifest snapshot into the output folder."""

        manifest = self._ensure_manifest_snapshot()
        if manifest is None:
            return None
        workspace = self.manifest_workspace
        if workspace is not None:
            if not self._confirm_overwrite_manifest():
                return None
            try:
                run_manifest.save_manifest(
                    manifest,
                    workspace.folder,
                    target_path=workspace.manifest_path,
                )
            except Exception as exc:
                self.create_toast(
                    f"Failed to overwrite manifest: {exc}",
                    severity="ERROR",
                    context="run",
                )
                return None
        else:
            output_root = Path(self._output_root())
            try:
                workspace = create_manifest_workspace(
                    output_root,
                    manifest,
                    folder_name=self._TEMP_MANIFEST_FOLDER,
                    manifest_filename=self._TEMP_MANIFEST_FILE,
                )
            except Exception as exc:
                self.create_toast(
                    f"Failed to save manifest: {exc}",
                    severity="ERROR",
                    context="run",
                )
                return None
            self.manifest_workspace = workspace
        self.summary.manifest_metadata = self._summarise_manifest()
        note = f"Manifest saved to {workspace.manifest_path}"
        self.summary.manifest_actions.append(note)
        if notify:
            self.create_toast(
                "Manifest stored in the output folder.",
                severity="INFO",
                context="run",
            )
        return workspace

    def confirm_start_run(self) -> None:
        """Generate a manifest snapshot and defer output creation."""

        if self.manifest_workspace is None or self.pending_manifest is None:
            self.create_toast(
                "Cannot start run without saving the manifest first.",
                severity="ERROR",
                context="run",
            )
            return
        run_start_ts = utils.get_timestamp()
        workspace = finalize_run_workspace(
            self.manifest_workspace,
            start_timestamp=run_start_ts,
        )
        self.manifest_workspace = workspace
        self.summary.manifest_metadata = self._summarise_manifest()
        self.summary.manifest_actions.append(
            f"Manifest finalised for run start: {workspace.manifest_path}"
        )
        try:
            worker_config = self._build_worker_config()
        except Exception as exc:
            self._log_run_event(str(exc), logging.ERROR)
            self.create_toast(
                f"Run aborted: {exc}",
                severity="ERROR",
                context="run",
            )
            return
        self.status = RunStatus.CONFIGURING
        self.current_phase = "Configuring"
        self._start_run_logging(self.pending_manifest)
        self.start_run()
        self._launch_worker_process(config=worker_config)

    def insert_manifest_from_builder(self) -> None:
        """Load the staged builder manifest without launching a run."""

        if self._staged_confirm_manifest is None:
            self._stage_confirm_manifest()
        if self._staged_confirm_manifest is None:
            return
        self.pending_manifest = copy.deepcopy(self._staged_confirm_manifest)
        self.summary.manifest_actions.append(
            "Inserted manifest from Run Builder"
        )
        self.summary.manifest_metadata = self._summarise_manifest()
        self.create_toast(
            "Manifest inserted. Review metadata before starting the run.",
            severity="INFO",
            context="run",
        )

    def import_manifest(self, path: str) -> dict:
        """Load a manifest and seed the builder selections from it."""

        manifest = run_manifest.load_manifest(path)
        self._delete_manifest_workspace()
        self.pending_manifest = manifest
        self.summary.manifest_metadata = self._summarise_manifest()
        self.summary.manifest_actions.append(f"Imported manifest from {path}")
        configuration = manifest.get("configuration", {})
        models = configuration.get("models", [])
        if isinstance(models, str):
            models = [models]
        self.selected_models = list(models)
        engine_meta = configuration.get("engine", {})
        self.selected_engine = engine_meta.get("name", "")
        datasets = configuration.get("datasets", [])
        if isinstance(datasets, str):
            datasets = [datasets]
        self.selected_datasets = []
        for dataset_id in datasets:
            entry = self.catalogue_index.get(dataset_id)
            if entry:
                self.selected_datasets.append(
                    {
                        "id": dataset_id,
                        "path": entry.get("path", ""),
                        "name": entry.get("name", dataset_id),
                        "version": entry.get("version", "unknown"),
                        "hashes": entry.get("hashes", {}),
                        "independence": entry.get("independence", []),
                        "type": entry.get("type", ""),
                        "license": entry.get("license", "unspecified"),
                        "badges": entry.get("badges", []),
                    }
                )
            else:
                self.selected_datasets.append(
                    {
                        "id": dataset_id,
                        "path": "",
                        "name": dataset_id,
                        "version": "unknown",
                        "hashes": {},
                        "independence": [],
                        "type": "",
                        "license": "unspecified",
                        "badges": [],
                    }
                )
        seed = manifest.get("seed")
        if seed is not None:
            self.draft.seed = str(seed)
        if models:
            self.draft.model = ", ".join(models)
        if datasets:
            self.draft.data = ", ".join(datasets)
        if self.selected_engine:
            self.draft.engine = self.selected_engine
        confirmation = manifest.get("confirmation", {})
        self.draft.plan = confirmation.get("plan", "Duplicate manifest")
        run_settings = configuration.get("run_settings", {})
        if isinstance(run_settings, dict):
            self._apply_run_settings_to_draft(run_settings)
        self.show_run_builder()
        return manifest

    def duplicate_manifest_for_editing(self, path: str) -> dict:
        """Load a manifest and mark Run Builder fields for edits."""

        manifest = self.import_manifest(path)
        confirmation = manifest.get("confirmation", {})
        if confirmation.get("notes"):
            self.draft.notes = confirmation["notes"]
        if confirmation.get("plan"):
            self.draft.plan = f"Duplicate & Edit: {confirmation['plan']}"
        else:
            self.draft.plan = "Duplicate & Edit"
        self.summary.manifest_actions.append(
            f"Duplicated manifest from {path} for editing"
        )
        return manifest

    def export_manifest(self, output_dir: str) -> str:
        """Persist the active manifest after ensuring it exists."""

        if self.pending_manifest is None:
            self.pending_manifest = self._generate_manifest_snapshot()
        path = run_manifest.save_manifest(self.pending_manifest, output_dir)
        self.summary.manifest_actions.append(f"Saved manifest to {path}")
        self.summary.manifest_metadata = self._summarise_manifest()
        return path

    def register_dataset(
        self, *, dataset_id: str, path: str, name: str
    ) -> None:
        """Record dataset metadata and compute its hashes for the manifest."""

        hashes: dict[str, str] = {}
        entry = self.catalogue_index.get(dataset_id)
        if entry:
            hashes.update(entry.get("hashes", {}))
        if os.path.exists(path) and os.path.isfile(path):
            hashes[os.path.basename(path)] = utils.compute_sha256(path)
        self.selected_datasets.append(
            {
                "id": dataset_id,
                "path": path,
                "name": name,
                "hashes": hashes,
                "version": entry.get("version", "unknown") if entry else "",
                "independence": entry.get("independence", []) if entry else [],
                "type": entry.get("type", "") if entry else "",
                "license": (
                    entry.get("license", "unspecified")
                    if entry
                    else "unspecified"
                ),
                "badges": entry.get("badges", []) if entry else [],
            }
        )

    def _collect_run_settings_snapshot(self) -> dict[str, object] | None:
        """Return canonical run settings from the builder inputs."""

        try:
            engine_entry = self._resolve_engine_entry()
            plan = self._build_sampling_plan_values(engine_entry["id"])
            snapshot = dict(plan)
            knob_settings = self._collect_engine_knob_settings()
            if knob_settings:
                snapshot.update(knob_settings)
            return snapshot
        except Exception:
            pass
        fallback_fields = {
            "n_walkers": self.draft.walkers,
            "burn_in_steps": self.draft.burn_in,
            "n_steps": self.draft.production_steps,
            "pool_size": self.draft.pool_size,
        }
        sanitized: dict[str, object] = {}
        for key, value in fallback_fields.items():
            if not isinstance(value, str):
                continue
            trimmed = value.strip()
            if not trimmed:
                continue
            parsed = self._safe_int(trimmed, None)
            sanitized[key] = parsed if parsed is not None else trimmed
        if sanitized:
            sanitized.setdefault("engine_kind", "mcmc")
        return sanitized or None

    def _apply_run_settings_to_draft(
        self, settings: Mapping[str, object]
    ) -> None:
        """Populate draft controls from manifest run settings."""

        def _set(field: str, key: str) -> None:
            value = settings.get(key)
            if value is None:
                setattr(self.draft, field, "")
            else:
                setattr(self.draft, field, str(value))

        _set("walkers", "n_walkers")
        _set("burn_in", "burn_in_steps")
        _set("production_steps", "n_steps")
        pool_value = settings.get("pool_size")
        if pool_value is None:
            pool_value = settings.get("pool_workers")
        if pool_value in ("", None):
            self.draft.pool_size = ""
        else:
            self.draft.pool_size = str(pool_value)

    def _generate_manifest_snapshot(self) -> dict:
        """Build a manifest using the current builder selections."""

        seed_value = int(self.draft.seed) if self.draft.seed.isdigit() else 0
        utils.set_random_seed(seed_value)
        engine_name = self.draft.engine or self.selected_engine or "engine"
        engine = SimpleNamespace(__name__=engine_name, ENGINE_VERSION="gui")
        models = self.selected_models or [self.draft.model or "model"]
        model_pairs = [
            (
                SimpleNamespace(
                    MODEL_NAME=model,
                    MODEL_FILENAME=f"{model}.yml",
                    PARAMETER_NAMES=[],
                    PARAMETER_PRIORS=[],
                    valid_for_cmb=False,
                ),
                "gui",
            )
            for model in models
        ]
        datasets: list[dict[str, object]] = []
        source_datasets = self.selected_datasets or [
            {
                "id": self.draft.data or "dataset",
                "name": self.draft.data or "dataset",
                "version": "unversioned",
                "path": "",
                "hashes": {},
                "independence": "GUI configured selection",
            }
        ]
        for dataset in source_datasets:
            independence = dataset.get("independence", [])
            if isinstance(independence, str):
                independence = [independence]
            datasets.append(
                {
                    "id": dataset.get("id", "dataset"),
                    "name": dataset.get("name", "dataset"),
                    "version": dataset.get("version", "unversioned"),
                    "path": dataset.get("path", ""),
                    "hashes": dataset.get("hashes", {}),
                    "independence": independence,
                    "type": dataset.get("type", "unknown") or "unknown",
                }
            )
        configuration = {
            "models": models,
            "engine": {"name": engine_name, "version": "gui"},
            "datasets": [dataset.get("id", "") for dataset in datasets],
            "notes": "Snapshot captured at run start confirmation.",
        }
        run_settings = self._collect_run_settings_snapshot()
        if run_settings:
            configuration["run_settings"] = run_settings
        manifest = run_manifest.build_manifest(
            models=model_pairs,
            engine_module=engine,
            datasets=datasets,
            state="pending",
            output_policy="unprepared",
            configuration=configuration,
        )
        manifest["confirmation"] = {
            "seed": seed_value,
            "notes": self.draft.notes,
            "plan": self.draft.plan,
        }
        return manifest

    def _summarise_manifest(self) -> list[str]:
        """Return a human-friendly digest of the active manifest."""

        if self.pending_manifest is None:
            return []
        summary: list[str] = []
        status = self.pending_manifest.get("status", {})
        summary.append(
            f"State: {status.get('state', 'unknown')} (outputs: "
            f"{status.get('outputs', 'n/a')})"
        )
        selection = self.pending_manifest.get("selection", {})
        models = selection.get("models", []) or []
        summary.append(f"Models: {', '.join(models)}")
        engine_meta = selection.get("engine", {})
        engine_desc = engine_meta.get("name", "engine")
        if engine_meta.get("version"):
            engine_desc += f" v{engine_meta['version']}"
        summary.append(f"Engine: {engine_desc}")
        dataset_lines = []
        for dataset_id, dataset in self.pending_manifest.get(
            "datasets", {}
        ).items():
            hashes = dataset.get("hashes", {})
            dataset_lines.append(
                f"{dataset_id}: "
                f"{len(hashes)} "
                f"hash{'es' if len(hashes) != 1 else ''} recorded"
            )
        summary.extend(dataset_lines)
        configuration = self.pending_manifest.get("configuration", {})
        settings = configuration.get("run_settings", {})
        if settings:
            formatted = ", ".join(
                f"{key.replace('_', ' ')}={value}"
                for key, value in settings.items()
            )
            summary.append(f"Run settings: {formatted}")
        if self.manifest_workspace is not None:
            summary.append(
                f"Manifest folder: {self.manifest_workspace.folder.name}"
            )
            summary.append(
                f"Manifest file: {self.manifest_workspace.manifest_path.name}"
            )
        return summary

    def _record_output_decision(self, disposition: str | None) -> None:
        """Record how outputs should be handled after cancellation."""

        self.output_retention_decision = (
            disposition or self.output_retention_decision
        )
        if not self.output_retention_decision:
            self.output_retention_decision = "kept"

    def run(self) -> None:
        """Start the Tk main loop when rendering is enabled."""

        if self.render and self.root is not None:
            self.root.mainloop()


__all__ = ["CopernicanGUI", "RunStatus", "RunDraft", "RunSummary"]
