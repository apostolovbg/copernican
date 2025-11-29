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

import importlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed only when Tk is missing
    tk = None
    ttk = None

import yaml

from copernican_lib import (
    console_output,
    dataset_registry,
    logger,
    run_manifest,
    utils,
)


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


@dataclass
class NavigationItem:
    """Describe a navigation rail entry and its callback."""

    name: str
    label: str
    shortcut: str
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
        "Plan",
        "Confirm",
    ]
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
        self.quick_actions: list[str] = [
            "New Run",
            "Open Run Monitor",
            "Open Output Folder",
        ]
        self.current_step_index = 0
        self.draft = RunDraft()
        self.summary = RunSummary()
        self.progress = 0
        self.nav_items = []
        self.selected_models: list[str] = []
        self.selected_engine: str = ""
        self.selected_datasets: list[dict[str, str]] = []
        self.pending_manifest: Optional[dict] = None
        self.catalogue_index: dict[str, dict] = {}
        self.model_index: dict[str, dict] = {}
        self.engine_index: dict[str, dict] = {}
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
        self._bootstrap_logging()
        self._build_navigation()
        self._initialise_rendering()
        self.refresh_inventory()

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

    def set_monitor_filter(self, severity: str) -> None:
        """Update the Run Monitor severity filter."""

        self.monitor_filter_level = severity.upper()

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
            self.root.title("Copernican Suite")
            self.root.geometry("1200x800")
            self.root.configure(padx=12, pady=12)
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
                    utils.compute_sha256(parser_path) if parser_path else ""
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
        return path

    def view_metadata_file(self, asset_id: str) -> str:
        """Return cached metadata content for the requested asset."""

        entry = self.catalogue_index.get(asset_id)
        if entry and entry.get("metadata_path"):
            cache_key = entry["metadata_path"]
        elif asset_id in self.model_index:
            cache_key = self.model_index[asset_id]["path"]
        else:
            cache_key = ""
            for record in self.model_index.values():
                if record.get("id") == asset_id:
                    cache_key = record.get("path", "")
                    break
        if not cache_key:
            if asset_id in self.engine_index:
                cache_key = self.engine_index[asset_id]["path"]
            else:
                for record in self.engine_index.values():
                    if record.get("id") == asset_id:
                        cache_key = record.get("path", "")
                        break
        if not cache_key:
            raise KeyError(asset_id)
        if cache_key in self.metadata_cache:
            return self.metadata_cache[cache_key]
        with open(cache_key, "r", encoding="utf-8") as handle:
            contents = handle.read()
        self.metadata_cache[cache_key] = contents
        return contents

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
            NavigationItem("home", "Home", "Ctrl+1", CopernicanGUI.show_home),
            NavigationItem(
                "run_builder",
                "Run Builder",
                "Ctrl+2",
                CopernicanGUI.show_run_builder,
            ),
            NavigationItem(
                "data", "Data", "Ctrl+3", CopernicanGUI.show_data_overview
            ),
            NavigationItem(
                "models", "Models", "Ctrl+4", CopernicanGUI.show_models
            ),
            NavigationItem(
                "engines", "Engines", "Ctrl+5", CopernicanGUI.show_engines
            ),
            NavigationItem(
                "settings", "Settings", "Ctrl+6", CopernicanGUI.show_settings
            ),
            NavigationItem("help", "Help", "Ctrl+7", CopernicanGUI.show_help),
        ]

    def _build_layout(self) -> None:
        """Construct the navigation rail and main content area."""

        if not self.render or self.root is None:
            return
        nav_frame = ttk.Frame(self.root, padding=(8, 8))
        nav_frame.grid(row=0, column=0, sticky="nsw")
        self.content_area = ttk.Frame(self.root, padding=(12, 12))
        self.content_area.grid(row=0, column=1, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        for item in self.nav_items:
            button = ttk.Button(
                nav_frame,
                text=f"{item.label} ({item.shortcut})",
                command=lambda i=item: i.action(self),
                takefocus=True,
            )
            button.pack(fill="x", pady=4)
            shortcut = item.shortcut.replace("Ctrl+", "<Control-") + ">"
            self.root.bind(shortcut.lower(), lambda _e, i=item: i.action(self))

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
            for action in self.quick_actions:
                ttk.Button(
                    frame,
                    text=action,
                    command=self.show_run_builder,
                    takefocus=True,
                ).pack(anchor="w", pady=2)

        self._swap_content(builder)

    def show_run_builder(self) -> None:
        """Render the Run Builder wizard with jump controls."""

        def builder(frame: tk.Frame) -> None:
            header = ttk.Label(
                frame, text="Run Builder", font=("Helvetica", 16)
            )
            header.pack(anchor="w", pady=(0, 8))
            step_frame = ttk.Frame(frame)
            step_frame.pack(fill="x", pady=(0, 12))
            for index, step in enumerate(self.builder_steps):
                indicator = ttk.Button(
                    step_frame,
                    text=f"{index + 1}. {step}",
                    command=lambda idx=index: self.jump_to_step(idx),
                    takefocus=True,
                )
                indicator.pack(side="left", padx=2)
            body = ttk.Label(
                frame,
                text=(
                    "Use Next/Previous to move between steps, jump directly "
                    "with the step buttons, or save a draft for later."
                ),
                wraplength=720,
                takefocus=True,
            )
            body.pack(anchor="w", pady=(8, 8))
            controls = ttk.Frame(frame)
            controls.pack(anchor="w")
            ttk.Button(
                controls,
                text="Previous",
                command=self.previous_step,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Next",
                command=self.next_step,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Save Draft",
                command=self.save_draft,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Cancel",
                command=self.cancel_builder,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Start Run",
                command=self.confirm_start_run,
                takefocus=True,
            ).pack(side="left", padx=4)

        self._swap_content(builder)

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
            for dataset in active:
                title = (
                    f"{dataset['name']} ({dataset['id']}) "
                    f"[{', '.join(dataset['badges'])}]"
                )
                ttk.Label(frame, text=title, takefocus=True).pack(anchor="w")
                ttk.Label(
                    frame,
                    text=(
                        f"Citation: {dataset.get('citation', 'missing')}\n"
                        f"License: {dataset.get('license', 'unspecified')}\n"
                        f"Parser SHA256: {dataset.get('parser_digest', 'n/a')}"
                    ),
                    wraplength=720,
                    justify="left",
                    takefocus=True,
                ).pack(anchor="w", pady=(0, 4))
                digest_line = (
                    f"Metadata SHA256: {dataset.get('metadata_digest', '')}"
                )
                ttk.Label(frame, text=digest_line, takefocus=True).pack(
                    anchor="w"
                )
                ttk.Label(
                    frame,
                    text="Compatibility badges: "
                    + ", ".join(dataset.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                ttk.Label(
                    frame,
                    text="Independence notes: "
                    + "; ".join(dataset.get("independence", [])),
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w", pady=(0, 4))
                actions = ttk.Frame(frame)
                actions.pack(anchor="w", pady=(0, 12))
                ttk.Button(
                    actions,
                    text="Open folder",
                    command=lambda p=dataset["path"]: self.open_folder(p),
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="View metadata",
                    command=lambda d=dataset["id"]: self.view_metadata_file(d),
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="Revalidate parser",
                    command=lambda d=dataset["id"]: self.revalidate_dataset(d),
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
            for model in self.model_index.values():
                heading = (
                    f"{model['id']} ({model['filename']}) "
                    f"v{model.get('version', 'unknown')}"
                )
                ttk.Label(frame, text=heading, takefocus=True).pack(anchor="w")
                ttk.Label(
                    frame,
                    text=(
                        f"Badges: {', '.join(model.get('badges', []))}\n"
                        f"SHA256: {model.get('hash', '')}\n"
                        f"License: {model.get('license', 'unspecified')}"
                    ),
                    wraplength=720,
                    takefocus=True,
                    justify="left",
                ).pack(anchor="w")
                actions = ttk.Frame(frame)
                actions.pack(anchor="w", pady=(0, 12))
                ttk.Button(
                    actions,
                    text="Open model folder",
                    command=lambda p=model["path"]: self.open_folder(
                        os.path.dirname(p)
                    ),
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="View YAML",
                    command=lambda m=model["id"]: self.view_metadata_file(m),
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
            for engine in self.engine_index.values():
                heading = f"{engine['label']} ({engine['filename']})"
                ttk.Label(frame, text=heading, takefocus=True).pack(anchor="w")
                ttk.Label(
                    frame,
                    text=(
                        f"Badges: {', '.join(engine.get('badges', []))}\n"
                        f"Version: {engine.get('version', 'unknown')}\n"
                        f"SHA256: {engine.get('hash', '')}"
                    ),
                    wraplength=720,
                    takefocus=True,
                    justify="left",
                ).pack(anchor="w")
                actions = ttk.Frame(frame)
                actions.pack(anchor="w", pady=(0, 12))
                ttk.Button(
                    actions,
                    text="Open engine folder",
                    command=lambda p=engine["path"]: self.open_folder(
                        os.path.dirname(p)
                    ),
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk.Button(
                    actions,
                    text="View module",
                    command=lambda e=engine["id"]: self.view_metadata_file(e),
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
                text=(
                    f"App log path: {self.application_log_path} "
                    f"(filter {self.diagnostics_filter_level}+)"
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w")
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
            for entry in self.get_application_log_entries()[-5:]:
                ttk.Label(
                    diag_frame,
                    text=f"[{entry.anchor}] {entry.formatted}",
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w")
            actions = ttk.Frame(diag_frame)
            actions.pack(anchor="w", pady=(6, 0))
            ttk.Button(
                actions,
                text="Copy filtered log",
                command=self.copy_application_logs,
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                actions,
                text="Download diagnostics",
                command=lambda: self.export_application_logs("logs/exports"),
                takefocus=True,
            ).pack(side="left", padx=2)

        self._swap_content(builder)

    def show_help(self) -> None:
        """Display contextual help panel."""

        def builder(frame: tk.Frame) -> None:
            ttk.Label(frame, text="Help", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "Use the navigation rail or keyboard shortcuts to move "
                    "between panels."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))

        self._swap_content(builder)

    def show_run_monitor(self) -> None:
        """Display live run status controls."""

        def builder(frame: tk.Frame) -> None:
            header = ttk.Label(
                frame, text="Run Monitor", font=("Helvetica", 16)
            )
            header.pack(anchor="w", pady=(0, 8))
            ttk.Label(
                frame, text=f"Status: {self.status.value}", takefocus=True
            ).pack(anchor="w")
            progress = ttk.Progressbar(
                frame, maximum=100, value=self.progress, length=320
            )
            progress.pack(anchor="w", pady=(8, 8))
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
            log_frame.pack(fill="x", pady=(8, 4))
            ttk.Label(
                log_frame,
                text=f"Filter: {self.monitor_filter_level}+",
                takefocus=True,
            ).pack(anchor="w")
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
            for entry in self.get_run_log_entries()[-5:]:
                ttk.Label(
                    log_frame,
                    text=f"[{entry.anchor}] {entry.formatted}",
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w")
            log_actions = ttk.Frame(log_frame)
            log_actions.pack(anchor="w", pady=(4, 0))
            ttk.Button(
                log_actions,
                text="Copy filtered log",
                command=self.copy_run_logs,
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk.Button(
                log_actions,
                text="Export run log",
                command=lambda: self.export_run_logs("logs/exports"),
                takefocus=True,
            ).pack(side="left", padx=2)
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
            ttk.Button(
                controls,
                text="Cancel",
                command=self.cancel_run,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Pause",
                command=self.pause_run,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls,
                text="Hard Stop",
                command=self.stop_run,
                takefocus=True,
            ).pack(side="left", padx=4)

        self._swap_content(builder)

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
        if self.current_step_index == len(self.builder_steps) - 1:
            self.summary.output_links = ["/output/latest/chains.nc"]
            self.summary.manifest_actions = [
                "Reuse manifest for new run",
                "Export manifest",
            ]
            self.show_summary()

    def previous_step(self) -> None:
        """Move back one builder step when possible."""

        if self.current_step_index > 0:
            self.current_step_index -= 1

    def jump_to_step(self, step_index: int) -> None:
        """Jump directly to any builder step."""

        if 0 <= step_index < len(self.builder_steps):
            self.current_step_index = step_index

    def cancel_builder(self) -> None:
        """Abandon the builder flow and reset its state."""

        self.current_step_index = 0
        self.draft = RunDraft()
        self.show_home()

    def save_draft(self) -> RunDraft:
        """Record the current builder selections and return the draft."""

        self.draft.completed_step = self.current_step_index
        return self.draft

    def start_run(self) -> None:
        """Move into the monitoring view with a running status."""

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

    def update_progress(self, value: int) -> None:
        """Update the monitor progress meter."""

        self.progress = max(0, min(100, value))
        self._log_run_event(
            f"Run progress updated to {self.progress}%", logging.INFO
        )
        if self.progress >= 100:
            self.status = RunStatus.IDLE
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

        self.status = RunStatus.CANCELLED
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

    def pause_run(self) -> None:
        """Pause the run while keeping the monitor visible."""

        self.status = RunStatus.PAUSED
        if self.pending_manifest is not None:
            self.pending_manifest = run_manifest.annotate_outcome(
                self.pending_manifest,
                state="paused",
                outputs=self.output_retention_decision,
                reason="Pause requested",
            )
            self.summary.manifest_metadata = self._summarise_manifest()
        self._log_run_event("Run paused by user", logging.WARNING)

    def stop_run(self, disposition: str | None = None) -> None:
        """Stop the run while keeping the monitor visible."""

        self.status = RunStatus.ABORTED
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

    def _noop(self) -> None:
        """Placeholder callback for summary actions."""

        return None

    def confirm_start_run(self) -> None:
        """Generate a manifest snapshot and defer output creation."""

        self.status = RunStatus.CONFIGURING
        self.pending_manifest = self._generate_manifest_snapshot()
        self.summary.manifest_metadata = self._summarise_manifest()
        if self.pending_manifest is not None:
            self._start_run_logging(self.pending_manifest)
        self.show_run_monitor()

    def import_manifest(self, path: str) -> dict:
        """Load a manifest and seed the builder selections from it."""

        manifest = run_manifest.load_manifest(path)
        self.pending_manifest = manifest
        self.summary.manifest_metadata = self._summarise_manifest()
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
                    }
                )
            else:
                self.selected_datasets.append(
                    {"id": dataset_id, "path": "", "name": dataset_id}
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
            }
        )

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
                }
            )
        configuration = {
            "models": models,
            "engine": {"name": engine_name, "version": "gui"},
            "datasets": [dataset.get("id", "") for dataset in datasets],
            "notes": "Snapshot captured at run start confirmation.",
        }
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
                f"{dataset_id} hashes: "
                f"{', '.join(sorted(hashes)) or 'none recorded'}"
            )
        summary.extend(dataset_lines)
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
