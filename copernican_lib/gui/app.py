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

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Callable, Dict, Optional

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed only when Tk is missing
    tk = None
    ttk = None

from copernican_lib import console_output, logger, run_manifest, utils


LINE_BREAK = "\n"


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
        """Initialise with a prefix so anchors stay deterministic."""

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
        """Prepare GUI state so tests can exercise navigation without Tk."""

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
        self.diagnostics_clipboard = LINE_BREAK.join(
            entry.formatted for entry in entries
        )
        return self.diagnostics_clipboard

    def copy_run_logs(self) -> str:
        """Copy filtered run logs into a clipboard buffer."""

        entries = self.get_run_log_entries()
        self.run_clipboard = LINE_BREAK.join(
            entry.formatted for entry in entries
        )
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
            """Render placeholder widgets so navigation stays predictable."""

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
            """Lay out the builder controls so keyboard jumps can be tested."""

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
        """Display placeholder data catalogue information."""

        def builder(frame: tk.Frame) -> None:
            """Provide a skeletal data view so navigation stays stable."""

            ttk.Label(frame, text="Data catalogue", takefocus=True).pack(
                anchor="w"
            )
            ttk.Label(
                frame,
                text=(
                    "Datasets remain selectable from the Run Builder; this "
                    "panel summarises installed catalogues."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))

        self._swap_content(builder)

    def show_models(self) -> None:
        """Display model listing placeholder content."""

        def builder(frame: tk.Frame) -> None:
            """Expose a simple model list placeholder for accessibility."""

            ttk.Label(frame, text="Models", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text="Model metadata will appear here for quick inspection.",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))

        self._swap_content(builder)

    def show_engines(self) -> None:
        """Display engine overview panel."""

        def builder(frame: tk.Frame) -> None:
            """Show engine placeholders so the navigation rail stays filled."""

            ttk.Label(frame, text="Engines", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "Engine compatibility and resource hints will be shown "
                    "here."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))

        self._swap_content(builder)

    def show_settings(self) -> None:
        """Display settings placeholder panel."""

        def builder(frame: tk.Frame) -> None:
            """Render diagnostics controls because log export needs hooks."""

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
            """List help affordances so keyboard navigation stays clear."""

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
            """Expose diagnostic streams because GUI mode surfaces logs."""

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
                    """Jump to log entries so alerts stay traceable."""

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
            """Summarise outputs so saved manifests remain discoverable."""

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
        self.selected_datasets = [
            {"id": dataset_id, "path": "", "name": dataset_id}
            for dataset_id in datasets
        ]
        seed = manifest.get("seed")
        if seed is not None:
            self.draft.seed = str(seed)
        self.show_run_builder()
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
        if os.path.exists(path) and os.path.isfile(path):
            hashes[os.path.basename(path)] = utils.compute_sha256(path)
        self.selected_datasets.append(
            {"id": dataset_id, "path": path, "name": name, "hashes": hashes}
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
        datasets = self.selected_datasets or [
            {
                "id": self.draft.data or "dataset",
                "name": self.draft.data or "dataset",
                "version": "unversioned",
                "path": "",
                "hashes": {},
                "independence": "GUI configured selection",
            }
        ]
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
