"""Tkinter-based GUI scaffold with headless fallbacks.

The GUI keeps a persistent navigation rail and modular content panes so
Copernican can expose future workflows without reworking the layout.
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
import platform
import re
import shutil
import subprocess  # nosec
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import matplotlib.image as mpimage
import yaml
from matplotlib.figure import Figure

import copernican.rng_minigames as rng_minigames
from copernican import validation as validation_utils
from copernican import version
from copernican.lib import (
    analysis,
    console_output,
    dataset_registry,
    logger,
    model_spec_validator,
    posterior_explorer,
    progress_state,
    run_manifest,
)
from copernican.lib import settings as settings_mod
from copernican.lib import utils
from copernican.lib.engine_capabilities import (
    EngineCapabilities,
    EngineSetting,
    get_engine_capabilities,
)
from copernican.lib.gui.plot_viewer import PlotViewer
from copernican.lib.run_lifecycle import (
    ManifestWorkspace,
    create_manifest_workspace,
    delete_manifest_workspace,
    finalize_run_workspace,
)

try:
    import tkinter as tkinter_module
    from tkinter import filedialog as filedialog_module
    from tkinter import messagebox as messagebox_module
    from tkinter import ttk as ttk_module
except ImportError:  # pragma: no cover - executed only when Tk is missing
    tkinter_module = None
    ttk_module = None
    filedialog_module = None
    messagebox_module = None

tk_gui = tkinter_module
ttk = ttk_module
filedialog = filedialog_module
messagebox = messagebox_module
tk_tcl_error = getattr(tkinter_module, "TclError", RuntimeError)

if tk_gui is not None:
    from copernican.lib.vendor.tkinterweb import HtmlFrame
else:
    HtmlFrame = None

_KATEX_VERSION = "0.16.4"
_EQUATION_EMPTY_BODY = (
    "<p class='hint'>Select a model to preview its symbolic equations.</p>"
)
_EQUATION_WINDOW_SHIM = r"""
  <script>
    (function() {
      const globalObj =
        typeof globalThis !== "undefined" ? globalThis : this;
      if (typeof window === "undefined") {
        window = globalObj;
      }
      if (typeof document === "undefined") {
        document = globalObj.document || {
          createElement() {
            const node = {
              style: {},
              appendChild() {},
              removeChild() {},
              setAttribute() {},
              innerHTML: "",
              className: "",
            };
            node.getBoundingClientRect = () => ({ width: 0, height: 0 });
            return node;
          },
          body: {
            appendChild() {},
            removeChild() {},
          },
        };
      }
    })();
  </script>"""

_EQUATION_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/katex@{version}/dist/katex.min.css"
  >
{shim}
  <style>
    body {{
      margin: 0;
      padding: 12px;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background-color: transparent;
      color: #111;
    }}
    .model-name {{
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 0.25rem;
      word-break: break-word;
    }}
    .expressions {{
      margin-top: 0.5rem;
    }}
    .expression-block {{
      margin-bottom: 0.95rem;
    }}
    .expression-title {{
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 0.18rem;
      word-break: break-word;
    }}
    .equation {{
      min-height: 1.4em;
      width: 100%;
      word-break: break-word;
      white-space: normal;
    }}
    .hint {{
      color: #666;
      font-style: italic;
      margin-top: 0.5rem;
    }}
  </style>
</head>
<body>
  <div class="model-name">{model_name}</div>
  <div class="expressions">
    {expressions}
  </div>
  <script
    defer
    src="https://cdn.jsdelivr.net/npm/katex@{version}/dist/katex.min.js"
  ></script>
  <script defer>
    (() => {{
      const containers = document.querySelectorAll(".equation");
      const fit = node => {{
        let size = parseFloat(getComputedStyle(node).fontSize) || 18;
        while (node.scrollWidth > node.clientWidth && size > 10) {{
          size -= 1;
          node.style.fontSize = size + "px";
        }}
      }};
      const render = () => {{
        containers.forEach(node => {{
          const latex = node.getAttribute("data-latex") || "";
          if (window.katex) {{
            try {{
              katex.render(
                latex,
                node,
                {{ throwOnError: false, displayMode: true }},
              );
            }} catch (_error) {{
              node.textContent = latex;
            }}
          }} else {{
            node.textContent = latex;
          }}
          fit(node);
        }});
      }};
      window.addEventListener("load", render);
      window.addEventListener("resize", () => {{
        containers.forEach(node => {{
          node.style.fontSize = "";
          fit(node);
        }});
      }});
    }})();
  </script>
</body>
</html>"""
log_mod = logger

_NAV_PANE_WIDTH = 140
_LOGO_PADDING = 12
_LOGO_SIDE = _NAV_PANE_WIDTH // 4
_ENGINE_SETTING_LIMITS: dict[str, dict[str, dict[str, float | int | str]]] = {
    "copernican.engines.engine_mcmc": {
        "n_steps": {"min": 1, "max": 500_000},
        "burn_in_steps": {"min": 0, "max": 100_000},
        "n_walkers": {"min": 1, "max": 10_000},
        "pool_size": {"min": 1, "max": "cpu"},
    },
    "copernican.engines.engine_nested": {
        "n_live_points": {"min": 1, "max": 20_000},
        "max_iterations": {"min": 1, "max": 1_000_000},
        "evidence_tolerance": {"min": 1e-6, "max": 1.0},
        "enlargement_fraction": {"min": 0.1, "max": 10.0},
    },
}

_HELP_PAGES = [
    {"id": "gui", "label": "GUI guide", "path": "docs/gui_guide.md"},
    {"id": "cli", "label": "CLI guide", "path": "docs/cli_guide.md"},
]


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
    dataset: str = ""
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
        """Initialize the handler and store the log anchor prefix."""
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
    _CONFIRM_STEP_NAME = "Confirm"
    _CONFIRM_HEADER_TEXT = "Confirm and start"
    builder_steps: list[str] = [
        "Seed",
        "Models",
        "Data",
        "Engine",
        "Manifest",
        _CONFIRM_STEP_NAME,
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
        """Initialize GUI state, frames and backend selectors."""
        self.render = render and tk_gui is not None
        self.root: Optional[tkinter_module.Tk] = None
        self.frames: Dict[str, tkinter_module.Frame] = {}
        self.content_area: Optional[tkinter_module.Frame] = None
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
        self.selected_engine_kind: str = "mcmc"
        self._selected_model_entry: dict | None = None
        self._selected_engine_entry: dict | None = None
        self._equation_html_frame: HtmlFrame | None = None
        self._equations_window: tkinter_module.Toplevel | None = None
        self._equations_text_widget: tkinter_module.Text | None = None
        self.engine_capabilities: EngineCapabilities | None = None
        self._engine_setting_vars: dict[str, tkinter_module.Variable] = {}
        self._engine_setting_specs: dict[str, EngineSetting] = {}
        self._engine_run_settings_frame: ttk_module.LabelFrame | None = None
        self._current_engine_module: str | None = None
        self.selected_datasets: list[dict[str, str]] = []
        self.help_page_index = 0
        self._current_help_page_id = _HELP_PAGES[0]["id"]
        self._help_page_buttons: dict[str, ttk_module.Button] = {}
        self._help_text_widget: tkinter_module.Text | None = None
        self._help_title_label: ttk_module.Label | None = None
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
        self.application_log_handler: _MemoryLogHandler | None = None
        self.run_log_path: str | None = None
        self.run_log_handler: _MemoryLogHandler | None = None
        self.run_logger: logging.Logger | None = None
        self.diagnostics_filter_level = "INFO"
        self.monitor_filter_level = "INFO"
        self.diagnostics_clipboard = ""
        self._minigame_catalog = rng_minigames.load_registry()
        self.run_clipboard = ""
        self.alerts: list[UIMessage] = []
        self.inline_messages: list[UIMessage] = []
        self.last_log_jump: str | None = None
        self._run_process: subprocess.Popen[str] | None = None
        self._run_config_path: str | None = None
        self._status_label: ttk_module.Label | None = None
        self._progress_state_path: str | None = None
        self._progress_snapshot: dict | None = None
        self._progress_poll_thread: threading.Thread | None = None
        self._progress_poll_stop: threading.Event | None = None
        self._monitor_refresh_job: str | None = None
        self._progress_status_label: ttk_module.Label | None = None
        self._batch_progressbar: ttk_module.Progressbar | None = None
        self._walker_progressbar: ttk_module.Progressbar | None = None
        self._monitor_log_widget: tkinter_module.Text | None = None
        self._monitor_filter_label: ttk_module.Label | None = None
        self._monitor_log_view_button: ttk_module.Button | None = None
        self._monitor_log_open_button: ttk_module.Button | None = None
        self._monitor_log_lock_var: tkinter_module.BooleanVar | None = None
        self._monitor_control_buttons: list[ttk_module.Button] = []
        self._monitor_button_style_name = "Copernican.RunControl.TButton"
        self._monitor_button_style_ready = False
        self._validation_progress_status_label: ttk_module.Label | None = None
        self._validation_batch_progressbar: ttk_module.Progressbar | None = (
            None
        )
        self._validation_walker_progressbar: ttk_module.Progressbar | None = (
            None
        )
        self._validation_progress_env_backup: str | None = None
        self._diagnostics_log_widget: tkinter_module.Text | None = None
        self._validation_status_label: ttk_module.Label | None = None
        self._validation_text_widget: tkinter_module.Text | None = None
        self._validation_log_lock_var: tkinter_module.BooleanVar | None = None
        self._validation_button: ttk_module.Button | None = None
        self._validation_cancel_button: ttk_module.Button | None = None
        self._validation_clear_button: ttk_module.Button | None = None
        self._validation_process: subprocess.Popen[str] | None = None
        self._validation_stdout_thread: threading.Thread | None = None
        self._validation_running = False
        self._validation_status_base = "Status: idle"
        self._validation_log_lines: list[str] = []
        self._validation_last_stage_label: str | None = None
        self._diagnostics_filter_label: ttk_module.Label | None = None
        self._cancel_button: ttk_module.Button | None = None
        self._pause_button: ttk_module.Button | None = None
        self._hard_stop_button: ttk_module.Button | None = None
        self._run_output_button: ttk_module.Button | None = None
        self.logo_image: tkinter_module.PhotoImage | None = None
        self._status_bar_frame: ttk_module.Frame | None = None
        self._brand_status_label: ttk_module.Label | None = None
        self._environment_status_label: ttk_module.Label | None = None
        self._settings_sections = [
            {
                "id": "datasets",
                "label": "Datasets",
                "description": (
                    "Toggle automatic discovery, manage dataset hash caching "
                    "and recalculate digests on demand to keep the catalogue "
                    "aligned with the registered parsers."
                ),
                "builder": self._build_settings_datasets_page,
            },
            {
                "id": "gui",
                "label": "GUI",
                "description": (
                    "Adjust detachment, managed environment requirements and "
                    "the environment hints shown in the status bar."
                ),
                "builder": self._build_settings_gui_page,
            },
            {
                "id": "tools",
                "label": "Tools",
                "description": (
                    "Run cache rebuilds, parser revalidation or reset the Run "
                    "Builder workspace without leaving the settings screen."
                ),
                "builder": self._build_settings_tools_page,
            },
        ]
        self._analysis_sections = [
            {
                "id": "run_summary",
                "label": "Run Summary",
                "description": (
                    "Select an existing output folder, inspect the summary "
                    "metadata and load diagnostics/χ² breakdowns without "
                    "replaying the CLI log."
                ),
                "builder": self._build_analysis_run_summary_page,
                "actions": [
                    {
                        "label": "Reload summary",
                        "command": self._load_analysis_summary,
                    },
                    {
                        "label": "Export summary",
                        "command": self._export_analysis_summary,
                    },
                    {
                        "label": "Copy JSON",
                        "command": self._copy_analysis_summary_to_clipboard,
                    },
                ],
            },
            {
                "id": "diagnostics",
                "label": "Diagnostics",
                "description": (
                    "Diagnostics explorers and runtime alerts will surface "
                    "here once implemented."
                ),
                "builder": lambda frame: self._build_analysis_placeholder_page(
                    frame, "Diagnostics"
                ),
                "actions": [
                    {
                        "label": "1",
                        "command": self._analysis_placeholder_action,
                    },
                    {
                        "label": "2",
                        "command": self._analysis_placeholder_action,
                    },
                    {
                        "label": "3",
                        "command": self._analysis_placeholder_action,
                    },
                ],
            },
            {
                "id": "posteriors",
                "label": "Posteriors",
                "description": (
                    "Posterior explorers, plots and trace analysis will "
                    "appear here in future phases."
                ),
                "builder": self._build_analysis_posterior_page,
                "actions": [
                    {
                        "label": "Refresh files",
                        "command": self._refresh_analysis_posterior_list,
                    },
                    {
                        "label": "Fit to screen",
                        "command": self._analysis_fit_posterior_to_screen,
                    },
                    {
                        "label": "Toggle zoom",
                        "command": self._analysis_toggle_posterior_zoom,
                    },
                ],
            },
            {
                "id": "comparisons",
                "label": "Comparisons",
                "description": (
                    "Compare two run directories side by side "
                    "to see Δχ², parameter shifts and dataset counts."
                ),
                "builder": self._build_analysis_comparison_page,
                "actions": [
                    {
                        "label": "Refresh diff",
                        "command": self._analysis_compare_runs,
                    },
                    {
                        "label": "Export summary",
                        "command": self._export_analysis_comparison,
                    },
                    {
                        "label": "Copy JSON",
                        "command": self._copy_analysis_comparison_to_clipboard,
                    },
                ],
            },
        ]
        self._analysis_current_index = 0
        self._analysis_section_buttons: list[ttk_module.Button] = []
        self._analysis_action_buttons: list[ttk_module.Button] = []
        self._analysis_header_label: ttk_module.Label | None = None
        self._analysis_description_label: ttk_module.Label | None = None
        self._analysis_page_body: ttk_module.Frame | None = None
        self._analysis_run_path_var: tkinter_module.StringVar | None = None
        self._analysis_summary_status_label: ttk_module.Label | None = None
        self._analysis_summary_text_widget: tkinter_module.Text | None = None
        self._analysis_summary_result: analysis.RunAnalysisResult | None = None
        self._analysis_comparison_base_var: tkinter_module.StringVar | None = (
            None
        )
        self._analysis_comparison_alt_var: tkinter_module.StringVar | None = (
            None
        )
        self._analysis_comparison_text_widget: tkinter_module.Text | None = (
            None
        )
        self._analysis_comparison_status_label: ttk_module.Label | None = None
        self._analysis_comparison_result: dict[str, Any] | None = None
        self._analysis_comparison_pair: (
            tuple[analysis.RunAnalysisResult, analysis.RunAnalysisResult]
            | None
        ) = None
        self._analysis_posterior_files: list[Path] = []
        self._analysis_posterior_file_var: tkinter_module.StringVar | None = (
            None
        )
        self._analysis_posterior_combobox: ttk_module.Combobox | None = None
        self._analysis_plot_viewer: PlotViewer | None = None
        self._analysis_posterior_status_label: ttk_module.Label | None = None
        self._analysis_current_posterior_path: Path | None = None
        self._analysis_corner_files: list[Path] = []
        self._analysis_histogram_files: list[Path] = []
        self._settings_section_buttons: list[ttk_module.Button] = []
        self._settings_current_index = 0
        self._settings_header_label: ttk_module.Label | None = None
        self._settings_description_label: ttk_module.Label | None = None
        self._settings_page_body: ttk_module.Frame | None = None
        self._settings_defaults_button: ttk_module.Button | None = None
        self._settings_cancel_button: ttk_module.Button | None = None
        self._settings_save_button: ttk_module.Button | None = None
        self._settings_dirty = False
        self._suppress_setting_events = False
        self._saved_settings = copy.deepcopy(settings_mod.get_settings())
        self._pending_settings = copy.deepcopy(self._saved_settings)
        self._settings_vars: dict[str, dict[str, tkinter_module.Variable]] = {}
        self._bootstrap_logging()
        self._build_navigation()
        self._initialise_rendering()
        self.refresh_inventory()
        console_output.write("GUI: inventory refreshed and navigation ready.")
        self.help_banner_image = None
        self._load_saved_manifest_workspace()
        self._builder_step_buttons: list[ttk_module.Button] = []

    def _bootstrap_logging(self) -> None:
        """Start the application log and capture environment details."""

        app_logger = logger.get_logger()
        self.application_log_handler = _MemoryLogHandler(prefix="app")
        self._attach_handler(app_logger, self.application_log_handler)
        logger.log_environment_info(
            target_logger=app_logger,
            console=False,
        )
        self._log_application_event(
            "GUI launch completed; diagnostics stream active",
            logging.INFO,
        )

    def _ensure_monitor_button_styles(self) -> None:
        """Configure ttk styles so disabled run controls appear grey."""

        if self._monitor_button_style_ready:
            return
        if not self.render or ttk_module is None or self.root is None:
            return
        style = ttk_module.Style(self.root)
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

    def _page_header(
        self,
        frame: tkinter_module.Frame,
        title: str,
    ) -> ttk_module.Label:
        """Render a standard page header label."""

        label = ttk_module.Label(
            frame,
            text=title,
            font=("Helvetica", 16),
            takefocus=True,
        )
        label.pack(anchor="w", pady=(0, 8))
        return label

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

    def _log_application_event(
        self, message: str, level: int
    ) -> Optional[str]:
        """Record an application-level message and return its anchor."""

        app_logger = logger.get_logger()
        handler = self.application_log_handler
        app_logger.log(level, message)
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

        log_tag = f"copernican-run_{utils.get_timestamp()}.txt"
        output_root = Path(self._output_root())
        run_output_dir = output_root / Path(log_tag).stem
        run_output_dir.mkdir(parents=True, exist_ok=True)
        self.run_logger, self.run_log_path = logger.setup_monitor_logging(
            log_dir=str(run_output_dir),
            log_tag=log_tag,
        )
        self._current_run_output_dir = str(run_output_dir)
        self.run_log_handler = _MemoryLogHandler(prefix="run")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        formatter.converter = time.gmtime
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
            return self._log_application_event(message, level)
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
        if self._widget_is_alive(self._diagnostics_filter_label):
            self._diagnostics_filter_label.configure(
                text=f"Filter: {self.diagnostics_filter_level}+"
            )
        else:
            self._diagnostics_filter_label = None
        self._refresh_diagnostics_widget()

    def set_monitor_filter(self, severity: str) -> None:
        """Update the Run Monitor severity filter."""

        self.monitor_filter_level = severity.upper()
        if self._widget_is_alive(self._monitor_filter_label):
            self._monitor_filter_label.configure(
                text=f"Filter: {self.monitor_filter_level}+"
            )
        else:
            self._monitor_filter_label = None
        self._refresh_run_log_widget()

    def copy_application_logs(self) -> str:
        """Copy filtered application logs into a clipboard buffer."""

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
        """Write filtered application logs to the requested directory."""

        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, "application_log.txt")
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
            logger_obj = self.run_logger or logger.get_logger()
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
            self.root = tkinter_module.Tk()
            self.root.title(f"Copernican {self.gui_version}")
            screen_width = max(self.root.winfo_screenwidth(), 1)
            screen_height = max(self.root.winfo_screenheight(), 1)
            preferred_width, preferred_height = (
                (1200, 900)
                if screen_width > 1280 and screen_height > 900
                else (1100, 670)
            )
            self.root.geometry(f"{preferred_width}x{preferred_height}")
            self.root.minsize(width=900, height=670)
            self.root.configure(padx=0, pady=0)
            self._ensure_monitor_button_styles()
            self._build_layout()
            if self.root is not None:
                self.root.after(10, self._raise_root_window)
        except (
            RuntimeError,
            OSError,
            ValueError,
            tk_tcl_error,
        ) as exc:  # pragma: no cover - only hits Tk failures
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
            except (OSError, RuntimeError, ValueError):
                pass

    def _raise_root_window(self) -> None:
        """Bring the GUI window to the foreground when it launches."""

        if not self.render or self.root is None:
            return
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.attributes("-topmost", True)
            self.root.after(
                1500,
                lambda: self.root and self.root.attributes("-topmost", False),
            )
        except (RuntimeError, ValueError, tk_tcl_error):
            pass

    def _data_root(self) -> str:
        """Return the absolute data directory used for catalogue scans.

        Centralising the discovery root keeps the GUI aligned with
        :func:`dataset_registry.discover_trusted_parsers`, preventing the
        headless controller from drifting away from the registry's own safety
        checks when the repository layout changes.
        """

        return str(Path(__file__).resolve().parents[2] / "datasets")

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

    def _collect_dataset_hashes(
        self, data_dir: str, metadata: dict
    ) -> dict[str, str]:
        """Mirror the CLI hashing rules so the catalogue lists identical
        digests."""

        return dataset_registry.collect_dataset_hashes(data_dir, metadata)

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

    def _discover_dataset_catalogue(
        self, force_discovery: bool = False
    ) -> dict[str, dict]:
        """Return dataset metadata indexed by dataset identifier.

        Discovery defers to :func:`dataset_registry.discover_trusted_parsers`
        so the GUI honours the same hash validation rules as the CLI.  The
        resulting catalogue surfaces parser digests, metadata digests and
        compatibility badges for the detail panes.
        """

        dataset_registry.discover_trusted_parsers(
            self._data_root(),
            force=force_discovery,
        )
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
                    "hashes": self._collect_dataset_hashes(data_dir, meta),
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
            logger.get_logger().warning(
                "Model metadata in %s is malformed; continuing with blanks",
                path,
            )
            return {}

    def _build_model_entry(self, path: Path) -> dict:
        """Return a model catalogue record for ``path``."""

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
        return {
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
            "metadata": meta,
            "is_external": False,
        }

    def _discover_model_library(self) -> dict[str, dict]:
        """Return model metadata keyed by filename stem."""

        models: dict[str, dict] = {}
        for pattern in ("*.yml", "*.yaml"):
            for path in sorted(Path(self._models_root()).glob(pattern)):
                if path.name.startswith("__"):
                    continue
                models[path.stem] = self._build_model_entry(path)
        return models

    def _model_entry_for_value(self, model_id: str | None) -> dict | None:
        """Return the model record matching ``model_id`` if possible."""

        if not model_id:
            return None
        if self._selected_model_entry and (
            model_id == self._selected_model_entry.get("path")
            or model_id == self._selected_model_entry.get("id")
            or model_id == self._selected_model_entry.get("filename")
        ):
            return self._selected_model_entry
        entry = self.model_index.get(model_id)
        if entry:
            return entry
        for record in self.model_index.values():
            if (
                record.get("id") == model_id
                or record.get("filename") == model_id
                or record.get("path") == model_id
            ):
                return record
        return None

    def _load_model_from_path(self, path: str) -> dict | None:
        """Load a model YAML file from an arbitrary filesystem path."""

        candidate = Path(path).expanduser()
        if candidate.suffix.lower() not in {".yml", ".yaml"}:
            self.create_toast(
                "Model files must use a .yml or .yaml file.",
                severity="ERROR",
                context="models",
            )
            return None
        if not candidate.is_file():
            self.create_toast(
                f"Model not found: {candidate}",
                severity="ERROR",
                context="models",
            )
            return None
        try:
            with tempfile.TemporaryDirectory(
                prefix="copernican-model-cache-"
            ) as cache_root:
                model_spec_validator.validate_and_cache_model(
                    candidate.resolve(),
                    cache_root,
                )
            entry = self._build_model_entry(candidate.resolve())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Failed to load model: {exc}",
                severity="ERROR",
                context="models",
            )
            return None
        entry["is_external"] = True
        entry["path"] = str(candidate.resolve())
        self._selected_model_entry = entry
        self.selected_models = [entry["path"]]
        self.draft.model = entry["path"]
        self.summary.manifest_actions.append(
            f"Loaded model from {entry['path']}"
        )
        return entry

    def _discover_engine_library(self) -> dict[str, dict]:
        """Return engine metadata keyed by module stem."""

        engines: dict[str, dict] = {}
        for path in sorted(Path(self._engines_root()).glob("*.py")):
            if path.name.startswith("__"):
                continue
            module_name = f"copernican.engines.{path.stem}"
            try:
                module = importlib.import_module(module_name)
                label = getattr(module, "ENGINE_LABEL", path.stem)
                version = getattr(module, "ENGINE_VERSION", "unknown")
            except (
                AttributeError,
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
            ):
                module = None
                label = path.stem
                version = "unavailable"
                logger.get_logger().warning(
                    "Engine metadata import failed for %s; using fallbacks",
                    module_name,
                )
            engines[module_name] = {
                "id": module_name,
                "filename": path.name,
                "path": str(path),
                "stem": path.stem,
                "citation": getattr(module, "__doc__", ""),
                "license": "Copernican default license; verify engines",
                "version": version,
                "label": label,
                "badges": ["SNE", "BAO", "CMB"],
                "hash": utils.compute_sha256(str(path)),
            }
        return engines

    def refresh_inventory(self, force_discovery: bool = False) -> None:
        """Refresh catalogue, model and engine metadata for list views.

        The method runs at startup and when panels request a revalidation.  It
        guarantees that GUI detail panes reflect the parser registration rules
        enforced by :mod:`dataset_registry` and that Run Builder defaults pull
        from the newest manifests and compatibility declarations.
        """

        self.validation_notes = []
        self.catalogue_index = self._discover_dataset_catalogue(
            force_discovery=force_discovery
        )
        self.model_index = self._discover_model_library()
        self.engine_index = self._discover_engine_library()

    def _catalogue_health_summary(self) -> dict[str, object]:
        """Return dataset, trust and filter metadata for the Home dashboard."""

        datasets = list(self.catalogue_index.values())
        type_counter = Counter(
            (entry.get("type") or "unknown").upper() for entry in datasets
        )
        untrusted = [
            {
                "id": entry.get("id", ""),
                "type": entry.get("type", "unknown"),
                "name": entry.get("name", entry.get("id", "")),
            }
            for entry in datasets
            if not entry.get("parser_trusted", True)
        ]
        return {
            "dataset_count": len(datasets),
            "type_counter": type_counter,
            "untrusted": untrusted,
            "notes": self.validation_notes[:3],
        }

    def _model_engine_health_summary(self) -> dict[str, object]:
        """Return compatibility and version health for models and engines."""

        model_badges = Counter()
        stale_models: list[tuple[str, str]] = []
        for entry in self.model_index.values():
            for badge in entry.get("badges", []):
                model_badges[badge.upper()] += 1
            version_label = (entry.get("version") or "").lower()
            if not version_label or version_label in {
                "unknown",
                "unavailable",
            }:
                stale_models.append(
                    (
                        entry.get("id") or entry.get("filename", "model"),
                        "missing",
                    )
                )
        stale_engines: list[tuple[str, str]] = []
        for entry in self.engine_index.values():
            version_label = (entry.get("version") or "").lower()
            if not version_label or version_label in {
                "unknown",
                "unavailable",
            }:
                stale_engines.append(
                    (
                        entry.get("label") or entry.get("id", "engine"),
                        "missing",
                    )
                )
        return {
            "model_count": len(self.model_index),
            "engine_count": len(self.engine_index),
            "model_badges": model_badges,
            "stale_models": stale_models,
            "stale_engines": stale_engines,
        }

    def filter_catalogue(self, types: list[str] | None = None) -> list[dict]:
        """Return datasets matching ``types`` while recording the filter."""

        if types is None:
            types = []
        resolved = [type_name.lower() for type_name in types if type_name]
        self.last_filter_types = resolved
        if not resolved:
            return list(self.catalogue_index.values())
        return [
            entry
            for entry in self.catalogue_index.values()
            if entry.get("type", "").lower() in resolved
        ]

    def _show_data_with_filter(self, dataset_types: list[str]) -> None:
        """Filter the catalogue and open the Data tab."""

        self.filter_catalogue(dataset_types)
        self.show_data_overview()

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

        return str(Path.home() / "copernican_output")

    def _launch_folder(self, path: str) -> None:
        """Open ``path`` in the native file manager when rendering."""

        if not self.render:
            return
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]  # nosec B606
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)  # nosec
            else:
                subprocess.run(["xdg-open", path], check=False)  # nosec
        except (OSError, RuntimeError, ValueError) as exc:
            console_output.write(
                f"Unable to open folder {path}: {exc}", error=True
            )

    def _launch_minigame(
        self, minigame_id: str, seed_var: "tkinter_module.StringVar"
    ) -> None:
        """Import and launch a mini-game on-demand."""

        try:
            launcher = rng_minigames.load_launcher(minigame_id)
        except KeyError:
            self.create_toast(
                f"Unknown mini-game '{minigame_id}'.",
                severity="ERROR",
                context="seed",
            )
            return
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
        ) as exc:  # pragma: no cover - import failure path
            log_mod.error(
                "Failed to import mini-game %s: %s",
                minigame_id,
                exc,
            )
            self.create_toast(
                f"Could not load mini-game: {exc}",
                severity="ERROR",
                context="seed",
            )
            return
        context = rng_minigames.MinigameContext(
            set_seed=lambda value: seed_var.set(value),
            notify=lambda msg, severity="INFO": self.create_toast(
                msg, severity=severity, context="seed"
            ),
            render=self.render,
            tk_root=self.root,
        )
        launcher(context)

    def open_output_directory(self) -> str:
        """Ensure the output directory exists and open it for the user."""

        output_dir = self._output_root()
        os.makedirs(output_dir, exist_ok=True)
        return self.open_folder(output_dir)

    def _open_path_with_system(self, path: str) -> None:
        """Open ``path`` using the operating system defaults."""

        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]  # nosec B606
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)  # nosec
            else:
                subprocess.run(["xdg-open", path], check=False)  # nosec
        except (OSError, RuntimeError, ValueError) as exc:
            console_output.write(f"Unable to open {path}: {exc}", error=True)

    def _show_metadata_dialog(
        self, title: str, content: str, source_path: str | None = None
    ) -> None:
        """Show metadata content in a scrollable dialog when rendering."""

        if not self.render or self.root is None:
            console_output.write(f"{title}:\n{content}")
            return
        window = tkinter_module.Toplevel(self.root)
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
        container = ttk_module.Frame(window, padding=(8, 6))
        container.pack(fill="both", expand=True)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        text = tkinter_module.Text(
            container,
            wrap="word",
            height=display_lines or 1,
        )
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk_module.Scrollbar(
            container, orient="vertical", command=text.yview
        )
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        buttons = ttk_module.Frame(container)
        buttons.grid(row=1, column=0, columnspan=2, pady=(8, 8))
        close_button = ttk_module.Button(
            buttons,
            text="Close",
            command=window.destroy,
        )
        close_button.pack(side="left", padx=4)
        if source_path:
            ttk_module.Button(
                buttons,
                text="Open file…",
                command=lambda: self._open_path_with_system(source_path),
            ).pack(side="left", padx=4)

    def _create_scrollable_panel(
        self, parent: tkinter_module.Frame, *, height: int | None = None
    ) -> tkinter_module.Frame:
        """Return a frame that scrolls vertically with the given parent."""

        container = ttk_module.Frame(parent)
        container.pack(fill="both", expand=True)
        canvas_kwargs: dict[str, int | str] = {
            "borderwidth": 0,
            "highlightthickness": 0,
        }
        if height is not None:
            canvas_kwargs["height"] = height
        canvas = tkinter_module.Canvas(container, **canvas_kwargs)
        scrollbar = ttk_module.Scrollbar(
            container, orient="vertical", command=canvas.yview
        )
        inner_frame = ttk_module.Frame(canvas)
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
            console_output.write(
                f"GUI: revalidation requested for dataset {dataset_id}."
            )
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

        if (
            self.help_banner_image is not None
            or not self.render
            or tk_gui is None
        ):
            return
        banner_path = (
            Path(__file__).resolve().parents[2] / "docs" / "banner_github.png"
        )
        if not banner_path.exists():
            return
        try:
            self.help_banner_image = tkinter_module.PhotoImage(
                file=str(banner_path)
            )
        except (OSError, RuntimeError, ValueError, tk_tcl_error) as exc:
            logger.get_logger().warning(
                "Failed to load help banner image: %s", exc
            )
            self.help_banner_image = None

    def _logo_image_path(self) -> Path:
        """Return the expected path to the resized navigation logo."""

        return Path(__file__).resolve().parents[2] / "img" / "logo_small.png"

    def _load_logo_image(self) -> None:
        """Load the navigation logo when rendering is enabled."""

        if self.logo_image is not None or not self.render or tk_gui is None:
            return
        logo_path = self._logo_image_path()
        if not logo_path.exists():
            return
        try:
            self.logo_image = tkinter_module.PhotoImage(file=str(logo_path))
        except (OSError, RuntimeError, ValueError, tk_tcl_error) as exc:
            logger.get_logger().warning(
                "Failed to load navigation logo: %s", exc
            )

    def _build_navigation_logo(self, nav_frame: tkinter_module.Frame) -> None:
        """Add the padded logo square above the navigation buttons."""

        if not self.render or tk_gui is None:
            return
        self._load_logo_image()
        logo_holder = ttk_module.Frame(nav_frame)
        logo_holder.pack(fill="x", pady=(20, _LOGO_PADDING + 10))
        logo_holder.pack_propagate(False)
        base_side = _LOGO_SIDE
        image_width = self.logo_image.width() if self.logo_image else base_side
        image_height = (
            self.logo_image.height() if self.logo_image else base_side
        )
        holder_height = image_height + 2 * _LOGO_PADDING + 6
        holder_width = image_width + 2 * _LOGO_PADDING
        logo_holder.configure(height=holder_height)
        square = ttk_module.Frame(
            logo_holder,
            width=holder_width,
            height=holder_height,
            padding=(
                _LOGO_PADDING,
                _LOGO_PADDING,
                _LOGO_PADDING,
                _LOGO_PADDING - 6,
            ),
        )
        square.pack_propagate(False)
        square.pack(anchor="center")
        square.columnconfigure(0, weight=1)
        square.rowconfigure(0, weight=1)
        if self.logo_image:
            logo_widget = ttk_module.Label(
                square,
                image=self.logo_image,
            )
            logo_widget.image = self.logo_image
        else:
            logo_widget = ttk_module.Label(
                square,
                text="Copernican",
                anchor="center",
                justify="center",
            )
        logo_widget.grid(row=0, column=0, sticky="nsew")

    def _load_help_markdown(self, relative_path: str) -> str:
        """Return the contents of the requested markdown asset."""

        doc_path = Path(__file__).resolve().parents[2] / relative_path
        try:
            return doc_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, RuntimeError, ValueError) as exc:
            message = f"Unable to read {relative_path} for Help panel: {exc}"
            logger.get_logger().warning(message)
            return message

    def _render_markdown_in_text_widget(
        self, widget: tkinter_module.Text, markdown: str
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
        widget: tkinter_module.Text,
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

    def _help_page_record(self, page_id: str | None = None) -> dict[str, str]:
        """Return the help page metadata for the provided identifier."""

        target_id = page_id or self._current_help_page_id
        for index, page in enumerate(_HELP_PAGES):
            if page["id"] == target_id:
                self.help_page_index = index
                return page
        self.help_page_index = 0
        self._current_help_page_id = _HELP_PAGES[0]["id"]
        return _HELP_PAGES[0]

    def _help_header_text(self) -> str:
        """Return the title string for the current help page."""

        record = self._help_page_record()
        return f"Help: {record['label']}"

    def _select_help_page(self, page_id: str) -> None:
        """Switch the help content to a new page and refresh widgets."""

        self._current_help_page_id = page_id
        self._help_page_record(page_id)
        self._refresh_help_page_view()

    def _refresh_help_page_view(self) -> None:
        """Update the help header, buttons and rendered markdown."""

        record = self._help_page_record()
        if self._help_title_label is not None:
            self._help_title_label.configure(text=self._help_header_text())
        for page in _HELP_PAGES:
            button = self._help_page_buttons.get(page["id"])
            if not button:
                continue
            if page["id"] == self._current_help_page_id:
                button.state(["disabled"])
            else:
                button.state(["!disabled"])
        if self._help_text_widget is None or not self.render or tk_gui is None:
            return
        markdown = self._load_help_markdown(record["path"])
        self._help_text_widget.configure(state="normal")
        self._help_text_widget.delete("1.0", tkinter_module.END)
        self._render_markdown_in_text_widget(self._help_text_widget, markdown)
        self._help_text_widget.yview_moveto(0.0)
        self._help_text_widget.configure(state="disabled")

    def view_metadata_file(self, asset_id: str) -> str:
        """Return cached metadata content for the requested asset."""

        path = self._resolve_asset_path(asset_id)
        return self._read_asset_text(path)

    def revalidate_dataset(self, dataset_id: str) -> dict[str, str]:
        """Re-run parser trust checks and return the refreshed record."""

        self.refresh_inventory(force_discovery=True)
        if dataset_id not in self.catalogue_index:
            raise KeyError(dataset_id)
        record = self.catalogue_index[dataset_id]
        console_output.write(
            f"GUI: dataset {dataset_id} diagnostics refreshed "
            f"(parser digest {record.get('parser_digest', 'n/a')})."
        )
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
                "analysis",
                "Analysis",
                CopernicanGUI.show_analysis,
            ),
            NavigationItem(
                "validation",
                "Validation",
                CopernicanGUI.show_validation,
            ),
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
        nav_frame = ttk_module.Frame(self.root, padding=(24, 12, 24, 12))
        nav_frame.configure(width=_NAV_PANE_WIDTH)
        nav_frame.grid(row=0, column=0, sticky="nsw")
        nav_frame.grid_propagate(False)
        self.root.grid_columnconfigure(0, minsize=_NAV_PANE_WIDTH)
        separator = ttk_module.Separator(self.root, orient="vertical")
        separator.grid(row=0, column=1, rowspan=2, sticky="ns")
        self.root.grid_columnconfigure(1, minsize=2)
        self.content_area = ttk_module.Frame(
            self.root,
            padding=(12, 12, 12, 12),
        )
        self.content_area.grid(row=0, column=2, sticky="nsew")
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)

        self._build_navigation_logo(nav_frame)
        nav_body = ttk_module.Frame(nav_frame)
        nav_body.pack(fill="both", expand=True)
        exit_frame = ttk_module.Frame(nav_frame)
        exit_frame.pack(fill="x", side="bottom")
        for nav_item in self.nav_items:
            target_frame = exit_frame if nav_item.name == "exit" else nav_body
            button = ttk_module.Button(
                target_frame,
                text=nav_item.label,
                command=lambda i=nav_item: i.action(self),
                takefocus=True,
            )
            button.pack(fill="x", pady=4)

        separator_bottom = ttk_module.Separator(
            self.root,
            orient="horizontal",
        )
        separator_bottom.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(4, 0),
        )
        self._build_status_bar()
        self.show_home()
        self._refresh_environment_status()

    def _swap_content(
        self,
        frame_builder: Callable[[tkinter_module.Frame], None],
    ) -> None:
        """Replace the right-hand content area with a new frame."""

        if not self.render or self.content_area is None:
            return
        for child in self.content_area.winfo_children():
            child.destroy()
        frame = ttk_module.Frame(self.content_area, padding=(8, 8))
        frame.pack(fill="both", expand=True)
        frame_builder(frame)

    def _build_status_bar(self) -> None:
        """Create the bottom status bar that shows environment metadata."""

        if not self.render or self.root is None:
            return
        if self._status_bar_frame is not None:
            return
        status_bar = ttk_module.Frame(
            self.root,
            padding=(8, 0, 8, 2),
            relief="flat",
            borderwidth=0,
        )
        status_bar.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 0),
        )
        status_bar.columnconfigure(0, weight=1)
        status_bar.columnconfigure(1, weight=1)
        self._brand_status_label = ttk_module.Label(
            status_bar,
            text="",
            foreground="#6c6c6c",
            anchor="w",
            takefocus=True,
        )
        self._brand_status_label.grid(row=0, column=0, sticky="w", pady=(0, 0))
        self._environment_status_label = ttk_module.Label(
            status_bar,
            text="",
            anchor="e",
            foreground="#6c6c6c",
            takefocus=True,
        )
        self._environment_status_label.grid(
            row=0, column=1, sticky="e", pady=(0, 0)
        )
        self._refresh_brand_status()
        self._status_bar_frame = status_bar

    def _environment_summary_text(self) -> str:
        """Return text describing the active Copernican environment."""

        python_label = platform.python_version()
        python_impl = platform.python_implementation()
        python_text = f"{python_impl} {python_label}"
        venv_path = os.environ.get("VIRTUAL_ENV") or ""
        if venv_path:
            venv_name = Path(venv_path).name
            if venv_name == ".venv":
                venv_text = "Managed .venv active"
            else:
                venv_text = f"VIRTUAL_ENV={venv_name}"
        else:
            venv_text = "Managed .venv inactive"
        summary_parts = [python_text, venv_text]
        return "  ".join(summary_parts)

    def _brand_status_text(self) -> str:
        """Return text describing the brand for the status strip."""

        return (
            f"Copernican {self.gui_version}  "
            "\u00a9 Apostol Apostolov & Black Epsilon Ltd."
        )

    def _refresh_environment_status(self) -> None:
        """Update the environment strip text."""

        self._refresh_brand_status()
        if self._environment_status_label is None:
            return
        self._environment_status_label.configure(
            text=self._environment_summary_text()
        )

    def _builder_status_message(self) -> str:
        """Return contextual instructions for the Run Builder header."""

        if self.current_step_index <= 3:
            return (
                "Work through the pages to assemble a reproducible manifest.\n"
                "After you have input all the settings, you will be able "
                "to proceed and save your manifest for a run."
            )
        current_step = self.builder_steps[self.current_step_index]
        if current_step == "Manifest":
            if self.manifest_workspace is None:
                return (
                    "Your manifest is currently only in Copernican's head. "
                    "You have to save it on your hard drive, solid state "
                    "drive, liquid state, gas or plasma drive to proceed to "
                    "Confirm and start your inference run."
                )
            return (
                "Now Copernican knows what to work with and won't look at the "
                "wrong model while you are sleeping. No need to babysit it "
                "anymore—just proceed to Confirm and run."
            )
        if current_step == self._CONFIRM_STEP_NAME:
            return (
                "If you are reading this, you are either on the brink of a "
                "groundbreaking discovery, or this diabolical software really "
                "works. The former is far more probable. Take a look at your "
                "run specs and start your run!"
            )
        return ""

    def _refresh_brand_status(self) -> None:
        """Update the left-aligned brand status label."""

        if self._brand_status_label is None:
            return
        self._brand_status_label.configure(text=self._brand_status_text())

    def show_home(self) -> None:
        """Render the project home panel with recents and quick actions."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the project home view inside the main panel."""
            self._page_header(frame, "Project Home")
            tiles = ttk_module.Frame(frame)
            tiles.pack(fill="x", pady=(0, 12))
            tiles.columnconfigure(0, weight=1)
            tiles.columnconfigure(1, weight=1)
            catalogue_health = self._catalogue_health_summary()
            cat_card = ttk_module.LabelFrame(
                tiles, text="Catalogue health", padding=(10, 8)
            )
            cat_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            dataset_summary = (
                f"{catalogue_health['dataset_count']} dataset(s) "
                f"/ {len(catalogue_health['untrusted'])} trust alert(s)"
            )
            dataset_label = ttk_module.Label(
                cat_card,
                text=dataset_summary,
                takefocus=True,
            )
            dataset_label.pack(anchor="w")
            type_counter = Counter(catalogue_health.get("type_counter") or {})
            if type_counter:
                type_summary = ", ".join(
                    f"{name}: {count}"
                    for name, count in sorted(type_counter.items())
                )
            else:
                type_summary = "No datasets discovered."
            type_label = ttk_module.Label(
                cat_card,
                text=type_summary,
                takefocus=True,
            )
            type_label.pack(anchor="w", pady=(2, 4))
            filter_row = ttk_module.Frame(cat_card)
            filter_row.pack(anchor="w", pady=(0, 6))
            filter_actions = [
                ("All data", ()),
                ("SNe", ("sne",)),
                ("BAO", ("bao",)),
                ("CMB", ("cmb",)),
            ]
            for label_text, filter_types in filter_actions:
                ttk_module.Button(
                    filter_row,
                    text=label_text,
                    command=(
                        lambda types=filter_types: self._show_data_with_filter(
                            list(types)
                        )
                    ),
                    takefocus=True,
                ).pack(side="left", padx=2)
            if catalogue_health["notes"]:
                ttk_module.Label(
                    cat_card,
                    text="; ".join(catalogue_health["notes"]),
                    wraplength=320,
                    takefocus=True,
                ).pack(anchor="w", pady=(0, 4))
            if catalogue_health["untrusted"]:
                ttk_module.Label(
                    cat_card,
                    text="Revalidate untrusted datasets:",
                    takefocus=True,
                ).pack(anchor="w", pady=(4, 0))
                for offender in catalogue_health["untrusted"][:3]:
                    display_name = offender["name"] or offender["id"]
                    ttk_module.Button(
                        cat_card,
                        text=f"Revalidate {display_name}",
                        command=lambda dataset_id=offender[
                            "id"
                        ]: self._handle_home_revalidate(dataset_id),
                        takefocus=True,
                    ).pack(anchor="w", pady=2)
            models_health = self._model_engine_health_summary()
            models_card = ttk_module.LabelFrame(
                tiles, text="Models & Engines", padding=(10, 8)
            )
            models_card.grid(row=0, column=1, sticky="nsew")
            badge_summary = (
                ", ".join(
                    f"{badge}: {count}"
                    for badge, count in sorted(
                        models_health["model_badges"].items()
                    )
                )
                if models_health["model_badges"]
                else "No compatibility badges recorded."
            )
            ttk_module.Label(
                models_card,
                text=(
                    f"{models_health['model_count']} model(s) / "
                    f"{models_health['engine_count']} engine(s)"
                ),
                takefocus=True,
            ).pack(anchor="w")
            ttk_module.Label(
                models_card,
                text=badge_summary,
                wraplength=320,
                takefocus=True,
            ).pack(anchor="w", pady=(2, 4))
            if models_health["stale_models"]:
                stale_models_text = ", ".join(
                    entry[0] for entry in models_health["stale_models"][:3]
                )
            else:
                stale_models_text = "None"
            if models_health["stale_engines"]:
                stale_engines_text = ", ".join(
                    entry[0] for entry in models_health["stale_engines"][:2]
                )
            else:
                stale_engines_text = "None"
            ttk_module.Label(
                models_card,
                text=f"Stale models: {stale_models_text}",
                wraplength=320,
                takefocus=True,
            ).pack(anchor="w")
            ttk_module.Label(
                models_card,
                text=f"Stale engines: {stale_engines_text}",
                wraplength=320,
                takefocus=True,
            ).pack(anchor="w")
            action_row = ttk_module.Frame(models_card)
            action_row.pack(anchor="w", pady=(6, 0))
            ttk_module.Button(
                action_row,
                text="View models",
                command=self.show_models,
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk_module.Button(
                action_row,
                text="View engines",
                command=self.show_engines,
                takefocus=True,
            ).pack(side="left", padx=2)
            if self.recent_runs or self.pinned_configs:
                ttk_module.Separator(frame, orient="horizontal").pack(
                    fill="x", pady=(4, 8)
                )
            if self.recent_runs:
                recent_runs_label = ttk_module.Label(
                    frame,
                    text="Recent runs",
                    takefocus=True,
                )
                recent_runs_label.pack(anchor="w")
                for run in self.recent_runs:
                    run_label = ttk_module.Label(
                        frame,
                        text=run,
                        takefocus=True,
                    )
                    run_label.pack(anchor="w")
            if self.pinned_configs:
                config_label = ttk_module.Label(
                    frame,
                    text="Quick configurations",
                    takefocus=True,
                )
                config_label.pack(anchor="w", pady=(12, 0))
                for config in self.pinned_configs:
                    config_entry_label = ttk_module.Label(
                        frame,
                        text=config,
                        takefocus=True,
                    )
                    config_entry_label.pack(anchor="w")
            quick_actions_label = ttk_module.Label(
                frame,
                text="Quick actions",
                takefocus=True,
            )
            quick_actions_label.pack(anchor="w", pady=(12, 0))
            for label_text, callback in self.quick_actions:
                action_button = ttk_module.Button(
                    frame,
                    text=label_text,
                    command=callback,
                    takefocus=True,
                )
                action_button.pack(anchor="w", pady=2)

        self._swap_content(builder)

    def show_validation(self) -> None:
        """Display the lightweight validation runner and latest summary."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the validation runner dashboard and controls."""
            self._validation_progress_status_label = None
            self._validation_batch_progressbar = None
            self._validation_walker_progressbar = None
            self._page_header(frame, "Validation")
            self._validation_status_label = ttk_module.Label(
                frame,
                text=self._validation_status_base,
                takefocus=True,
            )
            self._validation_status_label.pack(anchor="w")
            progress_frame = ttk_module.Frame(frame)
            progress_frame.pack(fill="x", pady=(8, 8))
            self._validation_progress_status_label = ttk_module.Label(
                progress_frame,
                text="Stage: Idle",
                font=("Helvetica", 12, "bold"),
                takefocus=True,
            )
            self._validation_progress_status_label.pack(anchor="w")
            ttk_module.Label(
                progress_frame,
                text="Overall validation progress",
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))
            self._validation_batch_progressbar = ttk_module.Progressbar(
                progress_frame, maximum=100, length=360
            )
            self._validation_batch_progressbar.pack(fill="x", pady=(2, 0))
            ttk_module.Label(
                progress_frame,
                text="Per-manifest progress",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            self._validation_walker_progressbar = ttk_module.Progressbar(
                progress_frame, maximum=100, length=360
            )
            self._validation_walker_progressbar.pack(fill="x", pady=(2, 0))
            controls = ttk_module.Frame(frame)
            controls.pack(fill="x", pady=(8, 4))
            button_style = self._monitor_button_kwargs()
            self._validation_button = ttk_module.Button(
                controls,
                text="Run validation suite",
                command=self._start_validation_run,
                takefocus=True,
                **button_style,
            )
            self._validation_button.pack(side="left", padx=4)
            self._validation_cancel_button = ttk_module.Button(
                controls,
                text="Cancel validation",
                command=self._cancel_validation_run,
                takefocus=True,
                **button_style,
            )
            self._validation_cancel_button.pack(side="left", padx=4)
            self._validation_clear_button = ttk_module.Button(
                controls,
                text="Clear validation",
                command=self._clear_validation_results,
                takefocus=True,
                **button_style,
            )
            self._validation_clear_button.pack(side="left", padx=4)
            summary_frame = ttk_module.LabelFrame(
                frame,
                text="Validation log",
            )
            summary_frame.pack(fill="x", pady=(8, 0))
            summary_frame.columnconfigure(0, weight=1)
            summary_frame.rowconfigure(0, weight=0)
            summary_frame.rowconfigure(1, weight=0)
            lock_frame = ttk_module.Frame(summary_frame)
            lock_frame.grid(row=0, column=0, sticky="w", pady=(0, 4))
            self._validation_log_lock_var = tkinter_module.BooleanVar(
                value=True
            )
            ttk_module.Checkbutton(
                lock_frame,
                text="Lock summary to latest entry",
                variable=self._validation_log_lock_var,
                takefocus=True,
            ).pack(side="left")
            text_panel = ttk_module.Frame(summary_frame)
            text_panel.grid(row=1, column=0, sticky="nsew")
            text_panel.columnconfigure(0, weight=1)
            text_panel.rowconfigure(0, weight=1)
            text_widget = tkinter_module.Text(
                text_panel,
                wrap="none",
                padx=8,
                pady=6,
                borderwidth=0,
                highlightthickness=0,
                height=20,
            )
            text_widget.grid(row=0, column=0, sticky="nsew")
            vscroll = ttk_module.Scrollbar(
                text_panel,
                orient="vertical",
                command=text_widget.yview,
            )
            vscroll.grid(row=0, column=1, sticky="ns")
            hscroll = ttk_module.Scrollbar(
                text_panel,
                orient="horizontal",
                command=text_widget.xview,
            )
            hscroll.grid(row=1, column=0, sticky="ew")
            text_widget.configure(
                yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
            )
            text_widget.configure(state="disabled")
            self._validation_text_widget = text_widget
            existing_summary = validation_utils.read_validation_summary()
            if not self._validation_running:
                self._validation_status_base = (
                    "Status: summary available"
                    if existing_summary
                    else "Status: idle"
                )
            if existing_summary and not self._validation_log_lines:
                self._append_validation_summary_text(existing_summary)
            self._refresh_validation_log_widget_from_history()
            self._update_validation_status_label(self._progress_snapshot)
            self._update_validation_control_states()
            self._refresh_validation_progress_widgets()

        self._swap_content(builder)

    def show_analysis(self) -> None:
        """Display an analysis workspace with tabbed sections."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the analysis workspace tabs and body content."""
            section = self._analysis_sections[self._analysis_current_index]
            header_text = f"Analysis: {section['label']}"
            self._analysis_header_label = self._page_header(frame, header_text)
            tab_bar = ttk_module.Frame(frame)
            tab_bar.pack(fill="x", pady=(0, 8))
            self._analysis_section_buttons = []
            button_style = self._monitor_button_kwargs()
            for index, sec in enumerate(self._analysis_sections):
                button = ttk_module.Button(
                    tab_bar,
                    text=sec["label"],
                    command=lambda idx=index: self._navigate_analysis_section(
                        idx
                    ),
                    takefocus=True,
                    **button_style,
                )
                button.pack(side="left", padx=4)
                self._analysis_section_buttons.append(button)
            self._analysis_description_label = ttk_module.Label(
                frame,
                text=section["description"],
                wraplength=720,
                takefocus=True,
            )
            self._analysis_description_label.pack(anchor="w", pady=(0, 8))
            action_row = ttk_module.Frame(frame)
            action_row.pack(fill="x", pady=(0, 20))
            self._analysis_action_buttons = []
            for _ in range(3):
                button = ttk_module.Button(
                    action_row,
                    text="",
                    command=self._analysis_placeholder_action,
                    takefocus=True,
                    **button_style,
                )
                button.pack(side="left", padx=4)
                self._analysis_action_buttons.append(button)
            body = ttk_module.Frame(frame)
            body.pack(fill="both", expand=True)
            self._analysis_page_body = body
            self._refresh_analysis_section()

        if not self.render or self.root is None:
            return
        self._swap_content(builder)

    def _navigate_analysis_section(self, index: int) -> None:
        """Switch to a different analysis section tab."""

        if index == self._analysis_current_index:
            return
        self._analysis_current_index = index
        self._refresh_analysis_section()

    def _refresh_analysis_section(self) -> None:
        """Update labels, buttons and builder content for the current tab."""

        if not self._analysis_sections:
            return
        section = self._analysis_sections[self._analysis_current_index]
        if self._analysis_header_label is not None:
            self._analysis_header_label.configure(
                text=f"Analysis: {section['label']}"
            )
        if self._analysis_description_label is not None:
            self._analysis_description_label.configure(
                text=section["description"]
            )
        for index, button in enumerate(self._analysis_section_buttons):
            if index == self._analysis_current_index:
                button.state(["disabled"])
            else:
                button.state(["!disabled"])
        self._refresh_analysis_action_buttons(section["actions"])
        if self._analysis_page_body is not None:
            for child in self._analysis_page_body.winfo_children():
                child.destroy()
            section["builder"](self._analysis_page_body)

    def _refresh_analysis_action_buttons(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> None:
        """Update the action buttons beneath the description."""

        for button, action in zip(self._analysis_action_buttons, actions):
            button.configure(text=action["label"], command=action["command"])

    def _analysis_placeholder_action(self) -> None:
        """No-op placeholder invoked by unfinished action buttons."""

        return

    def _build_analysis_run_summary_page(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the Run Summary tab content."""

        if tk_gui is None:
            ttk_module.Label(
                container,
                text="GUI is unavailable in this environment.",
                wraplength=720,
            ).pack(anchor="w")
            return

        run_path_var = self._analysis_run_path_var
        if run_path_var is None:
            run_path_var = tkinter_module.StringVar()
            self._analysis_run_path_var = run_path_var

        row = ttk_module.Frame(container)
        row.pack(fill="x", pady=(0, 6))
        ttk_module.Label(row, text="Run directory:").pack(side="left")
        entry = ttk_module.Entry(row, textvariable=run_path_var, width=60)
        entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        entry.bind("<Return>", lambda event: self._load_analysis_summary())
        ttk_module.Button(
            row,
            text="Browse…",
            command=self._browse_analysis_run_dir,
            takefocus=True,
            **self._monitor_button_kwargs(),
        ).pack(side="left")
        ttk_module.Button(
            row,
            text="Load summary",
            command=self._load_analysis_summary,
            takefocus=True,
            **self._monitor_button_kwargs(),
        ).pack(side="left", padx=(4, 0))

        self._analysis_summary_status_label = ttk_module.Label(
            container,
            text="Select a run output directory and load its summary.",
            wraplength=720,
            takefocus=True,
        )
        self._analysis_summary_status_label.pack(anchor="w", pady=(4, 6))

        text_panel = ttk_module.Frame(container)
        text_panel.pack(fill="both", expand=True)
        text_panel.columnconfigure(0, weight=1)
        text_panel.rowconfigure(0, weight=1)
        if tk_gui is not None:
            self._analysis_summary_text_widget = tkinter_module.Text(
                text_panel,
                wrap="none",
                padx=8,
                pady=6,
                borderwidth=0,
                highlightthickness=0,
                height=18,
            )
            self._analysis_summary_text_widget.grid(
                row=0, column=0, sticky="nsew"
            )
            vscroll = ttk_module.Scrollbar(
                text_panel,
                orient="vertical",
                command=self._analysis_summary_text_widget.yview,
            )
            vscroll.grid(row=0, column=1, sticky="ns")
            hscroll = ttk_module.Scrollbar(
                text_panel,
                orient="horizontal",
                command=self._analysis_summary_text_widget.xview,
            )
            hscroll.grid(row=1, column=0, sticky="ew")
            self._analysis_summary_text_widget.configure(
                yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
            )
            self._analysis_summary_text_widget.configure(state="disabled")

            summary_label = ttk_module.Label(
                container,
                text=(
                    "Loaded summaries show run metadata, diagnostics and "
                    "chi² breakdowns sourced from the manifest, parameter "
                    "summary and log."
                ),
                wraplength=720,
                justify="left",
            )
            summary_label.pack(anchor="w", pady=(8, 0))

    def _build_analysis_posterior_page(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the Posteriors tab with the shared plot viewer."""

        if tk_gui is None:
            ttk_module.Label(
                container,
                text="GUI is unavailable in this environment.",
                wraplength=720,
            ).pack(anchor="w")
            return

        run_dir = self._analysis_run_dir()
        run_label = (
            f"Current run: {run_dir}"
            if run_dir
            else "Current run: <select from Run Summary>"
        )
        run_label_widget = ttk_module.Label(
            container,
            text=run_label,
            wraplength=720,
            takefocus=True,
        )
        run_label_widget.pack(anchor="w", pady=(0, 4))

        selection_row = ttk_module.Frame(container)
        selection_row.pack(fill="x", pady=(0, 4))
        run_label_field = ttk_module.Label(
            selection_row,
            text="Posterior file:",
        )
        run_label_field.pack(side="left")
        if self._analysis_posterior_file_var is None:
            self._analysis_posterior_file_var = tkinter_module.StringVar()
        combobox = ttk_module.Combobox(
            selection_row,
            textvariable=self._analysis_posterior_file_var,
            values=[path.name for path in self._analysis_posterior_files],
            state="readonly",
            width=48,
        )
        combobox.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._analysis_posterior_combobox = combobox
        ttk_module.Button(
            selection_row,
            text="Load",
            command=self._analysis_load_selected_posterior,
            takefocus=True,
            **self._monitor_button_kwargs(),
        ).pack(side="left", padx=(4, 0))

        self._analysis_posterior_status_label = ttk_module.Label(
            container,
            text=(
                "Find a run directory and load one of its posterior "
                "snapshots."
            ),
            wraplength=720,
            takefocus=True,
        )
        self._analysis_posterior_status_label.pack(anchor="w", pady=(4, 6))

        viewer_holder = ttk_module.Frame(container)
        viewer_holder.pack(fill="both", expand=True)
        viewer_holder.columnconfigure(0, weight=1)
        viewer_holder.rowconfigure(0, weight=1)
        try:
            viewer = PlotViewer(viewer_holder)
        except RuntimeError as exc:  # pragma: no cover - unlikely in Tk env
            ttk_module.Label(
                viewer_holder,
                text=f"Plot viewer unavailable: {exc}",
                wraplength=720,
            ).grid(row=0, column=0, sticky="nsew")
            self._analysis_plot_viewer = None
        else:
            viewer.grid(row=0, column=0, sticky="nsew")
            self._analysis_plot_viewer = viewer

        controls = ttk_module.Frame(container)
        controls.pack(fill="x", pady=(6, 0))
        ttk_module.Button(
            controls,
            text="Fit all",
            command=self._analysis_fit_posterior_full,
            takefocus=True,
            **self._monitor_button_kwargs(),
        ).pack(side="left")
        ttk_module.Label(
            controls,
            text="Enable zoom mode above to pan the figure by dragging.",
            wraplength=520,
            takefocus=True,
        ).pack(side="left", padx=(12, 0))

        asset_controls = ttk_module.Frame(container)
        asset_controls.pack(fill="x", pady=(4, 0))
        corner_button = ttk_module.Button(
            asset_controls,
            text="Show corner plot",
            command=self._analysis_show_corner_plot,
            takefocus=True,
            **self._monitor_button_kwargs(),
        )
        corner_button.pack(side="left")
        histogram_button = ttk_module.Button(
            asset_controls,
            text="Show histograms",
            command=self._analysis_show_histogram_plot,
            takefocus=True,
            **self._monitor_button_kwargs(),
        )
        histogram_button.pack(side="left", padx=(8, 0))

        self._analysis_current_posterior_path = None
        self._analysis_refresh_plot_assets()
        self._refresh_analysis_posterior_list()

    def _build_analysis_comparison_page(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the Comparisons tab content that diff two runs."""

        if tk_gui is None:
            ttk_module.Label(
                container,
                text="GUI is unavailable in this environment.",
                wraplength=720,
            ).pack(anchor="w")
            return

        if self._analysis_comparison_base_var is None:
            self._analysis_comparison_base_var = tkinter_module.StringVar()
        if self._analysis_comparison_alt_var is None:
            self._analysis_comparison_alt_var = tkinter_module.StringVar()

        def _render_selector(
            label_text: str, selector_var: tkinter_module.StringVar
        ) -> None:
            """Render a labelled entry plus browse button for run paths."""
            row = ttk_module.Frame(container)
            row.pack(fill="x", pady=(0, 4))
            ttk_module.Label(row, text=label_text).pack(side="left")
            entry = ttk_module.Entry(row, textvariable=selector_var, width=60)
            entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
            entry.bind("<Return>", lambda event: self._analysis_compare_runs())
            ttk_module.Button(
                row,
                text="Browse…",
                command=lambda: self._analysis_browse_comparison_path(
                    selector_var
                ),
                takefocus=True,
                **self._monitor_button_kwargs(),
            ).pack(side="left")

        _render_selector(
            "Base run directory:", self._analysis_comparison_base_var
        )
        _render_selector(
            "Alternative run directory:", self._analysis_comparison_alt_var
        )

        self._analysis_comparison_status_label = ttk_module.Label(
            container,
            text="Provide both directories and refresh to see deltas.",
            wraplength=720,
            takefocus=True,
        )
        self._analysis_comparison_status_label.pack(anchor="w", pady=(4, 6))

        panel = ttk_module.Frame(container)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        text_widget = tkinter_module.Text(
            panel,
            wrap="none",
            padx=8,
            pady=6,
            borderwidth=0,
            highlightthickness=0,
            height=16,
        )
        text_widget.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk_module.Scrollbar(
            panel,
            orient="vertical",
            command=text_widget.yview,
        )
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll = ttk_module.Scrollbar(
            panel,
            orient="horizontal",
            command=text_widget.xview,
        )
        hscroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(
            yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
        )
        text_widget.configure(state="disabled")
        self._analysis_comparison_text_widget = text_widget

        ttk_module.Label(
            container,
            text=(
                "Comparisons highlight duration deltas, dataset row counts "
                "and χ²/parameter shifts between two runs."
            ),
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self._analysis_comparison_result = None
        self._analysis_comparison_pair = None
        self._analysis_set_comparison_status(
            "Configure the runs above and press 'Refresh diff'."
        )

    def _analysis_path_from_var(
        self, path_var: tkinter_module.StringVar | None
    ) -> Path | None:
        """Resolve the selected run directory from a StringVar."""

        if path_var:
            raw = path_var.get().strip()
            if raw:
                return Path(raw).expanduser()
        return None

    def _analysis_browse_comparison_path(
        self, path_var: tkinter_module.StringVar
    ) -> None:
        """Open a directory chooser and store the result in ``var``."""

        if filedialog_module is None:
            self.create_toast(
                "File dialogs are unavailable in this environment.",
                severity="ERROR",
                context="analysis",
            )
            return
        directory = filedialog_module.askdirectory(
            initialdir=self._output_root()
        )
        if directory:
            path_var.set(directory)

    def _analysis_set_comparison_status(
        self, message: str, *, severity: str = "INFO"
    ) -> None:
        """Update the status label for the Comparisons tab."""

        if self._analysis_comparison_status_label is not None:
            self._analysis_comparison_status_label.configure(text=message)
        if severity.upper() == "ERROR":
            logger.get_logger().warning("Analysis comparison: %s", message)

    def _analysis_compare_runs(self) -> None:
        """Analyse the two selected runs and populate the comparison pane."""

        base_path = self._analysis_path_from_var(
            self._analysis_comparison_base_var
        )
        alt_path = self._analysis_path_from_var(
            self._analysis_comparison_alt_var
        )
        if not base_path or not alt_path:
            self._analysis_set_comparison_status(
                "Select both run directories before comparing.",
                severity="ERROR",
            )
            return
        if not base_path.is_dir() or not alt_path.is_dir():
            self._analysis_set_comparison_status(
                "One of the specified paths is not a directory.",
                severity="ERROR",
            )
            return
        try:
            base_result = analysis.analyze_run(base_path)
            alt_result = analysis.analyze_run(alt_path)
            comparison = analysis.compare_runs(base_result, alt_result)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Comparison failed: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._analysis_set_comparison_status(
                "Comparison failed; check the logs.",
                severity="ERROR",
            )
            return
        self._analysis_comparison_result = comparison
        self._analysis_comparison_pair = (base_result, alt_result)
        if self._analysis_comparison_text_widget is not None:
            payload = json.dumps(comparison, indent=2)
            comparison_widget = self._analysis_comparison_text_widget
            comparison_widget.configure(state="normal")
            comparison_widget.delete("1.0", tkinter_module.END)
            comparison_widget.insert("1.0", payload)
            comparison_widget.configure(state="disabled")
        self._analysis_set_comparison_status("Comparison ready.")

    def _export_analysis_comparison(self) -> None:
        """Export the cached comparison payload to a directory."""

        if not self._analysis_comparison_pair:
            self._analysis_set_comparison_status(
                "Run a comparison before exporting.", severity="ERROR"
            )
            return
        if filedialog is None:
            self.create_toast(
                "File dialogs are unavailable in this environment.",
                severity="ERROR",
                context="analysis",
            )
            return
        target_dir = filedialog.askdirectory(
            initialdir=self._output_root(),
        )
        if not target_dir:
            return
        base_result, alt_result = self._analysis_comparison_pair
        try:
            saved = analysis.save_comparison_summary(
                base_result, alt_result, target_dir
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Export failed: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._analysis_set_comparison_status(
                "Export failed.", severity="ERROR"
            )
            return
        self.create_toast(
            "Analysis comparison exported.",
            severity="INFO",
            context="analysis",
        )
        self._analysis_set_comparison_status(
            "Comparison exported to "
            + ", ".join(str(path) for path in saved.values())
        )

    def _copy_analysis_comparison_to_clipboard(self) -> None:
        """Copy the JSON comparison payload into the clipboard."""

        if not self._analysis_comparison_result or tk_gui is None:
            self._analysis_set_comparison_status(
                "Run a comparison before copying.", severity="ERROR"
            )
            return
        payload = json.dumps(self._analysis_comparison_result, indent=2)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(payload)
        except (RuntimeError, ValueError, tk_tcl_error) as exc:
            self.create_toast(
                f"Clipboard copy failed: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._analysis_set_comparison_status("Clipboard copy failed.")
            return
        self._analysis_set_comparison_status("Comparison copied to clipboard.")

    def _build_analysis_placeholder_page(
        self, container: tkinter_module.Frame, label: str
    ) -> None:
        """Render a placeholder tab that is pending implementation."""

        ttk_module.Label(
            container,
            text=(
                f"{label} tools are coming soon. "
                "Check the run summary first."
            ),
            wraplength=720,
            justify="left",
        ).pack(anchor="w")
        ttk_module.Label(
            container,
            text="Use the numbered buttons above once those tabs are live.",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

    def _analysis_run_dir(self) -> Path | None:
        """Return the currently targeted run directory for analysis helpers."""

        if self._analysis_run_path_var:
            raw = self._analysis_run_path_var.get().strip()
            if raw:
                return Path(raw).expanduser()
        if self._analysis_summary_result:
            return self._analysis_summary_result.run_dir
        return None

    def _analysis_set_posterior_status(
        self, message: str, *, severity: str = "INFO"
    ) -> None:
        """Update the status label for the Posterior tab."""

        if self._analysis_posterior_status_label is not None:
            self._analysis_posterior_status_label.configure(text=message)
        if severity.upper() == "ERROR":
            logger.get_logger().warning("Analysis posteriors: %s", message)

    def _analysis_refresh_plot_assets(self) -> None:
        """Refresh references to saved corner/histogram PNGs for the run."""

        run_dir = self._analysis_run_dir()
        self._analysis_corner_files = []
        self._analysis_histogram_files = []
        if not run_dir or not run_dir.is_dir():
            return

        try:
            self._analysis_corner_files = sorted(
                run_dir.glob("corner-plot-*.png"),
                key=lambda entry: entry.stat().st_mtime,
            )
            self._analysis_histogram_files = sorted(
                run_dir.glob("parameter-histograms-*.png"),
                key=lambda entry: entry.stat().st_mtime,
            )
        except OSError:
            pass

    def _analysis_latest_plot(self, files: list[Path]) -> Path | None:
        """Return the newest file from the supplied plot list."""
        if not files:
            return None
        try:
            return max(files, key=lambda entry: entry.stat().st_mtime)
        except OSError:
            return files[-1]

    def _analysis_load_image(self, path: Path, title: str) -> None:
        """Load *path* into the plot viewer with the supplied *title*."""
        viewer = self._analysis_plot_viewer
        if viewer is None:
            self._analysis_set_posterior_status(
                "Plot viewer unavailable.", severity="ERROR"
            )
            return
        try:
            image = mpimage.imread(str(path))
        except (OSError, RuntimeError, ValueError) as exc:
            self._analysis_set_posterior_status(
                f"Failed to load {path.name}: {exc}", severity="ERROR"
            )
            return
        figure = Figure(figsize=(8, 6))
        plot_axes = figure.subplots()
        plot_axes.imshow(image)
        plot_axes.axis("off")
        figure.suptitle(title, fontsize=14)
        viewer.load_figure(figure)
        viewer.fit_to_screen()
        self._analysis_current_posterior_path = path
        self._analysis_set_posterior_status(f"Showing {path.name} ({title}).")

    def _analysis_show_corner_plot(self) -> None:
        """Display the latest corner plot for the selected run."""
        self._analysis_refresh_plot_assets()
        path = self._analysis_latest_plot(self._analysis_corner_files)
        if path is None:
            self._analysis_set_posterior_status(
                "No saved corner plot found for that run.", severity="ERROR"
            )
            return
        self._analysis_load_image(path, "Corner plot")

    def _analysis_show_histogram_plot(self) -> None:
        """Display the latest histogram plot for the selected run."""
        self._analysis_refresh_plot_assets()
        path = self._analysis_latest_plot(self._analysis_histogram_files)
        if path is None:
            self._analysis_set_posterior_status(
                "No saved histogram plot found for that run.", severity="ERROR"
            )
            return
        self._analysis_load_image(path, "Parameter histograms")

    def _analysis_summary_for_run(
        self, run_dir: Path
    ) -> analysis.RunAnalysisResult | None:
        """Return a cached summary for ``run_dir`` or recreate it."""

        current = self._analysis_summary_result
        if current and current.run_dir == run_dir:
            return current
        try:
            result = analysis.analyze_run(run_dir)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Failed to analyse run for posteriors: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._analysis_set_posterior_status(
                "Failed to analyse run directory; check the logs.",
                severity="ERROR",
            )
            return None
        self._analysis_summary_result = result
        if self._analysis_run_path_var is not None:
            self._analysis_run_path_var.set(str(run_dir))
        self._analysis_refresh_plot_assets()
        return result

    def _refresh_analysis_posterior_list(self) -> None:
        """Refresh the list of posterior NetCDF files for the current run."""

        run_dir = self._analysis_run_dir()
        if not run_dir:
            self._analysis_posterior_files = []
            if self._analysis_posterior_combobox is not None:
                self._analysis_posterior_combobox["values"] = []
            self._analysis_set_posterior_status(
                "Provide a run directory to discover posterior files.",
                severity="ERROR",
            )
            return
        if not run_dir.is_dir():
            self._analysis_set_posterior_status(
                "Selected run path is not a directory.", severity="ERROR"
            )
            return
        files = posterior_explorer.find_posterior_files(run_dir)
        self._analysis_posterior_files = files
        names = [path.name for path in files]
        if self._analysis_posterior_combobox is not None:
            self._analysis_posterior_combobox["values"] = names
        if self._analysis_posterior_file_var is not None:
            self._analysis_posterior_file_var.set(names[0] if names else "")
        self._analysis_set_posterior_status(
            f"{len(names)} posterior file(s) available."
        )

    def _analysis_load_selected_posterior(self) -> None:
        """Load the chosen posterior file and render it inside the viewer."""

        run_dir = self._analysis_run_dir()
        if not run_dir:
            self._analysis_set_posterior_status(
                "Select a run directory before loading posteriors.",
                severity="ERROR",
            )
            return
        if not run_dir.is_dir():
            self._analysis_set_posterior_status(
                "The configured run path does not exist.", severity="ERROR"
            )
            return
        if not self._analysis_posterior_files:
            self._refresh_analysis_posterior_list()
        if not self._analysis_posterior_files:
            self._analysis_set_posterior_status(
                "No posterior files were found for that run.", severity="ERROR"
            )
            return
        selection = (
            self._analysis_posterior_file_var.get()
            if self._analysis_posterior_file_var
            else ""
        )
        target = None
        for candidate in self._analysis_posterior_files:
            if candidate.name == selection:
                target = candidate
                break
        if target is None:
            target = self._analysis_posterior_files[0]
        result = self._analysis_summary_for_run(run_dir)
        if result is None:
            return
        try:
            figure = posterior_explorer.create_posterior_overview_figure(
                result, target
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._analysis_set_posterior_status(
                f"Failed to render {target.name}: {exc}", severity="ERROR"
            )
            return
        viewer = self._analysis_plot_viewer
        if viewer is None:
            self._analysis_set_posterior_status(
                "Plot viewer is unavailable.", severity="ERROR"
            )
            return
        viewer.load_figure(figure)
        viewer.fit_to_screen()
        self._analysis_current_posterior_path = target
        self._analysis_set_posterior_status(f"Showing {target.name}.")
        self._analysis_refresh_plot_assets()

    def _analysis_fit_posterior_to_screen(self) -> None:
        """Autoscale the current figure in the viewer."""

        viewer = self._analysis_plot_viewer
        if viewer is None:
            self._analysis_set_posterior_status(
                "Load a posterior before fitting the view.",
                severity="ERROR",
            )
            return
        viewer.fit_to_screen()
        self._analysis_set_posterior_status("Figure fitted to screen.")

    def _analysis_fit_posterior_full(self) -> None:
        """Restore the original figure limits captured at draw time."""

        viewer = self._analysis_plot_viewer
        if viewer is None:
            self._analysis_set_posterior_status(
                "Load a posterior before restoring the view.",
                severity="ERROR",
            )
            return
        viewer.fit_all()
        self._analysis_set_posterior_status(
            "View restored to original limits."
        )

    def _analysis_toggle_posterior_zoom(self) -> None:
        """Toggle pan/zoom mode for the viewer."""

        viewer = self._analysis_plot_viewer
        if viewer is None:
            self._analysis_set_posterior_status(
                "Load a posterior before enabling zoom.", severity="ERROR"
            )
            return
        viewer.toggle_zoom()
        state = "enabled" if viewer.zoom_active else "disabled"
        self._analysis_set_posterior_status(f"Zoom mode {state}.")

    def _browse_analysis_run_dir(self) -> None:
        """Ask the user to pick a run output folder."""

        if filedialog is None:
            self.create_toast(
                "File dialogs are unavailable in this environment.",
                severity="ERROR",
                context="analysis",
            )
            return
        directory = filedialog.askdirectory(initialdir=self._output_root())
        if directory:
            if self._analysis_run_path_var is None:
                self._analysis_run_path_var = tkinter_module.StringVar(
                    value=directory
                )
            else:
                self._analysis_run_path_var.set(directory)

    def _load_analysis_summary(self) -> None:
        """
        Load the selected run directory, analyse it and render the results.
        """

        if tk_gui is None:
            return
        run_path = (
            (self._analysis_run_path_var.get().strip())
            if self._analysis_run_path_var
            else ""
        )
        if not run_path:
            self._set_analysis_status(
                "Please choose a run directory first.", severity="ERROR"
            )
            return
        path = Path(run_path).expanduser()
        if not path.is_dir():
            self._set_analysis_status(
                "Selected path does not exist or is not a directory.",
                severity="ERROR",
            )
            return
        try:
            result = analysis.analyze_run(path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Failed to load run summary: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._set_analysis_status(
                "Summary load failed; check the logs for details.",
                severity="ERROR",
            )
            return
        self._analysis_summary_result = result
        formatted = analysis.format_run_summary_text(result)
        if self._analysis_summary_text_widget is not None:
            summary_widget = self._analysis_summary_text_widget
            summary_widget.configure(state="normal")
            summary_widget.delete("1.0", tkinter_module.END)
            summary_widget.insert("1.0", formatted)
            summary_widget.configure(state="disabled")
        if self._analysis_run_path_var is not None:
            self._analysis_run_path_var.set(str(result.run_dir))
        self._set_analysis_status("Run summary loaded successfully.")

    def _export_analysis_summary(self) -> None:
        """Export the currently loaded summary to disk."""

        if self._analysis_summary_result is None:
            self._set_analysis_status(
                "Load a run summary before exporting.",
                severity="ERROR",
            )
            return
        if filedialog is None:
            self.create_toast(
                "File dialogs are unavailable in this environment.",
                severity="ERROR",
                context="analysis",
            )
            return
        target_dir = filedialog.askdirectory(initialdir=self._output_root())
        if not target_dir:
            return
        try:
            saved = analysis.save_run_summary(
                self._analysis_summary_result.run_dir, target_dir
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Export failed: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._set_analysis_status("Export failed.")
            return
        self.create_toast(
            "Analysis summary exported.",
            severity="INFO",
            context="analysis",
        )
        self._set_analysis_status(
            "Summary exported to "
            + ", ".join(str(path) for path in saved.values())
        )

    def _copy_analysis_summary_to_clipboard(self) -> None:
        """Copy the JSON summary to the clipboard."""

        if self._analysis_summary_result is None or tk_gui is None:
            self._set_analysis_status(
                "Load a run summary before copying.", severity="ERROR"
            )
            return
        payload = json.dumps(self._analysis_summary_result.to_dict(), indent=2)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(payload)
        except (RuntimeError, ValueError, tk_tcl_error) as exc:
            self.create_toast(
                f"Clipboard copy failed: {exc}",
                severity="ERROR",
                context="analysis",
            )
            self._set_analysis_status("Clipboard copy failed.")
            return
        self._set_analysis_status("Summary copied to clipboard.")

    def _set_analysis_status(
        self, message: str, *, severity: str = "INFO"
    ) -> None:
        """Update the per-tab status label."""

        if self._analysis_summary_status_label is not None:
            self._analysis_summary_status_label.configure(text=message)
        if severity.upper() == "ERROR":
            logger.get_logger().warning("Analysis tab: %s", message)

    def _start_validation_run(self) -> None:
        """Kick off the validation suite inside a background thread."""

        if self.status in (RunStatus.RUNNING, RunStatus.CONFIGURING):
            self._notify_pipeline_conflict(
                "Production run active",
                "The pipeline is busy with a production run. Please wait "
                "for it to finish or cancel it so a validation run can be "
                "started.",
                context="validation",
            )
            return
        if self._validation_running:
            return
        self._validation_running = True
        if self._validation_button:
            self._validation_button.configure(state=tkinter_module.DISABLED)
        self._validation_status_base = "Status: running validation…"
        self._update_validation_status_label(self._progress_snapshot)
        self._prepare_validation_progress_monitor()
        self._refresh_validation_progress_widgets()
        self._reset_validation_log_widget(
            "Validation log streaming; progress updates appear below."
        )
        self._update_validation_control_states()
        console_output.write("GUI: validation suite queued for execution.")
        self._launch_validation_worker()

    def _notify_pipeline_conflict(
        self, title: str, message: str, *, context: str
    ) -> None:
        """Warn when a concurrent pipeline operation blocks the request."""

        if self.render and messagebox:
            messagebox.showwarning(title, message, parent=self.root)
        self.create_toast(message, severity="WARNING", context=context)

    def _launch_validation_worker(self) -> None:
        """Start the CLI helper that runs the validation suite."""

        if self._validation_process is not None:
            return
        command = [
            sys.executable,
            "-m",
            "copernican",
            "--run-validation",
        ]
        env = os.environ.copy()
        env.setdefault("COPERNICAN_HEADLESS_RUN", "1")
        if self._progress_state_path:
            env["COPERNICAN_GUI_PROGRESS_PATH"] = self._progress_state_path
        else:
            env.pop("COPERNICAN_GUI_PROGRESS_PATH", None)
        try:
            process = subprocess.Popen(  # nosec
                command,
                cwd=str(self._repo_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
            self.create_toast(
                f"Failed to launch validation: {exc}",
                severity="ERROR",
                context="validation",
            )
            self._validation_running = False
            self._cleanup_validation_progress_monitor()
            self._update_validation_control_states()
            return
        self._validation_process = process
        self._validation_stdout_thread = threading.Thread(
            target=self._stream_validation_output,
            args=(process,),
            daemon=True,
        )
        self._validation_stdout_thread.start()
        threading.Thread(
            target=self._monitor_validation_process,
            args=(process,),
            daemon=True,
        ).start()

    def _stream_validation_output(
        self, process: subprocess.Popen[str]
    ) -> None:
        """Append the worker stdout lines to the validation log."""

        if process.stdout is None:
            return
        for line in process.stdout:
            cleaned = line.rstrip()
            if cleaned:
                self._call_on_ui_thread(
                    self._append_validation_log_line, cleaned
                )

    def _monitor_validation_process(
        self, process: subprocess.Popen[str]
    ) -> None:
        """Watch the validation worker and react when it exits."""

        return_code = process.wait()
        summary = validation_utils.read_validation_summary()
        if self.content_area is None:
            self._complete_validation_run(return_code, summary)
            return
        self.content_area.after(
            0, lambda: self._complete_validation_run(return_code, summary)
        )

    def _complete_validation_run(self, code: int, summary: str) -> None:
        """Update the validation view after the worker finishes."""

        self._validation_running = False
        status = "passed" if code == 0 else "failed"
        self._validation_status_base = f"Status: validation {status}"
        self._update_validation_status_label(self._progress_snapshot)
        self._append_validation_summary_text(
            summary or "Validation completed without producing summary text."
        )
        console_output.write(
            f"GUI validation: completed with code {code} and summary length "
            f"{len(summary)}."
        )
        self._cleanup_validation_progress_monitor()
        self._validation_process = None
        self._validation_stdout_thread = None
        self._update_validation_control_states()

    def _append_validation_summary_text(self, summary: str) -> None:
        """Append the validation summary text to the log widget."""

        for line in summary.splitlines():
            self._append_validation_log_line(line)

    def _append_validation_log_line(self, line: str) -> None:
        """Append a single log *line* to the validation text widget."""

        self._validation_log_lines.append(line)
        if not self._validation_text_widget:
            return
        lock_tail = self._validation_log_lock_var is None or bool(
            self._validation_log_lock_var.get()
        )
        self._validation_text_widget.configure(state="normal")
        self._validation_text_widget.insert("end", f"{line}\n")
        self._validation_text_widget.configure(state="disabled")
        if lock_tail:
            try:
                self._validation_text_widget.yview_moveto(1.0)
            except (RuntimeError, ValueError, tk_tcl_error):
                pass

    def _reset_validation_log_widget(self, message: str | None = None) -> None:
        """Clear the log widget so new entries can stream in."""

        if not self._validation_text_widget:
            return
        self._validation_log_lines.clear()
        self._validation_text_widget.configure(state="normal")
        self._validation_text_widget.delete("1.0", tkinter_module.END)
        if message:
            self._append_validation_log_line(message)
        self._validation_text_widget.configure(state="disabled")
        if (
            self._validation_log_lock_var
            and self._validation_log_lock_var.get()
        ):
            try:
                self._validation_text_widget.yview_moveto(1.0)
            except (RuntimeError, ValueError, tk_tcl_error):
                pass

    def _refresh_validation_log_widget_from_history(self) -> None:
        """Repopulate the log widget when the UI is rebuilt."""

        if not self._validation_text_widget:
            return
        self._validation_text_widget.configure(state="normal")
        self._validation_text_widget.delete("1.0", tkinter_module.END)
        for line in self._validation_log_lines:
            self._validation_text_widget.insert("end", f"{line}\n")
        self._validation_text_widget.configure(state="disabled")
        if (
            self._validation_log_lock_var
            and self._validation_log_lock_var.get()
        ):
            try:
                self._validation_text_widget.yview_moveto(1.0)
            except (RuntimeError, ValueError, tk_tcl_error):
                pass

    def _update_validation_control_states(self) -> None:
        """Enable/disable buttons depending on the validation state."""

        if self._validation_button:
            self._validation_button.configure(
                state=(
                    tkinter_module.DISABLED
                    if self._validation_running
                    else tkinter_module.NORMAL
                )
            )
        if self._validation_cancel_button:
            self._validation_cancel_button.configure(
                state=(
                    tkinter_module.NORMAL
                    if self._validation_running
                    else tkinter_module.DISABLED
                )
            )
        if self._validation_clear_button:
            self._validation_clear_button.configure(
                state=(
                    tkinter_module.DISABLED
                    if self._validation_running
                    else tkinter_module.NORMAL
                )
            )

    def _cancel_validation_run(self) -> None:
        """Signal the running validation worker to stop early."""

        if not self._validation_running or self._validation_process is None:
            return
        try:
            self._validation_process.terminate()
        except (OSError, RuntimeError, ValueError) as exc:
            self.create_toast(
                f"Failed to cancel validation: {exc}",
                severity="WARNING",
                context="validation",
            )
            return
        self.create_toast(
            "Validation cancellation requested.",
            severity="INFO",
            context="validation",
        )

    def _clear_validation_results(self) -> None:
        """Remove every generated validation artifact and reset the view."""

        if self._validation_running:
            self.create_toast(
                "Stop the validation run before clearing outputs.",
                severity="WARNING",
                context="validation",
            )
            return
        output_dir = self._repo_root() / "validation" / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if validation_utils.VALIDATION_FILE.exists():
            validation_utils.VALIDATION_FILE.unlink(missing_ok=True)
        self._reset_validation_log_widget(
            "Validation outputs cleared. Run the suite to regenerate the log."
        )
        self._validation_status_base = "Status: idle"
        self._update_validation_status_label(self._progress_snapshot)
        self.create_toast(
            "Validation reports and outputs removed.",
            severity="INFO",
            context="validation",
        )
        self._update_validation_control_states()

        self._refresh_environment_status()

    def _prepare_validation_progress_monitor(self) -> None:
        """Configure progress state tracking for the validation runner."""

        if self._progress_state_path:
            return
        progress_path = self._prepare_progress_path()
        self._progress_state_path = progress_path
        progress_state.clear_progress(progress_path)
        self._validation_progress_env_backup = os.environ.get(
            "COPERNICAN_GUI_PROGRESS_PATH"
        )
        os.environ["COPERNICAN_GUI_PROGRESS_PATH"] = progress_path
        self._start_progress_poller()
        self._refresh_validation_progress_widgets()

    def _cleanup_validation_progress_monitor(self) -> None:
        """Stop monitoring and remove the temporary progress state."""

        self._stop_progress_poller()
        if self._progress_state_path:
            progress_state.clear_progress(self._progress_state_path)
            self._progress_state_path = None
        if self._validation_progress_env_backup is not None:
            os.environ["COPERNICAN_GUI_PROGRESS_PATH"] = (
                self._validation_progress_env_backup
            )
            self._validation_progress_env_backup = None
        else:
            os.environ.pop("COPERNICAN_GUI_PROGRESS_PATH", None)
        self._progress_snapshot = None
        self._validation_last_stage_label = None

    def _handle_home_revalidate(self, dataset_id: str) -> None:
        """Revalidate an untrusted dataset from the Home dashboard."""

        try:
            record = self.revalidate_dataset(dataset_id)
        except KeyError:
            self.create_toast(
                f"{dataset_id} is not present in the catalogue.",
                severity="ERROR",
                context="data",
            )
            self.show_home()
            return
        if record.get("parser_trusted"):
            severity = "INFO"
            message = f"{dataset_id} passed parser validation."
        else:
            severity = "WARNING"
            message = (
                f"{dataset_id} is still untrusted; verify parser digests."
            )
        self.create_toast(message, severity=severity, context="data")
        self.show_home()

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
            or self.draft.dataset.strip()
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
        self.selected_engine_kind = "mcmc"
        self._selected_model_entry = None
        self._selected_engine_entry = None
        self.engine_capabilities = None
        if self._engine_run_settings_frame is not None:
            self._engine_run_settings_frame.destroy()
        self._engine_setting_vars.clear()
        self._engine_setting_specs.clear()
        self._engine_run_settings_frame = None
        self._current_engine_module = None
        self.selected_datasets = []
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
            confirm_index = self.builder_steps.index(self._CONFIRM_STEP_NAME)
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
        initial_dir = self._output_root()
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
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:
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

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the Run Builder navigation controls and status."""
            current_step = self.builder_steps[self.current_step_index]
            self._page_header(frame, f"Run builder: {current_step}")
            step_frame = ttk_module.Frame(frame)
            step_frame.pack(fill="x", pady=(0, 12))
            self._builder_step_buttons = []
            jump_button_style = self._monitor_button_kwargs()
            for index, step in enumerate(self.builder_steps):
                indicator = ttk_module.Button(
                    step_frame,
                    text=step,
                    command=lambda idx=index: self.jump_to_step(idx),
                    takefocus=True,
                    **jump_button_style,
                )
                indicator.pack(side="left", padx=4)
                self._builder_step_buttons.append(indicator)
            status_message = self._builder_status_message()
            if status_message:
                body = ttk_module.Label(
                    frame,
                    text=status_message,
                    wraplength=720,
                    justify="left",
                    takefocus=True,
                )
                body.pack(anchor="w", pady=(8, 8))

            controls = ttk_module.Frame(frame)
            controls.pack(anchor="w")
            nav_button_style = self._monitor_button_kwargs()
            ttk_module.Button(
                controls,
                text="Previous",
                command=self.previous_step,
                state=(
                    tkinter_module.DISABLED
                    if self.current_step_index == 0
                    else tkinter_module.NORMAL
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
            ttk_module.Button(
                controls,
                text="Next",
                command=self._handle_builder_next,
                state=(
                    tkinter_module.DISABLED
                    if next_disabled
                    else tkinter_module.NORMAL
                ),
                takefocus=True,
                **nav_button_style,
            ).pack(side="left", padx=4)
            ttk_module.Button(
                controls,
                text="Cancel",
                command=self.cancel_builder,
                state=(
                    tkinter_module.NORMAL
                    if self._has_configuration()
                    else tkinter_module.DISABLED
                ),
                takefocus=True,
                **nav_button_style,
            ).pack(side="left", padx=4)
            content_container = ttk_module.Frame(frame)
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
        confirm_index = self.builder_steps.index(self._CONFIRM_STEP_NAME)
        for index, button in enumerate(self._builder_step_buttons):
            if index == manifest_index:
                desired_state = (
                    tkinter_module.NORMAL
                    if self._builder_ready()
                    else tkinter_module.DISABLED
                )
            elif index == confirm_index:
                desired_state = (
                    tkinter_module.NORMAL
                    if self._can_enter_confirm()
                    else tkinter_module.DISABLED
                )
            else:
                desired_state = tkinter_module.NORMAL
            if desired_state == tkinter_module.NORMAL:
                button.state(["!disabled"])
            else:
                button.state(["disabled"])

    def _build_run_builder_step(self, container: tkinter_module.Frame) -> None:
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
            entry = self._model_entry_for_value(model_id)
            if entry:
                counts.append(entry.get("parameter_count", 0))
                continue
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
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
        ):
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
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return default_steps, default_walkers, default_pool

        def _default_value(name: str, fallback: int) -> int:
            """Return the default parameter value declared by the engine."""
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

    def _render_builder_step_seed(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the Seed step UI, including seed entry and buttons."""
        ttk_module.Frame(container, height=30).pack(fill="x", pady=(0, 6))
        ttk_module.Label(
            container,
            text=(
                "Choose a deterministic seed, use the timestamp generator, or "
                "play one of the mini-games to forge a reproducible value."
            ),
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        seed_var = tkinter_module.StringVar(value=self.draft.seed)

        def _update_seed(*_args: object) -> None:
            """Mirror entry text to the draft seed as it changes."""
            self.draft.seed = seed_var.get().strip()
            self._refresh_builder_step_indicators()

        seed_var.trace_add("write", _update_seed)
        entry = ttk_module.Entry(
            container,
            textvariable=seed_var,
            width=24,
        )
        entry.pack(anchor="w", pady=(6, 8))
        button_column = ttk_module.Frame(container)
        button_column.pack(anchor="w", fill="x", pady=(0, 8))

        def _add_seed_button(
            label: str,
            action: Callable[[], None],
        ) -> None:
            """Add a configured button to the temporary seed button column."""
            seed_button = ttk_module.Button(
                button_column,
                text=label,
                command=action,
            )
            seed_button.pack(anchor="w", pady=2, ipadx=8, ipady=4)

        env_seed = os.environ.get("COPERNICAN_SEED")
        self._build_seed_button_column(button_column, seed_var, env_seed)

    def _build_seed_button_column(
        self,
        container: tkinter_module.Frame,
        seed_var: "tkinter_module.StringVar",
        env_seed: str | None,
    ) -> None:
        """Rebuild the mini-game seed buttons when the catalog changes."""
        for child in container.winfo_children():
            child.destroy()

        def _add_seed_button(label: str, action: Callable[[], None]) -> None:
            """Place a seed shortcut button inside the provided container."""
            ttk_module.Button(
                container,
                text=label,
                command=action,
            ).pack(anchor="w", pady=2, ipadx=8, ipady=4)

        _add_seed_button("Default (0)", lambda: seed_var.set("0"))
        _add_seed_button(
            "Random timestamp",
            lambda: seed_var.set(str(int(time.time()))),
        )
        for descriptor in self._minigame_catalog:
            _add_seed_button(
                descriptor.name,
                lambda game_id=descriptor.game_id: self._launch_minigame(
                    game_id, seed_var
                ),
            )
        if env_seed:
            _add_seed_button(
                "Use COPERNICAN_SEED",
                lambda: seed_var.set(env_seed),
            )
        _add_seed_button(
            "Refresh RNG mini-games",
            lambda: self._refresh_minigame_catalog(
                container, seed_var, env_seed
            ),
        )

    def _refresh_minigame_catalog(
        self,
        container: tkinter_module.Frame,
        seed_var: "tkinter_module.StringVar",
        env_seed: str | None,
    ) -> None:
        """Reload the RNG mini-game registry and refresh the buttons."""
        try:
            self._minigame_catalog = rng_minigames.refresh_registry()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Mini-game refresh failed: {exc}",
                severity="ERROR",
                context="seed",
            )
            return
        self.create_toast(
            "Mini-games refreshed from copernican/rng_minigames/",
            severity="INFO",
            context="seed",
        )
        self._build_seed_button_column(container, seed_var, env_seed)

    def _render_builder_step_models(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the Models step UI for selecting and previewing a model."""
        ttk_module.Frame(container, height=30).pack(fill="x", pady=(0, 6))
        ttk_module.Label(
            container,
            text="Pick one YAML-defined model to include in the run.",
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        available = sorted(
            self.model_index.values(), key=lambda entry: entry["id"]
        )
        if not available:
            ttk_module.Label(
                container,
                text="No models available; refresh inventory and try again.",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            return
        list_container = ttk_module.Frame(container)
        list_container.pack(fill="both", expand=True, pady=(8, 4))
        list_frame = ttk_module.Frame(list_container)
        list_frame.pack(side="left", fill="both", expand=True)
        listbox = tkinter_module.Listbox(
            list_frame,
            height=8,
            selectmode="browse",
            exportselection=False,
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk_module.Scrollbar(
            list_frame, orient="vertical", command=listbox.yview
        )
        scrollbar.pack(side="right", fill="y")
        listbox.configure(yscrollcommand=scrollbar.set)
        button_frame = ttk_module.Frame(list_container)
        button_frame.pack(side="left", fill="y", padx=(8, 0), anchor="n")

        def _apply_model_entry(entry: dict) -> None:
            """Persist the selected model entry into the builder draft."""

            self._selected_model_entry = entry
            model_value = (
                entry["path"] if entry.get("is_external") else entry["id"]
            )
            self.selected_models = [model_value]
            self.draft.model = model_value
            if entry.get("is_external"):
                summary_text = f"Loaded model: {entry['filename']}"
            else:
                summary_text = f"Selected model: {entry['id']}"
            summary.config(text=summary_text)
            _refresh_model_preview(entry)
            self._refresh_builder_step_indicators()

        def _view_selected_model() -> None:
            """Preview the currently selected model metadata."""
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
            """Launch the selected model file in the OS default editor."""
            entry = self._selected_model_entry
            if not entry:
                self.create_toast(
                    "Select a model before opening its YAML file.",
                    severity="WARNING",
                    context="models",
                )
                return
            self._open_path_with_system(entry["path"])

        def _load_model() -> None:
            """Open a picker and load a model YAML file."""

            if filedialog is None:
                self.create_toast(
                    "File dialogs are unavailable in this environment.",
                    severity="ERROR",
                    context="models",
                )
                return
            path = filedialog.askopenfilename(
                title="Load model...",
                initialdir=str(Path.home()),
                filetypes=[
                    ("YAML files", "*.yml *.yaml"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            entry = self._load_model_from_path(path)
            if entry is None:
                return
            self.create_toast(
                f"Loaded model {entry['filename']}.",
                severity="INFO",
                context="models",
            )
            self.show_run_builder()

        ttk_module.Button(
            button_frame,
            text="View model",
            command=_view_selected_model,
        ).pack(fill="x", pady=(0, 4))
        ttk_module.Button(
            button_frame,
            text="Load model...",
            command=_load_model,
        ).pack(fill="x", pady=(0, 4))
        ttk_module.Button(
            button_frame,
            text="Open model file...",
            command=_open_selected_model_file,
        ).pack(fill="x")
        summary = ttk_module.Label(
            container,
            text="",
            wraplength=720,
            takefocus=True,
        )
        summary.pack(anchor="w", pady=(4, 4))

        for index, model in enumerate(available):
            listbox.insert("end", f"{model['id']} ({model['filename']})")
            selected_model = (
                self.selected_models[0] if self.selected_models else None
            )
            selected_entry = self._model_entry_for_value(selected_model)
            if selected_entry and model["path"] == selected_entry.get("path"):
                listbox.select_set(index)

        preview_frame = ttk_module.LabelFrame(
            container,
            text="Model preview",
            padding=(8, 6),
        )
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        preview_frame.columnconfigure(0, weight=3)
        preview_frame.columnconfigure(1, weight=2)
        preview_frame.rowconfigure(0, weight=1)

        preview_panel = ttk_module.Frame(preview_frame)
        preview_panel.grid(row=0, column=0, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)
        preview_text = tkinter_module.Text(
            preview_panel,
            wrap="none",
            borderwidth=1,
            relief="solid",
            height=18,
        )
        preview_text.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk_module.Scrollbar(
            preview_panel, orient="vertical", command=preview_text.yview
        )
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll = ttk_module.Scrollbar(
            preview_panel, orient="horizontal", command=preview_text.xview
        )
        hscroll.grid(row=1, column=0, sticky="ew")
        preview_text.configure(
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )

        action_frame = ttk_module.Frame(container)
        action_frame.pack(fill="x", pady=(4, 4))
        ttk_module.Button(
            action_frame,
            text="Equations & expressions…",
            command=self._show_equations_window,
        ).pack(side="right")

        def _refresh_model_preview(entry: dict | None = None) -> None:
            """Update the model YAML preview and equation panel."""
            preview_text.configure(state="normal")
            preview_text.delete("1.0", "end")
            if entry:
                try:
                    content = self._read_asset_text(entry["path"])
                except (
                    OSError,
                    UnicodeError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    content = f"Unable to load {entry['id']}: {exc}"
                preview_text.insert("1.0", content)
            else:
                preview_text.insert(
                    "1.0",
                    "Select a model to preview its YAML definition.",
                )
            preview_text.configure(state="disabled")
            preview_text.yview_moveto(0)
            self._refresh_equation_panel(entry)

        def _refresh_model_selection(
            _: tkinter_module.Event | None = None,
        ) -> None:
            """React to model list changes and refresh selection state."""
            indices = listbox.curselection()
            if indices:
                entry = available[indices[0]]
                _apply_model_entry(entry)
            else:
                current = self._selected_model_entry
                if current and current.get("is_external"):
                    summary.config(text=f"Loaded model: {current['filename']}")
                    _refresh_model_preview(current)
                else:
                    self.selected_models = []
                    self.draft.model = ""
                    self._selected_model_entry = None
                    summary.config(text="No model selected yet.")
                    _refresh_model_preview(None)
            self._refresh_builder_step_indicators()

        listbox.bind("<<ListboxSelect>>", _refresh_model_selection)
        _refresh_model_selection()

    def _show_equations_window(self) -> None:
        """Open a standalone window that displays the current model's
        equations."""

        if not self.render or self.root is None:
            return
        entry = self._selected_model_entry
        if not entry:
            self.create_toast(
                "Select a model before viewing its equations.",
                severity="WARNING",
                context="models",
            )
            return
        if self._equations_window is not None:
            self._equations_window.deiconify()
            self._equations_window.lift()
            self._refresh_equation_panel(entry)
            return
        window = tkinter_module.Toplevel(self.root)
        window.title(f"Equations & expressions — {entry['id']}")
        window.geometry("460x400")
        window.minsize(360, 260)
        window.transient(self.root)
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        body = ttk_module.Frame(window, padding=(12, 12, 12, 12))
        body.grid(row=0, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        if HtmlFrame is not None:
            html = HtmlFrame(
                body,
                horizontal_scrollbar=False,
                vertical_scrollbar="auto",
                javascript_enabled=True,
            )
            html.grid(row=0, column=0, sticky="nsew")
            self._equation_html_frame = html
            self._equations_text_widget = None
        else:
            text_widget = tkinter_module.Text(
                body,
                wrap="word",
                borderwidth=1,
                relief="solid",
            )
            text_widget.grid(row=0, column=0, sticky="nsew")
            text_widget.configure(state="disabled")
            self._equations_text_widget = text_widget
            self._equation_html_frame = None
        window.protocol(
            "WM_DELETE_WINDOW",
            lambda: self._close_equations_window(window),
        )
        self._equations_window = window
        self._refresh_equation_panel(entry)

    def _close_equations_window(self, window: tkinter_module.Toplevel) -> None:
        """Cleanup references when the equations window closes."""

        if self._equations_window is window:
            self._equations_window = None
        self._equation_html_frame = None
        self._equations_text_widget = None
        window.destroy()

    def _collect_model_expressions(
        self, metadata: Mapping[str, Any] | None
    ) -> list[tuple[str, str]]:
        """Return a list of (title, LaTeX) tuples for the equation panel."""

        expressions: list[tuple[str, str]] = []

        def _append(title: str, expression_value: Any | None) -> None:
            """Add a non-empty expression entry to the list."""
            if isinstance(expression_value, str) and expression_value.strip():
                expressions.append((title, expression_value.strip()))

        meta = metadata or {}
        _append("H(z)", meta.get("Hz_expression"))
        _append("Sound horizon", meta.get("rs_expression"))

        equations = meta.get("equations")
        if isinstance(equations, Mapping):
            for section, equation_value in equations.items():
                if isinstance(equation_value, str):
                    _append(section, equation_value)
                elif isinstance(equation_value, Sequence) and not isinstance(
                    equation_value, str
                ):
                    for idx, entry in enumerate(equation_value):
                        _append(f"{section} {idx + 1}", entry)
        return expressions

    def _format_plain_expression_text(self, entry: dict | None) -> str:
        """Return a textual summary of the expressions for fallback views."""

        metadata = entry.get("metadata") if entry else None
        expressions = self._collect_model_expressions(metadata)
        if not expressions:
            return "Select a model to preview its symbolic equations."
        return "\n\n".join(f"{title}:\n{expr}" for title, expr in expressions)

    def _build_equation_html(self, entry: dict | None) -> str:
        """
        Return an HTML document that renders the model's LaTeX expressions.
        """

        if entry is None:
            expressions_html = _EQUATION_EMPTY_BODY
            model_name = "Model preview unloaded"
        else:
            metadata = entry.get("metadata")
            expressions = self._collect_model_expressions(metadata)
            if expressions:
                expressions_html = "".join(
                    (
                        "<div class='expression-block'>"
                        f"<div class='expression-title'>{escape(title)}</div>"
                        f"<div class='equation' data-latex=\"{escape(expr)}\">"
                        f"{escape(expr)}</div>"
                        "</div>"
                    )
                    for title, expr in expressions
                )
            else:
                expressions_html = _EQUATION_EMPTY_BODY
            model_name = (metadata or {}).get("model_name") or entry.get(
                "id", ""
            )

        return _EQUATION_HTML_TEMPLATE.format(
            version=_KATEX_VERSION,
            shim=_EQUATION_WINDOW_SHIM,
            model_name=escape(model_name),
            expressions=expressions_html,
        )

    def _refresh_equation_panel(self, entry: dict | None = None) -> None:
        """Update the KaTeX HTML view based on the selected model."""

        if self._equations_window is not None and entry:
            self._equations_window.title(
                f"Equations & expressions — {entry['id']}"
            )
        if self._equation_html_frame is not None:
            html = self._build_equation_html(entry)
            self._equation_html_frame.load_html(html)
        if self._equations_text_widget is not None:
            body = self._format_plain_expression_text(entry)
            self._equations_text_widget.configure(state="normal")
            self._equations_text_widget.delete("1.0", "end")
            self._equations_text_widget.insert("1.0", body)
            self._equations_text_widget.configure(state="disabled")

    def _render_builder_step_data(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render dataset selection panels grouped by observation type."""
        ttk_module.Frame(container, height=30).pack(fill="x", pady=(0, 6))
        ttk_module.Label(
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
            ttk_module.Label(
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
        catalogue_panel = ttk_module.Frame(container)
        catalogue_panel.pack(fill="x", pady=(6, 0))
        dropdown_widgets: dict[str, ttk_module.Combobox] = {}
        id_lookup: dict[str, tuple[str, int]] = {}

        def _add_dataset_section(
            parent: tkinter_module.Frame,
            dtype: str,
            records: list[dict],
            *,
            width_px: int = 500,
        ) -> None:
            """Insert a dataset selector row for the given observation type."""
            ttk_module.Label(
                parent,
                text=(
                    f"{dtype.upper()} datasets – {len(records)} "
                    f"{'candidate' if len(records) == 1 else 'candidates'}"
                ),
                takefocus=True,
            ).pack(anchor="w", pady=(4, 2))
            combo_frame = ttk_module.Frame(parent, width=width_px)
            combo_frame.pack(anchor="w", pady=(0, 16))
            placeholder = "Select dataset…"
            values = [
                f"{record['id']} – {record.get('name', record['id'])}"
                for record in records
            ]
            combo = ttk_module.Combobox(
                combo_frame,
                values=[placeholder] + values,
                state="readonly",
                width=max(40, width_px // 9),
            )
            combo.current(0)
            combo.pack(fill="both", expand=True)
            dropdown_widgets[dtype] = combo
            for index, record in enumerate(records):
                id_lookup[record["id"]] = (dtype, index)

        for dtype in ordered_types:
            _add_dataset_section(catalogue_panel, dtype, type_groups[dtype])
        detail_label = ttk_module.Label(
            container,
            text="Select datasets to preview details.",
            wraplength=720,
            takefocus=True,
        )
        detail_label.pack(anchor="w", pady=(6, 6))
        selection_map: dict[str, dict] = {}
        focus_state: dict[str, dict | None] = {"record": None}

        def _refresh_data_selection() -> None:
            """Update the builder draft whenever dataset choices change."""
            selection_map.clear()
            selected_records: list[dict] = []
            for dtype in ordered_types:
                combo = dropdown_widgets[dtype]
                index = combo.current()
                if index is None or index <= 0:
                    continue
                record = type_groups[dtype][index - 1]
                selection_map[dtype] = record
                selected_records.append(record)
            self.selected_datasets = [
                self._dataset_manifest_record(record)
                for record in selected_records
            ]
            ids = [record["id"] for record in selected_records]
            self.draft.dataset = ", ".join(ids)
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

        def _make_bind(dtype: str) -> None:
            """Attach selection listeners to each dataset combobox."""
            combo = dropdown_widgets[dtype]

            def _on_select(_event: tkinter_module.Event | None = None) -> None:
                """Refresh the focus record when a selection occurs."""
                index = combo.current()
                focus_state["record"] = (
                    type_groups[dtype][index - 1]
                    if index and index > 0
                    else None
                )
                _refresh_data_selection()

            combo.bind("<<ComboboxSelected>>", _on_select)

        for dtype in ordered_types:
            _make_bind(dtype)

        for dataset in self.selected_datasets:
            lookup = id_lookup.get(dataset["id"])
            if lookup:
                dtype, index = lookup
                combo = dropdown_widgets[dtype]
                combo.current(index + 1)
        _refresh_data_selection()
        action_row = ttk_module.Frame(container)
        action_row.pack(anchor="w")

        def _open_focused_folder() -> None:
            """Open the folder of the currently highlighted dataset."""
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
            """Show metadata for the highlighted dataset."""
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
            """Re-run parser trust checks for the focused dataset."""
            record = focus_state["record"]
            if record:
                self._revalidate_dataset_action(record["id"])
            else:
                self.create_toast(
                    "Choose a dataset to revalidate.",
                    severity="WARNING",
                    context="data",
                )

        ttk_module.Button(
            action_row,
            text="Open folder",
            command=_open_focused_folder,
        ).pack(side="left", padx=2)
        ttk_module.Button(
            action_row,
            text="View metadata",
            command=_view_focused_metadata,
        ).pack(side="left", padx=2)
        ttk_module.Button(
            action_row,
            text="Revalidate parser",
            command=_revalidate_focused_parser,
        ).pack(side="left", padx=2)

    def _render_builder_step_engine(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the engine picker and run settings controls."""
        ttk_module.Frame(container, height=30).pack(fill="x", pady=(0, 6))
        ttk_module.Label(
            container,
            text=("Choose the computational backend to run your models."),
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        options = sorted(
            self.engine_index.values(), key=lambda entry: entry["label"]
        )
        if not options:
            ttk_module.Label(
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
                if (
                    value["id"] == self.selected_engine
                    or value["stem"] == self.selected_engine
                    or value["filename"] == self.selected_engine
                )
            ),
            choices[0],
        )
        combo_var = tkinter_module.StringVar(value=initial_display)
        combo = ttk_module.Combobox(
            container,
            textvariable=combo_var,
            values=choices,
            state="readonly",
            width=48,
        )
        combo.pack(anchor="w", pady=(6, 6))

        def _apply_engine_selection(
            _: tkinter_module.Event | None = None,
        ) -> None:
            """Refresh the UI when a different engine is selected."""
            selection = combo_var.get()
            record = display_map.get(selection)
            if record:
                if self._current_engine_module != record["id"]:
                    self._engine_setting_vars.clear()
                    self._engine_setting_specs.clear()
                self._current_engine_module = record["id"]
                self.selected_engine = record["id"]
                self.draft.engine = record["id"]
                self._selected_engine_entry = record
                capabilities, engine_kind = self._load_engine_capabilities(
                    record["id"]
                )
                self.engine_capabilities = capabilities
                self.selected_engine_kind = engine_kind
                detail_label.config(
                    text=(
                        f"{record['label']} uses module {record['id']} "
                        f"with SHA256 {record.get('hash', '')}."
                    )
                )
            else:
                self.engine_capabilities = None
                self.selected_engine_kind = "mcmc"
                self._engine_setting_vars.clear()
                self._engine_setting_specs.clear()
                self._current_engine_module = None
                detail_label.config(
                    text="Select an engine to see details.",
                )
            self._render_engine_run_settings(container)
            self._refresh_builder_step_indicators()

        combo.bind("<<ComboboxSelected>>", _apply_engine_selection)
        detail_label = ttk_module.Label(
            container,
            text="Select an engine to see details.",
            wraplength=720,
            takefocus=True,
        )
        detail_label.pack(anchor="w", pady=(4, 4))
        _apply_engine_selection()
        button_frame = ttk_module.Frame(container)
        button_frame.pack(anchor="w", pady=(4, 0))

        def _open_selected_engine_folder() -> None:
            """Open the folder containing the selected engine module."""
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
            """Present metadata for the selected engine module."""
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

        ttk_module.Button(
            button_frame,
            text="Open engine folder",
            command=_open_selected_engine_folder,
        ).pack(side="left", padx=2)
        ttk_module.Button(
            button_frame,
            text="View module",
            command=_view_selected_engine_module,
        ).pack(side="left", padx=2)
        _apply_engine_selection()

    def _render_engine_run_settings(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render engine-run tuning inputs next to the engine selector."""

        if self._engine_run_settings_frame is not None:
            self._engine_run_settings_frame.destroy()
            self._engine_run_settings_frame = None
        settings_frame = ttk_module.LabelFrame(
            container,
            text="Run settings",
        )
        settings_frame.pack(fill="x", pady=(8, 0))
        self._engine_run_settings_frame = settings_frame
        capabilities = self.engine_capabilities
        if not capabilities or not capabilities.settings:
            ttk_module.Label(
                settings_frame,
                text="This engine exposes no adjustable settings.",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            return
        recommendations: dict[str, str] | None = None
        if self.selected_engine_kind == "mcmc":
            recommendations = self._mcmc_recommendation_texts(
                self._compute_run_recommendations()
            )

        for setting in capabilities.settings:
            dtype = (setting.dtype or "str").lower()
            setting_var = self._engine_setting_vars.get(setting.key)
            initial_value = self._initial_engine_setting_value(setting)
            if dtype == "bool":
                if not isinstance(setting_var, tkinter_module.BooleanVar):
                    setting_var = tkinter_module.BooleanVar(
                        value=bool(initial_value)
                    )
                    self._engine_setting_vars[setting.key] = setting_var
            else:
                if not isinstance(setting_var, tkinter_module.StringVar):
                    setting_var = tkinter_module.StringVar(
                        value=str(initial_value)
                    )
                    self._engine_setting_vars[setting.key] = setting_var
                elif not setting_var.get():
                    setting_var.set(str(initial_value))
            self._engine_setting_specs[setting.key] = setting
            row = ttk_module.Frame(settings_frame)
            row.pack(anchor="w", pady=(4, 0))
            ttk_module.Label(
                row,
                text=f"{setting.label}:",
                width=24,
                takefocus=True,
            ).pack(side="left")
            field_frame = ttk_module.Frame(row)
            field_frame.pack(side="left", padx=(6, 0))
            draft_field = self._draft_field_for_setting(setting.key)

            def _bind_update(
                variable: tkinter_module.Variable,
                key: str,
            ) -> None:
                """Attach a listener that mirrors widget writes
                into settings."""

                def _update(*_: object) -> None:
                    """Trigger the engine-setting update handler."""
                    self._handle_engine_setting_update(key)

                variable.trace_add("write", _update)

            if dtype == "bool":
                control = ttk_module.Checkbutton(
                    field_frame,
                    variable=setting_var,
                    takefocus=True,
                    command=partial(
                        self._handle_engine_setting_update, setting.key
                    ),
                )
                control.pack(anchor="w")
            else:
                min_value, max_value = self._engine_setting_bounds(setting)
                increment = 1 if dtype == "int" else 0.1

                def _validate(value_if_allowed: str, key: str) -> bool:
                    """Clamp engine setting inputs between the declared
                    bounds."""
                    if not value_if_allowed.strip():
                        return True
                    try:
                        parsed = (
                            int(value_if_allowed)
                            if dtype == "int"
                            else float(value_if_allowed)
                        )
                    except ValueError:
                        return False
                    if parsed < min_value or parsed > max_value:
                        return False
                    self._handle_engine_setting_update(key)
                    return True

                validate_cmd = field_frame.register(_validate)
                control = tkinter_module.Spinbox(
                    field_frame,
                    from_=min_value,
                    to=max_value,
                    increment=increment,
                    textvariable=setting_var,
                    width=14,
                    validate="focusout",
                    validatecommand=(validate_cmd, "%P", setting.key),
                )
                control.pack(anchor="w")
                if isinstance(setting_var, tkinter_module.StringVar):
                    _bind_update(setting_var, setting.key)
            if dtype == "bool":
                pass
            elif draft_field and isinstance(
                setting_var, tkinter_module.StringVar
            ):

                def _sync_draft(
                    *_: object,
                    attr: str = draft_field,
                    variable: tkinter_module.StringVar = setting_var,
                ) -> None:
                    """Mirror widget values to the manifest draft."""
                    setattr(self.draft, attr, variable.get())

                setting_var.trace_add("write", _sync_draft)
            metadata: list[str] = []
            if setting.description:
                metadata.append(setting.description)
            if setting.default is not None:
                metadata.append(f"Default: {setting.default}")
            if setting.hint:
                metadata.append(f"Recommendation: {setting.hint}")
            if recommendations and setting.key in recommendations:
                metadata.append(recommendations[setting.key])
            if metadata:
                ttk_module.Label(
                    settings_frame,
                    text=" ".join(metadata),
                    wraplength=720,
                    justify="left",
                    takefocus=True,
                ).pack(anchor="w", padx=(16, 0), pady=(0, 2))
            self._handle_engine_setting_update(setting.key)

    def _load_engine_capabilities(
        self, module_name: str
    ) -> tuple[EngineCapabilities | None, str]:
        """Return engine capability descriptors and its kind."""

        engine_kind = "mcmc"
        try:
            module = importlib.import_module(module_name)
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
        ) as exc:
            logger.get_logger().warning(
                "Failed to import engine %s: %s", module_name, exc
            )
            return None, engine_kind
        engine_kind = getattr(module, "ENGINE_KIND", "mcmc").lower()
        try:
            capabilities = get_engine_capabilities(module)
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.get_logger().warning(
                "Failed to load engine capabilities for %s: %s",
                module_name,
                exc,
            )
            capabilities = None
        return capabilities, engine_kind

    def _draft_field_for_setting(self, key: str) -> str | None:
        """Return the draft field name linked to the given engine key."""
        mapping = {
            "n_walkers": "walkers",
            "burn_in_steps": "burn_in",
            "n_steps": "production_steps",
            "pool_size": "pool_size",
        }
        return mapping.get(key)

    def _initial_engine_setting_value(self, setting: EngineSetting) -> object:
        """Return the initial value to display for a given engine setting."""
        field = self._draft_field_for_setting(setting.key)
        if field:
            current = getattr(self.draft, field, "")
            if current:
                dtype = (setting.dtype or "str").lower()
                if dtype == "bool":
                    return current.lower() in {"1", "true", "yes", "on"}
                return current
        if setting.default is not None:
            if (setting.dtype or "").lower() == "bool":
                return bool(setting.default)
            return str(setting.default)
        return False if (setting.dtype or "").lower() == "bool" else ""

    def _handle_engine_setting_update(self, key: str) -> None:
        """Hook fired whenever an engine setting var changes."""

        field = self._draft_field_for_setting(key)
        if not field:
            return
        setting_var = self._engine_setting_vars.get(key)
        if setting_var is None:
            return
        setting_value = setting_var.get()
        if isinstance(setting_var, tkinter_module.BooleanVar):
            setattr(self.draft, field, "true" if setting_value else "false")
        else:
            setattr(self.draft, field, setting_value)

    def _mcmc_recommendation_texts(
        self, values: dict[str, int | str]
    ) -> dict[str, str]:
        """Describe recommended ranges for MCMC run settings."""
        return {
            "n_steps": (
                "Production steps control the sampler iterations. "
                f"Recommended default: {values['recommended_steps']}. "
                f"Minimum suggested: {values['production_min']}."
            ),
            "burn_in_steps": (
                "Burn-in discards early samples so the chain stabilises. "
                f"Recommended warm-up: {values['burn_in_recommended']}. "
                f"Quicker option: {values['quick_burn']}."
            ),
            "n_walkers": (
                "Walkers explore the posterior in parallel; "
                f"minimum required: {values['minimum_walkers']}. "
                f"Recommended ensemble: {values['recommended_walkers']}."
            ),
            "pool_size": (
                "Pools spread walkers across processes. "
                f"Recommended size: {values['pool_max']} "
                f"(detected CPUs: {values['cpu_label']}). "
                "Enter 0 to disable multiprocessing entirely."
            ),
        }

    def _engine_setting_bounds(
        self, setting: EngineSetting
    ) -> tuple[float, float]:
        """Return enforced numeric bounds for an engine setting."""
        module_limits = _ENGINE_SETTING_LIMITS.get(
            self._current_engine_module or "", {}
        )
        setting_limits = module_limits.get(setting.key, {})
        min_value = setting_limits.get("min")
        max_value = setting_limits.get("max")

        def _resolve(
            bound_input: float | int | str | None,
        ) -> float | int | None:
            """Convert special labels into numeric capability bounds."""
            if isinstance(bound_input, str) and bound_input == "cpu":
                cores = os.cpu_count() or 1
                return max(1, cores)
            return bound_input

        min_value = _resolve(min_value)
        max_value = _resolve(max_value)
        dtype = (setting.dtype or "str").lower()
        if dtype == "int":
            if min_value is None:
                min_value = 0
            if max_value is None:
                max_value = sys.maxsize
            return float(int(min_value)), float(int(max_value))
        if dtype == "float":
            if min_value is None:
                min_value = 0.0
            if max_value is None:
                max_value = 1_000_000.0
            return float(min_value), float(max_value)
        return 0.0, float(sys.maxsize)

    def _render_builder_step_manifest(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render the manifest description/editor step for the builder."""
        ttk_module.Frame(container, height=30).pack(fill="x", pady=(0, 6))
        ttk_module.Label(
            container,
            text="Describe the production plan or testing notes for this run.",
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w")
        plan_var = tkinter_module.StringVar(value=self.draft.plan)

        def _update_plan(*_args: object) -> None:
            """Mirror the plan input back into the draft manifest."""
            self.draft.plan = plan_var.get()

        plan_var.trace_add("write", _update_plan)
        ttk_module.Entry(
            container,
            textvariable=plan_var,
            width=80,
        ).pack(anchor="w", pady=(6, 0))

        manifest_frame = ttk_module.LabelFrame(
            container,
            text="Manifest",
        )
        manifest_frame.pack(fill="both", expand=True, pady=(12, 0))
        status_text = (
            f"Saved: {self.manifest_workspace.manifest_path}"
            if self.manifest_workspace
            else "Manifest not saved yet."
        )
        ttk_module.Label(
            manifest_frame,
            text=status_text,
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w", pady=(0, 4))
        preview_panel = ttk_module.Frame(manifest_frame)
        preview_panel.pack(fill="both", expand=True)
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)
        manifest_text_widget = tkinter_module.Text(
            preview_panel,
            wrap="none",
            borderwidth=1,
            relief="solid",
            height=16,
        )
        manifest_text_widget.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk_module.Scrollbar(
            preview_panel,
            orient="vertical",
            command=manifest_text_widget.yview,
        )
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll = ttk_module.Scrollbar(
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

        if self.manifest_workspace is None:
            reminder_text = self._MANIFEST_REMINDER_MESSAGE
        else:
            reminder_text = "Manifest saved; proceed to confirmation."
        ttk_module.Label(
            container,
            text=reminder_text,
            wraplength=720,
            takefocus=True,
        ).pack(anchor="w", pady=(8, 0))

        actions_frame = ttk_module.Frame(container)
        actions_frame.pack(anchor="w", pady=(12, 0))

        def _save_manifest_action() -> None:
            """Persist the current manifest and return to the builder."""
            self._persist_manifest_workspace(notify=True)
            self.show_run_builder()

        save_state = (
            tkinter_module.NORMAL
            if self._builder_ready()
            else tkinter_module.DISABLED
        )
        clear_state = (
            tkinter_module.NORMAL
            if self._has_configuration()
            else tkinter_module.DISABLED
        )
        open_state = (
            tkinter_module.NORMAL
            if self.manifest_workspace is not None
            else tkinter_module.DISABLED
        )
        ttk_module.Button(
            actions_frame,
            text="Save manifest",
            command=_save_manifest_action,
            state=save_state,
        ).pack(side="left", padx=(0, 4))
        ttk_module.Button(
            actions_frame,
            text="Save and confirm",
            command=self._save_manifest_and_proceed,
            state=save_state,
        ).pack(side="left", padx=(0, 4))
        ttk_module.Button(
            actions_frame,
            text="Save to external folder...",
            command=self._save_manifest_to_external_folder,
            state=save_state,
        ).pack(side="left", padx=(0, 4))
        ttk_module.Button(
            actions_frame,
            text="Clear manifest",
            command=lambda: (
                self._clear_manifest_configuration(),
                self.show_run_builder(),
            ),
            state=clear_state,
        ).pack(side="left", padx=(0, 4))
        ttk_module.Button(
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
        except (
            OSError,
            RuntimeError,
            TypeError,
            UnicodeError,
            ValueError,
            yaml.YAMLError,
        ):
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

    def _render_builder_step_confirm(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Show the confirmation summary and start-run controls."""
        self._stage_confirm_manifest()
        ttk_module.Frame(container, height=30).pack(fill="x", pady=(0, 6))
        summary_frame = ttk_module.Frame(container)
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
        for label, entry_value in summary_entries:
            ttk_module.Label(
                summary_frame,
                text=f"{label}: {entry_value}",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w")
        button_frame = ttk_module.Frame(container)
        button_frame.pack(anchor="w", pady=(12, 0))
        start_state = (
            tkinter_module.NORMAL
            if self.manifest_workspace is not None
            else tkinter_module.DISABLED
        )
        ttk_module.Button(
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

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the dataset catalogue view with filters."""
            self._page_header(frame, "Data catalogue")
            ttk_module.Label(
                frame,
                text=(
                    "Datasets remain selectable from the Run Builder while "
                    "this catalogue surfaces parser digests, citations and "
                    "licensing to prove registration rules are honoured."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            filters = ttk_module.Frame(frame)
            filters.pack(anchor="w", pady=(0, 8))
            filters_label = ttk_module.Label(
                filters,
                text="Filters:",
                takefocus=True,
            )
            filters_label.pack(side="left", padx=(0, 4))
            ttk_module.Button(
                filters,
                text="All",
                command=lambda: (
                    self.filter_catalogue([]),
                    self.show_data_overview(),
                ),
                takefocus=True,
            ).pack(side="left", padx=2)
            for key in ("sne", "bao", "cmb"):
                ttk_module.Button(
                    filters,
                    text=key.upper(),
                    command=lambda k=key: (
                        self.filter_catalogue([k]),
                        self.show_data_overview(),
                    ),
                    takefocus=True,
                ).pack(side="left", padx=2)
            active = self.filter_catalogue(self.last_filter_types or [])
            ttk_module.Label(
                frame,
                text=(
                    f"Showing {len(active)} dataset(s); trust alerts: "
                    f"{len(self.validation_notes)}"
                ),
                takefocus=True,
            ).pack(anchor="w", pady=(0, 6))
            catalogue_panel = self._create_scrollable_panel(frame)
            for dataset in active:
                entry_frame = ttk_module.LabelFrame(
                    catalogue_panel,
                    text=f"{dataset['name']} ({dataset['id']})",
                    padding=(8, 6),
                )
                entry_frame.pack(fill="x", pady=4)
                ttk_module.Label(
                    entry_frame,
                    text=" | ".join(dataset.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                citation_value = dataset.get("citation", "missing")
                license_value = dataset.get("license", "unspecified")
                parser_digest = dataset.get("parser_digest", "n/a")
                metadata_digest = dataset.get("metadata_digest", "")
                ttk_module.Label(
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
                ttk_module.Label(
                    entry_frame,
                    text="Independence notes: "
                    + "; ".join(dataset.get("independence", [])),
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w", pady=(0, 4))
                actions = ttk_module.Frame(entry_frame)
                actions.pack(anchor="w")
                dataset_id = dataset["id"]

                def _open_current_folder(
                    path: str = dataset["path"],
                    dataset_label: str = dataset_id,
                ) -> None:
                    """Open the dataset directory currently listed."""
                    self._open_folder_or_warn(
                        path,
                        context="data",
                        subject=f"dataset {dataset_label}",
                    )

                ttk_module.Button(
                    actions,
                    text="Open folder",
                    command=_open_current_folder,
                    takefocus=True,
                ).pack(side="left", padx=2)

                def _view_current_metadata() -> None:
                    """Show metadata for the dataset currently displayed."""
                    self._present_metadata(
                        dataset_id, f"Dataset metadata: {dataset_id}"
                    )

                def _revalidate_current_parser() -> None:
                    """Revalidate the parser for the displayed dataset."""
                    self._revalidate_dataset_action(dataset_id)

                ttk_module.Button(
                    actions,
                    text="View metadata",
                    command=_view_current_metadata,
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk_module.Button(
                    actions,
                    text="Revalidate parser",
                    command=_revalidate_current_parser,
                    takefocus=True,
                ).pack(side="left", padx=2)

        self._swap_content(builder)

    def show_models(self) -> None:
        """Display installed model definitions and digests."""

        self.refresh_inventory()

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the model catalogue with metadata and hash info."""
            self._page_header(frame, "Models")
            ttk_module.Label(
                frame,
                text=(
                    "Model YAML files drive compatibility badges and priors. "
                    "This view lists their hashes so Run Builder choices can "
                    "be audited alongside dataset digests."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            ttk_module.Label(
                frame,
                text=f"Discovered {len(self.model_index)} model(s)",
                takefocus=True,
            ).pack(anchor="w", pady=(0, 6))
            catalogue_panel = self._create_scrollable_panel(frame)
            for model in sorted(
                self.model_index.values(), key=lambda entry: entry["id"]
            ):
                entry_frame = ttk_module.LabelFrame(
                    catalogue_panel,
                    text=f"{model['id']} ({model['filename']})",
                    padding=(8, 6),
                )
                entry_frame.pack(fill="x", pady=4)
                ttk_module.Label(
                    entry_frame,
                    text="Badges: " + ", ".join(model.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                ttk_module.Label(
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
                actions = ttk_module.Frame(entry_frame)
                actions.pack(anchor="w")
                model_folder = os.path.dirname(model["path"])
                model_id = model["id"]

                def _open_model_folder() -> None:
                    """Open the directory containing the selected model."""
                    self._open_folder_or_warn(
                        model_folder,
                        context="models",
                        subject=f"model {model_id}",
                    )

                def _view_model_yaml() -> None:
                    """Show the YAML content of the selected model."""
                    self._present_metadata(
                        model_id, f"Model definition: {model_id}"
                    )

                ttk_module.Button(
                    actions,
                    text="Open model folder",
                    command=_open_model_folder,
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk_module.Button(
                    actions,
                    text="View YAML",
                    command=_view_model_yaml,
                    takefocus=True,
                ).pack(side="left", padx=2)

        self._swap_content(builder)

    def show_engines(self) -> None:
        """Display engine overview panel with digests and health checks."""

        self.refresh_inventory()

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the engine catalogue with badges and hashes."""
            self._page_header(frame, "Engines")
            ttk_module.Label(
                frame,
                text=(
                    "Engines expose dataset compatibility and sampler labels. "
                    "Hashes appear here so health checks can confirm which "
                    "module executed a run."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            ttk_module.Label(
                frame,
                text=f"Discovered {len(self.engine_index)} engine(s)",
                takefocus=True,
            ).pack(anchor="w", pady=(0, 6))
            catalogue_panel = self._create_scrollable_panel(frame)
            for engine in sorted(
                self.engine_index.values(), key=lambda entry: entry["label"]
            ):
                entry_frame = ttk_module.LabelFrame(
                    catalogue_panel,
                    text=f"{engine['label']} ({engine['filename']})",
                    padding=(8, 6),
                )
                entry_frame.pack(fill="x", pady=4)
                ttk_module.Label(
                    entry_frame,
                    text="Badges: " + ", ".join(engine.get("badges", [])),
                    takefocus=True,
                ).pack(anchor="w")
                ttk_module.Label(
                    entry_frame,
                    text=(
                        f"Version: {engine.get('version', 'unknown')}\n"
                        f"SHA256: {engine.get('hash', '')}"
                    ),
                    wraplength=720,
                    takefocus=True,
                    justify="left",
                ).pack(anchor="w", pady=(4, 4))
                actions = ttk_module.Frame(entry_frame)
                actions.pack(anchor="w")
                engine_folder = os.path.dirname(engine["path"])
                engine_id = engine["id"]
                engine_label = engine["label"] or engine["stem"]

                def _open_engine_folder() -> None:
                    """Open the directory containing the current engine."""
                    self._open_folder_or_warn(
                        engine_folder,
                        context="engines",
                        subject=f"engine {engine_label}",
                    )

                def _view_engine_module() -> None:
                    """Show the metadata for the engine module."""
                    self._present_metadata(
                        engine_id, f"Engine module: {engine_label}"
                    )

                ttk_module.Button(
                    actions,
                    text="Open engine folder",
                    command=_open_engine_folder,
                    takefocus=True,
                ).pack(side="left", padx=2)
                ttk_module.Button(
                    actions,
                    text="View module",
                    command=_view_engine_module,
                    takefocus=True,
                ).pack(side="left", padx=2)

        self._swap_content(builder)

    def _handle_setting_change(
        self, section: str, key: str, *_args: Any
    ) -> None:
        """Persist the updated widget value into the pending settings."""

        if self._suppress_setting_events:
            return
        section_vars = self._settings_vars.get(section, {})
        setting_var = section_vars.get(key)
        if setting_var is None:
            return
        if isinstance(setting_var, tkinter_module.BooleanVar):
            new_value = bool(setting_var.get())
        elif isinstance(setting_var, tkinter_module.IntVar):
            try:
                new_value = int(setting_var.get())
            except (TypeError, ValueError):
                new_value = 0
        else:
            new_value = str(setting_var.get())
        old_value = self._pending_settings.get(section, {}).get(key)
        if old_value == new_value:
            return
        self._pending_settings[section][key] = new_value
        self._mark_settings_dirty()

    def _ensure_setting_var(
        self, section: str, key: str
    ) -> tkinter_module.Variable | None:
        """Return or create the Tk variable for *section/key*."""

        if not self.render or tk_gui is None:
            return None
        section_vars = self._settings_vars.setdefault(section, {})
        if key in section_vars:
            return section_vars[key]
        pending_value = self._pending_settings.get(section, {}).get(key)
        if isinstance(pending_value, bool):
            setting_var = tkinter_module.BooleanVar(value=pending_value)
        elif isinstance(pending_value, int):
            setting_var = tkinter_module.IntVar(value=pending_value)
        elif isinstance(pending_value, float):
            setting_var = tkinter_module.DoubleVar(value=pending_value)
        else:
            setting_var = tkinter_module.StringVar(value=str(pending_value))
        setting_var.trace_add(
            "write",
            partial(self._handle_setting_change, section, key),
        )
        section_vars[key] = setting_var
        return setting_var

    def _refresh_setting_vars(self) -> None:
        """Synchronise the Tk variables with the pending settings."""

        if self._suppress_setting_events:
            return
        self._suppress_setting_events = True
        try:
            for section, values in self._pending_settings.items():
                section_vars = self._settings_vars.get(section, {})
                for key, setting_var in section_vars.items():
                    pending_value = values.get(key)
                    if isinstance(setting_var, tkinter_module.BooleanVar):
                        setting_var.set(bool(pending_value))
                    elif isinstance(setting_var, tkinter_module.IntVar):
                        try:
                            setting_var.set(int(pending_value))
                        except (TypeError, ValueError):
                            setting_var.set(0)
                    elif isinstance(setting_var, tkinter_module.DoubleVar):
                        try:
                            setting_var.set(float(pending_value))
                        except (TypeError, ValueError):
                            setting_var.set(0.0)
                    else:
                        setting_var.set(str(pending_value))
        finally:
            self._suppress_setting_events = False

    def _set_settings_dirty(self, dirty_value: bool) -> None:
        """Update the dirty flag and refresh the action button states."""

        self._settings_dirty = dirty_value
        self._update_settings_action_states()

    def _update_settings_action_states(self) -> None:
        """Toggle the Save/Cancel buttons based on dirty state."""

        if self._settings_save_button is not None:
            if self._settings_dirty:
                self._settings_save_button.state(["!disabled"])
            else:
                self._settings_save_button.state(["disabled"])
        if self._settings_cancel_button is not None:
            if self._settings_dirty:
                self._settings_cancel_button.state(["!disabled"])
            else:
                self._settings_cancel_button.state(["disabled"])

    def _mark_settings_dirty(self) -> None:
        """Mark the settings as modified if they are not already."""

        if not self._settings_dirty:
            self._set_settings_dirty(True)

    def _refresh_settings_section_indicators(self) -> None:
        """Highlight which settings section is active."""

        # Buttons remain active; the current page is already indicated
        # by the header.

    def _render_settings_section(self) -> None:
        """Rebuild the active section's content."""

        if self._settings_page_body is None:
            return
        for child in self._settings_page_body.winfo_children():
            child.destroy()
        section = self._settings_sections[self._settings_current_index]
        if self._settings_header_label is not None:
            self._settings_header_label.configure(
                text=f"Settings: {section['label']}"
            )
        if self._settings_description_label is not None:
            self._settings_description_label.configure(
                text=section["description"]
            )
        section["builder"](self._settings_page_body)
        self._refresh_settings_section_indicators()

    def _navigate_settings_section(self, index: int) -> None:
        """Switch to another tab, guarding against unsaved changes."""

        if index == self._settings_current_index:
            return
        if not self._confirm_settings_navigation():
            return
        self._settings_current_index = index
        self._render_settings_section()

    def _confirm_settings_navigation(self) -> bool:
        """Ensure the user saves or reverts before leaving a dirty section."""

        if not self._settings_dirty:
            return True
        decision = self._prompt_settings_discard_dialog()
        if decision == "save":
            self._persist_pending_settings(notify=False)
            return True
        if decision == "revert":
            self._reset_settings_to_saved()
            return True
        return False

    def _prompt_settings_discard_dialog(self) -> str | None:
        """Ask the operator whether to save or revert unsaved changes."""

        if not (
            self.render
            and tk_gui is not None
            and messagebox is not None
            and self.root is not None
        ):
            self.create_toast(
                "Save or revert settings before navigating away.",
                severity="WARNING",
                context="settings",
            )
            return None

        dialog = tkinter_module.Toplevel(self.root)
        dialog.title("Unsaved settings")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        ttk_module.Label(
            dialog,
            text="You must either save or revert changes to navigate away.",
            wraplength=320,
            justify="left",
            takefocus=True,
        ).grid(row=0, column=0, padx=12, pady=(12, 8), sticky="w")
        button_frame = ttk_module.Frame(dialog)
        button_frame.grid(row=1, column=0, pady=(0, 12), padx=12)

        choice: dict[str, str | None] = {"value": None}

        def _close() -> None:
            """Close the unsaved settings dialog."""
            if dialog.grab_current() is dialog:
                dialog.grab_release()
            dialog.destroy()

        def _select(decision: str) -> None:
            """Capture the user's decision in the confirmation dialog."""
            choice["value"] = decision
            _close()

        ttk_module.Button(
            button_frame,
            text="Revert",
            command=lambda: _select("revert"),
            takefocus=True,
        ).pack(side="left", padx=4)
        ttk_module.Button(
            button_frame,
            text="Save and continue",
            command=lambda: _select("save"),
            takefocus=True,
        ).pack(side="left", padx=4)
        dialog.protocol("WM_DELETE_WINDOW", _close)
        dialog.wait_window()
        return choice["value"]

    def _persist_pending_settings(self, *, notify: bool = True) -> None:
        """Save the pending settings to disk."""

        settings_mod.save_settings(self._pending_settings)
        self._saved_settings = copy.deepcopy(self._pending_settings)
        self._set_settings_dirty(False)
        if notify:
            self.create_toast(
                "Settings saved.",
                severity="INFO",
                context="settings",
            )

    def _reset_settings_to_saved(self) -> None:
        """Revert the UI to the last-saved values."""

        self._pending_settings = copy.deepcopy(self._saved_settings)
        self._refresh_setting_vars()
        self._set_settings_dirty(False)
        self._render_settings_section()

    def _reset_settings_to_defaults(self) -> None:
        """Load the defaults so operators can start over."""

        self._pending_settings = copy.deepcopy(settings_mod.DEFAULT_SETTINGS)
        self._refresh_setting_vars()
        self._set_settings_dirty(True)
        self._render_settings_section()

    def show_settings(self) -> None:
        """Display the configurable settings panel with section tabs."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the settings tabs and actions for the current section."""
            section_label = self._settings_sections[
                self._settings_current_index
            ]["label"]
            self._settings_header_label = self._page_header(
                frame,
                f"Settings: {section_label}",
            )
            tab_bar = ttk_module.Frame(frame)
            tab_bar.pack(fill="x", pady=(0, 8))
            self._settings_section_buttons = []
            button_style = self._monitor_button_kwargs()
            for index, section in enumerate(self._settings_sections):
                button = ttk_module.Button(
                    tab_bar,
                    text=section["label"],
                    command=lambda idx=index: self._navigate_settings_section(
                        idx
                    ),
                    takefocus=True,
                    **button_style,
                )
                button.pack(side="left", padx=4)
                self._settings_section_buttons.append(button)
            self._settings_description_label = ttk_module.Label(
                frame,
                text=self._settings_sections[self._settings_current_index][
                    "description"
                ],
                wraplength=720,
                takefocus=True,
            )
            self._settings_description_label.pack(anchor="w", pady=(0, 8))
            action_row = ttk_module.Frame(frame)
            action_row.pack(fill="x", pady=(0, 32))
            self._settings_defaults_button = ttk_module.Button(
                action_row,
                text="Defaults",
                command=self._reset_settings_to_defaults,
                takefocus=True,
                **button_style,
            )
            self._settings_defaults_button.pack(side="left", padx=4)
            self._settings_cancel_button = ttk_module.Button(
                action_row,
                text="Cancel",
                command=self._reset_settings_to_saved,
                takefocus=True,
                **button_style,
            )
            self._settings_cancel_button.pack(side="left", padx=4)
            self._settings_save_button = ttk_module.Button(
                action_row,
                text="Save",
                command=self._persist_pending_settings,
                takefocus=True,
                **button_style,
            )
            self._settings_save_button.pack(side="left", padx=4)
            body = ttk_module.Frame(frame)
            body.pack(fill="both", expand=True)
            self._settings_page_body = body
            self._set_settings_dirty(self._settings_dirty)
            self._render_settings_section()

        self._swap_content(builder)

    def _build_settings_datasets_page(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Render dataset discovery toggles and digest helpers."""

        auto_var = self._ensure_setting_var(
            "datasets", "auto_dataset_discovery"
        )
        if auto_var is not None:
            ttk_module.Checkbutton(
                container,
                text="Enable automatic dataset discovery",
                variable=auto_var,
                takefocus=True,
            ).pack(anchor="w", pady=(0, 4))
        rebuild_var = self._ensure_setting_var(
            "datasets", "dataset_hash_auto_rebuild"
        )
        if rebuild_var is not None:
            ttk_module.Checkbutton(
                container,
                text="Rebuild dataset hashes automatically",
                variable=rebuild_var,
                takefocus=True,
            ).pack(anchor="w", pady=(0, 4))
        ttl_var = self._ensure_setting_var(
            "datasets", "dataset_hash_ttl_hours"
        )
        if ttl_var is not None:
            ttl_frame = ttk_module.Frame(container)
            ttl_frame.pack(fill="x", pady=(4, 6))
            hash_label = ttk_module.Label(
                ttl_frame,
                text="Hash TTL (hours):",
            )
            hash_label.pack(side="left")
            tkinter_module.Spinbox(
                ttl_frame,
                from_=1,
                to=168,
                width=4,
                textvariable=ttl_var,
            ).pack(side="left", padx=6)
            ttk_module.Label(
                ttl_frame,
                text="Rebuild caches after this many hours when flagged.",
                wraplength=520,
                justify="left",
            ).pack(anchor="w", pady=(4, 0))
        ttk_module.Button(
            container,
            text="Recalculate digests",
            command=self._rebuild_dataset_hashes_action,
            takefocus=True,
            **self._monitor_button_kwargs(),
        ).pack(anchor="w", pady=(8, 0))

    def _build_settings_gui_page(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Surface GUI-specific toggles and environment hints."""

        venv_var = self._ensure_setting_var("gui", "require_managed_venv")
        if venv_var is not None:
            ttk_module.Checkbutton(
                container,
                text="Require the managed .venv before running",
                variable=venv_var,
                takefocus=True,
            ).pack(anchor="w", pady=(0, 4))
        hints_var = self._ensure_setting_var("gui", "show_environment_hints")
        if hints_var is not None:
            ttk_module.Checkbutton(
                container,
                text="Show environment hints in the status bar",
                variable=hints_var,
                takefocus=True,
            ).pack(anchor="w", pady=(0, 4))

        env_frame = ttk_module.LabelFrame(
            container,
            text="Environment hints",
        )
        env_frame.pack(fill="x", pady=(8, 4))
        env_values = [
            ("COPERNICAN_SEED", os.environ.get("COPERNICAN_SEED", "unset")),
            (
                "COPERNICAN_STRICT_WARNINGS",
                os.environ.get("COPERNICAN_STRICT_WARNINGS", "0"),
            ),
        ]
        for env_name, env_value in env_values:
            ttk_module.Label(
                env_frame,
                text=f"{env_name}: {env_value}",
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w")

    def _build_settings_tools_page(
        self,
        container: tkinter_module.Frame,
    ) -> None:
        """Expose shared maintenance helpers."""

        button_style = self._monitor_button_kwargs()

        def _tool_row(
            text: str, command: Callable[[], None], desc: str
        ) -> None:
            """Add a helper row with an action button and description."""
            row = ttk_module.Frame(container)
            row.pack(fill="x", pady=(0, 14))
            ttk_module.Button(
                row,
                text=text,
                command=command,
                takefocus=True,
                **button_style,
            ).pack(anchor="w", padx=(0, 6), pady=(0, 6))
            ttk_module.Label(
                row,
                text=desc,
                wraplength=720,
                justify="left",
            ).pack(anchor="w", padx=(0, 6), pady=(0, 10))

        _tool_row(
            "Rebuild model cache",
            self._rebuild_model_cache_action,
            "Re-validate every YAML model and regenerate the sanitized cache.",
        )
        _tool_row(
            "Revalidate parsers",
            self._revalidate_all_parsers_action,
            (
                "Force the parser registry to re-run trust checks before "
                "refreshing the catalogue."
            ),
        )
        _tool_row(
            "Reset Run Builder",
            self._reset_builder_workspace_action,
            "Clear the Run Builder selections and draft manifest.",
        )

    def _rebuild_dataset_hashes_action(self) -> None:
        """Force the catalogue to recompute every dataset digest."""

        try:
            self.refresh_inventory(force_discovery=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Failed to refresh dataset digests: {exc}",
                severity="ERROR",
                context="settings",
            )
            logger.get_logger().warning(
                "Failed to refresh dataset hashes", exc_info=exc
            )
            return
        self.create_toast(
            "Dataset digests refreshed.",
            severity="INFO",
            context="settings",
        )

    def _revalidate_all_parsers_action(self) -> None:
        """Re-run parser trust checks across every observation."""

        try:
            dataset_registry.discover_trusted_parsers(
                self._data_root(),
                force=True,
            )
            self.refresh_inventory(force_discovery=True)
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self.create_toast(
                f"Parser revalidation failed: {exc}",
                severity="ERROR",
                context="settings",
            )
            logger.get_logger().warning(
                "Parser revalidation failed", exc_info=exc
            )
            return
        self.create_toast(
            "Parsers revalidated successfully.",
            severity="INFO",
            context="settings",
        )

    def _rebuild_model_cache_action(self) -> None:
        """Recreate the cached YAML models from disk."""

        models_root = Path(self._models_root())
        cache_dir = models_root / "cache"
        rebuilt = 0
        try:
            for pattern in ("*.yml", "*.yaml"):
                for yaml_path in sorted(models_root.glob(pattern)):
                    if yaml_path.name.startswith("__"):
                        continue
                    model_spec_validator.validate_and_cache_model(
                        yaml_path, cache_dir
                    )
                    rebuilt += 1
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.create_toast(
                f"Model cache rebuild failed: {exc}",
                severity="ERROR",
                context="settings",
            )
            logger.get_logger().warning(
                "Model cache rebuild failed", exc_info=exc
            )
            return
        self.create_toast(
            f"Rebuilt cache for {rebuilt} model(s).",
            severity="INFO",
            context="settings",
        )

    def _reset_builder_workspace_action(self) -> None:
        """Clear the Run Builder selections along with the draft manifest."""

        self._clear_builder_selections()
        self.manifest_workspace = None
        self.create_toast(
            "Run Builder selections reset.",
            severity="INFO",
            context="settings",
        )

    def show_help(self) -> None:
        """Display contextual help panel with GUI and CLI guides."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the help navigation buttons and markdown viewer."""
            self._help_page_buttons = {}
            self._help_text_widget = None
            self._help_title_label = self._page_header(
                frame, self._help_header_text()
            )
            ttk_module.Label(
                frame,
                text=(
                    "Select a Copernican guide to review the GUI workflow or "
                    "the CLI manifest pipeline. The Help view renders the "
                    "Markdown guides exactly as they ship in docs/."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 8))
            button_row = ttk_module.Frame(frame)
            button_row.pack(fill="x", pady=(0, 12))
            button_style = self._monitor_button_kwargs()
            for page in _HELP_PAGES:
                button = ttk_module.Button(
                    button_row,
                    text=page["label"],
                    command=lambda pid=page["id"]: self._select_help_page(pid),
                    width=12,
                    takefocus=True,
                    **button_style,
                )
                button.pack(side="left", padx=4)
                self._help_page_buttons[page["id"]] = button
            content_frame = ttk_module.Frame(frame)
            content_frame.pack(fill="both", expand=True)
            if not (self.render and tk_gui is not None):
                ttk_module.Label(
                    content_frame,
                    text=(
                        "Help content is available from docs/gui_guide.md and "
                        "docs/cli_guide.md in the project root."
                    ),
                    wraplength=720,
                    takefocus=True,
                ).pack(anchor="w")
                self._refresh_help_page_view()
                return
            self._load_help_banner()
            if self.help_banner_image:
                banner_label = ttk_module.Label(
                    content_frame, image=self.help_banner_image
                )
                banner_label.image = self.help_banner_image
                banner_label.pack(pady=(0, 8))
            text_frame = ttk_module.Frame(content_frame)
            text_frame.pack(fill="both", expand=True)
            text_frame.columnconfigure(0, weight=1)
            text_frame.rowconfigure(0, weight=1)
            text_widget = tkinter_module.Text(
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
            scrollbar = ttk_module.Scrollbar(
                text_frame,
                orient="vertical",
                command=text_widget.yview,
            )
            scrollbar.grid(row=0, column=1, sticky="ns")
            text_widget.configure(yscrollcommand=scrollbar.set)
            self._help_text_widget = text_widget
            self._refresh_help_page_view()

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
            "About Copernican",
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
        self._log_application_event("GUI exit requested", logging.INFO)
        try:
            import copernican

            copernican.exit_clean(0)
        except (OSError, RuntimeError, ValueError):
            sys.exit(0)

    def show_run_monitor(self) -> None:
        """Display live run status controls."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the run monitor view with progress bars and logs."""
            self._status_label = None
            self._progress_status_label = None
            self._batch_progressbar = None
            self._walker_progressbar = None
            self._monitor_log_widget = None
            self._monitor_filter_label = None
            self._page_header(frame, "Run Monitor")
            self._status_label = ttk_module.Label(
                frame,
                text=self._status_text(),
                takefocus=True,
            )
            self._status_label.pack(anchor="w")
            progress_frame = ttk_module.Frame(frame)
            progress_frame.pack(fill="x", pady=(8, 8))
            self._progress_status_label = ttk_module.Label(
                progress_frame,
                text="Stage: Idle",
                font=("Helvetica", 12, "bold"),
                takefocus=True,
            )
            self._progress_status_label.pack(anchor="w")
            ttk_module.Label(
                progress_frame,
                text="Overall batch progress",
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))
            self._batch_progressbar = ttk_module.Progressbar(
                progress_frame, maximum=100, length=360
            )
            self._batch_progressbar.pack(fill="x", pady=(2, 0))
            ttk_module.Label(
                progress_frame,
                text="Walker progress",
                takefocus=True,
            ).pack(anchor="w", pady=(6, 0))
            self._walker_progressbar = ttk_module.Progressbar(
                progress_frame, maximum=100, length=360
            )
            self._walker_progressbar.pack(fill="x", pady=(2, 0))
            meta_frame = ttk_module.Frame(frame)
            meta_frame.pack(anchor="w", pady=(4, 4))
            ttk_module.Label(
                meta_frame,
                text="Active manifest metadata:",
                takefocus=True,
            ).pack(anchor="w")
            for line in self.summary.manifest_metadata:
                metadata_label = ttk_module.Label(
                    meta_frame,
                    text=line,
                    takefocus=True,
                )
                metadata_label.pack(anchor="w")
            log_frame = ttk_module.LabelFrame(frame, text="Run logs")
            log_frame.pack(fill="both", expand=True, pady=(8, 4))
            self._monitor_filter_label = ttk_module.Label(
                log_frame,
                text=f"Filter: {self.monitor_filter_level}+",
                takefocus=True,
            )
            self._monitor_filter_label.pack(anchor="w")
            log_filters = ttk_module.Frame(log_frame)
            log_filters.pack(anchor="w", pady=(2, 4))
            ttk_module.Button(
                log_filters,
                text="Info",
                command=lambda: self.set_monitor_filter("INFO"),
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk_module.Button(
                log_filters,
                text="Warnings",
                command=lambda: self.set_monitor_filter("WARNING"),
                takefocus=True,
            ).pack(side="left", padx=2)
            ttk_module.Button(
                log_filters,
                text="Errors",
                command=lambda: self.set_monitor_filter("ERROR"),
                takefocus=True,
            ).pack(side="left", padx=2)
            lock_frame = ttk_module.Frame(log_frame)
            lock_frame.pack(anchor="w", pady=(0, 4))
            self._monitor_log_lock_var = tkinter_module.BooleanVar(value=True)
            ttk_module.Checkbutton(
                lock_frame,
                text="Lock log to latest entry",
                variable=self._monitor_log_lock_var,
                takefocus=True,
            ).pack(side="left")
            text_panel = ttk_module.Frame(log_frame)
            text_panel.pack(fill="both", expand=True)
            text_panel.columnconfigure(0, weight=1)
            text_panel.rowconfigure(0, weight=1)
            text_widget = tkinter_module.Text(
                text_panel,
                wrap="none",
                padx=8,
                pady=6,
                borderwidth=0,
                highlightthickness=0,
                height=12,
            )
            text_widget.grid(row=0, column=0, sticky="nsew")
            vscroll = ttk_module.Scrollbar(
                text_panel,
                orient="vertical",
                command=text_widget.yview,
            )
            vscroll.grid(row=0, column=1, sticky="ns")
            hscroll = ttk_module.Scrollbar(
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
            log_actions = ttk_module.Frame(log_frame)
            log_actions.pack(anchor="w", pady=(4, 0))
            button_style = self._monitor_button_kwargs()
            self._monitor_log_view_button = ttk_module.Button(
                log_actions,
                text="View log",
                command=self._view_run_log,
                takefocus=True,
                **button_style,
            )
            self._monitor_log_view_button.pack(side="left", padx=2)
            self._monitor_log_open_button = ttk_module.Button(
                log_actions,
                text="Open log…",
                command=self._open_run_log_file,
                takefocus=True,
                **button_style,
            )
            self._monitor_log_open_button.pack(side="left", padx=2)
            alerts = ttk_module.LabelFrame(frame, text="Active alerts")
            alerts.pack(fill="x", pady=(4, 8))
            for alert in self.alerts[-5:]:
                alert_row = ttk_module.Frame(alerts)
                alert_row.pack(anchor="w", fill="x", pady=(2, 0))
                ttk_module.Label(
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
                    """Scroll the log view to the specified anchor."""
                    self.jump_to_log_anchor(anchor)

                ttk_module.Button(
                    alert_row,
                    text="Jump to log",
                    command=_jump,
                    takefocus=True,
                ).pack(side="left", padx=2)
            controls = ttk_module.Frame(frame)
            controls.pack(anchor="w")
            button_style = self._monitor_button_kwargs()
            self._run_output_button = ttk_module.Button(
                controls,
                text="Open run output",
                command=self.open_current_run_output,
                takefocus=True,
                **button_style,
            )
            self._run_output_button.pack(side="left", padx=4)
            self._cancel_button = ttk_module.Button(
                controls,
                text="Cancel",
                command=self.cancel_run,
                takefocus=True,
                **button_style,
            )
            self._cancel_button.pack(side="left", padx=4)
            self._pause_button = ttk_module.Button(
                controls,
                text="Pause",
                command=self.pause_run,
                takefocus=True,
                **button_style,
            )
            self._pause_button.pack(side="left", padx=4)
            self._hard_stop_button = ttk_module.Button(
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

        if getattr(self, "current_phase", ""):
            suffix = f" – {self.current_phase}"
        else:
            suffix = ""
        return f"Status: {self.status.value}{suffix}"

    def _refresh_status_label(self) -> None:
        """Update the status label text if it is visible."""

        if self._widget_is_alive(self._status_label):
            self._status_label.configure(text=self._status_text())
        else:
            self._status_label = None

    def _update_monitor_controls_state(self) -> None:
        """Enable or disable monitor buttons based on current state."""

        run_active = self.status is RunStatus.RUNNING
        control_state = (
            tkinter_module.NORMAL if run_active else tkinter_module.DISABLED
        )
        alive_control_buttons: list[ttk_module.Button] = []
        for button in self._monitor_control_buttons:
            if self._widget_is_alive(button):
                button.configure(state=control_state)
                alive_control_buttons.append(button)
        self._monitor_control_buttons = alive_control_buttons
        log_available = bool(self.run_log_path)
        log_state = (
            tkinter_module.NORMAL if log_available else tkinter_module.DISABLED
        )
        if self._widget_is_alive(self._monitor_log_view_button):
            self._monitor_log_view_button.configure(state=log_state)
        else:
            self._monitor_log_view_button = None
        if self._widget_is_alive(self._monitor_log_open_button):
            self._monitor_log_open_button.configure(state=log_state)
        else:
            self._monitor_log_open_button = None
        output_available = bool(
            self._current_run_output_dir
        ) and os.path.isdir(self._current_run_output_dir)
        if self._widget_is_alive(self._run_output_button):
            self._run_output_button.configure(
                state=(
                    tkinter_module.NORMAL
                    if output_available
                    else tkinter_module.DISABLED
                )
            )
        else:
            self._run_output_button = None

    def show_summary(self) -> None:
        """Display the completion summary with manifest reuse actions."""

        def builder(frame: tkinter_module.Frame) -> None:
            """Render the run summary view with output links and metadata."""
            self._page_header(frame, "Run Summary")
            outputs_label = ttk_module.Label(
                frame,
                text="Outputs",
                takefocus=True,
            )
            outputs_label.pack(anchor="w")
            for link in self.summary.output_links:
                link_label = ttk_module.Label(
                    frame,
                    text=link,
                    takefocus=True,
                )
                link_label.pack(anchor="w")
            actions_label = ttk_module.Label(
                frame,
                text="Manifest actions",
                takefocus=True,
            )
            actions_label.pack(anchor="w", pady=(12, 0))
            for action in self.summary.manifest_actions:
                action_button = ttk_module.Button(
                    frame,
                    text=action,
                    command=self._noop,
                    takefocus=True,
                )
                action_button.pack(anchor="w", pady=2)
            metadata_label = ttk_module.Label(
                frame,
                text="Manifest metadata",
                takefocus=True,
            )
            metadata_label.pack(anchor="w", pady=(12, 0))
            for line in self.summary.manifest_metadata:
                metadata_entry_label = ttk_module.Label(
                    frame,
                    text=line,
                    takefocus=True,
                )
                metadata_entry_label.pack(anchor="w")

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

        confirm_index = self.builder_steps.index(self._CONFIRM_STEP_NAME)
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
        entry = self._model_entry_for_value(candidate)
        if entry:
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
                    or entry.get("stem") == candidate
                    or entry["filename"] == candidate
                ):
                    self._selected_engine_entry = entry
                    return entry
        raise RuntimeError("Select an engine before starting the run.")

    def _prepare_progress_path(self, base_dir: Path | None = None) -> str:
        """Return the path where the GUI stores progress snapshots."""

        directory = (
            base_dir
            if base_dir is not None
            else Path(self._output_root()) / ".progress"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / f"gui_progress_{utils.get_timestamp()}.json")

    def _build_worker_config(self) -> dict:
        """Return the serialized configuration passed to the CLI worker."""

        if self.manifest_workspace is None:
            raise RuntimeError("No saved manifest is available.")
        progress_path = self._prepare_progress_path(
            Path(self.manifest_workspace.folder)
        )
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
            "copernican.lib.gui.run_worker",
            config_path,
        ]
        env = os.environ.copy()
        try:
            process = subprocess.Popen(  # nosec
                command,
                cwd=str(self._repo_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
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
            self._call_on_ui_thread(self._handle_worker_success)
        else:
            self._call_on_ui_thread(self._handle_worker_failure)

    def _handle_worker_success(self) -> None:
        """Refresh UI elements after a successful run."""

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

    def _handle_worker_failure(self) -> None:
        """Refresh UI elements after a failed run."""

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
        if snapshot:
            stage_name = snapshot.get("stage_label")
            if stage_name:
                self._validation_last_stage_label = stage_name
        if snapshot.get("stage_label"):
            self.current_phase = snapshot["stage_label"]
        self._call_on_ui_thread(self._refresh_monitor_widgets)
        self._call_on_ui_thread(self._refresh_validation_progress_widgets)

    def _stage_label_text(self, snapshot: dict | None) -> str:
        """Return the textual stage label derived from the snapshot."""

        stage_label = "Stage: Idle"
        if snapshot:
            label = snapshot.get("stage_label", "Stage")
            event = snapshot.get("event", "").replace("_", " ")
            stage_label = f"{label} – {event}".strip(" –")
        return stage_label

    def _widget_is_alive(self, widget: tkinter_module.Widget | None) -> bool:
        """Return True if the widget has not been destroyed."""

        if widget is None:
            return False
        try:
            return bool(widget.winfo_exists())
        except tkinter_module.TclError:
            return False

    def _call_on_ui_thread(
        self, func: Callable[..., object], *args, **kwargs
    ) -> None:
        """Run *func* on the Tk main loop when rendering is active."""

        if self.render and self.root is not None:
            self.root.after(0, partial(func, *args, **kwargs))
        else:
            func(*args, **kwargs)

    def _refresh_monitor_widgets(self) -> None:
        """Update the progress bars, status label and log console."""

        snapshot = self._progress_snapshot
        stage_label = self._stage_label_text(snapshot)
        if snapshot and snapshot.get("stage_label"):
            self.current_phase = snapshot["stage_label"]
        if self._widget_is_alive(self._progress_status_label):
            self._progress_status_label.configure(text=stage_label)
        else:
            self._progress_status_label = None
        if self._widget_is_alive(self._batch_progressbar):
            percent = snapshot.get("batch_percent", 0) if snapshot else 0
            self._batch_progressbar["value"] = min(max(percent, 0), 100)
        else:
            self._batch_progressbar = None
        if self._widget_is_alive(self._walker_progressbar):
            walker_percent = (
                snapshot.get("walker_percent", 0) if snapshot else 0
            )
            self._walker_progressbar["value"] = min(
                max(walker_percent, 0), 100
            )
        else:
            self._walker_progressbar = None
        self._refresh_status_label()
        self._refresh_run_log_widget()
        self._update_monitor_controls_state()

    def _refresh_validation_progress_widgets(self) -> None:
        """Update the validation progress UI with the latest snapshot."""

        snapshot = self._progress_snapshot
        stage_label = self._stage_label_text(snapshot)
        if self._widget_is_alive(self._validation_progress_status_label):
            self._validation_progress_status_label.configure(text=stage_label)
        else:
            self._validation_progress_status_label = None
        if self._widget_is_alive(self._validation_batch_progressbar):
            percent = snapshot.get("batch_percent", 0) if snapshot else 0
            self._validation_batch_progressbar["value"] = min(
                max(percent, 0), 100
            )
        else:
            self._validation_batch_progressbar = None
        if self._widget_is_alive(self._validation_walker_progressbar):
            walker_percent = (
                snapshot.get("walker_percent", 0) if snapshot else 0
            )
            self._validation_walker_progressbar["value"] = min(
                max(walker_percent, 0), 100
            )
        else:
            self._validation_walker_progressbar = None
        self._update_validation_status_label(snapshot)

    def _update_validation_status_label(
        self, snapshot: dict | None = None
    ) -> None:
        """Keep the validation status label aligned with the base text."""

        if not self._widget_is_alive(self._validation_status_label):
            self._validation_status_label = None
            return
        text = self._validation_status_base
        if self._validation_running:
            stage_snapshot = snapshot
            if stage_snapshot is None and self._validation_last_stage_label:
                stage_snapshot = {
                    "stage_label": self._validation_last_stage_label
                }
            stage_label = self._stage_label_text(stage_snapshot)
            if stage_label and stage_label != "Stage: Idle":
                text = f"{text} – {stage_label}"
        self._validation_status_label.configure(text=text)

    def _refresh_run_log_widget(self) -> None:
        """Populate the run log text widget with the latest entries."""

        if not self._widget_is_alive(self._monitor_log_widget):
            self._monitor_log_widget = None
            return
        entries = self.get_run_log_entries()
        lock_tail = self._monitor_log_lock_var is None or bool(
            self._monitor_log_lock_var.get()
        )
        prev_view = None if lock_tail else self._monitor_log_widget.yview()
        self._monitor_log_widget.configure(state="normal")
        self._monitor_log_widget.delete("1.0", "end")
        for entry in entries[-200:]:
            self._monitor_log_widget.insert(
                "end",
                f"[{entry.anchor}] {entry.formatted}\n",
            )
        self._monitor_log_widget.configure(state="disabled")
        if lock_tail:
            try:
                self._monitor_log_widget.yview_moveto(1.0)
            except (RuntimeError, ValueError, tk_tcl_error):
                pass
        elif prev_view:
            try:
                self._monitor_log_widget.yview_moveto(
                    max(0.0, min(prev_view[0], 1.0))
                )
            except (RuntimeError, ValueError, tk_tcl_error):
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
        except (OSError, RuntimeError, ValueError) as exc:
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
            except (RuntimeError, ValueError, tk_tcl_error):
                pass

    def _terminate_run_process(self, *, force: bool) -> None:
        """Terminate the active worker process if it exists."""

        if self._run_process is None:
            return
        try:
            if force:
                self._run_process.kill()
            else:
                self._run_process.terminate()
        except (OSError, RuntimeError, ValueError) as exc:
            self._log_run_event(
                f"Failed to terminate worker: {exc}", logging.WARNING
            )

    def _repo_root(self) -> Path:
        """Return repository root path."""

        return Path(__file__).resolve().parents[2]

    def _safe_int(self, text_value: str, default: int | None) -> int | None:
        """Return ``int(text_value)`` or ``default`` when parsing fails."""

        try:
            stripped = text_value.strip()
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

    def _collect_engine_setting_values(self) -> dict[str, object]:
        """Return sanitized values entered into engine setting controls."""

        values: dict[str, object] = {}
        for key, setting_var in self._engine_setting_vars.items():
            spec = self._engine_setting_specs.get(key)
            if isinstance(setting_var, tkinter_module.BooleanVar):
                values[key] = bool(setting_var.get())
                continue
            raw_value = setting_var.get().strip()
            if not raw_value:
                continue
            dtype = spec.dtype if spec else "str"
            values[key] = self._parse_engine_setting_value(raw_value, dtype)
        return values

    @staticmethod
    def _parse_engine_setting_value(raw_value: str, dtype: str) -> object:
        """Convert a string knob value into the declared dtype."""

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

    def update_progress(self, progress_value: int) -> None:
        """Update the monitor progress meter."""

        self.progress = max(0, min(100, progress_value))
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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
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
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                yaml.YAMLError,
            ) as exc:
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
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                yaml.YAMLError,
            ) as exc:
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

        if self._validation_running:
            self._notify_pipeline_conflict(
                "Validation running",
                "The pipeline is busy with a validation run. Please wait "
                "for it to finish or cancel it, so a production run can "
                "be started.",
                context="run",
            )
            return

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
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
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
            self.draft.dataset = ", ".join(datasets)
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

        snapshot: dict[str, object] = {}
        try:
            engine_entry = self._resolve_engine_entry()
        except (OSError, RuntimeError, TypeError, ValueError):
            engine_entry = None
        engine_kind = "mcmc"
        if engine_entry:
            try:
                module = importlib.import_module(engine_entry["id"])
                engine_kind = getattr(module, "ENGINE_KIND", "mcmc").lower()
            except (
                AttributeError,
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
            ):
                engine_kind = "mcmc"
            if engine_kind == "mcmc":
                try:
                    snapshot = dict(
                        self._build_sampling_plan_values(engine_entry["id"])
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    snapshot = {}
            else:
                snapshot["engine_kind"] = engine_kind
        knob_settings = self._collect_engine_setting_values()
        if knob_settings:
            snapshot.update(knob_settings)
        if snapshot:
            return snapshot
        fallback_fields = {
            "n_walkers": self.draft.walkers,
            "burn_in_steps": self.draft.burn_in,
            "n_steps": self.draft.production_steps,
            "pool_size": self.draft.pool_size,
        }
        sanitized: dict[str, object] = {}
        for key, field_value in fallback_fields.items():
            if not isinstance(field_value, str):
                continue
            trimmed = field_value.strip()
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
            """Copy a numeric run-setting from the manifest snapshot."""
            setting_value = settings.get(key)
            if setting_value is None:
                setattr(self.draft, field, "")
            else:
                setattr(self.draft, field, str(setting_value))

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
        model_pairs = []
        for model in models:
            entry = self._model_entry_for_value(model)
            if entry:
                model_name = entry.get("id")
                model_filename = entry.get("filename")
            else:
                model_path = Path(model)
                model_name = model
                model_filename = (
                    model_path.name
                    if model_path.suffix.lower() in {".yml", ".yaml"}
                    else f"{model}.yml"
                )
            model_pairs.append(
                (
                    SimpleNamespace(
                        MODEL_NAME=model_name,
                        MODEL_FILENAME=model_filename,
                        PARAMETER_NAMES=[],
                        PARAMETER_PRIORS=[],
                        valid_for_cmb=False,
                    ),
                    "gui",
                )
            )
        datasets: list[dict[str, object]] = []
        source_datasets = self.selected_datasets or [
            {
                "id": self.draft.dataset or "dataset",
                "name": self.draft.dataset or "dataset",
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
