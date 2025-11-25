# Last Updated: 2025-11-25
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

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed only when Tk is missing
    tk = None
    ttk = None

from copernican_lib import console_output


class RunStatus(Enum):
    """Enumerate the run lifecycle states shown in the status strip."""

    IDLE = "Idle"
    CONFIGURING = "Configuring"
    RUNNING = "Running"
    PAUSED = "Paused"
    CANCELLED = "Cancelled"


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
        self._build_navigation()
        self._initialise_rendering()

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
                "run_builder", "Run Builder", "Ctrl+2",
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

        self._swap_content(builder)

    def show_data_overview(self) -> None:
        """Display placeholder data catalogue information."""

        def builder(frame: tk.Frame) -> None:
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
            ttk.Label(frame, text="Settings", takefocus=True).pack(anchor="w")
            ttk.Label(
                frame,
                text=(
                    "Adjust notification and logging preferences before "
                    "launching runs."
                ),
                wraplength=720,
                takefocus=True,
            ).pack(anchor="w", pady=(4, 0))

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
            controls = ttk.Frame(frame)
            controls.pack(anchor="w")
            ttk.Button(
                controls,
                text="Cancel",
                command=self.cancel_run,
                takefocus=True,
            ).pack(side="left", padx=4)
            ttk.Button(
                controls, text="Stop", command=self.stop_run, takefocus=True
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
            ttk.Label(
                frame, text="Manifest actions", takefocus=True
            ).pack(anchor="w", pady=(12, 0))
            for action in self.summary.manifest_actions:
                ttk.Button(
                    frame, text=action, command=self._noop, takefocus=True
                ).pack(anchor="w", pady=2)

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

        self.status = RunStatus.RUNNING
        self.progress = 0
        self.show_run_monitor()

    def update_progress(self, value: int) -> None:
        """Update the monitor progress meter."""

        self.progress = max(0, min(100, value))
        if self.progress >= 100:
            self.status = RunStatus.IDLE
            self.summary.output_links = ["/output/latest/chains.nc"]
            self.summary.manifest_actions = [
                "Clone manifest",
                "Open output directory",
            ]
            self.show_summary()

    def cancel_run(self) -> None:
        """Mark the run as cancelled and reset the progress."""

        self.status = RunStatus.CANCELLED
        self.progress = 0

    def stop_run(self) -> None:
        """Stop the run while keeping the monitor visible."""

        self.status = RunStatus.PAUSED

    def _noop(self) -> None:
        """Placeholder callback for summary actions."""

        return None

    def run(self) -> None:
        """Start the Tk main loop when rendering is enabled."""

        if self.render and self.root is not None:
            self.root.mainloop()


__all__ = ["CopernicanGUI", "RunStatus", "RunDraft", "RunSummary"]
