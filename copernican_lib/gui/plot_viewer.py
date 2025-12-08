"""Reusable Matplotlib viewer used across the GUI."""

from __future__ import annotations

import logging
import matplotlib.pyplot as plt

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - GUI disabled
    tk = None
    ttk = None

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PlotViewer(ttk.Frame):
    """A canvas that displays Matplotlib figures with pan/zoom controls."""

    def __init__(self, master, *, figure: Figure | None = None):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is required for the plot viewer.")
        super().__init__(master)
        self.figure: Figure = figure or plt.Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._zoom_active = False
        self._press_event = None
        self._original_limits: dict[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]] = {}
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("draw_event", self._save_original_limits)

    def load_figure(self, figure: Figure) -> None:
        """Replace the current figure with ``figure``."""

        self.figure = figure
        self.canvas.figure = figure
        self._zoom_active = False
        self._press_event = None
        self._save_original_limits(None)
        self.canvas.draw_idle()

    def fit_to_screen(self) -> None:
        """Autoscale all axes to their data limits."""

        for ax in self.figure.axes:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def fit_all(self) -> None:
        """Restore the original limits captured when the figure was drawn."""

        if not self._original_limits:
            self.fit_to_screen()
            return
        for ax in self.figure.axes:
            limits = self._original_limits.get(id(ax))
            if not limits:
                continue
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        self.canvas.draw_idle()

    def toggle_zoom(self) -> None:
        """Toggle the zoom/pan interaction."""

        self._zoom_active = not self._zoom_active
        state = "active" if self._zoom_active else "inactive"
        logger.debug("Plot viewer zoom mode %s", state)

    @property
    def zoom_active(self) -> bool:
        """Return True when zoom/pan interaction is enabled."""

        return self._zoom_active

    def _save_original_limits(self, event) -> None:
        """Remember the axes limits for later restoration."""

        self._original_limits = {}
        for ax in self.figure.axes:
            self._original_limits[id(ax)] = (
                ax.get_xlim(),
                ax.get_ylim(),
            )

    def _on_press(self, event) -> None:
        if not self._zoom_active or event.inaxes is None:
            return
        self._press_event = event

    def _on_release(self, event) -> None:
        if not self._zoom_active:
            return
        self._press_event = None

    def _on_motion(self, event) -> None:
        if not self._zoom_active or self._press_event is None:
            return
        if event.inaxes is None or self._press_event.inaxes is None:
            return
        dx = event.xdata - self._press_event.xdata
        dy = event.ydata - self._press_event.ydata
        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self._press_event = event
        self.canvas.draw_idle()
