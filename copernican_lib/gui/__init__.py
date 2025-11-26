"""GUI scaffolding package for the Copernican Suite."""

# Rationale: The GUI exports live here because consolidating widget setup keeps
# the CLI shim lightweight while still exposing a stable entry point for
# graphical runners.

from copernican_lib.gui.app import CopernicanGUI, RunStatus

__all__ = ["CopernicanGUI", "RunStatus"]
