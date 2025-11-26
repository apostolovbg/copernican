"""GUI scaffolding package for the Copernican Suite.

The GUI components live in a dedicated package so the command-line shim can
expose a stable import path without loading any graphical dependencies until
users explicitly request the interface.  Keeping the surface thin also makes
it easier to stub the GUI in headless environments while still exporting
consistent entry points for integration tests.
"""

# Rationale: The GUI exports live here because consolidating widget setup keeps
# the CLI shim lightweight while still exposing a stable entry point for
# graphical runners.

from copernican_lib.gui.app import CopernicanGUI, RunStatus

__all__ = ["CopernicanGUI", "RunStatus"]
