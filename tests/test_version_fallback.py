"""Regression tests for the resilient version helpers."""

from __future__ import annotations

import importlib
import unittest


class VersionFallbackTest(unittest.TestCase):
    """Ensure launchers continue working when version helpers are missing."""

    def setUp(self) -> None:  # noqa: D401
        """Record the original ``get_version`` for restoration."""

        import copernican_lib.version as version_module  # local import

        self.version_module = version_module
        self.original_getter = getattr(version_module, "get_version", None)

    def tearDown(self) -> None:  # noqa: D401
        """Restore ``get_version`` and reload dependants."""

        if self.original_getter is not None:
            self.version_module.get_version = self.original_getter
        else:  # pragma: no cover - defensive branch
            if hasattr(self.version_module, "get_version"):
                delattr(self.version_module, "get_version")
        import copernican_lib.plotter as plotter  # local import
        import copernican_lib.run_manifest as run_manifest  # local import

        importlib.reload(plotter)
        importlib.reload(run_manifest)

    def test_manifest_fallback_when_get_version_missing(self) -> None:
        """Manifest generation defaults to ``"0+unknown"`` when necessary."""

        import copernican_lib.run_manifest as run_manifest  # local import

        if hasattr(self.version_module, "get_version"):
            delattr(self.version_module, "get_version")
        importlib.reload(run_manifest)
        manifest = run_manifest.build_manifest([], object(), [])
        self.assertEqual(manifest["copernican"]["version"], "0+unknown")

    def test_plotter_constant_uses_safe_version_lookup(self) -> None:
        """Plotter initialisation tolerates missing ``get_version``."""

        import copernican_lib.plotter as plotter  # local import

        if hasattr(self.version_module, "get_version"):
            delattr(self.version_module, "get_version")
        importlib.reload(plotter)
        self.assertEqual(plotter.COPERNICAN_VERSION, "0+unknown")


__all__ = ["VersionFallbackTest"]
