"""Mirror test for the start-script parity package marker."""

from __future__ import annotations

import unittest

import devcovenant.custom.policies.start_script_parity as parity_package


class StartScriptParityPackageTest(unittest.TestCase):
    """Confirm the package marker module imports cleanly."""

    def test_package_imports(self) -> None:
        """The package marker module should remain importable."""
        self.assertEqual(
            parity_package.__name__,
            "devcovenant.custom.policies.start_script_parity",
        )

