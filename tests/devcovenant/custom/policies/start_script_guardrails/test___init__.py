"""Mirror test for the start-script guardrails package marker."""

from __future__ import annotations

import unittest

import devcovenant.custom.policies.start_script_guardrails as guardrails_package


class StartScriptGuardrailsPackageTest(unittest.TestCase):
    """Confirm the package marker module imports cleanly."""

    def test_package_imports(self) -> None:
        """The package marker module should remain importable."""
        self.assertEqual(
            guardrails_package.__name__,
            "devcovenant.custom.policies.start_script_guardrails",
        )

