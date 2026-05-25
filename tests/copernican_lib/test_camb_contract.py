"""Smoke tests for copernican_lib.camb_contract."""

from __future__ import annotations

import unittest

from copernican_lib import camb_contract


class TestCAMBContractExports(unittest.TestCase):
    """Verify the re-export module mirrors the adapter helpers."""

    def test_reexports_match_expected_helpers(self) -> None:
        self.assertTrue(hasattr(camb_contract, "CAMBContractEvaluator"))
        self.assertTrue(hasattr(camb_contract, "CAMBParameterEvaluator"))
        self.assertTrue(
            hasattr(camb_contract, "_validate_camb_contract_definition")
        )
        self.assertIn("CAMBContractEvaluator", camb_contract.__all__)
        self.assertIn(
            "_validate_camb_contract_definition",
            camb_contract.__all__,
        )


if __name__ == "__main__":
    unittest.main()
