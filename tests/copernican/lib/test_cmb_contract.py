"""Smoke tests for :mod:`copernican.lib.cmb_contract`."""

from __future__ import annotations

import unittest

from copernican.lib import cmb_contract


class TestCMBContractExports(unittest.TestCase):
    """Verify the contract module mirrors the native adapter helpers."""

    def test_reexports_match_expected_helpers(self) -> None:
        """The route-neutral evaluators should be the only public surface."""

        self.assertTrue(hasattr(cmb_contract, "CMBContractEvaluator"))
        self.assertTrue(hasattr(cmb_contract, "CMBParameterEvaluator"))
        self.assertFalse(hasattr(cmb_contract, "CMB_BACKEND_CAPABILITIES"))
        self.assertTrue(
            hasattr(cmb_contract, "_validate_cmb_contract_definition")
        )
        self.assertIn("CMBContractEvaluator", cmb_contract.__all__)
        self.assertNotIn("CMB_BACKEND_CAPABILITIES", cmb_contract.__all__)
        self.assertIn(
            "_validate_cmb_contract_definition",
            cmb_contract.__all__,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
