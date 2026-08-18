"""Tests for the canonical CCMBS solver identity."""

import unittest

from copernican.lib import cmb_identity


class CMBIdentityTestCase(unittest.TestCase):
    """Keep user-facing CMB identity stable and singular."""

    def test_native_identity_is_public_and_unambiguous(self) -> None:
        """Expose one CCMBS identifier and label."""

        self.assertEqual(
            cmb_identity.CCMBS_ID,
            "ccmbs_numpy",
        )
        self.assertEqual(
            cmb_identity.CCMBS_LABEL,
            "CCMBS — Copernican Cosmic Microwave Background Solver",
        )
        self.assertEqual(
            cmb_identity.__all__,
            ["CCMBS_ID", "CCMBS_LABEL"],
        )


if __name__ == "__main__":
    unittest.main()
