"""Tests for the canonical production CMB engine identity."""

import unittest

from copernican.lib import cmb_identity


class CMBIdentityTestCase(unittest.TestCase):
    """Keep user-facing CMB identity stable and singular."""

    def test_native_identity_is_public_and_unambiguous(self) -> None:
        """Expose one native engine identifier and label."""

        self.assertEqual(
            cmb_identity.NATIVE_CMB_ENGINE_ID,
            "copernican_native_declared_graph",
        )
        self.assertEqual(
            cmb_identity.NATIVE_CMB_ENGINE_LABEL,
            "Copernican native declared-graph CMB engine",
        )
        self.assertEqual(
            cmb_identity.__all__,
            ["NATIVE_CMB_ENGINE_ID", "NATIVE_CMB_ENGINE_LABEL"],
        )


if __name__ == "__main__":
    unittest.main()
