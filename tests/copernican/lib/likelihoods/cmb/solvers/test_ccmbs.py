"""Coverage for the single public CCMBS engine boundary."""

import unittest
from unittest import mock

from copernican.lib.likelihoods.cmb.solvers.ccmbs import CCMBS


class TestCCMBSEngine(unittest.TestCase):
    """Check public identity and internal backend selection evidence."""

    def test_engine_owns_backend_selection(self):
        """The public engine reports CPU fallback provenance."""

        solver = CCMBS()
        with mock.patch(
            "copernican.lib.likelihoods.cmb.solvers.ccmbs.taichi_device_probe",
            return_value={"backend": "taichi", "taichi_installed": False},
        ):
            capabilities = solver.capabilities()
        self.assertEqual(capabilities["solver_id"], "ccmbs")
        self.assertEqual(capabilities["execution_backend"], "cpu")
        provenance = capabilities["device_probe"]
        self.assertEqual(provenance["selected_backend"], "numpy_cpu")
        self.assertTrue(provenance["fallback"])
        self.assertEqual(
            provenance["fallback_reason"],
            "accelerator_unavailable_or_not_installed",
        )


if __name__ == "__main__":
    unittest.main()
