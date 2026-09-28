"""Coverage for the single CCMBS engine's CPU execution path."""

import unittest
from unittest import mock

from copernican.lib.likelihoods.cmb.solvers.ccmbs import CCMBS
from copernican.lib.likelihoods.cmb.solvers.ccmbs_numpy import CCMBSCPUBackend


class TestCCMBS(unittest.TestCase):
    """Check one public identity and internal CPU provenance."""

    def test_capabilities_and_lifecycle(self):
        """The reference adapter exposes a CPU capability probe."""

        solver = CCMBS()
        capabilities = solver.capabilities()
        self.assertEqual(solver.solver_id, "ccmbs")
        self.assertEqual(capabilities["execution_backend"], "cpu")
        self.assertEqual(
            capabilities["device_probe"]["selected_backend"], "numpy_cpu"
        )
        self.assertTrue(callable(solver.prepare))
        self.assertTrue(callable(solver.evaluate))
        self.assertTrue(callable(solver.evaluate_batch))
        solver.cleanup()

    def test_cpu_backend_symbols_remain_internal_and_callable(self):
        """The CPU implementation is an internal CCMBS backend."""

        self.assertEqual(CCMBSCPUBackend.solver_id, "ccmbs")
        self.assertTrue(callable(CCMBSCPUBackend.capabilities))

    def test_unvalidated_accelerator_stays_inside_ccmbs(self):
        """Available hardware does not become a second public solver."""

        solver = CCMBS()
        with mock.patch(
            "copernican.lib.likelihoods.cmb.solvers.ccmbs.taichi_device_probe",
            return_value={"backend": "taichi", "taichi_installed": True},
        ):
            capabilities = solver.capabilities()
        self.assertEqual(capabilities["solver_id"], "ccmbs")
        self.assertEqual(capabilities["execution_backend"], "cpu")
        self.assertEqual(
            capabilities["device_probe"]["selected_backend"], "numpy_cpu"
        )
        self.assertTrue(capabilities["device_probe"]["fallback"])


if __name__ == "__main__":
    unittest.main()
