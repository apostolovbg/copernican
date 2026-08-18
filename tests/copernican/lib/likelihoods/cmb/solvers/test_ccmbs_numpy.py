"""Coverage for the NumPy/SciPy CCMBS reference adapter."""

import unittest

from copernican.lib.likelihoods.cmb.solvers.ccmbs_numpy import CCMBSNumpySolver


class TestCCMBSNumpySolver(unittest.TestCase):
    """Check stable identity and capability metadata for CCMBS."""

    def test_capabilities_and_lifecycle(self):
        """The reference adapter exposes a CPU capability probe."""

        solver = CCMBSNumpySolver()
        capabilities = solver.capabilities()
        self.assertEqual(solver.solver_id, "ccmbs_numpy")
        self.assertEqual(capabilities["execution_backend"], "cpu")
        self.assertFalse(capabilities["device_probe"]["taichi_imported"])
        self.assertTrue(callable(solver.prepare))
        self.assertTrue(callable(solver.evaluate))
        self.assertTrue(callable(solver.evaluate_batch))
        solver.cleanup()


if __name__ == "__main__":
    unittest.main()
