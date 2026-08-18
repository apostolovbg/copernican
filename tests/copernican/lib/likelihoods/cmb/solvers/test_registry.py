"""Coverage for CMB solver registration and default resolution."""

import unittest

from copernican.lib.likelihoods.cmb.solvers.registry import (
    available_cmb_solvers,
    get_cmb_solver,
    register_cmb_solver,
    resolve_cmb_solver,
    solver_provenance,
)


class TestSolverRegistry(unittest.TestCase):
    """Check default registry discovery and identity resolution."""

    def test_default_solver_is_registered(self):
        """CCMBS is available without an eager backend import."""

        self.assertIn("ccmbs_numpy", available_cmb_solvers())
        self.assertEqual(
            resolve_cmb_solver("ccmbs_numpy").solver_id,
            "ccmbs_numpy",
        )
        solver = get_cmb_solver("ccmbs_numpy")
        self.assertEqual(solver_provenance(solver)["solver_id"], "ccmbs_numpy")
        self.assertTrue(callable(register_cmb_solver))


if __name__ == "__main__":
    unittest.main()
