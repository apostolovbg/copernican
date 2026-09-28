"""Coverage for CMB solver registration and default resolution."""

import unittest

from copernican.lib.likelihoods.cmb.errors import UnsupportedCapabilityError
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

        self.assertEqual(available_cmb_solvers(), ("ccmbs",))
        self.assertEqual(
            resolve_cmb_solver("ccmbs").solver_id,
            "ccmbs",
        )
        solver = get_cmb_solver("ccmbs")
        self.assertEqual(solver_provenance(solver)["solver_id"], "ccmbs")
        self.assertTrue(callable(register_cmb_solver))

    def test_backend_names_are_not_public_solver_choices(self):
        """CPU and accelerator implementation names cannot be selected."""

        with self.assertRaises(UnsupportedCapabilityError):
            resolve_cmb_solver("ccmbs_numpy")
        with self.assertRaises(UnsupportedCapabilityError):
            resolve_cmb_solver("ccmbs_taichi")

    def test_backend_fields_are_not_public_contract_controls(self):
        """Manifest backend controls are rejected at the public boundary."""

        with self.assertRaises(UnsupportedCapabilityError):
            resolve_cmb_solver({"id": "ccmbs", "backend": "taichi"})


if __name__ == "__main__":
    unittest.main()
