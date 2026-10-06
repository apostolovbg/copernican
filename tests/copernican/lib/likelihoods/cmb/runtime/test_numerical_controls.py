"""Direct tests for declared numerical grid and work controls."""

import unittest

from copernican.lib.likelihoods.cmb.runtime import numerical_controls


class NumericalControlsTestCase(unittest.TestCase):
    """Exercise numerical-control decisions without spectrum orchestration."""

    def test_runtime_work_estimate_is_deterministic_and_accounted(self):
        """Large bounded requests are accounted for, not rejected."""

        contract = {
            "perturbations": {
                "accuracy_controls": {"runtime_envelope": "bounded"}
            }
        }
        arguments = {
            "ell_count": 2500,
            "k_count": 2048,
            "eta_count": 2048,
            "state_slot_count": 64,
            "transfer_component_count": 3,
            "momentum_point_count": 256,
            "evolution_multiplier": 3,
        }
        first = numerical_controls._enforce_runtime_envelope(
            contract, **arguments
        )
        second = numerical_controls._enforce_runtime_envelope(
            contract, **arguments
        )
        self.assertEqual(first, second)
        self.assertEqual(first["work_accounting_mode"], "accounted")
        self.assertEqual(first["work_limits"], {})
        self.assertFalse(first["work_limits_enforced"])
        self.assertGreater(first["total_work_units"], 100_000_000)

    def test_explicit_work_limit_is_metadata_not_a_machine_ceiling(self):
        """A valid request is not rejected by an operator work hint."""

        contract = {
            "perturbations": {
                "accuracy_controls": {
                    "runtime_envelope": {
                        "maximum_total_work_units": 1,
                    }
                }
            }
        }
        envelope = numerical_controls._enforce_runtime_envelope(
            contract,
            ell_count=100,
            k_count=100,
            eta_count=100,
            state_slot_count=8,
            transfer_component_count=2,
            momentum_point_count=0,
        )
        self.assertEqual(
            envelope["work_limits"], {"maximum_total_work_units": 1}
        )
        self.assertFalse(envelope["work_limits_enforced"])

    def test_evolution_chunk_size_is_deterministic(self):
        """Evolution chunking is derived from declared array dimensions."""

        first = numerical_controls._resolve_evolution_chunk_size(
            k_count=2048,
            eta_count=2048,
            state_slot_count=64,
        )
        second = numerical_controls._resolve_evolution_chunk_size(
            k_count=2048,
            eta_count=2048,
            state_slot_count=64,
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first * 2048 * 64, 16_000_000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
