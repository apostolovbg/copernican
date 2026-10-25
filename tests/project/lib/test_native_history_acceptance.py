"""Reject misleading native component acceptance evidence."""

import copy
import unittest

from tests.project.lib import native_history_acceptance as acceptance


class NativeHistoryAcceptanceTestCase(unittest.TestCase):
    """Component acceptance requires measured, finite, complete arrays."""

    def test_component_verdict_rejects_missing_spikes_and_claimed_errors(self):
        names = (
            "Phi",
            "Psi",
            "theta_gamma0",
            "delta_b",
            "delta_c",
            "theta_b",
            "theta_c",
            "theta_gamma2",
            "e_gamma2",
            "polarization_source",
            "lensing_potential",
            "delta_nu",
            "temperature_monopole",
            "temperature_doppler",
            "temperature_quadrupole",
            "temperature_isw",
        )
        case = {
            "actual": {name: [1.0, 0.0, -1.0] for name in names},
            "reference": {name: [1.0, 0.0, -1.0] for name in names},
        }
        report = {
            "active_neutrino_density": "delta_nu",
            "eta": [10.0, 100.0, 1000.0],
            "cases": {
                "0.0001": copy.deepcopy(case),
                "0.02": copy.deepcopy(case),
            },
        }
        self.assertTrue(acceptance.assess_native_histories(report))
        fine = copy.deepcopy(report)
        self.assertTrue(acceptance.assess_native_refinement(report, fine))
        fine["cases"]["0.02"]["actual"]["Phi"][1] = 0.006
        self.assertTrue(acceptance.assess_native_histories(fine))
        self.assertFalse(acceptance.assess_native_refinement(report, fine))
        fine = copy.deepcopy(report)
        fine["reference_contract"] = {"param_map": {"H0": 70.0}}
        self.assertFalse(acceptance.assess_native_refinement(report, fine))
        for change in ("missing", "spike", "nonfinite", "wrong_shape"):
            invalid = copy.deepcopy(report)
            values = invalid["cases"]["0.02"]["actual"]
            if change == "missing":
                values.pop("polarization_source")
            elif change == "spike":
                values["e_gamma2"][1] = 0.03
            elif change == "nonfinite":
                values["Phi"][1] = float("nan")
            else:
                values["Psi"] = [1.0]
            invalid["cases"]["0.02"]["errors"] = {name: 0.0 for name in names}
            with self.subTest(change=change):
                self.assertFalse(acceptance.assess_native_histories(invalid))
        invalid = copy.deepcopy(report)
        invalid["cases"].pop("0.0001")
        self.assertFalse(acceptance.assess_native_histories(invalid))
        for bound in (0.0, 0.021, float("nan")):
            with self.assertRaises(ValueError):
                acceptance.assess_native_histories(report, bound=bound)


if __name__ == "__main__":
    unittest.main()
