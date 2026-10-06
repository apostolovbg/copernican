"""Direct tests for declared constraint-history validation."""

import unittest
from types import SimpleNamespace

import numpy

from copernican.lib.likelihoods.cmb.runtime import constraint_validation


class ConstraintValidationTestCase(unittest.TestCase):
    """Exercise constraint evidence independently from mode execution."""

    def test_scalar_constraint_acceptance_is_resolution_aware(
        self,
    ) -> None:
        """Reference tolerances apply only to sufficiently resolved grids."""

        context = {
            "einstein_energy_residual": numpy.full(8, 5.0e-3),
        }
        controls = {
            "scalar_constraint_reference_eta_samples": 16,
            "scalar_constraint_tolerances": {
                "einstein_energy_residual": 1.0e-3,
            },
        }
        diagnostics = (
            constraint_validation._validate_scalar_constraint_histories(
                perturbation_data=SimpleNamespace(conservation_rules={}),
                context=context,
                eta_grid=numpy.arange(8, dtype=float),
                accuracy_controls=controls,
                k_value=0.1,
            )
        )

        self.assertFalse(diagnostics["einstein_energy_residual"]["enforced"])
        self.assertFalse(
            diagnostics["einstein_energy_residual"]["reference_resolution_met"]
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["resolution_status"],
            "under_resolved",
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["physical_judgement"],
            "deferred",
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["normalization_source"],
            "residual_magnitude_fallback",
        )
        self.assertEqual(
            diagnostics["einstein_energy_residual"]["tolerance_kind"],
            "normalized",
        )
        self.assertGreater(
            float(diagnostics["einstein_energy_residual"]["maximum_absolute"]),
            float(diagnostics["einstein_energy_residual"]["tolerance"]),
        )

        controls["scalar_constraint_reference_eta_samples"] = 8
        with self.assertRaisesRegex(
            ValueError,
            "Scalar Einstein constraint exceeded tolerance",
        ):
            constraint_validation._validate_scalar_constraint_histories(
                perturbation_data=SimpleNamespace(conservation_rules={}),
                context=context,
                eta_grid=numpy.arange(8, dtype=float),
                accuracy_controls=controls,
                k_value=0.1,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
