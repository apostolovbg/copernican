"""Independent physical distribution and metadata regressions."""

import unittest
from types import SimpleNamespace

import numpy

from copernican.lib.phase_space import (
    phase_space_grid_name,
    phase_space_temperature,
    plan_distribution_grid,
    sample_distribution,
    validate_phase_space,
)


class PhaseSpaceTestCase(unittest.TestCase):
    """Distribution laws, derivatives and units must survive renamed grids."""

    def test_validation_rejects_unbound_laws_before_numerical_execution(self):
        law = {
            "distribution": "fermi_dirac",
            "mass_parameter": "mass",
            "temperature_ratio": "(4/11)**(1/3)*(heating/3)**0.25",
        }
        with self.assertRaisesRegex(ValueError, "undeclared names: heating"):
            validate_phase_space(law, available_parameters={"mass"})
        self.assertEqual(
            validate_phase_space(
                law, available_parameters={"mass", "heating"}
            ),
            law,
        )
        with self.assertRaisesRegex(
            ValueError, "mass_parameter is undeclared"
        ):
            validate_phase_space(law, available_parameters={"heating"})
        expression = {
            "mass_parameter": "mass",
            "temperature_ratio": "ratio",
            "occupation": "exp(-q / width)",
            "log_derivative": "-q / width",
        }
        self.assertEqual(
            validate_phase_space(
                expression, available_parameters={"mass", "ratio", "width"}
            )["distribution"],
            "expression",
        )
        expression["temperature_ratio"] = "q"
        with self.assertRaisesRegex(ValueError, "undeclared names: q"):
            validate_phase_space(
                expression, available_parameters={"mass", "ratio", "width"}
            )

    def test_occupation_and_derivative_match_independent_differences(self):
        q_values = numpy.geomspace(0.01, 100, 81)
        for name in ("fermi_dirac", "bose_einstein", "maxwell_boltzmann"):
            law = {"distribution": name}
            occupation, derivative = sample_distribution(q_values, law)
            plus = sample_distribution(q_values * numpy.exp(1e-5), law)[0]
            minus = sample_distribution(q_values * numpy.exp(-1e-5), law)[0]
            numerical = (numpy.log(plus) - numpy.log(minus)) / 2e-5
            numpy.testing.assert_allclose(derivative, numerical, rtol=1e-8)
            self.assertTrue(numpy.all(occupation > 0))

    def test_expression_law_retains_shifted_physical_feature(self):
        law = validate_phase_space(
            {
                "occupation": "exp(-(q-center)**2 / width**2)",
                "log_derivative": "-2*q*(q-center) / width**2",
                "temperature_ratio": "ratio",
                "mass_parameter": "mass",
            }
        )
        values, derivative = sample_distribution(
            (39.0, 40.0, 41.0), law, {"center": 40.0, "width": 0.5}
        )
        self.assertEqual(values[1], 1.0)
        self.assertEqual(derivative[1], 0.0)
        self.assertLess(values[0], 0.02)
        self.assertAlmostEqual(
            phase_space_temperature(law, {"ratio": 0.5}, 2.0),
            8.617333262145e-5,
        )
        self.assertEqual(phase_space_grid_name("renamed", law), "renamed_q")
        self.assertEqual(
            phase_space_grid_name(
                "renamed", SimpleNamespace(momentum_grid="physical_q")
            ),
            "physical_q",
        )

    def test_missing_laws_and_invalid_values_fail_explicitly(self):
        for declaration in (
            {},
            {"distribution": "unknown"},
            {"distribution": "fermi_dirac", "occupation": "exp(-q)"},
        ):
            with self.assertRaises(ValueError):
                validate_phase_space(declaration)
        with self.assertRaises(ValueError):
            sample_distribution((-1.0,), {"distribution": "fermi_dirac"})
        with self.assertRaises(ValueError):
            phase_space_temperature({"temperature_ratio": "-1"}, {}, 2.0)

    def test_relativistic_density_is_exclusive_and_declared(self):
        """Kinetic abundance has one explicit physical normalization."""
        law = {
            "distribution": "fermi_dirac",
            "mass_parameter": "mass",
            "temperature_ratio": "ratio",
            "relativistic_density": "early_density",
        }
        self.assertEqual(
            validate_phase_space(
                law, available_parameters={"mass", "ratio", "early_density"}
            ),
            law,
        )
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_phase_space(law, available_parameters={"mass", "ratio"})
        with self.assertRaisesRegex(ValueError, "one density"):
            validate_phase_space({**law, "density_parameter": "today"})

    def test_support_planning_finds_a_nonthermal_peak_beyond_neutrino_tail(
        self,
    ):
        law = {
            "occupation": "exp(-(q-center)**2 / width**2)",
            "log_derivative": "-2*q*(q-center)/width**2",
        }
        grid = plan_distribution_grid(law, {"center": 40.0, "width": 0.5})
        self.assertGreater(grid["q_max"], 40.0)
        self.assertTrue(grid["planning_evidence"]["converged"])
        # moments must resolve independent density and expanded support.
        self.assertGreater(grid["count"], 32)
