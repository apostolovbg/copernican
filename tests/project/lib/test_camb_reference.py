"""Focused tests for the independent CAMB reference helper."""

import unittest
from pathlib import Path

from tests.project.lib import camb_reference


class CambReferenceModuleTestCase(unittest.TestCase):
    """Exercise the independent CAMB reference helper surface."""

    def test_describe_camb_configuration_reports_expected_defaults(self):
        """The reference helper should report its default CAMB settings."""

        configuration = camb_reference.describe_camb_configuration()

        self.assertEqual(
            configuration["reionization_model"], "optical_depth_tau"
        )
        self.assertIn("lmax_padding", configuration)
        self.assertIn("lens_potential_accuracy", configuration)
        self.assertIn("accuracy", configuration)
        self.assertEqual(
            configuration["reference_identity"],
            f"camb:{camb_reference.camb.__version__}",
        )

    def test_reference_helper_is_test_owned(self):
        """The CAMB builder should remain outside the production package."""

        helper_path = Path(camb_reference.__file__).resolve()
        self.assertTrue(helper_path.is_relative_to(Path("tests").resolve()))
        self.assertEqual(
            camb_reference.CAMB_REFERENCE_IDENTITY,
            f"camb:{camb_reference.camb.__version__}",
        )

    def test_reference_symbols_are_exposed(self):
        """The test module should expose independent reference entrypoints."""

        self.assertIn(
            "compute_cmb_spectrum_from_camb_contract", camb_reference.__all__
        )
        self.assertIn(
            "compute_camb_background_observables", camb_reference.__all__
        )
        self.assertIn("describe_camb_configuration", camb_reference.__all__)
        self.assertIn("CAMB_REFERENCE_IDENTITY", camb_reference.__all__)
        self.assertIn("build_lcdm_reference_fixture", camb_reference.__all__)
        self.assertIn("reference_fixture_sha256", camb_reference.__all__)

    def test_fixed_lcdm_fixture_is_self_describing(self):
        """The frozen fixture records arrays, conventions, and its digest."""

        fixture = camb_reference.build_lcdm_reference_fixture(
            (2, 20, 100),
            spectra=("TT", "TE", "EE"),
        )

        self.assertEqual(fixture["ell_values"], (2, 20, 100))
        self.assertEqual(set(fixture["spectra"]), {"TT", "TE", "EE"})
        self.assertEqual(
            fixture["fixture_sha256"],
            camb_reference.reference_fixture_sha256(
                {
                    key: value
                    for key, value in fixture.items()
                    if key != "fixture_sha256"
                }
            ),
        )
        self.assertEqual(
            fixture["normalization"],
            "unlensed_scalar_D_ell_microkelvin_squared",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
