"""Focused tests for the independent CAMB reference helper."""

import unittest
from pathlib import Path

import numpy

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
        self.assertIn(
            "build_lcdm_full_reference_fixture", camb_reference.__all__
        )
        self.assertIn("compare_lcdm_reference_spectra", camb_reference.__all__)
        self.assertIn(
            "load_lcdm_full_reference_fixture", camb_reference.__all__
        )
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

    def test_full_fixture_is_frozen_complete_and_reversible(self):
        """The tracked fixture covers scalar observables in both units."""

        fixture = camb_reference.load_lcdm_full_reference_fixture()
        spectra = fixture["spectra"]
        expected_names = set(camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA)
        self.assertEqual(set(spectra), expected_names)
        self.assertEqual(
            tuple(fixture["declared_observables"]),
            camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA,
        )
        self.assertEqual(
            tuple(fixture["ell_values"]),
            (2, 20, 100, 200, 500, 1000, 1500, 2000),
        )
        self.assertEqual(fixture["applicability"]["scalar"]["omitted"], [])
        self.assertEqual(fixture["applicability"]["vector"]["included"], [])
        self.assertEqual(fixture["applicability"]["tensor"]["included"], [])
        ell = numpy.asarray(fixture["ell_values"], dtype=float)
        for name, values in spectra.items():
            self.assertEqual(set(values), {"C_ell", "D_ell"})
            raw = numpy.asarray(values["C_ell"], dtype=float)
            native = numpy.asarray(values["D_ell"], dtype=float)
            self.assertTrue(numpy.all(numpy.isfinite(raw)))
            self.assertTrue(numpy.all(numpy.isfinite(native)))
            if name == "PP":
                factor = ell**2 * (ell + 1.0) ** 2 / (2.0 * numpy.pi)
            elif name in {"TP", "EP"}:
                factor = (ell * (ell + 1.0)) ** 1.5 / (2.0 * numpy.pi)
            else:
                factor = ell * (ell + 1.0) / (2.0 * numpy.pi)
            numpy.testing.assert_allclose(
                native,
                raw * factor,
                rtol=1.0e-12,
                atol=1.0e-30,
            )

    def test_full_builder_matches_frozen_fixture(self):
        """The frozen arrays are reproducible from the pinned CAMB contract."""

        frozen = camb_reference.load_lcdm_full_reference_fixture()
        rebuilt = camb_reference.build_lcdm_full_reference_fixture()
        self.assertEqual(rebuilt["fixture_sha256"], frozen["fixture_sha256"])
        for name in camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA:
            for representation in ("C_ell", "D_ell"):
                numpy.testing.assert_allclose(
                    rebuilt["spectra"][name][representation],
                    frozen["spectra"][name][representation],
                    rtol=1.0e-12,
                    atol=1.0e-30,
                )

    def test_direct_helper_supports_every_frozen_observable(self):
        """The direct reference API exposes the complete scalar surface."""

        fixture = camb_reference.load_lcdm_full_reference_fixture()
        actual = camb_reference.compute_cmb_spectrum_from_camb_contract(
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
            fixture["ell_values"],
            spectra=camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA,
        )
        self.assertIsInstance(actual, dict)
        self.assertEqual(
            set(actual), set(camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA)
        )
        metrics = camb_reference.compare_lcdm_reference_spectra(
            actual,
            fixture["spectra"],
        )
        self.assertTrue(
            all(
                metric["max_fractional"] < 1.0e-12
                for metric in metrics.values()
            )
        )

    def test_full_comparator_requires_aligned_finite_arrays(self):
        """Comparison reports enforce shape and finite aligned arrays."""

        frozen = camb_reference.load_lcdm_full_reference_fixture()
        actual = {
            name: values["D_ell"] for name, values in frozen["spectra"].items()
        }
        metrics = camb_reference.compare_lcdm_reference_spectra(
            actual,
            frozen["spectra"],
        )
        self.assertEqual(
            set(metrics), set(camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA)
        )
        self.assertTrue(
            all(metric["max_fractional"] == 0.0 for metric in metrics.values())
        )
        with self.assertRaises(ValueError):
            camb_reference.compare_lcdm_reference_spectra(
                {"TT": [1.0]},
                {"TT": frozen["spectra"]["TT"]},
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
