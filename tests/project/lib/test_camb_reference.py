"""Focused tests for the independent CAMB reference helper."""

import unittest
from copy import deepcopy
from pathlib import Path

import numpy

from copernican.lib.likelihoods.cmb import diagnostics as cmb_diagnostics
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
        self.assertIn(
            "build_camb_full_reference_fixture", camb_reference.__all__
        )
        self.assertIn(
            "build_camb_parity_reference_set", camb_reference.__all__
        )
        self.assertIn("compare_lcdm_reference_spectra", camb_reference.__all__)
        self.assertIn("compare_scalar_history_waves", camb_reference.__all__)
        self.assertIn(
            "load_lcdm_full_reference_fixture", camb_reference.__all__
        )
        self.assertIn("reference_fixture_sha256", camb_reference.__all__)

    def test_scalar_time_evolution_returns_aligned_finite_histories(self):
        """The independent scalar-history reference preserves grid shape."""

        eta_values = numpy.asarray((10.0, 100.0, 1000.0), dtype=float)
        histories = camb_reference.compute_camb_scalar_time_evolution(
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
            0.02,
            eta_values,
            variables=("delta_photon", "v_photon", "T_source"),
        )

        self.assertEqual(
            set(histories),
            {"delta_photon", "v_photon", "T_source"},
        )
        for values in histories.values():
            self.assertEqual(values.shape, eta_values.shape)
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_scalar_history_wave_comparator_records_phase_evidence(self):
        """Wave comparison retains finite phase and residual diagnostics."""

        eta_values = numpy.linspace(0.0, 40.0, 401)
        reference = {"reference_wave": numpy.sin(eta_values)}
        actual = {"declared_wave": 2.5 * numpy.sin(eta_values + 0.1) + 3.0}
        report = camb_reference.compare_scalar_history_waves(
            actual,
            reference,
            eta_values,
            variable_map={"declared_wave": "reference_wave"},
            k_value=0.02,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["finite"])
        self.assertTrue(report["phase_coherent"])
        self.assertEqual(report["wave_count"], 1)
        metric = report["variables"]["declared_wave"]
        self.assertGreater(metric["actual_zero_crossings"], 0)
        self.assertGreater(metric["reference_zero_crossings"], 0)
        self.assertGreater(metric["phase_correlation"], 0.9)
        self.assertLess(metric["normalized_rms_residual"], 0.1)
        self.assertEqual(len(report["eta_sha256"]), 64)

    def test_scalar_history_wave_comparator_rejects_unaligned_grids(self):
        """Wave comparison must not interpolate silently between grids."""

        eta_values = numpy.linspace(1.0, 4.0, 16)
        with self.assertRaises(ValueError):
            camb_reference.compare_scalar_history_waves(
                {"actual": numpy.sin(eta_values[:-1])},
                {"reference": numpy.sin(eta_values)},
                eta_values,
                variable_map={"actual": "reference"},
            )

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

    def test_parity_reference_set_retains_multiple_fixed_points(self):
        """The scientific harness records complete CAMB surfaces per point."""

        shifted = dict(camb_reference.FIXED_LCDM_REFERENCE_CONTRACT)
        shifted["param_map"] = dict(shifted["param_map"])
        shifted["param_map"]["As"] = 2.2e-9
        report = camb_reference.build_camb_parity_reference_set(
            {
                "model_lcdm.yml": {
                    "initial": camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
                    "shifted_amplitude": shifted,
                }
            },
            ells=(2, 20, 100),
        )

        points = report["models"]["model_lcdm.yml"]
        self.assertEqual(set(points), {"initial", "shifted_amplitude"})
        for fixture in points.values():
            self.assertEqual(
                tuple(fixture["declared_observables"]),
                camb_reference.CAMB_PARITY_SPECTRA,
            )
            self.assertEqual(set(fixture["spectra"]["TT"]), {"C_ell", "D_ell"})
            self.assertTrue(
                all(
                    numpy.all(numpy.isfinite(values["D_ell"]))
                    for values in fixture["spectra"].values()
                )
            )

    def test_bounded_parity_rows_retain_request_and_raw_array_hashes(self):
        """Bounded CAMB rows retain independent, auditable parity evidence."""

        plan = {
            "ell_min": 2,
            "ell_max": 200,
            "k_sample_count": 64,
            "eta_sample_count": 192,
            "evolution_eta_sample_count": 128,
            "source_grid_multiplier": 2,
        }
        massive = deepcopy(camb_reference.FIXED_LCDM_REFERENCE_CONTRACT)
        massive["param_map"].update(
            {
                "omch2": 0.18 - 0.25 / 93.14,
                "sum_mnu": 0.25,
                "num_massive_neutrinos": 3,
            }
        )
        planck = deepcopy(camb_reference.FIXED_LCDM_REFERENCE_CONTRACT)
        planck["param_map"] = {
            "H0": 67.66,
            "ombh2": 0.04897 * (67.66 / 100.0) ** 2,
            "omch2": (
                (0.3111 - 0.04897) * (67.66 / 100.0) ** 2 - 0.06 / 93.14
            ),
            "omnuh2": 0.06 / 93.14,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.044,
            "num_massive_neutrinos": 3,
            "sum_mnu": 0.06,
            "YHe": 0.245,
        }
        dynamic = deepcopy(massive)
        dynamic["param_map"].update(
            {
                "omch2": 0.18 - 0.06 / 93.14,
                "sum_mnu": 0.06,
            }
        )
        wcdm = deepcopy(dynamic)
        wcdm["param_map"]["omk"] = 0.0
        wcdm["calls"] = [{"method": "set_dark_energy", "kwargs": {"w": -1.0}}]
        w0wa = deepcopy(wcdm)
        w0wa["calls"] = [
            {
                "method": "set_dark_energy",
                "kwargs": {"w": -1.0, "wa": 0.0},
            }
        ]
        report = camb_reference.build_camb_parity_reference_set(
            {
                "model_lcdm.yml": camb_reference.FIXED_LCDM_REFERENCE_CONTRACT,
                "model_lcdm_mnu.yml": massive,
                "model_ref_planck2018.yml": planck,
                "model_wcdm.yml": wcdm,
                "model_w0wa.yml": w0wa,
            },
            ells=(2, 20, 100, 200),
            spectra=camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA,
            numerical_plan=plan,
        )

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["numerical_plan"], plan)
        self.assertEqual(
            set(report["models"]),
            {
                "model_lcdm.yml",
                "model_lcdm_mnu.yml",
                "model_ref_planck2018.yml",
                "model_wcdm.yml",
                "model_w0wa.yml",
            },
        )
        for model_rows in report["models"].values():
            for fixture in model_rows.values():
                row = fixture["parity_row"]
                self.assertEqual(tuple(row["ell_values"]), (2, 20, 100, 200))
                self.assertEqual(row["numerical_plan"], plan)
                self.assertEqual(
                    tuple(row["spectra"]),
                    camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA,
                )
                self.assertEqual(
                    set(row["surface_sha256"]),
                    set(fixture["spectra"]),
                )
                self.assertTrue(
                    all(
                        len(value) == 64
                        for value in row["surface_sha256"].values()
                    )
                )
                self.assertEqual(
                    row["raw_artifact_sha256"],
                    camb_reference.reference_fixture_sha256(
                        fixture["spectra"]
                    ),
                )

    def test_bounded_reference_report_rejects_changed_raw_surface(self):
        """A changed array cannot pass the raw CAMB parity comparator."""

        fixture = camb_reference.build_lcdm_full_reference_fixture(
            (2, 20, 100, 200)
        )
        actual = {
            "sector": "scalar",
            "ell_values": fixture["ell_values"],
            "spectra": {
                name: {
                    "C_ell": values["C_ell"],
                    "D_ell": list(values["D_ell"]),
                }
                for name, values in fixture["spectra"].items()
            },
        }
        actual["spectra"]["TT"]["D_ell"][2] *= 1.01
        report = cmb_diagnostics.compare_full_cmb_observable_parity(
            actual,
            fixture,
            refinement={"converged": True},
            relative_tolerances={"TT": 1.0e-6},
            fixture_digest=fixture["fixture_sha256"],
            require_fixture_digest=True,
        )

        self.assertFalse(report["accepted"])
        tt_row = next(
            row for row in report["rows"] if row["row"] == "scalar:TT"
        )
        self.assertFalse(tt_row["metric"]["converged"])
        self.assertIn("raw-array parity tolerance failed", tt_row["issues"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
