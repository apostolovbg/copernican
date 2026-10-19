"""Explicit full-spectrum reference rebuild checks, outside discovery.

Run with python -m unittest tests.project.lib.camb_reference_acceptance -v.
The frozen-array assertions remain intact and use the pinned CAMB contract.
"""

import unittest
from copy import deepcopy

import numpy

from tests.project.lib import camb_reference, test_camb_reference


class CambReferenceRebuildTestCase(unittest.TestCase):
    """Recompute full CAMB spectra only when explicitly requested."""

    def _assert_rebuilt_spectrum_matches(self, actual, expected):
        """Share the tested compiler/platform numerical comparison bound."""

        owner = test_camb_reference.CambReferenceModuleTestCase
        owner._assert_rebuilt_spectrum_matches(self, actual, expected)

    def test_full_builder_matches_frozen_fixture(self):
        """Frozen metadata and numerical spectra reproduce across builds."""

        frozen = camb_reference.load_lcdm_full_reference_fixture()
        rebuilt = camb_reference.build_lcdm_full_reference_fixture()
        metadata = {
            key: value
            for key, value in rebuilt.items()
            if key not in {"spectra", "fixture_sha256"}
        }
        self.assertEqual(
            camb_reference._reference_jsonable(metadata),
            {
                key: value
                for key, value in frozen.items()
                if key not in {"spectra", "fixture_sha256"}
            },
        )
        self.assertEqual(
            rebuilt["fixture_sha256"],
            camb_reference.reference_fixture_sha256(
                {
                    key: value
                    for key, value in rebuilt.items()
                    if key != "fixture_sha256"
                }
            ),
        )
        for name in camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA:
            for representation in ("C_ell", "D_ell"):
                with self.subTest(name=name, representation=representation):
                    self._assert_rebuilt_spectrum_matches(
                        rebuilt["spectra"][name][representation],
                        frozen["spectra"][name][representation],
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
        for name in camb_reference.FIXED_LCDM_FULL_REFERENCE_SPECTRA:
            with self.subTest(name=name):
                self._assert_rebuilt_spectrum_matches(
                    actual[name], fixture["spectra"][name]["D_ell"]
                )

    def test_fixed_lcdm_parity_row_uses_the_canonical_sparse_contract(self):
        """The scientific handoff builder retains one exact CAMB row."""

        fixture = camb_reference.build_fixed_lcdm_camb_parity_row(
            ells=(2, 20, 100)
        )

        self.assertEqual(fixture["model"], "model_lcdm.yml")
        self.assertEqual(fixture["fixed_point"], "initial")
        self.assertEqual(
            tuple(fixture["declared_observables"]), ("TT", "TE", "EE")
        )
        self.assertEqual(tuple(fixture["ell_values"]), (2, 20, 100))
        self.assertEqual(fixture["resolved_parameters"]["num_nu_massive"], 0)
        self.assertEqual(fixture["resolved_parameters"]["omnuh2"], 0.0)

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
