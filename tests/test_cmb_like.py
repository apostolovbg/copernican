"""Unit tests for CAMB-backed CMB helpers.

"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import camb
import numpy as np

from copernican_lib import (
    engine_plugin_validation,
    model_coder,
    model_spec_validator,
)
from copernican_lib.likelihoods import cmb

class CMBBackgroundTestCase(unittest.TestCase):
    """Validate CAMB background helpers share settings with the spectra API."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare a ΛCDM plugin for evaluating CAMB helpers."""

        repo_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("VIRTUAL_ENV", str(repo_root / ".venv"))
        yaml_path = repo_root / "models" / "cosmo_model_lcdm.yml"
        cache_dir = repo_root / "models" / "cache"
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_plugin_validation.build_plugin(parsed, funcs)

    def test_background_observables_match_input_length(self) -> None:
        """Background helper should return one entry per requested redshift."""

        params = self.plugin.get_camb_params(self.plugin.INITIAL_GUESSES)
        redshifts = np.array([0.15, 0.35, 0.57])
        background = cmb.compute_camb_background_observables(params, redshifts)

        self.assertEqual(background["DM"].shape, redshifts.shape)
        self.assertEqual(background["DH"].shape, redshifts.shape)
        self.assertEqual(background["DV"].shape, redshifts.shape)
        self.assertGreater(background["rs_drag"], 0.0)
        self.assertTrue(np.all(np.isfinite(background["DM"])))

    def test_background_cache_collapses_duplicate_redshifts(self) -> None:
        """Repeated redshifts should produce identical background distances."""

        params = self.plugin.get_camb_params(self.plugin.INITIAL_GUESSES)
        redshifts = np.array([0.35, 0.35, 0.60])
        background = cmb.compute_camb_background_observables(params, redshifts)

        self.assertAlmostEqual(
            background["DM"][0], background["DM"][1], places=12
        )
        self.assertAlmostEqual(
            background["DH"][0], background["DH"][1], places=12
        )

    def test_neutrino_configuration_matches_direct_camb(self) -> None:
        """Neutrino sector parameters should propagate unchanged to CAMB."""

        custom_params = {
            "H0": 68.0,
            "ombh2": 0.023,
            "omch2": 0.118,
            "tau": 0.059,
            "As": 2.05e-9,
            "ns": 0.964,
            "Neff": 3.55,
            "standard_neutrino_neff": 3.044,
            "num_massive_neutrinos": 2,
            "mnu1": 0.06,
            "mnu2": 0.12,
            "neutrino_hierarchy": "normal",
        }
        ell_range = np.arange(2, 51, dtype=int)
        redshifts = np.array([0.15, 0.60, 1.0])

        helper_background = cmb.compute_camb_background_observables(
            custom_params, redshifts
        )
        helper_cls = cmb.compute_cmb_spectrum_from_dict(
            custom_params, ell_range, spectra=("TT", "EE", "TE")
        )

        manual = camb.CAMBparams()
        # Direct CAMB configuration mirroring the helper inputs.  This ensures
        # the regression asserts against the canonical API rather than the
        # helper internals.
        manual.set_cosmology(
            H0=custom_params["H0"],
            ombh2=custom_params["ombh2"],
            omch2=custom_params["omch2"],
            tau=custom_params["tau"],
            nnu=custom_params["Neff"],
            standard_neutrino_neff=custom_params["standard_neutrino_neff"],
            num_massive_neutrinos=custom_params["num_massive_neutrinos"],
            mnu=custom_params["mnu1"] + custom_params["mnu2"],
            neutrino_hierarchy=custom_params["neutrino_hierarchy"],
        )
        manual.set_for_lmax(
            int(ell_range.max()) + cmb._LMAX_PADDING,
            lens_potential_accuracy=0,
        )
        manual.InitPower.set_params(
            As=custom_params["As"], ns=custom_params["ns"]
        )

        manual_results = camb.get_results(manual)
        manual_cls = manual_results.get_unlensed_scalar_cls(
            lmax=int(ell_range.max()), CMB_unit="muK"
        )

        for column, spectrum in zip(
            (0, 1, 3), ("TT", "EE", "TE"), strict=True
        ):
            np.testing.assert_allclose(
                helper_cls[spectrum],
                manual_cls[:, column][ell_range],
                rtol=1e-7,
                atol=1e-7,
            )

        manual_background = {}
        # Derived parameters such as ``rdrag`` live on the CAMB results object,
        # so we fetch them explicitly to mirror the helper outputs.
        manual_background["rs_drag"] = float(
            manual_results.get_derived_params().get("rdrag")
        )
        manual_background["DM"] = np.asarray(
            [
                manual_results.comoving_radial_distance(float(z))
                for z in redshifts
            ]
        )
        manual_background["DA"] = np.asarray(
            [
                manual_results.angular_diameter_distance(float(z))
                for z in redshifts
            ]
        )
        manual_background["Hz"] = np.asarray(
            [manual_results.hubble_parameter(float(z)) for z in redshifts]
        )
        manual_background["DH"] = np.where(
            np.abs(manual_background["Hz"]) > 1e-12,
            cmb._C_LIGHT_KM_S / manual_background["Hz"],
            np.nan,
        )
        manual_background["DV"] = np.full_like(redshifts, np.nan, dtype=float)
        term = manual_background["DM"] * manual_background["DM"]
        term *= redshifts
        term *= manual_background["DH"]
        mask = np.isfinite(term) & (term >= 0.0)
        manual_background["DV"][mask] = np.power(term[mask], 1.0 / 3.0)
        manual_background["DV"][redshifts == 0.0] = 0.0

        np.testing.assert_allclose(
            helper_background["rs_drag"],
            manual_background["rs_drag"],
            rtol=5e-6,
            atol=1e-10,
        )
        for key in ("DM", "DA", "DH", "DV", "Hz"):
            np.testing.assert_allclose(
                helper_background[key],
                manual_background[key],
                rtol=1e-8,
                atol=1e-8,
            )

if __name__ == "__main__":
    unittest.main()
