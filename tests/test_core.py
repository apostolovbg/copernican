"""Basic functional tests for the Copernican Suite."""

import unittest
from pathlib import Path

import camb
import numpy as np
import pandas as pd

import copernican_lib.data_loaders as data_loaders
import copernican_lib.engine_interface as engine_interface
import copernican_lib.model_coder as model_coder
import copernican_lib.model_parser as model_parser

# Ensure compound BAO parser registration
import data.bao.compound.cosmo_parser_compound  # noqa: F401
import engines.cosmo_engine_comb as engine


class FunctionalTestCase(unittest.TestCase):
    """Run a minimal end-to-end check of the public API."""

    @classmethod
    def setUpClass(cls):
        """Prepare a validated ΛCDM plugin used by several test cases."""
        # Prepare a validated ΛCDM plugin used by several test cases.
        base = Path(__file__).resolve().parents[1]
        models_dir = base / "models"
        yaml_path = models_dir / "cosmo_model_lcdm.yml"
        cache_dir = models_dir / "cache"
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_interface.build_plugin(parsed, funcs)
        engine_interface.validate_plugin(cls.plugin)

    def test_plugin_validation(self):
        """Ensure the constructed plugin exposes the expected API."""
        self.assertTrue(hasattr(self.plugin, "distance_modulus_model"))

    def test_engine_routines(self):
        """Run a smoke test across the main engine routines."""
        sne_df = data_loaders.load_sne_data("JLA 2014")
        self.assertIsNotNone(sne_df)
        sne_df = sne_df.head(3)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :3, :3
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:3]

        bao_df = data_loaders.load_bao_data("Compound BAO dataset")
        self.assertIsNotNone(bao_df)
        bao_df = bao_df.head(3)

        cmb_df = data_loaders.load_cmb_data("Planck 2018 Lite TT/TE/EE")
        self.assertIsNotNone(cmb_df)

        params = self.plugin.INITIAL_GUESSES
        chi2_sne = engine.chi_squared_sne(
            params, self.plugin.distance_modulus_model, sne_df
        )
        self.assertTrue(np.isfinite(chi2_sne))

        pred_df, rs_mpc, _ = engine.calculate_bao_observables(
            bao_df, self.plugin, params
        )
        chi2_bao = engine.chi_squared_bao(bao_df, self.plugin, params, rs_mpc)
        self.assertTrue(np.isfinite(chi2_bao))

        camb_params = self.plugin.get_camb_params(params)
        chi2_cmb = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        spec = engine.compute_cmb_spectrum(
            camb_params, cmb_df["ell"].values, spectra=("TT", "TE", "EE")
        )
        self.assertTrue(np.isfinite(chi2_cmb))
        self.assertIn("TT", spec)
        self.assertEqual(len(spec["TT"]), len(cmb_df))

    def test_combined_fit(self):
        """Check that the combined fit pipeline returns finite χ² values."""
        sne_df = data_loaders.load_sne_data("JLA 2014").head(2)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :2, :2
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:2]
        bao_df = data_loaders.load_bao_data("Compound BAO dataset").head(2)
        cmb_df = data_loaders.load_cmb_data("Planck 2018 Lite TT/TE/EE")
        cmb_df = cmb_df.head(10)
        cmb_attrs = cmb_df.attrs
        cmb_attrs["covariance_matrix_inv"] = cmb_attrs[
            "covariance_matrix_inv"
        ][:10, :10]

        result = engine.fit_combined_parameters(
            sne_df,
            bao_df,
            cmb_df,
            self.plugin,
        )
        self.assertTrue(result["success"])
        self.assertIn("chi2_total", result)
        self.assertTrue(np.isfinite(result["chi2_total"]))

    def test_chi_squared_cmb_planck2018lite(self):
        """Verify that the Planck 2018 lite dataset yields finite χ²."""
        cmb_df = data_loaders.load_cmb_data("Planck 2018 Lite TT/TE/EE")
        params = self.plugin.INITIAL_GUESSES
        chi2 = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        self.assertTrue(np.isfinite(chi2))

    def test_chi_squared_sne_invalid_data(self):
        """chi_squared_sne should return inf when data is invalid."""
        bad_df = pd.DataFrame(
            {
                "zcmb": [0.1, np.nan],
                "mu_obs": [33.1, 34.5],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        chi2 = engine.chi_squared_sne(
            self.plugin.INITIAL_GUESSES,
            self.plugin.distance_modulus_model,
            bad_df,
        )
        self.assertTrue(np.isinf(chi2))

    def test_cmb_spectrum_is_d_ell(self):
        """Ensure cached CAMB spectra match Dl convention."""
        cmb_df = data_loaders.load_cmb_data("Planck 2018 Lite TT/TE/EE")
        ells = cmb_df["ell"].values[:5]
        camb_params = self.plugin.get_camb_params(self.plugin.INITIAL_GUESSES)
        result = engine.compute_cmb_spectrum_from_dict(
            camb_params, ells, spectra=("TT",)
        )

        params = camb.CAMBparams()
        params.set_cosmology(
            H0=camb_params["H0"],
            ombh2=camb_params["ombh2"],
            omch2=camb_params["omch2"],
            tau=camb_params["tau"],
        )
        params.omnuh2 = camb_params.get("omnuh2", 0.0)
        params.InitPower.set_params(As=camb_params["As"], ns=camb_params["ns"])
        params.set_for_lmax(int(np.max(ells)) + 300, lens_potential_accuracy=0)
        ref = camb.get_results(params).get_unlensed_scalar_cls(
            lmax=int(np.max(ells)), CMB_unit="muK"
        )
        np.testing.assert_allclose(result, ref[:, 0][ells], rtol=1e-7)


class PlotterUtilTestCase(unittest.TestCase):
    """Test helper utilities in ``plotter``."""

    def test_wrap_math_removes_size_macros(self):
        """Ensure size macros are stripped when wrapping math expressions."""
        from copernican_lib import plotter

        expr = r"\mu(z) = 5\log_{10}\bigl[d_L(z)/\mathrm{Mpc}\bigr] + 25"
        expected = r"$\mu(z) = 5\log_{10}[d_L(z)/{Mpc}] + 25$"
        self.assertEqual(plotter._wrap_math(expr), expected)

    def test_latex_utils_conversions(self):
        """Ensure LaTeX mappings are applied consistently."""
        from copernican_lib import latex_utils

        self.assertEqual(latex_utils.sanitize_name(r"\alpha_1"), "alpha_1")
        self.assertEqual(
            latex_utils.latex_to_sympy(r"\frac{1}{\infty}"), "(1)/(sympy.oo)"
        )
        self.assertEqual(latex_utils.latex_to_unicode(r"\Omega_{gamma}"), "Ωᵧ")
        self.assertEqual(latex_utils.latex_to_unicode("H_0"), "H₀")
        self.assertEqual(latex_utils.latex_to_unicode(r"z_{\rm rec}"), "zᵣₑc")
        self.assertEqual(latex_utils.latex_to_unicode("x_{(1+2)}"), "x₍₁₊₂₎")
        self.assertEqual(latex_utils.latex_to_unicode("y^{*}"), "y⁎")


if __name__ == "__main__":
    unittest.main()
