# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Last Updated: 2025-11-24

"""Basic functional tests for the Copernican Suite."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import camb
import numpy as np
import pandas as pd

import copernican_lib.data_loaders as data_loaders
import copernican_lib.engine_interface as engine_interface
import copernican_lib.model_coder as model_coder
import copernican_lib.model_parser as model_parser
import engines.cosmo_engine_mcmc as engine
from copernican_lib.likelihoods import cmb

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("VIRTUAL_ENV", str(REPO_ROOT / ".venv"))

COPERNICAN_PATH = REPO_ROOT / "copernican.py"
SPEC = importlib.util.spec_from_file_location("copernican", COPERNICAN_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Unable to resolve copernican module for testing")
COPERNICAN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COPERNICAN_MODULE)
extract_cosmological_param_vector = (
    COPERNICAN_MODULE.extract_cosmological_param_vector
)

# Ensure compound BAO parser registration without requiring package installs
parser_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "bao"
    / "compound"
    / "cosmo_parser_compound.py"
)
spec = importlib.util.spec_from_file_location(
    "cosmo_parser_compound",
    parser_path,
)
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


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
        sne_df = data_loaders.load_sne_data("jla_2014")
        self.assertIsNotNone(sne_df)
        sne_df = sne_df.head(3)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :3, :3
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:3]

        bao_df = data_loaders.load_bao_data("compound_bao_set")
        self.assertIsNotNone(bao_df)
        bao_df = bao_df.head(3)

        cmb_df = data_loaders.load_cmb_data("planck_2018_lite")
        self.assertIsNotNone(cmb_df)

        params = self.plugin.INITIAL_GUESSES
        chi2_sne = engine.chi_squared_sne(
            params, self.plugin.distance_modulus_model, sne_df
        )
        self.assertTrue(np.isfinite(chi2_sne))

        pred_df, rs_mpc, _ = engine.calculate_bao_observables(
            bao_df, self.plugin, params
        )
        z = bao_df["redshift"].to_numpy(dtype=float)
        obs_type = bao_df["observable_type"].to_numpy()
        obs_val = bao_df["value"].to_numpy(dtype=float)
        obs_err = bao_df["error"].to_numpy(dtype=float)
        cov_inv = bao_df.attrs.get("covariance_matrix_inv")
        chi2_bao = engine.chi_squared_bao(
            z,
            obs_type,
            obs_val,
            obs_err,
            self.plugin,
            params,
            rs_mpc,
            covariance_matrix_inv=cov_inv,
        )
        self.assertTrue(np.isfinite(chi2_bao))

        camb_params = self.plugin.get_camb_params(params)
        chi2_cmb = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        spec = engine.compute_cmb_spectrum(
            camb_params, cmb_df["ell"].values, spectra=("TT", "TE", "EE")
        )
        self.assertTrue(np.isfinite(chi2_cmb))
        self.assertIn("TT", spec)
        self.assertEqual(len(spec["TT"]), len(cmb_df))

    def test_mcmc_fit_returns_expected_fields(self):
        """Return posterior diagnostics and χ² totals from the MCMC engine."""
        sne_df = data_loaders.load_sne_data("jla_2014").head(3)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :3, :3
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:3]
        result = engine.fit_cosmology_parameters(
            sne_df,
            self.plugin,
            n_walkers=6,
            n_steps=8,
            pool_size=1,
            burn_in_steps=20,
        )
        self.assertTrue(result["success"])
        self.assertIn("samples", result)
        self.assertIn("chi2_total", result)
        self.assertTrue(np.isfinite(result["chi2_total"]))
        components = result.get("chi2_components", {})
        self.assertAlmostEqual(
            result["chi2_total"], sum(components.values()), places=7
        )
        self.assertAlmostEqual(result["chi2_sne"], components.get("sne", 0.0))
        self.assertAlmostEqual(
            result.get("chi2_bao", 0.0), components.get("bao", 0.0)
        )
        self.assertAlmostEqual(
            result.get("chi2_cmb", 0.0), components.get("cmb", 0.0)
        )
        self.assertIn("burn_in_steps", result)
        self.assertIn("production_steps", result)
        self.assertEqual(result["burn_in_steps"], 20)

    def test_chi_squared_cmb_planck2018lite(self):
        """Verify that the Planck 2018 lite dataset yields finite χ²."""
        cmb_df = data_loaders.load_cmb_data("planck_2018_lite")
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
        with self.assertLogs(level="ERROR") as cm:
            chi2 = engine.chi_squared_sne(
                self.plugin.INITIAL_GUESSES,
                self.plugin.distance_modulus_model,
                bad_df,
            )
        self.assertTrue(np.isinf(chi2))
        self.assertIn("non-finite zcmb or mu_obs", "".join(cm.output))

    def test_cmb_spectrum_is_d_ell(self):
        """Ensure cached CAMB spectra match Dl convention."""
        cmb_df = data_loaders.load_cmb_data("planck_2018_lite")
        ells = cmb_df["ell"].values[:5]
        camb_params = self.plugin.get_camb_params(self.plugin.INITIAL_GUESSES)
        result = engine.compute_cmb_spectrum_from_dict(
            camb_params, ells, spectra=("TT",)
        )

        # Mirror the likelihood helper's CAMB parameter construction so the
        # comparison exercises the same neutrino-sector mapping that feeds the
        # cached spectra.  Calling the internal builder keeps the functional
        # regression aligned with whichever optional neutrino knobs the plugin
        # exposes.
        params = cmb._make_camb_params(camb_params, lmax=int(np.max(ells)))
        params.InitPower.set_params(As=camb_params["As"], ns=camb_params["ns"])
        ref = camb.get_results(params).get_unlensed_scalar_cls(
            lmax=int(np.max(ells)), CMB_unit="muK"
        )
        np.testing.assert_allclose(result, ref[:, 0][ells], rtol=1e-7)

    def test_engine_metadata_constants(self):
        """Expose human-readable descriptors for UI logging."""
        self.assertTrue(hasattr(engine, "ENGINE_KIND"))
        self.assertEqual(engine.ENGINE_KIND, "mcmc")
        self.assertTrue(hasattr(engine, "ENGINE_LABEL"))
        self.assertIn("MCMC", engine.ENGINE_LABEL)


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


class CosmologicalParameterHelperTestCase(unittest.TestCase):
    """Validate parameter extraction fallbacks in ``copernican``."""

    def test_returns_ordered_values_when_available(self):
        """Preserve plugin parameter order when extraction succeeds."""

        plugin = SimpleNamespace(
            MODEL_NAME="Toy", PARAMETER_NAMES=["H0", "Om_m"]
        )
        fit_results = {
            "success": True,
            "fitted_cosmological_params": {
                "H0": 71.0,
                "Om_m": 0.3,
                "unused": 1.0,
            },
        }
        vector = extract_cosmological_param_vector(fit_results, plugin)
        self.assertEqual(vector, [71.0, 0.3])

    def test_returns_none_when_required_value_missing(self):
        """Refuse to extract vectors with incomplete parameter coverage."""

        plugin = SimpleNamespace(
            MODEL_NAME="Toy", PARAMETER_NAMES=["H0", "Om_m"]
        )
        fit_results = {
            "success": True,
            "fitted_cosmological_params": {"H0": 71.0},
        }
        vector = extract_cosmological_param_vector(fit_results, plugin)
        self.assertIsNone(vector)

    def test_returns_none_when_fit_unsuccessful(self):
        """Avoid leaking stale parameters when the fit itself failed."""

        plugin = SimpleNamespace(MODEL_NAME="Toy", PARAMETER_NAMES=["H0"])
        fit_results = {
            "success": False,
            "fitted_cosmological_params": {"H0": 71.0},
        }
        vector = extract_cosmological_param_vector(fit_results, plugin)
        self.assertIsNone(vector)


if __name__ == "__main__":
    unittest.main()
