# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Basic functional tests for the Copernican Suite."""

import importlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import camb
import numpy
import pandas

import copernican.engines.cosmo_engine_mcmc as engine
import copernican.lib.dataset_registry as dataset_registry
import copernican.lib.engine_adapter as engine_plugin_validation
import copernican.lib.model_coder as model_coder
import copernican.lib.model_spec_validator as model_spec_validator
from copernican.lib.likelihoods import cmb
from copernican.lib.run_pipeline import extract_cosmological_param_vector

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("VIRTUAL_ENV", str(REPO_ROOT / ".venv"))

# Ensure compound BAO parser registration without requiring package installs.
importlib.import_module(
    "copernican.datasets.bao.compound.cosmo_parser_compound"
)


class FunctionalTestCase(unittest.TestCase):
    """Run a minimal end-to-end check of the public API."""

    @classmethod
    def setUpClass(cls):
        """Prepare a validated reference plugin used by several tests."""
        # Prepare a validated reference plugin used by several tests.
        base = Path(__file__).resolve().parents[3]
        models_dir = base / "copernican" / "models"
        yaml_path = models_dir / "cosmo_model_lcdm.yml"
        cache_dir = models_dir / "cache"
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_plugin_validation.build_plugin(parsed, funcs)
        engine_plugin_validation.validate_plugin(cls.plugin)

    def test_plugin_validation(self):
        """Ensure the constructed plugin exposes the expected API."""
        self.assertTrue(hasattr(self.plugin, "distance_modulus_model"))

    def test_engine_routines(self):
        """Run a smoke test across the main engine routines."""
        sne_df = dataset_registry.load_sne_data("jla_2014")
        self.assertIsNotNone(sne_df)
        sne_df = sne_df.head(3)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :3, :3
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:3]

        bao_df = dataset_registry.load_bao_data("compound_bao_set")
        self.assertIsNotNone(bao_df)
        bao_df = bao_df.head(3)

        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        self.assertIsNotNone(cmb_df)

        params = self.plugin.INITIAL_GUESSES
        chi2_sne = engine.chi_squared_sne(
            params, self.plugin.distance_modulus_model, sne_df
        )
        self.assertTrue(numpy.isfinite(chi2_sne))

        pred_df, rs_mpc, _ = engine.calculate_bao_observables(
            bao_df, self.plugin, params
        )
        redshifts_array = bao_df["redshift"].to_numpy(dtype=float)
        observable_types_array = bao_df["observable_type"].to_numpy()
        observable_values_array = bao_df["value"].to_numpy(dtype=float)
        observable_errors_array = bao_df["error"].to_numpy(dtype=float)
        cov_inv = bao_df.attrs.get("covariance_matrix_inv")
        chi2_bao = engine.chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            params,
            rs_mpc,
            covariance_matrix_inv=cov_inv,
        )
        self.assertTrue(numpy.isfinite(chi2_bao))

        camb_params = self.plugin.get_camb_contract(params)
        camb_params["perturbations"] = (
            self.plugin.get_cmb_perturbation_contract(params)
        )
        chi2_cmb = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        spec = engine.compute_cmb_spectrum(
            camb_params, cmb_df["ell"].values, spectra=("TT", "TE", "EE")
        )
        self.assertTrue(numpy.isfinite(chi2_cmb))
        self.assertIn("TT", spec)
        self.assertEqual(len(spec["TT"]), len(cmb_df))

    def test_mcmc_fit_returns_expected_fields(self):
        """Return posterior diagnostics and χ² totals.

        The MCMC engine should report a finite total and component
        breakdown.
        """
        sne_df = dataset_registry.load_sne_data("jla_2014").head(3)
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
        self.assertTrue(numpy.isfinite(result["chi2_total"]))
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
        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        params = self.plugin.INITIAL_GUESSES
        chi2 = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        self.assertTrue(numpy.isfinite(chi2))

    def test_chi_squared_sne_invalid_data(self):
        """chi_squared_sne should return inf when data is invalid."""
        bad_df = pandas.DataFrame(
            {
                "zcmb": [0.1, numpy.nan],
                "mu_obs": [33.1, 34.5],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        with self.assertLogs(level="ERROR") as captured_logs:
            chi2 = engine.chi_squared_sne(
                self.plugin.INITIAL_GUESSES,
                self.plugin.distance_modulus_model,
                bad_df,
            )
        self.assertTrue(numpy.isinf(chi2))
        self.assertIn(
            "non-finite zcmb or mu_obs", "".join(captured_logs.output)
        )

    def test_cmb_spectrum_is_d_ell(self):
        """Ensure cached CAMB spectra match Dl convention."""
        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        ells = cmb_df["ell"].values[:5]
        camb_params = self.plugin.get_camb_contract(
            self.plugin.INITIAL_GUESSES
        )
        camb_params["perturbations"] = (
            self.plugin.get_cmb_perturbation_contract(
                self.plugin.INITIAL_GUESSES
            )
        )
        result = engine.compute_cmb_spectrum_from_dict(
            camb_params, ells, spectra=("TT",)
        )

        # Mirror the likelihood helper's CAMB parameter construction so the
        # comparison exercises the same neutrino-sector mapping that feeds the
        # cached spectra.  Calling the internal builder keeps the functional
        # regression aligned with whichever optional neutrino knobs the plugin
        # exposes.
        params = cmb._make_camb_params(camb_params, lmax=int(numpy.max(ells)))
        params.InitPower.set_params(
            As=camb_params["param_map"]["As"],
            ns=camb_params["param_map"]["ns"],
        )
        ref = camb.get_results(params).get_unlensed_scalar_cls(
            lmax=int(numpy.max(ells)), CMB_unit="muK"
        )
        numpy.testing.assert_allclose(result, ref[:, 0][ells], rtol=1e-7)

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
        from copernican.lib import plotter

        expr = r"\mu(z) = 5\log_{10}\bigl[d_L(z)/\mathrm{Mpc}\bigr] + 25"
        expected = r"$\mu(z) = 5\log_{10}[d_L(z)/{Mpc}] + 25$"
        self.assertEqual(plotter._wrap_math(expr), expected)

    def test_latex_utils_conversions(self):
        """Ensure LaTeX mappings are applied consistently."""
        from copernican.lib import latex_utils

        self.assertEqual(latex_utils.sanitize_name(r"\alpha_1"), "alpha_1")
        self.assertEqual(
            latex_utils.latex_to_sympy(r"\frac{1}{\infty}"), "(1)/(sympy.oo)"
        )
        self.assertEqual(
            latex_utils.latex_to_unicode(r"\Omega_{gamma}"),
            "Ωᵧ",
        )
        self.assertEqual(latex_utils.latex_to_unicode("H_0"), "H₀")
        self.assertEqual(
            latex_utils.latex_to_unicode(r"z_{\rm rec}"),
            "zᵣₑc",
        )
        self.assertEqual(
            latex_utils.latex_to_unicode("x_{(1+2)}"),
            "x₍₁₊₂₎",
        )
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
