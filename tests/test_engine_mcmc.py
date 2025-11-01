"""Integration tests for the ensemble MCMC engine.

**Last Updated:** 2025-11-01
"""

import importlib.util
import logging
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

if importlib.util.find_spec("arviz") is not None:
    from copernican_lib import chain_io

    ARVIZ_AVAILABLE = True
else:
    ARVIZ_AVAILABLE = False
from copernican_lib import engine_interface, model_coder, model_parser
from engines import cosmo_engine_mcmc
from engines.cosmo_engine_mcmc import (
    _ActiveLogProbability,
    _build_sne_logposterior,
    _classify_parameter_bounds,
    _estimate_condition_number,
    _initialise_active_walkers,
    _reseed_invalid_walkers,
)


def _build_model_plugin(yaml_filename: str):
    """Return a validated plugin for ``yaml_filename``.

    Tests construct plugins from disk instead of hard-coding dummy classes so
    that they exercise the same parsing pathway as the production workflow.
    """

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    yaml_path = os.path.join(models_dir, yaml_filename)
    cache_dir = os.path.join(models_dir, "cache")
    cache_path = model_parser.parse_model(yaml_path, cache_dir)
    func_dict, parsed = model_coder.generate_callables(cache_path)
    return engine_interface.build_plugin(parsed, func_dict)


@unittest.skipUnless(ARVIZ_AVAILABLE, "arviz not installed")
class TestMCMCEngine(unittest.TestCase):
    """Verify that the MCMC engine produces chains and NetCDF output."""

    def _build_lcdm_plugin(self):
        return _build_model_plugin("cosmo_model_lcdm.yml")

    def _build_cfsc_plugin(self):
        return _build_model_plugin("cosmo_model_cfsc.yml")

    def test_sampler_produces_netcdf(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=5,
            pool_size=1,
            burn_in_steps=12,
        )
        n_params = len(plugin.PARAMETER_NAMES)
        expected = (5, max(4, 2 * n_params), n_params)
        self.assertEqual(res["samples"].shape, expected)
        self.assertEqual(res["log_probability"].shape, expected[:2])
        self.assertTrue(res["success"])
        self.assertTrue(np.isfinite(res["chi2_min"]))
        components = res.get("chi2_components", {})
        total_components = (
            components.get("sne", 0.0)
            + components.get("bao", 0.0)
            + components.get("cmb", 0.0)
        )
        self.assertAlmostEqual(res["chi2_total"], total_components)
        self.assertAlmostEqual(res["chi2_sne"], components.get("sne", 0.0))
        self.assertAlmostEqual(
            res.get("chi2_bao", 0.0), components.get("bao", 0.0)
        )
        self.assertAlmostEqual(
            res.get("chi2_cmb", 0.0), components.get("cmb", 0.0)
        )
        self.assertSetEqual(
            set(res["fitted_cosmological_params"].keys()),
            set(plugin.PARAMETER_NAMES),
        )
        self.assertSetEqual(
            set(res["posterior_mean_params"].keys()),
            set(plugin.PARAMETER_NAMES),
        )
        self.assertIsInstance(res["burn_in_steps"], int)
        self.assertIsInstance(res["production_steps"], int)
        self.assertIsInstance(res["n_walkers"], int)
        self.assertIsInstance(res["pool_workers"], int)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chain.nc")
            chain_io.save_posterior(
                res["samples"],
                plugin.PARAMETER_NAMES,
                path,
                metadata={"model": plugin.MODEL_NAME},
            )
            # Use a context manager so Windows can remove the file when the
            # temporary directory cleans up. Without explicitly closing the
            # dataset the cleanup step fails because the file handle remains
            # open on that platform.
            with xr.open_dataset(path, group="posterior") as ds:
                for name in plugin.PARAMETER_NAMES:
                    self.assertIn(name, ds.data_vars)

    def test_progress_logging_reports_statistics(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )

        with self.assertLogs(level="INFO") as captured:
            cosmo_engine_mcmc.fit_sne_parameters(
                sne_df,
                plugin,
                n_walkers=4,
                n_steps=6,
                pool_size=1,
                progress_granularity=4,
                burn_in_steps=12,
            )

        joined = "\n".join(captured.output)
        self.assertIn("logP μ=", joined)
        self.assertIn("Walker[", joined)
        for name in plugin.PARAMETER_NAMES:
            self.assertIn(f"    {name}:", joined)
        self.assertNotIn("omitted", joined)

    def test_explicit_pool_size_respected(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=2,
            burn_in_steps=4,
        )
        self.assertEqual(res["pool_workers"], 2)
        self.assertGreaterEqual(res["n_walkers"], res["pool_workers"])

    def test_log_probability_penalty(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        posterior, _, _ = _build_sne_logposterior(
            plugin,
            sne_df,
        )
        bad = np.array([200.0] + list(plugin.INITIAL_GUESSES[1:]))
        lp = posterior(bad)
        self.assertTrue(np.isneginf(lp))

    def test_invalid_walkers_are_reseeded(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        bounds = plugin.PARAMETER_BOUNDS
        lower = np.array(
            [-np.inf if low is None else float(low) for low, _ in bounds]
        )
        upper = np.array(
            [np.inf if high is None else float(high) for _, high in bounds]
        )
        ndim = len(plugin.PARAMETER_NAMES)
        coords = np.vstack(
            [
                np.asarray(plugin.INITIAL_GUESSES, dtype=float),
                np.full(ndim, np.nan),
            ]
        )
        posterior, _, _ = _build_sne_logposterior(
            plugin,
            sne_df,
        )
        log_prob = np.array([posterior(coords[0]), np.nan])
        rng = np.random.default_rng(12345)
        new_coords, new_log_prob = _reseed_invalid_walkers(
            coords,
            log_prob,
            lower=lower,
            upper=upper,
            rng=rng,
            log_probability_fn=lambda pos: posterior(pos),
            reference_position=np.asarray(plugin.INITIAL_GUESSES, dtype=float),
        )
        self.assertTrue(np.all(np.isfinite(new_coords)))
        self.assertTrue(np.all(np.isfinite(new_log_prob)))

    def test_active_log_probability_expands_parameters(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        posterior, _, _ = _build_sne_logposterior(plugin, sne_df)
        bounds = plugin.PARAMETER_BOUNDS
        lower, upper, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logging.getLogger()
        )
        template = np.asarray(plugin.INITIAL_GUESSES, dtype=float)
        active_indices = np.flatnonzero(~fixed_mask)
        adapter = _ActiveLogProbability(posterior, template, active_indices)
        trial = template[active_indices]
        assembled = adapter.assemble_full(trial)
        self.assertTrue(np.allclose(assembled[active_indices], trial))
        self.assertTrue(
            np.allclose(assembled[fixed_mask], template[fixed_mask])
        )
        value = adapter(trial)
        self.assertIsInstance(value, float)
        self.assertTrue(math.isfinite(value) or math.isneginf(value))

        clipped = np.clip(
            trial + 0.1,
            lower[~fixed_mask],
            upper[~fixed_mask],
        )
        assembled_clipped = adapter.assemble_full(clipped)
        self.assertTrue(
            np.allclose(assembled_clipped[active_indices], clipped)
        )

    def test_sampler_runs_with_spawn_pool(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=6,
            pool_size=2,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["pool_workers"], 2)
        self.assertTrue(math.isfinite(result["log_posterior_best"]))

    def test_sampler_handles_fixed_bounds(self):
        plugin = self._build_cfsc_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=30,
            n_steps=4,
            pool_size=1,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        self.assertEqual(chain.shape[2], len(plugin.PARAMETER_NAMES))
        const_idx = plugin.PARAMETER_NAMES.index("c")
        fixed_spread = np.ptp(chain[:, :, const_idx])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)

    def test_likelihood_state_reported(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=1,
            burn_in_steps=12,
        )
        state = result["likelihood_state"]
        self.assertIn("components", state["metadata"])
        self.assertIn("sne", state["metadata"]["components"])
        self.assertTrue(math.isfinite(result["log_likelihood_best"]))
        self.assertTrue(math.isfinite(result["log_posterior_best"]))
        self.assertTrue(math.isfinite(result["log_prior_best"]))

    def test_joint_fit_component_chi2_totals(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        initial = np.asarray(plugin.INITIAL_GUESSES, dtype=float)
        z_bao = np.array([0.1])
        dm = plugin.get_comoving_distance_Mpc(z_bao, *initial)
        rs = plugin.get_sound_horizon_rs_Mpc(*initial)
        bao_df = pd.DataFrame(
            {
                "redshift": z_bao,
                "observable_type": ["DM_over_rs"],
                "value": dm / rs,
                "error": [0.05],
            }
        )
        bao_df.attrs["covariance_matrix_inv"] = np.eye(1)

        ells = np.arange(30, 34)
        camb_params = plugin.get_camb_params(initial)
        dl_vals = cosmo_engine_mcmc.compute_cmb_spectrum(
            camb_params,
            ells,
            spectra=("TT",),
        )
        cmb_df = pd.DataFrame({"ell": ells, "Dl_obs": dl_vals})
        cmb_df.attrs["covariance_matrix_inv"] = np.eye(len(ells))

        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            bao_data_df=bao_df,
            cmb_data_df=cmb_df,
            n_walkers=6,
            n_steps=6,
            pool_size=1,
            burn_in_steps=12,
        )
        components = result.get("chi2_components", {})
        total = sum(components.values())
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["chi2_total"], total, places=6)
        self.assertAlmostEqual(
            result["chi2_bao"], components.get("bao", float("nan"))
        )
        self.assertAlmostEqual(
            result["chi2_cmb"], components.get("cmb", float("nan"))
        )
        self.assertIn(
            "bao", result["likelihood_state"]["metadata"]["components"]
        )
        self.assertIn(
            "cmb", result["likelihood_state"]["metadata"]["components"]
        )

    def test_comoving_distance_vectorized(self):
        plugin = self._build_lcdm_plugin()
        params = plugin.INITIAL_GUESSES
        z_vals = np.array([0.1, 0.2, 0.3])
        arr = plugin.get_comoving_distance_Mpc(z_vals, *params)
        loop = np.array(
            [
                plugin.get_comoving_distance_Mpc(float(z), *params)
                for z in z_vals
            ]
        )
        np.testing.assert_allclose(arr, loop)


class TestMCMCHelpers(unittest.TestCase):
    """Exercise helper utilities that remain active without arviz."""

    def test_near_fixed_bounds_are_flagged(self):
        logger = logging.getLogger("test.mcmc.bounds")
        bounds = [(1.0, 1.0 + 5e-10), (0.0, 2.0)]
        lower, upper, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logger
        )
        self.assertTrue(fixed_mask[0])
        self.assertFalse(fixed_mask[1])
        self.assertAlmostEqual(lower[0], 1.0)
        self.assertAlmostEqual(upper[0], 1.0 + 5e-10)

    def test_initialise_walkers_relaxes_condition_number(self):
        initial = np.array([5.0, 5.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        rng = np.random.default_rng(42)

        def logp(_):
            return 0.0

        walkers, logp_vals = _initialise_active_walkers(
            initial,
            lower,
            upper,
            n_walkers=6,
            rng=rng,
            log_probability_fn=logp,
        )
        self.assertTrue(np.all(np.isfinite(logp_vals)))
        cond = _estimate_condition_number(walkers)
        if cond is not None:
            self.assertLessEqual(cond, 1e12)

    def test_sampler_handles_near_fixed_bounds(self):
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
        tight_value = plugin.INITIAL_GUESSES[0]
        plugin.PARAMETER_BOUNDS = list(plugin.PARAMETER_BOUNDS)
        plugin.PARAMETER_BOUNDS[0] = (
            tight_value - 5e-10,
            tight_value + 5e-10,
        )
        plugin.INITIAL_GUESSES = list(plugin.INITIAL_GUESSES)
        plugin.INITIAL_GUESSES[0] = tight_value

        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )

        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=10,
            n_steps=4,
            pool_size=1,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        fixed_spread = np.ptp(chain[:, :, 0])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
