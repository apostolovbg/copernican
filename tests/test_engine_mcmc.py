"""Tests for the MCMC engine."""

import importlib.util
import logging
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
    _classify_parameter_bounds,
    _estimate_condition_number,
    _initialise_active_walkers,
    _log_probability,
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
            sne_df, plugin, n_walkers=4, n_steps=5, pool_size=1
        )
        n_params = len(plugin.PARAMETER_NAMES)
        expected = (5, max(4, 2 * n_params), n_params)
        self.assertEqual(res["samples"].shape, expected)
        self.assertEqual(res["log_probability"].shape, expected[:2])
        self.assertTrue(res["success"])
        self.assertTrue(np.isfinite(res["chi2_min"]))
        self.assertAlmostEqual(res["chi2_total"], res["chi2_sne"])
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

    def test_log_probability_penalty(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        bad = np.array([200.0] + list(plugin.INITIAL_GUESSES[1:]))
        lp = cosmo_engine_mcmc._log_probability(bad, plugin, sne_df)
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
        log_prob = np.array(
            [
                _log_probability(coords[0], plugin, sne_df),
                np.nan,
            ]
        )
        rng = np.random.default_rng(12345)
        new_coords, new_log_prob = _reseed_invalid_walkers(
            coords,
            log_prob,
            lower=lower,
            upper=upper,
            rng=rng,
            log_probability_fn=lambda pos: _log_probability(
                pos, plugin, sne_df
            ),
            reference_position=np.asarray(plugin.INITIAL_GUESSES, dtype=float),
        )
        self.assertTrue(np.all(np.isfinite(new_coords)))
        self.assertTrue(np.all(np.isfinite(new_log_prob)))

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
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        self.assertEqual(chain.shape[2], len(plugin.PARAMETER_NAMES))
        const_idx = plugin.PARAMETER_NAMES.index("c")
        fixed_spread = np.ptp(chain[:, :, const_idx])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)

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
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        fixed_spread = np.ptp(chain[:, :, 0])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
