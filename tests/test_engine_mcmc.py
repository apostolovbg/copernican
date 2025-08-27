"""Tests for the MCMC engine."""

import os
import tempfile
import unittest

import pandas as pd
import xarray as xr

from copernican_lib import (
    chain_io,
    engine_interface,
    model_coder,
    model_parser,
)
from engines import cosmo_engine_mcmc


class TestMCMCEngine(unittest.TestCase):
    """Verify that the MCMC engine produces chains and NetCDF output."""

    def _build_lcdm_plugin(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        yaml_path = os.path.join(models_dir, "cosmo_model_lcdm.yml")
        cache_dir = os.path.join(models_dir, "cache")
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        func_dict, parsed = model_coder.generate_callables(cache_path)
        return engine_interface.build_plugin(parsed, func_dict)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chain.nc")
            chain_io.save_posterior(
                res["samples"],
                plugin.PARAMETER_NAMES,
                path,
                metadata={"model": plugin.MODEL_NAME},
            )
            ds = xr.open_dataset(path, group="posterior")
            for name in plugin.PARAMETER_NAMES:
                self.assertIn(name, ds.data_vars)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
