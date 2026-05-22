"""Tests for the parameter summary writer."""

import json
import logging
import numbers
import os
import tempfile
import unittest

import pandas as pd
import yaml

from copernican_lib import (
    engine_plugin_validation,
    model_coder,
    model_spec_validator,
    result_writer,
)
from engines import cosmo_engine_mcmc


class TestResultWriter(unittest.TestCase):
    """Ensure that result summaries are written with expected structure."""

    def _build_lcdm_plugin(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        yaml_path = os.path.join(models_dir, "cosmo_model_lcdm.yml")
        cache_dir = os.path.join(models_dir, "cache")
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        func_dict, parsed = model_coder.generate_callables(cache_path)
        return engine_plugin_validation.build_plugin(parsed, func_dict)

    def test_summary_contains_parameters(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = cosmo_engine_mcmc.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=6,
            pool_size=1,
            burn_in_steps=12,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, yaml_path = result_writer.save_summary(
                {plugin.MODEL_NAME: res}, tmpdir, timestamp="test"
            )
            with open(json_path, "r", encoding="utf-8") as fh:
                jdata = json.load(fh)
            with open(yaml_path, "r", encoding="utf-8") as fh:
                ydata = yaml.safe_load(fh)
            for data in (jdata, ydata):
                self.assertIn(plugin.MODEL_NAME, data)
                entry = data[plugin.MODEL_NAME]
                self.assertIn("parameters", entry)
                self.assertIn("errors_1sigma", entry)
                self.assertIn("covariance_matrix", entry)
                self.assertIn("sampling", entry)
                for val in entry["parameters"].values():
                    self.assertIsInstance(val, numbers.Real)
                for val in entry["errors_1sigma"].values():
                    self.assertIsInstance(val, numbers.Real)
                matrix = entry["covariance_matrix"]["matrix"]
                for row in matrix:
                    for val in row:
                        self.assertIsInstance(val, numbers.Real)
                sampling = entry["sampling"]
                self.assertIsInstance(sampling, dict)
                self.assertEqual(sampling.get("production_steps"), 6)
                self.assertEqual(sampling.get("burn_in_steps"), 12)
                _lower, _upper, fixed_mask = (
                    cosmo_engine_mcmc._classify_parameter_bounds(
                        plugin.PARAMETER_BOUNDS,
                        logger=logging.getLogger(),
                    )
                )
                active = int((~fixed_mask).sum())
                expected_walkers = max(4, 2 * active)
                self.assertEqual(sampling.get("n_walkers"), expected_walkers)
                self.assertEqual(sampling.get("pool_workers"), 0)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
