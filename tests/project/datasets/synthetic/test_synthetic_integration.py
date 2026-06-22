"""Integration test covering synthetic SNe, BAO and CMB stages."""

from __future__ import annotations

import importlib
import importlib.util as importlib_util
import os
import tempfile
import unittest

import numpy as numpy_module
import yaml

from copernican.engines import engine_mcmc, engine_nested
from copernican.lib import dataset_registry, result_writer, run_manifest, utils
from tests.project.datasets.synthetic import model_plugin

# Restore ``importlib.util`` attribute removed by the frozen importlib shim.
setattr(importlib, "util", importlib_util)


_EXPECTED_HASHES = {
    "bao.csv": (
        "cc98874d217c1fb3a6f1a4acef2ea8bf3a513496bb7d1979b1e8cb949e551654"
    ),
    "cmb.csv": (
        "75eeaa66c50c836a6aa5b86294b6fee2bd5122efd7e019902b78d1ef1bfb6083"
    ),
    "model.yml": (
        "4244f273d80e6456352c62ce877a3a5bfd2d4fb9fb3c491695aa8d1323aa656c"
    ),
    "metadata_synthetic.yml": (
        "61a268cc1df54bc1f901c13d4dc083d8c862977c2cd2fc199403a6d27daa2c47"
    ),
    "sne.csv": (
        "43be03513255fe62c358b19671c27918fb40fbb4bca89f39f8db914b3765831b"
    ),
}


def _load_datasets():
    import importlib

    importlib.import_module(
        "tests.project.datasets.synthetic.cosmo_parser_synthetic"
    )

    sne_df = dataset_registry.load_sne_data("synthetic_integration")
    bao_df = dataset_registry.load_bao_data("synthetic_integration")
    cmb_df = dataset_registry.load_cmb_data("synthetic_integration")
    return sne_df, bao_df, cmb_df


def _dataset_entry(dataset_frame):
    return {
        "id": dataset_frame.attrs["dataset_id"],
        "name": dataset_frame.attrs["dataset_name"],
        "version": dataset_frame.attrs["dataset_version"],
        "path": dataset_frame.attrs["data_path"],
        "hashes": dataset_frame.attrs["file_hashes"],
        "independence": dataset_frame.attrs.get(
            "independence_assumptions", []
        ),
    }


def _assert_hashes(testcase: unittest.TestCase, dataset_frame):
    hashes = dataset_frame.attrs["file_hashes"]
    for key, digest in _EXPECTED_HASHES.items():
        testcase.assertEqual(hashes.get(key), digest)


def _assert_manifest(testcase: unittest.TestCase, manifest, engine_name):
    testcase.assertEqual(manifest["seed"], utils.get_random_seed())
    testcase.assertTrue(manifest["engine"]["name"].endswith(engine_name))
    datasets = manifest["datasets"]
    testcase.assertEqual(set(datasets.keys()), {"synthetic_integration"})
    entry = datasets["synthetic_integration"]
    for key, digest in _EXPECTED_HASHES.items():
        testcase.assertEqual(entry["hashes"].get(key), digest)
    testcase.assertTrue(entry["independence"])


class TestSyntheticIntegration(unittest.TestCase):
    """Exercise the synthetic end-to-end pipeline against both engines."""

    def setUp(self) -> None:
        self._old_dont_write = os.environ.get("PYTHONDONTWRITEBYTECODE")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    def tearDown(self) -> None:
        if self._old_dont_write is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = self._old_dont_write

    def test_synthetic_pipeline(self) -> None:
        plugin = model_plugin.build_plugin()
        utils.set_random_seed(4)
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.set_random_seed(7)
            sne_dataframe, bao_dataframe, cmb_dataframe = _load_datasets()
            for dataset_frame in (
                sne_dataframe,
                bao_dataframe,
                cmb_dataframe,
            ):
                _assert_hashes(self, dataset_frame)

            for engine_module in (engine_mcmc, engine_nested):
                if engine_module is engine_mcmc:
                    fit_result = engine_module.fit_cosmology_parameters(
                        sne_dataframe,
                        plugin,
                        bao_data_df=bao_dataframe,
                        cmb_data_df=cmb_dataframe,
                        n_walkers=6,
                        n_steps=6,
                        burn_in_steps=2,
                        progress_granularity=2,
                        display_progress=False,
                    )
                else:
                    fit_result = engine_module.fit_cosmology_parameters(
                        sne_dataframe,
                        plugin,
                        bao_data_df=bao_dataframe,
                        cmb_data_df=cmb_dataframe,
                        n_live_points=12,
                        max_iterations=20,
                        evidence_tolerance=1e-3,
                        display_progress=False,
                    )
                self.assertTrue(fit_result["success"])
                self.assertTrue(
                    numpy_module.isfinite(
                        fit_result.get("chi2_total", numpy_module.nan)
                    )
                )
                self.assertTrue(fit_result.get("chi2_components", {}))

                engine_results = {plugin.MODEL_NAME: fit_result}
                summary_paths = result_writer.save_summary(
                    engine_results,
                    tmpdir,
                    timestamp="20000101_000000",
                )
                for summary_path in summary_paths:
                    self.assertTrue(utils.compute_sha256(summary_path))

                manifest = run_manifest.build_manifest(
                    models=[(plugin, "0.1")],
                    engine_module=engine_module,
                    datasets=[_dataset_entry(sne_dataframe)],
                )
                manifest_path = os.path.join(
                    tmpdir, "run_manifest_20000101_000000.yml"
                )
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(manifest, handle, sort_keys=False)

                with open(manifest_path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle)
                _assert_manifest(self, loaded, engine_module.__name__)

                manifest_hash = utils.compute_sha256(manifest_path)
                self.assertEqual(
                    manifest_hash, utils.compute_sha256(manifest_path)
                )
                self.assertTrue(manifest_hash)


if __name__ == "__main__":
    unittest.main()
