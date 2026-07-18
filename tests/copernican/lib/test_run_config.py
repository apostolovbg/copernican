"""Tests for run configuration extraction."""

import unittest

from copernican.lib import run_config as module
from copernican.lib.run_config import (
    DatasetDescriptor,
    build_config_from_manifest,
)


class TestRunConfig(unittest.TestCase):
    """Exercise manifest-to-config conversion."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "EngineDescriptor"))
        self.assertTrue(hasattr(module, "RunConfig"))
        self.assertTrue(hasattr(module, "RunSettings"))
        self.assertTrue(hasattr(module, "build_config_from_manifest"))

    def setUp(self) -> None:
        self.simple_manifest = {
            "seed": 42,
            "selection": {
                "models": ["LambdaCDM"],
                "engine": {
                    "name": "copernican.engines.engine_mcmc",
                    "version": "7.6.20",
                },
            },
            "datasets": {
                "sne/pantheon": {
                    "name": "Pantheon SNe",
                    "type": "sne",
                    "version": "1.0",
                    "path": "/copernican/datasets/sne/pantheon",
                    "hashes": {"data.csv": "abc123"},
                    "independence": ["sne"],
                }
            },
            "configuration": {
                "run_settings": {
                    "engine_kind": "mcmc",
                    "n_steps": 200,
                }
            },
        }

    def test_build_config_from_manifest(self) -> None:
        config = build_config_from_manifest(self.simple_manifest)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.models, ["LambdaCDM"])
        self.assertEqual(
            config.engine.module_name, "copernican.engines.engine_mcmc"
        )
        self.assertEqual(config.engine.version, "7.6.20")
        self.assertEqual(config.run_settings.engine_kind, "mcmc")
        self.assertEqual(config.run_settings.settings["n_steps"], 200)
        self.assertEqual(config.control_model, "model_lcdm.yml")
        self.assertEqual(config.test_model, "LambdaCDM")
        self.assertEqual(len(config.datasets), 1)
        descriptor = config.datasets[0]
        self.assertIsInstance(descriptor, DatasetDescriptor)
        self.assertEqual(descriptor.dataset_id, "sne/pantheon")
        self.assertEqual(descriptor.dataset_type, "sne")
        self.assertEqual(descriptor.version, "1.0")


if __name__ == "__main__":
    unittest.main()
