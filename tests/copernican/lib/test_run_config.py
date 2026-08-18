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
        self.assertTrue(hasattr(module, "SamplerDescriptor"))
        self.assertTrue(hasattr(module, "RunConfig"))
        self.assertTrue(hasattr(module, "RunSettings"))
        self.assertTrue(hasattr(module, "build_config_from_manifest"))

    def setUp(self) -> None:
        self.simple_manifest = {
            "seed": 42,
            "selection": {
                "models": ["ReferenceModel", "CandidateModel"],
                "comparison": {
                    "control": {"name": "ReferenceModel"},
                    "test": {"name": "CandidateModel"},
                },
                "sampler": {
                    "name": "copernican.samplers.sampler_mcmc",
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
                    "sampler_kind": "mcmc",
                    "n_steps": 200,
                }
            },
        }

    def test_build_config_from_manifest(self) -> None:
        config = build_config_from_manifest(self.simple_manifest)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.models, ["ReferenceModel", "CandidateModel"])
        self.assertEqual(
            config.sampler.module_name, "copernican.samplers.sampler_mcmc"
        )
        self.assertEqual(config.sampler.version, "7.6.20")
        self.assertEqual(config.run_settings.sampler_kind, "mcmc")
        self.assertEqual(config.run_settings.settings["n_steps"], 200)
        self.assertEqual(config.control_model, "ReferenceModel")
        self.assertEqual(config.test_model, "CandidateModel")
        self.assertEqual(len(config.datasets), 1)
        descriptor = config.datasets[0]
        self.assertIsInstance(descriptor, DatasetDescriptor)
        self.assertEqual(descriptor.dataset_id, "sne/pantheon")
        self.assertEqual(descriptor.dataset_type, "sne")
        self.assertEqual(descriptor.version, "1.0")

    def test_single_model_selection_is_rejected(self) -> None:
        self.simple_manifest["selection"]["models"] = ["CandidateModel"]
        with self.assertRaisesRegex(ValueError, "control and test"):
            build_config_from_manifest(self.simple_manifest)


if __name__ == "__main__":
    unittest.main()
