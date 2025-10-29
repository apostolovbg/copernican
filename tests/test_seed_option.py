"""Tests for RNG seed selection paths."""

import importlib
import os
import random
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

with mock.patch("sys.version_info", (3, 12, 0)):
    with mock.patch.dict(
        os.environ,
        {"VIRTUAL_ENV": str(Path(__file__).resolve().parents[1] / ".venv")},
    ):
        copernican = importlib.import_module("copernican")
from copernican_lib import utils


class SeedMenuTestCase(unittest.TestCase):
    """Verify environment, manual, random and default seed handling."""

    def test_env_seed_changes_rng(self):
        """Separate environment seeds yield distinct RNG outputs."""
        values = []
        for seed in (1, 2):
            with mock.patch.dict(
                os.environ, {"COPERNICAN_SEED": str(seed)}, clear=True
            ):
                copernican.select_seed()
            values.append((np.random.rand(), random.random()))
        self.assertNotEqual(values[0], values[1])

    def test_manual_seed_prompt(self):
        """User-entered seeds are applied."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input", side_effect=["2", "42"]):
                copernican.select_seed()
        self.assertEqual(utils.get_random_seed(), 42)

    def test_random_seed_prompt(self):
        """Random seed generation stores the value."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input", side_effect=["3"]):
                with mock.patch("random.randint", return_value=99):
                    copernican.select_seed()
        self.assertEqual(utils.get_random_seed(), 99)

    def test_default_seed_prompt(self):
        """Accepting the default seed stores zero."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input", side_effect=[""]):
                copernican.select_seed()
        self.assertEqual(utils.get_random_seed(), 0)


if __name__ == "__main__":
    unittest.main()
