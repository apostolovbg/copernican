"""Tests for the ``COPERNICAN_SEED`` environment variable."""

import importlib
import os
import random
import unittest
from unittest import mock

import numpy as np

with mock.patch("sys.version_info", (3, 12, 0)):
    copernican = importlib.import_module("copernican")
from copernican_lib import utils


class SeedOptionTestCase(unittest.TestCase):
    """Ensure different environment seeds produce distinct RNG states."""

    def test_env_seed_changes_rng(self):
        """Verify that separate seeds lead to different RNG outputs."""
        values = []
        for seed in (1, 2):
            with mock.patch.dict(
                os.environ, {"COPERNICAN_SEED": str(seed)}, clear=True
            ):
                opts = copernican.get_runtime_options()
            utils.set_random_seed(opts.seed)
            values.append((np.random.rand(), random.random()))
        self.assertNotEqual(values[0], values[1])


if __name__ == "__main__":
    unittest.main()
