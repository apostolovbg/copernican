"""Tests for ``--seed`` command line option."""

import importlib
import random
import unittest
from unittest import mock

import numpy as np

with mock.patch("sys.version_info", (3, 12, 0)):
    copernican = importlib.import_module("copernican")
from copernican_lib import utils


class SeedOptionTestCase(unittest.TestCase):
    """Ensure different CLI seeds produce distinct RNG states."""

    def test_cli_seed_changes_rng(self):
        """Verify that separate seeds lead to different RNG outputs."""
        values = []
        for seed in (1, 2):
            with mock.patch("sys.argv", ["copernican.py", f"--seed={seed}"]):
                args = copernican.parse_args()
            utils.set_random_seed(args.seed)
            values.append((np.random.rand(), random.random()))
        self.assertNotEqual(values[0], values[1])


if __name__ == "__main__":
    unittest.main()
