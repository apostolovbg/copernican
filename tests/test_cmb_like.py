"""Unit tests for CAMB-backed CMB helpers.

**Last Updated:** 2025-11-01
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from copernican_lib import engine_interface
from copernican_lib import model_coder
from copernican_lib import model_parser
from copernican_lib.likelihoods import cmb


class CMBBackgroundTestCase(unittest.TestCase):
    """Validate CAMB background helpers share settings with the spectra API."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare a ΛCDM plugin for evaluating CAMB helpers."""

        repo_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("VIRTUAL_ENV", str(repo_root / ".venv"))
        yaml_path = repo_root / "models" / "cosmo_model_lcdm.yml"
        cache_dir = repo_root / "models" / "cache"
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_interface.build_plugin(parsed, funcs)

    def test_background_observables_match_input_length(self) -> None:
        """Background helper should return one entry per requested redshift."""

        params = self.plugin.get_camb_params(self.plugin.INITIAL_GUESSES)
        redshifts = np.array([0.15, 0.35, 0.57])
        background = cmb.compute_camb_background_observables(params, redshifts)

        self.assertEqual(background["DM"].shape, redshifts.shape)
        self.assertEqual(background["DH"].shape, redshifts.shape)
        self.assertEqual(background["DV"].shape, redshifts.shape)
        self.assertGreater(background["rs_drag"], 0.0)
        self.assertTrue(np.all(np.isfinite(background["DM"])))

    def test_background_cache_collapses_duplicate_redshifts(self) -> None:
        """Repeated redshifts should produce identical background distances."""

        params = self.plugin.get_camb_params(self.plugin.INITIAL_GUESSES)
        redshifts = np.array([0.35, 0.35, 0.60])
        background = cmb.compute_camb_background_observables(params, redshifts)

        self.assertAlmostEqual(background["DM"][0], background["DM"][1], places=12)
        self.assertAlmostEqual(background["DH"][0], background["DH"][1], places=12)


if __name__ == "__main__":
    unittest.main()
