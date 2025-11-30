# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Regression tests for :mod:`copernican_lib.plugins` pickling helpers."""

import pickle
import unittest

from copernican_lib import plugins

def distance_modulus_model(z_val, h0):
    """Toy distance modulus helper that stays trivially picklable."""

    return float(z_val) + float(h0)

def get_comoving_distance_Mpc(z_val, h0):
    """Return a linearised comoving distance for testing only."""

    return float(z_val) * 100.0 / max(float(h0), 1.0)

def get_luminosity_distance_Mpc(z_val, h0):
    """Derive luminosity distance directly from the comoving result."""

    return (1.0 + float(z_val)) * get_comoving_distance_Mpc(z_val, h0)

def get_angular_diameter_distance_Mpc(z_val, h0):
    """Derive angular diameter distance from the comoving helper."""

    return get_comoving_distance_Mpc(z_val, h0) / (1.0 + float(z_val))

def get_Hz_per_Mpc(z_val, h0):
    """Return a monotonic H(z) scaling for deterministic assertions."""

    return float(h0) * (1.0 + float(z_val))

def get_DV_Mpc(z_val, h0):
    """Return a BAO-inspired helper anchored to the comoving distance."""

    dm_val = get_comoving_distance_Mpc(z_val, h0)
    numerator = dm_val * dm_val * 299792.458 * float(z_val)
    ratio = numerator / get_Hz_per_Mpc(z_val, h0)
    return ratio ** (1.0 / 3.0)

def get_sound_horizon_rs_Mpc(h0):
    """Simple sound horizon approximation suitable for tests."""

    return 144.0 / max(float(h0), 1.0)

def helper_extra_function():
    """Extra helper stored on the plugin to prove extras stay intact."""

    return "extra"

def _build_sample_plugin() -> plugins.EnginePlugin:
    """Create a minimal plugin suitable for pickling tests."""

    model_data = {
        "model_name": "TestModel",
        "description": "Synthetic plugin used for pickling tests.",
        "abstract": "Ensures FrozenMapping wrappers survive pickle.",
        "parameters": [
            {
                "name": "H0",
                "latex_name": "$H_0$",
                "unit": "km/s/Mpc",
                "bounds": (60.0, 80.0),
                "prior": {"type": "fixed", "value": 70.0},
            }
        ],
        "equations": {"sne": ["H_0"], "bao": []},
        "likelihood": {"datasets": ["sne"]},
    }
    func_dict = {
        "distance_modulus_model": distance_modulus_model,
        "get_comoving_distance_Mpc": get_comoving_distance_Mpc,
        "get_luminosity_distance_Mpc": get_luminosity_distance_Mpc,
        "get_angular_diameter_distance_Mpc": get_angular_diameter_distance_Mpc,
        "get_Hz_per_Mpc": get_Hz_per_Mpc,
        "get_DV_Mpc": get_DV_Mpc,
        "get_sound_horizon_rs_Mpc": get_sound_horizon_rs_Mpc,
        "custom_extra": helper_extra_function,
    }
    return plugins.build_engine_plugin(model_data, func_dict)

class FrozenMappingTests(unittest.TestCase):
    """Validate the FrozenMapping wrapper used across EnginePlugin fields."""

    def test_engine_plugin_pickles_with_frozen_mappings(self) -> None:
        """EnginePlugin should survive pickle round-trips under spawn pools."""

        plugin = _build_sample_plugin()
        payload = pickle.dumps(plugin)
        clone = pickle.loads(payload)

        self.assertIsInstance(plugin.extras, plugins.FrozenMapping)
        self.assertIsInstance(clone.extras, plugins.FrozenMapping)
        self.assertIn("custom_extra", clone.extras)
        self.assertEqual(clone.extras["custom_extra"](), "extra")
        self.assertAlmostEqual(clone.FIXED_PARAMS["H0"], 70.0)

    def test_frozen_mapping_to_dict_returns_copy(self) -> None:
        """The FrozenMapping copy helper must not expose internal state."""

        plugin = _build_sample_plugin()
        extras_copy = plugin.extras.to_dict()
        extras_copy["custom_extra"] = "shadowed"
        self.assertEqual(plugin.extras["custom_extra"](), "extra")

if __name__ == "__main__":  # pragma: no cover - unittest cli support
    unittest.main()
