# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Regression tests for :mod:`copernican_lib.engine_adapter` helpers."""

import multiprocessing as multiprocessing_module
import unittest

from copernican_lib import engine_adapter as plugins


def distance_modulus_model(z_val, hubble_parameter):
    """Toy distance modulus helper that stays trivially picklable."""

    return float(z_val) + float(hubble_parameter)


def get_comoving_distance_Mpc(z_val, hubble_parameter):
    """Return a linearised comoving distance for testing only."""

    return float(z_val) * 100.0 / max(float(hubble_parameter), 1.0)


def get_luminosity_distance_Mpc(z_val, hubble_parameter):
    """Derive luminosity distance directly from the comoving result."""

    return (1.0 + float(z_val)) * get_comoving_distance_Mpc(
        z_val, hubble_parameter
    )


def get_angular_diameter_distance_Mpc(z_val, hubble_parameter):
    """Derive angular diameter distance from the comoving helper."""

    return get_comoving_distance_Mpc(z_val, hubble_parameter) / (
        1.0 + float(z_val)
    )


def get_Hz_per_Mpc(z_val, hubble_parameter):
    """Return a monotonic H(z) scaling for deterministic assertions."""

    return float(hubble_parameter) * (1.0 + float(z_val))


def get_DV_Mpc(z_val, hubble_parameter):
    """Return a BAO-inspired helper anchored to the comoving distance."""

    dm_val = get_comoving_distance_Mpc(z_val, hubble_parameter)
    numerator = dm_val * dm_val * 299792.458 * float(z_val)
    ratio = numerator / get_Hz_per_Mpc(z_val, hubble_parameter)
    return ratio ** (1.0 / 3.0)


def get_sound_horizon_rs_Mpc(hubble_parameter):
    """Simple sound horizon approximation suitable for tests."""

    return 144.0 / max(float(hubble_parameter), 1.0)


def helper_extra_function():
    """Extra helper stored on the plugin to prove extras stay intact."""

    return "extra"


def _inspect_plugin(plugin: plugins.EnginePlugin):
    """Return round-trip observations from a worker process."""

    return (
        isinstance(plugin.extras, plugins.FrozenMapping),
        plugin.extras["custom_extra"](),
        plugin.FIXED_PARAMS["H0"],
    )


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
        with multiprocessing_module.get_context("spawn").Pool(1) as pool:
            is_frozen, custom_value, fixed_h0 = pool.apply(
                _inspect_plugin,
                (plugin,),
            )

        self.assertIsInstance(plugin.extras, plugins.FrozenMapping)
        self.assertTrue(is_frozen)
        self.assertEqual(custom_value, "extra")
        self.assertAlmostEqual(fixed_h0, 70.0)

    def test_frozen_mapping_to_dict_returns_copy(self) -> None:
        """The FrozenMapping copy helper must not expose internal state."""

        plugin = _build_sample_plugin()
        extras_copy = plugin.extras.to_dict()
        extras_copy["custom_extra"] = "shadowed"
        self.assertEqual(plugin.extras["custom_extra"](), "extra")


if __name__ == "__main__":  # pragma: no cover - unittest cli support
    unittest.main()
