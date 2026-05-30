"""Tests for CMB capability helpers."""

from __future__ import annotations

import unittest

from copernican_lib import model_coder as registry


class CMBCapabilityTestCase(unittest.TestCase):
    """Validate the explicit CMB backend capability flags."""

    def test_camb_capabilities_are_explicit(self) -> None:
        """CAMB should advertise the documented capability flags."""

        capabilities = registry.get_backend_capabilities("camb")
        self.assertTrue(capabilities["scalar_param_map"])
        self.assertTrue(capabilities["grids_values_calls"])
        self.assertTrue(capabilities["standard_perturbations"])
        self.assertTrue(capabilities["native_nonstandard_perturbations"])
        self.assertTrue(
            registry.backend_supports_standard_perturbations("camb")
        )
        self.assertTrue(
            registry.backend_supports_native_nonstandard_perturbations("camb")
        )

    def test_nonstandard_execution_requires_supported_backend(self) -> None:
        """Unsupported native perturbation execution should fail clearly."""

        with self.assertRaisesRegex(
            ValueError, "generic declarative executor is required"
        ):
            registry.validate_native_perturbation_execution(
                model_name="TemplateModel",
                backend="camb",
                standard=False,
                implemented=False,
            )


if __name__ == "__main__":
    unittest.main()
