"""Tests for the CMB backend registry helpers."""

from __future__ import annotations

import unittest

from copernican_lib import cmb_backend_registry as registry


class CMBBackendRegistryTestCase(unittest.TestCase):
    """Validate the explicit CMB backend capability registry."""

    def test_camb_capabilities_are_explicit(self) -> None:
        """CAMB should advertise the documented capability flags."""

        capabilities = registry.get_backend_capabilities("camb")
        self.assertTrue(capabilities["scalar_param_map"])
        self.assertTrue(capabilities["grids_values_calls"])
        self.assertTrue(capabilities["standard_perturbations"])
        self.assertFalse(capabilities["native_nonstandard_perturbations"])
        self.assertTrue(
            registry.backend_supports_standard_perturbations("camb")
        )
        self.assertFalse(
            registry.backend_supports_native_nonstandard_perturbations("camb")
        )
        self.assertIsNone(
            registry.get_registered_native_solver(
                "camb",
                "template_native_solver",
            )
        )
        self.assertFalse(
            registry.native_solver_is_registered(
                "camb",
                "template_native_solver",
            )
        )

    def test_nonstandard_execution_requires_supported_backend(self) -> None:
        """Unsupported native perturbation execution should fail clearly."""

        with self.assertRaisesRegex(ValueError, "native non-standard"):
            registry.validate_native_perturbation_execution(
                model_name="TemplateModel",
                backend="camb",
                standard=False,
                solver="template_native_solver",
                implemented=False,
            )


if __name__ == "__main__":
    unittest.main()
