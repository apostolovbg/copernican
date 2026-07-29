"""Tests for declared CMB projection role validation helpers."""

from __future__ import annotations

import unittest

from copernican.lib.cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTION_KERNELS,
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    DeclaredProjectionKernelSpec,
    DeclaredProjectionSpec,
    get_declared_projection_kernel_spec,
    get_declared_projection_spec,
    resolve_declared_projection_kernel,
    resolve_declared_source_kernel,
    validate_declared_projection_source_roles,
)


class CMBProjectionContractTestCase(unittest.TestCase):
    """Validate projection-role contracts for declared observables."""

    def test_supported_projection_list_includes_native_components(
        self,
    ) -> None:
        """The supported surface should list native dispatch keys."""

        self.assertIn(
            "line_of_sight_temperature",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )
        self.assertIn(
            "line_of_sight_polarization_e",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )
        self.assertIn(
            "spin2_b_mode",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )
        self.assertIn(
            "custom_line_of_sight",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )
        self.assertIn(
            "spin2_b_window",
            SUPPORTED_DECLARED_TRANSFER_PROJECTION_KERNELS,
        )
        self.assertIn(
            "spherical_bessel_second_derivative_window",
            SUPPORTED_DECLARED_TRANSFER_PROJECTION_KERNELS,
        )

    def test_required_roles_pass_validation(self) -> None:
        """Validation should accept the declared roles for a projection."""

        self.assertIsNone(
            validate_declared_projection_source_roles(
                projection="line_of_sight_temperature",
                observable_name="temperature",
                source_roles={"monopole", "doppler", "isw", "additive"},
            )
        )

    def test_custom_projection_accepts_declared_roles(self) -> None:
        """Generic custom projections should accept explicit role mappings."""

        self.assertIsNone(
            validate_declared_projection_source_roles(
                projection="custom_line_of_sight",
                observable_name="custom_signal",
                source_roles={"signal", "signal_aux"},
            )
        )

    def test_missing_required_role_fails(self) -> None:
        """Validation should reject incomplete declared role mappings."""

        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles",
        ):
            validate_declared_projection_source_roles(
                projection="line_of_sight_polarization_e",
                observable_name="polarization_e",
                source_roles=set(),
            )

    def test_custom_projection_requires_at_least_one_role(self) -> None:
        """Custom line-of-sight projections should reject empty bindings."""

        with self.assertRaisesRegex(
            ValueError,
            "requires at least one declared source-term role",
        ):
            validate_declared_projection_source_roles(
                projection="custom_line_of_sight",
                observable_name="custom_signal",
                source_roles=set(),
            )

    def test_unexpected_role_fails(self) -> None:
        """Validation should reject extra roles for strict projections."""

        with self.assertRaisesRegex(
            ValueError,
            "does not accept source-term roles",
        ):
            validate_declared_projection_source_roles(
                projection="line_of_sight_lensing_potential",
                observable_name="lensing",
                source_roles={"potential", "signal"},
            )

    def test_unsupported_projection_fails(self) -> None:
        """Validation should reject undeclared transfer projections."""

        with self.assertRaisesRegex(
            ValueError,
            "uses unsupported projection",
        ):
            validate_declared_projection_source_roles(
                projection="unsupported_projection",
                observable_name="custom_signal",
                source_roles={"signal"},
            )

    def test_projection_spec_tracks_declared_roles(self) -> None:
        """Projection specs should preserve the declared role contract."""

        spec = DeclaredProjectionSpec(
            name="custom_projection",
            required_roles=("potential",),
            allowed_roles=("potential", "signal"),
        )

        self.assertIsInstance(spec, DeclaredProjectionSpec)
        self.assertEqual(spec.name, "custom_projection")
        self.assertEqual(spec.required_roles, ("potential",))
        self.assertEqual(spec.allowed_roles, ("potential", "signal"))

    def test_kernel_spec_lookup_returns_native_contract(self) -> None:
        """Kernel lookups should expose immutable runtime metadata."""

        spec = get_declared_projection_kernel_spec("spin2_e_window")

        self.assertIsInstance(spec, DeclaredProjectionKernelSpec)
        self.assertEqual(spec.name, "spin2_e_window")
        self.assertEqual(spec.kind, "spin2_e")
        self.assertEqual(
            spec.description,
            "Spin-2 even-parity polarization kernel.",
        )

    def test_projection_spec_lookup_returns_native_contract(self) -> None:
        """Projection lookups should expose the native immutable contracts."""

        spec = get_declared_projection_spec("spin2_b_mode")

        self.assertEqual(spec.name, "spin2_b_mode")
        self.assertEqual(spec.required_roles, ("polarization_b",))
        self.assertEqual(spec.output_role, "polarization_b")
        self.assertEqual(spec.transfer_units, "dimensionless")
        self.assertEqual(spec.default_kernel, "spin2_b_window")
        self.assertTrue(spec.requires_odd_parity_source)

    def test_temperature_source_roles_select_reviewed_kernels(self) -> None:
        """Temperature roles retain their declared radial conventions."""

        self.assertEqual(
            resolve_declared_source_kernel(
                "line_of_sight_temperature",
                "monopole",
                kernel="temperature_mixed_window",
            ),
            "spherical_bessel_window",
        )
        self.assertEqual(
            resolve_declared_source_kernel(
                "line_of_sight_temperature",
                "doppler",
                kernel="temperature_mixed_window",
            ),
            "spherical_bessel_derivative_window",
        )
        self.assertEqual(
            resolve_declared_source_kernel(
                "line_of_sight_temperature",
                "additive_derivative",
                kernel="temperature_mixed_window",
            ),
            "spherical_bessel_second_derivative_window",
        )

    def test_temperature_source_role_rejects_undeclared_role(self) -> None:
        """Temperature projection cannot silently ignore a source role."""

        with self.assertRaisesRegex(ValueError, "does not define source role"):
            resolve_declared_source_kernel(
                "line_of_sight_temperature",
                "unclassified",
                kernel="temperature_mixed_window",
            )

    def test_projection_extension_inherits_and_overrides_kernel(self) -> None:
        """Declared projection extensions should expose the resolved spec."""

        extensions = {
            "signal_derivative_alias": {
                "base_projection": "custom_line_of_sight",
                "kernel": "spherical_bessel_derivative_window",
                "required_roles": ["signal"],
                "allowed_roles": ["signal"],
            }
        }

        spec = get_declared_projection_spec(
            "signal_derivative_alias",
            extensions=extensions,
        )

        self.assertEqual(spec.name, "signal_derivative_alias")
        self.assertEqual(spec.required_roles, ("signal",))
        self.assertEqual(spec.output_role, "signal")
        self.assertEqual(spec.transfer_units, "dimensionless")
        self.assertEqual(
            spec.default_kernel,
            "spherical_bessel_derivative_window",
        )
        self.assertEqual(
            spec.supported_kernels,
            ("spherical_bessel_derivative_window",),
        )

    def test_projection_extension_validates_declared_roles(self) -> None:
        """Declared projection extensions should reuse role validation."""

        extensions = {
            "signal_derivative_alias": {
                "base_projection": "custom_line_of_sight",
                "kernel": "spherical_bessel_derivative_window",
                "required_roles": ["signal"],
                "allowed_roles": ["signal"],
            }
        }

        self.assertIsNone(
            validate_declared_projection_source_roles(
                projection="signal_derivative_alias",
                observable_name="signal_transfer",
                source_roles={"signal"},
                extensions=extensions,
            )
        )
        self.assertEqual(
            resolve_declared_projection_kernel(
                "signal_derivative_alias",
                observable_name="signal_transfer",
                kernel=None,
                extensions=extensions,
            ),
            "spherical_bessel_derivative_window",
        )

    def test_custom_projection_requires_explicit_kernel(self) -> None:
        """Custom line-of-sight projections should require a kernel."""

        with self.assertRaisesRegex(ValueError, "must declare kernel"):
            resolve_declared_projection_kernel(
                "custom_line_of_sight",
                observable_name="custom_signal",
                kernel=None,
            )

    def test_builtin_projection_rejects_custom_kernel_override(self) -> None:
        """Builtin projections should keep their reviewed kernel contract."""

        with self.assertRaisesRegex(ValueError, "does not support kernel"):
            resolve_declared_projection_kernel(
                "line_of_sight_signal",
                observable_name="signal_transfer",
                kernel="spin2_e_window",
            )

    def test_custom_projection_accepts_supported_kernel(self) -> None:
        """Custom line-of-sight projections should allow reviewed kernels."""

        self.assertEqual(
            resolve_declared_projection_kernel(
                "custom_line_of_sight",
                observable_name="custom_signal",
                kernel="lensing_potential_window",
            ),
            "lensing_potential_window",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
