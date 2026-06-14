"""Tests for declared CMB projection role validation helpers."""

from __future__ import annotations

import unittest

from copernican.lib.cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    DeclaredProjectionSpec,
    get_declared_projection_spec,
    validate_declared_projection_source_roles,
)


class CMBProjectionContractTestCase(unittest.TestCase):
    """Validate projection-role contracts for declared observables."""

    def test_supported_projection_list_includes_native_components(
        self,
    ) -> None:
        """The supported surface should list native dispatch keys."""

        self.assertIn(
            "cmb_temperature_scalar",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )
        self.assertIn(
            "cmb_polarization_e_scalar",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )
        self.assertIn(
            "spin2_b_mode",
            SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
        )

    def test_required_roles_pass_validation(self) -> None:
        """Validation should accept the declared roles for a projection."""

        self.assertIsNone(
            validate_declared_projection_source_roles(
                projection="cmb_temperature_scalar",
                observable_name="temperature",
                source_roles={"monopole", "doppler", "isw", "additive"},
            )
        )

    def test_missing_required_role_fails(self) -> None:
        """Validation should reject incomplete declared role mappings."""

        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles",
        ):
            validate_declared_projection_source_roles(
                projection="cmb_polarization_e_scalar",
                observable_name="polarization_e",
                source_roles=set(),
            )

    def test_unexpected_role_fails(self) -> None:
        """Validation should reject extra roles for strict projections."""

        with self.assertRaisesRegex(
            ValueError,
            "does not accept source-term roles",
        ):
            validate_declared_projection_source_roles(
                projection="cmb_lensing_potential_scalar",
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

    def test_projection_spec_lookup_returns_native_contract(self) -> None:
        """Projection lookups should expose the native immutable contracts."""

        spec = get_declared_projection_spec("spin2_b_mode")

        self.assertEqual(spec.name, "spin2_b_mode")
        self.assertEqual(spec.required_roles, ("polarization_b",))
        self.assertTrue(spec.requires_odd_parity_source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
