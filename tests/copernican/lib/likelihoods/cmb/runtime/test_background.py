"""Focused tests for the declared CMB background module."""

import unittest
from pathlib import Path

import numpy

from copernican.lib.likelihoods.cmb.runtime import (
    background,
    cache,
    projection,
)


class BackgroundModuleTestCase(unittest.TestCase):
    """Exercise declared background helpers directly."""

    def test_manifest_summary_tracks_declared_background_metadata(self):
        """Manifest summaries should report declared background details."""

        contract = {
            "background": {
                "derived": {
                    "H": "1.0",
                    "rho_tot": "2.0",
                    "p_tot": "0.5",
                    "w_tot": "0.25",
                }
            },
            "param_map": {},
        }

        summary = background._summarize_declared_background_manifest_summary(
            contract
        )

        self.assertIn("H", summary["background_derived_names"])
        self.assertEqual(
            summary["recombination_runtime"]["hydrogen_model"],
            "peebles_case_b_ode",
        )
        self.assertTrue(hasattr(background._CustomCMBBackgroundData, "sample"))

    def test_custom_spectrum_accessors_return_named_payloads(self):
        """Spectrum payload accessors should expose stable arrays."""

        spectrum_data = background.CustomCMBSpectrumData(
            ell_grid=numpy.array([20.0, 30.0]),
            k_grid=numpy.array([0.1, 0.2]),
            transfer_components={
                "temperature": numpy.array([1.0, 2.0]),
                "polarization_e": numpy.array([3.0, 4.0]),
            },
            spectra={
                "TT": numpy.array([5.0, 6.0]),
                "TE": numpy.array([7.0, 8.0]),
                "EE": numpy.array([9.0, 10.0]),
            },
        )

        self.assertIs(
            type(spectrum_data),
            background.CustomCMBSpectrumData,
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.Delta_l_T, numpy.array([1.0, 2.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.Delta_l_E, numpy.array([3.0, 4.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_TT, numpy.array([5.0, 6.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_TE, numpy.array([7.0, 8.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_EE, numpy.array([9.0, 10.0]))
        )

    def test_custom_transfer_payload_is_read_only(self):
        """Transfer-only payloads should freeze grids and component arrays."""

        transfer_data = background.CustomCMBTransferData(
            ell_grid=numpy.array([20.0, 30.0]),
            k_grid=numpy.array([0.1, 0.2]),
            transfer_components={
                "temperature": numpy.array([[1.0, 2.0], [3.0, 4.0]])
            },
            runtime_envelope={"accuracy_tier": "final"},
        )

        self.assertIs(
            type(transfer_data),
            background.CustomCMBTransferData,
        )
        self.assertEqual(
            transfer_data.runtime_envelope["accuracy_tier"],
            "final",
        )
        self.assertFalse(transfer_data.ell_grid.flags.writeable)
        self.assertFalse(
            transfer_data.transfer_components["temperature"].flags.writeable
        )

    def test_spectrum_payload_is_read_only_and_missing_values_fail(self):
        """Cached outputs must not fabricate or permit mutated spectra."""

        spectrum_data = background.CustomCMBSpectrumData(
            ell_grid=numpy.array([20, 30]),
            k_grid=numpy.array([0.1, 0.2]),
            transfer_components={
                "temperature": numpy.array([1.0, 2.0]),
            },
            spectra={"TT": numpy.array([3.0, 4.0])},
            spectrum_availability={"TT": "computed", "EE": "unrequested"},
        )

        with self.assertRaisesRegex(KeyError, "EE.*unrequested"):
            _ = spectrum_data.C_l_EE
        with self.assertRaisesRegex(KeyError, "polarization_e.*unavailable"):
            _ = spectrum_data.Delta_l_E
        with self.assertRaises(ValueError):
            spectrum_data.C_l_TT[0] = 0.0
        with self.assertRaises(TypeError):
            spectrum_data.spectra["TT"] = numpy.zeros(2)

    def test_background_source_does_not_import_camb(self):
        """The declared background module should remain CAMB-free."""

        source_text = Path(background.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import camb", source_text)

    def test_mode_bessel_batch_matches_individual_grids(self):
        """Shared mode radial recurrence must preserve individual values."""

        ell_signature = (2, 7, 20)
        mode_x = numpy.asarray(
            (
                (0.0, 0.2, 1.0, 4.0),
                (0.0, 0.5, 2.0, 8.0),
                (0.0, 0.9, 3.0, 12.0),
            ),
            dtype=float,
        )
        batched_values, batched_derivatives = (
            background._compute_spherical_bessel_mode_batch(
                ell_signature,
                mode_x,
            )
        )
        for mode_index, x_values in enumerate(mode_x):
            values, derivatives = background._compute_spherical_bessel_batch(
                ell_signature,
                x_values,
            )
            numpy.testing.assert_allclose(
                batched_values[:, mode_index, :],
                values,
                rtol=1.0e-13,
                atol=1.0e-13,
            )
            numpy.testing.assert_allclose(
                batched_derivatives[:, mode_index, :],
                derivatives,
                rtol=1.0e-13,
                atol=1.0e-13,
            )

    def test_bessel_batch_matches_reference_at_zero_and_negative_arguments(
        self,
    ):
        """Radial values and derivatives preserve endpoint parity limits."""

        ell_signature = (0, 1, 2, 5, 12)
        x_values = numpy.asarray((-2.5, -0.4, 0.0, 0.4, 2.5))
        values, derivatives = background._compute_spherical_bessel_batch(
            ell_signature,
            x_values,
        )
        ell_array = numpy.asarray(ell_signature, dtype=int)[:, None]
        positive_x = numpy.abs(x_values)
        expected_values = background.spherical_jn(
            ell_array,
            positive_x[None, :],
        )
        expected_derivatives = background.spherical_jn(
            ell_array,
            positive_x[None, :],
            derivative=True,
        )
        negative_mask = x_values < 0.0
        value_parity = numpy.where(ell_array % 2 == 0, 1.0, -1.0)
        derivative_parity = -value_parity
        expected_values[:, negative_mask] *= value_parity
        expected_derivatives[:, negative_mask] *= derivative_parity
        numpy.testing.assert_allclose(
            values,
            expected_values,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            derivatives,
            expected_derivatives,
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    def test_sparse_high_order_bessel_batch_matches_reference(self):
        """Sparse high-order requests must retain the SciPy radial values."""

        ell_signature = (0, 1, 2, 80, 400, 1600)
        x_values = numpy.asarray((1.0e-8, 0.2, 12.0, 250.0, 1800.0))
        values, derivatives = background._compute_spherical_bessel_batch(
            ell_signature,
            x_values,
        )
        ell_array = numpy.asarray(ell_signature, dtype=int)[:, None]
        expected_values = background.spherical_jn(
            ell_array,
            x_values[None, :],
        )
        expected_derivatives = background.spherical_jn(
            ell_array,
            x_values[None, :],
            derivative=True,
        )

        numpy.testing.assert_allclose(
            values,
            expected_values,
            rtol=1.0e-10,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            derivatives,
            expected_derivatives,
            rtol=1.0e-10,
            atol=1.0e-14,
        )

    def test_projection_kernels_preserve_signed_parity_and_zero_limits(self):
        """All sector kernels remain finite and parity-consistent at ends."""

        x_values = numpy.asarray((-1.25, 0.0, 1.25), dtype=float)
        x_signature = "slice-twenty-two-kernel-endpoints"
        cache.store_bessel_inputs(x_signature, x_values)
        kernel_batch = background._get_cached_declared_projection_kernel_batch(
            (2, 3),
            x_signature,
            required_sectors=("vector", "tensor"),
        )
        for array in (
            kernel_batch.j_l_second_derivative,
            kernel_batch.e_kernel,
            kernel_batch.b_kernel,
            kernel_batch.vector_temperature_1,
            kernel_batch.vector_temperature_2,
            kernel_batch.vector_e,
            kernel_batch.vector_b,
            kernel_batch.tensor_temperature,
            kernel_batch.tensor_e,
            kernel_batch.tensor_b,
        ):
            self.assertTrue(numpy.all(numpy.isfinite(array)))
        self.assertAlmostEqual(
            float(kernel_batch.j_l_second_derivative[0, 1]),
            2.0 / 15.0,
        )
        self.assertAlmostEqual(
            float(kernel_batch.e_kernel[0, 1]),
            numpy.sqrt(24.0) / 15.0,
        )
        self.assertAlmostEqual(
            float(kernel_batch.vector_temperature_2[0, 1]),
            1.0 / 5.0,
        )
        self.assertAlmostEqual(
            float(kernel_batch.vector_e[0, 1]),
            numpy.sqrt(8.0) / 10.0,
        )
        self.assertAlmostEqual(
            float(kernel_batch.tensor_temperature[0, 1]),
            1.0 / 5.0,
        )
        self.assertAlmostEqual(
            float(kernel_batch.tensor_e[0, 1]),
            1.0 / 15.0,
        )
        parity = numpy.ones(2, dtype=float)
        numpy.testing.assert_allclose(
            kernel_batch.e_kernel[0, (0, 2)],
            kernel_batch.e_kernel[0, 2] * parity,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.b_kernel[0, (0, 2)],
            kernel_batch.b_kernel[0, 2] * numpy.asarray((-1.0, 1.0)),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        numpy.testing.assert_allclose(
            kernel_batch.tensor_e[0, (0, 2)],
            kernel_batch.tensor_e[0, 2] * parity,
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    def test_projection_rejects_incompatible_sector_before_integration(self):
        """The projector must reject a sector before multiplying histories."""

        x_signature = "slice-twenty-two-sector-rejection"
        cache.store_bessel_inputs(
            x_signature,
            numpy.asarray((0.5, 1.0), dtype=float),
        )
        kernel_batch = background._get_cached_declared_projection_kernel_batch(
            (2,),
            x_signature,
        )
        with self.assertRaisesRegex(ValueError, "incompatible with sector"):
            projection._declared_graph_projection(
                projection="line_of_sight_vector_temperature",
                kernel="spherical_bessel_window",
                sector="scalar",
                kernel_batch=kernel_batch,
                k_value=1.0,
                eta_weights=numpy.ones(2, dtype=float),
                chi_grid=numpy.zeros(2, dtype=float),
                source_chi=1.0,
                source_histories={
                    "signal": numpy.ones(2, dtype=float),
                },
            )

    def test_projection_kernel_rebuild_uses_mode_grid_after_cache_eviction(
        self,
    ) -> None:
        """An input-cache eviction must not lose the active mode grid."""

        cache.clear_cmb_parameter_caches()
        try:
            x_signature = "slice-deterministic-projection-eviction-active"
            x_values = numpy.asarray((0.25, 0.75, 1.5), dtype=float)
            for index in range(513):
                cache.store_bessel_inputs(
                    f"slice-deterministic-projection-eviction-{index}",
                    x_values,
                )
            self.assertIsNone(cache.get_bessel_inputs(x_signature))

            kernel_batch = (
                background._get_cached_declared_projection_kernel_batch(
                    (2, 5),
                    x_signature,
                    x_values=x_values,
                    required_sectors=("scalar",),
                )
            )
            expected_values = background._compute_spherical_bessel_batch(
                (2, 5),
                x_values,
            )[0]

            numpy.testing.assert_allclose(
                kernel_batch.j_l,
                expected_values,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        finally:
            cache.clear_cmb_parameter_caches()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
