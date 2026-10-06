"""Direct tests for line-of-sight projection and spectrum quadrature."""

import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy

from copernican.lib.likelihoods.cmb.runtime import spectrum_projection


class SpectrumProjectionTestCase(unittest.TestCase):
    """Exercise projection kernels without execution orchestration."""

    def test_production_grid_floor_keeps_refinement_nested_and_distinct(self):
        """A phase floor must not collapse the doubled production grid."""

        background = SimpleNamespace(
            eta0=1000.0,
            eta_rec=100.0,
            sound_horizon_mpc=10.0,
        )
        perturbation_data = SimpleNamespace(
            accuracy_controls={
                "accuracy_tier": "final",
                "phase_aware_k_quadrature": True,
                "require_phase_resolution": True,
            },
            manifest_summary={"generated_scalar_hierarchy": True},
        )
        base_numerics = SimpleNamespace(
            ell_min=2,
            ell_max=20,
            k_min=0.01,
            k_max=0.3,
            k_sample_count=8,
            k_grid_refinement_factor=1,
        )
        base = spectrum_projection._build_projection_k_grid(
            ell_arr=numpy.asarray((2, 20)),
            background=background,
            numerics=base_numerics,
            perturbation_data=perturbation_data,
            retain_declared_surface=True,
        )
        refined_numerics = SimpleNamespace(
            **{**vars(base_numerics), "k_grid_refinement_factor": 2}
        )
        refined = spectrum_projection._build_projection_k_grid(
            ell_arr=numpy.asarray((2, 20)),
            background=background,
            numerics=refined_numerics,
            perturbation_data=perturbation_data,
            retain_declared_surface=True,
            refinement_anchors=base,
        )

        self.assertGreater(refined.size, base.size)
        self.assertTrue(numpy.all(numpy.isin(base, refined)))

    def test_projection_quadrature_retains_endpoint_contributions(self):
        """Composite projection weights include both physical endpoints."""

        eta = numpy.asarray((0.0, 0.3, 0.7, 1.0))
        weights = spectrum_projection._simpson_weights(eta)
        endpoint_source = numpy.asarray((1.0, 0.0, 0.0, 2.0))

        self.assertGreater(float(weights[0]), 0.0)
        self.assertGreater(float(weights[-1]), 0.0)
        self.assertGreater(float(numpy.dot(weights, endpoint_source)), 0.0)

    def test_projection_source_does_not_import_camb(self):
        """The declared projection module should remain CAMB-free."""

        source_text = Path(spectrum_projection.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import camb", source_text)

    def test_irregular_log_k_quadrature_uses_stable_positive_weights(self):
        """Phase-aware nodes must not create negative Simpson lobes."""

        log_k = numpy.asarray(
            (-9.0, -7.0, -6.9, -5.0, -2.0, 0.0),
            dtype=numpy.longdouble,
        )
        transfer = numpy.asarray(
            ((1.0, -0.8, 0.7, -0.4, 0.3, -0.1),),
            dtype=numpy.longdouble,
        )
        actual = spectrum_projection._integrate_power_spectrum(
            numpy.ones(log_k.size, dtype=numpy.longdouble),
            log_k,
            transfer,
            transfer,
            auto_spectrum=True,
        )
        self.assertTrue(numpy.all(numpy.isfinite(actual)))
        self.assertGreaterEqual(float(actual[0]), 0.0)

    def test_coarse_projection_preserves_empty_optional_sectors(self):
        """Coarsening scalar kernels must not index absent vector sectors."""

        scalar = numpy.ones((2, 4), dtype=float)
        empty = numpy.empty((2, 0), dtype=float)
        kernel_batch = SimpleNamespace(
            j_l=scalar,
            j_l_derivative=scalar,
            j_l_second_derivative=scalar,
            e_kernel=scalar,
            b_kernel=scalar,
            vector_temperature_1=empty,
            vector_temperature_2=empty,
            vector_e=empty,
            vector_b=empty,
            tensor_temperature=empty,
            tensor_e=empty,
            tensor_b=empty,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DeprecationWarning)
            coarse = spectrum_projection._slice_projection_kernel_batch(
                kernel_batch,
                numpy.asarray((0, 3), dtype=int),
            )

        self.assertFalse(
            any(item.category is DeprecationWarning for item in captured)
        )
        self.assertEqual(coarse.j_l.shape, (2, 2))
        self.assertEqual(coarse.vector_temperature_1.shape, (2, 0))
        self.assertEqual(coarse.tensor_e.shape, (2, 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
