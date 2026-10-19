"""Direct tests for line-of-sight projection and spectrum quadrature."""

import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy
from scipy.special import sici

from copernican.lib.likelihoods.cmb.runtime import (
    background,
    spectrum_projection,
)
from copernican.lib.perturbation_contract import evaluate_compiled_expression
from tests.copernican.lib import test_perturbation_contract as contract_tests


class SpectrumProjectionTestCase(unittest.TestCase):
    """Exercise projection kernels without execution orchestration."""

    def test_integrated_temperature_quadrupole_matches_angular_thomson_source(
        self,
    ):
        """Integrate -W P_2(mu)/2 independently of LOS integration by parts."""

        compiled = contract_tests.PerturbationContractTestCase()._compile(
            contract_tests._scalar_metadata_only_contract()
        )
        eta = numpy.linspace(0, 1, 257)
        chi = 2 - eta
        kernels = background._get_cached_declared_projection_kernel_batch(
            (2,),
            "manufactured-angular-thomson",
            x_values=2 * chi,
            required_sectors=("scalar",),
        )
        weights = spectrum_projection._simpson_weights(eta)
        polar_cosine, angular_weights = numpy.polynomial.legendre.leggauss(48)
        legendre_two = (3 * polar_cosine**2 - 1) / 2
        # j_l(x) = integral P_l(mu) exp(i x mu) dmu / (2 i^l).
        angular = (
            numpy.real(
                numpy.exp(2j * chi[:, None] * polar_cosine[None, :])
                @ (angular_weights * legendre_two**2)
            )
            / 4
        )
        # Nonzero initial/final moments must retain their angular source.
        for endpoint_offset in (0.0, 0.5):
            weighted_moment = numpy.sin(numpy.pi * eta) ** 2 + endpoint_offset
            context = {
                "visibility": weighted_moment,
                "polarization_moment": 1.0,
            }
            source_histories = {
                compiled.sources[name].role: evaluate_compiled_expression(
                    compiled.sources[name].compiled_expression, context
                )
                for name in (
                    "temperature_quadrupole",
                    "temperature_quadrupole_derivative",
                )
            }
            actual = float(
                spectrum_projection._declared_graph_projection(
                    projection="line_of_sight_temperature",
                    kernel=None,
                    sector="scalar",
                    kernel_batch=kernels,
                    k_value=2.0,
                    eta_weights=weights,
                    chi_grid=chi,
                    source_chi=2.0,
                    source_histories=source_histories,
                )[0]
            )
            expected = float(weights @ (weighted_moment * angular))
            self.assertGreater(abs(expected), 1e-3)
            self.assertAlmostEqual(actual, expected, places=7)

    def test_constant_source_matches_sine_integral_and_detects_faults(
        self,
    ):
        """For ell=0 and chi=eta, integral j_0(eta) d eta = Si(eta)."""

        eta = numpy.linspace(0.0, numpy.pi, 129)
        kernels = background._get_cached_declared_projection_kernel_batch(
            (0,),
            "manufactured-j0-pi-129",
            x_values=eta,
            required_sectors=("scalar",),
        )
        weights = spectrum_projection._simpson_weights(eta)

        def project(source, local_weights=weights):
            return float(
                spectrum_projection._declared_graph_projection(
                    projection="line_of_sight_temperature",
                    kernel=None,
                    sector="scalar",
                    kernel_batch=kernels,
                    k_value=1.0,
                    eta_weights=local_weights,
                    chi_grid=eta,
                    source_chi=numpy.pi,
                    source_histories={"monopole": source},
                )[0]
            )

        expected = float(sici(numpy.pi)[0])
        self.assertAlmostEqual(
            project(numpy.ones(eta.size)), expected, places=8
        )
        spike = numpy.ones(eta.size)
        spike[64] += 1000.0
        missing_tail = numpy.ones(eta.size)
        missing_tail[65:] = 0.0
        for source in (-numpy.ones(eta.size), spike, missing_tail):
            with self.subTest(source=source[64]):
                self.assertGreater(abs(project(source) - expected), 0.1)

    def test_scalar_polarization_projection_has_the_spin_two_normalization(
        self,
    ):
        """j_2(x)/x^2 tends to 1/15; spin raising contributes sqrt(4!)."""

        x_values = numpy.asarray((0.0, 1.0e-5, 2.0e-5))
        kernels = background._get_cached_declared_projection_kernel_batch(
            (2, 3),
            "manufactured-spin2-origin",
            x_values=x_values,
            required_sectors=("scalar",),
        )
        numpy.testing.assert_allclose(
            kernels.e_kernel[0], numpy.sqrt(24.0) / 15.0, rtol=1e-9
        )
        self.assertAlmostEqual(float(kernels.e_kernel[1, 0]), 0.0)
        self.assertGreater(float(kernels.e_kernel[1, -1]), 0.0)

    def test_constant_transfer_power_has_the_logarithmic_support_integral(
        self,
    ):
        """A constant transfer integrates to 4 pi A_s T^2 log(kmax/kmin)."""

        nodes = numpy.log(numpy.geomspace(0.001, 0.1, 17))
        actual = spectrum_projection._integrate_power_spectrum(
            numpy.full(nodes.size, 2e-9),
            nodes,
            numpy.full((1, nodes.size), 3.0),
            numpy.full((1, nodes.size), 3.0),
            auto_spectrum=True,
        )
        expected = 4 * numpy.pi * 2e-9 * 9 * numpy.log(100.0)
        numpy.testing.assert_allclose(actual, (expected,), rtol=1e-12)
        truncated = spectrum_projection._integrate_power_spectrum(
            numpy.full(9, 2e-9),
            nodes[:9],
            numpy.full((1, 9), 3.0),
            numpy.full((1, 9), 3.0),
            auto_spectrum=True,
        )
        self.assertGreater(float(abs(truncated[0] / expected - 1)), 0.4)

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
