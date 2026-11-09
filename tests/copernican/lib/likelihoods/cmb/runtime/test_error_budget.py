"""Manufactured observable errors independent of the execution machinery."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy
from scipy.interpolate import CubicSpline

from copernican.lib.likelihoods.cmb.runtime.error_budget import (
    NativeSourceProbe,
    combine_observable_errors,
    project_source_uncertainty,
    source_representation_envelope,
    spline_integral_weights,
    transfer_spectrum_error,
    update_source_error_envelope,
)


class ObservableErrorBudgetTestCase(unittest.TestCase):
    """Operator conditioning and axis cancellation must remain visible."""

    def test_native_samples_bound_unresolved_source_time_oscillations(self):
        """Envelope resampling cannot erase an error between source nodes."""

        sparse = numpy.linspace(0, 1, 9)
        native = numpy.linspace(0, 1, 1001)
        samples = numpy.sin(12 * numpy.pi * sparse)
        truth = numpy.sin(12 * numpy.pi * native)
        relative, absolute, errors = source_representation_envelope(
            sparse, {"s": samples}, native, {"s": truth}
        )
        interpolated = CubicSpline(sparse, samples)(native)
        scale = numpy.max(abs(interpolated))
        bound = numpy.interp(
            native, sparse, relative["s"]
        ) * scale + numpy.interp(native, sparse, absolute["s"])
        self.assertTrue(numpy.all(abs(interpolated - truth) <= bound + 1e-14))
        self.assertGreater(errors["s"]["relative_error"], 0.01)
        relative, absolute, errors = source_representation_envelope(
            sparse,
            {"s": numpy.zeros(9)},
            native,
            {"s": numpy.sin(8 * numpy.pi * native) ** 2},
        )
        self.assertTrue(numpy.all(relative["s"] == 0))
        self.assertGreater(numpy.max(absolute["s"]), 0.9)
        self.assertEqual(errors["s"]["relative_error"], 1.0)

    def test_zero_source_retains_absolute_uncertainty_in_projection(self):
        from copernican.lib.likelihoods.cmb.runtime import (
            background,
            spectrum_projection,
        )

        eta = numpy.linspace(0, 10, 17)
        fine, comparison = (
            {"signal": numpy.zeros(17)},
            {"signal": numpy.full(17, 0.2)},
        )
        relative, absolute = {}, {}
        update_source_error_envelope(
            relative, fine, comparison, absolute_envelope=absolute
        )
        numpy.testing.assert_array_equal(absolute["signal"], 0.2)
        kernels = background._get_cached_declared_projection_kernel_batch(
            (2,),
            "zero-source-error",
            x_values=eta,
            required_sectors=("scalar",),
        )
        inputs = dict(
            kernel_batch=kernels,
            k_value=1.0,
            eta_weights=numpy.ones(17),
            chi_grid=eta,
            source_chi=10.0,
        )
        component = SimpleNamespace(
            source_terms={"monopole": "signal"},
            projection="line_of_sight_temperature",
            kernel=None,
            sector="scalar",
        )
        bound = project_source_uncertainty(
            component, fine, relative, absolute_profiles=absolute, **inputs
        )
        actual = spectrum_projection._declared_graph_projection(
            projection=component.projection,
            kernel=None,
            sector="scalar",
            source_histories={"monopole": comparison["signal"]},
            **inputs,
        )
        self.assertGreater(float(bound[0]), 0.0)
        self.assertTrue(numpy.all(bound >= abs(actual)))

    def test_geometry_changes_multiply_other_source_axis_errors(self):
        from copernican.lib.likelihoods.cmb.runtime import spectrum_projection

        calls = []

        def operators(*, operator_sink, **_kwargs):
            operator_sink["monopole"] = numpy.full(
                (1, 3), 2.0 if calls else 0.0
            )
            calls.append(True)

        component = SimpleNamespace(
            source_terms={"monopole": "signal"},
            projection="line_of_sight_temperature",
            kernel=None,
            sector="scalar",
        )
        with mock.patch.object(
            spectrum_projection,
            "_declared_graph_projection",
            side_effect=operators,
        ):
            bound = project_source_uncertainty(
                component,
                {"signal": numpy.zeros(3)},
                {"signal": numpy.zeros(3)},
                geometry_reference={
                    "ells": (2,),
                    "eta0_ratio": 2.0,
                    "source_chi": 3.0,
                },
                geometry_other_profiles=(
                    ({}, {"signal": numpy.full(3, 0.2)}),
                ),
                kernel_batch=None,
                k_value=1.0,
                eta_weights=numpy.ones(3),
                chi_grid=numpy.arange(1.0, 4.0),
                source_chi=3.0,
            )
        self.assertAlmostEqual(float(bound[0]), 1.2)

    def test_joint_source_axes_include_every_quadratic_interaction(self):
        nodes = numpy.linspace(-5.0, -1.0, 17)
        errors = numpy.stack(
            (numpy.full((2, 17), 0.01), numpy.full((2, 17), 0.02))
        )
        zero = numpy.zeros((2, 17))
        total = errors.sum(axis=0)
        bound = sum(
            transfer_spectrum_error(
                nodes,
                numpy.ones(17),
                zero,
                zero,
                error,
                error,
                total_primary_error=total,
                total_secondary_error=total,
            )
            for error in errors
        )
        exact = (
            4
            * numpy.pi
            * numpy.sum(
                spline_integral_weights(tuple(nodes)) * total**2, axis=-1
            )
        )
        numpy.testing.assert_allclose(bound, exact, rtol=1e-14)
        isolated = sum(
            transfer_spectrum_error(
                nodes, numpy.ones(17), zero, zero, error, error
            )
            for error in errors
        )
        self.assertTrue(numpy.all(bound > isolated))

    def test_disjoint_cross_spectrum_axes_still_have_a_joint_error(self):
        nodes = numpy.linspace(-5.0, -1.0, 17)
        zero = numpy.zeros((1, 17))
        first, second = numpy.full((1, 17), 0.01), numpy.full((1, 17), 0.02)
        bound = sum(
            transfer_spectrum_error(
                nodes,
                numpy.ones(17),
                zero,
                zero,
                primary,
                secondary,
                total_primary_error=first,
                total_secondary_error=second,
            )
            for primary, secondary in ((first, zero), (zero, second))
        )
        exact = (
            4
            * numpy.pi
            * numpy.sum(
                spline_integral_weights(tuple(nodes)) * first * second, axis=-1
            )
        )
        numpy.testing.assert_allclose(bound, exact, rtol=1e-14)
        self.assertGreater(float(bound[0]), 0.0)

    def test_native_source_probe_retains_coordinate_and_frozen_binding(
        self,
    ):
        from dataclasses import FrozenInstanceError

        eta = numpy.arange(3.0)
        probe = NativeSourceProbe(
            eta, {0.1: {"signal": eta}}, {}, 3.0, 2.0, eta
        )
        self.assertIsInstance(probe, NativeSourceProbe)
        self.assertEqual(probe.eta0, 3.0)
        with self.assertRaises(FrozenInstanceError):
            probe.eta0 = 9.0

    def test_source_envelope_and_projection_retain_cancelling_kernel_errors(
        self,
    ):
        from copernican.lib.likelihoods.cmb.runtime import background

        eta = numpy.linspace(0, 10, 129)
        fine = {"temperature": numpy.sin(eta)}
        lower = {"temperature": fine["temperature"] + 0.001}
        profile = {}
        update_source_error_envelope(profile, fine, lower)
        self.assertAlmostEqual(
            float(numpy.max(profile["temperature"])),
            0.001 / numpy.max(abs(fine["temperature"])),
        )
        kernels = background._get_cached_declared_projection_kernel_batch(
            (2,),
            "error-budget-independent-sine",
            x_values=eta,
            required_sectors=("scalar",),
        )
        bound = project_source_uncertainty(
            SimpleNamespace(
                source_terms={"monopole": "temperature"},
                projection="line_of_sight_temperature",
                kernel=None,
                sector="scalar",
            ),
            fine,
            profile,
            kernel_batch=kernels,
            k_value=1.0,
            eta_weights=numpy.full(eta.size, 10.0 / 128),
            chi_grid=eta,
            source_chi=10.0,
        )
        self.assertGreater(float(bound[0]), 0.0)
        # project_source_uncertainty retains kernel cancellation costs.
        self.assertAlmostEqual(
            float(numpy.max(profile["temperature"])),
            0.001 / numpy.max(abs(fine["temperature"])),
        )

    def test_adjoint_weights_match_independent_spline_basis(self):
        rng = numpy.random.default_rng(49)
        for count in (2, 3, 4, 17, 127):
            nodes = numpy.cumsum(rng.uniform(0.01, 3.0, count))
            primitive = CubicSpline(
                nodes, numpy.eye(count), axis=0
            ).antiderivative()
            independent = primitive(nodes[-1]) - primitive(nodes[0])
            numpy.testing.assert_allclose(
                spline_integral_weights(tuple(nodes)),
                independent,
                rtol=2e-13,
                atol=1e-13,
            )
        with self.assertRaises(ValueError):
            spline_integral_weights((1.0, 0.0))

    def test_error_propagation_bounds_signed_crosses_and_auto_spectra(self):
        rng = numpy.random.default_rng(9)
        nodes = numpy.log(numpy.geomspace(1e-5, 0.1, 25))
        for same in (True, False):
            first = rng.normal(size=(3, 25))
            second = first if same else rng.normal(size=(3, 25))
            error = rng.uniform(0, 0.02, first.shape)
            perturb_first = rng.uniform(-1, 1, first.shape) * error
            perturb_second = (
                perturb_first
                if same
                else (rng.uniform(-1, 1, second.shape) * error)
            )
            actual = transfer_spectrum_error(
                nodes, numpy.ones(25), first, second, error, error
            )
            weights = spline_integral_weights(tuple(nodes))
            difference = (
                4
                * numpy.pi
                * abs(
                    numpy.sum(
                        weights
                        * (
                            (first + perturb_first) * (second + perturb_second)
                            - first * second
                        ),
                        axis=1,
                    )
                )
            )
            self.assertTrue(numpy.all(actual >= difference))
            # transfer_spectrum_error includes quadratic signed-spline terms.

    def test_individually_small_axes_cannot_cancel_into_acceptance(self):
        spectrum = {"TE": numpy.array((-1.0, 0.0, 1.0))}
        report = combine_observable_errors(
            spectrum,
            {
                "evolution": {"TE": numpy.full(3, 0.006)},
                "projection": {"TE": numpy.full(3, 0.006)},
            },
            {"TE": 0.01},
        )
        self.assertFalse(report["converged"])
        self.assertAlmostEqual(
            report["products"]["TE"]["relative_error_bound"], 0.012
        )
        with self.assertRaises(ValueError):
            combine_observable_errors(spectrum, {"source": {}}, {"TE": 0.02})

    def test_tiny_power_and_cross_denominator_errors_are_not_erased(self):
        products = {
            "TT": numpy.ones(3) * 1e-15,
            "EE": numpy.ones(3) * 1e-20,
            "TE": numpy.array((-1.0, 0.0, 1.0)) * 1e-18,
        }
        errors = {name: numpy.zeros(3) for name in products}
        errors["EE"][:] = 1e-21
        report = combine_observable_errors(
            products, {"source": errors}, dict.fromkeys(products, 0.02)
        )
        self.assertFalse(report["converged"])
        self.assertAlmostEqual(
            report["products"]["EE"]["relative_error_bound"], 0.1
        )
        self.assertGreater(
            report["products"]["TE"]["normalized_cross_bound"], 0.01
        )
        for invalid in ({}, {"source": {"EE": numpy.full(3, -1.0)}}):
            with self.assertRaises(ValueError):
                combine_observable_errors(
                    {"EE": products["EE"]}, invalid, {"EE": 0.02}
                )

    def test_background_geometry_operator_bound_covers_kernel_shift(self):
        from copernican.lib.likelihoods.cmb.runtime import (
            background,
            spectrum_projection,
        )

        chi = numpy.linspace(0.0, 20.0, 129)
        source = numpy.cos(chi) + 0.2
        component = SimpleNamespace(
            source_terms={"monopole": "signal"},
            projection="line_of_sight_temperature",
            kernel=None,
            sector="scalar",
        )
        kernels = background._get_cached_declared_projection_kernel_batch(
            (2, 10),
            "geometry-bound-base",
            x_values=chi,
            required_sectors=("scalar",),
        )
        inputs = dict(
            kernel_batch=kernels,
            k_value=1.0,
            eta_weights=numpy.full(129, 20.0 / 128),
            chi_grid=chi,
            source_chi=20.0,
        )
        ratio = 1.03
        bound = project_source_uncertainty(
            component,
            {"signal": source},
            {"signal": numpy.zeros(129)},
            geometry_reference={
                "ells": (2, 10),
                "eta0_ratio": ratio,
                "source_chi": 20.0 * ratio,
            },
            **inputs,
        )
        shifted = background._get_cached_declared_projection_kernel_batch(
            (2, 10),
            "geometry-bound-shifted",
            x_values=chi * ratio,
            required_sectors=("scalar",),
        )
        base = spectrum_projection._declared_graph_projection(
            projection=component.projection,
            kernel=None,
            sector="scalar",
            source_histories={"monopole": source},
            **inputs,
        )
        independent = spectrum_projection._declared_graph_projection(
            projection=component.projection,
            kernel=None,
            sector="scalar",
            source_histories={"monopole": source},
            kernel_batch=shifted,
            k_value=1.0,
            eta_weights=inputs["eta_weights"] * ratio,
            chi_grid=chi * ratio,
            source_chi=20.0 * ratio,
        )
        self.assertTrue(numpy.all(bound >= abs(independent - base)))
        self.assertTrue(numpy.any(bound > 0))
