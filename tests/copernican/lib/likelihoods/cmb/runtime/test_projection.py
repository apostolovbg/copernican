"""Focused tests for the declared CMB projection module."""

import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.errors import ConvergenceError
from copernican.lib.likelihoods.cmb.runtime import projection


class ProjectionModuleTestCase(unittest.TestCase):
    """Exercise declared projection helpers directly."""

    def test_runtime_work_estimate_is_deterministic_and_accounted(self):
        """Large bounded requests are accounted for, not rejected."""

        contract = {
            "perturbations": {
                "accuracy_controls": {"runtime_envelope": "bounded"}
            }
        }
        arguments = {
            "ell_count": 2500,
            "k_count": 2048,
            "eta_count": 2048,
            "state_slot_count": 64,
            "transfer_component_count": 3,
            "momentum_point_count": 256,
            "evolution_multiplier": 3,
        }
        first = projection._enforce_runtime_envelope(contract, **arguments)
        second = projection._enforce_runtime_envelope(contract, **arguments)
        self.assertEqual(first, second)
        self.assertEqual(first["work_accounting_mode"], "accounted")
        self.assertEqual(first["work_limits"], {})
        self.assertFalse(first["work_limits_enforced"])
        self.assertGreater(first["total_work_units"], 100_000_000)

    def test_explicit_work_limit_is_metadata_not_a_machine_ceiling(self):
        """A valid request is not rejected by an operator work hint."""

        contract = {
            "perturbations": {
                "accuracy_controls": {
                    "runtime_envelope": {
                        "maximum_total_work_units": 1,
                    }
                }
            }
        }
        envelope = projection._enforce_runtime_envelope(
            contract,
            ell_count=100,
            k_count=100,
            eta_count=100,
            state_slot_count=8,
            transfer_component_count=2,
            momentum_point_count=0,
        )
        self.assertEqual(
            envelope["work_limits"], {"maximum_total_work_units": 1}
        )
        self.assertFalse(envelope["work_limits_enforced"])

    def test_evolution_chunk_size_is_deterministic(self):
        """Evolution chunking is derived from declared array dimensions."""

        first = projection._resolve_evolution_chunk_size(
            k_count=2048,
            eta_count=2048,
            state_slot_count=64,
        )
        second = projection._resolve_evolution_chunk_size(
            k_count=2048,
            eta_count=2048,
            state_slot_count=64,
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first * 2048 * 64, 16_000_000)

    def test_custom_spectrum_data_accessors_return_named_payloads(self):
        """Transfer and spectrum accessors should expose stable arrays."""

        spectrum_data = projection.CustomCMBSpectrumData(
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

    def test_production_scalar_rule_rejects_nonconverged_doubled_grid(self):
        """The production wrapper must reject a failed k-grid refinement."""

        base_values = {
            "TT": numpy.asarray([10.0, 20.0]),
            "TE": numpy.asarray([1.0, 2.0]),
            "EE": numpy.asarray([3.0, 4.0]),
        }
        contract = {
            "model_name": "test",
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "k_refinement_factor": 2,
                        "required_spectra": ["TT", "TE", "EE"],
                        "fail_on_nonconvergence": True,
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args, kwargs
            scale = 1.1 if request.get("_numerical_overrides") else 1.0
            return projection.CustomCMBSpectrumData(
                ell_grid=numpy.array([2, 3]),
                k_grid=numpy.array([0.1, 0.2]),
                transfer_components={},
                spectra={
                    name: values * scale
                    for name, values in base_values.items()
                },
            )

        with (
            mock.patch.object(
                projection,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(projection.cache, "set_cmb_spectrum"),
        ):
            with self.assertRaises(ConvergenceError):
                projection._compute_custom_cmb_spectrum_data(
                    contract,
                    (2, 3),
                    requested_spectra=("TT", "TE", "EE"),
                )

    def test_production_scalar_rule_records_converged_metrics(self):
        """A passing doubled-grid comparison remains in the envelope."""

        contract = {
            "model_name": "test",
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "k_refinement_factor": 2,
                        "required_spectra": ["TT", "TE", "EE"],
                        "relative_tolerances": {
                            "TT": 0.01,
                            "TE": 0.02,
                            "EE": 0.01,
                        },
                        "fail_on_nonconvergence": True,
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args, kwargs
            scale = 1.001 if request.get("_numerical_overrides") else 1.0
            values = {
                "TT": numpy.asarray([10.0, 20.0]),
                "TE": numpy.asarray([1.0, 2.0]),
                "EE": numpy.asarray([3.0, 4.0]),
            }
            return projection.CustomCMBSpectrumData(
                ell_grid=numpy.array([2, 3]),
                k_grid=numpy.array([0.1, 0.2]),
                transfer_components={},
                spectra={
                    name: value * scale for name, value in values.items()
                },
            )

        with (
            mock.patch.object(
                projection,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(projection.cache, "set_cmb_spectrum"),
        ):
            result = projection._compute_custom_cmb_spectrum_data(
                contract,
                (2, 3),
                requested_spectra=("TT", "TE", "EE"),
            )

        record = result.runtime_envelope["production_scalar_k_convergence"]
        self.assertTrue(record["converged"])
        self.assertEqual(record["base_count"], 2)
        self.assertEqual(record["refined_count"], 2)
        self.assertEqual(record["declared_base_count"], 8)
        self.assertEqual(record["declared_refined_count"], 16)
        self.assertEqual(set(record["metrics"]), {"TT", "TE", "EE"})

    def test_projection_source_does_not_import_camb(self):
        """The declared projection module should remain CAMB-free."""

        source_text = Path(projection.__file__).read_text(encoding="utf-8")
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
        actual = projection._integrate_power_spectrum(
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
            coarse = projection._slice_projection_kernel_batch(
                kernel_batch,
                numpy.asarray((0, 3), dtype=int),
            )

        self.assertFalse(
            any(item.category is DeprecationWarning for item in captured)
        )
        self.assertEqual(coarse.j_l.shape, (2, 2))
        self.assertEqual(coarse.vector_temperature_1.shape, (2, 0))
        self.assertEqual(coarse.tensor_e.shape, (2, 0))

    def test_batched_collision_overflow_is_handled_without_runtime_warnings(
        self,
    ):
        """Rejected stiff collision rows must not flood worker stderr."""

        blocks = numpy.asarray(
            [[[1.0e3, 0.0], [0.0, -1.0e3]]],
            dtype=float,
        )
        states = numpy.asarray([[1.0, 1.0]], dtype=float)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = projection._exact_batched_two_state_blocks(
                blocks,
                states,
            )

        self.assertIsNone(result)
        self.assertFalse(
            any(item.category is RuntimeWarning for item in captured)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
