"""Focused tests for declared CMB numerical execution."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import execution


class ExecutionModuleTestCase(unittest.TestCase):
    """Exercise public numerical execution composition directly."""

    def test_custom_spectrum_data_accessors_return_named_payloads(self):
        """Transfer and spectrum accessors should expose stable arrays."""

        spectrum_data = execution.CustomCMBSpectrumData(
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
            return execution.CustomCMBSpectrumData(
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
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(execution.cache, "set_cmb_spectrum"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid k-grid evidence"):
                execution._compute_custom_cmb_spectrum_data(
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
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.array([2, 3]),
                k_grid=(
                    numpy.array([0.1, 0.15, 0.2])
                    if request.get("_numerical_overrides")
                    else numpy.array([0.1, 0.2])
                ),
                transfer_components={},
                spectra={
                    name: value * scale for name, value in values.items()
                },
                runtime_envelope={
                    "evolution_modes_evolved": (
                        1 if request.get("_numerical_overrides") else 0
                    )
                },
            )

        with (
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(execution.cache, "set_cmb_spectrum"),
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 3),
                requested_spectra=("TT", "TE", "EE"),
            )

        record = result.runtime_envelope["production_scalar_k_convergence"]
        self.assertTrue(record["converged"])
        self.assertEqual(record["base_count"], 2)
        self.assertEqual(record["refined_count"], 3)
        self.assertTrue(record["distinct"])
        self.assertTrue(record["nested"])
        self.assertEqual(record["new_node_count"], 1)
        self.assertGreater(record["new_node_work_units"], 0)
        self.assertTrue(record["cold_refinement"])
        self.assertFalse(record["warm_reuse"])
        self.assertNotEqual(
            record["base_grid_sha256"], record["refined_grid_sha256"]
        )
        self.assertEqual(len(record["refinement_identity"]), 64)
        self.assertEqual(record["declared_base_count"], 8)
        self.assertEqual(record["declared_refined_count"], 16)
        self.assertEqual(set(record["metrics"]), {"TT", "TE", "EE"})

    def test_cold_refinement_requires_measured_new_work(self):
        """A nominal finer count cannot certify a cold production run."""

        contract = {
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "required_spectra": ["TT", "TE", "EE"],
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args, kwargs
            refined = bool(request.get("_numerical_overrides"))
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.asarray((2, 3)),
                k_grid=numpy.asarray(
                    (0.1, 0.15, 0.2) if refined else (0.1, 0.2)
                ),
                transfer_components={},
                spectra={
                    "TT": numpy.asarray((1.0, 2.0)),
                    "TE": numpy.asarray((0.1, 0.2)),
                    "EE": numpy.asarray((0.5, 0.8)),
                },
            )

        with mock.patch.object(
            execution,
            "_compute_custom_cmb_spectrum_data_impl",
            side_effect=fake_impl,
        ):
            with self.assertRaisesRegex(ValueError, "no measured new work"):
                execution._compute_custom_cmb_spectrum_data(
                    contract,
                    (2, 3),
                    requested_spectra=("TT", "TE", "EE"),
                )

    def test_warm_refinement_requires_matching_accepted_finer_grid(self):
        """Exact warm reuse records its matching nested finer calculation."""

        contract = {
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "required_spectra": ["TT", "TE", "EE"],
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args
            refined = bool(request.get("_numerical_overrides"))
            if refined:
                kwargs["performance_timer"].mark_cache_state("exact_cache_hit")
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.asarray((2, 3)),
                k_grid=numpy.asarray(
                    (0.1, 0.15, 0.2) if refined else (0.1, 0.2)
                ),
                transfer_components={},
                spectra={
                    "TT": numpy.asarray((1.0, 2.0)),
                    "TE": numpy.asarray((0.1, 0.2)),
                    "EE": numpy.asarray((0.5, 0.8)),
                },
            )

        with (
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(execution.cache, "set_cmb_spectrum"),
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 3),
                requested_spectra=("TT", "TE", "EE"),
            )

        record = result.runtime_envelope["production_scalar_k_convergence"]
        self.assertTrue(record["warm_reuse"])
        self.assertTrue(record["matching_accepted_finer_calculation"])
        self.assertEqual(record["new_node_work_units"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
