"""Tests for typed native CMB failures and request accounting."""

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy
import pandas

from copernican.lib.likelihoods.cmb import (
    cmb,
    native_cache,
    native_evolution,
    native_projection,
)
from copernican.lib.likelihoods.cmb.native_errors import (
    NativeCMBError,
    NativeConstraintViolationError,
    NativeContractError,
    NativeConvergenceError,
    NativeImplementationError,
    NativeInitialPointError,
    NativeNonFiniteEvolutionError,
    NativeParameterDomainError,
    NativePerformanceBudgetError,
    NativeUnsupportedCapabilityError,
    classify_native_exception,
    native_failure_context,
)


def _cmb_dataframe() -> pandas.DataFrame:
    """Return one valid one-point CMB likelihood table."""

    frame = pandas.DataFrame({"ell": [20], "Dl_obs": [1.0]})
    frame.attrs["covariance_matrix_inv"] = numpy.eye(1)
    return frame


class NativeErrorTaxonomyTestCase(unittest.TestCase):
    """Exercise typed failures at the production likelihood boundary."""

    def test_error_base_exposes_context_and_diagnostic_contract(self) -> None:
        """The base error should preserve additive structured diagnostics."""

        error = NativeCMBError("failure", context={"model": "A"})
        self.assertIs(error.add_context(gauge="newtonian"), error)
        self.assertEqual(error.diagnostic()["category"], "native_failure")

    def test_required_error_classes_preserve_taxonomy_relationships(
        self,
    ) -> None:
        """Every public failure class should remain a native CMB error."""

        self.assertTrue(issubclass(NativeContractError, NativeCMBError))
        self.assertTrue(issubclass(NativeConvergenceError, NativeCMBError))
        self.assertTrue(issubclass(NativeImplementationError, NativeCMBError))
        self.assertTrue(issubclass(NativeInitialPointError, NativeCMBError))
        self.assertTrue(
            issubclass(NativePerformanceBudgetError, NativeCMBError)
        )
        self.assertTrue(
            issubclass(NativeUnsupportedCapabilityError, NativeCMBError)
        )

    def test_failure_context_and_classifier_are_structured(self) -> None:
        """Boundary helpers should retain request identity and error type."""

        context_builder = native_failure_context
        self.assertTrue(callable(context_builder))
        context = context_builder(
            {"model_name": "Model A"},
            workload="joint_mcmc",
            spectra=("TT",),
        )
        self.assertEqual(context["model_name"], "Model A")
        self.assertIsInstance(
            classify_native_exception(ValueError("invalid contract")),
            NativeContractError,
        )

    def test_internal_failures_classify_into_distinct_categories(self) -> None:
        """Every required failure category must remain distinguishable."""

        cases = (
            (
                ValueError("requested unsupported projection"),
                NativeUnsupportedCapabilityError,
            ),
            (
                ValueError("contract did not converge"),
                NativeConvergenceError,
            ),
            (
                ValueError("evolution produced non-finite values"),
                NativeNonFiniteEvolutionError,
            ),
            (
                ValueError("Einstein constraint exceeded tolerance"),
                NativeConstraintViolationError,
            ),
            (
                ValueError("native performance budget exceeded"),
                NativePerformanceBudgetError,
            ),
            (ValueError("invalid contract field"), NativeContractError),
            (
                RuntimeError("unexpected runtime fault"),
                NativeImplementationError,
            ),
        )
        for raw_error, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                typed = classify_native_exception(
                    raw_error,
                    context={"workload": "joint_mcmc"},
                )
                self.assertIsInstance(typed, expected_type)
                self.assertEqual(
                    typed.context["workload"],
                    "joint_mcmc",
                )

    def test_cmb_like_rejects_only_parameter_domain_errors(self) -> None:
        """Expected proposals return -inf while invariant failures escape."""

        likelihood = cmb.CMBLike(_cmb_dataframe(), plugin=object())
        contract = {"model_name": "DomainModel", "param_map": {"x": 1.0}}
        with (
            mock.patch.object(
                cmb,
                "_resolve_plugin_cmb_contract",
                return_value=contract,
            ),
            mock.patch.object(
                cmb,
                "prepare_native_cmb_execution_contract",
                return_value=contract,
            ),
            mock.patch.object(
                cmb,
                "_compute_declared_perturbation_spectrum",
                side_effect=NativeParameterDomainError("outside domain"),
            ) as spectrum,
            self.assertLogs(level="DEBUG") as captured,
        ):
            self.assertTrue(numpy.isneginf(likelihood.loglike((1.0,))))

        self.assertFalse(
            any(record.levelno >= logging.ERROR for record in captured.records)
        )
        self.assertEqual(
            likelihood.state["metadata"]["proposal_rejections"], 1
        )
        self.assertEqual(spectrum.call_args.kwargs["workload"], "joint_mcmc")

        with (
            mock.patch.object(
                cmb,
                "_resolve_plugin_cmb_contract",
                return_value=contract,
            ),
            mock.patch.object(
                cmb,
                "prepare_native_cmb_execution_contract",
                return_value=contract,
            ),
            mock.patch.object(
                cmb,
                "_compute_declared_perturbation_spectrum",
                side_effect=NativeConstraintViolationError("broken invariant"),
            ),
            self.assertRaises(NativeConstraintViolationError),
        ):
            likelihood.loglike((1.0,))

        bounded = cmb.CMBLike(
            _cmb_dataframe(),
            plugin=SimpleNamespace(PARAMETER_BOUNDS=((0.0, 1.0),)),
        )
        self.assertTrue(numpy.isneginf(bounded.loglike((1.5,))))
        self.assertEqual(
            bounded.state["metadata"]["failure"]["category"],
            "parameter_domain",
        )

    def test_failed_request_retains_phase_and_workload_evidence(self) -> None:
        """A failed request must preserve all phase slots and stop location."""

        native_cache.clear_native_cmb_caches()

        def _fail(*args, performance_timer, **kwargs):
            del args, kwargs
            performance_timer.set_work_units({"evolution_work_units": 12})
            with performance_timer.phase("evolution"):
                raise ValueError("evolution produced non-finite values")

        with (
            mock.patch.object(
                native_projection,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=_fail,
            ),
            self.assertRaises(NativeNonFiniteEvolutionError),
        ):
            native_projection._compute_custom_cmb_spectrum_data(
                {"model_name": "FailureModel"},
                (20,),
                requested_spectra=("TT",),
                workload="joint_mcmc",
            )

        record = native_cache.latest_native_cmb_performance_record()
        self.assertEqual(record["workload"], "joint_mcmc")
        self.assertEqual(record["outcome"], "failure")
        self.assertEqual(record["stop_phase"], "evolution")
        self.assertEqual(record["work_units"]["evolution_work_units"], 12)
        for phase_name in (
            "background",
            "compilation",
            "evolution",
            "initial_data",
            "lensing",
            "likelihood_assembly",
            "projection",
        ):
            self.assertIn(f"{phase_name}_seconds", record["phase_seconds"])

    def test_scalar_failure_surfaces_report_distinct_provenance(self) -> None:
        """Initial and evolved scalar failures must retain separate context."""

        perturbation_data = SimpleNamespace(
            accuracy_controls={},
            collision_operators={},
            conservation_rules={},
            gauge="conformal_newtonian",
            manifest_summary={"generated_scalar_hierarchy": True},
        )
        initial_context = {
            "Phi": 0.0,
            "Psi": 0.0,
            "acoustic_k_sq": 1.0,
            "einstein_energy_residual": 0.02,
            "einstein_gravity_strength": 0.0,
            "metric_momentum_constraint": 0.0,
            "total_density_source": 0.0,
            "total_momentum_source": 0.0,
            "total_shear_source": 0.0,
        }
        with self.assertRaises(NativeConstraintViolationError) as initial:
            native_evolution._validate_generated_scalar_initial_constraints(
                perturbation_data=perturbation_data,
                context=initial_context,
                k_value=0.4,
            )
        self.assertEqual(
            initial.exception.context["tolerance_provenance"],
            "generated_initial_gauge_default",
        )

        with self.assertRaises(NativeConstraintViolationError) as evolved:
            native_projection._validate_scalar_constraint_histories(
                perturbation_data=perturbation_data,
                context={
                    "einstein_energy_residual": numpy.asarray((0.0, 0.004))
                },
                eta_grid=numpy.asarray((1.0, 2.0)),
                accuracy_controls={
                    "scalar_constraint_reference_eta_samples": 2,
                    "scalar_constraint_tolerances": {
                        "einstein_energy_residual": 0.003
                    },
                },
                k_value=0.1,
            )
        self.assertEqual(evolved.exception.context["eta"], 2.0)
        self.assertEqual(
            evolved.exception.context["tolerance_provenance"],
            "accuracy_controls.scalar_constraint_tolerances",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
