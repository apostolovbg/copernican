"""Tests for typed declared CMB failures and request accounting."""

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy
import pandas

from copernican.lib.likelihoods.cmb import cmb
from copernican.lib.likelihoods.cmb.errors import (
    CMBError,
    ConstraintViolationError,
    ContractError,
    ConvergenceError,
    EngineCapabilityError,
    ImplementationError,
    InitialPointError,
    ModelDeclarationError,
    ModelDiscoveryError,
    NonFiniteEvolutionError,
    ParameterDomainError,
    UnsupportedCapabilityError,
    classify_exception,
    failure_context,
)
from copernican.lib.likelihoods.cmb.runtime import cache, evolution, projection


def _cmb_dataframe() -> pandas.DataFrame:
    """Return one valid one-point CMB likelihood table."""

    frame = pandas.DataFrame({"ell": [20], "Dl_obs": [1.0]})
    frame.attrs["covariance_matrix_inv"] = numpy.eye(1)
    return frame


class CMBErrorTaxonomyTestCase(unittest.TestCase):
    """Exercise typed failures at the production likelihood boundary."""

    def test_error_base_exposes_context_and_diagnostic_contract(self) -> None:
        """The base error should preserve additive structured diagnostics."""

        error = CMBError("failure", context={"model": "A"})
        self.assertIs(error.add_context(gauge="newtonian"), error)
        self.assertEqual(error.diagnostic()["category"], "cmb_failure")

    def test_required_error_classes_preserve_taxonomy_relationships(
        self,
    ) -> None:
        """Every public failure class should remain a declared CMB error."""

        self.assertTrue(issubclass(ContractError, CMBError))
        self.assertTrue(issubclass(ConvergenceError, CMBError))
        self.assertTrue(issubclass(EngineCapabilityError, CMBError))
        self.assertTrue(issubclass(ImplementationError, CMBError))
        self.assertTrue(issubclass(InitialPointError, CMBError))
        self.assertTrue(issubclass(ModelDeclarationError, CMBError))
        self.assertTrue(issubclass(ModelDiscoveryError, CMBError))
        self.assertTrue(issubclass(UnsupportedCapabilityError, CMBError))

    def test_model_discovery_error_preserves_file_context(self) -> None:
        """Discovery failures retain a typed model-file diagnostic."""

        error = ModelDiscoveryError(
            "invalid future declaration",
            context={"model_filename": "model_future.yml"},
        )
        self.assertEqual(error.category, "model_discovery")
        self.assertEqual(
            error.diagnostic()["context"]["model_filename"],
            "model_future.yml",
        )

    def test_failure_context_and_classifier_are_structured(self) -> None:
        """Boundary helpers should retain request identity and error type."""

        context_builder = failure_context
        self.assertTrue(callable(context_builder))
        context = context_builder(
            {"model_name": "Model A"},
            workload="joint_mcmc",
            spectra=("TT",),
        )
        self.assertEqual(context["model_name"], "Model A")
        self.assertIsInstance(
            classify_exception(ValueError("invalid contract")),
            ContractError,
        )

    def test_internal_failures_classify_into_distinct_categories(self) -> None:
        """Every required failure category must remain distinguishable."""

        cases = (
            (
                ValueError("model does not provide requested TT"),
                UnsupportedCapabilityError,
            ),
            (
                ValueError("requested unsupported projection"),
                EngineCapabilityError,
            ),
            (
                ValueError("contract did not converge"),
                ConvergenceError,
            ),
            (
                ValueError("evolution produced non-finite values"),
                NonFiniteEvolutionError,
            ),
            (
                ValueError("Einstein constraint exceeded tolerance"),
                ConstraintViolationError,
            ),
            (ValueError("invalid contract field"), ContractError),
            (
                RuntimeError("unexpected runtime fault"),
                ImplementationError,
            ),
        )
        for raw_error, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                typed = classify_exception(
                    raw_error,
                    context={"workload": "joint_mcmc"},
                )
                self.assertIsInstance(typed, expected_type)
                self.assertEqual(
                    typed.context["workload"],
                    "joint_mcmc",
                )

    def test_universal_admission_categories_are_theory_neutral(self) -> None:
        """Declaration, engine, and request failures remain distinct."""

        self.assertEqual(
            ModelDeclarationError.category,
            "declaration_invalidity",
        )
        self.assertEqual(
            EngineCapabilityError.category,
            "engine_capability_gap",
        )
        self.assertEqual(
            UnsupportedCapabilityError.category,
            "request_not_declared",
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
                "prepare_cmb_execution_contract",
                return_value=contract,
            ),
            mock.patch.object(
                cmb,
                "_compute_declared_perturbation_spectrum",
                side_effect=ParameterDomainError("outside domain"),
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
                "prepare_cmb_execution_contract",
                return_value=contract,
            ),
            mock.patch.object(
                cmb,
                "_compute_declared_perturbation_spectrum",
                side_effect=ConstraintViolationError("broken invariant"),
            ),
            self.assertRaises(ConstraintViolationError),
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

        cache.clear_cmb_caches()

        def _fail(*args, performance_timer, **kwargs):
            del args, kwargs
            performance_timer.set_work_units({"evolution_work_units": 12})
            with performance_timer.phase("evolution"):
                raise ValueError("evolution produced non-finite values")

        with (
            mock.patch.object(
                projection,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=_fail,
            ),
            self.assertRaises(NonFiniteEvolutionError),
        ):
            projection._compute_custom_cmb_spectrum_data(
                {"model_name": "FailureModel"},
                (20,),
                requested_spectra=("TT",),
                workload="joint_mcmc",
            )

        record = cache.latest_cmb_performance_record()
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
        with self.assertRaises(ConstraintViolationError) as initial:
            evolution._validate_generated_scalar_initial_constraints(
                perturbation_data=perturbation_data,
                context=initial_context,
                k_value=0.4,
            )
        self.assertEqual(
            initial.exception.context["tolerance_provenance"],
            "generated_initial_fixed_normalized",
        )
        self.assertEqual(
            initial.exception.context["normalization_source"],
            "residual_magnitude_fallback",
        )
        self.assertEqual(
            initial.exception.context["normalization_terms"],
            {"declared_residual": 0.02},
        )

        with self.assertRaises(ConstraintViolationError) as evolved:
            projection._validate_scalar_constraint_histories(
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

    def test_generated_scalar_initial_surface_solves_all_einstein_terms(
        self,
    ) -> None:
        """A high-k initial solve should satisfy the coupled metric system."""

        perturbation_data = SimpleNamespace(
            gauge="conformal_newtonian",
            manifest_summary={"generated_scalar_hierarchy": True},
        )
        context = {
            "acoustic_k_sq": 0.09,
            "Hconf": 0.4,
            "einstein_gravity_strength": 0.02,
            "total_density_source": -0.7,
            "total_momentum_source": 0.13,
            "total_shear_source": 0.04,
        }
        solution = evolution._solve_generated_scalar_initial_einstein_surface(
            perturbation_data=perturbation_data,
            context=context,
            k_value=0.3,
        )
        resolved_context = {
            **context,
            **solution,
            "einstein_energy_residual": (
                context["acoustic_k_sq"] * solution["Phi"]
                + 3.0
                * context["Hconf"]
                * solution["metric_momentum_constraint"]
                + 1.5
                * context["einstein_gravity_strength"]
                * context["total_density_source"]
            ),
            "einstein_momentum_residual": (
                context["acoustic_k_sq"]
                * solution["metric_momentum_constraint"]
                - 1.5
                * context["einstein_gravity_strength"]
                * context["total_momentum_source"]
            ),
            "einstein_shear_residual": (
                context["acoustic_k_sq"] * (solution["Phi"] - solution["Psi"])
                - 3.0
                * context["einstein_gravity_strength"]
                * context["total_shear_source"]
            ),
        }

        diagnostics = evolution._validate_generated_scalar_initial_constraints(
            perturbation_data=perturbation_data,
            context=resolved_context,
            k_value=0.3,
        )

        self.assertEqual(
            set(diagnostics),
            {
                "einstein_energy_residual",
                "einstein_momentum_residual",
                "einstein_shear_residual",
            },
        )
        for metrics in diagnostics.values():
            self.assertLess(float(metrics["normalized_residual"]), 1.0e-12)
            self.assertEqual(
                metrics["normalization_source"],
                "sum_abs_declared_einstein_terms",
            )

    def test_shear_metrics_preserve_the_declared_closure_precision(
        self,
    ) -> None:
        """The shear diagnostic must avoid subtracting close potentials."""

        acoustic_k_sq = 0.04
        gravity = 0.02
        total_shear_source = 1.0e-12
        correction = 3.0 * gravity * total_shear_source / acoustic_k_sq
        phi_value = 1.0
        psi_value = phi_value - correction
        context = {
            "acoustic_k_sq": acoustic_k_sq,
            "einstein_gravity_strength": gravity,
            "total_shear_source": total_shear_source,
            "metric_constraint_scale": acoustic_k_sq,
            "metric_shear_correction": correction,
            "Phi": phi_value,
            "Psi": psi_value,
            "einstein_shear_residual": (
                acoustic_k_sq * (phi_value - psi_value)
                - 3.0 * gravity * total_shear_source
            ),
        }

        metrics = evolution._scalar_einstein_constraint_metrics(
            context,
            "einstein_shear_residual",
        )

        self.assertLess(
            float(numpy.max(metrics["normalized_values"])),
            1.0e-12,
        )
        self.assertAlmostEqual(
            float(metrics["term_values"]["metric_shear"]),
            3.0 * gravity * total_shear_source,
        )

    def test_generated_metrics_reject_missing_declared_term(self) -> None:
        """Generated constraints must not use residual-only normalization."""

        context = {
            "einstein_energy_residual": numpy.asarray((1.0,)),
        }
        with self.assertRaisesRegex(
            ConstraintViolationError,
            "omitted its declared term",
        ):
            evolution._scalar_einstein_constraint_metrics(
                context,
                "einstein_energy_residual",
                strict=True,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
