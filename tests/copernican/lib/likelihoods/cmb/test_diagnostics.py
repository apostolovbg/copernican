"""Tests for fixed-parameter CCMBS diagnostics and model discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy
import yaml

from copernican.lib import model_adapter, model_coder, model_spec_validator
from copernican.lib.likelihoods.cmb.contracts_audit import (
    assert_bundled_cmb_contracts,
    audit_bundled_cmb_contracts,
)
from copernican.lib.likelihoods.cmb.diagnostics import (
    BUNDLED_CMB_MODEL_FILENAMES,
    CMB_CERTIFICATION_TIER,
    CMB_CORPUS_BASELINE_REQUEST,
    CMB_USMF2_BASELINE_TIERS,
    CMBCorpusBaselineRow,
    CMBModelDiagnostic,
    _jsonable,
    _run_scalar_batch_cache_check,
    assess_acoustic_structure,
    assess_physical_spectrum_shape,
    assess_scalar_batch_cache_evidence,
    audit_source_history_residuals,
    build_bundled_cmb_matrix_report,
    build_cmb_certification_report,
    build_cmb_corpus_baseline_report,
    compare_cmb_spectra_to_reference,
    discover_bundled_cmb_plugins,
    resolve_source_residual_audit_controls,
    run_bundled_cmb_corpus_baseline,
    run_cmb_model_diagnostic,
    write_cmb_certification_report,
    write_cmb_corpus_baseline_report,
)
from copernican.lib.likelihoods.cmb.results import CMBBatchResult


class CCMBSDiagnosticTestCase(unittest.TestCase):
    """Verify raw fixed-point evidence is captured before plotting."""

    def test_jsonable_normalizes_extended_precision_arrays(self):
        """Canonical reports must encode NumPy extended scalars."""

        payload = _jsonable(
            {
                "array": numpy.asarray(
                    [numpy.longdouble("1.25"), numpy.longdouble("2.5")]
                ),
                "scalar": numpy.longdouble("3.75"),
            }
        )

        self.assertEqual(payload, {"array": [1.25, 2.5], "scalar": 3.75})

    @staticmethod
    def _low_resolution_plugin():
        """Build a small USMF2 graph suitable for a bounded unit test."""

        source_path = (
            Path(__file__).resolve().parents[5]
            / "copernican"
            / "models"
            / "model_usmf2.yml"
        )
        model_data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        for controls in (
            model_data["cmb"]["numerical"],
            model_data["cmb"]["perturbations"]["numerics"],
        ):
            controls.update(
                {
                    "ell_min": 2,
                    "ell_max": 20,
                    "k_min": 1.0e-4,
                    "k_max": 2.0e-2,
                    "k_sample_count": 8,
                    "eta_sample_count": 32,
                    "source_grid_multiplier": 1,
                    "a_min": 1.0e-2,
                    "initial_redshift": 99.0,
                }
            )
        accuracy_controls = model_data["cmb"]["perturbations"][
            "accuracy_controls"
        ]
        accuracy_controls["scalar_reference_ells"] = [2, 20]
        accuracy_controls["minimum_k_sample_count"] = 1
        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir) / source_path.name
            model_path.write_text(
                yaml.safe_dump(model_data, sort_keys=False),
                encoding="utf-8",
            )
            cache_path = model_spec_validator.validate_and_cache_model(
                model_path,
                Path(model_dir) / "cache",
            )
            functions, parsed = model_coder.generate_callables(cache_path)
        plugin = model_adapter.build_plugin(parsed, functions)
        plugin.MODEL_FILENAME = source_path.name
        return plugin

    @staticmethod
    def _corpus_fixture_report(
        filename: str,
        *,
        decision: str = "rejected",
    ) -> CMBModelDiagnostic:
        """Build complete raw evidence without invoking the costly runtime."""

        raw_spectra = {
            "TT": numpy.asarray([1.0, 2.0, 3.0]),
            "TE": numpy.asarray([-0.2, 0.0, 0.2]),
            "EE": numpy.asarray([0.1, 0.3, 0.5]),
        }
        failure = None
        availability = "measured"
        if decision == "rejected":
            availability = "rejected"
            failure = {
                "error_type": "FixtureRejection",
                "category": "rejected",
                "message": "The fixture preserves a pre-repair rejection.",
            }
        return CMBModelDiagnostic(
            model_filename=filename,
            model_name=filename.removeprefix("model_").removesuffix(".yml"),
            parameter_names=("H_0",),
            parameter_values=(70.0,),
            requested_ells=tuple(CMB_CORPUS_BASELINE_REQUEST["ells"]),
            requested_spectra=("TT", "TE", "EE"),
            spectra=raw_spectra,
            raw_spectra=raw_spectra,
            raw_transfer_components={
                "temperature_source": numpy.asarray([0.1, 0.2, 0.3]),
            },
            runtime_envelope={
                "configured_numerical_controls": {
                    "k_sample_count": 1024,
                    "eta_sample_count": 192,
                },
                "effective_numerical_controls": {
                    "k_sample_count": 1024,
                    "eta_sample_count": 192,
                },
                "k_grid_actual_count": 1024,
                "source_history_residual_sample_schema": 1,
                "source_history_residual_samples_by_k": {
                    "0.100000": {
                        "sample_count": 1,
                        "samples": ({"eta": 1.0, "Theta_0": 0.2},),
                    },
                },
            },
            refinement={
                "axis": "k_sample_count",
                "base_count": 1024,
                "refined_count": 2048,
                "converged": False,
            },
            shape={"finite": True},
            acoustic_structure={"available": True},
            source_residual_audit={
                "available": True,
                "converged": False,
                "residual_vectors": {"metric": [0.0, 0.0]},
            },
            contract_identity={"sha256": f"contract-{filename}"},
            cache_identity={
                "base": {"available": True, "sha256": f"base-{filename}"},
                "refined": {
                    "available": True,
                    "sha256": f"refined-{filename}",
                },
            },
            availability=availability,
            failure=failure,
        )

    @classmethod
    def _corpus_fixture_row(
        cls,
        filename: str,
        *,
        decision: str = "rejected",
    ) -> CMBCorpusBaselineRow:
        """Wrap one fixed raw diagnostic in a complete baseline row."""

        report = cls._corpus_fixture_report(filename, decision=decision)
        incomplete = decision == "unclassified"
        return CMBCorpusBaselineRow(
            model_filename=filename,
            model_name=report.model_name,
            decision=decision,
            diagnostic=report,
            contract_audit={"valid": True},
            source_graph_audit={"valid": True},
            request_identity={
                "baseline_request_sha256": "fixture-request",
                "parameter_source": "model_initial_guesses",
            },
            projection_metadata={
                "configured_numerical_controls": {
                    "k_sample_count": 1024,
                    "eta_sample_count": 192,
                },
                "effective_numerical_controls": {
                    "k_sample_count": 1024,
                    "eta_sample_count": 192,
                },
                "k_grid_actual_count": 1024,
            },
            source_history_metadata={
                "available": True,
                "raw_data_path": (
                    "diagnostic.runtime_envelope."
                    "source_history_residual_samples_by_k"
                ),
                "sample_schema": 1,
                "mode_count": 1,
                "sample_count": 1,
            },
            work_estimate={
                "unit": "grid_product_lower_bound",
                "not_a_wall_clock_estimate": True,
            },
            completion_state="incomplete" if incomplete else "completed",
            decision_context=(
                {"remaining_tiers": ({"id": "final"},)} if incomplete else {}
            ),
        )

    def test_corpus_baseline_request_is_explicit_and_versioned(self):
        """Every corpus record must derive from one direct fixed request."""

        request = CMB_CORPUS_BASELINE_REQUEST

        self.assertEqual(request["schema_version"], 1)
        self.assertEqual(request["id"], "ccmbs-corpus-baseline-v1")
        self.assertEqual(request["parameter_source"], "model_initial_guesses")
        self.assertEqual(request["ells"][0], 2)
        self.assertEqual(request["ells"][-1], 300)
        self.assertEqual(request["spectra"], ("TT", "TE", "EE"))
        self.assertEqual(
            request["numerical_overrides"],
            {"k_sample_count": 1024, "eta_sample_count": 192},
        )
        self.assertEqual(
            request["source_anchor_policy"],
            "quartiles-plus-visibility-peak-v1",
        )
        self.assertEqual(
            request["refinement"],
            {"axis": "k_sample_count", "factor": 2, "required": True},
        )

    def test_corpus_baseline_serializes_each_frozen_row_once(self):
        """The pre-repair record remains complete and deterministic."""

        rows = [
            self._corpus_fixture_row(
                filename,
                decision=(
                    "unclassified"
                    if filename == "model_usmf2.yml"
                    else "rejected"
                ),
            )
            for filename in BUNDLED_CMB_MODEL_FILENAMES
        ]

        first = build_cmb_corpus_baseline_report(reversed(rows))
        second = build_cmb_corpus_baseline_report(rows)

        self.assertTrue(first["complete"])
        self.assertTrue(first["evidence_complete"])
        self.assertFalse(first["decision_complete"])
        self.assertEqual(first["record_sha256"], second["record_sha256"])
        self.assertEqual(len(first["rows"]), 10)
        self.assertEqual(first["outcome_counts"]["rejected"], 9)
        self.assertEqual(first["unclassified_models"], ["model_usmf2.yml"])
        self.assertEqual(
            first["rows"][0]["diagnostic"]["raw_spectra"],
            {
                "EE": [0.1, 0.3, 0.5],
                "TE": [-0.2, 0.0, 0.2],
                "TT": [1.0, 2.0, 3.0],
            },
        )
        self.assertTrue(
            first["rows"][0]["source_history_metadata"]["available"]
        )

        with tempfile.TemporaryDirectory() as output_directory:
            destination = Path(output_directory) / "corpus-baseline.json"
            written = write_cmb_corpus_baseline_report(rows, destination)
            self.assertEqual(written["record_sha256"], first["record_sha256"])
            self.assertIn(
                "record_sha256",
                destination.read_text(encoding="utf-8"),
            )

    def test_corpus_baseline_keeps_usmf2_incomplete_until_final_tier(self):
        """Partial USMF2 work remains unclassified rather than unavailable."""

        class Plugin:
            """Provide the frozen filename and direct parameters to a mock."""

            valid_for_cmb = True
            PARAMETER_NAMES = ("H_0",)
            INITIAL_GUESSES = (70.0,)

            def __init__(self, filename: str) -> None:
                self.MODEL_FILENAME = filename
                self.MODEL_NAME = filename.removeprefix("model_").removesuffix(
                    ".yml"
                )

        class Audit:
            """Supply the immutable audit shape needed by a baseline row."""

            def __init__(self, filename: str) -> None:
                self.model_filename = filename

            def to_dict(self) -> dict[str, object]:
                return {"model_filename": self.model_filename, "valid": True}

        plugins = tuple(
            Plugin(filename) for filename in BUNDLED_CMB_MODEL_FILENAMES
        )
        audits = tuple(
            Audit(filename) for filename in BUNDLED_CMB_MODEL_FILENAMES
        )

        def fixture_run(plugin, **_kwargs):
            """Return raw fixed-point evidence without executing CCMBS."""

            return self._corpus_fixture_report(plugin.MODEL_FILENAME)

        with (
            mock.patch(
                "copernican.lib.likelihoods.cmb.diagnostics."
                "discover_bundled_cmb_plugins",
                return_value=plugins,
            ),
            mock.patch(
                "copernican.lib.likelihoods.cmb.diagnostics."
                "audit_bundled_cmb_contracts",
                return_value=audits,
            ),
            mock.patch(
                "copernican.lib.likelihoods.cmb.diagnostics."
                "audit_bundled_cmb_source_graphs",
                return_value=audits,
            ),
            mock.patch(
                "copernican.lib.likelihoods.cmb.diagnostics."
                "run_cmb_model_diagnostic",
                side_effect=fixture_run,
            ) as runner,
        ):
            partial = run_bundled_cmb_corpus_baseline(
                usmf2_progression=CMB_USMF2_BASELINE_TIERS[:1]
            )
            complete = run_bundled_cmb_corpus_baseline()

        partial_row = next(
            row
            for row in partial["rows"]
            if row["model_filename"] == "model_usmf2.yml"
        )
        complete_row = next(
            row
            for row in complete["rows"]
            if row["model_filename"] == "model_usmf2.yml"
        )
        self.assertTrue(partial["complete"])
        self.assertFalse(partial["decision_complete"])
        self.assertEqual(partial_row["decision"], "unclassified")
        self.assertEqual(partial_row["completion_state"], "incomplete")
        self.assertEqual(
            len(partial_row["decision_context"]["remaining_tiers"]),
            2,
        )
        self.assertNotIn("timeout", str(partial_row).lower())
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["decision_complete"])
        self.assertEqual(complete_row["decision"], "rejected")
        self.assertEqual(complete_row["completion_state"], "completed")
        self.assertEqual(len(complete_row["progression"]), 3)
        self.assertEqual(runner.call_count, 22)

    def test_discovery_covers_every_bundled_cmb_model(self) -> None:
        """The harness must enumerate the complete bundled CMB corpus."""

        plugins = discover_bundled_cmb_plugins()
        self.assertEqual(len(plugins), 10)
        self.assertEqual(
            {plugin.MODEL_FILENAME for plugin in plugins},
            {
                "model_lcdm.yml",
                "model_lcdm_mnu.yml",
                "model_qauc.yml",
                "model_qrsf.yml",
                "model_ref_planck2018.yml",
                "model_tog.yml",
                "model_torg.yml",
                "model_usmf2.yml",
                "model_w0wa.yml",
                "model_wcdm.yml",
            },
        )

    def test_fixed_point_report_contains_raw_spectra_and_refinement(self):
        """A report retains raw products, metadata, and grid evidence."""

        report = run_cmb_model_diagnostic(
            self._low_resolution_plugin(),
            ells=(2, 8, 20),
            spectra=("TT", "TE", "EE"),
        )

        self.assertFalse(report.success)
        self.assertEqual(report.failure["category"], "convergence_failure")
        self.assertEqual(set(report.spectra), {"TT", "TE", "EE"})
        self.assertEqual(set(report.raw_spectra), {"TT", "TE", "EE"})
        self.assertTrue(report.raw_transfer_components)
        self.assertIn("k_sample_count", report.runtime_envelope)
        self.assertIn(
            "source_history_residual_samples_by_k",
            report.runtime_envelope,
        )
        self.assertEqual(report.refinement["base_count"], 8)
        self.assertEqual(report.refinement["refined_count"], 16)
        self.assertEqual(report.refinement["declared_base_count"], 8)
        self.assertEqual(report.refinement["declared_refined_count"], 16)
        self.assertIn("metrics", report.refinement)
        serialized = report.to_dict()
        self.assertTrue(serialized["raw_transfer_components"])
        self.assertTrue(serialized["cache_identity"]["base"]["available"])
        self.assertTrue(serialized["cache_identity"]["refined"]["available"])
        self.assertEqual(serialized["success"], False)

    def test_bundled_contract_audit_is_complete_and_consistent(self) -> None:
        """Every CMB-enabled bundle passes the declaration-level inventory."""

        audits = audit_bundled_cmb_contracts()

        self.assertEqual(len(audits), 10)
        self.assertTrue(all(audit.valid for audit in audits))
        self.assertTrue(all(audit.contract_version == 2 for audit in audits))
        self.assertTrue(all("scalar" in audit.sectors for audit in audits))
        self.assertTrue(
            all(audit.numerical["k_sample_count"] >= 64 for audit in audits)
        )
        assert_bundled_cmb_contracts(audits)

    def test_shape_audit_rejects_quadrature_spikes(self) -> None:
        """Raw spectra with alternating aliases fail before plotting."""

        report = assess_physical_spectrum_shape(
            range(2, 22),
            {"TT": [100.0 if index % 2 else 1.0 for index in range(20)]},
        )
        self.assertFalse(report["smooth"])
        self.assertIn("unresolved", " ".join(report["issues"]))

    def test_acoustic_structure_records_peak_phase_and_damping_evidence(
        self,
    ) -> None:
        """The raw shape audit records features before any plot is made."""

        ells = numpy.arange(2, 42, dtype=int)
        phase = numpy.linspace(0.0, 7.0 * numpy.pi, ells.size)
        tt_spectrum = (
            120.0
            * numpy.exp(-0.012 * ells)
            * (1.0 + 0.7 * numpy.sin(phase) ** 2)
        )
        te_spectrum = 20.0 * numpy.exp(-0.015 * ells) * numpy.sin(phase)
        ee_spectrum = (
            18.0
            * numpy.exp(-0.014 * ells)
            * (1.0 + 0.6 * numpy.sin(phase + 0.4) ** 2)
        )
        evidence = assess_acoustic_structure(
            ells,
            {
                "TT": tt_spectrum,
                "TE": te_spectrum,
                "EE": ee_spectrum,
            },
        )

        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["peak_ordered"])
        self.assertGreaterEqual(evidence["tt"]["peak_count"], 3)
        self.assertGreater(evidence["te"]["sign_change_count"], 2)
        self.assertGreater(evidence["ee"]["peak_count"], 0)
        self.assertLess(float(evidence["damping_ratio"]), 1.0)

    def test_reference_comparison_reports_auto_and_cross_metrics(self) -> None:
        """Independent reference data receives explicit scientific metrics."""

        comparison = compare_cmb_spectra_to_reference(
            {
                "TT": [1.0, 2.0, 3.0],
                "TE": [-1.0, 0.0, 1.0],
            },
            {
                "TT": [1.0, 2.0, 3.0],
                "TE": [-1.0, 0.0, 1.0],
            },
        )

        self.assertTrue(comparison["converged"])
        self.assertTrue(callable(assess_physical_spectrum_shape))
        self.assertTrue(callable(assess_acoustic_structure))
        self.assertTrue(callable(compare_cmb_spectra_to_reference))
        self.assertTrue(callable(write_cmb_certification_report))
        self.assertEqual(comparison["metrics"]["TT"]["kind"], "auto")
        self.assertEqual(comparison["metrics"]["TE"]["kind"], "cross")

    def test_reference_comparison_rejects_spectrum_mismatch(self) -> None:
        """Reference acceptance fails when an auto spectrum is inaccurate."""

        comparison = compare_cmb_spectra_to_reference(
            {"TT": [2.0, 4.0, 6.0]},
            {"TT": [1.0, 2.0, 3.0]},
        )

        self.assertFalse(comparison["converged"])
        self.assertFalse(comparison["metrics"]["TT"]["converged"])

    def test_source_history_audit_recomputes_all_declared_closures(
        self,
    ) -> None:
        """Raw source samples independently satisfy every scalar closure."""

        acoustic_k = 0.2
        acoustic_k_sq = acoustic_k**2
        phi = 0.5
        psi = 0.1
        hconf = 0.3
        phi_tau = 0.2
        psi_tau = 0.1
        gravity = 2.0
        total_density = -(
            acoustic_k_sq * phi + 3.0 * hconf * (phi_tau + hconf * psi)
        ) / (1.5 * gravity)
        total_momentum = (
            acoustic_k_sq * (phi_tau + hconf * psi) / (1.5 * gravity)
        )
        metric_shear = 0.25
        total_shear = acoustic_k_sq * metric_shear / (3.0 * gravity)
        visibility = 0.2
        theta_gamma0 = 0.3
        theta_b = 0.4
        polarization_moment = 0.05
        tau = 0.7
        sample = {
            "eta": 1.0,
            "Phi": phi,
            "Psi": psi,
            "Phi_tau": phi_tau,
            "Psi_tau": psi_tau,
            "Phi_history_tau": 0.4,
            "Hconf": hconf,
            "acoustic_k": acoustic_k,
            "acoustic_k_sq": acoustic_k_sq,
            "einstein_gravity_strength": gravity,
            "metric_shear_correction": metric_shear,
            "total_density_source": total_density,
            "total_momentum_source": total_momentum,
            "total_shear_source": total_shear,
            "visibility": visibility,
            "tau": tau,
            "observable_theta_gamma0": theta_gamma0,
            "observable_theta_b": theta_b,
            "polarization_moment": polarization_moment,
            "temperature_monopole": visibility
            * (theta_gamma0 + psi + 0.25 * polarization_moment),
            "temperature_quadrupole": 0.0,
            "temperature_quadrupole_derivative": 0.0,
            "temperature_doppler": visibility * theta_b / acoustic_k,
            "temperature_isw": numpy.exp(-tau) * (0.4 + psi_tau),
            "polarization_source": 0.75 * visibility * polarization_moment,
        }
        audit = audit_source_history_residuals(
            {
                "source_history_residual_samples_by_k": {
                    "0.2": {
                        "k": acoustic_k,
                        "samples": (sample,),
                    }
                }
            }
        )

        self.assertTrue(audit["available"])
        self.assertTrue(audit["independent_recomputation"])
        self.assertTrue(audit["converged"], audit["issues"])
        self.assertTrue(callable(audit_source_history_residuals))
        self.assertTrue(callable(resolve_source_residual_audit_controls))
        self.assertTrue(
            all(metric["available"] for metric in audit["metrics"].values())
        )
        small_error = dict(sample)
        small_error["total_density_source"] += 5.0e-4
        absolute_audit = audit_source_history_residuals(
            {
                "source_residual_audit_controls": (
                    resolve_source_residual_audit_controls()
                ),
                "source_history_residual_samples_by_k": {
                    "0.2": {"k": acoustic_k, "samples": (small_error,)}
                },
            }
        )
        self.assertTrue(absolute_audit["converged"])
        self.assertEqual(
            absolute_audit["metrics"]["einstein_energy"]["convergence_basis"],
            "absolute",
        )

    def test_source_history_audit_records_explicit_absolute_fallback(self):
        """Small absolute residuals pass with their declared provenance."""

        controls = resolve_source_residual_audit_controls(
            {
                "source_residual_audit": {
                    "relative_tolerances": {"einstein_energy": 1.0e-8},
                    "absolute_tolerances": {
                        "einstein_energy": 1.0e-3,
                    },
                }
            }
        )
        self.assertEqual(
            controls["provenance"],
            "accuracy_controls.source_residual_audit",
        )
        self.assertEqual(controls["criterion"], "relative_or_absolute")

    def test_certification_rejects_missing_independent_reference(self):
        """A finite solver report is not final without reference evidence."""

        report = CMBModelDiagnostic(
            model_filename="model_lcdm.yml",
            model_name="LambdaCDM",
            parameter_names=(),
            parameter_values=(),
            requested_ells=(2, 20),
            requested_spectra=("TT", "TE", "EE"),
            spectra={
                "TT": numpy.array([1.0, 2.0]),
                "TE": numpy.array([0.1, -0.1]),
                "EE": numpy.array([0.2, 0.3]),
            },
            raw_spectra={
                "TT": numpy.array([1.0, 2.0]),
                "TE": numpy.array([0.1, -0.1]),
                "EE": numpy.array([0.2, 0.3]),
            },
            refinement={"converged": True},
            shape={
                "finite": True,
                "auto_spectra_nonnegative": True,
                "smooth": True,
            },
            source_residual_audit={
                "available": True,
                "converged": True,
            },
            reference_comparison={
                "available": False,
                "converged": False,
            },
        )

        certification = build_cmb_certification_report((report,))

        self.assertFalse(certification["success"])
        self.assertIn("model_lcdm.yml", certification["rejected_models"])
        self.assertIn(
            "independent reference comparison is unavailable",
            certification["rejected_models"]["model_lcdm.yml"],
        )

    def test_certification_serialization_retains_raw_arrays(self):
        """Final reports retain raw arrays and a deterministic digest."""

        report = CMBModelDiagnostic(
            model_filename="model_lcdm.yml",
            model_name="LambdaCDM",
            parameter_names=(),
            parameter_values=(),
            requested_ells=(2, 20),
            requested_spectra=("TT",),
            spectra={"TT": numpy.array([1.0, 2.0])},
            raw_spectra={"TT": numpy.array([1.0, 2.0])},
            refinement={"converged": True},
            shape={
                "finite": True,
                "auto_spectra_nonnegative": True,
                "smooth": True,
            },
            source_residual_audit={
                "available": True,
                "converged": True,
            },
            reference_comparison={
                "available": True,
                "converged": True,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certification.json"
            certification = write_cmb_certification_report(
                (report,),
                path,
                required_spectra=("TT",),
            )
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertTrue(certification["success"])
        self.assertEqual(
            loaded["record_sha256"], certification["record_sha256"]
        )
        self.assertEqual(
            loaded["reports"][0]["report"]["raw_spectra"]["TT"],
            [1.0, 2.0],
        )

    def test_matrix_corpus_is_frozen_and_rejects_duplicate_rows(self):
        """The matrix must contain one row for every frozen model filename."""

        self.assertEqual(len(BUNDLED_CMB_MODEL_FILENAMES), 10)
        self.assertEqual(
            tuple(sorted(BUNDLED_CMB_MODEL_FILENAMES)),
            BUNDLED_CMB_MODEL_FILENAMES,
        )
        row = CMBModelDiagnostic(
            model_filename=BUNDLED_CMB_MODEL_FILENAMES[0],
            model_name="LambdaCDM",
            parameter_names=(),
            parameter_values=(),
            requested_ells=(2,),
            requested_spectra=("TT",),
            spectra={"TT": numpy.array([1.0])},
            raw_spectra={"TT": numpy.array([1.0])},
            shape={"finite": True, "auto_spectra_nonnegative": True},
            refinement={"converged": True},
            source_residual_audit={"available": True, "converged": True},
            availability="measured",
            contract_identity={"sha256": "fixture"},
            scalar_batch_evidence={"available": True, "converged": True},
            cache_isolation_evidence={"available": True, "isolated": True},
        )
        audits = {
            row.model_filename: {"valid": True},
        }
        matrix = build_bundled_cmb_matrix_report(
            (row, row),
            required_model_filenames=(row.model_filename,),
            required_spectra=("TT",),
            contract_audits=audits,
            source_graph_audits=audits,
        )
        self.assertFalse(matrix["complete"])
        self.assertIn(row.model_filename, matrix["rejected_models"])
        self.assertIn(
            "appears more than once",
            " ".join(matrix["rejected_models"][row.model_filename]),
        )

    def test_matrix_report_records_tier_and_explicit_unavailability(self):
        """An unmeasured row is retained with a typed, non-passing reason."""

        filename = BUNDLED_CMB_MODEL_FILENAMES[0]
        row = CMBModelDiagnostic(
            model_filename=filename,
            model_name="LambdaCDM",
            parameter_names=(),
            parameter_values=(),
            requested_ells=tuple(CMB_CERTIFICATION_TIER["ells"]),
            requested_spectra=("TT", "TE", "EE"),
            availability="unavailable",
            contract_identity={"sha256": "fixture"},
            failure={
                "error_type": "diagnostic_execution_unavailable",
                "category": "unavailable",
                "message": "fixture did not run",
            },
        )
        audits = {filename: {"valid": True}}
        matrix = build_bundled_cmb_matrix_report(
            (row,),
            required_model_filenames=(filename,),
            contract_audits=audits,
            source_graph_audits=audits,
        )
        self.assertFalse(matrix["success"])
        self.assertTrue(matrix["decision_complete"])
        self.assertEqual(
            matrix["certification_tier"]["id"],
            CMB_CERTIFICATION_TIER["id"],
        )
        self.assertIn(filename, matrix["rejected_models"])
        self.assertIn(
            "not measured", " ".join(matrix["rejected_models"][filename])
        )

    def test_scalar_batch_cache_audit_requires_order_and_unique_identities(
        self,
    ):
        """Batch equality is accepted only with ordered isolated identities."""

        class Result:
            def __init__(self, index, identity):
                self.index = index
                self.spectrum = {"TT": numpy.array([1.0, 2.0])}
                self.failure = None
                self.cache_provenance = {"cache_identity": identity}

        evidence = assess_scalar_batch_cache_evidence(
            {"TT": numpy.array([1.0, 2.0])},
            (Result(0, "zero"), Result(1, "one")),
        )
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["converged"])
        self.assertTrue(evidence["ordering_preserved"])
        self.assertTrue(evidence["cache_isolated"])

    def test_scalar_batch_cache_audit_accepts_per_point_scalar_payloads(self):
        """Each batch point must equal its own scalar reference spectrum."""

        class Result:
            def __init__(self, index, values, identity):
                self.index = index
                self.spectrum = {"TT": numpy.asarray(values)}
                self.failure = None
                self.cache_provenance = {"cache_identity": identity}

        evidence = assess_scalar_batch_cache_evidence(
            (
                {"TT": numpy.array([1.0, 2.0])},
                {"TT": numpy.array([2.0, 3.0])},
            ),
            (
                Result(0, [1.0, 2.0], "zero"),
                Result(1, [2.0, 3.0], "one"),
            ),
        )

        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["scalar_count"], 2)
        self.assertEqual(evidence["identity_count"], 2)

    def test_scalar_batch_cache_check_records_ordered_runtime_evidence(self):
        """The opt-in matrix check records ordered cache evidence."""

        class Plugin:
            INITIAL_GUESSES = (1.0,)
            PARAMETER_BOUNDS = ((0.0, 2.0),)

            @staticmethod
            def get_cmb_declared_runtime(parameters):
                return {"parameters": tuple(parameters)}

        report = CMBModelDiagnostic(
            model_filename="model_fixture.yml",
            model_name="Fixture",
            parameter_names=("x",),
            parameter_values=(1.0,),
            requested_ells=(2, 3),
            requested_spectra=("TT",),
            spectra={"TT": numpy.array([1.0, 2.0])},
            availability="measured",
        )
        batch_results = (
            CMBBatchResult(
                index=0,
                spectrum={"TT": numpy.array([1.0, 2.0])},
                cache_provenance={"cache_identity": "zero"},
            ),
            CMBBatchResult(
                index=1,
                spectrum={"TT": numpy.array([2.0, 3.0])},
                cache_provenance={"cache_identity": "one"},
            ),
        )
        with (
            mock.patch(
                "copernican.lib.likelihoods.cmb.cmb."
                "compute_cmb_spectrum_cached",
                return_value={"TT": numpy.array([2.0, 3.0])},
            ) as scalar,
            mock.patch(
                "copernican.lib.likelihoods.cmb.cmb."
                "compute_cmb_spectrum_batch",
                return_value=batch_results,
            ) as batch,
        ):
            evidence, cache_evidence = _run_scalar_batch_cache_check(
                Plugin(),
                report,
                ells=(2, 3),
                spectra=("TT",),
            )

        scalar.assert_called_once()
        batch.assert_called_once()
        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["status"], "passed")
        self.assertTrue(cache_evidence["isolated"])


if __name__ == "__main__":
    unittest.main()
