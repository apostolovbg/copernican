"""Tests for fixed-parameter CCMBS diagnostics and model discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy
import yaml

from copernican.lib import model_adapter, model_coder, model_spec_validator
from copernican.lib.likelihoods.cmb.contracts_audit import (
    assert_bundled_cmb_contracts,
    audit_bundled_cmb_contracts,
)
from copernican.lib.likelihoods.cmb.diagnostics import (
    CMBModelDiagnostic,
    assess_acoustic_structure,
    assess_physical_spectrum_shape,
    audit_source_history_residuals,
    build_cmb_certification_report,
    compare_cmb_spectra_to_reference,
    discover_bundled_cmb_plugins,
    resolve_source_residual_audit_controls,
    run_cmb_model_diagnostic,
    write_cmb_certification_report,
)


class CCMBSDiagnosticTestCase(unittest.TestCase):
    """Verify raw fixed-point evidence is captured before plotting."""

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
        self.assertEqual(report.refinement["base_count"], 8)
        self.assertEqual(report.refinement["refined_count"], 16)
        self.assertEqual(report.refinement["declared_base_count"], 8)
        self.assertEqual(report.refinement["declared_refined_count"], 16)
        self.assertIn("metrics", report.refinement)
        serialized = report.to_dict()
        self.assertTrue(serialized["raw_transfer_components"])
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


if __name__ == "__main__":
    unittest.main()
