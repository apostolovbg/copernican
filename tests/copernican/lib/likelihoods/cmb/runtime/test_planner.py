"""Acceptance tests for engine-owned CMB numerical planning."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from copernican.lib.likelihoods.cmb.runtime.planner import (
    CMBNumericalPlan,
    build_cmb_planner_manifest,
    plan_cmb_numerics,
    planner_accuracy_controls,
)


class AutomaticCMBPlannerTestCase(unittest.TestCase):
    """Ensure plans are physical, request-aware, and model-name neutral."""

    @staticmethod
    def _lcdm_contract() -> dict:
        """Load one bundled physical graph without solver controls."""

        path = (
            Path(__file__).resolve().parents[6]
            / "copernican"
            / "models"
            / "model_lcdm.yml"
        )
        return yaml.safe_load(path.read_text(encoding="utf-8"))["cmb"]

    def test_bundled_declarations_have_no_solver_controls(self) -> None:
        """Bundled model files contain equations, not numerical recipes."""

        models = Path(__file__).resolve().parents[6] / "copernican"
        for path in sorted((models / "models").glob("model_*.yml")):
            with self.subTest(model=path.name):
                contract = yaml.safe_load(path.read_text(encoding="utf-8"))[
                    "cmb"
                ]
                self.assertNotIn("numerical", contract)
                perturbations = contract.get("perturbations", {})
                self.assertNotIn("numerics", perturbations)
                self.assertNotIn("accuracy_controls", perturbations)

    def test_plan_is_deterministic_and_request_aware(self) -> None:
        """Repeated plans match and a larger ell request is resolved finer."""

        contract = self._lcdm_contract()
        low = plan_cmb_numerics(contract, ells=range(2, 21))
        high = plan_cmb_numerics(contract, ells=range(2, 2001))
        repeat = plan_cmb_numerics(contract, ells=range(2, 21))
        self.assertEqual(low.signature, repeat.signature)
        self.assertGreaterEqual(
            high.numerical_controls["k_sample_count"],
            low.numerical_controls["k_sample_count"],
        )
        for value in low.numerical_controls.values():
            if value is not None:
                self.assertTrue(float(value) == float(value))

    def test_renaming_theory_does_not_change_plan(self) -> None:
        """Planner decisions must not depend on model names or filenames."""

        contract = self._lcdm_contract()
        renamed = dict(contract)
        renamed["model_name"] = "UnrelatedTheory"
        first = plan_cmb_numerics(contract, ells=(2, 30, 200))
        second = plan_cmb_numerics(renamed, ells=(2, 30, 200))
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_public_planner_symbols_are_exposed(self) -> None:
        """Planner API exposes the immutable plan and accuracy helper."""

        contract = self._lcdm_contract()
        self.assertTrue(callable(build_cmb_planner_manifest))
        plan = plan_cmb_numerics(contract, ells=(2, 30))
        self.assertIsInstance(plan, CMBNumericalPlan)
        self.assertEqual(
            planner_accuracy_controls(contract, ells=(2, 30)),
            {
                "runtime_envelope": "bounded",
                "source_history_reconstruction": True,
            },
        )

    def test_bundled_planner_manifest_covers_reference_models(self) -> None:
        """Raw planner evidence covers every required bundled declaration."""

        models_dir = (
            Path(__file__).resolve().parents[6] / "copernican" / "models"
        )
        required = {
            "model_lcdm.yml",
            "model_usmf2.yml",
            "model_qauc.yml",
            "model_qrsf.yml",
            "model_tog.yml",
            "model_torg.yml",
            "model_wcdm.yml",
            "model_w0wa.yml",
        }
        contracts = {
            path.name: yaml.safe_load(path.read_text(encoding="utf-8"))["cmb"]
            for path in sorted(models_dir.glob("model_*.yml"))
            if path.name in required
        }
        manifest = build_cmb_planner_manifest(
            contracts,
            ells=range(2, 201),
            spectra=("TT", "TE", "EE"),
        )
        self.assertEqual(set(manifest["models"]), required)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertTrue(
            all(row["signature"] for row in manifest["models"].values())
        )
        json.dumps(manifest, allow_nan=False)

    def test_plan_records_declared_schedule_evidence(self) -> None:
        """Plans expose hierarchy, collision, and initial-data decisions."""

        plan = plan_cmb_numerics(self._lcdm_contract(), ells=range(2, 21))
        evidence = plan.physical_scale_evidence
        hierarchy = evidence["hierarchy_resolution"]
        collisions = evidence["collision_schedule"]
        initial = evidence["initial_condition_resolution"]
        self.assertEqual(
            hierarchy["method"],
            "phase_visibility_and_declared_closure",
        )
        self.assertIn("photon_temperature", hierarchy["family_l_max"])
        self.assertEqual(
            collisions["method"],
            "declared_rate_phase_partition",
        )
        self.assertGreaterEqual(int(collisions["operator_count"]), 1)
        self.assertTrue(initial["hidden_prefix"])
        self.assertIn("conditions", initial)


if __name__ == "__main__":
    unittest.main()
