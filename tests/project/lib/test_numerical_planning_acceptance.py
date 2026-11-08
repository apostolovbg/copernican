"""Cheap evidence-reader checks; production requests are explicitly invoked."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from tests.project.lib import numerical_planning_acceptance as acceptance


class NumericalPlanningEvidenceTestCase(unittest.TestCase):
    """Incomplete or inconsistent measured evidence cannot earn closure."""

    @staticmethod
    def _operations():
        """Construct complete evidence for independent rejection tests."""

        products = {name: [1e-15, 2e-15, 3e-15] for name in ("TT", "TE", "EE")}
        rows = [
            {
                "name": name,
                "completed": True,
                "products": copy.deepcopy(products),
                "runtime_envelope": {
                    "observable_error_budget": {
                        "converged": True,
                        "products": {
                            "TT": {
                                "axis_relative_bounds": dict.fromkeys(
                                    (
                                        "background",
                                        "hierarchy_depth",
                                        "evolution",
                                        "source",
                                        "projection",
                                        "k",
                                    ),
                                    0.0,
                                )
                            }
                        },
                    },
                    "production_scalar_k_convergence": {"converged": True},
                },
                "performance": {"cache_state": "exact_cache_hit"},
            }
            for name in (
                "cold",
                "exact_repeat",
                "warm",
                "changed_request",
                "subset",
                "changed_parameter",
            )
        ]
        rows[4]["products"] = {name: [3e-15, 1e-15] for name in products}
        rows[3]["products"] = {
            name: [1e-15, 1.5e-15, 2e-15, 3e-15] for name in products
        }
        rows[5]["products"] = {
            name: [1.1e-15, 2.2e-15, 3.3e-15] for name in products
        }
        for row in rows:
            row["ells"] = [2, 10, 20]
            row["raw_spectra"] = copy.deepcopy(row["products"])
            budget = row["runtime_envelope"]["observable_error_budget"][
                "products"
            ]
            template = budget["TT"]
            row["runtime_envelope"]["observable_error_budget"]["products"] = {
                name: {
                    **copy.deepcopy(template),
                    "absolute_error_bound": 1e-20,
                }
                for name in products
            }
        rows[3]["ells"] = [2, 3, 10, 20]
        rows[4]["ells"] = [20, 2]
        rows.append(
            {
                "name": "batch",
                "completed": True,
                "indices": [0, 1, 2],
                "ordered_success": [True, False, True],
                "results": [
                    {"raw_spectra": copy.deepcopy(products)},
                    {},
                    {"raw_spectra": copy.deepcopy(products)},
                ],
            }
        )
        return rows

    def test_missing_failure_budget_subset_and_batch_mismatches_are_rejected(
        self,
    ):
        rows = self._operations()
        self.assertTrue(acceptance.assess_numerical_workload(rows)["passed"])
        damaged = []
        damaged.append(rows[:-1])
        failure = copy.deepcopy(rows)
        failure[0]["completed"] = False
        damaged.append(failure)
        budget = copy.deepcopy(rows)
        budget[0]["runtime_envelope"]["observable_error_budget"][
            "converged"
        ] = False
        damaged.append(budget)
        subset = copy.deepcopy(rows)
        subset[4]["products"]["EE"][0] *= 2
        damaged.append(subset)
        batch = copy.deepcopy(rows)
        batch[-1]["ordered_success"] = [False, False, True]
        damaged.append(batch)
        hit = copy.deepcopy(rows)
        hit[1]["performance"]["cache_state"] = "cold"
        damaged.append(hit)
        for report in damaged:
            self.assertFalse(
                acceptance.assess_numerical_workload(report)["passed"]
            )

    def test_subset_agreement_preserves_raw_and_public_uncertainties(
        self,
    ):
        rows = self._operations()
        subset = rows[4]
        subset["products"]["EE"][0] += 1.5e-20
        subset["raw_spectra"]["EE"][0] += 1.5e-20
        self.assertTrue(acceptance.assess_numerical_workload(rows)["passed"])
        subset["products"]["EE"][0] += 1e-20
        self.assertFalse(acceptance.assess_numerical_workload(rows)["passed"])
        subset["products"]["EE"][0] -= 1e-20
        subset["raw_spectra"]["EE"][0] += 1e-20
        self.assertFalse(acceptance.assess_numerical_workload(rows)["passed"])

    def test_main_persists_products_and_rejects_a_changed_producer(self):
        def run(_filename, persist):
            rows = self._operations()
            persist(rows)
            return rows

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with (
                mock.patch.object(
                    acceptance, "run_numerical_workload", side_effect=run
                ),
                mock.patch.object(
                    acceptance.diagnostics,
                    "cmb_completion_source_identity",
                    return_value={"sha256": "current"},
                ),
            ):
                self.assertEqual(
                    acceptance.main(
                        ["--model", "model_lcdm.yml", "--output", str(output)]
                    ),
                    0,
                )
            report = json.loads(output.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(
                len(report["models"]["model_lcdm.yml"]["operations"]), 7
            )
            with (
                mock.patch.object(
                    acceptance, "run_numerical_workload", side_effect=run
                ),
                mock.patch.object(
                    acceptance.diagnostics,
                    "cmb_completion_source_identity",
                    side_effect=iter(range(20)),
                ),
            ):
                self.assertEqual(
                    acceptance.main(
                        ["--model", "model_lcdm.yml", "--output", str(output)]
                    ),
                    1,
                )
        self.assertEqual(
            acceptance._encode(numpy.array([1.0, 2.0])), [1.0, 2.0]
        )
        with self.assertRaises(TypeError):
            acceptance._encode(object())

    def test_extended_precision_and_actual_result_telemetry_serialize(self):
        from copernican.lib.likelihoods.cmb import cmb
        from copernican.lib.likelihoods.cmb.runtime import cache
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        value = numpy.longdouble("1.0000000000000000001")
        encoded = json.loads(
            json.dumps(numpy.array([value]), default=acceptance._encode)
        )
        self.assertEqual(numpy.longdouble(encoded[0]), value)
        cache.clear_cmb_caches()
        self.addCleanup(cache.clear_cmb_caches)
        contract = test_cmb._prepare_declared_contract(
            test_cmb._analytic_signal_contract()
        )
        cmb.compute_cmb_spectrum_from_contract(contract, (2,), spectra=("TT",))
        result = cmb._LAST_CMB_RESULT.get()
        payload = {
            "raw": result.raw_spectra,
            "products": result.spectra,
            "provenance": result.cache_provenance,
            "performance": cache.latest_cmb_performance_record(),
            "runtime": acceptance.ccmbs._LAST_DECLARED_RUNTIME_EVIDENCE.get(),
        }
        restored = json.loads(json.dumps(payload, default=acceptance._encode))
        self.assertIn("TT", restored["raw"])
        numpy.testing.assert_allclose(
            numpy.asarray(restored["raw"]["TT"], dtype=float),
            result.raw_spectra["TT"],
            rtol=1e-15,
        )

    def test_readback_preserves_producer_and_rejects_source_drift(self):
        identity = {"sha256": "current"}
        producer = {
            "kind": "ordinary_numerical_runtime_evidence",
            "source_identity": identity,
            "final_source_identity": identity,
            "passed": False,
            "runner_sha256": "original-reader",
            "models": {"model_lcdm.yml": {"operations": self._operations()}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "producer.json"
            output = Path(directory) / "readback.json"
            path.write_text(json.dumps(producer))
            original = path.read_bytes()
            arguments = ["--assess-only", str(path), "--output", str(output)]
            with mock.patch.object(
                acceptance.diagnostics,
                "cmb_completion_source_identity",
                return_value=identity,
            ):
                self.assertEqual(acceptance.main(arguments), 0)
            result = json.loads(output.read_text())
            self.assertFalse(result["producer_assessment"])
            self.assertTrue(result["passed"])
            self.assertEqual(path.read_bytes(), original)
            with mock.patch.object(
                acceptance.diagnostics,
                "cmb_completion_source_identity",
                return_value={"sha256": "changed"},
            ):
                self.assertEqual(acceptance.main(arguments), 1)
            self.assertEqual(path.read_bytes(), original)
