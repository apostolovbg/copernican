"""Tests for the ordered declared CMB batch contract."""

import json
import unittest
from unittest import mock

import numpy
import pandas

from copernican.lib.likelihoods.cmb import cmb
from copernican.lib.likelihoods.cmb.errors import ParameterDomainError
from copernican.lib.likelihoods.cmb.results import CMBBatchResult


class CMBBatchContractTestCase(unittest.TestCase):
    """Exercise batch ordering, serialization, and failure isolation."""

    def test_single_item_batch_preserves_scalar_result_and_provenance(self):
        """One batch item returns the scalar spectrum at its input index."""

        def scalar(contract, ells, **_kwargs):
            return numpy.full(len(tuple(ells)), contract["value"])

        with mock.patch.object(
            cmb, "compute_cmb_spectrum_from_contract", side_effect=scalar
        ):
            results = cmb.compute_cmb_spectrum_batch(
                [{"value": 2.5}],
                [2, 3],
                requested_spectra=("TT",),
            )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, cmb.CMBBatchResult)
        self.assertEqual(result.index, 0)
        self.assertTrue(result.success)
        numpy.testing.assert_array_equal(result.spectrum, [2.5, 2.5])
        self.assertIsNone(result.failure)
        self.assertEqual(result.requested_ells, (2, 3))
        self.assertEqual(result.requested_spectra, ("TT",))
        self.assertEqual(result.solver_id, "ccmbs_numpy")
        json.dumps(result.to_dict(), sort_keys=True)

    def test_batch_preserves_input_order_and_isolates_typed_failures(self):
        """A failing item does not alter neighboring successful results."""

        def scalar(contract, ells, **_kwargs):
            if contract["kind"] == "invalid":
                raise ParameterDomainError("invalid parameter")
            return numpy.full(len(tuple(ells)), contract["value"])

        contracts = [
            {"kind": "valid", "value": 1.0},
            {"kind": "invalid"},
            {"kind": "valid", "value": 3.0},
        ]
        with mock.patch.object(
            cmb, "compute_cmb_spectrum_from_contract", side_effect=scalar
        ):
            results = cmb.compute_cmb_spectrum_batch(contracts, [2])

        self.assertEqual([result.index for result in results], [0, 1, 2])
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertIsInstance(results[1].failure, ParameterDomainError)
        self.assertEqual(results[1].failure.context["batch_index"], 1)
        self.assertEqual(results[1].requested_ells, (2,))
        self.assertEqual(results[1].requested_spectra, ("TT",))
        self.assertTrue(results[2].success)
        numpy.testing.assert_array_equal(results[0].spectrum, [1.0])
        numpy.testing.assert_array_equal(results[2].spectrum, [3.0])

    def test_batch_result_serializes_request_and_raw_metadata(self):
        """Batch results retain exact request and optional raw products."""

        result = CMBBatchResult(
            index=2,
            spectrum={"TT": numpy.array([1.0, 2.0])},
            requested_ells=(8, 3),
            requested_spectra=("TT",),
            diagnostics={"work_units": 4},
            phase_timings={"projection": 0.25},
            solver_id="ccmbs_numpy",
            solver_label="CCMBS NumPy",
            raw_spectra={"TT": numpy.array([0.1, 0.2])},
        )

        payload = result.to_dict()
        self.assertEqual(payload["requested_ells"], (8, 3))
        self.assertEqual(payload["requested_spectra"], ("TT",))
        self.assertEqual(payload["solver_id"], "ccmbs_numpy")
        self.assertEqual(payload["raw_spectra"]["TT"], [0.1, 0.2])
        json.dumps(payload, sort_keys=True)

    def test_cmb_likelihood_batch_matches_scalar_and_uses_one_cmb_call(
        self,
    ):
        """Batch likelihoods preserve scalar values and domain rejection."""

        plugin = mock.Mock(PARAMETER_BOUNDS=((0.0, 2.0),))
        cmb_data = pandas.DataFrame({"ell": [2, 3], "Dl_obs": [1.0, 2.0]})
        cmb_data.attrs["covariance_matrix_inv"] = numpy.eye(2)
        like = cmb.CMBLike(cmb_data, plugin)

        contracts = []

        def resolve(_plugin, params):
            return {"value": float(params[0])}

        def batch(contract_batch, ells, **_kwargs):
            contracts.extend(contract_batch)
            return tuple(
                CMBBatchResult(
                    index=index,
                    spectrum=numpy.asarray((1.0, 2.0)),
                )
                for index, _contract in enumerate(contract_batch)
            )

        with (
            mock.patch.object(cmb, "_resolve_plugin_cmb_contract", resolve),
            mock.patch.object(
                cmb,
                "prepare_cmb_execution_contract",
                side_effect=lambda contract: contract,
            ),
            mock.patch.object(
                cmb,
                "compute_cmb_spectrum_batch",
                side_effect=batch,
            ) as batch_call,
        ):
            values = like.loglike_batch(((1.0,), (3.0,)))

        self.assertEqual(values, (0.0, float("-inf")))
        batch_call.assert_called_once()
        self.assertEqual(contracts, [{"value": 1.0}])


if __name__ == "__main__":
    unittest.main()
