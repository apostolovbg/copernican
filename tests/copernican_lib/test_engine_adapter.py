"""Smoke tests for copernican_lib.engine_adapter."""

from __future__ import annotations

import unittest

import copernican_lib.cmb_backend_registry as cmb_backend_registry
from copernican_lib import camb_contract, engine_adapter


class TestEngineAdapterExports(unittest.TestCase):
    """Verify the root adapter module exports the expected surface."""

    def test_public_exports_are_present(self) -> None:
        self.assertTrue(callable(engine_adapter.build_engine_plugin))
        self.assertTrue(callable(engine_adapter.build_plugin))
        self.assertTrue(callable(engine_adapter.validate_plugin))
        self.assertTrue(hasattr(engine_adapter, "EnginePlugin"))
        self.assertTrue(hasattr(engine_adapter, "CAMBContractEvaluator"))
        self.assertTrue(hasattr(engine_adapter, "CAMBParameterEvaluator"))
        self.assertTrue(hasattr(engine_adapter, "FrozenMapping"))
        self.assertTrue(hasattr(engine_adapter, "PluginValidationError"))
        self.assertTrue(callable(engine_adapter.sanitize_equation))
        self.assertTrue(
            hasattr(engine_adapter.CAMBContractEvaluator, "evaluate_param_map")
        )
        self.assertTrue(
            hasattr(engine_adapter.EnginePlugin, "get_camb_params")
        )
        self.assertTrue(
            hasattr(engine_adapter.EnginePlugin, "get_camb_contract")
        )
        self.assertTrue(
            hasattr(
                engine_adapter.EnginePlugin, "get_cmb_perturbation_contract"
            )
        )
        self.assertTrue(
            hasattr(engine_adapter.EnginePlugin, "get_cmb_perturbation_ir")
        )
        self.assertTrue(hasattr(engine_adapter, "CMB_BACKEND_CAPABILITIES"))
        self.assertIs(
            engine_adapter.CMB_BACKEND_CAPABILITIES,
            cmb_backend_registry.CMB_BACKEND_CAPABILITIES,
        )
        self.assertIn("EnginePlugin", engine_adapter.__all__)
        self.assertIn("validate_plugin", engine_adapter.__all__)
        self.assertIs(
            camb_contract.CAMBContractEvaluator,
            engine_adapter.CAMBContractEvaluator,
        )
        self.assertIs(
            camb_contract.CAMBParameterEvaluator,
            engine_adapter.CAMBParameterEvaluator,
        )
        self.assertIs(
            camb_contract._validate_camb_contract_definition,
            engine_adapter._validate_camb_contract_definition,
        )

    def test_public_helpers_behave_as_expected(self) -> None:
        frozen = engine_adapter.FrozenMapping({"alpha": 1, "beta": [2, 3]})
        self.assertEqual(frozen.to_dict(), {"alpha": 1, "beta": [2, 3]})
        evaluator = engine_adapter.CAMBContractEvaluator(
            ("x",),
            ("x",),
            {
                "backend": "camb",
                "param_map": {"H0": "x"},
                "grids": {},
                "values": {},
                "calls": [],
                "perturbations": {
                    "contract_version": 1,
                    "standard": True,
                    "gauge": "unspecified",
                    "variables": {},
                    "derived": {},
                    "equations": {},
                    "closures": {},
                    "sources": {},
                    "validity": {"regimes": ["standard_camb"]},
                    "backend_mapping": {
                        "camb": {
                            "uses_standard_perturbations": True,
                        }
                    },
                },
            },
        )
        self.assertEqual(evaluator.evaluate_param_map((4.0,))["H0"], 4.0)
        self.assertIsInstance(
            engine_adapter.PluginValidationError("boom"),
            RuntimeError,
        )
        self.assertIsInstance(engine_adapter.sanitize_equation("x"), str)


if __name__ == "__main__":
    unittest.main()
