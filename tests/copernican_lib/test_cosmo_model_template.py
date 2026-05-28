"""Tests for the root model template."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from copernican_lib import engine_adapter, model_coder, model_spec_validator
from copernican_lib.likelihoods import cmb
from copernican_lib.perturbation_contract import PerturbationContractIR


class CosmoModelTemplateTestCase(unittest.TestCase):
    """Validate the root template's perturbation contract and execution."""

    def _build_template_plugin(self):
        """Build the template plugin through the normal model pipeline."""

        repo_root = Path(__file__).resolve().parents[2]
        template_path = repo_root / "cosmo_model_template.yml"
        with TemporaryDirectory() as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                template_path,
                cache_dir,
            )
            funcs, parsed = model_coder.generate_callables(cache_path)
            return engine_adapter.build_plugin(parsed, funcs)

    def test_template_schema_and_plugin_validation(self) -> None:
        """The root template should validate and expose typed IR."""

        plugin = self._build_template_plugin()
        self.assertTrue(engine_adapter.validate_plugin(plugin))
        self.assertIs(plugin.CMB_PERTURBATION_STANDARD, False)
        perturbation_ir = plugin.get_cmb_perturbation_ir(
            plugin.INITIAL_GUESSES
        )
        self.assertIsInstance(perturbation_ir, PerturbationContractIR)
        self.assertFalse(perturbation_ir.standard)
        self.assertEqual(perturbation_ir.gauge, "conformal_newtonian")
        self.assertIn("delta_x", perturbation_ir.variables)
        self.assertIn("theta_x", perturbation_ir.variables)
        self.assertIn("Phi_tau", perturbation_ir.derived)
        self.assertIn("delta_rho_eff", perturbation_ir.derived)
        self.assertIn("continuity_x", perturbation_ir.equations)
        self.assertIn("euler_x", perturbation_ir.equations)
        self.assertIn("poisson_x", perturbation_ir.sources)
        self.assertEqual(
            perturbation_ir.backend_mapping["camb"].solver,
            "template_native_solver",
        )
        self.assertFalse(perturbation_ir.backend_mapping["camb"].implemented)

    def test_template_execution_rejects_unsupported_native_solver(
        self,
    ) -> None:
        """The root template should fail on unsupported native execution."""

        plugin = self._build_template_plugin()
        with mock.patch.dict(
            os.environ,
            {"COPERNICAN_FAKE_CMB": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "native non-standard perturbations|native backend "
                "implementation is required",
            ):
                cmb.compute_cmb_spectrum_cached(
                    plugin,
                    plugin.INITIAL_GUESSES,
                    [2, 3],
                )


if __name__ == "__main__":
    unittest.main()
