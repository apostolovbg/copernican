"""Tests for the root model template."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from copernican.lib import engine_adapter, model_coder, model_spec_validator
from copernican.lib.likelihoods import cmb
from copernican.lib.perturbation_contract import PerturbationContractData


class CosmoModelTemplateTestCase(unittest.TestCase):
    """Validate the root template's perturbation contract and execution."""

    def _build_template_plugin(self):
        """Build the template plugin through the normal model pipeline."""

        repo_root = Path(__file__).resolve().parents[3]
        template_path = repo_root / "docs" / "model_template.yml"
        with TemporaryDirectory() as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                template_path,
                cache_dir,
            )
            funcs, parsed = model_coder.generate_callables(cache_path)
            return engine_adapter.build_plugin(parsed, funcs)

    def test_template_schema_and_plugin_validation(self) -> None:
        """The root template should validate and expose typed data."""

        plugin = self._build_template_plugin()
        self.assertTrue(engine_adapter.validate_plugin(plugin))
        self.assertIs(plugin.CMB_PERTURBATION_STANDARD, False)
        perturbation_data = plugin.get_cmb_perturbation_data(
            plugin.INITIAL_GUESSES
        )
        self.assertIsInstance(perturbation_data, PerturbationContractData)
        self.assertFalse(perturbation_data.standard)
        self.assertEqual(perturbation_data.gauge, "conformal_newtonian")
        self.assertIn("delta_x", perturbation_data.variables)
        self.assertIn("theta_x", perturbation_data.variables)
        self.assertIn("Phi_tau", perturbation_data.derived)
        self.assertIn("delta_rho_eff", perturbation_data.derived)
        self.assertIn("continuity_x", perturbation_data.equations)
        self.assertIn("euler_x", perturbation_data.equations)
        self.assertIn("poisson_x", perturbation_data.sources)
        self.assertFalse(perturbation_data.backend_mapping["camb"].implemented)

    def test_template_execution_rejects_unsupported_generic_execution(
        self,
    ) -> None:
        """The root template should fail on unsupported generic execution."""

        plugin = self._build_template_plugin()
        with mock.patch.dict(
            os.environ,
            {"COPERNICAN_FAKE_CMB": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "native non-standard perturbations|generic declarative "
                "executor is required",
            ):
                cmb.compute_cmb_spectrum_cached(
                    plugin,
                    plugin.INITIAL_GUESSES,
                    [2, 3],
                )


if __name__ == "__main__":
    unittest.main()
