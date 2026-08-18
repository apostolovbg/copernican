"""Tests for the root model template."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from copernican.lib import model_adapter, model_coder, model_spec_validator
from copernican.lib.likelihoods import cmb
from copernican.lib.perturbation_contract import PerturbationContractData


class ModelTemplateTestCase(unittest.TestCase):
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
            return model_adapter.build_plugin(parsed, funcs)

    def test_template_schema_and_plugin_validation(self) -> None:
        """The root template should validate and expose typed data."""

        plugin = self._build_template_plugin()
        self.assertTrue(model_adapter.validate_plugin(plugin))
        perturbation_data = plugin.get_cmb_perturbation_data(
            plugin.INITIAL_GUESSES
        )
        self.assertIsInstance(perturbation_data, PerturbationContractData)
        self.assertEqual(perturbation_data.gauge, "conformal_newtonian")
        self.assertIn("delta_x", perturbation_data.variables)
        self.assertIn("theta_x", perturbation_data.variables)
        self.assertIn("phi_aux", perturbation_data.variables)
        self.assertIn("psi_aux", perturbation_data.variables)
        self.assertIn("density_drive", perturbation_data.derived)
        self.assertIn("continuity_x", perturbation_data.equations)
        self.assertIn("euler_x", perturbation_data.equations)
        self.assertIn("poisson_phi_x", perturbation_data.constraints)
        self.assertIn("psi_equals_phi_x", perturbation_data.closures)
        self.assertIn("monopole_x", perturbation_data.sources)
        self.assertIn("temperature", perturbation_data.observables)
        self.assertIn("delta_x_seed", perturbation_data.initial_conditions)
        contract = plugin.get_cmb_contract(plugin.INITIAL_GUESSES)
        perturbations = plugin.get_cmb_perturbation_contract(
            plugin.INITIAL_GUESSES
        )
        self.assertNotIn("backend", contract)
        self.assertNotIn("standard", perturbations)
        self.assertNotIn("backend_mapping", perturbations)

    def test_template_executes_through_native_declared_graph(
        self,
    ) -> None:
        """The root template should execute through the native graph."""

        plugin = self._build_template_plugin()
        runtime_contract = plugin.get_cmb_native_runtime(
            plugin.INITIAL_GUESSES
        )
        contract = dict(runtime_contract)
        contract["numerical"] = {
            **dict(runtime_contract["numerical"]),
            "ell_max": 8,
            "k_sample_count": 4,
            "eta_sample_count": 64,
            "source_grid_multiplier": 1,
        }
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            [2, 3],
        )
        self.assertEqual(spectra.shape, (2,))
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))


if __name__ == "__main__":
    unittest.main()
