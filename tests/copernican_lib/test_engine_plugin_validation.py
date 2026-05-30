# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for ``copernican_lib.engine_adapter`` helpers."""

import copy
import math
import multiprocessing as multiprocessing_module
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from copernican_lib import engine_adapter as engine_plugin_validation
from copernican_lib import model_coder, model_spec_validator
from copernican_lib.engine_adapter import PluginValidationError
from copernican_lib.perturbation_contract import PerturbationContractData

# fmt: off
MAKE_POSTERIOR = engine_plugin_validation.make_logposterior


def _dummy_func(*_args, **_kwargs):
    """Return a placeholder numerical value."""
    return 0.0


def _linear_like(params):
    """Simple log-likelihood used to test pickling."""

    return -sum(params)


def _evaluate_posterior(posterior):
    """Return the posterior evaluation from a worker process."""

    return posterior([0.1])


class EngineInterfaceTestCase(unittest.TestCase):
    """Validate engine adapter construction and associated helpers."""

    def setUp(self):
        """Build a minimal engine adapter for reuse across tests."""
        self.base_param_map = {
            "H0": "H_0",
            "ombh2": 0.022,
            "omch2": 0.12,
            "Neff": 3.044,
            "num_massive_neutrinos": 3,
            "sum_mnu": 0.06,
        }
        self.base_cmb_contract = {
            "backend": "camb",
            "param_map": self.base_param_map,
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
                "validity": {
                    "regimes": ["standard_camb"],
                    "notes": (
                        "Uses the backend standard perturbation machinery."
                    ),
                },
                "backend_mapping": {
                    "camb": {
                        "uses_standard_perturbations": True,
                    }
                },
                "notes": (
                    "This model declares that its CMB perturbations are "
                    "represented by the selected backend's standard "
                    "perturbation system."
                ),
            },
        }
        self.model_data = {
            "model_name": "Dummy",
            "description": "desc",
            "abstract": "abs",
            "parameters": [
                {
                    "python_var": "hubble_parameter",
                    "latex_name": "H_0",
                    "bounds": [60, 80],
                }
            ],
            "equations": {"sne": ["$$E=mc^2$$"], "bao": []},
            "valid_for_cmb": True,
            "cmb": copy.deepcopy(self.base_cmb_contract),
        }
        req = engine_plugin_validation.REQUIRED_FUNCTIONS
        funcs = {name: _dummy_func for name in req}
        self.funcs = funcs
        build_plugin = engine_plugin_validation.build_plugin
        self.plugin = build_plugin(self.model_data, funcs)

    def _make_standard_perturbations(
        self, *, background_adapter: bool = False
    ) -> dict[str, object]:
        """Return a standard CAMB perturbation contract for tests."""

        perturbations = copy.deepcopy(self.base_cmb_contract["perturbations"])
        if background_adapter:
            perturbations["validity"]["regimes"] = [
                "standard_camb_with_declared_background_adapter"
            ]
            perturbations["validity"]["notes"] = (
                "Uses standard backend perturbations with the model's "
                "declared background adapter contract."
            )
            perturbations["notes"] = (
                "Native non-standard perturbation equations are not "
                "declared in this model file."
            )
        return perturbations

    def _make_nonstandard_perturbations(
        self,
        *,
        implemented: bool = False,
    ) -> dict[str, object]:
        """Return a fully declared non-standard perturbation contract."""

        return {
            "contract_version": 1,
            "standard": False,
            "gauge": "conformal_newtonian",
            "variables": {
                "delta_x": {
                    "kind": "density_contrast",
                    "description": "Example density perturbation.",
                },
                "theta_x": {
                    "kind": "velocity_divergence",
                    "description": "Example velocity perturbation.",
                },
                "rho_x": {
                    "kind": "background_density",
                    "description": "Example density source.",
                },
                "sigma_x": {
                    "kind": "anisotropic_stress",
                    "description": "Example stress source.",
                },
            },
            "derived": {
                "Phi_tau": {
                    "kind": "derivative_symbol",
                    "variable": "Phi",
                    "wrt": "tau",
                    "order": 1,
                    "description": "First conformal-time derivative.",
                },
                "delta_rho_eff": {
                    "expression": "rho_x * delta_x",
                    "description": "Effective density perturbation.",
                },
            },
            "equations": {
                "continuity_x": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "delta_x",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "-theta_x + 3 * Phi_tau",
                },
                "euler_x": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "theta_x",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "-Hconf * theta_x + k**2 * Psi",
                },
            },
            "closures": {
                "no_anisotropic_stress": {
                    "expression": "sigma_x",
                    "equals": "0",
                }
            },
            "sources": {
                "poisson": {
                    "expression": "delta_rho_eff + delta_x + theta_x",
                }
            },
            "validity": {
                "regimes": ["linear", "scalar"],
                "notes": "Declared for first-order scalar perturbations.",
            },
            "backend_mapping": {
                "camb": {
                    "native_solver_required": True,
                    "implemented": implemented,
                }
            },
            "notes": "Native perturbation mathematics are declared here.",
        }

    def test_plugin_validation(self):
        """Plugin built from minimal data should validate."""
        self.assertTrue(engine_plugin_validation.validate_plugin(self.plugin))

    def test_missing_attribute_fails_validation(self):
        """A plugin lacking required attributes is rejected."""
        bad = SimpleNamespace()
        with self.assertLogs(level="ERROR") as captured_logs:
            with self.assertRaises(PluginValidationError):
                engine_plugin_validation.validate_plugin(bad)
        self.assertIn("Plugin validation issue", "".join(captured_logs.output))

    def test_get_camb_params_expression(self):
        """LaTeX expressions in ``cmb.param_map`` evaluate correctly."""
        camb = self.plugin.get_camb_params([70.0])
        self.assertEqual(camb["H0"], 70.0)
        self.assertAlmostEqual(camb["ombh2"], 0.022)

    def test_empty_calls_preserve_scalar_camb_params(self):
        """An explicit empty call list keeps scalar CAMB mapping intact."""

        camb = self.plugin.get_camb_params([70.0])
        self.assertEqual(camb["Neff"], 3.044)
        self.assertEqual(camb["sum_mnu"], 0.06)

    def test_get_camb_contract_preserves_strings_and_arrays(self):
        """Structured contracts keep arrays and string kwargs intact."""

        model_data = copy.deepcopy(self.model_data)
        model_data["parameters"].append(
            {
                "python_var": "a_T",
                "latex_name": "a_T",
                "bounds": [0.5, 2.0],
            }
        )
        model_data["cmb"] = {
            "backend": "camb",
            "param_map": copy.deepcopy(self.base_param_map),
            "grids": {
                "a_grid": {
                    "symbol": "a",
                    "lower": 0.001,
                    "upper": 1.0,
                    "points": 4,
                    "spacing": "linear",
                }
            },
            "values": {
                "x": {
                    "grid": "a_grid",
                    "expression": "(a/a_T)**3",
                }
            },
            "calls": [
                {
                    "method": "set_dark_energy_w_a",
                    "args": {
                        "a": "@grid.a_grid",
                        "w": "@value.x",
                    },
                    "kwargs": {"dark_energy_model": "ppf"},
                }
            ],
            "perturbations": self._make_standard_perturbations(
                background_adapter=True
            ),
        }
        plugin = engine_plugin_validation.build_plugin(model_data, self.funcs)
        contract = plugin.get_camb_contract(plugin.INITIAL_GUESSES)
        self.assertEqual(contract["backend"], "camb")
        self.assertEqual(contract["param_map"]["H0"], 70.0)
        self.assertTrue(numpy.all(numpy.isfinite(contract["grids"]["a_grid"])))
        self.assertTrue(
            numpy.all(numpy.diff(contract["grids"]["a_grid"]) > 0.0)
        )
        self.assertTrue(numpy.all(numpy.isfinite(contract["values"]["x"])))
        numpy.testing.assert_allclose(
            contract["grids"]["a_grid"],
            numpy.linspace(0.001, 1.0, 4),
        )
        self.assertIsInstance(contract["values"]["x"], numpy.ndarray)
        self.assertEqual(
            contract["calls"][0]["kwargs"]["dark_energy_model"],
            "ppf",
        )
        self.assertIsInstance(contract["calls"][0]["args"]["a"], numpy.ndarray)
        self.assertIsInstance(contract["calls"][0]["args"]["w"], numpy.ndarray)

    def test_get_cmb_perturbation_contract_preserves_structure(self):
        """Perturbation contracts keep the declared YAML shape intact."""

        contract = self.plugin.get_cmb_perturbation_contract(
            self.plugin.INITIAL_GUESSES
        )
        self.assertEqual(contract["model_name"], self.plugin.MODEL_NAME)
        self.assertEqual(contract["backend"], "camb")
        self.assertTrue(contract["standard"])
        self.assertEqual(contract["gauge"], "unspecified")
        self.assertEqual(
            contract["backend_mapping"]["camb"]["uses_standard_perturbations"],
            True,
        )
        perturbation_data = self.plugin.get_cmb_perturbation_data(
            self.plugin.INITIAL_GUESSES
        )
        self.assertIsInstance(perturbation_data, PerturbationContractData)
        self.assertTrue(perturbation_data.standard)

    def test_get_camb_params_rejects_malicious_expression(self):
        """Expressions attempting attribute access raise ``ValueError``."""
        bad_expression = "np.__class__.__mro__[2].__subclasses__()"
        self.plugin.CMB_CONTRACT["param_map"]["bad"] = bad_expression
        with self.assertRaises(ValueError):
            self.plugin.get_camb_params([70.0])

    def test_get_camb_params_rejects_recursion_depth(self):
        """Deeply nested calls exceed the evaluator's recursion limit."""
        expr = "exp(" * 30 + "1" + ")" * 30
        self.plugin.CMB_CONTRACT["param_map"]["deep"] = expr
        with self.assertRaises(ValueError):
            self.plugin.get_camb_params([70.0])

    def test_get_camb_params_rejects_node_blowup(self):
        """Expressions with too many nodes trigger a ``ValueError``."""
        expr = "+".join(["1"] * 200)
        self.plugin.CMB_CONTRACT["param_map"]["wide"] = expr
        with self.assertRaises(ValueError):
            self.plugin.get_camb_params([70.0])

    def test_cmb_param_map_rejects_invalid_keys(self):
        """Engine interface should reject unsupported CAMB parameters."""

        bad_model = dict(self.model_data)
        bad_model["cmb"] = {
            "backend": "camb",
            "param_map": {"H0": "H_0", "bad_key": 1},
            "grids": {},
            "values": {},
            "calls": [],
            "perturbations": self._make_standard_perturbations(),
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_param_map_rejects_conflicting_neutrino_specs(self):
        """Sum and individual neutrino masses cannot be combined."""

        clash = dict(self.model_data)
        clash["cmb"] = {
            "backend": "camb",
            "param_map": {
                "H0": "H_0",
                "ombh2": 0.022,
                "omch2": 0.12,
                "sum_mnu": 0.06,
                "mnu1": 0.01,
            },
            "grids": {},
            "values": {},
            "calls": [],
            "perturbations": self._make_standard_perturbations(),
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(clash, self.funcs)

    def test_cmb_valid_model_without_backend_fails(self):
        """A CMB-capable model must declare its backend."""

        bad_model = copy.deepcopy(self.model_data)
        del bad_model["cmb"]["backend"]
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_without_calls_fails(self):
        """A CMB-capable model must declare its adapter calls."""

        bad_model = copy.deepcopy(self.model_data)
        del bad_model["cmb"]["calls"]
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_without_perturbations_fails(self):
        """A CMB-capable model must declare perturbations."""

        bad_model = copy.deepcopy(self.model_data)
        del bad_model["cmb"]["perturbations"]
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_without_perturbation_standard_fails(self):
        """The perturbation contract must declare the standard flag."""

        bad_model = copy.deepcopy(self.model_data)
        del bad_model["cmb"]["perturbations"]["standard"]
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_with_invalid_perturbation_gauge_fails(self):
        """Invalid perturbation gauges are rejected."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["perturbations"]["gauge"] = "galactic"
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_standard_false_perturbation_contract_validates(self):
        """A non-standard perturbation contract validates when declared."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"] = (
            self._make_nonstandard_perturbations()
        )
        plugin = engine_plugin_validation.build_plugin(model_data, self.funcs)
        self.assertTrue(plugin.valid_for_cmb)
        self.assertFalse(plugin.CMB_PERTURBATION_STANDARD)
        self.assertEqual(
            plugin.CMB_PERTURBATION_CONTRACT["gauge"],
            "conformal_newtonian",
        )
        self.assertIsInstance(
            plugin.get_cmb_perturbation_data(plugin.INITIAL_GUESSES),
            PerturbationContractData,
        )

    def test_standard_false_perturbation_contract_without_math_fails(self):
        """Non-standard perturbations need declared mathematical content."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"] = {
            "contract_version": 1,
            "standard": False,
            "gauge": "conformal_newtonian",
            "variables": {
                "delta_x": {
                    "kind": "density_contrast",
                    "description": "Example density perturbation.",
                }
            },
            "derived": {},
            "equations": {},
            "closures": {},
            "sources": {},
            "validity": {
                "regimes": ["linear"],
                "notes": "Declared but incomplete.",
            },
            "backend_mapping": {
                "camb": {
                    "native_solver_required": True,
                    "implemented": False,
                }
            },
            "notes": "Missing mathematical content.",
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_standard_true_rejects_non_empty_math_sections(self):
        """Standard perturbations must keep the math sections empty."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"]["variables"] = {
            "delta_x": {"kind": "density_contrast"}
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_free_text_equation_lhs_fails(self):
        """Equation left-hand sides must use typed derivative syntax."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"] = (
            self._make_nonstandard_perturbations()
        )
        model_data["cmb"]["perturbations"]["equations"]["continuity_x"][
            "lhs"
        ] = "delta_x"
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_undeclared_perturbation_symbol_fails(self):
        """Perturbation expressions must not reference undeclared symbols."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["derived"]["delta_rho_eff"]["expression"] = "unknown_x"
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_unsafe_perturbation_expression_fails(self):
        """Unsafe perturbation expressions are rejected."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["sources"]["poisson"]["expression"] = "delta_x.__class__"
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_unknown_perturbation_key_fails(self):
        """Unknown perturbation contract keys are rejected."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["unexpected"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_nonstandard_variables_fails(self):
        """Non-standard perturbations must declare variables."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["variables"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_nonstandard_equations_fails(self):
        """Non-standard perturbations must declare equations."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["equations"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_nonstandard_sources_fails(self):
        """Non-standard perturbations must declare sources."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["sources"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_nonstandard_backend_mapping_fails(self):
        """Non-standard perturbations must declare backend mapping."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["backend_mapping"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_nonstandard_standard_mapping_fails(self):
        """Non-standard perturbations cannot declare standard support."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["backend_mapping"]["camb"]["native_solver_required"] = (
            True
        )
        perturbations["backend_mapping"]["camb"][
            "uses_standard_perturbations"
        ] = True
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_derived_cycle_fails(self):
        """Derived perturbation expressions must not cycle."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_nonstandard_perturbations()
        perturbations["derived"]["alpha"] = {"expression": "beta"}
        perturbations["derived"]["beta"] = {"expression": "alpha"}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_cmb_invalid_model_does_not_require_cmb(self):
        """Models that opt out of CMB do not need a contract block."""

        model_data = copy.deepcopy(self.model_data)
        model_data["valid_for_cmb"] = False
        model_data.pop("cmb", None)
        plugin = engine_plugin_validation.build_plugin(model_data, self.funcs)
        self.assertFalse(plugin.valid_for_cmb)
        self.assertEqual(plugin.CMB_CONTRACT, {})
        self.assertIsNone(plugin.CMB_PERTURBATION_DATA)

    def test_migrated_cmb_models_validate(self):
        """All migrated CMB models should build and validate cleanly."""

        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / "models"
        cache_dir = models_dir / "cache"
        model_names = [
            "cosmo_model_lcdm.yml",
            "cosmo_model_lcdm_mnu.yml",
            "cosmo_model_ref_planck2018.yml",
            "cosmo_model_tog.yml",
            "cosmo_model_torg.yml",
            "cosmo_model_wcdm.yml",
            "cosmo_model_w0wa.yml",
            "cosmo_model_qauc.yml",
            "cosmo_model_qrsf.yml",
            "cosmo_model_usmf2.yml",
        ]
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                yaml_path = models_dir / model_name
                cache_path = model_spec_validator.validate_and_cache_model(
                    yaml_path, cache_dir
                )
                funcs, parsed = model_coder.generate_callables(cache_path)
                plugin = engine_plugin_validation.build_plugin(parsed, funcs)
                validate_plugin = engine_plugin_validation.validate_plugin
                self.assertTrue(validate_plugin(plugin))
                contract = plugin.get_camb_contract(plugin.INITIAL_GUESSES)
                self.assertEqual(contract["backend"], "camb")
                self.assertIsNotNone(
                    plugin.get_cmb_perturbation_data(plugin.INITIAL_GUESSES)
                )
                if model_name == "cosmo_model_torg.yml":
                    self.assertTrue(plugin.CMB_PERTURBATION_STANDARD)
                    self.assertEqual(
                        plugin.CMB_PERTURBATION_CONTRACT["backend_mapping"][
                            "camb"
                        ]["uses_standard_perturbations"],
                        True,
                    )

    def test_unknown_cmb_key_fails(self):
        """Unknown contract keys are rejected early."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_unknown_call_method_fails(self):
        """Unsupported CAMB methods are rejected."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["calls"] = [{"method": "set_unknown"}]
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_unknown_call_reference_fails(self):
        """Unknown grid and value references are rejected."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"] = {
            "backend": "camb",
            "param_map": copy.deepcopy(self.base_param_map),
            "grids": {
                "a_grid": {
                    "symbol": "a",
                    "lower": 0.001,
                    "upper": 1.0,
                    "points": 16,
                    "spacing": "linear",
                }
            },
            "values": {
                "w_tog": {
                    "grid": "a_grid",
                    "expression": "-1 - x",
                }
            },
            "calls": [
                {
                    "method": "set_dark_energy_w_a",
                    "args": {
                        "a": "@grid.a_grid",
                        "w": "@value.missing",
                    },
                    "kwargs": {"dark_energy_model": "ppf"},
                }
            ],
            "perturbations": self._make_standard_perturbations(
                background_adapter=True
            ),
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_declared_value_parameter_is_accepted(self):
        """Declared parameters used only by values pass validation."""

        model_data = copy.deepcopy(self.model_data)
        model_data["parameters"].append(
            {
                "python_var": "a_T",
                "latex_name": "a_T",
                "bounds": [0.5, 2.0],
            }
        )
        model_data["cmb"] = {
            "backend": "camb",
            "param_map": copy.deepcopy(self.base_param_map),
            "grids": {
                "a_grid": {
                    "symbol": "a",
                    "lower": 0.001,
                    "upper": 1.0,
                    "points": 8,
                    "spacing": "linear",
                }
            },
            "values": {
                "x": {
                    "grid": "a_grid",
                    "expression": "(a/a_T)**3",
                }
            },
            "calls": [],
            "perturbations": self._make_standard_perturbations(),
        }
        plugin = engine_plugin_validation.build_plugin(model_data, self.funcs)
        contract = plugin.get_camb_contract(plugin.INITIAL_GUESSES)
        self.assertIn("x", contract["values"])

    def test_undeclared_value_parameter_fails(self):
        """Values referencing undeclared parameters fail validation."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"] = {
            "backend": "camb",
            "param_map": copy.deepcopy(self.base_param_map),
            "grids": {
                "a_grid": {
                    "symbol": "a",
                    "lower": 0.001,
                    "upper": 1.0,
                    "points": 8,
                    "spacing": "linear",
                }
            },
            "values": {
                "x": {
                    "grid": "a_grid",
                    "expression": "(a/a_T)**3",
                }
            },
            "calls": [],
            "perturbations": self._make_standard_perturbations(),
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(model_data, self.funcs)

    def test_equation_sanitization(self):
        """Equations are sanitized into Matplotlib-friendly form."""
        self.assertEqual(self.plugin.MODEL_EQUATIONS_LATEX_SN[0], "$E=mc^2$")

    def test_make_logposterior_applies_priors(self):
        """Combined posterior should include prior contributions and bounds."""

        calls: list[tuple[float, ...]] = []

        def like(params):
            calls.append(tuple(params))
            return -0.5 * sum(value * value for value in params)

        like.parameter_bounds = [(0.0, 1.0), (None, None)]
        priors = [
            {"type": "uniform", "lower": 0.0, "upper": 1.0},
            {"type": "gaussian", "mean": 0.0, "sigma": 0.5},
        ]
        posterior = engine_plugin_validation.make_logposterior(like, priors)

        rejected = posterior((-0.1, 0.0))
        self.assertTrue(math.isinf(rejected) and rejected < 0)
        value = posterior((0.5, 0.0))
        base = like((0.5, 0.0))
        gaussian = -math.log(0.5 * math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(value, base + gaussian, places=7)
        self.assertIn((0.5, 0.0), calls)

    def test_make_logposterior_respects_transforms(self):
        """Transforms should apply Jacobian corrections automatically."""

        def like(params):
            return -0.5 * sum(value * value for value in params)

        def exp_transform(raw):
            transformed = math.exp(raw)
            return transformed, raw

        like.parameter_bounds = [(None, None), (0.5, 2.0)]
        like.parameter_transforms = [None, exp_transform]
        priors = [
            {"type": "uniform"},
            {"type": "uniform", "lower": 0.5, "upper": 2.0},
        ]
        posterior = engine_plugin_validation.make_logposterior(like, priors)

        # Raw value of 0.0 transforms to 1.0 and remains inside bounds.
        value = posterior((0.0, 0.0))
        expected = like((0.0, math.exp(0.0))) - math.log(2.0 - 0.5) + 0.0
        self.assertAlmostEqual(value, expected)
        # Raw value of log(3) transforms outside the allowed range and is
        # rejected.
        transformed = posterior((0.0, math.log(3.0)))
        self.assertTrue(math.isinf(transformed) and transformed < 0)

    def test_make_logposterior_is_picklable(self):
        """Posterior evaluator pickles with log-uniform transforms."""

        priors = [
            {"type": "loguniform", "lower": 1e-3, "upper": 1.0},
        ]
        posterior = MAKE_POSTERIOR(_linear_like, priors)
        with multiprocessing_module.get_context("spawn").Pool(1) as pool:
            restored_value = pool.apply(_evaluate_posterior, (posterior,))
        self.assertAlmostEqual(restored_value, posterior([0.1]))


if __name__ == "__main__":
    unittest.main()
# fmt: on
