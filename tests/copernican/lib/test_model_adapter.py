# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for ``copernican.lib.model_adapter`` helpers."""

import copy
import math
import multiprocessing as multiprocessing_module
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy
import yaml

from copernican import validation as validation_module
from copernican.lib import cmb_contract
from copernican.lib import model_adapter as model_plugin_validation
from copernican.lib import model_coder, model_spec_validator, run_manifest
from copernican.lib.cmb_identity import CCMBS_ID
from copernican.lib.likelihoods.cmb import cache, cmb, projection
from copernican.lib.model_adapter import PluginValidationError
from copernican.lib.perturbation_contract import PerturbationContractData

# fmt: off
MAKE_POSTERIOR = model_plugin_validation.make_logposterior


def _dummy_func(*_args, **_kwargs):
    """Return a placeholder numerical value."""
    return 0.0


def _linear_like(params):
    """Simple log-likelihood used to test pickling."""

    return -sum(params)


def _evaluate_posterior(posterior):
    """Return the posterior evaluation from a worker process."""

    return posterior([0.1])


def distance_modulus_model(z_val, hubble_parameter):
    """Toy distance modulus helper that stays trivially picklable."""

    return float(z_val) + float(hubble_parameter)


def get_comoving_distance_mpc(z_val, hubble_parameter):
    """Return a linearised comoving distance for testing only."""

    return float(z_val) * 100.0 / max(float(hubble_parameter), 1.0)


def get_luminosity_distance_mpc(z_val, hubble_parameter):
    """Derive luminosity distance directly from the comoving result."""

    return (1.0 + float(z_val)) * get_comoving_distance_mpc(
        z_val, hubble_parameter
    )


def get_angular_diameter_distance_mpc(z_val, hubble_parameter):
    """Derive angular diameter distance from the comoving helper."""

    return get_comoving_distance_mpc(z_val, hubble_parameter) / (
        1.0 + float(z_val)
    )


def get_hz_per_mpc(z_val, hubble_parameter):
    """Return a monotonic H(z) scaling for deterministic assertions."""

    return float(hubble_parameter) * (1.0 + float(z_val))


def get_dv_mpc(z_val, hubble_parameter):
    """Return a BAO-inspired helper anchored to the comoving distance."""

    dm_val = get_comoving_distance_mpc(z_val, hubble_parameter)
    numerator = dm_val * dm_val * 299792.458 * float(z_val)
    ratio = numerator / get_hz_per_mpc(z_val, hubble_parameter)
    return ratio ** (1.0 / 3.0)


def get_sound_horizon_rs_mpc(hubble_parameter):
    """Simple sound horizon approximation suitable for tests."""

    return 144.0 / max(float(hubble_parameter), 1.0)


def helper_extra_function():
    """Extra helper stored on the plugin to prove extras stay intact."""

    return "extra"


def _inspect_plugin(plugin: model_plugin_validation.ModelPlugin):
    """Return round-trip observations from a worker process."""

    return (
        isinstance(plugin.extras, model_plugin_validation.FrozenMapping),
        plugin.extras["custom_extra"](),
        plugin.FIXED_PARAMS["H0"],
    )


def _build_sample_plugin() -> model_plugin_validation.ModelPlugin:
    """Create a minimal plugin suitable for pickling tests."""

    model_data = {
        "model_name": "TestModel",
        "valid_for_cmb": False,
        "description": "Synthetic plugin used for pickling tests.",
        "abstract": "Ensures FrozenMapping wrappers survive pickle.",
        "parameters": [
            {
                "name": "H0",
                "latex_name": "$H_0$",
                "unit": "km/s/Mpc",
                "bounds": (60.0, 80.0),
                "prior": {"type": "fixed", "value": 70.0},
            }
        ],
        "equations": {"sne": ["H_0"], "bao": []},
        "likelihood": {"datasets": ["sne"]},
    }
    func_dict = {
        "distance_modulus_model": distance_modulus_model,
        "get_comoving_distance_Mpc": get_comoving_distance_mpc,
        "get_luminosity_distance_Mpc": get_luminosity_distance_mpc,
        "get_angular_diameter_distance_Mpc": (
            get_angular_diameter_distance_mpc
        ),
        "get_Hz_per_Mpc": get_hz_per_mpc,
        "get_DV_Mpc": get_dv_mpc,
        "get_sound_horizon_rs_Mpc": get_sound_horizon_rs_mpc,
        "custom_extra": helper_extra_function,
    }
    return model_plugin_validation.build_model_plugin(model_data, func_dict)


class TestModelAdapterExports(unittest.TestCase):
    """Verify the root adapter module exports the expected surface."""

    def test_public_exports_are_present(self) -> None:
        self.assertTrue(
            callable(model_plugin_validation.build_model_plugin)
        )
        self.assertTrue(callable(model_plugin_validation.build_plugin))
        self.assertTrue(callable(model_plugin_validation.validate_plugin))
        self.assertTrue(hasattr(model_plugin_validation, "ModelPlugin"))
        self.assertTrue(
            hasattr(model_plugin_validation, "CMBContractEvaluator")
        )
        self.assertTrue(
            hasattr(model_plugin_validation, "CMBParameterEvaluator")
        )
        self.assertTrue(hasattr(model_plugin_validation, "FrozenMapping"))
        self.assertTrue(
            hasattr(model_plugin_validation, "PluginValidationError")
        )
        self.assertTrue(callable(model_plugin_validation.sanitize_equation))
        contract_evaluator = model_plugin_validation.CMBContractEvaluator
        self.assertTrue(hasattr(contract_evaluator, "evaluate_param_map"))
        self.assertTrue(
            hasattr(model_plugin_validation.ModelPlugin, "get_cmb_params")
        )
        self.assertTrue(
            hasattr(
                model_plugin_validation.ModelPlugin,
                "get_cmb_contract",
            )
        )
        self.assertTrue(
            hasattr(
                model_plugin_validation.ModelPlugin,
                "get_cmb_declared_runtime",
            )
        )
        self.assertTrue(
            hasattr(
                model_plugin_validation.ModelPlugin,
                "get_cmb_perturbation_contract",
            )
        )
        self.assertTrue(
            hasattr(
                model_plugin_validation.ModelPlugin,
                "get_cmb_perturbation_data",
            )
        )
        self.assertFalse(
            hasattr(model_plugin_validation, "CMB_BACKEND_CAPABILITIES")
        )
        self.assertIn("ModelPlugin", model_plugin_validation.__all__)
        self.assertIn("validate_plugin", model_plugin_validation.__all__)
        self.assertIs(
            cmb_contract.CMBContractEvaluator,
            model_plugin_validation.CMBContractEvaluator,
        )
        self.assertIs(
            cmb_contract.CMBParameterEvaluator,
            model_plugin_validation.CMBParameterEvaluator,
        )
        self.assertIs(
            cmb_contract._validate_cmb_contract_definition,
            model_plugin_validation._validate_cmb_contract_definition,
        )

    def test_public_helpers_behave_as_expected(self) -> None:
        frozen = model_plugin_validation.FrozenMapping(
            {"alpha": 1, "beta": [2, 3]}
        )
        self.assertEqual(frozen.to_dict(), {"alpha": 1, "beta": [2, 3]})
        evaluator = model_plugin_validation.CMBParameterEvaluator(
            ("x",),
            ("x",),
            {"H0": "x"},
        )
        self.assertEqual(evaluator((4.0,))["H0"], 4.0)
        self.assertIsInstance(
            model_plugin_validation.PluginValidationError("boom"),
            RuntimeError,
        )
        self.assertIsInstance(
            model_plugin_validation.sanitize_equation("x"), str
        )


class ModelInterfaceTestCase(unittest.TestCase):
    """Validate model adapter construction and associated helpers."""

    def test_validation_summary_helpers_are_exposed(self):
        self.assertTrue(hasattr(validation_module, "read_validation_summary"))
        self.assertTrue(
            hasattr(validation_module, "write_validation_summary")
        )

    def setUp(self):
        """Build a minimal model adapter for reuse across tests."""
        self.base_param_map = {
            "H0": "H_0",
            "ombh2": 0.022,
            "omch2": 0.12,
            "Neff": 3.044,
            "num_massive_neutrinos": 3,
            "sum_mnu": 0.06,
        }
        self.base_cmb_contract = {
            "param_map": self.base_param_map,
            "grids": {},
            "values": {},
            "calls": [],
        }
        self.base_cmb_contract["perturbations"] = (
            self._make_declared_perturbations()
        )
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
        req = model_plugin_validation.REQUIRED_FUNCTIONS
        funcs = {name: _dummy_func for name in req}
        self.funcs = funcs
        build_plugin = model_plugin_validation.build_plugin
        self.plugin = build_plugin(self.model_data, funcs)

    def _make_declared_perturbations(
        self,
        *,
        background_adapter: bool = False,
    ) -> dict[str, object]:
        """Return a fully declared perturbation contract."""

        perturbations = {
            "contract_version": 2,
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
                "phi_aux": {
                    "kind": "metric_potential_phi",
                    "description": "Example Newtonian potential.",
                },
                "psi_aux": {
                    "kind": "metric_potential_psi",
                    "description": "Example curvature potential.",
                },
            },
            "derived": {
                "density_drive": {
                    "expression": "delta_x + phi_aux",
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
                    "rhs": "-theta_x + phi_aux",
                    "role": "continuity",
                },
                "euler_x": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "theta_x",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "-Hconf * theta_x + k * psi_aux",
                    "role": "euler",
                },
            },
            "constraints": {
                "poisson_phi": {
                    "target": "phi_aux",
                    "expression": "0.25 * delta_x",
                    "role": "constraint",
                }
            },
            "closures": {
                "no_anisotropic_stress": {
                    "target": "psi_aux",
                    "expression": "phi_aux",
                    "role": "closure",
                }
            },
            "sources": {
                "poisson": {
                    "expression": "visibility * density_drive",
                    "role": "monopole",
                },
                "polarization": {
                    "expression": "visibility * theta_x",
                    "role": "polarization",
                }
            },
            "observables": {
                "temperature": {
                    "kind": "transfer_component",
                    "projection": "line_of_sight_temperature",
                    "source_terms": {"monopole": "poisson"},
                },
                "polarization_e": {
                    "kind": "transfer_component",
                    "projection": "line_of_sight_polarization_e",
                    "source_terms": {"polarization": "polarization"},
                },
                "TT": {
                    "kind": "angular_power_spectrum",
                    "primary": "temperature",
                    "secondary": "temperature",
                },
                "TE": {
                    "kind": "angular_power_spectrum",
                    "primary": "temperature",
                    "secondary": "polarization_e",
                },
            },
            "initial_conditions": {
                "delta_seed": {
                    "target": {
                        "variable": "delta_x",
                        "wrt": "tau",
                        "order": 0,
                    },
                    "expression": "seed",
                },
                "theta_seed": {
                    "target": {
                        "variable": "theta_x",
                        "wrt": "tau",
                        "order": 0,
                    },
                    "expression": "0.1 * seed",
                },
            },
            "boundary_conditions": {},
            "numerics": {
                "ode_rtol": 1.0e-5,
                "ode_atol": 1.0e-8,
            },
            "validity": {
                "regimes": ["linear", "scalar"],
                "notes": "Declared for first-order scalar perturbations.",
            },
            "notes": "Declared perturbation mathematics are declared here.",
        }
        if background_adapter:
            perturbations["validity"]["regimes"].append(
                "declared_background_adapter"
            )
        return perturbations

    def test_plugin_validation(self):
        """Plugin built from minimal data should validate."""
        self.assertTrue(model_plugin_validation.validate_plugin(self.plugin))

    def test_missing_attribute_fails_validation(self):
        """A plugin lacking required attributes is rejected."""
        bad = SimpleNamespace()
        with self.assertLogs(level="ERROR") as captured_logs:
            with self.assertRaises(PluginValidationError):
                model_plugin_validation.validate_plugin(bad)
        self.assertIn("Plugin validation issue", "".join(captured_logs.output))

    def test_get_cmb_params_expression(self):
        """LaTeX expressions in ``cmb.param_map`` evaluate correctly."""
        cmb_params = self.plugin.get_cmb_params([70.0])
        self.assertEqual(cmb_params["H0"], 70.0)
        self.assertAlmostEqual(cmb_params["ombh2"], 0.022)

    def test_empty_calls_preserve_scalar_cmb_params(self):
        """An explicit empty call list keeps the scalar mapping intact."""

        cmb_params = self.plugin.get_cmb_params([70.0])
        self.assertEqual(cmb_params["Neff"], 3.044)
        self.assertEqual(cmb_params["sum_mnu"], 0.06)

    def test_get_cmb_contract_preserves_strings_and_arrays(self):
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
            "param_map": copy.deepcopy(self.base_param_map),
            "model_parameters": {"Tcmb_K": 2.7255},
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
            "perturbations": self._make_declared_perturbations(
                background_adapter=True
            ),
        }
        plugin = model_plugin_validation.build_plugin(model_data, self.funcs)
        contract = plugin.get_cmb_contract(plugin.INITIAL_GUESSES)
        self.assertNotIn("backend", contract)
        self.assertEqual(contract["param_map"]["H0"], 70.0)
        self.assertEqual(contract["model_parameters"]["Tcmb_K"], 2.7255)
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
        declared_runtime = plugin.get_cmb_declared_runtime(
            plugin.INITIAL_GUESSES
        )
        self.assertEqual(
            declared_runtime["model_parameters"]["Tcmb_K"],
            2.7255,
        )
        self.assertIsInstance(contract["calls"][0]["args"]["a"], numpy.ndarray)
        self.assertIsInstance(contract["calls"][0]["args"]["w"], numpy.ndarray)

    def test_get_cmb_perturbation_contract_preserves_structure(self):
        """Perturbation contracts keep the declared YAML shape intact."""

        self.assertTrue(
            callable(self.plugin.get_cmb_perturbation_contract)
        )
        contract = self.plugin.get_cmb_perturbation_contract(
            self.plugin.INITIAL_GUESSES
        )
        self.assertEqual(contract["model_name"], self.plugin.MODEL_NAME)
        self.assertNotIn("backend", contract)
        self.assertNotIn("standard", contract)
        self.assertNotIn("backend_mapping", contract)
        self.assertEqual(contract["gauge"], "conformal_newtonian")
        perturbation_data = self.plugin.get_cmb_perturbation_data(
            self.plugin.INITIAL_GUESSES
        )
        self.assertIsInstance(perturbation_data, PerturbationContractData)
        declared_runtime = self.plugin.get_cmb_declared_runtime(
            self.plugin.INITIAL_GUESSES
        )
        self.assertEqual(
            declared_runtime["model_name"],
            self.plugin.MODEL_NAME,
        )
        self.assertNotIn("backend", declared_runtime)
        self.assertIn("param_map", declared_runtime)
        self.assertIn("model_parameters", declared_runtime)
        self.assertIn("perturbation_data", declared_runtime)
        self.assertIs(declared_runtime["perturbation_data"], perturbation_data)

    def test_get_cmb_params_rejects_malicious_expression(self):
        """Expressions attempting attribute access raise ``ValueError``."""
        bad_expression = "np.__class__.__mro__[2].__subclasses__()"
        self.plugin.CMB_CONTRACT["param_map"]["bad"] = bad_expression
        with self.assertRaises(ValueError):
            self.plugin.get_cmb_params([70.0])

    def test_get_cmb_params_rejects_recursion_depth(self):
        """Deeply nested calls exceed the evaluator's recursion limit."""
        expr = "exp(" * 30 + "1" + ")" * 30
        self.plugin.CMB_CONTRACT["param_map"]["deep"] = expr
        with self.assertRaises(ValueError):
            self.plugin.get_cmb_params([70.0])

    def test_get_cmb_params_rejects_node_blowup(self):
        """Expressions with too many nodes trigger a ``ValueError``."""
        expr = "+".join(["1"] * 200)
        self.plugin.CMB_CONTRACT["param_map"]["wide"] = expr
        with self.assertRaises(ValueError):
            self.plugin.get_cmb_params([70.0])

    def test_cmb_param_map_rejects_invalid_keys(self):
        """Model interface should reject unsupported CMB parameters."""

        bad_model = dict(self.model_data)
        bad_model["cmb"] = {
            "param_map": {"H0": "H_0", "bad_key": 1},
            "grids": {},
            "values": {},
            "calls": [],
            "perturbations": self._make_declared_perturbations(),
        }
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_param_map_rejects_conflicting_neutrino_specs(self):
        """Sum and individual neutrino masses cannot be combined."""

        clash = dict(self.model_data)
        clash["cmb"] = {
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
            "perturbations": self._make_declared_perturbations(),
        }
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(clash, self.funcs)

    def test_cmb_backend_selector_fails(self):
        """A CMB-capable model must reject a backend selector."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["backend"] = "external"
        with self.assertRaisesRegex(ValueError, "removed route key.*backend"):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_without_calls_fails(self):
        """A CMB-capable model must declare its adapter calls."""

        bad_model = copy.deepcopy(self.model_data)
        del bad_model["cmb"]["calls"]
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_without_perturbations_fails(self):
        """A CMB-capable model must declare perturbations."""

        bad_model = copy.deepcopy(self.model_data)
        del bad_model["cmb"]["perturbations"]
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_perturbation_route_flag_fails(self):
        """The perturbation contract must reject a route flag."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["perturbations"]["standard"] = False
        with self.assertRaisesRegex(
            ValueError,
            "removed route key.*standard",
        ):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_cmb_valid_model_with_invalid_perturbation_gauge_fails(self):
        """Invalid perturbation gauges are rejected."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["perturbations"]["gauge"] = "galactic"
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_declared_perturbation_contract_validates(self):
        """A declared perturbation contract validates when declared."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"] = (
            self._make_declared_perturbations()
        )
        plugin = model_plugin_validation.build_plugin(model_data, self.funcs)
        self.assertTrue(plugin.valid_for_cmb)
        self.assertFalse(hasattr(plugin, "CMB_PERTURBATION_STANDARD"))
        self.assertEqual(
            plugin.CMB_PERTURBATION_CONTRACT["gauge"],
            "conformal_newtonian",
        )
        self.assertEqual(
            plugin.CMB_PERTURBATION_CONTRACT["contract_version"],
            2,
        )
        self.assertIsInstance(
            plugin.get_cmb_perturbation_data(plugin.INITIAL_GUESSES),
            PerturbationContractData,
        )
        declared_runtime = plugin.get_cmb_declared_runtime(
            plugin.INITIAL_GUESSES
        )
        self.assertNotIn("standard", declared_runtime["perturbations"])
        self.assertIs(
            declared_runtime["perturbation_data"],
            plugin.get_cmb_perturbation_data(plugin.INITIAL_GUESSES),
        )

    def test_declared_perturbation_contract_without_math_fails(self):
        """Declared perturbations need declared mathematical content."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"] = {
            "contract_version": 2,
            "gauge": "conformal_newtonian",
            "variables": {
                "delta_x": {
                    "kind": "density_contrast",
                    "description": "Example density perturbation.",
                }
            },
            "derived": {},
            "constraints": {},
            "equations": {},
            "closures": {},
            "sources": {},
            "observables": {},
            "initial_conditions": {},
            "boundary_conditions": {},
            "validity": {
                "regimes": ["linear"],
                "notes": "Declared but incomplete.",
            },
            "notes": "Missing mathematical content.",
        }
        with self.assertRaisesRegex(
            ValueError,
            "must declare equations",
        ):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_standard_route_flag_is_rejected(self):
        """Removed standard-route declarations must fail validation."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"]["standard"] = True
        with self.assertRaisesRegex(
            ValueError,
            "removed route key.*standard",
        ):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_free_text_equation_lhs_fails(self):
        """Equation left-hand sides must use typed derivative syntax."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"]["perturbations"] = (
            self._make_declared_perturbations()
        )
        model_data["cmb"]["perturbations"]["equations"]["continuity_x"][
            "lhs"
        ] = "delta_x"
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_undeclared_perturbation_symbol_fails(self):
        """Perturbation expressions must not reference undeclared symbols."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["derived"]["density_drive"]["expression"] = "unknown_x"
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_unsafe_perturbation_expression_fails(self):
        """Unsafe perturbation expressions are rejected."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["sources"]["poisson"]["expression"] = "delta_x.__class__"
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_unknown_perturbation_key_fails(self):
        """Unknown perturbation contract keys are rejected."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["unexpected"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_declared_variables_fails(self):
        """Declared perturbations must declare variables."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["variables"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_declared_equations_fail(self):
        """Declared perturbations must declare equations."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["equations"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaisesRegex(ValueError, "must declare equations"):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_missing_declared_initial_conditions_fail(self):
        """Declared perturbations must declare initial conditions."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["initial_conditions"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaisesRegex(
            ValueError,
            "must declare initial_conditions",
        ):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_backend_mapping_fails(self):
        """Declared perturbations must reject backend mappings."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["backend_mapping"] = {}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_backend_mapping_selector_fails(self):
        """Backend mapping selectors cannot enter declared contracts."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["backend_mapping"] = {
            "external": {"theory_selector": "default"}
        }
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_derived_cycle_fails(self):
        """Derived perturbation expressions must not cycle."""

        model_data = copy.deepcopy(self.model_data)
        perturbations = self._make_declared_perturbations()
        perturbations["derived"]["alpha"] = {"expression": "beta"}
        perturbations["derived"]["beta"] = {"expression": "alpha"}
        model_data["cmb"]["perturbations"] = perturbations
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

    def test_cmb_invalid_model_does_not_require_cmb(self):
        """Models that opt out of CMB do not need a contract block."""

        model_data = copy.deepcopy(self.model_data)
        model_data["valid_for_cmb"] = False
        model_data.pop("cmb", None)
        plugin = model_plugin_validation.build_plugin(model_data, self.funcs)
        self.assertFalse(plugin.valid_for_cmb)
        self.assertEqual(plugin.CMB_CONTRACT, {})
        self.assertIsNone(plugin.CMB_PERTURBATION_DATA)

    def test_declared_cmb_models_validate(self):
        """All CMB-capable model assets must expose their exact ontology."""

        repo_root = Path(__file__).resolve().parents[3]
        models_dir = repo_root / "copernican" / "models"
        model_names = [
            "model_lcdm.yml",
            "model_lcdm_mnu.yml",
            "model_ref_planck2018.yml",
            "model_tog.yml",
            "model_torg.yml",
            "model_wcdm.yml",
            "model_w0wa.yml",
            "model_qauc.yml",
            "model_qrsf.yml",
        ]
        expected_species = {
            "model_lcdm.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "photon",
            },
            "model_lcdm_mnu.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_ref_planck2018.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_tog.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_torg.yml": {
                "baryon",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_wcdm.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_w0wa.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_qauc.yml": {
                "baryon",
                "cdm",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
            "model_qrsf.yml": {
                "baryon",
                "massless_neutrino",
                "massive_neutrino",
                "photon",
            },
        }
        expected_source_closures = {
            "model_qrsf.yml": {
                "qrsf_matter_density",
                "qrsf_matter_momentum",
                "qrsf_baryon_euler",
            },
            "model_torg.yml": {
                "torg_matter_density",
                "torg_matter_momentum",
                "torg_baryon_euler",
            },
        }
        common_sources = {
            "lensing_potential",
            "polarization_b_source",
            "polarization_source",
            "temperature_doppler",
            "temperature_isw",
            "temperature_monopole",
            "temperature_quadrupole",
            "temperature_quadrupole_derivative",
        }
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                yaml_path = models_dir / model_name
                with tempfile.TemporaryDirectory() as cache_dir:
                    cache_path = (
                        model_spec_validator.validate_and_cache_model(
                            yaml_path,
                            cache_dir,
                        )
                    )
                    funcs, parsed = model_coder.generate_callables(cache_path)
                plugin = model_plugin_validation.build_plugin(parsed, funcs)
                validate_plugin = model_plugin_validation.validate_plugin
                self.assertTrue(validate_plugin(plugin))
                contract = plugin.get_cmb_contract(plugin.INITIAL_GUESSES)
                self.assertNotIn("backend", contract)
                self.assertIsNotNone(
                    plugin.get_cmb_perturbation_data(plugin.INITIAL_GUESSES)
                )
                self.assertIsNotNone(
                    plugin.get_cmb_declared_runtime(plugin.INITIAL_GUESSES)
                )
                self.assertFalse(
                    hasattr(plugin, "CMB_PERTURBATION_STANDARD")
                )
                summary = plugin.get_cmb_perturbation_data(
                    plugin.INITIAL_GUESSES
                ).manifest_summary
                self.assertEqual(
                    summary["execution_route"]["solver_id"],
                    CCMBS_ID,
                )
                self.assertTrue(summary["execution_route"]["ready"])
                self.assertEqual(
                    set(summary["species_names"]),
                    expected_species[model_name],
                )
                self.assertEqual(set(summary["sector_names"]), {"scalar"})
                self.assertEqual(
                    set(summary["initial_condition_family_names"]),
                    {"adiabatic_scalar"},
                )
                self.assertEqual(
                    set(summary["source_names"]),
                    common_sources
                    | expected_source_closures.get(model_name, set()),
                )
                self.assertEqual(
                    set(summary["angular_power_spectrum_targets"]),
                    {"TT", "TE", "EE", "BB", "PP", "TP", "EP"},
                )
                self.assertTrue(summary["equation_names"])
                self.assertTrue(summary["initial_condition_names"])
                equation_names = set(summary["equation_names"])
                initial_names = set(summary["initial_condition_names"])
                has_cdm = "cdm" in expected_species[model_name]
                self.assertEqual("evolve_delta_c" in equation_names, has_cdm)
                self.assertEqual("delta_c_seed" in initial_names, has_cdm)

    def test_declared_model_asset_cutover_is_complete(self):
        """The model corpus must contain one canonical declared LCDM asset."""

        repo_root = Path(__file__).resolve().parents[3]
        models_dir = repo_root / "copernican" / "models"
        expected_names = {
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
        }
        self.assertEqual(
            {path.name for path in models_dir.glob("model_*.yml")},
            expected_names,
        )
        forbidden_text = {
            "backend:",
            "backend_mapping",
            "ccmbs",
            "fallback integration",
            "legacy inference",
            "migration artifact",
            "standard:",
        }
        for yaml_path in sorted(models_dir.glob("model_*.yml")):
            with self.subTest(model_name=yaml_path.name):
                source_text = yaml_path.read_text(encoding="utf-8").casefold()
                for term in forbidden_text:
                    self.assertNotIn(term, source_text)

                model_data = yaml.safe_load(source_text)
                perturbations = model_data["cmb"]["perturbations"]
                declared_species = set(perturbations["species"])
                scalar_species = set(
                    perturbations["sectors"]["scalar"]["species"]
                )
                self.assertEqual(scalar_species, declared_species)

                param_map = model_data["cmb"]["param_map"]
                background = model_data["cmb"]["background"]["derived"]
                if "cdm" not in declared_species:
                    self.assertNotIn("omch2", param_map)
                    self.assertNotIn("omega_c0", background)
                if "massive_neutrino" not in declared_species:
                    self.assertNotIn("sum_mnu", param_map)
                    self.assertNotIn("num_massive_neutrinos", param_map)

    def test_usmf2_declared_route_is_available(self):
        """USMF2 exposes its complete closure through the declared solver."""

        repo_root = Path(__file__).resolve().parents[3]
        yaml_path = repo_root / "copernican" / "models" / "model_usmf2.yml"
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                yaml_path,
                cache_dir,
            )
            funcs, parsed = model_coder.generate_callables(cache_path)
        plugin = model_plugin_validation.build_plugin(parsed, funcs)

        self.assertTrue(plugin.valid_for_cmb)
        perturbations = parsed["cmb"]["perturbations"]
        self.assertTrue(perturbations["variables"])
        self.assertTrue(perturbations["equations"])
        self.assertTrue(perturbations["observables"])
        self.assertIn(
            "usmf2_declared_production",
            perturbations["validity"]["regimes"],
        )
        self.assertNotIn("standard", perturbations)
        self.assertNotIn("backend_mapping", perturbations)
        self.assertNotIn("cdm", perturbations["species"])
        self.assertIsNotNone(plugin.CMB_DECLARED_RUNTIME)
        self.assertIsNotNone(plugin.CMB_PERTURBATION_DATA)

    def test_declared_models_use_theory_neutral_scalar_metadata(self):
        """Scalar metadata must not smuggle LCDM assumptions into models."""

        repo_root = Path(__file__).resolve().parents[3]
        models_dir = repo_root / "copernican" / "models"
        forbidden_terms = ("lcdm", "flat", "cold dark matter", "standard")
        for yaml_path in sorted(models_dir.glob("model_*.yml")):
            with self.subTest(model_name=yaml_path.name):
                model_data = yaml.safe_load(
                    yaml_path.read_text(encoding="utf-8")
                )
                cmb_contract = model_data.get("cmb")
                if not isinstance(cmb_contract, dict):
                    continue
                perturbations = cmb_contract.get("perturbations")
                if not isinstance(perturbations, dict):
                    continue
                scalar = perturbations["sectors"]["scalar"]
                description = scalar["description"].casefold()
                self.assertNotIn("standard", perturbations)
                self.assertNotIn("backend_mapping", perturbations)
                self.assertFalse(
                    any(term in description for term in forbidden_terms)
                )

    def test_declared_cmb_models_execute_finite_declared_tt(self):
        """Every CMB-capable model must execute finite declared TT."""

        repo_root = Path(__file__).resolve().parents[3]
        models_dir = repo_root / "copernican" / "models"
        model_names = [
            "model_lcdm.yml",
            "model_lcdm_mnu.yml",
            "model_ref_planck2018.yml",
            "model_tog.yml",
            "model_torg.yml",
            "model_wcdm.yml",
            "model_w0wa.yml",
            "model_qauc.yml",
            "model_qrsf.yml",
        ]
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                model_data = yaml.safe_load(
                    (models_dir / model_name).read_text(encoding="utf-8")
                )
                model_data["cmb"]["numerical"].update(
                    {
                        "ell_min": 2,
                        "ell_max": 20,
                        "k_min": 1.0e-4,
                        "k_max": 1.0e-2,
                        "k_sample_count": 1,
                        "eta_sample_count": 16,
                        "source_grid_multiplier": 1,
                    }
                )
                model_data["cmb"]["perturbations"][
                    "accuracy_controls"
                ]["scalar_reference_ells"] = [2, 20]
                model_data["cmb"]["perturbations"]["accuracy_controls"][
                    "minimum_k_sample_count"
                ] = 1
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / model_name
                    temp_path.write_text(
                        yaml.safe_dump(model_data, sort_keys=False),
                        encoding="utf-8",
                    )
                    cache_path = (
                        model_spec_validator.validate_and_cache_model(
                            temp_path,
                            Path(temp_dir) / "cache",
                        )
                    )
                    functions, parsed = model_coder.generate_callables(
                        cache_path
                    )
                    plugin = model_plugin_validation.build_plugin(
                        parsed,
                        functions,
                    )
                    spectra = cmb.compute_cmb_spectrum_cached(
                        plugin,
                        plugin.INITIAL_GUESSES,
                        numpy.asarray([2], dtype=int),
                        spectra=("TT", "TE", "EE"),
                    )
                for values in spectra.values():
                    self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_unknown_cmb_key_fails(self):
        """Unknown contract keys are rejected early."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_unknown_call_method_fails(self):
        """Unsupported CMB contract methods are rejected."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"]["calls"] = [{"method": "set_unknown"}]
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

    def test_unknown_call_reference_fails(self):
        """Unknown grid and value references are rejected."""

        bad_model = copy.deepcopy(self.model_data)
        bad_model["cmb"] = {
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
            "perturbations": self._make_declared_perturbations(
                background_adapter=True
            ),
        }
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(bad_model, self.funcs)

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
            "perturbations": self._make_declared_perturbations(),
        }
        plugin = model_plugin_validation.build_plugin(model_data, self.funcs)
        contract = plugin.get_cmb_contract(plugin.INITIAL_GUESSES)
        self.assertIn("x", contract["values"])

    def test_undeclared_value_parameter_fails(self):
        """Values referencing undeclared parameters fail validation."""

        model_data = copy.deepcopy(self.model_data)
        model_data["cmb"] = {
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
            "perturbations": self._make_declared_perturbations(),
        }
        with self.assertRaises(ValueError):
            model_plugin_validation.build_plugin(model_data, self.funcs)

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
        posterior = model_plugin_validation.make_logposterior(like, priors)

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
        posterior = model_plugin_validation.make_logposterior(like, priors)

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


class FrozenMappingTests(unittest.TestCase):
    """Validate the FrozenMapping wrapper used across ModelPlugin fields."""

    def test_model_plugin_pickles_with_frozen_mappings(self) -> None:
        """ModelPlugin should survive pickle round-trips under spawn pools."""

        plugin = _build_sample_plugin()
        with multiprocessing_module.get_context("spawn").Pool(1) as pool:
            is_frozen, custom_value, fixed_h0 = pool.apply(
                _inspect_plugin,
                (plugin,),
            )

        self.assertIsInstance(
            plugin.extras,
            model_plugin_validation.FrozenMapping,
        )
        self.assertTrue(is_frozen)
        self.assertEqual(custom_value, "extra")
        self.assertAlmostEqual(fixed_h0, 70.0)

    def test_frozen_mapping_to_dict_returns_copy(self) -> None:
        """The FrozenMapping copy helper must not expose internal state."""

        plugin = _build_sample_plugin()
        extras_copy = plugin.extras.to_dict()
        extras_copy["custom_extra"] = "shadowed"
        self.assertEqual(plugin.extras["custom_extra"](), "extra")


class DeclaredLCDMModelTestCase(unittest.TestCase):
    """Verify that the declared LambdaCDM file reaches the declared solver."""

    @staticmethod
    def _build_plugin(model_name: str = "model_lcdm.yml"):
        """Build the declared model through the repository validation path."""

        model_path = (
            Path(__file__).resolve().parents[3]
            / "copernican"
            / "models"
            / model_name
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                model_path,
                cache_dir,
            )
            functions, model_data = model_coder.generate_callables(cache_path)
        plugin = model_plugin_validation.build_plugin(model_data, functions)
        plugin.MODEL_FILENAME = model_path.name
        return plugin

    def test_usmf2_declared_route_is_promoted(self) -> None:
        """USMF2 must expose its declared graph as a declared CMB route."""

        plugin = self._build_plugin("model_usmf2.yml")

        self.assertTrue(plugin.valid_for_cmb)
        self.assertIsNotNone(plugin.CMB_DECLARED_RUNTIME)
        self.assertIsNotNone(plugin.CMB_PERTURBATION_DATA)
        self.assertEqual(
            plugin.get_cmb_perturbation_data(
                plugin.INITIAL_GUESSES
            ).manifest_summary["execution_route"]["solver_id"],
            CCMBS_ID,
        )

    @staticmethod
    def _build_low_resolution_usmf2_plugin(eta_sample_count: int = 32):
        """Build USMF2 with a small deterministic grid for contract tests."""

        repo_root = Path(__file__).resolve().parents[3]
        source_path = repo_root / "copernican" / "models" / "model_usmf2.yml"
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
                    "k_sample_count": 3,
                    "eta_sample_count": eta_sample_count,
                    "source_grid_multiplier": 1,
                    "a_min": 1.0e-2,
                    "initial_redshift": 99.0,
                }
            )
        model_data["cmb"]["perturbations"]["accuracy_controls"][
            "scalar_reference_ells"
        ] = [2, 20]
        model_data["cmb"]["perturbations"]["accuracy_controls"][
            "minimum_k_sample_count"
        ] = 1
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
        plugin = model_plugin_validation.build_plugin(parsed, functions)
        plugin.MODEL_FILENAME = source_path.name
        return plugin

    def test_usmf2_declared_spectra_are_finite_and_responsive(self) -> None:
        """USMF2 spectra must be finite and respond to a model parameter."""

        plugin = self._build_low_resolution_usmf2_plugin()
        ell_grid = numpy.asarray([2, 8, 20], dtype=int)
        baseline = cmb.compute_cmb_spectrum_cached(
            plugin,
            plugin.INITIAL_GUESSES,
            ell_grid,
            spectra=("TT", "TE", "EE", "BB", "PP", "TP", "EP"),
        )
        changed = list(plugin.INITIAL_GUESSES)
        changed[1] += 0.1
        response = cmb.compute_cmb_spectrum_cached(
            plugin,
            tuple(changed),
            ell_grid,
            spectra=("TT", "TE", "EE", "BB", "PP", "TP", "EP"),
        )
        self.assertEqual(
            set(baseline),
            {"TT", "TE", "EE", "BB", "PP", "TP", "EP"},
        )
        for values in baseline.values():
            self.assertEqual(values.shape, ell_grid.shape)
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertFalse(
            numpy.allclose(
                baseline["TT"],
                response["TT"],
                rtol=1.0e-12,
                atol=0.0,
            )
        )

    def test_usmf2_default_declared_mode_is_finite(self) -> None:
        """The declared default history must support a finite TT mode."""

        plugin = self._build_plugin("model_usmf2.yml")
        spectra = cmb.compute_cmb_spectrum_cached(
            plugin,
            plugin.INITIAL_GUESSES,
            numpy.asarray([2], dtype=int),
            spectra=("TT",),
        )

        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_usmf2_declared_history_converges_on_declared_grid(self) -> None:
        """USMF2 transfer spectra must agree across declared resolutions."""

        ell_grid = numpy.asarray([2, 8, 20], dtype=int)
        coarse_plugin = self._build_low_resolution_usmf2_plugin(
            eta_sample_count=16
        )
        coarse = cmb.compute_cmb_spectrum_cached(
            coarse_plugin,
            coarse_plugin.INITIAL_GUESSES,
            ell_grid,
            spectra=("TT", "TE", "EE"),
        )
        reference_plugin = self._build_low_resolution_usmf2_plugin(
            eta_sample_count=32
        )
        reference = cmb.compute_cmb_spectrum_cached(
            reference_plugin,
            reference_plugin.INITIAL_GUESSES,
            ell_grid,
            spectra=("TT", "TE", "EE"),
        )

        for spectrum_name in ("TT", "TE", "EE"):
            numpy.testing.assert_allclose(
                coarse[spectrum_name],
                reference[spectrum_name],
                rtol=1.0e-8,
                atol=1.0e-12,
            )

    def test_declared_lcdm_declares_compiled_scalar_graph(self) -> None:
        """The artifact must compile a declared graph with declared outputs."""

        plugin = self._build_plugin()
        perturbation_data = plugin.get_cmb_perturbation_data(
            plugin.INITIAL_GUESSES
        )
        summary = perturbation_data.manifest_summary
        route = summary["execution_route"]

        self.assertFalse(hasattr(plugin, "CMB_PERTURBATION_STANDARD"))
        self.assertEqual(route["solver_id"], CCMBS_ID)
        self.assertTrue(route["ready"])
        self.assertIn("evolve_theta_gamma0", summary["equation_names"])
        self.assertIn(
            "adiabatic_scalar", summary["initial_condition_family_names"]
        )
        self.assertEqual(
            set(summary["angular_power_spectrum_targets"]),
            {"TT", "TE", "EE", "BB", "PP", "TP", "EP"},
        )

    def test_declared_lcdm_declared_spectra_are_finite(self) -> None:
        """Declared LCDM spectra must execute without an external solver."""

        plugin = self._build_plugin()
        ell_grid = numpy.asarray([2, 10, 20], dtype=int)
        spectra = cmb.compute_cmb_spectrum_cached(
            plugin,
            plugin.INITIAL_GUESSES,
            ell_grid,
            spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )

        self.assertEqual(
            set(spectra),
            {"TT", "TE", "EE", "PP", "TP", "EP"},
        )
        for values in spectra.values():
            self.assertEqual(values.shape, ell_grid.shape)
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_lcdm_and_torg_enforce_reference_scalar_constraints(self) -> None:
        """Reference histories retain declared scalar closure evidence."""

        for model_name in ("model_lcdm.yml", "model_torg.yml"):
            plugin = self._build_plugin(model_name)
            boundary_parameters = list(plugin.INITIAL_GUESSES)
            boundary_parameters[0] = 50.0
            for point_name, parameters in (
                ("interior", plugin.INITIAL_GUESSES),
                ("lower_hubble_boundary", tuple(boundary_parameters)),
            ):
                with self.subTest(
                    model_name=model_name,
                    point=point_name,
                ):
                    cache.clear_cmb_caches()
                    spectrum_data = (
                        projection._compute_custom_cmb_spectrum_data(
                            plugin.get_cmb_declared_runtime(parameters),
                            numpy.asarray((2, 10, 20), dtype=int),
                            requested_spectra=("TT",),
                        )
                    )
                    self.assertTrue(
                        numpy.all(
                            numpy.isfinite(spectrum_data.spectra["TT"])
                        )
                    )
                    diagnostics = spectrum_data.runtime_envelope[
                        "scalar_constraint_diagnostics"
                    ]
                    self.assertEqual(
                        set(diagnostics),
                        {
                            "einstein_energy_residual",
                            "einstein_momentum_residual",
                            "einstein_shear_residual",
                        },
                    )
                    for metrics in diagnostics.values():
                        self.assertTrue(metrics["enforced"])
                        self.assertTrue(metrics["reference_resolution_met"])
                        self.assertEqual(
                            metrics["resolution_status"],
                            "reference",
                        )
                        self.assertEqual(
                            metrics["physical_judgement"],
                            "evaluated",
                        )
                        self.assertEqual(
                            metrics["normalization_source"],
                            "sum_abs_declared_einstein_terms",
                        )
                        self.assertEqual(
                            metrics["tolerance_kind"],
                            "normalized",
                        )
                        self.assertEqual(
                            metrics["tolerance_source"],
                            "accuracy_controls.scalar_constraint_tolerances",
                        )
                        self.assertGreaterEqual(
                            float(metrics["maximum_eta"]),
                            0.0,
                        )
                        self.assertTrue(metrics["normalization_terms"])
                        self.assertLessEqual(
                            float(metrics["maximum_normalized"]),
                            float(metrics["tolerance"]),
                        )
                    reconstruction = spectrum_data.runtime_envelope[
                        "scalar_constraint_projection"
                    ]
                    self.assertEqual(
                        reconstruction["method"],
                        "source_history_coupled_einstein_reconstruction",
                    )
                    self.assertEqual(int(reconstruction["mode_count"]), 0)
                    self.assertFalse(
                        spectrum_data.runtime_envelope[
                            "source_history_reconstruction_enabled"
                        ]
                    )
                    self.assertTrue(
                        spectrum_data.runtime_envelope[
                            "source_history_reconstruction_diagnostic_only"
                        ]
                    )

    def test_declared_lcdm_full_spectrum_returns_finite_spectra(
        self,
    ) -> None:
        """A full declared LCDM request must return finite spectra."""

        plugin = self._build_plugin()
        ell_grid = numpy.arange(2, 2001, dtype=int)
        spectra = cmb.compute_cmb_spectrum_cached(
            plugin,
            plugin.INITIAL_GUESSES,
            ell_grid,
            spectra=("TT", "TE", "EE", "PP"),
        )
        self.assertEqual(
            set(spectra),
            {"TT", "TE", "EE", "PP"},
        )
        for values in spectra.values():
            self.assertEqual(values.shape, ell_grid.shape)
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_declared_lcdm_manifest_records_provenance(self) -> None:
        """A run manifest must identify declared execution and controls."""

        plugin = self._build_plugin()
        manifest = run_manifest.build_manifest(
            [(plugin, "1.0"), (plugin, "1.0")],
            SimpleNamespace(__name__="declared_test", SAMPLER_VERSION="test"),
            [],
        )
        model_entry = manifest["cmb"]["models"][0]
        route = model_entry["declared_cmb_execution"]

        self.assertEqual(
            model_entry["execution_solver"],
            CCMBS_ID,
        )
        self.assertEqual(route["solver_id"], CCMBS_ID)
        self.assertTrue(route["ready"])
        self.assertEqual(
            model_entry["declared_cmb_numerical_settings"]["ell_max"],
            2500,
        )
        numerical_envelope = model_entry["declared_cmb_numerical_envelope"]
        self.assertEqual(numerical_envelope["accuracy_tier"], "final")
        self.assertTrue(numerical_envelope["bounded"])
        self.assertEqual(
            numerical_envelope["numerical_controls"][
                "lensing_sampling_factor"
            ],
            1.4,
        )
        self.assertEqual(
            set(numerical_envelope["hierarchy_controls"]),
            {
                "massless_neutrino",
                "photon_polarization",
                "photon_temperature",
            },
        )
        self.assertEqual(
            model_entry["declared_cmb_runtime_manifest_summary"][
                "compile_diagnostics"
            ]["compiler"],
            "copernican.lib.model_coder.compile_declared_cmb_runtime",
        )

    def test_declared_lcdm_declares_converged_scalar_transfer_grid(
        self,
    ) -> None:
        """Production LCDM must not use the under-resolved 18-node grid."""

        plugin = self._build_plugin()
        runtime = plugin.get_cmb_declared_runtime(plugin.INITIAL_GUESSES)
        perturbation_data = runtime["perturbation_data"]
        controls = perturbation_data.accuracy_controls
        numerics = perturbation_data.numerics

        self.assertEqual(int(controls["minimum_k_sample_count"]), 64)
        self.assertGreaterEqual(int(numerics["k_sample_count"]), 64)


if __name__ == "__main__":
    unittest.main()
# fmt: on
