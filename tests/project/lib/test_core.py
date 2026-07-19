# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Basic functional tests for the Copernican Suite."""

import importlib
import json
import os
import subprocess  # nosec B404
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import camb
import numpy
import pandas

import copernican.engines.engine_mcmc as engine
import copernican.lib.dataset_registry as dataset_registry
import copernican.lib.engine_adapter as engine_plugin_validation
import copernican.lib.model_coder as model_coder
import copernican.lib.model_spec_validator as model_spec_validator
from copernican.lib.likelihoods.cmb import camb_solver
from copernican.lib.run_pipeline import extract_cosmological_param_vector
from tests.project import filesystem_helpers

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("VIRTUAL_ENV", str(REPO_ROOT / ".venv"))

# Ensure compound BAO parser registration without requiring package installs.
importlib.import_module(
    "copernican.datasets.bao.compound.cosmo_parser_compound"
)


class FunctionalTestCase(unittest.TestCase):
    """Run a minimal end-to-end check of the public API."""

    @classmethod
    def setUpClass(cls):
        """Prepare a validated reference plugin used by several tests."""
        # Prepare a validated reference plugin used by several tests.
        base = Path(__file__).resolve().parents[3]
        models_dir = base / "copernican" / "models"
        yaml_path = models_dir / "model_lcdm.yml"
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                yaml_path, cache_dir
            )
            funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_plugin_validation.build_plugin(parsed, funcs)
        engine_plugin_validation.validate_plugin(cls.plugin)

    def test_plugin_validation(self):
        """Ensure the constructed plugin exposes the expected API."""
        self.assertTrue(hasattr(self.plugin, "distance_modulus_model"))

    def test_engine_routines(self):
        """Run a smoke test across the main engine routines."""
        sne_df = dataset_registry.load_sne_data("jla_2014")
        self.assertIsNotNone(sne_df)
        sne_df = sne_df.head(3)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :3, :3
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:3]

        bao_df = dataset_registry.load_bao_data("compound_bao_set")
        self.assertIsNotNone(bao_df)
        bao_df = bao_df.head(3)

        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        self.assertIsNotNone(cmb_df)

        params = self.plugin.INITIAL_GUESSES
        chi2_sne = engine.chi_squared_sne(
            params, self.plugin.distance_modulus_model, sne_df
        )
        self.assertTrue(numpy.isfinite(chi2_sne))

        pred_df, rs_mpc, _ = engine.calculate_bao_observables(
            bao_df, self.plugin, params
        )
        redshifts_array = bao_df["redshift"].to_numpy(dtype=float)
        observable_types_array = bao_df["observable_type"].to_numpy()
        observable_values_array = bao_df["value"].to_numpy(dtype=float)
        observable_errors_array = bao_df["error"].to_numpy(dtype=float)
        cov_inv = bao_df.attrs.get("covariance_matrix_inv")
        chi2_bao = engine.chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            params,
            rs_mpc,
            covariance_matrix_inv=cov_inv,
        )
        self.assertTrue(numpy.isfinite(chi2_bao))

        camb_params = self.plugin.get_camb_contract(params)
        camb_params["perturbations"] = (
            self.plugin.get_cmb_perturbation_contract(params)
        )
        chi2_cmb = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        spec = engine.compute_cmb_spectrum(
            camb_params, cmb_df["ell"].values, spectra=("TT", "TE", "EE")
        )
        self.assertTrue(numpy.isfinite(chi2_cmb))
        self.assertIn("TT", spec)
        self.assertEqual(len(spec["TT"]), len(cmb_df))

    def test_mcmc_fit_returns_expected_fields(self):
        """Return posterior diagnostics and χ² totals.

        The MCMC engine should report a finite total and component
        breakdown.
        """
        sne_df = dataset_registry.load_sne_data("jla_2014").head(3)
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            attrs["covariance_matrix_inv"] = attrs["covariance_matrix_inv"][
                :3, :3
            ]
            attrs["diag_errors_for_plot"] = attrs["diag_errors_for_plot"][:3]
        result = engine.fit_cosmology_parameters(
            sne_df,
            self.plugin,
            n_walkers=6,
            n_steps=8,
            pool_size=1,
            burn_in_steps=20,
        )
        self.assertTrue(result["success"])
        self.assertIn("samples", result)
        self.assertIn("chi2_total", result)
        self.assertTrue(numpy.isfinite(result["chi2_total"]))
        components = result.get("chi2_components", {})
        self.assertAlmostEqual(
            result["chi2_total"], sum(components.values()), places=7
        )
        self.assertAlmostEqual(result["chi2_sne"], components.get("sne", 0.0))
        self.assertAlmostEqual(
            result.get("chi2_bao", 0.0), components.get("bao", 0.0)
        )
        self.assertAlmostEqual(
            result.get("chi2_cmb", 0.0), components.get("cmb", 0.0)
        )
        self.assertIn("burn_in_steps", result)
        self.assertIn("production_steps", result)
        self.assertEqual(result["burn_in_steps"], 20)

    def test_chi_squared_cmb_planck2018lite(self):
        """Verify that the Planck 2018 lite dataset yields finite χ²."""
        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        params = self.plugin.INITIAL_GUESSES
        chi2 = engine.chi_squared_cmb(params, cmb_df, self.plugin)
        self.assertTrue(numpy.isfinite(chi2))

    def test_chi_squared_sne_invalid_data(self):
        """chi_squared_sne should return inf when data is invalid."""
        bad_df = pandas.DataFrame(
            {
                "zcmb": [0.1, numpy.nan],
                "mu_obs": [33.1, 34.5],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        with self.assertLogs(level="ERROR") as captured_logs:
            chi2 = engine.chi_squared_sne(
                self.plugin.INITIAL_GUESSES,
                self.plugin.distance_modulus_model,
                bad_df,
            )
        self.assertTrue(numpy.isinf(chi2))
        self.assertIn(
            "non-finite zcmb or mu_obs", "".join(captured_logs.output)
        )

    def test_cmb_spectrum_is_d_ell(self):
        """Ensure the CAMB adapter returns spectra in Dl convention."""
        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        ells = cmb_df["ell"].values[:5]
        camb_params = self.plugin.get_camb_contract(
            self.plugin.INITIAL_GUESSES
        )
        result = camb_solver.compute_cmb_spectrum_from_camb_contract(
            camb_params,
            ells,
        )

        # Mirror the adapter's CAMB parameter construction so the comparison
        # exercises the same neutrino-sector mapping used by the reference.
        params = camb_solver._make_camb_params(
            camb_params,
            lmax=int(numpy.max(ells)),
        )
        params.InitPower.set_params(
            As=camb_params["param_map"]["As"],
            ns=camb_params["param_map"]["ns"],
        )
        ref = camb.get_results(params).get_unlensed_scalar_cls(
            lmax=int(numpy.max(ells)), CMB_unit="muK"
        )
        numpy.testing.assert_allclose(result, ref[:, 0][ells], rtol=1e-7)

    def test_engine_metadata_constants(self):
        """Expose human-readable descriptors for UI logging."""
        self.assertTrue(hasattr(engine, "ENGINE_KIND"))
        self.assertEqual(engine.ENGINE_KIND, "mcmc")
        self.assertTrue(hasattr(engine, "ENGINE_LABEL"))
        self.assertIn("MCMC", engine.ENGINE_LABEL)

    def test_installed_package_native_cmb_smoke(self):
        """A target install should run one native declared-graph spectrum."""

        contract = {
            "model_name": "InstalledSmokeNativeCMB",
            "backend": "camb",
            "param_map": {
                "H0": 67.4,
                "ombh2": 0.02237,
                "omch2": 0.12,
                "tau": 0.054,
                "As": 2.1e-9,
                "ns": 0.965,
                "Neff": 3.046,
            },
            "model_parameters": {
                "Tcmb_K": 2.7255,
                "YHe": 0.245,
                "source_scale": 1.0,
                "closure_scale": 1.0,
                "decay_rate": 0.02,
            },
            "background": {
                "derived": {
                    "h": "H0 / 100.0",
                    "Omega_k0": "0.0",
                    "Omega_b0": "ombh2 / (h * h)",
                    "Omega_c0": "omch2 / (h * h)",
                    "Omega_gamma0": (
                        "2.469e-5 * ((Tcmb_K / 2.7255) ** 4) / (h * h)"
                    ),
                    "Omega_nu0": "0.22710731766 * Neff * Omega_gamma0",
                    "Omega_r0": "Omega_gamma0 + Omega_nu0",
                    "Omega_m0": "Omega_b0 + Omega_c0",
                    "Omega_de0": "1.0 - Omega_m0 - Omega_r0 - Omega_k0",
                    "H": (
                        "H0 * sqrt("
                        "Omega_r0 / (a ** 4) + "
                        "Omega_m0 / (a ** 3) + "
                        "Omega_de0"
                        ")"
                    ),
                },
                "reionization": {
                    "calibration": {
                        "symbol": "reionization_log10_amplitude",
                        "target_optical_depth": "tau",
                        "lower": -24.0,
                        "upper": 32.0,
                    },
                    "quantities": {
                        "hydrogen_temperature_K": "1.0e4",
                        "helium_temperature_K": "1.0e4",
                        "helium_double_temperature_K": "2.0e4",
                        "hydrogen_ionization_rate": (
                            "(10 ** reionization_log10_amplitude) * H_SI"
                        ),
                        "helium_ionization_rate": (
                            "0.5 * hydrogen_ionization_rate"
                        ),
                        "helium_double_ionization_rate": (
                            "0.25 * hydrogen_ionization_rate"
                        ),
                    },
                },
            },
            "grids": {},
            "values": {},
            "calls": [],
            "numerical": {
                "ell_min": 20,
                "ell_max": 40,
                "k_min": 1.0e-4,
                "k_max": 0.08,
                "k_sample_count": 6,
                "eta_sample_count": 128,
                "source_grid_multiplier": 1,
                "ode_rtol": 1.0e-5,
                "ode_atol": 1.0e-8,
                "tight_coupling_ratio": 50.0,
                "a_min": 1.0e-6,
                "initial_redshift": 2.0e4,
            },
            "perturbations": {
                "contract_version": 2,
                "standard": False,
                "gauge": "conformal_newtonian",
                "variables": {
                    "signal_mode": {
                        "kind": "photon_temperature_monopole",
                        "tensor_character": "scalar_like",
                    },
                },
                "derived": {
                    "closure_drive": {
                        "expression": "closure_scale * signal_mode",
                    },
                },
                "equations": {
                    "evolve_signal_mode": {
                        "lhs": {
                            "kind": "derivative",
                            "variable": "signal_mode",
                            "wrt": "tau",
                            "order": 1,
                        },
                        "rhs": "-decay_rate * signal_mode",
                        "role": "continuity",
                    },
                },
                "constraints": {},
                "closures": {},
                "sources": {
                    "signal_source": {
                        "expression": "source_scale * closure_drive",
                        "role": "signal",
                    },
                },
                "observables": {
                    "signal_transfer": {
                        "kind": "transfer_component",
                        "projection": "line_of_sight_signal",
                        "source_terms": {"signal": "signal_source"},
                    },
                    "TT": {
                        "kind": "angular_power_spectrum",
                        "primary": "signal_transfer",
                        "secondary": "signal_transfer",
                    },
                },
                "initial_conditions": {
                    "signal_seed": {
                        "target": {
                            "variable": "signal_mode",
                            "wrt": "tau",
                            "order": 0,
                        },
                        "expression": "seed",
                    },
                },
                "boundary_conditions": {},
                "validity": {
                    "regimes": ["installed_smoke"],
                },
                "backend_mapping": {
                    "camb": {
                        "implemented": True,
                        "native_solver_required": True,
                    }
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_root = filesystem_helpers.stage_repo_snapshot(
                REPO_ROOT,
                tmp_path,
            )
            target_dir = tmp_path / "site-packages"
            install_result = subprocess.run(  # nosec B603
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--target",
                    str(target_dir),
                    str(source_root),
                ],
                check=False,
                cwd=source_root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                install_result.returncode,
                0,
                install_result.stderr,
            )

            smoke_script = (
                "import json\n"
                "import numpy\n"
                "from copernican.lib import model_coder\n"
                "from copernican.lib.likelihoods import cmb\n"
                f"contract = json.loads({json.dumps(json.dumps(contract))})\n"
                "prepared = model_coder.prepare_native_cmb_execution_contract("
                "contract)\n"
                "ells = numpy.arange(20, 25, dtype=int)\n"
                "spectra = cmb.compute_cmb_spectrum_from_contract("
                "prepared, ells, spectra=('TT',))\n"
                "values = numpy.asarray(spectra, dtype=float)\n"
                "assert values.shape == (ells.size,)\n"
                "assert numpy.all(numpy.isfinite(values))\n"
            )
            environment = dict(os.environ)
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(target_dir), existing_pythonpath) if part
            )
            smoke_result = subprocess.run(  # nosec B603
                [sys.executable, "-c", smoke_script],
                check=False,
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                smoke_result.returncode,
                0,
                smoke_result.stderr,
            )


class PlotterUtilTestCase(unittest.TestCase):
    """Test helper utilities in ``plotter``."""

    def test_wrap_math_removes_size_macros(self):
        """Ensure size macros are stripped when wrapping math expressions."""
        from copernican.lib import plotter

        expr = r"\mu(z) = 5\log_{10}\bigl[d_L(z)/\mathrm{Mpc}\bigr] + 25"
        expected = r"$\mu(z) = 5\log_{10}[d_L(z)/{Mpc}] + 25$"
        self.assertEqual(plotter._wrap_math(expr), expected)

    def test_latex_utils_conversions(self):
        """Ensure LaTeX mappings are applied consistently."""
        from copernican.lib import latex_utils

        self.assertEqual(latex_utils.sanitize_name(r"\alpha_1"), "alpha_1")
        self.assertEqual(
            latex_utils.latex_to_sympy(r"\frac{1}{\infty}"), "(1)/(sympy.oo)"
        )
        self.assertEqual(
            latex_utils.latex_to_unicode(r"\Omega_{gamma}"),
            "Ωᵧ",
        )
        self.assertEqual(latex_utils.latex_to_unicode("H_0"), "H₀")
        self.assertEqual(
            latex_utils.latex_to_unicode(r"z_{\rm rec}"),
            "zᵣₑc",
        )
        self.assertEqual(
            latex_utils.latex_to_unicode("x_{(1+2)}"),
            "x₍₁₊₂₎",
        )
        self.assertEqual(latex_utils.latex_to_unicode("y^{*}"), "y⁎")


class CosmologicalParameterHelperTestCase(unittest.TestCase):
    """Validate parameter extraction fallbacks in ``copernican``."""

    def test_returns_ordered_values_when_available(self):
        """Preserve plugin parameter order when extraction succeeds."""

        plugin = SimpleNamespace(
            MODEL_NAME="Toy", PARAMETER_NAMES=["H0", "Om_m"]
        )
        fit_results = {
            "success": True,
            "fitted_cosmological_params": {
                "H0": 71.0,
                "Om_m": 0.3,
                "unused": 1.0,
            },
        }
        vector = extract_cosmological_param_vector(fit_results, plugin)
        self.assertEqual(vector, [71.0, 0.3])

    def test_returns_none_when_required_value_missing(self):
        """Refuse to extract vectors with incomplete parameter coverage."""

        plugin = SimpleNamespace(
            MODEL_NAME="Toy", PARAMETER_NAMES=["H0", "Om_m"]
        )
        fit_results = {
            "success": True,
            "fitted_cosmological_params": {"H0": 71.0},
        }
        vector = extract_cosmological_param_vector(fit_results, plugin)
        self.assertIsNone(vector)

    def test_returns_none_when_fit_unsuccessful(self):
        """Avoid leaking stale parameters when the fit itself failed."""

        plugin = SimpleNamespace(MODEL_NAME="Toy", PARAMETER_NAMES=["H0"])
        fit_results = {
            "success": False,
            "fitted_cosmological_params": {"H0": 71.0},
        }
        vector = extract_cosmological_param_vector(fit_results, plugin)
        self.assertIsNone(vector)


if __name__ == "__main__":
    unittest.main()
