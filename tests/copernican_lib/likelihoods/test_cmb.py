"""Unit tests for CAMB-backed CMB helpers."""

from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path
from unittest import mock

import camb
import numpy
import pandas

from copernican_lib import engine_adapter as engine_plugin_validation
from copernican_lib import model_coder, model_spec_validator
from copernican_lib.likelihoods import cmb


class _FakeAccuracy:
    """Lightweight accuracy container used by CAMB adapter tests."""

    def __init__(self) -> None:
        self.AccuracyBoost = 1.0
        self.LAccuracyBoost = 1.0
        self.KAccuracyBoost = 1.0


class _FakeInitPower:
    """Minimal power-spectrum helper used by CAMB adapter tests."""

    def __init__(self) -> None:
        self.kwargs: dict[str, float] | None = None

    def set_params(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)


class _FakeCAMBParams:
    """Capture CAMB parameter calls without invoking the real backend."""

    def __init__(self) -> None:
        self.Accuracy = _FakeAccuracy()
        self.InitPower = _FakeInitPower()
        self.cosmology_kwargs: dict[str, object] | None = None
        self.dark_energy_kwargs: dict[str, object] | None = None
        self.dark_energy_w_a_kwargs: dict[str, object] | None = None
        self.lmax_args: tuple[int, dict[str, object]] | None = None

    def set_cosmology(self, **kwargs) -> None:
        self.cosmology_kwargs = dict(kwargs)

    def set_dark_energy(self, **kwargs) -> None:
        self.dark_energy_kwargs = dict(kwargs)

    def set_dark_energy_w_a(self, **kwargs) -> None:
        self.dark_energy_w_a_kwargs = dict(kwargs)

    def set_for_lmax(self, lmax, **kwargs) -> None:
        self.lmax_args = (int(lmax), dict(kwargs))


class CMBBackgroundTestCase(unittest.TestCase):
    """Validate CAMB background helpers share settings with the spectra API."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare a reference plugin for evaluating CAMB helpers."""

        repo_root = Path(__file__).resolve().parents[3]
        os.environ.setdefault("VIRTUAL_ENV", str(repo_root / ".venv"))
        yaml_path = repo_root / "models" / "cosmo_model_lcdm.yml"
        cache_dir = repo_root / "models" / "cache"
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_plugin_validation.build_plugin(parsed, funcs)

    def test_background_observables_match_input_length(self) -> None:
        """Background helper should return one entry per requested redshift."""

        params = self.plugin.get_camb_contract(self.plugin.INITIAL_GUESSES)
        redshifts = numpy.array([0.15, 0.35, 0.57])
        background = cmb.compute_camb_background_observables(params, redshifts)

        self.assertEqual(background["DM"].shape, redshifts.shape)
        self.assertEqual(background["DH"].shape, redshifts.shape)
        self.assertEqual(background["DV"].shape, redshifts.shape)
        self.assertGreater(background["rs_drag"], 0.0)
        self.assertTrue(numpy.all(numpy.isfinite(background["DM"])))

    def test_background_cache_collapses_duplicate_redshifts(self) -> None:
        """Repeated redshifts should produce identical background distances."""

        params = self.plugin.get_camb_contract(self.plugin.INITIAL_GUESSES)
        redshifts = numpy.array([0.35, 0.35, 0.60])
        background = cmb.compute_camb_background_observables(params, redshifts)

        self.assertAlmostEqual(
            background["DM"][0], background["DM"][1], places=12
        )
        self.assertAlmostEqual(
            background["DH"][0], background["DH"][1], places=12
        )

    def test_neutrino_configuration_matches_direct_camb(self) -> None:
        """Neutrino sector parameters should propagate unchanged to CAMB."""

        custom_params = {
            "H0": 68.0,
            "ombh2": 0.023,
            "omch2": 0.118,
            "tau": 0.059,
            "As": 2.05e-9,
            "ns": 0.964,
            "Neff": 3.55,
            "standard_neutrino_neff": 3.044,
            "num_massive_neutrinos": 2,
            "mnu1": 0.06,
            "mnu2": 0.12,
            "neutrino_hierarchy": "normal",
        }
        ell_range = numpy.arange(2, 51, dtype=int)
        redshifts = numpy.array([0.15, 0.60, 1.0])

        helper_background = cmb.compute_camb_background_observables(
            {
                "backend": "camb",
                "param_map": custom_params,
                "grids": {},
                "values": {},
                "calls": [],
            },
            redshifts,
        )
        helper_cls = cmb.compute_cmb_spectrum_from_legacy_params_for_tests(
            custom_params, ell_range, spectra=("TT", "EE", "TE")
        )

        manual = camb.CAMBparams()
        # Direct CAMB configuration mirroring the helper inputs.  This ensures
        # the regression asserts against the canonical API rather than the
        # helper internals.
        manual.set_cosmology(
            H0=custom_params["H0"],
            ombh2=custom_params["ombh2"],
            omch2=custom_params["omch2"],
            tau=custom_params["tau"],
            nnu=custom_params["Neff"],
            standard_neutrino_neff=custom_params["standard_neutrino_neff"],
            num_massive_neutrinos=custom_params["num_massive_neutrinos"],
            mnu=custom_params["mnu1"] + custom_params["mnu2"],
            neutrino_hierarchy=custom_params["neutrino_hierarchy"],
        )
        manual.set_for_lmax(
            int(ell_range.max()) + cmb._LMAX_PADDING,
            lens_potential_accuracy=0,
        )
        manual.InitPower.set_params(
            As=custom_params["As"], ns=custom_params["ns"]
        )

        manual_results = camb.get_results(manual)
        manual_cls = manual_results.get_unlensed_scalar_cls(
            lmax=int(ell_range.max()), CMB_unit="muK"
        )

        for column, spectrum in zip(
            (0, 1, 3), ("TT", "EE", "TE"), strict=True
        ):
            numpy.testing.assert_allclose(
                helper_cls[spectrum],
                manual_cls[:, column][ell_range],
                rtol=1e-7,
                atol=1e-7,
            )

        manual_background = {}
        # Derived parameters such as ``rdrag`` live on the CAMB results object,
        # so we fetch them explicitly to mirror the helper outputs.
        manual_background["rs_drag"] = float(
            manual_results.get_derived_params().get("rdrag")
        )
        manual_background["DM"] = numpy.asarray(
            [
                manual_results.comoving_radial_distance(float(z))
                for z in redshifts
            ]
        )
        manual_background["DA"] = numpy.asarray(
            [
                manual_results.angular_diameter_distance(float(z))
                for z in redshifts
            ]
        )
        manual_background["Hz"] = numpy.asarray(
            [manual_results.hubble_parameter(float(z)) for z in redshifts]
        )
        manual_background["DH"] = numpy.where(
            numpy.abs(manual_background["Hz"]) > 1e-12,
            cmb._C_LIGHT_KM_S / manual_background["Hz"],
            numpy.nan,
        )
        manual_background["DV"] = numpy.full_like(
            redshifts, numpy.nan, dtype=float
        )
        term = manual_background["DM"] * manual_background["DM"]
        term *= redshifts
        term *= manual_background["DH"]
        mask = numpy.isfinite(term) & (term >= 0.0)
        manual_background["DV"][mask] = numpy.power(term[mask], 1.0 / 3.0)
        manual_background["DV"][redshifts == 0.0] = 0.0

        numpy.testing.assert_allclose(
            helper_background["rs_drag"],
            manual_background["rs_drag"],
            rtol=5e-6,
            atol=1e-10,
        )
        for key in ("DM", "DA", "DH", "DV", "Hz"):
            numpy.testing.assert_allclose(
                helper_background[key],
                manual_background[key],
                rtol=1e-8,
                atol=1e-8,
            )

    def test_make_camb_params_reaches_set_dark_energy(self) -> None:
        """Structured contracts should call CAMB dark-energy hooks."""

        fake_params = _FakeCAMBParams()
        contract = {
            "backend": "camb",
            "param_map": {
                "H0": 68.0,
                "ombh2": 0.022,
                "omch2": 0.12,
                "tau": 0.054,
                "As": 2.1e-9,
                "ns": 0.965,
            },
            "grids": {},
            "values": {},
            "calls": [
                {
                    "method": "set_dark_energy",
                    "kwargs": {
                        "w0": -0.95,
                        "wa": 0.1,
                        "cs2": 1.0,
                        "dark_energy_model": "ppf",
                    },
                }
            ],
        }
        with mock.patch(
            "copernican_lib.likelihoods.cmb.camb.CAMBparams",
            return_value=fake_params,
        ):
            returned = cmb._make_camb_params(contract, lmax=12)

        self.assertIs(returned, fake_params)
        self.assertEqual(fake_params.dark_energy_kwargs["w"], -0.95)
        self.assertEqual(fake_params.dark_energy_kwargs["wa"], 0.1)
        self.assertEqual(fake_params.dark_energy_kwargs["cs2"], 1.0)
        self.assertEqual(
            fake_params.dark_energy_kwargs["dark_energy_model"],
            "ppf",
        )

    def test_make_camb_params_reaches_set_dark_energy_w_a(self) -> None:
        """Structured contracts should forward arrays to CAMB."""

        fake_params = _FakeCAMBParams()
        scale_factors = numpy.array([0.1, 0.4, 0.7, 1.0], dtype=float)
        equation_of_state = numpy.array([-1.0, -0.95, -0.9, -0.85])
        contract = {
            "backend": "camb",
            "param_map": {
                "H0": 68.0,
                "ombh2": 0.022,
                "omch2": 0.12,
                "tau": 0.054,
                "As": 2.1e-9,
                "ns": 0.965,
            },
            "grids": {},
            "values": {},
            "calls": [
                {
                    "method": "set_dark_energy_w_a",
                    "args": {
                        "a": scale_factors,
                        "w": equation_of_state,
                    },
                    "kwargs": {"dark_energy_model": "ppf"},
                }
            ],
        }
        with mock.patch(
            "copernican_lib.likelihoods.cmb.camb.CAMBparams",
            return_value=fake_params,
        ):
            returned = cmb._make_camb_params(contract, lmax=12)

        self.assertIs(returned, fake_params)
        self.assertIsInstance(
            fake_params.dark_energy_w_a_kwargs["a"], numpy.ndarray
        )
        self.assertIsInstance(
            fake_params.dark_energy_w_a_kwargs["w"], numpy.ndarray
        )
        numpy.testing.assert_allclose(
            fake_params.dark_energy_w_a_kwargs["a"], scale_factors
        )
        numpy.testing.assert_allclose(
            fake_params.dark_energy_w_a_kwargs["w"],
            equation_of_state,
        )
        self.assertEqual(
            fake_params.dark_energy_w_a_kwargs["dark_energy_model"],
            "ppf",
        )

    def test_cmb_loglike_preserves_structured_contract(self) -> None:
        """The CMB likelihood should pass the structured contract through."""

        captured: dict[str, object] = {}

        def fake_compute(contract, ells, *, spectra=("TT",)):
            captured["contract"] = contract
            captured["ells"] = tuple(ells)
            captured["spectra"] = tuple(spectra)
            return numpy.ones(len(tuple(ells)), dtype=float)

        class StructuredPlugin:
            """Plugin stub returning a structured CAMB contract."""

            def get_camb_contract(self, _params):
                return {
                    "model_name": "StructuredModel",
                    "backend": "camb",
                    "param_map": {
                        "H0": 68.0,
                        "ombh2": 0.022,
                        "omch2": 0.12,
                    },
                    "grids": {
                        "a_grid": {
                            "symbol": "a",
                            "lower": 0.1,
                            "upper": 1.0,
                            "points": 4,
                            "spacing": "linear",
                        }
                    },
                    "values": {
                        "w_tog": {
                            "grid": "a_grid",
                            "expression": "-1",
                        }
                    },
                    "calls": [
                        {
                            "method": "set_dark_energy_w_a",
                            "args": {
                                "a": numpy.array(
                                    [0.1, 0.4, 0.7, 1.0], dtype=float
                                ),
                                "w": numpy.array(
                                    [-1.0, -0.95, -0.9, -0.85],
                                    dtype=float,
                                ),
                            },
                            "kwargs": {
                                "dark_energy_model": "ppf",
                            },
                        }
                    ],
                }

            def get_cmb_perturbation_contract(self, _params):
                return {
                    "model_name": "StructuredModel",
                    "backend": "camb",
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
                }

        cmb_df = pandas.DataFrame(
            {"ell": [2, 3, 4], "Dl_obs": [1.0, 1.0, 1.0]}
        )
        cmb_df.attrs["covariance_matrix_inv"] = numpy.eye(3)
        with mock.patch(
            "copernican_lib.likelihoods.cmb.compute_cmb_spectrum_from_dict",
            side_effect=fake_compute,
        ):
            like = cmb.CMBLike(cmb_df, StructuredPlugin())
            loglike = like.loglike([68.0])

        self.assertTrue(numpy.isfinite(loglike))
        self.assertEqual(captured["spectra"], ("TT",))
        contract = captured["contract"]
        self.assertIsInstance(contract, dict)
        self.assertEqual(
            contract["calls"][0]["kwargs"]["dark_energy_model"], "ppf"
        )
        self.assertIsInstance(contract["calls"][0]["args"]["a"], numpy.ndarray)
        self.assertIsInstance(contract["calls"][0]["args"]["w"], numpy.ndarray)

    def test_cmb_loglike_rejects_unsupported_nonstandard_perturbations(self):
        """The CMB likelihood should reject unsupported perturbations."""

        class NonStandardPlugin:
            """Plugin stub for an unsupported non-standard perturbation set."""

            def get_camb_contract(self, _params):
                return {
                    "model_name": "NonStandardModel",
                    "backend": "camb",
                    "param_map": {
                        "H0": 68.0,
                        "ombh2": 0.022,
                        "omch2": 0.12,
                    },
                    "grids": {},
                    "values": {},
                    "calls": [],
                }

            def get_cmb_perturbation_contract(self, _params):
                return {
                    "model_name": "NonStandardModel",
                    "backend": "camb",
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
                        }
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
                        "regimes": ["linear"],
                        "notes": "Declared for setup failure coverage.",
                    },
                    "backend_mapping": {
                        "camb": {
                            "native_solver_required": True,
                            "implemented": False,
                        }
                    },
                    "notes": (
                        "Native perturbation mathematics are declared but "
                        "unsupported by CAMB."
                    ),
                }

        cmb_df = pandas.DataFrame(
            {"ell": [2, 3, 4], "Dl_obs": [1.0, 1.0, 1.0]}
        )
        cmb_df.attrs["covariance_matrix_inv"] = numpy.eye(3)
        like = cmb.CMBLike(cmb_df, NonStandardPlugin())
        self.assertEqual(like.loglike([68.0]), float("-inf"))

    def test_cmb_cached_spectrum_uses_generic_nonstandard_executor(self):
        """Structured non-standard models should use the generic executor."""

        class GenericNonStandardPlugin:
            """Plugin stub for a supported non-standard perturbation set."""

            PARAMETER_NAMES = (
                "hubble_constant",
                "baryon_density_h2",
                "cdm_density_h2",
            )
            INITIAL_GUESSES = (68.0, 0.022, 0.0)

            def get_camb_contract(self, values):
                hubble_constant, baryon_density_h2, cdm_density_h2 = values
                return {
                    "model_name": "GenericNonStandardModel",
                    "backend": "camb",
                    "param_map": {
                        "hubble_constant": hubble_constant,
                        "baryon_density_h2": baryon_density_h2,
                        "cdm_density_h2": cdm_density_h2,
                        "tau": 0.054,
                        "As": 2.1e-09,
                        "ns": 0.965,
                        "z_rec": 1089.92,
                    },
                    "grids": {},
                    "values": {},
                    "calls": [],
                    "model_parameters": {
                        "hubble_constant": hubble_constant,
                        "baryon_density_h2": baryon_density_h2,
                        "cdm_density_h2": cdm_density_h2,
                    },
                    "value_definitions": {},
                }

            def get_cmb_perturbation_contract(self, _values):
                return {
                    "model_name": "GenericNonStandardModel",
                    "backend": "camb",
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
                        "tensor_x": {
                            "kind": "tensor_mode",
                            "description": "Example tensor perturbation.",
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
                        }
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
                        },
                        "tensor_wave": {
                            "expression": "tensor_x",
                        },
                    },
                    "validity": {
                        "regimes": ["linear"],
                        "notes": "Declared for generic executor coverage.",
                    },
                    "backend_mapping": {
                        "camb": {
                            "native_solver_required": True,
                            "implemented": True,
                        }
                    },
                    "notes": (
                        "Native perturbation mathematics are declared and "
                        "supported by the generic executor."
                    ),
                }

            def get_Hz_per_Mpc(
                self,
                redshift,
                hubble_constant,
                baryon_density_h2,
                cdm_density_h2,
            ):
                redshift_arr = numpy.asarray(redshift, dtype=float)
                del baryon_density_h2, cdm_density_h2
                return hubble_constant * numpy.sqrt(
                    0.3 * numpy.power(1.0 + redshift_arr, 3.0) + 0.7
                )

            def get_comoving_distance_Mpc(
                self,
                redshift,
                hubble_constant,
                baryon_density_h2,
                cdm_density_h2,
            ):
                redshift_arr = numpy.asarray(redshift, dtype=float)
                hubble = numpy.maximum(
                    self.get_Hz_per_Mpc(
                        redshift_arr,
                        hubble_constant,
                        baryon_density_h2,
                        cdm_density_h2,
                    ),
                    1.0e-12,
                )
                return 299792.458 * redshift_arr / hubble

            def get_angular_diameter_distance_Mpc(
                self,
                redshift,
                hubble_constant,
                baryon_density_h2,
                cdm_density_h2,
            ):
                redshift_arr = numpy.asarray(redshift, dtype=float)
                return self.get_comoving_distance_Mpc(
                    redshift_arr,
                    hubble_constant,
                    baryon_density_h2,
                    cdm_density_h2,
                ) / (1.0 + redshift_arr)

            def get_DV_Mpc(
                self,
                redshift,
                hubble_constant,
                baryon_density_h2,
                cdm_density_h2,
            ):
                redshift_arr = numpy.asarray(redshift, dtype=float)
                comoving = self.get_comoving_distance_Mpc(
                    redshift_arr,
                    hubble_constant,
                    baryon_density_h2,
                    cdm_density_h2,
                )
                hubble = numpy.maximum(
                    self.get_Hz_per_Mpc(
                        redshift_arr,
                        hubble_constant,
                        baryon_density_h2,
                        cdm_density_h2,
                    ),
                    1.0e-12,
                )
                term = comoving * comoving
                term *= 299792.458 * redshift_arr / hubble
                return numpy.power(term, 1.0 / 3.0)

            def get_sound_horizon_rs_Mpc(
                self,
                hubble_constant,
                baryon_density_h2,
                cdm_density_h2,
            ):
                del hubble_constant, baryon_density_h2, cdm_density_h2
                return 147.0

        with mock.patch(
            "copernican_lib.likelihoods.cmb._compute_cmb_spectrum_direct",
            side_effect=AssertionError("standard CAMB path should not run"),
        ):
            with mock.patch(
                "copernican_lib.likelihoods.cmb."
                "compute_camb_background_observables",
                side_effect=AssertionError(
                    "CAMB background path should not run"
                ),
            ):
                spectrum = cmb.compute_cmb_spectrum_cached(
                    GenericNonStandardPlugin(),
                    GenericNonStandardPlugin.INITIAL_GUESSES,
                    numpy.array([2, 3, 4]),
                )

        self.assertEqual(spectrum.shape, (3,))
        self.assertTrue(numpy.all(numpy.isfinite(spectrum)))

    def test_cmb_spectrum_from_dict_uses_generic_nonstandard_executor(self):
        """Structured non-standard contracts should execute without CAMB."""

        contract = {
            "model_name": "GenericNonStandardModel",
            "backend": "camb",
            "param_map": {
                "hubble_constant": 68.0,
                "baryon_density_h2": 0.022,
                "cdm_density_h2": 0.0,
                "tau": 0.054,
                "As": 2.1e-09,
                "ns": 0.965,
                "z_rec": 1089.92,
            },
            "grids": {},
            "values": {},
            "calls": [],
            "perturbations": {
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
                    "tensor_x": {
                        "kind": "tensor_mode",
                        "description": "Example tensor perturbation.",
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
                    }
                },
                "closures": {
                    "velocity_seed": {
                        "expression": "theta_x",
                        "equals": "1e-3",
                    },
                    "no_anisotropic_stress": {
                        "expression": "sigma_x",
                        "equals": "0",
                    },
                },
                "sources": {
                    "poisson": {
                        "expression": "delta_rho_eff + delta_x + theta_x",
                    },
                    "tensor_wave": {
                        "expression": "tensor_x",
                    },
                },
                "validity": {
                    "regimes": ["linear"],
                    "notes": "Declared for setup coverage.",
                },
                "backend_mapping": {
                    "camb": {
                        "native_solver_required": True,
                        "implemented": True,
                    }
                },
                "notes": (
                    "Native perturbation mathematics are declared and "
                    "supported by the generic executor."
                ),
            },
        }

        with mock.patch(
            "copernican_lib.likelihoods.cmb._compute_cmb_spectrum_direct",
            side_effect=AssertionError("standard CAMB path should not run"),
        ):
            with mock.patch(
                "copernican_lib.likelihoods.cmb."
                "compute_camb_background_observables",
                side_effect=AssertionError(
                    "CAMB background path should not run"
                ),
            ):
                spectrum = cmb.compute_cmb_spectrum_from_dict(
                    contract,
                    numpy.array([2, 3, 4]),
                    spectra=("TT", "TE", "EE", "BB"),
                )

        self.assertIsInstance(spectrum, dict)
        self.assertEqual(set(spectrum), {"TT", "TE", "EE", "BB"})
        for spec in ("TT", "TE", "EE", "BB"):
            self.assertEqual(spectrum[spec].shape, (3,))
            self.assertTrue(numpy.all(numpy.isfinite(spectrum[spec])))
        self.assertTrue(numpy.any(numpy.abs(spectrum["BB"]) > 0.0))

        changed_contract = copy.deepcopy(contract)
        changed_contract["perturbations"]["equations"]["continuity_x"][
            "rhs"
        ] = "-10 * theta_x + 30 * Phi_tau"
        changed_spectrum = cmb.compute_cmb_spectrum_from_dict(
            changed_contract,
            numpy.array([2, 3, 4]),
            spectra=("TT",),
        )
        self.assertFalse(
            numpy.allclose(
                spectrum["TT"],
                changed_spectrum,
                rtol=1e-8,
                atol=0.0,
            )
        )
        changed_tensor_contract = copy.deepcopy(contract)
        changed_tensor_contract["perturbations"]["sources"]["tensor_wave"][
            "expression"
        ] = "2 * tensor_x"
        changed_tensor_spectrum = cmb.compute_cmb_spectrum_from_dict(
            changed_tensor_contract,
            numpy.array([2, 3, 4]),
            spectra=("BB",),
        )
        self.assertFalse(
            numpy.allclose(
                spectrum["BB"],
                changed_tensor_spectrum,
                rtol=1e-8,
                atol=0.0,
            )
        )
        visibility_shifted_contract = copy.deepcopy(contract)
        visibility_shifted_contract["param_map"]["z_rec"] = 900.0
        visibility_shifted_spectrum = cmb.compute_cmb_spectrum_from_dict(
            visibility_shifted_contract,
            numpy.array([2, 3, 4]),
            spectra=("TT",),
        )
        self.assertFalse(
            numpy.allclose(
                spectrum["TT"],
                visibility_shifted_spectrum,
                rtol=1e-8,
                atol=0.0,
            )
        )

    def test_compute_cmb_spectrum_from_dict_rejects_flat_params(self) -> None:
        """Scientific spectrum helpers must reject flat parameter maps."""

        with self.assertRaises(ValueError):
            cmb.compute_cmb_spectrum_from_dict(
                {"H0": 68.0, "ombh2": 0.022},
                numpy.array([2, 3, 4]),
            )

    def test_legacy_helper_accepts_flat_params(self) -> None:
        """The legacy-only helper still accepts flat parameter maps."""

        spectrum = cmb.compute_cmb_spectrum_from_legacy_params_for_tests(
            {
                "H0": 68.0,
                "ombh2": 0.022,
                "omch2": 0.12,
                "tau": 0.054,
                "As": 2.1e-9,
                "ns": 0.965,
            },
            numpy.array([2, 3, 4]),
        )
        self.assertEqual(spectrum.shape, (3,))

    def test_fake_cmb_mode_still_works(self) -> None:
        """The fake-CMB path still returns a deterministic spectrum."""

        contract = {
            "backend": "camb",
            "param_map": {
                "H0": 68.0,
                "ombh2": 0.022,
                "omch2": 0.12,
            },
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
                "validity": {},
                "backend_mapping": {
                    "camb": {
                        "uses_standard_perturbations": True,
                    }
                },
            },
        }
        with mock.patch.dict(os.environ, {"COPERNICAN_FAKE_CMB": "1"}):
            spectrum = cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.array([2, 3, 4]),
                spectra=("TT",),
            )
        self.assertTrue(numpy.all(numpy.isfinite(spectrum)))
        self.assertEqual(spectrum.shape, (3,))


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the CMB helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(cmb, "CMBLike"))
        self.assertTrue(callable(cmb.compute_cmb_spectrum))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_cached))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_from_dict))
        self.assertTrue(
            callable(cmb.compute_cmb_spectrum_from_legacy_params_for_tests)
        )
        self.assertTrue(callable(cmb.describe_camb_configuration))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        loglike = cmb.CMBLike.loglike
        state = cmb.CMBLike.state
        self.assertTrue(callable(loglike))
        self.assertTrue(hasattr(state, "__get__"))


if __name__ == "__main__":
    unittest.main()
