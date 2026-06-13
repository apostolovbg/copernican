"""Physics tests for the declared-graph CMB likelihood helpers."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest import mock

import numpy

try:
    import camb
except ImportError:  # pragma: no cover - optional external reference
    camb = None

from copernican.lib.likelihoods import cmb


def _declared_graph_perturbations(
    *,
    baryon_rhs: str = "-0.018 * delta_b + 0.008 * theta_b + 0.004 * Phi",
    photon_monopole_rhs: str = (
        "-0.025 * theta_gamma0 + 0.012 * theta_gamma1 + 0.005 * Phi"
    ),
    metric_closure_expression: str = "Phi",
    additive_source_expression: str = "0.0",
    include_bb: bool = False,
    include_lensing: bool = False,
) -> dict[str, object]:
    """Return a complete synthetic declared-math CMB graph."""

    perturbations: dict[str, object] = {
        "contract_version": 2,
        "standard": False,
        "gauge": "conformal_newtonian",
        "variables": {
            "theta_gamma0": {
                "kind": "photon_temperature_monopole",
                "tensor_character": "scalar_like",
            },
            "theta_gamma1": {
                "kind": "photon_temperature_dipole",
                "tensor_character": "scalar_like",
            },
            "theta_gamma2": {
                "kind": "photon_temperature_quadrupole",
                "tensor_character": "scalar_like",
            },
            "e_gamma2": {
                "kind": "photon_polarization_quadrupole",
                "tensor_character": "scalar_like",
            },
            "delta_b": {"kind": "baryon_density_contrast"},
            "theta_b": {"kind": "baryon_velocity_divergence"},
            "delta_c": {"kind": "cdm_density_contrast"},
            "theta_c": {"kind": "cdm_velocity_divergence"},
            "delta_nu": {
                "kind": "massless_neutrino_density_contrast",
            },
            "theta_nu": {
                "kind": "massless_neutrino_velocity_divergence",
            },
            "sigma_nu": {
                "kind": "massless_neutrino_anisotropic_stress",
            },
            "Phi": {
                "kind": "metric_potential_phi",
                "gauge_role": "newtonian_potential",
            },
            "Psi": {
                "kind": "metric_potential_psi",
                "gauge_role": "curvature_potential",
            },
        },
        "derived": {
            "temperature_drive": {
                "expression": "theta_gamma0 + Phi",
                "description": "Synthetic temperature drive.",
            }
        },
        "equations": {
            "evolve_theta_gamma0": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma0",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": photon_monopole_rhs,
                "role": "continuity",
            },
            "evolve_theta_gamma1": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma1",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "0.018 * theta_gamma0 - 0.022 * theta_gamma1 "
                    "+ 0.005 * Psi"
                ),
                "role": "euler",
            },
            "evolve_theta_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "0.009 * theta_gamma1 - 0.021 * theta_gamma2",
                "role": "hierarchy",
            },
            "evolve_e_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "0.006 * theta_gamma2 - 0.019 * e_gamma2",
                "role": "polarization",
            },
            "evolve_delta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": baryon_rhs,
                "role": "continuity",
            },
            "evolve_theta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-0.016 * theta_b + 0.007 * theta_gamma1 + 0.004 * Psi",
                "role": "euler",
            },
            "evolve_delta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-0.014 * delta_c + 0.006 * theta_c + 0.003 * Phi",
                "role": "continuity",
            },
            "evolve_theta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-0.018 * theta_c + 0.003 * Psi",
                "role": "euler",
            },
            "evolve_delta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-0.012 * delta_nu + 0.006 * theta_nu + 0.003 * Phi",
                "role": "continuity",
            },
            "evolve_theta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "0.008 * delta_nu - 0.017 * theta_nu "
                    "- 0.005 * sigma_nu + 0.004 * Psi"
                ),
                "role": "euler",
            },
            "evolve_sigma_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "0.004 * theta_nu - 0.018 * sigma_nu",
                "role": "hierarchy",
            },
        },
        "constraints": {
            "phi_constraint": {
                "target": "Phi",
                "expression": (
                    "0.08 * theta_gamma0 + 0.04 * delta_b "
                    "+ 0.03 * delta_c + 0.02 * delta_nu"
                ),
                "role": "constraint",
            }
        },
        "closures": {
            "psi_closure": {
                "target": "Psi",
                "expression": metric_closure_expression,
                "role": "closure",
            }
        },
        "sources": {
            "temperature_monopole": {
                "expression": (
                    "visibility * (theta_gamma0 + Psi "
                    "+ 0.25 * (theta_gamma2 + e_gamma2))"
                ),
                "role": "monopole",
            },
            "temperature_doppler": {
                "expression": "visibility * 3.0 * theta_gamma1",
                "role": "doppler",
            },
            "temperature_isw": {
                "expression": "exp(-tau) * (Psi - Phi)",
                "role": "isw",
            },
            "polarization_source": {
                "expression": "0.75 * visibility * (theta_gamma2 + e_gamma2)",
                "role": "polarization",
            },
            "temperature_additive": {
                "expression": additive_source_expression,
                "role": "additive",
            },
        },
        "observables": {
            "temperature": {
                "kind": "transfer_component",
                "projection": "cmb_temperature_scalar",
                "source_terms": {
                    "monopole": "temperature_monopole",
                    "doppler": "temperature_doppler",
                    "isw": "temperature_isw",
                    "additive": "temperature_additive",
                },
            },
            "polarization_e": {
                "kind": "transfer_component",
                "projection": "cmb_polarization_e_scalar",
                "source_terms": {"polarization": "polarization_source"},
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
            "EE": {
                "kind": "angular_power_spectrum",
                "primary": "polarization_e",
                "secondary": "polarization_e",
            },
        },
        "initial_conditions": {
            "theta_gamma0_seed": {
                "target": {
                    "variable": "theta_gamma0",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-0.5 * seed",
            },
            "theta_gamma1_seed": {
                "target": {
                    "variable": "theta_gamma1",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.1 * k * seed",
            },
            "theta_gamma2_seed": {
                "target": {
                    "variable": "theta_gamma2",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.01 * k * seed",
            },
            "e_gamma2_seed": {
                "target": {
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.005 * k * seed",
            },
            "delta_b_seed": {
                "target": {
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-0.75 * seed",
            },
            "theta_b_seed": {
                "target": {
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.3 * k * seed",
            },
            "delta_c_seed": {
                "target": {
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-0.75 * seed",
            },
            "theta_c_seed": {
                "target": {
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "delta_nu_seed": {
                "target": {
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-0.5 * seed",
            },
            "theta_nu_seed": {
                "target": {
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.2 * k * seed",
            },
            "sigma_nu_seed": {
                "target": {
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.05 * k * k * seed",
            },
        },
        "boundary_conditions": {},
        "numerics": {
            "ell_min": 20,
            "ell_max": 90,
            "k_min": 1.0e-4,
            "k_max": 0.3,
            "k_sample_count": 10,
            "eta_sample_count": 224,
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
            "tight_coupling_ratio": 50.0,
            "a_min": 1.0e-8,
            "source_grid_multiplier": 1,
        },
        "validity": {
            "regimes": ["linear", "scalar_like"],
            "notes": "Synthetic declared graph for runtime tests.",
        },
        "backend_mapping": {
            "camb": {
                "native_solver_required": True,
                "implemented": True,
            }
        },
    }
    if include_bb:
        perturbations["observables"]["BB"] = {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "polarization_e",
        }
    if include_lensing:
        perturbations["sources"]["lensing_potential"] = {
            "expression": "exp(-tau) * Phi",
            "role": "potential",
        }
        perturbations["observables"]["lensing_potential"] = {
            "kind": "transfer_component",
            "projection": "cmb_lensing_potential_scalar",
            "source_terms": {"potential": "lensing_potential"},
        }
        perturbations["observables"]["PP"] = {
            "kind": "angular_power_spectrum",
            "primary": "lensing_potential",
            "secondary": "lensing_potential",
        }
    return perturbations


def _base_custom_cmb_contract(
    **perturbation_kwargs: object,
) -> dict[str, object]:
    """Return a synthetic non-standard CMB contract used by the tests."""

    return {
        "model_name": "SyntheticCustomCMB",
        "backend": "camb",
        "param_map": {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
            "z_rec": 1090.0,
        },
        "grids": {},
        "values": {},
        "calls": [],
        "perturbations": _declared_graph_perturbations(
            **perturbation_kwargs,
        ),
        "numerical": {
            "ell_min": 20,
            "ell_max": 90,
            "k_min": 1.0e-4,
            "k_max": 0.3,
            "k_sample_count": 10,
            "eta_sample_count": 224,
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
            "tight_coupling_ratio": 50.0,
            "a_min": 1.0e-8,
            "source_grid_multiplier": 1,
        },
    }


def _base_standard_cmb_contract() -> dict[str, object]:
    """Return a standard CAMB contract used for reference comparisons."""

    return {
        "model_name": "SyntheticLCDM",
        "backend": "camb",
        "param_map": {
            "H0": 67.4,
            "ombh2": 0.02237,
            "omch2": 0.12,
            "tau": 0.054,
            "As": 2.1e-9,
            "ns": 0.965,
            "Neff": 3.046,
            "YHe": 0.245,
        },
        "grids": {},
        "values": {},
        "calls": [],
        "perturbations": {
            "contract_version": 2,
            "standard": True,
            "gauge": "unspecified",
            "variables": {},
            "derived": {},
            "equations": {},
            "constraints": {},
            "closures": {},
            "sources": {},
            "observables": {},
            "initial_conditions": {},
            "boundary_conditions": {},
            "validity": {
                "regimes": ["standard_camb"],
                "notes": "Uses backend standard perturbations.",
            },
            "backend_mapping": {
                "camb": {
                    "uses_standard_perturbations": True,
                }
            },
        },
    }


def _custom_contract(**perturbation_kwargs: object) -> dict[str, object]:
    """Return a deep-copied non-standard CMB fixture."""

    return copy.deepcopy(_base_custom_cmb_contract(**perturbation_kwargs))


def _standard_contract() -> dict[str, object]:
    """Return a deep-copied standard CAMB fixture."""

    return copy.deepcopy(_base_standard_cmb_contract())


def _custom_perturbations(**perturbation_kwargs: object) -> dict[str, object]:
    """Return the non-standard perturbation graph from the fixture."""

    return copy.deepcopy(
        _base_custom_cmb_contract(**perturbation_kwargs)["perturbations"]
    )


def _strip_perturbations(contract: dict[str, object]) -> dict[str, object]:
    """Return ``contract`` without the nested perturbation declaration."""

    stripped = copy.deepcopy(contract)
    stripped.pop("perturbations", None)
    return stripped


class _CustomCMBPlugin:
    """Plugin stub that exposes the synthetic declared graph fixture."""

    INITIAL_GUESSES = (
        67.4,
        0.02237,
        0.12,
        0.054,
        2.1e-9,
        0.965,
        3.046,
        0.245,
        1090.0,
    )

    def get_camb_contract(self, _params):
        """Return the structured CAMB contract used by the helper."""

        return _strip_perturbations(_custom_contract())

    def get_cmb_perturbation_contract(self, _params):
        """Return the synthetic non-standard perturbation graph."""

        return _custom_perturbations()


class CMBCustomPhysicsTestCase(unittest.TestCase):
    """Validate the declared-graph CMB engine."""

    def test_source_file_does_not_contain_fake_or_legacy_hacks(self) -> None:
        """The production module should not contain old compatibility code."""

        source_text = Path(cmb.__file__).read_text(encoding="utf-8")
        for needle in (
            "equation_mode",
            "mapped_sector",
            "declared_equations",
            "source_normalization",
            "transfer_amplitude",
            "_evolve_custom_cmb_mode_histories",
            "_CUSTOM_CMB_SOURCE_CHANNELS",
            "_CUSTOM_CMB_SECTOR_ALIASES",
            "_classify_custom_physical_sector",
            "visibility shift",
            "visibility rescale",
        ):
            self.assertNotIn(needle, source_text)

    def test_custom_background_matches_camb_recombination_reference(
        self,
    ) -> None:
        """The custom background should match CAMB recombination references."""

        if camb is None:
            self.skipTest("CAMB is not installed")

        contract = _custom_contract()
        physical = cmb._resolve_custom_cmb_physical_parameters(contract)
        numerics = cmb._resolve_custom_cmb_numerics(contract)
        background = cmb._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        reference_contract = _strip_perturbations(contract)
        reference_contract["param_map"].pop("z_rec", None)
        params = cmb._make_camb_params(reference_contract, lmax=32)
        results = camb.get_results(params)
        reference = results.get_background_time_evolution(
            background.eta_grid,
            vars=["x_e", "visibility", "opacity"],
            format="dict",
        )

        reference_peak_eta = float(results.tau_maxvis)
        reference_peak_z = float(
            results.redshift_at_conformal_time(reference_peak_eta)
        )
        reference_eta0 = float(results.conformal_time(0.0))
        reference_sound_horizon = float(
            results.sound_horizon(reference_peak_z)
        )
        reference_x_e = numpy.asarray(reference["x_e"], dtype=float)
        reference_visibility = numpy.asarray(
            reference["visibility"],
            dtype=float,
        )

        peak_index = int(numpy.argmax(background.visibility_grid))
        peak_z = float(background.z_grid[peak_index])
        median_relative_x_e_error = float(
            numpy.median(
                numpy.abs(background.x_e_grid - reference_x_e)
                / numpy.maximum(reference_x_e, 1.0e-8)
            )
        )

        self.assertTrue(numpy.all(numpy.isfinite(background.x_e_grid)))
        self.assertTrue(numpy.all(background.x_e_grid >= 0.0))
        self.assertTrue(numpy.all(background.x_e_grid <= 4.0))
        self.assertTrue(numpy.all(numpy.isfinite(reference_x_e)))
        self.assertTrue(numpy.all(numpy.isfinite(reference_visibility)))
        self.assertTrue(numpy.all(numpy.diff(background.tau_grid) <= 1.0e-8))
        self.assertLess(
            abs(peak_z - reference_peak_z) / reference_peak_z,
            0.01,
        )
        self.assertLess(
            abs(background.eta0 - reference_eta0) / reference_eta0,
            0.005,
        )
        self.assertLess(
            abs(background.sound_horizon_mpc - reference_sound_horizon)
            / reference_sound_horizon,
            0.005,
        )
        self.assertLess(median_relative_x_e_error, 0.18)
        self.assertLess(
            abs(background.reionization_tau - physical.tau_reio)
            / max(physical.tau_reio, 1.0e-12),
            0.05,
        )

    def test_custom_graph_runs_and_transfer_payloads_are_finite(self) -> None:
        """Transfer components and declared spectra should stay finite."""

        contract = _custom_contract()
        ells = numpy.arange(20, 45, dtype=int)
        spectrum_data = cmb._compute_custom_cmb_spectrum_data(contract, ells)

        self.assertIsInstance(spectrum_data, cmb.CustomCMBSpectrumData)
        self.assertTrue(numpy.array_equal(spectrum_data.ell_grid, ells))
        self.assertEqual(
            spectrum_data.transfer_components["temperature"].shape,
            (ells.size, spectrum_data.k_grid.size),
        )
        self.assertEqual(
            spectrum_data.transfer_components["polarization_e"].shape,
            (ells.size, spectrum_data.k_grid.size),
        )
        self.assertEqual(set(spectrum_data.spectra), {"TT", "TE", "EE"})
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.Delta_l_T,
                spectrum_data.transfer_components["temperature"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.Delta_l_E,
                spectrum_data.transfer_components["polarization_e"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.C_l_TT,
                spectrum_data.spectra["TT"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.C_l_TE,
                spectrum_data.spectra["TE"],
            )
        )
        self.assertTrue(
            numpy.array_equal(
                spectrum_data.C_l_EE,
                spectrum_data.spectra["EE"],
            )
        )
        for array in (
            spectrum_data.k_grid,
            spectrum_data.transfer_components["temperature"],
            spectrum_data.transfer_components["polarization_e"],
            spectrum_data.spectra["TT"],
            spectrum_data.spectra["TE"],
            spectrum_data.spectra["EE"],
        ):
            self.assertTrue(numpy.all(numpy.isfinite(array)))

    def test_custom_spectra_have_structure_and_parameter_response(
        self,
    ) -> None:
        """Declared spectra should be finite, structured, and responsive."""

        contract = _custom_contract()
        ells = numpy.arange(20, 90, dtype=int)
        base = cmb.compute_cmb_spectrum_from_dict(
            contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )
        hi_as_contract = _custom_contract()
        hi_as_contract["param_map"]["As"] = 4.2e-9
        hi_as = cmb.compute_cmb_spectrum_from_dict(
            hi_as_contract,
            ells,
            spectra=("TT",),
        )
        hi_h0_contract = _custom_contract()
        hi_h0_contract["param_map"]["H0"] = 74.0
        hi_h0 = cmb.compute_cmb_spectrum_from_dict(
            hi_h0_contract,
            ells,
            spectra=("TT",),
        )

        base_tt = numpy.asarray(base["TT"], dtype=float)
        base_te = numpy.asarray(base["TE"], dtype=float)
        base_ee = numpy.asarray(base["EE"], dtype=float)
        hi_as_tt = numpy.asarray(hi_as, dtype=float)
        hi_h0_tt = numpy.asarray(hi_h0, dtype=float)

        self.assertTrue(numpy.all(numpy.isfinite(base_tt)))
        self.assertTrue(numpy.all(numpy.isfinite(base_te)))
        self.assertTrue(numpy.all(numpy.isfinite(base_ee)))
        self.assertGreater(numpy.max(base_tt) - numpy.min(base_tt), 0.0)
        self.assertTrue(numpy.any(base_te < 0.0))
        self.assertTrue(numpy.any(base_te > 0.0))
        self.assertTrue(numpy.all(base_ee >= 0.0))
        self.assertAlmostEqual(
            float(numpy.mean(hi_as_tt / base_tt)),
            2.0,
            places=2,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(hi_h0_tt - base_tt))),
            0.0,
        )

    def test_custom_equations_change_spectrum(self) -> None:
        """Equation changes should move the requested observables."""

        baseline = _custom_contract()
        changed = _custom_contract(
            baryon_rhs="-0.012 * delta_b + 0.016 * theta_b + 0.012 * Phi"
        )
        ells = numpy.arange(20, 70, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_dict(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_dict(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(changed_tt - baseline_tt))),
            1.0e-12,
        )

    def test_custom_closures_change_spectrum(self) -> None:
        """Closure changes should alter TE or EE."""

        baseline = _custom_contract(metric_closure_expression="Phi")
        changed = _custom_contract(metric_closure_expression="1.15 * Phi")
        ells = numpy.arange(20, 70, dtype=int)
        baseline_spectra = cmb.compute_cmb_spectrum_from_dict(
            baseline,
            ells,
            spectra=("TE", "EE"),
        )
        changed_spectra = cmb.compute_cmb_spectrum_from_dict(
            changed,
            ells,
            spectra=("TE", "EE"),
        )
        te_delta = numpy.asarray(
            changed_spectra["TE"] - baseline_spectra["TE"],
            dtype=float,
        )
        ee_delta = numpy.asarray(
            changed_spectra["EE"] - baseline_spectra["EE"],
            dtype=float,
        )
        self.assertGreater(
            max(
                float(numpy.max(numpy.abs(te_delta))),
                float(numpy.max(numpy.abs(ee_delta))),
            ),
            1.0e-12,
        )

    def test_custom_source_expression_changes_observable(self) -> None:
        """Observable mappings should consume declared graph quantities."""

        baseline = _custom_contract(additive_source_expression="0.0")
        changed = _custom_contract(
            additive_source_expression="0.2 * theta_gamma0 + Phi"
        )
        ells = numpy.arange(20, 70, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_dict(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_dict(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(changed_tt - baseline_tt))),
            1.0e-12,
        )

    def test_bb_and_lensing_targets_run_when_declared(self) -> None:
        """Additional observable targets should run through the graph."""

        contract = _custom_contract(include_bb=True, include_lensing=True)
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_dict(
            contract,
            ells,
            spectra=("TT", "BB", "PP"),
        )

        self.assertEqual(set(spectra), {"TT", "BB", "PP"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_missing_initial_conditions_fail_loudly(self) -> None:
        """Missing initial conditions should fail before evolution."""

        contract = _custom_contract()
        del contract["perturbations"]["initial_conditions"]["theta_b_seed"]
        with self.assertRaisesRegex(
            ValueError,
            "missing required initial conditions",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_missing_observable_mappings_fail_loudly(self) -> None:
        """Missing observable mappings should fail clearly."""

        contract = _custom_contract()
        contract["perturbations"]["observables"] = {}
        with self.assertRaisesRegex(ValueError, "must declare observables"):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_unsupported_projection_fails_loudly(self) -> None:
        """Unsupported projections should be rejected at runtime."""

        contract = _custom_contract()
        contract["perturbations"]["observables"]["temperature"][
            "projection"
        ] = "bogus_projection"
        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_nonfinite_expression_results_fail_loudly(self) -> None:
        """Non-finite source expressions should fail clearly."""

        contract = _custom_contract()
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "sqrt(-1)"
        with self.assertRaisesRegex(
            ValueError,
            "Declared source term produced non-finite values",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_nonfinite_evolution_states_fail_loudly(self) -> None:
        """Non-finite evolution states should fail clearly."""

        contract = _custom_contract()
        theta_b_equation = contract["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        theta_b_equation["rhs"] = "1.0 / (delta_b - delta_b)"
        with self.assertRaisesRegex(
            ValueError,
            "must be finite",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_custom_cached_path_does_not_call_camb(self) -> None:
        """The cached plugin route should not use the standard CAMB path."""

        plugin = _CustomCMBPlugin()
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch.object(
            cmb,
            "_compute_cmb_spectrum_direct",
            side_effect=AssertionError("standard CAMB path should not run"),
        ):
            with mock.patch.object(
                cmb.camb,
                "get_results",
                side_effect=AssertionError(
                    "CAMB prediction path should not run"
                ),
            ):
                result = cmb.compute_cmb_spectrum_cached(
                    plugin,
                    plugin.INITIAL_GUESSES,
                    ells,
                    spectra=("TT", "TE", "EE"),
                )

        self.assertEqual(set(result), {"TT", "TE", "EE"})
        for spectrum in result.values():
            self.assertTrue(numpy.all(numpy.isfinite(spectrum)))
            self.assertEqual(spectrum.shape, (ells.size,))

    def test_standard_lcdm_matches_camb_when_available(self) -> None:
        """The standard path should stay aligned with CAMB."""

        if camb is None:
            self.skipTest("CAMB is not installed")

        standard_contract = _standard_contract()
        ells = numpy.arange(2, 35, dtype=int)
        actual = cmb.compute_cmb_spectrum_from_dict(
            standard_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )

        params = cmb._make_camb_params(standard_contract, lmax=int(ells.max()))
        results = camb.get_results(params)
        reference = results.get_unlensed_scalar_cls(
            lmax=int(ells.max()),
            CMB_unit="muK",
        )

        numpy.testing.assert_allclose(
            actual["TT"],
            reference[:, 0][ells],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        numpy.testing.assert_allclose(
            actual["EE"],
            reference[:, 1][ells],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        numpy.testing.assert_allclose(
            actual["TE"],
            reference[:, 3][ells],
            rtol=1.0e-5,
            atol=1.0e-5,
        )


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the public CMB helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        """The module should export the expected public helpers."""

        self.assertTrue(hasattr(cmb, "CMBLike"))
        self.assertTrue(callable(cmb.compute_cmb_spectrum))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_cached))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_from_dict))
        self.assertTrue(callable(cmb.compute_camb_background_observables))
        self.assertTrue(
            callable(cmb.compute_cmb_spectrum_from_legacy_params_for_tests)
        )
        self.assertTrue(callable(cmb.describe_camb_configuration))
        self.assertTrue(callable(cmb._CustomCMBBackgroundData.sample))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        """The likelihood protocol symbols should remain available."""

        self.assertTrue(callable(cmb.CMBLike.loglike))
        self.assertTrue(hasattr(cmb.CMBLike.state, "__get__"))


if __name__ == "__main__":
    unittest.main()
