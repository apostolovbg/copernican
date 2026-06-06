"""Physics tests for the CMB likelihood helpers."""

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


def _base_custom_cmb_contract() -> dict[str, object]:
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
        "perturbations": {
            "contract_version": 1,
            "standard": False,
            "equation_mode": "mapped_sector",
            "gauge": "conformal_newtonian",
            "variables": {
                "theta_gamma0": {"kind": "photon_temperature_monopole"},
                "theta_gamma1": {"kind": "photon_temperature_dipole"},
                "theta_gamma2": {"kind": "photon_temperature_quadrupole"},
                "e_gamma2": {"kind": "photon_polarization_quadrupole"},
                "delta_b": {"kind": "baryon_density_contrast"},
                "theta_b": {"kind": "baryon_velocity_divergence"},
                "delta_c": {"kind": "cdm_density_contrast"},
                "theta_c": {"kind": "cdm_velocity_divergence"},
                "delta_nu": {"kind": "massless_neutrino_density_contrast"},
                "theta_nu": {"kind": "massless_neutrino_velocity_divergence"},
                "sigma_nu": {"kind": "massless_neutrino_anisotropic_stress"},
                "Phi": {"kind": "metric_potential_phi"},
                "Psi": {"kind": "metric_potential_psi"},
            },
            "derived": {},
            "equations": {
                "evolve_delta_b": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "delta_b",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "-theta_b + 3 * Phi",
                }
            },
            "closures": {
                "metric_closure": {
                    "expression": "Psi - Phi",
                    "equals": "0",
                }
            },
            "sources": {
                "cmb_source": {
                    "channel": "temperature_additive",
                    "expression": "theta_gamma0 + theta_b + Phi + Psi",
                }
            },
            "validity": {
                "regimes": ["linear", "scalar"],
                "notes": "Synthetic non-standard scalar test fixture.",
            },
            "backend_mapping": {
                "camb": {
                    "native_solver_required": True,
                    "implemented": True,
                }
            },
            "notes": (
                "Synthetic non-standard scalar test fixture for the generic "
                "CMB engine."
            ),
        },
        "numerical": {
            "ell_min": 20,
            "ell_max": 90,
            "k_min": 1.0e-4,
            "k_max": 0.3,
            "k_sample_count": 24,
            "eta_sample_count": 256,
            "photon_hierarchy_l_max": 6,
            "neutrino_hierarchy_l_max": 6,
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
                "notes": "Uses backend standard perturbations.",
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


def _custom_contract() -> dict[str, object]:
    """Return a deep-copied custom CMB fixture."""

    return copy.deepcopy(_base_custom_cmb_contract())


def _standard_contract() -> dict[str, object]:
    """Return a deep-copied standard CAMB fixture."""

    return copy.deepcopy(_base_standard_cmb_contract())


def _custom_perturbations() -> dict[str, object]:
    """Return the custom perturbation contract from the fixture."""

    return copy.deepcopy(_base_custom_cmb_contract()["perturbations"])


def _declared_equation_perturbations(
    *,
    baryon_rhs: str = "-theta_b + 0.05 * Phi",
    photon_monopole_rhs: str = "-theta_gamma1",
    metric_closure_expression: str = "Psi - Phi",
    source_expression: str = "theta_gamma0 + theta_b + Phi + Psi",
    source_channel: str = "temperature_additive",
) -> dict[str, object]:
    """Return a full declared-equation perturbation contract."""

    return {
        "contract_version": 1,
        "standard": False,
        "equation_mode": "declared_equations",
        "gauge": "conformal_newtonian",
        "variables": {
            "theta_gamma0": {"kind": "photon_temperature_monopole"},
            "theta_gamma1": {"kind": "photon_temperature_dipole"},
            "theta_gamma2": {"kind": "photon_temperature_quadrupole"},
            "e_gamma2": {"kind": "photon_polarization_quadrupole"},
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
            "Phi": {"kind": "metric_potential_phi"},
            "Psi": {"kind": "metric_potential_psi"},
        },
        "derived": {},
        "equations": {
            "evolve_theta_gamma0": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma0",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": photon_monopole_rhs,
            },
            "evolve_theta_gamma1": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma1",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "k / 3 * (theta_gamma0 - 2 * theta_gamma2) + k * Psi "
                    "- (-tau_dot) * (theta_gamma1 - theta_b / (3 * k))"
                ),
            },
            "evolve_theta_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "k / 5 * (2 * theta_gamma1 - 3 * theta_gamma2) "
                    "- (-tau_dot) * theta_gamma2"
                ),
            },
            "evolve_e_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "k / 5 * (2 * theta_gamma1 - 3 * theta_gamma2) "
                    "- (-tau_dot) * e_gamma2 + 0.1 * theta_gamma2"
                ),
            },
            "evolve_delta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": baryon_rhs,
            },
            "evolve_theta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "-Hconf * theta_b + k ** 2 * Psi "
                    "+ (-tau_dot) * (theta_gamma1 - theta_b / (3 * k))"
                ),
            },
            "evolve_delta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_c + 0.25 * Phi",
            },
            "evolve_theta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-Hconf * theta_c + k ** 2 * Psi",
            },
            "evolve_delta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_nu + Phi",
            },
            "evolve_theta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "k / 3 * (delta_nu - 2 * sigma_nu) + k * Psi",
            },
            "evolve_sigma_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "k / 5 * (2 * theta_nu - 3 * sigma_nu) "
                    "- (-tau_dot) * sigma_nu"
                ),
            },
        },
        "closures": {
            "metric_closure": {
                "expression": metric_closure_expression,
                "equals": "0",
            }
        },
        "sources": {
            "cmb_source": {
                "channel": source_channel,
                "expression": source_expression,
            }
        },
        "validity": {
            "regimes": ["linear", "scalar"],
            "notes": "Synthetic non-standard scalar test fixture.",
        },
        "backend_mapping": {
            "camb": {
                "native_solver_required": True,
                "implemented": True,
            }
        },
        "notes": (
            "Synthetic non-standard scalar test fixture for the generic "
            "CMB engine."
        ),
    }


def _mapped_sector_only_contract() -> dict[str, object]:
    """Return a non-standard contract that uses built-in sector equations."""

    contract = _custom_contract()
    contract["perturbations"]["equation_mode"] = "mapped_sector"
    contract["perturbations"]["equations"] = {}
    return contract


def _declared_equation_contract(
    *,
    baryon_rhs: str = "-theta_b + 0.05 * Phi",
    photon_monopole_rhs: str = "-theta_gamma1",
    metric_closure_expression: str = "Psi - Phi",
    source_expression: str = "theta_gamma0 + theta_b + Phi + Psi",
    source_channel: str = "temperature_additive",
) -> dict[str, object]:
    """Return a contract that replaces the built-in sector equations."""

    contract = _custom_contract()
    contract["perturbations"] = _declared_equation_perturbations(
        baryon_rhs=baryon_rhs,
        photon_monopole_rhs=photon_monopole_rhs,
        metric_closure_expression=metric_closure_expression,
        source_expression=source_expression,
        source_channel=source_channel,
    )
    return contract


def _strip_perturbations(contract: dict[str, object]) -> dict[str, object]:
    """Return ``contract`` without the nested perturbation declaration."""

    stripped = copy.deepcopy(contract)
    stripped.pop("perturbations", None)
    return stripped


class _CustomCMBPlugin:
    """Plugin stub that exposes the synthetic custom CMB fixture."""

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
        """Return the synthetic non-standard perturbation contract."""

        return _custom_perturbations()


class CMBCustomPhysicsTestCase(unittest.TestCase):
    """Validate the non-standard scalar CMB engine."""

    def test_source_file_does_not_contain_fake_cmb_projection(self) -> None:
        """The production module should not contain the old fake path."""

        source_text = Path(cmb.__file__).read_text(encoding="utf-8")
        for needle in (
            "COPERNICAN_FAKE_CMB",
            "_FAKE_CMB_PROVIDER",
            "_FAKE_CMB_BASELINE",
            "_FAKE_CMB_OFFSET",
            "project_declared_perturbation_series",
            "damping template",
            "source amplitude",
        ):
            self.assertNotIn(needle, source_text)

    def test_custom_background_peaks_near_recombination(self) -> None:
        """The visibility function should peak near recombination."""

        contract = _custom_contract()
        physical = cmb._resolve_custom_cmb_physical_parameters(contract)
        numerics = cmb._resolve_custom_cmb_numerics(contract)
        background = cmb._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        sampled_background = background.sample(background.eta_rec)
        peak_index = int(numpy.argmax(background.visibility_grid))
        peak_z = float(background.z_grid[peak_index])

        self.assertGreater(background.visibility_grid.max(), 0.0)
        self.assertTrue(numpy.all(numpy.diff(background.tau_grid) <= 1.0e-8))
        self.assertLess(
            abs(peak_z - physical.z_rec) / physical.z_rec,
            0.15,
        )
        self.assertTrue(
            numpy.all(
                numpy.isfinite(
                    numpy.asarray(
                        sampled_background["visibility"],
                        dtype=float,
                    )
                )
            )
        )
        self.assertGreater(
            float(numpy.asarray(sampled_background["visibility"])),
            0.0,
        )

    def test_custom_transfer_outputs_are_finite(self) -> None:
        """Transfer-function and spectrum dataclasses should be finite."""

        contract = _custom_contract()
        ells = numpy.arange(20, 45, dtype=int)
        spectrum_data = cmb._compute_custom_cmb_spectrum_data(contract, ells)

        self.assertIsInstance(spectrum_data, cmb.CustomCMBSpectrumData)
        self.assertTrue(numpy.array_equal(spectrum_data.ell_grid, ells))
        self.assertEqual(
            spectrum_data.Delta_l_T.shape,
            (ells.size, spectrum_data.k_grid.size),
        )
        self.assertEqual(
            spectrum_data.Delta_l_E.shape,
            (ells.size, spectrum_data.k_grid.size),
        )
        self.assertEqual(spectrum_data.C_l_TT.shape, (ells.size,))
        self.assertEqual(spectrum_data.C_l_TE.shape, (ells.size,))
        self.assertEqual(spectrum_data.C_l_EE.shape, (ells.size,))
        for array in (
            spectrum_data.k_grid,
            spectrum_data.Delta_l_T,
            spectrum_data.Delta_l_E,
            spectrum_data.C_l_TT,
            spectrum_data.C_l_TE,
            spectrum_data.C_l_EE,
        ):
            self.assertTrue(numpy.all(numpy.isfinite(array)))

    def test_camb_background_observables_are_finite(self) -> None:
        """The structured CAMB background helper should stay finite."""

        if camb is None:
            self.skipTest("CAMB is not installed")

        standard_contract = _standard_contract()
        redshifts = numpy.asarray([0.0, 0.5, 1.0], dtype=float)
        observables = cmb.compute_camb_background_observables(
            standard_contract,
            redshifts,
        )

        self.assertEqual(
            set(observables),
            {"rs_drag", "DM", "DH", "DA", "DV", "Hz", "z"},
        )
        self.assertTrue(numpy.array_equal(observables["z"], redshifts))
        for name, values in observables.items():
            if name == "z":
                continue
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_custom_spectra_have_structure_and_parameter_response(
        self,
    ) -> None:
        """The custom spectra should be finite, oscillatory, and responsive."""

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
            spectra=("TT", "TE", "EE"),
        )
        hi_ns_contract = _custom_contract()
        hi_ns_contract["param_map"]["ns"] = 1.02
        hi_ns = cmb.compute_cmb_spectrum_from_dict(
            hi_ns_contract,
            ells,
            spectra=("TT",),
        )
        hi_omb_contract = _custom_contract()
        hi_omb_contract["param_map"]["ombh2"] = 0.0245
        hi_omb = cmb.compute_cmb_spectrum_from_dict(
            hi_omb_contract,
            ells,
            spectra=("TT",),
        )

        base_tt = numpy.asarray(base["TT"], dtype=float)
        base_te = numpy.asarray(base["TE"], dtype=float)
        base_ee = numpy.asarray(base["EE"], dtype=float)
        hi_as_tt = numpy.asarray(hi_as["TT"], dtype=float)
        hi_ns_tt = numpy.asarray(hi_ns, dtype=float)
        hi_omb_tt = numpy.asarray(hi_omb, dtype=float)

        self.assertTrue(numpy.all(numpy.isfinite(base_tt)))
        self.assertTrue(numpy.all(numpy.isfinite(base_te)))
        self.assertTrue(numpy.all(numpy.isfinite(base_ee)))
        self.assertGreater(numpy.max(base_tt) - numpy.min(base_tt), 0.0)
        peak_index = int(numpy.argmax(base_tt))
        self.assertGreater(base_tt[peak_index], base_tt[peak_index - 1])
        self.assertGreater(base_tt[peak_index], base_tt[peak_index + 1])
        self.assertTrue(numpy.any(base_te < 0.0))
        self.assertTrue(numpy.any(base_te > 0.0))
        self.assertTrue(numpy.all(base_ee > 0.0))
        self.assertAlmostEqual(
            float(numpy.mean(hi_as_tt / base_tt)),
            2.0,
            places=2,
        )

        base_tilt_ratio = float(base_tt[0] / base_tt[-1])
        hi_tilt_ratio = float(hi_ns_tt[0] / hi_ns_tt[-1])
        self.assertLess(hi_tilt_ratio, base_tilt_ratio)

        base_contrast = float(base_tt[peak_index] / base_tt[peak_index + 1])
        hi_omb_contrast = float(
            hi_omb_tt[peak_index] / hi_omb_tt[peak_index + 1]
        )
        self.assertNotAlmostEqual(base_contrast, hi_omb_contrast, places=3)

        h0_ells = numpy.arange(60, 120, dtype=int)
        base_h0 = cmb.compute_cmb_spectrum_from_dict(
            _custom_contract(),
            h0_ells,
            spectra=("TT",),
        )
        hi_h0_contract = _custom_contract()
        hi_h0_contract["param_map"]["H0"] = 74.0
        hi_h0 = cmb.compute_cmb_spectrum_from_dict(
            hi_h0_contract,
            h0_ells,
            spectra=("TT",),
        )
        self.assertNotEqual(
            int(numpy.argmax(base_h0)),
            int(numpy.argmax(hi_h0)),
        )

    def test_declared_baryon_equation_changes_tt_spectrum(self) -> None:
        """Declared baryon evolution should change the TT spectrum."""

        baseline = _mapped_sector_only_contract()
        declared = _declared_equation_contract(
            baryon_rhs="-theta_b + 0.2 * Phi",
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
        declared_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_dict(
                declared,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(declared_tt - baseline_tt))),
            1.0e-12,
        )

    def test_declared_photon_monopole_equation_changes_tt_spectrum(
        self,
    ) -> None:
        """Declared photon-monopole evolution should change TT."""

        baseline = _mapped_sector_only_contract()
        declared = _declared_equation_contract(
            photon_monopole_rhs="-theta_gamma1 + 0.15 * Phi",
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
        declared_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_dict(
                declared,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(declared_tt - baseline_tt))),
            1.0e-12,
        )

    def test_declared_metric_closure_changes_te_or_ee_spectrum(self) -> None:
        """Declared metric closures should move the TE/EE spectra."""

        baseline = _declared_equation_contract(
            metric_closure_expression="Psi - Phi",
        )
        changed = _declared_equation_contract(
            metric_closure_expression="Psi - 1.15 * Phi",
        )
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

    def test_declared_source_expression_changes_los_result(self) -> None:
        """Declared source expressions should affect the LOS temperature."""

        baseline = _declared_equation_contract(
            source_expression="0.0",
        )
        changed = _declared_equation_contract(
            source_expression="theta_gamma0 + theta_b + Phi + Psi",
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

    def test_declared_source_channels_affect_expected_los_terms(
        self,
    ) -> None:
        """Each declared source channel should move its intended LOS term."""

        baseline = _declared_equation_contract(
            source_expression="0.0",
            source_channel="temperature_additive",
        )
        baseline["numerical"].update(
            {
                "k_sample_count": 4,
                "eta_sample_count": 64,
                "photon_hierarchy_l_max": 4,
                "neutrino_hierarchy_l_max": 4,
            }
        )
        ells = numpy.arange(20, 35, dtype=int)
        baseline_spectra = cmb.compute_cmb_spectrum_from_dict(
            baseline,
            ells,
            spectra=("TT", "TE", "EE"),
        )

        for channel_name in (
            "temperature_monopole",
            "temperature_doppler",
            "temperature_isw",
            "temperature_additive",
        ):
            contract = _declared_equation_contract(
                source_expression="1.0",
                source_channel=channel_name,
            )
            contract["numerical"].update(
                {
                    "k_sample_count": 4,
                    "eta_sample_count": 64,
                    "photon_hierarchy_l_max": 4,
                    "neutrino_hierarchy_l_max": 4,
                }
            )
            spectra = cmb.compute_cmb_spectrum_from_dict(
                contract,
                ells,
                spectra=("TT", "TE", "EE"),
            )
            tt_delta = numpy.asarray(
                spectra["TT"] - baseline_spectra["TT"],
                dtype=float,
            )
            self.assertGreater(float(numpy.max(numpy.abs(tt_delta))), 0.0)

        polarization_contract = _declared_equation_contract(
            source_expression="1.0",
            source_channel="polarization",
        )
        polarization_contract["numerical"].update(
            {
                "k_sample_count": 4,
                "eta_sample_count": 64,
                "photon_hierarchy_l_max": 4,
                "neutrino_hierarchy_l_max": 4,
            }
        )
        polarization_spectra = cmb.compute_cmb_spectrum_from_dict(
            polarization_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        polarization_spectra["EE"] - baseline_spectra["EE"]
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        polarization_spectra["TE"] - baseline_spectra["TE"]
                    )
                )
            ),
            0.0,
        )

    def test_unsupported_symbol_in_declared_equation_fails_loudly(
        self,
    ) -> None:
        """Unsupported equation symbols should fail at compile time."""

        contract = _declared_equation_contract()
        equations = contract["perturbations"]["equations"]
        equations["evolve_delta_b"]["rhs"] = "mystery_symbol + Phi"
        with self.assertRaisesRegex(ValueError, "mystery_symbol"):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_missing_background_symbol_in_declared_equation_fails_loudly(
        self,
    ) -> None:
        """Missing background names should fail at compile time."""

        contract = _declared_equation_contract()
        equations = contract["perturbations"]["equations"]
        theta_b_equation = equations["evolve_theta_b"]
        theta_b_equation["rhs"] = "-H_background * theta_b + k ** 2 * Psi"
        with self.assertRaisesRegex(ValueError, "H_background"):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_duplicate_derivative_declaration_fails_loudly(self) -> None:
        """Duplicate derivative declarations should be rejected."""

        contract = _declared_equation_contract()
        contract["perturbations"]["variables"]["delta_b_alt"] = {
            "kind": "baryon_density_contrast",
        }
        contract["perturbations"]["equations"]["evolve_delta_b_alt"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "delta_b_alt",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-theta_b + Phi",
        }
        duplicate_sector_message = (
            "more than one derivative for mapped sector "
            "'baryon_density_contrast'"
        )
        with self.assertRaisesRegex(
            ValueError,
            duplicate_sector_message,
        ):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_declared_equation_override_mode_requires_required_equations(
        self,
    ) -> None:
        """Declared-equation mode should require the full sector set."""

        contract = _declared_equation_contract()
        contract["perturbations"]["equations"] = {
            "evolve_delta_b": contract["perturbations"]["equations"][
                "evolve_delta_b"
            ]
        }
        with self.assertRaisesRegex(
            ValueError,
            "Declared-equation mode is missing required sector equation",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_mapped_sector_mode_still_passes(self) -> None:
        """Mapped-sector mode should continue to produce valid spectra."""

        contract = _mapped_sector_only_contract()
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_dict(
            contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )

        self.assertEqual(set(spectra), {"TT", "TE", "EE"})
        for spectrum in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(spectrum)))

    def test_custom_contract_validation_fails_loudly(self) -> None:
        """Unsupported or incomplete custom contracts should fail clearly."""

        missing_contract = _custom_contract()
        del missing_contract["perturbations"]["variables"]["theta_gamma2"]
        with self.assertRaisesRegex(
            ValueError,
            "photon temperature quadrupole",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                missing_contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

        unsupported_contract = _custom_contract()
        unsupported_contract["perturbations"]["variables"]["bogus_mode"] = {
            "kind": "density_contrast"
        }
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported custom perturbation variable 'bogus_mode'",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                unsupported_contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

        solver_contract = _custom_contract()
        solver_contract["perturbations"]["solver"] = "toy"
        with self.assertRaisesRegex(
            ValueError,
            "Unknown perturbation contract key\\(s\\): solver",
        ):
            cmb.compute_cmb_spectrum_from_dict(
                solver_contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_custom_cached_path_does_not_call_camb(self) -> None:
        """The cached plugin route should also use the custom scalar engine."""

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
