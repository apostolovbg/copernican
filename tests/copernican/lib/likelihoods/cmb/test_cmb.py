"""Physics tests for the declared-graph CMB likelihood helpers."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest import mock

import numpy
import pandas

try:
    import camb
except ImportError:  # pragma: no cover - optional external reference
    camb = None

from copernican.lib import model_coder
from copernican.lib.likelihoods import cmb
from copernican.lib.likelihoods.cmb import camb_solver
from copernican.lib.likelihoods.cmb import (
    copernican_cmb_solver as native_cmb_solver,
)
from copernican.lib.likelihoods.cmb import (
    native_background,
    native_cache,
    native_evolution,
    native_projection,
)
from copernican.lib.perturbation_contract import evaluate_compiled_expression


def _named_limit_message(
    quantity: str,
    measured: float,
    tolerance: float,
) -> str:
    """Return a consistent failure message for validation thresholds."""

    return (
        f"{quantity} exceeded tolerance: measured={measured:.6g}, "
        f"tolerance={tolerance:.6g}"
    )


def _full_width_at_half_max(
    grid: numpy.ndarray,
    values: numpy.ndarray,
) -> float:
    """Return the width of ``values`` at half of its maximum."""

    grid_values = numpy.asarray(grid, dtype=float)
    profile = numpy.asarray(values, dtype=float)
    half_max = 0.5 * float(numpy.max(profile))
    support = numpy.flatnonzero(profile >= half_max)
    if support.size < 2:
        raise ValueError("Half-maximum support requires at least two points.")
    return float(grid_values[support[-1]] - grid_values[support[0]])


def _local_extrema_ells(
    ells: numpy.ndarray,
    values: numpy.ndarray,
    *,
    kind: str,
    ell_min: int,
    ell_max: int,
) -> list[int]:
    """Return local-extremum multipoles inside ``[ell_min, ell_max]``."""

    ell_values = numpy.asarray(ells, dtype=int)
    spectrum = numpy.asarray(values, dtype=float)
    extrema: list[int] = []
    for index in range(1, spectrum.size - 1):
        ell_value = int(ell_values[index])
        if ell_value < ell_min or ell_value > ell_max:
            continue
        if kind == "max":
            keep = spectrum[index - 1] < spectrum[index] >= spectrum[index + 1]
        elif kind == "min":
            keep = spectrum[index - 1] > spectrum[index] <= spectrum[index + 1]
        else:  # pragma: no cover - helper misuse
            raise ValueError(f"Unsupported extremum kind: {kind}")
        if keep:
            extrema.append(ell_value)
    return extrema


def _zero_crossing_ells(
    ells: numpy.ndarray,
    values: numpy.ndarray,
    *,
    ell_min: int,
    ell_max: int,
) -> list[int]:
    """Return the multipoles where ``values`` changes sign."""

    ell_values = numpy.asarray(ells, dtype=int)
    spectrum = numpy.asarray(values, dtype=float)
    zero_crossings: list[int] = []
    for index in range(1, spectrum.size):
        ell_value = int(ell_values[index])
        if ell_value < ell_min or ell_value > ell_max:
            continue
        if spectrum[index - 1] == 0.0 or spectrum[index] == 0.0:
            zero_crossings.append(ell_value)
            continue
        if (spectrum[index - 1] < 0.0) != (spectrum[index] < 0.0):
            zero_crossings.append(ell_value)
    return zero_crossings


def _declared_graph_perturbations(
    *,
    baryon_rhs: str = (
        "-Hconf * theta_b + k * k * sound_speed_sq * delta_b "
        "+ collision_rate * (3.0 * theta_gamma1 - theta_b) "
        "+ k * k * Psi"
    ),
    photon_monopole_rhs: str = "-k * theta_gamma1 + (k * Psi) / 3.0",
    metric_closure_expression: str = "Phi",
    additive_source_expression: str = "0.0",
    include_bb: bool = False,
    include_lensing: bool = False,
    include_vector: bool = False,
) -> dict[str, object]:
    """Return a physically structured declared-math CMB graph."""

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
            "theta_gamma3": {
                "kind": "photon_temperature_octopole",
                "tensor_character": "scalar_like",
            },
            "e_gamma0": {
                "kind": "photon_polarization_monopole",
                "tensor_character": "scalar_like",
            },
            "e_gamma1": {
                "kind": "photon_polarization_dipole",
                "tensor_character": "scalar_like",
            },
            "e_gamma2": {
                "kind": "photon_polarization_quadrupole",
                "tensor_character": "scalar_like",
            },
            "e_gamma3": {
                "kind": "photon_polarization_octopole",
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
            "nu_l3": {
                "kind": "massless_neutrino_multipole",
                "tensor_character": "scalar_like",
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
            "polarization_moment": {
                "expression": "theta_gamma2 + 6.0 * e_gamma2",
                "description": "Quadrupole source for polarization.",
            },
            "total_matter_density": {
                "expression": "Omega_c0 * delta_c + Omega_b0 * delta_b",
                "description": "Matter density source for the metric.",
            },
            "total_radiation_density": {
                "expression": (
                    "4.0 * Omega_gamma0 * theta_gamma0 + Omega_nu0 * delta_nu"
                ),
                "description": "Radiation density source for the metric.",
            },
            "total_momentum_density": {
                "expression": (
                    "Omega_b0 * theta_b + Omega_c0 * theta_c + "
                    "4.0 * Omega_gamma0 * theta_gamma1 + Omega_nu0 * theta_nu"
                ),
                "description": "Momentum source for the metric.",
            },
            "total_neutrino_shear": {
                "expression": "Omega_nu0 * sigma_nu",
                "description": "Neutrino shear source for the metric.",
            },
            "metric_denominator": {
                "expression": "k * k + 3.0 * Hconf * Hconf",
                "description": "Regularized Poisson denominator.",
            },
            "einstein_gravity_strength": {
                "expression": "H0_over_c_Mpc_inv * H0_over_c_Mpc_inv",
                "description": (
                    "Background gravity scale used by the scalar Einstein "
                    "constraints."
                ),
            },
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
                    "(k / 3.0) * (theta_gamma0 + Psi - 2.0 * theta_gamma2) "
                    "+ collision_rate * (theta_b / 3.0 - theta_gamma1)"
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
                "rhs": (
                    "(2.0 / 5.0) * k * theta_gamma1 "
                    "- (3.0 / 5.0) * k * theta_gamma3 "
                    "- collision_rate * "
                    "(theta_gamma2 - 0.1 * polarization_moment)"
                ),
                "role": "hierarchy",
            },
            "evolve_theta_gamma3": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_gamma3",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(3.0 / 7.0) * k * theta_gamma2 "
                    "- (4.0 / 7.0) * k * theta_gamma3"
                ),
                "role": "hierarchy",
            },
            "evolve_e_gamma0": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma0",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-k * e_gamma1",
                "role": "polarization",
            },
            "evolve_e_gamma1": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma1",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "(k / 3.0) * (e_gamma0 - 2.0 * e_gamma2)",
                "role": "polarization",
            },
            "evolve_e_gamma2": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(2.0 / 5.0) * k * e_gamma1 "
                    "- (3.0 / 5.0) * k * e_gamma3 "
                    "- collision_rate * "
                    "(e_gamma2 - 0.1 * polarization_moment)"
                ),
                "role": "polarization",
            },
            "evolve_e_gamma3": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "e_gamma3",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(3.0 / 7.0) * k * e_gamma2 "
                    "- (4.0 / 7.0) * k * e_gamma3"
                ),
                "role": "polarization",
            },
            "evolve_delta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_b",
                "role": "continuity",
            },
            "evolve_theta_b": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": baryon_rhs,
                "role": "euler",
            },
            "evolve_delta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_c",
                "role": "continuity",
            },
            "evolve_theta_c": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-Hconf * theta_c + k * k * Psi",
                "role": "euler",
            },
            "evolve_delta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-(4.0 / 3.0) * theta_nu",
                "role": "continuity",
            },
            "evolve_theta_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": ("k * k * (0.25 * delta_nu - sigma_nu + Psi)"),
                "role": "euler",
            },
            "evolve_sigma_nu": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(4.0 / 15.0) * theta_nu " "- (3.0 / 5.0) * k * nu_l3"
                ),
                "role": "hierarchy",
            },
            "evolve_nu_l3": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "nu_l3",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    "(3.0 / 7.0) * k * sigma_nu " "- (4.0 / 7.0) * k * nu_l3"
                ),
                "role": "hierarchy",
            },
        },
        "constraints": {
            "phi_constraint": {
                "target": "Phi",
                "expression": (
                    "-1.5 * einstein_gravity_strength * "
                    "(total_matter_density + total_radiation_density) "
                    "/ metric_denominator"
                ),
                "role": "constraint",
            }
        },
        "closures": {
            "psi_closure": {
                "target": "Psi",
                "expression": (
                    "("
                    + metric_closure_expression
                    + ") - 3.0 * einstein_gravity_strength * "
                    "Omega_nu0 * sigma_nu / metric_denominator"
                ),
                "role": "closure",
            }
        },
        "sources": {
            "temperature_monopole": {
                "expression": (
                    "visibility * (theta_gamma0 + Psi "
                    "+ 0.25 * polarization_moment)"
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
                "expression": "0.75 * visibility * polarization_moment",
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
                "projection": "line_of_sight_temperature",
                "source_terms": {
                    "monopole": "temperature_monopole",
                    "doppler": "temperature_doppler",
                    "isw": "temperature_isw",
                    "additive": "temperature_additive",
                },
            },
            "polarization_e": {
                "kind": "transfer_component",
                "projection": "line_of_sight_polarization_e",
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
            "theta_gamma3_seed": {
                "target": {
                    "variable": "theta_gamma3",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
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
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "theta_gamma2_seed": {
                "target": {
                    "variable": "theta_gamma2",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": (
                    "(k * eta_initial) * (k * eta_initial) * seed / 30.0"
                ),
            },
            "e_gamma2_seed": {
                "target": {
                    "variable": "e_gamma2",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "e_gamma0_seed": {
                "target": {
                    "variable": "e_gamma0",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "e_gamma1_seed": {
                "target": {
                    "variable": "e_gamma1",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "e_gamma3_seed": {
                "target": {
                    "variable": "e_gamma3",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
            "delta_b_seed": {
                "target": {
                    "variable": "delta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-1.5 * seed",
            },
            "theta_b_seed": {
                "target": {
                    "variable": "theta_b",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "delta_c_seed": {
                "target": {
                    "variable": "delta_c",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-1.5 * seed",
            },
            "theta_c_seed": {
                "target": {
                    "variable": "theta_c",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "delta_nu_seed": {
                "target": {
                    "variable": "delta_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "-2.0 * seed",
            },
            "theta_nu_seed": {
                "target": {
                    "variable": "theta_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "(k * eta_initial / 6.0) * seed",
            },
            "sigma_nu_seed": {
                "target": {
                    "variable": "sigma_nu",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": (
                    "(k * eta_initial) * (k * eta_initial) * seed / 15.0"
                ),
            },
            "nu_l3_seed": {
                "target": {
                    "variable": "nu_l3",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
        },
        "boundary_conditions": {},
        "numerics": {
            "ell_min": 20,
            "ell_max": 90,
            "k_min": 1.0e-4,
            "k_max": 0.3,
            "k_sample_count": 10,
            "eta_sample_count": 768,
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
            "tight_coupling_ratio": 50.0,
            "a_min": 1.0e-6,
            "source_grid_multiplier": 2,
            "initial_redshift": 2.0e4,
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
        perturbations["variables"]["tensor_b"] = {
            "kind": "custom_tensor_polarization_source",
            "rank": 2,
            "spin": 2.0,
            "parity": "odd",
            "tensor_character": "tensor_like",
        }
        perturbations["equations"]["evolve_tensor_b"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": ("0.2 * k * polarization_moment - 0.4 * Hconf * tensor_b"),
            "role": "odd_parity_polarization",
        }
        perturbations["initial_conditions"]["tensor_b_seed"] = {
            "target": {
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "(k * eta_initial) * seed / 120.0",
        }
        perturbations["sources"]["polarization_b_source"] = {
            "expression": "visibility * tensor_b",
            "role": "polarization_b",
        }
        perturbations["observables"]["polarization_b"] = {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {
                "polarization_b": "polarization_b_source",
            },
        }
        perturbations["observables"]["BB"] = {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
        }
    if include_lensing:
        perturbations["sources"]["lensing_potential"] = {
            "expression": "exp(-tau) * (Phi + Psi)",
            "role": "potential",
        }
        perturbations["observables"]["lensing_potential"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_lensing_potential",
            "source_terms": {"potential": "lensing_potential"},
        }
        perturbations["observables"]["PP"] = {
            "kind": "angular_power_spectrum",
            "primary": "lensing_potential",
            "secondary": "lensing_potential",
        }
        perturbations["observables"]["TP"] = {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "lensing_potential",
        }
        perturbations["observables"]["EP"] = {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "lensing_potential",
        }
    if include_vector:
        perturbations["variables"]["vector_signal"] = {
            "kind": "custom_vector_mode",
            "spin": 1.0,
            "parity": "even",
            "tensor_character": "vector_like",
        }
        perturbations["equations"]["evolve_vector_signal"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "vector_signal",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": ("0.15 * k * theta_gamma1 - 0.25 * Hconf * vector_signal"),
            "role": "vector_coupling",
        }
        perturbations["initial_conditions"]["vector_signal_seed"] = {
            "target": {
                "variable": "vector_signal",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "(k * eta_initial) * seed / 90.0",
        }
        perturbations["sources"]["vector_source"] = {
            "expression": "visibility * vector_signal",
            "role": "signal",
        }
        perturbations["observables"]["vector_signal"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_signal",
            "source_terms": {"signal": "vector_source"},
        }
        perturbations["observables"]["VV"] = {
            "kind": "angular_power_spectrum",
            "primary": "vector_signal",
            "secondary": "vector_signal",
        }
    return perturbations


def _declared_background() -> dict[str, object]:
    """Return the declared background and reionization contract."""

    return {
        "derived": {
            "h": "H0 / 100.0",
            "H0_over_c_Mpc_inv": "H0 / 299792.458",
            "Omega_k0": "0.0",
            "Tcmb": "Tcmb_K",
            "Omega_b0": "ombh2 / (h * h)",
            "Omega_c0": "omch2 / (h * h)",
            "Omega_gamma0": ("2.469e-5 * ((Tcmb_K / 2.7255) ** 4) / (h * h)"),
            "Omega_nu0": "0.22710731766 * Neff * Omega_gamma0",
            "Omega_r0": "Omega_gamma0 + Omega_nu0",
            "Omega_m0": "Omega_b0 + Omega_c0",
            "Omega_de0": "1.0 - Omega_m0 - Omega_r0 - Omega_k0",
            "H": (
                "H0 * sqrt("
                "Omega_r0 / (a ** 4) + "
                "Omega_m0 / (a ** 3) + "
                "Omega_k0 / (a ** 2) + "
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
                "stellar_temperature_K": 5.0e4,
                "quasar_temperature_K": 1.5e5,
                "hydrogen_temperature_K": 1.0e4,
                "helium_temperature_K": 1.0e4,
                "helium_double_temperature_K": 2.0e4,
                "collapse_threshold": 1.686,
                "collapse_source": ("exp(-collapse_threshold / (a + 1.0e-6))"),
                "hard_source": "collapse_source * collapse_source",
                "stellar_helium_hardness": (
                    "exp(-("
                    "(24.587387 - 13.605693122994) * 1.602176634e-19"
                    ") / (1.380649e-23 * stellar_temperature_K))"
                ),
                "quasar_helium_hardness": (
                    "exp(-("
                    "(54.417763 - 13.605693122994) * 1.602176634e-19"
                    ") / (1.380649e-23 * quasar_temperature_K))"
                ),
                "hydrogen_ionization_rate": (
                    "(10 ** reionization_log10_amplitude) "
                    "* H_SI * collapse_source"
                ),
                "helium_ionization_rate": (
                    "hydrogen_ionization_rate * stellar_helium_hardness"
                ),
                "helium_double_ionization_rate": (
                    "(10 ** reionization_log10_amplitude) "
                    "* H_SI * hard_source * quasar_helium_hardness"
                ),
            },
        },
    }


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
        },
        "model_parameters": {
            "Tcmb_K": 2.7255,
        },
        "background": _declared_background(),
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
            "eta_sample_count": 768,
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
            "tight_coupling_ratio": 50.0,
            "a_min": 1.0e-6,
            "source_grid_multiplier": 2,
            "initial_redshift": 2.0e4,
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


def _speedup_contract(contract: dict[str, object]) -> dict[str, object]:
    """Return ``contract`` with lighter numerics for governed behavior tests.

    Fast runtime-response coverage uses this helper so the full governed suite
    stays integral while the separate scientific reference checks keep their
    production-like numerics unchanged.
    """

    contract["numerical"].update(
        {
            "eta_sample_count": 128,
            "source_grid_multiplier": 1,
        }
    )
    return contract


def _prepare_native_contract(
    contract: dict[str, object],
) -> dict[str, object]:
    """Return ``contract`` with its native runtime prepared upstream."""

    return model_coder.prepare_native_cmb_execution_contract(
        copy.deepcopy(contract)
    )


def _standard_contract() -> dict[str, object]:
    """Return a deep-copied standard CAMB fixture."""

    return copy.deepcopy(_base_standard_cmb_contract())


def _native_scalar_hierarchy_contract(
    *,
    gauge: str = "conformal_newtonian",
    initial_mode: str = "adiabatic_scalar",
    include_massive_neutrino: bool = False,
    include_lensing: bool = False,
    sum_mnu: float = 0.06,
) -> dict[str, object]:
    """Return a scalar native hierarchy fixture."""

    numerics = {
        "ell_min": 20,
        "ell_max": 120,
        "k_min": 1.0e-4,
        "k_max": 0.3,
        "k_sample_count": 18,
        "eta_sample_count": 192,
        "ode_rtol": 1.0e-5,
        "ode_atol": 1.0e-8,
        "tight_coupling_ratio": 80.0,
        "a_min": 1.0e-6,
        "source_grid_multiplier": 1,
        "initial_redshift": 2.0e4,
        "photon_hierarchy_l_max": 8,
        "neutrino_hierarchy_l_max": 5,
    }
    if include_massive_neutrino:
        numerics["massive_neutrino_hierarchy_l_max"] = 5
        numerics["momentum_grids"] = {
            "massive_neutrino_default": {
                "count": 8,
                "q_min": 0.05,
                "q_max": 18.0,
                "mass_parameter": "sum_mnu",
            }
        }
    initial_condition_families = {
        initial_mode: {
            "sector": "scalar",
            "members": [],
        }
    }
    scalar_species = [
        "photon",
        "baryon",
        "cdm",
        "massless_neutrino",
    ]
    scalar_hierarchy_families = [
        "photon_temperature",
        "photon_polarization_e",
        "massless_neutrino",
    ]
    species = {
        "photon": {
            "sector": "scalar",
            "hierarchy_family": "photon_temperature",
            "collision_operators": ["thomson_drag"],
            "background_reference": "Omega_gamma0",
        },
        "baryon": {
            "sector": "scalar",
            "collision_operators": ["thomson_drag"],
            "background_reference": "Omega_b0",
        },
        "cdm": {
            "sector": "scalar",
            "background_reference": "Omega_c0",
        },
        "massless_neutrino": {
            "sector": "scalar",
            "hierarchy_family": "massless_neutrino",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        },
    }
    hierarchy_families = {
        "photon_temperature": {
            "sector": "scalar",
            "species": ["photon"],
            "closure": "free_streaming_scalar",
            "default_l_max": 8,
            "multipole_symbol": "theta_gamma_l",
        },
        "photon_polarization_e": {
            "sector": "scalar",
            "species": ["photon"],
            "closure": "free_streaming_scalar",
            "default_l_max": 8,
            "multipole_symbol": "e_gamma_l",
        },
        "massless_neutrino": {
            "sector": "scalar",
            "species": ["massless_neutrino"],
            "closure": "free_streaming_scalar",
            "default_l_max": 5,
            "multipole_symbol": "nu_l",
        },
    }
    if include_massive_neutrino:
        scalar_species.append("massive_neutrino")
        scalar_hierarchy_families.append("massive_neutrino")
        species["massive_neutrino"] = {
            "sector": "scalar",
            "hierarchy_family": "massive_neutrino",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        }
        hierarchy_families["massive_neutrino"] = {
            "sector": "scalar",
            "species": ["massive_neutrino"],
            "closure": "free_streaming_scalar",
            "default_l_max": 5,
            "multipole_symbol": "nu_massive_l",
            "momentum_grid": "massive_neutrino_default",
        }
    if include_lensing:
        hierarchy_families["lensing_potential"] = {
            "sector": "scalar",
            "species": ["photon", "massless_neutrino"],
            "closure": "line_of_sight_lensing",
            "default_l_max": 8,
            "multipole_symbol": "phi_l",
        }
    return {
        "model_name": "NativeScalarHierarchy",
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
            "sum_mnu": sum_mnu,
            "num_massive_neutrinos": 3,
        },
        "model_parameters": {
            "Tcmb_K": 2.7255,
        },
        "background": _declared_background(),
        "grids": {},
        "values": {},
        "calls": [],
        "numerical": dict(numerics),
        "perturbations": {
            "contract_version": 2,
            "standard": False,
            "gauge": gauge,
            "variables": {},
            "derived": {},
            "equations": {},
            "constraints": {},
            "closures": {},
            "collision_operators": {
                "thomson_drag": {
                    "sector": "scalar",
                    "species": ["photon", "baryon"],
                    "expression": (
                        "collision_rate * " "(theta_b / 3.0 - theta_gamma1)"
                    ),
                    "counterpart": "baryon_thomson_drag",
                }
            },
            "conservation_rules": {
                "thomson_drag_balance": {
                    "kind": "absolute_max",
                    "expression": (
                        "photon_baryon_momentum_ratio * thomson_drag + "
                        "baryon_thomson_drag"
                    ),
                    "tolerance": 1.0e-12,
                    "domain": "scalar",
                }
            },
            "initial_conditions": {},
            "initial_condition_families": initial_condition_families,
            "boundary_conditions": {},
            "sectors": {
                "scalar": {
                    "description": "Native scalar hierarchy sector.",
                    "species": scalar_species,
                    "hierarchy_families": scalar_hierarchy_families,
                    "supported_gauges": [
                        "conformal_newtonian",
                        "synchronous",
                    ],
                    "tensor_character": "scalar_like",
                }
            },
            "species": species,
            "hierarchy_families": hierarchy_families,
            "projection_typing": {
                "temperature_line_of_sight": {
                    "sector": "scalar",
                    "kernel": "temperature_mixed_window",
                    "source_roles": ["monopole", "doppler", "isw"],
                    "observable_kinds": ["transfer_component"],
                    "parity": "even",
                    "spin": 0.0,
                }
            },
            "accuracy_controls": {
                "scalar_reference_ells": [20, 60, 120],
                "runtime_envelope": "bounded",
            },
            "sources": {},
            "observables": {},
            "numerics": dict(numerics),
            "validity": {
                "regimes": ["linear", "native_scalar_hierarchy"],
                "notes": "Metadata-only native scalar hierarchy route.",
            },
            "backend_mapping": {
                "camb": {
                    "native_solver_required": True,
                    "implemented": True,
                }
            },
        },
    }


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


def _strip_native_runtime_sections(
    contract: dict[str, object],
) -> dict[str, object]:
    """Return a metadata-only native scalar hierarchy contract."""

    stripped = copy.deepcopy(contract)
    perturbations = stripped.get("perturbations")
    if isinstance(perturbations, dict):
        for section_name in (
            "variables",
            "derived",
            "equations",
            "constraints",
            "closures",
            "sources",
            "observables",
            "initial_conditions",
            "boundary_conditions",
        ):
            perturbations.pop(section_name, None)
    return stripped


def _analytic_signal_contract(
    *,
    source_scale: float = 1.0,
    closure_scale: float = 1.0,
    decay_rate: float = 0.02,
) -> dict[str, object]:
    """Return a one-mode graph with exact source and closure scaling."""

    contract = _base_custom_cmb_contract()
    contract["model_name"] = "AnalyticSignalCMB"
    model_parameters = dict(contract.get("model_parameters", {}))
    model_parameters.update(
        {
            "source_scale": source_scale,
            "closure_scale": closure_scale,
            "decay_rate": decay_rate,
        }
    )
    contract["model_parameters"] = model_parameters
    contract["numerical"].update(
        {
            "ell_max": 40,
            "k_sample_count": 6,
            "eta_sample_count": 128,
            "source_grid_multiplier": 1,
        }
    )
    contract["perturbations"] = {
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
            "regimes": ["analytic_signal"],
        },
        "backend_mapping": {
            "camb": {
                "implemented": True,
                "native_solver_required": True,
            }
        },
    }
    return contract


def _generic_background_custom_contract() -> dict[str, object]:
    """Return a custom CMB fixture with non-LCDM-style parameter names."""

    contract = _base_custom_cmb_contract()
    contract["model_name"] = "GenericBackgroundCustomCMB"
    contract["param_map"] = {
        "H0": 67.4,
        "tau": 0.054,
        "primordial_power_amplitude": 2.1e-9,
        "primordial_power_tilt": 0.965,
        "baryon_fraction_today": 0.049,
        "cold_dark_matter_fraction_today": 0.262,
        "photon_fraction_today": 5.38e-5,
        "relativistic_neutrino_fraction_today": 3.65e-5,
        "curvature_density_fraction": -0.01,
        "dark_fluid_eos_today": -0.85,
    }
    contract["model_parameters"] = {
        "cmb_temperature_K": 2.7255,
        "helium_mass_fraction": 0.245,
    }
    reionization = copy.deepcopy(_declared_background()["reionization"])
    contract["background"] = {
        "derived": {
            "Omega_b0": "baryon_fraction_today",
            "Omega_c0": "cold_dark_matter_fraction_today",
            "Omega_gamma0": "photon_fraction_today",
            "Omega_nu0": "relativistic_neutrino_fraction_today",
            "Omega_r0": "Omega_gamma0 + Omega_nu0",
            "Omega_k0": "curvature_density_fraction",
            "Omega_m0": "Omega_b0 + Omega_c0",
            "w0": "dark_fluid_eos_today",
            "Omega_de0": "1.0 - Omega_m0 - Omega_r0 - Omega_k0",
            "dark_energy_pressure_today": "w0 * Omega_de0",
            "H": (
                "H0 * sqrt("
                "Omega_r0 / (a ** 4) + "
                "Omega_m0 / (a ** 3) + "
                "Omega_k0 / (a ** 2) + "
                "Omega_de0 * (a ** (-3.0 * (1.0 + w0)))"
                ")"
            ),
        },
        "reionization": reionization,
    }
    return contract


def _critical_density_today_kg_m3(hubble_km_s_mpc: float) -> float:
    """Return the present-day critical density for ``hubble_km_s_mpc``."""

    hubble_si = float(hubble_km_s_mpc) * 1000.0 / 3.085_677_581_491_3673e22
    return float(3.0 * hubble_si * hubble_si / (8.0 * numpy.pi * 6.674_30e-11))


def _physical_density_custom_contract() -> dict[str, object]:
    """Return a native CMB fixture driven by direct physical densities."""

    contract = _base_custom_cmb_contract()
    contract["model_name"] = "PhysicalDensityCustomCMB"
    expansion_rate_today = 67.4
    baryon_fraction_today = 0.049
    dark_component_fraction_today = 0.262
    critical_density_today = _critical_density_today_kg_m3(
        expansion_rate_today
    )
    perturbations = _declared_graph_perturbations()
    perturbations["derived"]["total_matter_density"]["expression"] = (
        "dark_component_budget_today * delta_c + "
        "baryon_budget_today * delta_b"
    )
    perturbations["derived"]["total_radiation_density"]["expression"] = (
        "4.0 * photon_fraction_today * theta_gamma0 + "
        "relativistic_neutrino_fraction_today * delta_nu"
    )
    contract["param_map"] = {
        "expansion_rate_today": expansion_rate_today,
        "reionization_depth": 0.054,
        "scalar_power_amplitude": 2.1e-9,
        "scalar_tilt_index": 0.965,
        "critical_mass_density_today_kg_m3": critical_density_today,
        "baryon_rest_mass_density_today": (
            baryon_fraction_today * critical_density_today
        ),
        "cold_dark_matter_rest_mass_density_today": (
            dark_component_fraction_today * critical_density_today
        ),
        "photon_fraction_today": 5.38e-5,
        "relativistic_neutrino_fraction_today": 3.65e-5,
        "curvature_fraction_today": -0.01,
        "dark_component_eos_today": -0.85,
    }
    contract["model_parameters"] = {
        "cmb_temperature_kelvin": 2.7255,
        "helium_mass_fraction": 0.245,
    }
    contract["background"] = {
        "derived": {
            "baryon_budget_today": (
                "baryon_rest_mass_density_today / "
                "critical_mass_density_today_kg_m3"
            ),
            "dark_component_budget_today": (
                "cold_dark_matter_rest_mass_density_today / "
                "critical_mass_density_today_kg_m3"
            ),
            "light_budget_today": (
                "photon_fraction_today + relativistic_neutrino_fraction_today"
            ),
            "matter_budget_today": (
                "baryon_budget_today + dark_component_budget_today"
            ),
            "vacuum_budget_today": (
                "1.0 - matter_budget_today - light_budget_today - "
                "curvature_fraction_today"
            ),
            "dark_energy_pressure_today": (
                "dark_component_eos_today * vacuum_budget_today"
            ),
            "H": (
                "expansion_rate_today * sqrt("
                "light_budget_today / (a ** 4) + "
                "matter_budget_today / (a ** 3) + "
                "curvature_fraction_today / (a ** 2) + "
                "vacuum_budget_today * ("
                "a ** (-3.0 * (1.0 + dark_component_eos_today))"
                "))"
            ),
        },
        "reionization": copy.deepcopy(_declared_background()["reionization"]),
    }
    contract["background"]["reionization"]["calibration"][
        "target_optical_depth"
    ] = "reionization_depth"
    contract["perturbations"] = perturbations
    return contract


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

        raise AssertionError("native runtime should bypass get_camb_contract")

    def get_cmb_native_runtime(self, _params):
        """Return the synthetic native-runtime contract used by the helper."""

        return _speedup_contract(_custom_contract())

    def get_cmb_perturbation_contract(self, _params):
        """Return the synthetic non-standard perturbation graph."""

        return _custom_perturbations()


class CMBScientificReferenceValidationTestCase(unittest.TestCase):
    """CAMB-backed scientific reference checks for the CMB surface."""

    def test_slow_custom_background_matches_camb_recombination_reference(
        self,
    ) -> None:
        """Slow reference validation should catch named background defects."""

        if camb is None:
            self.skipTest("CAMB is not installed")

        contract = _prepare_native_contract(_custom_contract())
        physical = native_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        numerics = native_background._resolve_custom_cmb_numerics(contract)
        background = native_background._build_custom_cmb_background(
            contract,
            physical,
            numerics,
        )
        reference_contract = _strip_perturbations(contract)
        reference_contract["param_map"].pop("z_rec", None)
        params = camb_solver._make_camb_params(reference_contract, lmax=32)
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
        peak_eta = float(background.eta_grid[peak_index])
        recombination_band = (background.z_grid >= 800.0) & (
            background.z_grid <= 1600.0
        )
        recombination_median_x_e_error = float(
            numpy.median(
                numpy.abs(
                    background.x_e_grid[recombination_band]
                    - reference_x_e[recombination_band]
                )
                / numpy.maximum(reference_x_e[recombination_band], 1.0e-8)
            )
        )
        recombination_p90_error = float(
            numpy.percentile(
                numpy.abs(
                    background.x_e_grid[recombination_band]
                    - reference_x_e[recombination_band]
                )
                / numpy.maximum(reference_x_e[recombination_band], 1.0e-8),
                90.0,
            )
        )
        reionization_transition_band = (background.z_grid >= 6.0) & (
            background.z_grid <= 10.0
        )
        reionization_transition_error = float(
            numpy.median(
                numpy.abs(
                    background.x_e_grid[reionization_transition_band]
                    - reference_x_e[reionization_transition_band]
                )
            )
        )
        visibility_width_eta = _full_width_at_half_max(
            background.eta_grid,
            background.visibility_grid,
        )
        reference_visibility_width_eta = _full_width_at_half_max(
            background.eta_grid,
            reference_visibility,
        )
        visibility_width_z = _full_width_at_half_max(
            background.z_grid[::-1],
            background.visibility_grid[::-1],
        )
        reference_visibility_width_z = _full_width_at_half_max(
            background.z_grid[::-1],
            reference_visibility[::-1],
        )
        max_ionized_fraction = 1.0 + (
            physical.YHe / (2.0 * max(1.0 - physical.YHe, 1.0e-6))
        )

        self.assertTrue(numpy.all(numpy.isfinite(background.x_e_grid)))
        self.assertTrue(numpy.all(background.x_e_grid >= 0.0))
        self.assertTrue(
            numpy.all(background.x_e_grid <= max_ionized_fraction + 1.0e-6)
        )
        self.assertTrue(numpy.all(numpy.isfinite(reference_x_e)))
        self.assertTrue(numpy.all(numpy.isfinite(reference_visibility)))
        self.assertTrue(numpy.all(numpy.diff(background.tau_grid) <= 1.0e-8))

        peak_z_error = abs(peak_z - reference_peak_z) / reference_peak_z
        self.assertLess(
            peak_z_error,
            0.005,
            _named_limit_message(
                "visibility peak redshift",
                peak_z_error,
                0.005,
            ),
        )
        peak_eta_error = (
            abs(peak_eta - reference_peak_eta) / reference_peak_eta
        )
        self.assertLess(
            peak_eta_error,
            0.005,
            _named_limit_message(
                "visibility peak conformal time",
                peak_eta_error,
                0.005,
            ),
        )
        visibility_width_eta_error = (
            abs(visibility_width_eta - reference_visibility_width_eta)
            / reference_visibility_width_eta
        )
        self.assertLess(
            visibility_width_eta_error,
            0.02,
            _named_limit_message(
                "visibility FWHM in conformal time",
                visibility_width_eta_error,
                0.02,
            ),
        )
        visibility_width_z_error = (
            abs(visibility_width_z - reference_visibility_width_z)
            / reference_visibility_width_z
        )
        self.assertLess(
            visibility_width_z_error,
            0.02,
            _named_limit_message(
                "visibility FWHM in redshift",
                visibility_width_z_error,
                0.02,
            ),
        )
        eta0_error = abs(background.eta0 - reference_eta0) / reference_eta0
        self.assertLess(
            eta0_error,
            0.002,
            _named_limit_message("eta0", eta0_error, 0.002),
        )
        sound_horizon_error = (
            abs(background.sound_horizon_mpc - reference_sound_horizon)
            / reference_sound_horizon
        )
        self.assertLess(
            sound_horizon_error,
            0.002,
            _named_limit_message(
                "sound horizon at visibility peak",
                sound_horizon_error,
                0.002,
            ),
        )
        self.assertLess(
            recombination_median_x_e_error,
            0.01,
            _named_limit_message(
                "recombination x_e median relative error",
                recombination_median_x_e_error,
                0.01,
            ),
        )
        self.assertLess(
            recombination_p90_error,
            0.45,
            _named_limit_message(
                "recombination x_e p90 relative error",
                recombination_p90_error,
                0.45,
            ),
        )
        self.assertLess(
            reionization_transition_error,
            0.08,
            _named_limit_message(
                "reionization-band x_e median absolute error",
                reionization_transition_error,
                0.08,
            ),
        )
        tau_error = abs(background.reionization_tau - physical.tau_reio) / max(
            physical.tau_reio, 1.0e-12
        )
        self.assertLess(
            tau_error,
            0.03,
            _named_limit_message(
                "reionization optical depth",
                tau_error,
                0.03,
            ),
        )

    def test_standard_lcdm_reference_features_match_camb(self) -> None:
        """Standard-path scalar spectra should preserve CAMB features."""

        if camb is None:
            self.skipTest("CAMB is not installed")

        standard_contract = _standard_contract()
        ells = numpy.arange(2, 801, dtype=int)
        actual = cmb.compute_cmb_spectrum_from_contract(
            standard_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )

        params = camb_solver._make_camb_params(
            standard_contract,
            lmax=int(ells.max()),
        )
        results = camb.get_results(params)
        reference = results.get_unlensed_scalar_cls(
            lmax=int(ells.max()),
            CMB_unit="muK",
        )

        for spectrum_name, column_index in (("TT", 0), ("EE", 1), ("TE", 3)):
            actual_values = numpy.asarray(actual[spectrum_name], dtype=float)
            reference_values = numpy.asarray(
                reference[:, column_index][ells],
                dtype=float,
            )
            numpy.testing.assert_allclose(
                actual_values,
                reference_values,
                rtol=1.0e-5,
                atol=1.0e-5,
                err_msg=(
                    f"{spectrum_name} reference mismatch across "
                    "ell=2..800 for the standard CAMB route."
                ),
            )
            low_ell_mask = ells <= 30
            numpy.testing.assert_allclose(
                actual_values[low_ell_mask],
                reference_values[low_ell_mask],
                rtol=1.0e-5,
                atol=1.0e-5,
                err_msg=(
                    f"{spectrum_name} low-ell reference mismatch for the "
                    "standard CAMB route."
                ),
            )

        tt_actual_peaks = _local_extrema_ells(
            ells,
            numpy.asarray(actual["TT"], dtype=float),
            kind="max",
            ell_min=150,
            ell_max=650,
        )
        tt_reference_peaks = _local_extrema_ells(
            ells,
            numpy.asarray(reference[:, 0][ells], dtype=float),
            kind="max",
            ell_min=150,
            ell_max=650,
        )
        self.assertGreaterEqual(len(tt_actual_peaks), 2)
        self.assertGreaterEqual(len(tt_reference_peaks), 2)
        self.assertLessEqual(
            abs(tt_actual_peaks[0] - tt_reference_peaks[0]),
            1,
            (
                "TT first acoustic peak mismatch: "
                f"actual={tt_actual_peaks[0]}, "
                f"reference={tt_reference_peaks[0]}"
            ),
        )
        self.assertLessEqual(
            abs(tt_actual_peaks[1] - tt_reference_peaks[1]),
            1,
            (
                "TT second acoustic peak mismatch: "
                f"actual={tt_actual_peaks[1]}, "
                f"reference={tt_reference_peaks[1]}"
            ),
        )
        actual_tt_spacing = tt_actual_peaks[1] - tt_actual_peaks[0]
        reference_tt_spacing = tt_reference_peaks[1] - tt_reference_peaks[0]
        self.assertLessEqual(
            abs(actual_tt_spacing - reference_tt_spacing),
            1,
            (
                "TT acoustic peak spacing mismatch: "
                f"actual={actual_tt_spacing}, "
                f"reference={reference_tt_spacing}"
            ),
        )

        ee_actual_peaks = _local_extrema_ells(
            ells,
            numpy.asarray(actual["EE"], dtype=float),
            kind="max",
            ell_min=50,
            ell_max=450,
        )
        ee_reference_peaks = _local_extrema_ells(
            ells,
            numpy.asarray(reference[:, 1][ells], dtype=float),
            kind="max",
            ell_min=50,
            ell_max=450,
        )
        self.assertGreaterEqual(len(ee_actual_peaks), 2)
        self.assertGreaterEqual(len(ee_reference_peaks), 2)
        self.assertLessEqual(
            abs(ee_actual_peaks[0] - ee_reference_peaks[0]),
            1,
            (
                "EE first peak mismatch: "
                f"actual={ee_actual_peaks[0]}, "
                f"reference={ee_reference_peaks[0]}"
            ),
        )
        self.assertLessEqual(
            abs(ee_actual_peaks[1] - ee_reference_peaks[1]),
            1,
            (
                "EE second peak mismatch: "
                f"actual={ee_actual_peaks[1]}, "
                f"reference={ee_reference_peaks[1]}"
            ),
        )

        te_actual_zero_crossings = _zero_crossing_ells(
            ells,
            numpy.asarray(actual["TE"], dtype=float),
            ell_min=30,
            ell_max=650,
        )
        te_reference_zero_crossings = _zero_crossing_ells(
            ells,
            numpy.asarray(reference[:, 3][ells], dtype=float),
            ell_min=30,
            ell_max=650,
        )
        self.assertGreaterEqual(len(te_actual_zero_crossings), 3)
        self.assertGreaterEqual(len(te_reference_zero_crossings), 3)
        for index in range(3):
            self.assertLessEqual(
                abs(
                    te_actual_zero_crossings[index]
                    - te_reference_zero_crossings[index]
                ),
                1,
                (
                    "TE zero-crossing mismatch at index "
                    f"{index}: actual={te_actual_zero_crossings[index]}, "
                    f"reference={te_reference_zero_crossings[index]}"
                ),
            )

    def test_native_scalar_hierarchy_amplitude_response_tracks_camb(
        self,
    ) -> None:
        """Native scalar hierarchy should preserve CAMB amplitude response."""

        if camb is None:
            self.skipTest("CAMB is not installed")

        base_contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        shifted_contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        shifted_contract["param_map"]["As"] *= 1.1
        ells = numpy.asarray((30, 60, 90, 120), dtype=int)
        base_native = cmb.compute_cmb_spectrum_from_contract(
            base_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )
        shifted_native = cmb.compute_cmb_spectrum_from_contract(
            shifted_contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )

        base_params = camb_solver._make_camb_params(
            base_contract,
            lmax=int(ells.max()),
        )
        shifted_params = camb_solver._make_camb_params(
            shifted_contract,
            lmax=int(ells.max()),
        )
        base_reference = camb.get_results(base_params).get_unlensed_scalar_cls(
            lmax=int(ells.max()),
            CMB_unit="muK",
        )
        shifted_reference = camb.get_results(
            shifted_params
        ).get_unlensed_scalar_cls(
            lmax=int(ells.max()),
            CMB_unit="muK",
        )

        for spectrum_name, column_index in (("TT", 0), ("EE", 1), ("TE", 3)):
            native_base = numpy.asarray(
                base_native[spectrum_name], dtype=float
            )
            native_shifted = numpy.asarray(
                shifted_native[spectrum_name],
                dtype=float,
            )
            reference_base = numpy.asarray(
                base_reference[:, column_index][ells],
                dtype=float,
            )
            reference_shifted = numpy.asarray(
                shifted_reference[:, column_index][ells],
                dtype=float,
            )
            native_ratio = numpy.divide(
                numpy.abs(native_shifted),
                numpy.maximum(numpy.abs(native_base), 1.0e-30),
            )
            reference_ratio = numpy.divide(
                numpy.abs(reference_shifted),
                numpy.maximum(numpy.abs(reference_base), 1.0e-30),
            )
            numpy.testing.assert_allclose(
                native_ratio,
                reference_ratio,
                rtol=5.0e-2,
                atol=5.0e-2,
                err_msg=(
                    f"{spectrum_name} amplitude-response mismatch for the "
                    "native scalar hierarchy route."
                ),
            )


class CMBCustomAnalyticValidationTestCase(unittest.TestCase):
    """Analytic observable checks for the declared-graph runtime."""

    def test_custom_source_expression_changes_observable(self) -> None:
        """Source scalings should map to the expected quadratic TT response."""

        baseline = _analytic_signal_contract(source_scale=1.0)
        changed = _analytic_signal_contract(source_scale=1.5)
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, 1.5 * 1.5),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared source scaling should produce the exact quadratic "
                "TT power response."
            ),
        )

    def test_custom_closures_change_spectrum(self) -> None:
        """Closure scalings should map to quadratic TT power."""

        baseline = _analytic_signal_contract(closure_scale=1.0)
        changed = _analytic_signal_contract(closure_scale=1.7)
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, 1.7 * 1.7),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared closure scaling should produce the exact quadratic "
                "TT power response."
            ),
        )

    def test_custom_equations_change_spectrum(self) -> None:
        """Stronger damping should suppress the raw analytic TT response."""

        low_decay = _prepare_native_contract(
            _analytic_signal_contract(decay_rate=0.01)
        )
        high_decay = _prepare_native_contract(
            _analytic_signal_contract(decay_rate=0.05)
        )
        for contract in (low_decay, high_decay):
            contract["numerical"].update(
                {
                    "eta_sample_count": 256,
                    "source_grid_multiplier": 2,
                }
            )
        ells = numpy.arange(20, 30, dtype=int)
        low_decay_tt = numpy.asarray(
            native_projection._compute_custom_cmb_spectrum_data(
                low_decay,
                ells,
            ).spectra["TT"],
            dtype=float,
        )
        high_decay_tt = numpy.asarray(
            native_projection._compute_custom_cmb_spectrum_data(
                high_decay,
                ells,
            ).spectra["TT"],
            dtype=float,
        )
        self.assertTrue(
            numpy.all(numpy.abs(high_decay_tt) < numpy.abs(low_decay_tt)),
            (
                "Increasing the declared decay coefficient should reduce "
                "the TT amplitude for every tested multipole."
            ),
        )

    def test_declared_interactions_change_observable(self) -> None:
        """Interaction terms should feed declared sources before projection."""

        baseline = _analytic_signal_contract()
        changed = _analytic_signal_contract()
        for contract, coefficient in (
            (baseline, 0.5),
            (changed, 1.0),
        ):
            contract["perturbations"]["interactions"] = {
                "signal_bridge": {
                    "expression": f"{coefficient:.16g} * signal_mode",
                }
            }
            contract["perturbations"]["sources"]["signal_source"][
                "expression"
            ] = "closure_drive + signal_bridge"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, (2.0 / 1.5) ** 2),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared interactions should contribute to the compiled "
                "source pipeline before the TT projection."
            ),
        )

    def test_custom_projection_kernel_changes_observable(self) -> None:
        """Custom kernels should change the projected transfer response."""

        spherical = _analytic_signal_contract()
        derivative = _analytic_signal_contract()
        spherical_observable = spherical["perturbations"]["observables"][
            "signal_transfer"
        ]
        derivative_observable = derivative["perturbations"]["observables"][
            "signal_transfer"
        ]
        spherical_observable["projection"] = "custom_line_of_sight"
        spherical_observable["kernel"] = "spherical_bessel_window"
        derivative_observable["projection"] = "custom_line_of_sight"
        derivative_observable["kernel"] = "spherical_bessel_derivative_window"
        ells = numpy.arange(20, 30, dtype=int)
        spherical_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                spherical,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        derivative_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                derivative,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(derivative_tt - spherical_tt))),
            1.0e-12,
        )

    def test_projection_extension_alias_changes_observable(self) -> None:
        """Projection extensions should route through the reviewed alias."""

        baseline = _analytic_signal_contract()
        extended = _analytic_signal_contract()
        extended["perturbations"]["projection_extensions"] = {
            "signal_derivative_alias": {
                "base_projection": "custom_line_of_sight",
                "kernel": "spherical_bessel_derivative_window",
                "required_roles": ["signal"],
                "allowed_roles": ["signal"],
            }
        }
        extended["perturbations"]["observables"]["signal_transfer"][
            "projection"
        ] = "signal_derivative_alias"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        extended_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                extended,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertGreater(
            float(numpy.max(numpy.abs(extended_tt - baseline_tt))),
            1.0e-12,
        )

    def test_spin2_e_projection_sums_declared_sources(self) -> None:
        """Spin-2 E projections should not hide one declared source term."""

        baseline = _analytic_signal_contract()
        changed = _analytic_signal_contract()
        baseline_observable = baseline["perturbations"]["observables"][
            "signal_transfer"
        ]
        changed_observable = changed["perturbations"]["observables"][
            "signal_transfer"
        ]
        baseline_observable["projection"] = "spin2_e_mode"
        changed_observable["projection"] = "spin2_e_mode"
        baseline_observable["source_terms"] = {"signal": "signal_source"}
        changed["perturbations"]["sources"]["polarization_source"] = {
            "expression": "0.5 * closure_drive",
            "role": "polarization",
        }
        changed_observable["source_terms"] = {
            "signal": "signal_source",
            "polarization": "polarization_source",
        }
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_tt / baseline_tt,
            numpy.full_like(baseline_tt, 1.5 * 1.5),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Spin-2 E projections should sum every declared source term "
                "instead of silently substituting one role for another."
            ),
        )

    def test_lensing_source_changes_pp(self) -> None:
        """Lensing-source scalings should map to exact quadratic PP power."""

        baseline = _speedup_contract(_custom_contract(include_lensing=True))
        changed = _speedup_contract(_custom_contract(include_lensing=True))
        changed["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.35 * exp(-tau) * (Phi + Psi)"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_pp = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("PP",),
            ),
            dtype=float,
        )
        changed_pp = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("PP",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_pp / baseline_pp,
            numpy.full_like(baseline_pp, 1.35 * 1.35),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared lensing-source scaling should produce the exact "
                "quadratic PP power response."
            ),
        )

    def test_custom_lensing_kernel_changes_pp(self) -> None:
        """Custom lensing kernels should preserve the declared PP response."""

        baseline = _speedup_contract(_custom_contract(include_lensing=True))
        changed = _speedup_contract(_custom_contract(include_lensing=True))
        for contract in (baseline, changed):
            contract["perturbations"]["observables"]["lensing_potential"] = {
                "kind": "transfer_component",
                "projection": "custom_line_of_sight",
                "kernel": "lensing_potential_window",
                "source_terms": {"potential": "lensing_potential"},
            }
        changed["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.35 * exp(-tau) * (Phi + Psi)"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_pp = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("PP",),
            ),
            dtype=float,
        )
        changed_pp = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("PP",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_pp / baseline_pp,
            numpy.full_like(baseline_pp, 1.35 * 1.35),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Custom lensing kernels should preserve the exact quadratic "
                "PP response to declared source scaling."
            ),
        )

    def test_b_mode_source_changes_bb(self) -> None:
        """B-mode source scalings should map to exact quadratic BB power."""

        baseline = _speedup_contract(_custom_contract(include_bb=True))
        changed = _speedup_contract(_custom_contract(include_bb=True))
        changed["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "1.25 * visibility * tensor_b"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_bb = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("BB",),
            ),
            dtype=float,
        )
        changed_bb = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("BB",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_bb / baseline_bb,
            numpy.full_like(baseline_bb, 1.25 * 1.25),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Declared odd-parity source scaling should produce the "
                "exact quadratic BB power response."
            ),
        )

    def test_custom_b_mode_kernel_changes_bb(self) -> None:
        """Custom B-mode kernels should preserve the declared BB response."""

        baseline = _speedup_contract(_custom_contract(include_bb=True))
        changed = _speedup_contract(_custom_contract(include_bb=True))
        for contract in (baseline, changed):
            contract["perturbations"]["observables"]["polarization_b"] = {
                "kind": "transfer_component",
                "projection": "custom_line_of_sight",
                "kernel": "spin2_b_window",
                "source_terms": {"polarization_b": "polarization_b_source"},
                "required_projection_roles": ["b_mode"],
            }
        changed["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "1.25 * visibility * tensor_b"
        ells = numpy.arange(20, 36, dtype=int)
        baseline_bb = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("BB",),
            ),
            dtype=float,
        )
        changed_bb = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                changed,
                ells,
                spectra=("BB",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            changed_bb / baseline_bb,
            numpy.full_like(baseline_bb, 1.25 * 1.25),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=(
                "Custom B-mode kernels should preserve the exact quadratic "
                "BB response to declared odd-parity source scaling."
            ),
        )


class CMBCustomRuntimeBehaviorTestCase(unittest.TestCase):
    """Fast runtime-response coverage for declared-graph execution."""

    def test_source_file_does_not_contain_fake_or_legacy_hacks(self) -> None:
        """The production module should not contain old compatibility code."""

        source_text = "\n".join(
            Path(module.__file__).read_text(encoding="utf-8")
            for module in (
                native_cmb_solver,
                native_background,
                native_evolution,
                native_projection,
            )
        )
        for needle in (
            "equation_mode",
            "mapped_sector",
            "declared_equations",
            "source_normalization",
            "transfer_amplitude",
            "angular_projection_scale",
            "_evolve_custom_cmb_mode_histories",
            "_CUSTOM_CMB_SOURCE_CHANNELS",
            "_CUSTOM_CMB_SECTOR_ALIASES",
            "_classify_custom_physical_sector",
            "visibility shift",
            "visibility rescale",
            "_smooth_transition",
            "_gaussian_smooth_spectrum",
            "_assemble_approximate_lensed_spectra",
            "Keep the declared PP response visible",
        ):
            self.assertNotIn(needle, source_text)

    def test_custom_graph_runs_and_transfer_payloads_are_finite(self) -> None:
        """Transfer components and declared spectra should stay finite."""

        contract = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        ells = numpy.arange(20, 45, dtype=int)
        spectrum_data = native_projection._compute_custom_cmb_spectrum_data(
            contract,
            ells,
        )

        self.assertIsInstance(
            spectrum_data,
            native_projection.CustomCMBSpectrumData,
        )
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

        contract = _speedup_contract(_custom_contract())
        ells = numpy.arange(20, 90, dtype=int)
        base = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("TT", "TE", "EE"),
        )
        hi_as_contract = _speedup_contract(_custom_contract())
        hi_as_contract["param_map"]["As"] = 4.2e-9
        hi_as = cmb.compute_cmb_spectrum_from_contract(
            hi_as_contract,
            ells,
            spectra=("TT",),
        )
        hi_h0_contract = _speedup_contract(_custom_contract())
        hi_h0_contract["param_map"]["H0"] = 74.0
        hi_h0 = cmb.compute_cmb_spectrum_from_contract(
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
        self.assertGreater(
            float(numpy.max(numpy.abs(base_te))),
            0.0,
        )
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

    def test_primordial_tilt_changes_temperature_shape(self) -> None:
        """Primordial tilt should reshape the declared temperature spectrum."""

        ells = numpy.arange(20, 90, dtype=int)
        low_ns_contract = _speedup_contract(_custom_contract())
        high_ns_contract = _speedup_contract(_custom_contract())
        low_ns_contract["param_map"]["ns"] = 0.85
        high_ns_contract["param_map"]["ns"] = 1.15
        low_ns_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                low_ns_contract,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        high_ns_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                high_ns_contract,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        low_shape = float(
            numpy.mean(numpy.abs(low_ns_tt[ells >= 60]))
            / max(numpy.mean(numpy.abs(low_ns_tt[ells <= 35])), 1.0e-12)
        )
        high_shape = float(
            numpy.mean(numpy.abs(high_ns_tt[ells >= 60]))
            / max(numpy.mean(numpy.abs(high_ns_tt[ells <= 35])), 1.0e-12)
        )
        self.assertGreater(high_shape, low_shape)

    def test_reionization_tau_changes_background_and_temperature(self) -> None:
        """The physical reionization ODE should feed the spectrum response."""

        low_tau_contract = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        high_tau_contract = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        low_tau_contract["param_map"]["tau"] = 0.03
        high_tau_contract["param_map"]["tau"] = 0.08
        low_physical = (
            native_background._resolve_custom_cmb_physical_parameters(
                low_tau_contract
            )
        )
        high_physical = (
            native_background._resolve_custom_cmb_physical_parameters(
                high_tau_contract
            )
        )
        numerics = native_background._resolve_custom_cmb_numerics(
            low_tau_contract
        )
        low_background = native_background._build_custom_cmb_background(
            low_tau_contract,
            low_physical,
            numerics,
        )
        high_background = native_background._build_custom_cmb_background(
            high_tau_contract,
            high_physical,
            numerics,
        )
        z_probe = 8.0
        low_probe = float(
            low_background.x_e_grid[
                numpy.argmin(numpy.abs(low_background.z_grid - z_probe))
            ]
        )
        high_probe = float(
            high_background.x_e_grid[
                numpy.argmin(numpy.abs(high_background.z_grid - z_probe))
            ]
        )
        ells = numpy.arange(20, 30, dtype=int)
        low_tau_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                low_tau_contract,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        high_tau_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                high_tau_contract,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        self.assertLess(
            abs(low_background.reionization_tau - low_physical.tau_reio),
            0.01,
        )
        self.assertLess(
            abs(high_background.reionization_tau - high_physical.tau_reio),
            0.01,
        )
        self.assertGreater(high_probe, low_probe)
        self.assertGreater(
            float(numpy.max(numpy.abs(high_tau_tt - low_tau_tt))),
            1.0e-12,
        )

    def test_bb_and_lensing_targets_run_when_declared(self) -> None:
        """Additional observable targets should run through the graph."""

        contract = _speedup_contract(
            _custom_contract(include_bb=True, include_lensing=True)
        )
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("TT", "BB", "PP"),
        )

        self.assertEqual(set(spectra), {"TT", "BB", "PP"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["BB"], dtype=float)))
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["PP"], dtype=float)))
            ),
            0.0,
        )

    def test_lensing_cross_targets_run_when_declared(self) -> None:
        """Temperature and E-mode lensing cross terms should run natively."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("TP", "EP", "PP"),
        )

        self.assertEqual(set(spectra), {"TP", "EP", "PP"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["TP"], dtype=float)))
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(numpy.abs(numpy.asarray(spectra["EP"], dtype=float)))
            ),
            0.0,
        )

    def test_vector_sector_targets_run_when_declared(self) -> None:
        """Vector-like transfer components should run natively."""

        contract = _speedup_contract(_custom_contract(include_vector=True))
        ells = numpy.arange(20, 45, dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("VV",),
        )

        self.assertTrue(numpy.all(numpy.isfinite(spectra)))
        self.assertGreater(
            float(numpy.max(numpy.abs(numpy.asarray(spectra, dtype=float)))),
            0.0,
        )

    def test_sector_mismatch_cross_spectrum_fails_before_runtime(self) -> None:
        """Mixed scalar and vector transfer spectra should fail early."""

        contract = _speedup_contract(_custom_contract(include_vector=True))
        contract["perturbations"]["observables"]["TV"] = {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "vector_signal",
        }

        with self.assertRaisesRegex(
            ValueError,
            "incompatible sectors",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TV",),
            )

    def test_lensed_spectra_change_with_declared_lensing_strength(
        self,
    ) -> None:
        """Exact native lensed outputs should respond to PP strength."""

        baseline = _speedup_contract(_custom_contract(include_lensing=True))
        changed = _speedup_contract(_custom_contract(include_lensing=True))
        baseline["numerical"]["k_sample_count"] = 16
        changed["numerical"]["k_sample_count"] = 16
        baseline["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.0e4 * exp(-tau) * (Phi + Psi)"
        changed["perturbations"]["sources"]["lensing_potential"][
            "expression"
        ] = "1.6 * 1.0e4 * exp(-tau) * (Phi + Psi)"
        ells = numpy.arange(20, 60, dtype=int)
        baseline_unlensed = cmb.compute_cmb_spectrum_from_contract(
            baseline,
            ells,
            spectra=("TT", "EE", "TE"),
        )
        baseline_lensed = cmb.compute_cmb_spectrum_from_contract(
            baseline,
            ells,
            spectra=(
                "PP",
                "lensed_TT",
                "lensed_EE",
                "lensed_TE",
                "lensed_BB",
            ),
        )
        changed_lensed = cmb.compute_cmb_spectrum_from_contract(
            changed,
            ells,
            spectra=(
                "PP",
                "lensed_TT",
                "lensed_EE",
                "lensed_TE",
                "lensed_BB",
            ),
        )

        self.assertEqual(
            set(baseline_lensed),
            {"PP", "lensed_TT", "lensed_EE", "lensed_TE", "lensed_BB"},
        )
        for values in baseline_lensed.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            baseline_lensed["lensed_BB"], dtype=float
                        )
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            baseline_lensed["lensed_TT"],
                            dtype=float,
                        )
                        - numpy.asarray(baseline_unlensed["TT"], dtype=float)
                    )
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        numpy.asarray(changed_lensed["PP"], dtype=float)
                        - numpy.asarray(baseline_lensed["PP"], dtype=float)
                    )
                )
            ),
            0.0,
        )

    def test_projection_kernel_batches_reuse_across_scalar_rebinds(
        self,
    ) -> None:
        """Projection-kernel caches should survive scalar parameter rebinds."""

        native_cache.clear_native_cmb_caches()
        baseline = _prepare_native_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted = _prepare_native_contract(
            _speedup_contract(_custom_contract(include_lensing=True))
        )
        shifted["param_map"]["As"] *= 1.1
        ells = numpy.arange(20, 45, dtype=int)
        cmb.compute_cmb_spectrum_from_contract(
            baseline,
            ells,
            spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )
        first_stats = native_cache.native_cmb_cache_stats()[
            "declared_projection_kernel_batch"
        ]
        cmb.compute_cmb_spectrum_from_contract(
            shifted,
            ells,
            spectra=("TT", "TE", "EE", "PP", "TP", "EP"),
        )
        second_stats = native_cache.native_cmb_cache_stats()[
            "declared_projection_kernel_batch"
        ]

        self.assertEqual(second_stats["entries"], first_stats["entries"])
        self.assertGreater(second_stats["hits"], first_stats["hits"])

    def test_bb_requires_declared_b_mode_transfer_component(self) -> None:
        """BB should fail clearly when no odd-parity transfer is declared."""

        contract = _speedup_contract(_custom_contract())
        with self.assertRaisesRegex(
            ValueError,
            "does not provide requested spectra",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("BB",),
            )

    def test_start_boundary_conditions_can_seed_missing_state(self) -> None:
        """Start-anchored boundary conditions should seed the native solver."""

        contract = _speedup_contract(_custom_contract())
        theta_b_seed = contract["perturbations"]["initial_conditions"].pop(
            "theta_b_seed"
        )
        theta_b_seed["anchor"] = "start"
        boundary_conditions = contract["perturbations"]["boundary_conditions"]
        boundary_conditions["theta_b_start"] = theta_b_seed
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_initial_conditions_can_resolve_relation_targets(self) -> None:
        """Initial-condition expressions may depend on solved relations."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["initial_conditions"]["theta_b_seed"][
            "expression"
        ] = "(k * eta_initial / 6.0) * seed + 0.25 * Phi"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_relation_target_derivative_sources_run(self) -> None:
        """Array-valued sources may differentiate algebraic metric targets."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["derived"]["Phi_tau"] = {
            "kind": "derivative_symbol",
            "variable": "Phi",
            "wrt": "tau",
            "order": 1,
        }
        contract["perturbations"]["derived"]["Psi_tau"] = {
            "kind": "derivative_symbol",
            "variable": "Psi",
            "wrt": "tau",
            "order": 1,
        }
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "0.02 * (Psi_tau - Phi_tau)"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_end_boundary_conditions_can_seed_missing_state(self) -> None:
        """End-anchored boundary conditions should drive the shooter."""

        contract = _analytic_signal_contract(decay_rate=1.0e-4)
        contract["perturbations"]["initial_conditions"].pop("signal_seed")
        contract["perturbations"]["boundary_conditions"]["signal_end"] = {
            "target": {
                "variable": "signal_mode",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.5",
            "anchor": "end",
        }
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_declared_gauge_metadata_is_not_restricted(self) -> None:
        """The native graph solver should accept declared gauge metadata."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["gauge"] = "synchronous"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_declared_background_symbols_feed_native_equations(self) -> None:
        """Declared background symbols should flow into perturbation math."""

        baseline = _speedup_contract(_custom_contract())
        changed = _speedup_contract(_custom_contract())
        baseline["background"]["derived"]["metric_drive"] = "0.25 * Omega_b0"
        changed["background"]["derived"]["metric_drive"] = "0.75 * Omega_b0"
        baseline_baryon_equation = baseline["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        changed_baryon_equation = changed["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        baseline_baryon_equation["rhs"] += " + metric_drive * k * k * Psi"
        changed_baryon_equation["rhs"] += " + metric_drive * k * k * Psi"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
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

    def test_declared_collision_operator_changes_temperature_response(
        self,
    ) -> None:
        """Collision operators should alter the compiled drag response."""

        baseline = _speedup_contract(_native_scalar_hierarchy_contract())
        changed = _speedup_contract(_native_scalar_hierarchy_contract())
        changed["perturbations"]["collision_operators"]["thomson_drag"][
            "expression"
        ] = (
            "collision_rate * "
            "(theta_b / 3.0 - theta_gamma1) + 0.25 * theta_gamma0"
        )
        baseline = _prepare_native_contract(baseline)
        changed = _prepare_native_contract(changed)
        baseline_thomson_drag = baseline[
            "perturbation_data"
        ].collision_operators["thomson_drag"]
        changed_thomson_drag = changed[
            "perturbation_data"
        ].collision_operators["thomson_drag"]
        context = {
            "photon_baryon_momentum_ratio": 1.5,
            "theta_b": 0.12,
            "theta_gamma0": 0.31,
            "theta_gamma1": 0.08,
            "collision_rate": 0.75,
        }
        baseline_drag = evaluate_compiled_expression(
            baseline_thomson_drag.compiled_expression,
            context,
        )
        changed_drag = evaluate_compiled_expression(
            changed_thomson_drag.compiled_expression,
            context,
        )
        self.assertGreater(
            float(numpy.abs(changed_drag - baseline_drag)),
            1.0e-12,
        )
        baseline_baryon_drag = evaluate_compiled_expression(
            baseline["perturbation_data"]
            .derived["baryon_thomson_drag"]
            .compiled_expression,
            {
                **context,
                "thomson_drag": baseline_drag,
            },
        )
        changed_baryon_drag = evaluate_compiled_expression(
            changed["perturbation_data"]
            .derived["baryon_thomson_drag"]
            .compiled_expression,
            {
                **context,
                "thomson_drag": changed_drag,
            },
        )
        self.assertGreater(
            float(numpy.abs(changed_baryon_drag - baseline_baryon_drag)),
            1.0e-12,
        )

    def test_multiple_declared_coordinates_preserve_runtime_response(
        self,
    ) -> None:
        """Mixed `tau`/`a` equations should run through the transform."""

        baseline = _analytic_signal_contract(decay_rate=0.025)
        transformed = _analytic_signal_contract(decay_rate=0.025)
        transformed_equation = transformed["perturbations"]["equations"][
            "evolve_signal_mode"
        ]
        transformed_equation["lhs"]["wrt"] = "a"
        transformed_equation["rhs"] = (
            f"({transformed_equation['rhs']})"
            + " * (299792.458 / (a * a * H))"
        )
        transformed["perturbations"]["initial_conditions"]["signal_seed"][
            "target"
        ]["wrt"] = "a"
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        transformed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                transformed,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        numpy.testing.assert_allclose(
            transformed_tt,
            baseline_tt,
            rtol=1.0e-2,
            atol=1.0e-10,
        )

    def test_generic_background_aliases_run_without_lcdm_named_inputs(
        self,
    ) -> None:
        """Generic background aliases should supply the native solver."""

        contract = _speedup_contract(_generic_background_custom_contract())
        physical = native_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT", "TE", "EE"),
        )
        self.assertTrue(physical.has_cdm)
        self.assertIsNone(physical.Neff)
        self.assertAlmostEqual(physical.Omega_c0, 0.262)
        self.assertAlmostEqual(physical.primordial_amplitude, 2.1e-9)
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_physical_density_inputs_run_without_lcdm_density_aliases(
        self,
    ) -> None:
        """Direct physical densities should drive the native solver."""

        contract = _speedup_contract(_physical_density_custom_contract())
        physical = native_background._resolve_custom_cmb_physical_parameters(
            contract
        )
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT", "TE", "EE"),
        )
        self.assertTrue(physical.has_cdm)
        self.assertAlmostEqual(physical.Omega_b0, 0.049, places=6)
        self.assertAlmostEqual(physical.Omega_c0, 0.262, places=6)
        self.assertAlmostEqual(
            physical.rho_b0_kg_m3,
            float(contract["param_map"]["baryon_rest_mass_density_today"]),
            places=20,
        )
        self.assertAlmostEqual(
            physical.rho_c0_kg_m3,
            float(
                contract["param_map"][
                    "cold_dark_matter_rest_mass_density_today"
                ]
            ),
            places=20,
        )
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_native_source_refinement_above_two_changes_runtime_grid(
        self,
    ) -> None:
        """Source refinement above two should stay active at runtime."""

        baseline = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        refined = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        baseline["numerical"]["source_grid_multiplier"] = 3
        refined["numerical"]["source_grid_multiplier"] = 4
        ells = numpy.arange(20, 30, dtype=int)
        baseline_data = native_projection._compute_custom_cmb_spectrum_data(
            baseline,
            ells,
        )
        refined_data = native_projection._compute_custom_cmb_spectrum_data(
            refined,
            ells,
        )
        self.assertGreater(
            int(refined_data.runtime_envelope["eta_sample_count"]),
            int(baseline_data.runtime_envelope["eta_sample_count"]),
        )

    def test_native_eta_sample_count_changes_background_resolution(
        self,
    ) -> None:
        """Declared `eta_sample_count` should affect the native grid."""

        coarse = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        refined = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )
        coarse["numerical"]["eta_sample_count"] = 128
        refined["numerical"]["eta_sample_count"] = 256
        coarse_background = native_background._build_custom_cmb_background(
            coarse,
            native_background._resolve_custom_cmb_physical_parameters(coarse),
            native_background._resolve_custom_cmb_numerics(coarse),
        )
        refined_background = native_background._build_custom_cmb_background(
            refined,
            native_background._resolve_custom_cmb_physical_parameters(refined),
            native_background._resolve_custom_cmb_numerics(refined),
        )
        self.assertGreater(
            int(refined_background.eta_grid.size),
            int(coarse_background.eta_grid.size),
        )

    def test_requested_native_k_sample_count_is_not_capped(self) -> None:
        """Native `k_sample_count` should honor declared values above 48."""

        contract = _prepare_native_contract(
            _speedup_contract(_analytic_signal_contract())
        )
        contract["numerical"]["k_sample_count"] = 64
        spectrum_data = native_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )
        self.assertEqual(int(spectrum_data.k_grid.size), 64)

    def test_accuracy_controls_reject_underresolved_scalar_numerics(
        self,
    ) -> None:
        """Declared minimum controls should reject under-resolved numerics."""

        base_contract = _speedup_contract(_custom_contract())
        base_contract["numerical"].update(
            {
                "ell_max": 40,
                "photon_hierarchy_l_max": 2,
                "neutrino_hierarchy_l_max": 2,
            }
        )
        cases = (
            ("minimum_ell_max", 61, "ell_max"),
            ("minimum_k_sample_count", 17, "k_sample_count"),
            ("minimum_eta_sample_count", 129, "eta_sample_count"),
            ("minimum_source_grid_multiplier", 2, "source_grid_multiplier"),
            (
                "minimum_photon_hierarchy_l_max",
                3,
                "photon_hierarchy_l_max",
            ),
            (
                "minimum_neutrino_hierarchy_l_max",
                3,
                "neutrino_hierarchy_l_max",
            ),
        )

        for control_name, minimum_value, message in cases:
            contract = copy.deepcopy(base_contract)
            contract["perturbations"]["accuracy_controls"] = {
                control_name: minimum_value,
            }
            with self.subTest(control_name=control_name):
                with self.assertRaisesRegex(ValueError, message):
                    native_background._resolve_custom_cmb_numerics(contract)

    def test_runtime_envelope_records_governed_work_units(self) -> None:
        """Native spectra should carry the governed runtime envelope."""

        contract = _speedup_contract(_analytic_signal_contract())
        contract["perturbations"]["accuracy_controls"] = {
            "runtime_envelope": {
                "maximum_total_work_units": 200000,
            }
        }
        contract = _prepare_native_contract(contract)
        spectrum_data = native_projection._compute_custom_cmb_spectrum_data(
            contract,
            numpy.arange(20, 25, dtype=int),
        )

        self.assertGreater(
            int(spectrum_data.runtime_envelope["total_work_units"]),
            0,
        )
        self.assertLessEqual(
            int(spectrum_data.runtime_envelope["total_work_units"]),
            200000,
        )
        self.assertIn("evolution_work_units", spectrum_data.runtime_envelope)
        self.assertIn("projection_work_units", spectrum_data.runtime_envelope)

    def test_runtime_envelope_rejects_unbounded_work_units(self) -> None:
        """Declared runtime envelopes should fail before large runs start."""

        contract = _speedup_contract(_analytic_signal_contract())
        contract["perturbations"]["accuracy_controls"] = {
            "runtime_envelope": {
                "maximum_total_work_units": 1000,
            }
        }
        contract = _prepare_native_contract(contract)
        with self.assertRaisesRegex(
            ValueError,
            "runtime_envelope exceeded maximum_total_work_units",
        ):
            native_projection._compute_custom_cmb_spectrum_data(
                contract,
                numpy.arange(20, 25, dtype=int),
            )

    def test_conservation_rule_violation_fails_loudly(self) -> None:
        """Conservation residuals should fail once they exceed tolerance."""

        contract = _analytic_signal_contract()
        contract["perturbations"]["conservation_rules"] = {
            "signal_balance": {
                "kind": "absolute_max",
                "expression": "signal_mode",
                "tolerance": 1.0e-12,
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "conservation rule exceeded tolerance",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_declared_recombination_quantities_change_background(
        self,
    ) -> None:
        """Declared recombination hooks should alter the solved background."""

        def _recombination_contract(peebles_c: float) -> dict[str, object]:
            contract = _speedup_contract(_custom_contract())
            contract["background"]["recombination"] = {
                "quantities": {
                    "hydrogen_temperature_K": "2.7255 * (1.0 + z)",
                    "hydrogen_alpha_B": (
                        "1.0e-19 * "
                        "((hydrogen_temperature_K / 3000.0) ** -0.5)"
                    ),
                    "beta_continuum": (
                        "5.0e-20 * "
                        "((hydrogen_temperature_K / 3000.0) ** 0.5)"
                    ),
                    "peebles_c": f"{peebles_c:.16g}",
                }
            }
            return _prepare_native_contract(contract)

        baseline = _recombination_contract(0.4)
        changed = _recombination_contract(0.9)
        baseline_background = native_background._build_custom_cmb_background(
            baseline,
            native_background._resolve_custom_cmb_physical_parameters(
                baseline
            ),
            native_background._resolve_custom_cmb_numerics(baseline),
        )
        changed_background = native_background._build_custom_cmb_background(
            changed,
            native_background._resolve_custom_cmb_physical_parameters(changed),
            native_background._resolve_custom_cmb_numerics(changed),
        )

        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(
                        changed_background.x_e_grid
                        - baseline_background.x_e_grid
                    )
                )
            ),
            1.0e-8,
        )

    def test_declared_recombination_quantities_require_full_hook_set(
        self,
    ) -> None:
        """Partial recombination hooks should fail with a named error."""

        contract = _speedup_contract(_custom_contract())
        contract["background"]["recombination"] = {
            "quantities": {
                "hydrogen_temperature_K": "3000.0",
            }
        }
        with self.assertRaisesRegex(
            ValueError,
            "must define 'hydrogen_alpha_B'",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 25, dtype=int),
                spectra=("TT",),
            )

    def test_momentum_grid_accuracy_control_rejects_short_grid(
        self,
    ) -> None:
        """Massive-neutrino grids should honor declared count floors."""

        contract = _speedup_contract(
            _native_scalar_hierarchy_contract(
                include_massive_neutrino=True,
            )
        )
        contract["perturbations"]["numerics"]["momentum_grids"] = {
            "massive_neutrino_default": {
                "count": 4,
            }
        }
        contract["perturbations"]["accuracy_controls"][
            "minimum_momentum_grid_count"
        ] = {
            "massive_neutrino_default": 5,
        }
        with self.assertRaisesRegex(
            ValueError,
            "momentum_grids.massive_neutrino_default.count >= 5",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                _prepare_native_contract(contract),
                numpy.arange(20, 24, dtype=int),
                spectra=("TT",),
            )

    def test_background_pressure_and_curvature_symbols_change_outputs(
        self,
    ) -> None:
        """Pressure/equation-of-state background outputs should feed math."""

        baseline = _speedup_contract(_generic_background_custom_contract())
        changed = _speedup_contract(_generic_background_custom_contract())
        source_adjustment = (
            " + 0.05 * (dark_energy_pressure_today + Omega_k0) * k * Psi"
        )
        baseline_equations = baseline["perturbations"]["equations"]
        changed_equations = changed["perturbations"]["equations"]
        baseline_equations["evolve_theta_b"]["rhs"] += source_adjustment
        changed_equations["evolve_theta_b"]["rhs"] += source_adjustment
        changed_background = changed["background"]["derived"]
        changed_background["dark_energy_pressure_today"] = (
            "1.4 * w0 * Omega_de0"
        )
        ells = numpy.arange(20, 30, dtype=int)
        baseline_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
                baseline,
                ells,
                spectra=("TT",),
            ),
            dtype=float,
        )
        changed_tt = numpy.asarray(
            cmb.compute_cmb_spectrum_from_contract(
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

    def test_missing_declared_background_h_fails_loudly(self) -> None:
        """Declared native contracts must provide the background H."""

        contract = _speedup_contract(_custom_contract())
        contract["background"]["derived"].pop("H", None)
        with self.assertRaisesRegex(
            ValueError,
            "must provide a derived expansion history",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_incomplete_background_declarations_fail_early(self) -> None:
        """Missing required physical background inputs should fail early."""

        contract = _speedup_contract(_generic_background_custom_contract())
        contract["background"]["derived"].pop("Omega_b0", None)
        contract["background"]["derived"].pop("Omega_m0", None)
        contract["param_map"].pop("baryon_fraction_today", None)
        derived_background = contract["background"]["derived"]
        derived_background["matter_total"] = "0.311"
        derived_background["Omega_c0"] = "matter_total"
        derived_background["Omega_de0"] = (
            "1.0 - matter_total - Omega_r0 - Omega_k0"
        )
        derived_background["H"] = (
            "H0 * sqrt("
            "Omega_r0 / (a ** 4) + "
            "matter_total / (a ** 3) + "
            "Omega_k0 / (a ** 2) + "
            "Omega_de0 * (a ** (-3.0 * (1.0 + w0)))"
            ")"
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires explicit baryon density",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_invalid_background_curvature_override_fails_loudly(self) -> None:
        """Invalid declared curvature histories should fail before runtime."""

        contract = _speedup_contract(_generic_background_custom_contract())
        derived_background = contract["background"]["derived"]
        derived_background["chi"] = "z"
        derived_background["angular_diameter_distance"] = "-0.5 * z"
        with self.assertRaisesRegex(
            ValueError,
            "curvature histories must stay non-negative",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_multi_hop_derived_context_resolves_fully(self) -> None:
        """Derived chains should resolve across more than one dependency."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["derived"]["temperature_drive_mid"] = {
            "expression": "theta_gamma0 + Phi",
        }
        contract["perturbations"]["derived"]["temperature_drive_outer"] = {
            "expression": "temperature_drive_mid + Psi",
        }
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "0.15 * temperature_drive_outer"
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            numpy.arange(20, 30, dtype=int),
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(spectra)))

    def test_missing_initial_conditions_fail_loudly(self) -> None:
        """Missing initial conditions should fail before evolution."""

        contract = _speedup_contract(_custom_contract())
        del contract["perturbations"]["initial_conditions"]["theta_b_seed"]
        with self.assertRaisesRegex(
            ValueError,
            "missing required initial conditions",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_missing_observable_mappings_fail_loudly(self) -> None:
        """Missing observable mappings should fail clearly."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["observables"] = {}
        with self.assertRaisesRegex(ValueError, "must declare observables"):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_unsupported_projection_fails_loudly(self) -> None:
        """Unsupported projections should be rejected before evolution."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["observables"]["temperature"][
            "projection"
        ] = "bogus_projection"
        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_missing_projection_source_role_fails_loudly(self) -> None:
        """Projection-role mismatches should fail during compilation."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["observables"]["polarization_e"][
            "source_terms"
        ] = {"signal": "polarization_source"}
        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("EE",),
            )

    def test_b_mode_projection_requires_odd_parity_source(self) -> None:
        """BB should fail before runtime on scalar-like B-source plumbing."""

        contract = _speedup_contract(_custom_contract(include_bb=True))
        contract["perturbations"]["sources"]["polarization_b_source"][
            "expression"
        ] = "visibility * theta_gamma2"
        with self.assertRaisesRegex(
            ValueError,
            "requires an odd-parity declared source ancestry",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("BB",),
            )

    def test_custom_b_mode_projection_rejects_scalar_multi_source(
        self,
    ) -> None:
        """Custom B-mode kernels should reject scalar-only extra sources."""

        contract = _speedup_contract(_custom_contract(include_bb=True))
        contract["perturbations"]["sources"]["polarization_aux_source"] = {
            "expression": "visibility * theta_gamma2",
            "role": "polarization_aux",
        }
        contract["perturbations"]["observables"]["polarization_b"] = {
            "kind": "transfer_component",
            "projection": "custom_line_of_sight",
            "kernel": "spin2_b_window",
            "source_terms": {
                "polarization_b": "polarization_b_source",
                "polarization_aux": "polarization_aux_source",
            },
            "required_projection_roles": ["b_mode"],
        }
        with self.assertRaisesRegex(
            ValueError,
            "requires source 'polarization_aux_source' to provide declared "
            "projection roles: b_mode",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("BB",),
            )

    def test_lensing_projection_requires_potential_role(self) -> None:
        """PP should fail before runtime without a declared potential role."""

        contract = _speedup_contract(_custom_contract(include_lensing=True))
        contract["perturbations"]["observables"]["lensing_potential"][
            "source_terms"
        ] = {"signal": "lensing_potential"}
        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles: potential",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("PP",),
            )

    def test_nonfinite_expression_results_fail_loudly(self) -> None:
        """Non-finite source expressions should fail clearly."""

        contract = _speedup_contract(_custom_contract())
        contract["perturbations"]["sources"]["temperature_additive"][
            "expression"
        ] = "sqrt(-1)"
        with self.assertRaisesRegex(
            ValueError,
            "Declared source term produced non-finite values",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_nonfinite_evolution_states_fail_loudly(self) -> None:
        """Non-finite evolution states should fail clearly."""

        contract = _speedup_contract(_custom_contract())
        theta_b_equation = contract["perturbations"]["equations"][
            "evolve_theta_b"
        ]
        theta_b_equation["rhs"] = "1.0 / (delta_b - delta_b)"
        with self.assertRaisesRegex(
            ValueError,
            "must be finite",
        ):
            cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.arange(20, 30, dtype=int),
                spectra=("TT",),
            )

    def test_custom_cached_path_does_not_call_camb(self) -> None:
        """The cached plugin route should not use the standard CAMB path."""

        plugin = _CustomCMBPlugin()
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch.object(
            camb_solver,
            "_compute_cmb_spectrum_direct",
            side_effect=AssertionError("standard CAMB path should not run"),
        ):
            with mock.patch.object(
                camb_solver.camb,
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

    def test_custom_cached_path_uses_precompiled_native_runtime(self) -> None:
        """The cached route should reuse precompiled native runtime data."""

        contract = _prepare_native_contract(
            _speedup_contract(_custom_contract())
        )

        class _PrecompiledRuntimePlugin(_CustomCMBPlugin):
            """Plugin stub exposing one precompiled native runtime."""

            def get_cmb_native_runtime(self, _params):
                return contract

        plugin = _PrecompiledRuntimePlugin()
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            side_effect=AssertionError(
                "native runtime should reuse precompiled perturbation data"
            ),
        ):
            result = cmb.compute_cmb_spectrum_cached(
                plugin,
                plugin.INITIAL_GUESSES,
                ells,
                spectra=("TT",),
            )

        self.assertTrue(numpy.all(numpy.isfinite(result)))
        self.assertEqual(result.shape, (ells.size,))

    def test_native_scalar_hierarchy_materializes_generated_hierarchy(
        self,
    ) -> None:
        """The native scalar route should compile hierarchy data."""

        contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]

        self.assertTrue(
            perturbation_data.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertIn("theta_gamma8", perturbation_data.variables)
        self.assertIn("e_gamma8", perturbation_data.variables)
        self.assertIn("nu_l5", perturbation_data.variables)
        self.assertIn("TT", perturbation_data.observables)
        self.assertIn("TE", perturbation_data.observables)
        self.assertIn("EE", perturbation_data.observables)
        self.assertIn("PP", perturbation_data.observables)
        self.assertIn("TP", perturbation_data.observables)
        self.assertIn("EP", perturbation_data.observables)
        self.assertEqual(
            perturbation_data.derived["polarization_moment"].expression,
            "theta_gamma2 + 6.0 * e_gamma2",
        )

    def test_native_scalar_hierarchy_materializes_massive_neutrinos(
        self,
    ) -> None:
        """The native scalar route should compile massive neutrinos."""

        contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        perturbation_data = contract["perturbation_data"]

        self.assertIn("delta_nu_massive", perturbation_data.variables)
        self.assertIn("theta_nu_massive", perturbation_data.variables)
        self.assertIn("sigma_nu_massive", perturbation_data.variables)
        self.assertIn("nu_massive_l5", perturbation_data.variables)
        self.assertEqual(
            perturbation_data.hierarchy_families[
                "massive_neutrino"
            ].momentum_grid,
            "massive_neutrino_default",
        )

    def test_native_scalar_hierarchy_materializes_massive_neutrino_q_bins(
        self,
    ) -> None:
        """Massive-neutrino q bins should materialize resolved states."""

        contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        perturbation_data = contract["perturbation_data"]

        self.assertIn("delta_nu_massive_q0", perturbation_data.variables)
        self.assertIn("theta_nu_massive_q0", perturbation_data.variables)
        self.assertIn("sigma_nu_massive_q0", perturbation_data.variables)
        self.assertIn("nu_massive_q0_l5", perturbation_data.variables)

    def test_native_scalar_hierarchy_materializer_uses_q_weights(
        self,
    ) -> None:
        """Generated q bins should keep the physical momentum weights."""

        contract = _prepare_native_contract(
            _strip_native_runtime_sections(
                _native_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                )
            )
        )
        perturbation_data = contract["perturbation_data"]

        self.assertTrue(
            perturbation_data.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertIn(
            "massive_neutrino_q0_weight * delta_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_density_q0"
            ].expression,
        )
        self.assertIn(
            "massive_neutrino_q0_weight * theta_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_momentum_q0"
            ].expression,
        )
        self.assertIn(
            "massive_neutrino_q0_weight * sigma_nu_massive_q0",
            perturbation_data.derived[
                "massive_neutrino_metric_shear_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_mass_fraction",
            perturbation_data.derived[
                "massive_neutrino_metric_density_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_velocity_ratio",
            perturbation_data.derived[
                "massive_neutrino_metric_momentum_q0"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_pressure_ratio",
            perturbation_data.derived[
                "massive_neutrino_metric_shear_q0"
            ].expression,
        )
        self.assertIn(
            "massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_delta_nu_massive_q0"].rhs,
        )
        self.assertNotIn(
            "massive_neutrino_q0_velocity_ratio",
            perturbation_data.equations["evolve_delta_nu_massive_q0"].rhs,
        )
        self.assertIn(
            "massive_neutrino_q0_streaming_speed",
            perturbation_data.equations["evolve_theta_nu_massive_q0"].rhs,
        )
        self.assertIn(
            "massive_neutrino_streaming_speed",
            perturbation_data.equations["evolve_nu_massive_l3"].rhs,
        )
        self.assertNotIn(
            "massive_neutrino_velocity_ratio",
            perturbation_data.equations["evolve_nu_massive_l3"].rhs,
        )
        self.assertIn(
            "massive_neutrino_metric_momentum",
            perturbation_data.derived["total_momentum_density"].expression,
        )
        self.assertEqual(
            perturbation_data.derived["baryon_thomson_drag"].expression,
            "- photon_baryon_momentum_ratio * thomson_drag",
        )
        self.assertIn(
            "thomson_drag_balance",
            perturbation_data.conservation_rules,
        )
        self.assertEqual(
            perturbation_data.collision_operators["thomson_drag"].counterpart,
            "baryon_thomson_drag",
        )
        self.assertEqual(
            perturbation_data.conservation_rules[
                "thomson_drag_balance"
            ].expression,
            "photon_baryon_momentum_ratio * thomson_drag + "
            "baryon_thomson_drag",
        )
        self.assertIn(
            "Phi_tau",
            perturbation_data.equations["evolve_delta_b"].rhs,
        )
        self.assertIn(
            "baryon_thomson_drag",
            perturbation_data.equations["evolve_theta_b"].rhs,
        )
        self.assertIn(
            "theta_gamma3",
            perturbation_data.equations["evolve_theta_gamma2"].rhs,
        )
        self.assertIn(
            "collision_rate",
            perturbation_data.equations["evolve_theta_gamma2"].rhs,
        )
        self.assertIn(
            "e_gamma3",
            perturbation_data.equations["evolve_e_gamma2"].rhs,
        )
        self.assertIn(
            "collision_rate",
            perturbation_data.equations["evolve_e_gamma2"].rhs,
        )
        self.assertEqual(
            perturbation_data.equations["evolve_e_gamma0"].rhs,
            "-acoustic_k * e_gamma1",
        )
        self.assertNotIn(
            "tight_coupling_drag",
            perturbation_data.equations["evolve_theta_gamma3"].rhs,
        )
        self.assertNotIn(
            "tight_coupling_drag",
            perturbation_data.equations["evolve_e_gamma3"].rhs,
        )
        self.assertNotIn(
            "massive_neutrino_pressure_ratio",
            perturbation_data.equations["evolve_delta_nu_massive"].rhs,
        )
        self.assertIn(
            "massive_neutrino_metric_density",
            perturbation_data.equations["evolve_theta_nu_massive"].rhs,
        )
        self.assertNotIn(
            "massive_neutrino_velocity_ratio",
            perturbation_data.equations["evolve_theta_nu_massive"].rhs,
        )
        self.assertIn(
            "massive_neutrino_metric_momentum",
            perturbation_data.equations["evolve_sigma_nu_massive"].rhs,
        )
        self.assertNotIn(
            "massive_neutrino_velocity_ratio",
            perturbation_data.equations["evolve_sigma_nu_massive"].rhs,
        )
        self.assertIn(
            "acoustic_k * eta_initial / 6.0",
            perturbation_data.initial_conditions[
                "theta_gamma1_seed"
            ].expression,
        )
        self.assertIn(
            "acoustic_k_sq * eta_initial / 2.0",
            perturbation_data.initial_conditions["theta_b_seed"].expression,
        )
        self.assertIn(
            "distribution_log_derivative",
            perturbation_data.initial_conditions[
                "delta_nu_massive_q0_seed"
            ].expression,
        )
        self.assertIn(
            "acoustic_k * eta_initial / 6.0",
            perturbation_data.initial_conditions[
                "theta_nu_massive_q0_seed"
            ].expression,
        )
        self.assertNotIn(
            "massive_neutrino_q0_velocity_ratio",
            perturbation_data.initial_conditions[
                "theta_nu_massive_q0_seed"
            ].expression,
        )

    def test_native_power_spectrum_scale_factor_is_physical(
        self,
    ) -> None:
        """Native spectrum scaling should stay on physical units only."""

        ell_factor = numpy.asarray((2.0, 6.0, 12.0), dtype=numpy.longdouble)
        t_cmb_muK = numpy.longdouble("2.7255e6")

        tt_scale = native_cmb_solver._power_spectrum_scale_factor(
            None,
            "TT",
            ell_factor=ell_factor,
            t_cmb_muK=float(t_cmb_muK),
            lensing_mode=False,
        )
        pp_scale = native_cmb_solver._power_spectrum_scale_factor(
            None,
            "PP",
            ell_factor=ell_factor,
            t_cmb_muK=float(t_cmb_muK),
            lensing_mode=True,
        )

        numpy.testing.assert_allclose(
            numpy.asarray(tt_scale, dtype=numpy.longdouble),
            ell_factor * t_cmb_muK * t_cmb_muK,
        )
        numpy.testing.assert_allclose(
            numpy.asarray(pp_scale, dtype=numpy.longdouble),
            numpy.ones_like(ell_factor, dtype=numpy.longdouble),
        )

    def test_native_scalar_spectrum_aliases_round_trip(self) -> None:
        """Declared phiphi, Tphi, and Ephi aliases should round-trip."""

        contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        ells = numpy.asarray((20, 40, 60, 90, 120), dtype=int)
        spectra = cmb.compute_cmb_spectrum_from_contract(
            contract,
            ells,
            spectra=("PP", "phiphi", "TP", "Tphi", "EP", "Ephi"),
        )

        self.assertEqual(
            set(spectra),
            {"PP", "phiphi", "TP", "Tphi", "EP", "Ephi"},
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["PP"], dtype=float),
            numpy.asarray(spectra["phiphi"], dtype=float),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["TP"], dtype=float),
            numpy.asarray(spectra["Tphi"], dtype=float),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(spectra["EP"], dtype=float),
            numpy.asarray(spectra["Ephi"], dtype=float),
        )

    def test_native_scalar_hierarchy_synchronous_matches_newtonian(
        self,
    ) -> None:
        """Generated synchronous and Newtonian routes should agree."""

        newtonian = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        synchronous = _prepare_native_contract(
            _native_scalar_hierarchy_contract(gauge="synchronous")
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        spectra = (
            "TT",
            "TE",
            "EE",
            "BB",
            "PP",
            "TP",
            "EP",
            "lensed_TT",
            "lensed_TE",
            "lensed_EE",
            "lensed_BB",
        )
        baseline = cmb.compute_cmb_spectrum_from_contract(
            newtonian,
            ells,
            spectra=spectra,
        )
        comparison = cmb.compute_cmb_spectrum_from_contract(
            synchronous,
            ells,
            spectra=spectra,
        )

        for spectrum_name in spectra:
            numpy.testing.assert_allclose(
                numpy.asarray(comparison[spectrum_name], dtype=float),
                numpy.asarray(baseline[spectrum_name], dtype=float),
                rtol=1.0e-10,
                atol=1.0e-10,
            )

    def test_native_scalar_hierarchy_standard_modes_generate_distinct(
        self,
    ) -> None:
        """Generated standard initial-condition modes should change TT."""

        adiabatic = _prepare_native_contract(
            _native_scalar_hierarchy_contract(initial_mode="adiabatic_scalar")
        )
        cdm_mode = _prepare_native_contract(
            _native_scalar_hierarchy_contract(initial_mode="cdm_isocurvature")
        )
        ells = numpy.asarray((20, 30, 40, 60, 90, 120), dtype=int)
        adiabatic_tt = numpy.asarray(
            native_projection._compute_custom_cmb_spectrum_data(
                adiabatic,
                ells,
            ).spectra["TT"],
            dtype=numpy.longdouble,
        )
        cdm_tt = numpy.asarray(
            native_projection._compute_custom_cmb_spectrum_data(
                cdm_mode,
                ells,
            ).spectra["TT"],
            dtype=numpy.longdouble,
        )

        self.assertTrue(numpy.all(numpy.isfinite(adiabatic_tt)))
        self.assertTrue(numpy.all(numpy.isfinite(cdm_tt)))
        self.assertGreater(
            float(numpy.max(numpy.abs(adiabatic_tt - cdm_tt))),
            1.0e-12,
        )
        self.assertEqual(
            cdm_mode["perturbation_data"]
            .initial_conditions["delta_c_seed"]
            .expression,
            "seed",
        )
        self.assertEqual(
            adiabatic["perturbation_data"]
            .initial_conditions["delta_c_seed"]
            .expression,
            "-1.5 * seed",
        )

    def test_native_scalar_hierarchy_massive_neutrino_response(
        self,
    ) -> None:
        """Massive-neutrino momentum grids should affect q-bin physics."""

        light = _prepare_native_contract(
            _speedup_contract(
                _native_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                    sum_mnu=0.06,
                )
            )
        )
        light["numerical"].update(
            {
                "eta_sample_count": 32,
                "k_sample_count": 8,
            }
        )
        light["numerical"]["momentum_grids"]["massive_neutrino_default"][
            "count"
        ] = 4
        heavy = _prepare_native_contract(
            _speedup_contract(
                _native_scalar_hierarchy_contract(
                    include_massive_neutrino=True,
                    sum_mnu=2.0,
                )
            )
        )
        heavy["numerical"].update(
            {
                "eta_sample_count": 32,
                "k_sample_count": 8,
            }
        )
        heavy["numerical"]["momentum_grids"]["massive_neutrino_default"][
            "count"
        ] = 4
        light_physical = (
            native_background._resolve_custom_cmb_physical_parameters(light)
        )
        heavy_physical = (
            native_background._resolve_custom_cmb_physical_parameters(heavy)
        )
        light_q_context = native_evolution._declared_momentum_grid_context(
            light["perturbation_data"],
            model_parameters=light["param_map"],
            physical_params=light_physical,
            scale_factor=0.5,
        )
        heavy_q_context = native_evolution._declared_momentum_grid_context(
            heavy["perturbation_data"],
            model_parameters=heavy["param_map"],
            physical_params=heavy_physical,
            scale_factor=0.5,
        )
        light_streaming_speed = numpy.asarray(
            light_q_context["massive_neutrino_q0_streaming_speed"],
            dtype=float,
        )
        heavy_streaming_speed = numpy.asarray(
            heavy_q_context["massive_neutrino_q0_streaming_speed"],
            dtype=float,
        )
        light_mass_fraction = numpy.asarray(
            light_q_context["massive_neutrino_q0_mass_fraction"],
            dtype=float,
        )
        heavy_mass_fraction = numpy.asarray(
            heavy_q_context["massive_neutrino_q0_mass_fraction"],
            dtype=float,
        )

        self.assertTrue(numpy.isfinite(light_streaming_speed))
        self.assertTrue(numpy.isfinite(heavy_streaming_speed))
        self.assertGreater(
            float(
                numpy.max(
                    numpy.abs(light_streaming_speed - heavy_streaming_speed)
                )
            ),
            1.0e-12,
        )
        self.assertGreater(
            float(
                numpy.max(numpy.abs(light_mass_fraction - heavy_mass_fraction))
            ),
            1.0e-12,
        )

    def test_native_scalar_hierarchy_momentum_grid_cache_reuses(
        self,
    ) -> None:
        """Momentum-grid quadrature should reuse bounded cache entries."""

        native_cache.clear_native_cmb_caches()
        baseline = _prepare_native_contract(
            _native_scalar_hierarchy_contract(include_massive_neutrino=True)
        )
        shifted = _native_scalar_hierarchy_contract(
            include_massive_neutrino=True
        )
        shifted["param_map"]["As"] *= 1.1
        shifted_contract = _prepare_native_contract(shifted)
        ells = numpy.asarray((20, 40, 60), dtype=int)

        cmb.compute_cmb_spectrum_from_contract(
            baseline,
            ells,
            spectra=("TT",),
        )
        first_stats = native_cache.native_cmb_cache_stats()[
            "declared_momentum_grid"
        ]
        cmb.compute_cmb_spectrum_from_contract(
            shifted_contract,
            ells,
            spectra=("TT",),
        )
        second_stats = native_cache.native_cmb_cache_stats()[
            "declared_momentum_grid"
        ]

        self.assertEqual(first_stats["entries"], second_stats["entries"])
        self.assertGreaterEqual(second_stats["hits"], first_stats["hits"] + 1)

    def test_native_scalar_hierarchy_reuses_structural_runtime_bundle(
        self,
    ) -> None:
        """Scalar runtime signatures should survive bound-value changes."""

        baseline = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        shifted = _native_scalar_hierarchy_contract()
        shifted["param_map"]["As"] *= 1.1
        shifted["param_map"]["H0"] = 68.1
        shifted_prepared = _prepare_native_contract(shifted)

        self.assertEqual(
            baseline["runtime_signature"],
            shifted_prepared["runtime_signature"],
        )

    def test_native_scalar_hierarchy_runs_camb_free(self) -> None:
        """The native scalar route should stay CAMB-free at runtime."""

        contract = _prepare_native_contract(
            _native_scalar_hierarchy_contract()
        )
        ells = numpy.arange(20, 60, dtype=int)
        with mock.patch.object(
            camb_solver.camb,
            "get_results",
            side_effect=AssertionError(
                "native scalar hierarchy should not call CAMB"
            ),
        ):
            spectra = cmb.compute_cmb_spectrum_from_contract(
                contract,
                ells,
                spectra=("TT", "TE", "EE"),
            )

        self.assertEqual(set(spectra), {"TT", "TE", "EE"})
        for values in spectra.values():
            self.assertTrue(numpy.all(numpy.isfinite(values)))

    def test_direct_custom_path_does_not_call_camb(self) -> None:
        """The direct declared-graph route should stay CAMB-free."""

        contract = _speedup_contract(_custom_contract())
        ells = numpy.arange(20, 35, dtype=int)
        with mock.patch.object(
            camb_solver.camb,
            "get_results",
            side_effect=AssertionError("CAMB prediction path should not run"),
        ):
            result = cmb.compute_cmb_spectrum_from_contract(
                contract,
                ells,
                spectra=("TT", "TE", "EE"),
            )

        self.assertEqual(set(result), {"TT", "TE", "EE"})
        for spectrum in result.values():
            self.assertTrue(numpy.all(numpy.isfinite(spectrum)))
            self.assertEqual(spectrum.shape, (ells.size,))


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the public CMB helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        """The module should export the expected public helpers."""

        self.assertEqual(
            set(cmb.__all__),
            {
                "CMBLike",
                "compute_camb_background_observables",
                "compute_cmb_spectrum",
                "compute_cmb_spectrum_cached",
                "compute_cmb_spectrum_from_contract",
                "compute_cmb_spectrum_from_legacy_params_for_tests",
                "describe_camb_configuration",
            },
        )
        self.assertTrue(hasattr(cmb, "CMBLike"))
        self.assertTrue(callable(cmb.compute_cmb_spectrum))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_cached))
        self.assertTrue(callable(cmb.compute_cmb_spectrum_from_contract))
        self.assertTrue(callable(cmb.compute_camb_background_observables))
        self.assertTrue(
            callable(cmb.compute_cmb_spectrum_from_legacy_params_for_tests)
        )
        self.assertTrue(callable(cmb.describe_camb_configuration))
        self.assertFalse(hasattr(cmb, "_CustomCMBBackgroundData"))
        self.assertFalse(hasattr(cmb, "_make_camb_params"))
        self.assertFalse(hasattr(cmb, "camb"))
        self.assertFalse(hasattr(cmb, "compute_cmb_spectrum_from_dict"))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        """The likelihood protocol symbols should remain available."""

        self.assertTrue(callable(cmb.CMBLike.loglike))
        self.assertTrue(hasattr(cmb.CMBLike.state, "__get__"))


class CMBLikeMultiSpectrumTestCase(unittest.TestCase):
    """Exercise the public CMB likelihood surface on spectrum blocks."""

    def test_loglike_supports_block_spectra(self) -> None:
        """Block spectra should flatten into one consistent likelihood."""

        class _BlockSpectrumPlugin:
            """Return a minimal structured CAMB contract for testing."""

            def get_camb_contract(self, _params):
                return {
                    "backend": "camb",
                    "calls": [],
                    "grids": {},
                    "param_map": {},
                    "perturbations": {"standard": True},
                    "values": {},
                }

        cmb_data = pandas.DataFrame(
            {
                "ell": [2, 3, 2, 3],
                "spectrum": ["TT", "TT", "TE", "TE"],
                "Dl_obs": [1.0, 2.0, 0.5, 0.25],
            }
        )
        cmb_data.attrs["covariance_matrix_inv"] = numpy.eye(4, dtype=float)
        theory = {
            "TT": numpy.asarray([0.0, 0.0, 1.0, 2.0], dtype=float),
            "TE": numpy.asarray([0.0, 0.0, 0.5, 0.25], dtype=float),
        }
        with mock.patch(
            (
                "copernican.lib.likelihoods.cmb.cmb."
                "compute_cmb_spectrum_from_contract"
            ),
            return_value=theory,
        ):
            likelihood = cmb.CMBLike(cmb_data, _BlockSpectrumPlugin())
            loglike = likelihood.loglike(())

        self.assertEqual(likelihood.state["chi2"], 0.0)
        self.assertEqual(loglike, 0.0)
        self.assertEqual(likelihood._observed.shape, (4,))


if __name__ == "__main__":
    unittest.main()
