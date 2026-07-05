"""Compile declared CMB graph contracts into immutable runtime data.

`standard: false` contracts describe one declared-math graph rather than
selecting a hard-coded solver family. The compiler validates symbols,
dependencies, observables, and runtime requirements before the numerical CMB
engine evolves the system.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy

from .cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_spec,
    resolve_declared_projection_kernel,
    validate_declared_projection_source_roles,
)
from .engine_adapter import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES,
    FrozenMapping,
    _build_parameter_replacements,
    _freeze_for_cache,
    _parse_safe_expression,
    _replace_latex_tokens,
    _validate_safe_expression,
)

_COMPILED_BINARY_OPCODE_NAMES = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
}
_COMPILED_UNARY_OPCODE_NAMES = {
    ast.UAdd: "uadd",
    ast.USub: "usub",
}
_COMPILED_BINARY_OPERATORS = {
    "add": numpy.add,
    "sub": numpy.subtract,
    "mul": numpy.multiply,
    "div": numpy.divide,
    "pow": numpy.power,
}
_COMPILED_UNARY_OPERATORS = {
    "uadd": lambda value: value,
    "usub": numpy.negative,
}

_RUNTIME_REFERENCE_NAMES = {
    "H0_km_s_Mpc",
    "H0_over_c_Mpc_inv",
    "Omega_b0",
    "Omega_c0",
    "Omega_de0",
    "Omega_gamma0",
    "Omega_k0",
    "Omega_m0",
    "Omega_nu0",
    "Omega_r0",
    "Neff",
    "Tcmb_K",
    "YHe",
    "a_initial",
    "angular_diameter_distance",
    "chi",
    "collision_rate",
    "eta_initial",
    "free_streaming",
    "hubble_ratio",
    "massive_neutrino_mass_eV",
    "massive_neutrino_mass_fraction",
    "massive_neutrino_pressure_ratio",
    "massive_neutrino_streaming_speed",
    "massive_neutrino_velocity_ratio",
    "n_H0_m3",
    "n_b0_m3",
    "ombh2",
    "omch2",
    "seed",
    "rho_b0_kg_m3",
    "rho_c0_kg_m3",
    "rho_crit0_kg_m3",
    "sound_horizon",
    "sound_speed",
    "sound_speed_sq",
    "tight_coupling_drag",
    "tight_coupling_ratio",
    "w0",
    "wa",
    "primordial_amplitude",
    "primordial_spectral_index",
}

_SUPPORTED_PERTURBATION_KEYS = {
    "accuracy_controls",
    "backend_mapping",
    "boundary_conditions",
    "conservation_rules",
    "collision_operators",
    "closures",
    "constraints",
    "contract_version",
    "derived",
    "equations",
    "gauge",
    "hierarchy_families",
    "initial_conditions",
    "initial_condition_families",
    "interactions",
    "notes",
    "numerics",
    "observables",
    "projection_extensions",
    "projection_typing",
    "sectors",
    "species",
    "sources",
    "standard",
    "validity",
    "variables",
}
_SUPPORTED_VARIABLE_KEYS = {
    "description",
    "domain",
    "gauge_role",
    "kind",
    "notes",
    "parity",
    "projection_role",
    "rank",
    "source_role",
    "spin",
    "tensor_character",
    "units",
}
_SUPPORTED_DERIVED_KEYS = {
    "description",
    "domain",
    "expression",
    "kind",
    "notes",
    "order",
    "variable",
    "wrt",
}
_SUPPORTED_EQUATION_KEYS = {
    "dependencies",
    "description",
    "domain",
    "lhs",
    "notes",
    "rhs",
    "role",
}
_SUPPORTED_LHS_KEYS = {"kind", "order", "variable", "wrt"}
_SUPPORTED_RELATION_KEYS = {
    "dependencies",
    "description",
    "domain",
    "expression",
    "notes",
    "role",
    "target",
}
_SUPPORTED_SOURCE_KEYS = {
    "dependencies",
    "description",
    "domain",
    "expression",
    "notes",
    "role",
}
_SUPPORTED_OBSERVABLE_KEYS = {
    "dependencies",
    "description",
    "domain",
    "kernel",
    "kind",
    "notes",
    "primary",
    "projection",
    "required_projection_roles",
    "secondary",
    "source_terms",
}
_SUPPORTED_CONDITION_KEYS = {
    "anchor",
    "dependencies",
    "description",
    "domain",
    "expression",
    "notes",
    "target",
}
_SUPPORTED_CONDITION_TARGET_KEYS = {"order", "variable", "wrt"}
_SUPPORTED_VALIDITY_KEYS = {"notes", "regimes"}
_SUPPORTED_BACKEND_KEYS = {"camb"}
_SUPPORTED_SECTOR_KEYS = {
    "description",
    "hierarchy_families",
    "notes",
    "species",
    "supported_gauges",
    "tensor_character",
}
_SUPPORTED_SPECIES_KEYS = {
    "anisotropic_stress",
    "background_reference",
    "collision_operators",
    "description",
    "equation_of_state",
    "hierarchy_family",
    "notes",
    "sector",
    "sound_speed",
}
_SUPPORTED_HIERARCHY_FAMILY_KEYS = {
    "closure",
    "default_l_max",
    "description",
    "momentum_grid",
    "multipole_symbol",
    "notes",
    "sector",
    "species",
}
_SUPPORTED_COLLISION_OPERATOR_KEYS = {
    "counterpart",
    "dependencies",
    "description",
    "expression",
    "notes",
    "sector",
    "species",
}
_SUPPORTED_INTERACTION_KEYS = _SUPPORTED_COLLISION_OPERATOR_KEYS
_SUPPORTED_INITIAL_CONDITION_FAMILY_KEYS = {
    "description",
    "members",
    "notes",
    "sector",
}
_SUPPORTED_CONSERVATION_RULE_KEYS = {
    "description",
    "domain",
    "expression",
    "kind",
    "notes",
    "tolerance",
}
_SUPPORTED_PROJECTION_EXTENSION_KEYS = {
    "allowed_roles",
    "base_projection",
    "description",
    "kernel",
    "notes",
    "required_projection_roles",
    "required_roles",
    "requires_odd_parity_source",
}
_SUPPORTED_PROJECTION_TYPING_KEYS = {
    "description",
    "kernel",
    "notes",
    "observable_kinds",
    "parity",
    "sector",
    "source_roles",
    "spin",
}
_SCALAR_HIERARCHY_REQUIRED_SECTOR = "scalar"
_SCALAR_HIERARCHY_REQUIRED_SPECIES = {
    "baryon",
    "cdm",
    "massless_neutrino",
    "photon",
}
_SCALAR_HIERARCHY_REQUIRED_FAMILIES = {
    "massless_neutrino",
    "photon_polarization_e",
    "photon_temperature",
}
_SCALAR_HIERARCHY_REQUIRED_COLLISION = "thomson_drag"
_SCALAR_HIERARCHY_STANDARD_INITIAL_MODES = (
    "adiabatic_scalar",
    "baryon_isocurvature",
    "cdm_isocurvature",
    "neutrino_density_isocurvature",
    "neutrino_velocity_isocurvature",
    "tensor_mode",
)
_NEWTONIAN_GAUGE_ROLES = frozenset(
    {"curvature_potential", "newtonian_potential"}
)
_SYNCHRONOUS_GAUGE_ROLES = frozenset(
    {"synchronous_metric_shear", "synchronous_metric_trace"}
)


def _has_explicit_native_runtime_graph(
    contract: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract`` already declares runtime nodes."""

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
        if contract.get(section_name):
            return True
    return False


def _scalar_temperature_name(moment: int) -> str:
    """Return the declared photon-temperature variable name."""

    return f"theta_gamma{int(moment)}"


def _scalar_polarization_name(moment: int) -> str:
    """Return the declared photon-polarization variable name."""

    return f"e_gamma{int(moment)}"


def _scalar_neutrino_name(moment: int) -> str:
    """Return the declared massless-neutrino variable name."""

    if moment == 0:
        return "delta_nu"
    if moment == 1:
        return "theta_nu"
    if moment == 2:
        return "sigma_nu"
    return f"nu_l{int(moment)}"


def _scalar_massive_neutrino_name(moment: int) -> str:
    """Return the declared massive-neutrino variable name."""

    if moment == 0:
        return "delta_nu_massive"
    if moment == 1:
        return "theta_nu_massive"
    if moment == 2:
        return "sigma_nu_massive"
    return f"nu_massive_l{int(moment)}"


def _scalar_massive_neutrino_q_name(
    index: int,
    moment: int,
) -> str:
    """Return the q-resolved massive-neutrino variable name."""

    q_index = int(index)
    if moment == 0:
        return f"delta_nu_massive_q{q_index}"
    if moment == 1:
        return f"theta_nu_massive_q{q_index}"
    if moment == 2:
        return f"sigma_nu_massive_q{q_index}"
    return f"nu_massive_q{q_index}_l{int(moment)}"


def _scalar_massive_neutrino_distribution_log_derivative_name(
    index: int,
) -> str:
    """Return the thermal momentum-distribution log-derivative name."""

    return f"massive_neutrino_q{int(index)}_distribution_log_derivative"


def _scalar_massive_neutrino_q_streaming_speed_name(
    index: int,
) -> str:
    """Return the q-bin streaming-speed name for massive neutrinos."""

    return f"massive_neutrino_q{int(index)}_streaming_speed"


def _select_standard_initial_mode(
    family_defs: Mapping[str, Any],
) -> str | None:
    """Return the declared auto-generated initial-condition mode."""

    for family_name in _SCALAR_HIERARCHY_STANDARD_INITIAL_MODES:
        if family_name in family_defs:
            return family_name
    return None


def _scalar_hierarchy_base_seed_expressions(
    mode: str,
) -> dict[str, str]:
    """Return base seed expressions for one scalar hierarchy mode."""

    adiabatic_brightness_dipole = "(acoustic_k * eta_initial / 6.0) * seed"
    adiabatic_velocity_divergence = (
        "(acoustic_k_sq * eta_initial / 2.0) * seed"
    )
    adiabatic_quadrupole = (
        "(acoustic_k * eta_initial) * (acoustic_k * eta_initial) "
        "* seed / 30.0"
    )
    adiabatic_neutrino_quadrupole = (
        "(acoustic_k * eta_initial) * (acoustic_k * eta_initial) "
        "* seed / 15.0"
    )
    by_mode = {
        "adiabatic_scalar": {
            "theta_gamma0": "-0.5 * seed",
            "theta_gamma1": adiabatic_brightness_dipole,
            "theta_gamma2": adiabatic_quadrupole,
            "e_gamma0": "0.0",
            "e_gamma1": "0.0",
            "e_gamma2": "0.0",
            "delta_b": "-1.5 * seed",
            "theta_b": adiabatic_velocity_divergence,
            "delta_c": "-1.5 * seed",
            "theta_c": adiabatic_velocity_divergence,
            "delta_nu": "-2.0 * seed",
            "theta_nu": adiabatic_velocity_divergence,
            "sigma_nu": adiabatic_neutrino_quadrupole,
            "delta_nu_massive": "-2.0 * seed",
            "theta_nu_massive": adiabatic_velocity_divergence,
            "sigma_nu_massive": adiabatic_neutrino_quadrupole,
        },
        "baryon_isocurvature": {
            "delta_b": "seed",
        },
        "cdm_isocurvature": {
            "theta_gamma0": "0.25 * seed",
            "delta_b": "-0.5 * seed",
            "delta_c": "seed",
            "delta_nu": "-0.5 * seed",
        },
        "neutrino_density_isocurvature": {
            "delta_nu": "seed",
            "delta_nu_massive": "seed",
        },
        "neutrino_velocity_isocurvature": {
            "theta_nu": "seed",
            "theta_nu_massive": "seed",
        },
        "tensor_mode": {},
    }
    return dict(by_mode.get(mode, {}))


def _auto_initial_condition_expression(
    *,
    variable_entry: "PerturbationVariableData",
    target_order: int,
    mode: str,
) -> str:
    """Return one generated seed expression for ``variable_entry``."""

    if target_order > 0:
        return "0.0"
    explicit_seed = _scalar_hierarchy_base_seed_expressions(mode).get(
        variable_entry.name
    )
    if explicit_seed is not None:
        return explicit_seed
    kind = str(variable_entry.kind)
    if mode == "tensor_mode":
        if variable_entry.tensor_character == "tensor_like":
            return "seed"
        return "0.0"
    if "density" in kind:
        return "seed" if mode == "adiabatic_scalar" else "0.0"
    if "velocity" in kind or "dipole" in kind:
        return (
            "(acoustic_k * eta_initial / 6.0) * seed"
            if mode == "adiabatic_scalar"
            else "0.0"
        )
    # Use the leading-order super-horizon series for metric roles instead of
    # the older physical-series constants so Newtonian and synchronous gauge
    # seeds stay physically shaped.
    if variable_entry.gauge_role in _NEWTONIAN_GAUGE_ROLES:
        if mode == "adiabatic_scalar":
            return "seed"
        if mode == "cdm_isocurvature":
            return "0.25 * seed"
        return "0.0"
    if variable_entry.gauge_role in _SYNCHRONOUS_GAUGE_ROLES:
        if mode != "adiabatic_scalar":
            return "0.0"
        if variable_entry.name == "h_sync_metric":
            return (
                "(acoustic_k * eta_initial) * "
                "(acoustic_k * eta_initial) * seed"
            )
        return "2.0 * seed"
    return "0.0"


def _scalar_hierarchy_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    collision_term: str | None = None,
) -> str:
    """Return one hierarchy recurrence RHS for the generated scalar route."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        pieces.append(
            f"- {float(moment + 1):.16g} / {float((2 * moment) + 1):.16g} "
            f"* acoustic_k * {name}"
        )
    else:
        next_coeff = float(moment + 1) / float((2 * moment) + 1)
        pieces.append(f"- {next_coeff:.16g} * acoustic_k * {next_name}")
    if collision_term is not None:
        pieces.append(collision_term)
    return " ".join(pieces)


def _materialize_native_scalar_hierarchy_contract(
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    """Return a generated scalar hierarchy contract when metadata is enough."""

    if contract.get("standard") is not False:
        return contract, False
    if _has_explicit_native_runtime_graph(contract):
        return contract, False

    sectors = contract.get("sectors", {}) or {}
    species = contract.get("species", {}) or {}
    hierarchy_families = contract.get("hierarchy_families", {}) or {}
    collision_operators = contract.get("collision_operators", {}) or {}
    initial_condition_families = (
        contract.get("initial_condition_families", {}) or {}
    )
    if not isinstance(sectors, Mapping):
        return contract, False
    if not isinstance(species, Mapping):
        return contract, False
    if not isinstance(hierarchy_families, Mapping):
        return contract, False
    if not isinstance(collision_operators, Mapping):
        return contract, False
    if not isinstance(initial_condition_families, Mapping):
        return contract, False
    if _SCALAR_HIERARCHY_REQUIRED_SECTOR not in sectors:
        return contract, False
    if not _SCALAR_HIERARCHY_REQUIRED_SPECIES.issubset(species):
        return contract, False
    if not _SCALAR_HIERARCHY_REQUIRED_FAMILIES.issubset(hierarchy_families):
        return contract, False
    initial_mode = _select_standard_initial_mode(initial_condition_families)
    if initial_mode is None:
        return contract, False

    numerics = contract.get("numerics", {}) or {}
    if not isinstance(numerics, Mapping):
        numerics = {}
    gauge = str(contract.get("gauge") or "conformal_newtonian")
    sync_gauge = gauge == "synchronous"
    has_massive_neutrino = (
        "massive_neutrino" in hierarchy_families
        and "massive_neutrino" in species
    )
    photon_default_l_max = hierarchy_families["photon_temperature"].get(
        "default_l_max", 6
    )
    polarization_default_l_max = hierarchy_families[
        "photon_polarization_e"
    ].get(
        "default_l_max",
        photon_default_l_max,
    )
    neutrino_default_l_max = hierarchy_families["massless_neutrino"].get(
        "default_l_max", 4
    )
    photon_l_max = max(
        3,
        int(
            numerics.get(
                "photon_hierarchy_l_max",
                max(photon_default_l_max, polarization_default_l_max),
            )
        ),
    )
    neutrino_l_max = max(
        3,
        int(
            numerics.get(
                "neutrino_hierarchy_l_max",
                neutrino_default_l_max,
            )
        ),
    )
    massive_neutrino_l_max = max(
        3,
        int(
            numerics.get(
                "massive_neutrino_hierarchy_l_max",
                hierarchy_families.get("massive_neutrino", {}).get(
                    "default_l_max",
                    neutrino_l_max,
                ),
            )
        ),
    )
    massive_neutrino_grid_count = 0
    if has_massive_neutrino:
        momentum_grid_name = str(
            hierarchy_families.get("massive_neutrino", {}).get(
                "momentum_grid",
                "",
            )
        ).strip()
        momentum_grid_defs = numerics.get("momentum_grids", {}) or {}
        if (
            momentum_grid_name
            and isinstance(momentum_grid_defs, Mapping)
            and momentum_grid_name in momentum_grid_defs
        ):
            momentum_grid_def = momentum_grid_defs.get(momentum_grid_name, {})
            if isinstance(momentum_grid_def, Mapping):
                massive_neutrino_grid_count = max(
                    1,
                    int(momentum_grid_def.get("count", 1)),
                )
    materialized = copy.deepcopy(dict(contract))
    if sync_gauge:
        phi_state_name = "h_sync_metric"
        psi_state_name = "eta_sync_metric"
        metric_variables = {
            phi_state_name: {
                "kind": "synchronous_metric_trace",
                "gauge_role": "synchronous_metric_trace",
            },
            psi_state_name: {
                "kind": "synchronous_metric_shear",
                "gauge_role": "synchronous_metric_shear",
            },
        }
    else:
        phi_state_name = "Phi"
        psi_state_name = "Psi"
        metric_variables = {
            phi_state_name: {
                "kind": "metric_potential_phi",
                "gauge_role": "newtonian_potential",
            },
            psi_state_name: {
                "kind": "metric_potential_psi",
                "gauge_role": "curvature_potential",
            },
        }
    variables: dict[str, Any] = {
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
        **metric_variables,
    }
    if has_massive_neutrino:
        variables.update(
            {
                "delta_nu_massive": {
                    "kind": "massive_neutrino_density_contrast",
                },
                "theta_nu_massive": {
                    "kind": "massive_neutrino_velocity_divergence",
                },
                "sigma_nu_massive": {
                    "kind": "massive_neutrino_anisotropic_stress",
                },
            }
        )
    for moment in range(photon_l_max + 1):
        if moment == 0:
            kind = "photon_temperature_monopole"
        elif moment == 1:
            kind = "photon_temperature_dipole"
        elif moment == 2:
            kind = "photon_temperature_quadrupole"
        elif moment == 3:
            kind = "photon_temperature_octopole"
        else:
            kind = "photon_temperature_multipole"
        variables[_scalar_temperature_name(moment)] = {
            "kind": kind,
            "tensor_character": "scalar_like",
        }
    for moment in range(photon_l_max + 1):
        if moment == 0:
            kind = "photon_polarization_monopole"
        elif moment == 1:
            kind = "photon_polarization_dipole"
        elif moment == 2:
            kind = "photon_polarization_quadrupole"
        elif moment == 3:
            kind = "photon_polarization_octopole"
        else:
            kind = "photon_polarization_multipole"
        variables[_scalar_polarization_name(moment)] = {
            "kind": kind,
            "tensor_character": "scalar_like",
        }
    variables["polarization_b_mode_seed"] = {
        "kind": "polarization_b_seed",
        "projection_role": "b_mode",
    }
    for moment in range(3, neutrino_l_max + 1):
        variables[_scalar_neutrino_name(moment)] = {
            "kind": "massless_neutrino_multipole",
            "tensor_character": "scalar_like",
        }
    if has_massive_neutrino:
        for moment in range(3, massive_neutrino_l_max + 1):
            variables[_scalar_massive_neutrino_name(moment)] = {
                "kind": "massive_neutrino_multipole",
                "tensor_character": "scalar_like",
            }
        for q_index in range(massive_neutrino_grid_count):
            for moment in range(massive_neutrino_l_max + 1):
                q_name = _scalar_massive_neutrino_q_name(
                    q_index,
                    moment,
                )
                if moment == 0:
                    kind = "massive_neutrino_momentum_bin_density_contrast"
                elif moment == 1:
                    kind = (
                        "massive_neutrino_momentum_bin_" "velocity_divergence"
                    )
                elif moment == 2:
                    kind = (
                        "massive_neutrino_momentum_bin_" "anisotropic_stress"
                    )
                else:
                    kind = "massive_neutrino_momentum_bin_multipole"
                variables[q_name] = {
                    "kind": kind,
                    "tensor_character": "scalar_like",
                }

    equations: dict[str, Any] = {
        "evolve_theta_gamma0": {
            "lhs": {
                "kind": "derivative",
                "variable": "theta_gamma0",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-acoustic_k * theta_gamma1 - Phi_tau",
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
                "(acoustic_k / 3.0) * "
                "(theta_gamma0 + Psi - 2.0 * theta_gamma2) "
                "+ thomson_drag"
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
                f"{2.0 / 5.0:.16g} * acoustic_k * theta_gamma1 "
                f"- {3.0 / 5.0:.16g} * acoustic_k * "
                f"{_scalar_temperature_name(3)} "
                "- collision_rate * "
                "(theta_gamma2 - 0.1 * polarization_moment)"
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
            "rhs": "-acoustic_k * e_gamma1",
            "role": "polarization",
        },
        "evolve_e_gamma1": {
            "lhs": {
                "kind": "derivative",
                "variable": "e_gamma1",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "(acoustic_k / 3.0) * (e_gamma0 - 2.0 * e_gamma2)",
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
                f"{3.0 / 5.0:.16g} * acoustic_k * "
                f"{_scalar_polarization_name(3)} "
                "- collision_rate * "
                "(e_gamma2 - 0.1 * polarization_moment)"
            ),
            "role": "polarization",
        },
        "evolve_polarization_b_mode_seed": {
            "lhs": {
                "kind": "derivative",
                "variable": "polarization_b_mode_seed",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "0.0",
            "role": "polarization_b",
        },
        "evolve_delta_b": {
            "lhs": {
                "kind": "derivative",
                "variable": "delta_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-theta_b + 3.0 * Phi_tau",
            "role": "continuity",
        },
        "evolve_theta_b": {
            "lhs": {
                "kind": "derivative",
                "variable": "theta_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-Hconf * theta_b + acoustic_k_sq * sound_speed_sq * "
                "delta_b "
                "+ baryon_thomson_drag "
                "+ acoustic_k_sq * Psi"
            ),
            "role": "euler",
        },
        "evolve_delta_c": {
            "lhs": {
                "kind": "derivative",
                "variable": "delta_c",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-theta_c + 3.0 * Phi_tau",
            "role": "continuity",
        },
        "evolve_theta_c": {
            "lhs": {
                "kind": "derivative",
                "variable": "theta_c",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-Hconf * theta_c + acoustic_k_sq * Psi",
            "role": "euler",
        },
        "evolve_delta_nu": {
            "lhs": {
                "kind": "derivative",
                "variable": "delta_nu",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-(4.0 / 3.0) * theta_nu + 4.0 * Phi_tau",
            "role": "continuity",
        },
        "evolve_theta_nu": {
            "lhs": {
                "kind": "derivative",
                "variable": "theta_nu",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": ("acoustic_k_sq * (0.25 * delta_nu + Psi - sigma_nu)"),
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
                f"{4.0 / 15.0:.16g} * theta_nu "
                f"- {3.0 / 5.0:.16g} * acoustic_k * "
                f"{_scalar_neutrino_name(3)}"
            ),
            "role": "hierarchy",
        },
    }
    for moment in range(3, photon_l_max + 1):
        name = _scalar_temperature_name(moment)
        next_name = None
        if moment < photon_l_max:
            next_name = _scalar_temperature_name(moment + 1)
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _scalar_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=_scalar_temperature_name(moment - 1),
                next_name=next_name,
            ),
            "role": "hierarchy",
        }
    for moment in range(3, photon_l_max + 1):
        name = _scalar_polarization_name(moment)
        next_name = None
        if moment < photon_l_max:
            next_name = _scalar_polarization_name(moment + 1)
        previous_name = (
            "e_gamma2"
            if moment == 3
            else _scalar_polarization_name(moment - 1)
        )
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _scalar_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=previous_name,
                next_name=next_name,
            ),
            "role": "polarization",
        }
    for moment in range(3, neutrino_l_max + 1):
        name = _scalar_neutrino_name(moment)
        next_name = None
        if moment < neutrino_l_max:
            next_name = _scalar_neutrino_name(moment + 1)
        previous_name = (
            "sigma_nu" if moment == 3 else _scalar_neutrino_name(moment - 1)
        )
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _scalar_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=previous_name,
                next_name=next_name,
            ),
            "role": "hierarchy",
        }
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        equations.update(
            {
                "evolve_delta_nu_massive": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "delta_nu_massive",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "-theta_nu_massive + 3.0 * Phi_tau",
                    "role": "continuity",
                },
                "evolve_theta_nu_massive": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "theta_nu_massive",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": (
                        "acoustic_k_sq * (0.25 * "
                        "massive_neutrino_metric_density + "
                        "Psi - massive_neutrino_metric_shear)"
                    ),
                    "role": "euler",
                },
                "evolve_sigma_nu_massive": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "sigma_nu_massive",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": (
                        f"{4.0 / 15.0:.16g} * "
                        "massive_neutrino_metric_momentum "
                        f"- {3.0 / 5.0:.16g} * acoustic_k * "
                        f"{_scalar_massive_neutrino_name(3)}"
                    ),
                    "role": "hierarchy",
                },
            }
        )
        for moment in range(3, massive_neutrino_l_max + 1):
            name = _scalar_massive_neutrino_name(moment)
            next_name = None
            if moment < massive_neutrino_l_max:
                next_name = _scalar_massive_neutrino_name(moment + 1)
            previous_name = (
                "sigma_nu_massive"
                if moment == 3
                else _scalar_massive_neutrino_name(moment - 1)
            )
            previous_coeff = float(moment) / float((2 * moment) + 1)
            next_coeff = float(moment + 1) / float((2 * moment) + 1)
            if next_name is None:
                next_term = (
                    f"- {next_coeff:.16g} * acoustic_k * "
                    f"massive_neutrino_streaming_speed * {name}"
                )
            else:
                next_term = (
                    f"- {next_coeff:.16g} * acoustic_k * "
                    f"massive_neutrino_streaming_speed * {next_name}"
                )
            equations[f"evolve_{name}"] = {
                "lhs": {
                    "kind": "derivative",
                    "variable": name,
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    f"{previous_coeff:.16g} * acoustic_k * "
                    f"massive_neutrino_streaming_speed * {previous_name} "
                    f"{next_term}"
                ),
                "role": "hierarchy",
            }
        for q_index in range(massive_neutrino_grid_count):
            q_streaming_speed_name = (
                _scalar_massive_neutrino_q_streaming_speed_name(q_index)
            )
            q_log_derivative_name = (
                _scalar_massive_neutrino_distribution_log_derivative_name(
                    q_index
                )
            )
            q_delta_name = _scalar_massive_neutrino_q_name(
                q_index,
                0,
            )
            q_theta_name = _scalar_massive_neutrino_q_name(
                q_index,
                1,
            )
            q_sigma_name = _scalar_massive_neutrino_q_name(
                q_index,
                2,
            )
            q_l3_name = _scalar_massive_neutrino_q_name(
                q_index,
                3,
            )
            equations[f"evolve_{q_delta_name}"] = {
                "lhs": {
                    "kind": "derivative",
                    "variable": q_delta_name,
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    f"-acoustic_k * {q_streaming_speed_name} * {q_theta_name} "
                    f"- Phi_tau * {q_log_derivative_name}"
                ),
                "role": "continuity",
            }
            equations[f"evolve_{q_theta_name}"] = {
                "lhs": {
                    "kind": "derivative",
                    "variable": q_theta_name,
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    f"(acoustic_k / 3.0) * {q_streaming_speed_name} * "
                    f"({q_delta_name} - 2.0 * {q_sigma_name}) - "
                    f"(acoustic_k / 3.0) * "
                    f"(1.0 / {q_streaming_speed_name}) * Psi * "
                    f"{q_log_derivative_name}"
                ),
                "role": "euler",
            }
            equations[f"evolve_{q_sigma_name}"] = {
                "lhs": {
                    "kind": "derivative",
                    "variable": q_sigma_name,
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": (
                    f"{2.0 / 5.0:.16g} * acoustic_k * "
                    f"{q_streaming_speed_name} * {q_theta_name} "
                    f"- {3.0 / 5.0:.16g} * acoustic_k * "
                    f"{q_streaming_speed_name} * {q_l3_name}"
                ),
                "role": "hierarchy",
            }
            for moment in range(3, massive_neutrino_l_max + 1):
                name = _scalar_massive_neutrino_q_name(
                    q_index,
                    moment,
                )
                next_name = None
                if moment < massive_neutrino_l_max:
                    next_name = _scalar_massive_neutrino_q_name(
                        q_index,
                        moment + 1,
                    )
                previous_name = (
                    q_sigma_name
                    if moment == 3
                    else _scalar_massive_neutrino_q_name(
                        q_index,
                        moment - 1,
                    )
                )
                previous_coeff = float(moment) / float((2 * moment) + 1)
                next_coeff = float(moment + 1) / float((2 * moment) + 1)
                if next_name is None:
                    next_term = (
                        f"- {next_coeff:.16g} * acoustic_k * "
                        f"{q_streaming_speed_name} * {name}"
                    )
                else:
                    next_term = (
                        f"- {next_coeff:.16g} * acoustic_k * "
                        f"{q_streaming_speed_name} * {next_name}"
                    )
                equations[f"evolve_{name}"] = {
                    "lhs": {
                        "kind": "derivative",
                        "variable": name,
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": (
                        f"{previous_coeff:.16g} * acoustic_k * "
                        f"{q_streaming_speed_name} * {previous_name} "
                        f"{next_term}"
                    ),
                    "role": "hierarchy",
                }

    materialized["variables"] = variables
    massless_fraction_expression = "Omega_nu0"
    total_radiation_expression = (
        "4.0 * Omega_gamma0 * theta_gamma0 + "
        "massless_neutrino_fraction * delta_nu"
    )
    total_neutrino_shear_expression = "massless_neutrino_fraction * sigma_nu"
    total_momentum_expression = (
        "Omega_b0 * theta_b + Omega_c0 * theta_c + "
        "4.0 * Omega_gamma0 * theta_gamma1 + "
        "massless_neutrino_fraction * theta_nu"
    )
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        total_radiation_expression += " + massive_neutrino_metric_density"
        total_neutrino_shear_expression += " + massive_neutrino_metric_shear"
        total_momentum_expression += " + massive_neutrino_metric_momentum"
    derived_entries: dict[str, Any] = {
        "polarization_moment": {
            "expression": "theta_gamma2 + 6.0 * e_gamma2",
            "description": "Scalar polarization source moment.",
        },
        "acoustic_k": {
            "expression": "k",
            "description": "Shifted scalar acoustic wave number.",
        },
        "acoustic_k_sq": {
            "expression": "acoustic_k * acoustic_k",
            "description": "Squared shifted scalar acoustic wave number.",
        },
        "total_matter_density": {
            "expression": "Omega_c0 * delta_c + Omega_b0 * delta_b",
            "description": "Matter source for the scalar metric.",
        },
        "massless_neutrino_fraction": {
            "expression": massless_fraction_expression,
            "description": "Effective relativistic-neutrino metric weight.",
        },
        "total_radiation_density": {
            "expression": total_radiation_expression,
            "description": "Radiation source for the scalar metric.",
        },
        "total_neutrino_shear": {
            "expression": total_neutrino_shear_expression,
            "description": "Neutrino shear source for metric closure.",
        },
        "total_momentum_density": {
            "expression": total_momentum_expression,
            "description": "Momentum source for the scalar metric.",
        },
        "metric_denominator": {
            "expression": "k * k",
            "description": "Scalar Poisson denominator for the metric.",
        },
        "einstein_gravity_strength": {
            "expression": "H0_over_c_Mpc_inv * H0_over_c_Mpc_inv",
            "description": (
                "Background gravity scale used by the scalar Einstein "
                "constraints."
            ),
        },
        "photon_baryon_momentum_ratio": {
            "expression": "(4.0 * Omega_gamma0) / (3.0 * Omega_b0 * a)",
            "description": "Photon-to-baryon momentum-transfer ratio.",
        },
        "baryon_thomson_drag": {
            "expression": "- photon_baryon_momentum_ratio * thomson_drag",
            "description": "Baryon-side Thomson drag counterpart.",
        },
    }
    derived_entries.update(
        {
            "Phi_tau": {
                "kind": "metric_potential_time_derivative",
                "expression": (
                    "-Hconf * Psi + "
                    "1.5 * einstein_gravity_strength * "
                    "total_momentum_density / metric_denominator"
                ),
                "description": (
                    "Momentum-constraint relation for the scalar "
                    "potential time derivative."
                ),
            },
            "Psi_tau": {
                "kind": "metric_potential_time_derivative",
                "expression": (
                    "-Hconf * Psi + "
                    "1.5 * einstein_gravity_strength * "
                    "total_neutrino_shear / metric_denominator"
                ),
                "description": (
                    "Anisotropic-stress relation for the curvature "
                    "potential time derivative."
                ),
            },
        }
    )
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        q_density_component_names = []
        q_momentum_component_names = []
        q_shear_component_names = []
        for q_index in range(massive_neutrino_grid_count):
            q_prefix = f"massive_neutrino_q{q_index}"
            q_density_name = f"massive_neutrino_metric_density_q{q_index}"
            q_momentum_name = f"massive_neutrino_metric_momentum_q{q_index}"
            q_shear_name = f"massive_neutrino_metric_shear_q{q_index}"
            q_log_derivative_name = (
                _scalar_massive_neutrino_distribution_log_derivative_name(
                    q_index
                )
            )
            q_streaming_speed_name = (
                _scalar_massive_neutrino_q_streaming_speed_name(q_index)
            )
            q_density_component_names.append(q_density_name)
            q_momentum_component_names.append(q_momentum_name)
            q_shear_component_names.append(q_shear_name)
            derived_entries[q_log_derivative_name] = {
                "expression": (
                    f"-{q_prefix}_point / (1.0 + exp(-{q_prefix}_point))"
                ),
                "description": (
                    "Logarithmic derivative of the thermal distribution."
                ),
            }
            derived_entries[q_streaming_speed_name] = {
                "expression": (
                    f"{q_prefix}_point / sqrt(("
                    f"{q_prefix}_point * {q_prefix}_point) + "
                    "(a * massive_neutrino_mass_eV) * "
                    "(a * massive_neutrino_mass_eV))"
                ),
                "description": (
                    "Streaming speed for one massive-neutrino momentum bin."
                ),
            }
            derived_entries[q_density_name] = {
                "expression": (
                    f"{q_prefix}_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 0)}"
                ),
                "description": (
                    "Momentum-grid-weighted q-bin density moment."
                ),
            }
            derived_entries[q_momentum_name] = {
                "expression": (
                    f"{q_prefix}_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 1)}"
                ),
                "description": (
                    "Momentum-grid-weighted q-bin momentum moment."
                ),
            }
            derived_entries[q_shear_name] = {
                "expression": (
                    f"{q_prefix}_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 2)}"
                ),
                "description": ("Momentum-grid-weighted q-bin shear moment."),
            }
        density_sum_expression = " + ".join(q_density_component_names)
        momentum_sum_expression = " + ".join(q_momentum_component_names)
        shear_sum_expression = " + ".join(q_shear_component_names)
        if len(q_density_component_names) > 1:
            density_sum_expression = f"({density_sum_expression})"
        if len(q_momentum_component_names) > 1:
            momentum_sum_expression = f"({momentum_sum_expression})"
        if len(q_shear_component_names) > 1:
            shear_sum_expression = f"({shear_sum_expression})"
        derived_entries.update(
            {
                "massive_neutrino_metric_density": {
                    "expression": density_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino density."
                    ),
                },
                "massive_neutrino_metric_momentum": {
                    "expression": momentum_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino momentum."
                    ),
                },
                "massive_neutrino_metric_shear": {
                    "expression": shear_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino shear."
                    ),
                },
            }
        )
    if sync_gauge:
        derived_entries.update(
            {
                "Phi": {
                    "expression": f"0.5 * {phi_state_name}",
                    "description": "Gauge-stable metric-potential alias.",
                },
                "Psi": {
                    "expression": f"0.5 * {psi_state_name}",
                    "description": "Gauge-stable metric-curvature alias.",
                },
            }
        )
    materialized["derived"] = derived_entries
    materialized["equations"] = equations
    collision_operator_entries = dict(
        materialized.get("collision_operators", {}) or {}
    )
    collision_operator_entries.setdefault(
        "thomson_drag",
        {
            "sector": "scalar",
            "species": ["photon", "baryon"],
            "expression": (
                "collision_rate * " "(theta_b / 3.0 - theta_gamma1)"
            ),
            "counterpart": "baryon_thomson_drag",
        },
    )
    materialized["collision_operators"] = collision_operator_entries
    conservation_rule_entries = dict(
        materialized.get("conservation_rules", {}) or {}
    )
    conservation_rule_entries.setdefault(
        "thomson_drag_balance",
        {
            "kind": "absolute_max",
            "expression": (
                "photon_baryon_momentum_ratio * thomson_drag + "
                "baryon_thomson_drag"
            ),
            "tolerance": 1.0e-12,
            "domain": "scalar",
        },
    )
    materialized["conservation_rules"] = conservation_rule_entries
    phi_constraint_expression = (
        "-1.5 * einstein_gravity_strength * "
        "(total_matter_density + total_radiation_density) "
        "/ metric_denominator"
    )
    psi_closure_expression = (
        "Phi - 3.0 * einstein_gravity_strength * total_neutrino_shear "
        "/ metric_denominator"
    )
    materialized["constraints"] = {
        "phi_constraint": {
            "target": phi_state_name,
            "expression": (
                f"2.0 * ({phi_constraint_expression})"
                if sync_gauge
                else phi_constraint_expression
            ),
            "role": "constraint",
        }
    }
    materialized["closures"] = {
        "psi_closure": {
            "target": psi_state_name,
            "expression": (
                f"2.0 * ({psi_closure_expression})"
                if sync_gauge
                else psi_closure_expression
            ),
            "role": "closure",
        }
    }
    materialized["sources"] = {
        "temperature_monopole": {
            "expression": (
                "visibility * (theta_gamma0 + Psi + "
                "0.25 * polarization_moment)"
            ),
            "role": "monopole",
        },
        "temperature_doppler": {
            "expression": "visibility * 3.0 * theta_gamma1",
            "role": "doppler",
        },
        "temperature_isw": {
            "expression": "exp(-tau) * (Psi_tau - Phi_tau)",
            "role": "isw",
        },
        "polarization_source": {
            "expression": "0.75 * visibility * polarization_moment",
            "role": "polarization",
        },
        "polarization_b_source": {
            "expression": "polarization_b_mode_seed",
            "role": "polarization_b",
        },
        "lensing_potential": {
            "expression": "exp(-tau) * (Phi + Psi)",
            "role": "potential",
        },
    }
    materialized["observables"] = {
        "temperature": {
            "kind": "transfer_component",
            "projection": "line_of_sight_temperature",
            "source_terms": {
                "monopole": "temperature_monopole",
                "doppler": "temperature_doppler",
                "isw": "temperature_isw",
            },
        },
        "polarization_e": {
            "kind": "transfer_component",
            "projection": "line_of_sight_polarization_e",
            "source_terms": {"polarization": "polarization_source"},
        },
        "polarization_b": {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {
                "polarization_b": "polarization_b_source",
            },
        },
        "lensing_potential": {
            "kind": "transfer_component",
            "projection": "line_of_sight_lensing_potential",
            "source_terms": {
                "potential": "lensing_potential",
            },
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
        "BB": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
        },
        "PP": {
            "kind": "angular_power_spectrum",
            "primary": "lensing_potential",
            "secondary": "lensing_potential",
        },
        "TP": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "lensing_potential",
        },
        "EP": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "lensing_potential",
        },
    }
    initial_conditions: dict[str, Any] = {}
    for variable_name, expression in sorted(
        _scalar_hierarchy_base_seed_expressions(initial_mode).items()
    ):
        if variable_name not in variables:
            continue
        initial_conditions[f"{variable_name}_seed"] = {
            "target": {
                "variable": variable_name,
                "wrt": "tau",
                "order": 0,
            },
            "expression": expression,
        }
    for required_name in (
        "theta_gamma0",
        "theta_gamma1",
        "theta_gamma2",
        "e_gamma0",
        "e_gamma1",
        "e_gamma2",
        "polarization_b_mode_seed",
        "delta_b",
        "theta_b",
        "delta_c",
        "theta_c",
        "delta_nu",
        "theta_nu",
        "sigma_nu",
    ):
        initial_conditions.setdefault(
            f"{required_name}_seed",
            {
                "target": {
                    "variable": required_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            },
        )
    if has_massive_neutrino:
        for required_name in (
            "delta_nu_massive",
            "theta_nu_massive",
            "sigma_nu_massive",
        ):
            initial_conditions.setdefault(
                f"{required_name}_seed",
                {
                    "target": {
                        "variable": required_name,
                        "wrt": "tau",
                        "order": 0,
                    },
                    "expression": "0.0",
                },
            )
    for moment in range(3, photon_l_max + 1):
        initial_conditions[f"{_scalar_temperature_name(moment)}_seed"] = {
            "target": {
                "variable": _scalar_temperature_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
        initial_conditions[f"{_scalar_polarization_name(moment)}_seed"] = {
            "target": {
                "variable": _scalar_polarization_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    for moment in range(3, neutrino_l_max + 1):
        initial_conditions[f"{_scalar_neutrino_name(moment)}_seed"] = {
            "target": {
                "variable": _scalar_neutrino_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        for moment in range(3, massive_neutrino_l_max + 1):
            initial_conditions[
                f"{_scalar_massive_neutrino_name(moment)}_seed"
            ] = {
                "target": {
                    "variable": _scalar_massive_neutrino_name(moment),
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.0",
            }
        for q_index in range(massive_neutrino_grid_count):
            q_log_derivative_name = (
                _scalar_massive_neutrino_distribution_log_derivative_name(
                    q_index
                )
            )
            q_delta_name = _scalar_massive_neutrino_q_name(
                q_index,
                0,
            )
            q_theta_name = _scalar_massive_neutrino_q_name(
                q_index,
                1,
            )
            q_sigma_name = _scalar_massive_neutrino_q_name(
                q_index,
                2,
            )
            initial_conditions[f"{q_delta_name}_seed"] = {
                "target": {
                    "variable": q_delta_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": f"-2.0 * seed * {q_log_derivative_name}",
            }
            initial_conditions[f"{q_theta_name}_seed"] = {
                "target": {
                    "variable": q_theta_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": (
                    f"(acoustic_k * eta_initial / 6.0) * seed * "
                    f"{q_log_derivative_name}"
                ),
            }
            initial_conditions[f"{q_sigma_name}_seed"] = {
                "target": {
                    "variable": q_sigma_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": (
                    "(acoustic_k * eta_initial) * "
                    "(acoustic_k * eta_initial) * seed / 15.0 * "
                    f"{q_log_derivative_name}"
                ),
            }
            for moment in range(3, massive_neutrino_l_max + 1):
                q_name = _scalar_massive_neutrino_q_name(
                    q_index,
                    moment,
                )
                initial_conditions[f"{q_name}_seed"] = {
                    "target": {
                        "variable": q_name,
                        "wrt": "tau",
                        "order": 0,
                    },
                    "expression": "0.0",
                }
    materialized["initial_conditions"] = initial_conditions
    family_entries = copy.deepcopy(dict(initial_condition_families))
    if initial_mode in family_entries:
        family_entries[initial_mode]["members"] = list(
            sorted(initial_conditions)
        )
    materialized["initial_condition_families"] = family_entries
    materialized["boundary_conditions"] = {}
    return materialized, True


_STANDARD_BACKEND_KEYS = {"uses_standard_perturbations"}
_NONSTANDARD_BACKEND_KEYS = {
    "implemented",
    "native_solver_required",
}
_SUPPORTED_GAUGES = {
    "conformal_newtonian",
    "gauge_invariant",
    "synchronous",
    "unspecified",
}
_SUPPORTED_CONDITION_ANCHORS = {"end", "start"}
_SUPPORTED_OBSERVABLE_KINDS = {
    "angular_power_spectrum",
    "transfer_component",
}
_COMPILED_CONTRACT_RESULTS: dict[
    tuple[Any, ...], "PerturbationContractData"
] = {}


@lru_cache(maxsize=256)
def _get_cached_perturbation_contract(
    cache_key: tuple[Any, ...],
) -> "PerturbationContractData":
    """Return a cached contract for ``cache_key``."""

    return _COMPILED_CONTRACT_RESULTS[cache_key]


@dataclass(frozen=True, slots=True)
class PerturbationVariableData:
    """Immutable metadata for one declared graph variable."""

    name: str
    kind: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    units: str | None = None
    gauge_role: str | None = None
    source_role: str | None = None
    projection_role: str | None = None
    tensor_character: str | None = None
    parity: str | None = None
    rank: int | None = None
    spin: float | None = None


@dataclass(frozen=True, slots=True)
class PerturbationCompiledExpressionData:
    """Picklable stack program for one validated declared expression."""

    expression: str
    dependencies: tuple[str, ...]
    program: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PerturbationDerivedData:
    """Immutable metadata for one declared derived graph symbol."""

    name: str
    kind: str
    expression: str | None = None
    variable: str | None = None
    wrt: str | None = None
    order: int | None = None
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationDerivativeLhsData:
    """Immutable typed left-hand side for one differential equation."""

    kind: str
    variable: str
    wrt: str
    order: int


@dataclass(frozen=True, slots=True)
class PerturbationEquationData:
    """Immutable metadata for one declared differential equation."""

    name: str
    lhs: PerturbationDerivativeLhsData
    rhs: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_rhs: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationClosureData:
    """Immutable metadata for one declared algebraic closure relation."""

    name: str
    target: str
    expression: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationConstraintData:
    """Immutable metadata for one declared algebraic constraint."""

    name: str
    target: str
    expression: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationSourceData:
    """Immutable metadata for one declared observable source term."""

    name: str
    expression: str
    role: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationObservableData:
    """Immutable metadata for one declared observable mapping."""

    name: str
    kind: str
    projection: str | None = None
    kernel: str | None = None
    primary: str | None = None
    secondary: str | None = None
    source_terms: FrozenMapping = field(default_factory=FrozenMapping)
    required_projection_roles: tuple[str, ...] = ()
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    output_role: str | None = None
    sector: str | None = None
    parity: str | None = None
    spin: float | None = None
    tensor_character: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationConditionTargetData:
    """Immutable target descriptor for an initial/boundary condition."""

    variable: str
    wrt: str
    order: int


@dataclass(frozen=True, slots=True)
class PerturbationConditionData:
    """Immutable metadata for an initial or boundary condition."""

    name: str
    target: PerturbationConditionTargetData
    expression: str
    anchor: str = "start"
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationValidityData:
    """Immutable validity declaration for the declared graph."""

    regimes: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationBackendMappingData:
    """Immutable backend execution metadata."""

    backend: str
    uses_standard_perturbations: bool | None = None
    native_solver_required: bool | None = None
    implemented: bool | None = None


@dataclass(frozen=True, slots=True)
class PerturbationSectorData:
    """Immutable sector metadata for one hierarchy-capable contract."""

    name: str
    description: str | None = None
    notes: str | None = None
    tensor_character: str | None = None
    hierarchy_families: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    supported_gauges: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationSpeciesData:
    """Immutable species metadata for one hierarchy-capable contract."""

    name: str
    sector: str | None = None
    hierarchy_family: str | None = None
    description: str | None = None
    notes: str | None = None
    equation_of_state: str | None = None
    sound_speed: str | None = None
    anisotropic_stress: str | None = None
    background_reference: str | None = None
    collision_operators: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationHierarchyFamilyData:
    """Immutable hierarchy-family metadata for native CMB contracts."""

    name: str
    sector: str | None = None
    description: str | None = None
    notes: str | None = None
    species: tuple[str, ...] = ()
    multipole_symbol: str | None = None
    closure: str | None = None
    default_l_max: int | None = None
    momentum_grid: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationCollisionOperatorData:
    """Immutable collision-operator metadata for one declared contract."""

    name: str
    description: str | None = None
    notes: str | None = None
    sector: str | None = None
    species: tuple[str, ...] = ()
    expression: str | None = None
    counterpart: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationInteractionData:
    """Immutable interaction metadata for one declared contract."""

    name: str
    description: str | None = None
    notes: str | None = None
    sector: str | None = None
    species: tuple[str, ...] = ()
    expression: str | None = None
    counterpart: str | None = None
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationConservationRuleData:
    """Immutable conservation-check metadata for one declared contract."""

    name: str
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    kind: str | None = None
    expression: str | None = None
    tolerance: float = 0.0
    dependencies: tuple[str, ...] = ()
    compiled_expression: PerturbationCompiledExpressionData | None = None


@dataclass(frozen=True, slots=True)
class PerturbationInitialConditionFamilyData:
    """Immutable metadata grouping related initial-condition declarations."""

    name: str
    sector: str | None = None
    description: str | None = None
    notes: str | None = None
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PerturbationProjectionTypingData:
    """Immutable projection-typing metadata for native observables."""

    name: str
    sector: str | None = None
    description: str | None = None
    notes: str | None = None
    kernel: str | None = None
    source_roles: tuple[str, ...] = ()
    observable_kinds: tuple[str, ...] = ()
    parity: str | None = None
    spin: float | None = None


@dataclass(frozen=True, slots=True)
class PerturbationProjectionExtensionData:
    """Immutable declared projection-extension metadata."""

    name: str
    base_projection: str
    description: str | None = None
    notes: str | None = None
    kernel: str | None = None
    required_roles: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ()
    required_projection_roles: tuple[str, ...] = ()
    requires_odd_parity_source: bool = False


@dataclass(frozen=True, slots=True)
class PerturbationDependencyGraphSummaryData:
    """Immutable summary of declared graph dependencies."""

    variable_names: tuple[str, ...]
    derived_names: tuple[str, ...]
    equation_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    closure_names: tuple[str, ...]
    interaction_names: tuple[str, ...]
    conservation_rule_names: tuple[str, ...]
    source_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    initial_condition_names: tuple[str, ...]
    boundary_condition_names: tuple[str, ...]
    independent_variables_used: tuple[str, ...]
    model_parameters_used: tuple[str, ...]
    background_references_used: tuple[str, ...]
    derived_dependencies: FrozenMapping
    equation_dependencies: FrozenMapping
    constraint_dependencies: FrozenMapping
    closure_dependencies: FrozenMapping
    interaction_dependencies: FrozenMapping
    conservation_rule_dependencies: FrozenMapping
    source_dependencies: FrozenMapping
    observable_dependencies: FrozenMapping
    initial_condition_dependencies: FrozenMapping
    boundary_condition_dependencies: FrozenMapping
    evaluation_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerturbationContractData:
    """Immutable internal representation of a declared CMB graph."""

    model_name: str
    backend: str
    contract_version: int
    standard: bool
    gauge: str
    variables: FrozenMapping
    derived: FrozenMapping
    equations: FrozenMapping
    constraints: FrozenMapping
    closures: FrozenMapping
    sources: FrozenMapping
    observables: FrozenMapping
    initial_conditions: FrozenMapping
    boundary_conditions: FrozenMapping
    numerics: FrozenMapping
    validity: PerturbationValidityData
    backend_mapping: FrozenMapping
    dependency_graph_summary: PerturbationDependencyGraphSummaryData
    manifest_summary: FrozenMapping
    sectors: FrozenMapping = field(default_factory=FrozenMapping)
    species: FrozenMapping = field(default_factory=FrozenMapping)
    hierarchy_families: FrozenMapping = field(default_factory=FrozenMapping)
    collision_operators: FrozenMapping = field(default_factory=FrozenMapping)
    interactions: FrozenMapping = field(default_factory=FrozenMapping)
    conservation_rules: FrozenMapping = field(default_factory=FrozenMapping)
    initial_condition_families: FrozenMapping = field(
        default_factory=FrozenMapping
    )
    projection_extensions: FrozenMapping = field(default_factory=FrozenMapping)
    projection_typing: FrozenMapping = field(default_factory=FrozenMapping)
    accuracy_controls: FrozenMapping = field(default_factory=FrozenMapping)


@lru_cache(maxsize=4096)
def _collect_expression_names(expr: str) -> tuple[str, ...]:
    """Return the symbol names referenced by ``expr``."""

    node = _parse_safe_expression(expr)
    names: list[str] = []
    seen: set[str] = set()
    for current in ast.walk(node):
        if not isinstance(current, ast.Name):
            continue
        if current.id in _ALLOWED_MATH_FUNCS:
            continue
        if current.id in _ALLOWED_CONSTANTS:
            continue
        if current.id in seen:
            continue
        seen.add(current.id)
        names.append(current.id)
    return tuple(names)


@lru_cache(maxsize=4096)
def _compile_expression_program(
    expr: str,
) -> tuple[tuple[str, Any], ...]:
    """Return a picklable stack program for one validated expression."""

    node = _parse_safe_expression(expr)
    program: list[tuple[str, Any]] = []

    def _visit(current: ast.AST) -> None:
        """Append stack-machine instructions for ``current``."""

        if isinstance(current, ast.Expression):
            _visit(current.body)
            return
        if isinstance(current, ast.Constant):
            if not isinstance(current.value, (int, float)):
                raise ValueError("non-numeric literal")
            program.append(("const", float(current.value)))
            return
        if isinstance(current, ast.Name):
            program.append(("name", current.id))
            return
        if isinstance(current, ast.BinOp):
            opcode = _COMPILED_BINARY_OPCODE_NAMES.get(type(current.op))
            if opcode is None:
                raise ValueError("operator not allowed")
            _visit(current.left)
            _visit(current.right)
            program.append(("binary", opcode))
            return
        if isinstance(current, ast.UnaryOp):
            opcode = _COMPILED_UNARY_OPCODE_NAMES.get(type(current.op))
            if opcode is None:
                raise ValueError("operator not allowed")
            _visit(current.operand)
            program.append(("unary", opcode))
            return
        if isinstance(current, ast.Call):
            if not isinstance(current.func, ast.Name):
                raise ValueError("invalid function call")
            if current.keywords:
                raise ValueError("keyword arguments not supported")
            for argument in current.args:
                _visit(argument)
            program.append(
                ("call", (current.func.id, len(tuple(current.args))))
            )
            return
        raise ValueError("expression not allowed")

    _visit(node)
    return tuple(program)


@lru_cache(maxsize=4096)
def _compile_expression_plan(
    expr: str,
    *,
    dependencies: tuple[str, ...] | None = None,
) -> PerturbationCompiledExpressionData:
    """Return picklable evaluator metadata for one validated expression."""

    dependency_names = (
        _collect_expression_names(expr)
        if dependencies is None
        else tuple(dependencies)
    )
    return PerturbationCompiledExpressionData(
        expression=expr,
        dependencies=dependency_names,
        program=_compile_expression_program(expr),
    )


def _evaluate_compiled_expression_noerr(
    expression_data: PerturbationCompiledExpressionData,
    env: Mapping[str, Any],
) -> Any:
    """Evaluate one compiled expression against ``env`` without errstate."""

    stack: list[Any] = []
    for opcode, payload in expression_data.program:
        if opcode == "const":
            stack.append(payload)
            continue
        if opcode == "name":
            if payload in env:
                stack.append(env[payload])
                continue
            if payload in _ALLOWED_CONSTANTS:
                stack.append(_ALLOWED_CONSTANTS[payload])
                continue
            raise ValueError(f"name '{payload}' not allowed")
        if opcode == "binary":
            right = stack.pop()
            left = stack.pop()
            stack.append(_COMPILED_BINARY_OPERATORS[payload](left, right))
            continue
        if opcode == "unary":
            stack.append(_COMPILED_UNARY_OPERATORS[payload](stack.pop()))
            continue
        if opcode == "call":
            func_name, arg_count = payload
            func = _ALLOWED_MATH_FUNCS.get(func_name)
            if func is None:
                raise ValueError(f"function '{func_name}' not allowed")
            args = [stack.pop() for _ in range(int(arg_count))]
            args.reverse()
            stack.append(func(*args))
            continue
        raise ValueError("expression not allowed")
    if len(stack) != 1:
        raise ValueError(
            "Compiled expression evaluation did not produce one result"
        )
    return stack[0]


def evaluate_compiled_expression(
    expression_data: PerturbationCompiledExpressionData,
    env: Mapping[str, Any],
) -> Any:
    """Evaluate one compiled declared expression against ``env``."""

    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _evaluate_compiled_expression_noerr(expression_data, env)


def _validate_entry_keys(
    *,
    entry: Mapping[str, Any],
    allowed_keys: set[str],
    label: str,
) -> None:
    """Reject unknown keys inside one contract entry."""

    entry_keys = {str(key) for key in entry.keys()}
    invalid_keys = entry_keys - allowed_keys
    if invalid_keys:
        invalid_str = ", ".join(sorted(invalid_keys))
        raise ValueError(f"Unknown key(s) in {label}: {invalid_str}")


def _validate_string(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    """Return ``value`` as a non-empty string unless ``allow_empty``."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ValueError(f"{label} must not be empty")
    return cleaned


def _validate_optional_string(
    value: Any,
    *,
    label: str,
) -> str | None:
    """Return ``value`` as ``str`` when present."""

    if value is None:
        return None
    return _validate_string(value, label=label)


def _validate_optional_int(
    value: Any,
    *,
    label: str,
) -> int | None:
    """Return ``value`` as an integer when present."""

    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _validate_optional_float(
    value: Any,
    *,
    label: str,
) -> float | None:
    """Return ``value`` as a float when present."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _validate_regimes(value: Any) -> tuple[str, ...]:
    """Return a validated tuple of declared validity regimes."""

    if not isinstance(value, list):
        raise ValueError("cmb.perturbations.validity.regimes must be a list")
    cleaned: list[str] = []
    for item in value:
        cleaned.append(
            _validate_string(
                item,
                label="cmb.perturbations.validity.regimes entry",
            )
        )
    if not cleaned:
        raise ValueError(
            "cmb.perturbations.validity.regimes must not be empty"
        )
    return tuple(cleaned)


def _validate_optional_string_list(
    value: Any,
    *,
    label: str,
) -> tuple[str, ...]:
    """Return ``value`` as a deduplicated string tuple when present."""

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    cleaned: list[str] = []
    for item in value:
        cleaned.append(_validate_string(item, label=f"{label} entry"))
    return _dedupe_names(cleaned)


def _replace_and_validate_expression(
    expression: Any,
    *,
    label: str,
    replacements: Mapping[str, str],
    allowed_names: set[str],
) -> tuple[str, tuple[str, ...]]:
    """Return a cleaned, validated expression and its dependencies."""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"{label} must be a non-empty string expression")
    clean_expression = _replace_latex_tokens(expression, replacements)
    names = _collect_expression_names(clean_expression)
    unknown = sorted(set(names) - allowed_names)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(
            f"{label} references unknown symbol(s): {unknown_str}"
        )
    _validate_safe_expression(clean_expression, allowed_names)
    return clean_expression, names


def _dedupe_names(names: Sequence[str]) -> tuple[str, ...]:
    """Return ``names`` in the order of first appearance without duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _normalize_accuracy_control_value(
    value: Any,
    *,
    label: str,
) -> Any:
    """Return a manifest-safe accuracy-control value."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, numpy.integer, numpy.floating)):
        return float(value)
    if isinstance(value, str):
        return _validate_string(value, label=label)
    if isinstance(value, (list, tuple)):
        normalized: list[Any] = []
        for index, item in enumerate(value):
            normalized.append(
                _normalize_accuracy_control_value(
                    item,
                    label=f"{label}[{index}]",
                )
            )
        return tuple(normalized)
    if isinstance(value, Mapping):
        return FrozenMapping(
            {
                _validate_string(
                    key,
                    label=f"{label} key",
                ): _normalize_accuracy_control_value(
                    item,
                    label=f"{label}.{key}",
                )
                for key, item in value.items()
            }
        )
    raise ValueError(
        f"{label} must contain scalar, mapping, or list-like scalar values"
    )


def _compile_sector_metadata(
    sector_defs: Mapping[str, Any],
) -> dict[str, PerturbationSectorData]:
    """Return validated sector metadata declarations."""

    compiled: dict[str, PerturbationSectorData] = {}
    for sector_name, sector_def in sector_defs.items():
        name = _validate_string(
            sector_name,
            label="Perturbation sector name",
        )
        if not isinstance(sector_def, Mapping):
            raise ValueError(f"Perturbation sector '{name}' must be a mapping")
        _validate_entry_keys(
            entry=sector_def,
            allowed_keys=_SUPPORTED_SECTOR_KEYS,
            label=f"cmb.perturbations.sectors.{name}",
        )
        supported_gauges = _validate_optional_string_list(
            sector_def.get("supported_gauges"),
            label=f"cmb.perturbations.sectors.{name}.supported_gauges",
        )
        for gauge_name in supported_gauges:
            if gauge_name not in _SUPPORTED_GAUGES:
                raise ValueError(
                    "cmb.perturbations.sectors."
                    f"{name}.supported_gauges contains unsupported gauge "
                    f"'{gauge_name}'"
                )
        compiled[name] = PerturbationSectorData(
            name=name,
            description=_validate_optional_string(
                sector_def.get("description"),
                label=f"cmb.perturbations.sectors.{name}.description",
            ),
            notes=_validate_optional_string(
                sector_def.get("notes"),
                label=f"cmb.perturbations.sectors.{name}.notes",
            ),
            tensor_character=_validate_optional_string(
                sector_def.get("tensor_character"),
                label=f"cmb.perturbations.sectors.{name}.tensor_character",
            ),
            hierarchy_families=_validate_optional_string_list(
                sector_def.get("hierarchy_families"),
                label=(
                    "cmb.perturbations.sectors." f"{name}.hierarchy_families"
                ),
            ),
            species=_validate_optional_string_list(
                sector_def.get("species"),
                label=f"cmb.perturbations.sectors.{name}.species",
            ),
            supported_gauges=supported_gauges,
        )
    return compiled


def _compile_hierarchy_family_metadata(
    hierarchy_defs: Mapping[str, Any],
    *,
    sector_names: set[str],
) -> dict[str, PerturbationHierarchyFamilyData]:
    """Return validated hierarchy-family metadata declarations."""

    compiled: dict[str, PerturbationHierarchyFamilyData] = {}
    for family_name, family_def in hierarchy_defs.items():
        name = _validate_string(
            family_name,
            label="Perturbation hierarchy family name",
        )
        if not isinstance(family_def, Mapping):
            raise ValueError(
                f"Perturbation hierarchy family '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=family_def,
            allowed_keys=_SUPPORTED_HIERARCHY_FAMILY_KEYS,
            label=f"cmb.perturbations.hierarchy_families.{name}",
        )
        sector = _validate_optional_string(
            family_def.get("sector"),
            label=f"cmb.perturbations.hierarchy_families.{name}.sector",
        )
        if sector is not None and sector not in sector_names:
            raise ValueError(
                "cmb.perturbations.hierarchy_families."
                f"{name}.sector references unknown sector '{sector}'"
            )
        compiled[name] = PerturbationHierarchyFamilyData(
            name=name,
            sector=sector,
            description=_validate_optional_string(
                family_def.get("description"),
                label=(
                    "cmb.perturbations.hierarchy_families."
                    f"{name}.description"
                ),
            ),
            notes=_validate_optional_string(
                family_def.get("notes"),
                label=f"cmb.perturbations.hierarchy_families.{name}.notes",
            ),
            species=_validate_optional_string_list(
                family_def.get("species"),
                label=f"cmb.perturbations.hierarchy_families.{name}.species",
            ),
            multipole_symbol=_validate_optional_string(
                family_def.get("multipole_symbol"),
                label=(
                    "cmb.perturbations.hierarchy_families."
                    f"{name}.multipole_symbol"
                ),
            ),
            closure=_validate_optional_string(
                family_def.get("closure"),
                label=f"cmb.perturbations.hierarchy_families.{name}.closure",
            ),
            default_l_max=_validate_optional_int(
                family_def.get("default_l_max"),
                label=(
                    "cmb.perturbations.hierarchy_families."
                    f"{name}.default_l_max"
                ),
            ),
            momentum_grid=_validate_optional_string(
                family_def.get("momentum_grid"),
                label=(
                    "cmb.perturbations.hierarchy_families."
                    f"{name}.momentum_grid"
                ),
            ),
        )
    return compiled


def _compile_species_metadata(
    species_defs: Mapping[str, Any],
    *,
    sector_names: set[str],
    hierarchy_family_names: set[str],
) -> dict[str, PerturbationSpeciesData]:
    """Return validated species metadata declarations."""

    compiled: dict[str, PerturbationSpeciesData] = {}
    for species_name, species_def in species_defs.items():
        name = _validate_string(
            species_name,
            label="Perturbation species name",
        )
        if not isinstance(species_def, Mapping):
            raise ValueError(
                f"Perturbation species '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=species_def,
            allowed_keys=_SUPPORTED_SPECIES_KEYS,
            label=f"cmb.perturbations.species.{name}",
        )
        sector = _validate_optional_string(
            species_def.get("sector"),
            label=f"cmb.perturbations.species.{name}.sector",
        )
        if sector is not None and sector not in sector_names:
            raise ValueError(
                "cmb.perturbations.species."
                f"{name}.sector references unknown sector '{sector}'"
            )
        hierarchy_family = _validate_optional_string(
            species_def.get("hierarchy_family"),
            label=f"cmb.perturbations.species.{name}.hierarchy_family",
        )
        if (
            hierarchy_family is not None
            and hierarchy_family not in hierarchy_family_names
        ):
            raise ValueError(
                "cmb.perturbations.species."
                f"{name}.hierarchy_family references unknown family "
                f"'{hierarchy_family}'"
            )
        compiled[name] = PerturbationSpeciesData(
            name=name,
            sector=sector,
            hierarchy_family=hierarchy_family,
            description=_validate_optional_string(
                species_def.get("description"),
                label=f"cmb.perturbations.species.{name}.description",
            ),
            notes=_validate_optional_string(
                species_def.get("notes"),
                label=f"cmb.perturbations.species.{name}.notes",
            ),
            equation_of_state=_validate_optional_string(
                species_def.get("equation_of_state"),
                label=(
                    "cmb.perturbations.species." f"{name}.equation_of_state"
                ),
            ),
            sound_speed=_validate_optional_string(
                species_def.get("sound_speed"),
                label=f"cmb.perturbations.species.{name}.sound_speed",
            ),
            anisotropic_stress=_validate_optional_string(
                species_def.get("anisotropic_stress"),
                label=(
                    "cmb.perturbations.species." f"{name}.anisotropic_stress"
                ),
            ),
            background_reference=_validate_optional_string(
                species_def.get("background_reference"),
                label=(
                    "cmb.perturbations.species." f"{name}.background_reference"
                ),
            ),
            collision_operators=_validate_optional_string_list(
                species_def.get("collision_operators"),
                label=(
                    "cmb.perturbations.species." f"{name}.collision_operators"
                ),
            ),
        )
    return compiled


def _compile_collision_operator_metadata(
    collision_defs: Mapping[str, Any],
    *,
    sector_names: set[str],
    species_names: set[str],
    allowed_names: set[str],
    replacements: Mapping[str, str],
) -> dict[str, PerturbationCollisionOperatorData]:
    """Return validated collision-operator metadata declarations."""

    compiled: dict[str, PerturbationCollisionOperatorData] = {}
    for operator_name, operator_def in collision_defs.items():
        name = _validate_string(
            operator_name,
            label="Perturbation collision operator name",
        )
        if not isinstance(operator_def, Mapping):
            raise ValueError(
                f"Perturbation collision operator '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=operator_def,
            allowed_keys=_SUPPORTED_COLLISION_OPERATOR_KEYS,
            label=f"cmb.perturbations.collision_operators.{name}",
        )
        sector = _validate_optional_string(
            operator_def.get("sector"),
            label=f"cmb.perturbations.collision_operators.{name}.sector",
        )
        if sector is not None and sector not in sector_names:
            raise ValueError(
                "cmb.perturbations.collision_operators."
                f"{name}.sector references unknown sector '{sector}'"
            )
        species = _validate_optional_string_list(
            operator_def.get("species"),
            label=f"cmb.perturbations.collision_operators.{name}.species",
        )
        unknown_species = sorted(set(species) - species_names)
        if unknown_species:
            unknown_str = ", ".join(unknown_species)
            raise ValueError(
                "cmb.perturbations.collision_operators."
                f"{name}.species references unknown species: {unknown_str}"
            )
        expression = operator_def.get("expression")
        compiled_expression = None
        dependencies: tuple[str, ...] = ()
        clean_expression = None
        if expression is not None:
            clean_expression, dependencies = _replace_and_validate_expression(
                expression,
                label=(
                    "cmb.perturbations.collision_operators."
                    f"{name}.expression"
                ),
                replacements=replacements,
                allowed_names=allowed_names,
            )
            compiled_expression = _compile_expression_plan(
                clean_expression,
                dependencies=dependencies,
            )
        compiled[name] = PerturbationCollisionOperatorData(
            name=name,
            description=_validate_optional_string(
                operator_def.get("description"),
                label=(
                    "cmb.perturbations.collision_operators."
                    f"{name}.description"
                ),
            ),
            notes=_validate_optional_string(
                operator_def.get("notes"),
                label=f"cmb.perturbations.collision_operators.{name}.notes",
            ),
            sector=sector,
            species=species,
            expression=clean_expression,
            counterpart=_validate_optional_string(
                operator_def.get("counterpart"),
                label=(
                    "cmb.perturbations.collision_operators."
                    f"{name}.counterpart"
                ),
            ),
            dependencies=dependencies,
            compiled_expression=compiled_expression,
        )
    return compiled


def _compile_interaction_metadata(
    interaction_defs: Mapping[str, Any],
    *,
    sector_names: set[str],
    species_names: set[str],
    allowed_names: set[str],
    replacements: Mapping[str, str],
) -> dict[str, PerturbationInteractionData]:
    """Return validated interaction metadata declarations."""

    compiled: dict[str, PerturbationInteractionData] = {}
    for interaction_name, interaction_def in interaction_defs.items():
        name = _validate_string(
            interaction_name,
            label="Perturbation interaction name",
        )
        if not isinstance(interaction_def, Mapping):
            raise ValueError(
                f"Perturbation interaction '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=interaction_def,
            allowed_keys=_SUPPORTED_INTERACTION_KEYS,
            label=f"cmb.perturbations.interactions.{name}",
        )
        sector = _validate_optional_string(
            interaction_def.get("sector"),
            label=f"cmb.perturbations.interactions.{name}.sector",
        )
        if sector is not None and sector not in sector_names:
            raise ValueError(
                "cmb.perturbations.interactions."
                f"{name}.sector references unknown sector '{sector}'"
            )
        species = _validate_optional_string_list(
            interaction_def.get("species"),
            label=f"cmb.perturbations.interactions.{name}.species",
        )
        unknown_species = sorted(set(species) - species_names)
        if unknown_species:
            unknown_str = ", ".join(unknown_species)
            raise ValueError(
                "cmb.perturbations.interactions."
                f"{name}.species references unknown species: {unknown_str}"
            )
        expression = interaction_def.get("expression")
        compiled_expression = None
        dependencies: tuple[str, ...] = ()
        clean_expression = None
        if expression is not None:
            clean_expression, dependencies = _replace_and_validate_expression(
                expression,
                label=f"cmb.perturbations.interactions.{name}.expression",
                replacements=replacements,
                allowed_names=allowed_names,
            )
            compiled_expression = _compile_expression_plan(
                clean_expression,
                dependencies=dependencies,
            )
        compiled[name] = PerturbationInteractionData(
            name=name,
            description=_validate_optional_string(
                interaction_def.get("description"),
                label=f"cmb.perturbations.interactions.{name}.description",
            ),
            notes=_validate_optional_string(
                interaction_def.get("notes"),
                label=f"cmb.perturbations.interactions.{name}.notes",
            ),
            sector=sector,
            species=species,
            expression=clean_expression,
            counterpart=_validate_optional_string(
                interaction_def.get("counterpart"),
                label=f"cmb.perturbations.interactions.{name}.counterpart",
            ),
            dependencies=dependencies,
            compiled_expression=compiled_expression,
        )
    return compiled


def _compile_conservation_rule_metadata(
    rule_defs: Mapping[str, Any],
    *,
    allowed_names: set[str],
    replacements: Mapping[str, str],
) -> dict[str, PerturbationConservationRuleData]:
    """Return validated conservation-rule declarations."""

    compiled: dict[str, PerturbationConservationRuleData] = {}
    for rule_name, rule_def in rule_defs.items():
        name = _validate_string(
            rule_name,
            label="Perturbation conservation rule name",
        )
        if not isinstance(rule_def, Mapping):
            raise ValueError(
                f"Perturbation conservation rule '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=rule_def,
            allowed_keys=_SUPPORTED_CONSERVATION_RULE_KEYS,
            label=f"cmb.perturbations.conservation_rules.{name}",
        )
        expression, dependencies = _replace_and_validate_expression(
            rule_def.get("expression"),
            label=f"cmb.perturbations.conservation_rules.{name}.expression",
            replacements=replacements,
            allowed_names=allowed_names,
        )
        tolerance = _validate_optional_float(
            rule_def.get("tolerance"),
            label=f"cmb.perturbations.conservation_rules.{name}.tolerance",
        )
        if tolerance is None or tolerance <= 0.0:
            raise ValueError(
                "cmb.perturbations.conservation_rules."
                f"{name}.tolerance must be a positive float"
            )
        compiled[name] = PerturbationConservationRuleData(
            name=name,
            description=_validate_optional_string(
                rule_def.get("description"),
                label=(
                    "cmb.perturbations.conservation_rules."
                    f"{name}.description"
                ),
            ),
            notes=_validate_optional_string(
                rule_def.get("notes"),
                label=f"cmb.perturbations.conservation_rules.{name}.notes",
            ),
            domain=_validate_optional_string(
                rule_def.get("domain"),
                label=f"cmb.perturbations.conservation_rules.{name}.domain",
            ),
            kind=_validate_optional_string(
                rule_def.get("kind"),
                label=f"cmb.perturbations.conservation_rules.{name}.kind",
            )
            or "absolute_max",
            expression=expression,
            tolerance=float(tolerance),
            dependencies=dependencies,
            compiled_expression=_compile_expression_plan(
                expression,
                dependencies=dependencies,
            ),
        )
    return compiled


def _compile_projection_extension_metadata(
    extension_defs: Mapping[str, Any],
) -> dict[str, PerturbationProjectionExtensionData]:
    """Return validated projection-extension metadata."""

    compiled: dict[str, PerturbationProjectionExtensionData] = {}
    for extension_name, extension_def in extension_defs.items():
        name = _validate_string(
            extension_name,
            label="Perturbation projection extension name",
        )
        if name in compiled:
            raise ValueError(
                f"Perturbation projection extension '{name}' is duplicated"
            )
        if name in SUPPORTED_DECLARED_TRANSFER_PROJECTIONS:
            raise ValueError(
                "Perturbation projection extension "
                f"'{name}' collides with a built-in projection"
            )
        if not isinstance(extension_def, Mapping):
            raise ValueError(
                f"Perturbation projection extension '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=extension_def,
            allowed_keys=_SUPPORTED_PROJECTION_EXTENSION_KEYS,
            label=f"cmb.perturbations.projection_extensions.{name}",
        )
        base_projection = _validate_string(
            extension_def.get("base_projection"),
            label=(
                "cmb.perturbations.projection_extensions."
                f"{name}.base_projection"
            ),
        )
        base_spec = get_declared_projection_spec(base_projection)
        kernel = _validate_optional_string(
            extension_def.get("kernel"),
            label=f"cmb.perturbations.projection_extensions.{name}.kernel",
        )
        if kernel is not None:
            resolve_declared_projection_kernel(
                base_projection,
                observable_name=f"projection extension '{name}'",
                kernel=kernel,
            )
        required_roles = _validate_optional_string_list(
            extension_def.get("required_roles"),
            label=(
                "cmb.perturbations.projection_extensions."
                f"{name}.required_roles"
            ),
        )
        allowed_roles = _validate_optional_string_list(
            extension_def.get("allowed_roles"),
            label=(
                "cmb.perturbations.projection_extensions."
                f"{name}.allowed_roles"
            ),
        )
        if (
            required_roles
            and allowed_roles
            and not set(required_roles).issubset(allowed_roles)
        ):
            raise ValueError(
                "cmb.perturbations.projection_extensions."
                f"{name}.required_roles must be included in allowed_roles"
            )
        raw_requires_odd_parity = extension_def.get(
            "requires_odd_parity_source"
        )
        if raw_requires_odd_parity is None:
            requires_odd_parity_source = base_spec.requires_odd_parity_source
        elif isinstance(raw_requires_odd_parity, bool):
            requires_odd_parity_source = raw_requires_odd_parity
        else:
            raise ValueError(
                "cmb.perturbations.projection_extensions."
                f"{name}.requires_odd_parity_source must be a boolean"
            )
        compiled[name] = PerturbationProjectionExtensionData(
            name=name,
            base_projection=base_projection,
            description=_validate_optional_string(
                extension_def.get("description"),
                label=(
                    "cmb.perturbations.projection_extensions."
                    f"{name}.description"
                ),
            ),
            notes=_validate_optional_string(
                extension_def.get("notes"),
                label=f"cmb.perturbations.projection_extensions.{name}.notes",
            ),
            kernel=kernel,
            required_roles=(
                required_roles
                if required_roles
                else tuple(base_spec.required_roles)
            ),
            allowed_roles=(
                allowed_roles
                if allowed_roles
                else tuple(base_spec.allowed_roles)
            ),
            required_projection_roles=_validate_optional_string_list(
                extension_def.get("required_projection_roles"),
                label=(
                    "cmb.perturbations.projection_extensions."
                    f"{name}.required_projection_roles"
                ),
            )
            or tuple(base_spec.required_projection_roles),
            requires_odd_parity_source=requires_odd_parity_source,
        )
    return compiled


def _compile_initial_condition_family_metadata(
    family_defs: Mapping[str, Any],
    *,
    sector_names: set[str],
    initial_condition_names: set[str],
) -> dict[str, PerturbationInitialConditionFamilyData]:
    """Return validated initial-condition family declarations."""

    compiled: dict[str, PerturbationInitialConditionFamilyData] = {}
    for family_name, family_def in family_defs.items():
        name = _validate_string(
            family_name,
            label="Perturbation initial-condition family name",
        )
        if not isinstance(family_def, Mapping):
            raise ValueError(
                "Perturbation initial-condition family "
                f"'{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=family_def,
            allowed_keys=_SUPPORTED_INITIAL_CONDITION_FAMILY_KEYS,
            label=f"cmb.perturbations.initial_condition_families.{name}",
        )
        sector = _validate_optional_string(
            family_def.get("sector"),
            label=(
                "cmb.perturbations.initial_condition_families."
                f"{name}.sector"
            ),
        )
        if sector is not None and sector not in sector_names:
            raise ValueError(
                "cmb.perturbations.initial_condition_families."
                f"{name}.sector references unknown sector '{sector}'"
            )
        members = _validate_optional_string_list(
            family_def.get("members"),
            label=(
                "cmb.perturbations.initial_condition_families."
                f"{name}.members"
            ),
        )
        unknown_members = sorted(set(members) - initial_condition_names)
        if unknown_members:
            unknown_str = ", ".join(unknown_members)
            raise ValueError(
                "cmb.perturbations.initial_condition_families."
                f"{name}.members references unknown initial conditions: "
                f"{unknown_str}"
            )
        compiled[name] = PerturbationInitialConditionFamilyData(
            name=name,
            sector=sector,
            description=_validate_optional_string(
                family_def.get("description"),
                label=(
                    "cmb.perturbations.initial_condition_families."
                    f"{name}.description"
                ),
            ),
            notes=_validate_optional_string(
                family_def.get("notes"),
                label=(
                    "cmb.perturbations.initial_condition_families."
                    f"{name}.notes"
                ),
            ),
            members=members,
        )
    return compiled


def _compile_projection_typing_metadata(
    projection_defs: Mapping[str, Any],
    *,
    sector_names: set[str],
) -> dict[str, PerturbationProjectionTypingData]:
    """Return validated projection-typing metadata declarations."""

    compiled: dict[str, PerturbationProjectionTypingData] = {}
    for projection_name, projection_def in projection_defs.items():
        name = _validate_string(
            projection_name,
            label="Perturbation projection typing name",
        )
        if not isinstance(projection_def, Mapping):
            raise ValueError(
                f"Perturbation projection typing '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=projection_def,
            allowed_keys=_SUPPORTED_PROJECTION_TYPING_KEYS,
            label=f"cmb.perturbations.projection_typing.{name}",
        )
        sector = _validate_optional_string(
            projection_def.get("sector"),
            label=f"cmb.perturbations.projection_typing.{name}.sector",
        )
        if sector is not None and sector not in sector_names:
            raise ValueError(
                "cmb.perturbations.projection_typing."
                f"{name}.sector references unknown sector '{sector}'"
            )
        compiled[name] = PerturbationProjectionTypingData(
            name=name,
            sector=sector,
            description=_validate_optional_string(
                projection_def.get("description"),
                label=(
                    "cmb.perturbations.projection_typing."
                    f"{name}.description"
                ),
            ),
            notes=_validate_optional_string(
                projection_def.get("notes"),
                label=f"cmb.perturbations.projection_typing.{name}.notes",
            ),
            kernel=_validate_optional_string(
                projection_def.get("kernel"),
                label=f"cmb.perturbations.projection_typing.{name}.kernel",
            ),
            source_roles=_validate_optional_string_list(
                projection_def.get("source_roles"),
                label=(
                    "cmb.perturbations.projection_typing."
                    f"{name}.source_roles"
                ),
            ),
            observable_kinds=_validate_optional_string_list(
                projection_def.get("observable_kinds"),
                label=(
                    "cmb.perturbations.projection_typing."
                    f"{name}.observable_kinds"
                ),
            ),
            parity=_validate_optional_string(
                projection_def.get("parity"),
                label=f"cmb.perturbations.projection_typing.{name}.parity",
            ),
            spin=_validate_optional_float(
                projection_def.get("spin"),
                label=f"cmb.perturbations.projection_typing.{name}.spin",
            ),
        )
    return compiled


def _relation_target_nodes(
    constraints: Mapping[str, PerturbationConstraintData],
    closures: Mapping[str, PerturbationClosureData],
) -> dict[str, tuple[str, str]]:
    """Return graph nodes for algebraic relation targets."""

    nodes: dict[str, tuple[str, str]] = {}
    for name, entry in constraints.items():
        nodes[entry.target] = ("constraint", name)
    for name, entry in closures.items():
        if entry.target in nodes:
            previous_kind, previous_name = nodes[entry.target]
            raise ValueError(
                "Declared graph defines algebraic target "
                f"'{entry.target}' more than once via "
                f"{previous_kind} '{previous_name}' and closure '{name}'"
            )
        nodes[entry.target] = ("closure", name)
    return nodes


def _relation_target_entries(
    constraints: Mapping[str, PerturbationConstraintData],
    closures: Mapping[str, PerturbationClosureData],
) -> dict[str, PerturbationConstraintData | PerturbationClosureData]:
    """Return algebraic relation entries keyed by target variable name."""

    relation_entries: dict[
        str, PerturbationConstraintData | PerturbationClosureData
    ] = {}
    for entry in constraints.values():
        relation_entries[entry.target] = entry
    for entry in closures.values():
        relation_entries[entry.target] = entry
    return relation_entries


def _topological_evaluation_order(
    *,
    derived: Mapping[str, PerturbationDerivedData],
    constraints: Mapping[str, PerturbationConstraintData],
    closures: Mapping[str, PerturbationClosureData],
    interactions: Mapping[str, PerturbationInteractionData],
    collision_operators: Mapping[str, PerturbationCollisionOperatorData],
) -> tuple[str, ...]:
    """Return a topological order for expression and algebraic nodes."""

    relation_nodes = _relation_target_nodes(constraints, closures)
    graph: dict[str, tuple[str, ...]] = {}
    expression_names = {
        name for name, entry in derived.items() if entry.expression is not None
    }
    interaction_names = {
        name
        for name, entry in interactions.items()
        if entry.expression is not None
    }
    collision_names = {
        name
        for name, entry in collision_operators.items()
        if entry.expression is not None
    }
    node_names = (
        expression_names
        | set(relation_nodes)
        | interaction_names
        | collision_names
    )
    for name, entry in derived.items():
        if entry.expression is None:
            continue
        graph[name] = tuple(
            dependency
            for dependency in entry.dependencies
            if dependency in node_names
        )
    for name, entry in interactions.items():
        if entry.expression is None:
            continue
        graph[name] = tuple(
            dependency
            for dependency in entry.dependencies
            if dependency in node_names
        )
    for name, entry in collision_operators.items():
        if entry.expression is None:
            continue
        graph[name] = tuple(
            dependency
            for dependency in entry.dependencies
            if dependency in node_names
        )
    for target_name, (kind, entry_name) in relation_nodes.items():
        if kind == "constraint":
            entry = constraints[entry_name]
        else:
            entry = closures[entry_name]
        graph[target_name] = tuple(
            dependency
            for dependency in entry.dependencies
            if dependency in node_names
        )

    active: set[str] = set()
    completed: set[str] = set()
    visiting: list[str] = []
    order: list[str] = []

    def _visit(node: str) -> None:
        """Depth-first walk that detects declared-graph dependency cycles."""

        if node in completed:
            return
        if node in active:
            cycle_start = visiting.index(node)
            cycle = visiting[cycle_start:] + [node]
            raise ValueError(
                "Declared graph contains a cycle: " + " -> ".join(cycle)
            )
        active.add(node)
        visiting.append(node)
        for dependency in graph.get(node, ()):
            _visit(dependency)
        visiting.pop()
        active.remove(node)
        completed.add(node)
        order.append(node)

    for node in sorted(graph):
        _visit(node)
    return tuple(order)


def _build_manifest_summary(
    *,
    model_name: str,
    backend: str,
    contract_version: int,
    standard: bool,
    gauge: str,
    variables: tuple[str, ...],
    derived: tuple[str, ...],
    equations: tuple[str, ...],
    constraints: tuple[str, ...],
    closures: tuple[str, ...],
    interactions: tuple[str, ...],
    conservation_rules: tuple[str, ...],
    sources: tuple[str, ...],
    observables: tuple[str, ...],
    initial_conditions: tuple[str, ...],
    boundary_conditions: tuple[str, ...],
    sectors: tuple[str, ...],
    species: tuple[str, ...],
    hierarchy_families: tuple[str, ...],
    collision_operators: tuple[str, ...],
    initial_condition_families: tuple[str, ...],
    projection_extensions: tuple[str, ...],
    projection_typing: tuple[str, ...],
    validity: PerturbationValidityData,
    numerics: Mapping[str, Any],
    accuracy_controls: Mapping[str, Any],
    backend_mapping: PerturbationBackendMappingData,
    dependency_summary: PerturbationDependencyGraphSummaryData,
    generated_scalar_hierarchy: bool,
    equation_wrt_by_variable: Mapping[str, str],
    boundary_condition_anchors: Mapping[str, str],
    transfer_component_contracts: Mapping[str, Mapping[str, Any]],
    angular_power_spectrum_targets: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Return a manifest-friendly summary of the compiled graph."""

    execution_route = _build_execution_route_summary(
        backend=backend,
        standard=standard,
        backend_mapping=backend_mapping,
    )
    return {
        "model_name": model_name,
        "backend": backend,
        "contract_version": contract_version,
        "standard": standard,
        "gauge": gauge,
        "variable_names": variables,
        "derived_names": derived,
        "equation_names": equations,
        "constraint_names": constraints,
        "closure_names": closures,
        "interaction_names": interactions,
        "conservation_rule_names": conservation_rules,
        "source_names": sources,
        "observable_names": observables,
        "initial_condition_names": initial_conditions,
        "boundary_condition_names": boundary_conditions,
        "sector_names": sectors,
        "species_names": species,
        "hierarchy_family_names": hierarchy_families,
        "collision_operator_names": collision_operators,
        "initial_condition_family_names": initial_condition_families,
        "projection_extension_names": projection_extensions,
        "projection_typing_names": projection_typing,
        "validity_regimes": validity.regimes,
        "validity_notes": validity.notes,
        "numerics_keys": tuple(sorted(str(key) for key in numerics)),
        "accuracy_control_keys": tuple(
            sorted(str(key) for key in accuracy_controls)
        ),
        "backend_implemented": backend_mapping.implemented,
        "backend_native_solver_required": (
            backend_mapping.native_solver_required
        ),
        "backend_uses_standard_perturbations": (
            backend_mapping.uses_standard_perturbations
        ),
        "independent_variables_used": (
            dependency_summary.independent_variables_used
        ),
        "model_parameters_used": dependency_summary.model_parameters_used,
        "background_references_used": (
            dependency_summary.background_references_used
        ),
        "evaluation_order": dependency_summary.evaluation_order,
        "equation_wrt_by_variable": {
            str(name): str(wrt_name)
            for name, wrt_name in equation_wrt_by_variable.items()
        },
        "boundary_condition_anchors": {
            str(name): str(anchor_name)
            for name, anchor_name in boundary_condition_anchors.items()
        },
        "execution_route": execution_route,
        "compilation_ownership": {
            "compiler": (
                "copernican.lib.model_coder.compile_native_cmb_runtime"
            ),
            "compiled_upstream": True,
            "hot_path_recompilation_allowed": False,
        },
        "generated_scalar_hierarchy": generated_scalar_hierarchy,
        "transfer_component_contracts": {
            str(name): {
                str(key): value for key, value in contract_data.items()
            }
            for name, contract_data in transfer_component_contracts.items()
        },
        "angular_power_spectrum_targets": {
            str(name): {
                str(key): str(value) for key, value in target_data.items()
            }
            for name, target_data in angular_power_spectrum_targets.items()
        },
    }


def _build_execution_route_summary(
    *,
    backend: str,
    standard: bool,
    backend_mapping: PerturbationBackendMappingData,
) -> dict[str, Any]:
    """Return manifest-friendly execution-route metadata."""

    if standard:
        route_id = "backend_standard_perturbations"
        prediction_engine = backend
        transfer_function_path = f"{backend}.standard"
        solver = f"{backend}_standard"
    else:
        route_id = "native_declared_graph"
        prediction_engine = "copernican_native_declared_graph"
        transfer_function_path = (
            "copernican.lib.likelihoods.cmb.copernican_cmb_solver"
        )
        solver = "declared_math_graph"
    uses_camb_prediction = bool(
        standard and str(backend).strip().lower() == "camb"
    )
    return {
        "route_id": route_id,
        "prediction_engine": prediction_engine,
        "transfer_function_path": transfer_function_path,
        "solver": solver,
        "route_ready_for_execution": bool(
            standard
            or (
                backend_mapping.native_solver_required is True
                and backend_mapping.implemented is True
            )
        ),
        "uses_backend_standard_perturbations": bool(standard),
        "uses_native_declared_graph": bool(not standard),
        "uses_camb_prediction": uses_camb_prediction,
        "uses_camb_standard_perturbations": uses_camb_prediction,
        "backend_mapping_implemented": backend_mapping.implemented,
        "backend_mapping_native_solver_required": (
            backend_mapping.native_solver_required
        ),
        "backend_mapping_uses_standard_perturbations": (
            backend_mapping.uses_standard_perturbations
        ),
    }


def compile_perturbation_contract(
    contract: Mapping[str, Any],
    *,
    model_name: str,
    backend: str,
    parameter_names: Sequence[str],
    latex_names: Sequence[str],
    background_reference_names: Sequence[str],
) -> PerturbationContractData:
    """Validate and compile a declared CMB graph contract."""

    if not isinstance(contract, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")
    contract, materialized_scalar_hierarchy = (
        _materialize_native_scalar_hierarchy_contract(contract)
    )

    cache_key = (
        _freeze_for_cache(contract),
        str(model_name),
        str(backend),
        tuple(str(name) for name in parameter_names),
        tuple(str(name) for name in latex_names),
        tuple(str(name) for name in background_reference_names),
    )
    cached_result = _COMPILED_CONTRACT_RESULTS.get(cache_key)
    if cached_result is not None:
        return cached_result

    contract_keys = {str(key) for key in contract.keys()}
    required_sections = {
        "backend_mapping",
        "contract_version",
        "gauge",
        "standard",
        "validity",
    }
    missing_keys = required_sections - contract_keys
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Missing perturbation contract key(s): {missing_str}"
        )
    invalid_keys = contract_keys - _SUPPORTED_PERTURBATION_KEYS
    if invalid_keys:
        invalid_str = ", ".join(sorted(invalid_keys))
        raise ValueError(
            f"Unknown perturbation contract key(s): {invalid_str}"
        )

    contract_version = contract.get("contract_version")
    if isinstance(contract_version, bool) or not isinstance(
        contract_version, int
    ):
        raise ValueError("cmb.perturbations.contract_version must be an int")
    standard = contract.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")
    if standard:
        if contract_version not in {1, 2}:
            raise ValueError(
                "Standard perturbations must declare contract_version 1 or 2"
            )
    elif contract_version != 2:
        raise ValueError(
            "Non-standard perturbations must declare contract_version: 2"
        )

    gauge = _validate_string(
        contract.get("gauge"),
        label="cmb.perturbations.gauge",
    )
    if gauge not in _SUPPORTED_GAUGES:
        raise ValueError("cmb.perturbations.gauge is invalid")
    _validate_optional_string(
        contract.get("notes"),
        label="cmb.perturbations.notes",
    )

    sections = {
        "accuracy_controls": contract.get("accuracy_controls", {}),
        "variables": contract.get("variables", {}),
        "derived": contract.get("derived", {}),
        "equations": contract.get("equations", {}),
        "constraints": contract.get("constraints", {}),
        "closures": contract.get("closures", {}),
        "conservation_rules": contract.get("conservation_rules", {}),
        "collision_operators": contract.get("collision_operators", {}),
        "interactions": contract.get("interactions", {}),
        "sources": contract.get("sources", {}),
        "observables": contract.get("observables", {}),
        "initial_conditions": contract.get("initial_conditions", {}),
        "initial_condition_families": (
            contract.get("initial_condition_families", {})
        ),
        "boundary_conditions": contract.get("boundary_conditions", {}),
        "sectors": contract.get("sectors", {}),
        "species": contract.get("species", {}),
        "hierarchy_families": contract.get("hierarchy_families", {}),
        "projection_extensions": contract.get("projection_extensions", {}),
        "projection_typing": contract.get("projection_typing", {}),
        "validity": contract.get("validity", {}),
        "backend_mapping": contract.get("backend_mapping"),
        "numerics": contract.get("numerics", {}),
    }
    for section_name, section_value in sections.items():
        if not isinstance(section_value, Mapping):
            raise ValueError(
                f"cmb.perturbations.{section_name} must be a mapping"
            )
    projection_extension_entries = _compile_projection_extension_metadata(
        sections["projection_extensions"],
    )
    projection_typing_entries = _compile_projection_typing_metadata(
        sections["projection_typing"],
        sector_names={str(name) for name in sections["sectors"].keys()},
    )

    parameter_name_set = {str(name) for name in parameter_names}
    background_reference_set = {
        str(name) for name in background_reference_names
    } | _RUNTIME_REFERENCE_NAMES
    replacements = _build_parameter_replacements(
        parameter_names,
        latex_names,
    )

    backend_keys = {str(key) for key in sections["backend_mapping"].keys()}
    invalid_backend_keys = backend_keys - _SUPPORTED_BACKEND_KEYS
    if invalid_backend_keys:
        invalid_str = ", ".join(sorted(invalid_backend_keys))
        raise ValueError(f"Unknown perturbation backend(s): {invalid_str}")
    backend_contract = sections["backend_mapping"].get(backend)
    if not isinstance(backend_contract, Mapping):
        raise ValueError(
            f"cmb.perturbations.backend_mapping must include {backend}"
        )
    backend_contract_keys = {str(key) for key in backend_contract.keys()}
    if standard:
        invalid_standard_keys = backend_contract_keys - _STANDARD_BACKEND_KEYS
        if invalid_standard_keys:
            invalid_str = ", ".join(sorted(invalid_standard_keys))
            raise ValueError(
                "Standard perturbation mappings may only declare "
                f"uses_standard_perturbations: {invalid_str}"
            )
        if backend_contract.get("uses_standard_perturbations") is not True:
            raise ValueError(
                "cmb.perturbations.backend_mapping.camb must declare "
                "uses_standard_perturbations: true"
            )
    else:
        invalid_nonstandard_keys = (
            backend_contract_keys - _NONSTANDARD_BACKEND_KEYS
        )
        if invalid_nonstandard_keys:
            invalid_str = ", ".join(sorted(invalid_nonstandard_keys))
            raise ValueError(
                "Non-standard perturbation mappings may only declare "
                f"native_solver_required, implemented: {invalid_str}"
            )
        if backend_contract.get("native_solver_required") is not True:
            raise ValueError(
                "cmb.perturbations.backend_mapping.camb must declare "
                "native_solver_required: true"
            )
        implemented = backend_contract.get("implemented")
        if not isinstance(implemented, bool):
            raise ValueError(
                "cmb.perturbations.backend_mapping.camb.implemented must be "
                "boolean"
            )

    variable_entries: dict[str, PerturbationVariableData] = {}
    for variable_name, variable_def in sections["variables"].items():
        name = _validate_string(
            variable_name,
            label="Perturbation variable name",
        )
        if not isinstance(variable_def, Mapping):
            raise ValueError(
                f"Perturbation variable '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=variable_def,
            allowed_keys=_SUPPORTED_VARIABLE_KEYS,
            label=f"cmb.perturbations.variables.{name}",
        )
        if name in parameter_name_set or name in background_reference_set:
            raise ValueError(
                f"Perturbation variable '{name}' collides with an "
                "existing background symbol"
            )
        if name in variable_entries:
            raise ValueError(f"Perturbation variable '{name}' is duplicated")
        variable_entries[name] = PerturbationVariableData(
            name=name,
            kind=_validate_string(
                variable_def.get("kind"),
                label=f"cmb.perturbations.variables.{name}.kind",
            ),
            description=_validate_optional_string(
                variable_def.get("description"),
                label=f"cmb.perturbations.variables.{name}.description",
            ),
            notes=_validate_optional_string(
                variable_def.get("notes"),
                label=f"cmb.perturbations.variables.{name}.notes",
            ),
            domain=_validate_optional_string(
                variable_def.get("domain"),
                label=f"cmb.perturbations.variables.{name}.domain",
            ),
            units=_validate_optional_string(
                variable_def.get("units"),
                label=f"cmb.perturbations.variables.{name}.units",
            ),
            gauge_role=_validate_optional_string(
                variable_def.get("gauge_role"),
                label=f"cmb.perturbations.variables.{name}.gauge_role",
            ),
            source_role=_validate_optional_string(
                variable_def.get("source_role"),
                label=f"cmb.perturbations.variables.{name}.source_role",
            ),
            projection_role=_validate_optional_string(
                variable_def.get("projection_role"),
                label=(f"cmb.perturbations.variables.{name}.projection_role"),
            ),
            tensor_character=_validate_optional_string(
                variable_def.get("tensor_character"),
                label=(f"cmb.perturbations.variables.{name}.tensor_character"),
            ),
            parity=_validate_optional_string(
                variable_def.get("parity"),
                label=f"cmb.perturbations.variables.{name}.parity",
            ),
            rank=_validate_optional_int(
                variable_def.get("rank"),
                label=f"cmb.perturbations.variables.{name}.rank",
            ),
            spin=_validate_optional_float(
                variable_def.get("spin"),
                label=f"cmb.perturbations.variables.{name}.spin",
            ),
        )

    allowed_name_pool: set[str] = set(parameter_name_set)
    allowed_name_pool.update(background_reference_set)
    allowed_name_pool.update(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
    allowed_name_pool.update(variable_entries)
    allowed_name_pool.update(
        str(name) for name in sections["collision_operators"]
    )
    allowed_name_pool.update(str(name) for name in sections["interactions"])
    momentum_grid_defs = sections["numerics"].get("momentum_grids", {})
    if isinstance(momentum_grid_defs, Mapping):
        for family_name, family_def in sections["hierarchy_families"].items():
            if not isinstance(family_def, Mapping):
                continue
            momentum_grid_name = family_def.get("momentum_grid")
            if not momentum_grid_name:
                continue
            grid_def = momentum_grid_defs.get(str(momentum_grid_name), {})
            if not isinstance(grid_def, Mapping):
                continue
            grid_count = max(0, int(grid_def.get("count", 0)))
            for index in range(grid_count):
                for suffix in (
                    "point",
                    "weight",
                    "velocity_ratio",
                    "pressure_ratio",
                    "mass_fraction",
                ):
                    allowed_name_pool.add(f"{family_name}_q{index}_{suffix}")
                    allowed_name_pool.add(
                        f"momentum_grid_{momentum_grid_name}_q{index}_{suffix}"
                    )

    derived_entries: dict[str, PerturbationDerivedData] = {}
    declared_derived_names = {str(name) for name in sections["derived"]}
    expression_derived_names: list[str] = []
    for derived_name, derived_def in sections["derived"].items():
        name = _validate_string(
            derived_name,
            label="Derived perturbation name",
        )
        if not isinstance(derived_def, Mapping):
            raise ValueError(
                f"Perturbation derived '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=derived_def,
            allowed_keys=_SUPPORTED_DERIVED_KEYS,
            label=f"cmb.perturbations.derived.{name}",
        )
        if (
            name in parameter_name_set
            or name in background_reference_set
            or name in variable_entries
            or name in derived_entries
        ):
            raise ValueError(
                f"Perturbation derived '{name}' collides with an "
                "existing symbol"
            )
        expression = derived_def.get("expression")
        derivative_variable = derived_def.get("variable")
        derivative_wrt = derived_def.get("wrt")
        derivative_order = derived_def.get("order")
        if expression is None:
            if derivative_variable is None:
                raise ValueError(
                    f"Perturbation derived '{name}' must declare either "
                    "expression or variable"
                )
            variable_name = _validate_string(
                derivative_variable,
                label=f"cmb.perturbations.derived.{name}.variable",
            )
            if (
                variable_name not in variable_entries
                and variable_name not in background_reference_set
            ):
                raise ValueError(
                    f"Derivative symbol '{name}' references unknown "
                    f"variable '{variable_name}'"
                )
            wrt_name = _validate_string(
                derivative_wrt,
                label=f"cmb.perturbations.derived.{name}.wrt",
            )
            if wrt_name not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
                raise ValueError(
                    f"Derivative symbol '{name}' uses unsupported wrt "
                    f"'{wrt_name}'"
                )
            order_value = _validate_optional_int(
                derivative_order,
                label=f"cmb.perturbations.derived.{name}.order",
            )
            if order_value is None or order_value < 1:
                raise ValueError(
                    f"Derivative symbol '{name}' order must be a positive "
                    "integer"
                )
            derived_entries[name] = PerturbationDerivedData(
                name=name,
                kind=_validate_string(
                    derived_def.get("kind") or "derivative_symbol",
                    label=f"cmb.perturbations.derived.{name}.kind",
                ),
                variable=variable_name,
                wrt=wrt_name,
                order=order_value,
                description=_validate_optional_string(
                    derived_def.get("description"),
                    label=f"cmb.perturbations.derived.{name}.description",
                ),
                notes=_validate_optional_string(
                    derived_def.get("notes"),
                    label=f"cmb.perturbations.derived.{name}.notes",
                ),
                domain=_validate_optional_string(
                    derived_def.get("domain"),
                    label=f"cmb.perturbations.derived.{name}.domain",
                ),
                dependencies=(variable_name,),
            )
            continue
        if derivative_variable is not None:
            raise ValueError(
                f"Perturbation derived '{name}' cannot declare both "
                "expression and variable"
            )
        clean_expression, dependencies = _replace_and_validate_expression(
            expression,
            label=f"cmb.perturbations.derived.{name}.expression",
            replacements=replacements,
            allowed_names=allowed_name_pool
            | declared_derived_names
            | set(derived_entries)
            | set(expression_derived_names),
        )
        derived_entries[name] = PerturbationDerivedData(
            name=name,
            kind=_validate_string(
                derived_def.get("kind") or "expression",
                label=f"cmb.perturbations.derived.{name}.kind",
            ),
            expression=clean_expression,
            description=_validate_optional_string(
                derived_def.get("description"),
                label=f"cmb.perturbations.derived.{name}.description",
            ),
            notes=_validate_optional_string(
                derived_def.get("notes"),
                label=f"cmb.perturbations.derived.{name}.notes",
            ),
            domain=_validate_optional_string(
                derived_def.get("domain"),
                label=f"cmb.perturbations.derived.{name}.domain",
            ),
            dependencies=dependencies,
            compiled_expression=_compile_expression_plan(
                clean_expression,
                dependencies=dependencies,
            ),
        )
        expression_derived_names.append(name)

    expression_names = {
        name
        for name, entry in derived_entries.items()
        if entry.expression is not None
    }
    all_expression_names = (
        allowed_name_pool | set(derived_entries) | expression_names
    )

    equation_entries: dict[str, PerturbationEquationData] = {}
    equation_targets: set[tuple[str, int]] = set()
    for equation_name, equation_def in sections["equations"].items():
        name = _validate_string(
            equation_name,
            label="Differential equation name",
        )
        if not isinstance(equation_def, Mapping):
            raise ValueError(
                f"Perturbation equation '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=equation_def,
            allowed_keys=_SUPPORTED_EQUATION_KEYS,
            label=f"cmb.perturbations.equations.{name}",
        )
        lhs = equation_def.get("lhs")
        if not isinstance(lhs, Mapping):
            raise ValueError(
                f"Perturbation equation '{name}' must declare typed lhs"
            )
        _validate_entry_keys(
            entry=lhs,
            allowed_keys=_SUPPORTED_LHS_KEYS,
            label=f"cmb.perturbations.equations.{name}.lhs",
        )
        lhs_kind = _validate_string(
            lhs.get("kind"),
            label=f"cmb.perturbations.equations.{name}.lhs.kind",
        )
        if lhs_kind != "derivative":
            raise ValueError(
                f"Perturbation equation '{name}' lhs kind must be "
                "derivative"
            )
        lhs_variable = _validate_string(
            lhs.get("variable"),
            label=f"cmb.perturbations.equations.{name}.lhs.variable",
        )
        if lhs_variable not in variable_entries:
            raise ValueError(
                f"Perturbation equation '{name}' references unknown "
                f"variable '{lhs_variable}'"
            )
        lhs_wrt = _validate_string(
            lhs.get("wrt"),
            label=f"cmb.perturbations.equations.{name}.lhs.wrt",
        )
        if lhs_wrt not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
            raise ValueError(
                f"Perturbation equation '{name}' uses unsupported wrt "
                f"'{lhs_wrt}'"
            )
        lhs_order = _validate_optional_int(
            lhs.get("order"),
            label=f"cmb.perturbations.equations.{name}.lhs.order",
        )
        if lhs_order is None or lhs_order < 1:
            raise ValueError(
                f"Perturbation equation '{name}' lhs order must be a "
                "positive integer"
            )
        target_key = (lhs_variable, lhs_order)
        if target_key in equation_targets:
            raise ValueError(
                f"Perturbation equation '{name}' duplicates the derivative "
                f"target for variable '{lhs_variable}' order {lhs_order}"
            )
        rhs_expression, dependencies = _replace_and_validate_expression(
            equation_def.get("rhs"),
            label=f"cmb.perturbations.equations.{name}.rhs",
            replacements=replacements,
            allowed_names=all_expression_names,
        )
        equation_entries[name] = PerturbationEquationData(
            name=name,
            lhs=PerturbationDerivativeLhsData(
                kind="derivative",
                variable=lhs_variable,
                wrt=lhs_wrt,
                order=lhs_order,
            ),
            rhs=rhs_expression,
            role=_validate_string(
                equation_def.get("role") or "differential",
                label=f"cmb.perturbations.equations.{name}.role",
            ),
            description=_validate_optional_string(
                equation_def.get("description"),
                label=f"cmb.perturbations.equations.{name}.description",
            ),
            notes=_validate_optional_string(
                equation_def.get("notes"),
                label=f"cmb.perturbations.equations.{name}.notes",
            ),
            domain=_validate_optional_string(
                equation_def.get("domain"),
                label=f"cmb.perturbations.equations.{name}.domain",
            ),
            dependencies=dependencies,
            compiled_rhs=_compile_expression_plan(
                rhs_expression,
                dependencies=dependencies,
            ),
        )
        equation_targets.add(target_key)

    evolved_variable_names = {
        entry.lhs.variable for entry in equation_entries.values()
    }

    def _compile_relations(
        relation_defs: Mapping[str, Any],
        *,
        label_prefix: str,
        relation_kind: str,
    ) -> dict[str, Any]:
        """Compile constraint or closure mappings into typed relation data."""

        compiled: dict[str, Any] = {}
        seen_targets: set[str] = set()
        for relation_name, relation_def in relation_defs.items():
            name = _validate_string(
                relation_name,
                label=f"{relation_kind.title()} relation name",
            )
            if not isinstance(relation_def, Mapping):
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' must be a mapping"
                )
            _validate_entry_keys(
                entry=relation_def,
                allowed_keys=_SUPPORTED_RELATION_KEYS,
                label=f"{label_prefix}.{name}",
            )
            target_name = _validate_string(
                relation_def.get("target"),
                label=f"{label_prefix}.{name}.target",
            )
            if target_name in parameter_name_set:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' cannot target "
                    f"parameter '{target_name}'"
                )
            if target_name in background_reference_set:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' cannot target "
                    f"background symbol '{target_name}'"
                )
            if target_name in seen_targets:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' duplicates "
                    f"target '{target_name}'"
                )
            if target_name in evolved_variable_names:
                raise ValueError(
                    f"Perturbation {relation_kind} '{name}' cannot target "
                    f"evolved variable '{target_name}'"
                )
            expression_text, dependencies = _replace_and_validate_expression(
                relation_def.get("expression"),
                label=f"{label_prefix}.{name}.expression",
                replacements=replacements,
                allowed_names=all_expression_names
                | {target_name}
                | seen_targets,
            )
            entry_kwargs = {
                "name": name,
                "target": target_name,
                "expression": expression_text,
                "role": _validate_string(
                    relation_def.get("role") or relation_kind,
                    label=f"{label_prefix}.{name}.role",
                ),
                "description": _validate_optional_string(
                    relation_def.get("description"),
                    label=f"{label_prefix}.{name}.description",
                ),
                "notes": _validate_optional_string(
                    relation_def.get("notes"),
                    label=f"{label_prefix}.{name}.notes",
                ),
                "domain": _validate_optional_string(
                    relation_def.get("domain"),
                    label=f"{label_prefix}.{name}.domain",
                ),
                "dependencies": dependencies,
                "compiled_expression": _compile_expression_plan(
                    expression_text,
                    dependencies=dependencies,
                ),
            }
            if relation_kind == "constraint":
                compiled[name] = PerturbationConstraintData(**entry_kwargs)
            else:
                compiled[name] = PerturbationClosureData(**entry_kwargs)
            seen_targets.add(target_name)
        return compiled

    constraint_entries = _compile_relations(
        sections["constraints"],
        label_prefix="cmb.perturbations.constraints",
        relation_kind="constraint",
    )
    closure_entries = _compile_relations(
        sections["closures"],
        label_prefix="cmb.perturbations.closures",
        relation_kind="closure",
    )

    source_entries: dict[str, PerturbationSourceData] = {}
    for source_name, source_def in sections["sources"].items():
        name = _validate_string(
            source_name,
            label="Source-term name",
        )
        if not isinstance(source_def, Mapping):
            raise ValueError(f"Perturbation source '{name}' must be a mapping")
        _validate_entry_keys(
            entry=source_def,
            allowed_keys=_SUPPORTED_SOURCE_KEYS,
            label=f"cmb.perturbations.sources.{name}",
        )
        expression_text, dependencies = _replace_and_validate_expression(
            source_def.get("expression"),
            label=f"cmb.perturbations.sources.{name}.expression",
            replacements=replacements,
            allowed_names=all_expression_names
            | set(_relation_target_nodes(constraint_entries, closure_entries)),
        )
        source_entries[name] = PerturbationSourceData(
            name=name,
            expression=expression_text,
            role=_validate_string(
                source_def.get("role"),
                label=f"cmb.perturbations.sources.{name}.role",
            ),
            description=_validate_optional_string(
                source_def.get("description"),
                label=f"cmb.perturbations.sources.{name}.description",
            ),
            notes=_validate_optional_string(
                source_def.get("notes"),
                label=f"cmb.perturbations.sources.{name}.notes",
            ),
            domain=_validate_optional_string(
                source_def.get("domain"),
                label=f"cmb.perturbations.sources.{name}.domain",
            ),
            dependencies=dependencies,
            compiled_expression=_compile_expression_plan(
                expression_text,
                dependencies=dependencies,
            ),
        )

    relation_entries = _relation_target_entries(
        constraint_entries,
        closure_entries,
    )

    def _reachable_variable_names(
        dependencies: Sequence[str],
        *,
        seen: set[str] | None = None,
    ) -> set[str]:
        """Return transitive variable ancestry for ``dependencies``."""

        if seen is None:
            seen = set()
        reachable: set[str] = set()
        for dependency in dependencies:
            if dependency in seen:
                continue
            seen.add(dependency)
            if dependency in variable_entries:
                reachable.add(dependency)
                continue
            if dependency in derived_entries:
                reachable.update(
                    _reachable_variable_names(
                        derived_entries[dependency].dependencies,
                        seen=seen,
                    )
                )
                continue
            if dependency in relation_entries:
                reachable.update(
                    _reachable_variable_names(
                        relation_entries[dependency].dependencies,
                        seen=seen,
                    )
                )
        return reachable

    def _supports_odd_parity_projection(
        variable_name: str,
    ) -> bool:
        """Return whether ``variable_name`` can feed a B-like projection."""

        variable_entry = variable_entries[variable_name]
        explicit_projection_role = (
            variable_entry.source_role == "polarization_b"
            or variable_entry.projection_role == "b_mode"
        )
        odd_parity = variable_entry.parity == "odd"
        has_non_scalar_character = (
            (
                variable_entry.spin is not None
                and abs(float(variable_entry.spin)) >= 1.0
            )
            or (
                variable_entry.rank is not None
                and int(variable_entry.rank) >= 1
            )
            or variable_entry.tensor_character
            in {"vector_like", "tensor_like"}
        )
        return explicit_projection_role or (
            odd_parity and has_non_scalar_character
        )

    def _supports_projection_role(
        variable_name: str,
        projection_role: str,
    ) -> bool:
        """Return whether ``variable_name`` satisfies ``projection_role``."""

        if projection_role == "b_mode":
            return _supports_odd_parity_projection(variable_name)
        variable_entry = variable_entries[variable_name]
        return variable_entry.projection_role == projection_role

    def _source_ancestry_supports_projection_roles(
        source_name: str,
        required_roles: Sequence[str],
    ) -> bool:
        """Return whether ``source_name`` ancestry satisfies all roles."""

        reachable_variables = sorted(
            _reachable_variable_names(source_entries[source_name].dependencies)
        )
        if not reachable_variables:
            return False
        return all(
            any(
                _supports_projection_role(variable_name, role_name)
                for variable_name in reachable_variables
            )
            for role_name in required_roles
        )

    def _infer_variable_sector(variable_name: str) -> str:
        """Return one coarse physical sector label for ``variable_name``."""

        variable_entry = variable_entries[variable_name]
        if variable_entry.tensor_character == "vector_like":
            return "vector"
        if variable_entry.tensor_character == "tensor_like":
            return "tensor"
        if variable_entry.tensor_character == "scalar_like":
            return "scalar"
        if variable_entry.rank is not None and int(variable_entry.rank) >= 2:
            return "tensor"
        if variable_entry.spin is not None:
            abs_spin = abs(float(variable_entry.spin))
            if abs_spin >= 2.0:
                return "tensor"
            if abs_spin >= 1.0:
                return "vector"
        return "scalar"

    def _merge_tensor_character(
        values: Sequence[str | None],
    ) -> str | None:
        """Return the strongest tensor-character label in ``values``."""

        labels = {str(value) for value in values if value is not None}
        if "tensor_like" in labels:
            return "tensor_like"
        if "vector_like" in labels:
            return "vector_like"
        if "scalar_like" in labels:
            return "scalar_like"
        return None

    def _infer_source_ancestry_metadata(
        source_name: str,
    ) -> dict[str, Any]:
        """Return sector and projection metadata for ``source_name``."""

        reachable_variables = sorted(
            _reachable_variable_names(source_entries[source_name].dependencies)
        )
        sectors = tuple(
            sorted(
                {
                    _infer_variable_sector(variable_name)
                    for variable_name in reachable_variables
                }
            )
        )
        tensor_character = _merge_tensor_character(
            tuple(
                variable_entries[variable_name].tensor_character
                for variable_name in reachable_variables
            )
        )
        parities = {
            str(variable_entries[variable_name].parity)
            for variable_name in reachable_variables
            if variable_entries[variable_name].parity is not None
        }
        spins = [
            abs(float(variable_entries[variable_name].spin))
            for variable_name in reachable_variables
            if variable_entries[variable_name].spin is not None
        ]
        return {
            "reachable_variables": tuple(reachable_variables),
            "sectors": sectors,
            "tensor_character": tensor_character,
            "parity": next(iter(parities)) if len(parities) == 1 else None,
            "spin": max(spins) if spins else None,
        }

    def _projection_output_role(projection: str) -> str:
        """Return the observable role emitted by ``projection``."""

        if projection in projection_extension_entries:
            projection = str(
                projection_extension_entries[projection].base_projection
            )
        if projection == "line_of_sight_temperature":
            return "temperature"
        if projection in {
            "line_of_sight_polarization_e",
            "spin2_e_mode",
        }:
            return "polarization_e"
        if projection == "spin2_b_mode":
            return "polarization_b"
        if projection in {
            "line_of_sight_lensing_potential",
            "line_of_sight_potential",
        }:
            return "potential"
        return "signal"

    def _match_projection_typing_entry(
        *,
        observable_name: str,
        observable_kind: str,
        kernel: str | None,
        source_roles: Mapping[str, str],
    ) -> PerturbationProjectionTypingData | None:
        """Return one matching projection-typing entry when present."""

        explicit_entry = projection_typing_entries.get(observable_name)
        source_role_names = set(source_roles)

        def _entry_matches(
            entry: PerturbationProjectionTypingData,
        ) -> bool:
            """Return whether ``entry`` matches the observable contract."""

            if (
                entry.observable_kinds
                and observable_kind not in entry.observable_kinds
            ):
                return False
            if entry.kernel is not None and entry.kernel != kernel:
                return False
            if (
                entry.source_roles
                and set(entry.source_roles) != source_role_names
            ):
                return False
            return True

        if explicit_entry is not None:
            if not _entry_matches(explicit_entry):
                raise ValueError(
                    f"Perturbation observable '{observable_name}' "
                    "projection_typing metadata does not match its "
                    "kernel or source-term roles"
                )
            return explicit_entry

        matches = [
            entry
            for entry in projection_typing_entries.values()
            if _entry_matches(entry)
        ]
        if len(matches) > 1:
            match_names = ", ".join(sorted(entry.name for entry in matches))
            raise ValueError(
                f"Perturbation observable '{observable_name}' matches more "
                "than one projection_typing entry: "
                f"{match_names}"
            )
        if not matches:
            return None
        return matches[0]

    def _resolve_transfer_component_metadata(
        *,
        observable_name: str,
        projection: str,
        observable_kind: str,
        kernel: str | None,
        source_term_refs: Mapping[str, str],
    ) -> tuple[str, str | None, str | None, float | None, str | None]:
        """Return output-role and sector metadata for one component."""

        source_sector_names: set[str] = set()
        tensor_character_values: list[str | None] = []
        parity_values: set[str] = set()
        spin_values: list[float] = []
        for source_name in source_term_refs.values():
            ancestry_metadata = _infer_source_ancestry_metadata(source_name)
            source_sector_names.update(ancestry_metadata["sectors"])
            tensor_character_values.append(
                ancestry_metadata["tensor_character"]
            )
            source_parity = ancestry_metadata["parity"]
            if source_parity is not None:
                parity_values.add(str(source_parity))
            source_spin = ancestry_metadata["spin"]
            if source_spin is not None:
                spin_values.append(abs(float(source_spin)))
        if len(source_sector_names) > 1:
            mixed = ", ".join(sorted(source_sector_names))
            raise ValueError(
                f"Perturbation observable '{observable_name}' mixes "
                f"declared source sectors: {mixed}"
            )
        sector = (
            next(iter(source_sector_names)) if source_sector_names else None
        )
        typing_entry = _match_projection_typing_entry(
            observable_name=observable_name,
            observable_kind=observable_kind,
            kernel=kernel,
            source_roles=source_term_refs,
        )
        if (
            typing_entry is not None
            and typing_entry.sector is not None
            and sector is not None
            and typing_entry.sector != sector
        ):
            raise ValueError(
                f"Perturbation observable '{observable_name}' "
                "projection_typing sector does not match declared source "
                f"ancestry: {typing_entry.sector} vs {sector}"
            )
        output_role = _projection_output_role(projection)
        if output_role == "potential" and sector not in {None, "scalar"}:
            raise ValueError(
                f"Perturbation observable '{observable_name}' lensing or "
                "potential projections require scalar source ancestry"
            )
        tensor_character = _merge_tensor_character(
            tuple(tensor_character_values)
        )
        if tensor_character is None and sector is not None:
            tensor_character = f"{sector}_like"
        parity = None
        if output_role == "polarization_b":
            parity = "odd"
        elif typing_entry is not None and typing_entry.parity is not None:
            parity = str(typing_entry.parity)
        elif len(parity_values) == 1:
            parity = next(iter(parity_values))
        elif output_role in {"polarization_e", "temperature", "potential"}:
            parity = "even"
        spin = None
        if typing_entry is not None and typing_entry.spin is not None:
            spin = float(typing_entry.spin)
        elif output_role in {"polarization_e", "polarization_b"}:
            spin = 2.0
        elif output_role in {"temperature", "potential"}:
            spin = 0.0
        elif spin_values:
            spin = max(spin_values)
        if (
            typing_entry is not None
            and typing_entry.sector is not None
            and sector is None
        ):
            sector = str(typing_entry.sector)
        return output_role, sector, parity, spin, tensor_character

    observable_entries: dict[str, PerturbationObservableData] = {}
    observable_names: set[str] = set()
    transfer_component_names: set[str] = set()
    for observable_name, observable_def in sections["observables"].items():
        name = _validate_string(
            observable_name,
            label="Observable name",
        )
        if not isinstance(observable_def, Mapping):
            raise ValueError(
                f"Perturbation observable '{name}' must be a mapping"
            )
        _validate_entry_keys(
            entry=observable_def,
            allowed_keys=_SUPPORTED_OBSERVABLE_KEYS,
            label=f"cmb.perturbations.observables.{name}",
        )
        observable_kind = _validate_string(
            observable_def.get("kind"),
            label=f"cmb.perturbations.observables.{name}.kind",
        )
        if observable_kind not in _SUPPORTED_OBSERVABLE_KINDS:
            raise ValueError(
                f"Perturbation observable '{name}' uses unsupported kind "
                f"'{observable_kind}'"
            )
        projection = _validate_optional_string(
            observable_def.get("projection"),
            label=f"cmb.perturbations.observables.{name}.projection",
        )
        primary = _validate_optional_string(
            observable_def.get("primary"),
            label=f"cmb.perturbations.observables.{name}.primary",
        )
        secondary = _validate_optional_string(
            observable_def.get("secondary"),
            label=f"cmb.perturbations.observables.{name}.secondary",
        )
        source_term_mapping = observable_def.get("source_terms", {})
        if source_term_mapping is None:
            source_term_mapping = {}
        if not isinstance(source_term_mapping, Mapping):
            raise ValueError(
                f"Perturbation observable '{name}' source_terms must be a "
                "mapping"
            )
        source_term_refs: dict[str, str] = {}
        for role_name, source_name in source_term_mapping.items():
            role = _validate_string(
                role_name,
                label=f"cmb.perturbations.observables.{name}.source_terms key",
            )
            source_ref = _validate_string(
                source_name,
                label=(
                    f"cmb.perturbations.observables.{name}.source_terms."
                    f"{role}"
                ),
            )
            if source_ref not in source_entries:
                raise ValueError(
                    f"Perturbation observable '{name}' references unknown "
                    f"source term '{source_ref}'"
                )
            source_term_refs[role] = source_ref
        kernel = _validate_optional_string(
            observable_def.get("kernel"),
            label=f"cmb.perturbations.observables.{name}.kernel",
        )
        required_projection_roles = _validate_optional_string_list(
            observable_def.get("required_projection_roles"),
            label=(
                f"cmb.perturbations.observables.{name}"
                ".required_projection_roles"
            ),
        )
        effective_projection_roles = required_projection_roles
        if observable_kind == "transfer_component":
            if projection is None:
                raise ValueError(
                    f"Perturbation observable '{name}' must declare "
                    "projection"
                )
            declared_projection = str(projection)
            if primary is not None or secondary is not None:
                raise ValueError(
                    f"Perturbation observable '{name}' kind "
                    "'transfer_component' must not declare primary or "
                    "secondary"
                )
            declared_source_roles = {
                role_name: source_entries[source_name].role
                for role_name, source_name in source_term_refs.items()
            }
            validate_declared_projection_source_roles(
                declared_projection,
                observable_name=name,
                source_roles=declared_source_roles,
                extensions=projection_extension_entries,
            )
            for role_name, source_role in declared_source_roles.items():
                if source_role is None:
                    continue
                if source_role != role_name:
                    source_name = source_term_refs[role_name]
                    raise ValueError(
                        f"Perturbation observable '{name}' binds source term "
                        f"role '{role_name}' to source '{source_name}' "
                        f"with declared role '{source_role}'"
                    )
            projection_spec = get_declared_projection_spec(
                declared_projection,
                extensions=projection_extension_entries,
            )
            kernel = resolve_declared_projection_kernel(
                declared_projection,
                observable_name=name,
                kernel=kernel,
                extensions=projection_extension_entries,
            )
            runtime_projection = (
                str(
                    projection_extension_entries[
                        declared_projection
                    ].base_projection
                )
                if declared_projection in projection_extension_entries
                else declared_projection
            )
            effective_projection_roles = _dedupe_names(
                projection_spec.required_projection_roles
                + required_projection_roles
            )
            if projection_spec.requires_odd_parity_source and (
                "b_mode" not in effective_projection_roles
            ):
                effective_projection_roles = effective_projection_roles + (
                    "b_mode",
                )
            if effective_projection_roles:
                for source_name in source_term_refs.values():
                    if _source_ancestry_supports_projection_roles(
                        source_name,
                        effective_projection_roles,
                    ):
                        continue
                    if (
                        projection_spec.requires_odd_parity_source
                        and effective_projection_roles == ("b_mode",)
                    ):
                        raise ValueError(
                            f"Perturbation observable '{name}' projection "
                            f"'{declared_projection}' requires an odd-parity "
                            "declared source ancestry"
                        )
                    raise ValueError(
                        f"Perturbation observable '{name}' projection "
                        f"'{declared_projection}' requires source "
                        f"'{source_name}' "
                        "to provide declared projection roles: "
                        + ", ".join(effective_projection_roles)
                    )
            transfer_component_names.add(name)
            (
                output_role,
                sector,
                parity,
                spin,
                tensor_character,
            ) = _resolve_transfer_component_metadata(
                observable_name=name,
                projection=runtime_projection,
                observable_kind=observable_kind,
                kernel=kernel,
                source_term_refs=source_term_refs,
            )
            projection = runtime_projection
        else:
            output_role = None
            sector = None
            parity = None
            spin = None
            tensor_character = None
            if primary is None or secondary is None:
                raise ValueError(
                    f"Perturbation observable '{name}' must declare "
                    "primary and secondary"
                )
            if projection is not None or source_term_refs:
                raise ValueError(
                    f"Perturbation observable '{name}' kind "
                    "'angular_power_spectrum' must not declare projection "
                    "or source_terms"
                )
            if kernel is not None or required_projection_roles:
                raise ValueError(
                    f"Perturbation observable '{name}' kind "
                    "'angular_power_spectrum' must not declare kernel or "
                    "required_projection_roles"
                )
        dependencies = _dedupe_names(
            tuple(source_term_refs.values())
            + (() if primary is None else (primary,))
            + (() if secondary is None else (secondary,))
        )
        observable_entries[name] = PerturbationObservableData(
            name=name,
            kind=observable_kind,
            projection=projection,
            kernel=kernel,
            primary=primary,
            secondary=secondary,
            source_terms=FrozenMapping(source_term_refs),
            required_projection_roles=effective_projection_roles,
            description=_validate_optional_string(
                observable_def.get("description"),
                label=f"cmb.perturbations.observables.{name}.description",
            ),
            notes=_validate_optional_string(
                observable_def.get("notes"),
                label=f"cmb.perturbations.observables.{name}.notes",
            ),
            domain=_validate_optional_string(
                observable_def.get("domain"),
                label=f"cmb.perturbations.observables.{name}.domain",
            ),
            dependencies=dependencies,
            output_role=output_role,
            sector=sector,
            parity=parity,
            spin=spin,
            tensor_character=tensor_character,
        )
        observable_names.add(name)
    for observable_name, observable_entry in observable_entries.items():
        if observable_entry.kind != "angular_power_spectrum":
            continue
        if observable_entry.primary not in transfer_component_names:
            raise ValueError(
                f"Perturbation observable '{observable_name}' references "
                f"unknown transfer component '{observable_entry.primary}'"
            )
        if observable_entry.secondary not in transfer_component_names:
            raise ValueError(
                f"Perturbation observable '{observable_name}' references "
                f"unknown transfer component '{observable_entry.secondary}'"
            )
        primary_entry = observable_entries[str(observable_entry.primary)]
        secondary_entry = observable_entries[str(observable_entry.secondary)]
        primary_sector = primary_entry.sector
        secondary_sector = secondary_entry.sector
        if (
            primary_sector is not None
            and secondary_sector is not None
            and primary_sector != secondary_sector
        ):
            raise ValueError(
                f"Perturbation observable '{observable_name}' mixes "
                "transfer components from incompatible sectors: "
                f"{primary_sector} vs {secondary_sector}"
            )
        primary_role = primary_entry.output_role
        secondary_role = secondary_entry.output_role
        role_names = {primary_role, secondary_role}
        if "potential" in role_names:
            other_roles = role_names - {"potential"}
            if other_roles and not other_roles.issubset(
                {"polarization_e", "signal", "temperature"}
            ):
                raise ValueError(
                    f"Perturbation observable '{observable_name}' uses an "
                    "unsupported lensing-potential cross spectrum"
                )
            if primary_sector not in {
                None,
                "scalar",
            } or secondary_sector not in {None, "scalar"}:
                raise ValueError(
                    f"Perturbation observable '{observable_name}' "
                    "lensing-potential spectra require scalar transfer "
                    "components"
                )
        if "polarization_b" in role_names and role_names != {"polarization_b"}:
            raise ValueError(
                f"Perturbation observable '{observable_name}' uses an "
                "unsupported odd-parity B-mode cross spectrum"
            )
        if role_names == {"potential"}:
            output_role = "potential_power"
        elif "potential" in role_names:
            output_role = "temperature_potential_cross"
        elif role_names.issubset(
            {"polarization_b", "polarization_e", "temperature"}
        ):
            output_role = "temperature_power"
        else:
            output_role = "signal_power"
        observable_entries[observable_name] = replace(
            observable_entry,
            output_role=output_role,
            sector=primary_sector or secondary_sector,
            parity=(
                primary_entry.parity
                if primary_entry.parity == secondary_entry.parity
                else None
            ),
            spin=max(
                [
                    abs(float(spin_value))
                    for spin_value in (
                        primary_entry.spin,
                        secondary_entry.spin,
                    )
                    if spin_value is not None
                ],
                default=None,
            ),
            tensor_character=_merge_tensor_character(
                (
                    primary_entry.tensor_character,
                    secondary_entry.tensor_character,
                )
            ),
        )

    def _compile_conditions(
        condition_defs: Mapping[str, Any],
        *,
        label_prefix: str,
        default_anchor: str,
    ) -> dict[str, PerturbationConditionData]:
        """Compile initial or boundary-condition mappings into typed data."""

        compiled: dict[str, PerturbationConditionData] = {}
        seen_targets: set[tuple[str, str, int]] = set()
        for condition_name, condition_def in condition_defs.items():
            name = _validate_string(
                condition_name,
                label="Condition name",
            )
            if not isinstance(condition_def, Mapping):
                raise ValueError(
                    f"Perturbation condition '{name}' must be a mapping"
                )
            _validate_entry_keys(
                entry=condition_def,
                allowed_keys=_SUPPORTED_CONDITION_KEYS,
                label=f"{label_prefix}.{name}",
            )
            target = condition_def.get("target")
            if not isinstance(target, Mapping):
                raise ValueError(
                    f"{label_prefix}.{name}.target must be a mapping"
                )
            _validate_entry_keys(
                entry=target,
                allowed_keys=_SUPPORTED_CONDITION_TARGET_KEYS,
                label=f"{label_prefix}.{name}.target",
            )
            variable_name = _validate_string(
                target.get("variable"),
                label=f"{label_prefix}.{name}.target.variable",
            )
            if variable_name not in variable_entries:
                raise ValueError(
                    f"Perturbation condition '{name}' references unknown "
                    f"variable '{variable_name}'"
                )
            wrt_name = _validate_string(
                target.get("wrt"),
                label=f"{label_prefix}.{name}.target.wrt",
            )
            if wrt_name not in _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES:
                raise ValueError(
                    f"Perturbation condition '{name}' uses unsupported wrt "
                    f"'{wrt_name}'"
                )
            order_value = _validate_optional_int(
                target.get("order"),
                label=f"{label_prefix}.{name}.target.order",
            )
            if order_value is None or order_value < 0:
                raise ValueError(
                    f"Perturbation condition '{name}' order must be a "
                    "non-negative integer"
                )
            target_key = (variable_name, wrt_name, order_value)
            if target_key in seen_targets:
                raise ValueError(
                    f"Perturbation condition '{name}' duplicates target "
                    f"{target_key}"
                )
            expression_text, dependencies = _replace_and_validate_expression(
                condition_def.get("expression"),
                label=f"{label_prefix}.{name}.expression",
                replacements=replacements,
                allowed_names=all_expression_names
                | set(
                    _relation_target_nodes(constraint_entries, closure_entries)
                )
                | {entry_name for entry_name in variable_entries},
            )
            anchor_name = str(
                _validate_optional_string(
                    condition_def.get("anchor"),
                    label=f"{label_prefix}.{name}.anchor",
                )
                or default_anchor
            )
            if anchor_name not in _SUPPORTED_CONDITION_ANCHORS:
                raise ValueError(
                    f"Perturbation condition '{name}' uses unsupported "
                    f"anchor '{anchor_name}'"
                )
            compiled[name] = PerturbationConditionData(
                name=name,
                target=PerturbationConditionTargetData(
                    variable=variable_name,
                    wrt=wrt_name,
                    order=order_value,
                ),
                expression=expression_text,
                anchor=anchor_name,
                description=_validate_optional_string(
                    condition_def.get("description"),
                    label=f"{label_prefix}.{name}.description",
                ),
                notes=_validate_optional_string(
                    condition_def.get("notes"),
                    label=f"{label_prefix}.{name}.notes",
                ),
                domain=_validate_optional_string(
                    condition_def.get("domain"),
                    label=f"{label_prefix}.{name}.domain",
                ),
                dependencies=dependencies,
                compiled_expression=_compile_expression_plan(
                    expression_text,
                    dependencies=dependencies,
                ),
            )
            seen_targets.add(target_key)
        return compiled

    GeneratedInitialConditions = tuple[
        dict[str, Any],
        dict[str, tuple[str, ...]],
    ]

    def _generated_initial_conditions_from_families() -> (
        GeneratedInitialConditions
    ):
        """Return auto-generated initial conditions for standard modes."""

        family_defs = sections["initial_condition_families"]
        initial_mode = _select_standard_initial_mode(family_defs)
        if initial_mode is None:
            return {}, {}
        family_entry = family_defs.get(initial_mode)
        if not isinstance(family_entry, Mapping):
            return {}, {}
        if family_entry.get("members"):
            return {}, {}
        if (
            sum(
                1
                for family_name in family_defs
                if family_name in _SCALAR_HIERARCHY_STANDARD_INITIAL_MODES
            )
            > 1
        ):
            raise ValueError(
                "Declare at most one auto-generated initial-condition "
                "family per perturbation contract."
            )

        required_targets = {
            (
                entry.lhs.variable,
                entry.lhs.wrt,
                derivative_order,
            )
            for entry in equation_entries.values()
            for derivative_order in range(entry.lhs.order)
        }
        covered_targets = {
            (
                condition_def["target"]["variable"],
                condition_def["target"]["wrt"],
                int(condition_def["target"]["order"]),
            )
            for condition_def in sections["initial_conditions"].values()
            if isinstance(condition_def, Mapping)
            and isinstance(condition_def.get("target"), Mapping)
        } | {
            (
                condition_def["target"]["variable"],
                condition_def["target"]["wrt"],
                int(condition_def["target"]["order"]),
            )
            for condition_def in sections["boundary_conditions"].values()
            if isinstance(condition_def, Mapping)
            and isinstance(condition_def.get("target"), Mapping)
        }
        missing_targets = sorted(required_targets - covered_targets)
        if not missing_targets:
            return {}, {}

        generated: dict[str, Any] = {}
        generated_names: list[str] = []
        for variable_name, wrt_name, order_value in missing_targets:
            variable_entry = variable_entries[variable_name]
            condition_name = (
                f"{initial_mode}_{variable_name}_{wrt_name}_{order_value}_seed"
            )
            generated[condition_name] = {
                "target": {
                    "variable": variable_name,
                    "wrt": wrt_name,
                    "order": order_value,
                },
                "expression": _auto_initial_condition_expression(
                    variable_entry=variable_entry,
                    target_order=order_value,
                    mode=initial_mode,
                ),
                "description": (
                    "Auto-generated native initial condition for "
                    f"{initial_mode}."
                ),
            }
            generated_names.append(condition_name)
        return generated, {initial_mode: tuple(generated_names)}

    generated_initial_conditions, generated_condition_members = (
        _generated_initial_conditions_from_families()
    )
    raw_initial_conditions = dict(sections["initial_conditions"])
    raw_initial_conditions.update(generated_initial_conditions)
    initial_condition_entries = _compile_conditions(
        raw_initial_conditions,
        label_prefix="cmb.perturbations.initial_conditions",
        default_anchor="start",
    )
    boundary_condition_entries = _compile_conditions(
        sections["boundary_conditions"],
        label_prefix="cmb.perturbations.boundary_conditions",
        default_anchor="start",
    )

    sector_entries = _compile_sector_metadata(sections["sectors"])
    hierarchy_family_entries = _compile_hierarchy_family_metadata(
        sections["hierarchy_families"],
        sector_names=set(sector_entries),
    )
    species_entries = _compile_species_metadata(
        sections["species"],
        sector_names=set(sector_entries),
        hierarchy_family_names=set(hierarchy_family_entries),
    )
    collision_operator_entries = _compile_collision_operator_metadata(
        sections["collision_operators"],
        sector_names=set(sector_entries),
        species_names=set(species_entries),
        allowed_names=(
            all_expression_names
            | set(source_entries)
            | set(observable_entries)
            | set(sections["interactions"])
        ),
        replacements=replacements,
    )
    interaction_entries = _compile_interaction_metadata(
        sections["interactions"],
        sector_names=set(sector_entries),
        species_names=set(species_entries),
        allowed_names=(
            all_expression_names
            | set(sections["collision_operators"])
            | set(sections["interactions"])
            | set(_relation_target_nodes(constraint_entries, closure_entries))
        ),
        replacements=replacements,
    )
    conflicting_dynamic_names = sorted(
        (set(collision_operator_entries) | set(interaction_entries))
        & (
            set(variable_entries)
            | set(derived_entries)
            | set(
                _relation_target_nodes(
                    constraint_entries,
                    closure_entries,
                )
            )
            | set(source_entries)
        )
    )
    if conflicting_dynamic_names:
        readable = ", ".join(conflicting_dynamic_names)
        raise ValueError(
            "Declared interaction or collision names collide with graph "
            f"symbols: {readable}"
        )
    duplicate_dynamic_names = sorted(
        set(collision_operator_entries) & set(interaction_entries)
    )
    if duplicate_dynamic_names:
        readable = ", ".join(duplicate_dynamic_names)
        raise ValueError(
            "Declared interactions duplicate collision operators: "
            f"{readable}"
        )
    conservation_rule_entries = _compile_conservation_rule_metadata(
        sections["conservation_rules"],
        allowed_names=(
            all_expression_names
            | set(_relation_target_nodes(constraint_entries, closure_entries))
            | set(source_entries)
            | set(collision_operator_entries)
            | set(interaction_entries)
        ),
        replacements=replacements,
    )
    initial_condition_family_entries = (
        _compile_initial_condition_family_metadata(
            {
                str(name): {
                    **dict(entry),
                    "members": list(
                        dict.fromkeys(
                            list(entry.get("members", []) or [])
                            + list(
                                generated_condition_members.get(
                                    str(name),
                                    (),
                                )
                            )
                        )
                    ),
                }
                for name, entry in sections[
                    "initial_condition_families"
                ].items()
            },
            sector_names=set(sector_entries),
            initial_condition_names=set(initial_condition_entries),
        )
    )
    projection_typing_entries = _compile_projection_typing_metadata(
        sections["projection_typing"],
        sector_names=set(sector_entries),
    )
    accuracy_controls_mapping = FrozenMapping(
        {
            str(key): _normalize_accuracy_control_value(
                value,
                label=f"cmb.perturbations.accuracy_controls.{key}",
            )
            for key, value in sections["accuracy_controls"].items()
        }
    )
    momentum_grid_defs = sections["numerics"].get("momentum_grids", {})
    if momentum_grid_defs in (None, {}):
        momentum_grid_defs = {}
    if not isinstance(momentum_grid_defs, Mapping):
        raise ValueError(
            "cmb.perturbations.numerics.momentum_grids must be a mapping"
        )
    for family_name, family_entry in hierarchy_family_entries.items():
        if family_entry.momentum_grid is None:
            continue
        if family_entry.momentum_grid not in momentum_grid_defs:
            raise ValueError(
                "cmb.perturbations.hierarchy_families."
                f"{family_name}.momentum_grid references unknown "
                f"numerics.momentum_grids entry "
                f"'{family_entry.momentum_grid}'"
            )
    if gauge not in {"gauge_invariant", "unspecified"} and any(
        gauge in (entry.supported_gauges or ())
        for entry in sector_entries.values()
    ):
        disallowed_roles = (
            _SYNCHRONOUS_GAUGE_ROLES
            if gauge == "conformal_newtonian"
            else _NEWTONIAN_GAUGE_ROLES
        )
        conflicting_variables = sorted(
            name
            for name, entry in variable_entries.items()
            if entry.gauge_role in disallowed_roles
        )
        if conflicting_variables:
            readable = ", ".join(conflicting_variables)
            raise ValueError(
                "Perturbation variables declare gauge roles that conflict "
                f"with gauge '{gauge}': {readable}"
            )
        incompatible_sectors = sorted(
            sector_name
            for sector_name, sector_entry in sector_entries.items()
            if sector_entry.supported_gauges
            and gauge not in sector_entry.supported_gauges
        )
        if incompatible_sectors:
            readable = ", ".join(incompatible_sectors)
            raise ValueError(
                "Perturbation sectors do not support gauge "
                f"'{gauge}': {readable}"
            )
    for sector_name, sector_entry in sector_entries.items():
        unknown_families = sorted(
            set(sector_entry.hierarchy_families)
            - set(hierarchy_family_entries)
        )
        if unknown_families:
            unknown_str = ", ".join(unknown_families)
            raise ValueError(
                "cmb.perturbations.sectors."
                f"{sector_name}.hierarchy_families references unknown "
                f"families: {unknown_str}"
            )
        unknown_species = sorted(
            set(sector_entry.species) - set(species_entries)
        )
        if unknown_species:
            unknown_str = ", ".join(unknown_species)
            raise ValueError(
                "cmb.perturbations.sectors."
                f"{sector_name}.species references unknown species: "
                f"{unknown_str}"
            )
    for family_name, family_entry in hierarchy_family_entries.items():
        unknown_species = sorted(
            set(family_entry.species) - set(species_entries)
        )
        if unknown_species:
            unknown_str = ", ".join(unknown_species)
            raise ValueError(
                "cmb.perturbations.hierarchy_families."
                f"{family_name}.species references unknown species: "
                f"{unknown_str}"
            )
    for species_name, species_entry in species_entries.items():
        unknown_operators = sorted(
            set(species_entry.collision_operators)
            - set(collision_operator_entries)
        )
        if unknown_operators:
            unknown_str = ", ".join(unknown_operators)
            raise ValueError(
                "cmb.perturbations.species."
                f"{species_name}.collision_operators references unknown "
                f"operators: {unknown_str}"
            )

    if standard:
        for section_name in (
            "variables",
            "derived",
            "equations",
            "constraints",
            "closures",
            "conservation_rules",
            "collision_operators",
            "interactions",
            "sources",
            "observables",
            "initial_conditions",
            "initial_condition_families",
            "boundary_conditions",
            "sectors",
            "species",
            "hierarchy_families",
            "projection_extensions",
            "projection_typing",
            "accuracy_controls",
        ):
            if sections[section_name]:
                raise ValueError(
                    f"Standard perturbations require {section_name}: {{}}"
                )
    else:
        if not variable_entries:
            raise ValueError(
                "Non-standard perturbations must declare variables"
            )
        if not equation_entries:
            raise ValueError(
                "Non-standard perturbations must declare equations"
            )
        if not initial_condition_entries and not boundary_condition_entries:
            raise ValueError(
                "Non-standard perturbations must declare initial_conditions "
                "or boundary_conditions"
            )
        if not observable_entries:
            raise ValueError(
                "Non-standard perturbations must declare observables"
            )
        if not sections["validity"]:
            raise ValueError(
                "Non-standard perturbations must declare validity"
            )

    validity_notes = _validate_optional_string(
        sections["validity"].get("notes"),
        label="cmb.perturbations.validity.notes",
    )
    validity_regimes = sections["validity"].get("regimes")
    regimes = ()
    if validity_regimes is not None:
        regimes = _validate_regimes(validity_regimes)
    elif not standard:
        raise ValueError(
            "Non-standard perturbations must declare validity.regimes"
        )
    validity_data = PerturbationValidityData(
        regimes=regimes,
        notes=validity_notes,
    )

    numerics_mapping = FrozenMapping(
        {str(key): value for key, value in sections["numerics"].items()}
    )

    derivative_symbol_orders: dict[tuple[str, str], int] = {}
    for entry in derived_entries.values():
        if entry.expression is not None:
            continue
        key = (str(entry.variable), str(entry.wrt))
        derivative_symbol_orders[key] = max(
            derivative_symbol_orders.get(key, 0),
            int(entry.order or 0),
        )
    equation_orders: dict[tuple[str, str], int] = {}
    for entry in equation_entries.values():
        key = (entry.lhs.variable, entry.lhs.wrt)
        equation_orders[key] = max(
            equation_orders.get(key, 0),
            entry.lhs.order,
        )
    evolved_variable_names = {
        entry.lhs.variable for entry in equation_entries.values()
    }
    relation_target_names = set(
        _relation_target_nodes(constraint_entries, closure_entries)
    )
    for key, required_order in derivative_symbol_orders.items():
        if key not in equation_orders:
            variable_name, wrt_name = key
            if variable_name in relation_target_names:
                continue
            raise ValueError(
                "Derivative symbol requires an evolved variable: "
                f"{variable_name} wrt {wrt_name}"
            )
        if required_order >= equation_orders[key]:
            variable_name, wrt_name = key
            raise ValueError(
                "Derivative symbol order exceeds the declared differential "
                f"state for variable '{variable_name}' wrt '{wrt_name}'"
            )

    required_initial_targets = {
        (
            equation_entry.lhs.variable,
            equation_entry.lhs.wrt,
            derivative_order,
        )
        for equation_entry in equation_entries.values()
        for derivative_order in range(equation_entry.lhs.order)
    }
    declared_condition_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in initial_condition_entries.values()
    } | {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in boundary_condition_entries.values()
    }
    unsupported_condition_targets = sorted(
        declared_condition_targets - required_initial_targets
    )
    if unsupported_condition_targets:
        readable = ", ".join(
            f"{variable} wrt {wrt} order {order}"
            for variable, wrt, order in unsupported_condition_targets
        )
        raise ValueError(
            "Perturbation conditions may only target declared differential "
            f"state slots: {readable}"
        )
    declared_initial_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in initial_condition_entries.values()
    }
    declared_start_boundary_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in boundary_condition_entries.values()
        if condition_entry.anchor == "start"
    }
    declared_boundary_targets = {
        (
            condition_entry.target.variable,
            condition_entry.target.wrt,
            condition_entry.target.order,
        )
        for condition_entry in boundary_condition_entries.values()
    }
    duplicate_start_targets = sorted(
        declared_initial_targets & declared_start_boundary_targets
    )
    if duplicate_start_targets:
        readable = ", ".join(
            f"{variable} wrt {wrt} order {order}"
            for variable, wrt, order in duplicate_start_targets
        )
        raise ValueError(
            "Initial conditions and start-anchored boundary conditions "
            f"duplicate targets: {readable}"
        )
    missing_initial_targets = sorted(
        required_initial_targets
        - declared_initial_targets
        - declared_boundary_targets
    )
    if missing_initial_targets:
        readable = ", ".join(
            f"{variable} wrt {wrt} order {order}"
            for variable, wrt, order in missing_initial_targets
        )
        raise ValueError(
            "Non-standard perturbations are missing required initial "
            f"conditions: {readable}"
        )

    solved_variable_names = evolved_variable_names | relation_target_names
    referenced_unsolved_variables = sorted(
        {
            dependency
            for entry in (
                list(derived_entries.values())
                + list(equation_entries.values())
                + list(constraint_entries.values())
                + list(closure_entries.values())
                + list(interaction_entries.values())
                + list(source_entries.values())
                + list(initial_condition_entries.values())
                + list(boundary_condition_entries.values())
            )
            for dependency in entry.dependencies
            if (
                dependency in variable_entries
                and dependency not in solved_variable_names
            )
        }
    )
    if referenced_unsolved_variables:
        readable = ", ".join(referenced_unsolved_variables)
        raise ValueError(
            "Declared graph references variable(s) without evolution or "
            f"algebraic definitions: {readable}"
        )

    evaluation_order = _topological_evaluation_order(
        derived=derived_entries,
        constraints=constraint_entries,
        closures=closure_entries,
        interactions=interaction_entries,
        collision_operators=collision_operator_entries,
    )
    dependency_summary = PerturbationDependencyGraphSummaryData(
        variable_names=tuple(sorted(variable_entries)),
        derived_names=tuple(sorted(derived_entries)),
        equation_names=tuple(sorted(equation_entries)),
        constraint_names=tuple(sorted(constraint_entries)),
        closure_names=tuple(sorted(closure_entries)),
        interaction_names=tuple(sorted(interaction_entries)),
        conservation_rule_names=tuple(sorted(conservation_rule_entries)),
        source_names=tuple(sorted(source_entries)),
        observable_names=tuple(sorted(observable_entries)),
        initial_condition_names=tuple(sorted(initial_condition_entries)),
        boundary_condition_names=tuple(sorted(boundary_condition_entries)),
        independent_variables_used=tuple(
            sorted(
                set().union(
                    *(
                        set(entry.dependencies)
                        for entry in (
                            list(derived_entries.values())
                            + list(equation_entries.values())
                            + list(constraint_entries.values())
                            + list(closure_entries.values())
                            + list(interaction_entries.values())
                            + list(conservation_rule_entries.values())
                            + list(source_entries.values())
                            + list(observable_entries.values())
                            + list(initial_condition_entries.values())
                            + list(boundary_condition_entries.values())
                        )
                    )
                )
                & set(_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES)
            )
        ),
        model_parameters_used=tuple(
            sorted(
                set().union(
                    *(
                        set(entry.dependencies)
                        for entry in (
                            list(derived_entries.values())
                            + list(equation_entries.values())
                            + list(constraint_entries.values())
                            + list(closure_entries.values())
                            + list(interaction_entries.values())
                            + list(conservation_rule_entries.values())
                            + list(source_entries.values())
                            + list(initial_condition_entries.values())
                            + list(boundary_condition_entries.values())
                        )
                    )
                )
                & parameter_name_set
            )
        ),
        background_references_used=tuple(
            sorted(
                set().union(
                    *(
                        set(entry.dependencies)
                        for entry in (
                            list(derived_entries.values())
                            + list(equation_entries.values())
                            + list(constraint_entries.values())
                            + list(closure_entries.values())
                            + list(interaction_entries.values())
                            + list(conservation_rule_entries.values())
                            + list(source_entries.values())
                            + list(initial_condition_entries.values())
                            + list(boundary_condition_entries.values())
                        )
                    )
                )
                & background_reference_set
            )
        ),
        derived_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in derived_entries.items()
            }
        ),
        equation_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in equation_entries.items()
            }
        ),
        constraint_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in constraint_entries.items()
            }
        ),
        closure_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in closure_entries.items()
            }
        ),
        interaction_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in interaction_entries.items()
            }
        ),
        conservation_rule_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in conservation_rule_entries.items()
            }
        ),
        source_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in source_entries.items()
            }
        ),
        observable_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in observable_entries.items()
            }
        ),
        initial_condition_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in initial_condition_entries.items()
            }
        ),
        boundary_condition_dependencies=FrozenMapping(
            {
                name: entry.dependencies
                for name, entry in boundary_condition_entries.items()
            }
        ),
        evaluation_order=evaluation_order,
    )

    backend_data = PerturbationBackendMappingData(
        backend=backend,
        uses_standard_perturbations=backend_contract.get(
            "uses_standard_perturbations"
        ),
        native_solver_required=backend_contract.get("native_solver_required"),
        implemented=backend_contract.get("implemented"),
    )
    transfer_component_contracts = {
        name: {
            "declared_projection": str(
                (
                    (sections["observables"].get(name, {}) or {}).get(
                        "projection"
                    )
                    or entry.projection
                    or ""
                )
            ),
            "projection": str(entry.projection or ""),
            "kernel": (None if entry.kernel is None else str(entry.kernel)),
            "source_term_roles": tuple(
                str(role) for role in entry.source_terms
            ),
            "source_term_names": {
                str(role): str(source_name)
                for role, source_name in entry.source_terms.items()
            },
            "required_projection_roles": tuple(
                str(role) for role in entry.required_projection_roles
            ),
        }
        for name, entry in observable_entries.items()
        if entry.kind == "transfer_component"
    }
    angular_power_spectrum_targets = {
        name: {
            "primary": str(entry.primary or ""),
            "secondary": str(entry.secondary or ""),
        }
        for name, entry in observable_entries.items()
        if entry.kind == "angular_power_spectrum"
    }
    manifest_summary = FrozenMapping(
        _build_manifest_summary(
            model_name=model_name,
            backend=backend,
            contract_version=contract_version,
            standard=standard,
            gauge=gauge,
            variables=dependency_summary.variable_names,
            derived=dependency_summary.derived_names,
            equations=dependency_summary.equation_names,
            constraints=dependency_summary.constraint_names,
            closures=dependency_summary.closure_names,
            interactions=dependency_summary.interaction_names,
            conservation_rules=(dependency_summary.conservation_rule_names),
            sources=dependency_summary.source_names,
            observables=dependency_summary.observable_names,
            initial_conditions=dependency_summary.initial_condition_names,
            boundary_conditions=(dependency_summary.boundary_condition_names),
            sectors=tuple(sorted(sector_entries)),
            species=tuple(sorted(species_entries)),
            hierarchy_families=tuple(sorted(hierarchy_family_entries)),
            collision_operators=tuple(sorted(collision_operator_entries)),
            initial_condition_families=tuple(
                sorted(initial_condition_family_entries)
            ),
            projection_extensions=tuple(sorted(projection_extension_entries)),
            projection_typing=tuple(sorted(projection_typing_entries)),
            validity=validity_data,
            numerics=numerics_mapping,
            accuracy_controls=accuracy_controls_mapping,
            backend_mapping=backend_data,
            dependency_summary=dependency_summary,
            generated_scalar_hierarchy=materialized_scalar_hierarchy,
            equation_wrt_by_variable={
                entry.lhs.variable: entry.lhs.wrt
                for entry in equation_entries.values()
            },
            boundary_condition_anchors={
                name: entry.anchor
                for name, entry in boundary_condition_entries.items()
            },
            transfer_component_contracts=transfer_component_contracts,
            angular_power_spectrum_targets=angular_power_spectrum_targets,
        )
    )

    compiled = PerturbationContractData(
        model_name=model_name,
        backend=backend,
        contract_version=contract_version,
        standard=standard,
        gauge=gauge,
        variables=FrozenMapping(variable_entries),
        derived=FrozenMapping(derived_entries),
        equations=FrozenMapping(equation_entries),
        constraints=FrozenMapping(constraint_entries),
        closures=FrozenMapping(closure_entries),
        sources=FrozenMapping(source_entries),
        observables=FrozenMapping(observable_entries),
        initial_conditions=FrozenMapping(initial_condition_entries),
        boundary_conditions=FrozenMapping(boundary_condition_entries),
        numerics=numerics_mapping,
        validity=validity_data,
        backend_mapping=FrozenMapping({backend: backend_data}),
        dependency_graph_summary=dependency_summary,
        manifest_summary=manifest_summary,
        sectors=FrozenMapping(sector_entries),
        species=FrozenMapping(species_entries),
        hierarchy_families=FrozenMapping(hierarchy_family_entries),
        collision_operators=FrozenMapping(collision_operator_entries),
        interactions=FrozenMapping(interaction_entries),
        conservation_rules=FrozenMapping(conservation_rule_entries),
        initial_condition_families=FrozenMapping(
            initial_condition_family_entries
        ),
        projection_extensions=FrozenMapping(projection_extension_entries),
        projection_typing=FrozenMapping(projection_typing_entries),
        accuracy_controls=accuracy_controls_mapping,
    )
    _COMPILED_CONTRACT_RESULTS[cache_key] = compiled
    return _get_cached_perturbation_contract(cache_key)


__all__ = [
    "PerturbationBackendMappingData",
    "PerturbationCollisionOperatorData",
    "PerturbationClosureData",
    "PerturbationCompiledExpressionData",
    "PerturbationConservationRuleData",
    "PerturbationConditionData",
    "PerturbationConditionTargetData",
    "PerturbationConstraintData",
    "PerturbationContractData",
    "PerturbationDependencyGraphSummaryData",
    "PerturbationDerivedData",
    "PerturbationDerivativeLhsData",
    "PerturbationEquationData",
    "PerturbationHierarchyFamilyData",
    "PerturbationInitialConditionFamilyData",
    "PerturbationInteractionData",
    "PerturbationObservableData",
    "PerturbationProjectionExtensionData",
    "PerturbationProjectionTypingData",
    "PerturbationSectorData",
    "PerturbationSourceData",
    "PerturbationSpeciesData",
    "PerturbationValidityData",
    "PerturbationVariableData",
    "compile_perturbation_contract",
    "evaluate_compiled_expression",
]
