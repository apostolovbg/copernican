"""Compile declared CMB graph contracts into immutable runtime data.

Each contract describes one declared mathematical graph. The compiler
validates symbols, dependencies, observables, and runtime requirements before
the numerical CMB solver evolves the system.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy

from .cmb_identity import CCMBS_ID, CCMBS_LABEL
from .cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_spec,
    resolve_declared_projection_kernel,
    resolve_declared_source_kernel,
    validate_declared_projection_sector,
    validate_declared_projection_source_roles,
)
from .model_adapter import (
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
_COMPILED_EXPRESSION_GLOBALS = {
    "__builtins__": {},
    **_ALLOWED_CONSTANTS,
    **_ALLOWED_MATH_FUNCS,
}
_RUNTIME_REFERENCE_NAMES = {
    "H0_km_s_Mpc",
    "H0_over_c_Mpc_inv",
    "Hconf_tau",
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
    "a",
    "a_initial",
    "angular_diameter_distance",
    "chi",
    "collision_rate",
    "eta_initial",
    "free_streaming",
    "hubble_ratio",
    "k",
    "massive_neutrino_mass_eV",
    "neutrino_temperature_eV",
    "massive_neutrino_mass_fraction",
    "massive_neutrino_density_fraction",
    "massive_neutrino_momentum_fraction",
    "massive_neutrino_pressure_fraction",
    "massive_neutrino_shear_fraction",
    "massive_neutrino_pressure_ratio",
    "massive_neutrino_streaming_speed",
    "massive_neutrino_velocity_ratio",
    "num_massive_neutrinos",
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
    "baryon_sound_speed_sq",
    "tight_coupling_drag",
    "tight_coupling_ratio",
    "tensor_spectral_index",
    "tensor_to_scalar_ratio",
    "w0",
    "wa",
    "primordial_amplitude",
    "primordial_spectral_index",
}

_SUPPORTED_PERTURBATION_KEYS = {
    "accuracy_controls",
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
    "binding",
    "description",
    "domain",
    "expression",
    "kind",
    "notes",
    "order",
    "units",
    "variable",
    "wrt",
}
_SUPPORTED_DERIVED_BINDINGS = {"runtime_history_gradient"}
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
    "units",
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
    "units",
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
    "activation_strategy",
    "counterpart",
    "dependencies",
    "description",
    "expression",
    "exact_form",
    "integration_strategy",
    "linear_block",
    "notes",
    "rate_expression",
    "sector",
    "species",
}
_SUPPORTED_INTERACTION_KEYS = {
    "counterpart",
    "dependencies",
    "description",
    "expression",
    "notes",
    "sector",
    "species",
}
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
    "photon",
}
_SCALAR_HIERARCHY_REQUIRED_FAMILIES = {
    "photon_polarization_e",
    "photon_temperature",
}
_SCALAR_HIERARCHY_REQUIRED_COLLISION = "thomson_drag"
_VECTOR_HIERARCHY_REQUIRED_SECTOR = "vector"
_VECTOR_HIERARCHY_REQUIRED_SPECIES = {
    "baryon",
    "massless_neutrino",
    "photon",
}
_VECTOR_HIERARCHY_REQUIRED_FAMILIES = {
    "massless_neutrino_vector",
    "photon_polarization_b_vector",
    "photon_polarization_e_vector",
    "photon_temperature_vector",
}
_TENSOR_HIERARCHY_REQUIRED_SECTOR = "tensor"
_TENSOR_HIERARCHY_REQUIRED_SPECIES = {
    "massless_neutrino",
    "photon",
}
_TENSOR_HIERARCHY_REQUIRED_FAMILIES = {
    "massless_neutrino_tensor",
    "photon_polarization_b_tensor",
    "photon_polarization_e_tensor",
    "photon_temperature_tensor",
}
_SCALAR_HIERARCHY_STANDARD_INITIAL_MODES = (
    "adiabatic_scalar",
    "baryon_isocurvature",
    "cdm_isocurvature",
    "neutrino_density_isocurvature",
    "neutrino_velocity_isocurvature",
    "tensor_mode",
)
_VECTOR_HIERARCHY_STANDARD_INITIAL_MODES = ("regular_vector_mode",)
_TENSOR_HIERARCHY_STANDARD_INITIAL_MODES = ("tensor_mode",)
_NEWTONIAN_GAUGE_ROLES = frozenset(
    {"curvature_potential", "newtonian_potential"}
)
_SYNCHRONOUS_GAUGE_ROLES = frozenset(
    {"synchronous_metric_shear", "synchronous_metric_trace"}
)
_DIMENSIONLESS_UNITS = "dimensionless"
_INVERSE_MPC_UNITS = "1/Mpc"
_INVERSE_MPC_SQUARED_UNITS = "1/Mpc^2"
_INVERSE_MPC_CUBED_UNITS = "1/Mpc^3"
_LINE_OF_SIGHT_SOURCE_UNITS = "1/Mpc"


def _has_explicit_declared_runtime_graph(
    contract: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract`` already declares runtime nodes."""

    for section_name in (
        "variables",
        "derived",
        "equations",
        "constraints",
        "closures",
        "observables",
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


def _vector_temperature_name(moment: int) -> str:
    """Return the declared vector photon-temperature variable name."""

    if moment == 1:
        return "q_gamma_vector"
    if moment == 2:
        return "pi_gamma_vector"
    return f"theta_gamma_v{int(moment)}"


def _vector_polarization_e_name(moment: int) -> str:
    """Return the declared vector E-polarization variable name."""

    return f"e_gamma_v{int(moment)}"


def _vector_polarization_b_name(moment: int) -> str:
    """Return the declared vector B-polarization variable name."""

    return f"b_gamma_v{int(moment)}"


def _vector_neutrino_name(moment: int) -> str:
    """Return the declared vector massless-neutrino variable name."""

    if moment == 1:
        return "q_nu_vector"
    if moment == 2:
        return "pi_nu_vector"
    return f"nu_v{int(moment)}"


def _tensor_temperature_name(moment: int) -> str:
    """Return the declared tensor photon-temperature variable name."""

    if moment == 2:
        return "pi_gamma_tensor"
    return f"theta_gamma_t{int(moment)}"


def _tensor_polarization_e_name(moment: int) -> str:
    """Return the declared tensor E-polarization variable name."""

    return f"e_gamma_t{int(moment)}"


def _tensor_polarization_b_name(moment: int) -> str:
    """Return the declared tensor B-polarization variable name."""

    return f"b_gamma_t{int(moment)}"


def _tensor_neutrino_name(moment: int) -> str:
    """Return the declared tensor massless-neutrino variable name."""

    if moment == 2:
        return "pi_nu_tensor"
    return f"nu_t{int(moment)}"


def _metadata_entry(
    kind: str,
    description: str,
    *,
    units: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return one metadata-rich declared entry."""

    entry: dict[str, Any] = {
        "kind": kind,
        "description": description,
    }
    if units is not None:
        entry["units"] = units
    entry.update(extra)
    return entry


def _select_standard_initial_mode(
    family_defs: Mapping[str, Any],
) -> str | None:
    """Return the declared auto-generated initial-condition mode."""

    for family_name in _SCALAR_HIERARCHY_STANDARD_INITIAL_MODES:
        if family_name in family_defs:
            return family_name
    return None


def _standard_initial_mode_names(
    family_defs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return supported standard modes declared by one family mapping."""

    return tuple(
        family_name
        for family_name in _SCALAR_HIERARCHY_STANDARD_INITIAL_MODES
        if family_name in family_defs
    )


def _select_standard_vector_initial_mode(
    family_defs: Mapping[str, Any],
) -> str | None:
    """Return the declared auto-generated vector initial-condition mode."""

    for family_name in _VECTOR_HIERARCHY_STANDARD_INITIAL_MODES:
        if family_name in family_defs:
            return family_name
    return None


def _select_standard_tensor_initial_mode(
    family_defs: Mapping[str, Any],
) -> str | None:
    """Return the declared auto-generated tensor initial-condition mode."""

    for family_name in _TENSOR_HIERARCHY_STANDARD_INITIAL_MODES:
        if family_name in family_defs:
            return family_name
    return None


def _scalar_metric_seed_amplitude(mode: str) -> str:
    """Return the leading super-horizon metric amplitude for ``mode``."""

    by_mode = {
        "adiabatic_scalar": "scalar_potential_seed",
        "baryon_isocurvature": "0.0",
        "cdm_isocurvature": "0.25 * seed",
        "neutrino_density_isocurvature": "0.0",
        "neutrino_velocity_isocurvature": "0.0",
        "tensor_mode": "0.0",
    }
    return str(by_mode.get(mode, "0.0"))


def _scalar_hierarchy_base_seed_expressions(
    mode: str,
    *,
    gauge: str = "conformal_newtonian",
) -> dict[str, str]:
    """Return base seed expressions for one scalar hierarchy mode."""

    k_eta = "acoustic_k * scalar_initial_conformal_time"
    k_eta_sq = f"({k_eta}) * ({k_eta})"
    isocurvature_velocity = (
        "(acoustic_k_sq * scalar_initial_conformal_time / 4.0) * seed"
    )
    compensated_brightness_dipole = f"({k_eta} / 12.0) * seed"
    neutrino_velocity_divergence = "acoustic_k * seed"
    neutrino_velocity_dipole = "seed / 3.0"
    neutrino_velocity_quadrupole = f"({k_eta} / 6.0) * seed"
    neutrino_velocity_density_constraint = (
        "-(acoustic_k_sq * Phi + 3.0 * Hconf * "
        "metric_momentum_constraint) * a * a / "
        "(6.0 * einstein_gravity_strength * Omega_gamma0)"
    )
    adiabatic_seed = "scalar_lapse_seed"
    adiabatic_k_eta = "acoustic_k * scalar_initial_conformal_time"
    adiabatic_k_eta_sq = f"({adiabatic_k_eta}) * ({adiabatic_k_eta})"
    adiabatic_velocity_divergence = (
        "(acoustic_k_sq * scalar_initial_conformal_time / 2.0) * "
        f"{adiabatic_seed}"
    )
    by_mode = {
        "adiabatic_scalar": {
            "theta_gamma0": f"-0.5 * {adiabatic_seed}",
            "theta_gamma1": f"({adiabatic_k_eta} / 6.0) * "
            f"{adiabatic_seed}",
            "theta_gamma2": f"{adiabatic_k_eta_sq} * "
            f"{adiabatic_seed} / 30.0",
            "e_gamma0": "0.0",
            "e_gamma1": "0.0",
            "e_gamma2": "0.0",
            "delta_b": f"-1.5 * {adiabatic_seed}",
            "theta_b": adiabatic_velocity_divergence,
            "delta_c": f"-1.5 * {adiabatic_seed}",
            "theta_c": adiabatic_velocity_divergence,
            "delta_nu": f"-2.0 * {adiabatic_seed}",
            "theta_nu": adiabatic_velocity_divergence,
            "sigma_nu": f"{adiabatic_k_eta_sq} * " f"{adiabatic_seed} / 15.0",
            "delta_nu_massive": f"-2.0 * {adiabatic_seed}",
            "theta_nu_massive": adiabatic_velocity_divergence,
            "sigma_nu_massive": f"{adiabatic_k_eta_sq} * "
            f"{adiabatic_seed} / 15.0",
        },
        "baryon_isocurvature": {
            "delta_b": "seed",
            "theta_b": isocurvature_velocity,
            "theta_gamma0": "-0.25 * seed",
            "theta_gamma1": compensated_brightness_dipole,
            "theta_gamma2": f"{k_eta_sq} * seed / 60.0",
            "delta_nu": "-0.25 * seed",
            "theta_nu": isocurvature_velocity,
            "sigma_nu": f"{k_eta_sq} * seed / 30.0",
            "delta_nu_massive": "-0.25 * seed",
            "theta_nu_massive": isocurvature_velocity,
            "sigma_nu_massive": f"{k_eta_sq} * seed / 30.0",
        },
        "cdm_isocurvature": {
            "theta_gamma0": "0.25 * seed",
            "theta_gamma1": compensated_brightness_dipole,
            "theta_gamma2": f"{k_eta_sq} * seed / 60.0",
            "delta_b": "-0.5 * seed",
            "theta_b": isocurvature_velocity,
            "delta_c": "seed",
            "theta_c": "0.0",
            "delta_nu": "-0.5 * seed",
            "theta_nu": isocurvature_velocity,
            "sigma_nu": f"{k_eta_sq} * seed / 30.0",
            "delta_nu_massive": "-0.5 * seed",
            "theta_nu_massive": isocurvature_velocity,
            "sigma_nu_massive": f"{k_eta_sq} * seed / 30.0",
        },
        "neutrino_density_isocurvature": {
            "delta_nu": "seed",
            "theta_gamma0": "-0.25 * seed",
            "delta_b": "-0.75 * seed",
            "delta_c": "-0.75 * seed",
            "theta_gamma1": compensated_brightness_dipole,
            "theta_b": isocurvature_velocity,
            "theta_c": isocurvature_velocity,
            "theta_nu": isocurvature_velocity,
            "sigma_nu": f"{k_eta_sq} * seed / 15.0",
            "delta_nu_massive": "seed",
            "theta_nu_massive": isocurvature_velocity,
            "sigma_nu_massive": f"{k_eta_sq} * seed / 15.0",
        },
        "neutrino_velocity_isocurvature": {
            "theta_gamma0": neutrino_velocity_density_constraint,
            "theta_gamma1": neutrino_velocity_dipole,
            "theta_b": neutrino_velocity_divergence,
            "theta_c": neutrino_velocity_divergence,
            "theta_nu": neutrino_velocity_divergence,
            "sigma_nu": neutrino_velocity_quadrupole,
            "theta_nu_massive": neutrino_velocity_divergence,
            "sigma_nu_massive": neutrino_velocity_quadrupole,
        },
        "tensor_mode": {},
    }
    seed_map = dict(by_mode.get(mode, {}))
    if gauge == "synchronous":
        metric_seed = _scalar_metric_seed_amplitude(mode)
        seed_map.update(
            {
                "h_sync_metric": (
                    "0.5 * (acoustic_k * scalar_initial_conformal_time) * "
                    "(acoustic_k * scalar_initial_conformal_time) * "
                    f"({metric_seed})"
                ),
                "eta_sync_metric": f"2.0 * ({metric_seed})",
                "gauge_shift_alpha": (
                    "0.5 * scalar_initial_conformal_time * " f"({metric_seed})"
                ),
            }
        )
    return seed_map


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
    previous_coefficient: float | None = None,
    use_physical_terminal_closure: bool = False,
) -> str:
    """Return one hierarchy recurrence RHS for the generated scalar route."""

    previous_coeff = (
        float(moment) / float((2 * moment) + 1)
        if previous_coefficient is None
        else float(previous_coefficient)
    )
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        if use_physical_terminal_closure:
            pieces[0] = f"1 * acoustic_k * {previous_name}"
            closure_scale = (
                "sqrt((acoustic_k * eta) * (acoustic_k * eta) + "
                f"{float(moment + 1):.16g} * {float(moment + 1):.16g})"
            )
            pieces.append(
                f"- acoustic_k * {float(moment + 1):.16g} * {name} / "
                f"{closure_scale}"
            )
        else:
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


def _scalar_streaming_hierarchy_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    streaming_speed_name: str,
    use_physical_terminal_closure: bool = False,
) -> str:
    """Return one q-resolved hierarchy RHS with one streaming factor."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    pieces = [
        f"{previous_coeff:.16g} * acoustic_k * "
        f"{streaming_speed_name} * {previous_name}"
    ]
    if next_name is None:
        if use_physical_terminal_closure:
            pieces[0] = (
                f"acoustic_k * {streaming_speed_name} * {previous_name}"
            )
            closure_scale = (
                "sqrt((acoustic_k * eta) * (acoustic_k * eta) + "
                f"{float(moment + 1):.16g} * {float(moment + 1):.16g})"
            )
            pieces.append(
                f"- acoustic_k * {streaming_speed_name} * "
                f"{float(moment + 1):.16g} * {name} / {closure_scale}"
            )
        else:
            pieces.append(
                f"- {float(moment + 1):.16g} / {float((2 * moment) + 1):.16g} "
                f"* acoustic_k * {streaming_speed_name} * {name}"
            )
    else:
        next_coeff = float(moment + 1) / float((2 * moment) + 1)
        pieces.append(
            f"- {next_coeff:.16g} * acoustic_k * "
            f"{streaming_speed_name} * {next_name}"
        )
    return " ".join(pieces)


def _scalar_polarization_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    collision_term: str | None = None,
    use_physical_terminal_closure: bool = False,
) -> str:
    """Return one scalar E hierarchy RHS with spin-2 streaming factors."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        if use_physical_terminal_closure:
            pieces[0] = (
                f"{float(moment) / max(float(moment - 2), 1.0):.16g} * "
                f"acoustic_k * {previous_name}"
            )
            closure_scale = (
                "sqrt((acoustic_k * eta) * (acoustic_k * eta) + "
                f"{float(moment + 3):.16g} * {float(moment + 3):.16g})"
            )
            pieces.append(
                f"- acoustic_k * {float(moment + 3):.16g} * {name} / "
                f"{closure_scale}"
            )
        else:
            pieces.append(
                f"- {float(moment + 3):.16g} / "
                f"{float((2 * moment) + 1):.16g} * acoustic_k * {name}"
            )
    else:
        next_coeff = float((moment + 3) * (moment - 1)) / float(
            ((2 * moment) + 1) * (moment + 1)
        )
        pieces.append(f"- {next_coeff:.16g} * acoustic_k * {next_name}")
    if collision_term is not None:
        pieces.append(collision_term)
    return " ".join(pieces)


def _vector_hierarchy_next_coeff(moment: int) -> float:
    """Return the vector hierarchy coefficient multiplying ``F_{l+1}``."""

    moment_value = float(moment)
    return (
        moment_value
        * (moment_value + 2.0)
        / ((2.0 * moment_value + 1.0) * (moment_value + 1.0))
    )


def _vector_polarization_next_coeff(moment: int) -> float:
    """Return the vector polarization coefficient for ``E_{l+1}`` or ``B``."""

    moment_value = float(moment)
    return (
        (moment_value + 3.0)
        * moment_value
        * (moment_value - 1.0)
        * (moment_value + 2.0)
        / ((2.0 * moment_value + 1.0) * (moment_value + 1.0) ** 3)
    )


def _vector_hierarchy_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    collision_term: str | None = None,
) -> str:
    """Return one vector hierarchy RHS using the CAMB flat-space closure."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        pieces[0] = (
            f"{float(moment) / float(moment - 1):.16g} * acoustic_k * "
            f"{previous_name}"
        )
        pieces.append(f"- {float(moment + 2):.16g} * {name} / vector_eta_safe")
    else:
        next_coeff = _vector_hierarchy_next_coeff(moment)
        pieces.append(f"- {next_coeff:.16g} * acoustic_k * {next_name}")
    if collision_term is not None:
        pieces.append(collision_term)
    return " ".join(pieces)


def _vector_polarization_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    opposite_name: str,
    sign: int,
    collision_term: str | None = None,
) -> str:
    """Return one vector polarization hierarchy RHS."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    opposite_coeff = 2.0 / (float(moment) * float(moment + 1))
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        pieces[0] = (
            f"{float(moment) / float(moment - 1):.16g} * acoustic_k * "
            f"{previous_name}"
        )
        pieces.append(f"- {float(moment + 2):.16g} * {name} / vector_eta_safe")
    else:
        next_coeff = _vector_polarization_next_coeff(moment)
        pieces.append(f"- {next_coeff:.16g} * acoustic_k * {next_name}")
    pieces.append(
        f"{'+' if sign > 0 else '-'} "
        f"{opposite_coeff:.16g} * acoustic_k * {opposite_name}"
    )
    if collision_term is not None:
        pieces.append(collision_term)
    return " ".join(pieces)


def _tensor_hierarchy_next_coeff(moment: int) -> float:
    """Return the tensor hierarchy coefficient multiplying ``F_{l+1}``."""

    moment_value = float(moment)
    return (((moment_value + 1.0) * (moment_value + 1.0)) - 4.0) / (
        (2.0 * moment_value + 1.0) * (moment_value + 1.0)
    )


def _tensor_hierarchy_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    collision_term: str | None = None,
) -> str:
    """Return one tensor hierarchy RHS with spin-2 free streaming."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        pieces[0] = (
            f"{float(moment) / float(moment - 2):.16g} * acoustic_k * "
            f"{previous_name}"
        )
        pieces.append(f"- {float(moment + 3):.16g} * {name} / tensor_eta_safe")
    else:
        next_coeff = _tensor_hierarchy_next_coeff(moment)
        pieces.append(f"- {next_coeff:.16g} * acoustic_k * {next_name}")
    if collision_term is not None:
        pieces.append(collision_term)
    return " ".join(pieces)


def _tensor_streaming_hierarchy_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    streaming_speed_name: str,
) -> str:
    """Return one q-resolved tensor hierarchy RHS."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    pieces = [
        f"{previous_coeff:.16g} * acoustic_k * "
        f"{streaming_speed_name} * {previous_name}"
    ]
    if next_name is None:
        pieces[0] = (
            f"{float(moment) / float(moment - 2):.16g} * acoustic_k * "
            f"{streaming_speed_name} * {previous_name}"
        )
        pieces.append(f"- {float(moment + 3):.16g} * {name} / tensor_eta_safe")
    else:
        next_coeff = _tensor_hierarchy_next_coeff(moment)
        pieces.append(
            f"- {next_coeff:.16g} * acoustic_k * "
            f"{streaming_speed_name} * {next_name}"
        )
    return " ".join(pieces)


def _tensor_polarization_recurrence_rhs(
    *,
    name: str,
    moment: int,
    previous_name: str,
    next_name: str | None,
    opposite_name: str,
    sign: int,
    collision_term: str | None = None,
) -> str:
    """Return one tensor polarization hierarchy RHS."""

    previous_coeff = float(moment) / float((2 * moment) + 1)
    opposite_coeff = 4.0 / (float(moment) * float(moment + 1))
    pieces = [f"{previous_coeff:.16g} * acoustic_k * {previous_name}"]
    if next_name is None:
        pieces[0] = (
            f"{float(moment) / float(moment - 2):.16g} * acoustic_k * "
            f"{previous_name}"
        )
        pieces.append(f"- {float(moment + 3):.16g} * {name} / tensor_eta_safe")
    else:
        moment_value = float(moment)
        tensor_factor = (
            (moment_value + 3.0) * (moment_value - 1.0) / (moment_value + 1.0)
        )
        next_coeff = (
            tensor_factor
            * tensor_factor
            / ((2.0 * moment_value + 1.0) * (moment_value + 1.0))
        )
        pieces.append(f"- {next_coeff:.16g} * acoustic_k * {next_name}")
    pieces.append(
        f"{'+' if sign > 0 else '-'} "
        f"{opposite_coeff:.16g} * acoustic_k * {opposite_name}"
    )
    if collision_term is not None:
        pieces.append(collision_term)
    return " ".join(pieces)


def _materialize_bounded_derived_sum(
    derived_entries: dict[str, Any],
    component_names: Iterable[str],
    *,
    name_prefix: str,
    description: str,
    units: str,
) -> str:
    """Return a bounded-AST sum, materializing intermediate partials."""

    active_names = [str(name) for name in component_names]
    if not active_names:
        return "0.0"
    level = 0
    while len(active_names) > 32:
        partial_names: list[str] = []
        for chunk_index, start in enumerate(range(0, len(active_names), 32)):
            partial_name = f"{name_prefix}_level{level}_{chunk_index}"
            derived_entries[partial_name] = {
                "expression": " + ".join(active_names[start : start + 32]),
                "description": description,
                "units": units,
            }
            partial_names.append(partial_name)
        active_names = partial_names
        level += 1
    return " + ".join(active_names)


def _materialize_declared_scalar_hierarchy_contract(
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    """Return a generated scalar hierarchy contract when metadata is enough."""

    if _has_explicit_declared_runtime_graph(contract):
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
    declared_modes = _standard_initial_mode_names(initial_condition_families)
    if len(declared_modes) > 1:
        raise ValueError(
            "Declare at most one auto-generated initial-condition family "
            "per perturbation contract: " + ", ".join(declared_modes)
        )
    initial_mode = _select_standard_initial_mode(initial_condition_families)
    if initial_mode is None:
        return contract, False

    numerics = contract.get("numerics", {}) or {}
    if not isinstance(numerics, Mapping):
        numerics = {}
    gauge = str(contract.get("gauge") or "conformal_newtonian")
    sync_gauge = gauge == "synchronous"
    invariant_gauge = gauge == "gauge_invariant"
    has_cdm = "cdm" in species
    has_massless_neutrino = (
        "massless_neutrino" in hierarchy_families
        and "massless_neutrino" in species
    )
    has_massive_neutrino = (
        "massive_neutrino" in hierarchy_families
        and "massive_neutrino" in species
    )
    if initial_mode == "cdm_isocurvature" and not has_cdm:
        raise ValueError("cdm_isocurvature requires a declared cdm species")
    if (
        initial_mode
        in {"neutrino_density_isocurvature", "neutrino_velocity_isocurvature"}
        and not has_massless_neutrino
    ):
        raise ValueError(
            f"{initial_mode} requires a declared massless_neutrino species"
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
    neutrino_default_l_max = (
        hierarchy_families.get("massless_neutrino", {}).get("default_l_max", 4)
        if has_massless_neutrino
        else 3
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
    declared_source_definitions = copy.deepcopy(
        materialized.get("sources", {}) or {}
    )

    def _declared_source_expression(role: str) -> str | None:
        """Return one model-declared scalar source closure by role."""

        matches = [
            definition.get("expression")
            for definition in declared_source_definitions.values()
            if isinstance(definition, Mapping)
            and definition.get("role") == role
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Declared scalar hierarchy permits one '{role}' source "
                "closure"
            )
        if not matches:
            return None
        expression = matches[0]
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(
                f"Declared scalar hierarchy '{role}' source closure must "
                "declare an expression"
            )
        return expression

    scalar_sector = dict(
        (materialized.get("sectors", {}) or {}).get("scalar", {}) or {}
    )
    if scalar_sector:
        supported_gauges = list(
            scalar_sector.get("supported_gauges", []) or []
        )
        if gauge not in supported_gauges:
            supported_gauges.append(gauge)
        scalar_sector["supported_gauges"] = supported_gauges
        materialized["sectors"] = dict(materialized.get("sectors", {}) or {})
        materialized["sectors"]["scalar"] = scalar_sector
    if sync_gauge:
        phi_state_name = "h_sync_metric"
        psi_state_name = "eta_sync_metric"
        alpha_state_name = "gauge_shift_alpha"
        metric_variables = {
            phi_state_name: _metadata_entry(
                "synchronous_metric_trace",
                "Synchronous-gauge trace metric perturbation h.",
                units=_DIMENSIONLESS_UNITS,
                gauge_role="synchronous_metric_trace",
            ),
            psi_state_name: _metadata_entry(
                "synchronous_metric_shear",
                "Synchronous-gauge shear metric perturbation eta.",
                units=_DIMENSIONLESS_UNITS,
                gauge_role="synchronous_metric_shear",
            ),
            alpha_state_name: _metadata_entry(
                "scalar_gauge_shift_generator",
                "Scalar gauge-shift generator alpha.",
                units=_DIMENSIONLESS_UNITS,
            ),
            "Phi": _metadata_entry(
                "observable_curvature_potential",
                "Observable-basis curvature potential.",
                units=_DIMENSIONLESS_UNITS,
            ),
            "Psi": _metadata_entry(
                "observable_lapse_potential",
                "Observable-basis lapse potential.",
                units=_DIMENSIONLESS_UNITS,
            ),
            "Phi_gi": _metadata_entry(
                "gauge_invariant_curvature_potential",
                "Gauge-invariant curvature potential evolved with the "
                "synchronous metric states.",
                units=_DIMENSIONLESS_UNITS,
            ),
        }
    elif invariant_gauge:
        phi_state_name = "Phi_gi"
        psi_state_name = "Psi_gi"
        alpha_state_name = None
        metric_variables = {
            phi_state_name: _metadata_entry(
                "gauge_invariant_curvature_potential",
                "Gauge-invariant Bardeen curvature potential Phi.",
                units=_DIMENSIONLESS_UNITS,
            ),
            psi_state_name: _metadata_entry(
                "gauge_invariant_lapse_potential",
                "Gauge-invariant Bardeen lapse potential Psi.",
                units=_DIMENSIONLESS_UNITS,
            ),
            "Phi": _metadata_entry(
                "observable_curvature_potential",
                "Observable-basis curvature potential.",
                units=_DIMENSIONLESS_UNITS,
            ),
            "Psi": _metadata_entry(
                "observable_lapse_potential",
                "Observable-basis lapse potential.",
                units=_DIMENSIONLESS_UNITS,
            ),
        }
    else:
        phi_state_name = "Phi"
        psi_state_name = "Psi"
        alpha_state_name = None
        metric_variables = {
            phi_state_name: _metadata_entry(
                "metric_potential_phi",
                "Conformal-Newtonian spatial-curvature potential Phi.",
                units=_DIMENSIONLESS_UNITS,
                gauge_role="curvature_potential",
            ),
            psi_state_name: _metadata_entry(
                "metric_potential_psi",
                "Conformal-Newtonian lapse potential Psi.",
                units=_DIMENSIONLESS_UNITS,
                gauge_role="newtonian_potential",
            ),
        }
    variables: dict[str, Any] = {
        "delta_b": _metadata_entry(
            "baryon_density_contrast",
            "Baryon density contrast delta_b.",
            units=_DIMENSIONLESS_UNITS,
        ),
        "theta_b": _metadata_entry(
            "baryon_velocity_divergence",
            "Baryon velocity divergence theta_b.",
            units=_INVERSE_MPC_UNITS,
        ),
        **metric_variables,
    }
    if has_cdm:
        variables.update(
            {
                "delta_c": _metadata_entry(
                    "cdm_density_contrast",
                    "Cold-dark-matter density contrast delta_c.",
                    units=_DIMENSIONLESS_UNITS,
                ),
                "theta_c": _metadata_entry(
                    "cdm_velocity_divergence",
                    "Cold-dark-matter velocity divergence theta_c.",
                    units=_INVERSE_MPC_UNITS,
                ),
            }
        )
    if has_massless_neutrino:
        variables.update(
            {
                "delta_nu": _metadata_entry(
                    "massless_neutrino_density_contrast",
                    "Massless-neutrino density contrast delta_nu.",
                    units=_DIMENSIONLESS_UNITS,
                ),
                "theta_nu": _metadata_entry(
                    "massless_neutrino_velocity_divergence",
                    "Massless-neutrino velocity divergence theta_nu.",
                    units=_INVERSE_MPC_UNITS,
                ),
                "sigma_nu": _metadata_entry(
                    "massless_neutrino_anisotropic_stress",
                    "Massless-neutrino anisotropic stress sigma_nu.",
                    units=_DIMENSIONLESS_UNITS,
                ),
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
        variables[_scalar_temperature_name(moment)] = _metadata_entry(
            kind,
            f"Photon temperature multipole Theta_gamma,{int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="scalar_like",
        )
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
        variables[_scalar_polarization_name(moment)] = _metadata_entry(
            kind,
            f"Photon E-polarization multipole E_gamma,{int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="scalar_like",
        )
    variables["polarization_b_mode_seed"] = _metadata_entry(
        "polarization_b_seed",
        "Declared primordial or sourced B-mode transfer seed.",
        units=_DIMENSIONLESS_UNITS,
        projection_role="b_mode",
        parity="odd",
        spin=2.0,
    )
    variables["visibility_polarization_moment"] = _metadata_entry(
        "photon_scalar_visibility_weighted_source_moment",
        "Visibility-weighted scalar polarization source moment.",
        units=_INVERSE_MPC_UNITS,
        tensor_character="scalar_like",
    )
    if has_massless_neutrino:
        for moment in range(3, neutrino_l_max + 1):
            variables[_scalar_neutrino_name(moment)] = _metadata_entry(
                "massless_neutrino_multipole",
                f"Massless-neutrino multipole F_nu,{int(moment)}.",
                units=_DIMENSIONLESS_UNITS,
                tensor_character="scalar_like",
            )
    for q_index in range(massive_neutrino_grid_count):
        for moment in range(massive_neutrino_l_max + 1):
            q_name = _scalar_massive_neutrino_q_name(
                q_index,
                moment,
            )
            if moment == 0:
                kind = "massive_neutrino_momentum_bin_density_contrast"
            elif moment == 1:
                kind = "massive_neutrino_momentum_bin_velocity_dipole"
            elif moment == 2:
                kind = "massive_neutrino_momentum_bin_anisotropic_stress"
            else:
                kind = "massive_neutrino_momentum_bin_multipole"
            variables[q_name] = _metadata_entry(
                kind,
                "Massive-neutrino momentum-bin perturbation for q index "
                f"{int(q_index)} and multipole {int(moment)}.",
                units=_DIMENSIONLESS_UNITS,
                tensor_character="scalar_like",
            )

    # ``theta_gamma1`` is the temperature dipole.  Its streaming coefficient
    # is k (the photon velocity divergence is the separately derived
    # ``3*k*theta_gamma1``), while the Newtonian-gauge continuity equation
    # carries the negative curvature-potential derivative.
    photon_monopole_rhs = "-acoustic_k * theta_gamma1 - Phi_tau"
    photon_dipole_rhs = (
        "(acoustic_k / 3.0) * "
        "(theta_gamma0 + Psi - 2.0 * theta_gamma2) + thomson_drag"
    )
    photon_quadrupole_metric_drive = "0.0"
    baryon_density_rhs = "-theta_b + 3.0 * Phi_tau"
    baryon_euler_rhs = (
        "-Hconf * theta_b + acoustic_k_sq * baryon_sound_speed_sq * "
        "delta_b + baryon_thomson_drag + acoustic_k_sq * Psi"
    )
    declared_baryon_euler_rhs = _declared_source_expression("baryon_euler")
    if declared_baryon_euler_rhs is not None:
        baryon_euler_rhs = declared_baryon_euler_rhs
    cdm_density_rhs = "-theta_c + 3.0 * Phi_tau"
    cdm_euler_rhs = "-Hconf * theta_c + acoustic_k_sq * Psi"
    neutrino_density_rhs = "-(4.0 / 3.0) * theta_nu + 4.0 * Phi_tau"
    neutrino_euler_rhs = "acoustic_k_sq * (0.25 * delta_nu + Psi - sigma_nu)"
    neutrino_quadrupole_metric_drive = "0.0"

    equations: dict[str, Any] = {
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
            "rhs": photon_dipole_rhs,
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
                f"+ {photon_quadrupole_metric_drive}"
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
            "rhs": "0.0",
            "role": "polarization",
        },
        "evolve_e_gamma1": {
            "lhs": {
                "kind": "derivative",
                "variable": "e_gamma1",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "0.0",
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
                f"- {1.0 / 3.0:.16g} * acoustic_k * "
                f"{_scalar_polarization_name(3)}"
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
            "rhs": baryon_density_rhs,
            "role": "continuity",
        },
        "evolve_theta_b": {
            "lhs": {
                "kind": "derivative",
                "variable": "theta_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": baryon_euler_rhs,
            "role": "euler",
        },
    }
    if has_cdm:
        equations.update(
            {
                "evolve_delta_c": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "delta_c",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": cdm_density_rhs,
                    "role": "continuity",
                },
                "evolve_theta_c": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "theta_c",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": cdm_euler_rhs,
                    "role": "euler",
                },
            }
        )
    if has_massless_neutrino:
        equations.update(
            {
                "evolve_delta_nu": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "delta_nu",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": neutrino_density_rhs,
                    "role": "continuity",
                },
                "evolve_theta_nu": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "theta_nu",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": neutrino_euler_rhs,
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
                        f"{_scalar_neutrino_name(3)} "
                        f"+ {neutrino_quadrupole_metric_drive}"
                    ),
                    "role": "hierarchy",
                },
            }
        )
    metric_evolution_state_name = "Phi_gi" if sync_gauge else phi_state_name
    if metric_evolution_state_name is not None:
        equations[f"evolve_{metric_evolution_state_name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": metric_evolution_state_name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "Phi_tau",
            "role": "metric",
        }
    if sync_gauge:
        equations.update(
            {
                "evolve_h_sync_metric": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "h_sync_metric",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "h_sync_metric_tau",
                    "role": "metric",
                },
                "evolve_eta_sync_metric": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "eta_sync_metric",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "eta_sync_metric_tau",
                    "role": "metric",
                },
                "evolve_gauge_shift_alpha": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "gauge_shift_alpha",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "gauge_shift_alpha_tau",
                    "role": "metric",
                },
            }
        )

    def _uses_scalar_terminal_closure(family_name: str) -> bool:
        """Validate and select the declared scalar hierarchy closure."""

        family_entry = hierarchy_families.get(family_name, {})
        closure_name = str(family_entry.get("closure", "")).strip()
        if closure_name != "free_streaming_scalar":
            raise ValueError(
                "Declared scalar hierarchy family "
                f"'{family_name}' must declare the supported terminal "
                "closure 'free_streaming_scalar'"
            )
        return True

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
                use_physical_terminal_closure=_uses_scalar_terminal_closure(
                    "photon_temperature"
                ),
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
            "rhs": _scalar_polarization_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=previous_name,
                next_name=next_name,
                use_physical_terminal_closure=_uses_scalar_terminal_closure(
                    "photon_polarization_e"
                ),
            ),
            "role": "polarization",
        }
    if has_massless_neutrino:
        for moment in range(3, neutrino_l_max + 1):
            name = _scalar_neutrino_name(moment)
            next_name = None
            if moment < neutrino_l_max:
                next_name = _scalar_neutrino_name(moment + 1)
            previous_name = (
                "sigma_nu"
                if moment == 3
                else _scalar_neutrino_name(moment - 1)
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
                    use_physical_terminal_closure=(
                        _uses_scalar_terminal_closure("massless_neutrino")
                    ),
                ),
                "role": "hierarchy",
            }
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        mode_seed_expressions = _scalar_hierarchy_base_seed_expressions(
            initial_mode,
            gauge=gauge,
        )
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
            q_streaming_speed_name = (
                _scalar_massive_neutrino_q_streaming_speed_name(q_index)
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
                    f"-acoustic_k * {q_theta_name} "
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
                    f"- Hconf * (1.0 - {q_streaming_speed_name} * "
                    f"{q_streaming_speed_name}) * {q_theta_name} + "
                    f"(acoustic_k / 3.0) * "
                    f"{q_streaming_speed_name} * {q_streaming_speed_name} * "
                    f"({q_delta_name} - 2.0 * {q_sigma_name}) - "
                    f"(acoustic_k / 3.0) * "
                    f"Psi * "
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
                    f"{q_theta_name} "
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
                equations[f"evolve_{name}"] = {
                    "lhs": {
                        "kind": "derivative",
                        "variable": name,
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": _scalar_streaming_hierarchy_recurrence_rhs(
                        name=name,
                        moment=moment,
                        previous_name=previous_name,
                        next_name=next_name,
                        streaming_speed_name=q_streaming_speed_name,
                        use_physical_terminal_closure=(
                            _uses_scalar_terminal_closure("massive_neutrino")
                        ),
                    ),
                    "role": "hierarchy",
                }

    observable_theta_gamma0_expression = "theta_gamma0"
    observable_theta_gamma1_expression = "theta_gamma1"
    observable_delta_b_expression = "delta_b"
    observable_theta_b_expression = "theta_b"
    observable_delta_nu_expression = "delta_nu"
    observable_theta_nu_expression = "theta_nu"
    observable_delta_nu_massive_expression = "delta_nu_massive"
    observable_theta_nu_massive_expression = "theta_nu_massive"
    if has_cdm:
        observable_delta_c_expression = "delta_c"
        observable_theta_c_expression = "theta_c"

    materialized["variables"] = variables
    massless_fraction_expression = "0.0"
    if has_massless_neutrino:
        massless_fraction_expression = "Omega_nu0"
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        massless_fraction_expression = (
            "Omega_nu0 * 0.5 * (Neff - num_massive_neutrinos + "
            "abs(Neff - num_massive_neutrinos)) / Neff"
        )
    matter_density_terms = ["Omega_b0 * observable_delta_b"]
    if has_cdm:
        matter_density_terms.insert(0, "Omega_c0 * observable_delta_c")
    declared_matter_density_expression = _declared_source_expression(
        "matter_density"
    )
    if declared_matter_density_expression is not None:
        matter_density_terms = [declared_matter_density_expression]
    matter_density_source_expression = (
        f"({ ' + '.join(matter_density_terms) }) / a"
    )
    radiation_density_source_expression = (
        "(4.0 * Omega_gamma0 * observable_theta_gamma0"
    )
    if has_massless_neutrino:
        neutrino_density_weight = (
            "massless_neutrino_fraction"
            if has_massive_neutrino and massive_neutrino_grid_count > 0
            else "Omega_nu0"
        )
        radiation_density_source_expression += (
            f" + {neutrino_density_weight} * observable_delta_nu"
        )
    radiation_density_source_expression += ") / (a * a)"
    total_density_source_expression = (
        "matter_density_source + radiation_density_source"
    )
    matter_momentum_terms = ["Omega_b0 * observable_theta_b"]
    if has_cdm:
        matter_momentum_terms.insert(0, "Omega_c0 * observable_theta_c")
    declared_matter_momentum_expression = _declared_source_expression(
        "matter_momentum"
    )
    if declared_matter_momentum_expression is not None:
        matter_momentum_terms = [declared_matter_momentum_expression]
    radiation_momentum_terms = [
        "(4.0 / 3.0) * Omega_gamma0 * photon_velocity_divergence"
    ]
    if has_massless_neutrino:
        neutrino_momentum_weight = (
            "massless_neutrino_fraction"
            if has_massive_neutrino and massive_neutrino_grid_count > 0
            else "Omega_nu0"
        )
        radiation_momentum_terms.append(
            "(4.0 / 3.0) * "
            f"{neutrino_momentum_weight} * observable_theta_nu"
        )
    total_momentum_source_expression = (
        f"({ ' + '.join(matter_momentum_terms) }) / a + "
        f"({ ' + '.join(radiation_momentum_terms) }) / (a * a)"
    )
    shear_terms = ["4.0 * Omega_gamma0 * observable_theta_gamma2"]
    if has_massless_neutrino:
        neutrino_shear_weight = (
            "massless_neutrino_fraction"
            if has_massive_neutrino and massive_neutrino_grid_count > 0
            else "Omega_nu0"
        )
        shear_terms.append(f"2.0 * {neutrino_shear_weight} * sigma_nu")
    total_shear_source_expression = f"({ ' + '.join(shear_terms) }) / (a * a)"
    derived_entries: dict[str, Any] = {
        "polarization_moment": {
            "expression": "theta_gamma2 + e_gamma0 + e_gamma2",
            "description": (
                "Scalar polarization source moment Pi = Theta_gamma,2 + "
                "E_gamma,0 + E_gamma,2 in the declared scalar hierarchy."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "scalar_initial_conformal_time": {
            "expression": "1.0 / (Hconf + 1.0e-30)",
            "description": (
                "Local conformal-Hubble time used to scale the regular "
                "scalar initial-condition series for each declared "
                "background."
            ),
            "units": "Mpc",
        },
        "scalar_neutrino_fraction": {
            "expression": ("Omega_nu0 / (Omega_gamma0 + Omega_nu0 + 1.0e-30)"),
            "description": (
                "Relativistic neutrino fraction used by the regular scalar "
                "initial-condition series."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "scalar_potential_seed": {
            "expression": (
                "(10.0 / (15.0 + 4.0 * scalar_neutrino_fraction)) * seed"
            ),
            "description": (
                "Radiation-era curvature potential sourced by primordial "
                "curvature seed."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "scalar_lapse_seed": {
            "expression": "scalar_potential_seed",
            "description": (
                "Radiation-era lapse potential for regular adiabatic scalar "
                "initial data."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "visibility_polarization_moment_tau_tau": {
            "kind": "scalar_source_time_derivative",
            "variable": "visibility_polarization_moment",
            "wrt": "tau",
            "order": 2,
            "description": (
                "Second conformal-time derivative of the visibility-weighted "
                "scalar polarization source moment."
            ),
            "units": _INVERSE_MPC_CUBED_UNITS,
        },
        "acoustic_k": {
            "expression": "k",
            "description": "Scalar acoustic wave number.",
            "units": _INVERSE_MPC_UNITS,
        },
        "acoustic_k_sq": {
            "expression": "acoustic_k * acoustic_k",
            "description": "Squared scalar acoustic wave number.",
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "observable_theta_gamma0": {
            "expression": observable_theta_gamma0_expression,
            "description": ("Observable-basis photon temperature monopole."),
            "units": _DIMENSIONLESS_UNITS,
        },
        "observable_theta_gamma1": {
            "expression": observable_theta_gamma1_expression,
            "description": ("Observable-basis photon temperature dipole."),
            "units": _DIMENSIONLESS_UNITS,
        },
        "observable_theta_gamma2": {
            "expression": "theta_gamma2",
            "description": ("Observable-basis photon temperature quadrupole."),
            "units": _DIMENSIONLESS_UNITS,
        },
        "observable_delta_b": {
            "expression": observable_delta_b_expression,
            "description": "Observable-basis baryon density contrast.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "observable_theta_b": {
            "expression": observable_theta_b_expression,
            "description": "Observable-basis baryon velocity divergence.",
            "units": _INVERSE_MPC_UNITS,
        },
        "photon_velocity_divergence": {
            "expression": "3.0 * acoustic_k * observable_theta_gamma1",
            "description": "Photon velocity divergence theta_gamma.",
            "units": _INVERSE_MPC_UNITS,
        },
        "matter_density_source": {
            "expression": matter_density_source_expression,
            "description": (
                "Time-dependent matter density source for the "
                "scalar Einstein energy constraint."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "massless_neutrino_fraction": {
            "expression": massless_fraction_expression,
            "description": "Present-day massless-neutrino density fraction.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "radiation_density_source": {
            "expression": radiation_density_source_expression,
            "description": (
                "Time-dependent radiation density source for the "
                "scalar Einstein energy constraint."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "total_density_source": {
            "expression": total_density_source_expression,
            "description": (
                "Total density source for the scalar Einstein system."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "total_shear_source": {
            "expression": total_shear_source_expression,
            "description": (
                "Total shear source for the scalar Einstein system."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "total_momentum_source": {
            "expression": total_momentum_source_expression,
            "description": (
                "Total momentum source for the scalar Einstein system."
            ),
            "units": _INVERSE_MPC_UNITS,
        },
        "einstein_gravity_strength": {
            "expression": "H0_over_c_Mpc_inv * H0_over_c_Mpc_inv",
            "description": (
                "Background gravity scale used by the scalar Einstein "
                "constraints."
            ),
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "metric_constraint_scale": {
            "expression": "acoustic_k_sq",
            "description": (
                "Exact scalar Einstein constraint wave-number scale."
            ),
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "metric_momentum_source_drive": {
            "expression": (
                "1.5 * einstein_gravity_strength * total_momentum_source "
                "/ acoustic_k_sq"
            ),
            "description": ("Source-side scalar momentum-constraint drive."),
            "units": _INVERSE_MPC_UNITS,
        },
        "metric_shear_correction": {
            "expression": (
                "3.0 * einstein_gravity_strength * total_shear_source "
                "/ metric_constraint_scale"
            ),
            "description": ("Scalar anisotropic-stress correction Phi - Psi."),
            "units": _DIMENSIONLESS_UNITS,
        },
        "photon_baryon_momentum_ratio": {
            "expression": "(4.0 * Omega_gamma0) / (3.0 * Omega_b0 * a)",
            "description": "Photon-to-baryon momentum-transfer ratio.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "baryon_thomson_drag": {
            "expression": (
                "- 3.0 * acoustic_k * photon_baryon_momentum_ratio * "
                "thomson_drag"
            ),
            "description": "Baryon-side Thomson drag counterpart.",
        },
    }
    if has_cdm:
        derived_entries.update(
            {
                "observable_delta_c": {
                    "expression": observable_delta_c_expression,
                    "description": "Observable-basis CDM density contrast.",
                    "units": _DIMENSIONLESS_UNITS,
                },
                "observable_theta_c": {
                    "expression": observable_theta_c_expression,
                    "description": "Observable-basis CDM velocity divergence.",
                    "units": _INVERSE_MPC_UNITS,
                },
            }
        )
    if has_massless_neutrino:
        derived_entries.update(
            {
                "observable_delta_nu": {
                    "expression": observable_delta_nu_expression,
                    "description": (
                        "Observable-basis massless-neutrino density "
                        "contrast."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "observable_theta_nu": {
                    "expression": observable_theta_nu_expression,
                    "description": (
                        "Observable-basis massless-neutrino velocity "
                        "divergence."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
            }
        )
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        derived_entries.update(
            {
                "massive_neutrino_density_source": {
                    "expression": (
                        "a * a * massive_neutrino_density_fraction * "
                        "observable_delta_nu_massive"
                    ),
                    "description": (
                        "Current massive-neutrino density source moment "
                        "for the scalar Einstein system."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "massive_neutrino_momentum_source": {
                    "expression": (
                        "(4.0 / 3.0) * a * a * "
                        "massive_neutrino_momentum_fraction * "
                        "observable_theta_nu_massive"
                    ),
                    "description": (
                        "Current massive-neutrino momentum source moment "
                        "for the scalar Einstein system."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
                "massive_neutrino_shear_source": {
                    "expression": (
                        "a * a * massive_neutrino_shear_fraction * "
                        "massive_neutrino_metric_shear"
                    ),
                    "description": (
                        "Current massive-neutrino shear source moment for "
                        "the scalar Einstein system."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
            }
        )
        derived_entries["total_density_source"] = {
            "expression": (
                "matter_density_source + radiation_density_source + "
                "massive_neutrino_density_source"
            ),
            "description": (
                "Total density source for the scalar Einstein system."
            ),
            "units": _DIMENSIONLESS_UNITS,
        }
        derived_entries["total_momentum_source"] = {
            "expression": (
                total_momentum_source_expression
                + " + massive_neutrino_momentum_source"
            ),
            "description": (
                "Total momentum source for the scalar Einstein system."
            ),
            "units": _INVERSE_MPC_UNITS,
        }
        derived_entries["total_shear_source"] = {
            "expression": (
                total_shear_source_expression
                + " + 2.0 * massive_neutrino_shear_source"
            ),
            "description": (
                "Total shear source for the scalar Einstein system."
            ),
            "units": _DIMENSIONLESS_UNITS,
        }
    if sync_gauge:
        derived_entries.update(
            {
                "eta_sync_metric_tau": {
                    "expression": (
                        "metric_momentum_source_drive + "
                        "(Hconf_tau - Hconf * Hconf) * gauge_shift_alpha"
                    ),
                    "description": (
                        "Numerically stable synchronous-gauge shear evolution "
                        "eta'."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
                "h_sync_metric_tau": {
                    "expression": (
                        "2.0 * acoustic_k_sq * gauge_shift_alpha - "
                        "6.0 * eta_sync_metric_tau"
                    ),
                    "description": ("Synchronous-gauge trace evolution h'."),
                    "units": _INVERSE_MPC_UNITS,
                },
                "gauge_shift_alpha_tau": {
                    "expression": "Psi - Hconf * gauge_shift_alpha",
                    "description": (
                        "Conformal-time derivative of the gauge shift."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
                "Phi_from_synchronous": {
                    "expression": (
                        "eta_sync_metric - Hconf * gauge_shift_alpha"
                    ),
                    "description": (
                        "Curvature potential reconstructed from the "
                        "synchronous variables."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "Psi_from_synchronous": {
                    "expression": (
                        "gauge_shift_alpha_tau + " "Hconf * gauge_shift_alpha"
                    ),
                    "description": (
                        "Lapse potential reconstructed from the synchronous "
                        "variables."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "Phi_tau": {
                    "kind": "metric_potential_time_derivative",
                    "expression": "metric_momentum_source_drive - Hconf * Psi",
                    "description": (
                        "Observable-basis curvature-potential derivative."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
            }
        )
    elif invariant_gauge:
        derived_entries.update(
            {
                "Phi_tau": {
                    "kind": "metric_potential_time_derivative",
                    "expression": "metric_momentum_source_drive - Hconf * Psi",
                    "description": (
                        "Gauge-invariant curvature-potential derivative."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
            }
        )
    else:
        derived_entries.update(
            {
                "Phi_tau": {
                    "kind": "metric_potential_time_derivative",
                    "expression": "metric_momentum_source_drive - Hconf * Psi",
                    "description": (
                        "Scalar Einstein-system relation for the curvature "
                        "potential time derivative."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
            }
        )
    derived_entries.update(
        {
            "metric_momentum_constraint": {
                "expression": "Phi_tau + Hconf * Psi",
                "description": (
                    "Scalar momentum-constraint combination "
                    "Phi_tau + Hconf Psi."
                ),
                "units": _INVERSE_MPC_UNITS,
            },
            "Psi_tau": {
                "kind": "metric_potential_time_derivative",
                "binding": "runtime_history_gradient",
                "variable": "Psi",
                "wrt": "tau",
                "order": 1,
                "description": (
                    "History-derived lapse-potential time derivative."
                ),
                "units": _INVERSE_MPC_UNITS,
            },
            "Phi_history_tau": {
                "kind": "metric_history_time_derivative",
                "binding": "runtime_history_gradient",
                "variable": "Phi",
                "wrt": "tau",
                "order": 1,
                "description": (
                    "History-derived curvature-potential time derivative "
                    "used by the integrated Sachs-Wolfe source; the runtime "
                    "binds this symbol to the evolved Phi history gradient."
                ),
                "units": _INVERSE_MPC_UNITS,
            },
            "einstein_energy_residual": {
                "expression": (
                    "acoustic_k_sq * Phi + "
                    "3.0 * Hconf * metric_momentum_constraint + "
                    "1.5 * einstein_gravity_strength * total_density_source"
                ),
                "description": ("Scalar Einstein energy-constraint residual."),
                "units": _INVERSE_MPC_SQUARED_UNITS,
            },
            "einstein_momentum_residual": {
                "expression": (
                    "acoustic_k_sq * (Phi_tau + Hconf * Psi) - "
                    "1.5 * einstein_gravity_strength * total_momentum_source"
                ),
                "description": (
                    "Scalar Einstein momentum-constraint residual."
                ),
                "units": _INVERSE_MPC_CUBED_UNITS,
            },
            "einstein_shear_residual": {
                "expression": (
                    "metric_constraint_scale * metric_shear_correction - "
                    "3.0 * einstein_gravity_strength * total_shear_source"
                ),
                "description": (
                    "Scalar Einstein anisotropic-stress residual evaluated "
                    "from the declared shear closure."
                ),
                "units": _INVERSE_MPC_SQUARED_UNITS,
            },
        }
    )
    if has_massive_neutrino and massive_neutrino_grid_count > 0:
        q_density_component_names = []
        q_pressure_component_names = []
        q_momentum_component_names = []
        q_shear_component_names = []
        aggregate_hierarchy_component_names: dict[int, list[str]] = {
            moment: [] for moment in range(3, massive_neutrino_l_max + 1)
        }
        for q_index in range(massive_neutrino_grid_count):
            q_prefix = f"massive_neutrino_q{q_index}"
            q_density_name = f"massive_neutrino_metric_density_q{q_index}"
            q_pressure_name = f"massive_neutrino_metric_pressure_q{q_index}"
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
            q_pressure_component_names.append(q_pressure_name)
            q_momentum_component_names.append(q_momentum_name)
            q_shear_component_names.append(q_shear_name)
            derived_entries[q_log_derivative_name] = {
                "expression": (
                    f"-{q_prefix}_point / (1.0 + exp(-{q_prefix}_point))"
                ),
                "description": (
                    "Logarithmic derivative of the thermal distribution."
                ),
                "units": _DIMENSIONLESS_UNITS,
            }
            derived_entries[q_streaming_speed_name] = {
                "expression": (
                    f"{q_prefix}_point / sqrt(("
                    f"{q_prefix}_point * {q_prefix}_point) + "
                    "(a * massive_neutrino_mass_eV / "
                    "neutrino_temperature_eV) * "
                    "(a * massive_neutrino_mass_eV / "
                    "neutrino_temperature_eV))"
                ),
                "description": (
                    "Streaming speed for one massive-neutrino momentum bin."
                ),
                "units": _DIMENSIONLESS_UNITS,
            }
            derived_entries[q_density_name] = {
                "expression": (
                    f"{q_prefix}_density_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 0)}"
                ),
                "description": (
                    "Momentum-grid-weighted q-bin density moment."
                ),
                "units": _DIMENSIONLESS_UNITS,
            }
            derived_entries[q_pressure_name] = {
                "expression": (
                    f"{q_prefix}_pressure_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 0)}"
                ),
                "description": (
                    "Momentum-grid-weighted q-bin pressure moment."
                ),
                "units": _DIMENSIONLESS_UNITS,
            }
            derived_entries[q_momentum_name] = {
                "expression": (
                    "acoustic_k * "
                    f"{q_prefix}_momentum_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 1)}"
                ),
                "description": (
                    "Momentum-grid-weighted q-bin momentum moment."
                ),
                "units": _INVERSE_MPC_UNITS,
            }
            derived_entries[q_shear_name] = {
                "expression": (
                    f"{q_prefix}_shear_weight * "
                    f"{_scalar_massive_neutrino_q_name(q_index, 2)}"
                ),
                "description": ("Momentum-grid-weighted q-bin shear moment."),
                "units": _DIMENSIONLESS_UNITS,
            }
            for moment in range(3, massive_neutrino_l_max + 1):
                component_name = (
                    f"massive_neutrino_metric_l{moment}_q{q_index}"
                )
                derived_entries[component_name] = {
                    "expression": (
                        f"{q_prefix}_shear_weight * "
                        f"{_scalar_massive_neutrino_q_name(q_index, moment)}"
                    ),
                    "description": (
                        "Momentum-grid-weighted q-bin higher multipole "
                        f"F_nu_m,{int(moment)}."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                }
                aggregate_hierarchy_component_names[moment].append(
                    component_name
                )
        density_sum_expression = _materialize_bounded_derived_sum(
            derived_entries,
            q_density_component_names,
            name_prefix="massive_neutrino_density_sum",
            description="Partial massive-neutrino density quadrature.",
            units=_DIMENSIONLESS_UNITS,
        )
        pressure_sum_expression = _materialize_bounded_derived_sum(
            derived_entries,
            q_pressure_component_names,
            name_prefix="massive_neutrino_pressure_sum",
            description="Partial massive-neutrino pressure quadrature.",
            units=_DIMENSIONLESS_UNITS,
        )
        momentum_sum_expression = _materialize_bounded_derived_sum(
            derived_entries,
            q_momentum_component_names,
            name_prefix="massive_neutrino_momentum_sum",
            description="Partial massive-neutrino momentum quadrature.",
            units=_INVERSE_MPC_UNITS,
        )
        shear_sum_expression = _materialize_bounded_derived_sum(
            derived_entries,
            q_shear_component_names,
            name_prefix="massive_neutrino_shear_sum",
            description="Partial massive-neutrino shear quadrature.",
            units=_DIMENSIONLESS_UNITS,
        )
        derived_entries.update(
            {
                "massive_neutrino_metric_density": {
                    "expression": density_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino density."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "massive_neutrino_metric_momentum": {
                    "expression": momentum_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino momentum."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
                "massive_neutrino_metric_pressure": {
                    "expression": pressure_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino pressure."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "massive_neutrino_metric_shear": {
                    "expression": shear_sum_expression,
                    "description": (
                        "Momentum-grid-weighted massive-neutrino shear."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "delta_nu_massive": {
                    "expression": "massive_neutrino_metric_density",
                    "description": (
                        "Q-integrated massive-neutrino density contrast."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "theta_nu_massive": {
                    "expression": "massive_neutrino_metric_momentum",
                    "description": (
                        "Q-integrated massive-neutrino momentum divergence."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
                "sigma_nu_massive": {
                    "expression": "massive_neutrino_metric_shear",
                    "description": (
                        "Q-integrated massive-neutrino anisotropic stress."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "observable_delta_nu_massive": {
                    "expression": observable_delta_nu_massive_expression,
                    "description": (
                        "Observable-basis massive-neutrino density contrast."
                    ),
                    "units": _DIMENSIONLESS_UNITS,
                },
                "observable_theta_nu_massive": {
                    "expression": observable_theta_nu_massive_expression,
                    "description": (
                        "Observable-basis massive-neutrino momentum "
                        "divergence."
                    ),
                    "units": _INVERSE_MPC_UNITS,
                },
            }
        )
        for moment in range(3, massive_neutrino_l_max + 1):
            component_expressions = aggregate_hierarchy_component_names[moment]
            aggregate_expression = _materialize_bounded_derived_sum(
                derived_entries,
                component_expressions,
                name_prefix=f"massive_neutrino_l{moment}_sum",
                description=(
                    "Partial massive-neutrino higher-multipole quadrature."
                ),
                units=_DIMENSIONLESS_UNITS,
            )
            derived_entries[_scalar_massive_neutrino_name(moment)] = {
                "expression": aggregate_expression,
                "description": (
                    "Q-integrated massive-neutrino higher multipole "
                    f"F_nu_m,{int(moment)}."
                ),
                "units": _DIMENSIONLESS_UNITS,
            }
    materialized["derived"] = derived_entries
    materialized["equations"] = equations
    collision_operator_entries = dict(
        materialized.get("collision_operators", {}) or {}
    )
    thomson_drag_entry = dict(
        collision_operator_entries.get("thomson_drag", {}) or {}
    )
    thomson_drag_entry.setdefault("sector", "scalar")
    thomson_drag_entry.setdefault("species", ["photon", "baryon"])
    thomson_drag_entry.setdefault(
        "expression",
        "collision_rate * ((theta_b / acoustic_k) / 3.0 - theta_gamma1)",
    )
    thomson_drag_entry.setdefault("counterpart", "baryon_thomson_drag")
    thomson_drag_entry.setdefault("integration_strategy", "exact")
    thomson_drag_entry.setdefault("activation_strategy", "always")
    thomson_drag_entry.setdefault("rate_expression", "collision_rate")
    thomson_drag_entry.setdefault(
        "exact_form",
        {
            "targets": [
                {"kind": "photon_temperature_dipole"},
                {"kind": "baryon_velocity_divergence"},
                {"kind": "photon_temperature_quadrupole"},
                {"kind": "photon_polarization_quadrupole"},
            ],
            "matrix": [
                [
                    "-1.0",
                    "1.0 / (3.0 * acoustic_k)",
                    "0.0",
                    "0.0",
                ],
                [
                    "3.0 * acoustic_k * photon_baryon_momentum_ratio",
                    "-photon_baryon_momentum_ratio",
                    "0.0",
                    "0.0",
                ],
                ["0.0", "0.0", "-0.8", "0.1"],
                ["0.0", "0.0", "0.05", "-0.25"],
            ],
            "damping_targets": [
                {"kind": "photon_temperature_octopole"},
                {"kind": "photon_temperature_multipole"},
                {"kind": "photon_polarization_octopole"},
                {"kind": "photon_polarization_multipole"},
            ],
            "damping_coefficient": "-1.0",
            "fast_manifold": True,
            "activation_strategy": "always",
        },
    )
    collision_operator_entries["thomson_drag"] = thomson_drag_entry
    materialized["collision_operators"] = collision_operator_entries
    conservation_rule_entries = dict(
        materialized.get("conservation_rules", {}) or {}
    )
    conservation_rule_entries.setdefault(
        "thomson_drag_balance",
        {
            "kind": "absolute_max",
            "expression": (
                "3.0 * acoustic_k * photon_baryon_momentum_ratio * "
                "thomson_drag + "
                "baryon_thomson_drag"
            ),
            "tolerance": 1.0e-12,
            "domain": "scalar",
        },
    )
    materialized["conservation_rules"] = conservation_rule_entries
    psi_closure_expression = "Phi - metric_shear_correction"
    if sync_gauge:
        materialized["constraints"] = {}
        materialized["closures"] = {
            "phi_closure": {
                "target": "Phi",
                "expression": "Phi_gi",
                "role": "closure",
            },
            "psi_closure": {
                "target": "Psi",
                "expression": "Phi_gi - metric_shear_correction",
                "role": "closure",
            },
        }
    elif invariant_gauge:
        materialized["constraints"] = {
            "observable_phi_constraint": {
                "target": "Phi",
                "expression": phi_state_name,
                "role": "constraint",
            },
        }
        materialized["closures"] = {
            "psi_closure": {
                "target": psi_state_name,
                "expression": psi_closure_expression,
                "role": "closure",
            },
            "observable_psi_closure": {
                "target": "Psi",
                "expression": psi_state_name,
                "role": "closure",
            },
        }
    else:
        materialized["constraints"] = {}
        materialized["closures"] = {
            "psi_closure": {
                "target": psi_state_name,
                "expression": psi_closure_expression,
                "role": "closure",
            }
        }
    materialized["closures"]["visibility_polarization_moment_closure"] = {
        "target": "visibility_polarization_moment",
        "expression": "visibility * polarization_moment",
        "role": "closure",
        "description": (
            "Visibility-weighted scalar polarization source moment."
        ),
    }
    generated_sources = {
        "temperature_monopole": {
            "expression": (
                "visibility * (observable_theta_gamma0 + Psi "
                "+ 0.25 * polarization_moment)"
            ),
            "role": "monopole",
            "description": "Visibility-weighted temperature monopole source.",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
            "notes": (
                "Uses Delta_gamma / 4 + Psi + Pi / 4 on the visibility "
                "surface."
            ),
        },
        "temperature_quadrupole": {
            "expression": "0.0",
            "role": "additive",
            "description": (
                "Deprecated split temperature quadrupole term; the scalar "
                "polarization contribution is included in the monopole "
                "line-of-sight source as Pi / 4."
            ),
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "temperature_quadrupole_derivative": {
            "expression": "0.0",
            "role": "additive_derivative",
            "description": (
                "Deprecated second-derivative temperature source retained "
                "as an explicit zero for contract compatibility."
            ),
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "temperature_doppler": {
            "expression": "visibility * observable_theta_b / acoustic_k",
            "role": "doppler",
            "description": "Visibility-weighted Doppler source.",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
            "notes": (
                "Uses the baryon velocity divergence theta_b after the "
                "line-of-sight derivative is transferred to the radial "
                "Bessel kernel."
            ),
        },
        "temperature_isw": {
            "expression": "exp(-tau) * (Phi_history_tau + Psi_tau)",
            "role": "isw",
            "description": (
                "Integrated Sachs-Wolfe source from the Weyl-potential "
                "time derivative."
            ),
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "polarization_source": {
            "expression": "0.75 * visibility * polarization_moment",
            "role": "polarization",
            "description": ("Standard scalar visibility-weighted E source."),
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "polarization_b_source": {
            "expression": "polarization_b_mode_seed",
            "role": "polarization_b",
            "description": "Declared odd-parity B-polarization source seed.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "lensing_potential": {
            "expression": "Phi + Psi",
            "role": "potential",
            "description": "Scalar Weyl-potential source for CMB lensing.",
            "units": _DIMENSIONLESS_UNITS,
        },
    }
    generated_sources.update(declared_source_definitions)
    materialized["sources"] = generated_sources
    materialized["observables"] = {
        "temperature": {
            "kind": "transfer_component",
            "projection": "line_of_sight_temperature",
            "source_terms": {
                "monopole": "temperature_monopole",
                "doppler": "temperature_doppler",
                "isw": "temperature_isw",
            },
            "description": "Temperature transfer function Delta_ell^T(k).",
        },
        "polarization_e": {
            "kind": "transfer_component",
            "projection": "line_of_sight_polarization_e",
            "source_terms": {"polarization": "polarization_source"},
            "description": "E-polarization transfer function Delta_ell^E(k).",
        },
        "polarization_b": {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {
                "polarization_b": "polarization_b_source",
            },
            "description": "B-polarization transfer function Delta_ell^B(k).",
        },
        "lensing_potential": {
            "kind": "transfer_component",
            "projection": "line_of_sight_lensing_potential",
            "source_terms": {
                "potential": "lensing_potential",
            },
            "description": (
                "Lensing-potential transfer function Delta_ell^phi(k)."
            ),
        },
        "TT": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "temperature",
            "description": "Temperature auto spectrum.",
            "notes": "Public solver returns D_ell^TT in muK^2.",
        },
        "TE": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "polarization_e",
            "description": "Temperature and E-polarization cross spectrum.",
            "notes": "Public solver returns D_ell^TE in muK^2.",
        },
        "EE": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "polarization_e",
            "description": "E-polarization auto spectrum.",
            "notes": "Public solver returns D_ell^EE in muK^2.",
        },
        "BB": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
            "description": "B-polarization auto spectrum.",
            "notes": "Public solver returns D_ell^BB in muK^2.",
        },
        "PP": {
            "kind": "angular_power_spectrum",
            "primary": "lensing_potential",
            "secondary": "lensing_potential",
            "description": "Lensing-potential auto spectrum.",
            "notes": (
                "Public solver returns clpp = [ell(ell+1)]^2 "
                "C_ell^{phiphi} / (2*pi)."
            ),
        },
        "TP": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "lensing_potential",
            "units": "dimensionless",
            "description": "Temperature and lensing-potential cross spectrum.",
            "notes": (
                "Public solver returns ell(ell+1) C_ell^{Tphi} / (2*pi)."
            ),
        },
        "EP": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "lensing_potential",
            "units": "dimensionless",
            "description": (
                "E-polarization and lensing-potential cross spectrum."
            ),
            "notes": (
                "Public solver returns ell(ell+1) C_ell^{Ephi} / (2*pi)."
            ),
        },
    }
    initial_conditions = copy.deepcopy(
        dict(materialized.get("initial_conditions", {}) or {})
    )
    for variable_name, expression in sorted(
        _scalar_hierarchy_base_seed_expressions(
            initial_mode,
            gauge=gauge,
        ).items()
    ):
        if variable_name not in variables or variable_name in {
            "theta_gamma2",
            "e_gamma2",
        }:
            continue
        initial_conditions.setdefault(
            f"{variable_name}_seed",
            {
                "target": {
                    "variable": variable_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": expression,
            },
        )
    initial_conditions.setdefault(
        "theta_gamma2_seed",
        {
            "target": {
                "variable": "theta_gamma2",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "(8.0 / 15.0) * acoustic_k * theta_gamma1 / " "collision_rate"
            ),
        },
    )
    initial_conditions.setdefault(
        "e_gamma2_seed",
        {
            "target": {
                "variable": "e_gamma2",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "theta_gamma2 / 4.0",
        },
    )
    if metric_evolution_state_name is not None:
        initial_conditions[f"{metric_evolution_state_name}_seed"] = {
            "target": {
                "variable": metric_evolution_state_name,
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                f"({_scalar_metric_seed_amplitude(initial_mode)}) + "
                "metric_shear_correction"
            ),
        }
    if sync_gauge:
        initial_conditions["eta_sync_metric_seed"] = {
            "target": {
                "variable": "eta_sync_metric",
                "wrt": "tau",
                "order": 0,
            },
            "expression": ("Phi_gi + Hconf * gauge_shift_alpha"),
        }
        initial_conditions["h_sync_metric_seed"] = {
            "target": {
                "variable": "h_sync_metric",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "0.5 * (acoustic_k * scalar_initial_conformal_time) * "
                "(acoustic_k * scalar_initial_conformal_time) * seed"
            ),
        }
    required_initial_names = [
        "theta_gamma0",
        "theta_gamma1",
        "theta_gamma2",
        "e_gamma0",
        "e_gamma1",
        "e_gamma2",
        "polarization_b_mode_seed",
        "delta_b",
        "theta_b",
    ]
    if has_cdm:
        required_initial_names.extend(("delta_c", "theta_c"))
    if has_massless_neutrino:
        required_initial_names.extend(("delta_nu", "theta_nu", "sigma_nu"))
    if sync_gauge:
        required_initial_names.extend(
            ("h_sync_metric", "eta_sync_metric", "gauge_shift_alpha")
        )
    for required_name in required_initial_names:
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
        temperature_expression = "0.0"
        polarization_expression = "0.0"
        if moment == 3:
            temperature_expression = (
                "(3.0 / 7.0) * acoustic_k * theta_gamma2 / collision_rate"
            )
            polarization_expression = (
                "(3.0 / 28.0) * acoustic_k * theta_gamma2 / collision_rate"
            )
        initial_conditions[f"{_scalar_temperature_name(moment)}_seed"] = {
            "target": {
                "variable": _scalar_temperature_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": temperature_expression,
        }
        initial_conditions[f"{_scalar_polarization_name(moment)}_seed"] = {
            "target": {
                "variable": _scalar_polarization_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": polarization_expression,
        }
    if has_massless_neutrino:
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
            if initial_mode == "adiabatic_scalar":
                q_delta_expression = (
                    f"0.5 * scalar_lapse_seed * {q_log_derivative_name}"
                )
                q_theta_expression = (
                    "-(acoustic_k * scalar_initial_conformal_time / 8.0) "
                    "* scalar_lapse_seed * "
                    f"{q_log_derivative_name} * "
                    f"{q_streaming_speed_name}"
                )
                q_sigma_expression = (
                    "-(acoustic_k * scalar_initial_conformal_time) * "
                    "(acoustic_k * scalar_initial_conformal_time) * "
                    "scalar_lapse_seed / 60.0 * "
                    f"{q_log_derivative_name}"
                )
            else:
                q_delta_expression = mode_seed_expressions.get(
                    "delta_nu_massive",
                    "0.0",
                )
                q_theta_expression = mode_seed_expressions.get(
                    "theta_nu_massive",
                    "0.0",
                )
                q_sigma_expression = mode_seed_expressions.get(
                    "sigma_nu_massive",
                    "0.0",
                )
            initial_conditions[f"{q_delta_name}_seed"] = {
                "target": {
                    "variable": q_delta_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": q_delta_expression,
            }
            initial_conditions[f"{q_theta_name}_seed"] = {
                "target": {
                    "variable": q_theta_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": q_theta_expression,
            }
            initial_conditions[f"{q_sigma_name}_seed"] = {
                "target": {
                    "variable": q_sigma_name,
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": q_sigma_expression,
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


def _materialize_declared_vector_hierarchy_contract(
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    """Return a generated vector hierarchy contract when metadata is enough."""

    if _has_explicit_declared_runtime_graph(contract):
        return contract, False

    sectors = contract.get("sectors", {}) or {}
    species = contract.get("species", {}) or {}
    hierarchy_families = contract.get("hierarchy_families", {}) or {}
    initial_condition_families = (
        contract.get("initial_condition_families", {}) or {}
    )
    if not isinstance(sectors, Mapping):
        return contract, False
    if not isinstance(species, Mapping):
        return contract, False
    if not isinstance(hierarchy_families, Mapping):
        return contract, False
    if not isinstance(initial_condition_families, Mapping):
        return contract, False
    if {str(name) for name in sectors} != {_VECTOR_HIERARCHY_REQUIRED_SECTOR}:
        return contract, False
    if _VECTOR_HIERARCHY_REQUIRED_SECTOR not in sectors:
        return contract, False
    if not _VECTOR_HIERARCHY_REQUIRED_SPECIES.issubset(species):
        return contract, False
    if not _VECTOR_HIERARCHY_REQUIRED_FAMILIES.issubset(hierarchy_families):
        return contract, False
    initial_mode = _select_standard_vector_initial_mode(
        initial_condition_families
    )
    if initial_mode is None:
        return contract, False

    gauge = str(contract.get("gauge") or "conformal_newtonian")
    if gauge == "unspecified":
        gauge = "conformal_newtonian"
    if gauge != "conformal_newtonian":
        return contract, False

    numerics = contract.get("numerics", {}) or {}
    if not isinstance(numerics, Mapping):
        numerics = {}
    photon_default_l_max = hierarchy_families["photon_temperature_vector"].get(
        "default_l_max", 6
    )
    polarization_default_l_max = max(
        hierarchy_families["photon_polarization_e_vector"].get(
            "default_l_max",
            photon_default_l_max,
        ),
        hierarchy_families["photon_polarization_b_vector"].get(
            "default_l_max",
            photon_default_l_max,
        ),
    )
    neutrino_default_l_max = hierarchy_families[
        "massless_neutrino_vector"
    ].get("default_l_max", 4)
    photon_l_max = max(
        3,
        int(numerics.get("photon_hierarchy_l_max", photon_default_l_max)),
    )
    polarization_l_max = max(
        2,
        int(
            numerics.get(
                "photon_polarization_hierarchy_l_max",
                max(photon_l_max, polarization_default_l_max),
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
    has_cdm = "cdm" in species

    materialized = copy.deepcopy(dict(contract))
    materialized["gauge"] = gauge
    vector_sector = dict(
        (materialized.get("sectors", {}) or {}).get("vector", {}) or {}
    )
    supported_gauges = list(vector_sector.get("supported_gauges", []) or [])
    if gauge not in supported_gauges:
        supported_gauges.append(gauge)
    vector_sector["supported_gauges"] = supported_gauges
    materialized["sectors"] = dict(materialized.get("sectors", {}) or {})
    materialized["sectors"]["vector"] = vector_sector

    variables: dict[str, Any] = {
        "sigma_vector": _metadata_entry(
            "vector_metric_shear",
            "Vector metric shear amplitude sigma.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "v_b_vector": _metadata_entry(
            "baryon_vector_vorticity",
            "Baryon vector-vorticity amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "q_gamma_vector": _metadata_entry(
            "photon_vector_heat_flux",
            "Photon vector heat-flux amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "pi_gamma_vector": _metadata_entry(
            "photon_vector_anisotropic_stress",
            "Photon vector anisotropic-stress amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "vector_polarization_moment": _metadata_entry(
            "photon_vector_source_moment",
            "Vector polarization source moment.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "vector_visibility_polarization_moment": _metadata_entry(
            "photon_vector_visibility_weighted_source_moment",
            "Visibility-weighted vector polarization source moment.",
            units=_INVERSE_MPC_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "q_nu_vector": _metadata_entry(
            "massless_neutrino_vector_heat_flux",
            "Massless-neutrino vector heat-flux amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
        "pi_nu_vector": _metadata_entry(
            "massless_neutrino_vector_anisotropic_stress",
            "Massless-neutrino vector anisotropic stress amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        ),
    }
    if has_cdm:
        variables["v_c_vector"] = _metadata_entry(
            "cdm_vector_vorticity",
            "Cold-dark-matter vector-vorticity amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        )
    for moment in range(3, photon_l_max + 1):
        variables[_vector_temperature_name(moment)] = _metadata_entry(
            "photon_vector_temperature_multipole",
            f"Photon vector temperature multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        )
    for moment in range(2, polarization_l_max + 1):
        variables[_vector_polarization_e_name(moment)] = _metadata_entry(
            "photon_vector_polarization_e_multipole",
            f"Photon vector E-polarization multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=2.0,
        )
        variables[_vector_polarization_b_name(moment)] = _metadata_entry(
            "photon_vector_polarization_b_multipole",
            f"Photon vector B-polarization multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="odd",
            projection_role="b_mode",
            spin=2.0,
        )
    for moment in range(3, neutrino_l_max + 1):
        variables[_vector_neutrino_name(moment)] = _metadata_entry(
            "massless_neutrino_vector_multipole",
            f"Massless-neutrino vector multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="vector_like",
            parity="even",
            spin=1.0,
        )
    materialized["variables"] = variables

    vector_momentum_source_expression = (
        "(Omega_b0 * v_b_vector) / a + "
        "(Omega_gamma0 * q_gamma_vector + "
        "vector_neutrino_density * q_nu_vector) / (a * a)"
    )
    vector_matter_density_expression = "Omega_b0"
    if has_cdm:
        vector_matter_density_expression = "Omega_b0 + Omega_c0"
        vector_momentum_source_expression = (
            "(Omega_b0 * v_b_vector + Omega_c0 * v_c_vector) / a + "
            "(Omega_gamma0 * q_gamma_vector + "
            "vector_neutrino_density * q_nu_vector) / (a * a)"
        )
    vector_cdm_fraction_expression = (
        f"Omega_c0 / ({vector_matter_density_expression})"
        if has_cdm
        else "0.0"
    )
    vector_initial_conformal_time_expression = (
        "a_initial / (H0_over_c_Mpc_inv * sqrt("
        "Omega_gamma0 + vector_neutrino_density + 1.0e-30))"
    )
    vector_initial_matter_loading_expression = (
        f"a_initial * ({vector_matter_density_expression}) / "
        "(Omega_gamma0 + vector_neutrino_density)"
    )
    vector_heat_flux_regular_correction_expression = (
        "1.0 - 0.75 * "
        f"({vector_initial_matter_loading_expression}) * "
        "(vector_cdm_matter_fraction - 1.0) / "
        "(vector_neutrino_fraction - 1.0) * (1.0 - 0.25 * "
        f"({vector_initial_matter_loading_expression}) * "
        "(3.0 * vector_cdm_matter_fraction - 2.0 - "
        "vector_neutrino_fraction) / "
        "(vector_neutrino_fraction - 1.0))"
    )
    derived_entries: dict[str, Any] = {
        "acoustic_k": {
            "expression": "k",
            "description": "Vector acoustic wave number.",
            "units": _INVERSE_MPC_UNITS,
        },
        "acoustic_k_sq": {
            "expression": "acoustic_k * acoustic_k",
            "description": "Squared vector acoustic wave number.",
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "vector_eta_safe": {
            "expression": "sqrt(eta * eta + 1.0e-24)",
            "description": "Regularized conformal time for vector closures.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_radial_argument": {
            "expression": "acoustic_k * chi",
            "description": "Vector line-of-sight radial argument x = k chi.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_radial_argument_safe": {
            "expression": (
                "sqrt(vector_radial_argument * vector_radial_argument + "
                "1.0e-24)"
            ),
            "description": "Regularized radial argument for vector sources.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_neutrino_density": {
            "expression": "Omega_nu0",
            "description": "Massless-neutrino radiation density today.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_neutrino_fraction": {
            "expression": "Omega_nu0 / (Omega_nu0 + Omega_gamma0)",
            "description": "Massless-neutrino fraction in the radiation bath.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_photon_fraction": {
            "expression": "Omega_gamma0 / (Omega_nu0 + Omega_gamma0)",
            "description": "Photon fraction in the radiation bath.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_photon_baryon_loading": {
            "expression": "Omega_gamma0 / (Omega_b0 * a)",
            "description": "Photon-to-baryon vector momentum-loading ratio.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_initial_conformal_time": {
            "expression": vector_initial_conformal_time_expression,
            "description": (
                "Radiation-era conformal-time estimate used by vector "
                "initial conditions."
            ),
            "units": "Mpc",
        },
        "vector_initial_matter_loading": {
            "expression": vector_initial_matter_loading_expression,
            "description": (
                "Radiation-era matter loading used by vector initial "
                "conditions."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_cdm_matter_fraction": {
            "expression": vector_cdm_fraction_expression,
            "description": (
                "Cold-dark-matter share of the pressureless matter sector."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_heat_flux_regular_correction": {
            "expression": vector_heat_flux_regular_correction_expression,
            "description": (
                "Regular vector-mode correction applied to the photon and "
                "baryon heat flux seeds."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "baryon_vector_thomson_drag": {
            "expression": (
                "-vector_photon_baryon_loading * thomson_vector_drag"
            ),
            "description": "Baryon counterpart of the vector Thomson drag.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_polarization_moment_tau": {
            "kind": "vector_source_time_derivative",
            "variable": "vector_polarization_moment",
            "wrt": "tau",
            "order": 1,
            "description": (
                "History-derived conformal-time derivative of the vector "
                "polarization source moment."
            ),
            "units": _INVERSE_MPC_UNITS,
        },
        "vector_visibility_polarization_moment_tau": {
            "kind": "vector_source_time_derivative",
            "variable": "vector_visibility_polarization_moment",
            "wrt": "tau",
            "order": 1,
            "description": (
                "History-derived conformal-time derivative of the "
                "visibility-weighted vector polarization source moment."
            ),
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "vector_total_momentum_source": {
            "expression": vector_momentum_source_expression,
            "description": (
                "Total vector momentum source entering the Einstein system."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_total_shear_source": {
            "expression": (
                "(Omega_gamma0 * pi_gamma_vector + "
                "vector_neutrino_density * pi_nu_vector) / (a * a)"
            ),
            "description": (
                "Total vector anisotropic-stress source for the Einstein "
                "system."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "einstein_gravity_strength": {
            "expression": "H0_over_c_Mpc_inv * H0_over_c_Mpc_inv",
            "description": "Background gravity scale used by vector modes.",
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "vector_sigma_constraint": {
            "expression": (
                "6.0 * einstein_gravity_strength * "
                "vector_total_momentum_source / acoustic_k_sq"
            ),
            "description": "Vector Einstein momentum-constraint solution.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "vector_metric_shear_rhs": {
            "expression": (
                "-2.0 * Hconf * sigma_vector - "
                "3.0 * einstein_gravity_strength * "
                "vector_total_shear_source / acoustic_k"
            ),
            "description": "Vector metric-shear evolution RHS.",
            "units": _INVERSE_MPC_UNITS,
        },
        "vector_einstein_momentum_residual": {
            "expression": "sigma_vector - vector_sigma_constraint",
            "description": "Vector Einstein momentum-constraint residual.",
            "units": _DIMENSIONLESS_UNITS,
        },
    }
    materialized["derived"] = derived_entries

    equations: dict[str, Any] = {
        "evolve_sigma_vector": {
            "lhs": {
                "kind": "derivative",
                "variable": "sigma_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "vector_metric_shear_rhs",
            "role": "metric",
        },
        "evolve_v_b_vector": {
            "lhs": {
                "kind": "derivative",
                "variable": "v_b_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-Hconf * v_b_vector + baryon_vector_thomson_drag",
            "role": "vector_euler",
        },
        "evolve_q_gamma_vector": {
            "lhs": {
                "kind": "derivative",
                "variable": "q_gamma_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-0.5 * acoustic_k * pi_gamma_vector + " "thomson_vector_drag"
            ),
            "role": "vector_hierarchy",
        },
        "evolve_pi_gamma_vector": {
            "lhs": {
                "kind": "derivative",
                "variable": "pi_gamma_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "(2.0 / 5.0) * acoustic_k * q_gamma_vector - "
                "(8.0 / 15.0) * acoustic_k * theta_gamma_v3 + "
                "(8.0 / 15.0) * acoustic_k * sigma_vector - "
                "vector_quadrupole_collision"
            ),
            "role": "vector_hierarchy",
        },
        "evolve_q_nu_vector": {
            "lhs": {
                "kind": "derivative",
                "variable": "q_nu_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-0.5 * acoustic_k * pi_nu_vector",
            "role": "vector_hierarchy",
        },
        "evolve_pi_nu_vector": {
            "lhs": {
                "kind": "derivative",
                "variable": "pi_nu_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "(2.0 / 5.0) * acoustic_k * q_nu_vector - "
                "(8.0 / 15.0) * acoustic_k * nu_v3 + "
                "(8.0 / 15.0) * acoustic_k * sigma_vector"
            ),
            "role": "vector_hierarchy",
        },
        "evolve_e_gamma_v2": {
            "lhs": {
                "kind": "derivative",
                "variable": "e_gamma_v2",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-(8.0 / 27.0) * acoustic_k * e_gamma_v3 + "
                "(1.0 / 3.0) * acoustic_k * b_gamma_v2 - "
                "vector_e_quadrupole_collision"
            ),
            "role": "vector_polarization",
        },
        "evolve_b_gamma_v2": {
            "lhs": {
                "kind": "derivative",
                "variable": "b_gamma_v2",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-(8.0 / 27.0) * acoustic_k * b_gamma_v3 - "
                "(1.0 / 3.0) * acoustic_k * e_gamma_v2 - "
                "collision_rate * b_gamma_v2"
            ),
            "role": "vector_polarization_b",
        },
    }
    if has_cdm:
        equations["evolve_v_c_vector"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "v_c_vector",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-Hconf * v_c_vector",
            "role": "vector_euler",
        }
    for moment in range(3, photon_l_max + 1):
        name = _vector_temperature_name(moment)
        next_name = None
        if moment < photon_l_max:
            next_name = _vector_temperature_name(moment + 1)
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _vector_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=(
                    "pi_gamma_vector"
                    if moment == 3
                    else _vector_temperature_name(moment - 1)
                ),
                next_name=next_name,
                collision_term=f"- collision_rate * {name}",
            ),
            "role": "vector_hierarchy",
        }
    for moment in range(3, polarization_l_max + 1):
        e_name = _vector_polarization_e_name(moment)
        b_name = _vector_polarization_b_name(moment)
        e_next_name = None
        b_next_name = None
        if moment < polarization_l_max:
            e_next_name = _vector_polarization_e_name(moment + 1)
            b_next_name = _vector_polarization_b_name(moment + 1)
        equations[f"evolve_{e_name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": e_name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _vector_polarization_recurrence_rhs(
                name=e_name,
                moment=moment,
                previous_name=_vector_polarization_e_name(moment - 1),
                next_name=e_next_name,
                opposite_name=b_name,
                sign=1,
                collision_term=f"- collision_rate * {e_name}",
            ),
            "role": "vector_polarization",
        }
        equations[f"evolve_{b_name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": b_name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _vector_polarization_recurrence_rhs(
                name=b_name,
                moment=moment,
                previous_name=_vector_polarization_b_name(moment - 1),
                next_name=b_next_name,
                opposite_name=e_name,
                sign=-1,
                collision_term=f"- collision_rate * {b_name}",
            ),
            "role": "vector_polarization_b",
        }
    for moment in range(3, neutrino_l_max + 1):
        name = _vector_neutrino_name(moment)
        next_name = None
        if moment < neutrino_l_max:
            next_name = _vector_neutrino_name(moment + 1)
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _vector_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=(
                    "pi_nu_vector"
                    if moment == 3
                    else _vector_neutrino_name(moment - 1)
                ),
                next_name=next_name,
            ),
            "role": "vector_hierarchy",
        }
    materialized["equations"] = equations

    collision_operator_entries = dict(
        materialized.get("collision_operators", {}) or {}
    )
    collision_operator_entries["thomson_vector_drag"] = {
        "sector": "vector",
        "species": ["photon", "baryon"],
        "expression": (
            "collision_rate * ((4.0 / 3.0) * v_b_vector - q_gamma_vector)"
        ),
        "counterpart": "baryon_vector_thomson_drag",
        "integration_strategy": "implicit",
        "activation_strategy": "tight_coupling",
        "rate_expression": "collision_rate",
        "linear_block": {
            "targets": [
                {"variable": "q_gamma_vector"},
                {"variable": "v_b_vector"},
            ],
            "matrix": [
                ["-1.0", "4.0 / 3.0"],
                [
                    "vector_photon_baryon_loading",
                    "-(4.0 / 3.0) * vector_photon_baryon_loading",
                ],
            ],
            "activation_strategy": "tight_coupling",
        },
    }
    collision_operator_entries["vector_quadrupole_collision"] = {
        "sector": "vector",
        "species": ["photon"],
        "expression": (
            "collision_rate * (pi_gamma_vector - "
            "vector_polarization_moment)"
        ),
    }
    collision_operator_entries["vector_e_quadrupole_collision"] = {
        "sector": "vector",
        "species": ["photon"],
        "expression": (
            "collision_rate * (e_gamma_v2 - vector_polarization_moment)"
        ),
    }
    materialized["collision_operators"] = collision_operator_entries
    conservation_rule_entries = dict(
        materialized.get("conservation_rules", {}) or {}
    )
    conservation_rule_entries["thomson_vector_drag_balance"] = {
        "kind": "absolute_max",
        "expression": (
            "vector_photon_baryon_loading * thomson_vector_drag + "
            "baryon_vector_thomson_drag"
        ),
        "tolerance": 1.0e-12,
        "domain": "vector",
    }
    materialized["conservation_rules"] = conservation_rule_entries
    materialized["constraints"] = {}
    materialized["closures"] = {
        "vector_polarization_moment_closure": {
            "target": "vector_polarization_moment",
            "expression": "0.1 * pi_gamma_vector + 0.6 * e_gamma_v2",
            "role": "closure",
            "description": "Vector polarization source moment.",
        },
        "vector_visibility_polarization_moment_closure": {
            "target": "vector_visibility_polarization_moment",
            "expression": "visibility * vector_polarization_moment",
            "role": "closure",
            "description": (
                "Visibility-weighted vector polarization source moment."
            ),
        },
    }

    materialized["sources"] = {
        "vector_temperature_source": {
            "expression": (
                "("
                "4.0 * (v_b_vector + sigma_vector) * visibility + "
                "7.5 * vector_visibility_polarization_moment_tau / "
                "acoustic_k + "
                "4.0 * exp(-tau) * vector_metric_shear_rhs"
                ")"
            ),
            "role": "signal",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "vector_polarization_e_source": {
            "expression": (
                "15.0 * visibility * vector_polarization_moment + "
                "7.5 * vector_visibility_polarization_moment_tau / "
                "acoustic_k"
            ),
            "role": "signal",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "vector_polarization_b_source": {
            "expression": ("-7.5 * visibility * vector_polarization_moment"),
            "role": "signal",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
    }
    materialized["observables"] = {
        "temperature": {
            "kind": "transfer_component",
            "projection": "line_of_sight_vector_temperature",
            "source_terms": {"signal": "vector_temperature_source"},
            "description": "Vector temperature transfer function.",
        },
        "polarization_e": {
            "kind": "transfer_component",
            "projection": "line_of_sight_vector_polarization_e",
            "source_terms": {"signal": "vector_polarization_e_source"},
            "description": "Vector E-polarization transfer function.",
        },
        "polarization_b": {
            "kind": "transfer_component",
            "projection": "line_of_sight_vector_polarization_b",
            "source_terms": {"signal": "vector_polarization_b_source"},
            "description": "Vector B-polarization transfer function.",
        },
        "TT": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "temperature",
            "description": "Vector temperature auto spectrum.",
            "notes": "Public solver returns D_ell^TT in muK^2.",
        },
        "TE": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "polarization_e",
            "description": "Vector temperature and E-mode cross spectrum.",
            "notes": "Public solver returns D_ell^TE in muK^2.",
        },
        "EE": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "polarization_e",
            "description": "Vector E-polarization auto spectrum.",
            "notes": "Public solver returns D_ell^EE in muK^2.",
        },
        "BB": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
            "description": "Vector B-polarization auto spectrum.",
            "notes": "Public solver returns D_ell^BB in muK^2.",
        },
    }

    initial_conditions: dict[str, Any] = {
        "sigma_vector_seed": {
            "target": {
                "variable": "sigma_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "seed * (1.0 - 7.5 * vector_initial_matter_loading / "
                "(4.0 * vector_neutrino_fraction + 15.0))"
            ),
        },
        "v_b_vector_seed": {
            "target": {
                "variable": "v_b_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "0.25 * seed * (4.0 * vector_neutrino_fraction + 5.0) / "
                "vector_photon_fraction * "
                "vector_heat_flux_regular_correction"
            ),
        },
        "q_gamma_vector_seed": {
            "target": {
                "variable": "q_gamma_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "(seed / 3.0) * "
                "(4.0 * vector_neutrino_fraction + 5.0) / "
                "vector_photon_fraction * "
                "vector_heat_flux_regular_correction"
            ),
        },
        "pi_gamma_vector_seed": {
            "target": {
                "variable": "pi_gamma_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        },
        "q_nu_vector_seed": {
            "target": {
                "variable": "q_nu_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "-(seed / 3.0) * "
                "(4.0 * vector_neutrino_fraction + 5.0) / "
                "vector_neutrino_fraction + "
                "(acoustic_k * vector_initial_conformal_time) * "
                "(acoustic_k * vector_initial_conformal_time) * seed / "
                "6.0 / "
                "vector_neutrino_fraction"
            ),
        },
        "pi_nu_vector_seed": {
            "target": {
                "variable": "pi_nu_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "-(2.0 / 3.0) * acoustic_k * "
                "vector_initial_conformal_time * seed / "
                "vector_neutrino_fraction"
            ),
        },
    }
    if has_cdm:
        initial_conditions["v_c_vector_seed"] = {
            "target": {
                "variable": "v_c_vector",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    for moment in range(3, photon_l_max + 1):
        initial_conditions[f"{_vector_temperature_name(moment)}_seed"] = {
            "target": {
                "variable": _vector_temperature_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    for moment in range(2, polarization_l_max + 1):
        initial_conditions[f"{_vector_polarization_e_name(moment)}_seed"] = {
            "target": {
                "variable": _vector_polarization_e_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
        initial_conditions[f"{_vector_polarization_b_name(moment)}_seed"] = {
            "target": {
                "variable": _vector_polarization_b_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    for moment in range(3, neutrino_l_max + 1):
        initial_conditions[f"{_vector_neutrino_name(moment)}_seed"] = {
            "target": {
                "variable": _vector_neutrino_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    materialized["initial_conditions"] = initial_conditions
    family_entries = copy.deepcopy(dict(initial_condition_families))
    family_entries[initial_mode]["members"] = list(sorted(initial_conditions))
    materialized["initial_condition_families"] = family_entries
    materialized["boundary_conditions"] = {}
    return materialized, True


def _materialize_declared_tensor_hierarchy_contract(
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    """Return a generated tensor hierarchy contract when metadata is enough."""

    if _has_explicit_declared_runtime_graph(contract):
        return contract, False

    sectors = contract.get("sectors", {}) or {}
    species = contract.get("species", {}) or {}
    hierarchy_families = contract.get("hierarchy_families", {}) or {}
    initial_condition_families = (
        contract.get("initial_condition_families", {}) or {}
    )
    if not isinstance(sectors, Mapping):
        return contract, False
    if not isinstance(species, Mapping):
        return contract, False
    if not isinstance(hierarchy_families, Mapping):
        return contract, False
    if not isinstance(initial_condition_families, Mapping):
        return contract, False
    if {str(name) for name in sectors} != {_TENSOR_HIERARCHY_REQUIRED_SECTOR}:
        return contract, False
    if _TENSOR_HIERARCHY_REQUIRED_SECTOR not in sectors:
        return contract, False
    if not _TENSOR_HIERARCHY_REQUIRED_SPECIES.issubset(species):
        return contract, False
    if not _TENSOR_HIERARCHY_REQUIRED_FAMILIES.issubset(hierarchy_families):
        return contract, False
    initial_mode = _select_standard_tensor_initial_mode(
        initial_condition_families
    )
    if initial_mode is None:
        return contract, False

    gauge = str(contract.get("gauge") or "conformal_newtonian")
    if gauge == "unspecified":
        gauge = "conformal_newtonian"
    if gauge != "conformal_newtonian":
        return contract, False

    numerics = contract.get("numerics", {}) or {}
    if not isinstance(numerics, Mapping):
        numerics = {}
    photon_default_l_max = hierarchy_families["photon_temperature_tensor"].get(
        "default_l_max",
        6,
    )
    polarization_default_l_max = max(
        hierarchy_families["photon_polarization_e_tensor"].get(
            "default_l_max",
            photon_default_l_max,
        ),
        hierarchy_families["photon_polarization_b_tensor"].get(
            "default_l_max",
            photon_default_l_max,
        ),
    )
    neutrino_default_l_max = hierarchy_families[
        "massless_neutrino_tensor"
    ].get("default_l_max", 4)
    photon_l_max = max(
        3,
        int(numerics.get("photon_hierarchy_l_max", photon_default_l_max)),
    )
    polarization_l_max = max(
        2,
        int(
            numerics.get(
                "photon_polarization_hierarchy_l_max",
                max(photon_l_max, polarization_default_l_max),
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

    materialized = copy.deepcopy(dict(contract))
    materialized["gauge"] = gauge
    tensor_sector = dict(
        (materialized.get("sectors", {}) or {}).get("tensor", {}) or {}
    )
    supported_gauges = list(tensor_sector.get("supported_gauges", []) or [])
    if gauge not in supported_gauges:
        supported_gauges.append(gauge)
    tensor_sector["supported_gauges"] = supported_gauges
    materialized["sectors"] = dict(materialized.get("sectors", {}) or {})
    materialized["sectors"]["tensor"] = tensor_sector

    variables: dict[str, Any] = {
        "h_tensor": _metadata_entry(
            "tensor_metric_wave",
            "Tensor metric-wave amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        ),
        "h_tensor_tau": _metadata_entry(
            "tensor_metric_wave_derivative",
            "Conformal-time derivative of the tensor metric wave.",
            units=_INVERSE_MPC_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        ),
        "pi_gamma_tensor": _metadata_entry(
            "photon_tensor_anisotropic_stress",
            "Photon tensor anisotropic-stress amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        ),
        "pi_nu_tensor": _metadata_entry(
            "massless_neutrino_tensor_anisotropic_stress",
            "Massless-neutrino tensor anisotropic-stress amplitude.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        ),
    }
    for moment in range(3, photon_l_max + 1):
        variables[_tensor_temperature_name(moment)] = _metadata_entry(
            "photon_tensor_temperature_multipole",
            f"Photon tensor temperature multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        )
    for moment in range(2, polarization_l_max + 1):
        variables[_tensor_polarization_e_name(moment)] = _metadata_entry(
            "photon_tensor_polarization_e_multipole",
            f"Photon tensor E-polarization multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        )
    for moment in range(2, polarization_l_max + 1):
        variables[_tensor_polarization_b_name(moment)] = _metadata_entry(
            "photon_tensor_polarization_b_multipole",
            f"Photon tensor B-polarization multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="odd",
            projection_role="b_mode",
            spin=2.0,
        )
    for moment in range(3, neutrino_l_max + 1):
        variables[_tensor_neutrino_name(moment)] = _metadata_entry(
            "massless_neutrino_tensor_multipole",
            f"Massless-neutrino tensor multipole at l={int(moment)}.",
            units=_DIMENSIONLESS_UNITS,
            tensor_character="tensor_like",
            parity="even",
            spin=2.0,
        )
    materialized["variables"] = variables

    derived_entries: dict[str, Any] = {
        "acoustic_k": {
            "expression": "k",
            "description": "Tensor acoustic wave number.",
            "units": _INVERSE_MPC_UNITS,
        },
        "acoustic_k_sq": {
            "expression": "acoustic_k * acoustic_k",
            "description": "Squared tensor acoustic wave number.",
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "tensor_eta_safe": {
            "expression": "sqrt(eta * eta + 1.0e-24)",
            "description": "Regularized conformal time for tensor closures.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "tensor_shear": {
            "expression": "-h_tensor_tau / (acoustic_k + 1.0e-30)",
            "description": "Tensor shear obtained from the metric derivative.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "tensor_neutrino_density": {
            "expression": "Omega_nu0",
            "description": "Massless-neutrino tensor density today.",
            "units": _DIMENSIONLESS_UNITS,
        },
        "tensor_free_streaming_fraction": {
            "expression": ("Omega_nu0 / (Omega_gamma0 + Omega_nu0 + 1.0e-30)"),
            "description": (
                "Radiation-era free-streaming fraction used by tensor "
                "initial conditions."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "tensor_initial_series_denominator": {
            "expression": ("15.0 + 4.0 * tensor_free_streaming_fraction"),
            "description": (
                "Regular tensor initial-series denominator including "
                "free-streaming neutrino stress."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "tensor_polarization_moment": {
            "expression": "0.1 * pi_gamma_tensor + 0.6 * e_gamma_t2",
            "description": (
                "Tensor polarization source moment from the photon "
                "temperature and polarization hierarchy."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "tensor_total_shear_source": {
            "expression": (
                "Omega_gamma0 * pi_gamma_tensor + "
                "tensor_neutrino_density * pi_nu_tensor"
            ),
            "description": (
                "Total tensor anisotropic-stress source from photon and "
                "massless-neutrino moments."
            ),
            "units": _DIMENSIONLESS_UNITS,
        },
        "einstein_gravity_strength": {
            "expression": "H0_over_c_Mpc_inv * H0_over_c_Mpc_inv",
            "description": "Background gravity scale used by tensor modes.",
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
        "tensor_metric_wave_rhs": {
            "expression": (
                "-2.0 * Hconf * h_tensor_tau - acoustic_k_sq * h_tensor + "
                "3.0 * einstein_gravity_strength * "
                "tensor_total_shear_source / (a * a)"
            ),
            "description": "Tensor metric-wave evolution RHS.",
            "units": _INVERSE_MPC_SQUARED_UNITS,
        },
    }
    materialized["derived"] = derived_entries

    equations: dict[str, Any] = {
        "evolve_h_tensor": {
            "lhs": {
                "kind": "derivative",
                "variable": "h_tensor",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "h_tensor_tau",
            "role": "metric",
        },
        "evolve_h_tensor_tau": {
            "lhs": {
                "kind": "derivative",
                "variable": "h_tensor_tau",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "tensor_metric_wave_rhs",
            "role": "metric",
        },
        "evolve_pi_gamma_tensor": {
            "lhs": {
                "kind": "derivative",
                "variable": "pi_gamma_tensor",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-(1.0 / 3.0) * acoustic_k * theta_gamma_t3 + "
                "(8.0 / 15.0) * acoustic_k * tensor_shear"
            ),
            "role": "tensor_hierarchy",
        },
        "evolve_pi_nu_tensor": {
            "lhs": {
                "kind": "derivative",
                "variable": "pi_nu_tensor",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-(1.0 / 3.0) * acoustic_k * nu_t3 + "
                "(8.0 / 15.0) * acoustic_k * tensor_shear"
            ),
            "role": "tensor_hierarchy",
        },
        "evolve_e_gamma_t2": {
            "lhs": {
                "kind": "derivative",
                "variable": "e_gamma_t2",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-(5.0 / 27.0) * acoustic_k * e_gamma_t3 + "
                "(2.0 / 3.0) * acoustic_k * b_gamma_t2"
            ),
            "role": "tensor_polarization",
        },
        "evolve_b_gamma_t2": {
            "lhs": {
                "kind": "derivative",
                "variable": "b_gamma_t2",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": (
                "-(5.0 / 27.0) * acoustic_k * b_gamma_t3 - "
                "(2.0 / 3.0) * acoustic_k * e_gamma_t2"
            ),
            "role": "tensor_polarization_b",
        },
    }
    for moment in range(3, photon_l_max + 1):
        name = _tensor_temperature_name(moment)
        next_name = None
        if moment < photon_l_max:
            next_name = _tensor_temperature_name(moment + 1)
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _tensor_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=(
                    "pi_gamma_tensor"
                    if moment == 3
                    else _tensor_temperature_name(moment - 1)
                ),
                next_name=next_name,
            ),
            "role": "tensor_hierarchy",
        }
    for moment in range(3, polarization_l_max + 1):
        e_name = _tensor_polarization_e_name(moment)
        b_name = _tensor_polarization_b_name(moment)
        e_next_name = None
        b_next_name = None
        if moment < polarization_l_max:
            e_next_name = _tensor_polarization_e_name(moment + 1)
            b_next_name = _tensor_polarization_b_name(moment + 1)
        equations[f"evolve_{e_name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": e_name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _tensor_polarization_recurrence_rhs(
                name=e_name,
                moment=moment,
                previous_name=_tensor_polarization_e_name(moment - 1),
                next_name=e_next_name,
                opposite_name=b_name,
                sign=1,
            ),
            "role": "tensor_polarization",
        }
        equations[f"evolve_{b_name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": b_name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _tensor_polarization_recurrence_rhs(
                name=b_name,
                moment=moment,
                previous_name=_tensor_polarization_b_name(moment - 1),
                next_name=b_next_name,
                opposite_name=e_name,
                sign=-1,
            ),
            "role": "tensor_polarization_b",
        }
    for moment in range(3, neutrino_l_max + 1):
        name = _tensor_neutrino_name(moment)
        next_name = None
        if moment < neutrino_l_max:
            next_name = _tensor_neutrino_name(moment + 1)
        equations[f"evolve_{name}"] = {
            "lhs": {
                "kind": "derivative",
                "variable": name,
                "wrt": "tau",
                "order": 1,
            },
            "rhs": _tensor_hierarchy_recurrence_rhs(
                name=name,
                moment=moment,
                previous_name=(
                    "pi_nu_tensor"
                    if moment == 3
                    else _tensor_neutrino_name(moment - 1)
                ),
                next_name=next_name,
            ),
            "role": "tensor_hierarchy",
        }
    materialized["equations"] = equations

    collision_operator_entries = dict(
        materialized.get("collision_operators", {}) or {}
    )
    tensor_damping_targets = [
        *[
            {"variable": _tensor_temperature_name(moment)}
            for moment in range(3, photon_l_max + 1)
        ],
        *[
            {"variable": _tensor_polarization_e_name(moment)}
            for moment in range(3, polarization_l_max + 1)
        ],
        *[
            {"variable": _tensor_polarization_b_name(moment)}
            for moment in range(2, polarization_l_max + 1)
        ],
    ]
    collision_operator_entries["tensor_thomson_collision"] = {
        "sector": "tensor",
        "species": ["photon"],
        "expression": "collision_rate * tensor_polarization_moment",
        "integration_strategy": "exact",
        "activation_strategy": "always",
        "rate_expression": "collision_rate",
        "exact_form": {
            "targets": [
                {"variable": "pi_gamma_tensor"},
                {"variable": "e_gamma_t2"},
            ],
            "matrix": [
                ["-0.9", "0.6"],
                ["0.1", "-0.4"],
            ],
            "damping_targets": tensor_damping_targets,
            "damping_coefficient": "-1.0",
            "activation_strategy": "always",
        },
    }
    materialized["collision_operators"] = collision_operator_entries
    materialized["conservation_rules"] = dict(
        materialized.get("conservation_rules", {}) or {}
    )
    materialized["constraints"] = {
        "tensor_initial_metric_constraint": {
            "target": "tensor_initial_metric_residual",
            "expression": (
                "h_tensor_tau + 5.0 * acoustic_k_sq * eta * h_tensor / "
                "tensor_initial_series_denominator"
            ),
            "role": "initial_series",
            "description": (
                "Regular superhorizon tensor metric-series residual."
            ),
        },
        "tensor_initial_neutrino_constraint": {
            "target": "tensor_initial_neutrino_residual",
            "expression": (
                "pi_nu_tensor - 4.0 * acoustic_k_sq * eta * eta * "
                "h_tensor / (3.0 * tensor_initial_series_denominator)"
            ),
            "role": "initial_series",
            "description": (
                "Regular superhorizon tensor neutrino-stress residual."
            ),
        },
        "tensor_initial_collision_constraint": {
            "target": "tensor_initial_collision_residual",
            "expression": (
                "collision_rate * pi_gamma_tensor + "
                "(32.0 / 45.0) * h_tensor_tau"
            ),
            "role": "initial_series",
            "description": ("Tensor photon tight-collision initial residual."),
        },
    }
    materialized["closures"] = {}
    materialized["sources"] = {
        "tensor_temperature_source": {
            "expression": (
                "-exp(-tau) * h_tensor_tau + "
                "(15.0 / 8.0) * visibility * tensor_polarization_moment"
            ),
            "role": "signal",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "tensor_polarization_e_source": {
            "expression": (
                "(15.0 / 2.0) * sqrt(3.0 / 8.0) * visibility * "
                "tensor_polarization_moment"
            ),
            "role": "polarization",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
        "tensor_polarization_b_source": {
            "expression": (
                "(15.0 / 2.0) * sqrt(3.0 / 8.0) * visibility * "
                "tensor_polarization_moment + "
                "0.0 * visibility * b_gamma_t2"
            ),
            "role": "polarization_b",
            "units": _LINE_OF_SIGHT_SOURCE_UNITS,
        },
    }
    materialized["observables"] = {
        "temperature": {
            "kind": "transfer_component",
            "projection": "line_of_sight_signal",
            "source_terms": {"signal": "tensor_temperature_source"},
            "description": "Tensor temperature transfer function.",
        },
        "polarization_e": {
            "kind": "transfer_component",
            "projection": "spin2_e_mode",
            "source_terms": {"polarization": "tensor_polarization_e_source"},
            "description": "Tensor E-polarization transfer function.",
        },
        "polarization_b": {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {
                "polarization_b": "tensor_polarization_b_source",
            },
            "description": "Tensor B-polarization transfer function.",
        },
        "TT": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "temperature",
            "description": "Tensor temperature auto spectrum.",
            "notes": "Public solver returns D_ell^TT in muK^2.",
        },
        "TE": {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "polarization_e",
            "description": "Tensor temperature and E-mode cross spectrum.",
            "notes": "Public solver returns D_ell^TE in muK^2.",
        },
        "EE": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_e",
            "secondary": "polarization_e",
            "description": "Tensor E-polarization auto spectrum.",
            "notes": "Public solver returns D_ell^EE in muK^2.",
        },
        "BB": {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
            "description": "Tensor B-polarization auto spectrum.",
            "notes": "Public solver returns D_ell^BB in muK^2.",
        },
    }

    initial_conditions: dict[str, Any] = {
        "h_tensor_seed": {
            "target": {
                "variable": "h_tensor",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "seed",
        },
        "h_tensor_tau_seed": {
            "target": {
                "variable": "h_tensor_tau",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "-5.0 * acoustic_k_sq * eta_initial * seed / "
                "tensor_initial_series_denominator"
            ),
        },
        "pi_gamma_tensor_seed": {
            "target": {
                "variable": "pi_gamma_tensor",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "-(32.0 / 45.0) * h_tensor_tau / " "(collision_rate + 1.0e-30)"
            ),
        },
        "pi_nu_tensor_seed": {
            "target": {
                "variable": "pi_nu_tensor",
                "wrt": "tau",
                "order": 0,
            },
            "expression": (
                "4.0 * acoustic_k_sq * eta_initial * eta_initial * seed / "
                "(3.0 * tensor_initial_series_denominator)"
            ),
        },
    }
    for moment in range(3, photon_l_max + 1):
        initial_conditions[f"{_tensor_temperature_name(moment)}_seed"] = {
            "target": {
                "variable": _tensor_temperature_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    for moment in range(2, polarization_l_max + 1):
        initial_conditions[f"{_tensor_polarization_e_name(moment)}_seed"] = {
            "target": {
                "variable": _tensor_polarization_e_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": ("pi_gamma_tensor / 4.0" if moment == 2 else "0.0"),
        }
    for moment in range(2, polarization_l_max + 1):
        initial_conditions[f"{_tensor_polarization_b_name(moment)}_seed"] = {
            "target": {
                "variable": _tensor_polarization_b_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    for moment in range(3, neutrino_l_max + 1):
        initial_conditions[f"{_tensor_neutrino_name(moment)}_seed"] = {
            "target": {
                "variable": _tensor_neutrino_name(moment),
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
    materialized["initial_conditions"] = initial_conditions
    family_entries = copy.deepcopy(dict(initial_condition_families))
    family_entries[initial_mode]["members"] = list(sorted(initial_conditions))
    materialized["initial_condition_families"] = family_entries
    materialized["boundary_conditions"] = {}
    return materialized, True


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
    binding: str | None = None
    expression: str | None = None
    variable: str | None = None
    wrt: str | None = None
    order: int | None = None
    description: str | None = None
    notes: str | None = None
    domain: str | None = None
    units: str | None = None
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
    units: str | None = None
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
    units: str | None = None
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
    """Immutable hierarchy-family metadata for declared CMB contracts."""

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
class PerturbationCollisionTargetSelectorData:
    """Immutable selector for one collision-managed state target."""

    variable: str | None = None
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationCollisionLinearFormData:
    """Immutable matrix-form metadata for exact or implicit operators."""

    targets: tuple[PerturbationCollisionTargetSelectorData, ...] = ()
    matrix: tuple[tuple[str, ...], ...] = ()
    dependencies: tuple[str, ...] = ()
    compiled_matrix: tuple[
        tuple[PerturbationCompiledExpressionData, ...],
        ...,
    ] = ()
    damping_targets: tuple[PerturbationCollisionTargetSelectorData, ...] = ()
    damping_coefficient: str | None = None
    damping_dependencies: tuple[str, ...] = ()
    compiled_damping_coefficient: PerturbationCompiledExpressionData | None = (
        None
    )
    fast_manifold: bool = False
    activation_strategy: str = "always"


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
    integration_strategy: str = "explicit"
    activation_strategy: str = "always"
    rate_expression: str | None = None
    rate_dependencies: tuple[str, ...] = ()
    compiled_rate_expression: PerturbationCompiledExpressionData | None = None
    exact_form: PerturbationCollisionLinearFormData | None = None
    linear_block: PerturbationCollisionLinearFormData | None = None


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
    """Immutable projection-typing metadata for declared observables."""

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
    contract_version: int
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


@lru_cache(maxsize=4096)
def _compile_expression_code(expr: str) -> Any:
    """Return a validated Python expression code object for hot evaluation."""

    node = _parse_safe_expression(expr)
    # The expression has already passed the restricted AST validator during
    # contract compilation.  Keeping the globals closed prevents builtins or
    # imports from entering this generated evaluation path.
    return compile(node, "<declared-cmb-expression>", "eval")


def _evaluate_compiled_expression_noerr(
    expression_data: PerturbationCompiledExpressionData,
    env: Mapping[str, Any],
) -> Any:
    """Evaluate one compiled expression against ``env`` without errstate."""

    # Declared evolution evaluates the same validated expressions millions of
    # times.  Python bytecode avoids allocating a stack program for every
    # call while retaining the restricted AST and globals contract.
    # security-scanner: allow validated expression evaluation.
    return eval(  # nosec B307 - code comes from the validated AST grammar.
        _compile_expression_code(expression_data.expression),
        _COMPILED_EXPRESSION_GLOBALS,
        env,
    )


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


def _validate_derived_binding(
    value: Any,
    *,
    label: str,
) -> str | None:
    """Validate an explicit runtime binding for a derived history symbol."""

    binding = _validate_optional_string(value, label=label)
    if binding is not None and binding not in _SUPPORTED_DERIVED_BINDINGS:
        supported = ", ".join(sorted(_SUPPORTED_DERIVED_BINDINGS))
        raise ValueError(f"{label} must be one of: {supported}")
    return binding


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


def _compile_collision_target_selector(
    selector_def: Any,
    *,
    label: str,
) -> PerturbationCollisionTargetSelectorData:
    """Return one validated collision-state selector."""

    if not isinstance(selector_def, Mapping):
        raise ValueError(f"{label} must be a mapping")
    variable = _validate_optional_string(
        selector_def.get("variable"),
        label=f"{label}.variable",
    )
    kind = _validate_optional_string(
        selector_def.get("kind"),
        label=f"{label}.kind",
    )
    if (variable is None) == (kind is None):
        raise ValueError(
            f"{label} must declare exactly one of 'variable' or 'kind'"
        )
    return PerturbationCollisionTargetSelectorData(
        variable=variable,
        kind=kind,
    )


def _compile_collision_linear_form(
    linear_form_def: Any,
    *,
    label: str,
    allowed_names: set[str],
    replacements: Mapping[str, str],
    allow_damping: bool,
) -> PerturbationCollisionLinearFormData:
    """Return one validated exact-form or implicit linear block."""

    if not isinstance(linear_form_def, Mapping):
        raise ValueError(f"{label} must be a mapping")
    targets_def = linear_form_def.get("targets")
    if not isinstance(targets_def, Sequence) or isinstance(
        targets_def, (str, bytes)
    ):
        raise ValueError(f"{label}.targets must be a non-string sequence")
    targets = tuple(
        _compile_collision_target_selector(
            selector_def,
            label=f"{label}.targets[{index}]",
        )
        for index, selector_def in enumerate(targets_def)
    )
    if not targets:
        raise ValueError(f"{label}.targets must not be empty")
    matrix_def = linear_form_def.get("matrix")
    if not isinstance(matrix_def, Sequence) or isinstance(
        matrix_def, (str, bytes)
    ):
        raise ValueError(f"{label}.matrix must be a non-string sequence")
    target_count = len(targets)
    matrix_rows: list[tuple[str, ...]] = []
    compiled_rows: list[tuple[PerturbationCompiledExpressionData, ...]] = []
    dependency_names: set[str] = set()
    if len(matrix_def) != target_count:
        raise ValueError(f"{label}.matrix must have {target_count} row(s)")
    for row_index, row_def in enumerate(matrix_def):
        if not isinstance(row_def, Sequence) or isinstance(
            row_def, (str, bytes)
        ):
            raise ValueError(
                f"{label}.matrix[{row_index}] must be a non-string sequence"
            )
        if len(row_def) != target_count:
            raise ValueError(
                f"{label}.matrix[{row_index}] must have {target_count} "
                "column(s)"
            )
        clean_row: list[str] = []
        compiled_row: list[PerturbationCompiledExpressionData] = []
        for column_index, entry in enumerate(row_def):
            clean_expression, entry_dependencies = (
                _replace_and_validate_expression(
                    entry,
                    label=(f"{label}.matrix[{row_index}][{column_index}]"),
                    replacements=replacements,
                    allowed_names=allowed_names,
                )
            )
            dependency_names.update(entry_dependencies)
            clean_row.append(clean_expression)
            compiled_row.append(
                _compile_expression_plan(
                    clean_expression,
                    dependencies=entry_dependencies,
                )
            )
        matrix_rows.append(tuple(clean_row))
        compiled_rows.append(tuple(compiled_row))
    damping_targets: tuple[PerturbationCollisionTargetSelectorData, ...] = ()
    damping_coefficient = None
    damping_dependencies: tuple[str, ...] = ()
    compiled_damping_coefficient = None
    if "damping_targets" in linear_form_def:
        if not allow_damping:
            raise ValueError(
                f"{label}.damping_targets is only supported for exact forms"
            )
        damping_def = linear_form_def.get("damping_targets")
        if not isinstance(damping_def, Sequence) or isinstance(
            damping_def, (str, bytes)
        ):
            raise ValueError(
                f"{label}.damping_targets must be a non-string sequence"
            )
        damping_targets = tuple(
            _compile_collision_target_selector(
                selector_def,
                label=f"{label}.damping_targets[{index}]",
            )
            for index, selector_def in enumerate(damping_def)
        )
    if "damping_coefficient" in linear_form_def:
        if not allow_damping:
            raise ValueError(
                f"{label}.damping_coefficient is only supported for exact "
                "forms"
            )
        damping_coefficient, damping_dependency_names = (
            _replace_and_validate_expression(
                linear_form_def.get("damping_coefficient"),
                label=f"{label}.damping_coefficient",
                replacements=replacements,
                allowed_names=allowed_names,
            )
        )
        damping_dependencies = tuple(sorted(damping_dependency_names))
        compiled_damping_coefficient = _compile_expression_plan(
            damping_coefficient,
            dependencies=damping_dependencies,
        )
    activation_strategy = _validate_optional_string(
        linear_form_def.get("activation_strategy"),
        label=f"{label}.activation_strategy",
    )
    if activation_strategy is None:
        activation_strategy = "always"
    if activation_strategy not in {"always", "tight_coupling"}:
        raise ValueError(
            f"{label}.activation_strategy must be 'always' or "
            "'tight_coupling'"
        )
    fast_manifold = linear_form_def.get("fast_manifold", False)
    if not isinstance(fast_manifold, bool):
        raise ValueError(f"{label}.fast_manifold must be a boolean")
    return PerturbationCollisionLinearFormData(
        targets=targets,
        matrix=tuple(matrix_rows),
        dependencies=tuple(sorted(dependency_names)),
        compiled_matrix=tuple(compiled_rows),
        damping_targets=damping_targets,
        damping_coefficient=damping_coefficient,
        damping_dependencies=damping_dependencies,
        compiled_damping_coefficient=compiled_damping_coefficient,
        fast_manifold=fast_manifold,
        activation_strategy=activation_strategy,
    )


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
        integration_strategy = _validate_optional_string(
            operator_def.get("integration_strategy"),
            label=(
                "cmb.perturbations.collision_operators."
                f"{name}.integration_strategy"
            ),
        )
        if integration_strategy is None:
            integration_strategy = "explicit"
        if integration_strategy not in {"explicit", "exact", "implicit"}:
            raise ValueError(
                "cmb.perturbations.collision_operators."
                f"{name}.integration_strategy must be one of "
                "'explicit', 'exact', or 'implicit'"
            )
        activation_strategy = _validate_optional_string(
            operator_def.get("activation_strategy"),
            label=(
                "cmb.perturbations.collision_operators."
                f"{name}.activation_strategy"
            ),
        )
        if activation_strategy is None:
            activation_strategy = "always"
        if activation_strategy not in {"always", "tight_coupling"}:
            raise ValueError(
                "cmb.perturbations.collision_operators."
                f"{name}.activation_strategy must be 'always' or "
                "'tight_coupling'"
            )
        rate_expression = operator_def.get("rate_expression")
        clean_rate_expression = None
        rate_dependencies: tuple[str, ...] = ()
        compiled_rate_expression = None
        if rate_expression is not None:
            clean_rate_expression, rate_dependencies = (
                _replace_and_validate_expression(
                    rate_expression,
                    label=(
                        "cmb.perturbations.collision_operators."
                        f"{name}.rate_expression"
                    ),
                    replacements=replacements,
                    allowed_names=allowed_names,
                )
            )
            compiled_rate_expression = _compile_expression_plan(
                clean_rate_expression,
                dependencies=rate_dependencies,
            )
        exact_form = None
        if "exact_form" in operator_def:
            exact_form = _compile_collision_linear_form(
                operator_def.get("exact_form"),
                label=(
                    "cmb.perturbations.collision_operators."
                    f"{name}.exact_form"
                ),
                allowed_names=allowed_names,
                replacements=replacements,
                allow_damping=True,
            )
        linear_block = None
        if "linear_block" in operator_def:
            linear_block = _compile_collision_linear_form(
                operator_def.get("linear_block"),
                label=(
                    "cmb.perturbations.collision_operators."
                    f"{name}.linear_block"
                ),
                allowed_names=allowed_names,
                replacements=replacements,
                allow_damping=False,
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
            integration_strategy=integration_strategy,
            activation_strategy=activation_strategy,
            rate_expression=clean_rate_expression,
            rate_dependencies=rate_dependencies,
            compiled_rate_expression=compiled_rate_expression,
            exact_form=exact_form,
            linear_block=linear_block,
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


def _build_gauge_equivalence_summary(gauge: str) -> dict[str, Any]:
    """Describe the explicit scalar gauge bridge in a compiled manifest."""

    normalized_gauge = str(gauge or "unspecified")
    if normalized_gauge == "synchronous":
        return {
            "route": normalized_gauge,
            "observable_basis": "newtonian",
            "transformation": "scalar_first_order",
            "explicit": True,
            "metric_state_names": (
                "h_sync_metric",
                "eta_sync_metric",
                "gauge_shift_alpha",
                "Phi_gi",
            ),
            "observable_state_names": ("Phi", "Psi"),
            "derived_transform_names": (
                "Phi_from_synchronous",
                "Psi_from_synchronous",
            ),
        }
    if normalized_gauge == "gauge_invariant":
        return {
            "route": normalized_gauge,
            "observable_basis": "newtonian",
            "transformation": "bardeen_invariant",
            "explicit": True,
            "metric_state_names": ("Phi_gi", "Psi_gi"),
            "observable_state_names": ("Phi", "Psi"),
            "derived_transform_names": (),
        }
    if normalized_gauge == "conformal_newtonian":
        return {
            "route": normalized_gauge,
            "observable_basis": "newtonian",
            "transformation": "observable_identity",
            "explicit": True,
            "metric_state_names": ("Phi", "Psi"),
            "observable_state_names": ("Phi", "Psi"),
            "derived_transform_names": (),
        }
    return {
        "route": normalized_gauge,
        "observable_basis": None,
        "transformation": "declared_custom",
        "explicit": False,
        "metric_state_names": (),
        "observable_state_names": (),
        "derived_transform_names": (),
    }


def _build_vector_hierarchy_summary(
    *,
    generated_vector_hierarchy: bool,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    """Describe vector state roles and analytic projection kernels."""

    if not generated_vector_hierarchy:
        return {
            "implemented": False,
            "sector": None,
            "metric_state": None,
            "closure": None,
            "parity": (),
            "radial_kernels": (),
            "temperature_states": (),
            "polarization_e_states": (),
            "polarization_b_states": (),
            "neutrino_states": (),
        }
    variable_names = tuple(str(name) for name in variables)
    return {
        "implemented": True,
        "sector": "vector",
        "metric_state": "sigma_vector",
        "closure": "free_streaming_vector",
        "parity": ("even", "odd"),
        "radial_kernels": (
            "vector_temperature_1",
            "vector_temperature_2",
            "vector_e",
            "vector_b",
        ),
        "temperature_states": tuple(
            name for name in variable_names if name.startswith("theta_gamma_v")
        ),
        "polarization_e_states": tuple(
            name for name in variable_names if name.startswith("e_gamma_v")
        ),
        "polarization_b_states": tuple(
            name for name in variable_names if name.startswith("b_gamma_v")
        ),
        "neutrino_states": tuple(
            name for name in variable_names if name.startswith("nu_v")
        ),
    }


def _build_manifest_summary(
    *,
    model_name: str,
    contract_version: int,
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
    dependency_summary: PerturbationDependencyGraphSummaryData,
    generated_scalar_hierarchy: bool,
    generated_vector_hierarchy: bool,
    generated_tensor_hierarchy: bool,
    equation_wrt_by_variable: Mapping[str, str],
    boundary_condition_anchors: Mapping[str, str],
    transfer_component_contracts: Mapping[str, Mapping[str, Any]],
    angular_power_spectrum_targets: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Return a manifest-friendly summary of the compiled graph."""

    return {
        "model_name": model_name,
        "contract_version": contract_version,
        "gauge": gauge,
        "gauge_equivalence": _build_gauge_equivalence_summary(gauge),
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
        "execution_route": _build_execution_route_summary(),
        "compilation_ownership": {
            "compiler": (
                "copernican.lib.model_coder.compile_declared_cmb_runtime"
            ),
            "compiled_upstream": True,
            "hot_path_recompilation_allowed": False,
        },
        "generated_scalar_hierarchy": generated_scalar_hierarchy,
        "generated_vector_hierarchy": generated_vector_hierarchy,
        "generated_tensor_hierarchy": generated_tensor_hierarchy,
        "vector_hierarchy": _build_vector_hierarchy_summary(
            generated_vector_hierarchy=generated_vector_hierarchy,
            variables=variables,
        ),
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


def _build_execution_route_summary() -> dict[str, Any]:
    """Return the single declared execution-route metadata surface."""

    return {
        "solver_id": CCMBS_ID,
        "solver_label": CCMBS_LABEL,
        "runtime_module": (
            "copernican.lib.likelihoods.cmb.orchestrators.ccmbs"
        ),
        "ready": True,
    }


def validate_generated_scalar_source_graph(contract: Any) -> None:
    """Validate the generated scalar source graph before runtime use.

    Generated scalar contracts are materialized from one shared template, but
    the compiled graph remains the authority consumed by the solver.  Keep
    this validation at that boundary so a future template or model-specific
    override cannot omit metric derivatives, source roles, or closures and
    have the runtime silently substitute zeros.  Bundled initial-condition
    completeness is audited separately because small generated fixtures may
    intentionally declare only the regular mode under test.
    """

    manifest = getattr(contract, "manifest_summary", {}) or {}
    if not bool(manifest.get("generated_scalar_hierarchy")):
        return

    variables = getattr(contract, "variables", {}) or {}
    derived = getattr(contract, "derived", {}) or {}
    sources = getattr(contract, "sources", {}) or {}
    closures = getattr(contract, "closures", {}) or {}
    issues: list[str] = []

    if not {"Phi", "Psi"}.issubset(variables):
        issues.append("generated scalar hierarchy must expose Phi and Psi")
    required_derivatives = {"Phi_tau", "Psi_tau", "Phi_history_tau"}
    missing_derivatives = sorted(required_derivatives - set(derived))
    if missing_derivatives:
        issues.append(
            "missing generated metric derivative(s): "
            + ", ".join(missing_derivatives)
        )
    phi_tau = derived.get("Phi_tau")
    if phi_tau is not None:
        if getattr(phi_tau, "kind", None) != (
            "metric_potential_time_derivative"
        ):
            issues.append(
                "Phi_tau must declare the metric-potential derivative kind"
            )
        if not str(getattr(phi_tau, "expression", "") or "").strip():
            issues.append("Phi_tau must have an explicit graph expression")
        phi_tau_dependencies = set(getattr(phi_tau, "dependencies", ()) or ())
        for dependency in ("metric_momentum_source_drive", "Hconf", "Psi"):
            if dependency not in phi_tau_dependencies:
                issues.append(
                    "Phi_tau must depend on the declared "
                    f"{dependency} source"
                )
    psi_tau = derived.get("Psi_tau")
    if psi_tau is not None:
        if getattr(psi_tau, "kind", None) != (
            "metric_potential_time_derivative"
        ):
            issues.append(
                "Psi_tau must declare the metric-potential derivative kind"
            )
        if (
            str(getattr(psi_tau, "variable", "")) != "Psi"
            or str(getattr(psi_tau, "wrt", "")) != "tau"
            or int(getattr(psi_tau, "order", 0) or 0) != 1
        ):
            issues.append("Psi_tau must be the first tau derivative of Psi")
        if getattr(psi_tau, "expression", None) is not None:
            issues.append("Psi_tau must not replace its history derivative")
        if getattr(psi_tau, "binding", None) != "runtime_history_gradient":
            issues.append(
                "Psi_tau must declare the runtime history-gradient binding"
            )
    phi_history_tau = derived.get("Phi_history_tau")
    if phi_history_tau is not None:
        if getattr(phi_history_tau, "kind", None) != (
            "metric_history_time_derivative"
        ):
            issues.append(
                "Phi_history_tau must declare the metric-history derivative "
                "kind"
            )
        if (
            str(getattr(phi_history_tau, "variable", "")) != "Phi"
            or str(getattr(phi_history_tau, "wrt", "")) != "tau"
            or int(getattr(phi_history_tau, "order", 0) or 0) != 1
        ):
            issues.append(
                "Phi_history_tau must be the first tau derivative of Phi"
            )
        if getattr(phi_history_tau, "expression", None) is not None:
            issues.append(
                "Phi_history_tau must not use a zero-expression fallback"
            )
        if getattr(phi_history_tau, "binding", None) != (
            "runtime_history_gradient"
        ):
            issues.append(
                "Phi_history_tau must declare the runtime history-gradient "
                "binding"
            )
        if (
            "runtime binds"
            not in str(
                getattr(phi_history_tau, "description", "") or ""
            ).lower()
        ):
            issues.append("Phi_history_tau must document runtime binding")

    required_roles = {
        "monopole",
        "additive",
        "additive_derivative",
        "doppler",
        "isw",
        "polarization",
        "polarization_b",
        "potential",
    }
    source_roles = {
        str(getattr(entry, "role", "")) for entry in sources.values()
    }
    missing_roles = sorted(required_roles - source_roles)
    if missing_roles:
        issues.append(
            "missing generated source role(s): " + ", ".join(missing_roles)
        )
    required_residuals = {
        "einstein_energy_residual",
        "einstein_momentum_residual",
        "einstein_shear_residual",
    }
    missing_residuals = sorted(required_residuals - set(derived))
    if missing_residuals:
        issues.append(
            "missing generated scalar residual(s): "
            + ", ".join(missing_residuals)
        )

    # Initial-condition family completeness is audited against the bundled
    # model inventory.  Small generated fixtures may intentionally exercise a
    # single regular mode, so the compiler only enforces the source graph
    # symbols needed by every generated runtime here.
    for closure_name in (
        "psi_closure",
        "visibility_polarization_moment_closure",
    ):
        if closure_name not in closures:
            issues.append(f"generated hierarchy must declare {closure_name}")
    for source_name, source in sources.items():
        if getattr(source, "compiled_expression", None) is None:
            issues.append(f"source '{source_name}' is not compiler-backed")
    for closure_name, closure in closures.items():
        if getattr(closure, "compiled_expression", None) is None:
            issues.append(f"closure '{closure_name}' is not compiler-backed")

    if issues:
        raise ValueError(
            "Generated scalar source graph validation failed: "
            + "; ".join(sorted(set(issues)))
        )


def _generated_scalar_source_closure_summary(contract: Any) -> dict[str, Any]:
    """Return immutable-friendly provenance for a validated scalar graph.

    The summary is attached to the compiled manifest and is consumed by the
    runtime diagnostics.  It records the distinction between an algebraic
    Einstein derivative (``Phi_tau``) and history-bound derivatives used by
    line-of-sight sources.  Keeping this provenance in the contract prevents
    a later runtime path from silently replacing one kind with the other.
    """

    manifest = getattr(contract, "manifest_summary", {}) or {}
    if not bool(manifest.get("generated_scalar_hierarchy")):
        return {
            "schema_version": 1,
            "status": "not_applicable",
            "metric_derivatives": {},
            "source_roles": {},
            "closure_names": (),
            "residual_names": (),
        }
    derived = getattr(contract, "derived", {}) or {}
    sources = getattr(contract, "sources", {}) or {}
    closures = getattr(contract, "closures", {}) or {}
    metric_derivatives = {}
    for name in ("Phi_tau", "Psi_tau", "Phi_history_tau"):
        entry = derived.get(name)
        if entry is None:
            continue
        metric_derivatives[name] = {
            "kind": str(getattr(entry, "kind", "")),
            "variable": (
                None
                if getattr(entry, "variable", None) is None
                else str(entry.variable)
            ),
            "wrt": (
                None if getattr(entry, "wrt", None) is None else str(entry.wrt)
            ),
            "order": (
                None
                if getattr(entry, "order", None) is None
                else int(entry.order)
            ),
            "binding": (
                None
                if getattr(entry, "binding", None) is None
                else str(entry.binding)
            ),
            "expression": (
                None
                if getattr(entry, "expression", None) is None
                else str(entry.expression)
            ),
            "dependencies": tuple(
                str(value)
                for value in (getattr(entry, "dependencies", ()) or ())
            ),
        }
    role_to_sources: dict[str, list[str]] = {}
    for source_name, source in sources.items():
        role_to_sources.setdefault(str(getattr(source, "role", "")), [])
        role_to_sources[str(getattr(source, "role", ""))].append(
            str(source_name)
        )
    return {
        "schema_version": 1,
        "status": "validated",
        "metric_derivatives": metric_derivatives,
        "source_roles": {
            role: tuple(sorted(names))
            for role, names in sorted(role_to_sources.items())
        },
        "closure_names": tuple(sorted(str(name) for name in closures)),
        "residual_names": tuple(
            name
            for name in (
                "einstein_energy_residual",
                "einstein_momentum_residual",
                "einstein_shear_residual",
            )
            if name in derived
        ),
    }


def compile_perturbation_contract(
    contract: Mapping[str, Any],
    *,
    model_name: str,
    parameter_names: Sequence[str],
    latex_names: Sequence[str],
    background_reference_names: Sequence[str],
) -> PerturbationContractData:
    """Validate and compile a declared CMB graph contract."""

    if not isinstance(contract, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")
    contract, materialized_scalar_hierarchy = (
        _materialize_declared_scalar_hierarchy_contract(contract)
    )
    contract, materialized_vector_hierarchy = (
        _materialize_declared_vector_hierarchy_contract(contract)
    )
    contract, materialized_tensor_hierarchy = (
        _materialize_declared_tensor_hierarchy_contract(contract)
    )

    cache_key = (
        _freeze_for_cache(contract),
        str(model_name),
        tuple(str(name) for name in parameter_names),
        tuple(str(name) for name in latex_names),
        tuple(str(name) for name in background_reference_names),
    )
    cached_result = _COMPILED_CONTRACT_RESULTS.get(cache_key)
    if cached_result is not None:
        return cached_result

    contract_keys = {str(key) for key in contract.keys()}
    required_sections = {
        "contract_version",
        "gauge",
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
    if contract_version != 2:
        raise ValueError(
            "Declared perturbations must declare contract_version: 2"
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

    if materialized_scalar_hierarchy:
        expected_scalar_units = {
            "einstein_energy_residual": _INVERSE_MPC_SQUARED_UNITS,
            "einstein_momentum_residual": _INVERSE_MPC_CUBED_UNITS,
            "einstein_shear_residual": _INVERSE_MPC_SQUARED_UNITS,
        }
        for variable_name, variable_entry in variable_entries.items():
            expected_units = (
                _INVERSE_MPC_UNITS
                if (
                    "velocity_divergence" in variable_entry.kind
                    or "visibility_weighted_source_moment"
                    in variable_entry.kind
                )
                else _DIMENSIONLESS_UNITS
            )
            if variable_entry.units != expected_units:
                raise ValueError(
                    "Generated scalar variable "
                    f"'{variable_name}' must declare units "
                    f"'{expected_units}'"
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
                    "distribution_weight",
                    "velocity_ratio",
                    "pressure_ratio",
                    "mass_fraction",
                    "density_weight",
                    "momentum_weight",
                    "pressure_weight",
                    "shear_weight",
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
                and variable_name not in declared_derived_names
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
                binding=_validate_derived_binding(
                    derived_def.get("binding"),
                    label=f"cmb.perturbations.derived.{name}.binding",
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
                units=_validate_optional_string(
                    derived_def.get("units"),
                    label=f"cmb.perturbations.derived.{name}.units",
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
            binding=_validate_derived_binding(
                derived_def.get("binding"),
                label=f"cmb.perturbations.derived.{name}.binding",
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
            units=_validate_optional_string(
                derived_def.get("units"),
                label=f"cmb.perturbations.derived.{name}.units",
            ),
            dependencies=dependencies,
            compiled_expression=_compile_expression_plan(
                clean_expression,
                dependencies=dependencies,
            ),
        )
        expression_derived_names.append(name)

    if materialized_scalar_hierarchy:
        for derived_name, expected_units in expected_scalar_units.items():
            derived_entry = derived_entries.get(derived_name)
            if derived_entry is None:
                raise ValueError(
                    "Generated scalar hierarchy must declare derived "
                    f"diagnostic '{derived_name}'"
                )
            if derived_entry.units != expected_units:
                raise ValueError(
                    "Generated scalar diagnostic "
                    f"'{derived_name}' must declare units "
                    f"'{expected_units}'"
                )

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
            units=_validate_optional_string(
                source_def.get("units"),
                label=f"cmb.perturbations.sources.{name}.units",
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

        return str(
            get_declared_projection_spec(
                projection,
                extensions=projection_extension_entries,
            ).output_role
        )

    def _projection_transfer_units(projection: str) -> str | None:
        """Return the canonical transfer-component units for ``projection``."""

        return get_declared_projection_spec(
            projection,
            extensions=projection_extension_entries,
        ).transfer_units

    def _power_spectrum_units(
        primary_role: str | None,
        secondary_role: str | None,
    ) -> str | None:
        """Return the default public units for one spectrum pair."""

        role_names = {primary_role, secondary_role}
        if role_names == {"potential"}:
            return "dimensionless"
        if "potential" in role_names:
            return "muK"
        if role_names.issubset(
            {"polarization_b", "polarization_e", "temperature"}
        ):
            return "muK^2"
        return None

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
        observable_units = _validate_optional_string(
            observable_def.get("units"),
            label=f"cmb.perturbations.observables.{name}.units",
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
            validate_declared_projection_sector(
                declared_projection,
                sector,
                observable_name=name,
                kernel=kernel,
                extensions=projection_extension_entries,
            )
            for role_name in source_term_refs:
                source_kernel = resolve_declared_source_kernel(
                    declared_projection,
                    role_name,
                    kernel=kernel,
                    extensions=projection_extension_entries,
                )
                validate_declared_projection_sector(
                    declared_projection,
                    sector,
                    observable_name=name,
                    kernel=source_kernel,
                    extensions=projection_extension_entries,
                )
            if observable_units is None:
                observable_units = _projection_transfer_units(
                    declared_projection
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
            units=observable_units,
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
            units=(
                observable_entry.units
                if observable_entry.units is not None
                else _power_spectrum_units(primary_role, secondary_role)
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
                    "Auto-generated declared initial condition for "
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

    if not variable_entries:
        raise ValueError("Declared perturbations must declare variables")
    if not equation_entries:
        raise ValueError("Declared perturbations must declare equations")
    if not initial_condition_entries and not boundary_condition_entries:
        raise ValueError(
            "Declared perturbations must declare initial_conditions or "
            "boundary_conditions"
        )
    if not observable_entries:
        raise ValueError("Declared perturbations must declare observables")
    if not sections["validity"]:
        raise ValueError("Declared perturbations must declare validity")

    validity_notes = _validate_optional_string(
        sections["validity"].get("notes"),
        label="cmb.perturbations.validity.notes",
    )
    validity_regimes = sections["validity"].get("regimes")
    regimes = ()
    if validity_regimes is not None:
        regimes = _validate_regimes(validity_regimes)
    else:
        raise ValueError(
            "Declared perturbations must declare validity.regimes"
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
        runtime_bound = any(
            entry.expression is None
            and entry.binding == "runtime_history_gradient"
            and (str(entry.variable), str(entry.wrt)) == key
            for entry in derived_entries.values()
        )
        if key not in equation_orders:
            variable_name, wrt_name = key
            if variable_name in relation_target_names:
                continue
            raise ValueError(
                "Derivative symbol requires an evolved variable: "
                f"{variable_name} wrt {wrt_name}"
            )
        if required_order >= equation_orders[key] and not runtime_bound:
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
            "Declared perturbations are missing required initial "
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
            "output_role": (
                None if entry.output_role is None else str(entry.output_role)
            ),
            "units": (None if entry.units is None else str(entry.units)),
        }
        for name, entry in observable_entries.items()
        if entry.kind == "transfer_component"
    }
    angular_power_spectrum_targets = {
        name: {
            "primary": str(entry.primary or ""),
            "secondary": str(entry.secondary or ""),
            "output_role": str(entry.output_role or ""),
            "units": str(entry.units or ""),
        }
        for name, entry in observable_entries.items()
        if entry.kind == "angular_power_spectrum"
    }
    manifest_summary_data = _build_manifest_summary(
        model_name=model_name,
        contract_version=contract_version,
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
        dependency_summary=dependency_summary,
        generated_scalar_hierarchy=materialized_scalar_hierarchy,
        generated_vector_hierarchy=materialized_vector_hierarchy,
        generated_tensor_hierarchy=materialized_tensor_hierarchy,
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

    compiled = PerturbationContractData(
        model_name=model_name,
        contract_version=contract_version,
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
        dependency_graph_summary=dependency_summary,
        manifest_summary=FrozenMapping(manifest_summary_data),
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
    validate_generated_scalar_source_graph(compiled)
    manifest_summary_data["generated_scalar_source_closure"] = (
        _generated_scalar_source_closure_summary(compiled)
    )
    compiled = replace(
        compiled,
        manifest_summary=FrozenMapping(manifest_summary_data),
    )
    _COMPILED_CONTRACT_RESULTS[cache_key] = compiled
    return _get_cached_perturbation_contract(cache_key)


__all__ = [
    "PerturbationCollisionLinearFormData",
    "PerturbationCollisionOperatorData",
    "PerturbationCollisionTargetSelectorData",
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
    "validate_generated_scalar_source_graph",
]
