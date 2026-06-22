r"""Native declared-graph CMB solver support.

This module owns the internal execution helpers for the `standard: false`
native CMB path. It compiles and evaluates the declared background,
perturbation, and projection runtime without exposing a second public CMB
façade beside :mod:`copernican.lib.likelihoods.cmb.cmb`.
"""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import astuple, dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.special import spherical_jn

from ...cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_kernel_spec,
)
from ...engine_adapter import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _SUPPORTED_CMB_BACKEND,
    FrozenMapping,
    _evaluate_safe_expression,
    _freeze_for_cache,
)
from ...model_coder import validate_native_perturbation_execution
from ...perturbation_contract import (
    _compile_expression_plan,
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)

_C_LIGHT_KM_S = 299_792.458
_CACHE_PRECISION = 15
_LEGACY_DECLARED_EVOLUTION_COORDINATES = {"eta", "tau"}
_PHYSICAL_QUANTITY_ALIASES = {
    "H0_km_s_Mpc": (
        "H0",
        "hubble_constant",
        "Hubble constant",
    ),
    "Omega_b0": (
        "Omega_b",
        "Omega_b0",
        "baryon_density_fraction",
        "baryon_fraction_today",
        "omega_b",
    ),
    "ombh2": (
        "Omega_b_h2",
        "baryon_density_h2",
        "ombh2",
        "omega_b_h2",
    ),
    "Omega_c0": (
        "Omega_c",
        "Omega_c0",
        "Omega_cdm",
        "Omega_cdm0",
        "cdm_density_fraction",
        "cold_dark_matter_fraction_today",
        "omega_c",
        "omega_cdm",
    ),
    "omch2": (
        "Omega_c_h2",
        "cdm_density_h2",
        "cold_dark_matter_density_h2",
        "omch2",
        "omega_c_h2",
    ),
    "Tcmb_K": (
        "T_cmb",
        "Tcmb",
        "Tcmb_K",
        "cmb_temperature_K",
    ),
    "YHe": (
        "YHe",
        "Yp",
        "helium_fraction",
        "helium_mass_fraction",
    ),
    "Neff": (
        "N_eff",
        "Neff",
        "effective_neutrino_count",
    ),
    "Omega_gamma0": (
        "Omega_gamma",
        "Omega_gamma0",
        "omgamma",
        "omega_gamma",
        "photon_density_fraction",
        "photon_fraction_today",
    ),
    "Omega_nu0": (
        "Omega_nu",
        "Omega_nu0",
        "omega_nu",
        "relativistic_neutrino_density_fraction",
        "relativistic_neutrino_fraction_today",
    ),
    "Omega_r0": (
        "Omega_r",
        "Omega_r0",
        "omega_r",
        "radiation_density_fraction",
        "radiation_fraction_today",
    ),
    "Omega_k0": (
        "Omega_k",
        "Omega_k0",
        "curvature_density_fraction",
        "omk",
        "omega_k",
    ),
    "Omega_m0": (
        "Omega_m",
        "Omega_m0",
        "matter_density_fraction",
        "matter_fraction_today",
        "omega_m",
    ),
    "Omega_de0": (
        "Omega_Lambda",
        "Omega_de",
        "Omega_de0",
        "Omega_lambda",
        "dark_energy_density_fraction",
        "dark_energy_fraction_today",
        "omega_de",
    ),
    "w0": (
        "dark_energy_eos0",
        "dark_energy_w0",
        "equation_of_state_today",
        "w",
        "w0",
    ),
    "wa": (
        "dark_energy_eos1",
        "equation_of_state_derivative",
        "wa",
    ),
    "primordial_amplitude": (
        "A_s",
        "As",
        "primordial_amplitude",
        "primordial_power_amplitude",
    ),
    "primordial_spectral_index": (
        "n_s",
        "ns",
        "primordial_power_tilt",
        "primordial_spectral_index",
        "primordial_tilt",
    ),
    "chi": (
        "chi",
        "comoving_distance",
    ),
    "angular_diameter_distance": (
        "D_A",
        "angular_diameter_distance",
        "da",
    ),
}
_BACKGROUND_PROVENANCE_ROLE_KEYS = {
    "expansion": ("H0_km_s_Mpc",),
    "density": (
        "Omega_b0",
        "ombh2",
        "Omega_c0",
        "omch2",
        "Omega_gamma0",
        "Omega_nu0",
        "Omega_r0",
        "Omega_m0",
        "Omega_de0",
    ),
    "pressure": ("w0",),
    "equation_of_state": ("w0", "wa"),
    "curvature": ("Omega_k0", "chi", "angular_diameter_distance"),
    "primordial": (
        "primordial_amplitude",
        "primordial_spectral_index",
    ),
}


def _normalise_value(entry_value: Any) -> Any:
    """Return a cache-friendly representation of ``entry_value``."""

    if isinstance(entry_value, (int, float, numpy.integer, numpy.floating)):
        return float(entry_value)
    return str(entry_value)


def _normalise_items(
    param_dict: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """Convert ``param_dict`` into a deterministic tuple of items."""

    normalised: list[tuple[str, Any]] = []
    for key in sorted(param_dict):
        normalised.append((str(key), _normalise_value(param_dict[key])))
    return tuple(normalised)


def _coerce_numeric_scalar(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite scalar float."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 0:
        raise ValueError(f"{name} must evaluate to a scalar")
    scalar = float(array_value)
    if not numpy.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _coerce_numeric_array(value: Any, *, name: str) -> numpy.ndarray:
    """Return ``value`` as a finite one-dimensional array."""

    array_value = numpy.asarray(value, dtype=float)
    if array_value.ndim != 1:
        raise ValueError(f"{name} must evaluate to a one-dimensional array")
    if array_value.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not numpy.all(numpy.isfinite(array_value)):
        raise ValueError(f"{name} must contain only finite values")
    return array_value


def _get_declared_background_section(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the declared background mapping for native CMB execution."""

    section = contract.get("background")
    if not isinstance(section, Mapping):
        raise ValueError(
            "Declared CMB native execution requires a 'background' mapping."
        )
    derived = section.get("derived", {})
    if not isinstance(derived, Mapping):
        raise ValueError("background.derived must be a mapping.")
    reionization = section.get("reionization", {})
    if reionization is None:
        reionization = {}
    if not isinstance(reionization, Mapping):
        raise ValueError("background.reionization must be a mapping.")
    calibration = reionization.get("calibration", {})
    if calibration is None:
        calibration = {}
    if not isinstance(calibration, Mapping):
        raise ValueError(
            "background.reionization.calibration must be a mapping."
        )
    quantities = reionization.get("quantities", {})
    if quantities is None:
        quantities = {}
    if not isinstance(quantities, Mapping):
        raise ValueError(
            "background.reionization.quantities must be a mapping."
        )
    return section


def _expression_symbol_names(expression: str) -> set[str]:
    """Return bare symbol names referenced by ``expression``."""

    try:
        node = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid declared background expression: {expression!r}"
        ) from exc
    names = {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    names.difference_update(_ALLOWED_CONSTANTS)
    names.difference_update(_ALLOWED_MATH_FUNCS)
    return names


@dataclass(frozen=True, slots=True)
class _DeclaredSymbolPlanEntry:
    """One ordered declared-background evaluation step."""

    kind: str
    name: str
    payload: Any


_DECLARED_SYMBOL_PLAN_RESULTS: dict[
    Any, tuple[_DeclaredSymbolPlanEntry, ...]
] = {}


@lru_cache(maxsize=256)
def _get_cached_declared_symbol_plan(
    cache_key: Any,
) -> tuple[_DeclaredSymbolPlanEntry, ...]:
    """Return one cached declared-symbol evaluation plan."""

    return _DECLARED_SYMBOL_PLAN_RESULTS[cache_key]


def _compile_declared_symbol_plan(
    entries: Mapping[str, Any],
) -> tuple[_DeclaredSymbolPlanEntry, ...]:
    """Return one ordered declared symbol plan for background helpers."""

    cache_key = _freeze_for_cache(entries)
    cached = _DECLARED_SYMBOL_PLAN_RESULTS.get(cache_key)
    if cached is not None:
        return _get_cached_declared_symbol_plan(cache_key)

    pending = {str(name): value for name, value in entries.items()}
    local_names = set(pending)
    resolved_names: set[str] = set()
    plan: list[_DeclaredSymbolPlanEntry] = []
    while pending:
        progress = False
        for name, raw_value in tuple(sorted(pending.items())):
            if isinstance(raw_value, bool):
                raise ValueError(
                    "Declared background entries must be numeric or "
                    "string expressions."
                )
            if isinstance(
                raw_value, (int, float, numpy.integer, numpy.floating)
            ):
                plan.append(
                    _DeclaredSymbolPlanEntry(
                        kind="literal",
                        name=name,
                        payload=float(raw_value),
                    )
                )
                resolved_names.add(name)
                pending.pop(name)
                progress = True
                continue
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(
                    "Declared background entries must be numeric or "
                    "string expressions."
                )
            expression_plan = _compile_expression_plan(raw_value.strip())
            unresolved_locals = {
                dependency
                for dependency in expression_plan.dependencies
                if dependency in local_names
                and dependency not in resolved_names
            }
            if unresolved_locals:
                continue
            plan.append(
                _DeclaredSymbolPlanEntry(
                    kind="expression",
                    name=name,
                    payload=expression_plan,
                )
            )
            resolved_names.add(name)
            pending.pop(name)
            progress = True
        if progress:
            continue
        unresolved_names = ", ".join(sorted(pending))
        raise ValueError(
            "Declared background expressions contain circular or "
            f"unresolved names: {unresolved_names}"
        )
    compiled_plan = tuple(plan)
    _DECLARED_SYMBOL_PLAN_RESULTS[cache_key] = compiled_plan
    return _get_cached_declared_symbol_plan(cache_key)


def _evaluate_declared_symbol_plan(
    plan: Sequence[_DeclaredSymbolPlanEntry],
    *,
    base_context: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Evaluate one ordered declared symbol plan against ``base_context``."""

    resolved: dict[str, Any] = dict(base_context)
    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for entry in plan:
            if isinstance(entry, _DeclaredSymbolPlanEntry):
                kind = entry.kind
                name = entry.name
                payload = entry.payload
            else:
                kind, name, payload = entry
            if kind == "literal":
                resolved[name] = float(payload)
                continue
            compiled_expression = payload
            missing_names = sorted(
                dependency
                for dependency in compiled_expression.dependencies
                if dependency not in resolved
            )
            if missing_names:
                missing_text = ", ".join(missing_names)
                raise ValueError(
                    f"{label}.{name} references unknown symbol(s): "
                    f"{missing_text}"
                )
            resolved[name] = _evaluate_compiled_expression_noerr(
                compiled_expression,
                resolved,
            )
    return resolved


def _resolve_declared_symbol_context(
    entries: Mapping[str, Any],
    *,
    base_context: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Resolve numeric values and safe expressions from one declaration map."""

    return _evaluate_declared_symbol_plan(
        _compile_declared_symbol_plan(entries),
        base_context=base_context,
        label=label,
    )


def _resolve_declared_background_context(
    contract: Mapping[str, Any],
    *,
    a_values: Any,
    z_values: Any,
) -> dict[str, Any]:
    """Return the resolved declared background graph context."""

    section = _get_declared_background_section(contract)
    env: dict[str, Any] = {
        "a": a_values,
        "z": z_values,
    }
    for source in (
        contract.get("param_map", {}) or {},
        contract.get("model_parameters", {}) or {},
    ):
        if not isinstance(source, Mapping):
            continue
        for name, value in source.items():
            key = str(name)
            if key in env:
                continue
            if isinstance(value, (int, float, numpy.integer, numpy.floating)):
                env[key] = float(value)
    background_runtime = contract.get("background_runtime")
    if background_runtime is not None:
        return _evaluate_declared_symbol_plan(
            getattr(background_runtime, "derived_plan", ()),
            base_context=env,
            label="background.derived",
        )
    return _resolve_declared_symbol_context(
        section.get("derived", {}) or {},
        base_context=env,
        label="background.derived",
    )


def _get_declared_reionization_section(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the declared reionization mapping for native CMB execution."""

    section = _get_declared_background_section(contract)
    reionization = section.get("reionization", {}) or {}
    return reionization


def _resolve_declared_reionization_context(
    contract: Mapping[str, Any],
    *,
    base_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return resolved declared reionization quantities."""

    background_runtime = contract.get("background_runtime")
    if background_runtime is not None:
        return _evaluate_declared_symbol_plan(
            getattr(background_runtime, "reionization_quantity_plan", ()),
            base_context=base_context,
            label="background.reionization.quantities",
        )
    reionization = _get_declared_reionization_section(contract)
    return _resolve_declared_symbol_context(
        reionization.get("quantities", {}) or {},
        base_context=base_context,
        label="background.reionization.quantities",
    )


def _lookup_declared_background_scalar(
    contract: Mapping[str, Any],
    background_context: Mapping[str, Any],
    names: Sequence[str],
) -> float | None:
    """Return the first declared scalar value among ``names``."""

    value = _extract_contract_scalar(contract, names, default=None)
    if value is not None:
        return float(value)
    for name in names:
        if name not in background_context:
            continue
        return _coerce_numeric_scalar(
            background_context[name],
            name=f"background.derived.{name}",
        )
    return None


def _lookup_declared_background_scalar_with_source(
    contract: Mapping[str, Any],
    background_context: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[float, str] | None:
    """Return the first declared scalar among ``names`` and its source."""

    contract_value = _extract_contract_scalar_with_source(contract, names)
    if contract_value is not None:
        return contract_value
    for name in names:
        if name not in background_context:
            continue
        return (
            _coerce_numeric_scalar(
                background_context[name],
                name=f"background.derived.{name}",
            ),
            f"background.derived:{name}",
        )
    return None


def _physical_quantity_names(quantity_key: str) -> tuple[str, ...]:
    """Return accepted aliases for one physical quantity."""

    return tuple(_PHYSICAL_QUANTITY_ALIASES.get(quantity_key, (quantity_key,)))


def _summarize_declared_background_manifest_summary(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return manifest-friendly background provenance for a CMB contract."""

    section = _get_declared_background_section(contract)
    reionization = section.get("reionization", {}) or {}
    calibration = reionization.get("calibration", {}) or {}
    derived_names = tuple(
        sorted(str(name) for name in (section.get("derived", {}) or {}))
    )
    reionization_names = tuple(
        sorted(
            str(name) for name in ((reionization.get("quantities", {})) or {})
        )
    )
    contract_scalar_names = {
        str(name)
        for source_name in ("param_map", "model_parameters")
        for name in (
            (contract.get(source_name, {}) or {})
            if isinstance(contract.get(source_name, {}) or {}, Mapping)
            else {}
        )
    }
    available_names = set(derived_names) | contract_scalar_names
    quantity_aliases = {
        quantity_name: tuple(
            sorted(
                alias
                for alias in _physical_quantity_names(quantity_name)
                if alias in available_names
            )
        )
        for quantity_name in _PHYSICAL_QUANTITY_ALIASES
    }
    quantity_aliases = {
        key: value for key, value in quantity_aliases.items() if value
    }
    role_names = {}
    for role_name, quantity_names in _BACKGROUND_PROVENANCE_ROLE_KEYS.items():
        role_aliases = tuple(
            sorted(
                alias
                for quantity_name in quantity_names
                for alias in quantity_aliases.get(quantity_name, ())
            )
        )
        if role_aliases:
            role_names[role_name] = role_aliases
    return {
        "background_derived_names": derived_names,
        "background_reionization_quantity_names": reionization_names,
        "background_quantity_aliases": quantity_aliases,
        "background_quantity_role_names": role_names,
        "reionization_calibration": {
            "symbol": calibration.get("symbol"),
            "target_optical_depth": calibration.get("target_optical_depth"),
            "lower": calibration.get("lower"),
            "upper": calibration.get("upper"),
        },
        "recombination_runtime": {
            "hydrogen_model": "peebles_case_b_ode",
            "helium_electron_contribution": True,
            "reionization_ode": True,
        },
    }


def _is_structured_camb_contract(
    contract_or_params: Mapping[str, Any],
) -> bool:
    """Return ``True`` when ``contract_or_params`` includes perturbations."""

    keys = {str(key) for key in contract_or_params.keys()}
    required = {
        "backend",
        "calls",
        "grids",
        "param_map",
        "perturbations",
        "values",
    }
    return required.issubset(keys)


def _combine_camb_contracts(
    background_contract: Mapping[str, Any],
    perturbation_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a structured CAMB contract with perturbation metadata."""

    combined = dict(background_contract)
    if perturbation_contract:
        combined["perturbations"] = dict(perturbation_contract)
    return combined


def _validate_camb_perturbation_execution(
    contract: Mapping[str, Any],
) -> None:
    """Reject unsupported perturbation declarations before CAMB runs."""

    perturbations = contract.get("perturbations")
    if perturbations is None:
        raise ValueError("Structured CAMB contract is missing perturbations")
    if not isinstance(perturbations, Mapping):
        raise ValueError("cmb.perturbations must be a mapping")

    model_name = contract.get("model_name", "unknown model")
    backend = str(contract.get("backend", "camb"))

    standard = perturbations.get("standard")
    if not isinstance(standard, bool):
        raise ValueError("cmb.perturbations.standard must be boolean")
    if standard:
        return

    backend_mapping = perturbations.get("backend_mapping", {})
    backend_entry = {}
    if isinstance(backend_mapping, Mapping):
        backend_entry = backend_mapping.get(backend, {}) or {}

    implemented = None
    if isinstance(backend_entry, Mapping):
        implemented = backend_entry.get("implemented")

    validate_native_perturbation_execution(
        model_name=str(model_name),
        backend=backend,
        standard=standard,
        implemented=implemented if isinstance(implemented, bool) else None,
    )


def _compile_declared_perturbation_contract(
    contract: Mapping[str, Any],
):
    """Return the compiled perturbation contract for generic execution."""

    precompiled = contract.get("perturbation_data")
    if precompiled is not None:
        return precompiled

    from ...perturbation_contract import compile_perturbation_contract

    model_parameters = contract.get("model_parameters", {})
    if isinstance(model_parameters, Mapping) and model_parameters:
        parameter_names = tuple(str(key) for key in model_parameters)
    else:
        parameter_names = tuple(
            str(key) for key in (contract.get("param_map", {}) or {})
        )

    latex_names = tuple("" for _ in parameter_names)
    background_reference_names = {
        str(key) for key in (contract.get("param_map", {}) or {})
    }
    grid_defs = contract.get("grids", {}) or {}
    for grid_def in grid_defs.values():
        if not isinstance(grid_def, Mapping):
            continue
        symbol = grid_def.get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            background_reference_names.add(symbol.strip())
    background_reference_names.update(
        str(key) for key in (contract.get("values", {}) or {})
    )
    background_reference_names.update(parameter_names)
    background_section = contract.get("background")
    if isinstance(background_section, Mapping):
        background_reference_names.update(
            str(key) for key in (background_section.get("derived", {}) or {})
        )

    perturbations = contract.get("perturbations", {})
    if isinstance(perturbations, Mapping):
        perturbations = {
            key: value
            for key, value in perturbations.items()
            if key not in {"backend", "model_name"}
        }
    return compile_perturbation_contract(
        perturbations,
        model_name=str(contract.get("model_name", "unknown model")),
        backend=str(contract.get("backend", _SUPPORTED_CMB_BACKEND)),
        parameter_names=parameter_names,
        latex_names=latex_names,
        background_reference_names=tuple(sorted(background_reference_names)),
    )


@dataclass(slots=True)
class _CustomCMBNumerics:
    """Numerical settings used by the declared-graph CMB engine."""

    ell_min: int = 2
    ell_max: int = 2500
    k_min: float = 1.0e-5
    k_max: float = 0.4
    k_sample_count: int = 64
    eta_sample_count: int = 1024
    photon_hierarchy_l_max: int = 8
    neutrino_hierarchy_l_max: int = 8
    ode_rtol: float = 1.0e-6
    ode_atol: float = 1.0e-9
    tight_coupling_ratio: float = 50.0
    a_min: float = 1.0e-8
    source_grid_multiplier: int = 2
    initial_redshift: float = 1.0e5


@dataclass(slots=True)
class _CustomCMBPhysicalParameters:
    """Resolved physical background inputs for the native CMB solver."""

    H0_km_s_Mpc: float
    hubble_ratio: float
    H0_over_c_Mpc_inv: float
    ombh2: float
    omch2: float | None
    Omega_b0: float
    Omega_c0: float | None
    Omega_m0_background: float | None
    Omega_gamma0: float
    Omega_nu0: float | None
    Omega_r0: float | None
    Omega_k0: float | None
    Omega_de0: float | None
    dark_energy_eos0: float | None
    dark_energy_eos1: float | None
    YHe: float
    Neff: float | None
    primordial_amplitude: float
    primordial_spectral_index: float
    z_rec: float
    tau_reio: float
    Tcmb_K: float
    n_b0_m3: float
    n_H0_m3: float
    rho_b0_kg_m3: float
    has_cdm: bool
    has_dark_energy: bool
    quantity_provenance: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True)
class _CustomCMBBackgroundData:
    """Background and recombination tables for the generic CMB solver."""

    a_grid: numpy.ndarray
    z_grid: numpy.ndarray
    eta_grid: numpy.ndarray
    eta0: float
    chi_grid: numpy.ndarray
    da_grid: numpy.ndarray
    H_grid: numpy.ndarray
    Hconf_grid: numpy.ndarray
    tau_grid: numpy.ndarray
    tau_dot_grid: numpy.ndarray
    visibility_grid: numpy.ndarray
    x_e_grid: numpy.ndarray
    n_e_grid: numpy.ndarray
    n_H_grid: numpy.ndarray
    sound_speed_grid: numpy.ndarray
    sound_horizon_mpc: float
    reionization_z: float
    reionization_tau: float
    eta_rec: float
    a_rec: float
    z_rec: float
    eta_of_a: PchipInterpolator
    a_of_eta: PchipInterpolator
    z_of_eta: PchipInterpolator
    H_of_eta: PchipInterpolator
    chi_of_eta: PchipInterpolator
    da_of_eta: PchipInterpolator
    tau_of_eta: PchipInterpolator
    tau_dot_of_eta: PchipInterpolator
    visibility_of_eta: PchipInterpolator
    x_e_of_eta: PchipInterpolator
    sound_speed_of_eta: PchipInterpolator

    def sample(
        self, eta_values: numpy.ndarray | float
    ) -> dict[str, numpy.ndarray]:
        """Return the background quantities interpolated at ``eta_values``."""

        eta_arr = numpy.asarray(eta_values, dtype=float)
        return {
            "a": numpy.asarray(self.a_of_eta(eta_arr), dtype=float),
            "z": numpy.asarray(self.z_of_eta(eta_arr), dtype=float),
            "H": numpy.asarray(self.H_of_eta(eta_arr), dtype=float),
            "chi": numpy.asarray(self.chi_of_eta(eta_arr), dtype=float),
            "angular_diameter_distance": numpy.asarray(
                self.da_of_eta(eta_arr), dtype=float
            ),
            "tau": numpy.asarray(self.tau_of_eta(eta_arr), dtype=float),
            "tau_dot": numpy.asarray(
                self.tau_dot_of_eta(eta_arr), dtype=float
            ),
            "visibility": numpy.asarray(
                self.visibility_of_eta(eta_arr), dtype=float
            ),
            "x_e": numpy.asarray(self.x_e_of_eta(eta_arr), dtype=float),
            "sound_speed": numpy.asarray(
                self.sound_speed_of_eta(eta_arr), dtype=float
            ),
        }


@dataclass(slots=True)
class CustomCMBSpectrumData:
    """Internal transfer-component and spectrum payload for CMB outputs."""

    ell_grid: numpy.ndarray
    k_grid: numpy.ndarray
    transfer_components: Mapping[str, numpy.ndarray]
    spectra: Mapping[str, numpy.ndarray]

    @property
    def Delta_l_T(self) -> numpy.ndarray:
        """Return the temperature transfer component when present."""

        return numpy.asarray(
            self.transfer_components.get("temperature", []),
            dtype=float,
        )

    @property
    def Delta_l_E(self) -> numpy.ndarray:
        """Return the E-mode transfer component when present."""

        return numpy.asarray(
            self.transfer_components.get("polarization_e", []),
            dtype=float,
        )

    @property
    def C_l_TT(self) -> numpy.ndarray:
        """Return the TT power spectrum when present."""

        return numpy.asarray(self.spectra.get("TT", []), dtype=float)

    @property
    def C_l_TE(self) -> numpy.ndarray:
        """Return the TE power spectrum when present."""

        return numpy.asarray(self.spectra.get("TE", []), dtype=float)

    @property
    def C_l_EE(self) -> numpy.ndarray:
        """Return the EE power spectrum when present."""

        return numpy.asarray(self.spectra.get("EE", []), dtype=float)


@dataclass(frozen=True, slots=True)
class _DeclaredProjectionKernelBatch:
    """Cache one ell-batched set of spherical-Bessel projection kernels."""

    j_l: numpy.ndarray
    j_l_derivative: numpy.ndarray
    e_kernel: numpy.ndarray
    b_kernel: numpy.ndarray


_CUSTOM_CMB_BACKGROUND_RESULTS: dict[
    tuple[Any, ...], "_CustomCMBBackgroundData"
] = {}
_CUSTOM_CMB_SPECTRUM_RESULTS: dict[
    tuple[Any, ...], "CustomCMBSpectrumData"
] = {}
_CUSTOM_CMB_BESSEL_INPUTS: dict[str, numpy.ndarray] = {}
_CUSTOM_CMB_BESSEL_BATCH_RESULTS: dict[
    tuple[tuple[int, ...], str], _DeclaredProjectionKernelBatch
] = {}


@lru_cache(maxsize=64)
def _get_cached_custom_cmb_background(
    cache_key: tuple[Any, ...],
) -> "_CustomCMBBackgroundData":
    """Return a cached custom background payload."""

    return _CUSTOM_CMB_BACKGROUND_RESULTS[cache_key]


@lru_cache(maxsize=64)
def _get_cached_custom_cmb_spectrum_data(
    cache_key: tuple[Any, ...],
) -> "CustomCMBSpectrumData":
    """Return a cached custom spectrum payload."""

    return _CUSTOM_CMB_SPECTRUM_RESULTS[cache_key]


@lru_cache(maxsize=4096)
def _get_cached_spherical_bessel_values(
    ell: int,
    x_signature: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return cached spherical Bessel values for one ell and x-grid."""

    x_values = _CUSTOM_CMB_BESSEL_INPUTS[x_signature]
    return (
        spherical_jn(int(ell), x_values),
        spherical_jn(int(ell), x_values, derivative=True),
    )


@lru_cache(maxsize=512)
def _get_cached_declared_projection_kernel_batch(
    ell_signature: tuple[int, ...],
    x_signature: str,
) -> _DeclaredProjectionKernelBatch:
    """Return cached ell-batched spherical-Bessel kernels for one x-grid."""

    cache_key = (ell_signature, x_signature)
    cached = _CUSTOM_CMB_BESSEL_BATCH_RESULTS.get(cache_key)
    if cached is not None:
        return cached
    x_values = _CUSTOM_CMB_BESSEL_INPUTS[x_signature]
    shape = (len(ell_signature), x_values.size)
    j_l_matrix = numpy.empty(shape, dtype=float)
    j_l_derivative_matrix = numpy.empty(shape, dtype=float)
    e_kernel = numpy.zeros(shape, dtype=float)
    b_kernel = numpy.zeros(shape, dtype=float)
    inverse_x = 1.0 / numpy.maximum(numpy.abs(x_values), 1.0e-12)
    inverse_x_sq = inverse_x * inverse_x
    for ell_index, ell_value in enumerate(ell_signature):
        j_l, j_l_derivative = _get_cached_spherical_bessel_values(
            int(ell_value),
            x_signature,
        )
        j_l_matrix[ell_index] = j_l
        j_l_derivative_matrix[ell_index] = j_l_derivative
        if int(ell_value) < 2:
            continue
        prefactor = math.exp(
            0.5
            * (
                math.lgamma(int(ell_value) + 3)
                - math.lgamma(int(ell_value) - 1)
            )
        )
        e_kernel[ell_index] = prefactor * j_l * inverse_x_sq
        b_kernel[ell_index] = prefactor * j_l_derivative * inverse_x
    batch = _DeclaredProjectionKernelBatch(
        j_l=j_l_matrix,
        j_l_derivative=j_l_derivative_matrix,
        e_kernel=e_kernel,
        b_kernel=b_kernel,
    )
    _CUSTOM_CMB_BESSEL_BATCH_RESULTS[cache_key] = batch
    return batch


def _custom_cmb_provider_key(background_provider: Any | None) -> int:
    """Return a stable cache key for a custom-CMB background provider."""

    if background_provider is None:
        return 0
    return object.__hash__(background_provider)


_BACKGROUND_CACHE_PHYSICAL_FIELDS = (
    "H0_km_s_Mpc",
    "hubble_ratio",
    "H0_over_c_Mpc_inv",
    "ombh2",
    "omch2",
    "Omega_b0",
    "Omega_c0",
    "Omega_m0_background",
    "Omega_gamma0",
    "Omega_nu0",
    "Omega_r0",
    "Omega_k0",
    "Omega_de0",
    "dark_energy_eos0",
    "dark_energy_eos1",
    "YHe",
    "Neff",
    "z_rec",
    "tau_reio",
    "Tcmb_K",
    "n_b0_m3",
    "n_H0_m3",
    "rho_b0_kg_m3",
    "has_cdm",
    "has_dark_energy",
)


def _background_parameter_values_for_cache(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the declared parameter values that affect the background."""

    section = _get_declared_background_section(contract)
    dependency_names: set[str] = set()
    for raw_value in (section.get("derived", {}) or {}).values():
        if isinstance(raw_value, str) and raw_value.strip():
            dependency_names.update(
                _expression_symbol_names(raw_value.strip())
            )
    reionization = section.get("reionization", {}) or {}
    for raw_value in (reionization.get("quantities", {}) or {}).values():
        if isinstance(raw_value, str) and raw_value.strip():
            dependency_names.update(
                _expression_symbol_names(raw_value.strip())
            )
    target_tau = (reionization.get("calibration", {}) or {}).get(
        "target_optical_depth"
    )
    if isinstance(target_tau, str) and target_tau.strip():
        dependency_names.update(_expression_symbol_names(target_tau.strip()))
    parameter_values: dict[str, Any] = {}
    for source_name in ("param_map", "model_parameters"):
        source = contract.get(source_name, {}) or {}
        if not isinstance(source, Mapping):
            continue
        for dependency_name in sorted(dependency_names):
            if dependency_name in source:
                parameter_values[dependency_name] = source[dependency_name]
    return parameter_values


def _custom_cmb_background_cache_key(
    contract: Mapping[str, Any],
    physical_params: _CustomCMBPhysicalParameters,
    numerics: _CustomCMBNumerics,
    background_provider: Any | None,
) -> tuple[Any, ...]:
    """Return a cache key for the custom CMB background tables."""

    physical_key = tuple(
        (
            field_name,
            _freeze_for_cache(getattr(physical_params, field_name)),
        )
        for field_name in _BACKGROUND_CACHE_PHYSICAL_FIELDS
    )
    return (
        _freeze_for_cache(_get_declared_background_section(contract)),
        _freeze_for_cache(_background_parameter_values_for_cache(contract)),
        physical_key,
        astuple(numerics),
        _custom_cmb_provider_key(background_provider),
    )


def _custom_cmb_spectrum_cache_key(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    background_provider: Any | None,
) -> tuple[Any, ...]:
    """Return a cache key for the custom spectrum transfer data."""

    ell_key = tuple(int(ell) for ell in numpy.asarray(list(ells), dtype=int))
    return (
        _freeze_for_cache(_contract_cache_view(contract_or_params)),
        ell_key,
        _custom_cmb_provider_key(background_provider),
    )


def _contract_cache_view(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the cache-relevant view of a native-runtime contract."""

    if "perturbation_data" not in contract:
        return contract
    return {
        key: value
        for key, value in contract.items()
        if key != "perturbation_data"
    }


def _expression_symbol_name(expression: str) -> str | None:
    """Return the bare symbol name in ``expression`` when one is present."""

    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    body = node.body
    if isinstance(body, ast.Name):
        return body.id
    return None


def _extract_contract_scalar(
    contract: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: float | None = None,
) -> float | None:
    """Return the first finite scalar matching ``names`` from ``contract``."""

    for source in (
        contract.get("param_map", {}) or {},
        contract.get("model_parameters", {}) or {},
    ):
        if not isinstance(source, Mapping):
            continue
        for name in names:
            if name not in source:
                continue
            try:
                return _coerce_numeric_scalar(source[name], name=name)
            except ValueError:
                continue
    if default is None:
        return None
    return float(default)


def _extract_contract_scalar_with_source(
    contract: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[float, str] | None:
    """Return the first finite scalar with its contract source."""

    for source_name in ("param_map", "model_parameters"):
        source = contract.get(source_name, {}) or {}
        if not isinstance(source, Mapping):
            continue
        for name in names:
            if name not in source:
                continue
            try:
                return (
                    _coerce_numeric_scalar(source[name], name=name),
                    f"{source_name}:{name}",
                )
            except ValueError:
                continue
    return None


def _resolve_custom_cmb_numerics(
    contract: Mapping[str, Any],
) -> _CustomCMBNumerics:
    """Return numerical settings for the generic custom CMB solver."""

    raw = contract.get("numerical", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("cmb.numerical must be a mapping when declared")
    defaults = _CustomCMBNumerics()

    def _read_int(name: str, default: int) -> int:
        """Return a positive integer from ``raw`` or ``default``."""

        value = raw.get(name, default)
        numeric = int(_coerce_numeric_scalar(value, name=name))
        if numeric < 1:
            raise ValueError(f"cmb.numerical.{name} must be positive")
        return numeric

    def _read_float(name: str, default: float) -> float:
        """Return a positive float from ``raw`` or ``default``."""

        value = raw.get(name, default)
        numeric = float(_coerce_numeric_scalar(value, name=name))
        if numeric <= 0.0:
            raise ValueError(f"cmb.numerical.{name} must be positive")
        return numeric

    ell_min = max(2, _read_int("ell_min", defaults.ell_min))
    ell_max = max(ell_min, _read_int("ell_max", defaults.ell_max))
    k_min = _read_float("k_min", defaults.k_min)
    k_max = _read_float("k_max", defaults.k_max)
    k_sample_count = max(
        16, _read_int("k_sample_count", defaults.k_sample_count)
    )
    eta_sample_count = max(
        128,
        _read_int("eta_sample_count", defaults.eta_sample_count),
    )
    photon_hierarchy_l_max = max(
        2,
        _read_int(
            "photon_hierarchy_l_max",
            defaults.photon_hierarchy_l_max,
        ),
    )
    neutrino_hierarchy_l_max = max(
        2,
        _read_int(
            "neutrino_hierarchy_l_max",
            defaults.neutrino_hierarchy_l_max,
        ),
    )
    ode_rtol = _read_float("ode_rtol", defaults.ode_rtol)
    ode_atol = _read_float("ode_atol", defaults.ode_atol)
    tight_coupling_ratio = _read_float(
        "tight_coupling_ratio",
        defaults.tight_coupling_ratio,
    )
    a_min = _read_float("a_min", defaults.a_min)
    source_grid_multiplier = max(
        1,
        _read_int(
            "source_grid_multiplier",
            defaults.source_grid_multiplier,
        ),
    )
    initial_redshift = _read_float(
        "initial_redshift",
        defaults.initial_redshift,
    )
    return _CustomCMBNumerics(
        ell_min=ell_min,
        ell_max=ell_max,
        k_min=k_min,
        k_max=k_max,
        k_sample_count=k_sample_count,
        eta_sample_count=eta_sample_count,
        photon_hierarchy_l_max=photon_hierarchy_l_max,
        neutrino_hierarchy_l_max=neutrino_hierarchy_l_max,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        tight_coupling_ratio=tight_coupling_ratio,
        a_min=a_min,
        source_grid_multiplier=source_grid_multiplier,
        initial_redshift=initial_redshift,
    )


def _resolve_custom_cmb_physical_parameters(
    contract: Mapping[str, Any],
    background_provider: Any | None = None,
) -> _CustomCMBPhysicalParameters:
    """Return physical CMB parameters from the structured contract."""

    del background_provider
    background_scalar_context = _resolve_declared_background_context(
        contract,
        a_values=1.0,
        z_values=0.0,
    )
    quantity_provenance: dict[str, str] = {}

    def _record_quantity(
        quantity_name: str,
        source: str,
        *,
        derived_suffix: str | None = None,
    ) -> None:
        """Store one resolved-quantity provenance record."""

        provenance = source
        if derived_suffix is not None:
            provenance = f"derived:{derived_suffix}[{source}]"
        quantity_provenance[quantity_name] = provenance

    def _lookup_quantity(
        quantity_name: str,
    ) -> tuple[float, str] | None:
        """Return one resolved physical quantity and its source."""

        return _lookup_declared_background_scalar_with_source(
            contract,
            background_scalar_context,
            _physical_quantity_names(quantity_name),
        )

    def _require_quantity(
        quantity_name: str,
        *,
        label: str,
    ) -> tuple[float, str]:
        """Return a declared scalar quantity or fail clearly."""

        entry = _lookup_quantity(quantity_name)
        if entry is not None:
            return entry
        names_text = ", ".join(_physical_quantity_names(quantity_name))
        raise ValueError(
            "Declared CMB native execution requires explicit "
            f"{label}. Provide one of: {names_text}"
        )

    hubble_entry = _lookup_quantity("H0_km_s_Mpc")
    if hubble_entry is None:
        hubble_entry = _lookup_declared_background_scalar_with_source(
            contract,
            background_scalar_context,
            ("H",),
        )
        if hubble_entry is None:
            raise ValueError(
                "Declared CMB native execution requires explicit background "
                "H(z) at a=1 or an H0 scalar."
            )
        _record_quantity("H0_km_s_Mpc", hubble_entry[1], derived_suffix="H")
    else:
        _record_quantity("H0_km_s_Mpc", hubble_entry[1])
    hubble_km_s_mpc = hubble_entry[0]
    hubble_km_s_mpc = max(float(hubble_km_s_mpc), 1.0e-6)
    hubble_ratio = hubble_km_s_mpc / 100.0
    hubble_over_c = hubble_km_s_mpc / _C_LIGHT_KM_S

    baryon_entry = _lookup_quantity("Omega_b0")
    ombh2_entry = _lookup_quantity("ombh2")
    if baryon_entry is None and ombh2_entry is not None:
        Omega_b0 = ombh2_entry[0] / (hubble_ratio * hubble_ratio)
        _record_quantity("Omega_b0", ombh2_entry[1], derived_suffix="ombh2")
    elif baryon_entry is not None:
        Omega_b0 = baryon_entry[0]
        _record_quantity("Omega_b0", baryon_entry[1])
    else:
        raise ValueError(
            "Declared CMB native execution requires explicit baryon density."
        )
    if ombh2_entry is None:
        ombh2 = Omega_b0 * hubble_ratio * hubble_ratio
        _record_quantity(
            "ombh2", quantity_provenance["Omega_b0"], derived_suffix="Omega_b0"
        )
    else:
        ombh2 = ombh2_entry[0]
        _record_quantity("ombh2", ombh2_entry[1])

    cdm_entry = _lookup_quantity("Omega_c0")
    omch2_entry = _lookup_quantity("omch2")
    Omega_c0: float | None = None
    omch2: float | None = None
    if cdm_entry is not None:
        Omega_c0 = cdm_entry[0]
        _record_quantity("Omega_c0", cdm_entry[1])
    elif omch2_entry is not None:
        Omega_c0 = omch2_entry[0] / (hubble_ratio * hubble_ratio)
        _record_quantity("Omega_c0", omch2_entry[1], derived_suffix="omch2")
    if omch2_entry is not None:
        omch2 = omch2_entry[0]
        _record_quantity("omch2", omch2_entry[1])
    elif Omega_c0 is not None:
        omch2 = Omega_c0 * hubble_ratio * hubble_ratio
        _record_quantity(
            "omch2",
            quantity_provenance["Omega_c0"],
            derived_suffix="Omega_c0",
        )
    has_cdm = Omega_c0 is not None or omch2 is not None

    Tcmb_K, Tcmb_source = _require_quantity(
        "Tcmb_K",
        label="CMB temperature",
    )
    _record_quantity("Tcmb_K", Tcmb_source)
    YHe, YHe_source = _require_quantity(
        "YHe",
        label="helium fraction",
    )
    _record_quantity("YHe", YHe_source)
    Neff_entry = _lookup_quantity("Neff")
    Neff = None if Neff_entry is None else Neff_entry[0]
    if Neff_entry is not None:
        _record_quantity("Neff", Neff_entry[1])

    photon_entry = _lookup_quantity("Omega_gamma0")
    if photon_entry is None:
        omega_gamma_h2 = 2.469e-5 * (Tcmb_K / 2.7255) ** 4
        Omega_gamma0 = omega_gamma_h2 / (hubble_ratio * hubble_ratio)
        _record_quantity(
            "Omega_gamma0",
            quantity_provenance["Tcmb_K"],
            derived_suffix="Tcmb_K_blackbody",
        )
    else:
        Omega_gamma0 = photon_entry[0]
        _record_quantity("Omega_gamma0", photon_entry[1])

    radiation_entry = _lookup_quantity("Omega_r0")
    neutrino_entry = _lookup_quantity("Omega_nu0")
    Omega_nu0: float | None = None
    Omega_r0: float | None = None
    if neutrino_entry is not None:
        Omega_nu0 = neutrino_entry[0]
        _record_quantity("Omega_nu0", neutrino_entry[1])
    elif radiation_entry is not None:
        Omega_nu0 = radiation_entry[0] - Omega_gamma0
        _record_quantity(
            "Omega_nu0",
            f"derived:Omega_r0_minus_Omega_gamma0[{radiation_entry[1]}]",
        )
    elif Neff is not None:
        Omega_nu0 = max(0.0, Omega_gamma0) * 0.2271 * max(Neff, 0.0)
        _record_quantity(
            "Omega_nu0",
            quantity_provenance["Neff"],
            derived_suffix="Neff_radiation_closure",
        )
    if radiation_entry is not None:
        Omega_r0 = radiation_entry[0]
        _record_quantity("Omega_r0", radiation_entry[1])
    elif Omega_nu0 is not None:
        Omega_r0 = Omega_gamma0 + Omega_nu0
        _record_quantity(
            "Omega_r0",
            quantity_provenance["Omega_gamma0"],
            derived_suffix="Omega_gamma0_plus_Omega_nu0",
        )

    Omega_k0_entry = _lookup_quantity("Omega_k0")
    Omega_k0 = None if Omega_k0_entry is None else Omega_k0_entry[0]
    if Omega_k0_entry is not None:
        _record_quantity("Omega_k0", Omega_k0_entry[1])
    matter_entry = _lookup_quantity("Omega_m0")
    Omega_m0_background = None if matter_entry is None else matter_entry[0]
    if matter_entry is not None:
        _record_quantity("Omega_m0", matter_entry[1])
    dark_energy_entry = _lookup_quantity("Omega_de0")
    Omega_de0 = None if dark_energy_entry is None else dark_energy_entry[0]
    if dark_energy_entry is not None:
        _record_quantity("Omega_de0", dark_energy_entry[1])
    has_dark_energy = Omega_de0 is not None and abs(float(Omega_de0)) > 1.0e-12

    dark_energy_eos0_entry = _lookup_quantity("w0")
    dark_energy_eos0 = (
        None if dark_energy_eos0_entry is None else dark_energy_eos0_entry[0]
    )
    if dark_energy_eos0_entry is not None:
        _record_quantity("w0", dark_energy_eos0_entry[1])
    dark_energy_eos1_entry = _lookup_quantity("wa")
    dark_energy_eos1 = (
        None if dark_energy_eos1_entry is None else dark_energy_eos1_entry[0]
    )
    if dark_energy_eos1_entry is not None:
        _record_quantity("wa", dark_energy_eos1_entry[1])

    primordial_amplitude, primordial_amplitude_source = _require_quantity(
        "primordial_amplitude",
        label="primordial amplitude",
    )
    _record_quantity(
        "primordial_amplitude",
        primordial_amplitude_source,
    )
    primordial_spectral_index, primordial_tilt_source = _require_quantity(
        "primordial_spectral_index",
        label="primordial tilt",
    )
    _record_quantity(
        "primordial_spectral_index",
        primordial_tilt_source,
    )
    z_rec = _lookup_declared_background_scalar(
        contract,
        background_scalar_context,
        ("z_rec",),
    )
    if z_rec is None or z_rec <= 0.0:
        z_rec = 0.0
    tau_reio = _lookup_declared_background_scalar(
        contract,
        background_scalar_context,
        ("tau", "tau_reio", "reionization_tau"),
    )
    if tau_reio is None:
        tau_reio = 0.0

    G_NEWTON = 6.674_30e-11
    MPC_M = 3.085_677_581_491_3673e22
    hubble_si = hubble_km_s_mpc * 1000.0 / MPC_M
    rho_crit0 = 3.0 * hubble_si * hubble_si / (8.0 * math.pi * G_NEWTON)
    rho_b0 = Omega_b0 * rho_crit0
    n_b0_m3 = rho_b0 / 1.672_621_923_69e-27
    n_H0_m3 = n_b0_m3 * max(0.0, 1.0 - YHe)
    return _CustomCMBPhysicalParameters(
        H0_km_s_Mpc=hubble_km_s_mpc,
        hubble_ratio=hubble_ratio,
        H0_over_c_Mpc_inv=hubble_over_c,
        ombh2=ombh2,
        omch2=omch2,
        Omega_b0=Omega_b0,
        Omega_c0=Omega_c0,
        Omega_m0_background=Omega_m0_background,
        Omega_gamma0=Omega_gamma0,
        Omega_nu0=Omega_nu0,
        Omega_r0=Omega_r0,
        Omega_k0=Omega_k0,
        Omega_de0=Omega_de0,
        dark_energy_eos0=dark_energy_eos0,
        dark_energy_eos1=dark_energy_eos1,
        YHe=YHe,
        Neff=Neff,
        primordial_amplitude=primordial_amplitude,
        primordial_spectral_index=primordial_spectral_index,
        z_rec=z_rec,
        tau_reio=tau_reio,
        Tcmb_K=Tcmb_K,
        n_b0_m3=n_b0_m3,
        n_H0_m3=n_H0_m3,
        rho_b0_kg_m3=rho_b0,
        has_cdm=has_cdm,
        has_dark_energy=has_dark_energy,
        quantity_provenance=tuple(sorted(quantity_provenance.items())),
    )


def _build_custom_cmb_background(
    contract: Mapping[str, Any],
    physical_params: _CustomCMBPhysicalParameters,
    numerics: _CustomCMBNumerics,
    *,
    background_provider: Any | None = None,
) -> _CustomCMBBackgroundData:
    """Return the interpolated background and recombination solution."""

    cache_key = _custom_cmb_background_cache_key(
        contract,
        physical_params,
        numerics,
        background_provider,
    )
    cached_background = _CUSTOM_CMB_BACKGROUND_RESULTS.get(cache_key)
    if cached_background is not None:
        return _get_cached_custom_cmb_background(cache_key)
    del background_provider

    MPC_M = 3.085_677_581_491_3673e22
    SIGMA_T_M2 = 6.652_458_7321e-29

    a_min = max(numerics.a_min, 1.0e-8)
    log_a = numpy.geomspace(a_min, 1.0, numerics.eta_sample_count)

    helium_ionization_energy_j = 24.587_387 * 1.602_176_634e-19
    helium_double_ionization_energy_j = 54.417_763 * 1.602_176_634e-19
    boltzmann_j_k = 1.380_649e-23
    planck_j_s = 6.626_070_15e-34
    electron_mass_kg = 9.109_383_7015e-31
    helium_number_ratio = max(
        0.0,
        physical_params.YHe / (4.0 * max(1.0 - physical_params.YHe, 1.0e-6)),
    )
    recombination_window = numpy.geomspace(
        1.0 / 5_000.0,
        1.0 / 30.0,
        max(256, numerics.eta_sample_count),
    )
    reionization_window = numpy.geomspace(
        1.0 / 30.0,
        1.0,
        max(128, numerics.eta_sample_count // 2),
    )
    a_grid = numpy.unique(
        numpy.clip(
            numpy.concatenate(
                (log_a, recombination_window, reionization_window)
            ),
            a_min,
            1.0,
        )
    )
    a_grid.sort()
    z_grid = numpy.maximum(1.0 / a_grid - 1.0, 0.0)
    background_grid_context = _resolve_declared_background_context(
        contract,
        a_values=a_grid,
        z_values=z_grid,
    )

    def _coerce_background_history(
        *,
        names: Sequence[str],
        label: str,
    ) -> tuple[numpy.ndarray, str] | None:
        """Return one declared background history on the LOS grid."""

        for name in names:
            if name not in background_grid_context:
                continue
            history = numpy.asarray(background_grid_context[name], dtype=float)
            if history.ndim == 0:
                history = numpy.full_like(z_grid, float(history), dtype=float)
            if history.shape != z_grid.shape:
                raise ValueError(
                    "Declared CMB background quantity produced an invalid "
                    f"shape for {label}: {name}"
                )
            if not numpy.all(numpy.isfinite(history)):
                raise ValueError(
                    "Declared CMB background quantity produced non-finite "
                    f"values for {label}: {name}"
                )
            return history, name
        return None

    hubble_entry = _coerce_background_history(
        names=("H", "expansion_rate"),
        label="expansion rate",
    )
    if hubble_entry is None:
        raise ValueError(
            "Declared CMB background must provide a derived expansion "
            "history such as 'H'."
        )
    if numpy.any(hubble_entry[0] <= 0.0):
        raise ValueError(
            "Declared CMB background expansion history must stay positive."
        )
    H_grid = numpy.asarray(hubble_entry[0], dtype=float)
    eta_grid = cumulative_trapezoid(
        _C_LIGHT_KM_S / numpy.maximum(a_grid * a_grid * H_grid, 1.0e-12),
        a_grid,
        initial=0.0,
    )
    eta_grid = numpy.asarray(eta_grid, dtype=float)
    eta0 = float(eta_grid[-1])

    z_asc = z_grid[::-1]
    H_asc = H_grid[::-1]
    chi_asc = cumulative_trapezoid(
        _C_LIGHT_KM_S / numpy.maximum(H_asc, 1.0e-12),
        z_asc,
        initial=0.0,
    )
    derived_chi_grid = chi_asc[::-1]
    chi_entry = _coerce_background_history(
        names=_physical_quantity_names("chi"),
        label="comoving distance",
    )
    angular_diameter_entry = _coerce_background_history(
        names=_physical_quantity_names("angular_diameter_distance"),
        label="angular diameter distance",
    )
    if (chi_entry is None) != (angular_diameter_entry is None):
        raise ValueError(
            "Declared CMB background curvature override must provide both "
            "chi and angular_diameter_distance together."
        )
    if chi_entry is not None and angular_diameter_entry is not None:
        chi_grid = numpy.asarray(chi_entry[0], dtype=float)
        da_grid = numpy.asarray(angular_diameter_entry[0], dtype=float)
        if numpy.any(chi_grid < 0.0) or numpy.any(da_grid < 0.0):
            raise ValueError(
                "Declared CMB background curvature histories must stay "
                "non-negative."
            )
    else:
        chi_grid = derived_chi_grid
        omega_k = physical_params.Omega_k0
        if omega_k is None:
            raise ValueError(
                "Declared CMB background must declare Omega_k0 when "
                "curvature distances are not provided explicitly."
            )
        if abs(omega_k) > 1.0e-8:
            sqrt_ok = math.sqrt(abs(omega_k))
            hubble_distance_mpc = _C_LIGHT_KM_S / physical_params.H0_km_s_Mpc
            arg = sqrt_ok * chi_grid / max(hubble_distance_mpc, 1.0e-12)
            if omega_k > 0.0:
                d_m = hubble_distance_mpc / sqrt_ok * numpy.sinh(arg)
            else:
                d_m = hubble_distance_mpc / sqrt_ok * numpy.sin(arg)
        else:
            d_m = chi_grid
        da_grid = numpy.divide(d_m, 1.0 + z_grid, dtype=float)
    n_H_grid = physical_params.n_H0_m3 * numpy.power(1.0 + z_grid, 3.0)
    if physical_params.Omega_gamma0 <= 0.0:
        raise ValueError(
            "Declared CMB native execution requires a positive photon "
            "density."
        )

    def _safe_exp(log_value: float) -> float:
        """Return ``exp(log_value)`` while avoiding overflow."""

        if log_value > 700.0:
            return float("inf")
        if log_value < -700.0:
            return 0.0
        return math.exp(log_value)

    def _saha_ratio(
        temperature_k: float,
        ionization_energy_j: float,
        statistical_weight_ratio: float,
    ) -> float:
        """Return the Saha equilibrium ratio for a bound state transition."""

        if temperature_k <= 0.0:
            return 0.0
        thermal_prefactor = (
            2.0
            * math.pi
            * electron_mass_kg
            * boltzmann_j_k
            * temperature_k
            / (planck_j_s * planck_j_s)
        )
        log_ratio = 1.5 * math.log(max(thermal_prefactor, 1.0e-300))
        log_ratio += math.log(max(statistical_weight_ratio, 1.0e-12))
        log_ratio -= ionization_energy_j / (boltzmann_j_k * temperature_k)
        return _safe_exp(log_ratio)

    def _hydrogen_alpha_coefficient(temperature_k: float) -> float:
        """Return the case-B hydrogen recombination coefficient in m^3/s."""

        temperature_10k_ratio = max(temperature_k / 1.0e4, 1.0e-4)
        numerator = 4.309e-19 * temperature_10k_ratio**-0.6166
        denominator = 1.0 + 0.6703 * temperature_10k_ratio**0.5300
        return numerator / denominator

    def _helium_electron_fraction(
        z_value: float,
        hydrogen_fraction: float,
        n_h_value: float,
    ) -> tuple[float, float]:
        """Return the total and helium electron fractions at ``z_value``."""

        temperature_k = physical_params.Tcmb_K * (1.0 + z_value)
        total_fraction = hydrogen_fraction + 2.0 * helium_number_ratio
        helium_fraction = 2.0 * helium_number_ratio
        for _ in range(5):
            n_e_value = max(total_fraction * n_h_value, 1.0e-30)
            he_ii_ratio = (
                _saha_ratio(
                    temperature_k,
                    helium_ionization_energy_j,
                    4.0,
                )
                / n_e_value
            )
            he_iii_ratio = (
                _saha_ratio(
                    temperature_k,
                    helium_double_ionization_energy_j,
                    1.0,
                )
                / n_e_value
            )
            he_denominator = 1.0 + he_ii_ratio + he_ii_ratio * he_iii_ratio
            helium_fraction = (
                helium_number_ratio
                * (he_ii_ratio + 2.0 * he_ii_ratio * he_iii_ratio)
                / he_denominator
            )
            updated_fraction = hydrogen_fraction + helium_fraction
            if abs(updated_fraction - total_fraction) <= 1.0e-8 * max(
                1.0, updated_fraction
            ):
                total_fraction = updated_fraction
                break
            total_fraction = 0.5 * (total_fraction + updated_fraction)
        return total_fraction, helium_fraction

    hydrogen_ground_state_energy_j = 13.605_693_122_994
    hydrogen_ground_state_energy_j *= 1.602_176_634e-19
    hydrogen_n2_binding_energy_j = 0.25 * hydrogen_ground_state_energy_j
    lyman_alpha_energy_j = (
        hydrogen_ground_state_energy_j - hydrogen_n2_binding_energy_j
    )
    lambda_alpha_m = planck_j_s * 299_792_458.0 / lyman_alpha_energy_j
    hydrogen_two_photon_decay_rate = 8.22458
    hydrogen_rate_grid = H_grid * 1000.0 / MPC_M

    def _hydrogen_saha_fraction(
        z_value: float,
        helium_fraction: float,
        n_h_value: float,
    ) -> float:
        """Return the Saha hydrogen ionization fraction at ``z_value``."""

        temperature_k = physical_params.Tcmb_K * (1.0 + z_value)
        saha_ratio = _saha_ratio(
            temperature_k,
            hydrogen_ground_state_energy_j,
            1.0,
        ) / max(n_h_value, 1.0e-30)
        coefficient_b = helium_fraction + saha_ratio
        discriminant = coefficient_b * coefficient_b + 4.0 * saha_ratio
        return float(
            numpy.clip(
                0.5 * (-coefficient_b + math.sqrt(max(discriminant, 0.0))),
                1.0e-8,
                1.0,
            )
        )

    def _hydrogen_recombination_rate(
        *,
        a_value: float,
        hydrogen_fraction: float,
        z_value: float,
        n_h_value: float,
        hubble_rate: float,
    ) -> float:
        """Return the Peebles hydrogen derivative with respect to ``a``."""

        temperature_k = physical_params.Tcmb_K * (1.0 + z_value)
        alpha_b = _hydrogen_alpha_coefficient(temperature_k)
        beta_n2 = alpha_b * _saha_ratio(
            temperature_k,
            hydrogen_n2_binding_energy_j,
            1.0,
        )
        beta_continuum = beta_n2 * math.exp(
            -lyman_alpha_energy_j / (boltzmann_j_k * temperature_k)
        )
        total_fraction, _ = _helium_electron_fraction(
            z_value,
            float(numpy.clip(hydrogen_fraction, 1.0e-8, 1.0)),
            n_h_value,
        )
        neutral_fraction = max(1.0 - hydrogen_fraction, 1.0e-12)
        peebles_k = lambda_alpha_m**3 / (
            8.0 * math.pi * max(hubble_rate, 1.0e-30)
        )
        peebles_c = (
            1.0
            + peebles_k
            * hydrogen_two_photon_decay_rate
            * n_h_value
            * neutral_fraction
        ) / (
            1.0
            + peebles_k
            * (hydrogen_two_photon_decay_rate + beta_n2)
            * n_h_value
            * neutral_fraction
        )
        dx_dt = peebles_c * (
            beta_continuum * neutral_fraction
            - n_h_value * alpha_b * total_fraction * hydrogen_fraction
        )
        return dx_dt / max(a_value * hubble_rate, 1.0e-30)

    hydrogen_saha_grid = numpy.ones_like(z_grid, dtype=float)
    hydrogen_recombination_start_index = int(
        numpy.searchsorted(
            a_grid,
            float(recombination_window[0]),
            side="left",
        )
    )
    saha_break_index = z_grid.size
    peebles_switch_fraction = 0.99
    hydrogen_guess = 1.0
    for index in range(hydrogen_recombination_start_index, z_grid.size):
        z_value = float(z_grid[index])
        n_h_value = float(n_H_grid[index])
        _, helium_fraction_guess = _helium_electron_fraction(
            z_value,
            hydrogen_guess,
            n_h_value,
        )
        hydrogen_saha_grid[index] = _hydrogen_saha_fraction(
            z_value,
            float(helium_fraction_guess),
            n_h_value,
        )
        hydrogen_guess = float(hydrogen_saha_grid[index])
        if (
            saha_break_index == z_grid.size
            and float(hydrogen_saha_grid[index]) < peebles_switch_fraction
        ):
            saha_break_index = index

    x_h_grid = numpy.empty_like(z_grid, dtype=float)
    if saha_break_index > 0:
        x_h_grid[:saha_break_index] = hydrogen_saha_grid[:saha_break_index]
    if saha_break_index < z_grid.size:
        initial_fraction = float(hydrogen_saha_grid[saha_break_index])

        def _hydrogen_ode(
            a_value: float,
            state: numpy.ndarray,
        ) -> numpy.ndarray:
            """Return the stiff hydrogen recombination derivative."""

            z_value = float(numpy.interp(a_value, a_grid, z_grid))
            n_h_value = float(numpy.interp(a_value, a_grid, n_H_grid))
            hubble_rate = float(
                numpy.interp(a_value, a_grid, hydrogen_rate_grid)
            )
            derivative = _hydrogen_recombination_rate(
                a_value=float(a_value),
                hydrogen_fraction=float(numpy.clip(state[0], 1.0e-8, 1.0)),
                z_value=z_value,
                n_h_value=n_h_value,
                hubble_rate=hubble_rate,
            )
            return numpy.asarray((derivative,), dtype=float)

        recombination_solution = solve_ivp(
            _hydrogen_ode,
            (
                float(a_grid[saha_break_index]),
                float(a_grid[-1]),
            ),
            numpy.asarray((initial_fraction,), dtype=float),
            method="Radau",
            t_eval=numpy.asarray(a_grid[saha_break_index:], dtype=float),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        if not recombination_solution.success:
            raise ValueError(
                "Hydrogen recombination integration failed: "
                f"{recombination_solution.message}"
            )
        x_h_grid[saha_break_index:] = numpy.clip(
            recombination_solution.y[0],
            1.0e-8,
            1.0,
        )
    else:
        x_h_grid[:] = hydrogen_saha_grid
    helium_electron_grid = numpy.empty_like(z_grid, dtype=float)
    x_e_recomb_grid = numpy.empty_like(z_grid, dtype=float)
    for index, (z_value, n_h_value, hydrogen_fraction) in enumerate(
        zip(z_grid, n_H_grid, x_h_grid, strict=True)
    ):
        total_fraction, helium_fraction = _helium_electron_fraction(
            float(z_value),
            float(hydrogen_fraction),
            float(n_h_value),
        )
        helium_electron_grid[index] = helium_fraction
        x_e_recomb_grid[index] = total_fraction

    reionization_section = _get_declared_reionization_section(contract)
    calibration_section = reionization_section.get("calibration", {}) or {}
    reionization_quantities = reionization_section.get("quantities", {}) or {}
    background_runtime = contract.get("background_runtime")
    hubble0_si = physical_params.H0_km_s_Mpc * 1000.0 / MPC_M
    helium_floor_grid = numpy.minimum(
        helium_electron_grid,
        helium_number_ratio,
    )

    def _resolve_reionization_target_tau() -> float | None:
        """Return the declared reionization optical-depth target."""

        target_entry = calibration_section.get("target_optical_depth")
        if background_runtime is not None:
            target_entry = getattr(
                background_runtime,
                "reionization_target_tau",
                target_entry,
            )
        if target_entry is None:
            return None
        scalar_context = _resolve_declared_background_context(
            contract,
            a_values=1.0,
            z_values=0.0,
        )
        if isinstance(
            target_entry, (int, float, numpy.integer, numpy.floating)
        ):
            return float(target_entry)
        if hasattr(target_entry, "program"):
            return _coerce_numeric_scalar(
                evaluate_compiled_expression(target_entry, scalar_context),
                name=(
                    "background.reionization.calibration."
                    "target_optical_depth"
                ),
            )
        if not isinstance(target_entry, str) or not target_entry.strip():
            raise ValueError(
                "background.reionization.calibration.target_optical_depth "
                "must be numeric or a string expression."
            )
        return _coerce_numeric_scalar(
            _evaluate_safe_expression(target_entry.strip(), scalar_context),
            name="background.reionization.calibration.target_optical_depth",
        )

    calibration_symbol = (
        getattr(
            background_runtime,
            "reionization_calibration_symbol",
            calibration_section.get("symbol"),
        )
        if background_runtime is not None
        else calibration_section.get("symbol")
    )
    if calibration_symbol is not None:
        if (
            not isinstance(calibration_symbol, str)
            or not calibration_symbol.strip()
        ):
            raise ValueError(
                "background.reionization.calibration.symbol must be a "
                "non-empty string."
            )
        calibration_symbol = calibration_symbol.strip()
    target_reionization_tau = _resolve_reionization_target_tau()
    if target_reionization_tau is not None:
        if not reionization_quantities:
            raise ValueError(
                "Declared reionization calibration requires "
                "background.reionization.quantities."
            )
        if calibration_symbol is None:
            raise ValueError(
                "background.reionization.calibration.symbol is required "
                "when target_optical_depth is declared."
            )
        if (
            "lower" not in calibration_section
            or "upper" not in calibration_section
        ):
            raise ValueError(
                "background.reionization.calibration must declare lower "
                "and upper bounds."
            )
        calibration_lower = _coerce_numeric_scalar(
            calibration_section["lower"],
            name="background.reionization.calibration.lower",
        )
        calibration_upper = _coerce_numeric_scalar(
            calibration_section["upper"],
            name="background.reionization.calibration.upper",
        )
        if calibration_upper <= calibration_lower:
            raise ValueError(
                "background.reionization.calibration.upper must be "
                "greater than lower."
            )
    else:
        calibration_lower = 0.0
        calibration_upper = 0.0

    def _helium_recombination_coefficient(temperature_k: float) -> float:
        """Return an effective case-B He I recombination coefficient."""

        temperature_10k_ratio = max(temperature_k / 1.0e4, 1.0e-4)
        return 1.54e-19 * temperature_10k_ratio**-0.486

    def _helium_double_recombination_coefficient(
        temperature_k: float,
    ) -> float:
        """Return the hydrogenic case-B coefficient for He III -> He II."""

        return 4.0 * _hydrogen_alpha_coefficient(temperature_k / 4.0)

    def _resolve_stage_reionization_quantities(
        *,
        a_value: float,
        z_value: float,
        n_h_value: float,
        x_h_floor: float,
        helium_electron_floor: float,
        x_e_floor: float,
        hubble_rate: float,
        calibration_value: float | None,
    ) -> Mapping[str, Any]:
        """Return the resolved declared reionization quantities."""

        if not reionization_quantities:
            return {}
        background_context = _resolve_declared_background_context(
            contract,
            a_values=float(a_value),
            z_values=float(z_value),
        )
        reionization_context = dict(background_context)
        reionization_context.update(
            {
                "n_H": float(n_h_value),
                "x_h_floor": float(x_h_floor),
                "helium_electron_floor": float(helium_electron_floor),
                "x_e_floor": float(x_e_floor),
                "neutral_h_floor": max(1.0 - float(x_h_floor), 0.0),
                "neutral_he_floor": max(
                    helium_number_ratio - float(helium_electron_floor),
                    0.0,
                ),
                "helium_number_ratio": float(helium_number_ratio),
                "H_SI": float(hubble_rate),
                "H0_SI": float(hubble0_si),
            }
        )
        if calibration_symbol is not None and calibration_value is not None:
            reionization_context[calibration_symbol] = float(calibration_value)
        return _resolve_declared_reionization_context(
            contract,
            base_context=reionization_context,
        )

    def _reionization_state_da(
        a_value: float,
        state_vector: numpy.ndarray,
        *,
        calibration_value: float | None,
        a_left: float,
        a_right: float,
        floor_h_left: float,
        floor_h_right: float,
        floor_he_left: float,
        floor_he_right: float,
        x_e_floor_left: float,
        x_e_floor_right: float,
        n_h_left: float,
        n_h_right: float,
        rate_left: float,
        rate_right: float,
    ) -> numpy.ndarray:
        """Return reionization-state derivatives with respect to ``a``."""

        interval = max(a_right - a_left, 1.0e-30)
        blend = numpy.clip((a_value - a_left) / interval, 0.0, 1.0)
        floor_h = float((1.0 - blend) * floor_h_left + blend * floor_h_right)
        floor_he = float(
            (1.0 - blend) * floor_he_left + blend * floor_he_right
        )
        x_e_floor = float(
            (1.0 - blend) * x_e_floor_left + blend * x_e_floor_right
        )
        n_h_value = float((1.0 - blend) * n_h_left + blend * n_h_right)
        hubble_rate = float((1.0 - blend) * rate_left + blend * rate_right)
        delta_h = float(
            numpy.clip(
                state_vector[0],
                0.0,
                max(1.0 - floor_h, 0.0),
            )
        )
        delta_he_double = float(
            numpy.clip(
                state_vector[2],
                0.0,
                helium_number_ratio,
            )
        )
        delta_he_single = float(
            numpy.clip(
                state_vector[1],
                0.0,
                max(helium_number_ratio - delta_he_double, 0.0),
            )
        )
        total_x_e = (
            x_e_floor + delta_h + delta_he_single + 2.0 * delta_he_double
        )
        n_e_value = max(total_x_e * n_h_value, 1.0e-30)
        neutral_h = max(1.0 - floor_h - delta_h, 0.0)
        neutral_he = max(
            helium_number_ratio - floor_he - delta_he_single - delta_he_double,
            0.0,
        )
        reionization_context = _resolve_stage_reionization_quantities(
            a_value=float(a_value),
            z_value=z_value,
            n_h_value=n_h_value,
            x_h_floor=floor_h,
            helium_electron_floor=floor_he,
            x_e_floor=x_e_floor,
            hubble_rate=hubble_rate,
            calibration_value=calibration_value,
        )
        if not reionization_context:
            return numpy.zeros(3, dtype=float)
        for required_name in (
            "hydrogen_ionization_rate",
            "helium_ionization_rate",
            "helium_double_ionization_rate",
            "hydrogen_temperature_K",
            "helium_temperature_K",
            "helium_double_temperature_K",
        ):
            if required_name not in reionization_context:
                raise ValueError(
                    "Declared reionization quantities must define "
                    f"'{required_name}'."
                )
        gamma_h = _coerce_numeric_scalar(
            reionization_context["hydrogen_ionization_rate"],
            name="background.reionization.quantities.hydrogen_ionization_rate",
        )
        gamma_he = _coerce_numeric_scalar(
            reionization_context["helium_ionization_rate"],
            name="background.reionization.quantities.helium_ionization_rate",
        )
        gamma_he_double = _coerce_numeric_scalar(
            reionization_context["helium_double_ionization_rate"],
            name=(
                "background.reionization.quantities."
                "helium_double_ionization_rate"
            ),
        )
        hydrogen_temperature_k = _coerce_numeric_scalar(
            reionization_context["hydrogen_temperature_K"],
            name="background.reionization.quantities.hydrogen_temperature_K",
        )
        helium_temperature_k = _coerce_numeric_scalar(
            reionization_context["helium_temperature_K"],
            name="background.reionization.quantities.helium_temperature_K",
        )
        helium_double_temperature_k = _coerce_numeric_scalar(
            reionization_context["helium_double_temperature_K"],
            name=(
                "background.reionization.quantities."
                "helium_double_temperature_K"
            ),
        )
        alpha_h = _hydrogen_alpha_coefficient(hydrogen_temperature_k)
        alpha_he = _helium_recombination_coefficient(helium_temperature_k)
        alpha_he_double = _helium_double_recombination_coefficient(
            helium_double_temperature_k
        )
        hydrogen_dt = gamma_h * neutral_h - alpha_h * n_e_value * delta_h
        helium_single_dt = (
            gamma_he * neutral_he
            - alpha_he * n_e_value * delta_he_single
            - gamma_he_double * delta_he_single
            + alpha_he_double * n_e_value * delta_he_double
        )
        helium_double_dt = (
            gamma_he_double * delta_he_single
            - alpha_he_double * n_e_value * delta_he_double
        )
        return numpy.asarray(
            (
                hydrogen_dt,
                helium_single_dt,
                helium_double_dt,
            ),
            dtype=float,
        ) / max(a_value * hubble_rate, 1.0e-30)

    def _integrate_reionization_history(
        calibration_value: float | None,
    ) -> tuple[numpy.ndarray, float, float]:
        """Return reionization electrons, tau, and midpoint redshift."""

        if not reionization_quantities:
            return (
                numpy.zeros_like(z_grid, dtype=float),
                0.0,
                0.0,
            )
        state = numpy.zeros(3, dtype=float)
        delta_h_grid = numpy.zeros_like(z_grid, dtype=float)
        delta_he_single_grid = numpy.zeros_like(z_grid, dtype=float)
        delta_he_double_grid = numpy.zeros_like(z_grid, dtype=float)
        for index, a_value in enumerate(a_grid):
            delta_h_grid[index] = state[0]
            delta_he_single_grid[index] = state[1]
            delta_he_double_grid[index] = state[2]
            if index == a_grid.size - 1:
                break
            a_left = float(a_value)
            a_right = float(a_grid[index + 1])
            step = a_right - a_left
            if step <= 0.0:
                continue

            def _stage_derivative(
                stage_a: float,
                stage_state: numpy.ndarray,
            ) -> numpy.ndarray:
                """Return one RK4 stage for the reionization state."""

                return _reionization_state_da(
                    stage_a,
                    stage_state,
                    calibration_value=calibration_value,
                    a_left=a_left,
                    a_right=a_right,
                    floor_h_left=float(x_h_grid[index]),
                    floor_h_right=float(x_h_grid[index + 1]),
                    floor_he_left=float(helium_floor_grid[index]),
                    floor_he_right=float(helium_floor_grid[index + 1]),
                    x_e_floor_left=float(x_e_recomb_grid[index]),
                    x_e_floor_right=float(x_e_recomb_grid[index + 1]),
                    n_h_left=float(n_H_grid[index]),
                    n_h_right=float(n_H_grid[index + 1]),
                    rate_left=float(hydrogen_rate_grid[index]),
                    rate_right=float(hydrogen_rate_grid[index + 1]),
                )

            slope_start = _stage_derivative(a_left, state)
            midpoint_a = a_left + 0.5 * step
            slope_mid_a = _stage_derivative(
                midpoint_a,
                state + 0.5 * step * slope_start,
            )
            slope_mid_b = _stage_derivative(
                midpoint_a,
                state + 0.5 * step * slope_mid_a,
            )
            slope_end = _stage_derivative(
                a_right,
                state + step * slope_mid_b,
            )
            candidate_state = state + (step / 6.0) * (
                slope_start + 2.0 * slope_mid_a + 2.0 * slope_mid_b + slope_end
            )
            if not numpy.all(numpy.isfinite(candidate_state)):
                raise ValueError(
                    "Physical reionization history produced non-finite "
                    "state values"
                )
            state = numpy.asarray(candidate_state, dtype=float)
            state[0] = float(
                numpy.clip(
                    state[0],
                    0.0,
                    max(1.0 - float(x_h_grid[index + 1]), 0.0),
                )
            )
            state[2] = float(
                numpy.clip(
                    state[2],
                    0.0,
                    helium_number_ratio,
                )
            )
            state[1] = float(
                numpy.clip(
                    state[1],
                    0.0,
                    max(helium_number_ratio - state[2], 0.0),
                )
            )

        reionization_xe_grid = (
            delta_h_grid + delta_he_single_grid + 2.0 * delta_he_double_grid
        )
        tau_reion_grid = -cumulative_trapezoid(
            (a_grid * n_H_grid * reionization_xe_grid * SIGMA_T_M2 * MPC_M)[
                ::-1
            ],
            eta_grid[::-1],
            initial=0.0,
        )[::-1]
        ionization_progress = numpy.maximum.accumulate(reionization_xe_grid)
        midpoint_electrons = 0.5 * float(ionization_progress[-1])
        midpoint_z = 0.0
        if midpoint_electrons > 0.0:
            midpoint_z = float(
                numpy.interp(
                    midpoint_electrons,
                    ionization_progress,
                    z_grid,
                )
            )
        return (
            numpy.asarray(reionization_xe_grid, dtype=float),
            float(tau_reion_grid[0]),
            midpoint_z,
        )

    reionization_history_cache: dict[
        float | None, tuple[numpy.ndarray, float, float]
    ] = {}

    def _get_reionization_history(
        calibration_value: float | None,
    ) -> tuple[numpy.ndarray, float, float]:
        """Return the cached reionization history for one calibration value."""

        if calibration_value not in reionization_history_cache:
            reionization_history_cache[calibration_value] = (
                _integrate_reionization_history(calibration_value)
            )
        return reionization_history_cache[calibration_value]

    if target_reionization_tau is not None:
        _, lower_tau, _ = _get_reionization_history(calibration_lower)
        _, upper_tau, _ = _get_reionization_history(calibration_upper)
        lower_offset = lower_tau - target_reionization_tau
        upper_offset = upper_tau - target_reionization_tau
        if lower_offset == 0.0:
            chosen_calibration = calibration_lower
        elif upper_offset == 0.0:
            chosen_calibration = calibration_upper
        elif lower_offset * upper_offset > 0.0:
            raise ValueError(
                "Declared reionization calibration range does not bracket "
                "the requested optical depth."
            )
        else:
            chosen_calibration = float(
                brentq(
                    lambda value: _get_reionization_history(value)[1]
                    - target_reionization_tau,
                    calibration_lower,
                    calibration_upper,
                    maxiter=96,
                )
            )
        reionization_xe_grid, reionization_tau, z_reion = (
            _get_reionization_history(chosen_calibration)
        )
    else:
        reionization_xe_grid, reionization_tau, z_reion = (
            _get_reionization_history(None)
        )

    x_e_grid = numpy.clip(
        x_e_recomb_grid + reionization_xe_grid,
        1.0e-8,
        1.0 + 2.0 * helium_number_ratio + 1.0e-6,
    )
    n_e_grid = x_e_grid * n_H_grid
    tau_dot_grid = -a_grid * n_e_grid * SIGMA_T_M2 * MPC_M
    tau_grid = -cumulative_trapezoid(
        (-tau_dot_grid)[::-1],
        eta_grid[::-1],
        initial=0.0,
    )[::-1]
    tau_grid = numpy.minimum(tau_grid, 700.0)
    visibility_grid = -tau_dot_grid * numpy.exp(-tau_grid)
    peak_index = int(numpy.argmax(visibility_grid))
    peak_z = float(z_grid[peak_index])
    visibility_integral = float(
        cumulative_trapezoid(visibility_grid, eta_grid, initial=0.0)[-1]
    )
    if not numpy.isfinite(visibility_integral) or visibility_integral <= 0.0:
        raise ValueError("Failed to construct a physical visibility function")
    sound_speed_grid = _C_LIGHT_KM_S / numpy.sqrt(
        3.0
        * (
            1.0
            + 3.0
            * physical_params.Omega_b0
            * a_grid
            / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
        )
    )
    a_rec_value = float(a_grid[peak_index])
    eta_rec = float(eta_grid[peak_index])
    sound_speed_over_a2_h = sound_speed_grid / numpy.maximum(
        a_grid * a_grid * H_grid,
        1.0e-12,
    )
    sound_horizon_grid = cumulative_trapezoid(
        sound_speed_over_a2_h,
        a_grid,
        initial=0.0,
    )
    sound_horizon_mpc = float(
        numpy.interp(
            a_rec_value,
            a_grid,
            sound_horizon_grid,
            left=float(sound_horizon_grid[0]),
            right=float(sound_horizon_grid[-1]),
        )
    )

    eta_of_a = PchipInterpolator(a_grid, eta_grid, extrapolate=True)
    a_of_eta = PchipInterpolator(eta_grid, a_grid, extrapolate=True)
    z_of_eta = PchipInterpolator(eta_grid, z_grid, extrapolate=True)
    H_of_eta = PchipInterpolator(eta_grid, H_grid, extrapolate=True)
    chi_of_eta = PchipInterpolator(eta_grid, chi_grid, extrapolate=True)
    da_of_eta = PchipInterpolator(eta_grid, da_grid, extrapolate=True)
    tau_of_eta = PchipInterpolator(eta_grid, tau_grid, extrapolate=True)
    tau_dot_of_eta = PchipInterpolator(
        eta_grid,
        tau_dot_grid,
        extrapolate=True,
    )
    visibility_of_eta = PchipInterpolator(
        eta_grid,
        visibility_grid,
        extrapolate=True,
    )
    x_e_of_eta = PchipInterpolator(eta_grid, x_e_grid, extrapolate=True)
    sound_speed_of_eta = PchipInterpolator(
        eta_grid,
        sound_speed_grid,
        extrapolate=True,
    )

    background_data = _CustomCMBBackgroundData(
        a_grid=a_grid,
        z_grid=z_grid,
        eta_grid=eta_grid,
        eta0=eta0,
        chi_grid=chi_grid,
        da_grid=da_grid,
        H_grid=H_grid,
        Hconf_grid=a_grid * H_grid / _C_LIGHT_KM_S,
        tau_grid=tau_grid,
        tau_dot_grid=tau_dot_grid,
        visibility_grid=visibility_grid,
        x_e_grid=x_e_grid,
        n_e_grid=n_e_grid,
        n_H_grid=n_H_grid,
        sound_speed_grid=sound_speed_grid,
        sound_horizon_mpc=sound_horizon_mpc,
        reionization_z=float(z_reion),
        reionization_tau=float(reionization_tau),
        eta_rec=eta_rec,
        a_rec=a_rec_value,
        z_rec=peak_z,
        eta_of_a=eta_of_a,
        a_of_eta=a_of_eta,
        z_of_eta=z_of_eta,
        H_of_eta=H_of_eta,
        chi_of_eta=chi_of_eta,
        da_of_eta=da_of_eta,
        tau_of_eta=tau_of_eta,
        tau_dot_of_eta=tau_dot_of_eta,
        visibility_of_eta=visibility_of_eta,
        x_e_of_eta=x_e_of_eta,
        sound_speed_of_eta=sound_speed_of_eta,
    )
    _CUSTOM_CMB_BACKGROUND_RESULTS[cache_key] = background_data
    return _get_cached_custom_cmb_background(cache_key)


_CMB_TEMPERATURE_SPECTRA = {"BB", "EE", "TE", "TT"}


@dataclass(frozen=True, slots=True)
class _DeclaredStateSlot:
    """Describe one state-vector slot for the declared graph solver."""

    variable: str
    wrt: str
    order: int
    index: int


@dataclass(frozen=True, slots=True)
class _DeclaredGraphRuntimeSpec:
    """Prepared runtime metadata for the declared graph solver."""

    evolution_variable: str
    state_slots: tuple[_DeclaredStateSlot, ...]
    state_index_by_key: FrozenMapping
    equation_by_variable: FrozenMapping
    equation_orders: FrozenMapping
    equation_wrt_by_variable: FrozenMapping


@dataclass(frozen=True, slots=True)
class _DeclaredDerivativeStep:
    """Prepared derivative-symbol resolution metadata."""

    output_name: str
    variable: str
    wrt: str
    order: int
    slot_name: str


@dataclass(frozen=True, slots=True)
class _DeclaredValueStep:
    """Prepared expression or algebraic relation evaluation step."""

    output_name: str
    compiled_expression: Any
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DeclaredEquationSlotPlan:
    """Prepared derivative update rule for one state-vector slot."""

    state_index: int
    wrt: str
    promote_from_index: int | None
    compiled_rhs: Any | None
    equation_name: str | None


@dataclass(frozen=True, slots=True)
class _DeclaredGraphExecutionPlan:
    """Immutable compiled execution plan for the declared graph solver."""

    runtime_spec: _DeclaredGraphRuntimeSpec
    derivative_steps: tuple[_DeclaredDerivativeStep, ...]
    value_steps: tuple[_DeclaredValueStep, ...]
    source_steps: tuple[_DeclaredValueStep, ...]
    start_condition_entries: tuple[Any, ...]
    end_condition_entries: tuple[Any, ...]
    equation_slot_plans: tuple[_DeclaredEquationSlotPlan, ...]


_DECLARED_GRAPH_EXECUTION_PLAN_CACHE: dict[
    int, _DeclaredGraphExecutionPlan
] = {}


def _prepare_declared_graph_runtime_spec(
    perturbation_data: Any,
) -> _DeclaredGraphRuntimeSpec:
    """Return state-vector metadata for the declared graph contract."""

    equation_by_variable: dict[str, Any] = {}
    equation_orders: dict[str, int] = {}
    equation_wrt_by_variable: dict[str, str] = {}
    for equation_name, equation_entry in perturbation_data.equations.items():
        variable_name = str(equation_entry.lhs.variable)
        if variable_name in equation_by_variable:
            previous_name = equation_by_variable[variable_name].name
            raise ValueError(
                "Declared CMB graph defines more than one differential "
                f"equation for variable '{variable_name}' via "
                f"'{previous_name}' and '{equation_name}'"
            )
        equation_by_variable[variable_name] = equation_entry
        equation_orders[variable_name] = int(equation_entry.lhs.order)
        equation_wrt_by_variable[variable_name] = str(equation_entry.lhs.wrt)

    state_slots: list[_DeclaredStateSlot] = []
    state_index_by_key: dict[tuple[str, str, int], int] = {}
    for variable_name in sorted(equation_by_variable):
        order = equation_orders[variable_name]
        variable_wrt = equation_wrt_by_variable[variable_name]
        for derivative_order in range(order):
            index = len(state_slots)
            slot = _DeclaredStateSlot(
                variable=variable_name,
                wrt=variable_wrt,
                order=derivative_order,
                index=index,
            )
            state_slots.append(slot)
            state_index_by_key[
                (variable_name, variable_wrt, derivative_order)
            ] = index

    return _DeclaredGraphRuntimeSpec(
        evolution_variable="eta",
        state_slots=tuple(state_slots),
        state_index_by_key=FrozenMapping(state_index_by_key),
        equation_by_variable=FrozenMapping(equation_by_variable),
        equation_orders=FrozenMapping(equation_orders),
        equation_wrt_by_variable=FrozenMapping(equation_wrt_by_variable),
    )


def _compile_declared_graph_execution_plan(
    perturbation_data: Any,
) -> _DeclaredGraphExecutionPlan:
    """Return the compiled execution plan for one declared graph."""

    cache_token = object.__hash__(perturbation_data)
    cached = _DECLARED_GRAPH_EXECUTION_PLAN_CACHE.get(cache_token)
    if cached is not None:
        return cached

    runtime_spec = _prepare_declared_graph_runtime_spec(perturbation_data)
    derivative_steps = tuple(
        _DeclaredDerivativeStep(
            output_name=entry.name,
            variable=str(entry.variable or ""),
            wrt=str(entry.wrt or ""),
            order=int(entry.order or 1),
            slot_name=(
                f"__d{int(entry.order or 1)}_"
                f"{entry.variable}_"
                f"{entry.wrt}"
            ),
        )
        for entry in perturbation_data.derived.values()
        if entry.expression is None
    )
    relation_entries = {
        entry.target: entry for entry in perturbation_data.constraints.values()
    }
    relation_entries.update(
        {entry.target: entry for entry in perturbation_data.closures.values()}
    )
    value_steps: list[_DeclaredValueStep] = []
    for (
        node_name
    ) in perturbation_data.dependency_graph_summary.evaluation_order:
        derived_entry = perturbation_data.derived.get(node_name)
        if derived_entry is not None and derived_entry.expression is not None:
            value_steps.append(
                _DeclaredValueStep(
                    output_name=node_name,
                    compiled_expression=derived_entry.compiled_expression,
                    dependencies=tuple(derived_entry.dependencies),
                )
            )
            continue
        relation_entry = relation_entries.get(node_name)
        if relation_entry is None:
            continue
        value_steps.append(
            _DeclaredValueStep(
                output_name=node_name,
                compiled_expression=relation_entry.compiled_expression,
                dependencies=tuple(relation_entry.dependencies),
            )
        )
    source_steps = tuple(
        _DeclaredValueStep(
            output_name=entry.name,
            compiled_expression=entry.compiled_expression,
            dependencies=tuple(entry.dependencies),
        )
        for entry in perturbation_data.sources.values()
    )
    start_condition_entries = tuple(
        sorted(
            tuple(perturbation_data.initial_conditions.values())
            + tuple(
                entry
                for entry in perturbation_data.boundary_conditions.values()
                if str(getattr(entry, "anchor", "start")) == "start"
            ),
            key=lambda entry: (
                str(entry.target.variable),
                str(entry.target.wrt),
                int(entry.target.order),
                str(entry.name),
            ),
        )
    )
    end_condition_entries = tuple(
        sorted(
            (
                entry
                for entry in perturbation_data.boundary_conditions.values()
                if str(getattr(entry, "anchor", "start")) == "end"
            ),
            key=lambda entry: (
                str(entry.target.variable),
                str(entry.target.wrt),
                int(entry.target.order),
                str(entry.name),
            ),
        )
    )
    equation_slot_plans: list[_DeclaredEquationSlotPlan] = []
    for slot in runtime_spec.state_slots:
        promote_from_index = None
        compiled_rhs = None
        equation_name = None
        if slot.order + 1 < runtime_spec.equation_orders[slot.variable]:
            promote_from_index = runtime_spec.state_index_by_key[
                (
                    slot.variable,
                    slot.wrt,
                    slot.order + 1,
                )
            ]
        else:
            equation_entry = runtime_spec.equation_by_variable[slot.variable]
            compiled_rhs = equation_entry.compiled_rhs
            equation_name = equation_entry.name
        equation_slot_plans.append(
            _DeclaredEquationSlotPlan(
                state_index=int(slot.index),
                wrt=str(slot.wrt),
                promote_from_index=promote_from_index,
                compiled_rhs=compiled_rhs,
                equation_name=equation_name,
            )
        )
    compiled_plan = _DeclaredGraphExecutionPlan(
        runtime_spec=runtime_spec,
        derivative_steps=derivative_steps,
        value_steps=tuple(value_steps),
        source_steps=source_steps,
        start_condition_entries=start_condition_entries,
        end_condition_entries=end_condition_entries,
        equation_slot_plans=tuple(equation_slot_plans),
    )
    _DECLARED_GRAPH_EXECUTION_PLAN_CACHE[cache_token] = compiled_plan
    return compiled_plan


def _declared_runtime_seed(
    *,
    k_value: float,
    physical_params: _CustomCMBPhysicalParameters,
    model_parameters: Mapping[str, float],
) -> float:
    """Return the declared-graph initial-condition normalization."""

    del k_value
    del physical_params
    for parameter_name in ("seed", "primordial_seed", "transfer_seed"):
        if parameter_name not in model_parameters:
            continue
        return _coerce_numeric_scalar(
            model_parameters[parameter_name],
            name=parameter_name,
        )
    # Keep transfer functions unit-normalized unless the contract declares
    # an explicit seed for its initial conditions.
    return 1.0


def _build_declared_base_context(
    *,
    model_parameters: Mapping[str, float],
    physical_params: _CustomCMBPhysicalParameters,
    numerics: _CustomCMBNumerics,
    k_value: float,
    eta_value: float,
    background_scalars: Mapping[str, float],
) -> dict[str, Any]:
    """Return scalar runtime values shared by equations and conditions."""

    tight_coupling_drag = _compute_tight_coupling_drag(
        collision_rate=float(background_scalars["collision_rate"]),
        k_value=float(k_value),
        tight_coupling_ratio=float(numerics.tight_coupling_ratio),
    )
    context: dict[str, Any] = dict(model_parameters)
    context.update(background_scalars)
    context["k"] = float(k_value)
    context["seed"] = _declared_runtime_seed(
        k_value=float(k_value),
        physical_params=physical_params,
        model_parameters=model_parameters,
    )
    context["a_initial"] = float(background_scalars["a"])
    context["eta_initial"] = float(eta_value)
    for name, value in (
        ("Omega_b0", physical_params.Omega_b0),
        ("ombh2", physical_params.ombh2),
        ("Omega_c0", physical_params.Omega_c0),
        ("omch2", physical_params.omch2),
        ("Omega_m0", physical_params.Omega_m0_background),
        ("Omega_gamma0", physical_params.Omega_gamma0),
        ("Omega_nu0", physical_params.Omega_nu0),
        ("Omega_r0", physical_params.Omega_r0),
        ("Omega_k0", physical_params.Omega_k0),
        ("Omega_de0", physical_params.Omega_de0),
        ("w0", physical_params.dark_energy_eos0),
        ("wa", physical_params.dark_energy_eos1),
        ("Neff", physical_params.Neff),
        ("primordial_amplitude", physical_params.primordial_amplitude),
        (
            "primordial_spectral_index",
            physical_params.primordial_spectral_index,
        ),
    ):
        if value is None:
            continue
        context.setdefault(name, float(value))
    context["sound_horizon"] = float(background_scalars["sound_horizon"])
    context["sound_speed_sq"] = float(background_scalars["sound_speed_sq"])
    context["collision_rate"] = float(background_scalars["collision_rate"])
    context["free_streaming"] = float(background_scalars["free_streaming"])
    context["tight_coupling_drag"] = float(tight_coupling_drag)
    context["tight_coupling_ratio"] = float(numerics.tight_coupling_ratio)
    return context


def _compute_tight_coupling_drag(
    *,
    collision_rate: float | numpy.ndarray,
    k_value: float,
    tight_coupling_ratio: float,
) -> float | numpy.ndarray:
    """Return the capped collision rate used by declared CMB graphs."""

    tight_coupling_cap = max(
        float(k_value) * float(tight_coupling_ratio),
        1.0e-12,
    )
    collision_rate_array = numpy.asarray(collision_rate, dtype=float)
    drag = collision_rate_array / (
        1.0 + collision_rate_array / tight_coupling_cap
    )
    if drag.ndim == 0:
        return float(drag)
    return drag


def _resolve_declared_graph_context(
    context: dict[str, Any],
    perturbation_data: Any,
    *,
    allow_partial: bool = False,
    eta_grid: numpy.ndarray | None,
    execution_plan: _DeclaredGraphExecutionPlan | None,
) -> dict[str, Any]:
    """Resolve derivative symbols, derived expressions, and relations."""

    if execution_plan is None:
        execution_plan = _compile_declared_graph_execution_plan(
            perturbation_data
        )
    runtime_spec = execution_plan.runtime_spec

    pending_derivatives = list(execution_plan.derivative_steps)
    pending_values = list(execution_plan.value_steps)
    while pending_derivatives or pending_values:
        progress = False
        next_derivatives: list[_DeclaredDerivativeStep] = []
        for step in pending_derivatives:
            target_name = step.variable
            if target_name not in context:
                next_derivatives.append(step)
                continue
            target_value = context[target_name]
            derivative_order = int(step.order)
            if eta_grid is None:
                slot_index = runtime_spec.state_index_by_key.get(
                    (target_name, step.wrt, derivative_order)
                )
                if slot_index is None or step.slot_name not in context:
                    next_derivatives.append(step)
                    continue
                context[step.output_name] = context[step.slot_name]
                progress = True
                continue
            coordinate_name = str(step.wrt or runtime_spec.evolution_variable)
            derivative_value = numpy.asarray(target_value, dtype=float)
            if coordinate_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
                coordinate_history = numpy.asarray(eta_grid, dtype=float)
            else:
                if coordinate_name not in context:
                    next_derivatives.append(step)
                    continue
                coordinate_history = numpy.asarray(
                    context[coordinate_name],
                    dtype=float,
                )
                if coordinate_history.ndim == 0:
                    coordinate_history = numpy.full_like(
                        eta_grid,
                        float(coordinate_history),
                        dtype=float,
                    )
                if coordinate_history.shape != eta_grid.shape:
                    raise ValueError(
                        "Declared coordinate history must match the eta "
                        f"grid for derivative symbol '{step.output_name}'."
                    )
            for _ in range(derivative_order):
                derivative_eta = numpy.asarray(
                    numpy.gradient(
                        derivative_value,
                        eta_grid,
                        edge_order=1,
                    ),
                    dtype=float,
                )
                if coordinate_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
                    derivative_value = derivative_eta
                    continue
                coordinate_rate = numpy.asarray(
                    numpy.gradient(
                        coordinate_history,
                        eta_grid,
                        edge_order=1,
                    ),
                    dtype=float,
                )
                if not numpy.all(numpy.isfinite(coordinate_rate)):
                    raise ValueError(
                        "Declared coordinate history produced non-finite "
                        f"rates for derivative symbol '{step.output_name}'."
                    )
                if numpy.any(numpy.abs(coordinate_rate) <= 1.0e-12):
                    raise ValueError(
                        "Declared coordinate history is singular for "
                        f"derivative symbol '{step.output_name}'."
                    )
                derivative_value = derivative_eta / coordinate_rate
            context[step.output_name] = derivative_value
            progress = True

        next_values: list[_DeclaredValueStep] = []
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for step in pending_values:
                missing = [
                    dependency
                    for dependency in step.dependencies
                    if dependency not in context
                ]
                if missing:
                    next_values.append(step)
                    continue
                context[step.output_name] = (
                    _evaluate_compiled_expression_noerr(
                        step.compiled_expression,
                        context,
                    )
                )
                progress = True

        if progress:
            pending_derivatives = next_derivatives
            pending_values = next_values
            continue
        if allow_partial:
            return context
        pending_names = sorted(
            [step.output_name for step in next_derivatives]
            + [step.output_name for step in next_values]
        )
        pending_str = ", ".join(pending_names)
        raise ValueError(
            "Declared CMB graph references unresolved symbol(s): "
            f"{pending_str}"
        )
    return context


def _evaluate_declared_initial_state(
    *,
    perturbation_data: Any,
    execution_plan: _DeclaredGraphExecutionPlan,
    base_context: Mapping[str, Any],
) -> tuple[numpy.ndarray, tuple[tuple[str, str, int], ...]]:
    """Return the initial state vector for one Fourier mode."""

    runtime_spec = execution_plan.runtime_spec
    state_vector = numpy.zeros(len(runtime_spec.state_slots), dtype=float)
    assigned_targets: list[tuple[str, str, int]] = []
    context = dict(base_context)
    condition_entries = execution_plan.start_condition_entries
    pending = list(condition_entries)
    while pending:
        context = _resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
        )
        progress = False
        next_round: list[Any] = []
        for entry in pending:
            missing = [
                dependency
                for dependency in entry.dependencies
                if dependency not in context
            ]
            if missing:
                next_round.append(entry)
                continue
            value = _coerce_numeric_scalar(
                evaluate_compiled_expression(
                    entry.compiled_expression,
                    context,
                ),
                name=f"condition '{entry.name}'",
            )
            state_index = runtime_spec.state_index_by_key[
                (
                    str(entry.target.variable),
                    str(entry.target.wrt),
                    int(entry.target.order),
                )
            ]
            state_vector[state_index] = value
            assigned_targets.append(
                (
                    str(entry.target.variable),
                    str(entry.target.wrt),
                    int(entry.target.order),
                )
            )
            if int(entry.target.order) == 0:
                context[str(entry.target.variable)] = value
            else:
                context[
                    "__d"
                    f"{int(entry.target.order)}_"
                    f"{entry.target.variable}_"
                    f"{entry.target.wrt}"
                ] = value
            progress = True
        if not progress and next_round:
            pending_names = ", ".join(entry.name for entry in next_round)
            raise ValueError(
                "Declared CMB start conditions could not be resolved: "
                f"{pending_names}"
            )
        pending = next_round
    _resolve_declared_graph_context(
        context,
        perturbation_data,
        allow_partial=True,
        eta_grid=None,
        execution_plan=execution_plan,
    )
    return state_vector, tuple(assigned_targets)


def _declared_graph_projection(
    *,
    projection: str,
    kernel: str | None,
    kernel_batch: _DeclaredProjectionKernelBatch,
    eta_weights: numpy.ndarray,
    chi_grid: numpy.ndarray,
    source_chi: float,
    source_histories: Mapping[str, numpy.ndarray],
) -> numpy.ndarray:
    """Return projected transfer component values for every ell."""

    j_l = kernel_batch.j_l
    j_l_derivative = kernel_batch.j_l_derivative
    e_kernel = kernel_batch.e_kernel
    b_kernel = kernel_batch.b_kernel

    def _apply_kernel(kernel_name: str) -> numpy.ndarray:
        """Return the ell-batched kernel selected by ``kernel_name``."""

        kernel_spec = get_declared_projection_kernel_spec(kernel_name)
        if kernel_spec.kind == "temperature_mixed":
            raise ValueError(
                "Temperature mixed kernels must use the dedicated "
                "temperature projection dispatch."
            )
        if kernel_spec.kind == "spherical_bessel":
            return j_l
        if kernel_spec.kind == "spherical_bessel_derivative":
            return j_l_derivative
        if kernel_spec.kind == "spin2_e":
            return e_kernel
        if kernel_spec.kind == "spin2_b":
            return b_kernel
        if kernel_spec.kind == "lensing_potential":
            geometry = numpy.clip(source_chi - chi_grid, 0.0, None) / (
                max(float(source_chi), 1.0e-12)
                * numpy.maximum(chi_grid, 1.0e-12)
            )
            return 2.0 * j_l * geometry[numpy.newaxis, :]
        raise ValueError(
            "Declared observable requests unsupported kernel "
            f"'{kernel_name}'"
        )

    def _project_history(
        kernel_values: numpy.ndarray,
        history: numpy.ndarray,
    ) -> numpy.ndarray:
        """Project one source history through one ell-batched kernel."""

        return numpy.asarray(
            kernel_values @ (eta_weights * history),
            dtype=float,
        )

    def _sum_projected_sources(kernel_name: str) -> numpy.ndarray:
        """Project every declared source through one shared kernel."""

        kernel_values = _apply_kernel(kernel_name)
        source = numpy.zeros_like(eta_weights, dtype=float)
        for history in source_histories.values():
            source += history
        return _project_history(kernel_values, source)

    if projection == "line_of_sight_temperature":
        projected = numpy.zeros(j_l.shape[0], dtype=float)
        if "monopole" in source_histories:
            projected += _project_history(
                j_l,
                source_histories["monopole"],
            )
        if "doppler" in source_histories:
            projected += _project_history(
                j_l_derivative,
                source_histories["doppler"],
            )
        if "isw" in source_histories:
            projected += _project_history(
                j_l,
                source_histories["isw"],
            )
        if "additive" in source_histories:
            projected += _project_history(
                j_l,
                source_histories["additive"],
            )
        return projected
    if projection in {
        "line_of_sight_polarization_e",
        "line_of_sight_signal",
        "line_of_sight_signal_derivative",
        "spin2_e_mode",
        "spin2_b_mode",
        "line_of_sight_potential",
        "line_of_sight_lensing_potential",
        "custom_line_of_sight",
    }:
        if kernel is None:
            raise ValueError(
                f"Declared observable projection '{projection}' did not "
                "resolve a kernel."
            )
        return _sum_projected_sources(kernel)
    if projection in SUPPORTED_DECLARED_TRANSFER_PROJECTIONS:
        raise ValueError(
            "Declared observable projection dispatch is incomplete for "
            f"'{projection}'"
        )
    raise ValueError(
        "Declared observable requests unsupported projection "
        f"'{projection}'"
    )


def _trapezoid_weights(grid: numpy.ndarray) -> numpy.ndarray:
    """Return the integration weights for one strictly increasing grid."""

    step_sizes = numpy.diff(grid)
    if step_sizes.size == 0 or not numpy.all(numpy.isfinite(step_sizes)):
        raise ValueError("eta_los_grid must be a finite grid")
    if numpy.any(step_sizes <= 0.0):
        raise ValueError("eta_los_grid must be strictly increasing")
    weights = numpy.empty_like(grid, dtype=float)
    weights[0] = 0.5 * step_sizes[0]
    weights[-1] = 0.5 * step_sizes[-1]
    if grid.size > 2:
        weights[1:-1] = 0.5 * (step_sizes[:-1] + step_sizes[1:])
    return weights


def _compute_custom_cmb_spectrum_data(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
) -> CustomCMBSpectrumData:
    """Return transfer functions and spectra for a declared CMB graph."""

    cache_key = _custom_cmb_spectrum_cache_key(
        contract_or_params,
        ells,
        background_provider,
    )
    cached_spectrum = _CUSTOM_CMB_SPECTRUM_RESULTS.get(cache_key)
    if cached_spectrum is not None:
        return _get_cached_custom_cmb_spectrum_data(cache_key)

    perturbation_data = _compile_declared_perturbation_contract(
        contract_or_params
    )
    if perturbation_data.standard:
        raise ValueError("Standard perturbation contracts must use CAMB.")

    execution_plan = _compile_declared_graph_execution_plan(perturbation_data)
    runtime_spec = execution_plan.runtime_spec
    physical_params = _resolve_custom_cmb_physical_parameters(
        contract_or_params,
        background_provider,
    )
    numerics = _resolve_custom_cmb_numerics(contract_or_params)
    background = _build_custom_cmb_background(
        contract_or_params,
        physical_params,
        numerics,
        background_provider=background_provider,
    )

    ell_arr = numpy.asarray(list(ells), dtype=int)
    if ell_arr.size == 0:
        raise ValueError("ells must not be empty")

    a_initial = max(
        background.a_grid[0],
        1.0 / (max(numerics.initial_redshift, 1.0) + 1.0),
    )
    eta_start = float(background.eta_of_a(a_initial))
    eta_los_grid = numpy.asarray(
        background.eta_grid[background.eta_grid >= eta_start],
        dtype=float,
    )
    eta_los_refinement = max(1, min(numerics.source_grid_multiplier, 2))
    for _ in range(eta_los_refinement - 1):
        midpoint_grid = 0.5 * (eta_los_grid[:-1] + eta_los_grid[1:])
        eta_los_grid = numpy.unique(
            numpy.concatenate((eta_los_grid, midpoint_grid))
        )
    if eta_los_grid.size < 128:
        eta_los_grid = numpy.linspace(
            eta_start,
            float(background.eta_grid[-1]),
            128,
        )
    eta_los_background = background.sample(eta_los_grid)
    a_los_grid = numpy.asarray(eta_los_background["a"], dtype=float)
    z_los_grid = numpy.asarray(eta_los_background["z"], dtype=float)
    H_los_grid = numpy.asarray(eta_los_background["H"], dtype=float)
    tau_los_grid = numpy.asarray(eta_los_background["tau"], dtype=float)
    tau_dot_los_grid = numpy.asarray(
        eta_los_background["tau_dot"],
        dtype=float,
    )
    visibility_los_grid = numpy.asarray(
        eta_los_background["visibility"],
        dtype=float,
    )
    chi_los_grid = numpy.asarray(
        eta_los_background["chi"],
        dtype=float,
    )
    angular_diameter_distance_grid = numpy.asarray(
        eta_los_background["angular_diameter_distance"],
        dtype=float,
    )
    sound_speed_los_grid = numpy.asarray(
        eta_los_background["sound_speed"],
        dtype=float,
    )
    Hconf_los_grid = a_los_grid * H_los_grid / _C_LIGHT_KM_S
    baryon_loading_grid = (
        3.0
        * physical_params.Omega_b0
        * a_los_grid
        / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
    )
    collision_rate_grid = numpy.maximum(-tau_dot_los_grid, 0.0)
    free_streaming_grid = 1.0 / (
        1.0 + collision_rate_grid / max(float(collision_rate_grid.max()), 1.0)
    )
    sound_speed_sq_grid = 1.0 / (3.0 * (1.0 + baryon_loading_grid))
    declared_background_los = _resolve_declared_background_context(
        contract_or_params,
        a_values=a_los_grid,
        z_values=z_los_grid,
    )
    declared_background_histories: dict[str, numpy.ndarray] = {}
    for name, raw_value in declared_background_los.items():
        if name in {"a", "z"}:
            continue
        history = numpy.asarray(raw_value, dtype=float)
        if history.ndim == 0:
            history = numpy.full_like(
                eta_los_grid,
                float(history),
                dtype=float,
            )
        if history.shape != eta_los_grid.shape:
            raise ValueError(
                "Declared background symbol did not match the "
                f"line-of-sight grid: {name}"
            )
        if not numpy.all(numpy.isfinite(history)):
            raise ValueError(
                "Declared background symbol produced non-finite values: "
                f"{name}"
            )
        declared_background_histories[name] = history
    coordinate_histories = {
        "a": a_los_grid,
        "z": z_los_grid,
        "eta": eta_los_grid,
        "H": H_los_grid,
        "Hconf": Hconf_los_grid,
        "tau": tau_los_grid,
        "tau_dot": tau_dot_los_grid,
        "visibility": visibility_los_grid,
        "chi": chi_los_grid,
        "angular_diameter_distance": angular_diameter_distance_grid,
        "sound_speed": sound_speed_los_grid,
    }
    for name, history in declared_background_histories.items():
        coordinate_histories.setdefault(name, history)
    coordinate_rate_histories = {
        "eta": numpy.ones_like(eta_los_grid, dtype=float)
    }
    for name, history in coordinate_histories.items():
        if name == "eta":
            continue
        coordinate_rate_histories[name] = numpy.asarray(
            numpy.gradient(history, eta_los_grid, edge_order=1),
            dtype=float,
        )

    eta0_floor = max(background.eta0, 1.0e-6)
    k_min = max(
        numerics.k_min,
        0.2 * max(float(ell_arr.min()), 2.0) / eta0_floor,
    )
    eta_rec_distance = max(background.eta0 - background.eta_rec, 1.0)
    required_k_max = 1.5 * ((float(ell_arr.max()) + 16.0) / eta_rec_distance)
    k_max = max(
        required_k_max,
        min(numerics.k_max, max(12.0 * k_min, 0.08)),
    )
    k_values = numpy.logspace(
        math.log10(k_min),
        math.log10(k_max),
        max(16, min(numerics.k_sample_count, 48)),
    )
    k_values = numpy.asarray(k_values, dtype=float)

    eta0 = background.eta0
    source_chi = float(background.chi_of_eta(background.eta_rec))
    source_parameters: dict[str, float] = {}
    for source in (
        contract_or_params.get("param_map", {}) or {},
        contract_or_params.get("model_parameters", {}) or {},
    ):
        if not isinstance(source, Mapping):
            continue
        for name, value in source.items():
            if str(name) in source_parameters:
                continue
            try:
                source_parameters[str(name)] = _coerce_numeric_scalar(
                    value,
                    name=str(name),
                )
            except ValueError:
                continue

    transfer_component_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "transfer_component"
    }
    power_spectrum_observables = {
        name: entry
        for name, entry in perturbation_data.observables.items()
        if entry.kind == "angular_power_spectrum"
    }
    transfer_components = {
        name: numpy.zeros((ell_arr.size, k_values.size), dtype=float)
        for name in transfer_component_observables
    }
    eta_integration_weights = _trapezoid_weights(eta_los_grid)

    def _blend_history(
        history: numpy.ndarray,
        *,
        step_index: int,
        blend: float,
    ) -> float:
        """Return one linearly interpolated history value."""

        next_index = min(step_index + 1, eta_los_grid.size - 1)
        weight_next = float(blend)
        weight_current = 1.0 - weight_next
        return float(
            weight_current * history[step_index]
            + weight_next * history[next_index]
        )

    def _scalar_background_context(
        step_index: int,
        blend: float,
    ) -> tuple[float, dict[str, float]]:
        """Return one interpolated scalar background context."""

        eta_value = _blend_history(
            eta_los_grid,
            step_index=step_index,
            blend=blend,
        )
        scalar_context = {
            "a": _blend_history(
                a_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "z": _blend_history(
                z_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "eta": float(eta_value),
            "H": _blend_history(
                H_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "Hconf": _blend_history(
                Hconf_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "tau": _blend_history(
                tau_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "tau_dot": _blend_history(
                tau_dot_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "visibility": _blend_history(
                visibility_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "chi": _blend_history(
                chi_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "angular_diameter_distance": _blend_history(
                angular_diameter_distance_grid,
                step_index=step_index,
                blend=blend,
            ),
            "sound_speed": _blend_history(
                sound_speed_los_grid,
                step_index=step_index,
                blend=blend,
            ),
            "sound_speed_sq": _blend_history(
                sound_speed_sq_grid,
                step_index=step_index,
                blend=blend,
            ),
            "collision_rate": _blend_history(
                collision_rate_grid,
                step_index=step_index,
                blend=blend,
            ),
            "free_streaming": _blend_history(
                free_streaming_grid,
                step_index=step_index,
                blend=blend,
            ),
            "sound_horizon": float(background.sound_horizon_mpc),
        }
        for name, history in declared_background_histories.items():
            scalar_context[name] = _blend_history(
                history,
                step_index=step_index,
                blend=blend,
            )
        return float(eta_value), scalar_context

    def _resolve_coordinate_rate(
        *,
        wrt_name: str,
        scalar_context: Mapping[str, float],
        step_index: int,
        blend: float,
        k_value: float,
    ) -> float:
        """Return ``dwrt/deta`` for one declared runtime coordinate."""

        if wrt_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
            return 1.0
        for legacy_name in _LEGACY_DECLARED_EVOLUTION_COORDINATES:
            derivative_symbol = f"__d1_{wrt_name}_{legacy_name}"
            if derivative_symbol not in scalar_context:
                continue
            rate = float(scalar_context[derivative_symbol])
            break
        else:
            if wrt_name not in coordinate_rate_histories:
                raise ValueError(
                    "Declared CMB coordinate transform does not support "
                    f"wrt '{wrt_name}'."
                )
            rate = _blend_history(
                coordinate_rate_histories[wrt_name],
                step_index=step_index,
                blend=blend,
            )
        if not numpy.isfinite(rate) or abs(rate) <= 1.0e-12:
            eta_value = _blend_history(
                eta_los_grid,
                step_index=step_index,
                blend=blend,
            )
            raise ValueError(
                "Declared CMB coordinate transform is singular for "
                f"wrt '{wrt_name}' at eta={eta_value}, k={k_value}"
            )
        return rate

    def _build_scalar_state_context(
        state_vector: numpy.ndarray,
        *,
        k_value: float,
        eta_value: float,
        background_scalars: Mapping[str, float],
    ) -> dict[str, Any]:
        """Return the scalar expression environment for one solver stage."""

        context = _build_declared_base_context(
            model_parameters=source_parameters,
            physical_params=physical_params,
            numerics=numerics,
            k_value=float(k_value),
            eta_value=float(eta_value),
            background_scalars=background_scalars,
        )
        for slot in runtime_spec.state_slots:
            value = float(state_vector[slot.index])
            if slot.order == 0:
                context[slot.variable] = value
            else:
                context[f"__d{slot.order}_{slot.variable}_{slot.wrt}"] = value
        return _resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=True,
            eta_grid=None,
            execution_plan=execution_plan,
        )

    def _build_array_context(
        histories: Mapping[str, numpy.ndarray],
        *,
        k_value: float,
    ) -> dict[str, Any]:
        """Return the array-valued expression environment for one mode."""

        context = {
            "a": a_los_grid,
            "z": z_los_grid,
            "eta": eta_los_grid,
            "H": H_los_grid,
            "Hconf": Hconf_los_grid,
            "tau": tau_los_grid,
            "tau_dot": tau_dot_los_grid,
            "visibility": visibility_los_grid,
            "chi": chi_los_grid,
            "angular_diameter_distance": numpy.asarray(
                angular_diameter_distance_grid,
                dtype=float,
            ),
            "sound_speed": sound_speed_los_grid,
            "sound_speed_sq": sound_speed_sq_grid,
            "collision_rate": collision_rate_grid,
            "free_streaming": free_streaming_grid,
            "tight_coupling_drag": _compute_tight_coupling_drag(
                collision_rate=collision_rate_grid,
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            ),
            "sound_horizon": numpy.full_like(
                eta_los_grid,
                float(background.sound_horizon_mpc),
                dtype=float,
            ),
            "k": float(k_value),
            "seed": _declared_runtime_seed(
                k_value=float(k_value),
                physical_params=physical_params,
                model_parameters=source_parameters,
            ),
        }
        for name, value in (
            ("Omega_b0", physical_params.Omega_b0),
            ("ombh2", physical_params.ombh2),
            ("Omega_c0", physical_params.Omega_c0),
            ("omch2", physical_params.omch2),
            ("Omega_m0", physical_params.Omega_m0_background),
            ("Omega_gamma0", physical_params.Omega_gamma0),
            ("Omega_nu0", physical_params.Omega_nu0),
            ("Omega_r0", physical_params.Omega_r0),
            ("Omega_k0", physical_params.Omega_k0),
            ("Omega_de0", physical_params.Omega_de0),
            ("w0", physical_params.dark_energy_eos0),
            ("wa", physical_params.dark_energy_eos1),
            ("Neff", physical_params.Neff),
            ("primordial_amplitude", physical_params.primordial_amplitude),
            (
                "primordial_spectral_index",
                physical_params.primordial_spectral_index,
            ),
        ):
            if value is None:
                continue
            context[name] = numpy.full_like(
                eta_los_grid,
                float(value),
                dtype=float,
            )
        for name, history in declared_background_histories.items():
            context[name] = numpy.asarray(history, dtype=float)
        for name, value in source_parameters.items():
            context[name] = float(value)
        for slot in runtime_spec.state_slots:
            if slot.order != 0:
                continue
            context[slot.variable] = numpy.asarray(
                histories[slot.variable],
                dtype=float,
            )
        return _resolve_declared_graph_context(
            context,
            perturbation_data,
            allow_partial=False,
            eta_grid=eta_los_grid,
            execution_plan=execution_plan,
        )

    def _evaluate_declared_sources(
        context: Mapping[str, Any],
        *,
        k_value: float,
    ) -> dict[str, numpy.ndarray]:
        """Return source arrays keyed by source-term name."""

        source_arrays: dict[str, numpy.ndarray] = {}
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for source_step in execution_plan.source_steps:
                value = numpy.asarray(
                    _evaluate_compiled_expression_noerr(
                        source_step.compiled_expression,
                        context,
                    ),
                    dtype=float,
                )
                if value.ndim == 0:
                    value = numpy.full_like(
                        eta_los_grid,
                        float(value),
                        dtype=float,
                    )
                if value.shape != eta_los_grid.shape:
                    raise ValueError(
                        "Source term "
                        f"'{source_step.output_name}' did not evaluate to "
                        "an eta-grid history."
                    )
                if not numpy.all(numpy.isfinite(value)):
                    raise ValueError(
                        "Declared source term produced non-finite values: "
                        f"{source_step.output_name} at k={k_value}"
                    )
                source_arrays[source_step.output_name] = value
        return source_arrays

    def _mode_rhs(
        state_vector: numpy.ndarray,
        *,
        step_index: int,
        blend: float,
        k_value: float,
    ) -> numpy.ndarray:
        """Return the state derivative for one RK stage."""

        eta_value, background_scalars = _scalar_background_context(
            step_index,
            blend,
        )
        scalar_context = _build_scalar_state_context(
            state_vector,
            k_value=float(k_value),
            eta_value=float(eta_value),
            background_scalars=background_scalars,
        )
        derivative = numpy.zeros_like(state_vector, dtype=float)
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for slot_plan in execution_plan.equation_slot_plans:
                coordinate_rate = _resolve_coordinate_rate(
                    wrt_name=slot_plan.wrt,
                    scalar_context=scalar_context,
                    step_index=step_index,
                    blend=blend,
                    k_value=float(k_value),
                )
                if slot_plan.promote_from_index is not None:
                    derivative[slot_plan.state_index] = (
                        float(state_vector[slot_plan.promote_from_index])
                        * coordinate_rate
                    )
                    continue
                derivative[slot_plan.state_index] = (
                    _coerce_numeric_scalar(
                        _evaluate_compiled_expression_noerr(
                            slot_plan.compiled_rhs,
                            scalar_context,
                        ),
                        name=f"equation '{slot_plan.equation_name}'",
                    )
                    * coordinate_rate
                )
        if not numpy.all(numpy.isfinite(derivative)):
            bad_indices = numpy.flatnonzero(~numpy.isfinite(derivative))
            bad_index = int(bad_indices[0]) if bad_indices.size else -1
            raise ValueError(
                "Declared CMB evolution produced non-finite derivatives at "
                f"eta={eta_value}, k={k_value}, state_index={bad_index}"
            )
        return derivative

    def _evolve_declared_mode(
        k_value: float,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]:
        """Integrate one Fourier mode through the declared graph."""

        end_boundary_entries = execution_plan.end_condition_entries

        def _advance_declared_interval(
            state_vector: numpy.ndarray,
            *,
            step_index: int,
            dt: float,
            k_value: float,
        ) -> numpy.ndarray:
            """Advance one LOS interval with adaptive RK4 sub-stepping."""

            _, start_scalars = _scalar_background_context(step_index, 0.0)
            _, end_scalars = _scalar_background_context(step_index, 1.0)
            start_drag = _compute_tight_coupling_drag(
                collision_rate=float(start_scalars["collision_rate"]),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
            end_drag = _compute_tight_coupling_drag(
                collision_rate=float(end_scalars["collision_rate"]),
                k_value=float(k_value),
                tight_coupling_ratio=float(numerics.tight_coupling_ratio),
            )
            stiffness_scale = max(
                abs(float(k_value)),
                abs(float(start_scalars["Hconf"])),
                abs(float(end_scalars["Hconf"])),
                abs(float(start_drag)),
                abs(float(end_drag)),
                1.0e-12,
            )
            target_stage_scale = 0.25
            required_substeps = max(
                1,
                int(
                    math.ceil(
                        abs(float(dt)) * stiffness_scale / target_stage_scale
                    )
                ),
            )
            substep_count = 1
            while substep_count < required_substeps:
                substep_count *= 2
            max_substep_count = 512
            while substep_count <= max_substep_count:
                trial_state = numpy.asarray(state_vector, dtype=float).copy()
                sub_dt = dt / float(substep_count)
                failed = False
                for substep_index in range(substep_count):
                    blend_start = substep_index / substep_count
                    blend_mid = (substep_index + 0.5) / substep_count
                    blend_end = (substep_index + 1.0) / substep_count
                    stage_rhs_initial = _mode_rhs(
                        trial_state,
                        step_index=step_index,
                        blend=blend_start,
                        k_value=float(k_value),
                    )
                    stage_rhs_mid_a = _mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_initial,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                    )
                    stage_rhs_mid_b = _mode_rhs(
                        trial_state + 0.5 * sub_dt * stage_rhs_mid_a,
                        step_index=step_index,
                        blend=blend_mid,
                        k_value=float(k_value),
                    )
                    stage_rhs_final = _mode_rhs(
                        trial_state + sub_dt * stage_rhs_mid_b,
                        step_index=step_index,
                        blend=blend_end,
                        k_value=float(k_value),
                    )
                    candidate_state = trial_state + (sub_dt / 6.0) * (
                        stage_rhs_initial
                        + 2.0 * stage_rhs_mid_a
                        + 2.0 * stage_rhs_mid_b
                        + stage_rhs_final
                    )
                    if not numpy.all(numpy.isfinite(candidate_state)):
                        failed = True
                        break
                    trial_state = candidate_state
                if not failed:
                    return trial_state
                substep_count *= 2
            raise ValueError(
                "Declared CMB evolution produced non-finite state values "
                f"at k={k_value}, step_index={step_index}"
            )

        def _integrate_declared_state_history(
            initial_state: numpy.ndarray,
        ) -> tuple[dict[str, numpy.ndarray], numpy.ndarray]:
            """Return mode histories and the final state vector."""

            histories = {
                slot.variable: numpy.empty_like(eta_los_grid, dtype=float)
                for slot in runtime_spec.state_slots
                if slot.order == 0
            }
            state = numpy.asarray(initial_state, dtype=float).copy()
            for step_index, eta_value in enumerate(eta_los_grid):
                for slot in runtime_spec.state_slots:
                    if slot.order != 0:
                        continue
                    histories[slot.variable][step_index] = state[slot.index]
                if step_index == eta_los_grid.size - 1:
                    break
                dt = float(eta_los_grid[step_index + 1] - eta_value)
                state = _advance_declared_interval(
                    state,
                    step_index=step_index,
                    dt=dt,
                    k_value=float(k_value),
                )
            return histories, state

        def _evaluate_end_boundary_residuals(
            final_state: numpy.ndarray,
        ) -> numpy.ndarray:
            """Return end-boundary residuals for one integrated mode."""

            if not end_boundary_entries:
                return numpy.zeros(0, dtype=float)
            final_eta, final_background = _scalar_background_context(
                eta_los_grid.size - 1,
                0.0,
            )
            final_context = _build_scalar_state_context(
                final_state,
                k_value=float(k_value),
                eta_value=float(final_eta),
                background_scalars=final_background,
            )
            residuals = []
            for entry in end_boundary_entries:
                state_index = runtime_spec.state_index_by_key[
                    (
                        str(entry.target.variable),
                        str(entry.target.wrt),
                        int(entry.target.order),
                    )
                ]
                expected_value = _coerce_numeric_scalar(
                    evaluate_compiled_expression(
                        entry.compiled_expression,
                        final_context,
                    ),
                    name=f"end boundary '{entry.name}'",
                )
                residuals.append(
                    float(final_state[state_index]) - float(expected_value)
                )
            return numpy.asarray(residuals, dtype=float)

        initial_eta, initial_background = _scalar_background_context(0, 0.0)
        initial_context = _build_declared_base_context(
            model_parameters=source_parameters,
            physical_params=physical_params,
            numerics=numerics,
            k_value=float(k_value),
            eta_value=float(initial_eta),
            background_scalars=initial_background,
        )
        initial_state, assigned_targets = _evaluate_declared_initial_state(
            perturbation_data=perturbation_data,
            execution_plan=execution_plan,
            base_context=initial_context,
        )
        state = numpy.asarray(initial_state, dtype=float)
        if end_boundary_entries:
            assigned_target_set = set(assigned_targets)
            free_target_keys = tuple(
                sorted(
                    (
                        slot.variable,
                        slot.wrt,
                        slot.order,
                    )
                    for slot in runtime_spec.state_slots
                    if (
                        slot.variable,
                        slot.wrt,
                        slot.order,
                    )
                    not in assigned_target_set
                )
            )
            end_target_keys = tuple(
                sorted(
                    (
                        str(entry.target.variable),
                        str(entry.target.wrt),
                        int(entry.target.order),
                    )
                    for entry in end_boundary_entries
                )
            )
            if free_target_keys != end_target_keys:
                raise ValueError(
                    "Declared end boundary solver requires end anchors to "
                    "replace exactly the missing start-state slots."
                )
            free_indices = numpy.asarray(
                [
                    runtime_spec.state_index_by_key[target_key]
                    for target_key in free_target_keys
                ],
                dtype=int,
            )
            initial_guess_context = _build_scalar_state_context(
                state,
                k_value=float(k_value),
                eta_value=float(initial_eta),
                background_scalars=initial_background,
            )
            boundary_guess = []
            for entry in end_boundary_entries:
                try:
                    boundary_guess.append(
                        _coerce_numeric_scalar(
                            evaluate_compiled_expression(
                                entry.compiled_expression,
                                initial_guess_context,
                            ),
                            name=f"end boundary '{entry.name}' guess",
                        )
                    )
                except ValueError:
                    boundary_guess.append(
                        float(
                            state[
                                runtime_spec.state_index_by_key[
                                    (
                                        str(entry.target.variable),
                                        str(entry.target.wrt),
                                        int(entry.target.order),
                                    )
                                ]
                            ]
                        )
                    )

            def _boundary_objective(
                unknown_values: numpy.ndarray,
            ) -> numpy.ndarray:
                """Return end-boundary residuals for one shooting guess."""

                trial_state = numpy.asarray(state, dtype=float).copy()
                trial_state[free_indices] = numpy.asarray(
                    unknown_values,
                    dtype=float,
                )
                _, final_state = _integrate_declared_state_history(trial_state)
                return _evaluate_end_boundary_residuals(final_state)

            boundary_solution = least_squares(
                _boundary_objective,
                numpy.asarray(boundary_guess, dtype=float),
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
            residual_scale = max(float(numerics.ode_atol) * 50.0, 1.0e-8)
            final_residuals = numpy.asarray(
                boundary_solution.fun,
                dtype=float,
            )
            if (
                not boundary_solution.success
                or not numpy.all(numpy.isfinite(boundary_solution.x))
                or not numpy.all(numpy.isfinite(final_residuals))
                or numpy.max(numpy.abs(final_residuals), initial=0.0)
                > residual_scale
            ):
                message = str(getattr(boundary_solution, "message", "unknown"))
                raise ValueError(
                    "Declared end boundary solver failed to converge: "
                    f"{message}"
                )
            state[free_indices] = numpy.asarray(
                boundary_solution.x,
                dtype=float,
            )
        histories, final_state = _integrate_declared_state_history(state)
        final_residuals = _evaluate_end_boundary_residuals(final_state)
        if final_residuals.size and numpy.max(
            numpy.abs(final_residuals), initial=0.0
        ) > max(float(numerics.ode_atol) * 50.0, 1.0e-8):
            raise ValueError(
                "Declared end boundary conditions remained unsatisfied "
                "after integration."
            )
        array_context = _build_array_context(histories, k_value=float(k_value))
        source_arrays = _evaluate_declared_sources(
            array_context,
            k_value=float(k_value),
        )
        return histories, source_arrays

    log_k_values = numpy.log(k_values)
    primordial_grid = physical_params.primordial_amplitude * numpy.power(
        k_values / 0.05,
        physical_params.primordial_spectral_index - 1.0,
    )
    ell_signature = tuple(int(ell_value) for ell_value in ell_arr)

    for k_index, k_value in enumerate(k_values):
        _, source_arrays = _evolve_declared_mode(float(k_value))
        x_values = k_value * (eta0 - eta_los_grid)
        x_signature = hashlib.sha256(
            numpy.asarray(x_values, dtype=float).tobytes()
        ).hexdigest()
        _CUSTOM_CMB_BESSEL_INPUTS.setdefault(
            x_signature,
            numpy.asarray(x_values, dtype=float).copy(),
        )
        kernel_batch = _get_cached_declared_projection_kernel_batch(
            ell_signature,
            x_signature,
        )
        for (
            component_name,
            component_entry,
        ) in transfer_component_observables.items():
            component_source_terms = component_entry.source_terms.items()
            source_histories = {
                role_name: source_arrays[source_name]
                for role_name, source_name in component_source_terms
            }
            transfer_components[component_name][:, k_index] = (
                _declared_graph_projection(
                    projection=str(component_entry.projection or ""),
                    kernel=(
                        None
                        if component_entry.kernel is None
                        else str(component_entry.kernel)
                    ),
                    kernel_batch=kernel_batch,
                    eta_weights=eta_integration_weights,
                    chi_grid=chi_los_grid,
                    source_chi=source_chi,
                    source_histories=source_histories,
                )
            )
    for component_name, component_matrix in transfer_components.items():
        if not numpy.all(numpy.isfinite(component_matrix)):
            raise ValueError(
                "Declared transfer component produced non-finite values: "
                f"{component_name}"
            )

    spectra_results: dict[str, numpy.ndarray] = {}
    for (
        observable_name,
        observable_entry,
    ) in power_spectrum_observables.items():
        primary = numpy.asarray(
            transfer_components[str(observable_entry.primary)],
            dtype=float,
        )
        secondary = numpy.asarray(
            transfer_components[str(observable_entry.secondary)],
            dtype=float,
        )
        weighted = primordial_grid[numpy.newaxis, :] * (primary * secondary)
        spectra_results[observable_name] = (
            4.0
            * math.pi
            * numpy.trapz(
                weighted,
                log_k_values,
                axis=1,
            )
        )

    spectrum_data = CustomCMBSpectrumData(
        ell_grid=ell_arr,
        k_grid=k_values,
        transfer_components=FrozenMapping(
            {name: matrix for name, matrix in transfer_components.items()}
        ),
        spectra=FrozenMapping(spectra_results),
    )
    _CUSTOM_CMB_SPECTRUM_RESULTS[cache_key] = spectrum_data
    return _get_cached_custom_cmb_spectrum_data(cache_key)


def _compute_declared_perturbation_spectrum(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    background_payload: Mapping[str, Any] | None = None,
    background_provider: Any | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Return spectra from a declared non-standard perturbation contract."""

    del background_payload
    custom_data = _compute_custom_cmb_spectrum_data(
        contract_or_params,
        ells,
        background_provider=background_provider,
    )
    ell_factor = (
        custom_data.ell_grid.astype(float)
        * (custom_data.ell_grid.astype(float) + 1.0)
        / (2.0 * math.pi)
    )
    t_cmb_muK = 2.7255e6
    spectra_results: dict[str, numpy.ndarray] = {}
    for spectrum_name, spectrum_values in custom_data.spectra.items():
        raw_values = numpy.asarray(spectrum_values, dtype=float)
        if spectrum_name in _CMB_TEMPERATURE_SPECTRA:
            raw_values = ell_factor * raw_values * t_cmb_muK * t_cmb_muK
        spectra_results[str(spectrum_name)] = raw_values
    for spectrum_name, spectrum_values in spectra_results.items():
        if not numpy.all(numpy.isfinite(spectrum_values)):
            raise ValueError(
                "Custom CMB spectrum calculation produced non-finite "
                f"{spectrum_name} values"
            )
    result = {
        spec: numpy.asarray(spectra_results[spec], dtype=float)
        for spec in spectra
        if spec in spectra_results
    }
    if len(result) != len(tuple(spectra)):
        missing = sorted(set(spectra) - set(result))
        missing_str = ", ".join(missing)
        raise ValueError(
            "Declared CMB graph does not provide requested spectra: "
            f"{missing_str}"
        )
    if len(result) == 1:
        return next(iter(result.values()))
    return result
