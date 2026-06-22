r"""Declared native background, recombination, and shared cache helpers."""

from __future__ import annotations

import ast
import math
from dataclasses import astuple, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import spherical_jn

from ...engine_adapter import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _evaluate_safe_expression,
    _freeze_for_cache,
)
from ...perturbation_contract import (
    _compile_expression_plan,
    _evaluate_compiled_expression_noerr,
    evaluate_compiled_expression,
)
from . import native_cache

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


def _get_cached_declared_symbol_plan(
    cache_key: Any,
) -> tuple[_DeclaredSymbolPlanEntry, ...]:
    """Return one cached declared-symbol evaluation plan."""

    cached = native_cache.get_declared_symbol_plan(cache_key)
    if cached is None:  # pragma: no cover - callers guard existence first
        raise KeyError(cache_key)
    return cached


def _compile_declared_symbol_plan(
    entries: Mapping[str, Any],
) -> tuple[_DeclaredSymbolPlanEntry, ...]:
    """Return one ordered declared symbol plan for background helpers."""

    cache_key = _freeze_for_cache(entries)
    cached = native_cache.get_declared_symbol_plan(cache_key)
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
    native_cache.set_declared_symbol_plan(cache_key, compiled_plan)
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


def _get_cached_custom_cmb_background(
    cache_key: tuple[Any, ...],
) -> "_CustomCMBBackgroundData":
    """Return a cached custom background payload."""

    cached = native_cache.get_custom_cmb_background(cache_key)
    if cached is None:  # pragma: no cover - callers guard existence first
        raise KeyError(cache_key)
    return cached


def _get_cached_custom_cmb_spectrum_data(
    cache_key: tuple[Any, ...],
) -> "CustomCMBSpectrumData":
    """Return a cached custom spectrum payload."""

    cached = native_cache.get_custom_cmb_spectrum(cache_key)
    if cached is None:  # pragma: no cover - callers guard existence first
        raise KeyError(cache_key)
    return cached


def _get_cached_spherical_bessel_values(
    ell: int,
    x_signature: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return cached spherical Bessel values for one ell and x-grid."""

    cache_key = (int(ell), x_signature)
    cached = native_cache.get_bessel_values(cache_key)
    if cached is not None:
        return cached
    x_values = native_cache.get_bessel_inputs(x_signature)
    if x_values is None:  # pragma: no cover - projection stores inputs first
        raise KeyError(x_signature)
    values = (
        spherical_jn(int(ell), x_values),
        spherical_jn(int(ell), x_values, derivative=True),
    )
    native_cache.set_bessel_values(cache_key, values)
    return values


def _get_cached_declared_projection_kernel_batch(
    ell_signature: tuple[int, ...],
    x_signature: str,
) -> _DeclaredProjectionKernelBatch:
    """Return cached ell-batched spherical-Bessel kernels for one x-grid."""

    cache_key = (ell_signature, x_signature)
    cached = native_cache.get_declared_projection_kernel_batch(cache_key)
    if cached is not None:
        return cached
    x_values = native_cache.get_bessel_inputs(x_signature)
    if x_values is None:  # pragma: no cover - projection stores inputs first
        raise KeyError(x_signature)
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
    native_cache.set_declared_projection_kernel_batch(cache_key, batch)
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
    cached_background = native_cache.get_custom_cmb_background(cache_key)
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
    native_cache.set_custom_cmb_background(cache_key, background_data)
    return _get_cached_custom_cmb_background(cache_key)
