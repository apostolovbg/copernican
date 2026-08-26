"""Numerical-envelope and cross-sector convergence contracts.

The contracts govern declared CMB calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy

FINAL_SPECTRUM_RELATIVE_TOLERANCES = {
    "TT": 1.0e-2,
    "EE": 1.0e-2,
    "TE": 2.0e-2,
    "PP": 3.0e-2,
    "lensed_BB": 5.0e-2,
}
FINAL_Q_GRID_RELATIVE_TOLERANCE = 2.0e-2
FINAL_HIERARCHY_RELATIVE_TOLERANCE = 1.0e-2

_PRODUCTION_SCALAR_DEFAULT_TOLERANCES = {
    "TT": FINAL_SPECTRUM_RELATIVE_TOLERANCES["TT"],
    "TE": FINAL_SPECTRUM_RELATIVE_TOLERANCES["TE"],
    "EE": FINAL_SPECTRUM_RELATIVE_TOLERANCES["EE"],
}

RUNTIME_WORK_LIMIT_NAMES = (
    "maximum_evolution_work_units",
    "maximum_momentum_work_units",
    "maximum_projection_work_units",
    "maximum_total_work_units",
)

_NUMERICAL_DEFAULTS = {
    "ell_min": 2,
    "ell_max": 2500,
    "k_min": 1.0e-5,
    "k_max": 0.4,
    "k_sample_count": 64,
    "eta_sample_count": 1024,
    "evolution_eta_sample_count": None,
    "evolution_phase_step": 0.5,
    "ode_rtol": 1.0e-6,
    "ode_atol": 1.0e-9,
    "tight_coupling_ratio": 50.0,
    "tight_coupling_exit_ratio": 0.1,
    "a_min": 1.0e-8,
    "source_grid_multiplier": 2,
    "initial_redshift": 1.0e5,
    "lensing_sampling_factor": 1.4,
}

_FINAL_MINIMUM_CONTROLS = {
    "ell_max": 2000.0,
    "k_max": 0.3,
    "k_sample_count": 18.0,
    "eta_sample_count": 192.0,
    "evolution_eta_sample_count": 128.0,
    "tight_coupling_ratio": 1600.0,
    "source_grid_multiplier": 2.0,
    "initial_redshift": 2.0e4,
    "lensing_sampling_factor": 1.4,
}

_FINAL_MAXIMUM_CONTROLS = {
    "ell_min": 2.0,
    "k_min": 1.0e-4,
    "evolution_phase_step": 2.0,
    "ode_rtol": 1.0e-5,
    "ode_atol": 1.0e-8,
    "tight_coupling_exit_ratio": 0.1,
    "a_min": 1.0e-6,
}

_FINAL_HIERARCHY_MINIMUMS_BY_SECTOR = {
    "scalar": {
        "photon_temperature": 10,
        "photon_polarization": 10,
        "massless_neutrino": 7,
        "massive_neutrino": 7,
    },
    "vector": {
        "photon_temperature": 8,
        "photon_polarization": 8,
        "massless_neutrino": 5,
    },
    "tensor": {
        "photon_temperature": 12,
        "photon_polarization": 12,
        "massless_neutrino": 9,
        "massive_neutrino": 7,
    },
}

_FINAL_MOMENTUM_GRID_MINIMUMS = {
    "count": 16,
    "q_min_maximum": 5.0e-2,
    "q_max_minimum": 15.0,
    "quadrature_order": 2,
}


@dataclass(frozen=True, slots=True)
class NumericalEnvelope:
    """Resolved controls and bounds for one declared CMB request."""

    accuracy_tier: str | None
    bounded: bool
    sectors: tuple[str, ...]
    numerical_controls: Mapping[str, Any]
    hierarchy_controls: Mapping[str, int]
    momentum_grid_controls: Mapping[str, Mapping[str, Any]]
    runtime_limits: Mapping[str, int]
    spectrum_relative_tolerances: Mapping[str, float]
    q_grid_relative_tolerance: float
    hierarchy_relative_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe representation of the envelope."""

        return {
            "accuracy_tier": self.accuracy_tier,
            "bounded": bool(self.bounded),
            "sectors": list(self.sectors),
            "numerical_controls": dict(self.numerical_controls),
            "hierarchy_controls": dict(self.hierarchy_controls),
            "momentum_grid_controls": {
                str(name): dict(values)
                for name, values in self.momentum_grid_controls.items()
            },
            "runtime_limits": dict(self.runtime_limits),
            "spectrum_relative_tolerances": dict(
                self.spectrum_relative_tolerances
            ),
            "q_grid_relative_tolerance": float(self.q_grid_relative_tolerance),
            "hierarchy_relative_tolerance": float(
                self.hierarchy_relative_tolerance
            ),
        }


@dataclass(frozen=True, slots=True)
class RefinementMetric:
    """One measured final-refinement error and its acceptance threshold."""

    name: str
    relative_error: float
    tolerance: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe representation of the metric."""

        return {
            "name": self.name,
            "relative_error": float(self.relative_error),
            "tolerance": float(self.tolerance),
            "converged": bool(self.converged),
        }


@dataclass(frozen=True, slots=True)
class ProductionScalarConvergenceControls:
    """Declared production rule for a doubled scalar wave-number grid."""

    enabled: bool
    k_refinement_factor: int
    required_spectra: tuple[str, ...]
    relative_tolerances: Mapping[str, float]
    fail_on_nonconvergence: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe representation of the production rule."""

        return {
            "enabled": bool(self.enabled),
            "k_refinement_factor": int(self.k_refinement_factor),
            "required_spectra": list(self.required_spectra),
            "relative_tolerances": dict(self.relative_tolerances),
            "fail_on_nonconvergence": bool(self.fail_on_nonconvergence),
        }


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """Named final-refinement metrics across declared physical surfaces."""

    metrics: Mapping[str, RefinementMetric]

    @property
    def converged(self) -> bool:
        """Return whether every measured surface meets its threshold."""

        return bool(self.metrics) and all(
            metric.converged for metric in self.metrics.values()
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe representation of the report."""

        return {
            "converged": self.converged,
            "metrics": {
                str(name): metric.to_dict()
                for name, metric in self.metrics.items()
            },
        }


def _accuracy_controls(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return compiled or raw accuracy controls from ``contract``."""

    perturbation_data = contract.get("perturbation_data")
    if perturbation_data is not None:
        return getattr(perturbation_data, "accuracy_controls", {}) or {}
    perturbations = contract.get("perturbations", {}) or {}
    if isinstance(perturbations, Mapping):
        controls = perturbations.get("accuracy_controls", {}) or {}
        if isinstance(controls, Mapping):
            return controls
    return {}


def resolve_production_scalar_convergence(
    contract: Mapping[str, Any],
) -> ProductionScalarConvergenceControls:
    """Resolve the optional production scalar grid-refinement rule."""

    controls = _accuracy_controls(contract)
    raw = controls.get("production_scalar_convergence", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "production_scalar_convergence must be a mapping when declared"
        )
    enabled = bool(raw.get("enabled", False))
    factor = int(raw.get("k_refinement_factor", 2))
    if factor < 2:
        raise ValueError(
            "production_scalar_convergence.k_refinement_factor must be "
            "at least 2"
        )
    raw_required = raw.get(
        "required_spectra",
        tuple(_PRODUCTION_SCALAR_DEFAULT_TOLERANCES),
    )
    if isinstance(raw_required, (str, bytes)):
        raise ValueError(
            "production_scalar_convergence.required_spectra must be a list"
        )
    required = tuple(str(name).upper() for name in raw_required)
    if not required:
        raise ValueError(
            "production_scalar_convergence.required_spectra must not be empty"
        )
    unknown = [
        name
        for name in required
        if name not in FINAL_SPECTRUM_RELATIVE_TOLERANCES
    ]
    if unknown:
        raise ValueError(
            "Unknown production scalar spectrum(s): " + ", ".join(unknown)
        )
    raw_tolerances = raw.get("relative_tolerances", {}) or {}
    if not isinstance(raw_tolerances, Mapping):
        raise ValueError(
            "production_scalar_convergence.relative_tolerances must be a "
            "mapping"
        )
    tolerances = {
        name: float(
            raw_tolerances.get(
                name,
                _PRODUCTION_SCALAR_DEFAULT_TOLERANCES.get(
                    name,
                    FINAL_SPECTRUM_RELATIVE_TOLERANCES[name],
                ),
            )
        )
        for name in required
    }
    if any(
        not numpy.isfinite(value) or value <= 0.0
        for value in tolerances.values()
    ):
        raise ValueError(
            "production scalar relative tolerances must be finite and positive"
        )
    return ProductionScalarConvergenceControls(
        enabled=enabled,
        k_refinement_factor=factor,
        required_spectra=required,
        relative_tolerances=tolerances,
        fail_on_nonconvergence=bool(raw.get("fail_on_nonconvergence", True)),
    )


def _numerical_controls(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return merged background and hierarchy numerical declarations."""

    merged = dict(_NUMERICAL_DEFAULTS)
    raw_numerical = contract.get("numerical", {}) or {}
    if isinstance(raw_numerical, Mapping):
        merged.update(raw_numerical)
    perturbation_data = contract.get("perturbation_data")
    if perturbation_data is not None:
        hierarchy_numerics = getattr(perturbation_data, "numerics", {}) or {}
    else:
        perturbations = contract.get("perturbations", {}) or {}
        hierarchy_numerics = (
            perturbations.get("numerics", {})
            if isinstance(perturbations, Mapping)
            else {}
        )
    if isinstance(hierarchy_numerics, Mapping):
        merged.update(hierarchy_numerics)
    overrides = contract.get("_numerical_overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("_numerical_overrides must be a mapping")
    merged.update(overrides)
    return merged


def _family_entries(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return compiled or raw hierarchy-family declarations."""

    perturbation_data = contract.get("perturbation_data")
    if perturbation_data is not None:
        return getattr(perturbation_data, "hierarchy_families", {}) or {}
    perturbations = contract.get("perturbations", {}) or {}
    if isinstance(perturbations, Mapping):
        entries = perturbations.get("hierarchy_families", {}) or {}
        if isinstance(entries, Mapping):
            return entries
    return {}


def _entry_value(entry: Any, name: str, default: Any = None) -> Any:
    """Return one field from a compiled dataclass or raw mapping entry."""

    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _hierarchy_kind(family_name: str, entry: Any) -> str | None:
    """Return the physical hierarchy kind represented by one family."""

    name = str(family_name).lower()
    species = {
        str(value).lower()
        for value in (_entry_value(entry, "species", ()) or ())
    }
    if "massive_neutrino" in species or "massive_neutrino" in name:
        return "massive_neutrino"
    if "massless_neutrino" in species or "neutrino" in name:
        return "massless_neutrino"
    if "polarization" in name or "photon_polarization" in species:
        return "photon_polarization"
    if "photon" in species or "photon_temperature" in name:
        return "photon_temperature"
    return None


def _resolved_hierarchy_controls(
    contract: Mapping[str, Any],
    numerical: Mapping[str, Any],
) -> dict[str, int]:
    """Resolve active hierarchy depths from declarations and overrides."""

    defaults: dict[str, int] = {}
    for family_name, entry in _family_entries(contract).items():
        hierarchy_kind = _hierarchy_kind(str(family_name), entry)
        if hierarchy_kind is None:
            continue
        raw_default = _entry_value(entry, "default_l_max")
        if raw_default is None:
            continue
        defaults[hierarchy_kind] = max(
            defaults.get(hierarchy_kind, 0),
            int(raw_default),
        )
    resolved: dict[str, int] = {}
    if "photon_temperature" in defaults:
        resolved["photon_temperature"] = int(
            numerical.get(
                "photon_hierarchy_l_max",
                defaults["photon_temperature"],
            )
        )
    if "photon_polarization" in defaults:
        resolved["photon_polarization"] = int(
            numerical.get(
                "photon_polarization_hierarchy_l_max",
                max(
                    defaults["photon_polarization"],
                    resolved.get("photon_temperature", 0),
                ),
            )
        )
    if "massless_neutrino" in defaults:
        resolved["massless_neutrino"] = int(
            numerical.get(
                "neutrino_hierarchy_l_max",
                defaults["massless_neutrino"],
            )
        )
    if "massive_neutrino" in defaults:
        resolved["massive_neutrino"] = int(
            numerical.get(
                "massive_neutrino_hierarchy_l_max",
                defaults["massive_neutrino"],
            )
        )
    return resolved


def _momentum_grid_controls(
    numerical: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return resolved momentum-grid controls without materializing nodes."""

    raw_grids = numerical.get("momentum_grids", {}) or {}
    if not isinstance(raw_grids, Mapping):
        raise ValueError("Declared momentum-grid numerics must be a mapping")
    resolved: dict[str, dict[str, Any]] = {}
    for grid_name, raw_grid in raw_grids.items():
        if not isinstance(raw_grid, Mapping):
            raise ValueError(
                f"Declared momentum grid '{grid_name}' must be a mapping"
            )
        resolved[str(grid_name)] = {
            "count": int(raw_grid.get("count", 8)),
            "q_min": float(raw_grid.get("q_min", 0.05)),
            "q_max": float(raw_grid.get("q_max", 15.0)),
            "quadrature_order": int(raw_grid.get("quadrature_order", 2)),
        }
    return resolved


def _runtime_limits(
    controls: Mapping[str, Any],
) -> tuple[bool, dict[str, int]]:
    """Resolve optional operator limits without imposing preset ceilings.

    ``bounded`` means that requested work is accounted for and can be
    chunked; it is deliberately not a fixed numerical budget.  Explicit
    limits remain part of the manifest for diagnostics and malformed values
    are rejected, but the projection runtime does not use them to reject a
    valid physical request.
    """

    raw_envelope = controls.get("runtime_envelope")
    if raw_envelope == "bounded":
        return True, {}
    if not isinstance(raw_envelope, Mapping):
        return False, {}
    limits: dict[str, int] = {}
    for name in RUNTIME_WORK_LIMIT_NAMES:
        if name not in raw_envelope:
            continue
        value = int(raw_envelope[name])
        if value < 1:
            raise ValueError(
                f"Declared runtime limit '{name}' must be positive"
            )
        limits[name] = value
    return bool(limits), limits


def _active_sectors(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the canonical active perturbation-sector names."""

    perturbation_data = contract.get("perturbation_data")
    if perturbation_data is not None:
        sectors = getattr(perturbation_data, "sectors", {}) or {}
        variables = getattr(perturbation_data, "variables", {}) or {}
        observables = getattr(perturbation_data, "observables", {}) or {}
    else:
        perturbations = contract.get("perturbations", {}) or {}
        sectors = (
            perturbations.get("sectors", {})
            if isinstance(perturbations, Mapping)
            else {}
        )
        variables = (
            perturbations.get("variables", {})
            if isinstance(perturbations, Mapping)
            else {}
        )
        observables = (
            perturbations.get("observables", {})
            if isinstance(perturbations, Mapping)
            else {}
        )
    if isinstance(sectors, Mapping) and sectors:
        return tuple(sorted(str(name) for name in sectors))

    inferred: set[str] = set()
    tensor_character_sectors = {
        "scalar_like": "scalar",
        "vector_like": "vector",
        "tensor_like": "tensor",
    }
    for entries in (variables, observables):
        if not isinstance(entries, Mapping):
            continue
        for entry in entries.values():
            sector = _entry_value(entry, "sector")
            if sector in {"scalar", "vector", "tensor"}:
                inferred.add(str(sector))
            tensor_character = _entry_value(entry, "tensor_character")
            inferred_sector = tensor_character_sectors.get(
                str(tensor_character)
            )
            if inferred_sector is not None:
                inferred.add(inferred_sector)
    return tuple(sorted(inferred))


def _finite_control(numerical: Mapping[str, Any], name: str) -> float:
    """Return one finite scalar numerical control."""

    value = numerical.get(name)
    if value is None:
        raise ValueError(
            f"Declared accuracy tier requires numerical control '{name}'"
        )
    numeric = float(numpy.asarray(value, dtype=float))
    if not numpy.isfinite(numeric):
        raise ValueError(f"Declared numerical control '{name}' must be finite")
    return numeric


def _validate_final_tier(
    *,
    numerical: Mapping[str, Any],
    hierarchies: Mapping[str, int],
    momentum_grids: Mapping[str, Mapping[str, Any]],
    bounded: bool,
    sectors: Iterable[str],
    low_resolution_override: bool = False,
) -> None:
    """Reject unresolved final controls unless a smoke override exists.

    A generated model may explicitly declare ``minimum_k_sample_count`` below
    the production floor for a finite smoke evaluation.  The projection path
    still promotes that request to its generated production floor; the
    override only prevents the preflight validator from rejecting the request
    before that promotion can occur.
    """

    complaints: list[str] = []
    if not bounded:
        complaints.append(
            "runtime_envelope must be 'bounded' or declare a positive "
            "work-accounting limit"
        )
    if not low_resolution_override:
        for name, minimum in _FINAL_MINIMUM_CONTROLS.items():
            try:
                value = _finite_control(numerical, name)
            except ValueError as exc:
                complaints.append(str(exc))
                continue
            if value < minimum:
                complaints.append(f"{name}={value:g} < {minimum:g}")
    for name, maximum in _FINAL_MAXIMUM_CONTROLS.items():
        try:
            value = _finite_control(numerical, name)
        except ValueError as exc:
            complaints.append(str(exc))
            continue
        if value > maximum:
            complaints.append(f"{name}={value:g} > {maximum:g}")
    hierarchy_minimums: dict[str, int] = {}
    for sector in sectors:
        for name, minimum in _FINAL_HIERARCHY_MINIMUMS_BY_SECTOR.get(
            str(sector),
            {},
        ).items():
            hierarchy_minimums[name] = max(
                hierarchy_minimums.get(name, 0),
                int(minimum),
            )
    if not low_resolution_override:
        for name, minimum in hierarchy_minimums.items():
            if name not in hierarchies:
                continue
            value = int(hierarchies[name])
            if value < minimum:
                complaints.append(f"{name}_l_max={value} < {minimum}")
        for grid_name, grid in momentum_grids.items():
            count = int(grid["count"])
            q_min = float(grid["q_min"])
            q_max = float(grid["q_max"])
            order = int(grid["quadrature_order"])
            if count < int(_FINAL_MOMENTUM_GRID_MINIMUMS["count"]):
                complaints.append(
                    f"momentum_grids.{grid_name}.count={count} < "
                    f"{_FINAL_MOMENTUM_GRID_MINIMUMS['count']}"
                )
            if q_min > float(_FINAL_MOMENTUM_GRID_MINIMUMS["q_min_maximum"]):
                complaints.append(
                    f"momentum_grids.{grid_name}.q_min={q_min:g} > "
                    f"{_FINAL_MOMENTUM_GRID_MINIMUMS['q_min_maximum']:g}"
                )
            if q_max < float(_FINAL_MOMENTUM_GRID_MINIMUMS["q_max_minimum"]):
                complaints.append(
                    f"momentum_grids.{grid_name}.q_max={q_max:g} < "
                    f"{_FINAL_MOMENTUM_GRID_MINIMUMS['q_max_minimum']:g}"
                )
            if order != int(_FINAL_MOMENTUM_GRID_MINIMUMS["quadrature_order"]):
                complaints.append(
                    f"momentum_grids.{grid_name}.quadrature_order={order} != "
                    f"{_FINAL_MOMENTUM_GRID_MINIMUMS['quadrature_order']}"
                )
    if complaints:
        raise ValueError(
            "Declared accuracy tier 'final' is under-resolved: "
            + "; ".join(complaints)
        )


def resolve_declared_numerical_envelope(
    contract: Mapping[str, Any],
) -> NumericalEnvelope:
    """Resolve and validate the active declared numerical accuracy envelope."""

    controls = _accuracy_controls(contract)
    raw_tier = controls.get("accuracy_tier")
    accuracy_tier = None if raw_tier is None else str(raw_tier).strip()
    if accuracy_tier not in {None, "final"}:
        raise ValueError(
            "cmb.perturbations.accuracy_controls.accuracy_tier must be "
            "'final' when declared"
        )
    numerical = _numerical_controls(contract)
    hierarchies = _resolved_hierarchy_controls(contract, numerical)
    momentum_grids = _momentum_grid_controls(numerical)
    bounded, runtime_limits = _runtime_limits(controls)
    sectors = _active_sectors(contract)
    if accuracy_tier == "final":
        raw_minimum_k = controls.get("minimum_k_sample_count")
        low_resolution_override = False
        if raw_minimum_k is not None:
            low_resolution_override = int(raw_minimum_k) < int(
                _FINAL_MINIMUM_CONTROLS["k_sample_count"]
            )
        _validate_final_tier(
            numerical=numerical,
            hierarchies=hierarchies,
            momentum_grids=momentum_grids,
            bounded=bounded,
            sectors=sectors,
            low_resolution_override=low_resolution_override,
        )
    numerical_names = tuple(_NUMERICAL_DEFAULTS)
    resolved_numerical = {
        name: numerical.get(name) for name in numerical_names
    }
    return NumericalEnvelope(
        accuracy_tier=accuracy_tier,
        bounded=bounded,
        sectors=sectors,
        numerical_controls=resolved_numerical,
        hierarchy_controls=hierarchies,
        momentum_grid_controls=momentum_grids,
        runtime_limits=runtime_limits,
        spectrum_relative_tolerances=FINAL_SPECTRUM_RELATIVE_TOLERANCES,
        q_grid_relative_tolerance=FINAL_Q_GRID_RELATIVE_TOLERANCE,
        hierarchy_relative_tolerance=FINAL_HIERARCHY_RELATIVE_TOLERANCE,
    )


def _finite_array(values: Any, *, name: str) -> numpy.ndarray:
    """Return one finite floating refinement surface."""

    array = numpy.asarray(values, dtype=numpy.longdouble)
    if array.size == 0 or not numpy.all(numpy.isfinite(array)):
        raise ValueError(
            f"Declared refinement surface '{name}' must be finite"
        )
    return array


def _fractional_refinement_error(
    coarse: numpy.ndarray,
    fine: numpy.ndarray,
) -> float:
    """Return the relative L-infinity change between two surfaces."""

    if coarse.shape != fine.shape:
        raise ValueError("Declared refinement surfaces must have equal shapes")
    peak = numpy.max(numpy.abs(fine), initial=numpy.longdouble(0.0))
    scale = max(peak, numpy.finfo(numpy.longdouble).tiny)
    return float(numpy.max(numpy.abs(fine - coarse), initial=0.0) / scale)


def _normalized_te(
    spectra: Mapping[str, Any],
    *,
    label: str,
) -> numpy.ndarray:
    """Return the TE correlation coefficient for one refinement level."""

    missing = [name for name in ("TT", "TE", "EE") if name not in spectra]
    if missing:
        raise ValueError(
            f"Normalized TE {label} refinement requires " + ", ".join(missing)
        )
    temperature_power = _finite_array(spectra["TT"], name=f"{label} TT")
    temperature_electric_power = _finite_array(
        spectra["TE"], name=f"{label} TE"
    )
    electric_power = _finite_array(spectra["EE"], name=f"{label} EE")
    if (
        temperature_power.shape != temperature_electric_power.shape
        or electric_power.shape != temperature_electric_power.shape
    ):
        raise ValueError("Normalized TE surfaces must have equal shapes")
    denominator = numpy.sqrt(numpy.abs(temperature_power)) * numpy.sqrt(
        numpy.abs(electric_power)
    )
    floor = max(
        numpy.max(denominator, initial=numpy.longdouble(0.0))
        * numpy.longdouble("1.0e-12"),
        numpy.finfo(numpy.longdouble).tiny,
    )
    return numpy.asarray(
        temperature_electric_power / numpy.maximum(denominator, floor),
        dtype=numpy.longdouble,
    )


def evaluate_spectrum_refinement(
    coarse_spectra: Mapping[str, Any],
    fine_spectra: Mapping[str, Any],
    *,
    required_spectra: Iterable[str] | None = None,
    relative_tolerances: Mapping[str, float] | None = None,
) -> ConvergenceReport:
    """Measure final declared spectrum refinement against thresholds."""

    required = tuple(
        required_spectra
        if required_spectra is not None
        else FINAL_SPECTRUM_RELATIVE_TOLERANCES
    )
    tolerance_overrides = dict(relative_tolerances or {})
    metrics: dict[str, RefinementMetric] = {}
    for raw_name in required:
        name = str(raw_name)
        if name not in FINAL_SPECTRUM_RELATIVE_TOLERANCES:
            raise ValueError(f"Unknown final convergence spectrum '{name}'")
        if name == "TE":
            coarse_te = _finite_array(coarse_spectra["TE"], name="coarse TE")
            fine_te = _finite_array(fine_spectra["TE"], name="fine TE")
            coarse = _normalized_te(coarse_spectra, label="coarse")
            fine = _normalized_te(fine_spectra, label="fine")
            coarse_tt = _finite_array(coarse_spectra["TT"], name="coarse TT")
            coarse_ee = _finite_array(coarse_spectra["EE"], name="coarse EE")
            fine_tt = _finite_array(fine_spectra["TT"], name="fine TT")
            fine_ee = _finite_array(fine_spectra["EE"], name="fine EE")
            denominator = numpy.minimum(
                numpy.sqrt(numpy.abs(coarse_tt))
                * numpy.sqrt(numpy.abs(coarse_ee)),
                numpy.sqrt(numpy.abs(fine_tt))
                * numpy.sqrt(numpy.abs(fine_ee)),
            )
            signal_floor = numpy.max(denominator, initial=0.0) * 1.0e-2
            supported = denominator >= signal_floor
            normalized_error = (
                float(
                    numpy.max(numpy.abs(fine[supported] - coarse[supported]))
                )
                if numpy.any(supported)
                else 0.0
            )
            te_scale = max(
                numpy.max(numpy.abs(coarse_te), initial=0.0),
                numpy.max(numpy.abs(fine_te), initial=0.0),
                numpy.finfo(numpy.longdouble).tiny,
            )
            amplitude_error = float(
                numpy.max(numpy.abs(fine_te - coarse_te), initial=0.0)
                / te_scale
            )
            relative_error = max(normalized_error, amplitude_error)
        else:
            if name not in coarse_spectra or name not in fine_spectra:
                raise ValueError(
                    f"Declared spectrum refinement requires '{name}' at "
                    "both resolutions"
                )
            coarse = _finite_array(coarse_spectra[name], name=f"coarse {name}")
            fine = _finite_array(fine_spectra[name], name=f"fine {name}")
            relative_error = _fractional_refinement_error(coarse, fine)
        tolerance = float(
            tolerance_overrides.get(
                name,
                FINAL_SPECTRUM_RELATIVE_TOLERANCES[name],
            )
        )
        if not numpy.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                f"Spectrum refinement tolerance for {name} must be positive"
            )
        metrics[name] = RefinementMetric(
            name=name,
            relative_error=relative_error,
            tolerance=tolerance,
            converged=bool(relative_error < tolerance),
        )
    return ConvergenceReport(metrics=metrics)


def evaluate_control_refinement(
    coarse: Any,
    fine: Any,
    *,
    name: str,
    tolerance: float,
) -> RefinementMetric:
    """Measure one hierarchy, q-grid, background, or grid refinement."""

    coarse_values = _finite_array(coarse, name=f"coarse {name}")
    fine_values = _finite_array(fine, name=f"fine {name}")
    relative_error = _fractional_refinement_error(coarse_values, fine_values)
    threshold = float(tolerance)
    if not numpy.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Declared refinement tolerance must be positive")
    return RefinementMetric(
        name=str(name),
        relative_error=relative_error,
        tolerance=threshold,
        converged=bool(relative_error < threshold),
    )


def require_convergence(
    report: ConvergenceReport | RefinementMetric,
) -> None:
    """Raise a named error when final numerical refinement is unresolved."""

    if isinstance(report, RefinementMetric):
        metrics = {report.name: report}
    else:
        metrics = report.metrics
    failed = [metric for metric in metrics.values() if not metric.converged]
    if not failed:
        return
    details = ", ".join(
        f"{metric.name}={metric.relative_error:.6g} "
        f">= {metric.tolerance:.6g}"
        for metric in failed
    )
    raise ValueError(f"Declared final numerical refinement failed: {details}")


__all__ = [
    "FINAL_HIERARCHY_RELATIVE_TOLERANCE",
    "FINAL_Q_GRID_RELATIVE_TOLERANCE",
    "FINAL_SPECTRUM_RELATIVE_TOLERANCES",
    "ConvergenceReport",
    "NumericalEnvelope",
    "ProductionScalarConvergenceControls",
    "RUNTIME_WORK_LIMIT_NAMES",
    "RefinementMetric",
    "evaluate_control_refinement",
    "evaluate_spectrum_refinement",
    "require_convergence",
    "resolve_production_scalar_convergence",
    "resolve_declared_numerical_envelope",
]
