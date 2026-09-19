"""Engine-owned numerical planning for declared CMB calculations.

Model files describe equations and physical domains.  This module turns the
requested observable surface and the compiled physical graph into a stable
numerical plan; no model name or model-provided solver setting is consulted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_ENGINE_VERSION = "cmb-engine-planner-v3"
_PLANNER_REQUEST_MODES = frozenset({"production", "diagnostic"})


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` when it is a mapping, otherwise an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _field(entry: Any, name: str, default: Any = None) -> Any:
    """Read one field from either a raw declaration or compiled metadata."""

    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _tuple_strings(value: Any) -> tuple[str, ...]:
    """Normalize a declaration list for deterministic planner evidence."""

    if value is None or isinstance(value, (str, bytes)):
        return () if value is None else (str(value),)
    try:
        return tuple(sorted(str(item) for item in value))
    except TypeError:
        return (str(value),)


def _requested_ells(ells: Sequence[int] | None) -> tuple[int, ...]:
    """Normalize an optional multipole request to a deterministic tuple."""

    if ells is None:
        return tuple(range(2, 2501))
    values = sorted({max(2, int(value)) for value in ells})
    return tuple(values) or (2,)


def _next_power_of_two(value: int) -> int:
    """Return the least power of two greater than or equal to ``value``."""

    result = 1
    while result < value:
        result *= 2
    return result


@dataclass(frozen=True, slots=True)
class CMBNumericalPlan:
    """Immutable numerical decision made by the CCMBS engine."""

    planner_version: str
    ell_range: tuple[int, int]
    requested_spectra: tuple[str, ...]
    numerical_controls: Mapping[str, Any]
    hierarchy_controls: Mapping[str, int]
    momentum_grid_controls: Mapping[str, Mapping[str, Any]]
    accuracy_controls: Mapping[str, Any]
    physical_scale_evidence: Mapping[str, Any]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe plan and its physical decision evidence."""

        return {
            "planner_version": self.planner_version,
            "ell_range": list(self.ell_range),
            "requested_spectra": list(self.requested_spectra),
            "numerical_controls": dict(self.numerical_controls),
            "hierarchy_controls": dict(self.hierarchy_controls),
            "momentum_grid_controls": {
                str(name): dict(values)
                for name, values in self.momentum_grid_controls.items()
            },
            "accuracy_controls": dict(self.accuracy_controls),
            "physical_scale_evidence": dict(self.physical_scale_evidence),
            "signature": self.signature,
        }


def plan_cmb_numerics(
    contract: Mapping[str, Any],
    *,
    ells: Sequence[int] | None = None,
    spectra: Sequence[str] = (),
    request_mode: str = "production",
) -> CMBNumericalPlan:
    """Plan all solver grids from physical declarations and request shape.

    The planner deliberately accepts both a raw declaration and a compiled
    runtime contract.  Only physical graph metadata is inspected.  Legacy
    ``numerical`` and ``accuracy_controls`` keys are ignored here so a model
    cannot steer the engine's resolution.  ``diagnostic`` is reserved for
    explicit reduced-grid fixtures; normal public requests are production.
    """

    request_mode = str(request_mode).strip().lower()
    if request_mode not in _PLANNER_REQUEST_MODES:
        allowed_modes = ", ".join(sorted(_PLANNER_REQUEST_MODES))
        raise ValueError(
            f"Unsupported CMB planner request mode {request_mode!r}; "
            f"expected one of: {allowed_modes}"
        )

    requested_ell_values = _requested_ells(ells)
    # Scalar transfer histories always include the physical quadrupole.  A
    # request that starts at a later ell therefore reuses the same low-ell
    # plan instead of changing already-computed multipoles.
    ell_min = 2
    ell_max = int(requested_ell_values[-1])
    spectrum_order = ("TT", "TE", "EE", "BB", "PP", "TP", "EP")
    requested_spectra = tuple(
        name
        for name in spectrum_order
        if name in {str(value).upper() for value in spectra}
    )
    perturbations = _mapping(contract.get("perturbations"))
    compiled_perturbations = contract.get("perturbation_data")
    if compiled_perturbations is not None:
        sectors = _mapping(getattr(compiled_perturbations, "sectors", {}))
        species = _mapping(getattr(compiled_perturbations, "species", {}))
        families = _mapping(
            getattr(compiled_perturbations, "hierarchy_families", {})
        )
        observables = _mapping(
            getattr(compiled_perturbations, "observables", {})
        )
    else:
        sectors = _mapping(perturbations.get("sectors"))
        species = _mapping(perturbations.get("species"))
        families = _mapping(perturbations.get("hierarchy_families"))
        observables = _mapping(perturbations.get("observables"))

    sector_names = tuple(sorted(str(name) for name in sectors)) or ("scalar",)
    family_names = tuple(sorted(str(name) for name in families))
    has_massive = any(
        "massive" in name.lower() for name in family_names
    ) or any("massive" in str(name).lower() for name in species)

    family_rows = tuple(
        {
            "name": str(name),
            "sector": str(_field(entry, "sector", "") or ""),
            "species": _tuple_strings(_field(entry, "species", ())),
            "closure": str(_field(entry, "closure", "") or ""),
            "multipole_symbol": str(
                _field(entry, "multipole_symbol", "") or ""
            ),
            "momentum_grid": str(_field(entry, "momentum_grid", "") or ""),
        }
        for name, entry in sorted(
            families.items(), key=lambda item: str(item[0])
        )
    )
    collision_entries = _mapping(
        getattr(
            contract.get("perturbation_data"),
            "collision_operators",
            {},
        )
        if contract.get("perturbation_data") is not None
        else perturbations.get("collision_operators")
    )
    collision_rows = tuple(
        {
            "name": str(name),
            "sector": str(_field(entry, "sector", "") or ""),
            "species": _tuple_strings(_field(entry, "species", ())),
            "integration_strategy": str(
                _field(entry, "integration_strategy", "explicit")
            ),
            "activation_strategy": str(
                _field(entry, "activation_strategy", "always")
            ),
            "counterpart": str(_field(entry, "counterpart", "") or ""),
            "rate_dependencies": _tuple_strings(
                _field(entry, "rate_dependencies", ())
            ),
            "has_exact_form": bool(
                _field(entry, "exact_form", None) is not None
            ),
            "has_linear_block": bool(
                _field(entry, "linear_block", None) is not None
            ),
        }
        for name, entry in sorted(
            collision_entries.items(), key=lambda item: str(item[0])
        )
    )
    initial_entries = _mapping(
        getattr(
            contract.get("perturbation_data"),
            "initial_conditions",
            {},
        )
        if contract.get("perturbation_data") is not None
        else perturbations.get("initial_conditions")
    )
    boundary_entries = _mapping(
        getattr(
            contract.get("perturbation_data"),
            "boundary_conditions",
            {},
        )
        if contract.get("perturbation_data") is not None
        else perturbations.get("boundary_conditions")
    )
    initial_condition_rows = tuple(
        {
            "name": str(name),
            "anchor": "start",
            "target": str(
                _field(_field(entry, "target", None), "variable", "") or ""
            ),
        }
        for name, entry in sorted(
            initial_entries.items(), key=lambda item: str(item[0])
        )
    ) + tuple(
        {
            "name": str(name),
            "anchor": str(_field(entry, "anchor", "start")),
            "target": str(
                _field(_field(entry, "target", None), "variable", "") or ""
            ),
        }
        for name, entry in sorted(
            boundary_entries.items(), key=lambda item: str(item[0])
        )
    )
    equation_entries = _mapping(
        getattr(contract.get("perturbation_data"), "equations", {})
        if contract.get("perturbation_data") is not None
        else perturbations.get("equations")
    )
    closure_entries = _mapping(
        getattr(contract.get("perturbation_data"), "closures", {})
        if contract.get("perturbation_data") is not None
        else perturbations.get("closures")
    )
    constraint_entries = _mapping(
        getattr(contract.get("perturbation_data"), "constraints", {})
        if contract.get("perturbation_data") is not None
        else perturbations.get("constraints")
    )
    graph_complexity = (
        len(equation_entries)
        + len(closure_entries)
        + len(constraint_entries)
        + len(collision_entries)
    )
    transfer_routes = tuple(
        {
            "name": str(name),
            "projection": str(_field(entry, "projection", "") or ""),
            "kernel": str(_field(entry, "kernel", "") or ""),
            "sector": str(_field(entry, "sector", "") or "scalar"),
            "source_roles": _tuple_strings(
                _field(entry, "source_terms", {}).keys()
                if isinstance(_field(entry, "source_terms", {}), Mapping)
                else ()
            ),
        }
        for name, entry in sorted(
            observables.items(), key=lambda item: str(item[0])
        )
        if str(_field(entry, "kind", "")) == "transfer_component"
    )
    spectrum_edges = tuple(
        {
            "name": str(name),
            "primary": str(_field(entry, "primary", "") or ""),
            "secondary": str(_field(entry, "secondary", "") or ""),
        }
        for name, entry in sorted(
            observables.items(), key=lambda item: str(item[0])
        )
        if str(_field(entry, "kind", "")) == "angular_power_spectrum"
    )

    # The phase scale k(eta_0-eta_*) grows approximately linearly with ell.
    # The bounded formulas are deterministic and intentionally conservative;
    # later refinement is driven by measured residuals, never by model knobs.
    phase_scale = max(1.0e-5, float(ell_max) / 14000.0)
    k_max = min(1.0, max(0.30, 2.0 * phase_scale + 0.05))
    k_min = max(1.0e-5, min(1.0e-4, 0.02 / max(ell_max, 2)))
    k_nodes = min(
        512,
        max(64, _next_power_of_two(32 + int(3.0 * math.sqrt(ell_max)))),
    )
    eta_nodes = min(1024, max(192, 128 + int(math.sqrt(ell_max) * 8.0)))
    evolution_nodes = min(512, max(128, eta_nodes // 2))
    if request_mode == "production" and ell_max >= 200:
        # Recombination and reionization histories need the same final-tier
        # background floor used by the bundled production runtime.  A purely
        # ell-scaled diagnostic budget is too sparse for the visibility
        # refinement even when the requested surface reaches the first
        # acoustic feature.
        eta_nodes = max(528, eta_nodes)
        evolution_nodes = max(264, evolution_nodes)

    photon_l_max = max(10, 4 + int(math.ceil(math.sqrt(ell_max) / 4.0)))
    polarization_l_max = max(photon_l_max, 10)
    neutrino_l_max = max(8, photon_l_max - 2)
    massive_l_max = max(7, neutrino_l_max - 1)
    hierarchy_controls = {
        "photon_temperature": int(photon_l_max),
        "photon_polarization": int(polarization_l_max),
        "massless_neutrino": int(neutrino_l_max),
        "massive_neutrino": int(massive_l_max if has_massive else 0),
    }
    hierarchy_family_controls = {
        row["name"]: int(
            max(
                3,
                (
                    photon_l_max
                    if "photon" in row["name"].lower()
                    else (
                        massive_l_max
                        if row["momentum_grid"]
                        else (
                            neutrino_l_max
                            if "neutrino" in row["name"].lower()
                            else photon_l_max
                        )
                    )
                ),
            )
        )
        for row in family_rows
    }
    momentum_grid_controls: dict[str, Mapping[str, Any]] = {}
    if has_massive:
        momentum_grid_controls["massive_neutrino_default"] = {
            # The q support is selected from the thermal tail.  Thirty-two
            # log nodes resolve the relativistic-to-nonrelativistic moments;
            # the runtime performs an independent doubled-count check when
            # a massive family is present.
            "count": 32,
            "q_min": 0.02,
            "q_max": 24.0,
            "mass_parameter": "sum_mnu",
        }

    numerical_controls = {
        "ell_min": ell_min,
        "ell_max": ell_max,
        "k_min": float(k_min),
        "k_max": float(k_max),
        "k_sample_count": int(k_nodes),
        "eta_sample_count": int(eta_nodes),
        "evolution_eta_sample_count": int(evolution_nodes),
        "evolution_phase_step": 0.5,
        "photon_hierarchy_l_max": int(photon_l_max),
        "photon_polarization_hierarchy_l_max": int(polarization_l_max),
        "neutrino_hierarchy_l_max": int(neutrino_l_max),
        "massive_neutrino_hierarchy_l_max": int(massive_l_max),
        "ode_rtol": 1.0e-6,
        "ode_atol": 1.0e-9,
        # The transition detector is a physical stiffness criterion, not a
        # model knob.  Keep the engine envelope above the final-tier minimum
        # so tight coupling is exited only after the collision rate is safely
        # resolved.
        "tight_coupling_ratio": 2000.0,
        "tight_coupling_exit_ratio": 0.1,
        "a_min": 1.0e-8,
        "source_grid_multiplier": 2,
        "initial_redshift": 1.0e5,
        "lensing_sampling_factor": 1.4,
        "background_refinement_factor": 2,
        "background_refinement_tolerance": 1.0e-2,
    }
    accuracy_controls = {
        "runtime_envelope": "bounded",
        "source_history_reconstruction": True,
    }
    if request_mode == "production":
        # Every normal request, including an explicit ell array, carries the
        # final numerical envelope owned by the engine.  Reduced-grid tests
        # must opt into the diagnostic boundary explicitly.
        required_spectra = tuple(
            name
            for name in requested_spectra
            if name in {"TT", "TE", "EE", "PP"}
        ) or ("TT", "TE", "EE")
        accuracy_controls.update(
            {
                # Scalar Einstein checks are engine-owned acceptance
                # diagnostics. The model declaration supplies the equations;
                # these tolerances and history density belong to the solver.
                "scalar_constraint_normalization": (
                    "sum_abs_declared_einstein_terms"
                ),
                "scalar_constraint_reference_eta_samples": int(eta_nodes),
                "scalar_constraint_tolerances": {
                    "einstein_energy_residual": 1.0e-3,
                    # The normalized momentum equation is a subtraction of
                    # the radiation and matter source terms.  At the first
                    # super-horizon samples its float64 cancellation envelope
                    # is about five parts in 10^6 even when the evolved
                    # histories are otherwise converged.
                    "einstein_momentum_residual": 5.0e-6,
                    "einstein_shear_residual": 1.0e-6,
                },
                "accuracy_tier": "final",
                "phase_aware_k_quadrature": True,
                "minimum_k_sample_count": 64,
                "minimum_eta_sample_count": 192,
                "minimum_evolution_eta_sample_count": 128,
                "minimum_source_grid_multiplier": 2,
                "minimum_photon_hierarchy_l_max": 10,
                "minimum_photon_polarization_hierarchy_l_max": 10,
                "minimum_neutrino_hierarchy_l_max": 7,
                "minimum_massive_neutrino_hierarchy_l_max": 7,
                "minimum_lensing_sampling_factor": 1.4,
                "background_refinement_factor": 2,
                "background_refinement_tolerance": 1.0e-2,
                "production_scalar_convergence": {
                    "enabled": True,
                    "k_refinement_factor": 2,
                    "required_spectra": required_spectra,
                    "fail_on_nonconvergence": True,
                },
            }
        )
    # Runtime consumers use this envelope for diagnostics, while its values
    # remain engine-owned and are not accepted from model declarations.
    physical_scale_evidence = {
        "sector_names": sector_names,
        "hierarchy_family_names": family_names,
        "massive_neutrino_sector": bool(has_massive),
        "ell_count": len(requested_ell_values),
        "request_mode": request_mode,
        "phase_scale_k_eta": float(phase_scale),
        "selection": "request_shape_and_declared_physical_graph",
        "background_resolution": "equation_scales_visibility_width_drag_depth",
        "massive_neutrino_resolution": (
            "thermal_fermi_dirac_q_moments" if has_massive else "not_declared"
        ),
        "graph_complexity": int(graph_complexity),
        "hierarchy_resolution": {
            "method": "phase_visibility_and_declared_closure",
            "ell_max": int(ell_max),
            "phase_scale_k_eta": float(phase_scale),
            "family_l_max": hierarchy_family_controls,
            "state_equation_count": int(len(equation_entries)),
            "sector_count": int(len(sector_names)),
        },
        "collision_schedule": {
            "method": "declared_rate_phase_partition",
            "operator_count": int(len(collision_rows)),
            "operators": collision_rows,
            "tight_coupling_entry": "collision_rate >= k * engine_ratio",
            "tight_coupling_exit": "collision_rate < exit_ratio * entry",
        },
        "initial_condition_resolution": {
            "method": "declared_start_and_boundary_conditions",
            "conditions": initial_condition_rows,
            "hidden_prefix": True,
            "state_slot_count": int(len(equation_entries)),
        },
        "projection_resolution": {
            "method": "declared_source_route_and_phase_kernel",
            "transfer_route_count": int(len(transfer_routes)),
            "spectrum_edge_count": int(len(spectrum_edges)),
            "sectors": tuple(
                sorted(
                    {
                        str(row["sector"])
                        for row in transfer_routes
                        if row["sector"]
                    }
                )
            ),
            "kernels": tuple(
                sorted(
                    {
                        str(row["kernel"])
                        for row in transfer_routes
                        if row["kernel"]
                    }
                )
            ),
            "phase_anchors": {
                "ell_min": int(ell_min),
                "ell_max": int(ell_max),
                "k_nodes": int(k_nodes),
                "eta_nodes": int(eta_nodes),
            },
            "independent_refinement_axes": (
                "k",
                "eta",
                "source",
                "projection",
            ),
        },
    }
    payload = {
        "planner_version": _ENGINE_VERSION,
        "ell_range": (ell_min, ell_max),
        "requested_spectra": requested_spectra,
        "request_mode": request_mode,
        "numerical_controls": numerical_controls,
        "hierarchy_controls": hierarchy_controls,
        "momentum_grid_controls": momentum_grid_controls,
        "accuracy_controls": accuracy_controls,
        "physical_scale_evidence": physical_scale_evidence,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()[:24]
    return CMBNumericalPlan(
        planner_version=_ENGINE_VERSION,
        ell_range=(ell_min, ell_max),
        requested_spectra=requested_spectra,
        numerical_controls=numerical_controls,
        hierarchy_controls=hierarchy_controls,
        momentum_grid_controls=momentum_grid_controls,
        accuracy_controls=accuracy_controls,
        physical_scale_evidence=physical_scale_evidence,
        signature=signature,
    )


def planner_accuracy_controls(
    contract: Mapping[str, Any],
    *,
    ells: Sequence[int] | None = None,
    spectra: Sequence[str] = (),
    request_mode: str = "production",
) -> Mapping[str, Any]:
    """Return the engine-generated accuracy envelope for a request."""

    plan = plan_cmb_numerics(
        contract,
        ells=ells,
        spectra=spectra,
        request_mode=request_mode,
    )
    return dict(plan.accuracy_controls)


def build_cmb_planner_manifest(
    contracts: Mapping[str, Mapping[str, Any]],
    *,
    ells: Sequence[int] | None = None,
    spectra: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic raw planner manifest for declared models.

    The manifest is deliberately a pure serialization of planner decisions;
    it does not execute a model or alter the runtime cache.  Callers can
    persist it beside diagnostic products as reproducible planning evidence.
    """

    requested_ell_values = _requested_ells(ells)
    requested_spectra = tuple(sorted({str(name) for name in spectra}))
    models = {
        str(model_key): plan_cmb_numerics(
            contract,
            ells=requested_ell_values,
            spectra=requested_spectra,
        ).to_dict()
        for model_key, contract in sorted(
            contracts.items(), key=lambda entry: str(entry[0])
        )
    }
    return {
        "schema_version": 1,
        "planner_version": _ENGINE_VERSION,
        "request": {
            "ells": list(requested_ell_values),
            "spectra": list(requested_spectra),
        },
        "models": models,
    }


__all__ = [
    "CMBNumericalPlan",
    "build_cmb_planner_manifest",
    "plan_cmb_numerics",
    "planner_accuracy_controls",
]
