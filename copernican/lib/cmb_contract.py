# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Declared CMB contract helpers for solver integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model_adapter import (
    _SUPPORTED_CMB_CALL_KEYS,
    _SUPPORTED_CMB_CALL_METHODS,
    _SUPPORTED_CMB_CONTRACT_KEYS,
    _SUPPORTED_CMB_GRID_KEYS,
    _SUPPORTED_CMB_GRID_SPACING,
    _SUPPORTED_CMB_PARAMETER_KEYS,
    _SUPPORTED_CMB_PERTURBATION_GAUGES,
    _SUPPORTED_CMB_PERTURBATION_KEYS,
    _SUPPORTED_CMB_VALUE_KEYS,
    _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES,
    CMBContractEvaluator,
    CMBParameterEvaluator,
    FrozenMapping,
    _validate_cmb_contract_definition,
)
from .perturbation_contract import PerturbationContractData


@dataclass(frozen=True, slots=True)
class CMBObservableRequirement:
    """Minimum transfer capabilities for one public CMB spectrum."""

    name: str
    required_transfer_roles: tuple[str, ...]
    required_sectors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CMBObservableCapability:
    """Machine-readable result for one requested observable."""

    name: str
    available: bool
    required_transfer_roles: tuple[str, ...]
    required_sectors: tuple[str, ...]
    declared_components: tuple[str, ...] = ()
    declared_sector: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CMBCapabilityAudit:
    """Complete declared capability inventory for one compiled contract."""

    model_name: str
    contract_version: int
    gauge: str
    sectors: tuple[str, ...]
    species: tuple[str, ...]
    hierarchy_families: tuple[str, ...]
    collision_operators: tuple[str, ...]
    interactions: tuple[str, ...]
    closures: tuple[str, ...]
    initial_condition_families: tuple[str, ...]
    projection_typing: tuple[str, ...]
    background_references: tuple[str, ...]
    numerical_controls: tuple[str, ...]
    validity_regimes: tuple[str, ...]
    declared_observables: tuple[str, ...]
    capability_matrix: tuple[CMBObservableCapability, ...]
    generated_hierarchies: tuple[str, ...]
    execution_solver_id: str
    execution_runtime_module: str

    @property
    def supported_observables(self) -> tuple[str, ...]:
        """Return public observables whose declared graph is complete."""

        return tuple(
            entry.name for entry in self.capability_matrix if entry.available
        )

    @property
    def unsupported_observables(self) -> tuple[str, ...]:
        """Return public observables unavailable from this graph."""

        return tuple(
            entry.name
            for entry in self.capability_matrix
            if not entry.available
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic manifest-safe capability mapping."""

        return {
            "model_name": self.model_name,
            "contract_version": self.contract_version,
            "gauge": self.gauge,
            "sectors": self.sectors,
            "species": self.species,
            "hierarchy_families": self.hierarchy_families,
            "collision_operators": self.collision_operators,
            "interactions": self.interactions,
            "closures": self.closures,
            "initial_condition_families": self.initial_condition_families,
            "projection_typing": self.projection_typing,
            "background_references": self.background_references,
            "numerical_controls": self.numerical_controls,
            "validity_regimes": self.validity_regimes,
            "declared_observables": self.declared_observables,
            "supported_observables": self.supported_observables,
            "unsupported_observables": self.unsupported_observables,
            "capability_matrix": {
                entry.name: {
                    "available": entry.available,
                    "required_transfer_roles": entry.required_transfer_roles,
                    "required_sectors": entry.required_sectors,
                    "declared_components": entry.declared_components,
                    "declared_sector": entry.declared_sector,
                    "reason": entry.reason,
                }
                for entry in self.capability_matrix
            },
            "execution_route": {
                "solver_id": self.execution_solver_id,
                "runtime_module": self.execution_runtime_module,
                "routing_basis": "declared_contract_and_universal_rules",
            },
            "generated_hierarchies": self.generated_hierarchies,
        }


CMB_OBSERVABLE_REQUIREMENTS = FrozenMapping(
    {
        "TT": CMBObservableRequirement("TT", ("temperature", "temperature")),
        "TE": CMBObservableRequirement(
            "TE", ("temperature", "polarization_e")
        ),
        "EE": CMBObservableRequirement(
            "EE", ("polarization_e", "polarization_e")
        ),
        "BB": CMBObservableRequirement(
            "BB", ("polarization_b", "polarization_b")
        ),
        "PP": CMBObservableRequirement(
            "PP", ("potential", "potential"), ("scalar",)
        ),
        "TP": CMBObservableRequirement(
            "TP", ("temperature", "potential"), ("scalar",)
        ),
        "EP": CMBObservableRequirement(
            "EP", ("polarization_e", "potential"), ("scalar",)
        ),
    }
)


def _capability_entry(
    contract: PerturbationContractData,
    requirement: CMBObservableRequirement,
) -> CMBObservableCapability:
    """Derive one capability row from compiled observable metadata."""

    observable = contract.observables.get(requirement.name)
    if observable is None:
        return CMBObservableCapability(
            name=requirement.name,
            available=False,
            required_transfer_roles=requirement.required_transfer_roles,
            required_sectors=requirement.required_sectors,
            declared_sector=None,
            reason=(
                "missing declared angular_power_spectrum observable "
                f"'{requirement.name}'"
            ),
        )
    if observable.kind != "angular_power_spectrum":
        return CMBObservableCapability(
            name=requirement.name,
            available=False,
            required_transfer_roles=requirement.required_transfer_roles,
            required_sectors=requirement.required_sectors,
            declared_sector=None,
            reason=(
                f"declared observable has kind '{observable.kind}', not "
                "angular_power_spectrum"
            ),
        )
    primary = contract.observables.get(observable.primary)
    secondary = contract.observables.get(observable.secondary)
    if primary is None or secondary is None:
        missing = (
            observable.primary if primary is None else observable.secondary
        )
        return CMBObservableCapability(
            name=requirement.name,
            available=False,
            required_transfer_roles=requirement.required_transfer_roles,
            required_sectors=requirement.required_sectors,
            declared_sector=None,
            reason=f"missing declared transfer component '{missing}'",
        )
    actual_roles = (str(primary.output_role), str(secondary.output_role))
    components = (str(primary.name), str(secondary.name))
    if tuple(sorted(actual_roles)) != tuple(
        sorted(requirement.required_transfer_roles)
    ):
        return CMBObservableCapability(
            name=requirement.name,
            available=False,
            required_transfer_roles=requirement.required_transfer_roles,
            required_sectors=requirement.required_sectors,
            declared_components=components,
            declared_sector=observable.sector,
            reason=(
                "declared transfer roles "
                f"{actual_roles!r} do not provide "
                f"{requirement.required_transfer_roles!r}"
            ),
        )
    if (
        requirement.required_sectors
        and observable.sector not in requirement.required_sectors
    ):
        return CMBObservableCapability(
            name=requirement.name,
            available=False,
            required_transfer_roles=requirement.required_transfer_roles,
            required_sectors=requirement.required_sectors,
            declared_components=components,
            declared_sector=observable.sector,
            reason=(
                f"declared sector '{observable.sector}' is not compatible "
                f"with {requirement.required_sectors!r}"
            ),
        )
    return CMBObservableCapability(
        name=requirement.name,
        available=True,
        required_transfer_roles=requirement.required_transfer_roles,
        required_sectors=requirement.required_sectors,
        declared_components=components,
        declared_sector=observable.sector,
    )


def audit_cmb_capabilities(
    contract: PerturbationContractData,
) -> CMBCapabilityAudit:
    """Inventory declared graph capabilities without theory-name routing."""

    if not isinstance(contract, PerturbationContractData):
        raise TypeError("CMB capability audits require compiled contract data")
    route = contract.manifest_summary.get("execution_route", {})
    background_references = (
        contract.dependency_graph_summary.background_references_used
    )
    return CMBCapabilityAudit(
        model_name=contract.model_name,
        contract_version=contract.contract_version,
        gauge=contract.gauge,
        sectors=tuple(sorted(str(name) for name in contract.sectors)),
        species=tuple(sorted(str(name) for name in contract.species)),
        hierarchy_families=tuple(
            sorted(str(name) for name in contract.hierarchy_families)
        ),
        collision_operators=tuple(
            sorted(str(name) for name in contract.collision_operators)
        ),
        interactions=tuple(
            sorted(str(name) for name in contract.interactions)
        ),
        closures=tuple(sorted(str(name) for name in contract.closures)),
        initial_condition_families=tuple(
            sorted(str(name) for name in contract.initial_condition_families)
        ),
        projection_typing=tuple(
            sorted(str(name) for name in contract.projection_typing)
        ),
        background_references=tuple(
            sorted(str(name) for name in background_references)
        ),
        numerical_controls=tuple(
            sorted(
                str(name)
                for name in contract.manifest_summary.get("numerics_keys", ())
            )
        ),
        validity_regimes=tuple(
            str(name) for name in contract.validity.regimes
        ),
        declared_observables=tuple(
            sorted(str(name) for name in contract.observables)
        ),
        capability_matrix=tuple(
            _capability_entry(contract, requirement)
            for requirement in CMB_OBSERVABLE_REQUIREMENTS.values()
        ),
        generated_hierarchies=tuple(
            name
            for name, enabled in (
                (
                    "scalar",
                    contract.manifest_summary.get(
                        "generated_scalar_hierarchy", False
                    ),
                ),
                (
                    "vector",
                    contract.manifest_summary.get(
                        "generated_vector_hierarchy", False
                    ),
                ),
                (
                    "tensor",
                    contract.manifest_summary.get(
                        "generated_tensor_hierarchy", False
                    ),
                ),
            )
            if enabled
        ),
        execution_solver_id=str(route.get("solver_id", "")),
        execution_runtime_module=str(route.get("runtime_module", "")),
    )


def build_cmb_capability_matrix(
    contracts: Mapping[str, PerturbationContractData],
) -> FrozenMapping:
    """Build a deterministic model-by-capability matrix from compiled data."""

    audits: dict[str, CMBCapabilityAudit] = {}
    for label, contract in sorted(
        contracts.items(), key=lambda item: str(item[0])
    ):
        audit = audit_cmb_capabilities(contract)
        key = audit.model_name or str(label)
        if key in audits:
            raise ValueError(f"Duplicate CMB capability model '{key}'")
        audits[key] = audit
    return FrozenMapping(audits)


def require_cmb_capability(
    audit: CMBCapabilityAudit,
    observable_name: str,
) -> CMBObservableCapability:
    """Return a capability row or fail with an explicit early diagnostic."""

    name = str(observable_name).upper()
    requirement = CMB_OBSERVABLE_REQUIREMENTS.get(name)
    if requirement is None:
        supported = ", ".join(CMB_OBSERVABLE_REQUIREMENTS)
        raise ValueError(
            f"Unsupported CMB observable '{observable_name}'; "
            f"supported public observables are: {supported}"
        )
    row = next(
        entry for entry in audit.capability_matrix if entry.name == name
    )
    if not row.available:
        detail = row.reason or "declared capability is incomplete"
        raise ValueError(
            f"Unsupported CMB observable '{name}' for model "
            f"'{audit.model_name}': {detail}"
        )
    return row


__all__ = [
    "CMBContractEvaluator",
    "CMBCapabilityAudit",
    "CMBObservableCapability",
    "CMBObservableRequirement",
    "CMBParameterEvaluator",
    "CMB_OBSERVABLE_REQUIREMENTS",
    "audit_cmb_capabilities",
    "build_cmb_capability_matrix",
    "require_cmb_capability",
    "_SUPPORTED_CMB_CALL_KEYS",
    "_SUPPORTED_CMB_CALL_METHODS",
    "_SUPPORTED_CMB_CONTRACT_KEYS",
    "_SUPPORTED_CMB_GRID_KEYS",
    "_SUPPORTED_CMB_GRID_SPACING",
    "_SUPPORTED_CMB_PARAMETER_KEYS",
    "_SUPPORTED_CMB_PERTURBATION_GAUGES",
    "_SUPPORTED_CMB_PERTURBATION_KEYS",
    "_SUPPORTED_CMB_VALUE_KEYS",
    "_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES",
    "_validate_cmb_contract_definition",
]
