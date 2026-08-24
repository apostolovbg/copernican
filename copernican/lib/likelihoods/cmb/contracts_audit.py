"""Contract-level inventory checks for bundled CCMBS model declarations.

The audit verifies the shape and internal consistency of each bundled CMB
contract.  It intentionally does not claim that a declared hierarchy is
physically correct; that evidence belongs to the scientific solver tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

_SUPPORTED_GAUGES = {"conformal_newtonian", "synchronous"}
_REQUIRED_SPECTRA = {"TT", "TE", "EE"}
_REQUIRED_NUMERICS = (
    "ell_min",
    "ell_max",
    "k_min",
    "k_max",
    "k_sample_count",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping for an omitted declaration."""

    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    """Return one finite numeric declaration, or ``None`` when invalid."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class CMBContractAudit:
    """Machine-readable contract inventory result for one bundled model."""

    model_filename: str
    model_name: str
    valid_for_cmb: bool
    contract_version: int | None
    gauge: str | None
    sectors: tuple[str, ...]
    hierarchy_families: tuple[str, ...]
    spectra: tuple[str, ...]
    numerical: Mapping[str, Any] = field(default_factory=dict)
    accuracy_controls: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the declaration passes all structural checks."""

        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a stable serialized representation of the audit record."""

        return {
            "model_filename": self.model_filename,
            "model_name": self.model_name,
            "valid_for_cmb": self.valid_for_cmb,
            "contract_version": self.contract_version,
            "gauge": self.gauge,
            "sectors": list(self.sectors),
            "hierarchy_families": list(self.hierarchy_families),
            "spectra": list(self.spectra),
            "numerical": dict(self.numerical),
            "accuracy_controls": dict(self.accuracy_controls),
            "issues": list(self.issues),
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class CMBSourceGraphAudit:
    """Machine-testable audit of one generated hierarchy/source graph.

    This is deliberately separate from :class:`CMBContractAudit`: the latter
    inventories declarations, while this record checks that the declaration
    exposes the metric derivatives, visibility/polarization sources, closures,
    and compiled expressions required by the CCMBS runtime.
    """

    model_filename: str
    generated_scalar_hierarchy: bool
    metric_state_names: tuple[str, ...]
    metric_derivative_names: tuple[str, ...]
    source_roles: tuple[str, ...]
    closure_targets: tuple[str, ...]
    compiled_source_count: int
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the generated source graph is complete."""

        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a stable serialized graph-audit record."""

        return {
            "model_filename": self.model_filename,
            "generated_scalar_hierarchy": self.generated_scalar_hierarchy,
            "metric_state_names": list(self.metric_state_names),
            "metric_derivative_names": list(self.metric_derivative_names),
            "source_roles": list(self.source_roles),
            "closure_targets": list(self.closure_targets),
            "compiled_source_count": self.compiled_source_count,
            "issues": list(self.issues),
            "valid": self.valid,
        }


def _audit_source_graph_plugin(plugin: Any) -> CMBSourceGraphAudit:
    """Audit generated metric, source, closure, and derivative metadata."""

    perturbation_data = getattr(plugin, "CMB_PERTURBATION_DATA", None)
    manifest = _mapping(getattr(perturbation_data, "manifest_summary", {}))
    generated_scalar = bool(manifest.get("generated_scalar_hierarchy"))
    filename = str(getattr(plugin, "MODEL_FILENAME", "<unknown-model>"))
    variables = {
        str(name)
        for name in _mapping(getattr(perturbation_data, "variables", {}))
    }
    derived = _mapping(getattr(perturbation_data, "derived", {}))
    sources = _mapping(getattr(perturbation_data, "sources", {}))
    closures = _mapping(getattr(perturbation_data, "closures", {}))
    initial_conditions = _mapping(
        getattr(perturbation_data, "initial_conditions", {})
    )
    initial_families = _mapping(
        getattr(perturbation_data, "initial_condition_families", {})
    )
    issues: list[str] = []
    metric_state_names = tuple(
        sorted(name for name in ("Phi", "Psi") if name in variables)
    )
    if generated_scalar and set(metric_state_names) != {"Phi", "Psi"}:
        issues.append("generated scalar hierarchy must expose Phi and Psi")

    required_metric_derivatives = {"Phi_tau", "Psi_tau", "Phi_history_tau"}
    metric_derivative_names = tuple(
        sorted(required_metric_derivatives.intersection(derived))
    )
    if generated_scalar:
        missing = sorted(
            required_metric_derivatives - set(metric_derivative_names)
        )
        if missing:
            issues.append(
                "missing generated metric derivative(s): " + ", ".join(missing)
            )
    phi_tau = derived.get("Phi_tau")
    if generated_scalar and phi_tau is not None:
        if not str(getattr(phi_tau, "expression", "") or "").strip():
            issues.append("Phi_tau must have an explicit graph expression")
        if (
            str(getattr(phi_tau, "kind", ""))
            != "metric_potential_time_derivative"
        ):
            issues.append("Phi_tau must be typed as a metric derivative")
        dependencies = set(getattr(phi_tau, "dependencies", ()) or ())
        for dependency in ("metric_momentum_source_drive", "Hconf", "Psi"):
            if dependency not in dependencies:
                issues.append(
                    "Phi_tau must depend on the declared "
                    f"{dependency} source"
                )
    psi_tau = derived.get("Psi_tau")
    if generated_scalar and psi_tau is not None:
        if (
            str(getattr(psi_tau, "variable", "")) != "Psi"
            or str(getattr(psi_tau, "wrt", "")) != "tau"
            or int(getattr(psi_tau, "order", 0) or 0) != 1
        ):
            issues.append(
                "Psi_tau must target the first tau derivative of Psi"
            )
        if getattr(psi_tau, "expression", None) is not None:
            issues.append("Psi_tau must not use an expression fallback")
        if getattr(psi_tau, "binding", None) != "runtime_history_gradient":
            issues.append(
                "Psi_tau must declare the runtime history-gradient binding"
            )
    phi_history_tau = derived.get("Phi_history_tau")
    if generated_scalar and phi_history_tau is not None:
        if (
            str(getattr(phi_history_tau, "variable", "")) != "Phi"
            or str(getattr(phi_history_tau, "wrt", "")) != "tau"
            or int(getattr(phi_history_tau, "order", 0) or 0) != 1
        ):
            issues.append(
                "Phi_history_tau must target the first tau derivative of Phi"
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
        description = str(getattr(phi_history_tau, "description", "") or "")
        if "runtime binds" not in description.lower():
            issues.append(
                "Phi_history_tau must document its evolved-history binding"
            )

    source_roles = tuple(
        sorted(
            str(getattr(entry, "role", ""))
            for entry in sources.values()
            if str(getattr(entry, "role", ""))
        )
    )
    if generated_scalar:
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
        missing_roles = sorted(required_roles - set(source_roles))
        if missing_roles:
            issues.append(
                "missing generated source role(s): " + ", ".join(missing_roles)
            )
    compiled_source_count = 0
    for name, entry in sources.items():
        expression = str(getattr(entry, "expression", "") or "").strip()
        if not expression:
            issues.append(f"source '{name}' has no expression")
        if getattr(entry, "compiled_expression", None) is None:
            issues.append(f"source '{name}' is not compiler-backed")
        else:
            compiled_source_count += 1

    closure_targets: list[str] = []
    for name, entry in closures.items():
        target = str(getattr(entry, "target", "") or "")
        closure_targets.append(target)
        if target not in variables and target not in derived:
            issues.append(
                f"closure '{name}' targets unknown symbol '{target}'"
            )
        if getattr(entry, "compiled_expression", None) is None:
            issues.append(f"closure '{name}' is not compiler-backed")
    if generated_scalar:
        if "adiabatic_scalar" not in initial_families:
            issues.append("generated hierarchy must declare adiabatic_scalar")
        required_initial_conditions = {
            "Phi_seed",
            "theta_gamma0_seed",
            "theta_gamma1_seed",
            "theta_gamma2_seed",
            "e_gamma2_seed",
            "delta_b_seed",
            "theta_b_seed",
        }
        missing_initial = sorted(
            required_initial_conditions - set(initial_conditions)
        )
        if missing_initial:
            issues.append(
                "missing generated initial condition(s): "
                + ", ".join(missing_initial)
            )
        closure_names = set(closures)
        if "psi_closure" not in closure_names:
            issues.append("generated hierarchy must declare psi_closure")
        if "visibility_polarization_moment_closure" not in closure_names:
            issues.append(
                "generated hierarchy must declare visibility "
                "polarization closure"
            )
    return CMBSourceGraphAudit(
        model_filename=filename,
        generated_scalar_hierarchy=generated_scalar,
        metric_state_names=metric_state_names,
        metric_derivative_names=metric_derivative_names,
        source_roles=source_roles,
        closure_targets=tuple(sorted(set(closure_targets))),
        compiled_source_count=compiled_source_count,
        issues=tuple(sorted(set(issues))),
    )


def audit_bundled_cmb_source_graphs(
    model_directory: str | None = None,
) -> tuple[CMBSourceGraphAudit, ...]:
    """Audit generated source graphs for every bundled CMB model."""

    from .diagnostics import discover_bundled_cmb_plugins

    return tuple(
        _audit_source_graph_plugin(plugin)
        for plugin in discover_bundled_cmb_plugins(model_directory)
    )


def assert_bundled_cmb_source_graphs(
    audits: Iterable[CMBSourceGraphAudit],
) -> None:
    """Raise when a bundled source graph omits required runtime metadata."""

    failures = [
        f"{audit.model_filename}: {'; '.join(audit.issues)}"
        for audit in audits
        if not audit.valid
    ]
    if failures:
        raise ValueError(
            "Bundled CMB source-graph audit failed: " + " | ".join(failures)
        )


def _audit_plugin(plugin: Any) -> CMBContractAudit:
    """Audit one already-compiled bundled model plugin."""

    perturbation_data = getattr(plugin, "CMB_PERTURBATION_DATA", None)
    issues: list[str] = []
    model_filename = str(getattr(plugin, "MODEL_FILENAME", "<unknown-model>"))
    model_name = str(getattr(plugin, "MODEL_NAME", model_filename))
    valid_for_cmb = bool(getattr(plugin, "valid_for_cmb", False))
    if not valid_for_cmb:
        issues.append("valid_for_cmb must be true")
    contract_version_value = getattr(
        perturbation_data,
        "contract_version",
        None,
    )
    try:
        contract_version = int(contract_version_value)
    except (TypeError, ValueError):
        contract_version = None
    if contract_version != 2:
        issues.append("contract_version must be 2")
    gauge_value = getattr(perturbation_data, "gauge", None)
    gauge = None if gauge_value is None else str(gauge_value)
    if gauge not in _SUPPORTED_GAUGES:
        issues.append("gauge must be a supported declared gauge")
    sectors_mapping = _mapping(getattr(perturbation_data, "sectors", {}))
    sectors = tuple(sorted(str(name) for name in sectors_mapping))
    if "scalar" not in sectors:
        issues.append("a scalar perturbation sector is required")
    hierarchy_mapping = _mapping(
        getattr(perturbation_data, "hierarchy_families", {})
    )
    hierarchy_families = tuple(sorted(str(name) for name in hierarchy_mapping))
    if not hierarchy_families:
        issues.append("at least one hierarchy family is required")
    observables = _mapping(getattr(perturbation_data, "observables", {}))
    spectra = tuple(
        sorted(
            str(name).upper()
            for name, entry in observables.items()
            if str(getattr(entry, "kind", "")).lower()
            == "angular_power_spectrum"
        )
    )
    missing_spectra = sorted(_REQUIRED_SPECTRA - set(spectra))
    if missing_spectra:
        issues.append(
            "missing required angular spectra: " + ", ".join(missing_spectra)
        )
    projection_typing = _mapping(
        getattr(perturbation_data, "projection_typing", {})
    )
    if "temperature_line_of_sight" not in projection_typing:
        issues.append("temperature_line_of_sight projection is required")
    numerical = dict(_mapping(getattr(perturbation_data, "numerics", {})))
    runtime_contract = {}
    get_runtime = getattr(plugin, "get_cmb_declared_runtime", None)
    if callable(get_runtime):
        try:
            runtime_contract = _mapping(
                get_runtime(getattr(plugin, "INITIAL_GUESSES", ()))
            )
        except (TypeError, ValueError, RuntimeError) as error:
            issues.append(f"runtime contract cannot be prepared: {error}")
    runtime_numerical = _mapping(runtime_contract.get("numerical", {}))
    for name in _REQUIRED_NUMERICS:
        value = _number(numerical.get(name))
        if value is None:
            issues.append(f"numerics.{name} must be finite")
            continue
        runtime_value = _number(runtime_numerical.get(name))
        if runtime_value is None or runtime_value != value:
            issues.append(
                f"runtime numerical declaration disagrees for {name}"
            )
    ell_min = _number(numerical.get("ell_min"))
    ell_max = _number(numerical.get("ell_max"))
    k_min = _number(numerical.get("k_min"))
    k_max = _number(numerical.get("k_max"))
    if ell_min is not None and ell_max is not None and ell_max < ell_min:
        issues.append("ell_max must not be below ell_min")
    if k_min is not None and k_max is not None and k_max <= k_min:
        issues.append("k_max must be greater than k_min")
    accuracy_controls = dict(
        _mapping(getattr(perturbation_data, "accuracy_controls", {}))
    )
    minimum_k = _number(accuracy_controls.get("minimum_k_sample_count"))
    k_count = _number(numerical.get("k_sample_count"))
    if minimum_k is not None and k_count is not None and k_count < minimum_k:
        issues.append("k_sample_count is below minimum_k_sample_count")
    references = accuracy_controls.get("scalar_reference_ells", ()) or ()
    try:
        reference_values = tuple(float(value) for value in references)
    except (TypeError, ValueError):
        reference_values = ()
        issues.append("scalar_reference_ells must be numeric")
    if (
        reference_values
        and tuple(sorted(set(reference_values))) != reference_values
    ):
        issues.append("scalar_reference_ells must be strictly increasing")
    if reference_values and ell_min is not None and ell_max is not None:
        if min(reference_values) < ell_min or max(reference_values) > ell_max:
            issues.append("scalar_reference_ells must lie within ell bounds")
    if not accuracy_controls.get("runtime_envelope"):
        issues.append("runtime_envelope must be declared")
    return CMBContractAudit(
        model_filename=model_filename,
        model_name=model_name,
        valid_for_cmb=valid_for_cmb,
        contract_version=contract_version,
        gauge=gauge,
        sectors=sectors,
        hierarchy_families=hierarchy_families,
        spectra=spectra,
        numerical=numerical,
        accuracy_controls=accuracy_controls,
        issues=tuple(issues),
    )


def audit_bundled_cmb_contracts(
    model_directory: str | None = None,
) -> tuple[CMBContractAudit, ...]:
    """Audit every bundled model declaration that enables CMB output."""

    from .diagnostics import discover_bundled_cmb_plugins

    return tuple(
        _audit_plugin(plugin)
        for plugin in discover_bundled_cmb_plugins(model_directory)
    )


def assert_bundled_cmb_contracts(
    audits: Iterable[CMBContractAudit],
) -> None:
    """Raise a concise error when any bundled contract fails its audit."""

    failures = [
        f"{audit.model_filename}: {'; '.join(audit.issues)}"
        for audit in audits
        if not audit.valid
    ]
    if failures:
        raise ValueError(
            "Bundled CMB contract audit failed: " + " | ".join(failures)
        )


__all__ = [
    "CMBContractAudit",
    "CMBSourceGraphAudit",
    "assert_bundled_cmb_contracts",
    "assert_bundled_cmb_source_graphs",
    "audit_bundled_cmb_contracts",
    "audit_bundled_cmb_source_graphs",
]
