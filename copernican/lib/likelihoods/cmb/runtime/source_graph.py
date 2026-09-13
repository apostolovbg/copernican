"""Compile deterministic source and projection graphs for CCMBS runtime use.

The perturbation contract already contains the physical equations and
observable declarations.  This module turns that immutable metadata into one
explicit source-to-kernel graph used by projection and diagnostics.  Keeping
the graph independent of model names makes cross-model and renamed-contract
comparisons exercise exactly the same routing rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ....cmb_output import canonical_cmb_spectrum_name
from ....cmb_projection_contract import (
    get_declared_projection_kernel_spec,
    resolve_declared_projection_kernel,
    resolve_declared_source_kernel,
    validate_declared_projection_sector,
    validate_declared_projection_source_roles,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping for optional sections."""

    return value if isinstance(value, Mapping) else {}


def _field(entry: Any, name: str, default: Any = None) -> Any:
    """Read one field from a compiled dataclass or raw mapping."""

    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _section(contract: Any, name: str) -> Mapping[str, Any]:
    """Read one graph section from a mapping or compiled contract."""

    if isinstance(contract, Mapping):
        return _mapping(contract.get(name))
    return _mapping(getattr(contract, name, {}))


def _strings(value: Any) -> tuple[str, ...]:
    """Normalize declaration names for stable graph serialization."""

    if value is None or isinstance(value, (str, bytes)):
        return () if value is None else (str(value),)
    try:
        return tuple(sorted(str(item) for item in value))
    except TypeError:
        return (str(value),)


def _canonical(value: Any) -> Any:
    """Convert graph metadata to JSON-safe deterministic values."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_canonical(item) for item in sorted(value, key=str)]
    return value


@dataclass(frozen=True, slots=True)
class DeclaredSourceGraph:
    """Immutable source, transfer-route, and spectrum-edge graph."""

    schema_version: int
    source_nodes: tuple[Mapping[str, Any], ...]
    transfer_routes: tuple[Mapping[str, Any], ...]
    spectrum_edges: tuple[Mapping[str, Any], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe graph including its content digest."""

        return {
            "schema_version": int(self.schema_version),
            "source_nodes": [dict(row) for row in self.source_nodes],
            "transfer_routes": [dict(row) for row in self.transfer_routes],
            "spectrum_edges": [dict(row) for row in self.spectrum_edges],
            "digest": self.digest,
        }


def compile_declared_source_graph(
    perturbation_data: Any,
    *,
    requested_spectra: Sequence[str] | None = None,
) -> DeclaredSourceGraph:
    """Compile all declared sources and validated line-of-sight routes.

    The graph includes every source declaration and every transfer component,
    while marking routes active when they feed the requested power spectra.
    Validation is performed before numerical evolution, so a missing source,
    incompatible sector, or unsupported kernel cannot turn into a fabricated
    zero transfer.  The digest excludes the model name and request-local
    numerical controls.
    """

    sources = _section(perturbation_data, "sources")
    observables = _section(perturbation_data, "observables")
    extensions = _section(perturbation_data, "projection_extensions")
    requested = None
    if requested_spectra is not None:
        requested = {
            canonical_cmb_spectrum_name(str(name))
            for name in requested_spectra
        }
    active_components: set[str] = set()
    transfer_component_names = {
        str(name)
        for name, entry in observables.items()
        if str(_field(entry, "kind", "")) == "transfer_component"
    }
    spectrum_edges: list[Mapping[str, Any]] = []
    for name, entry in sorted(
        observables.items(), key=lambda item: str(item[0])
    ):
        if str(_field(entry, "kind", "")) != "angular_power_spectrum":
            continue
        spectrum_name = canonical_cmb_spectrum_name(str(name))
        primary = str(_field(entry, "primary", "") or "")
        secondary = str(_field(entry, "secondary", "") or "")
        missing_components = sorted(
            name
            for name in (primary, secondary)
            if name not in transfer_component_names
        )
        if missing_components:
            raise ValueError(
                f"Declared spectrum '{spectrum_name}' references unknown "
                "transfer components: " + ", ".join(missing_components)
            )
        active = requested is None or spectrum_name in requested
        if active:
            active_components.update((primary, secondary))
        spectrum_edges.append(
            {
                "name": spectrum_name,
                "primary": primary,
                "secondary": secondary,
                "sector": str(_field(entry, "sector", "") or ""),
                "active": bool(active),
            }
        )

    source_nodes = tuple(
        {
            "name": str(name),
            "role": str(_field(entry, "role", "") or ""),
            "expression": str(_field(entry, "expression", "") or ""),
            "dependencies": _strings(_field(entry, "dependencies", ())),
            "units": (
                None
                if _field(entry, "units", None) is None
                else str(_field(entry, "units"))
            ),
            "domain": (
                None
                if _field(entry, "domain", None) is None
                else str(_field(entry, "domain"))
            ),
        }
        for name, entry in sorted(
            sources.items(), key=lambda item: str(item[0])
        )
    )

    transfer_rows: list[Mapping[str, Any]] = []
    for name, entry in sorted(
        observables.items(), key=lambda item: str(item[0])
    ):
        if str(_field(entry, "kind", "")) != "transfer_component":
            continue
        component_name = str(name)
        projection = str(_field(entry, "projection", "") or "")
        kernel_value = _field(entry, "kernel", None)
        kernel = resolve_declared_projection_kernel(
            projection,
            observable_name=component_name,
            kernel=None if kernel_value is None else str(kernel_value),
            extensions=extensions,
        )
        sector_value = _field(entry, "sector", None)
        sector = None if sector_value is None else str(sector_value)
        validate_declared_projection_sector(
            projection,
            sector,
            observable_name=component_name,
            kernel=kernel,
            extensions=extensions,
        )
        raw_terms = _mapping(_field(entry, "source_terms", {}))
        source_terms = {
            str(role): str(source_name)
            for role, source_name in sorted(
                raw_terms.items(), key=lambda item: str(item[0])
            )
        }
        validate_declared_projection_source_roles(
            projection,
            observable_name=component_name,
            source_roles=set(
                str(_field(sources.get(source_name), "role", role))
                for role, source_name in source_terms.items()
            ),
            extensions=extensions,
        )
        missing_sources = sorted(
            source_name
            for source_name in source_terms.values()
            if source_name not in sources
        )
        if missing_sources:
            raise ValueError(
                f"Declared transfer component '{component_name}' references "
                "unknown source histories: " + ", ".join(missing_sources)
            )
        kernels = {
            role: resolve_declared_source_kernel(
                projection,
                role,
                kernel=kernel,
                extensions=extensions,
            )
            for role in source_terms
        }
        kernel_kinds = {
            role: get_declared_projection_kernel_spec(source_kernel).kind
            for role, source_kernel in kernels.items()
        }
        transfer_rows.append(
            {
                "name": component_name,
                "active": component_name in active_components,
                "sector": sector or "scalar",
                "projection": projection,
                "kernel": kernel,
                "kernel_kind": (
                    None
                    if kernel is None
                    else get_declared_projection_kernel_spec(kernel).kind
                ),
                "output_role": str(_field(entry, "output_role", "") or ""),
                "parity": str(_field(entry, "parity", "") or ""),
                "spin": _field(entry, "spin", None),
                "units": (
                    None
                    if _field(entry, "units", None) is None
                    else str(_field(entry, "units"))
                ),
                "source_terms": source_terms,
                "source_kernels": kernels,
                "source_kernel_kinds": kernel_kinds,
                "required_projection_roles": _strings(
                    _field(entry, "required_projection_roles", ())
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "source_nodes": source_nodes,
        "transfer_routes": tuple(transfer_rows),
        "spectrum_edges": tuple(spectrum_edges),
    }
    canonical = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return DeclaredSourceGraph(
        schema_version=1,
        source_nodes=source_nodes,
        transfer_routes=tuple(transfer_rows),
        spectrum_edges=tuple(spectrum_edges),
        digest=digest,
    )


__all__ = ["DeclaredSourceGraph", "compile_declared_source_graph"]
