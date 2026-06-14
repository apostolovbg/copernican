"""Shared declared-projection rules for the native CMB graph engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DeclaredProjectionSpec:
    """Describe the source-role contract for one declared projection."""

    name: str
    required_roles: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    requires_odd_parity_source: bool = False


_DECLARED_PROJECTION_SPECS = {
    "cmb_lensing_potential_scalar": DeclaredProjectionSpec(
        name="cmb_lensing_potential_scalar",
        required_roles=("potential",),
        allowed_roles=("potential",),
    ),
    "cmb_polarization_e_scalar": DeclaredProjectionSpec(
        name="cmb_polarization_e_scalar",
        required_roles=("polarization",),
        allowed_roles=("polarization",),
    ),
    "cmb_temperature_scalar": DeclaredProjectionSpec(
        name="cmb_temperature_scalar",
        required_roles=(),
        allowed_roles=("additive", "doppler", "isw", "monopole"),
    ),
    "scalar_e_mode": DeclaredProjectionSpec(
        name="scalar_e_mode",
        required_roles=("signal",),
        allowed_roles=("signal",),
    ),
    "scalar_jl": DeclaredProjectionSpec(
        name="scalar_jl",
        required_roles=("signal",),
        allowed_roles=("signal",),
    ),
    "scalar_jl_derivative": DeclaredProjectionSpec(
        name="scalar_jl_derivative",
        required_roles=("signal",),
        allowed_roles=("signal",),
    ),
    "scalar_potential": DeclaredProjectionSpec(
        name="scalar_potential",
        required_roles=("potential",),
        allowed_roles=("potential",),
    ),
    "spin2_b_mode": DeclaredProjectionSpec(
        name="spin2_b_mode",
        required_roles=("polarization_b",),
        allowed_roles=("polarization_b",),
        requires_odd_parity_source=True,
    ),
    "spin2_e_mode": DeclaredProjectionSpec(
        name="spin2_e_mode",
        required_roles=(),
        allowed_roles=("polarization", "signal"),
    ),
}

SUPPORTED_DECLARED_TRANSFER_PROJECTIONS = frozenset(_DECLARED_PROJECTION_SPECS)


def get_declared_projection_spec(projection: str) -> DeclaredProjectionSpec:
    """Return the immutable source-role contract for ``projection``."""

    try:
        return _DECLARED_PROJECTION_SPECS[projection]
    except KeyError as exc:
        raise ValueError(
            f"Declared observable requests unsupported projection "
            f"'{projection}'"
        ) from exc


def validate_declared_projection_source_roles(
    projection: str,
    *,
    observable_name: str,
    source_roles: set[str] | Mapping[str, str | None],
) -> None:
    """Raise ``ValueError`` when a declared projection contract is invalid."""

    if isinstance(source_roles, Mapping):
        role_names = set(source_roles)
    else:
        role_names = set(source_roles)
    try:
        spec = get_declared_projection_spec(projection)
    except ValueError as exc:
        raise ValueError(
            f"Perturbation observable '{observable_name}' uses unsupported "
            f"projection '{projection}'"
        ) from exc
    missing_roles = [
        role for role in spec.required_roles if role not in role_names
    ]
    if missing_roles:
        expected = ", ".join(spec.required_roles)
        raise ValueError(
            f"Perturbation observable '{observable_name}' projection "
            f"'{projection}' requires the source-term roles: {expected}"
        )
    unexpected_roles = sorted(role_names - set(spec.allowed_roles))
    if unexpected_roles:
        unexpected = ", ".join(unexpected_roles)
        allowed = ", ".join(spec.allowed_roles)
        raise ValueError(
            f"Perturbation observable '{observable_name}' projection "
            f"'{projection}' does not accept source-term roles: {unexpected}; "
            f"allowed roles: {allowed}"
        )
    if role_names:
        return
    allowed = ", ".join(spec.allowed_roles)
    raise ValueError(
        f"Perturbation observable '{observable_name}' projection "
        f"'{projection}' requires at least one source-term role from: "
        f"{allowed}"
    )


__all__ = [
    "SUPPORTED_DECLARED_TRANSFER_PROJECTIONS",
    "DeclaredProjectionSpec",
    "get_declared_projection_spec",
    "validate_declared_projection_source_roles",
]
