"""Shared declared-projection rules for the native CMB graph engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeclaredProjectionSpec:
    """Describe the source-role contract for one declared projection."""

    name: str
    any_of_roles: tuple[str, ...]


_DECLARED_PROJECTION_SPECS = {
    "cmb_lensing_potential_scalar": DeclaredProjectionSpec(
        name="cmb_lensing_potential_scalar",
        any_of_roles=("potential", "signal"),
    ),
    "cmb_polarization_e_scalar": DeclaredProjectionSpec(
        name="cmb_polarization_e_scalar",
        any_of_roles=("polarization",),
    ),
    "cmb_temperature_scalar": DeclaredProjectionSpec(
        name="cmb_temperature_scalar",
        any_of_roles=("additive", "doppler", "isw", "monopole"),
    ),
    "scalar_e_mode": DeclaredProjectionSpec(
        name="scalar_e_mode",
        any_of_roles=("signal",),
    ),
    "scalar_jl": DeclaredProjectionSpec(
        name="scalar_jl",
        any_of_roles=("signal",),
    ),
    "scalar_jl_derivative": DeclaredProjectionSpec(
        name="scalar_jl_derivative",
        any_of_roles=("signal",),
    ),
    "scalar_potential": DeclaredProjectionSpec(
        name="scalar_potential",
        any_of_roles=("potential", "signal"),
    ),
    "spin2_b_mode": DeclaredProjectionSpec(
        name="spin2_b_mode",
        any_of_roles=("polarization", "polarization_b", "signal"),
    ),
    "spin2_e_mode": DeclaredProjectionSpec(
        name="spin2_e_mode",
        any_of_roles=("polarization", "signal"),
    ),
}

SUPPORTED_DECLARED_TRANSFER_PROJECTIONS = frozenset(_DECLARED_PROJECTION_SPECS)


def validate_declared_projection_source_roles(
    projection: str,
    *,
    observable_name: str,
    source_roles: set[str],
) -> None:
    """Raise ``ValueError`` when a declared projection contract is invalid."""

    if projection not in _DECLARED_PROJECTION_SPECS:
        raise ValueError(
            f"Perturbation observable '{observable_name}' uses unsupported "
            f"projection '{projection}'"
        )
    spec = _DECLARED_PROJECTION_SPECS[projection]
    if any(role in source_roles for role in spec.any_of_roles):
        return
    expected = ", ".join(spec.any_of_roles)
    raise ValueError(
        f"Perturbation observable '{observable_name}' projection "
        f"'{projection}' requires one of the source-term roles: {expected}"
    )


__all__ = [
    "SUPPORTED_DECLARED_TRANSFER_PROJECTIONS",
    "validate_declared_projection_source_roles",
]
