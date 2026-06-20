"""Shared declared-projection rules for the native CMB graph engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DeclaredProjectionKernelSpec:
    """Describe one reviewed line-of-sight kernel shape."""

    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class DeclaredProjectionSpec:
    """Describe the declared contract for one transfer projection."""

    name: str
    required_roles: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    default_kernel: str | None = None
    supported_kernels: tuple[str, ...] = ()
    allows_custom_source_roles: bool = False
    required_projection_roles: tuple[str, ...] = ()
    requires_odd_parity_source: bool = False


_DECLARED_PROJECTION_KERNEL_SPECS = {
    "temperature_mixed_window": DeclaredProjectionKernelSpec(
        name="temperature_mixed_window",
        kind="temperature_mixed",
    ),
    "spherical_bessel_window": DeclaredProjectionKernelSpec(
        name="spherical_bessel_window",
        kind="spherical_bessel",
    ),
    "spherical_bessel_derivative_window": DeclaredProjectionKernelSpec(
        name="spherical_bessel_derivative_window",
        kind="spherical_bessel_derivative",
    ),
    "spin2_e_window": DeclaredProjectionKernelSpec(
        name="spin2_e_window",
        kind="spin2_e",
    ),
    "spin2_b_window": DeclaredProjectionKernelSpec(
        name="spin2_b_window",
        kind="spin2_b",
    ),
    "lensing_potential_window": DeclaredProjectionKernelSpec(
        name="lensing_potential_window",
        kind="lensing_potential",
    ),
}
_CUSTOM_LINE_OF_SIGHT_KERNELS = (
    "spherical_bessel_window",
    "spherical_bessel_derivative_window",
    "spin2_e_window",
    "spin2_b_window",
    "lensing_potential_window",
)
_DECLARED_PROJECTION_SPECS = {
    "line_of_sight_lensing_potential": DeclaredProjectionSpec(
        name="line_of_sight_lensing_potential",
        required_roles=("potential",),
        allowed_roles=("potential",),
        default_kernel="lensing_potential_window",
        supported_kernels=("lensing_potential_window",),
    ),
    "line_of_sight_polarization_e": DeclaredProjectionSpec(
        name="line_of_sight_polarization_e",
        required_roles=("polarization",),
        allowed_roles=("polarization",),
        default_kernel="spin2_e_window",
        supported_kernels=("spin2_e_window",),
    ),
    "line_of_sight_temperature": DeclaredProjectionSpec(
        name="line_of_sight_temperature",
        required_roles=(),
        allowed_roles=("additive", "doppler", "isw", "monopole"),
        default_kernel="temperature_mixed_window",
        supported_kernels=("temperature_mixed_window",),
    ),
    "line_of_sight_signal": DeclaredProjectionSpec(
        name="line_of_sight_signal",
        required_roles=("signal",),
        allowed_roles=("signal",),
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "line_of_sight_signal_derivative": DeclaredProjectionSpec(
        name="line_of_sight_signal_derivative",
        required_roles=("signal",),
        allowed_roles=("signal",),
        default_kernel="spherical_bessel_derivative_window",
        supported_kernels=("spherical_bessel_derivative_window",),
    ),
    "line_of_sight_potential": DeclaredProjectionSpec(
        name="line_of_sight_potential",
        required_roles=("potential",),
        allowed_roles=("potential",),
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "spin2_b_mode": DeclaredProjectionSpec(
        name="spin2_b_mode",
        required_roles=("polarization_b",),
        allowed_roles=("polarization_b",),
        default_kernel="spin2_b_window",
        supported_kernels=("spin2_b_window",),
        required_projection_roles=("b_mode",),
        requires_odd_parity_source=True,
    ),
    "spin2_e_mode": DeclaredProjectionSpec(
        name="spin2_e_mode",
        required_roles=(),
        allowed_roles=("polarization", "signal"),
        default_kernel="spin2_e_window",
        supported_kernels=("spin2_e_window",),
    ),
    "custom_line_of_sight": DeclaredProjectionSpec(
        name="custom_line_of_sight",
        required_roles=(),
        allowed_roles=(),
        supported_kernels=_CUSTOM_LINE_OF_SIGHT_KERNELS,
        allows_custom_source_roles=True,
    ),
}

SUPPORTED_DECLARED_TRANSFER_PROJECTIONS = frozenset(_DECLARED_PROJECTION_SPECS)
SUPPORTED_DECLARED_TRANSFER_PROJECTION_KERNELS = frozenset(
    _DECLARED_PROJECTION_KERNEL_SPECS
)


def get_declared_projection_spec(projection: str) -> DeclaredProjectionSpec:
    """Return the immutable source-role contract for ``projection``."""

    try:
        return _DECLARED_PROJECTION_SPECS[projection]
    except KeyError as exc:
        raise ValueError(
            f"Declared observable requests unsupported projection "
            f"'{projection}'"
        ) from exc


def get_declared_projection_kernel_spec(
    kernel: str,
) -> DeclaredProjectionKernelSpec:
    """Return the immutable kernel contract for ``kernel``."""

    try:
        return _DECLARED_PROJECTION_KERNEL_SPECS[kernel]
    except KeyError as exc:
        raise ValueError(
            f"Declared observable requests unsupported kernel '{kernel}'"
        ) from exc


def resolve_declared_projection_kernel(
    projection: str,
    *,
    observable_name: str,
    kernel: str | None,
) -> str | None:
    """Return the validated kernel name for one observable."""

    spec = get_declared_projection_spec(projection)
    if kernel is None:
        if spec.name == "custom_line_of_sight":
            raise ValueError(
                f"Perturbation observable '{observable_name}' projection "
                f"'{projection}' must declare kernel"
            )
        return spec.default_kernel
    kernel_name = str(kernel)
    get_declared_projection_kernel_spec(kernel_name)
    if kernel_name not in spec.supported_kernels:
        raise ValueError(
            f"Perturbation observable '{observable_name}' projection "
            f"'{projection}' does not support kernel '{kernel_name}'"
        )
    return kernel_name


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
    if spec.allows_custom_source_roles:
        if role_names:
            return
        raise ValueError(
            f"Perturbation observable '{observable_name}' projection "
            f"'{projection}' requires at least one declared source-term role"
        )
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
    "SUPPORTED_DECLARED_TRANSFER_PROJECTION_KERNELS",
    "DeclaredProjectionKernelSpec",
    "DeclaredProjectionSpec",
    "get_declared_projection_kernel_spec",
    "get_declared_projection_spec",
    "resolve_declared_projection_kernel",
    "validate_declared_projection_source_roles",
]
