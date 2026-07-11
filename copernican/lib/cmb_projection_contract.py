"""Shared declared-projection rules for the native CMB graph engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DeclaredProjectionKernelSpec:
    """Describe one reviewed line-of-sight kernel shape."""

    name: str
    kind: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DeclaredProjectionSpec:
    """Describe the declared contract for one transfer projection."""

    name: str
    required_roles: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    description: str | None = None
    output_role: str = "signal"
    transfer_units: str | None = None
    default_kernel: str | None = None
    supported_kernels: tuple[str, ...] = ()
    allows_custom_source_roles: bool = False
    required_projection_roles: tuple[str, ...] = ()
    requires_odd_parity_source: bool = False


_DECLARED_PROJECTION_KERNEL_SPECS = {
    "temperature_mixed_window": DeclaredProjectionKernelSpec(
        name="temperature_mixed_window",
        kind="temperature_mixed",
        description="Mixed scalar temperature line-of-sight kernel.",
    ),
    "spherical_bessel_window": DeclaredProjectionKernelSpec(
        name="spherical_bessel_window",
        kind="spherical_bessel",
        description="Ordinary spherical-Bessel line-of-sight kernel.",
    ),
    "spherical_bessel_derivative_window": DeclaredProjectionKernelSpec(
        name="spherical_bessel_derivative_window",
        kind="spherical_bessel_derivative",
        description="Derivative spherical-Bessel line-of-sight kernel.",
    ),
    "spin2_e_window": DeclaredProjectionKernelSpec(
        name="spin2_e_window",
        kind="spin2_e",
        description="Spin-2 even-parity polarization kernel.",
    ),
    "spin2_b_window": DeclaredProjectionKernelSpec(
        name="spin2_b_window",
        kind="spin2_b",
        description="Spin-2 odd-parity polarization kernel.",
    ),
    "lensing_potential_window": DeclaredProjectionKernelSpec(
        name="lensing_potential_window",
        kind="lensing_potential",
        description="Scalar lensing-potential projection kernel.",
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
        description="Scalar lensing-potential transfer component.",
        output_role="potential",
        transfer_units="dimensionless",
        default_kernel="lensing_potential_window",
        supported_kernels=("lensing_potential_window",),
    ),
    "line_of_sight_polarization_e": DeclaredProjectionSpec(
        name="line_of_sight_polarization_e",
        required_roles=("polarization",),
        allowed_roles=("polarization",),
        description="Spin-2 even-parity polarization transfer component.",
        output_role="polarization_e",
        transfer_units="dimensionless",
        default_kernel="spin2_e_window",
        supported_kernels=("spin2_e_window",),
    ),
    "line_of_sight_temperature": DeclaredProjectionSpec(
        name="line_of_sight_temperature",
        required_roles=(),
        allowed_roles=("additive", "doppler", "isw", "monopole"),
        description="Temperature line-of-sight transfer component.",
        output_role="temperature",
        transfer_units="dimensionless",
        default_kernel="temperature_mixed_window",
        supported_kernels=("temperature_mixed_window",),
    ),
    "line_of_sight_vector_temperature": DeclaredProjectionSpec(
        name="line_of_sight_vector_temperature",
        required_roles=("signal",),
        allowed_roles=("signal",),
        description="Vector temperature transfer component.",
        output_role="temperature",
        transfer_units="dimensionless",
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "line_of_sight_vector_polarization_e": DeclaredProjectionSpec(
        name="line_of_sight_vector_polarization_e",
        required_roles=("signal",),
        allowed_roles=("signal",),
        description="Vector E-polarization transfer component.",
        output_role="polarization_e",
        transfer_units="dimensionless",
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "line_of_sight_vector_polarization_b": DeclaredProjectionSpec(
        name="line_of_sight_vector_polarization_b",
        required_roles=("signal",),
        allowed_roles=("signal",),
        description="Vector B-polarization transfer component.",
        output_role="polarization_b",
        transfer_units="dimensionless",
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "line_of_sight_signal": DeclaredProjectionSpec(
        name="line_of_sight_signal",
        required_roles=("signal",),
        allowed_roles=("signal",),
        description="Generic scalar or sector-tagged signal component.",
        output_role="signal",
        transfer_units="dimensionless",
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "line_of_sight_signal_derivative": DeclaredProjectionSpec(
        name="line_of_sight_signal_derivative",
        required_roles=("signal",),
        allowed_roles=("signal",),
        description="Derivative generic signal component.",
        output_role="signal",
        transfer_units="dimensionless",
        default_kernel="spherical_bessel_derivative_window",
        supported_kernels=("spherical_bessel_derivative_window",),
    ),
    "line_of_sight_potential": DeclaredProjectionSpec(
        name="line_of_sight_potential",
        required_roles=("potential",),
        allowed_roles=("potential",),
        description="Scalar potential transfer component.",
        output_role="potential",
        transfer_units="dimensionless",
        default_kernel="spherical_bessel_window",
        supported_kernels=("spherical_bessel_window",),
    ),
    "spin2_b_mode": DeclaredProjectionSpec(
        name="spin2_b_mode",
        required_roles=("polarization_b",),
        allowed_roles=("polarization_b",),
        description="Spin-2 odd-parity polarization transfer component.",
        output_role="polarization_b",
        transfer_units="dimensionless",
        default_kernel="spin2_b_window",
        supported_kernels=("spin2_b_window",),
        required_projection_roles=("b_mode",),
        requires_odd_parity_source=True,
    ),
    "spin2_e_mode": DeclaredProjectionSpec(
        name="spin2_e_mode",
        required_roles=(),
        allowed_roles=("polarization", "signal"),
        description="Spin-2 even-parity generic transfer component.",
        output_role="polarization_e",
        transfer_units="dimensionless",
        default_kernel="spin2_e_window",
        supported_kernels=("spin2_e_window",),
    ),
    "custom_line_of_sight": DeclaredProjectionSpec(
        name="custom_line_of_sight",
        required_roles=(),
        allowed_roles=(),
        description="Reviewed custom line-of-sight transfer component.",
        output_role="signal",
        transfer_units="dimensionless",
        supported_kernels=_CUSTOM_LINE_OF_SIGHT_KERNELS,
        allows_custom_source_roles=True,
    ),
}

SUPPORTED_DECLARED_TRANSFER_PROJECTIONS = frozenset(_DECLARED_PROJECTION_SPECS)
SUPPORTED_DECLARED_TRANSFER_PROJECTION_KERNELS = frozenset(
    _DECLARED_PROJECTION_KERNEL_SPECS
)


def _extension_field(
    extension_entry: Any,
    field_name: str,
) -> Any:
    """Return ``field_name`` from one projection-extension entry."""

    if isinstance(extension_entry, Mapping):
        return extension_entry.get(field_name)
    return getattr(extension_entry, field_name, None)


def _resolve_projection_spec(
    projection: str,
    *,
    extensions: Mapping[str, Any] | None = None,
) -> DeclaredProjectionSpec:
    """Return the built-in or declared extension spec for ``projection``."""

    if extensions and projection in extensions:
        extension_entry = extensions[projection]
        base_projection = _extension_field(
            extension_entry,
            "base_projection",
        )
        if not isinstance(base_projection, str) or not base_projection.strip():
            raise ValueError(
                f"Declared observable requests invalid projection "
                f"extension '{projection}'"
            )
        base_spec = _resolve_projection_spec(base_projection.strip())
        kernel = _extension_field(extension_entry, "kernel")
        if kernel is not None:
            kernel = str(kernel)
        raw_required_roles = _extension_field(
            extension_entry,
            "required_roles",
        )
        raw_allowed_roles = _extension_field(
            extension_entry,
            "allowed_roles",
        )
        raw_projection_roles = _extension_field(
            extension_entry,
            "required_projection_roles",
        )
        raw_odd_parity = _extension_field(
            extension_entry,
            "requires_odd_parity_source",
        )
        required_roles = (
            tuple(str(role) for role in raw_required_roles)
            if raw_required_roles is not None
            else base_spec.required_roles
        )
        allowed_roles = (
            tuple(str(role) for role in raw_allowed_roles)
            if raw_allowed_roles is not None
            else base_spec.allowed_roles
        )
        required_projection_roles = (
            tuple(str(role) for role in raw_projection_roles)
            if raw_projection_roles is not None
            else base_spec.required_projection_roles
        )
        requires_odd_parity_source = (
            bool(raw_odd_parity)
            if raw_odd_parity is not None
            else base_spec.requires_odd_parity_source
        )
        supported_kernels = (
            (kernel,) if kernel is not None else base_spec.supported_kernels
        )
        return DeclaredProjectionSpec(
            name=projection,
            required_roles=required_roles,
            allowed_roles=allowed_roles,
            description=base_spec.description,
            output_role=base_spec.output_role,
            transfer_units=base_spec.transfer_units,
            default_kernel=kernel or base_spec.default_kernel,
            supported_kernels=supported_kernels,
            allows_custom_source_roles=base_spec.allows_custom_source_roles,
            required_projection_roles=required_projection_roles,
            requires_odd_parity_source=requires_odd_parity_source,
        )
    try:
        return _DECLARED_PROJECTION_SPECS[projection]
    except KeyError as exc:
        raise ValueError(
            f"Declared observable requests unsupported projection "
            f"'{projection}'"
        ) from exc


def get_declared_projection_spec(
    projection: str,
    *,
    extensions: Mapping[str, Any] | None = None,
) -> DeclaredProjectionSpec:
    """Return the immutable source-role contract for ``projection``."""

    return _resolve_projection_spec(projection, extensions=extensions)


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
    extensions: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the validated kernel name for one observable."""

    spec = get_declared_projection_spec(
        projection,
        extensions=extensions,
    )
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
    extensions: Mapping[str, Any] | None = None,
) -> None:
    """Raise ``ValueError`` when a declared projection contract is invalid."""

    if isinstance(source_roles, Mapping):
        role_names = set(source_roles)
    else:
        role_names = set(source_roles)
    try:
        spec = get_declared_projection_spec(
            projection,
            extensions=extensions,
        )
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
