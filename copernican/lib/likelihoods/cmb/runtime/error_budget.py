"""Propagate numerical uncertainty through the actual observable operators.

These bounds condition on the supplied source/transfer uncertainty. They
do not turn a sampled history comparison into an independent physics proof.
Signed cross spectra and signed spline weights require absolute bounds;
errors from different axes cannot cancel one another to earn acceptance.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class NativeSourceProbe:
    """Actual native sources for an independent numerical-axis comparison."""

    eta: numpy.ndarray
    sources: Mapping[float, Mapping[str, numpy.ndarray]]
    runtime_envelope: Mapping
    eta0: float
    source_chi: float
    chi: numpy.ndarray


@lru_cache(maxsize=32)
def spline_integral_weights(nodes: tuple[float, ...]) -> numpy.ndarray:
    """Return exact not-a-knot cubic integration weights in linear work.

    Integrating each Hermite interval gives trapezoid weights plus a linear
    functional of endpoint slopes. An adjoint tridiagonal solve applies the
    derivative operator without building a quadratic matrix of basis curves.
    """

    x = numpy.asarray(nodes, dtype=float)
    interval_widths = numpy.diff(x)
    if (
        x.ndim != 1
        or x.size < 2
        or not numpy.all(numpy.isfinite(x))
        or (numpy.any(interval_widths <= 0))
    ):
        raise ValueError("Spline quadrature needs ordered finite nodes")
    if x.size <= 3:
        primitive = CubicSpline(x, numpy.eye(x.size), axis=0).antiderivative()
        weights = primitive(x[-1]) - primitive(x[0])
    else:
        count = x.size
        diagonal = numpy.zeros((3, count))
        diagonal[1, 1:-1] = 2 * (interval_widths[:-1] + interval_widths[1:])
        diagonal[0, 2:] = interval_widths[:-1]
        diagonal[2, :-2] = interval_widths[1:]
        diagonal[1, 0], diagonal[0, 1] = (
            interval_widths[1],
            interval_widths[0] + interval_widths[1],
        )
        diagonal[1, -1], diagonal[2, -2] = (
            interval_widths[-2],
            interval_widths[-1] + interval_widths[-2],
        )
        transposed = numpy.zeros_like(diagonal)
        transposed[1] = diagonal[1]
        transposed[0, 1:] = diagonal[2, :-1]
        transposed[2, :-1] = diagonal[0, 1:]
        functional = numpy.zeros(count)
        functional[:-1] += interval_widths**2 / 12
        functional[1:] -= interval_widths**2 / 12
        adjoint = solve_banded((1, 1), transposed, functional)
        weights = numpy.zeros(count)
        weights[:-1] += interval_widths / 2
        weights[1:] += interval_widths / 2
        left = 3 * adjoint[1:-1] * interval_widths[1:] / interval_widths[:-1]
        right = 3 * adjoint[1:-1] * interval_widths[:-1] / interval_widths[1:]
        weights[:-2] -= left
        weights[1:-1] += left - right
        weights[2:] += right
        for index, first, second, width, other, coefficient in (
            (0, 0, 1, interval_widths[0], interval_widths[1], adjoint[0]),
            (
                count - 1,
                count - 2,
                count - 3,
                interval_widths[-1],
                interval_widths[-2],
                adjoint[-1],
            ),
        ):
            span = width + other
            edge = coefficient * (width + 2 * span) * other / span / width
            adjacent = coefficient * width**2 / span / other
            if index == 0:
                weights[0] -= edge
                weights[1] += edge - adjacent
                weights[2] += adjacent
            else:
                weights[-1] += edge
                weights[-2] += adjacent - edge
                weights[-3] -= adjacent
    weights.flags.writeable = False
    return weights


def transfer_spectrum_error(
    log_k: numpy.ndarray,
    primordial: numpy.ndarray,
    primary: numpy.ndarray,
    secondary: numpy.ndarray,
    primary_error: numpy.ndarray,
    secondary_error: numpy.ndarray,
    *,
    total_primary_error: numpy.ndarray | None = None,
    total_secondary_error: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Bound cross/auto spectrum error including the quadratic error term."""

    arrays = [
        numpy.asarray(value, dtype=float)
        for value in (primary, secondary, primary_error, secondary_error)
    ]
    if any(not numpy.all(numpy.isfinite(value)) for value in arrays):
        raise ValueError("Transfer error propagation requires finite arrays")
    first, second, first_error, second_error = arrays
    if numpy.any(first_error < 0) or numpy.any(second_error < 0):
        raise ValueError("Transfer uncertainty must be nonnegative")
    total_first = (
        first_error
        if total_primary_error is None
        else numpy.asarray(total_primary_error)
    )
    total_second = (
        second_error
        if total_secondary_error is None
        else numpy.asarray(total_secondary_error)
    )
    if not all(
        numpy.all(numpy.isfinite(value))
        for value in (total_first, total_second)
    ) or (
        numpy.any(total_first < first_error)
        or numpy.any(total_second < second_error)
    ):
        raise ValueError("Joint transfer uncertainty must include this axis")
    envelope = (
        abs(first) * second_error
        + abs(second) * first_error
        # Allocate each cross-axis quadratic term equally to its two axes.
        + 0.5 * (first_error * total_second + second_error * total_first)
    )
    weights = abs(spline_integral_weights(tuple(float(x) for x in log_k)))
    return (
        4
        * numpy.pi
        * numpy.sum(
            envelope * abs(numpy.asarray(primordial)) * weights, axis=-1
        )
    )


def combine_observable_errors(
    spectra: Mapping[str, numpy.ndarray],
    axis_errors: Mapping[str, Mapping[str, numpy.ndarray]],
    tolerances: Mapping[str, float],
    *,
    absolute_tolerance: float = 1.0e-30,
) -> dict:
    """Require the sum of axis uncertainties to fit each observable budget."""

    if not spectra or not axis_errors:
        raise ValueError(
            "Observable budgets require products and measured axes"
        )
    if not numpy.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
        raise ValueError("The absolute observable allowance must be positive")
    products = {}
    total_errors = {}
    for name, spectrum in spectra.items():
        values = numpy.asarray(spectrum, dtype=float)
        if not values.size or not numpy.all(numpy.isfinite(values)):
            raise ValueError("Observable products must be finite and nonempty")
        scale = max(float(numpy.max(abs(values))), absolute_tolerance)
        contributions = {}
        total = numpy.zeros_like(values)
        for axis, errors in axis_errors.items():
            if name not in errors:
                raise ValueError(f"Missing {axis} error for observable {name}")
            bound = numpy.asarray(errors[name], dtype=float)
            if bound.shape != values.shape or not numpy.all(
                numpy.isfinite(bound) & (bound >= 0)
            ):
                raise ValueError("Observable error bounds must match products")
            total += bound
            contributions[axis] = float(numpy.max(bound)) / scale
        tolerance = float(tolerances[name])
        if not numpy.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Observable relative tolerances must be positive")
        total_errors[name] = total
        peak = float(numpy.max(total))
        products[name] = {
            "relative_error_bound": peak / scale,
            "absolute_error_bound": peak,
            "relative_tolerance": tolerance,
            "axis_relative_bounds": contributions,
            "converged": bool(
                peak <= max(absolute_tolerance, tolerance * scale)
            ),
        }
    for cross, first, second in (
        ("TE", "TT", "EE"),
        ("TP", "TT", "PP"),
        ("EP", "EE", "PP"),
    ):
        if not {cross, first, second}.issubset(spectra):
            continue
        power_first = numpy.asarray(spectra[first])
        power_second = numpy.asarray(spectra[second])
        denominator = numpy.sqrt(abs(power_first * power_second))
        supported = denominator >= max(
            float(numpy.max(denominator)) * 0.01, absolute_tolerance
        )
        lower = numpy.sqrt(
            numpy.maximum(power_first - total_errors[first], 0.0)
            * numpy.maximum(power_second - total_errors[second], 0.0)
        )
        lower = numpy.maximum(lower, numpy.finfo(float).tiny)
        correlation = abs(numpy.asarray(spectra[cross])) / numpy.maximum(
            denominator, numpy.finfo(float).tiny
        )
        propagated = total_errors[cross] / lower + correlation * (
            denominator / lower - 1.0
        )
        normalized = float(numpy.max(propagated[supported], initial=0.0))
        products[cross]["normalized_cross_bound"] = normalized
        products[cross]["converged"] &= normalized <= float(tolerances[cross])
    return {
        "method": "absolute_operator_propagation_and_axis_triangle_bound",
        "products": products,
        "converged": all(row["converged"] for row in products.values()),
    }


def update_source_error_envelope(
    envelope: dict[str, numpy.ndarray],
    fine: Mapping[str, numpy.ndarray],
    comparison: Mapping[str, numpy.ndarray],
    *,
    absolute_envelope: dict[str, numpy.ndarray] | None = None,
) -> None:
    """Retain pointwise, per-source discrepancies on the validation domain.

    Each source is scaled separately; a large monopole must not erase a
    small polarized or late-time source. Zero or unscalable histories
    retain an absolute envelope instead of losing a newly resolved signal.
    Projection uses absolute kernels so phases cannot cancel uncertainty.
    """

    for name, raw in fine.items():
        values = numpy.asarray(raw, dtype=float)
        lower = numpy.asarray(comparison[name], dtype=float)
        if values.shape != lower.shape or not numpy.all(
            numpy.isfinite(values) & numpy.isfinite(lower)
        ):
            raise ValueError("Source uncertainty requires matching histories")
        scale = float(numpy.max(abs(values)))
        difference = abs(values - lower)
        maximum_error = float(numpy.max(difference))
        if scale == 0 or scale < maximum_error / numpy.finfo(float).max:
            if absolute_envelope is None:
                raise ValueError(
                    "An unscalable source requires an absolute envelope"
                )
            absolute_envelope[name] = numpy.maximum(
                absolute_envelope.get(name, 0.0), difference
            )
            envelope.setdefault(name, numpy.zeros_like(values))
            continue
        error = difference / scale
        envelope[name] = numpy.maximum(envelope.get(name, 0.0), error)


def project_source_uncertainty(
    component,
    source_arrays,
    profiles,
    *,
    geometry_reference=None,
    absolute_profiles=None,
    geometry_other_profiles=(),
    **projection_inputs,
) -> numpy.ndarray:
    """Apply absolute LOS operators to a measured per-source envelope."""

    from .spectrum_projection import _declared_graph_projection

    histories = {}
    geometry_histories = {}
    absolute_profiles = absolute_profiles or {}
    for role, name in component.source_terms.items():
        values = numpy.asarray(source_arrays[name], dtype=float)
        if name not in profiles and name not in absolute_profiles:
            raise ValueError(f"Missing source uncertainty: {name}")
        scale = float(numpy.max(abs(values)))
        histories[role] = profiles.get(
            name, 0.0
        ) * scale + absolute_profiles.get(name, 0.0)
        geometry_histories[role] = abs(values) + sum(
            relative.get(name, 0.0) * scale + absolute.get(name, 0.0)
            for relative, absolute in geometry_other_profiles
        )
    if geometry_reference is not None:
        import hashlib

        from .background import _get_cached_declared_projection_kernel_batch

        bound_operators, reference_operators = {}, {}
        bound_histories = {
            role: numpy.asarray(source_arrays[name])
            for role, name in component.source_terms.items()
        }
        _declared_graph_projection(
            projection=str(component.projection),
            kernel=component.kernel,
            sector=component.sector,
            source_histories=bound_histories,
            operator_sink=bound_operators,
            **projection_inputs,
        )
        ratio = float(geometry_reference["eta0_ratio"])
        comparison_chi = numpy.asarray(
            geometry_reference.get(
                "chi_grid",
                numpy.asarray(projection_inputs["chi_grid"]) * ratio,
            )
        )
        radial_distance = numpy.asarray(
            geometry_reference.get("radial_distance", comparison_chi)
        )
        x_values = float(projection_inputs["k_value"]) * radial_distance
        reference_kernels = _get_cached_declared_projection_kernel_batch(
            tuple(geometry_reference["ells"]),
            hashlib.sha256(x_values.tobytes()).hexdigest(),
            x_values=x_values,
            required_sectors=(str(component.sector or "scalar"),),
        )
        _declared_graph_projection(
            projection=str(component.projection),
            kernel=component.kernel,
            sector=component.sector,
            source_histories=bound_histories,
            operator_sink=reference_operators,
            kernel_batch=reference_kernels,
            k_value=projection_inputs["k_value"],
            eta_weights=numpy.asarray(projection_inputs["eta_weights"])
            * ratio,
            chi_grid=comparison_chi,
            source_chi=geometry_reference["source_chi"],
        )
        return sum(
            (
                abs(bound_operators[role])
                + abs(reference_operators[role] - bound_operators[role])
            )
            @ histories[role]
            + abs(reference_operators[role] - bound_operators[role])
            @ geometry_histories[role]
            for role in histories
        )
    return _declared_graph_projection(
        projection=str(component.projection),
        kernel=component.kernel,
        sector=component.sector,
        source_histories=histories,
        absolute_error=True,
        **projection_inputs,
    )
