"""Line-of-sight projection and angular-spectrum quadrature."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from ....cmb_projection_contract import (
    SUPPORTED_DECLARED_TRANSFER_PROJECTIONS,
    get_declared_projection_kernel_spec,
    resolve_declared_source_kernel,
    validate_declared_projection_sector,
)
from .adaptive import phase_aware_k_grid, phase_aware_k_grid_requirements
from .background import (
    _accuracy_control_value,
    _coerce_numeric_scalar,
    _CustomCMBPhysicalParameters,
    _DeclaredProjectionKernelBatch,
)


def _integrate_power_spectrum(
    primordial_grid: numpy.ndarray,
    log_k_values: numpy.ndarray,
    primary: numpy.ndarray,
    secondary: numpy.ndarray,
    *,
    auto_spectrum: bool = False,
    use_cubic_spline: bool = True,
    use_positive_trapezoid: bool = False,
) -> numpy.ndarray:
    """Return one finite power-spectrum quadrature in extended precision.

    Adaptive projection paths may merge logarithmic anchors with a dense
    phase ladder.  The merged nodes are sorted and deduplicated here before
    applying Simpson integration on a uniform log grid or the positive
    composite trapezoid rule on an irregular grid.  Auto spectra reject
    material negative power instead of hiding a quadrature failure.
    """

    primordial_ld = numpy.asarray(primordial_grid, dtype=numpy.longdouble)
    log_k_ld = numpy.asarray(log_k_values, dtype=numpy.longdouble)
    primary_ld = numpy.asarray(primary, dtype=numpy.longdouble)
    secondary_ld = numpy.asarray(secondary, dtype=numpy.longdouble)
    if primary_ld.ndim == 1:
        primary_ld = primary_ld[numpy.newaxis, :]
    if secondary_ld.ndim == 1:
        secondary_ld = secondary_ld[numpy.newaxis, :]
    if log_k_ld.ndim != 1 or primordial_ld.ndim != 1:
        raise ValueError("log-k quadrature nodes must be one-dimensional")
    if (
        log_k_ld.size != primordial_ld.size
        or primary_ld.shape[-1] != log_k_ld.size
        or secondary_ld.shape[-1] != log_k_ld.size
    ):
        raise ValueError(
            "log-k quadrature arrays must have matching node counts"
        )
    if not (
        numpy.all(numpy.isfinite(log_k_ld))
        and numpy.all(numpy.isfinite(primordial_ld))
    ):
        raise ValueError("log-k quadrature inputs must be finite")
    # Adaptive source/transfer paths merge a dense local ladder with the
    # declared scaffold.  Sort that union and collapse nodes which round to
    # the same long-double logarithm before constructing quadrature weights.
    # This preserves every distinct physical node while making the numerical
    # integration contract independent of how the local ladder was assembled.
    order = numpy.argsort(log_k_ld, kind="stable")
    log_k_ld = log_k_ld[order]
    primordial_ld = primordial_ld[order]
    primary_ld = primary_ld[..., order]
    secondary_ld = secondary_ld[..., order]
    if log_k_ld.size > 1:
        keep = numpy.concatenate(
            (
                numpy.asarray((True,), dtype=bool),
                numpy.diff(log_k_ld) > 0.0,
            )
        )
        log_k_ld = log_k_ld[keep]
        primordial_ld = primordial_ld[keep]
        primary_ld = primary_ld[..., keep]
        secondary_ld = secondary_ld[..., keep]
    weighted = primordial_ld[numpy.newaxis, :] * (primary_ld * secondary_ld)
    log_k_steps = numpy.diff(log_k_ld)

    # The phase-aware grid intentionally combines a logarithmic scaffold with
    # linear phase nodes.  A cubic spline antiderivative integrates that
    # irregular grid without treating a refinement as a relocation of the
    # anchors.  Production transfer products are instead allowed to request
    # the positive trapezoid explicitly: interpolation between oscillatory
    # transfer nodes can invent auto-spectrum power and cross-spectrum lobes.
    if use_positive_trapezoid:
        integral = numpy.sum(
            0.5
            * (weighted[:, :-1] + weighted[:, 1:])
            * log_k_steps[numpy.newaxis, :],
            axis=1,
        )
    elif use_cubic_spline and log_k_ld.size >= 3:
        spline_scale = numpy.maximum(
            numpy.max(numpy.abs(weighted), axis=1),
            numpy.longdouble(1.0),
        )
        scaled_weighted = weighted / spline_scale[:, numpy.newaxis]
        spline = CubicSpline(
            numpy.asarray(log_k_ld, dtype=float),
            numpy.asarray(scaled_weighted, dtype=float),
            axis=1,
        )
        antiderivative = spline.antiderivative()
        scaled_integral = numpy.asarray(
            antiderivative(float(log_k_ld[-1]))
            - antiderivative(float(log_k_ld[0])),
            dtype=numpy.longdouble,
        )
        integral = scaled_integral * spline_scale
    else:
        if log_k_ld.size >= 3 and not use_cubic_spline:
            integral = numpy.asarray(
                simpson(weighted, x=log_k_ld, axis=1),
                dtype=numpy.longdouble,
            )
        else:
            integral = numpy.asarray(
                numpy.sum(
                    0.5
                    * (weighted[:, :-1] + weighted[:, 1:])
                    * log_k_steps[numpy.newaxis, :],
                    axis=1,
                ),
                dtype=numpy.longdouble,
            )
    if auto_spectrum and numpy.any(integral < 0.0):
        # Generalized Simpson weights can become negative on a sparse
        # anchor grid.  Re-evaluate only those rows with the positive
        # composite trapezoid rule; replacing every row makes a spectrum
        # depend on whether an unrelated multipole happens to be negative.
        positive_integral = numpy.sum(
            0.5
            * (weighted[:, :-1] + weighted[:, 1:])
            * log_k_steps[numpy.newaxis, :],
            axis=1,
        )
        negative_rows = integral < 0.0
        integral = numpy.where(negative_rows, positive_integral, integral)
    if auto_spectrum:
        # Auto spectra are positive by construction.  Clamp only tiny
        # negative roundoff after the stable positive quadrature; a material
        # negative value is an invariant failure, not a numerical fallback.
        scale = numpy.maximum(numpy.max(numpy.abs(weighted), axis=1), 1.0)
        roundoff = numpy.finfo(float).eps * scale
        if numpy.any(integral < -roundoff):
            raise ValueError(
                "Auto-spectrum quadrature produced a negative power"
            )
        integral = numpy.maximum(integral, 0.0)
    integrated = 4.0 * numpy.longdouble(math.pi) * integral
    # Keep the raw spectrum in extended precision until the public solver
    # applies its final float conversion.
    return numpy.asarray(integrated, dtype=numpy.longdouble)


def _configured_reference_ells(
    perturbation_data: Any,
    *,
    maximum_ell: int | None = None,
) -> tuple[int, ...]:
    """Return all declared reference multipoles for the declared run."""

    controls = getattr(perturbation_data, "accuracy_controls", {}) or {}
    anchor_ells: list[int] = []
    for control_name in (
        "scalar_reference_ells",
        "vector_reference_ells",
        "tensor_reference_ells",
    ):
        raw_values = controls.get(control_name)
        if not isinstance(raw_values, (tuple, list)):
            continue
        for index, raw_value in enumerate(raw_values):
            ell_value = int(
                _coerce_numeric_scalar(
                    raw_value,
                    name=(
                        "cmb.perturbations.accuracy_controls."
                        f"{control_name}[{index}]"
                    ),
                )
            )
            if maximum_ell is None or ell_value <= int(maximum_ell):
                anchor_ells.append(ell_value)
    return tuple(sorted(set(anchor_ells)))


def _projection_anchor_ells(
    ell_arr: numpy.ndarray,
    *,
    perturbation_data: Any,
    node_budget: int,
) -> tuple[int, ...]:
    """Return ell anchors that steer the declared projection k-grid."""

    ell_values = numpy.asarray(ell_arr, dtype=int)
    ell_min = int(ell_values.min())
    ell_max = int(ell_values.max())
    required_ells = {
        ell_min,
        ell_max,
        *_configured_reference_ells(
            perturbation_data,
            maximum_ell=ell_max,
        ),
    }
    if node_budget <= len(required_ells):
        return tuple(sorted(required_ells))
    sample_count = min(int(node_budget), int(ell_values.size))
    sampled_indices = numpy.linspace(
        0,
        ell_values.size - 1,
        num=sample_count,
        dtype=int,
    )
    sampled_ells = {int(ell_values[index]) for index in sampled_indices}
    optional_ells = sorted(sampled_ells - required_ells)
    optional_budget = max(0, int(node_budget) - len(required_ells))
    if len(optional_ells) > optional_budget:
        selected_optional_indices = numpy.linspace(
            0,
            len(optional_ells) - 1,
            num=optional_budget,
            dtype=int,
        )
        optional_ells = [
            optional_ells[index]
            for index in sorted(
                set(int(index) for index in selected_optional_indices)
            )
        ]
    return tuple(sorted(required_ells | set(optional_ells)))


def _build_projection_k_grid(
    *,
    ell_arr: numpy.ndarray,
    background: Any,
    numerics: Any,
    perturbation_data: Any,
    allow_final_production_floor: bool = True,
    diagnostic_matrix_fast_path: bool = False,
    surface_ell_max_override: int | None = None,
    retain_declared_surface: bool = False,
    refinement_anchors: Sequence[float] | None = None,
) -> numpy.ndarray:
    """Return a projection k-grid that satisfies declared numerical bounds.

    Diagnostic requests own their k bounds.  Final generated production
    requests retain the declared ell ceiling so sparse observations share
    one physical transfer surface.
    """

    ell_values = numpy.asarray(ell_arr, dtype=int)
    diagnostic_minimum = 1 if diagnostic_matrix_fast_path else 8
    sample_count = max(diagnostic_minimum, int(numerics.k_sample_count))
    declared_ell_max = int(getattr(numerics, "ell_max", int(ell_values.max())))
    declared_k_min = float(numerics.k_min)
    declared_k_max = float(numerics.k_max)
    accuracy_controls = (
        getattr(
            perturbation_data,
            "accuracy_controls",
            {},
        )
        or {}
    )
    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    generated_final_hierarchy = bool(
        manifest_summary.get("generated_scalar_hierarchy")
        and accuracy_controls.get("accuracy_tier") == "final"
    )
    if not numpy.isfinite(declared_k_min) or not numpy.isfinite(
        declared_k_max
    ):
        raise ValueError("Declared numerical k limits must be finite")
    if declared_k_min <= 0.0 or declared_k_max < declared_k_min:
        raise ValueError(
            "Declared numerical k limits must satisfy " "0 < k_min <= k_max"
        )
    eta0_floor = max(float(background.eta0), 1.0e-6)
    eta_rec_distance = max(
        float(background.eta0) - float(background.eta_rec),
        1.0,
    )
    if surface_ell_max_override is not None:
        requested_surface_ell_max = int(surface_ell_max_override)
        if requested_surface_ell_max < int(ell_values.max()):
            raise ValueError(
                "Projection surface ell ceiling cannot exclude a requested "
                "multipole"
            )
        surface_ell_max = min(declared_ell_max, requested_surface_ell_max)
    else:
        declared_required_k_max = 1.5 * (
            (float(declared_ell_max) + 16.0) / eta_rec_distance
        )
        if (
            declared_required_k_max <= declared_k_max
            and retain_declared_surface
        ):
            # Production requests retain the model-declared physical surface
            # even when the caller selects sparse multipoles from it.  The
            # high-k tail contributes to low-ell power through the same
            # transfer graph and is required for a stable acoustic shape.
            surface_ell_max = declared_ell_max
        else:
            # A low-ell diagnostic request can still be evaluated when a
            # model declares an ell ceiling whose k ceiling is too small to
            # support the full range.
            surface_ell_max = int(ell_values.max())
    configured_reference_ells = _configured_reference_ells(
        perturbation_data,
        maximum_ell=max(int(ell_values.max()), surface_ell_max),
    )
    # The declared numerical interval, rather than the selected dataset
    # rows, owns the projection surface.  A likelihood commonly requests a
    # sparse band beginning at ell=30; letting that band raise k_min would
    # remove legitimate low-k power and change the same multipoles when a
    # caller later requests the full spectrum.
    grid_ell_min = min(
        (
            int(numerics.ell_min),
            *configured_reference_ells,
        )
    )
    # The declared numerical ell ceiling defines the physical projection
    # surface.  Deriving this bound from each request makes the quadrature
    # nodes depend on which other multipoles happen to be requested, so the
    # same low-ell spectrum changes when a caller adds high-ell observations.
    # Keep the surface fixed; sparse likelihood rows only select values from
    # this shared quadrature rather than changing its physical boundaries.
    grid_ell_max = max(
        (
            int(ell_values.max()),
            surface_ell_max,
            *configured_reference_ells,
        )
    )
    k_min = max(
        declared_k_min,
        0.2 * max(float(grid_ell_min), 2.0) / eta0_floor,
    )
    required_k_max = 1.5 * ((float(grid_ell_max) + 16.0) / eta_rec_distance)
    refinement_factor = max(
        1,
        int(getattr(numerics, "k_grid_refinement_factor", 1)),
    )
    if refinement_factor > 1 and not generated_final_hierarchy:
        # Explicit graphs do not use the generated final-grid floor below.
        # Promote the requested node count before applying the minimum so a
        # doubled production request is a real physical refinement even when
        # both declared counts are below the eight-node safety floor.
        sample_count = max(
            sample_count * refinement_factor,
            int(numerics.k_sample_count) * refinement_factor,
        )
    if manifest_summary.get("generated_tensor_hierarchy"):
        # Tensor spin-2 kernels retain an oscillatory high-k tail beyond the
        # scalar projection envelope.  Keep that tail in the fixed node
        # budget so the absolute tensor surfaces converge at the reference
        # multipoles instead of biasing EE and BB low.
        # Round the tail outward by one representable float.  The requested
        # bound and the test-side physical estimate use the same expression,
        # but independent evaluation order can otherwise leave the generated
        # endpoint one ulp below the declared spin-2 requirement.
        tensor_tail_floor = numpy.nextafter(
            5.0 * required_k_max,
            numpy.inf,
        )
        k_floor = max(12.0 * k_min, tensor_tail_floor)
    else:
        # Keep scalar quadrature nodes on the requested projection surface.
        # A fixed 0.08/Mpc floor spends the declared node budget on modes
        # that cannot project to the requested ell range and leaves the
        # visibility-scale Bessel oscillations under-resolved.
        k_floor = max(12.0 * k_min, required_k_max)
    if k_min > declared_k_max or required_k_max > declared_k_max:
        raise ValueError(
            "Requested projection k-grid exceeds declared numerical limits: "
            f"requested=[{k_min}, {required_k_max}], "
            f"declared=[{declared_k_min}, {declared_k_max}]"
        )
    k_max = max(required_k_max, min(declared_k_max, k_floor))
    if not numpy.isfinite(k_min) or not numpy.isfinite(k_max):
        raise ValueError("Declared projection k-grid requires finite bounds")
    if k_max <= k_min:
        return numpy.asarray((k_min,), dtype=float)

    if generated_final_hierarchy and allow_final_production_floor:
        # A 64-node ladder is adequate for contract smoke tests but cannot
        # resolve the rapidly oscillating spherical-Bessel projection at the
        # public ell ceiling.  Keep the declared value as the lower bound and
        # promote generated final spectra to a deterministic production grid.
        # Production refinement carries an explicit factor so the doubled
        # request actually doubles the physical grid rather than being hidden
        # by this floor.
        # Scale the declared ladder before applying the floor.  Otherwise a
        # 64-node base request and a 96-node refinement both collapse to the
        # same production grid, so the convergence comparison does not
        # actually measure a refinement.  The 4x floor keeps the base and
        # doubled ladders distinct without making every public request pay
        # for an unnecessary 2048-mode refinement.
        sample_count = max(
            sample_count * refinement_factor * 4,
            512 * refinement_factor,
        )
    production_probe = accuracy_controls.get(
        "production_scalar_convergence",
        {},
    )
    if (
        isinstance(production_probe, Mapping)
        and bool(production_probe.get("enabled", False))
        and not generated_final_hierarchy
    ):
        # Explicit-graph compatibility probes retain their declared counts in
        # the evidence record, but the actual quadrature must have enough
        # phase samples for a meaningful doubled-grid comparison.  This is
        # an engine floor, not a model-authored solver control.
        sample_count = max(sample_count * refinement_factor, 128)
    has_lensing_surface = any(
        str(getattr(entry, "projection", ""))
        in {
            "line_of_sight_lensing_potential",
            "lensing_potential",
        }
        for entry in getattr(perturbation_data, "observables", {}).values()
    )
    if (
        has_lensing_surface
        and not generated_final_hierarchy
        and not diagnostic_matrix_fast_path
        and str(accuracy_controls.get("accuracy_tier", "")) == "final"
    ):
        # Lensing and temperature-potential cross surfaces retain rapidly
        # varying k structure even on a low-ell request. Resolve that phase
        # with a proportional engine-owned floor while preserving the
        # declared count as the lower bound and in runtime evidence.
        sample_count = max(128, 2 * sample_count)
    phase_setting = accuracy_controls.get("phase_aware_k_quadrature")
    phase_aware_k_enabled = not diagnostic_matrix_fast_path and (
        bool(phase_setting)
        if phase_setting is not None
        else generated_final_hierarchy
    )
    require_phase_resolution = bool(
        accuracy_controls.get("require_phase_resolution", False)
    )
    if phase_aware_k_enabled and require_phase_resolution:
        # A final request must resolve the physical radial/acoustic phase
        # before mode evolution.  The old bounded path capped the linear
        # phase ladder at its nominal sample count and merely recorded an
        # under-resolved status; that allowed aliased Bessel oscillations to
        # enter an otherwise successful production spectrum.  Promote the
        # engine-owned budget to the uncapped requirement, while preserving
        # explicit diagnostic budgets and their evidence-only status.
        phase_requirements = phase_aware_k_grid_requirements(
            k_min,
            k_max,
            phase_points_per_cycle=float(
                _accuracy_control_value(
                    accuracy_controls,
                    "phase_points_per_cycle",
                )
                or 8.0
            ),
            eta_distance=eta_rec_distance,
            sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
        )
        # The physical phase floor is a lower bound for the base ladder.  A
        # refined production ladder must remain independently finer even when
        # that floor dominates the declared sample count; otherwise the two
        # convergence products are identical and provide no evidence.
        phase_floor = int(phase_requirements["required_nodes"])
        if refinement_factor > 1:
            phase_floor *= refinement_factor
        sample_count = max(int(sample_count), phase_floor)
    # Keep the physical anchor set independent of the requested node count.
    # Otherwise a 64-node and a 96-node refinement choose different ell
    # anchors before refinement even begins, measuring an anchor relocation
    # rather than convergence of the same quadrature.  The bounded 48-point
    # anchor scaffold is reused by every ladder and the remaining nodes are
    # deterministic gap subdivisions.
    anchor_ell_count = min(48, grid_ell_max - grid_ell_min + 1)
    projection_ell_values = numpy.linspace(
        grid_ell_min,
        grid_ell_max,
        num=max(2, anchor_ell_count),
        dtype=int,
    )
    # Use one stable physical-anchor budget throughout production-sized
    # refinements.  The remaining nodes are deterministic midpoint
    # subdivisions, so the 64-node final ladder is retained at 96 nodes.
    anchor_node_budget = min(48, max(2, sample_count - 2))
    if phase_aware_k_enabled:
        # Preserve a logarithmic low-k scaffold for the largest-scale modes.
        # A full 48-point ell anchor set consumes nearly the whole 64-node
        # budget and leaves only a handful of nodes below the first acoustic
        # projection scale, which aliases the low-ell spectrum.
        anchor_node_budget = min(16, max(4, sample_count // 4))
    anchor_ells = _projection_anchor_ells(
        projection_ell_values,
        perturbation_data=perturbation_data,
        node_budget=anchor_node_budget,
    )
    k_nodes = {
        float(k_min),
        float(k_max),
        *(
            float(
                numpy.clip(
                    (float(ell_value) + 0.5) / eta_rec_distance,
                    k_min,
                    k_max,
                )
            )
            for ell_value in anchor_ells
        ),
    }
    if refinement_anchors is not None:
        for raw_anchor in refinement_anchors:
            try:
                anchor = float(raw_anchor)
            except (TypeError, ValueError):
                continue
            if numpy.isfinite(anchor) and k_min <= anchor <= k_max:
                k_nodes.add(anchor)
    if not phase_aware_k_enabled:
        ordered_nodes = sorted(k_nodes)
        if len(ordered_nodes) > sample_count:
            interior_nodes = ordered_nodes[1:-1]
            interior_budget = max(0, sample_count - 2)
            if len(interior_nodes) > interior_budget:
                selected_indices = numpy.linspace(
                    0,
                    len(interior_nodes) - 1,
                    num=interior_budget,
                    dtype=int,
                )
                interior_nodes = [
                    interior_nodes[index]
                    for index in sorted(
                        set(int(index) for index in selected_indices)
                    )
                ]
            ordered_nodes = [
                ordered_nodes[0],
                *interior_nodes,
                ordered_nodes[-1],
            ]
        while len(ordered_nodes) < sample_count:
            linear_nodes = numpy.asarray(ordered_nodes, dtype=float)
            log_nodes = numpy.log(linear_nodes)
            widest_gap_index = int(numpy.argmax(numpy.diff(log_nodes)))
            midpoint = float(
                numpy.exp(
                    0.5
                    * (
                        log_nodes[widest_gap_index]
                        + log_nodes[widest_gap_index + 1]
                    )
                )
            )
            if (
                not numpy.isfinite(midpoint)
                or midpoint <= ordered_nodes[widest_gap_index]
                or midpoint >= ordered_nodes[widest_gap_index + 1]
            ):
                break
            ordered_nodes.insert(widest_gap_index + 1, midpoint)
        result = numpy.asarray(ordered_nodes, dtype=float)
    else:
        anchor_nodes = tuple(sorted(float(value) for value in k_nodes))
        phase_points_per_cycle = float(
            _accuracy_control_value(
                accuracy_controls,
                "phase_points_per_cycle",
            )
            or 8.0
        )
        result = phase_aware_k_grid(
            k_min,
            k_max,
            minimum_nodes=sample_count,
            maximum_nodes=sample_count,
            phase_points_per_cycle=phase_points_per_cycle,
            eta_distance=eta_rec_distance,
            sound_horizon=max(float(background.sound_horizon_mpc), 1.0),
            anchors=anchor_nodes,
            require_phase_resolution=bool(require_phase_resolution),
        )
    if (
        result.ndim != 1
        or result.size == 0
        or not numpy.all(numpy.isfinite(result))
        or numpy.any(result < declared_k_min)
        or numpy.any(result > declared_k_max)
        or (result.size > 1 and numpy.any(numpy.diff(result) <= 0.0))
    ):
        raise ValueError(
            "Requested projection k-grid does not satisfy declared numerical "
            "limits"
        )
    return result


def _projection_ell_limit_for_mode(
    *,
    ell_values: numpy.ndarray,
    x_values: numpy.ndarray,
) -> int:
    """Return the non-negligible radial-order limit for one Fourier mode.

    Spherical Bessel functions are exponentially suppressed above ``ell``
    larger than their largest radial argument.  Leaving those rows at zero
    avoids evaluating a large high-ell recurrence for low-k modes without
    changing the line-of-sight integral at floating-point precision relevant
    to the requested surface.
    """

    if ell_values.size == 0 or x_values.size == 0:
        return 0
    maximum_ell = int(numpy.max(ell_values))
    maximum_x = float(numpy.max(numpy.abs(x_values)))
    if not numpy.isfinite(maximum_x):
        raise ValueError("Projection radial arguments must be finite")
    radial_limit = int(
        math.ceil(maximum_x + 32.0 + 8.0 * math.sqrt(max(maximum_x, 0.0)))
    )
    return min(maximum_ell, max(int(numpy.min(ell_values)), radial_limit))


def _primordial_power_grid_for_observable(
    *,
    physical_params: _CustomCMBPhysicalParameters,
    perturbation_data: Any,
    observable_entry: Any,
    k_values: numpy.ndarray,
) -> numpy.ndarray:
    """Return the primordial power grid driving ``observable_entry``."""

    sector = str(getattr(observable_entry, "sector", "") or "")
    manifest_summary = getattr(perturbation_data, "manifest_summary", {}) or {}
    if sector == "tensor" and bool(
        manifest_summary.get("generated_tensor_hierarchy")
    ):
        tensor_ratio = getattr(physical_params, "tensor_to_scalar_ratio", None)
        tensor_tilt = getattr(
            physical_params,
            "tensor_spectral_index",
            None,
        )
        amplitude = float(physical_params.primordial_amplitude) * float(
            0.0 if tensor_ratio is None else max(float(tensor_ratio), 0.0)
        )
        # The declared tensor metric seed is h=1. CAMB/CLASS tensor power
        # conventions put the compensating 1/6 in the primordial spectrum.
        amplitude /= 6.0
        exponent = 0.0 if tensor_tilt is None else float(tensor_tilt)
    else:
        amplitude = float(physical_params.primordial_amplitude)
        exponent = float(physical_params.primordial_spectral_index) - 1.0
    return amplitude * numpy.power(k_values / 0.05, exponent)


def _integrate_declared_spectra(
    *,
    physical_params: _CustomCMBPhysicalParameters,
    perturbation_data: Any,
    power_spectrum_observables: Mapping[str, Any],
    transfer_components: Mapping[str, numpy.ndarray],
    k_values: numpy.ndarray,
    log_k_values: numpy.ndarray,
) -> dict[str, numpy.ndarray]:
    """Integrate declared spectra from transfer products and current tilt."""

    spectra_results: dict[str, numpy.ndarray] = {}
    for (
        observable_name,
        observable_entry,
    ) in power_spectrum_observables.items():
        primordial_grid = _primordial_power_grid_for_observable(
            physical_params=physical_params,
            perturbation_data=perturbation_data,
            observable_entry=observable_entry,
            k_values=k_values,
        )
        primary = numpy.asarray(
            transfer_components[str(observable_entry.primary)],
            dtype=numpy.longdouble,
        )
        secondary = numpy.asarray(
            transfer_components[str(observable_entry.secondary)],
            dtype=numpy.longdouble,
        )
        spectra_results[observable_name] = _integrate_power_spectrum(
            primordial_grid=primordial_grid,
            log_k_values=log_k_values,
            primary=primary,
            secondary=secondary,
            auto_spectrum=(
                str(observable_entry.primary)
                == str(observable_entry.secondary)
            ),
            use_cubic_spline=(
                str(getattr(observable_entry, "sector", "")) != "tensor"
            ),
        )
    return spectra_results


def _declared_graph_projection(
    *,
    projection: str,
    kernel: str | None,
    sector: str | None = None,
    kernel_batch: _DeclaredProjectionKernelBatch,
    k_value: float,
    eta_weights: numpy.ndarray,
    chi_grid: numpy.ndarray,
    source_chi: float,
    source_histories: Mapping[str, numpy.ndarray],
) -> numpy.ndarray:
    """Return projected transfer component values for every ell."""

    if not source_histories:
        raise ValueError(
            f"Declared projection '{projection}' has no available source "
            "histories"
        )

    j_l = kernel_batch.j_l
    j_l_derivative = kernel_batch.j_l_derivative
    j_l_second_derivative = kernel_batch.j_l_second_derivative
    e_kernel = kernel_batch.e_kernel
    b_kernel = kernel_batch.b_kernel
    sector_name = "" if sector is None else str(sector)
    validate_declared_projection_sector(
        projection,
        sector_name or None,
        observable_name=projection,
        kernel=kernel,
    )

    if sector_name == "vector":
        temperature_kernel = kernel_batch.vector_temperature_1
        e_projection_kernel = kernel_batch.vector_e
        b_projection_kernel = kernel_batch.vector_b
    elif sector_name == "tensor":
        temperature_kernel = kernel_batch.tensor_temperature
        e_projection_kernel = kernel_batch.tensor_e
        b_projection_kernel = kernel_batch.tensor_b
    else:
        temperature_kernel = j_l
        e_projection_kernel = e_kernel
        b_projection_kernel = b_kernel

    def _apply_kernel(kernel_name: str) -> numpy.ndarray:
        """Return the ell-batched kernel selected by ``kernel_name``."""

        kernel_spec = get_declared_projection_kernel_spec(kernel_name)
        if kernel_spec.kind == "temperature_mixed":
            raise ValueError(
                "Temperature mixed kernels must use the dedicated "
                "temperature projection dispatch."
            )
        if kernel_spec.kind == "spherical_bessel":
            if (
                sector_name == "vector"
                and projection == "line_of_sight_vector_polarization_e"
            ):
                return e_projection_kernel
            if (
                sector_name == "vector"
                and projection == "line_of_sight_vector_polarization_b"
            ):
                return b_projection_kernel
            return temperature_kernel
        if kernel_spec.kind == "spherical_bessel_derivative":
            return j_l_derivative
        if kernel_spec.kind == "spherical_bessel_second_derivative":
            return j_l_second_derivative
        if kernel_spec.kind == "spin2_e":
            return e_projection_kernel
        if kernel_spec.kind == "spin2_b":
            return b_projection_kernel
        if kernel_spec.kind == "lensing_potential":
            geometry = numpy.clip(source_chi - chi_grid, 0.0, None) / (
                max(float(source_chi), 1.0e-12)
                * numpy.maximum(chi_grid, 1.0e-12)
            )
            return -j_l * geometry[numpy.newaxis, :]
        raise ValueError(
            "Declared observable requests unsupported kernel "
            f"'{kernel_name}'"
        )

    def _project_history(
        kernel_values: numpy.ndarray,
        history: numpy.ndarray,
    ) -> numpy.ndarray:
        """Project one source history through one ell-batched kernel."""

        return numpy.asarray(
            kernel_values @ (eta_weights * history),
            dtype=float,
        )

    if projection in SUPPORTED_DECLARED_TRANSFER_PROJECTIONS:
        projected = numpy.zeros(j_l.shape[0], dtype=float)
        for role_name, history in source_histories.items():
            source_kernel = resolve_declared_source_kernel(
                projection,
                role_name,
                kernel=kernel,
            )
            validate_declared_projection_sector(
                projection,
                sector_name or None,
                observable_name=projection,
                kernel=source_kernel,
            )
            projected += _project_history(
                _apply_kernel(source_kernel),
                history,
            )
        return projected
    raise ValueError(
        "Declared observable requests unsupported projection "
        f"'{projection}'"
    )


def _bind_declared_source_histories(
    *,
    component_name: str,
    component_entry: Any,
    source_arrays: Mapping[str, numpy.ndarray],
) -> dict[str, numpy.ndarray]:
    """Resolve a component's declared roles without fabricating sources."""

    source_terms = {
        str(role_name): str(source_name)
        for role_name, source_name in component_entry.source_terms.items()
    }
    missing = sorted(
        source_name
        for source_name in source_terms.values()
        if source_name not in source_arrays
    )
    if missing:
        raise ValueError(
            f"Declared transfer component '{component_name}' source "
            "histories unavailable: " + ", ".join(missing)
        )
    if not source_terms:
        raise ValueError(
            f"Declared transfer component '{component_name}' has no "
            "declared source histories"
        )
    return {
        role_name: numpy.asarray(source_arrays[source_name], dtype=float)
        for role_name, source_name in source_terms.items()
    }


def _slice_projection_kernel_batch(
    kernel_batch: _DeclaredProjectionKernelBatch,
    indices: numpy.ndarray,
) -> _DeclaredProjectionKernelBatch:
    """Return one radial-kernel batch restricted to an eta subset."""

    selected = numpy.asarray(indices, dtype=int)

    def _slice_kernel(values: numpy.ndarray) -> numpy.ndarray:
        """Slice an eta kernel, preserving intentionally empty sectors."""

        array = numpy.asarray(values)
        if array.ndim != 2:
            raise ValueError("Projection kernels must be two-dimensional")
        # Optional vector/tensor sectors are represented by zero-width
        # arrays when the request does not declare them.  NumPy 2.x warns
        # (and will eventually raise) when non-empty indices are applied to
        # such an axis even though the sector is never consumed.  Preserve
        # the empty-sector sentinel instead of indexing it.
        if array.shape[1] == 0:
            return array[:, :0].copy()
        return array[:, selected]

    return _DeclaredProjectionKernelBatch(
        j_l=_slice_kernel(kernel_batch.j_l),
        j_l_derivative=_slice_kernel(kernel_batch.j_l_derivative),
        j_l_second_derivative=_slice_kernel(
            kernel_batch.j_l_second_derivative
        ),
        e_kernel=_slice_kernel(kernel_batch.e_kernel),
        b_kernel=_slice_kernel(kernel_batch.b_kernel),
        vector_temperature_1=_slice_kernel(kernel_batch.vector_temperature_1),
        vector_temperature_2=_slice_kernel(kernel_batch.vector_temperature_2),
        vector_e=_slice_kernel(kernel_batch.vector_e),
        vector_b=_slice_kernel(kernel_batch.vector_b),
        tensor_temperature=_slice_kernel(kernel_batch.tensor_temperature),
        tensor_e=_slice_kernel(kernel_batch.tensor_e),
        tensor_b=_slice_kernel(kernel_batch.tensor_b),
    )


def _trapezoid_weights(grid: numpy.ndarray) -> numpy.ndarray:
    """Return composite trapezoid weights for a strictly ordered grid."""

    coordinates = numpy.asarray(grid, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("A trapezoid grid requires at least two samples")
    steps = numpy.diff(coordinates)
    if not numpy.all(numpy.isfinite(coordinates)) or numpy.any(steps <= 0.0):
        raise ValueError("A trapezoid grid must be finite and increasing")
    weights = numpy.zeros_like(coordinates, dtype=float)
    weights[0] = 0.5 * steps[0]
    weights[-1] = 0.5 * steps[-1]
    if coordinates.size > 2:
        weights[1:-1] = 0.5 * (steps[:-1] + steps[1:])
    return weights


def _simpson_weights(grid: numpy.ndarray) -> numpy.ndarray:
    """Return linear weights for nonuniform composite Simpson quadrature."""

    eta_grid = numpy.asarray(grid, dtype=float)
    step_sizes = numpy.diff(eta_grid)
    if eta_grid.size < 2 or not numpy.all(numpy.isfinite(eta_grid)):
        raise ValueError("eta_los_grid must contain a finite grid")
    if not numpy.all(numpy.isfinite(step_sizes)) or numpy.any(
        step_sizes <= 0.0
    ):
        raise ValueError("eta_los_grid must be strictly increasing")

    weights = numpy.zeros_like(eta_grid, dtype=float)
    simpson_stop = eta_grid.size
    if eta_grid.size % 2 == 0:
        # Apply Simpson to the odd-sized prefix and retain a second-order
        # endpoint rule for the unavoidable final interval.
        simpson_stop -= 1
        weights[-2:] += 0.5 * step_sizes[-1]
    for start in range(0, simpson_stop - 2, 2):
        left_step = step_sizes[start]
        right_step = step_sizes[start + 1]
        ratio_limit = 2.0 * (1.0 + 1.0e-12)
        if (
            right_step > ratio_limit * left_step
            or left_step > ratio_limit * right_step
        ):
            # Generalized Simpson weights become negative when adjacent
            # intervals differ by more than two.  Merged physical grids can
            # contain near-coincident background anchors, so use the
            # positive trapezoid rule for only that unstable interval pair.
            weights[start] += 0.5 * left_step
            weights[start + 1] += 0.5 * (left_step + right_step)
            weights[start + 2] += 0.5 * right_step
            continue
        total_step = left_step + right_step
        weights[start] += (
            total_step * (2.0 * left_step - right_step) / (6.0 * left_step)
        )
        weights[start + 1] += total_step**3 / (6.0 * left_step * right_step)
        weights[start + 2] += (
            total_step * (2.0 * right_step - left_step) / (6.0 * right_step)
        )
    return weights
