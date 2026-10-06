"""Line-of-sight grid planning and background-history sampling."""

from __future__ import annotations

from typing import Any, Mapping

import numpy

from .adaptive import phase_aware_eta_grid
from .background import _C_LIGHT_KM_S, _resolve_declared_background_context
from .evolution import _nonuniform_gradient


def build_phase_aware_projection_ladder(
    base_eta: numpy.ndarray,
    *,
    background: Any,
    k_max: float,
    phase_points_per_cycle: float,
    minimum_nodes: int,
    maximum_nodes: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Build a nested coarse/fine LOS phase ladder."""

    coarse_maximum = max(
        int(minimum_nodes),
        int(numpy.floor(0.875 * int(maximum_nodes))),
    )
    if coarse_maximum >= int(maximum_nodes):
        raise ValueError(
            "adaptive_projection requires distinct coarse and fine "
            "node budgets"
        )
    coarse_eta = phase_aware_eta_grid(
        base_eta,
        visibility=numpy.asarray(
            background.visibility_of_eta(base_eta),
            dtype=float,
        ),
        k_max=float(k_max),
        minimum_nodes=int(minimum_nodes),
        maximum_nodes=coarse_maximum,
        phase_points_per_cycle=phase_points_per_cycle,
    )
    fine_minimum = int(maximum_nodes)
    fine_eta = phase_aware_eta_grid(
        coarse_eta,
        visibility=numpy.asarray(
            background.visibility_of_eta(coarse_eta),
            dtype=float,
        ),
        k_max=float(k_max),
        minimum_nodes=fine_minimum,
        maximum_nodes=int(maximum_nodes),
        phase_points_per_cycle=phase_points_per_cycle,
    )
    if fine_eta.size <= coarse_eta.size:
        raise ValueError(
            "adaptive_projection produced an identical effective grid"
        )
    return fine_eta, coarse_eta


def sample_eta_background_grids(
    eta_grid: numpy.ndarray,
    *,
    background: Any,
    physical_params: Any,
    contract_or_params: Mapping[str, Any],
) -> tuple[
    dict[str, numpy.ndarray],
    dict[str, numpy.ndarray],
    dict[str, numpy.ndarray],
]:
    """Return sampled background histories and coordinate rates."""

    eta_background = background.sample(eta_grid)
    a_grid = numpy.asarray(eta_background["a"], dtype=float)
    z_grid = numpy.asarray(eta_background["z"], dtype=float)
    H_grid = numpy.asarray(eta_background["H"], dtype=float)
    tau_grid = numpy.asarray(eta_background["tau"], dtype=float)
    tau_dot_grid = numpy.asarray(
        eta_background["tau_dot"],
        dtype=float,
    )
    visibility_grid = numpy.asarray(
        eta_background["visibility"],
        dtype=float,
    )
    chi_grid = numpy.asarray(
        eta_background["chi"],
        dtype=float,
    )
    angular_diameter_distance_grid = numpy.asarray(
        eta_background["angular_diameter_distance"],
        dtype=float,
    )
    sound_speed_grid = numpy.asarray(
        eta_background["sound_speed"],
        dtype=float,
    )
    baryon_sound_speed_sq_grid = numpy.asarray(
        eta_background["baryon_sound_speed_sq"],
        dtype=float,
    )
    Hconf_grid = a_grid * H_grid / _C_LIGHT_KM_S
    Hconf_tau_grid = _nonuniform_gradient(Hconf_grid, eta_grid)
    baryon_loading_grid = (
        3.0
        * physical_params.Omega_b0
        * a_grid
        / (4.0 * max(physical_params.Omega_gamma0, 1.0e-12))
    )
    collision_rate_grid = numpy.maximum(-tau_dot_grid, 0.0)
    free_streaming_grid = 1.0 / (
        1.0 + collision_rate_grid / max(float(collision_rate_grid.max()), 1.0)
    )
    sound_speed_sq_grid = 1.0 / (3.0 * (1.0 + baryon_loading_grid))
    declared_background = _resolve_declared_background_context(
        contract_or_params,
        a_values=a_grid,
        z_values=z_grid,
    )
    declared_background_histories: dict[str, numpy.ndarray] = {}
    for name, raw_value in declared_background.items():
        if name in {"a", "z"}:
            continue
        history = numpy.asarray(raw_value, dtype=float)
        if history.ndim == 0:
            history = numpy.full_like(
                eta_grid,
                float(history),
                dtype=float,
            )
        if history.shape != eta_grid.shape:
            raise ValueError(
                "Declared background symbol did not match the "
                f"line-of-sight grid: {name}"
            )
        if not numpy.all(numpy.isfinite(history)):
            raise ValueError(
                "Declared background symbol produced non-finite values: "
                f"{name}"
            )
        declared_background_histories[name] = history
    coordinate_histories = {
        "a": a_grid,
        "z": z_grid,
        "eta": eta_grid,
        "H": H_grid,
        "Hconf": Hconf_grid,
        "Hconf_tau": Hconf_tau_grid,
        "tau": tau_grid,
        "tau_dot": tau_dot_grid,
        "visibility": visibility_grid,
        "chi": chi_grid,
        "angular_diameter_distance": angular_diameter_distance_grid,
        "sound_speed": sound_speed_grid,
        "baryon_sound_speed_sq": baryon_sound_speed_sq_grid,
    }
    for name, history in declared_background_histories.items():
        coordinate_histories.setdefault(name, history)
    coordinate_rate_histories = {"eta": numpy.ones_like(eta_grid, dtype=float)}
    for name, history in coordinate_histories.items():
        if name == "eta":
            continue
        coordinate_rate_histories[name] = _nonuniform_gradient(
            history,
            eta_grid,
        )
    coordinate_rate_histories["a"] = numpy.asarray(
        a_grid * Hconf_grid,
        dtype=float,
    )
    coordinate_rate_histories["z"] = numpy.asarray(
        -(1.0 + z_grid) * Hconf_grid,
        dtype=float,
    )
    return (
        {
            "eta": numpy.asarray(eta_grid, dtype=float),
            "a": a_grid,
            "z": z_grid,
            "H": H_grid,
            "Hconf": Hconf_grid,
            "Hconf_tau": Hconf_tau_grid,
            "tau": tau_grid,
            "tau_dot": tau_dot_grid,
            "visibility": visibility_grid,
            "chi": chi_grid,
            "angular_diameter_distance": angular_diameter_distance_grid,
            "sound_speed": sound_speed_grid,
            "sound_speed_sq": sound_speed_sq_grid,
            "baryon_sound_speed_sq": baryon_sound_speed_sq_grid,
            "baryon_loading": baryon_loading_grid,
            "collision_rate": collision_rate_grid,
            "free_streaming": free_streaming_grid,
        },
        declared_background_histories,
        coordinate_rate_histories,
    )
