"""Evaluate the actual declared background at scheduled RK coordinates.

Vectorization changes neither coordinates nor interpolants. It avoids
rebuilding the complete kinetic background separately for every stage.
The scalar evolution owner remains an independent comparison authority.
"""

import numpy

from .background import _C_LIGHT_KM_S, _resolve_declared_background_context
from .evolution import _compute_tight_coupling_drag


def sample_declared_stage_contexts(
    *,
    background,
    grids,
    declared_histories,
    contract,
    physical,
    numerics,
    indices,
    blends,
    wave,
):
    """Return exact generated-hierarchy stage values as broadcast vectors."""

    indices = numpy.asarray(indices, dtype=int)
    blends = numpy.asarray(blends, dtype=float)
    if (
        indices.ndim != 1
        or indices.shape != blends.shape
        or numpy.any(indices < 0)
        or numpy.any(indices >= len(grids["eta"]))
        or not numpy.all(numpy.isfinite(blends))
        or numpy.any((blends < 0) | (blends > 1))
    ):
        raise ValueError(
            "Stage coordinates require supported matching vectors"
        )
    following = numpy.minimum(indices + 1, len(grids["eta"]) - 1)

    def blend(history):
        """Preserve the scalar owner's arithmetic order at every coordinate."""

        values = numpy.asarray(history)
        return (1.0 - blends) * values[indices] + blends * values[following]

    eta = blend(grids["eta"])
    context = {
        name: blend(grids[name])
        for name in (
            "a",
            "z",
            "H",
            "Hconf",
            "Hconf_tau",
            "tau",
            "tau_dot",
            "visibility",
            "chi",
            "angular_diameter_distance",
            "sound_speed",
            "baryon_sound_speed_sq",
            "sound_speed_sq",
            "baryon_loading",
            "free_streaming",
        )
    }
    context["eta"] = eta
    context["sound_horizon"] = float(background.sound_horizon_mpc)
    sampled = background.sample(eta)
    sampled_a = numpy.asarray(sampled["a"], dtype=float)
    try:
        first = background.a_of_eta.derivative()(eta)
        second = background.a_of_eta.derivative(2)(eta)
        hconf = first / numpy.maximum(sampled_a, 1.0e-30)
        hubble = hconf * _C_LIGHT_KM_S / numpy.maximum(sampled_a, 1.0e-30)
        hconf_tau = second / numpy.maximum(sampled_a, 1.0e-30) - hconf**2
    except (AttributeError, TypeError, ValueError):
        hubble = numpy.asarray(sampled["H"], dtype=float)
        hconf = sampled_a * hubble / _C_LIGHT_KM_S
        hconf_tau = context["Hconf_tau"]
    for name in (
        "a",
        "z",
        "tau",
        "tau_dot",
        "visibility",
        "chi",
        "angular_diameter_distance",
        "sound_speed",
        "baryon_sound_speed_sq",
    ):
        context[name] = numpy.asarray(sampled[name], dtype=float)
    collision = numpy.maximum(-numpy.asarray(sampled["tau_dot"]), 0.0)
    loading = (
        3.0
        * physical.Omega_b0
        * sampled_a
        / max(4.0 * physical.Omega_gamma0, 1.0e-12)
    )
    context.update(
        H=hubble,
        Hconf=hconf,
        Hconf_tau=hconf_tau,
        collision_rate=collision,
        baryon_loading=loading,
        sound_speed_sq=1.0 / (3.0 * (1.0 + loading)),
        tight_coupling_drag=_compute_tight_coupling_drag(
            collision_rate=collision,
            k_value=float(wave),
            tight_coupling_ratio=float(numerics.tight_coupling_ratio),
        ),
    )
    # Keep the scalar owner's precedence: declared sampled histories precede
    # a fresh evaluation of the declared graph at this stage's a and z.
    context.update(
        {name: blend(values) for name, values in declared_histories.items()}
    )
    declared = _resolve_declared_background_context(
        contract,
        a_values=context["a"],
        z_values=context["z"],
    )
    context.update(
        {
            name: value
            for name, value in declared.items()
            if name not in {"a", "z"}
        }
    )
    if "Hconf" not in declared:
        context["Hconf"] = context["a"] * context["H"] / _C_LIGHT_KM_S
    return eta, {
        name: numpy.broadcast_to(numpy.asarray(value, dtype=float), eta.shape)
        for name, value in context.items()
    }
