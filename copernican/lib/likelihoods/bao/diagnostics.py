"""Independent BAO evidence checks used by the final CMB boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy

from ...model_coder import SoundHorizonComputationError


def _same_value(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    """Compare nested BAO values without depending on object identity."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(
            _same_value(left[key], right[key], rtol=rtol, atol=atol)
            for key in left
        )
    if isinstance(left, (list, tuple, numpy.ndarray)) or isinstance(
        right, (list, tuple, numpy.ndarray)
    ):
        try:
            left_array = numpy.asarray(left)
            right_array = numpy.asarray(right)
        except (TypeError, ValueError):
            return False
        if left_array.shape != right_array.shape:
            return False
        if numpy.issubdtype(left_array.dtype, numpy.number) and (
            numpy.issubdtype(right_array.dtype, numpy.number)
        ):
            return bool(
                numpy.allclose(
                    left_array,
                    right_array,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            )
        return bool(numpy.array_equal(left_array, right_array))
    if isinstance(left, (float, int, numpy.number)) and isinstance(
        right, (float, int, numpy.number)
    ):
        return bool(numpy.isclose(left, right, rtol=rtol, atol=atol))
    return left == right


def assess_bao_cmb_isolation(
    baseline: Mapping[str, Any],
    isolated: Mapping[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Prove fixed-background BAO output is independent of CMB execution.

    ``baseline`` and ``isolated`` are caller-captured BAO outputs.  The
    helper intentionally performs no CMB import or invocation: the caller
    is responsible for making the entrypoint unavailable in the isolated
    capture.  Every supplied value, including covariance metadata and typed
    failure payloads, must be preserved exactly (or within the declared
    numerical tolerance).
    """

    if not isinstance(baseline, Mapping) or not isinstance(isolated, Mapping):
        raise TypeError("BAO isolation evidence must be mappings")
    keys = tuple(sorted(set(baseline) | set(isolated), key=str))
    missing = sorted(
        (
            str(key)
            for key in keys
            if key not in baseline or key not in isolated
        ),
        key=str,
    )
    values_preserved = not missing and all(
        _same_value(baseline[key], isolated[key], rtol=rtol, atol=atol)
        for key in keys
    )
    covariance_keys = tuple(
        key
        for key in keys
        if "cov" in str(key).lower()
        or str(key).lower() in {"covariance", "covariance_mode"}
    )
    failure_keys = tuple(
        key
        for key in keys
        if any(
            token in str(key).lower()
            for token in ("failure", "error_type", "error_category")
        )
    )
    covariance_preserved = not covariance_keys or all(
        _same_value(baseline[key], isolated[key], rtol=rtol, atol=atol)
        for key in covariance_keys
    )
    typed_failures_preserved = not failure_keys or all(
        _same_value(baseline[key], isolated[key], rtol=rtol, atol=atol)
        for key in failure_keys
    )
    return {
        "schema_version": 1,
        "available": bool(keys),
        "converged": bool(
            bool(keys)
            and not missing
            and values_preserved
            and covariance_preserved
            and typed_failures_preserved
        ),
        "keys": tuple(str(key) for key in keys),
        "missing_keys": tuple(missing),
        "values_preserved": values_preserved,
        "covariance_preserved": covariance_preserved,
        "typed_failures_preserved": typed_failures_preserved,
        "rtol": float(rtol),
        "atol": float(atol),
    }


def assess_bao_sound_horizon_epochs(
    model_plugin: Any,
    params: Any,
) -> dict[str, Any]:
    """Report the independent recombination and BAO drag horizons.

    CMB diagnostics use recombination-era quantities, whereas BAO ratios
    require the drag-epoch sound horizon.  Generated model plugins expose
    both callables so this check can prove that the BAO route selects the
    latter without importing or invoking the CMB solver.
    """

    rec_fn = getattr(model_plugin, "get_sound_horizon_rs_rec_Mpc", None)
    if rec_fn is None:
        rec_fn = getattr(model_plugin, "get_sound_horizon_rs_Mpc", None)
    drag_fn = getattr(model_plugin, "get_sound_horizon_rs_drag_Mpc", None)
    z_drag_fn = getattr(model_plugin, "get_bao_drag_redshift", None)
    if not callable(rec_fn) or not callable(drag_fn):
        return {
            "available": False,
            "finite": False,
            "distinct": False,
            "sound_horizon_epoch": "unavailable",
        }
    try:
        rec_value = float(rec_fn(*params))
    except SoundHorizonComputationError as exc:
        return {
            "available": True,
            "finite": False,
            "distinct": False,
            "sound_horizon_epoch": "recombination",
            "failure_type": type(exc).__name__,
            "failure_stage": "recombination",
        }
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return {
            "available": True,
            "finite": False,
            "distinct": False,
            "sound_horizon_epoch": "recombination",
            "failure_type": "invalid_value",
            "failure_stage": "recombination",
        }
    try:
        drag_value = float(drag_fn(*params))
        z_drag = None if z_drag_fn is None else float(z_drag_fn(*params))
    except SoundHorizonComputationError as exc:
        return {
            "available": True,
            "finite": False,
            "distinct": False,
            "sound_horizon_epoch": "drag",
            "failure_type": type(exc).__name__,
            "failure_stage": "drag",
        }
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return {
            "available": True,
            "finite": False,
            "distinct": False,
            "sound_horizon_epoch": "drag",
            "failure_type": "invalid_value",
            "failure_stage": "drag",
        }
    finite = bool(
        numpy.isfinite(rec_value)
        and numpy.isfinite(drag_value)
        and rec_value > 0.0
        and drag_value > 0.0
        and (z_drag is None or (numpy.isfinite(z_drag) and z_drag > 0.0))
    )
    if not finite:
        if not (numpy.isfinite(rec_value) and rec_value > 0.0):
            failure_stage = "recombination"
        else:
            failure_stage = "drag"
        return {
            "available": True,
            "finite": False,
            "distinct": False,
            "sound_horizon_epoch": failure_stage,
            "recombination_rs_Mpc": rec_value,
            "drag_rs_Mpc": drag_value,
            "z_drag": z_drag,
            "failure_type": "invalid_value",
            "failure_stage": failure_stage,
        }
    return {
        "available": True,
        "finite": finite,
        "distinct": bool(not numpy.isclose(rec_value, drag_value)),
        "recombination_rs_Mpc": rec_value,
        "drag_rs_Mpc": drag_value,
        "z_drag": z_drag,
        "sound_horizon_epoch": "drag",
    }


__all__ = [
    "assess_bao_cmb_isolation",
    "assess_bao_sound_horizon_epochs",
]
