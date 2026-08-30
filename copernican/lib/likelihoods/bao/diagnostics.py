"""Independent BAO evidence checks used by the final CMB boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy


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


__all__ = ["assess_bao_cmb_isolation"]
