"""Fail-closed continuation of the pinned SciPy BDF trajectory."""

from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from typing import Any, Callable

import numpy
import scipy
from scipy.integrate import BDF
from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate._ivp.bdf import (
    BdfDenseOutput,
    change_D,
    compute_R,
    solve_bdf_system,
)
from scipy.sparse import csc_matrix, issparse

from .history_store import allocate_native_history


@lru_cache(maxsize=1)
def bdf_implementation_identity():
    """Identify the exact pinned Python algorithms used by a checkpoint."""

    if scipy.__version__ != "1.12.0":
        raise ValueError("BDF checkpoints require the pinned SciPy version")
    functions = (
        BDF.__init__,
        BDF._validate_jac,
        BDF._step_impl,
        BDF._dense_output_impl,
        OdeSolver.step,
        BdfDenseOutput._call_impl,
        change_D,
        compute_R,
        solve_bdf_system,
    )
    digest = hashlib.sha256()
    for function in functions:
        digest.update(inspect.getsource(function).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _encode_bdf_matrix(value):
    """Represent a dense or CSC matrix without executable serialization."""

    if issparse(value):
        matrix = value.tocsc()
        return {
            "format": "csc",
            "shape": list(matrix.shape),
            "data": numpy.asarray(matrix.data).copy(),
            "indices": numpy.asarray(matrix.indices).copy(),
            "indptr": numpy.asarray(matrix.indptr).copy(),
        }
    return {"format": "dense", "values": numpy.asarray(value).copy()}


def _decode_bdf_matrix(payload, dimension):
    """Reject malformed matrix storage before invoking numerical libraries."""

    if not isinstance(payload, dict):
        raise ValueError("BDF checkpoint matrix is invalid")
    if payload.get("format") == "dense":
        values = payload.get("values")
        if (
            not isinstance(values, numpy.ndarray)
            or values.shape != (dimension, dimension)
            or values.dtype.kind != "f"
            or not numpy.all(numpy.isfinite(values))
        ):
            raise ValueError("BDF checkpoint dense matrix is invalid")
        return values.copy()
    if payload.get("format") != "csc":
        raise ValueError("BDF checkpoint matrix format is invalid")
    matrix_values, indices, indptr = (
        payload.get("data"),
        payload.get("indices"),
        payload.get("indptr"),
    )
    if (
        payload.get("shape") != [dimension, dimension]
        or not isinstance(matrix_values, numpy.ndarray)
        or not isinstance(indices, numpy.ndarray)
        or not isinstance(indptr, numpy.ndarray)
        or matrix_values.ndim != 1
        or matrix_values.dtype.kind != "f"
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or indptr.ndim != 1
        or indptr.dtype.kind not in "iu"
        or indptr.size != dimension + 1
        or indices.size != matrix_values.size
        or indptr[0] != 0
        or indptr[-1] != matrix_values.size
        or numpy.any(indptr[1:] < indptr[:-1])
        or numpy.any(indices < 0)
        or numpy.any(indices >= dimension)
        or not numpy.all(numpy.isfinite(matrix_values))
    ):
        raise ValueError("BDF checkpoint sparse matrix is invalid")
    return csc_matrix(
        (matrix_values.copy(), indices.copy(), indptr.copy()),
        shape=(dimension, dimension),
    )


def _bdf_factor_arrays(factor):
    """Expose every numerical LU factor and permutation for exact checking."""

    if isinstance(factor, tuple):
        return {
            "kind": "dense",
            "lu": numpy.asarray(factor[0]).copy(),
            "pivots": numpy.asarray(factor[1]).copy(),
        }
    return {
        "kind": "sparse",
        "lower": _encode_bdf_matrix(factor.L),
        "upper": _encode_bdf_matrix(factor.U),
        "row_permutation": numpy.asarray(factor.perm_r).copy(),
        "column_permutation": numpy.asarray(factor.perm_c).copy(),
    }


def _identical_bdf_product(left, right):
    """Compare typed nested numerical factors without tolerances."""

    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(
                _identical_bdf_product(left[key], right[key]) for key in left
            )
        )
    if isinstance(left, numpy.ndarray):
        return (
            isinstance(right, numpy.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    return type(left) is type(right) and left == right


def _validate_bdf_snapshot(state, eta, dimension):
    """Validate mutable multistep state before restoring a SciPy solver."""

    expected = {
        "t",
        "t_old",
        "y",
        "h_abs",
        "order",
        "differences",
        "jacobian",
        "factor_matrix",
        "factors",
        "n_equal_steps",
        "nfev",
        "njev",
        "nlu",
        "accepted_steps",
        "evaluated",
        "scalar_rhs_calls",
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("BDF checkpoint state fields differ")
    ints = (
        "order",
        "n_equal_steps",
        "nfev",
        "njev",
        "nlu",
        "accepted_steps",
        "evaluated",
        "scalar_rhs_calls",
    )
    if any(
        not isinstance(state[name], int)
        or isinstance(state[name], bool)
        or state[name] < 0
        for name in ints
    ):
        raise ValueError("BDF checkpoint counters are invalid")
    order, epoch, t_old, h_abs = (
        state["order"],
        state["t"],
        state["t_old"],
        state["h_abs"],
    )
    values, differences = state["y"], state["differences"]
    if (
        order not in range(1, 6)
        or state["accepted_steps"] < 1
        or not all(
            isinstance(value, (int, float)) and numpy.isfinite(value)
            for value in (epoch, t_old, h_abs)
        )
        or not float(eta[0]) <= t_old < epoch < float(eta[-1])
        or h_abs <= 0
        or state["evaluated"]
        != int(numpy.searchsorted(eta, epoch, side="right"))
        or not isinstance(values, numpy.ndarray)
        or values.shape != (dimension,)
        or values.dtype != numpy.dtype(float)
        or not numpy.all(numpy.isfinite(values))
        or not isinstance(differences, numpy.ndarray)
        or differences.shape != (order + 2, dimension)
        or differences.dtype != numpy.dtype(float)
        or not numpy.all(numpy.isfinite(differences))
    ):
        raise ValueError("BDF checkpoint numerical state is invalid")
    jacobian = _decode_bdf_matrix(state["jacobian"], dimension)
    matrix = state["factor_matrix"]
    factors = state["factors"]
    if (matrix is None) != (factors is None):
        raise ValueError("BDF checkpoint factor metadata differs")
    if matrix is not None:
        matrix = _decode_bdf_matrix(matrix, dimension)
        if not isinstance(factors, dict):
            raise ValueError("BDF checkpoint factors are invalid")
    return jacobian, matrix


def integrate_bdf_checkpointed(
    rhs: Callable[..., numpy.ndarray],
    jacobian: Callable[..., Any],
    eta: numpy.ndarray,
    initial: numpy.ndarray,
    *,
    rtol: float,
    atol: float,
    checkpoint: Any,
    rhs_calls: Callable[[], int],
    restore_rhs_calls: Callable[[int], None],
    initial_rhs_calls: int,
) -> numpy.ndarray:
    """Continue pinned BDF steps and the same dense-output sampling."""

    bdf_implementation_identity()
    eta = numpy.asarray(eta, dtype=float)
    initial = numpy.asarray(initial, dtype=float)
    if (
        eta.ndim != 1
        or eta.size < 2
        or not numpy.all(numpy.isfinite(eta))
        or not numpy.all(numpy.diff(eta) > 0)
        or initial.ndim != 1
        or not numpy.all(numpy.isfinite(initial))
    ):
        raise ValueError("BDF checkpoint requires a finite increasing grid")
    storage = allocate_native_history((1, eta.size, initial.size))
    history = storage[0].T
    solver = BDF(
        rhs,
        float(eta[0]),
        initial,
        float(eta[-1]),
        jac=jacobian,
        rtol=float(rtol),
        atol=float(atol),
    )
    original_lu = solver.lu
    last_factor_matrix = [None]

    def traced_lu(matrix):
        """Retain the exact matrix behind SciPy's reusable LU factors."""

        last_factor_matrix[0] = matrix.copy()
        return original_lu(matrix)

    solver.lu = traced_lu
    sparse_jacobian = issparse(solver.J)
    state = checkpoint.restore(history)
    accepted_steps = 0
    evaluated = 0
    if state is not None:
        saved_jacobian, factor_matrix = _validate_bdf_snapshot(
            state, eta, initial.size
        )
        if issparse(saved_jacobian) != sparse_jacobian:
            raise ValueError("BDF checkpoint Jacobian backend differs")
        solver.t = float(state["t"])
        solver.t_old = float(state["t_old"])
        solver.y = state["y"].copy()
        solver.h_abs = float(state["h_abs"])
        solver.order = int(state["order"])
        solver.n_equal_steps = int(state["n_equal_steps"])
        solver.D.fill(0.0)
        solver.D[: solver.order + 2] = state["differences"]
        solver.J = saved_jacobian
        solver.nfev = int(state["nfev"])
        solver.njev = int(state["njev"])
        solver.nlu = int(state["nlu"])
        if factor_matrix is None:
            solver.LU = None
        else:
            nlu = solver.nlu
            solver.LU = solver.lu(factor_matrix)
            solver.nlu = nlu
            if not _identical_bdf_product(
                _bdf_factor_arrays(solver.LU), state["factors"]
            ):
                raise ValueError("BDF checkpoint LU factors differ")
        accepted_steps = int(state["accepted_steps"])
        evaluated = int(state["evaluated"])
        restore_rhs_calls(initial_rhs_calls + int(state["scalar_rhs_calls"]))
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise ValueError(
                "Declared CMB continuous evolution failed: " + message
            )
        accepted_steps += 1
        end = int(numpy.searchsorted(eta, solver.t, side="right"))
        if end > evaluated:
            history[:, evaluated:end] = solver.dense_output()(
                eta[evaluated:end]
            )
            evaluated = end
        if solver.status == "running" and checkpoint.due():
            if solver.LU is not None and last_factor_matrix[0] is None:
                raise ValueError("BDF checkpoint is missing its factor matrix")
            snapshot = {
                "t": float(solver.t),
                "t_old": float(solver.t_old),
                "y": solver.y.copy(),
                "h_abs": float(solver.h_abs),
                "order": int(solver.order),
                "differences": solver.D[: solver.order + 2].copy(),
                "jacobian": _encode_bdf_matrix(solver.J),
                "factor_matrix": (
                    None
                    if solver.LU is None
                    else _encode_bdf_matrix(last_factor_matrix[0])
                ),
                "factors": (
                    None
                    if solver.LU is None
                    else _bdf_factor_arrays(solver.LU)
                ),
                "n_equal_steps": int(solver.n_equal_steps),
                "nfev": int(solver.nfev),
                "njev": int(solver.njev),
                "nlu": int(solver.nlu),
                "accepted_steps": int(accepted_steps),
                "evaluated": int(evaluated),
                "scalar_rhs_calls": int(rhs_calls() - initial_rhs_calls),
            }
            _validate_bdf_snapshot(snapshot, eta, initial.size)
            checkpoint.record(history, evaluated, snapshot)
    if evaluated != eta.size:
        raise ValueError(
            "Declared CMB continuous evolution returned an "
            "incomplete state history"
        )
    return numpy.ascontiguousarray(history)
