"""Optional Taichi accelerator boundary for fixed-shape CMB kernels.

The module is import-safe when Taichi is not installed.  The default CCMBS
route never imports or initializes a device runtime; callers must select this
backend explicitly and receive a typed capability failure when its optional
runtime or full declared-graph route is unavailable.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy

from ..contracts import CMBResult, CMBSolverCapabilities
from ..errors import EngineCapabilityError

TAICHI_CMB_SOLVER_ID = "ccmbs_taichi"
TAICHI_CMB_SOLVER_LABEL = "CCMBS — optional Taichi accelerator boundary"
_ACCELERATOR_ARCHITECTURES = ("metal", "vulkan", "cuda")


def _candidate_architectures() -> tuple[str, ...]:
    """Return device architectures relevant to the current host."""

    if sys.platform == "darwin":
        return ("metal", "vulkan")
    if sys.platform.startswith("win"):
        return ("vulkan", "cuda")
    return ("vulkan", "cuda", "metal")


def taichi_device_probe() -> dict[str, Any]:
    """Describe optional Taichi availability without importing Taichi."""

    try:
        installed = importlib.util.find_spec("taichi") is not None
    except (ImportError, ModuleNotFoundError):
        installed = False
    return {
        "backend": "taichi",
        "taichi_imported": False,
        "taichi_installed": bool(installed),
        "platform": sys.platform,
        "machine": platform.machine(),
        "candidate_architectures": tuple(
            name
            for name in _candidate_architectures()
            if name in _ACCELERATOR_ARCHITECTURES
        ),
        "hardware_probe": "deferred_until_explicit_selection",
    }


def _unavailable_error(
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> EngineCapabilityError:
    """Create a typed non-applicability result for the optional route."""

    return EngineCapabilityError(message, context=dict(context or {}))


def _load_taichi(*, architecture: str | None = None) -> tuple[Any, str]:
    """Load and initialize one explicitly requested accelerator runtime."""

    probe = taichi_device_probe()
    if not probe["taichi_installed"]:
        raise _unavailable_error(
            "The optional Taichi runtime is not installed",
            context={"device_probe": probe},
        )
    try:
        taichi = importlib.import_module("taichi")
    except (ImportError, ModuleNotFoundError) as exc:
        raise _unavailable_error(
            "The optional Taichi runtime could not be imported",
            context={"device_probe": probe},
        ) from exc
    requested = str(architecture or "").strip().lower()
    candidates = (
        (requested,) if requested else tuple(probe["candidate_architectures"])
    )
    for candidate in candidates:
        if candidate not in _ACCELERATOR_ARCHITECTURES:
            continue
        arch = getattr(taichi, candidate, None)
        if arch is None:
            continue
        try:
            taichi.init(
                arch=arch,
                default_fp=taichi.f32,
                offline_cache=False,
                log_level=taichi.ERROR,
            )
        # DEVCOV_ALLOW_BROAD_ONCE optional device initialization boundary.
        except Exception as exc:  # DEVCOV_ALLOW_BROAD_ONCE
            last_error = str(exc)
            continue
        return taichi, candidate
    raise _unavailable_error(
        "No requested Taichi accelerator architecture is available",
        context={
            "device_probe": probe,
            "requested_architecture": requested or None,
            "initialization_error": locals().get("last_error"),
        },
    )


def apply_taichi_two_state_collision(
    operator_matrices: numpy.ndarray,
    target_states: numpy.ndarray,
    operator_scales: numpy.ndarray,
    dt: float,
    *,
    architecture: str | None = None,
) -> numpy.ndarray:
    """Apply an exact two-state collision block on an explicit device.

    This fixed-shape pilot is independent from full CCMBS orchestration.  It
    has no CPU fallback: unavailable Taichi or device initialization raises a
    typed engine-capability error.
    """

    matrices = numpy.asarray(operator_matrices, dtype=float)
    states = numpy.asarray(target_states, dtype=float)
    scales = numpy.asarray(operator_scales, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
        raise ValueError("Taichi collision matrices must have shape (n, 2, 2)")
    if states.shape != (matrices.shape[0], 2):
        raise ValueError("Taichi collision states do not match matrices")
    if scales.shape != (matrices.shape[0],):
        raise ValueError("Taichi collision scales do not match matrices")
    if not (
        numpy.all(numpy.isfinite(matrices))
        and numpy.all(numpy.isfinite(states))
        and numpy.all(numpy.isfinite(scales))
        and numpy.isfinite(float(dt))
    ):
        raise ValueError("Taichi collision inputs must be finite")
    taichi, selected_architecture = _load_taichi(
        architecture=architecture,
    )
    mode_count = int(matrices.shape[0])
    matrix_field = taichi.field(dtype=taichi.f32, shape=(mode_count, 2, 2))
    state_field = taichi.field(dtype=taichi.f32, shape=(mode_count, 2))
    scale_field = taichi.field(dtype=taichi.f32, shape=mode_count)
    output_field = taichi.field(dtype=taichi.f32, shape=(mode_count, 2))
    matrix_field.from_numpy(numpy.asarray(matrices, dtype=numpy.float32))
    state_field.from_numpy(numpy.asarray(states, dtype=numpy.float32))
    scale_field.from_numpy(numpy.asarray(scales, dtype=numpy.float32))

    @taichi.kernel
    def apply(dt_value: taichi.f32):
        """Evaluate the exact real 2x2 exponential for every mode."""

        for mode_index in range(mode_count):
            matrix_00 = (
                matrix_field[mode_index, 0, 0] * scale_field[mode_index]
            )
            matrix_01 = (
                matrix_field[mode_index, 0, 1] * scale_field[mode_index]
            )
            matrix_10 = (
                matrix_field[mode_index, 1, 0] * scale_field[mode_index]
            )
            matrix_11 = (
                matrix_field[mode_index, 1, 1] * scale_field[mode_index]
            )
            matrix_00 *= dt_value
            matrix_01 *= dt_value
            matrix_10 *= dt_value
            matrix_11 *= dt_value
            trace_half = 0.5 * (matrix_00 + matrix_11)
            centered = 0.5 * (matrix_00 - matrix_11)
            discriminant = centered * centered + matrix_01 * matrix_10
            if discriminant >= 0.0:
                delta = taichi.sqrt(discriminant)
                if delta <= 1.0e-6:
                    hyperbolic = 1.0
                    cosine = 1.0
                else:
                    hyperbolic = taichi.sinh(delta) / delta
                    cosine = taichi.cosh(delta)
            else:
                frequency = taichi.sqrt(-discriminant)
                if frequency <= 1.0e-6:
                    hyperbolic = 1.0
                    cosine = 1.0
                else:
                    hyperbolic = taichi.sin(frequency) / frequency
                    cosine = taichi.cos(frequency)
            factor = taichi.exp(trace_half)
            first_centered = centered * state_field[mode_index, 0]
            first_centered += matrix_01 * state_field[mode_index, 1]
            second_centered = matrix_10 * state_field[mode_index, 0]
            second_centered -= centered * state_field[mode_index, 1]
            output_field[mode_index, 0] = factor * (
                cosine * state_field[mode_index, 0]
                + hyperbolic * first_centered
            )
            output_field[mode_index, 1] = factor * (
                cosine * state_field[mode_index, 1]
                + hyperbolic * second_centered
            )

    try:
        apply(float(dt))
    # DEVCOV_ALLOW_BROAD_ONCE optional kernel execution boundary.
    except Exception as exc:  # DEVCOV_ALLOW_BROAD_ONCE
        raise _unavailable_error(
            "Taichi collision-kernel execution failed",
            context={
                "architecture": selected_architecture,
                "mode_count": mode_count,
            },
        ) from exc
    return numpy.asarray(output_field.to_numpy(), dtype=float)


class CCMBSTaichiSolver:
    """Expose explicit accelerator capability without a hidden CPU fallback."""

    solver_id = TAICHI_CMB_SOLVER_ID
    solver_label = TAICHI_CMB_SOLVER_LABEL

    def capabilities(
        self,
        contract: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        """Return deterministic optional-device capability metadata."""

        del contract
        probe = taichi_device_probe()
        return CMBSolverCapabilities(
            solver_id=self.solver_id,
            solver_label=self.solver_label,
            execution_backend="accelerator",
            implementation="taichi_fixed_shape_collision_pilot",
            supported_spectra=(),
            supported_grids={},
            accuracy_tiers=(),
            batch_mode="ordered_accelerator_boundary",
            preparation=True,
            cleanup=True,
            device_probe={
                **probe,
                "route_status": "fixed_shape_kernel_pilot",
                "full_declared_graph": False,
            },
        ).to_mapping()

    def prepare(self, contract: Mapping[str, object]) -> Mapping[str, object]:
        """Retain the contract without allocating a device implicitly."""

        if not isinstance(contract, Mapping):
            raise TypeError("Taichi solver contracts must be mappings")
        return contract

    def evaluate(
        self,
        prepared: object,
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> CMBResult:
        """Return typed non-applicability until the full graph is ported."""

        del prepared
        failure = _unavailable_error(
            "The Taichi backend currently exposes fixed-shape kernels only; "
            "the full declared CMB graph has no accelerator route",
            context={
                "workload": str(workload),
                "requested_spectra": tuple(str(name) for name in spectra),
                "device_probe": taichi_device_probe(),
            },
        )
        return CMBResult(
            requested_ells=tuple(int(value) for value in ells),
            requested_spectra=tuple(str(name) for name in spectra),
            failure=failure,
            solver_id=self.solver_id,
            solver_label=self.solver_label,
        )

    def evaluate_batch(
        self,
        prepared: Sequence[object],
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> tuple[CMBResult, ...]:
        """Return one typed, ordered non-applicability result per contract."""

        return tuple(
            self.evaluate(
                item,
                ells,
                spectra=spectra,
                workload=workload,
            )
            for item in prepared
        )

    def cleanup(self) -> None:
        """Release no process-global device state implicitly."""


__all__ = [
    "CCMBSTaichiSolver",
    "TAICHI_CMB_SOLVER_ID",
    "TAICHI_CMB_SOLVER_LABEL",
    "apply_taichi_two_state_collision",
    "taichi_device_probe",
]
