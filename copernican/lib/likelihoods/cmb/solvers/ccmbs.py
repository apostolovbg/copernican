"""The single CCMBS engine and its internal hardware execution policy."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy
from scipy.linalg import expm

from ....cmb_contract import audit_cmb_capabilities
from ....cmb_identity import CCMBS_ID, CCMBS_LABEL
from ....model_coder import prepare_declared_cmb_execution_contract
from ..contracts import CMBResult, CMBSolverCapabilities
from ..errors import EngineCapabilityError, classify_exception, failure_context
from ..orchestrators.ccmbs import (
    _compute_declared_perturbation_spectrum,
    last_declared_postprocessing_evidence,
    last_declared_raw_spectra,
)
from ..runtime import cache
from ..runtime.evolution import prepare_runtime_assets

_PUBLIC_SPECTRA = ("TT", "TE", "EE", "BB", "PP", "TP", "EP")
_ACCELERATOR_ARCHITECTURES = ("metal", "vulkan", "cuda")


def _performance_record_after(
    previous_index: int,
) -> Mapping[str, Any] | None:
    """Return the latest declared performance record for one request."""

    record = cache.latest_cmb_performance_record()
    if record is None or int(record.get("request_index", 0)) <= previous_index:
        return None
    return record


def _result_provenance(
    record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Split performance metadata into diagnostics and phase timings."""

    if record is None:
        return {}, {}
    diagnostics = {
        "performance_record": dict(record),
        "outcome": str(record.get("outcome", "success")),
        "cache_state": str(record.get("cache_state", "cold")),
    }
    phases = {
        str(name): float(value)
        for name, value in dict(record.get("phase_seconds", {})).items()
    }
    return diagnostics, phases


def _candidate_architectures() -> tuple[str, ...]:
    """Return device architectures relevant to the current host."""

    if sys.platform == "darwin":
        return ("metal", "vulkan")
    if sys.platform.startswith("win"):
        return ("vulkan", "cuda")
    return ("vulkan", "cuda", "metal")


def _device_probe() -> dict[str, Any]:
    """Describe optional accelerator availability without importing it."""

    try:
        installed = importlib.util.find_spec("taichi") is not None
    except (ImportError, ModuleNotFoundError):
        installed = False
    return {
        "backend": "accelerator",
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
    """Create a typed non-applicability result for optional hardware."""

    return EngineCapabilityError(message, context=dict(context or {}))


def _load_accelerator(
    *,
    architecture: str | None = None,
) -> tuple[Any, str]:
    """Load one explicitly requested optional accelerator runtime."""

    probe = _device_probe()
    if not probe["taichi_installed"]:
        raise _unavailable_error(
            "The optional accelerator runtime is not installed",
            context={"device_probe": probe},
        )
    try:
        taichi = importlib.import_module("taichi")
    except (ImportError, ModuleNotFoundError) as exc:
        raise _unavailable_error(
            "The optional accelerator runtime could not be imported",
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
        "No requested accelerator architecture is available",
        context={
            "device_probe": probe,
            "requested_architecture": requested or None,
            "initialization_error": locals().get("last_error"),
        },
    )


def _cpu_two_state_collision(
    operator_matrices: numpy.ndarray,
    target_states: numpy.ndarray,
    operator_scales: numpy.ndarray,
    dt: float,
) -> numpy.ndarray:
    """Return the scalar CPU reference for the fixed-shape pilot."""

    return numpy.asarray(
        tuple(
            expm(
                numpy.asarray(matrix, dtype=float) * float(scale) * float(dt)
            ).dot(numpy.asarray(state, dtype=float))
            for matrix, state, scale in zip(
                operator_matrices,
                target_states,
                operator_scales,
            )
        ),
        dtype=float,
    )


def _apply_accelerated_collision(
    operator_matrices: numpy.ndarray,
    target_states: numpy.ndarray,
    operator_scales: numpy.ndarray,
    dt: float,
    *,
    architecture: str | None = None,
) -> numpy.ndarray:
    """Apply and validate one fixed-shape collision action on a device."""

    matrices = numpy.asarray(operator_matrices, dtype=float)
    states = numpy.asarray(target_states, dtype=float)
    scales = numpy.asarray(operator_scales, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
        raise ValueError("Accelerator matrices must have shape (n, 2, 2)")
    if states.shape != (matrices.shape[0], 2):
        raise ValueError("Accelerator states do not match matrices")
    if scales.shape != (matrices.shape[0],):
        raise ValueError("Accelerator scales do not match matrices")
    if not (
        numpy.all(numpy.isfinite(matrices))
        and numpy.all(numpy.isfinite(states))
        and numpy.all(numpy.isfinite(scales))
        and numpy.isfinite(float(dt))
    ):
        raise ValueError("Accelerator inputs must be finite")
    accelerator, selected_architecture = _load_accelerator(
        architecture=architecture,
    )
    mode_count = int(matrices.shape[0])
    matrix_field = accelerator.field(
        dtype=accelerator.f32,
        shape=(mode_count, 2, 2),
    )
    state_field = accelerator.field(
        dtype=accelerator.f32,
        shape=(mode_count, 2),
    )
    scale_field = accelerator.field(dtype=accelerator.f32, shape=mode_count)
    output_field = accelerator.field(
        dtype=accelerator.f32,
        shape=(mode_count, 2),
    )
    matrix_field.from_numpy(numpy.asarray(matrices, dtype=numpy.float32))
    state_field.from_numpy(numpy.asarray(states, dtype=numpy.float32))
    scale_field.from_numpy(numpy.asarray(scales, dtype=numpy.float32))

    @accelerator.kernel
    def apply(dt_value: accelerator.f32):
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
                delta = accelerator.sqrt(discriminant)
                if delta <= 1.0e-6:
                    hyperbolic = 1.0
                    cosine = 1.0
                else:
                    hyperbolic = accelerator.sinh(delta) / delta
                    cosine = accelerator.cosh(delta)
            else:
                frequency = accelerator.sqrt(-discriminant)
                if frequency <= 1.0e-6:
                    hyperbolic = 1.0
                    cosine = 1.0
                else:
                    hyperbolic = accelerator.sin(frequency) / frequency
                    cosine = accelerator.cos(frequency)
            factor = accelerator.exp(trace_half)
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
            "Accelerator collision-kernel execution failed",
            context={
                "architecture": selected_architecture,
                "mode_count": mode_count,
            },
        ) from exc
    result = numpy.asarray(output_field.to_numpy(), dtype=float)
    reference = _cpu_two_state_collision(
        matrices,
        states,
        scales,
        dt,
    )
    if not numpy.allclose(result, reference, rtol=2.0e-5, atol=2.0e-6):
        delta = numpy.abs(result - reference)
        raise _unavailable_error(
            "Accelerator collision pilot failed CPU equivalence validation",
            context={
                "architecture": selected_architecture,
                "max_absolute_error": float(numpy.max(delta, initial=0.0)),
                "max_relative_error": float(
                    numpy.max(
                        delta / numpy.maximum(numpy.abs(reference), 1.0e-30),
                        initial=0.0,
                    )
                ),
            },
        )
    return result


class CCMBS:
    """Expose one engine identity and select execution internally."""

    solver_id = CCMBS_ID
    solver_label = CCMBS_LABEL

    def __init__(self) -> None:
        """Initialize one CCMBS engine without probing a device."""

    @staticmethod
    def _backend_provenance() -> dict[str, Any]:
        """Return deterministic selected-device and fallback evidence."""

        accelerator = _device_probe()
        return {
            "selected_backend": "cpu",
            "backend_kind": "cpu",
            "fallback": True,
            "fallback_reason": (
                "accelerator_full_declared_graph_not_validated"
                if accelerator.get("taichi_installed")
                else "accelerator_unavailable_or_not_installed"
            ),
            "accelerator": accelerator,
        }

    def capabilities(
        self,
        contract: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        """Return stable CCMBS and device capabilities."""

        supported = _PUBLIC_SPECTRA
        accuracy_tiers: tuple[str, ...] = ()
        grids: dict[str, Any] = {}
        if contract is not None:
            perturbation_data = contract.get("perturbation_data")
            if perturbation_data is not None:
                audit = audit_cmb_capabilities(perturbation_data)
                supported = tuple(audit.supported_observables)
                controls = (
                    getattr(
                        perturbation_data,
                        "accuracy_controls",
                        {},
                    )
                    or {}
                )
                tier = controls.get("accuracy_tier")
                if tier is not None:
                    accuracy_tiers = (str(tier),)
                numerical = getattr(perturbation_data, "numerics", {}) or {}
                if isinstance(numerical, Mapping):
                    grids = {
                        str(key): numerical[key]
                        for key in sorted(numerical, key=str)
                    }
        capabilities = CMBSolverCapabilities(
            solver_id=self.solver_id,
            solver_label=self.solver_label,
            execution_backend="cpu",
            implementation="ccmbs_declared_graph",
            supported_spectra=tuple(str(name) for name in supported),
            supported_grids=grids,
            accuracy_tiers=accuracy_tiers,
            batch_mode="ordered_scalar_adapter",
            preparation=True,
            cleanup=True,
            device_probe=self._backend_provenance(),
        ).to_mapping()
        capabilities["solver_id"] = self.solver_id
        capabilities["solver_label"] = self.solver_label
        return capabilities

    def prepare(self, contract: Mapping[str, object]) -> Mapping[str, Any]:
        """Prepare a declared contract and its structural graph assets."""

        normalizer = prepare_declared_cmb_execution_contract
        try:
            from .. import cmb as cmb_api

            normalizer = getattr(
                cmb_api,
                "prepare_cmb_execution_contract",
                normalizer,
            )
        except ImportError:
            pass
        prepared = normalizer(contract)
        perturbation_data = prepared.get("perturbation_data")
        if perturbation_data is not None and hasattr(
            perturbation_data,
            "equations",
        ):
            prepare_runtime_assets(
                str(prepared.get("runtime_signature", "")),
                perturbation_data,
            )
        return prepared

    def evaluate(
        self,
        prepared: object,
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> CMBResult:
        """Evaluate the declared graph on CCMBS's selected execution path."""

        requested_ells = tuple(int(value) for value in ells)
        requested_spectra = tuple(str(value) for value in spectra)
        previous = cache.latest_cmb_performance_record()
        previous_index = (
            0 if previous is None else int(previous.get("request_index", 0))
        )
        contract = prepared
        if not isinstance(contract, Mapping):
            failure = classify_exception(
                TypeError("Prepared CMB solver contract must be a mapping"),
                context={
                    "workload": str(workload),
                    "requested_spectra": requested_spectra,
                },
            )
            return CMBResult(
                requested_ells=requested_ells,
                requested_spectra=requested_spectra,
                failure=failure,
                solver_id=self.solver_id,
                solver_label=self.solver_label,
            )
        started = perf_counter()
        try:
            # Do not allow a previous successful request to leak raw products
            # into a failed result.
            from ..orchestrators import ccmbs as ccmbs_orchestrator

            ccmbs_orchestrator._LAST_DECLARED_RAW_SPECTRA.set(None)
            ccmbs_orchestrator._LAST_DECLARED_POSTPROCESSING_EVIDENCE.set(None)
            executor = _compute_declared_perturbation_spectrum
            try:
                from .. import cmb as cmb_api

                executor = getattr(
                    cmb_api,
                    "_compute_declared_perturbation_spectrum",
                    executor,
                )
            except ImportError:
                pass
            background_provider = contract.get("_background_provider")
            spectra_result = executor(
                contract,
                requested_ells,
                spectra=requested_spectra,
                workload=str(workload),
                background_provider=background_provider,
            )
        # DEVCOV_ALLOW_BROAD_ONCE solver boundary: classify failures.
        except Exception as exc:  # DEVCOV_ALLOW_BROAD_ONCE
            failure = classify_exception(
                exc,
                context=failure_context(
                    contract,
                    workload=str(workload),
                    spectra=requested_spectra,
                ),
            )
            record = _performance_record_after(previous_index)
            diagnostics, phases = _result_provenance(record)
            diagnostics["elapsed_seconds"] = max(perf_counter() - started, 0.0)
            diagnostics["ccmbs_backend"] = self._backend_provenance()
            return CMBResult(
                requested_ells=requested_ells,
                requested_spectra=requested_spectra,
                diagnostics=diagnostics,
                cache_provenance=cache.cmb_cache_stats(),
                phase_timings=phases,
                failure=failure,
                solver_id=self.solver_id,
                solver_label=self.solver_label,
            )
        record = _performance_record_after(previous_index)
        diagnostics, phases = _result_provenance(record)
        diagnostics["elapsed_seconds"] = max(perf_counter() - started, 0.0)
        diagnostics["ccmbs_backend"] = self._backend_provenance()
        evidence = last_declared_postprocessing_evidence()
        if evidence is not None:
            diagnostics["postprocessing_evidence"] = dict(evidence)
        return CMBResult(
            spectra=spectra_result,
            requested_ells=requested_ells,
            requested_spectra=requested_spectra,
            diagnostics=diagnostics,
            cache_provenance=cache.cmb_cache_stats(),
            phase_timings=phases,
            solver_id=self.solver_id,
            solver_label=self.solver_label,
            raw_spectra=last_declared_raw_spectra(),
        )

    def evaluate_batch(
        self,
        prepared: Sequence[object],
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> tuple[CMBResult, ...]:
        """Evaluate each prepared contract in order with isolated outcomes."""

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
        """Release process-owned CCMBS resources."""


__all__ = ["CCMBS"]
