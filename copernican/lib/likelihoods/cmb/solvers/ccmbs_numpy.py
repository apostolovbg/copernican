"""NumPy/SciPy reference implementation of the CCMBS solver contract.

This adapter preserves the established declared-graph numerical path while
adding protocol-level preparation, typed results, cache provenance, and
phase timing for callers that do not need implementation details.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping, Sequence

from ....cmb_contract import audit_cmb_capabilities
from ....cmb_identity import CCMBS_ID, CCMBS_LABEL
from ....model_coder import prepare_declared_cmb_execution_contract
from ..contracts import CMBResult, CMBSolverCapabilities
from ..errors import classify_exception, failure_context
from ..orchestrators.ccmbs import _compute_declared_perturbation_spectrum
from ..runtime import cache
from ..runtime.evolution import prepare_runtime_assets

_PUBLIC_SPECTRA = ("TT", "TE", "EE", "BB", "PP", "TP", "EP")


def _performance_record_after(previous_index: int) -> Mapping[str, Any] | None:
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


class CCMBSNumpySolver:
    """Adapt the exact declared-graph executor to the CCMBS protocol."""

    solver_id = CCMBS_ID
    solver_label = CCMBS_LABEL

    def capabilities(
        self,
        contract: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        """Return stable CPU, grid, spectrum, and accuracy capabilities."""

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
        return CMBSolverCapabilities(
            solver_id=self.solver_id,
            solver_label=self.solver_label,
            execution_backend="cpu",
            implementation="numpy_scipy_reference",
            supported_spectra=tuple(str(name) for name in supported),
            supported_grids=grids,
            accuracy_tiers=accuracy_tiers,
            batch_mode="ordered_scalar_adapter",
            preparation=True,
            cleanup=True,
            device_probe={"backend": "cpu", "taichi_imported": False},
        ).to_mapping()

    def prepare(self, contract: Mapping[str, object]) -> Mapping[str, Any]:
        """Normalize a contract and materialize its structural graph assets."""

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
            perturbation_data, "equations"
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
        """Evaluate one prepared contract without changing declared kernels."""

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
        # DEVCOV_ALLOW_BROAD_ONCE solver adapter boundary: declared execution
        # classifies all backend failures into the public typed taxonomy.
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
        return CMBResult(
            spectra=spectra_result,
            requested_ells=requested_ells,
            requested_spectra=requested_spectra,
            diagnostics=diagnostics,
            cache_provenance=cache.cmb_cache_stats(),
            phase_timings=phases,
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
        """Release reference-backend resources; caches remain process-owned."""


__all__ = ["CCMBSNumpySolver"]
