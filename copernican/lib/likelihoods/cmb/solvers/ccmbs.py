"""The single public CCMBS engine and its internal backend policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from ....cmb_identity import CCMBS_ID, CCMBS_LABEL
from ..contracts import CMBResult
from .ccmbs_numpy import CCMBSCPUBackend
from .ccmbs_taichi import taichi_device_probe


class CCMBS:
    """Expose one solver identity while selecting execution internally.

    The declared graph currently remains on the NumPy/SciPy correctness path.
    The Taichi probe is retained as capability evidence; it is not selected
    for the full graph until its fixed-shape kernels have passed the CPU
    equivalence contract.
    """

    solver_id = CCMBS_ID
    solver_label = CCMBS_LABEL

    def __init__(self) -> None:
        """Initialize the portable backend without probing a device."""

        self._cpu = CCMBSCPUBackend()

    @staticmethod
    def _backend_provenance() -> dict[str, Any]:
        """Return deterministic selected-backend and fallback evidence."""

        accelerator = taichi_device_probe()
        return {
            "selected_backend": "numpy_cpu",
            "backend_kind": "cpu",
            "fallback": True,
            "fallback_reason": (
                "taichi_full_declared_graph_not_validated"
                if accelerator.get("taichi_installed")
                else "accelerator_unavailable_or_not_installed"
            ),
            "accelerator": accelerator,
        }

    def capabilities(
        self,
        contract: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        """Return CCMBS capabilities with backend provenance."""

        capabilities = dict(self._cpu.capabilities(contract))
        capabilities.update(
            {
                "solver_id": self.solver_id,
                "solver_label": self.solver_label,
                "implementation": "ccmbs_declared_graph",
                "execution_backend": "cpu",
                "device_probe": self._backend_provenance(),
            }
        )
        return capabilities

    def prepare(self, contract: Mapping[str, object]) -> Mapping[str, Any]:
        """Prepare a declared contract through the CPU correctness backend."""

        return self._cpu.prepare(contract)

    def _annotate(self, result: CMBResult) -> CMBResult:
        """Attach backend evidence without changing the solver identity."""

        diagnostics = dict(result.diagnostics or {})
        diagnostics["ccmbs_backend"] = self._backend_provenance()
        return replace(
            result,
            diagnostics=diagnostics,
            solver_id=self.solver_id,
            solver_label=self.solver_label,
        )

    def evaluate(
        self,
        prepared: object,
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> CMBResult:
        """Evaluate the declared graph on the selected internal backend."""

        return self._annotate(
            self._cpu.evaluate(
                prepared,
                ells,
                spectra=spectra,
                workload=workload,
            )
        )

    def evaluate_batch(
        self,
        prepared: Sequence[object],
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> tuple[CMBResult, ...]:
        """Evaluate contracts in order with one CCMBS identity."""

        return tuple(
            self._annotate(result)
            for result in self._cpu.evaluate_batch(
                prepared,
                ells,
                spectra=spectra,
                workload=workload,
            )
        )

    def cleanup(self) -> None:
        """Release internal backend resources."""

        self._cpu.cleanup()


__all__ = ["CCMBS"]
