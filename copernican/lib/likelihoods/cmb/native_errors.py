"""Typed failure taxonomy for native CMB execution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class NativeCMBError(ValueError):
    """Base class for native failures that must cross likelihood boundaries."""

    category = "native_failure"
    proposal_rejection = False

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Store a stable message and structured diagnostic context."""

        super().__init__(str(message))
        self.context = dict(context or {})

    def add_context(self, **entries: Any) -> "NativeCMBError":
        """Fill absent diagnostic fields and return this exception."""

        for key, value in entries.items():
            if value is not None and key not in self.context:
                self.context[str(key)] = value
        return self

    def diagnostic(self) -> dict[str, Any]:
        """Return a serializable diagnostic payload for logs and tests."""

        return {
            "category": self.category,
            "message": str(self),
            "proposal_rejection": bool(self.proposal_rejection),
            "context": dict(self.context),
        }


class NativeParameterDomainError(NativeCMBError):
    """Identify a scientifically valid rejection of one parameter point."""

    category = "parameter_domain"
    proposal_rejection = True


class NativeUnsupportedCapabilityError(NativeCMBError):
    """Identify a requested capability absent from the declared graph."""

    category = "unsupported_capability"


class NativeContractError(NativeCMBError):
    """Identify an invalid model, dataset, or execution contract."""

    category = "contract_invalidity"


class NativeConvergenceError(NativeCMBError):
    """Identify a numerical solve or refinement that did not converge."""

    category = "convergence_failure"


class NativeNonFiniteEvolutionError(NativeCMBError):
    """Identify non-finite native states, sources, or spectra."""

    category = "nonfinite_evolution"


class NativeConstraintViolationError(NativeCMBError):
    """Identify a declared physical or numerical constraint violation."""

    category = "constraint_violation"


class NativePerformanceBudgetError(NativeCMBError):
    """Identify a native request that exceeded its workload budget."""

    category = "performance_budget"


class NativeImplementationError(NativeCMBError):
    """Identify an unexpected implementation or infrastructure fault."""

    category = "implementation_failure"


class NativeInitialPointError(NativeParameterDomainError):
    """Identify a configured initial point rejected before walker creation."""

    category = "initial_point_rejection"
    proposal_rejection = False


def native_failure_context(
    contract: Mapping[str, Any] | None,
    *,
    workload: str,
    spectra: Sequence[str] = (),
) -> dict[str, Any]:
    """Return stable model, parameter, gauge, and accuracy diagnostics."""

    contract = contract or {}
    perturbation_data = contract.get("perturbation_data")
    perturbations = contract.get("perturbations", {}) or {}
    controls = (
        getattr(perturbation_data, "accuracy_controls", {}) or {}
        if perturbation_data is not None
        else (
            perturbations.get("accuracy_controls", {}) or {}
            if isinstance(perturbations, Mapping)
            else {}
        )
    )
    gauge = (
        getattr(perturbation_data, "gauge", None)
        if perturbation_data is not None
        else (
            perturbations.get("gauge")
            if isinstance(perturbations, Mapping)
            else None
        )
    )
    parameters: dict[str, Any] = {}
    for source_name in ("model_parameters", "param_map"):
        source = contract.get(source_name, {}) or {}
        if isinstance(source, Mapping):
            parameters.update(
                {str(key): value for key, value in source.items()}
            )
    return {
        "model_name": str(contract.get("model_name", "")),
        "parameters": dict(sorted(parameters.items())),
        "gauge": None if gauge is None else str(gauge),
        "numerical_tier": controls.get("accuracy_tier"),
        "requested_spectra": tuple(str(name) for name in spectra),
        "workload": str(workload),
    }


def classify_native_exception(
    exc: BaseException,
    *,
    context: Mapping[str, Any] | None = None,
) -> NativeCMBError:
    """Translate an untyped internal exception at the native boundary."""

    if isinstance(exc, NativeCMBError):
        if context:
            exc.add_context(**dict(context))
        return exc

    message = str(exc)
    normalized = message.casefold()
    error_type: type[NativeCMBError]
    if "performance budget" in normalized:
        error_type = NativePerformanceBudgetError
    elif "non-finite" in normalized or "nonfinite" in normalized:
        error_type = NativeNonFiniteEvolutionError
    elif (
        "failed to converge" in normalized
        or "did not converge" in normalized
        or "under-resolved" in normalized
        or "incomplete state history" in normalized
    ):
        error_type = NativeConvergenceError
    elif (
        "constraint" in normalized
        or "conservation rule exceeded" in normalized
    ):
        error_type = NativeConstraintViolationError
    elif (
        "unsupported" in normalized
        or "does not provide requested" in normalized
    ):
        error_type = NativeUnsupportedCapabilityError
    elif isinstance(exc, (KeyError, TypeError, ValueError)):
        error_type = NativeContractError
    else:
        error_type = NativeImplementationError
    return error_type(message, context=context)


__all__ = [
    "NativeCMBError",
    "NativeConstraintViolationError",
    "NativeContractError",
    "NativeConvergenceError",
    "NativeImplementationError",
    "NativeInitialPointError",
    "NativeNonFiniteEvolutionError",
    "NativeParameterDomainError",
    "NativePerformanceBudgetError",
    "NativeUnsupportedCapabilityError",
    "classify_native_exception",
    "native_failure_context",
]
