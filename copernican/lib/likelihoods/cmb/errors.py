"""Typed failure taxonomy for declared CMB execution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class CMBError(ValueError):
    """Base class for declared failures that cross likelihood boundaries."""

    category = "cmb_failure"
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

    def add_context(self, **entries: Any) -> "CMBError":
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


class ParameterDomainError(CMBError):
    """Identify a scientifically valid rejection of one parameter point."""

    category = "parameter_domain"
    proposal_rejection = True


class UnsupportedCapabilityError(CMBError):
    """Identify a requested capability absent from the declared graph."""

    category = "unsupported_capability"


class ContractError(CMBError):
    """Identify an invalid model, dataset, or execution contract."""

    category = "contract_invalidity"


class ModelDiscoveryError(CMBError):
    """Identify a model file that cannot be compiled for CCMBS."""

    category = "model_discovery"


class ConvergenceError(CMBError):
    """Identify a numerical solve or refinement that did not converge."""

    category = "convergence_failure"


class NonFiniteEvolutionError(CMBError):
    """Identify non-finite declared states, sources, or spectra."""

    category = "nonfinite_evolution"


class ConstraintViolationError(CMBError):
    """Identify a declared physical or numerical constraint violation."""

    category = "constraint_violation"


class ImplementationError(CMBError):
    """Identify an unexpected implementation or infrastructure fault."""

    category = "implementation_failure"


class InitialPointError(ParameterDomainError):
    """Identify a configured initial point rejected before walker creation."""

    category = "initial_point_rejection"
    proposal_rejection = False


def failure_context(
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


def classify_exception(
    exc: BaseException,
    *,
    context: Mapping[str, Any] | None = None,
) -> CMBError:
    """Translate an untyped internal exception at the declared boundary."""

    if isinstance(exc, CMBError):
        if context:
            exc.add_context(**dict(context))
        return exc

    message = str(exc)
    normalized = message.casefold()
    error_type: type[CMBError]
    if "non-finite" in normalized or "nonfinite" in normalized:
        error_type = NonFiniteEvolutionError
    elif (
        "failed to converge" in normalized
        or "did not converge" in normalized
        or "under-resolved" in normalized
        or "incomplete state history" in normalized
    ):
        error_type = ConvergenceError
    elif (
        "constraint" in normalized
        or "conservation rule exceeded" in normalized
    ):
        error_type = ConstraintViolationError
    elif (
        "unsupported" in normalized
        or "does not provide requested" in normalized
    ):
        error_type = UnsupportedCapabilityError
    elif isinstance(exc, (KeyError, TypeError, ValueError)):
        error_type = ContractError
    else:
        error_type = ImplementationError
    return error_type(message, context=context)


__all__ = [
    "CMBError",
    "ConstraintViolationError",
    "ContractError",
    "ConvergenceError",
    "ImplementationError",
    "InitialPointError",
    "ModelDiscoveryError",
    "NonFiniteEvolutionError",
    "ParameterDomainError",
    "UnsupportedCapabilityError",
    "classify_exception",
    "failure_context",
]
