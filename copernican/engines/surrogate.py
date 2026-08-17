"""Deterministic local surrogates and exact delayed-acceptance control.

The helpers in this module deliberately keep the approximation boundary
small.  Training values are always supplied by an exact target evaluator,
unsupported points require an exact fallback, and every proposal records the
stage that made its decision.  The default MCMC engine does not import or use
these classes unless delayed acceptance is explicitly enabled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy

_SURROGATE_DEFAULTS = {
    "min_support": 3,
    "neighbor_count": 8,
    "uncertainty_threshold": 1.0,
    "max_samples": 256,
    "proposal_scale": 0.05,
}
_SURROGATE_KEYS = frozenset(_SURROGATE_DEFAULTS)


def validate_delayed_acceptance_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize the explicit delayed-acceptance settings."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("delayed-acceptance configuration must be a mapping")
    unknown = sorted(set(config) - _SURROGATE_KEYS)
    if unknown:
        raise ValueError(
            "unknown delayed-acceptance setting(s): " + ", ".join(unknown)
        )
    normalized = dict(_SURROGATE_DEFAULTS)
    normalized.update(config)
    for key in ("min_support", "neighbor_count", "max_samples"):
        try:
            normalized[key] = int(normalized[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if normalized[key] < 1:
            raise ValueError(f"{key} must be at least one")
    for key in ("uncertainty_threshold", "proposal_scale"):
        try:
            normalized[key] = float(normalized[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(normalized[key]) or normalized[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return normalized


def _as_float_vector(
    values: Sequence[float], size: int | None = None
) -> numpy.ndarray:
    """Return a finite one-dimensional parameter vector."""

    vector = numpy.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("parameter values must be one-dimensional")
    if size is not None and vector.size != int(size):
        raise ValueError("parameter vector has an unexpected dimension")
    if not numpy.all(numpy.isfinite(vector)):
        raise ValueError("parameter values must be finite")
    return vector


def _jsonable(value: Any) -> Any:
    """Convert diagnostics to deterministic JSON-compatible values."""

    if isinstance(value, numpy.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, numpy.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class SurrogateResult:
    """Describe one deterministic surrogate prediction and its support."""

    prediction: float | None
    uncertainty: float
    support: int
    training_sample_ids: tuple[str, ...]
    domain_status: str
    exact_required: bool
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return stable diagnostics suitable for manifests and logs."""

        return {
            "prediction": self.prediction,
            "uncertainty": float(self.uncertainty),
            "support": int(self.support),
            "training_sample_ids": list(self.training_sample_ids),
            "domain_status": self.domain_status,
            "exact_required": bool(self.exact_required),
            "provenance": _jsonable(self.provenance),
        }


class DeterministicLocalSurrogate:
    """Interpolate exact samples in a normalized bounded parameter domain."""

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        min_support: int = 3,
        neighbor_count: int = 8,
        uncertainty_threshold: float = 1.0,
        max_samples: int = 256,
    ) -> None:
        """Validate bounds and configure deterministic support thresholds."""

        self.lower = _as_float_vector(lower)
        self.upper = _as_float_vector(upper, self.lower.size)
        if numpy.any(self.upper <= self.lower):
            raise ValueError("surrogate bounds must have positive widths")
        self.width = self.upper - self.lower
        self.min_support = max(1, int(min_support))
        self.neighbor_count = max(1, int(neighbor_count))
        self.uncertainty_threshold = float(uncertainty_threshold)
        if self.uncertainty_threshold < 0 or not math.isfinite(
            self.uncertainty_threshold
        ):
            raise ValueError(
                "uncertainty_threshold must be finite and nonnegative"
            )
        self.max_samples = max(1, int(max_samples))
        self._points: list[numpy.ndarray] = []
        self._values: list[float] = []
        self._sample_ids: list[str] = []

    @property
    def dimension(self) -> int:
        """Return the number of active parameters represented by the model."""

        return int(self.lower.size)

    @property
    def sample_count(self) -> int:
        """Return the number of exact training samples retained."""

        return len(self._points)

    @property
    def training_sample_ids(self) -> tuple[str, ...]:
        """Return retained exact-sample identities in insertion order."""

        return tuple(self._sample_ids)

    @property
    def cache_identity(self) -> str:
        """Return a stable identity for bounds and interpolation controls."""

        payload = {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "min_support": self.min_support,
            "neighbor_count": self.neighbor_count,
            "uncertainty_threshold": self.uncertainty_threshold,
            "max_samples": self.max_samples,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return (
            "surrogate:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )

    def normalized(self, params: Sequence[float]) -> numpy.ndarray:
        """Return normalized coordinates without silently clipping them."""

        vector = _as_float_vector(params, self.dimension)
        return (vector - self.lower) / self.width

    def add_exact_sample(
        self,
        params: Sequence[float],
        exact_value: float,
        *,
        sample_id: str | None = None,
    ) -> str:
        """Add one exact target value and return its stable sample identity."""

        vector = _as_float_vector(params, self.dimension)
        value = float(exact_value)
        if not math.isfinite(value):
            raise ValueError("surrogate training values must be finite")
        normalized = self.normalized(vector)
        if numpy.any(normalized < 0.0) or numpy.any(normalized > 1.0):
            raise ValueError(
                "surrogate training point lies outside its domain"
            )
        for index, point in enumerate(self._points):
            if numpy.array_equal(point, vector):
                return self._sample_ids[index]
        identity = str(sample_id or f"exact-{len(self._sample_ids):06d}")
        if len(self._points) >= self.max_samples:
            self._points.pop(0)
            self._values.pop(0)
            self._sample_ids.pop(0)
        self._points.append(vector.copy())
        self._values.append(value)
        self._sample_ids.append(identity)
        return identity

    def predict(self, params: Sequence[float]) -> SurrogateResult:
        """Predict one value, requiring exact evaluation when support is
        weak.
        """

        vector = _as_float_vector(params, self.dimension)
        normalized = self.normalized(vector)
        if numpy.any(normalized < 0.0) or numpy.any(normalized > 1.0):
            return SurrogateResult(
                prediction=None,
                uncertainty=float("inf"),
                support=0,
                training_sample_ids=(),
                domain_status="outside_domain",
                exact_required=True,
                provenance={"cache_identity": self.cache_identity},
            )
        if not self._points:
            return self._unsupported_result("insufficient_support", 0, ())

        points = numpy.asarray(self._points, dtype=float)
        values = numpy.asarray(self._values, dtype=float)
        distances = numpy.linalg.norm(
            (points - self.lower) / self.width - normalized,
            axis=1,
        )
        order = numpy.argsort(distances, kind="stable")
        exact_index = int(order[0])
        if distances[exact_index] <= 1e-14:
            return SurrogateResult(
                prediction=float(values[exact_index]),
                uncertainty=0.0,
                support=1,
                training_sample_ids=(self._sample_ids[exact_index],),
                domain_status="supported",
                exact_required=False,
                provenance={
                    "cache_identity": self.cache_identity,
                    "interpolant": "exact_training_sample",
                },
            )

        count = min(len(order), self.neighbor_count)
        selected = order[:count]
        selected_distances = distances[selected]
        ids = tuple(self._sample_ids[int(index)] for index in selected)
        if count < self.min_support:
            return self._unsupported_result("insufficient_support", count, ids)
        weights = 1.0 / numpy.maximum(selected_distances, 1e-12)
        weights /= numpy.sum(weights)
        prediction = float(numpy.sum(weights * values[selected]))
        uncertainty = float(
            numpy.sqrt(
                numpy.sum(weights * (values[selected] - prediction) ** 2)
            )
        )
        uncertain = uncertainty > self.uncertainty_threshold
        return SurrogateResult(
            prediction=prediction,
            uncertainty=uncertainty,
            support=count,
            training_sample_ids=ids,
            domain_status="uncertain_support" if uncertain else "supported",
            exact_required=uncertain,
            provenance={
                "cache_identity": self.cache_identity,
                "interpolant": "inverse_distance_weighted",
                "normalized_distance": float(selected_distances[0]),
            },
        )

    def _unsupported_result(
        self,
        status: str,
        support: int,
        sample_ids: tuple[str, ...],
    ) -> SurrogateResult:
        """Build a consistent result for points lacking safe support."""

        return SurrogateResult(
            prediction=None,
            uncertainty=float("inf"),
            support=int(support),
            training_sample_ids=sample_ids,
            domain_status=status,
            exact_required=True,
            provenance={"cache_identity": self.cache_identity},
        )


@dataclass(frozen=True, slots=True)
class DelayedAcceptanceOutcome:
    """Record one proposal decision and its exact correction status."""

    params: tuple[float, ...]
    exact_log_probability: float
    accepted: bool
    stage: str
    exact_called: bool
    acceptance_probability: float
    correction_log_ratio: float
    surrogate: SurrogateResult

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe proposal record."""

        return {
            "params": list(self.params),
            "exact_log_probability": self.exact_log_probability,
            "accepted": self.accepted,
            "stage": self.stage,
            "exact_called": self.exact_called,
            "acceptance_probability": self.acceptance_probability,
            "correction_log_ratio": self.correction_log_ratio,
            "surrogate": self.surrogate.to_dict(),
        }


class DelayedAcceptanceController:
    """Apply a symmetric-proposal delayed-acceptance correction exactly."""

    def __init__(
        self,
        exact_log_probability: Callable[[Sequence[float]], float],
        surrogate: DeterministicLocalSurrogate,
        *,
        rng: numpy.random.Generator,
        proposal_scale: float = 0.05,
    ) -> None:
        """Store the exact target, surrogate, and deterministic RNG."""

        if proposal_scale <= 0 or not math.isfinite(float(proposal_scale)):
            raise ValueError("proposal_scale must be finite and positive")
        self.exact_log_probability = exact_log_probability
        self.surrogate = surrogate
        self.rng = rng
        self.proposal_scale = float(proposal_scale)
        self.counters: dict[str, int] = {
            "proposals": 0,
            "surrogate_predictions": 0,
            "stage_one_rejections": 0,
            "exact_corrections": 0,
            "exact_rejections": 0,
            "support_fallbacks": 0,
            "exact_calls": 0,
            "exact_failures": 0,
            "accepted": 0,
        }
        self.proposal_records: list[dict[str, Any]] = []

    def step(
        self,
        current_params: Sequence[float],
        current_exact: float,
        proposed_params: Sequence[float],
        *,
        log_q_forward: float = 0.0,
        log_q_reverse: float = 0.0,
    ) -> DelayedAcceptanceOutcome:
        """Evaluate one proposal with a surrogate screen and exact
        correction.
        """

        current = _as_float_vector(current_params, self.surrogate.dimension)
        proposed = _as_float_vector(proposed_params, self.surrogate.dimension)
        self.counters["proposals"] += 1
        current_surrogate = self.surrogate.predict(current)
        proposed_surrogate = self.surrogate.predict(proposed)
        self.counters["surrogate_predictions"] += 2
        log_q_ratio = float(log_q_reverse) - float(log_q_forward)
        if (
            current_surrogate.exact_required
            or proposed_surrogate.exact_required
        ):
            self.counters["support_fallbacks"] += 1
            return self._exact_stage(
                current,
                float(current_exact),
                proposed,
                proposed_surrogate,
                log_q_ratio,
                stage="exact_fallback",
            )

        surrogate_ratio = (
            float(proposed_surrogate.prediction)
            - float(current_surrogate.prediction)
            + log_q_ratio
        )
        stage_one_probability = _acceptance_probability(surrogate_ratio)
        if not self._accept(stage_one_probability):
            self.counters["stage_one_rejections"] += 1
            outcome = DelayedAcceptanceOutcome(
                params=tuple(float(value) for value in current),
                exact_log_probability=float(current_exact),
                accepted=False,
                stage="surrogate_screen",
                exact_called=False,
                acceptance_probability=stage_one_probability,
                correction_log_ratio=0.0,
                surrogate=proposed_surrogate,
            )
            self.proposal_records.append(outcome.to_dict())
            return outcome

        self.counters["exact_corrections"] += 1
        return self._exact_stage(
            current,
            float(current_exact),
            proposed,
            proposed_surrogate,
            log_q_ratio - surrogate_ratio,
            stage="exact_correction",
            correction_base=surrogate_ratio,
        )

    def _exact_stage(
        self,
        current: numpy.ndarray,
        current_exact: float,
        proposed: numpy.ndarray,
        proposed_surrogate: SurrogateResult,
        log_ratio: float,
        *,
        stage: str,
        correction_base: float = 0.0,
    ) -> DelayedAcceptanceOutcome:
        """Run exact evaluation and either accept or reject the proposal."""

        del correction_base
        exact_called = True
        self.counters["exact_calls"] += 1
        try:
            proposed_exact = float(self.exact_log_probability(tuple(proposed)))
        # DEVCOV_ALLOW_BROAD_ONCE exact target failure boundary.
        except Exception as exc:  # noqa: BLE001
            self.counters["exact_failures"] += 1
            proposed_exact = float("nan")
            proposed_surrogate = SurrogateResult(
                prediction=proposed_surrogate.prediction,
                uncertainty=proposed_surrogate.uncertainty,
                support=proposed_surrogate.support,
                training_sample_ids=proposed_surrogate.training_sample_ids,
                domain_status="exact_failure:" + type(exc).__name__,
                exact_required=True,
                provenance=proposed_surrogate.provenance,
            )
        if math.isfinite(proposed_exact):
            self.surrogate.add_exact_sample(
                proposed,
                proposed_exact,
                sample_id=f"exact-{self.counters['exact_calls']:06d}",
            )
        if not math.isfinite(proposed_exact):
            accepted = False
            probability = 0.0
        else:
            exact_ratio = proposed_exact - current_exact + log_ratio
            probability = _acceptance_probability(exact_ratio)
            accepted = self._accept(probability)
        if accepted:
            self.counters["accepted"] += 1
        elif stage == "exact_correction":
            self.counters["exact_rejections"] += 1
        outcome = DelayedAcceptanceOutcome(
            params=tuple(
                float(value) for value in (proposed if accepted else current)
            ),
            exact_log_probability=(
                proposed_exact if accepted else float(current_exact)
            ),
            accepted=accepted,
            stage=stage,
            exact_called=exact_called,
            acceptance_probability=probability,
            correction_log_ratio=float(log_ratio),
            surrogate=proposed_surrogate,
        )
        self.proposal_records.append(outcome.to_dict())
        return outcome

    def _accept(self, probability: float) -> bool:
        """Draw one deterministic acceptance decision."""

        return probability >= 1.0 or float(self.rng.random()) < probability


def _acceptance_probability(log_ratio: float) -> float:
    """Return ``min(1, exp(log_ratio))`` without numerical overflow."""

    if not math.isfinite(float(log_ratio)):
        return 1.0 if float(log_ratio) > 0 else 0.0
    if log_ratio >= 0.0:
        return 1.0
    return float(math.exp(log_ratio))


def run_delayed_acceptance_chain(
    initial_params: Sequence[float],
    exact_log_probability: Callable[[Sequence[float]], float],
    surrogate: DeterministicLocalSurrogate,
    *,
    n_steps: int,
    rng: numpy.random.Generator,
    proposal_scale: float = 0.05,
) -> dict[str, Any]:
    """Run one bounded random-walk chain for analytic validation fixtures."""

    current = _as_float_vector(initial_params, surrogate.dimension)
    exact_value = float(exact_log_probability(tuple(current)))
    if not math.isfinite(exact_value):
        raise ValueError("initial exact target value must be finite")
    surrogate.add_exact_sample(
        current, exact_value, sample_id="initial-000000"
    )
    controller = DelayedAcceptanceController(
        exact_log_probability,
        surrogate,
        rng=rng,
        proposal_scale=proposal_scale,
    )
    width = surrogate.width * float(proposal_scale)
    positions = numpy.empty(
        (max(int(n_steps), 0), surrogate.dimension),
        dtype=float,
    )
    log_probability = numpy.empty(max(int(n_steps), 0), dtype=float)
    for index in range(max(int(n_steps), 0)):
        proposal = numpy.clip(
            current + rng.normal(0.0, width, surrogate.dimension),
            surrogate.lower,
            surrogate.upper,
        )
        outcome = controller.step(current, exact_value, proposal)
        current = numpy.asarray(outcome.params, dtype=float)
        exact_value = float(outcome.exact_log_probability)
        positions[index] = current
        log_probability[index] = exact_value
    counters = dict(controller.counters)
    counters["exact_calls"] += 1
    counters["rejected"] = counters["proposals"] - counters["accepted"]
    return {
        "positions": positions,
        "log_probability": log_probability,
        "counters": counters,
        "proposal_records": list(controller.proposal_records),
        "cache_identity": surrogate.cache_identity,
        "training_sample_ids": list(surrogate.training_sample_ids),
    }


__all__ = [
    "DelayedAcceptanceController",
    "DelayedAcceptanceOutcome",
    "DeterministicLocalSurrogate",
    "SurrogateResult",
    "run_delayed_acceptance_chain",
    "validate_delayed_acceptance_config",
]
