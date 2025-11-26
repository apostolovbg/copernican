"""Parameter prior definitions and helper utilities.


This module centralises the construction of parameter priors so every
engine evaluates them consistently.  Historically the project stored raw
dictionary structures in the sanitised YAML cache and reimplemented the
log-density calculations in multiple places.  The new helpers collapse
that logic into a single location, provide descriptive validation errors
when users supply incomplete metadata and expose transform callables so
Jacobian corrections remain synchronised with the prior configuration.

In addition to probabilistic distributions the helper also understands a
``fixed`` prior.  Fixed priors represent parameters whose lower and upper
bounds coincide and therefore become deterministic inputs.  Treating
them explicitly keeps manifests self-documenting and allows engines to
expose the constant values to downstream utilities without special-case
code.

Each prior exposes three pieces of information:

``log_density``
    Evaluates the natural logarithm of the probability density at a given
    parameter value.  Invalid regions return ``-inf`` so sampling code can
    reject proposals deterministically.

``to_mapping``
    Returns a canonical dictionary representation written back to the
    sanitised YAML cache.  This keeps manifests and downstream tooling
    human-readable while ensuring equivalent priors serialise identically.

``create_transform``
    Provides an optional callable that yields a ``(value, log_jacobian)``
    tuple for :func:`copernican_lib.posterior.make_logposterior`. Uniform and
    Gaussian priors default to an identity transform, whereas log-uniform
    priors apply the standard ``log`` reparameterisation to maintain proper
    normalisation in log-space. Transform implementations are now realised as
    module-level classes so multiprocessing pools can pickle them safely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, MutableMapping, Optional


class PriorError(ValueError):
    """Raised when a prior definition is incomplete or inconsistent."""


TransformCallable = Callable[[float], tuple[float, float]]


@dataclass(slots=True)
class LogUniformTransform:
    """Picklable helper implementing the log-uniform Jacobian term."""

    def __call__(self, value: float) -> tuple[float, float]:
        """Return the value and Jacobian because log priors rescale density."""

        value = float(value)
        if value <= 0.0:
            raise ValueError(
                "Log-uniform prior expects strictly positive values"
            )
        return value, -math.log(value)


def _ensure_number(value: object, field: str) -> float:
    """Return ``value`` as a finite float or raise :class:`PriorError`.

    Parameters
    ----------
    value : object
        Incoming YAML field.
    field : str
        Human-readable field name for error reporting.
    """

    if value is None:
        raise PriorError(f"Prior field '{field}' is required")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise PriorError(f"Prior field '{field}' must be a number") from exc
    if not math.isfinite(number):
        raise PriorError(f"Prior field '{field}' must be finite")
    return number


@dataclass(slots=True)
class BasePrior:
    """Common behaviour shared by all prior implementations."""

    kind: str

    def log_density(self, value: float) -> float:
        """Return log p(value) or ``-inf`` when outside the support."""

        raise NotImplementedError

    def to_mapping(self) -> dict[str, float | str]:
        """Return a serialisable representation suitable for YAML."""

        raise NotImplementedError

    def create_transform(self) -> Optional[TransformCallable]:
        """Return an optional parameter transform for log-posterior use."""

        return None


@dataclass(slots=True)
class UniformPrior(BasePrior):
    """Continuous uniform distribution on ``[lower, upper]``."""

    lower: float
    upper: float
    _log_width: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate bounds so uniform priors cannot hide inverted intervals."""

        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise PriorError("Uniform priors require finite lower and upper")
        if self.upper <= self.lower:
            raise PriorError("Uniform prior requires upper > lower")
        self._log_width = -math.log(self.upper - self.lower)

    def log_density(self, value: float) -> float:
        """Return a constant log-density or ``-inf`` outside the support."""

        if value < self.lower or value > self.upper:
            return float("-inf")
        return self._log_width

    def to_mapping(self) -> dict[str, float | str]:
        """Serialise the prior so manifests record its exact bounds."""

        return {
            "type": "uniform",
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(slots=True)
class NormalPrior(BasePrior):
    """Gaussian distribution with mean ``mean`` and standard deviation."""

    mean: float
    sigma: float
    _norm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute the normalisation to avoid recomputing per evaluation."""

        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise PriorError("Gaussian prior requires finite sigma > 0")
        self._norm = -math.log(self.sigma * math.sqrt(2.0 * math.pi))

    def log_density(self, value: float) -> float:
        """Return log p(value) for the Gaussian prior."""

        delta = (value - self.mean) / self.sigma
        return self._norm - 0.5 * delta * delta

    def to_mapping(self) -> dict[str, float | str]:
        """Record the mean and sigma so cached YAML mirrors runtime use."""

        return {
            "type": "gaussian",
            "mean": self.mean,
            "sigma": self.sigma,
        }


@dataclass(slots=True)
class LogUniformPrior(BasePrior):
    """Log-uniform distribution across a positive interval.

    The distribution is uniform in ``log(value)`` which implies a ``1/x``
    density over ``[lower, upper]``.  The transform returned by
    :meth:`create_transform` contributes the ``-log(value)`` Jacobian term so
    samplers operating in linear space remain correctly normalised.
    """

    lower: float
    upper: float
    _log_interval: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate positive bounds because the log transform needs them."""

        if self.lower <= 0 or self.upper <= 0:
            raise PriorError("Log-uniform priors require positive bounds")
        if self.upper <= self.lower:
            raise PriorError("Log-uniform prior requires upper > lower")
        log_interval = math.log(self.upper) - math.log(self.lower)
        if log_interval <= 0:
            raise PriorError("Invalid log-uniform interval")
        self._log_interval = -math.log(log_interval)

    def log_density(self, value: float) -> float:
        """Return the log-density while honouring the ``1/x`` behaviour."""

        if value <= 0:
            return float("-inf")
        if value < self.lower or value > self.upper:
            return float("-inf")
        return self._log_interval - math.log(value)

    def to_mapping(self) -> dict[str, float | str]:
        """Serialise bounds and transform so Jacobians remain documented."""

        return {
            "type": "loguniform",
            "lower": self.lower,
            "upper": self.upper,
            "transform": "log",
        }

    def create_transform(self) -> TransformCallable:
        """Expose log-uniform transform so posterior maths stay explicit."""

        return LogUniformTransform()


@dataclass(slots=True)
class FixedPrior(BasePrior):
    """Degenerate prior fixing a parameter to a single value.

    Sampling engines treat this prior as a hard constraint rather than a
    probability density.  The log-density is ``0`` when evaluated at the
    prescribed value and ``-inf`` elsewhere so optimisers reject any
    deviation immediately.  A tight absolute/relative tolerance avoids
    floating-point round-off from falsely rejecting valid inputs while still
    behaving deterministically.
    """

    value: float
    _abs_tol: float = field(default=1e-12, init=False, repr=False)
    _rel_tol: float = field(default=1e-12, init=False, repr=False)

    def __post_init__(self) -> None:
        """Ensure the fixed value is finite so errors surface early."""

        if not math.isfinite(self.value):
            raise PriorError("Fixed priors require a finite value")

    def log_density(self, value: float) -> float:
        """Return zero at the fixed point and ``-inf`` elsewhere."""

        if math.isclose(
            value, self.value, rel_tol=self._rel_tol, abs_tol=self._abs_tol
        ):
            return 0.0
        return float("-inf")

    def to_mapping(self) -> dict[str, float | str]:
        """Serialise the deterministic value for manifests and caching."""

        return {"type": "fixed", "value": self.value}


PRIOR_TYPES = {
    "uniform": UniformPrior,
    "gaussian": NormalPrior,
    "normal": NormalPrior,
    "loguniform": LogUniformPrior,
    "log-uniform": LogUniformPrior,
    "fixed": FixedPrior,
}


def prior_from_mapping(
    mapping: Mapping[str, object] | None,
) -> Optional[BasePrior]:
    """Return a prior instance built from ``mapping``.

    ``None`` or empty dictionaries return ``None`` so callers can treat
    missing priors as uninformative.  The function validates required fields
    and raises :class:`PriorError` for incomplete specifications.
    """

    if not mapping:
        return None
    if not isinstance(mapping, Mapping):
        raise PriorError("Prior definitions must be mappings")
    prior_type = mapping.get("type")
    if not prior_type:
        raise PriorError("Prior definition missing 'type'")
    prior_type = str(prior_type).lower()
    if prior_type not in PRIOR_TYPES:
        raise PriorError(f"Unsupported prior type '{prior_type}'")
    cls = PRIOR_TYPES[prior_type]
    if cls is UniformPrior:
        lower = _ensure_number(mapping.get("lower"), "lower")
        upper = _ensure_number(mapping.get("upper"), "upper")
        return UniformPrior("uniform", lower, upper)
    if cls is NormalPrior:
        mean = _ensure_number(mapping.get("mean"), "mean")
        sigma = _ensure_number(mapping.get("sigma"), "sigma")
        return NormalPrior("gaussian", mean, sigma)
    if cls is LogUniformPrior:
        lower = _ensure_number(mapping.get("lower"), "lower")
        upper = _ensure_number(mapping.get("upper"), "upper")
        return LogUniformPrior("loguniform", lower, upper)
    if cls is FixedPrior:
        value = _ensure_number(mapping.get("value"), "value")
        return FixedPrior("fixed", value)
    raise PriorError(f"Unsupported prior implementation for '{prior_type}'")


def normalise_prior_mapping(mapping: MutableMapping[str, object]) -> None:
    """In-place normalisation of ``mapping`` using canonical keys.

    The caller typically supplies a dictionary loaded from YAML.  The helper
    replaces the content with the canonical output produced by
    :meth:`BasePrior.to_mapping`.  Providing a dedicated function keeps the
    parser agnostic to individual prior subclasses.
    """

    prior = prior_from_mapping(mapping)
    if prior is None:
        mapping.clear()
        return
    canonical = prior.to_mapping()
    mapping.clear()
    mapping.update(canonical)


def transform_from_mapping(
    mapping: Mapping[str, object],
) -> Optional[TransformCallable]:
    """Return the transform declared in ``mapping`` when supported."""

    prior = prior_from_mapping(mapping)
    if prior is None:
        return None
    return prior.create_transform()


__all__ = [
    "BasePrior",
    "FixedPrior",
    "NormalPrior",
    "LogUniformPrior",
    "PriorError",
    "UniformPrior",
    "normalise_prior_mapping",
    "prior_from_mapping",
    "transform_from_mapping",
]
