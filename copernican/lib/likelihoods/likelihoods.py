"""Shared likelihood contracts and the joint likelihood helper.

This module defines the common protocol and mutable state container used by
all likelihood helpers together with :class:`JointLike`, the aggregator that
combines enabled dataset likelihoods into one total log-likelihood.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol, Sequence


class LikelihoodProtocol(Protocol):
    """Runtime contract implemented by all likelihood helpers."""

    enabled: bool

    def loglike(self, params: Sequence[float]) -> float:
        """Return the natural logarithm of the likelihood for ``params``."""

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostic information captured during the last call."""


@dataclass(slots=True)
class LikelihoodState:
    """Mutable container storing likelihood diagnostics."""

    chi2: float = float("inf")
    loglike: float = float("-inf")
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> Mapping[str, Any]:
        """Return an immutable representation of the stored state."""

        return {
            "chi2": self.chi2,
            "loglike": self.loglike,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class JointLike(LikelihoodProtocol):
    """Sum log-likelihoods from enabled components."""

    components: Mapping[str, LikelihoodProtocol]
    config: Mapping[str, bool] | None = None
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )

    def __post_init__(self) -> None:
        """Apply configuration toggles to component enabled flags."""

        toggles = {
            key: bool(value) for key, value in (self.config or {}).items()
        }
        for name, component in self.components.items():
            if name in toggles:
                component.enabled = component.enabled and toggles[name]
        self._toggles = toggles

    def prepare_worker_runtime(self) -> None:
        """Prepare enabled process-local likelihood runtime assets once."""

        for name, component in self.components.items():
            if not getattr(component, "enabled", True):
                continue
            if name in self._toggles and not self._toggles[name]:
                continue
            prepare = getattr(component, "prepare_worker_runtime", None)
            if callable(prepare):
                prepare()

    def loglike(self, params: Sequence[float]) -> float:
        """Return the total log-likelihood across all enabled components."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        total_loglike = 0.0
        total_chi2 = 0.0
        combined: MutableMapping[str, Any] = {}
        enabled_components: list[str] = []

        for name, component in self.components.items():
            use_component = getattr(component, "enabled", True)
            if name in getattr(self, "_toggles", {}):
                use_component = use_component and self._toggles[name]
                component.enabled = use_component

            if not use_component:
                combined[name] = {
                    "chi2": 0.0,
                    "loglike": 0.0,
                    "metadata": {"enabled": False},
                }
                continue

            loglike_val = component.loglike(params)
            state = component.state
            combined[name] = state | {
                "metadata": dict(state.get("metadata", {}))
            }
            combined[name]["metadata"]["enabled"] = True
            enabled_components.append(name)

            if not math.isfinite(loglike_val):
                total_loglike = float("-inf")
                total_chi2 = float("inf")
                logger.debug(
                    "(joint_like): Component %s returned non-finite logL.",
                    name,
                )
                break

            total_loglike += loglike_val
            total_chi2 += float(state.get("chi2", float("inf")))

        metadata = {
            "components": combined,
            "enabled_components": tuple(enabled_components),
        }
        self._state = LikelihoodState(
            chi2=total_chi2,
            loglike=total_loglike,
            metadata=metadata,
        )
        return total_loglike

    def loglike_batch(
        self, params_batch: Sequence[Sequence[float]]
    ) -> tuple[float, ...]:
        """Evaluate an ordered batch while sharing batch-capable components."""

        batch_items = list(params_batch)
        if not batch_items:
            return ()
        cmb = self.components.get("cmb")
        cmb_batch = getattr(cmb, "loglike_batch", None)
        if not callable(cmb_batch) or not getattr(cmb, "enabled", True):
            return tuple(self.loglike(params) for params in batch_items)

        cmb_values = tuple(float(value) for value in cmb_batch(batch_items))
        totals = [0.0] * len(batch_items)
        for index, params in enumerate(batch_items):
            total = cmb_values[index]
            if not math.isfinite(total):
                totals[index] = float("-inf")
                continue
            for name, component in self.components.items():
                if name == "cmb":
                    continue
                use_component = getattr(component, "enabled", True)
                if name in getattr(self, "_toggles", {}):
                    use_component = use_component and self._toggles[name]
                    component.enabled = use_component
                if use_component:
                    value = float(component.loglike(params))
                    if not math.isfinite(value):
                        total = float("-inf")
                        break
                    total += value
            totals[index] = total
        return tuple(totals)

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostics captured during the last evaluation."""

        return self._state.as_mapping()


__all__ = ["JointLike", "LikelihoodProtocol", "LikelihoodState"]
