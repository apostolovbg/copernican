"""Posterior assembly utilities shared across engine implementations.

**Last Updated:** 2025-10-31

The posterior helper formerly lived in :mod:`copernican_lib.engine_interface`
where it relied on nested closures. Multiprocessing pools that use the
``spawn`` start method must pickle the callable they execute, yet closures
defined inside ``make_logposterior`` produced ``AttributeError: Can't pickle
local object`` whenever engines requested worker pools. This module extracts
the evaluation logic into the picklable :class:`PosteriorEvaluator` class and
keeps the normalisation and validation steps readable for future backends.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import priors as prior_lib


@dataclass(slots=True)
class PosteriorEvaluator:
    """Callable that evaluates priors, bounds and the wrapped likelihood."""

    like: Callable[[Sequence[float]], float]
    priors: tuple[prior_lib.BasePrior | None, ...]
    bounds: tuple[tuple[float | None, float | None], ...] | None
    transforms: tuple[Callable[[float], Any] | None, ...] | None
    logger: logging.Logger

    def __call__(self, params: Sequence[float]) -> float:
        try:
            raw_values = tuple(float(val) for val in params)
        except (TypeError, ValueError):
            self.logger.debug("(PosteriorEvaluator): received non-numeric params")
            return float("-inf")

        transformed: list[float] = []
        log_jacobian = 0.0
        transforms = self.transforms or ()

        for idx, value in enumerate(raw_values):
            transform = transforms[idx] if idx < len(transforms) else None
            if transform is None:
                transformed.append(value)
                continue
            try:
                result = transform(value)
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.debug(
                    "(PosteriorEvaluator): transform %d failed: %s", idx, exc
                )
                return float("-inf")
            if isinstance(result, tuple):
                if len(result) != 2:
                    self.logger.debug(
                        "(PosteriorEvaluator): transform %d returned %s", idx, result
                    )
                    return float("-inf")
                new_val, jac = result
            else:
                new_val, jac = result, 0.0
            try:
                transformed.append(float(new_val))
                log_jacobian += float(jac)
            except (TypeError, ValueError):
                self.logger.debug(
                    "(PosteriorEvaluator): transform %d produced non-float", idx
                )
                return float("-inf")

        bounds = self.bounds
        if bounds is not None:
            for idx, value in enumerate(transformed):
                try:
                    low_val, high_val = bounds[idx]
                except IndexError:
                    low_val = high_val = None
                if low_val is not None and value < low_val:
                    return float("-inf")
                if high_val is not None and value > high_val:
                    return float("-inf")

        log_prior = log_jacobian
        for value, prior in zip_longest(transformed, self.priors, fillvalue=None):
            if prior is None:
                continue
            density = prior.log_density(value)
            if not math.isfinite(density):
                return float("-inf")
            log_prior += density

        like_value = self.like(transformed)
        if not math.isfinite(like_value):
            return float("-inf")
        return float(like_value + log_prior)


def make_logposterior(
    like: Callable[[Sequence[float]], float],
    priors: Iterable[prior_lib.BasePrior | Mapping[str, Any]] | None,
) -> PosteriorEvaluator:
    """Return a :class:`PosteriorEvaluator` for ``like`` and ``priors``."""

    logger = logging.getLogger()
    prior_objects: list[prior_lib.BasePrior | None] = []
    for entry in priors or []:
        if isinstance(entry, prior_lib.BasePrior):
            prior_objects.append(entry)
            continue
        if isinstance(entry, Mapping):
            try:
                prior_objects.append(prior_lib.prior_from_mapping(entry))
            except prior_lib.PriorError as exc:
                logger.warning("(make_logposterior): invalid prior skipped: %s", exc)
                prior_objects.append(None)
            continue
        logger.warning(
            "(make_logposterior): unsupported prior entry %r skipped", entry
        )
        prior_objects.append(None)

    bounds_attr = getattr(like, "parameter_bounds", None)
    bounds: tuple[tuple[float | None, float | None], ...] | None = None
    if bounds_attr is not None:
        bounds = tuple(
            (
                None if low is None else float(low),
                None if high is None else float(high),
            )
            for low, high in bounds_attr
        )

    transforms_attr = getattr(like, "parameter_transforms", None)
    transforms: tuple[Callable[[float], Any] | None, ...] | None = None
    if transforms_attr is not None:
        transforms = tuple(transforms_attr)

    return PosteriorEvaluator(
        like=like,
        priors=tuple(prior_objects),
        bounds=bounds,
        transforms=transforms,
        logger=logger,
    )


__all__ = ["PosteriorEvaluator", "make_logposterior"]
