# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Posterior assembly utilities shared across sampler implementations.

These helpers stay picklable so multiprocessing pools that use the ``spawn``
start method can execute them without closure-related failures. The module
exposes the :class:`PosteriorEvaluator` class and the ``make_logposterior``
factory used by :mod:`copernican.lib.model_adapter` and the sampler backends
that wrap it.
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

    def _prepare(
        self, params: Sequence[float]
    ) -> tuple[tuple[float, ...], float] | None:
        """Transform one point and return values plus its log prior."""

        try:
            raw_values = tuple(float(param_val) for param_val in params)
        except (TypeError, ValueError):
            self.logger.debug(
                "(PosteriorEvaluator): received non-numeric params"
            )
            return None

        transformed: list[float] = []
        log_jacobian = 0.0
        transforms = self.transforms or ()

        for idx, raw_param in enumerate(raw_values):
            transform = transforms[idx] if idx < len(transforms) else None
            if transform is None:
                transformed.append(raw_param)
                continue
            try:
                result = transform(raw_param)
            except (
                ArithmeticError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ) as exc:  # pragma: no cover - defensive guard
                self.logger.debug(
                    "(PosteriorEvaluator): transform %d failed: %s", idx, exc
                )
                return None
            if isinstance(result, tuple):
                if len(result) != 2:
                    self.logger.debug(
                        "(PosteriorEvaluator): transform %d returned %s",
                        idx,
                        result,
                    )
                    return None
                new_val, jac = result
            else:
                new_val, jac = result, 0.0
            try:
                transformed.append(float(new_val))
                log_jacobian += float(jac)
            except (TypeError, ValueError):
                self.logger.debug(
                    "(PosteriorEvaluator): transform %d produced non-float",
                    idx,
                )
                return None

        bounds = self.bounds
        if bounds is not None:
            for idx, transformed_value in enumerate(transformed):
                try:
                    low_val, high_val = bounds[idx]
                except IndexError:
                    low_val = high_val = None
                if low_val is not None and transformed_value < low_val:
                    return None
                if high_val is not None and transformed_value > high_val:
                    return None

        log_prior = log_jacobian
        for transformed_value, prior in zip_longest(
            transformed, self.priors, fillvalue=None
        ):
            if prior is None:
                continue
            density = prior.log_density(transformed_value)
            if not math.isfinite(density):
                return None
            log_prior += density
        return tuple(transformed), float(log_prior)

    def __call__(self, params: Sequence[float]) -> float:
        """Transform parameters and evaluate the log posterior."""
        prepared = self._prepare(params)
        if prepared is None:
            return float("-inf")
        transformed, log_prior = prepared

        like_value = self.like(transformed)
        if not math.isfinite(like_value):
            return float("-inf")
        return float(like_value + log_prior)

    def evaluate_batch(
        self, params_batch: Sequence[Sequence[float]]
    ) -> tuple[float, ...]:
        """Evaluate an ordered batch with exact prior and bound handling."""

        prepared: list[tuple[tuple[float, ...], float] | None] = [
            self._prepare(params) for params in params_batch
        ]
        valid = [entry for entry in prepared if entry is not None]
        evaluate_batch = getattr(self.like, "evaluate_batch", None)
        if callable(evaluate_batch) and valid:
            like_values = iter(evaluate_batch([entry[0] for entry in valid]))
        else:
            like_values = iter(self.like(entry[0]) for entry in valid)

        values: list[float] = []
        for entry in prepared:
            if entry is None:
                values.append(float("-inf"))
                continue
            like_value = float(next(like_values))
            if not math.isfinite(like_value):
                values.append(float("-inf"))
            else:
                values.append(float(like_value + entry[1]))
        return tuple(values)


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
                logger.warning(
                    "(make_logposterior): invalid prior skipped: %s", exc
                )
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
