"""Supernova Ia likelihood helper.

**Last Updated:** 2025-10-30

Wraps the covariance-aware χ² evaluation previously implemented in
``copernican_lib.statistics`` so engines can reuse it without duplicating
validation logic.  The helper exposes a :meth:`loglike` method returning the
natural logarithm of the likelihood and records χ² diagnostics in
:pyattr:`state`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import LikelihoodProtocol, LikelihoodState


@dataclass(slots=True)
class SNeLike(LikelihoodProtocol):
    """Evaluate the Supernova Ia log-likelihood for a given dataset."""

    mu_model: Callable[[np.ndarray, *Sequence[float]], np.ndarray]
    data: pd.DataFrame
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )

    def loglike(self, params: Sequence[float]) -> float:
        """Return the log-likelihood for ``params`` with the stored data."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        df = self.data
        required = ("zcmb", "mu_obs")
        if not all(col in df.columns for col in required):
            logger.error(
                "SNe DataFrame missing required columns %s",
                required,
            )
            self._state = LikelihoodState()
            return float("-inf")

        z_data = df["zcmb"].to_numpy(dtype=float)
        mu_obs = df["mu_obs"].to_numpy(dtype=float)
        if np.any(~np.isfinite(z_data)) or np.any(~np.isfinite(mu_obs)):
            logger.error(
                "SNe data contains non-finite zcmb or mu_obs values"
            )
            self._state = LikelihoodState()
            return float("-inf")

        try:
            mu_model = self.mu_model(z_data, *params)
        except Exception:  # pragma: no cover - logged upstream
            self._state = LikelihoodState()
            return float("-inf")

        if not isinstance(mu_model, np.ndarray) or (
            mu_model.shape != mu_obs.shape
        ):
            self._state = LikelihoodState()
            return float("-inf")

        if np.any(~np.isfinite(mu_model)):
            self._state = LikelihoodState()
            return float("-inf")

        resid = mu_obs - mu_model
        cov_inv = df.attrs.get("covariance_matrix_inv")
        metadata: dict[str, Any] = {}

        chi2 = float("inf")
        if cov_inv is not None:
            try:
                if cov_inv.shape[0] != len(resid):
                    raise ValueError("Covariance mismatch for SNe data")
                chi2 = float(resid @ cov_inv @ resid)
                metadata["covariance"] = "full"
            except Exception as exc:
                logger.warning(
                    "Falling back to diagonal covariance due to issue: %s",
                    exc,
                )
                cov_inv = None

        if cov_inv is None:
            if "e_mu_obs" not in df.columns:
                logger.error("No diagonal errors available for SNe data.")
                self._state = LikelihoodState()
                return float("-inf")
            err = df["e_mu_obs"].to_numpy(dtype=float)
            err = np.where(~np.isfinite(err) | (err <= 0), 1e-12, err)
            chi2 = float(np.sum((resid / err) ** 2))
            metadata["covariance"] = "diagonal"

        loglike = -0.5 * chi2 if np.isfinite(chi2) else float("-inf")
        self._state = LikelihoodState(
            chi2=chi2,
            loglike=loglike,
            metadata=metadata,
        )
        return loglike

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostics captured during the last evaluation."""

        return self._state.as_mapping()


__all__ = ["SNeLike"]
