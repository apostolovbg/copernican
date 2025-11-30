"""Supernova Ia likelihood helper.

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

from ._protocol import LikelihoodProtocol, LikelihoodState

@dataclass(slots=True)
class SNeLike(LikelihoodProtocol):
    """Evaluate the Supernova Ia log-likelihood for a given dataset."""

    mu_model: Callable[..., np.ndarray]
    data: pd.DataFrame
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )
    _z_values: np.ndarray = field(init=False, repr=False)
    _mu_observed: np.ndarray = field(init=False, repr=False)
    _covariance_matrix_inv: np.ndarray | None = field(init=False, repr=False)
    _diag_errors: np.ndarray | None = field(init=False, repr=False)
    _residual_buffer: np.ndarray = field(init=False, repr=False)
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Cache immutable arrays so log-likelihood calls avoid copies."""

        df = self.data
        required = ("zcmb", "mu_obs")
        if not all(col in df.columns for col in required):
            missing = tuple(col for col in required if col not in df.columns)
            self._setup_error = "SNe DataFrame missing required columns %s" % (
                missing,
            )
            self._z_values = np.empty(0, dtype=float)
            self._mu_observed = np.empty(0, dtype=float)
            self._covariance_matrix_inv = None
            self._diag_errors = None
            self._residual_buffer = np.empty(0, dtype=float)
            return

        self._z_values = df["zcmb"].to_numpy(dtype=float, copy=True)
        self._mu_observed = df["mu_obs"].to_numpy(dtype=float, copy=True)
        if np.any(~np.isfinite(self._z_values)) or np.any(
            ~np.isfinite(self._mu_observed)
        ):
            self._setup_error = (
                "SNe data contains non-finite zcmb or mu_obs values"
            )

        cov_attr = df.attrs.get("covariance_matrix_inv")
        self._covariance_matrix_inv = None
        if cov_attr is not None:
            self._covariance_matrix_inv = np.asarray(cov_attr, dtype=float)

        self._diag_errors = None
        if "e_mu_obs" in df.columns:
            errs = df["e_mu_obs"].to_numpy(dtype=float, copy=True)
            errs = np.where(~np.isfinite(errs) | (errs <= 0), 1e-12, errs)
            self._diag_errors = errs

        self._residual_buffer = np.empty_like(self._mu_observed, dtype=float)

    def loglike(self, params: Sequence[float]) -> float:
        """Return the log-likelihood for ``params`` with the stored data."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        if self._setup_error is not None:
            logger.error(self._setup_error)
            self._state = LikelihoodState()
            return float("-inf")

        try:
            mu_model = self.mu_model(self._z_values, *params)
        except Exception:  # pragma: no cover - logged upstream
            self._state = LikelihoodState()
            return float("-inf")

        if not isinstance(mu_model, np.ndarray) or (
            mu_model.shape != self._mu_observed.shape
        ):
            self._state = LikelihoodState()
            return float("-inf")

        if np.any(~np.isfinite(mu_model)):
            self._state = LikelihoodState()
            return float("-inf")

        np.subtract(
            self._mu_observed,
            mu_model,
            out=self._residual_buffer,
            casting="unsafe",
        )
        if np.any(~np.isfinite(self._residual_buffer)):
            self._state = LikelihoodState()
            return float("-inf")

        cov_inv = self._covariance_matrix_inv
        metadata: dict[str, Any] = {}

        chi2 = float("inf")
        if cov_inv is not None:
            try:
                if cov_inv.shape[0] != self._residual_buffer.size:
                    raise ValueError("Covariance mismatch for SNe data")
                chi2 = float(
                    self._residual_buffer @ cov_inv @ self._residual_buffer
                )
                metadata["covariance"] = "full"
            except Exception as exc:
                logger.warning(
                    "Falling back to diagonal covariance due to issue: %s",
                    exc,
                )
                cov_inv = None

        if cov_inv is None:
            if self._diag_errors is None:
                logger.error("No diagonal errors available for SNe data.")
                self._state = LikelihoodState()
                return float("-inf")
            chi2 = float(
                np.sum((self._residual_buffer / self._diag_errors) ** 2)
            )
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
