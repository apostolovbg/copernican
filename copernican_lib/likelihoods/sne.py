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

import numpy
import pandas

from ._protocol import LikelihoodProtocol, LikelihoodState


def compute_sne_intercept_delta(
    residuals: numpy.ndarray,
    *,
    covariance_matrix_inv: numpy.ndarray | None = None,
    diag_errors: numpy.ndarray | None = None,
) -> float:
    """Return the additive intercept that minimizes SNe residuals."""

    residual_vector = numpy.asarray(residuals, dtype=float)
    if residual_vector.ndim != 1:
        raise ValueError("SNe residual vector must be one-dimensional.")
    if not numpy.all(numpy.isfinite(residual_vector)):
        raise ValueError("SNe residual vector must be finite.")

    if covariance_matrix_inv is not None:
        cov_inv = numpy.asarray(covariance_matrix_inv, dtype=float)
        if cov_inv.ndim != 2 or cov_inv.shape[0] != cov_inv.shape[1]:
            raise ValueError("SNe covariance inverse must be square.")
        if cov_inv.shape[0] != residual_vector.size:
            raise ValueError("SNe covariance mismatch for intercept fit.")
        ones = numpy.ones(residual_vector.size, dtype=float)
        numerator = float(ones @ cov_inv @ residual_vector)
        denominator = float(ones @ cov_inv @ ones)
        if not numpy.isfinite(denominator) or denominator == 0.0:
            raise ValueError("SNe intercept denominator is invalid.")
        delta_mu = -numerator / denominator
        if not numpy.isfinite(delta_mu):
            raise ValueError("SNe intercept delta is invalid.")
        return float(delta_mu)

    if diag_errors is None:
        raise ValueError("SNe diagonal errors are required for intercept fit.")

    errors = numpy.asarray(diag_errors, dtype=float)
    if errors.ndim != 1 or errors.size != residual_vector.size:
        raise ValueError("SNe diagonal errors mismatch for intercept fit.")
    errors = numpy.where(
        ~numpy.isfinite(errors) | (errors <= 0), 1e-12, errors
    )
    weights = 1.0 / (errors**2)
    denominator = float(numpy.sum(weights))
    if not numpy.isfinite(denominator) or denominator == 0.0:
        raise ValueError("SNe intercept denominator is invalid.")
    delta_mu = -float(numpy.sum(weights * residual_vector)) / denominator
    if not numpy.isfinite(delta_mu):
        raise ValueError("SNe intercept delta is invalid.")
    return float(delta_mu)


@dataclass(slots=True)
class SNeLike(LikelihoodProtocol):
    """Evaluate the Supernova Ia log-likelihood for a given dataset."""

    mu_model: Callable[..., numpy.ndarray]
    observations: pandas.DataFrame
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )
    _z_values: numpy.ndarray = field(init=False, repr=False)
    _mu_observed: numpy.ndarray = field(init=False, repr=False)
    _covariance_matrix_inv: numpy.ndarray | None = field(
        init=False, repr=False
    )
    _diag_errors: numpy.ndarray | None = field(init=False, repr=False)
    _residual_buffer: numpy.ndarray = field(init=False, repr=False)
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Cache immutable arrays so log-likelihood calls avoid copies."""

        observations_df = self.observations
        required = ("zcmb", "mu_obs")
        if not all(col in observations_df.columns for col in required):
            missing = tuple(
                col for col in required if col not in observations_df.columns
            )
            self._setup_error = "SNe DataFrame missing required columns %s" % (
                missing,
            )
            self._z_values = numpy.empty(0, dtype=float)
            self._mu_observed = numpy.empty(0, dtype=float)
            self._covariance_matrix_inv = None
            self._diag_errors = None
            self._residual_buffer = numpy.empty(0, dtype=float)
            return

        self._z_values = observations_df["zcmb"].to_numpy(
            dtype=float, copy=True
        )
        self._mu_observed = observations_df["mu_obs"].to_numpy(
            dtype=float, copy=True
        )
        if numpy.any(~numpy.isfinite(self._z_values)) or numpy.any(
            ~numpy.isfinite(self._mu_observed)
        ):
            self._setup_error = (
                "SNe data contains non-finite zcmb or mu_obs values"
            )

        cov_attr = observations_df.attrs.get("covariance_matrix_inv")
        self._covariance_matrix_inv = None
        if cov_attr is not None:
            self._covariance_matrix_inv = numpy.asarray(cov_attr, dtype=float)

        self._diag_errors = None
        if "e_mu_obs" in observations_df.columns:
            errs = observations_df["e_mu_obs"].to_numpy(dtype=float, copy=True)
            errs = numpy.where(
                ~numpy.isfinite(errs) | (errs <= 0), 1e-12, errs
            )
            self._diag_errors = errs

        self._residual_buffer = numpy.empty_like(
            self._mu_observed, dtype=float
        )

    def loglike(self, params: Sequence[float]) -> float:
        """Return the log-likelihood for ``params`` with the stored data."""

        observations_df = self.observations
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
        except (RuntimeError, TypeError, ValueError, ZeroDivisionError):
            self._state = LikelihoodState()
            return float("-inf")

        if not isinstance(mu_model, numpy.ndarray) or (
            mu_model.shape != self._mu_observed.shape
        ):
            self._state = LikelihoodState()
            return float("-inf")

        if numpy.any(~numpy.isfinite(mu_model)):
            self._state = LikelihoodState()
            return float("-inf")

        numpy.subtract(
            self._mu_observed,
            mu_model,
            out=self._residual_buffer,
            casting="unsafe",
        )
        if numpy.any(~numpy.isfinite(self._residual_buffer)):
            self._state = LikelihoodState()
            return float("-inf")

        cov_inv = self._covariance_matrix_inv
        metadata: dict[str, Any] = {}
        use_full_covariance = False
        if cov_inv is not None:
            try:
                if cov_inv.shape[0] != self._residual_buffer.size:
                    raise ValueError("Covariance mismatch for SNe data")
                use_full_covariance = True
            except (
                numpy.linalg.LinAlgError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Falling back to diagonal covariance due to issue: %s",
                    exc,
                )
                cov_inv = None

        if bool(
            observations_df.attrs.get(
                "requires_sne_intercept_marginalization",
            )
        ):
            try:
                delta_mu = compute_sne_intercept_delta(
                    self._residual_buffer,
                    covariance_matrix_inv=(
                        cov_inv if use_full_covariance else None
                    ),
                    diag_errors=self._diag_errors,
                )
            except ValueError as exc:
                logger.error(
                    "Unable to marginalize SNe intercept: %s",
                    exc,
                )
                self._state = LikelihoodState()
                return float("-inf")
            numpy.add(
                self._residual_buffer,
                delta_mu,
                out=self._residual_buffer,
            )
            metadata["sne_intercept_marginalized"] = True
            metadata["sne_intercept_delta_mu"] = float(delta_mu)
            metadata["sne_intercept_name"] = str(
                observations_df.attrs.get("sne_intercept_name", "Delta_mu")
            )
        else:
            metadata["sne_intercept_marginalized"] = False

        chi2 = float("inf")
        if use_full_covariance and cov_inv is not None:
            try:
                chi2 = float(
                    self._residual_buffer @ cov_inv @ self._residual_buffer
                )
                metadata["covariance"] = "full"
            except (
                numpy.linalg.LinAlgError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Falling back to diagonal covariance due to issue: %s",
                    exc,
                )
                cov_inv = None
                use_full_covariance = False

        if not use_full_covariance or cov_inv is None:
            if self._diag_errors is None:
                logger.error("No diagonal errors available for SNe data.")
                self._state = LikelihoodState()
                return float("-inf")
            chi2 = float(
                numpy.sum((self._residual_buffer / self._diag_errors) ** 2)
            )
            metadata["covariance"] = "diagonal"

        loglike = -0.5 * chi2 if numpy.isfinite(chi2) else float("-inf")
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
