"""Baryon Acoustic Oscillation likelihood helper.

**Last Updated:** 2025-11-01

Computes BAO observables using CAMB background distances aligned with the CMB
likelihood configuration.  Previous revisions mixed direct model integrals with
sound-horizon fallbacks, producing unphysical predictions for exotic models.
Tying the calculations to CAMB eliminates the fallback path and guarantees that
Stage 2 evaluates a self-consistent cosmology across SNe, BAO and CMB data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._protocol import LikelihoodProtocol, LikelihoodState
from .cmb import compute_camb_background_observables


@dataclass(slots=True)
class BAOLike(LikelihoodProtocol):
    """Evaluate BAO log-likelihoods for pre-extracted observables."""

    z: np.ndarray
    obs_type: np.ndarray
    obs_val: np.ndarray
    obs_err: np.ndarray
    model_plugin: Any
    covariance_matrix_inv: np.ndarray | None = None
    rs_override: float | None = None
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )
    _z_values: np.ndarray = field(init=False, repr=False)
    _obs_type: np.ndarray = field(init=False, repr=False)
    _observed: np.ndarray = field(init=False, repr=False)
    _errors: np.ndarray = field(init=False, repr=False)
    _cov_inv: np.ndarray | None = field(init=False, repr=False)
    _rs_override: float | None = field(init=False, repr=False)
    _mask_dm: np.ndarray = field(init=False, repr=False)
    _mask_dh: np.ndarray = field(init=False, repr=False)
    _mask_dv: np.ndarray = field(init=False, repr=False)
    _prediction_buffer: np.ndarray = field(init=False, repr=False)
    _residual_buffer: np.ndarray = field(init=False, repr=False)
    _get_camb_params: Callable[[Sequence[float]], Mapping[str, Any]] | None = field(
        init=False,
        repr=False,
    )
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalise arrays and cache model callables for fast evaluation."""

        self._z_values = np.asarray(self.z, dtype=float).copy()
        self._obs_type = np.asarray(self.obs_type, dtype=object).copy()
        self._observed = np.asarray(self.obs_val, dtype=float).copy()
        self._errors = np.asarray(self.obs_err, dtype=float).copy()
        self._cov_inv = (
            None
            if self.covariance_matrix_inv is None
            else np.asarray(self.covariance_matrix_inv, dtype=float)
        )
        self._rs_override = (
            None if self.rs_override is None else float(self.rs_override)
        )

        self._mask_dm = self._obs_type == "DM_over_rs"
        self._mask_dh = self._obs_type == "DH_over_rs"
        self._mask_dv = self._obs_type == "DV_over_rs"

        self._prediction_buffer = np.empty_like(self._observed, dtype=float)
        self._prediction_buffer.fill(np.nan)
        self._residual_buffer = np.empty_like(self._observed, dtype=float)

        self._get_camb_params = getattr(
            self.model_plugin, "get_camb_params", None
        )
        if self._get_camb_params is None:
            self._setup_error = (
                "(bao_like): Model plugin does not expose get_camb_params."
            )

        if self._z_values.size == 0:
            self._setup_error = "(bao_like): BAO redshift array is empty."

    def loglike(self, params: Sequence[float]) -> float:
        """Return the BAO log-likelihood for ``params``."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        if self._setup_error is not None:
            logger.error(self._setup_error)
            self._state = LikelihoodState()
            return float("-inf")

        if self._get_camb_params is None:
            self._state = LikelihoodState()
            return float("-inf")

        try:
            camb_params = self._get_camb_params(params)
        except Exception:
            self._state = LikelihoodState()
            return float("-inf")

        background = compute_camb_background_observables(
            camb_params,
            self._z_values,
        )

        rs_mpc = self._rs_override
        if rs_mpc is None:
            rs_mpc = float(background.get("rs_drag", float("nan")))

        if not (np.isfinite(rs_mpc) and rs_mpc > 0):
            self._state = LikelihoodState()
            return float("-inf")

        self._prediction_buffer.fill(np.nan)

        dm_vals = background.get("DM")
        dh_vals = background.get("DH")
        dv_vals = background.get("DV")

        if dm_vals is None or dh_vals is None or dv_vals is None:
            self._state = LikelihoodState()
            return float("-inf")

        if np.any(self._mask_dm):
            self._prediction_buffer[self._mask_dm] = (
                dm_vals[self._mask_dm] / rs_mpc
            )

        if np.any(self._mask_dh):
            self._prediction_buffer[self._mask_dh] = dh_vals[self._mask_dh] / rs_mpc

        if np.any(self._mask_dv):
            self._prediction_buffer[self._mask_dv] = dv_vals[self._mask_dv] / rs_mpc

        if np.all(~np.isfinite(self._prediction_buffer)):
            logger.warning(
                "(bao_like): Model returned no finite BAO predictions."
            )
            self._state = LikelihoodState()
            return float("-inf")

        np.subtract(
            self._observed,
            self._prediction_buffer,
            out=self._residual_buffer,
            casting="unsafe",
        )
        if np.any(~np.isfinite(self._residual_buffer)):
            logger.warning("(bao_like): Non-finite residuals in BAO data.")
            self._state = LikelihoodState()
            return float("-inf")

        cov_inv = self._cov_inv
        metadata: dict[str, Any] = {"points": int(self._observed.size)}
        chi2 = float("inf")
        if cov_inv is not None:
            try:
                if cov_inv.shape[0] != self._observed.size:
                    raise ValueError("Covariance size mismatch")
                chi2 = float(
                    self._residual_buffer @ cov_inv @ self._residual_buffer
                )
                metadata["covariance"] = "full"
            except Exception as exc:
                logger.warning(
                    "(bao_like): Falling back to diagonal covariance: %s",
                    exc,
                )
                cov_inv = None

        if cov_inv is None:
            valid = np.isfinite(self._errors) & (self._errors > 1e-9)
            if not np.any(valid):
                logger.warning("(bao_like): No valid BAO errors available.")
                self._state = LikelihoodState()
                return float("-inf")
            chi2 = float(
                np.sum(
                    (self._residual_buffer[valid] / self._errors[valid]) ** 2
                )
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


__all__ = ["BAOLike"]
