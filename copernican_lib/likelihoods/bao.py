"""Baryon Acoustic Oscillation likelihood helper.

**Last Updated:** 2025-02-14

Reimplements the χ² evaluation previously exposed via
``copernican_lib.statistics.chi_squared_bao`` while preserving support for
full covariance matrices and diagonal fallbacks.  Engines can toggle BAO data
on or off without branching through the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ._protocol import LikelihoodProtocol, LikelihoodState


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

    def loglike(self, params: Sequence[float]) -> float:
        """Return the BAO log-likelihood for ``params``."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        if self.z is None or len(self.z) == 0:
            logger.error("(bao_like): BAO redshift array is empty.")
            self._state = LikelihoodState()
            return float("-inf")

        try:
            get_DM = getattr(self.model_plugin, "get_comoving_distance_Mpc")
            get_Hz = getattr(self.model_plugin, "get_Hz_per_Mpc")
            get_DV = getattr(self.model_plugin, "get_DV_Mpc", None)
            get_rs = getattr(self.model_plugin, "get_sound_horizon_rs_Mpc")
            C_LIGHT = self.model_plugin.FIXED_PARAMS.get(
                "C_LIGHT_KM_S",
                299792.458,
            )
        except AttributeError as exc:
            logger.error("(bao_like): Model plugin missing BAO API: %s", exc)
            self._state = LikelihoodState()
            return float("-inf")

        rs_mpc = self.rs_override
        if rs_mpc is None:
            try:
                rs_mpc = get_rs(*params)
            except Exception:
                self._state = LikelihoodState()
                return float("-inf")

        if not (np.isfinite(rs_mpc) and rs_mpc > 0):
            self._state = LikelihoodState()
            return float("-inf")

        pred = np.full_like(self.obs_val, np.nan, dtype=float)

        mask = self.obs_type == "DM_over_rs"
        if np.any(mask):
            pred[mask] = get_DM(self.z[mask], *params) / rs_mpc

        mask = self.obs_type == "DH_over_rs"
        if np.any(mask):
            hz = get_Hz(self.z[mask], *params)
            dh = np.where(
                np.isfinite(hz) & (np.abs(hz) > 1e-9),
                C_LIGHT / hz,
                np.nan,
            )
            pred[mask] = dh / rs_mpc

        mask = self.obs_type == "DV_over_rs"
        if np.any(mask):
            if get_DV is not None:
                dv = get_DV(self.z[mask], *params)
            else:
                dm_val = get_DM(self.z[mask], *params)
                hz_val = get_Hz(self.z[mask], *params)
                term = dm_val**2 * C_LIGHT * self.z[mask] / hz_val
                valid = (
                    np.isfinite(dm_val)
                    & (dm_val >= 0)
                    & np.isfinite(hz_val)
                    & (np.abs(hz_val) > 1e-9)
                    & (self.z[mask] > 1e-9)
                )
                dv = np.full_like(dm_val, np.nan)
                dv[valid] = np.where(
                    term[valid] >= 0,
                    term[valid] ** (1.0 / 3.0),
                    np.nan,
                )
                dv[np.abs(self.z[mask]) < 1e-9] = 0.0
            pred[mask] = dv / rs_mpc

        if np.all(~np.isfinite(pred)):
            logger.warning(
                "(bao_like): Model returned no finite BAO predictions."
            )
            self._state = LikelihoodState()
            return float("-inf")

        resid = self.obs_val - pred
        if np.any(~np.isfinite(resid)):
            logger.warning("(bao_like): Non-finite residuals in BAO data.")
            self._state = LikelihoodState()
            return float("-inf")

        cov_inv = self.covariance_matrix_inv
        metadata: dict[str, Any] = {"points": int(resid.size)}
        chi2 = float("inf")
        if cov_inv is not None:
            try:
                if cov_inv.shape[0] != resid.size:
                    raise ValueError("Covariance size mismatch")
                chi2 = float(resid @ cov_inv @ resid)
                metadata["covariance"] = "full"
            except Exception as exc:
                logger.warning(
                    "(bao_like): Falling back to diagonal covariance: %s",
                    exc,
                )
                cov_inv = None

        if cov_inv is None:
            valid = np.isfinite(self.obs_err) & (self.obs_err > 1e-9)
            if not np.any(valid):
                logger.warning("(bao_like): No valid BAO errors available.")
                self._state = LikelihoodState()
                return float("-inf")
            chi2 = float(np.sum((resid[valid] / self.obs_err[valid]) ** 2))
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
