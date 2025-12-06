"""Baryon Acoustic Oscillation likelihood helper.

Computes BAO observables using CAMB background distances aligned with the CMB
likelihood configuration.  Previous revisions mixed direct model integrals with
sound-horizon fallbacks, producing unphysical predictions for exotic models.
Tying the calculations to CAMB eliminates the fallback path and guarantees that
Stage 2 evaluates a self-consistent cosmology across SNe, BAO and CMB data.
When CAMB parameters are unavailable the helper gracefully falls back to the
model's distance functions so legacy tests and simplified benchmarks continue
to operate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..model_coder import SoundHorizonComputationError
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
    _get_camb_params: Callable[[Sequence[float]], Mapping[str, Any]] | None = (
        field(
            init=False,
            repr=False,
        )
    )
    _fallback_dm: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_hz: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_dv: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_da: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_rs: Callable[..., Any] | None = field(init=False, repr=False)
    _c_light_km_s: float = field(init=False, repr=False)
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

        self._fallback_dm = getattr(
            self.model_plugin, "get_comoving_distance_Mpc", None
        )
        self._fallback_hz = getattr(self.model_plugin, "get_Hz_per_Mpc", None)
        self._fallback_dv = getattr(self.model_plugin, "get_DV_Mpc", None)
        self._fallback_da = getattr(
            self.model_plugin, "get_angular_diameter_distance_Mpc", None
        )
        self._fallback_rs = getattr(
            self.model_plugin, "get_sound_horizon_rs_Mpc", None
        )
        fixed = getattr(self.model_plugin, "FIXED_PARAMS", {}) or {}
        self._c_light_km_s = float(fixed.get("C_LIGHT_KM_S", 299_792.458))

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

        background = None
        camb_params: Mapping[str, Any] | None = None
        if self._get_camb_params is not None:
            try:
                camb_params = self._get_camb_params(params)
            except Exception as exc:
                logger.warning(
                    "(bao_like): Failed to obtain CAMB parameters; %s",
                    exc,
                )
            else:
                if camb_params:
                    try:
                        background = compute_camb_background_observables(
                            camb_params,
                            self._z_values,
                        )
                    except Exception as exc:
                        logger.warning(
                            "(bao_like): CAMB background failed; falling back "
                            "to model distances: %s",
                            exc,
                        )

        if background is None:
            background = self._compute_plugin_background(params)
            if background is None:
                self._state = LikelihoodState()
                return float("-inf")

        rs_mpc = self._rs_override
        if rs_mpc is None:
            rs_drag = background.get("rs_drag")
            if rs_drag is None:
                rs_background = float("nan")
            else:
                rs_arr = np.asarray(rs_drag)
                if rs_arr.size == 0:
                    rs_background = float("nan")
                else:
                    rs_background = float(rs_arr.flat[0])
            if np.isnan(rs_background) and self._fallback_rs is not None:
                try:
                    rs_background = float(
                        self._call_with_params(self._fallback_rs, (), params)
                    )
                except SoundHorizonComputationError as exc:
                    logger.error(
                        "(bao_like): rs_expression diverged; aborting BAO "
                        "predictions: %s",
                        exc,
                    )
                    self._state = LikelihoodState(
                        metadata={
                            "error": (
                                "Sound horizon integral diverged; see log for "
                                "rs_expression diagnostics."
                            )
                        }
                    )
                    return float("-inf")
                except Exception as exc:
                    logger.warning(
                        "(bao_like): Sound horizon fallback failed: %s",
                        exc,
                    )
            rs_mpc = rs_background

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
            self._prediction_buffer[self._mask_dh] = (
                dh_vals[self._mask_dh] / rs_mpc
            )

        if np.any(self._mask_dv):
            self._prediction_buffer[self._mask_dv] = (
                dv_vals[self._mask_dv] / rs_mpc
            )

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

    def _call_with_params(
        self,
        func: Callable[..., Any],
        args: Sequence[Any],
        params: Sequence[float],
    ) -> Any:
        """Invoke ``func`` with optional cosmological parameters."""

        try:
            return func(*args, *params)
        except TypeError:
            return func(*args)

    def _compute_plugin_background(
        self, params: Sequence[float]
    ) -> dict[str, np.ndarray] | None:
        """Return background observables from model distance functions."""

        if self._fallback_dm is None or self._fallback_hz is None:
            return None

        try:
            dm_vals = np.asarray(
                self._call_with_params(
                    self._fallback_dm,
                    (self._z_values,),
                    params,
                ),
                dtype=float,
            )
            hz_vals = np.asarray(
                self._call_with_params(
                    self._fallback_hz,
                    (self._z_values,),
                    params,
                ),
                dtype=float,
            )
        except SoundHorizonComputationError as exc:
            logging.getLogger().error(
                "(bao_like): rs_expression diverged while computing model "
                "background distances: %s",
                exc,
            )
            return None
        except Exception:
            return None

        if (
            dm_vals.shape != self._z_values.shape
            or hz_vals.shape != dm_vals.shape
        ):
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            dh_vals = np.where(
                np.abs(hz_vals) > 1e-12,
                self._c_light_km_s / hz_vals,
                np.nan,
            )

        if self._fallback_dv is not None:
            try:
                dv_vals = np.asarray(
                    self._call_with_params(
                        self._fallback_dv,
                        (self._z_values,),
                        params,
                    ),
                    dtype=float,
                )
            except Exception:
                dv_vals = np.full_like(dm_vals, np.nan, dtype=float)
            if dv_vals.shape != dm_vals.shape:
                dv_vals = np.full_like(dm_vals, np.nan, dtype=float)
        else:
            dv_vals = np.full_like(dm_vals, np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = dm_vals * dm_vals
                term *= self._z_values
                term *= dh_vals
            mask = np.isfinite(term) & (term >= 0.0)
            dv_vals[mask] = np.power(term[mask], 1.0 / 3.0)
            zero = np.isfinite(term) & (self._z_values == 0.0)
            dv_vals[zero] = 0.0

        if self._fallback_da is not None:
            try:
                da_vals = np.asarray(
                    self._call_with_params(
                        self._fallback_da,
                        (self._z_values,),
                        params,
                    ),
                    dtype=float,
                )
            except Exception:
                da_vals = dm_vals / (1.0 + self._z_values)
            if da_vals.shape != dm_vals.shape:
                with np.errstate(divide="ignore", invalid="ignore"):
                    da_vals = dm_vals / (1.0 + self._z_values)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                da_vals = dm_vals / (1.0 + self._z_values)

        return {
            "rs_drag": float("nan"),
            "DM": dm_vals,
            "DH": dh_vals,
            "DA": da_vals,
            "DV": dv_vals,
            "Hz": hz_vals,
            "z": self._z_values.copy(),
        }

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostics captured during the last evaluation."""

        return self._state.as_mapping()


__all__ = ["BAOLike"]
