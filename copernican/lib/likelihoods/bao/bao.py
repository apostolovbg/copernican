"""Baryon Acoustic Oscillation likelihood helper.

Model plugins provide their own background distances and sound horizon.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy

from ...model_coder import SoundHorizonComputationError
from ..likelihoods import LikelihoodProtocol, LikelihoodState


@dataclass(slots=True)
class BAOLike(LikelihoodProtocol):
    """Evaluate BAO log-likelihoods for pre-extracted observables."""

    redshifts: numpy.ndarray
    observable_types: numpy.ndarray
    observable_values: numpy.ndarray
    observable_errors: numpy.ndarray
    model_plugin: Any
    covariance_matrix_inv: numpy.ndarray | None = None
    rs_override: float | None = None
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )
    _z_values: numpy.ndarray = field(init=False, repr=False)
    _obs_type: numpy.ndarray = field(init=False, repr=False)
    _observed: numpy.ndarray = field(init=False, repr=False)
    _errors: numpy.ndarray = field(init=False, repr=False)
    _cov_inv: numpy.ndarray | None = field(init=False, repr=False)
    _rs_override: float | None = field(init=False, repr=False)
    _mask_dm: numpy.ndarray = field(init=False, repr=False)
    _mask_dh: numpy.ndarray = field(init=False, repr=False)
    _mask_dv: numpy.ndarray = field(init=False, repr=False)
    _prediction_buffer: numpy.ndarray = field(init=False, repr=False)
    _residual_buffer: numpy.ndarray = field(init=False, repr=False)
    _fallback_dm: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_hz: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_dv: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_da: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_rs: Callable[..., Any] | None = field(init=False, repr=False)
    _fallback_rs_drag: Callable[..., Any] | None = field(
        init=False, repr=False
    )
    _fallback_z_drag: Callable[..., Any] | None = field(init=False, repr=False)
    _c_light_km_s: float = field(init=False, repr=False)
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalise arrays and cache model callables for fast evaluation."""

        self._z_values = numpy.asarray(self.redshifts, dtype=float).copy()
        self._obs_type = numpy.asarray(
            self.observable_types, dtype=object
        ).copy()
        self._observed = numpy.asarray(
            self.observable_values, dtype=float
        ).copy()
        self._errors = numpy.asarray(
            self.observable_errors, dtype=float
        ).copy()
        self._cov_inv = (
            None
            if self.covariance_matrix_inv is None
            else numpy.asarray(self.covariance_matrix_inv, dtype=float)
        )
        self._rs_override = (
            None if self.rs_override is None else float(self.rs_override)
        )

        self._mask_dm = self._obs_type == "DM_over_rs"
        self._mask_dh = self._obs_type == "DH_over_rs"
        self._mask_dv = self._obs_type == "DV_over_rs"

        self._prediction_buffer = numpy.empty_like(self._observed, dtype=float)
        self._prediction_buffer.fill(numpy.nan)
        self._residual_buffer = numpy.empty_like(self._observed, dtype=float)

        if self._z_values.size == 0:
            self._setup_error = "(bao_like): BAO redshift array is empty."
        elif any(
            values.shape != self._z_values.shape
            for values in (self._obs_type, self._observed, self._errors)
        ):
            self._setup_error = (
                "(bao_like): BAO redshifts, observations, and errors must "
                "have identical shapes."
            )
        elif not numpy.all(numpy.isfinite(self._z_values)):
            self._setup_error = "(bao_like): BAO redshifts must be finite."
        elif not numpy.all(numpy.isfinite(self._observed)):
            self._setup_error = "(bao_like): BAO observations must be finite."
        elif not numpy.all(
            numpy.isfinite(self._errors) & (self._errors > 0.0)
        ):
            self._setup_error = (
                "(bao_like): BAO errors must be finite and positive."
            )

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
        self._fallback_rs_drag = getattr(
            self.model_plugin, "get_sound_horizon_rs_drag_Mpc", None
        )
        self._fallback_z_drag = getattr(
            self.model_plugin, "get_bao_drag_redshift", None
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
            self._record_failure(
                self._setup_error,
                failure_type="invalid_dataset",
                failure_stage="setup",
            )
            return float("-inf")

        background = self._compute_plugin_background(params)
        if background is None:
            if not self._state.metadata:
                self._record_failure(
                    "Model background helpers returned no usable values.",
                    failure_type="invalid_background",
                    failure_stage="background",
                )
            return float("-inf")

        rs_mpc = self._rs_override
        sound_horizon_source = "explicit_override"
        sound_horizon_epoch = "override"
        if rs_mpc is None:
            rs_drag = background.get("rs_drag")
            if rs_drag is None:
                rs_background = float("nan")
            else:
                rs_arr = numpy.asarray(rs_drag)
                if rs_arr.size != 1:
                    self._record_failure(
                        "Background drag sound horizon must be scalar.",
                        failure_type="invalid_shape",
                        failure_stage="drag",
                    )
                    return float("-inf")
                rs_background = float(rs_arr.flat[0])

            if numpy.isfinite(rs_background) and rs_background > 0.0:
                rs_mpc = rs_background
                sound_horizon_source = "background.rs_drag"
                sound_horizon_epoch = "drag"
            elif self._fallback_rs_drag is not None:
                # The canonical BAO ruler is drag-era.  Do not probe the
                # recombination helper first: a broken legacy helper must
                # never prevent a valid drag ruler from being used.
                try:
                    rs_mpc = float(
                        self._call_with_params(
                            self._fallback_rs_drag, (), params
                        )
                    )
                except SoundHorizonComputationError as exc:
                    logger.error(
                        "(bao_like): drag-epoch sound-horizon integral "
                        "diverged; aborting BAO predictions: %s",
                        exc,
                    )
                    self._record_failure(
                        str(exc),
                        failure_type=type(exc).__name__,
                        failure_stage="drag",
                    )
                    return float("-inf")
                except (
                    AttributeError,
                    ImportError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                ) as exc:
                    logger.error(
                        "(bao_like): canonical drag sound-horizon helper "
                        "failed: %s",
                        exc,
                    )
                    self._record_failure(
                        str(exc),
                        failure_type="helper_error",
                        failure_stage="drag",
                    )
                    return float("-inf")
                if not (numpy.isfinite(rs_mpc) and rs_mpc > 0.0):
                    logger.error(
                        "(bao_like): canonical drag sound-horizon helper "
                        "returned an invalid value: %s",
                        rs_mpc,
                    )
                    self._record_failure(
                        "Canonical drag sound-horizon helper returned a "
                        "non-finite or non-positive value.",
                        failure_type="invalid_value",
                        failure_stage="drag",
                    )
                    return float("-inf")
                sound_horizon_source = (
                    "model_plugin.get_sound_horizon_rs_drag_Mpc"
                )
                sound_horizon_epoch = "drag"
            elif self._fallback_rs is not None:
                # External legacy plugins may expose only the historical
                # recombination helper.  This compatibility path is reached
                # only when no canonical drag helper exists, so BAO never
                # silently substitutes recombination for drag on a modern
                # plugin.
                try:
                    rs_background = float(
                        self._call_with_params(self._fallback_rs, (), params)
                    )
                except SoundHorizonComputationError as exc:
                    logger.error(
                        "(bao_like): legacy sound-horizon integral diverged; "
                        "aborting BAO predictions: %s",
                        exc,
                    )
                    self._record_failure(
                        str(exc),
                        failure_type=type(exc).__name__,
                        failure_stage="recombination",
                    )
                    return float("-inf")
                except (
                    AttributeError,
                    ImportError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                ) as exc:
                    logger.error(
                        "(bao_like): legacy sound-horizon helper failed: %s",
                        exc,
                    )
                    self._record_failure(
                        str(exc),
                        failure_type="helper_error",
                        failure_stage="recombination",
                    )
                    return float("-inf")
                rs_mpc = rs_background
                sound_horizon_source = "model_plugin.get_sound_horizon_rs_Mpc"
                sound_horizon_epoch = "legacy_recombination"

        if not (numpy.isfinite(rs_mpc) and rs_mpc > 0):
            self._record_failure(
                "BAO sound horizon is non-finite or non-positive.",
                failure_type="invalid_value",
                failure_stage="drag",
            )
            return float("-inf")

        self._prediction_buffer.fill(numpy.nan)

        dm_vals = background.get("DM")
        dh_vals = background.get("DH")
        dv_vals = background.get("DV")

        if dm_vals is None or dh_vals is None or dv_vals is None:
            self._record_failure(
                "Model returned no complete BAO distance background.",
                failure_type="missing_helper",
                failure_stage="background",
            )
            return float("-inf")

        if numpy.any(self._mask_dm):
            self._prediction_buffer[self._mask_dm] = (
                dm_vals[self._mask_dm] / rs_mpc
            )

        if numpy.any(self._mask_dh):
            self._prediction_buffer[self._mask_dh] = (
                dh_vals[self._mask_dh] / rs_mpc
            )

        if numpy.any(self._mask_dv):
            self._prediction_buffer[self._mask_dv] = (
                dv_vals[self._mask_dv] / rs_mpc
            )

        if numpy.all(~numpy.isfinite(self._prediction_buffer)):
            logger.warning(
                "(bao_like): Model returned no finite BAO predictions."
            )
            self._record_failure(
                "Model returned no finite BAO predictions.",
                failure_type="nonfinite_prediction",
                failure_stage="background",
            )
            return float("-inf")

        numpy.subtract(
            self._observed,
            self._prediction_buffer,
            out=self._residual_buffer,
            casting="unsafe",
        )
        if numpy.any(~numpy.isfinite(self._residual_buffer)):
            logger.warning("(bao_like): Non-finite residuals in BAO data.")
            self._record_failure(
                "BAO residuals are non-finite.",
                failure_type="nonfinite_prediction",
                failure_stage="likelihood",
            )
            return float("-inf")

        cov_inv = self._cov_inv
        metadata: dict[str, Any] = {
            "points": int(self._observed.size),
            "sound_horizon_epoch": sound_horizon_epoch,
            "sound_horizon_source": sound_horizon_source,
        }
        if self._fallback_z_drag is not None and sound_horizon_epoch == "drag":
            try:
                z_drag = float(
                    self._call_with_params(self._fallback_z_drag, (), params)
                )
            except SoundHorizonComputationError as exc:
                self._record_failure(
                    str(exc),
                    failure_type=type(exc).__name__,
                    failure_stage="drag",
                )
                return float("-inf")
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                ZeroDivisionError,
                OverflowError,
            ) as exc:
                self._record_failure(
                    str(exc),
                    failure_type="invalid_value",
                    failure_stage="drag",
                )
                return float("-inf")
            if not (numpy.isfinite(z_drag) and z_drag > 0.0):
                self._record_failure(
                    "BAO drag redshift is non-finite or non-positive.",
                    failure_type="invalid_value",
                    failure_stage="drag",
                )
                return float("-inf")
            metadata["z_drag"] = z_drag
        chi2 = float("inf")
        if cov_inv is not None:
            try:
                if cov_inv.shape[0] != self._observed.size:
                    raise ValueError("Covariance size mismatch")
                chi2 = float(
                    self._residual_buffer @ cov_inv @ self._residual_buffer
                )
                metadata["covariance"] = "full"
            except (
                RuntimeError,
                TypeError,
                ValueError,
                numpy.linalg.LinAlgError,
            ) as exc:
                logger.warning(
                    "(bao_like): Falling back to diagonal covariance: %s",
                    exc,
                )
                cov_inv = None

        if cov_inv is None:
            valid = numpy.isfinite(self._errors) & (self._errors > 1e-9)
            if not numpy.any(valid):
                logger.warning("(bao_like): No valid BAO errors available.")
                self._record_failure(
                    "No valid BAO errors are available.",
                    failure_type="invalid_dataset",
                    failure_stage="likelihood",
                )
                return float("-inf")
            chi2 = float(
                numpy.sum(
                    (self._residual_buffer[valid] / self._errors[valid]) ** 2
                )
            )
            metadata["covariance"] = "diagonal"

        loglike = -0.5 * chi2 if numpy.isfinite(chi2) else float("-inf")
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

        # Select the supported calling convention before invocation. Retrying
        # after an arbitrary ``TypeError`` can execute a different convention
        # and hide a genuine error raised inside a model helper.
        supplied = (*args, *params)
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return func(*supplied)
        try:
            signature.bind(*supplied)
        except TypeError:
            signature.bind(*args)
            return func(*args)
        return func(*supplied)

    def _record_failure(
        self,
        message: str,
        *,
        failure_type: str,
        failure_stage: str,
    ) -> None:
        """Store a typed BAO failure without exposing a fake likelihood."""

        self._state = LikelihoodState(
            metadata={
                "error": str(message),
                "failure_type": str(failure_type),
                "failure_stage": str(failure_stage),
            }
        )

    def _compute_plugin_background(
        self, params: Sequence[float]
    ) -> dict[str, numpy.ndarray] | None:
        """Return background observables from model distance functions."""

        if self._fallback_dm is None or self._fallback_hz is None:
            self._record_failure(
                "BAO model is missing distance or Hubble helpers.",
                failure_type="missing_helper",
                failure_stage="background",
            )
            return None

        try:
            dm_vals = numpy.asarray(
                self._call_with_params(
                    self._fallback_dm,
                    (self._z_values,),
                    params,
                ),
                dtype=float,
            )
            hz_vals = numpy.asarray(
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
            self._record_failure(
                str(exc),
                failure_type=type(exc).__name__,
                failure_stage="background",
            )
            return None
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._record_failure(
                str(exc),
                failure_type="invalid_background",
                failure_stage="background",
            )
            return None

        if (
            dm_vals.shape != self._z_values.shape
            or hz_vals.shape != dm_vals.shape
        ):
            self._record_failure(
                "BAO distance helpers returned incompatible shapes.",
                failure_type="invalid_shape",
                failure_stage="background",
            )
            return None

        if not (
            numpy.all(numpy.isfinite(dm_vals))
            and numpy.all(numpy.isfinite(hz_vals))
        ):
            self._record_failure(
                "BAO distance helpers returned non-finite values.",
                failure_type="nonfinite_background",
                failure_stage="background",
            )
            return None

        with numpy.errstate(divide="ignore", invalid="ignore"):
            dh_vals = numpy.where(
                numpy.abs(hz_vals) > 1e-12,
                self._c_light_km_s / hz_vals,
                numpy.nan,
            )

        if self._fallback_dv is not None:
            try:
                dv_vals = numpy.asarray(
                    self._call_with_params(
                        self._fallback_dv,
                        (self._z_values,),
                        params,
                    ),
                    dtype=float,
                )
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                dv_vals = numpy.full_like(dm_vals, numpy.nan, dtype=float)
            if dv_vals.shape != dm_vals.shape:
                dv_vals = numpy.full_like(dm_vals, numpy.nan, dtype=float)
        else:
            dv_vals = numpy.full_like(dm_vals, numpy.nan, dtype=float)
            with numpy.errstate(divide="ignore", invalid="ignore"):
                term = dm_vals * dm_vals
                term *= self._z_values
                term *= dh_vals
            mask = numpy.isfinite(term) & (term >= 0.0)
            dv_vals[mask] = numpy.power(term[mask], 1.0 / 3.0)
            zero = numpy.isfinite(term) & (self._z_values == 0.0)
            dv_vals[zero] = 0.0

        if self._fallback_da is not None:
            try:
                da_vals = numpy.asarray(
                    self._call_with_params(
                        self._fallback_da,
                        (self._z_values,),
                        params,
                    ),
                    dtype=float,
                )
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                da_vals = dm_vals / (1.0 + self._z_values)
            if da_vals.shape != dm_vals.shape:
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    da_vals = dm_vals / (1.0 + self._z_values)
        else:
            with numpy.errstate(divide="ignore", invalid="ignore"):
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
