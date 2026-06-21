"""Public CMB likelihood entrypoint.

This module owns the public CMB surface and dispatches between the standard
CAMB-backed solver and the native declared-graph solver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import camb
import numpy
import pandas

from ..shared import LikelihoodProtocol, LikelihoodState
from .camb_solver import (
    _compute_cmb_spectrum_direct,
    _make_camb_params,
    compute_camb_background_observables,
    compute_cmb_spectrum_from_camb_contract,
    compute_cmb_spectrum_from_legacy_params_for_tests,
    describe_camb_configuration,
)
from .copcmb_solver import (
    CustomCMBSpectrumData,
    _build_custom_cmb_background,
    _combine_camb_contracts,
    _compute_custom_cmb_spectrum_data,
    _compute_declared_perturbation_spectrum,
    _CustomCMBBackgroundData,
    _is_structured_camb_contract,
    _resolve_custom_cmb_numerics,
    _resolve_custom_cmb_physical_parameters,
    _summarize_declared_background_manifest_summary,
    _validate_camb_perturbation_execution,
)


def _resolve_plugin_cmb_contract(
    plugin: Any,
    cosmo_params: Sequence[float],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Return the runtime CMB contract and optional perturbation metadata."""

    get_native_runtime = getattr(plugin, "get_cmb_native_runtime", None)
    if callable(get_native_runtime):
        native_runtime = get_native_runtime(cosmo_params)
        perturbations = native_runtime.get("perturbations", {}) or {}
        if (
            isinstance(perturbations, Mapping)
            and perturbations.get("standard") is False
        ):
            return native_runtime, perturbations

    get_contract = getattr(plugin, "get_camb_contract", None)
    if not callable(get_contract):
        raise ValueError("Model plugin does not expose a CAMB contract")
    camb_contract = get_contract(cosmo_params)

    perturbation_contract: Mapping[str, Any] | None = None
    get_perturbation_contract = getattr(
        plugin, "get_cmb_perturbation_contract", None
    )
    if callable(get_perturbation_contract):
        perturbation_contract = get_perturbation_contract(cosmo_params)
        if perturbation_contract:
            camb_contract = _combine_camb_contracts(
                camb_contract,
                perturbation_contract,
            )
    return camb_contract, perturbation_contract


def _with_extra_params(
    contract: Mapping[str, Any],
    extra_params: Mapping[str, float] | None,
) -> Mapping[str, Any]:
    """Return ``contract`` with extra scalar parameters merged in."""

    if not extra_params:
        return contract
    updated = dict(contract)
    param_map = dict(updated.get("param_map", {}))
    param_map.update(
        {
            str(param_key): float(value)
            for param_key, value in extra_params.items()
        }
    )
    updated["param_map"] = param_map
    return updated


def compute_cmb_spectrum_from_dict(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra from one CMB contract."""

    if not _is_structured_camb_contract(contract_or_params):
        raise ValueError(
            "Structured CAMB contracts must include perturbations"
        )

    _validate_camb_perturbation_execution(contract_or_params)
    perturbations = contract_or_params.get("perturbations", {}) or {}
    if (
        isinstance(perturbations, Mapping)
        and perturbations.get("standard") is False
    ):
        return _compute_declared_perturbation_spectrum(
            contract_or_params,
            ells,
            spectra=spectra,
        )
    return compute_cmb_spectrum_from_camb_contract(
        contract_or_params,
        ells,
        spectra=spectra,
    )


def compute_cmb_spectrum_cached(
    plugin: Any,
    cosmo_params: Sequence[float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    camb_contract, perturbation_contract = _resolve_plugin_cmb_contract(
        plugin,
        cosmo_params,
    )
    if isinstance(perturbation_contract, Mapping) and (
        perturbation_contract.get("standard") is False
    ):
        _validate_camb_perturbation_execution(camb_contract)
        return _compute_declared_perturbation_spectrum(
            camb_contract,
            ells,
            spectra=spectra,
            background_provider=plugin,
        )
    return compute_cmb_spectrum_from_dict(
        camb_contract,
        ells,
        spectra=spectra,
    )


def compute_cmb_spectrum(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return spectra using a structured CMB contract."""

    return compute_cmb_spectrum_from_dict(param_dict, ells, spectra=spectra)


@dataclass(slots=True)
class CMBLike(LikelihoodProtocol):
    """Evaluate CMB log-likelihoods for tabulated spectra."""

    cmb_data_df: pandas.DataFrame
    plugin: Any
    extra_params: Mapping[str, float] | None = None
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )
    _ells: numpy.ndarray = field(init=False, repr=False)
    _observed: numpy.ndarray = field(init=False, repr=False)
    _cov_inv: numpy.ndarray | None = field(init=False, repr=False)
    _residual_buffer: numpy.ndarray = field(init=False, repr=False)
    _extra_params_cached: dict[str, float] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Extract immutable arrays so log-likelihood evaluation stays lean."""

        cmb_df = self.cmb_data_df
        if cmb_df is None or cmb_df.empty:
            self._setup_error = "(cmb_like): CMB data is empty."
            self._ells = numpy.empty(0, dtype=int)
            self._observed = numpy.empty(0, dtype=float)
            self._cov_inv = None
            self._residual_buffer = numpy.empty(0, dtype=float)
            return

        self._ells = cmb_df["ell"].to_numpy(dtype=int, copy=True)
        self._observed = cmb_df["Dl_obs"].to_numpy(dtype=float, copy=True)
        if numpy.any(~numpy.isfinite(self._observed)):
            self._setup_error = (
                "(cmb_like): Observed spectrum contains non-finite values."
            )

        cov_attr = cmb_df.attrs.get("covariance_matrix_inv")
        self._cov_inv = (
            None if cov_attr is None else numpy.asarray(cov_attr, dtype=float)
        )
        if self._cov_inv is None:
            self._setup_error = (
                "(cmb_like): Missing inverse covariance matrix."
            )

        self._residual_buffer = numpy.empty_like(self._observed, dtype=float)

        if self.extra_params:
            cached: dict[str, float] = {}
            for param_key, param_value in self.extra_params.items():
                cached[str(param_key)] = float(param_value)
            self._extra_params_cached = cached

    def loglike(self, params: Sequence[float]) -> float:
        """Return the CMB log-likelihood for ``params``."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        if self._setup_error is not None:
            logger.error(self._setup_error)
            self._state = LikelihoodState()
            return float("-inf")

        perturbation_contract: Mapping[str, Any] | None = None
        try:
            camb_contract, perturbation_contract = (
                _resolve_plugin_cmb_contract(
                    self.plugin,
                    params,
                )
            )
            camb_contract = _with_extra_params(
                camb_contract,
                self._extra_params_cached,
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error("(cmb_like): %s", exc)
            self._state = LikelihoodState()
            return float("-inf")

        if not isinstance(camb_contract, Mapping):
            self._state = LikelihoodState()
            return float("-inf")

        try:
            if isinstance(perturbation_contract, Mapping) and (
                perturbation_contract.get("standard") is False
            ):
                theory = _compute_declared_perturbation_spectrum(
                    camb_contract,
                    self._ells,
                    spectra=("TT",),
                    background_provider=self.plugin,
                )
            else:
                theory = compute_cmb_spectrum_from_dict(
                    camb_contract,
                    self._ells,
                    spectra=("TT",),
                )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error("(cmb_like): %s", exc)
            self._state = LikelihoodState()
            return float("-inf")
        if not isinstance(theory, numpy.ndarray):
            theory = numpy.asarray(theory, dtype=float)
        if theory.shape != self._observed.shape or numpy.any(
            ~numpy.isfinite(theory)
        ):
            self._state = LikelihoodState()
            return float("-inf")

        numpy.subtract(
            self._observed,
            theory,
            out=self._residual_buffer,
            casting="unsafe",
        )

        cov_inv = self._cov_inv
        if cov_inv is None:
            self._state = LikelihoodState()
            return float("-inf")

        try:
            chi2 = float(
                self._residual_buffer @ cov_inv @ self._residual_buffer
            )
        except (
            FloatingPointError,
            numpy.linalg.LinAlgError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.error("(cmb_like): Linear algebra failure: %s", exc)
            self._state = LikelihoodState()
            return float("-inf")

        loglike = -0.5 * chi2 if numpy.isfinite(chi2) else float("-inf")
        self._state = LikelihoodState(
            chi2=chi2,
            loglike=loglike,
            metadata={
                "covariance": "full",
                "points": int(self._observed.size),
            },
        )
        return loglike

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostics captured during the last evaluation."""

        return self._state.as_mapping()


__all__ = [
    "CMBLike",
    "CustomCMBSpectrumData",
    "_CustomCMBBackgroundData",
    "_build_custom_cmb_background",
    "_compute_cmb_spectrum_direct",
    "_compute_custom_cmb_spectrum_data",
    "_make_camb_params",
    "_resolve_custom_cmb_numerics",
    "_resolve_custom_cmb_physical_parameters",
    "_summarize_declared_background_manifest_summary",
    "compute_camb_background_observables",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_dict",
    "compute_cmb_spectrum_from_legacy_params_for_tests",
    "camb",
    "describe_camb_configuration",
]
