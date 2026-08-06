"""Public native declared-graph CMB likelihood entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy
import pandas

from ...model_coder import prepare_native_cmb_execution_contract
from ..likelihoods import LikelihoodProtocol, LikelihoodState
from .copernican_cmb_solver import _compute_declared_perturbation_spectrum


def _resolve_plugin_cmb_contract(
    plugin: Any,
    cosmo_params: Sequence[float],
) -> Mapping[str, Any]:
    """Return the plugin's required native CMB runtime contract."""

    get_native_runtime = getattr(plugin, "get_cmb_native_runtime", None)
    if not callable(get_native_runtime):
        raise ValueError("Model plugin does not expose a native CMB runtime")
    native_runtime = get_native_runtime(cosmo_params)
    if not isinstance(native_runtime, Mapping):
        raise ValueError("Model native CMB runtime must be a mapping")
    return native_runtime


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


def compute_cmb_spectrum_from_contract(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra from one CMB contract."""

    prepared_contract = prepare_native_cmb_execution_contract(
        contract_or_params
    )
    return _compute_declared_perturbation_spectrum(
        prepared_contract,
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

    native_contract = _resolve_plugin_cmb_contract(
        plugin,
        cosmo_params,
    )
    prepared_contract = prepare_native_cmb_execution_contract(native_contract)
    return _compute_declared_perturbation_spectrum(
        prepared_contract,
        ells,
        spectra=spectra,
        background_provider=plugin,
    )


def compute_cmb_spectrum(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return spectra using one structured CMB contract."""

    return compute_cmb_spectrum_from_contract(
        param_dict,
        ells,
        spectra=spectra,
    )


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
    _observed_spectra: tuple[str, ...] = field(init=False, repr=False)
    _observed_spectrum_labels: numpy.ndarray = field(
        init=False,
        repr=False,
    )
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
            self._observed_spectra = ()
            self._observed_spectrum_labels = numpy.empty(0, dtype=object)
            self._cov_inv = None
            self._residual_buffer = numpy.empty(0, dtype=float)
            return

        if "spectrum" in cmb_df.columns:
            spectrum_series = cmb_df["spectrum"].astype(str)
            observed_spectra = tuple(dict.fromkeys(spectrum_series.tolist()))
            ordered_spectra = pandas.Categorical(
                spectrum_series,
                categories=list(observed_spectra),
                ordered=True,
            )
            ordered_df = cmb_df.assign(
                _spectrum_order=ordered_spectra
            ).sort_values(
                ["_spectrum_order", "ell"],
                kind="stable",
            )
            self._observed_spectra = observed_spectra
            self._ells = ordered_df["ell"].to_numpy(dtype=int, copy=True)
            self._observed = ordered_df["Dl_obs"].to_numpy(
                dtype=float,
                copy=True,
            )
            self._observed_spectrum_labels = ordered_df["spectrum"].to_numpy(
                dtype=object, copy=True
            )
        else:
            self._observed_spectra = ("TT",)
            self._ells = cmb_df["ell"].to_numpy(dtype=int, copy=True)
            self._observed = cmb_df["Dl_obs"].to_numpy(dtype=float, copy=True)
            self._observed_spectrum_labels = numpy.full(
                self._observed.shape,
                "TT",
                dtype=object,
            )
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
        elif self._cov_inv.shape != (
            self._observed.size,
            self._observed.size,
        ):
            self._setup_error = (
                "(cmb_like): Inverse covariance matrix has unexpected "
                "shape."
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

        try:
            native_contract = _resolve_plugin_cmb_contract(
                self.plugin,
                params,
            )
            native_contract = _with_extra_params(
                native_contract,
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

        if not isinstance(native_contract, Mapping):
            self._state = LikelihoodState()
            return float("-inf")

        requested_spectra = self._observed_spectra or ("TT",)
        try:
            native_contract = prepare_native_cmb_execution_contract(
                native_contract
            )
            theory = _compute_declared_perturbation_spectrum(
                native_contract,
                self._ells,
                spectra=requested_spectra,
                background_provider=self.plugin,
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

        if isinstance(theory, Mapping):
            theory_blocks = {
                str(name): numpy.asarray(values, dtype=float)
                for name, values in theory.items()
            }
            theory_vector = numpy.empty_like(self._observed, dtype=float)
            for index, (spectrum_name, _ell_value) in enumerate(
                zip(self._observed_spectrum_labels, self._ells)
            ):
                block = theory_blocks.get(str(spectrum_name))
                if block is None or index >= block.size:
                    self._state = LikelihoodState()
                    return float("-inf")
                theory_vector[index] = float(block[index])
        else:
            theory_vector = numpy.asarray(theory, dtype=float)
            if len(requested_spectra) > 1:
                self._state = LikelihoodState()
                return float("-inf")
            if theory_vector.shape != self._observed.shape:
                self._state = LikelihoodState()
                return float("-inf")

        if numpy.any(~numpy.isfinite(theory_vector)):
            self._state = LikelihoodState()
            return float("-inf")

        numpy.subtract(
            self._observed,
            theory_vector,
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
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_contract",
]
