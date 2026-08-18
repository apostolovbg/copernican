"""Public native declared-graph CMB likelihood entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy
import pandas

from ...cmb_output import (
    CMBObservationBlock,
    assemble_cmb_theory_vector,
    cmb_observation_blocks,
)
from ...model_coder import prepare_native_cmb_execution_contract
from ..likelihoods import LikelihoodProtocol, LikelihoodState
from . import native_cache
from .copernican_cmb_solver import _compute_declared_perturbation_spectrum
from .native_batch import NativeCMBBatchResult
from .native_errors import (
    NativeContractError,
    NativeInitialPointError,
    NativeNonFiniteEvolutionError,
    NativeParameterDomainError,
    classify_native_exception,
    native_failure_context,
)
from .native_evolution import prepare_native_runtime_assets


def _resolve_plugin_cmb_contract(
    plugin: Any,
    model_params: Sequence[float],
) -> Mapping[str, Any]:
    """Return the plugin's required native CMB runtime contract."""

    get_native_runtime = getattr(plugin, "get_cmb_native_runtime", None)
    if not callable(get_native_runtime):
        raise ValueError("Model plugin does not expose a native CMB runtime")
    native_runtime = get_native_runtime(model_params)
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
    workload: str = "full_spectrum",
    background_provider: Any | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra from one CMB contract."""

    try:
        prepared_contract = prepare_native_cmb_execution_contract(
            contract_or_params
        )
    # DEVCOV_ALLOW_BROAD_ONCE native contract normalization boundary.
    except Exception as exc:
        context = native_failure_context(
            contract_or_params,
            workload=workload,
            spectra=spectra,
        )
        raise classify_native_exception(exc, context=context) from exc
    return _compute_declared_perturbation_spectrum(
        prepared_contract,
        ells,
        spectra=spectra,
        workload=workload,
        background_provider=background_provider,
    )


def _batch_cache_provenance(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
    *,
    include_identity: bool = True,
) -> dict[str, Any]:
    """Summarize shared and parameter-dependent cache activity for one item."""

    inventory = native_cache.native_cmb_cache_inventory()
    caches: dict[str, dict[str, Any]] = {}
    for name, after_stats in after.items():
        before_stats = before.get(name, {})
        caches[name] = {
            "category": inventory.get(name, {}).get("category", "unknown"),
            "entries": int(after_stats.get("entries", 0)),
            "hits": int(after_stats.get("hits", 0))
            - int(before_stats.get("hits", 0)),
            "misses": int(after_stats.get("misses", 0))
            - int(before_stats.get("misses", 0)),
            "evictions": int(after_stats.get("evictions", 0))
            - int(before_stats.get("evictions", 0)),
        }
    identity = (
        native_cache.latest_native_cmb_request_identity()
        if include_identity
        else None
    )
    identity_payload = None
    if identity is not None:
        identity_payload = {
            "contract_static": repr(identity.contract_static),
            "model_static": repr(identity.model_static),
            "request_specific": repr(identity.request_specific),
            "execution_solver": str(identity.execution_solver),
        }
    return {
        "cache_identity": identity_payload,
        "caches": caches,
    }


def compute_cmb_spectrum_batch(
    contracts: Sequence[Mapping[str, Any]],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
    requested_spectra: Iterable[str] | None = None,
) -> tuple[NativeCMBBatchResult, ...]:
    """Evaluate native CMB contracts in order with isolated typed outcomes.

    The first implementation deliberately adapts the exact scalar executor
    item-by-item. This establishes the ordering, failure, cache, and
    serialization contract before any shared numerical kernel is enabled.
    """

    ell_values = tuple(int(ell) for ell in ells)
    spectra = tuple(str(name) for name in (requested_spectra or ("TT",)))
    results: list[NativeCMBBatchResult] = []
    for index, contract in enumerate(contracts):
        before_cache = native_cache.native_cmb_cache_stats()
        before_requests = int(
            native_cache.native_cmb_performance_stats().get("requests", 0)
        )
        try:
            spectrum = compute_cmb_spectrum_from_contract(
                contract,
                ell_values,
                spectra=spectra,
                background_provider=background_provider,
            )
            failure = None
        # DEVCOV_ALLOW_BROAD_ONCE batch item boundary: isolate native failures.
        except Exception as exc:
            failure = classify_native_exception(
                exc,
                context=native_failure_context(
                    contract,
                    workload="full_spectrum",
                    spectra=spectra,
                ),
            )
            failure.add_context(batch_index=index)
            spectrum = None
        after_cache = native_cache.native_cmb_cache_stats()
        performance_record = (
            native_cache.latest_native_cmb_performance_record()
        )
        after_requests = int(
            native_cache.native_cmb_performance_stats().get("requests", 0)
        )
        if after_requests <= before_requests:
            performance_record = None
        performance = (
            {} if performance_record is None else dict(performance_record)
        )
        performance.setdefault("batch_index", index)
        results.append(
            NativeCMBBatchResult(
                index=index,
                spectrum=spectrum,
                failure=failure,
                performance_envelope=performance,
                cache_provenance=_batch_cache_provenance(
                    before_cache,
                    after_cache,
                    include_identity=performance_record is not None,
                ),
            )
        )
    return tuple(results)


def compute_cmb_spectrum_cached(
    plugin: Any,
    model_params: Sequence[float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    workload: str = "full_spectrum",
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    try:
        native_contract = _resolve_plugin_cmb_contract(
            plugin,
            model_params,
        )
        prepared_contract = prepare_native_cmb_execution_contract(
            native_contract
        )
    # DEVCOV_ALLOW_BROAD_ONCE plugin contract normalization boundary.
    except Exception as exc:
        raise classify_native_exception(
            exc,
            context={
                "parameters": tuple(float(value) for value in model_params),
                "requested_spectra": tuple(str(name) for name in spectra),
                "workload": str(workload),
            },
        ) from exc
    return _compute_declared_perturbation_spectrum(
        prepared_contract,
        ells,
        spectra=spectra,
        background_provider=plugin,
        workload=workload,
    )


def compute_cmb_spectrum(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    workload: str = "full_spectrum",
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return spectra using one structured CMB contract."""

    return compute_cmb_spectrum_from_contract(
        param_dict,
        ells,
        spectra=spectra,
        workload=workload,
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
    _observation_blocks: tuple[CMBObservationBlock, ...] = field(
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
    _proposal_rejection_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        """Extract immutable arrays so log-likelihood evaluation stays lean."""

        cmb_df = self.cmb_data_df
        if cmb_df is None or cmb_df.empty:
            self._setup_error = "(cmb_like): CMB data is empty."
            self._ells = numpy.empty(0, dtype=int)
            self._observed = numpy.empty(0, dtype=float)
            self._observed_spectra = ()
            self._observation_blocks = ()
            self._cov_inv = None
            self._residual_buffer = numpy.empty(0, dtype=float)
            return

        blocks = cmb_observation_blocks(cmb_df)
        self._observation_blocks = blocks
        if "spectrum" in cmb_df.columns:
            self._observed_spectra = tuple(
                block.metadata.canonical_name for block in blocks
            )
            self._ells = cmb_df["ell"].to_numpy(dtype=int, copy=True)
            self._observed = cmb_df["Dl_obs"].to_numpy(
                dtype=float,
                copy=True,
            )
        else:
            self._observed_spectra = ("TT",)
            self._observation_blocks = tuple(
                block
                for block in blocks
                if block.metadata.canonical_name == "TT"
            )
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
            self._state = LikelihoodState()
            raise NativeContractError(self._setup_error)

        domain_error = self._parameter_domain_error(params)
        if domain_error is not None:
            return self._reject_parameter_point(domain_error, logger=logger)

        try:
            native_contract = _resolve_plugin_cmb_contract(
                self.plugin,
                params,
            )
            native_contract = _with_extra_params(
                native_contract,
                self._extra_params_cached,
            )
        # DEVCOV_ALLOW_BROAD_ONCE plugin contract normalization boundary.
        except Exception as exc:
            typed_error = classify_native_exception(
                exc,
                context={"workload": "joint_mcmc"},
            )
            if isinstance(typed_error, NativeParameterDomainError):
                return self._reject_parameter_point(typed_error, logger=logger)
            raise typed_error from exc

        if not isinstance(native_contract, Mapping):
            self._state = LikelihoodState()
            raise NativeContractError(
                "(cmb_like): Model native CMB runtime must be a mapping."
            )

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
                workload="joint_mcmc",
            )
        # DEVCOV_ALLOW_BROAD_ONCE native likelihood normalization boundary.
        except Exception as exc:
            typed_error = classify_native_exception(
                exc,
                context=native_failure_context(
                    native_contract,
                    workload="joint_mcmc",
                    spectra=requested_spectra,
                ),
            )
            if isinstance(typed_error, NativeParameterDomainError):
                return self._reject_parameter_point(typed_error, logger=logger)
            raise typed_error from exc

        return self._loglike_from_theory(theory, requested_spectra)

    def loglike_batch(
        self, params_batch: Sequence[Sequence[float]]
    ) -> tuple[float, ...]:
        """Evaluate ordered parameter points through the native batch API."""

        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return tuple(0.0 for _ in params_batch)
        if self._setup_error is not None:
            self._state = LikelihoodState()
            raise NativeContractError(self._setup_error)

        logger = logging.getLogger()
        requested_spectra = self._observed_spectra or ("TT",)
        contracts: list[Mapping[str, Any]] = []
        positions: list[int] = []
        values = [float("-inf")] * len(params_batch)
        for index, params in enumerate(params_batch):
            domain_error = self._parameter_domain_error(params)
            if domain_error is not None:
                values[index] = self._reject_parameter_point(
                    domain_error,
                    logger=logger,
                )
                continue
            try:
                native_contract = _resolve_plugin_cmb_contract(
                    self.plugin,
                    params,
                )
                native_contract = _with_extra_params(
                    native_contract,
                    self._extra_params_cached,
                )
                if not isinstance(native_contract, Mapping):
                    raise NativeContractError(
                        "(cmb_like): Model native CMB runtime must be a "
                        "mapping."
                    )
                native_contract = prepare_native_cmb_execution_contract(
                    native_contract
                )
            # DEVCOV_ALLOW_BROAD_ONCE batch boundary: isolate failures.
            except Exception as exc:
                typed_error = classify_native_exception(
                    exc,
                    context={"workload": "joint_mcmc"},
                )
                if isinstance(typed_error, NativeParameterDomainError):
                    values[index] = self._reject_parameter_point(
                        typed_error,
                        logger=logger,
                    )
                    continue
                raise typed_error from exc
            positions.append(index)
            contracts.append(native_contract)

        if contracts:
            results = compute_cmb_spectrum_batch(
                contracts,
                self._ells,
                background_provider=self.plugin,
                requested_spectra=requested_spectra,
            )
            for position, result in zip(positions, results):
                if result.failure is not None:
                    if isinstance(
                        result.failure,
                        NativeParameterDomainError,
                    ):
                        values[position] = self._reject_parameter_point(
                            result.failure,
                            logger=logger,
                        )
                        continue
                    raise result.failure
                values[position] = self._loglike_from_theory(
                    result.spectrum,
                    requested_spectra,
                )
        return tuple(float(value) for value in values)

    def _loglike_from_theory(
        self,
        theory: numpy.ndarray | Mapping[str, numpy.ndarray] | None,
        requested_spectra: Sequence[str],
    ) -> float:
        """Assemble one native theory result into the CMB likelihood."""

        if theory is None:
            self._state = LikelihoodState()
            raise NativeContractError(
                "(cmb_like): Native CMB batch returned no spectrum."
            )
        if isinstance(theory, Mapping):
            try:
                theory_vector = assemble_cmb_theory_vector(
                    theory,
                    self._observation_blocks,
                    total_row_count=self._observed.size,
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._state = LikelihoodState()
                raise NativeContractError(
                    "(cmb_like): CMB theory blocks do not match observations."
                ) from exc
        else:
            theory_vector = numpy.asarray(theory, dtype=float)
            if len(requested_spectra) > 1:
                self._state = LikelihoodState()
                raise NativeContractError(
                    "(cmb_like): Multi-spectrum data require named theory."
                )
            if theory_vector.shape != self._observed.shape:
                self._state = LikelihoodState()
                raise NativeContractError(
                    "(cmb_like): CMB theory vector has unexpected shape."
                )

        if numpy.any(~numpy.isfinite(theory_vector)):
            self._state = LikelihoodState()
            raise NativeNonFiniteEvolutionError(
                "(cmb_like): Native CMB theory contains non-finite values."
            )

        numpy.subtract(
            self._observed,
            theory_vector,
            out=self._residual_buffer,
            casting="unsafe",
        )

        cov_inv = self._cov_inv
        if cov_inv is None:
            self._state = LikelihoodState()
            raise NativeContractError(
                "(cmb_like): Missing inverse covariance matrix."
            )

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
            self._state = LikelihoodState()
            raise NativeContractError(
                f"(cmb_like): Linear algebra failure: {exc}"
            ) from exc

        if not numpy.isfinite(chi2):
            self._state = LikelihoodState()
            raise NativeContractError(
                "(cmb_like): CMB likelihood assembly produced non-finite chi2."
            )
        loglike = -0.5 * chi2
        self._state = LikelihoodState(
            chi2=chi2,
            loglike=loglike,
            metadata={
                "covariance": "full",
                "points": int(self._observed.size),
                "proposal_rejections": self._proposal_rejection_count,
            },
        )
        return loglike

    def _parameter_domain_error(
        self,
        params: Sequence[float],
    ) -> NativeParameterDomainError | None:
        """Return a typed rejection for non-finite or out-of-bound values."""

        try:
            parameter_values = tuple(float(value) for value in params)
        except (TypeError, ValueError) as exc:
            raise NativeContractError(
                "CMB likelihood parameters must be numeric scalars."
            ) from exc
        if not all(numpy.isfinite(value) for value in parameter_values):
            return NativeParameterDomainError(
                "CMB likelihood parameters must be finite.",
                context={"parameters": parameter_values},
            )
        bounds = tuple(getattr(self.plugin, "PARAMETER_BOUNDS", ()) or ())
        if bounds and len(bounds) != len(parameter_values):
            raise NativeContractError(
                "CMB likelihood parameter vector does not match model bounds."
            )
        for index, (value, bound) in enumerate(zip(parameter_values, bounds)):
            lower, upper = bound
            if (lower is not None and value < float(lower)) or (
                upper is not None and value > float(upper)
            ):
                return NativeParameterDomainError(
                    "CMB likelihood parameter lies outside its model bounds.",
                    context={
                        "index": index,
                        "lower": lower,
                        "parameters": parameter_values,
                        "upper": upper,
                        "value": value,
                    },
                )
        return None

    def _reject_parameter_point(
        self,
        error: NativeParameterDomainError,
        *,
        logger: logging.Logger,
    ) -> float:
        """Record one expected proposal rejection without an error storm."""

        self._proposal_rejection_count += 1
        if self._proposal_rejection_count == 1 or (
            self._proposal_rejection_count % 100 == 0
        ):
            logger.debug(
                "(cmb_like): rejected %d parameter-domain proposal(s); "
                "latest: %s",
                self._proposal_rejection_count,
                error,
            )
        self._state = LikelihoodState(
            metadata={
                "failure": error.diagnostic(),
                "proposal_rejections": self._proposal_rejection_count,
            }
        )
        return float("-inf")

    def prepare_worker_runtime(self) -> None:
        """Materialize immutable graph assets once in the current worker."""

        if not self.enabled:
            return
        runtime = getattr(self.plugin, "CMB_NATIVE_RUNTIME", None)
        if runtime is None:
            raise NativeContractError(
                "Enabled CMB likelihood requires a compiled native runtime."
            )
        prepare_native_runtime_assets(
            runtime.runtime_signature,
            runtime.perturbation_data,
        )

    def preflight(self, params: Sequence[float]) -> float:
        """Validate the configured initial point before walker creation."""

        value = float(self.loglike(params))
        if not numpy.isfinite(value):
            raise NativeInitialPointError(
                "Initial native CMB parameter point was rejected.",
                context={
                    "parameters": tuple(float(value) for value in params),
                    "likelihood_state": self.state,
                },
            )
        return value

    @property
    def state(self) -> Mapping[str, Any]:
        """Return diagnostics captured during the last evaluation."""

        return self._state.as_mapping()


__all__ = [
    "CMBLike",
    "NativeCMBBatchResult",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_batch",
    "compute_cmb_spectrum_from_contract",
]
