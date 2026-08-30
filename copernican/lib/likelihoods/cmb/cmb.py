"""Public declared-graph CMB likelihood entrypoint."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy
import pandas

from ...cmb_output import (
    CMBObservationBlock,
    assemble_cmb_theory_vector,
    cmb_observation_blocks,
)
from ...model_coder import (
    prepare_declared_cmb_execution_contract as _prepare_cmb_contract,
)
from ..likelihoods import LikelihoodProtocol, LikelihoodState
from .contracts import CMBResult, CMBSolverProtocol
from .errors import (
    ContractError,
    InitialPointError,
    NonFiniteEvolutionError,
    ParameterDomainError,
    classify_exception,
    failure_context,
)
from .orchestrators import ccmbs as _ccmbs_orchestrator
from .results import CMBBatchResult
from .runtime import cache
from .solvers.registry import resolve_cmb_solver, solver_provenance

prepare_cmb_execution_contract = _prepare_cmb_contract
prepare_declared_cmb_execution_contract = _prepare_cmb_contract

_LAST_CMB_RESULT: ContextVar[CMBResult | None] = ContextVar(
    "last_cmb_result",
    default=None,
)


def _compute_declared_perturbation_spectrum(
    contract_or_params: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    background_payload: Mapping[str, Any] | None = None,
    background_provider: Any | None = None,
    workload: str = "full_spectrum",
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Forward the legacy patch seam to the CCMBS orchestrator."""

    return _ccmbs_orchestrator._compute_declared_perturbation_spectrum(
        contract_or_params,
        ells,
        spectra=spectra,
        background_payload=background_payload,
        background_provider=background_provider,
        workload=workload,
    )


def _select_cmb_solver(
    solver: Any | str | None,
    cmb_solver: Any | str | None,
) -> CMBSolverProtocol:
    """Resolve the explicit solver argument or the CCMBS default."""

    if (
        solver is not None
        and cmb_solver is not None
        and solver is not cmb_solver
    ):
        raise ValueError("Specify either solver or cmb_solver, not both")
    return resolve_cmb_solver(solver if solver is not None else cmb_solver)


def _unwrap_cmb_result(
    result: CMBResult,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    """Raise typed solver failures and return the public spectra payload."""

    result.raise_for_failure()
    if result.spectra is None:
        raise ContractError("CMB solver returned no spectra")
    return result.spectra


def _resolve_plugin_cmb_contract(
    plugin: Any,
    model_params: Sequence[float],
) -> Mapping[str, Any]:
    """Return the plugin's required declared CMB runtime contract."""

    get_declared_runtime = getattr(plugin, "get_cmb_declared_runtime", None)
    if not callable(get_declared_runtime):
        raise ValueError("Model plugin does not expose a declared CMB runtime")
    declared_runtime = get_declared_runtime(model_params)
    if not isinstance(declared_runtime, Mapping):
        raise ValueError("Model declared CMB runtime must be a mapping")
    return declared_runtime


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
    solver: Any | str | None = None,
    cmb_solver: Any | str | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra from one CMB contract."""

    selected_solver = _select_cmb_solver(solver, cmb_solver)
    requested_ells = tuple(int(value) for value in ells)
    requested_spectra = tuple(str(value) for value in spectra)
    _LAST_CMB_RESULT.set(None)
    prepared_contract = contract_or_params
    if background_provider is not None and isinstance(
        prepared_contract, Mapping
    ):
        prepared_contract = dict(prepared_contract)
        prepared_contract["_background_provider"] = background_provider
    try:
        prepared = selected_solver.prepare(prepared_contract)
        result = selected_solver.evaluate(
            prepared,
            requested_ells,
            spectra=requested_spectra,
            workload=workload,
        )
        _LAST_CMB_RESULT.set(result)
    # DEVCOV_ALLOW_BROAD_ONCE declared solver boundary.
    except Exception as exc:
        context = failure_context(
            prepared_contract,
            workload=workload,
            spectra=spectra,
        )
        failure = classify_exception(exc, context=context)
        _LAST_CMB_RESULT.set(
            CMBResult(
                requested_ells=requested_ells,
                requested_spectra=requested_spectra,
                failure=failure,
                solver_id=getattr(selected_solver, "solver_id", ""),
                solver_label=getattr(selected_solver, "solver_label", ""),
            )
        )
        raise failure from exc
    return _unwrap_cmb_result(result)


def _batch_cache_provenance(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
    *,
    include_identity: bool = True,
) -> dict[str, Any]:
    """Summarize shared and parameter-dependent cache activity for one item."""

    inventory = cache.cmb_cache_inventory()
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
        cache.latest_cmb_request_identity() if include_identity else None
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
    workload: str = "full_spectrum",
    solver: Any | str | None = None,
    cmb_solver: Any | str | None = None,
) -> tuple[CMBBatchResult, ...]:
    """Evaluate declared CMB contracts in order with isolated typed outcomes.

    The first implementation deliberately adapts the exact scalar executor
    item-by-item. This establishes the ordering, failure, cache, and
    serialization contract before any shared numerical kernel is enabled.
    """

    ell_values = tuple(int(ell) for ell in ells)
    spectra = tuple(str(name) for name in (requested_spectra or ("TT",)))
    selected_solver = _select_cmb_solver(solver, cmb_solver)
    results: list[CMBBatchResult] = []
    for index, contract in enumerate(contracts):
        before_cache = cache.cmb_cache_stats()
        before_requests = int(cache.cmb_performance_stats().get("requests", 0))
        solver_result: CMBResult | None = None
        try:
            _LAST_CMB_RESULT.set(None)
            spectrum = compute_cmb_spectrum_from_contract(
                contract,
                ell_values,
                spectra=spectra,
                background_provider=background_provider,
                workload=workload,
                solver=selected_solver,
            )
            solver_result = _LAST_CMB_RESULT.get()
            failure = None
        # DEVCOV_ALLOW_BROAD_ONCE: isolate batch item failures.
        except Exception as exc:
            failure = classify_exception(
                exc,
                context=failure_context(
                    contract,
                    workload=workload,
                    spectra=spectra,
                ),
            )
            failure.add_context(batch_index=index)
            spectrum = None
            solver_result = _LAST_CMB_RESULT.get()
        after_cache = cache.cmb_cache_stats()
        performance_record = cache.latest_cmb_performance_record()
        after_requests = int(cache.cmb_performance_stats().get("requests", 0))
        if after_requests <= before_requests:
            performance_record = None
        performance = (
            {} if performance_record is None else dict(performance_record)
        )
        performance.setdefault("batch_index", index)
        results.append(
            CMBBatchResult(
                index=index,
                spectrum=spectrum,
                failure=failure,
                performance_envelope=performance,
                cache_provenance=_batch_cache_provenance(
                    before_cache,
                    after_cache,
                    include_identity=performance_record is not None,
                ),
                requested_ells=ell_values,
                requested_spectra=spectra,
                diagnostics=(
                    {} if solver_result is None else solver_result.diagnostics
                ),
                phase_timings=(
                    {}
                    if solver_result is None
                    else solver_result.phase_timings
                ),
                raw_spectra=(
                    None
                    if solver_result is None
                    else solver_result.raw_spectra
                ),
                solver_id=(
                    selected_solver.solver_id
                    if solver_result is None
                    else solver_result.solver_id
                ),
                solver_label=(
                    selected_solver.solver_label
                    if solver_result is None
                    else solver_result.solver_label
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
    numerical_overrides: Mapping[str, Any] | None = None,
    diagnostic_matrix_fast_path: bool = False,
    solver: Any | str | None = None,
    cmb_solver: Any | str | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    selected_solver = _select_cmb_solver(solver, cmb_solver)
    requested_ells = tuple(int(value) for value in ells)
    requested_spectra = tuple(str(value) for value in spectra)
    _LAST_CMB_RESULT.set(None)
    try:
        declared_contract = _resolve_plugin_cmb_contract(
            plugin,
            model_params,
        )
        prepared_contract = dict(declared_contract)
        prepared_contract["_background_provider"] = plugin
        if numerical_overrides:
            numerical = dict(prepared_contract.get("numerical", {}) or {})
            numerical.update(
                {
                    str(name): value
                    for name, value in numerical_overrides.items()
                }
            )
            prepared_contract["numerical"] = numerical
        if diagnostic_matrix_fast_path:
            prepared_contract["_diagnostic_matrix_fast_path"] = True
        prepared = selected_solver.prepare(prepared_contract)
        result = selected_solver.evaluate(
            prepared,
            requested_ells,
            spectra=requested_spectra,
            workload=workload,
        )
        _LAST_CMB_RESULT.set(result)
    # DEVCOV_ALLOW_BROAD_ONCE plugin and solver boundary.
    except Exception as exc:
        context = failure_context(
            prepared_contract if "prepared_contract" in locals() else {},
            workload=workload,
            spectra=requested_spectra,
        )
        failure = classify_exception(exc, context=context)
        _LAST_CMB_RESULT.set(
            CMBResult(
                requested_ells=requested_ells,
                requested_spectra=requested_spectra,
                failure=failure,
                solver_id=getattr(selected_solver, "solver_id", ""),
                solver_label=getattr(selected_solver, "solver_label", ""),
            )
        )
        raise failure from exc
    return _unwrap_cmb_result(result)


def compute_cmb_spectrum(
    param_dict: Mapping[str, Any],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
    workload: str = "joint_mcmc",
    solver: Any | str | None = None,
    cmb_solver: Any | str | None = None,
) -> numpy.ndarray | Mapping[str, numpy.ndarray]:
    r"""Return spectra using one structured CMB contract."""

    return compute_cmb_spectrum_from_contract(
        param_dict,
        ells,
        spectra=spectra,
        workload=workload,
        solver=solver,
        cmb_solver=cmb_solver,
    )


@dataclass(slots=True)
class CMBLike(LikelihoodProtocol):
    """Evaluate CMB log-likelihoods for tabulated spectra."""

    cmb_data_df: pandas.DataFrame
    plugin: Any
    extra_params: Mapping[str, float] | None = None
    enabled: bool = True
    cmb_solver: Any | str | None = None
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
    _solver_provenance: Mapping[str, Any] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Extract immutable arrays so log-likelihood evaluation stays lean."""

        self.cmb_solver = resolve_cmb_solver(self.cmb_solver)
        self._solver_provenance = solver_provenance(self.cmb_solver)

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
            raise ContractError(self._setup_error)

        domain_error = self._parameter_domain_error(params)
        if domain_error is not None:
            return self._reject_parameter_point(domain_error, logger=logger)

        try:
            declared_contract = _resolve_plugin_cmb_contract(
                self.plugin,
                params,
            )
            declared_contract = _with_extra_params(
                declared_contract,
                self._extra_params_cached,
            )
        # DEVCOV_ALLOW_BROAD_ONCE plugin contract normalization boundary.
        except Exception as exc:
            typed_error = classify_exception(
                exc,
                context={"workload": "joint_mcmc"},
            )
            if isinstance(typed_error, ParameterDomainError):
                return self._reject_parameter_point(typed_error, logger=logger)
            raise typed_error from exc

        if not isinstance(declared_contract, Mapping):
            self._state = LikelihoodState()
            raise ContractError(
                "(cmb_like): Model declared CMB runtime must be a mapping."
            )

        requested_spectra = self._observed_spectra or ("TT",)
        try:
            declared_contract = dict(declared_contract)
            declared_contract["_background_provider"] = self.plugin
            declared_contract["_joint_mcmc_fast_path"] = True
            prepared = self.cmb_solver.prepare(declared_contract)
            result = self.cmb_solver.evaluate(
                prepared,
                tuple(int(value) for value in self._ells),
                spectra=requested_spectra,
                workload="joint_mcmc",
            )
            theory = _unwrap_cmb_result(result)
        # DEVCOV_ALLOW_BROAD_ONCE declared likelihood normalization boundary.
        except Exception as exc:
            typed_error = classify_exception(
                exc,
                context=failure_context(
                    declared_contract,
                    workload="joint_mcmc",
                    spectra=requested_spectra,
                ),
            )
            if isinstance(typed_error, ParameterDomainError):
                return self._reject_parameter_point(typed_error, logger=logger)
            raise typed_error from exc

        return self._loglike_from_theory(theory, requested_spectra)

    def loglike_batch(
        self, params_batch: Sequence[Sequence[float]]
    ) -> tuple[float, ...]:
        """Evaluate ordered parameter points through the declared batch API."""

        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return tuple(0.0 for _ in params_batch)
        if self._setup_error is not None:
            self._state = LikelihoodState()
            raise ContractError(self._setup_error)

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
                declared_contract = _resolve_plugin_cmb_contract(
                    self.plugin,
                    params,
                )
                declared_contract = _with_extra_params(
                    declared_contract,
                    self._extra_params_cached,
                )
                if not isinstance(declared_contract, Mapping):
                    raise ContractError(
                        "(cmb_like): Model declared CMB runtime must be a "
                        "mapping."
                    )
            # DEVCOV_ALLOW_BROAD_ONCE batch boundary: isolate failures.
            except Exception as exc:
                typed_error = classify_exception(
                    exc,
                    context={"workload": "joint_mcmc"},
                )
                if isinstance(typed_error, ParameterDomainError):
                    values[index] = self._reject_parameter_point(
                        typed_error,
                        logger=logger,
                    )
                    continue
                raise typed_error from exc
            positions.append(index)
            contracts.append(declared_contract)

        if contracts:
            results = compute_cmb_spectrum_batch(
                contracts,
                self._ells,
                background_provider=self.plugin,
                requested_spectra=requested_spectra,
                solver=self.cmb_solver,
            )
            for position, result in zip(positions, results):
                if result.failure is not None:
                    if isinstance(
                        result.failure,
                        ParameterDomainError,
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
        """Assemble one declared theory result into the CMB likelihood."""

        if theory is None:
            self._state = LikelihoodState()
            raise ContractError(
                "(cmb_like): Declared CMB batch returned no spectrum."
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
                raise ContractError(
                    "(cmb_like): CMB theory blocks do not match observations."
                ) from exc
        else:
            theory_vector = numpy.asarray(theory, dtype=float)
            if len(requested_spectra) > 1:
                self._state = LikelihoodState()
                raise ContractError(
                    "(cmb_like): Multi-spectrum data require named theory."
                )
            if theory_vector.shape != self._observed.shape:
                self._state = LikelihoodState()
                raise ContractError(
                    "(cmb_like): CMB theory vector has unexpected shape."
                )

        if numpy.any(~numpy.isfinite(theory_vector)):
            self._state = LikelihoodState()
            raise NonFiniteEvolutionError(
                "(cmb_like): Declared CMB theory contains non-finite values."
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
            raise ContractError(
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
            raise ContractError(
                f"(cmb_like): Linear algebra failure: {exc}"
            ) from exc

        if not numpy.isfinite(chi2):
            self._state = LikelihoodState()
            raise ContractError(
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
                "cmb_solver": dict(self._solver_provenance),
            },
        )
        return loglike

    def _parameter_domain_error(
        self,
        params: Sequence[float],
    ) -> ParameterDomainError | None:
        """Return a typed rejection for non-finite or out-of-bound values."""

        try:
            parameter_values = tuple(float(value) for value in params)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "CMB likelihood parameters must be numeric scalars."
            ) from exc
        if not all(numpy.isfinite(value) for value in parameter_values):
            return ParameterDomainError(
                "CMB likelihood parameters must be finite.",
                context={"parameters": parameter_values},
            )
        bounds = tuple(getattr(self.plugin, "PARAMETER_BOUNDS", ()) or ())
        if bounds and len(bounds) != len(parameter_values):
            raise ContractError(
                "CMB likelihood parameter vector does not match model bounds."
            )
        for index, (value, bound) in enumerate(zip(parameter_values, bounds)):
            lower, upper = bound
            if (lower is not None and value < float(lower)) or (
                upper is not None and value > float(upper)
            ):
                return ParameterDomainError(
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
        error: ParameterDomainError,
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
        runtime = getattr(self.plugin, "CMB_DECLARED_RUNTIME", None)
        if runtime is None:
            raise ContractError(
                "Enabled CMB likelihood requires a compiled declared runtime."
            )
        self.cmb_solver.prepare(
            {
                "runtime_signature": runtime.runtime_signature,
                "perturbation_data": runtime.perturbation_data,
                "background_runtime": getattr(
                    runtime, "background_runtime", None
                ),
            }
        )

    def preflight(self, params: Sequence[float]) -> float:
        """Validate the configured initial point before walker creation."""

        value = float(self.loglike(params))
        if not numpy.isfinite(value):
            raise InitialPointError(
                "Initial declared CMB parameter point was rejected.",
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
    "CMBBatchResult",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_batch",
    "compute_cmb_spectrum_from_contract",
]
