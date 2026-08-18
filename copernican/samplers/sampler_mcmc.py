# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Markov Chain Monte Carlo sampler using :mod:`emcee`.

The combined optimiser has been retired entirely, leaving this sampler as the
sole runtime sampler. It continues to focus on Supernova Ia posteriors while
delegating shared χ² helpers to :mod:`copernican.lib.statistics` so the
module
acts as the canonical sampler façade. Future backends can slot in beside it
without changing the orchestration code. Verbose progress logging tracks both
burn-in and production phases with percentage updates so long chains always
report their status. Version 6.2.0 routes all likelihood evaluations through
the :class:`copernican.lib.likelihoods.JointLike` aggregator and the new
:func:`copernican.lib.model_adapter.make_logposterior` helper so posterior
calculations automatically honour per-parameter priors, declared bounds and
optional reparameterisation transforms while exposing diagnostic metadata
alongside sampled chains.

Version 7.6.20 removes walker snapshot logging entirely, dedicates the output
channel to concise diagnostics, and now emits simple counter lines instead of
maintaining a repaint pump. The carriage-return spinner has been retired but
each batch still announces its completion so transcripts never retain stale
bars. The counter updates log the stage label, completed steps and percentage,
keeping the GUI monitor and log in sync while the logger focuses on statistics.

Version 7.2.10 extends the reproducibility contract by constructing every
NumPy :class:`~numpy.random.Generator` from the shared
:func:`copernican.lib.utils.get_random_seed` value.  The helper captures the
seed supplied through the CLI prompt or ``COPERNICAN_SEED`` so subsequent
samplers observe the same pseudo-random stream without requiring callers to
seed multiple subsystems manually.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as multiprocessing_module
import os
import warnings
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Sequence

# ArviZ expects ``scipy.signal.gaussian`` which moved in newer SciPy releases.
try:  # pragma: no cover - compatibility shim
    from scipy.signal import gaussian  # type: ignore # noqa: F401
except ImportError:  # pragma: no cover - SciPy layout varies
    try:
        import scipy.signal as _signal
        from scipy.signal.windows import gaussian  # type: ignore # noqa: F401

        _signal.gaussian = gaussian
    except (AttributeError, ImportError):  # pragma: no cover
        pass

try:
    import arviz as arviz_module
except ModuleNotFoundError:  # pragma: no cover - optional fallback for tests
    arviz_module = None
import emcee
import numpy
import pandas

from copernican.lib import model_adapter as model_plugin_validation
from copernican.lib.likelihoods import BAOLike, CMBLike, JointLike, SNeLike
from copernican.lib.likelihoods.cmb.errors import InitialPointError
from copernican.lib.likelihoods.cmb.solvers.registry import (
    resolve_cmb_solver,
    solver_provenance,
)
from copernican.lib.progress import BatchProgressBar
from copernican.lib.sampler_capabilities import (
    SamplerProgressChunk,
    SamplerSetting,
)
from copernican.lib.statistics import (
    calculate_bao_observables,
    chi_squared_bao,
    chi_squared_cmb,
    chi_squared_sne,
    compute_cmb_spectrum,
    compute_cmb_spectrum_from_contract,
)
from copernican.lib.utils import get_random_seed

warnings.filterwarnings(
    "ignore",
    message=r"More chains \(\d+\) than draws \(\d+\)",
    module=r"arviz\\.data\\.base",
    category=UserWarning,
)

SAMPLER_KIND = "mcmc"
SAMPLER_LABEL = "Ensemble MCMC sampler"
SAMPLER_VERSION = "7.6.20"

SAMPLER_SETTINGS = (
    SamplerSetting(
        key="n_steps",
        label="Production steps",
        description="Iterations performed during the production phase.",
        dtype="int",
        default=200,
    ),
    SamplerSetting(
        key="burn_in_steps",
        label="Burn-in steps",
        description="Warm-up iterations discarded before the main chain.",
        dtype="int",
        default=50,
    ),
    SamplerSetting(
        key="n_walkers",
        label="Walkers",
        description="Size of the ensemble sampling the posterior.",
        dtype="int",
        default=32,
    ),
    SamplerSetting(
        key="pool_size",
        label="Worker pool size",
        description=(
            "Multiprocessing pool size; 0 leaves the decision to the " "suite."
        ),
        dtype="int",
        default=0,
        hint="0=auto",
    ),
    SamplerSetting(
        key="display_progress",
        label="Display progress",
        description="Emit live progress updates to the console.",
        dtype="bool",
        default=True,
    ),
    SamplerSetting(
        key="cmb_batch_size",
        label="CMB batch size",
        description=(
            "Opt-in bounded declared CMB batches; zero keeps scalar "
            "evaluation."
        ),
        dtype="int",
        default=0,
        hint="0=disabled",
    ),
)
SAMPLER_PROGRESS_CHUNKS = (
    SamplerProgressChunk(name="burn_in", label="Burn-in"),
    SamplerProgressChunk(name="production", label="Production"),
)

# ``emcee`` triggers its condition number guard when walkers occupy an almost
# degenerate subspace.  The suite accepts wildly different model definitions,
# so the sampler must adaptively identify fixed or near-fixed parameters and
# spread walkers enough to avoid singular ensembles.  These heuristics rely on
# a small mix of absolute and relative tolerances.  The defaults below flag
# intervals narrower than roughly one billionth of the parameter scale while
# still allowing legitimate, tight priors to remain active.
_FIXED_BOUNDS_RTOL = 1e-9
_FIXED_BOUNDS_ATOL = 1e-12
_MAX_INITIAL_CONDITION = 1e12
_MAX_INITIAL_ATTEMPTS = 12


class _ActiveLogProbability:
    """Picklable adapter that expands active coordinates to full parameters.

    ``emcee`` evaluates only the actively sampled parameters, yet the
    likelihood helper expects the full cosmological vector, including entries
    that are numerically fixed.  The managed start scripts launch the suite
    under the ``spawn`` multiprocessing context, so any callable passed to
    worker processes must be picklable.  Local closures fail that constraint
    and previously triggered ``AttributeError: Can't pickle local object``
    when Stage 2 activated multiprocessing.  A dedicated adapter class keeps
    the restoration logic together with the posterior callable while remaining
    serialisable for ``multiprocessing.Pool``.
    """

    __slots__ = ("_posterior", "_template", "_active_indices")

    def __init__(
        self,
        posterior: Callable[[numpy.ndarray], float],
        template_params: numpy.ndarray,
        active_indices: numpy.ndarray,
    ) -> None:
        """Save the wrapped posterior metadata for multiprocessing."""
        # ``posterior`` already encapsulates priors and likelihood terms via
        # ``_build_joint_logposterior``.  We retain it verbatim and only manage
        # the vector assembly around it.
        self._posterior = posterior
        # ``template_params`` stores the baseline parameter vector with fixed
        # entries included.  Copy and coerce it to ``float`` so all workers see
        # a consistent array and accidental mutations never bleed between
        # processes.
        self._template = numpy.asarray(template_params, dtype=float)
        # ``active_indices`` pinpoints which coordinates ``emcee`` manipulates.
        self._active_indices = numpy.asarray(active_indices, dtype=int)

    def assemble_full(self, position: numpy.ndarray) -> numpy.ndarray:
        """Return a full parameter vector for ``position``.

        The method centralises the reconstruction so tests can assert that
        fixed coordinates survive untouched when the adapter is invoked
        directly.  Each call returns a new array, keeping ``self._template``
        immutable for safe multiprocessing pickling.
        """

        full = self._template.copy()
        full[self._active_indices] = numpy.asarray(position, dtype=float)
        return full

    def __call__(self, position: numpy.ndarray) -> float:
        """Evaluate the posterior for ``position`` in the active subspace."""

        full = self.assemble_full(position)
        posterior_value = self._posterior(full)
        # ``emcee`` expects a Python ``float`` rather than a NumPy scalar for
        # predictable ``isfinite`` checks.  Coercing here guarantees downstream
        # consumers see the same type under both serial and multiprocessing
        # execution.
        return float(posterior_value)

    def evaluate_batch(self, positions: numpy.ndarray) -> tuple[float, ...]:
        """Evaluate a bounded ordered batch in the active parameter space."""

        coordinates = numpy.asarray(positions, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[numpy.newaxis, :]
        full_vectors = numpy.asarray(
            [self.assemble_full(position) for position in coordinates],
            dtype=float,
        )
        evaluate_batch = getattr(self._posterior, "evaluate_batch", None)
        if callable(evaluate_batch):
            values = evaluate_batch(full_vectors)
        else:
            values = [self._posterior(vector) for vector in full_vectors]
        return tuple(float(value) for value in values)

    def prepare_worker_runtime(self) -> None:
        """Prepare the wrapped likelihood's process-local runtime assets."""

        likelihood = getattr(self._posterior, "like", None)
        prepare = getattr(likelihood, "prepare_worker_runtime", None)
        if callable(prepare):
            prepare()


_WORKER_LOG_PROBABILITY: _ActiveLogProbability | None = None


def _initialize_mcmc_worker(
    log_probability: _ActiveLogProbability,
) -> None:
    """Install one sampler callable and compile worker-local static assets."""

    global _WORKER_LOG_PROBABILITY
    runtime_started = perf_counter()
    # Keep one numerical thread per worker so BLAS/OpenMP pools cannot multiply
    # the process cap into an unbounded numerical workload.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    _WORKER_LOG_PROBABILITY = log_probability
    log_probability.prepare_worker_runtime()
    logging.getLogger().info(
        "MCMC worker runtime prepared in %.3fs.",
        max(perf_counter() - runtime_started, 0.0),
    )


def _worker_log_probability(position: numpy.ndarray) -> float:
    """Evaluate one proposal using the worker's initialized callable."""

    if _WORKER_LOG_PROBABILITY is None:
        raise RuntimeError("MCMC worker runtime was not initialized")
    return _WORKER_LOG_PROBABILITY(position)


def _worker_batch_log_probability(
    positions: numpy.ndarray,
) -> tuple[float, ...]:
    """Evaluate one bounded proposal batch in the worker process."""

    if _WORKER_LOG_PROBABILITY is None:
        raise RuntimeError("MCMC worker runtime was not initialized")
    return _WORKER_LOG_PROBABILITY.evaluate_batch(positions)


def _worker_indexed_log_probability(
    item: tuple[int, numpy.ndarray],
) -> tuple[int, float]:
    """Evaluate one indexed proposal for unordered progress reporting."""

    index, position = item
    return int(index), _worker_log_probability(position)


class _JointLogLikelihood:
    """Picklable adapter that proxies :class:`JointLike.loglike`."""

    __slots__ = (
        "_joint_like",
        "parameter_bounds",
        "parameter_transforms",
    )

    def __init__(
        self,
        joint_like: JointLike,
        parameter_bounds: Iterable[tuple[float | None, float | None]] | None,
        parameter_transforms: (
            Iterable[Callable[[float], tuple[float, float]]] | None
        ),
    ) -> None:
        """Record the joint likelihood plus bounds/transforms for __call__."""
        self._joint_like = joint_like
        self.parameter_bounds = list(parameter_bounds or [])
        if parameter_transforms is not None:
            self.parameter_transforms = list(parameter_transforms)

    def __call__(self, params: Sequence[float]) -> float:
        """Return the combined log-likelihood for ``params``."""

        return float(self._joint_like.loglike(params))

    def evaluate_batch(
        self, params_batch: Sequence[Sequence[float]]
    ) -> tuple[float, ...]:
        """Evaluate a parameter batch through the joint likelihood."""

        return self._joint_like.loglike_batch(params_batch)

    def prepare_worker_runtime(self) -> None:
        """Prepare enabled likelihood assets in the current worker."""

        self._joint_like.prepare_worker_runtime()


class _SamplingProgressReporter:
    """Emit compact diagnostics for ensemble sampler updates.

    The reporter reconstructs the full parameter vectors from the active
    coordinates tracked by :mod:`emcee`, computes running summary statistics
    and prepares human-readable log lines.  Each stage instantiates its own
    reporter so burn-in and production maintain distinct Δχ² baselines while
    sharing the same formatting logic.
    """

    def __init__(
        self,
        param_names: Sequence[str],
        template_params: numpy.ndarray,
        active_indices: numpy.ndarray,
        *,
        progress_granularity: int = 20,
        max_params_to_show: int | None = None,
    ) -> None:
        """Capture sampling state and formatting hints for progress reports."""
        self._param_names = list(param_names)
        self._template = numpy.asarray(template_params, dtype=float)
        self._active_indices = numpy.asarray(active_indices, dtype=int)
        self._reference_log_prob: float | None = None
        self._report_count = 0
        self._show_all = max_params_to_show is None
        if self._show_all:
            self._max_params_to_show = len(self._param_names)
        else:
            self._max_params_to_show = int(max(1, max_params_to_show))
        self._sample_interval = max(1, int(progress_granularity // 4) or 1)
        # Reuse a scratch buffer so percentile calculations avoid allocating a
        # fresh ``(n_walkers, n_params)`` array on every progress callback.
        self._scratch: numpy.ndarray | None = None
        self._wrap_width = 72

    def __call__(
        self,
        step_index: int,
        state: emcee.State,
    ) -> Sequence[str]:
        """Return formatted diagnostics for ``state`` at ``step_index``."""

        del step_index  # The index is tracked via ``_report_count``.
        self._report_count += 1

        coords = numpy.asarray(state.coords, dtype=float)
        log_prob = numpy.asarray(state.log_prob, dtype=float)

        lines: list[str] = []
        finite_mask = numpy.isfinite(log_prob)
        if numpy.any(finite_mask):
            finite = log_prob[finite_mask]
            mean_lp = float(numpy.mean(finite))
            std_lp = float(numpy.std(finite))
            max_lp = float(numpy.max(finite))
            if self._reference_log_prob is None:
                self._reference_log_prob = max_lp
            delta_chi2 = -2.0 * (max_lp - self._reference_log_prob)
            lines.append(
                "    logP μ=% .3e σ=% .3e max=% .3e Δχ²≈%+.3f"
                % (mean_lp, std_lp, max_lp, delta_chi2)
            )
        else:
            lines.append(
                "    logP diagnostics unavailable (non-finite values)."
            )

        if coords.ndim == 1:
            coords = coords[numpy.newaxis, :]

        n_walkers = coords.shape[0]
        if n_walkers == 0:
            return lines

        expanded = self._expand_coordinates(coords)

        for idx, name in enumerate(
            self._param_names[: self._max_params_to_show]
        ):
            values = expanded[:, idx]
            finite_vals = values[numpy.isfinite(values)]
            if finite_vals.size == 0:
                lines.append(
                    f"    {name}: statistics unavailable (non-finite samples)."
                )
                continue
            q16, q50, q84 = numpy.percentile(finite_vals, [16.0, 50.0, 84.0])
            minus = q50 - q16
            plus = q84 - q50
            lines.append(
                "    %s: med=% .4g (-%.2g/+%.2g)" % (name, q50, minus, plus)
            )

        if not self._show_all:
            remaining = len(self._param_names) - self._max_params_to_show
            if remaining > 0:
                lines.append(
                    "    ... %d additional parameter(s) omitted." % remaining
                )

        return lines

    def _expand_coordinates(self, coords: numpy.ndarray) -> numpy.ndarray:
        """Return full parameter coordinates for ``coords`` walkers."""

        if coords.ndim == 1:
            coords = coords[numpy.newaxis, :]

        n_walkers = coords.shape[0]
        n_params = self._template.size
        if self._scratch is None or self._scratch.shape != (
            n_walkers,
            n_params,
        ):
            self._scratch = numpy.broadcast_to(
                self._template, (n_walkers, n_params)
            ).copy()
        else:
            self._scratch[...] = self._template

        if coords.size:
            self._scratch[:, self._active_indices] = coords

        return self._scratch


class _OrderedPoolMap:
    """Expose ordered worker mapping with a bounded task chunk size."""

    def __init__(
        self,
        pool: object,
        *,
        chunksize: int = 1,
        batch_size: int = 0,
        batch_function: (
            Callable[[numpy.ndarray], Sequence[float]] | None
        ) = None,
    ) -> None:
        """Wrap a multiprocessing pool without changing its lifecycle."""
        self._pool = pool
        self._chunksize = max(int(chunksize), 1)
        self._batch_size = max(int(batch_size), 0)
        self._batch_function = batch_function
        self.batch_count = 0
        self.batch_items = 0
        self.batch_elapsed_seconds = 0.0

    def map(self, function, iterable):
        """Map ``function`` in input order using the configured chunk size."""
        if self._batch_size > 1 and self._batch_function is not None:
            items = list(iterable)
            batches = [
                items[start : start + self._batch_size]
                for start in range(0, len(items), self._batch_size)
            ]
            started = perf_counter()
            batch_results = self._pool.map(
                self._batch_function,
                batches,
                chunksize=self._chunksize,
            )
            self.batch_count += len(batches)
            self.batch_items += len(items)
            self.batch_elapsed_seconds += max(perf_counter() - started, 0.0)
            return [value for result in batch_results for value in result]
        return self._pool.map(function, iterable, chunksize=self._chunksize)


def _build_joint_logposterior(
    model_plugin: Any,
    sne_data_df: Any,
    bao_data_df: Any | None = None,
    cmb_data_df: Any | None = None,
    cmb_solver: Any | str | None = None,
) -> tuple[
    Callable[[Sequence[float]], float],
    Callable[[Sequence[float]], float],
    JointLike,
]:
    """Return posterior, likelihood and diagnostics for joint datasets.

    Samplers evaluate the returned posterior repeatedly during sampling.  The
    helper therefore pre-computes the reusable :class:`JointLike` aggregator
    once, attaches the plugin's bounds and optional transformations to the
    underlying log-likelihood callable and finally hands everything to
    :func:`copernican.lib.model_adapter.make_logposterior` so priors and
    Jacobian adjustments remain consistent across samplers.
    """

    sne_like = SNeLike(model_plugin.distance_modulus_model, sne_data_df)

    # Recreate the BAO arrays in NumPy form so the helper can compute
    # residuals without repeatedly converting the DataFrame on every
    # likelihood evaluation. Missing datasets fall back to empty arrays while
    # the ``enabled`` flag records whether the component should contribute to
    # the joint likelihood.
    if bao_data_df is not None:
        bao_z = bao_data_df.get("redshift")
        bao_types = bao_data_df.get("observable_type")
        bao_val = bao_data_df.get("value")
        bao_err = bao_data_df.get("error")
    else:
        bao_z = bao_types = bao_val = bao_err = None

    bao_enabled = bool(
        bao_data_df is not None
        and getattr(model_plugin, "valid_for_bao", True)
        and hasattr(bao_data_df, "__len__")
        and len(bao_data_df) > 0
    )
    bao_like = BAOLike(
        numpy.asarray(bao_z if bao_z is not None else [], dtype=float),
        numpy.asarray(
            bao_types if bao_types is not None else [], dtype=object
        ),
        numpy.asarray(bao_val if bao_val is not None else [], dtype=float),
        numpy.asarray(bao_err if bao_err is not None else [], dtype=float),
        model_plugin,
        covariance_matrix_inv=(
            None
            if bao_data_df is None
            else bao_data_df.attrs.get("covariance_matrix_inv")
        ),
        enabled=bao_enabled,
    )

    # The CMB helper operates directly on the tabulated DataFrame so we can
    # preserve metadata such as the inverse covariance matrix.  When the model
    # declares it is incompatible with CMB analyses or the dataset lacks the
    # required covariance the component remains disabled.
    cmb_enabled = bool(
        cmb_data_df is not None
        and getattr(model_plugin, "valid_for_cmb", True)
        and not getattr(cmb_data_df, "empty", True)
        and "covariance_matrix_inv" in getattr(cmb_data_df, "attrs", {})
    )
    cmb_like = CMBLike(
        cmb_data_df if cmb_data_df is not None else pandas.DataFrame(),
        model_plugin,
        enabled=cmb_enabled,
        cmb_solver=cmb_solver,
    )

    # Preserve any model-provided dataset toggles while ensuring every
    # component defaults to the availability detected above.  The resulting
    # configuration keeps backwards compatibility with lightweight models that
    # only targeted Supernova data yet lets richer plugins disable specific
    # datasets explicitly.
    likelihood_config = dict(
        getattr(model_plugin, "LIKELIHOOD_CONFIG", {}) or {}
    )
    likelihood_config.setdefault(
        "sne",
        sne_data_df is not None
        and hasattr(sne_data_df, "__len__")
        and len(sne_data_df) > 0,
    )
    likelihood_config.setdefault("bao", bao_enabled)
    likelihood_config.setdefault("cmb", cmb_enabled)

    joint_like = JointLike(
        {"sne": sne_like, "bao": bao_like, "cmb": cmb_like},
        config=likelihood_config,
    )

    transforms = getattr(model_plugin, "PARAMETER_TRANSFORMS", None)
    loglike = _JointLogLikelihood(
        joint_like,
        getattr(model_plugin, "PARAMETER_BOUNDS", []),
        transforms,
    )

    priors = getattr(model_plugin, "PARAMETER_PRIOR_OBJECTS", None)
    if priors is None:
        priors = getattr(model_plugin, "PARAMETER_PRIORS", [])
    posterior = model_plugin_validation.make_logposterior(loglike, priors)
    return posterior, loglike, joint_like


def _preflight_initial_model_point(
    posterior: Callable[[Sequence[float]], float],
    parameters: Sequence[float],
) -> float:
    """Require a finite initial posterior before any walkers or pool exist."""

    parameter_values = tuple(float(value) for value in parameters)
    value = float(posterior(parameter_values))
    if not numpy.isfinite(value):
        raise InitialPointError(
            "Initial model point has non-finite posterior probability.",
            context={"parameters": parameter_values},
        )
    return value


# Backward compatibility for legacy imports that still reference the
# supernova-specific helper name.
_build_sne_logposterior = _build_joint_logposterior


def _reseed_invalid_walkers(
    coords: numpy.ndarray,
    log_prob: numpy.ndarray,
    *,
    lower: numpy.ndarray,
    upper: numpy.ndarray,
    rng: numpy.random.Generator,
    log_probability_fn: Callable[[numpy.ndarray], float],
    map_fn: Callable[..., Any] | None = None,
    reference_position: numpy.ndarray | None = None,
    max_attempts: int = 8,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Replace non-finite walker states with fresh proposals.

    The :mod:`emcee` stretch move occasionally propagates ``nan`` coordinates
    when proposals sample highly implausible regions.  Those invalid walkers
    subsequently trigger ``RuntimeWarning`` messages inside emcee's internal
    subtraction logic.  To maintain a clean log and avoid undefined
    transitions we reseed any problematic walkers by drawing small Gaussian
    jitters around the mean of the valid ensemble before continuing the run.
    ``log_probability_fn`` evaluates the sampler's objective for the proposed
    coordinates so fixed-parameter expansions can remain encapsulated inside a
    caller-provided closure.  ``reference_position`` supplies a fallback
    centroid when every walker is invalid so reseeding still succeeds even if
    the ensemble collapses entirely.
    """

    logger = logging.getLogger()
    coords = numpy.asarray(coords, dtype=float).copy()
    log_prob = numpy.asarray(log_prob, dtype=float).copy()

    invalid = (~numpy.isfinite(coords).all(axis=1)) | (
        ~numpy.isfinite(log_prob)
    )
    if not numpy.any(invalid):
        return coords, log_prob

    logger.warning(
        "Detected %d invalid walkers after burn-in; reseeding them.",
        int(numpy.sum(invalid)),
    )

    valid_coords = coords[~invalid]
    if valid_coords.size == 0:
        if reference_position is None:
            raise RuntimeError("No baseline available for reseeding walkers.")
        valid_coords = numpy.asarray(reference_position, dtype=float)[None, :]

    centre = numpy.mean(valid_coords, axis=0)
    spread = numpy.std(valid_coords, axis=0)
    finite_width = numpy.where(
        numpy.isfinite(lower) & numpy.isfinite(upper), upper - lower, numpy.nan
    )
    fallback = numpy.where(
        numpy.isfinite(finite_width), finite_width / 6.0, 1.0
    )
    spread = numpy.where(spread > 0, spread, fallback)
    spread = numpy.where(spread > 0, spread, 1.0)

    bad_idx = numpy.flatnonzero(invalid)
    attempts = 0
    while bad_idx.size and attempts < max_attempts:
        attempts += 1
        jitter = rng.standard_normal((bad_idx.size, centre.size))
        proposals = centre + jitter * numpy.maximum(spread, 1e-3)
        proposals = numpy.clip(proposals, lower, upper)
        if map_fn is None:
            new_log_prob = numpy.array(
                [log_probability_fn(pos) for pos in proposals]
            )
        else:
            new_log_prob = numpy.asarray(
                list(map_fn(log_probability_fn, proposals)), dtype=float
            )
        finite = numpy.isfinite(new_log_prob)
        coords[bad_idx[finite]] = proposals[finite]
        log_prob[bad_idx[finite]] = new_log_prob[finite]
        bad_idx = bad_idx[~finite]

    if bad_idx.size:
        raise RuntimeError(
            "Unable to reseed %d walkers with finite log probability"
            % bad_idx.size
        )

    return coords, log_prob


def _run_stage_with_progress(
    sampler: emcee.EnsembleSampler,
    initial_state: numpy.ndarray,
    n_steps: int,
    *,
    stage_name: str,
    logger: logging.Logger,
    progress_granularity: int = 20,
    summary_callback: (
        Callable[[int, emcee.State], Sequence[str]] | None
    ) = None,
    progress_label: str | None = None,
    display_progress: bool = True,
    progress_listener: Callable[[dict[str, object]], None] | None = None,
    stage_metadata: dict[str, str] | None = None,
):
    """Iterate ``sampler.sample`` while logging percentage progress.

    When ``summary_callback`` is provided it receives the step index and the
    :class:`emcee.State` instance for each progress update and must return an
    iterable of strings to append to the log output.  This keeps statistics
    generation orthogonal to the sampling loop while ensuring all logging
    honours ``progress_granularity``.
    """

    if n_steps <= 0:
        logger.info("Skipping %s stage; zero steps requested.", stage_name)
        return sampler.get_last_sample()

    logger.info("Starting MCMC %s stage for %d steps...", stage_name, n_steps)

    if progress_granularity <= 0:
        progress_granularity = 1

    label = progress_label or f"{stage_name.title()} stage"
    progress_bar = BatchProgressBar(
        label,
        n_steps,
        display=display_progress,
        subunit_labels=("iteration", "iterations"),
        walker_total=n_steps * sampler.nwalkers,
        progress_listener=progress_listener,
        stage_metadata=stage_metadata,
    )
    if n_steps <= progress_granularity:
        interval = max(1, n_steps)
    else:
        interval = max(1, math.ceil(n_steps / progress_granularity))
    batch_start = 1
    batch_end = min(interval, n_steps)
    progress_bar.start_batch(batch_start, batch_end)

    state = None
    iterator: Iterator[emcee.State] | None = None
    try:
        iterator = sampler.sample(
            initial_state, iterations=n_steps, progress=False
        )
        for idx in range(1, n_steps + 1):
            state = next(iterator)
            progress_bar.update(
                idx,
                processed=idx,
                total=n_steps,
                step_progress=1.0,
                walker_processed=idx * sampler.nwalkers,
                walker_total=n_steps * sampler.nwalkers,
            )
            if idx == batch_end and idx < n_steps:
                progress_bar.finish_batch()
                batch_start = idx + 1
                batch_end = min(batch_start + interval - 1, n_steps)
                progress_bar.start_batch(batch_start, batch_end)
            elif idx == n_steps:
                progress_bar.finish_batch()

            if summary_callback is not None and (
                idx == 1 or idx % interval == 0 or idx == n_steps
            ):
                with progress_bar.suspend_display():
                    for line in summary_callback(idx, state):
                        logger.info("%s", line)
    finally:
        progress_bar.finish_batch()

    if state is None:
        raise RuntimeError("Sampler produced no states during %s" % stage_name)

    logger.info("Completed MCMC %s stage.", stage_name)
    return state


def _resolve_mcmc_pool_processes(
    *,
    requested_pool: int | None,
    n_walkers: int,
) -> int | None:
    """Return a bounded worker count that leaves one CPU for the parent."""

    try:
        cpu_total = int(multiprocessing_module.cpu_count())
    except NotImplementedError:
        cpu_total = 1
    available_workers = min(max(cpu_total - 1, 0), int(n_walkers))
    if requested_pool is None:
        return available_workers if available_workers > 1 else None
    requested_workers = int(requested_pool)
    if requested_workers <= 1:
        return None
    return min(requested_workers, available_workers) or None


def _mcmc_cpu_count() -> int:
    """Return a positive CPU count for sampler resource accounting."""

    try:
        return max(1, int(multiprocessing_module.cpu_count()))
    except (NotImplementedError, TypeError, ValueError):
        return 1


def _ensemble_performance_envelope(
    *,
    started: float,
    phase_seconds: dict[str, float],
    requested_pool_workers: int,
    pool_workers: int,
    cpu_count: int,
    n_walkers: int,
    burn_in_steps: int,
    production_steps: int,
    failed_requests: int = 0,
    cmb_batch_size: int = 0,
    batch_count: int = 0,
    batch_items: int = 0,
    batch_elapsed_seconds: float = 0.0,
) -> dict[str, object]:
    """Build serialisable timing and worker-resource provenance."""

    elapsed = max(0.0, perf_counter() - started)
    worker_limit = min(max(cpu_count - 1, 0), max(int(n_walkers), 0))
    oversubscribed = pool_workers > worker_limit
    nominal_evaluations = (
        n_walkers
        * (max(int(burn_in_steps), 0) + max(int(production_steps), 0))
        + n_walkers
    )
    return {
        "workload": "ensemble_mcmc",
        "elapsed_seconds": elapsed,
        "phase_seconds": {
            name: max(0.0, float(value))
            for name, value in phase_seconds.items()
        },
        "requested_pool_workers": int(requested_pool_workers),
        "pool_workers": int(pool_workers),
        "cpu_count": int(cpu_count),
        "worker_limit": int(worker_limit),
        "numerical_threads_per_worker": 1,
        "oversubscribed": bool(oversubscribed),
        "nominal_evaluations": int(nominal_evaluations),
        "failed_requests": int(max(0, failed_requests)),
        "cmb_batch_size": int(max(0, cmb_batch_size)),
        "batch_count": int(max(0, batch_count)),
        "batch_items": int(max(0, batch_items)),
        "batch_elapsed_seconds": max(0.0, float(batch_elapsed_seconds)),
        "batch_items_per_second": (
            float(batch_items) / float(batch_elapsed_seconds)
            if batch_elapsed_seconds > 0 and batch_items > 0
            else 0.0
        ),
        "budget_seconds": 1800.0,
        "budget_passed": bool(elapsed <= 1800.0 and not oversubscribed),
    }


def sample_parameters(
    sne_data_df: Any,
    model_plugin: Any,
    *,
    bao_data_df: Any | None = None,
    cmb_data_df: Any | None = None,
    n_walkers: int = 32,
    n_steps: int = 200,
    pool_size: int | None = None,
    progress_granularity: int = 20,
    burn_in_steps: int | None = None,
    display_progress: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cmb_batch_size: int = 0,
    cmb_solver: Any | str | None = None,
) -> dict[str, Any]:
    """Sample cosmological parameters with joint dataset support.

    The routine initialises walkers within the declared parameter bounds, runs
    a configurable burn-in stage and returns summary statistics alongside the
    raw chain. ``burn_in_steps`` overrides the adaptive ``max(100, n_steps //
    5)`` heuristic, letting tests and scripted workflows trim the warm-up cost
    without affecting the production phase.  ``progress_granularity`` controls
    how many progress updates appear per stage and therefore also the cadence
    of the accompanying diagnostics.  When ``pool_size`` is provided the
    walker ensemble expands as needed so every worker process remains busy
    throughout burn-in and production.  ``display_progress`` disables the
    console progress bar when ``False`` so automated pipelines can execute
    quietly.
    """

    logger = logging.getLogger()
    cmb_batch_size = max(int(cmb_batch_size), 0)
    if cmb_batch_size == 1:
        cmb_batch_size = 0
    if cmb_batch_size > 1:
        logger.info(
            "Declared CMB batch adapter enabled: max batch size=%d.",
            cmb_batch_size,
        )
    ensemble_started = perf_counter()
    phase_seconds = {
        "initialization": 0.0,
        "burn_in": 0.0,
        "production": 0.0,
    }
    cpu_count = _mcmc_cpu_count()
    requested_pool_workers = int(pool_size or 0)

    def _failure_result(
        *,
        pool_workers: int = 0,
        burn_steps: int = 0,
    ) -> dict[str, object]:
        """Return a failure payload with the same resource provenance."""

        return {
            "success": False,
            "samples": None,
            "cmb_solver": solver_provenance(selected_cmb_solver),
            "ensemble_performance": _ensemble_performance_envelope(
                started=ensemble_started,
                phase_seconds=phase_seconds,
                requested_pool_workers=requested_pool_workers,
                pool_workers=pool_workers,
                cpu_count=cpu_count,
                n_walkers=n_walkers,
                burn_in_steps=burn_steps,
                production_steps=n_steps,
                cmb_batch_size=cmb_batch_size,
            ),
        }

    model_plugin_validation.validate_plugin(model_plugin)
    selected_cmb_solver = resolve_cmb_solver(cmb_solver)

    posterior_full, loglike_full, joint_like = _build_joint_logposterior(
        model_plugin,
        sne_data_df,
        bao_data_df,
        cmb_data_df,
        selected_cmb_solver,
    )
    names: Iterable[str] = getattr(model_plugin, "PARAMETER_NAMES", [])
    names = list(names)
    initial = numpy.asarray(
        getattr(model_plugin, "INITIAL_GUESSES", []), float
    )
    bounds = list(getattr(model_plugin, "PARAMETER_BOUNDS", []))

    ndim_total = len(initial)
    if ndim_total == 0 or len(bounds) != ndim_total:
        logger.error("Model plugin missing parameter definitions")
        return _failure_result()

    try:
        lower_all, upper_all, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logger
        )
    except ValueError:
        return _failure_result()
    active_mask = ~fixed_mask
    active_indices = numpy.flatnonzero(active_mask)
    fixed_indices = numpy.flatnonzero(fixed_mask)

    fixed_only = active_indices.size == 0
    if fixed_only:
        logger.info("All parameters are fixed; mirroring reference values.")

    if fixed_indices.size:
        fixed_names = ", ".join(names[idx] for idx in fixed_indices)
        logger.info(
            "Treating %d parameter(s) as fixed or numerically locked: %s",
            int(fixed_indices.size),
            fixed_names,
        )

    template_params = numpy.clip(initial, lower_all, upper_all)
    _preflight_initial_model_point(posterior_full, template_params)
    initial_active = template_params[active_indices]
    lower = lower_all[active_indices]
    upper = upper_all[active_indices]

    seed = get_random_seed()
    if seed is None:
        seed = 0
    rng = numpy.random.default_rng(seed)
    logger.debug(
        "Initialising sampler RNG with seed %s for deterministic chains.",
        seed,
    )

    ndim_active = active_indices.size
    requested_pool = pool_size if pool_size not in (None, 0) else None
    # ``emcee`` requires at least ``2 * ndim`` walkers.  Honour that rule and
    # ensure the bounded worker pool never idles because the ensemble is too
    # small.
    minimum_walkers = max(2 * ndim_active, 2)
    candidate_pool_processes = _resolve_mcmc_pool_processes(
        requested_pool=requested_pool,
        n_walkers=max(n_walkers, minimum_walkers),
    )
    if candidate_pool_processes is not None:
        minimum_walkers = max(minimum_walkers, candidate_pool_processes)

    n_walkers = max(n_walkers, minimum_walkers)
    logger.info(
        "Using %d walkers for %d active parameter(s).",
        int(n_walkers),
        int(ndim_active),
    )

    sampler: emcee.EnsembleSampler | None = None
    sampler_pool: _OrderedPoolMap | None = None
    chain_active: numpy.ndarray | None = None
    log_prob_chain: numpy.ndarray | None = None
    flat_log_prob: numpy.ndarray | None = None
    acceptance_fraction: numpy.ndarray | None = None
    pool_processes: int | None = None

    burn_in = (
        burn_in_steps if burn_in_steps is not None else max(100, n_steps // 5)
    )
    burn_in = max(1, int(burn_in))
    if not fixed_only:
        log_probability_active = _ActiveLogProbability(
            posterior_full,
            template_params,
            active_indices,
        )
        pool = None
        pool_processes = _resolve_mcmc_pool_processes(
            requested_pool=requested_pool,
            n_walkers=n_walkers,
        )
        worker_progress = None
        if requested_pool is None:
            if pool_processes is not None:
                logger.info(
                    "Auto-configured multiprocessing pool with %d worker(s).",
                    pool_processes,
                )
        elif pool_processes is not None:
            logger.info(
                "Using bounded multiprocessing pool with %d worker(s).",
                pool_processes,
            )

        if pool_processes is not None:
            worker_progress = BatchProgressBar(
                f"{model_plugin.MODEL_NAME} worker pool",
                pool_processes,
                display=display_progress,
                progress_listener=progress_callback,
                stage_metadata={
                    "phase": "worker_pool_launch",
                    "model": getattr(model_plugin, "MODEL_NAME", ""),
                },
            )
            worker_progress.start_batch(1, pool_processes)
            pool_started = perf_counter()
            worker_flag = "COPERNICAN_MCMC_WORKER"
            previous_worker_flag = os.environ.get(worker_flag)
            os.environ[worker_flag] = "1"
            try:
                pool = multiprocessing_module.get_context("spawn").Pool(
                    processes=pool_processes,
                    initializer=_initialize_mcmc_worker,
                    initargs=(log_probability_active,),
                )
            finally:
                if previous_worker_flag is None:
                    os.environ.pop(worker_flag, None)
                else:
                    os.environ[worker_flag] = previous_worker_flag
            pool_elapsed = max(perf_counter() - pool_started, 0.0)
            worker_progress.update(
                pool_processes,
                processed=pool_processes,
                total=pool_processes,
                force=True,
            )
            worker_progress.finish_batch()
            logger.info(
                "MCMC worker pool launched: workers=%d, elapsed=%.3fs.",
                pool_processes,
                pool_elapsed,
            )
        try:
            initial_log_probability = (
                _worker_log_probability
                if pool is not None
                else log_probability_active
            )
            initial_map = pool.map if pool is not None else None
            unordered_map = pool.imap_unordered if pool is not None else None
            walker_progress = BatchProgressBar(
                f"{model_plugin.MODEL_NAME} walker initialization",
                n_walkers,
                display=display_progress,
                progress_listener=progress_callback,
                stage_metadata={
                    "phase": "walker_initialization",
                    "model": getattr(model_plugin, "MODEL_NAME", ""),
                },
            )
            walker_progress.start_batch(1, n_walkers)

            def _report_walker_progress(completed: int, total: int) -> None:
                """Forward completed initialization evaluations."""

                walker_progress.update(
                    completed,
                    processed=completed,
                    total=total,
                    force=True,
                )

            initialization_started = perf_counter()
            initial_positions, logp = _initialise_active_walkers(
                initial_active,
                lower,
                upper,
                n_walkers,
                rng,
                initial_log_probability,
                map_fn=initial_map,
                unordered_map_fn=unordered_map,
                progress_callback=_report_walker_progress,
            )
            walker_progress.finish_batch()
            logger.info(
                "Walker initialization completed: walkers=%d, elapsed=%.3fs.",
                n_walkers,
                max(perf_counter() - initialization_started, 0.0),
            )
            sampler_log_probability = (
                _worker_log_probability
                if pool is not None
                else log_probability_active
            )
            sampler_pool = (
                _OrderedPoolMap(
                    pool,
                    chunksize=1,
                    batch_size=cmb_batch_size,
                    batch_function=_worker_batch_log_probability,
                )
                if pool is not None
                else None
            )
            sampler = emcee.EnsembleSampler(
                n_walkers,
                ndim_active,
                sampler_log_probability,
                pool=sampler_pool,
            )
            phase_seconds["initialization"] = (
                perf_counter() - initialization_started
            )
            burnin_reporter = _SamplingProgressReporter(
                names,
                template_params,
                active_indices,
                progress_granularity=progress_granularity,
            )
            burn_in_started = perf_counter()
            last = _run_stage_with_progress(
                sampler,
                initial_positions,
                burn_in,
                stage_name="burn-in",
                logger=logger,
                progress_granularity=progress_granularity,
                summary_callback=burnin_reporter,
                progress_label=f"{model_plugin.MODEL_NAME} burn-in",
                display_progress=display_progress,
                progress_listener=progress_callback,
                stage_metadata={
                    "phase": "burn-in",
                    "model": getattr(model_plugin, "MODEL_NAME", ""),
                },
            )
            phase_seconds["burn_in"] = perf_counter() - burn_in_started
            try:
                coords, log_prob = _reseed_invalid_walkers(
                    last.coords,
                    last.log_prob,
                    lower=lower,
                    upper=upper,
                    rng=rng,
                    log_probability_fn=initial_log_probability,
                    map_fn=initial_map,
                    reference_position=initial_active,
                )
            except RuntimeError as exc:
                logger.error("%s", exc)
                return _failure_result(
                    pool_workers=int(pool_processes or 0),
                    burn_steps=burn_in,
                )
            sampler.reset()
            production_reporter = _SamplingProgressReporter(
                names,
                template_params,
                active_indices,
                progress_granularity=progress_granularity,
            )
            production_started = perf_counter()
            _run_stage_with_progress(
                sampler,
                coords,
                n_steps,
                stage_name="production",
                logger=logger,
                progress_granularity=progress_granularity,
                summary_callback=production_reporter,
                progress_label=f"{model_plugin.MODEL_NAME} production",
                display_progress=display_progress,
                progress_listener=progress_callback,
                stage_metadata={
                    "phase": "production",
                    "model": getattr(model_plugin, "MODEL_NAME", ""),
                },
            )
            phase_seconds["production"] = perf_counter() - production_started
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chain_active = sampler.get_chain()
        log_prob_chain = sampler.get_log_prob()
        flat_log_prob = sampler.get_log_prob(flat=True)
        acceptance_fraction = sampler.acceptance_fraction
    else:
        n_effective_walkers = int(n_walkers)
        n_production = max(int(max(n_steps, 1)), 1)
        chain_active = numpy.zeros(
            (n_production, n_effective_walkers, 0), dtype=float
        )
        log_prob_value = float(posterior_full(template_params))
        log_prob_chain = numpy.full(
            (n_production, n_effective_walkers), log_prob_value
        )
        flat_log_prob = log_prob_chain.ravel()
        acceptance_fraction = numpy.zeros(n_effective_walkers, dtype=float)

    n_production, n_effective_walkers, _ = chain_active.shape
    chain = numpy.empty(
        (n_production, n_effective_walkers, ndim_total),
        dtype=chain_active.dtype,
    )
    chain[:] = template_params
    chain[:, :, active_indices] = chain_active

    flat_chain = chain.reshape(-1, ndim_total)

    best_index = int(numpy.argmax(flat_log_prob))
    best_params = flat_chain[best_index]
    mean_params = numpy.mean(flat_chain, axis=0)

    covariance = numpy.cov(flat_chain, rowvar=False)
    errors = numpy.sqrt(numpy.diag(covariance))
    error_dict = {
        name: error_value for name, error_value in zip(names, errors)
    }

    fitted = {name: value for name, value in zip(names, best_params)}
    posterior_mean = {name: value for name, value in zip(names, mean_params)}

    loglike_best = float(loglike_full(best_params))
    log_posterior_best = float(posterior_full(best_params))
    likelihood_state = dict(joint_like.state)
    metadata = likelihood_state.get("metadata", {})
    components = metadata.get("components", {})
    chi2_sne = float(components.get("sne", {}).get("chi2", float("inf")))
    chi2_bao = float(components.get("bao", {}).get("chi2", 0.0))
    chi2_cmb = float(components.get("cmb", {}).get("chi2", 0.0))
    chi2_best = float(
        likelihood_state.get("chi2", chi2_sne + chi2_bao + chi2_cmb)
    )

    sne_points = len(sne_data_df) if sne_data_df is not None else 0
    bao_points = components.get("bao", {}).get("metadata", {}).get("points", 0)
    cmb_points = components.get("cmb", {}).get("metadata", {}).get("points", 0)
    try:
        bao_points = int(bao_points)
    except (TypeError, ValueError):
        bao_points = len(bao_data_df) if bao_data_df is not None else 0
    try:
        cmb_points = int(cmb_points)
    except (TypeError, ValueError):
        cmb_points = (
            len(cmb_data_df)
            if cmb_data_df is not None and not cmb_data_df.empty
            else 0
        )
    total_points = sne_points + bao_points + cmb_points
    dof = total_points - ndim_total
    reduced = chi2_best / dof if dof > 0 else numpy.nan

    failed_requests = 0
    for component in components.values():
        component_metadata = component.get("metadata", {})
        try:
            failed_requests += int(
                component_metadata.get("proposal_rejections", 0)
            )
        except (TypeError, ValueError):
            continue
    ensemble_performance = _ensemble_performance_envelope(
        started=ensemble_started,
        phase_seconds=phase_seconds,
        requested_pool_workers=requested_pool_workers,
        pool_workers=int(pool_processes or 0),
        cpu_count=cpu_count,
        n_walkers=n_walkers,
        burn_in_steps=burn_in,
        production_steps=n_steps,
        failed_requests=failed_requests,
        cmb_batch_size=cmb_batch_size,
        batch_count=(0 if sampler_pool is None else sampler_pool.batch_count),
        batch_items=0 if sampler_pool is None else sampler_pool.batch_items,
        batch_elapsed_seconds=(
            0.0 if sampler_pool is None else sampler_pool.batch_elapsed_seconds
        ),
    )
    log_prior_best = float("-inf")
    if math.isfinite(log_posterior_best) and math.isfinite(loglike_best):
        log_prior_best = log_posterior_best - loglike_best

    acceptance = (
        acceptance_fraction
        if acceptance_fraction is not None
        else numpy.zeros(n_effective_walkers, dtype=float)
    )
    diagnostics: dict[str, dict[str, float]] = {
        "rhat": {},
        "ess_bulk": {},
        "ess_tail": {},
    }
    if arviz_module is not None and active_indices.size:
        try:
            # ``arviz`` expects chains ordered as ``(n_chains, n_draws, ...)``.
            # ``emcee`` stores them as ``(n_draws, n_chains, n_params)``,
            # so swap the leading axes. Fixed coordinates are excluded from
            # rank diagnostics because their zero variance makes R-hat
            # undefined.
            active_names = [names[int(index)] for index in active_indices]
            active_chain = chain[:, :, active_indices]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                inference_data = arviz_module.from_dict(
                    posterior={
                        "parameters": numpy.swapaxes(active_chain, 0, 1)
                    },
                    coords={"parameter": active_names},
                    dims={"parameters": ["parameter"]},
                )
                rhat_dataset = arviz_module.rhat(inference_data, method="rank")
                ess_bulk_dataset = arviz_module.ess(
                    inference_data, method="bulk"
                )
                ess_tail_dataset = arviz_module.ess(
                    inference_data, method="tail"
                )

            def _dataset_to_dict(dataset: Any) -> dict[str, float]:
                """Return scalar diagnostics keyed by parameter name."""

                series = dataset["parameters"].to_series()
                return {
                    str(idx): float(value) for idx, value in series.items()
                }

            diagnostics = {
                "rhat": _dataset_to_dict(rhat_dataset),
                "ess_bulk": _dataset_to_dict(ess_bulk_dataset),
                "ess_tail": _dataset_to_dict(ess_tail_dataset),
            }
            total_draws = float(n_effective_walkers * n_production)
            for fixed_index in fixed_indices:
                fixed_name = names[int(fixed_index)]
                diagnostics["rhat"][fixed_name] = 1.0
                diagnostics["ess_bulk"][fixed_name] = total_draws
                diagnostics["ess_tail"][fixed_name] = total_draws
            if any(
                not math.isfinite(value)
                for values in diagnostics.values()
                for value in values.values()
            ):
                raise ValueError("ArviZ returned non-finite diagnostics")
            if diagnostics["rhat"]:
                rhat_values = numpy.fromiter(
                    diagnostics["rhat"].values(),
                    dtype=float,
                    count=len(diagnostics["rhat"]),
                )
                logger.info(
                    "Rank-normalised R-hat summary: min=%.3f median=%.3f "
                    "max=%.3f",
                    float(numpy.min(rhat_values)),
                    float(numpy.median(rhat_values)),
                    float(numpy.max(rhat_values)),
                )
            if diagnostics["ess_bulk"]:
                bulk_values = numpy.fromiter(
                    diagnostics["ess_bulk"].values(),
                    dtype=float,
                    count=len(diagnostics["ess_bulk"]),
                )
                tail_values = numpy.fromiter(
                    diagnostics["ess_tail"].values(),
                    dtype=float,
                    count=len(diagnostics["ess_tail"]),
                )
                logger.info(
                    "Effective sample sizes: bulk median=%.1f tail median="
                    "%.1f",
                    float(numpy.median(bulk_values)),
                    float(numpy.median(tail_values)),
                )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive logging path
            logger.warning(
                "Falling back to internal diagnostics after ArviZ failure: %s",
                exc,
            )
            diagnostics = _compute_basic_diagnostics(
                chain, names, logger=logger
            )
    else:
        if arviz_module is None:
            logger.warning(
                "ArviZ is unavailable; computing conservative diagnostics "
                "without it."
            )
        diagnostics = _compute_basic_diagnostics(chain, names, logger=logger)

    logger.info(
        "MCMC acceptance for %s: mean=%.3f, min=%.3f, max=%.3f",
        getattr(model_plugin, "MODEL_NAME", "Unknown"),
        float(numpy.mean(acceptance)),
        float(numpy.min(acceptance)),
        float(numpy.max(acceptance)),
    )

    # ``emcee`` estimates autocorrelation times with a minimum window of 32
    # steps (``min_n``) and typically raises ``AutocorrError`` when shorter
    # chains are analysed.  Guard the call so tiny diagnostic runs return
    # ``None`` rather than emitting a ``RuntimeWarning`` downstream.
    autocorr = None
    min_autocorr_window = max(32, 2 * max(int(ndim_active), 1))
    if n_production >= min_autocorr_window:
        try:
            autocorr = sampler.get_autocorr_time()
        except (
            AttributeError,
            emcee.autocorr.AutocorrError,
            RuntimeError,
            ValueError,
        ):
            autocorr = None
    else:
        logger.debug(
            "Skipping autocorrelation estimate: production steps %d < %d",
            int(n_production),
            int(min_autocorr_window),
        )

    return {
        "success": numpy.isfinite(chi2_best)
        and math.isfinite(log_posterior_best),
        "samples": chain,
        "log_probability": log_prob_chain,
        "fitted_model_params": fitted,
        "posterior_mean_params": posterior_mean,
        "model_name": getattr(model_plugin, "MODEL_NAME", "Unknown"),
        "param_names": list(names),
        "parameter_errors": error_dict,
        "covariance_matrix": covariance,
        "chi2_min": chi2_best,
        "chi2_sne": chi2_sne,
        "chi2_bao": chi2_bao,
        "chi2_cmb": chi2_cmb,
        "chi2_total": chi2_best,
        "log_likelihood_best": loglike_best,
        "log_posterior_best": log_posterior_best,
        "log_prior_best": log_prior_best,
        "dof": dof,
        "reduced_chi2": reduced,
        "acceptance_fraction": acceptance,
        "burn_in_steps": burn_in,
        "production_steps": n_steps,
        "n_walkers": int(n_walkers),
        "autocorrelation_time": autocorr,
        "pool_workers": int(pool_processes or 0),
        "ensemble_performance": ensemble_performance,
        "diagnostics": diagnostics,
        "progress_granularity": int(progress_granularity),
        "likelihood_state": likelihood_state,
        "chi2_components": {
            "sne": chi2_sne,
            "bao": chi2_bao,
            "cmb": chi2_cmb,
        },
        "data_points": {
            "sne": sne_points,
            "bao": bao_points,
            "cmb": cmb_points,
            "total": total_points,
        },
        "cmb_solver": solver_provenance(selected_cmb_solver),
    }


__all__ = [
    "SAMPLER_KIND",
    "SAMPLER_LABEL",
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_from_contract",
    "sample_parameters",
]


def _estimate_condition_number(samples: numpy.ndarray) -> float | None:
    """Return the condition number of ``samples`` or ``None`` when undefined.

    ``emcee`` inspects the condition number of the initial walker ensemble to
    ensure the stretch move can generate proposals effectively.  The function
    below mirrors that logic without importing private ``emcee`` helpers so the
    sampler can deliberately inflate the walker spread before the library
    raises
    ``ValueError``.  When the ensemble contains fewer than two walkers the
    condition number is undefined; in that situation we return ``None`` and let
    the caller continue with additional attempts.
    """

    if samples.shape[0] < 2:
        return None
    centred = samples - numpy.mean(samples, axis=0, keepdims=True)
    try:
        singular_values = numpy.linalg.svd(
            centred, full_matrices=False, compute_uv=False
        )
    except numpy.linalg.LinAlgError:
        return float("inf")
    if singular_values.size == 0:
        return float("inf")
    largest = float(singular_values[0])
    tolerance = (
        largest * max(centred.shape) * numpy.finfo(singular_values.dtype).eps
    )
    # Match a rank-style cutoff so tiny SVD noise stays numerically zero.
    positive = singular_values[singular_values > tolerance]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())


def _classify_parameter_bounds(
    bounds: Iterable[tuple[float | None, float | None]],
    *,
    logger: logging.Logger,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Return lower/upper bounds and a mask of effectively fixed parameters.

    Each entry in ``bounds`` is converted to a floating interval.  ``None``
    values map to ``-numpy.inf`` or ``numpy.inf`` as appropriate.  Bounds
    where the upper edge falls below the lower edge signal malformed model
    definitions and trigger an error log.  Parameters whose admissible range
    shrinks to a
    single point—or a numerically indistinguishable sliver—are flagged as
    fixed so the active sampling subspace contains only degrees of freedom
    that ``emcee`` can explore without tripping its linear-independence
    checks.
    """

    lower = numpy.empty(len(bounds), dtype=float)
    upper = numpy.empty(len(bounds), dtype=float)
    for idx, (low, high) in enumerate(bounds):
        lower[idx] = -numpy.inf if low is None else float(low)
        upper[idx] = numpy.inf if high is None else float(high)
        if (
            numpy.isfinite(lower[idx])
            and numpy.isfinite(upper[idx])
            and upper[idx] < lower[idx]
        ):
            logger.error(
                "Parameter %d declares inverted bounds [%f, %f]",
                idx,
                lower[idx],
                upper[idx],
            )
            raise ValueError("invalid parameter bounds: lower exceeds upper")

    with numpy.errstate(invalid="ignore"):
        widths = upper - lower
        centres = (upper + lower) / 2.0
        scale = numpy.maximum(numpy.abs(centres), 1.0)
        threshold = scale * _FIXED_BOUNDS_RTOL + _FIXED_BOUNDS_ATOL
        fixed_mask = numpy.isfinite(widths) & (widths <= threshold)

    return lower, upper, fixed_mask


def _compute_basic_diagnostics(
    chain: numpy.ndarray,
    names: Sequence[str],
    *,
    logger: logging.Logger,
) -> dict[str, dict[str, float]]:
    """Return conservative R-hat and ESS estimates without ArviZ.

    Each walker is treated as an independent chain so the classic
    Gelman–Rubin
    estimator can operate without external dependencies.  When ArviZ is
    unavailable the helper keeps diagnostics finite, albeit deliberately
    conservative, by collapsing effective sample sizes to the total draw count.
    """

    walkers_first = numpy.swapaxes(chain, 0, 1)
    n_chains, n_draws, _ = walkers_first.shape
    if n_chains <= 0 or n_draws <= 1:
        logger.warning(
            "Unable to compute R-hat with %d chain(s) and %d draw(s); "
            "returning NaNs.",
            int(n_chains),
            int(n_draws),
        )
        rhat_values = numpy.full(len(names), 1.0, dtype=float)
    else:
        chain_means = numpy.mean(walkers_first, axis=1)
        chain_vars = numpy.var(walkers_first, axis=1, ddof=1)
        mean_overall = numpy.mean(chain_means, axis=0)
        between = numpy.sum((chain_means - mean_overall) ** 2, axis=0)
        between *= n_draws / max(n_chains - 1, 1)
        within = numpy.mean(chain_vars, axis=0)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            var_hat = ((n_draws - 1) / n_draws) * within + between / n_draws
            ratio = numpy.where(within > 0, var_hat / within, 1.0)
            rhat_values = numpy.sqrt(numpy.clip(ratio, 0.0, numpy.inf))

    rhat = {name: float(value) for name, value in zip(names, rhat_values)}
    total_draws = float(max(n_chains, 1) * max(n_draws, 0))
    ess = {name: total_draws for name in names}
    return {"rhat": rhat, "ess_bulk": ess.copy(), "ess_tail": ess.copy()}


def _initialise_active_walkers(
    initial_active: numpy.ndarray,
    lower: numpy.ndarray,
    upper: numpy.ndarray,
    n_walkers: int,
    rng: numpy.random.Generator,
    log_probability_fn: Callable[[numpy.ndarray], float],
    map_fn: Callable[..., Any] | None = None,
    unordered_map_fn: Callable[..., Any] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return initial walker positions with finite log probabilities.

    The generator gradually inflates the proposal scatter whenever the
    resulting ensemble either falls outside the declared bounds, yields
    non-finite log probabilities or remains dangerously close to a
    degenerate hyperplane.  The strategy favours uniform draws when both
    bounds are finite because those intervals already encode acceptable
    ranges.  Otherwise walkers jitter around the initial guess with adaptive
    Gaussian noise that widens on every retry.  The first walker remains
    anchored to ``initial_active`` so the sampler always includes the
    model's nominal parameter set.
    """

    ndim_active = initial_active.size
    uniform_mask = numpy.isfinite(lower) & numpy.isfinite(upper)
    width = upper - lower
    jitter = numpy.maximum(numpy.abs(initial_active), 1.0) * 1e-3
    jitter = numpy.where(
        numpy.isfinite(width), numpy.maximum(width / 10.0, jitter), jitter
    )

    def _evaluate_log_probabilities(
        proposals: numpy.ndarray,
    ) -> numpy.ndarray:
        """Evaluate proposals while reporting completed walker work."""

        total = int(len(proposals))
        if unordered_map_fn is not None:
            values = numpy.empty(total, dtype=float)
            completed = 0
            indexed_proposals = enumerate(proposals)
            for index, value in unordered_map_fn(
                _worker_indexed_log_probability,
                indexed_proposals,
                chunksize=1,
            ):
                values[int(index)] = float(value)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
            return values

        if map_fn is None:
            values = numpy.empty(total, dtype=float)
            for index, proposal in enumerate(proposals):
                values[index] = float(log_probability_fn(proposal))
                if progress_callback is not None:
                    progress_callback(index + 1, total)
            return values

        values = numpy.asarray(
            list(map_fn(log_probability_fn, proposals)), dtype=float
        )
        if progress_callback is not None:
            for completed in range(1, total + 1):
                progress_callback(completed, total)
        return values

    attempts = 0
    scatter_multiplier = 1.0
    while attempts < _MAX_INITIAL_ATTEMPTS:
        attempts += 1
        if uniform_mask.all():
            proposals = rng.uniform(
                lower, upper, size=(n_walkers, ndim_active)
            )
        else:
            noise = rng.standard_normal((n_walkers, ndim_active))
            proposals = initial_active + noise * jitter * scatter_multiplier
            proposals = numpy.clip(proposals, lower, upper)
        proposals[0] = numpy.clip(initial_active, lower, upper)

        logp = _evaluate_log_probabilities(proposals)
        if not numpy.all(numpy.isfinite(logp)):
            scatter_multiplier *= 2.0
            continue

        cond = _estimate_condition_number(proposals)
        if cond is None or cond <= _MAX_INITIAL_CONDITION:
            return proposals, logp

        scatter_multiplier *= 5.0

    raise RuntimeError(
        "Unable to initialise walkers with stable condition number"
    )
