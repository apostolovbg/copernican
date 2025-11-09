# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Last Updated: 2025-11-07

"""Markov Chain Monte Carlo engine using :mod:`emcee`.

**Last Updated:** 2025-11-09

The combined optimiser has been retired entirely, leaving this sampler as the
sole runtime engine.  It continues to focus on Supernova Ia posteriors while
delegating shared χ² helpers to :mod:`copernican_lib.statistics` so the module
acts as the canonical engine façade.  Future backends can slot in beside it
without changing the orchestration code.  Verbose progress logging tracks both
burn-in and production phases with percentage updates so long chains always
report their status.  Version 6.2.0 routes all likelihood evaluations through
the :class:`copernican_lib.likelihoods.JointLike` aggregator and the
new :func:`copernican_lib.engine_interface.make_logposterior` helper so that
posterior calculations automatically honour per-parameter priors, declared
bounds and optional reparameterisation transforms while exposing diagnostic
metadata alongside sampled chains.

Version 7.2.10 extends the reproducibility contract by constructing every
NumPy :class:`~numpy.random.Generator` from the shared
:func:`copernican_lib.utils.get_random_seed` value.  The helper captures the
seed supplied through the CLI prompt or ``COPERNICAN_SEED`` so subsequent
engines observe the same pseudo-random stream without requiring callers to
seed multiple subsystems manually.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import textwrap
import warnings
from typing import Any, Callable, Iterable, Sequence

import arviz as az
import emcee
import numpy as np
import pandas as pd

from copernican_lib import console_output as console
from copernican_lib import engine_interface
from copernican_lib.likelihoods import BAOLike, CMBLike, JointLike, SNeLike
from copernican_lib.statistics import (
    calculate_bao_observables,
    chi_squared_bao,
    chi_squared_cmb,
    chi_squared_sne,
    compute_cmb_spectrum,
    compute_cmb_spectrum_from_dict,
)
from copernican_lib.utils import get_random_seed

warnings.filterwarnings(
    "ignore",
    message=r"More chains \(\d+\) than draws \(\d+\)",
    module=r"arviz\\.data\\.base",
    category=UserWarning,
)

ENGINE_KIND = "mcmc"
ENGINE_LABEL = "Ensemble MCMC sampler"

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
        posterior: Callable[[np.ndarray], float],
        template_params: np.ndarray,
        active_indices: np.ndarray,
    ) -> None:
        # ``posterior`` already encapsulates priors and likelihood terms via
        # ``_build_sne_logposterior``.  We retain it verbatim and only manage
        # the vector assembly around it.
        self._posterior = posterior
        # ``template_params`` stores the baseline parameter vector with fixed
        # entries included.  Copy and coerce it to ``float`` so all workers see
        # a consistent array and accidental mutations never bleed between
        # processes.
        self._template = np.asarray(template_params, dtype=float)
        # ``active_indices`` pinpoints which coordinates ``emcee`` manipulates.
        self._active_indices = np.asarray(active_indices, dtype=int)

    def assemble_full(self, position: np.ndarray) -> np.ndarray:
        """Return a full parameter vector for ``position``.

        The method centralises the reconstruction so tests can assert that
        fixed coordinates survive untouched when the adapter is invoked
        directly.  Each call returns a new array, keeping ``self._template``
        immutable for safe multiprocessing pickling.
        """

        full = self._template.copy()
        full[self._active_indices] = np.asarray(position, dtype=float)
        return full

    def __call__(self, position: np.ndarray) -> float:
        """Evaluate the posterior for ``position`` in the active subspace."""

        full = self.assemble_full(position)
        value = self._posterior(full)
        # ``emcee`` expects a Python ``float`` rather than a NumPy scalar for
        # predictable ``isfinite`` checks.  Coercing here guarantees downstream
        # consumers see the same type under both serial and multiprocessing
        # execution.
        return float(value)


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
        self._joint_like = joint_like
        self.parameter_bounds = list(parameter_bounds or [])
        if parameter_transforms is not None:
            self.parameter_transforms = list(parameter_transforms)

    def __call__(self, params: Sequence[float]) -> float:
        """Return the combined log-likelihood for ``params``."""

        return float(self._joint_like.loglike(params))


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
        template_params: np.ndarray,
        active_indices: np.ndarray,
        *,
        progress_granularity: int = 20,
        max_params_to_show: int | None = None,
    ) -> None:
        self._param_names = list(param_names)
        self._template = np.asarray(template_params, dtype=float)
        self._active_indices = np.asarray(active_indices, dtype=int)
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
        self._scratch: np.ndarray | None = None
        self._wrap_width = 72

    def __call__(
        self,
        step_index: int,
        state: emcee.State,
    ) -> Sequence[str]:
        """Return formatted diagnostics for ``state`` at ``step_index``."""

        del step_index  # The index is tracked via ``_report_count``.
        self._report_count += 1

        coords = np.asarray(state.coords, dtype=float)
        log_prob = np.asarray(state.log_prob, dtype=float)

        lines: list[str] = []
        finite_mask = np.isfinite(log_prob)
        if np.any(finite_mask):
            finite = log_prob[finite_mask]
            mean_lp = float(np.mean(finite))
            std_lp = float(np.std(finite))
            max_lp = float(np.max(finite))
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
            coords = coords[np.newaxis, :]

        n_walkers = coords.shape[0]
        if n_walkers == 0:
            return lines

        expanded = self._expand_coordinates(coords)

        for idx, name in enumerate(
            self._param_names[: self._max_params_to_show]
        ):
            values = expanded[:, idx]
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size == 0:
                lines.append(
                    f"    {name}: statistics unavailable (non-finite samples)."
                )
                continue
            q16, q50, q84 = np.percentile(finite_vals, [16.0, 50.0, 84.0])
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

        if np.any(finite_mask) and (
            self._report_count == 1
            or self._report_count % self._sample_interval == 0
        ):
            walker_idx = int(np.argmax(log_prob[finite_mask]))
            full_idx = np.flatnonzero(finite_mask)[walker_idx]
            snapshot_vals = expanded[full_idx]
            snapshot_pairs = [
                f"{name}={snapshot_vals[idx]:.4g}"
                for idx, name in enumerate(
                    self._param_names[: self._max_params_to_show]
                )
            ]
            snapshot_text = ", ".join(snapshot_pairs)
            lines.append(
                textwrap.fill(
                    snapshot_text,
                    width=self._wrap_width,
                    initial_indent=f"    Walker[{full_idx}] snapshot: ",
                    subsequent_indent=" " * 8,
                )
            )

        return lines

    def _expand_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Return full parameter coordinates for ``coords`` walkers."""

        if coords.ndim == 1:
            coords = coords[np.newaxis, :]

        n_walkers = coords.shape[0]
        n_params = self._template.size
        if self._scratch is None or self._scratch.shape != (
            n_walkers,
            n_params,
        ):
            self._scratch = np.broadcast_to(
                self._template, (n_walkers, n_params)
            ).copy()
        else:
            self._scratch[...] = self._template

        if coords.size:
            self._scratch[:, self._active_indices] = coords

        return self._scratch


def _build_sne_logposterior(
    model_plugin: Any,
    sne_data_df: Any,
    bao_data_df: Any | None = None,
    cmb_data_df: Any | None = None,
) -> tuple[
    Callable[[Sequence[float]], float],
    Callable[[Sequence[float]], float],
    JointLike,
]:
    """Return posterior, likelihood and diagnostics for joint datasets.

    Engines evaluate the returned posterior repeatedly during sampling.  The
    helper therefore pre-computes the reusable :class:`JointLike` aggregator
    once, attaches the plugin's bounds and optional transformations to the
    underlying log-likelihood callable and finally hands everything to
    :func:`engine_interface.make_logposterior` so priors and Jacobian
    adjustments remain consistent across engines.
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
        np.asarray(bao_z if bao_z is not None else [], dtype=float),
        np.asarray(bao_types if bao_types is not None else [], dtype=object),
        np.asarray(bao_val if bao_val is not None else [], dtype=float),
        np.asarray(bao_err if bao_err is not None else [], dtype=float),
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
        cmb_data_df if cmb_data_df is not None else pd.DataFrame(),
        model_plugin,
        enabled=cmb_enabled,
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
    posterior = engine_interface.make_logposterior(loglike, priors)
    return posterior, loglike, joint_like


def _reseed_invalid_walkers(
    coords: np.ndarray,
    log_prob: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    log_probability_fn: Callable[[np.ndarray], float],
    reference_position: np.ndarray | None = None,
    max_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
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
    coords = np.asarray(coords, dtype=float).copy()
    log_prob = np.asarray(log_prob, dtype=float).copy()

    invalid = (~np.isfinite(coords).all(axis=1)) | (~np.isfinite(log_prob))
    if not np.any(invalid):
        return coords, log_prob

    logger.warning(
        "Detected %d invalid walkers after burn-in; reseeding them.",
        int(np.sum(invalid)),
    )

    valid_coords = coords[~invalid]
    if valid_coords.size == 0:
        if reference_position is None:
            raise RuntimeError("No baseline available for reseeding walkers.")
        valid_coords = np.asarray(reference_position, dtype=float)[None, :]

    centre = np.mean(valid_coords, axis=0)
    spread = np.std(valid_coords, axis=0)
    finite_width = np.where(
        np.isfinite(lower) & np.isfinite(upper), upper - lower, np.nan
    )
    fallback = np.where(np.isfinite(finite_width), finite_width / 6.0, 1.0)
    spread = np.where(spread > 0, spread, fallback)
    spread = np.where(spread > 0, spread, 1.0)

    bad_idx = np.flatnonzero(invalid)
    attempts = 0
    while bad_idx.size and attempts < max_attempts:
        attempts += 1
        jitter = rng.standard_normal((bad_idx.size, centre.size))
        proposals = centre + jitter * np.maximum(spread, 1e-3)
        proposals = np.clip(proposals, lower, upper)
        new_log_prob = np.array([log_probability_fn(pos) for pos in proposals])
        finite = np.isfinite(new_log_prob)
        coords[bad_idx[finite]] = proposals[finite]
        log_prob[bad_idx[finite]] = new_log_prob[finite]
        bad_idx = bad_idx[~finite]

    if bad_idx.size:
        raise RuntimeError(
            "Unable to reseed %d walkers with finite log probability"
            % bad_idx.size
        )

    return coords, log_prob


class _BatchProgressBar:
    """Render textual progress bars for sampler batches.

    Stage 2 reports sampler progress in fixed-size batches so long runs do
    not spam the console. This helper draws a bar that fills gradually to
    100% for each batch, prints an empty line once the batch is complete and
    then announces the next batch. The spacing mirrors the interactive design
    brief from the orchestrator so operators can see how far the current
    batch has advanced before the next update arrives.
    """

    _BAR_WIDTH = 28

    def __init__(
        self,
        stage_label: str,
        total_steps: int,
        *,
        display: bool = True,
    ) -> None:
        self._stage_label = stage_label
        self._total_steps = max(int(total_steps), 0)
        self._display = bool(display and self._total_steps > 0)
        self._batch_index = 0
        self._current_start = 1
        self._current_end = 0
        self._active = False
        self._last_percent = -1

    def start_batch(self, batch_start: int, batch_end: int) -> None:
        """Announce a new batch spanning ``batch_start`` to ``batch_end``."""

        if not self._display or batch_end < batch_start:
            self._active = False
            return
        self._batch_index += 1
        self._current_start = int(batch_start)
        self._current_end = int(batch_end)
        self._active = True
        self._last_percent = -1
        span = self._current_end - self._current_start + 1
        console.write(
            f"{self._stage_label} batch {self._batch_index} "
            f"({span} step(s)) progress:"
        )

    def update(self, step_index: int) -> None:
        """Update the bar to reflect ``step_index`` progress."""

        if not self._active or not self._display:
            return
        batch_size = self._current_end - self._current_start + 1
        if batch_size <= 0:
            return
        completed = max(0, step_index - self._current_start + 1)
        completed = min(completed, batch_size)
        fraction = completed / batch_size
        percent = int(round(fraction * 100))
        if percent == self._last_percent and completed < batch_size:
            return
        self._last_percent = percent
        filled = int(round(fraction * self._BAR_WIDTH))
        filled = max(0, min(self._BAR_WIDTH, filled))
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        remaining = max(self._current_end - step_index, 0)
        # ``console.write`` flushes after every call, so emitting a carriage
        # return keeps the latest bar on a single line while it fills.
        line = (
            f"\r[{bar}] {percent:>3d}% "
            f"(batch {completed}/{batch_size}, {remaining} step(s) remaining)"
        )
        console.write(line, end="")

    def finish_batch(self) -> None:
        """Close the current batch, inserting required spacing."""

        if not self._active or not self._display:
            return
        self._active = False
        if self._last_percent >= 0:
            console.write("")
        console.write("")
        self._last_percent = -1


def _run_stage_with_progress(
    sampler: emcee.EnsembleSampler,
    initial_state: np.ndarray,
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
    progress_bar = _BatchProgressBar(label, n_steps, display=display_progress)
    interval = max(1, n_steps // progress_granularity)
    batch_start = 1
    batch_end = min(interval, n_steps)
    progress_bar.start_batch(batch_start, batch_end)

    state = None
    for idx, state in enumerate(
        sampler.sample(initial_state, iterations=n_steps, progress=False),
        start=1,
    ):
        progress_bar.update(idx)
        if idx == batch_end and idx < n_steps:
            progress_bar.finish_batch()
            batch_start = idx + 1
            batch_end = min(batch_start + interval - 1, n_steps)
            progress_bar.start_batch(batch_start, batch_end)
        elif idx == n_steps:
            progress_bar.finish_batch()

        if idx == 1 or idx % interval == 0 or idx == n_steps:
            percent = int(round(idx / n_steps * 100))
            logger.info(
                "MCMC %s progress: %3d%% (%d/%d steps)",
                stage_name,
                percent,
                idx,
                n_steps,
            )
            if summary_callback is not None:
                for line in summary_callback(idx, state):
                    logger.info("%s", line)

    if state is None:
        raise RuntimeError("Sampler produced no states during %s" % stage_name)

    logger.info("Completed MCMC %s stage.", stage_name)
    return state


def fit_sne_parameters(
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
    console progress bar when ``False`` so automated runtime estimations can
    execute quietly.
    """

    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)

    posterior_full, loglike_full, joint_like = _build_sne_logposterior(
        model_plugin,
        sne_data_df,
        bao_data_df,
        cmb_data_df,
    )
    names: Iterable[str] = getattr(model_plugin, "PARAMETER_NAMES", [])
    names = list(names)
    initial = np.asarray(getattr(model_plugin, "INITIAL_GUESSES", []), float)
    bounds = list(getattr(model_plugin, "PARAMETER_BOUNDS", []))

    ndim_total = len(initial)
    if ndim_total == 0 or len(bounds) != ndim_total:
        logger.error("Model plugin missing parameter definitions")
        return {"success": False, "samples": None}

    try:
        lower_all, upper_all, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logger
        )
    except ValueError:
        return {"success": False, "samples": None}
    active_mask = ~fixed_mask
    active_indices = np.flatnonzero(active_mask)
    fixed_indices = np.flatnonzero(fixed_mask)

    if active_indices.size == 0:
        logger.error("All parameters are fixed; cannot run the sampler.")
        return {"success": False, "samples": None}

    if fixed_indices.size:
        fixed_names = ", ".join(names[idx] for idx in fixed_indices)
        logger.info(
            "Treating %d parameter(s) as fixed or numerically locked: %s",
            int(fixed_indices.size),
            fixed_names,
        )

    template_params = np.clip(initial, lower_all, upper_all)
    initial_active = template_params[active_indices]
    lower = lower_all[active_indices]
    upper = upper_all[active_indices]

    seed = get_random_seed()
    if seed is None:
        seed = 0
    rng = np.random.default_rng(seed)
    logger.debug(
        "Initialising sampler RNG with seed %s for deterministic chains.",
        seed,
    )

    ndim_active = active_indices.size
    requested_pool = pool_size if pool_size not in (None, 0) else None

    # ``emcee`` requires at least ``2 * ndim`` walkers.  Honour that rule and
    # also guarantee that a user-specified worker pool never idles because
    # the ensemble is too small.
    minimum_walkers = max(2 * ndim_active, 2)
    if requested_pool is not None:
        minimum_walkers = max(minimum_walkers, int(requested_pool))

    n_walkers = max(n_walkers, minimum_walkers)
    logger.info(
        "Using %d walkers for %d active parameter(s).",
        int(n_walkers),
        int(ndim_active),
    )

    log_probability_active = _ActiveLogProbability(
        posterior_full,
        template_params,
        active_indices,
    )

    try:
        p0, logp = _initialise_active_walkers(
            initial_active,
            lower,
            upper,
            n_walkers,
            rng,
            log_probability_active,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return {"success": False, "samples": None}

    pool = None
    pool_processes = requested_pool
    if pool_processes is None:
        try:
            cpu_total = mp.cpu_count()
        except NotImplementedError:
            cpu_total = 1
        if cpu_total > 1:
            pool_processes = min(max(cpu_total - 1, 1), n_walkers)
            if pool_processes <= 1:
                pool_processes = None
        if pool_processes is not None:
            logger.info(
                "Auto-configured multiprocessing pool with %d worker(s).",
                pool_processes,
            )
    elif pool_processes > 1:
        logger.info(
            "Using requested multiprocessing pool with %d worker(s).",
            pool_processes,
        )
    else:
        pool_processes = None

    if pool_processes is not None:
        pool = mp.get_context("spawn").Pool(processes=pool_processes)
    burn_in = (
        burn_in_steps if burn_in_steps is not None else max(100, n_steps // 5)
    )
    burn_in = max(1, int(burn_in))
    try:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim_active,
            log_probability_active,
            pool=pool,
        )
        burnin_reporter = _SamplingProgressReporter(
            names,
            template_params,
            active_indices,
            progress_granularity=progress_granularity,
        )
        last = _run_stage_with_progress(
            sampler,
            p0,
            burn_in,
            stage_name="burn-in",
            logger=logger,
            progress_granularity=progress_granularity,
            summary_callback=burnin_reporter,
            progress_label=f"{model_plugin.MODEL_NAME} burn-in",
            display_progress=display_progress,
        )
        try:
            coords, log_prob = _reseed_invalid_walkers(
                last.coords,
                last.log_prob,
                lower=lower,
                upper=upper,
                rng=rng,
                log_probability_fn=log_probability_active,
                reference_position=initial_active,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return {"success": False, "samples": None}
        sampler.reset()
        production_reporter = _SamplingProgressReporter(
            names,
            template_params,
            active_indices,
            progress_granularity=progress_granularity,
        )
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
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain_active = sampler.get_chain()
    log_prob_chain = sampler.get_log_prob()
    flat_log_prob = sampler.get_log_prob(flat=True)

    n_production, n_effective_walkers, _ = chain_active.shape
    chain = np.empty(
        (n_production, n_effective_walkers, ndim_total),
        dtype=chain_active.dtype,
    )
    chain[:] = template_params
    chain[:, :, active_indices] = chain_active

    flat_chain = chain.reshape(-1, ndim_total)

    best_index = int(np.argmax(flat_log_prob))
    best_params = flat_chain[best_index]
    mean_params = np.mean(flat_chain, axis=0)

    covariance = np.cov(flat_chain, rowvar=False)
    errors = np.sqrt(np.diag(covariance))
    error_dict = {n: e for n, e in zip(names, errors)}

    fitted = {n: v for n, v in zip(names, best_params)}
    posterior_mean = {n: v for n, v in zip(names, mean_params)}

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
    reduced = chi2_best / dof if dof > 0 else np.nan

    log_prior_best = float("-inf")
    if math.isfinite(log_posterior_best) and math.isfinite(loglike_best):
        log_prior_best = log_posterior_best - loglike_best

    acceptance = sampler.acceptance_fraction
    diagnostics: dict[str, dict[str, float]] = {
        "rhat": {},
        "ess_bulk": {},
        "ess_tail": {},
    }
    try:
        # ``arviz`` expects chains ordered as ``(n_chains, n_draws, ...)``.
        # ``emcee`` stores them as ``(n_draws, n_chains, n_params)``, so swap
        # the leading axes before building the ``InferenceData`` container.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            inference_data = az.from_dict(
                posterior={"parameters": np.swapaxes(chain, 0, 1)},
                coords={"parameter": list(names)},
                dims={"parameters": ["parameter"]},
            )
            rhat_dataset = az.rhat(inference_data, method="rank")
            ess_bulk_dataset = az.ess(inference_data, method="bulk")
            ess_tail_dataset = az.ess(inference_data, method="tail")

        def _dataset_to_dict(dataset: Any) -> dict[str, float]:
            """Return scalar diagnostics keyed by parameter name."""

            series = dataset["parameters"].to_series()
            return {str(idx): float(value) for idx, value in series.items()}

        diagnostics = {
            "rhat": _dataset_to_dict(rhat_dataset),
            "ess_bulk": _dataset_to_dict(ess_bulk_dataset),
            "ess_tail": _dataset_to_dict(ess_tail_dataset),
        }
        if diagnostics["rhat"]:
            rhat_values = np.fromiter(
                diagnostics["rhat"].values(),
                dtype=float,
                count=len(diagnostics["rhat"]),
            )
            logger.info(
                "Rank-normalised R-hat summary: min=%.3f median=%.3f max=%.3f",
                float(np.min(rhat_values)),
                float(np.median(rhat_values)),
                float(np.max(rhat_values)),
            )
        if diagnostics["ess_bulk"]:
            bulk_values = np.fromiter(
                diagnostics["ess_bulk"].values(),
                dtype=float,
                count=len(diagnostics["ess_bulk"]),
            )
            tail_values = np.fromiter(
                diagnostics["ess_tail"].values(),
                dtype=float,
                count=len(diagnostics["ess_tail"]),
            )
            logger.info(
                "Effective sample sizes: bulk median=%.1f tail median=%.1f",
                float(np.median(bulk_values)),
                float(np.median(tail_values)),
            )
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.debug("Failed to compute ArviZ diagnostics: %s", exc)

    logger.info(
        "MCMC acceptance for %s: mean=%.3f, min=%.3f, max=%.3f",
        getattr(model_plugin, "MODEL_NAME", "Unknown"),
        float(np.mean(acceptance)),
        float(np.min(acceptance)),
        float(np.max(acceptance)),
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
        except Exception:
            autocorr = None
    else:
        logger.debug(
            "Skipping autocorrelation estimate: production steps %d < %d",
            int(n_production),
            int(min_autocorr_window),
        )

    return {
        "success": np.isfinite(chi2_best)
        and math.isfinite(log_posterior_best),
        "samples": chain,
        "log_probability": log_prob_chain,
        "fitted_cosmological_params": fitted,
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
    }


__all__ = [
    "ENGINE_KIND",
    "ENGINE_LABEL",
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_from_dict",
    "fit_sne_parameters",
]


def _estimate_condition_number(samples: np.ndarray) -> float | None:
    """Return the condition number of ``samples`` or ``None`` when undefined.

    ``emcee`` inspects the condition number of the initial walker ensemble to
    ensure the stretch move can generate proposals effectively.  The function
    below mirrors that logic without importing private ``emcee`` helpers so the
    engine can deliberately inflate the walker spread before the library raises
    ``ValueError``.  When the ensemble contains fewer than two walkers the
    condition number is undefined; in that situation we return ``None`` and let
    the caller continue with additional attempts.
    """

    if samples.shape[0] < 2:
        return None
    centred = samples - np.mean(samples, axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(
            centred, full_matrices=False, hermitian=False
        )[1]
    except np.linalg.LinAlgError:
        return float("inf")
    positive = singular_values[singular_values > 0]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())


def _classify_parameter_bounds(
    bounds: Iterable[tuple[float | None, float | None]],
    *,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper bounds and a mask of effectively fixed parameters.

    Each entry in ``bounds`` is converted to a floating interval.  ``None``
    values map to ``-np.inf`` or ``np.inf`` as appropriate.  Bounds where the
    upper edge falls below the lower edge signal malformed model definitions
    and trigger an error log.  Parameters whose admissible range shrinks to a
    single point—or a numerically indistinguishable sliver—are flagged as
    fixed so the active sampling subspace contains only degrees of freedom
    that ``emcee`` can explore without tripping its linear-independence
    checks.
    """

    lower = np.empty(len(bounds), dtype=float)
    upper = np.empty(len(bounds), dtype=float)
    for idx, (low, high) in enumerate(bounds):
        lower[idx] = -np.inf if low is None else float(low)
        upper[idx] = np.inf if high is None else float(high)
        if (
            np.isfinite(lower[idx])
            and np.isfinite(upper[idx])
            and upper[idx] < lower[idx]
        ):
            logger.error(
                "Parameter %d declares inverted bounds [%f, %f]",
                idx,
                lower[idx],
                upper[idx],
            )
            raise ValueError("invalid parameter bounds: lower exceeds upper")

    with np.errstate(invalid="ignore"):
        widths = upper - lower
        centres = (upper + lower) / 2.0
        scale = np.maximum(np.abs(centres), 1.0)
        threshold = scale * _FIXED_BOUNDS_RTOL + _FIXED_BOUNDS_ATOL
        fixed_mask = np.isfinite(widths) & (widths <= threshold)

    return lower, upper, fixed_mask


def _initialise_active_walkers(
    initial_active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_walkers: int,
    rng: np.random.Generator,
    log_probability_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
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
    uniform_mask = np.isfinite(lower) & np.isfinite(upper)
    width = upper - lower
    jitter = np.maximum(np.abs(initial_active), 1.0) * 1e-3
    jitter = np.where(
        np.isfinite(width), np.maximum(width / 10.0, jitter), jitter
    )

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
            proposals = np.clip(proposals, lower, upper)
        proposals[0] = np.clip(initial_active, lower, upper)

        logp = np.array([log_probability_fn(pos) for pos in proposals])
        if not np.all(np.isfinite(logp)):
            scatter_multiplier *= 2.0
            continue

        cond = _estimate_condition_number(proposals)
        if cond is None or cond <= _MAX_INITIAL_CONDITION:
            return proposals, logp

        scatter_multiplier *= 5.0

    raise RuntimeError(
        "Unable to initialise walkers with stable condition number"
    )
