# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Utilities for writing MCMC chains to NetCDF files.

This module stores posterior samples produced by samplers such as the
``sampler_mcmc`` backend.  Chains are saved in the NetCDF format using
ArviZ so that users can analyse results with a broad ecosystem of Bayesian
tools.  The helper aims to keep file handling consistent across samplers and to
centralise metadata attachment so future formats can be supported from one
location.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy
import xarray as xarray_module

# ArviZ expects ``scipy.signal.gaussian`` which moved in newer SciPy.
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
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised in minimal test envs
    arviz_module = None

from .logger import get_logger
from .utils import ensure_dir_exists


def save_posterior(
    chain: numpy.ndarray,
    param_names: Iterable[str],
    filepath: str,
    *,
    metadata: dict | None = None,
) -> None:
    """Persist ``chain`` and ``metadata`` to ``filepath``.

    Parameters
    ----------
    chain : ndarray
        Array of shape ``(n_steps, n_walkers, n_params)`` containing the raw
        MCMC samples.
    param_names : iterable of str
        Names of parameters corresponding to the last dimension of
        ``chain``.
    filepath : str
        Target NetCDF file path.  Parent directories are created
        automatically.
    metadata : dict, optional
        Additional attributes stored under ``InferenceData.attrs``.  These
        typically include the dataset identifier and model name so that
        provenance is fully captured inside the file.
    """

    logger = get_logger()
    ensure_dir_exists(os.path.dirname(filepath))

    # ArviZ expects arrays in ``(chain, draw, ...)`` order.  The sampler in the
    # MCMC sampler yields ``(step, walker, param)``; transpose accordingly.
    transposed = numpy.transpose(chain, (1, 0, 2))
    posterior_dict = {
        name: transposed[:, :, i] for i, name in enumerate(param_names)
    }

    metadata = metadata or {}

    if arviz_module is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            idata = arviz_module.from_dict(posterior=posterior_dict)
        if metadata:
            idata.attrs.update(metadata)
            # Persist metadata on the posterior group as well so callers that
            # open only that group still recover provenance details such as the
            # model name and dataset identifier when using NetCDF backends that
            # support grouping.
            idata.posterior.attrs.update(metadata)

        try:
            idata.to_netcdf(filepath)
            logger.info("Posterior samples saved to %s", filepath)
        except (OSError, ValueError) as exc:  # pragma: no cover
            logger.error("Failed saving posterior to %s: %s", filepath, exc)
        return

    # Minimal fallback when ArviZ is unavailable during lightweight test runs.
    coords = {
        "chain": numpy.arange(transposed.shape[0], dtype=int),
        "draw": numpy.arange(transposed.shape[1], dtype=int),
    }
    xarray_dataset = xarray_module.Dataset(
        {
            name: (("chain", "draw"), transposed[:, :, i])
            for i, name in enumerate(param_names)
        }
    )
    xarray_dataset = xarray_dataset.assign_coords(coords)
    if metadata:
        xarray_dataset.attrs.update(metadata)
    # The SciPy NetCDF backend does not support groups, so the fallback writes
    # everything to the root group and records a flag for downstream callers.
    xarray_dataset.attrs.setdefault("posterior_group", "/")

    try:
        xarray_dataset.to_netcdf(filepath)
        logger.info(
            "Posterior samples saved to %s using xarray fallback "
            "(ArviZ missing)",
            filepath,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover
        logger.error("Failed saving posterior to %s: %s", filepath, exc)


__all__ = ["save_posterior"]
