# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Utilities for writing MCMC chains to NetCDF files.

This module stores posterior samples produced by engines such as the
``cosmo_engine_mcmc`` backend.  Chains are saved in the NetCDF format using
ArviZ so that users can analyse results with a broad ecosystem of Bayesian
tools.  The helper aims to keep file handling consistent across engines and to
centralise metadata attachment so future formats can be supported from one
location.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import xarray as xr

# ArviZ expects ``scipy.signal.gaussian`` which moved in newer SciPy.
try:  # pragma: no cover - compatibility shim
    from scipy.signal import gaussian  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - SciPy layout varies
    try:
        import scipy.signal as _signal
        from scipy.signal.windows import gaussian  # type: ignore # noqa: F401

        _signal.gaussian = gaussian
    except Exception:  # pragma: no cover
        pass

try:
    import arviz as az
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised in minimal test envs
    az = None

from .logger import get_logger
from .utils import ensure_dir_exists

def save_posterior(
    chain: np.ndarray,
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
    # MCMC engine yields ``(step, walker, param)`` so we transpose accordingly.
    transposed = np.transpose(chain, (1, 0, 2))
    posterior_dict = {
        name: transposed[:, :, i] for i, name in enumerate(param_names)
    }

    metadata = metadata or {}

    if az is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            idata = az.from_dict(posterior=posterior_dict)
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
        except Exception as exc:  # pragma: no cover - file errors uncommon
            logger.error("Failed saving posterior to %s: %s", filepath, exc)
        return

    # Minimal fallback when ArviZ is unavailable during lightweight test runs.
    coords = {
        "chain": np.arange(transposed.shape[0], dtype=int),
        "draw": np.arange(transposed.shape[1], dtype=int),
    }
    dataset = xr.Dataset(
        {
            name: (("chain", "draw"), transposed[:, :, i])
            for i, name in enumerate(param_names)
        }
    )
    dataset = dataset.assign_coords(coords)
    if metadata:
        dataset.attrs.update(metadata)
    # The SciPy NetCDF backend does not support groups, so the fallback writes
    # everything to the root group and records a flag for downstream callers.
    dataset.attrs.setdefault("posterior_group", "/")

    try:
        dataset.to_netcdf(filepath)
        logger.info(
            "Posterior samples saved to %s using xarray fallback "
            "(ArviZ missing)",
            filepath,
        )
    except Exception as exc:  # pragma: no cover - file errors uncommon
        logger.error("Failed saving posterior to %s: %s", filepath, exc)

__all__ = ["save_posterior"]
