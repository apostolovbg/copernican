"""Cosmic Microwave Background likelihood helper.

**Last Updated:** 2025-02-14

Provides the CAMB spectrum wrappers and covariance-aware χ² evaluation for the
Planck lite dataset as well as future CMB releases.  The helper mirrors the
previous logic from :mod:`copernican_lib.statistics` so external APIs remain
stable while consolidating likelihood behaviour inside this package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import camb
import numpy as np
import pandas as pd

from ._protocol import LikelihoodProtocol, LikelihoodState


@lru_cache(maxsize=128)
def _cached_cmb(
    key: tuple[str, tuple[tuple[str, float], ...], int, tuple[str, ...]]
):
    """Return unlensed CAMB spectra for a given cache key."""

    _, param_tuple, lmax, spectra = key
    param_dict = dict(param_tuple)
    params = camb.CAMBparams()
    params.set_cosmology(
        H0=param_dict["H0"],
        ombh2=param_dict["ombh2"],
        omch2=param_dict["omch2"],
        tau=param_dict["tau"],
    )
    params.omnuh2 = param_dict.get("omnuh2", 0.0)
    params.InitPower.set_params(As=param_dict["As"], ns=param_dict["ns"])
    params.set_for_lmax(lmax + 300, lens_potential_accuracy=0)
    results = camb.get_results(params)
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    out: dict[str, np.ndarray] = {}
    if "TT" in spectra:
        out["TT"] = cls[:, 0]
    if "EE" in spectra:
        out["EE"] = cls[:, 1]
    if "TE" in spectra:
        out["TE"] = cls[:, 3]
    return out


def compute_cmb_spectrum_from_dict(
    param_dict: Mapping[str, float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> np.ndarray | Mapping[str, np.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using CAMB with caching."""

    logger = logging.getLogger()
    try:
        pairs: list[tuple[str, float]] = []
        for key, value in sorted(param_dict.items()):
            pairs.append((key, float(f"{float(value):.6g}")))
        key_tuple = tuple(pairs)
        lmax = int(np.max(list(ells)))
        cache_key = ("dict", key_tuple, lmax, tuple(sorted(spectra)))
        full = _cached_cmb(cache_key)
    except Exception as exc:  # pragma: no cover - camb errors are logged
        logger.error("(compute_cmb_spectrum_from_dict): %s", exc)
        return np.full_like(np.asarray(list(ells)), np.nan, dtype=float)

    ell_arr = np.asarray(list(ells), dtype=int)
    result = {spec: full[spec][ell_arr] for spec in spectra}
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def compute_cmb_spectrum_cached(
    plugin: Any,
    cosmo_params: Sequence[float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> np.ndarray | Mapping[str, np.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    logger = logging.getLogger()
    try:
        camb_params = plugin.get_camb_params(cosmo_params)
    except Exception as exc:
        logger.error("(compute_cmb_spectrum_cached): %s", exc)
        return np.full_like(np.asarray(list(ells)), np.nan, dtype=float)

    return compute_cmb_spectrum_from_dict(camb_params, ells, spectra=spectra)


def compute_cmb_spectrum(
    param_dict: Mapping[str, float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> np.ndarray | Mapping[str, np.ndarray]:
    r"""Backward-compatible wrapper accepting a CAMB parameter dictionary."""

    dummy = type(
        "_Dummy",
        (),
        {
            "MODEL_NAME": "direct",
            "get_camb_params": lambda self, _: param_dict,
        },
    )()
    return compute_cmb_spectrum_cached(dummy, [], ells, spectra=spectra)


@dataclass(slots=True)
class CMBLike(LikelihoodProtocol):
    """Evaluate CMB log-likelihoods for tabulated spectra."""

    data: pd.DataFrame
    plugin: Any
    extra_params: Mapping[str, float] | None = None
    enabled: bool = True
    _state: LikelihoodState = field(
        default_factory=LikelihoodState,
        init=False,
    )

    def loglike(self, params: Sequence[float]) -> float:
        """Return the CMB log-likelihood for ``params``."""

        logger = logging.getLogger()
        if not self.enabled:
            self._state = LikelihoodState(chi2=0.0, loglike=0.0)
            return 0.0

        if self.data is None or self.data.empty:
            logger.error("(cmb_like): CMB data is empty.")
            self._state = LikelihoodState()
            return float("-inf")
        if "covariance_matrix_inv" not in self.data.attrs:
            logger.error("(cmb_like): Missing inverse covariance matrix.")
            self._state = LikelihoodState()
            return float("-inf")

        ells = self.data["ell"].to_numpy(dtype=int)
        obs = self.data["Dl_obs"].to_numpy(dtype=float)
        try:
            camb_params = self.plugin.get_camb_params(params)
        except Exception:
            self._state = LikelihoodState()
            return float("-inf")
        if self.extra_params:
            camb_params.update(self.extra_params)

        theory = compute_cmb_spectrum_from_dict(
            camb_params, ells, spectra=("TT",)
        )
        if not isinstance(theory, np.ndarray):
            theory = np.asarray(theory, dtype=float)
        if theory.shape != obs.shape or np.any(~np.isfinite(theory)):
            self._state = LikelihoodState()
            return float("-inf")

        resid = obs - theory
        cov_inv = self.data.attrs["covariance_matrix_inv"]
        try:
            chi2 = float(resid @ cov_inv @ resid)
        except Exception as exc:
            logger.error("(cmb_like): Linear algebra failure: %s", exc)
            self._state = LikelihoodState()
            return float("-inf")

        loglike = -0.5 * chi2 if np.isfinite(chi2) else float("-inf")
        self._state = LikelihoodState(
            chi2=chi2,
            loglike=loglike,
            metadata={
                "covariance": "full",
                "points": int(resid.size),
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
    "compute_cmb_spectrum_from_dict",
]
