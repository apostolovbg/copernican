r"""Cosmic Microwave Background likelihood helper.

**Last Updated:** 2025-11-01

Provides cache-aware CAMB interfaces shared by the CMB likelihood and the BAO
background evaluator.  Earlier revisions duplicated CAMB configuration across
modules which let the BAO pipeline drift from the spectra settings used during
Stage 2.  The refactor below consolidates parameter normalisation, neutrino
sector handling and accuracy knobs so every observable consumes the same
cosmology and the run manifest can record the exact CAMB controls.  The spectra
returned here are expressed as :math:`D_\ell` so downstream tests comparing
against published Planck-lite tables use consistent conventions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import camb
import numpy as np
import pandas as pd

from ._protocol import LikelihoodProtocol, LikelihoodState

_C_LIGHT_KM_S = 299_792.458
_LMAX_PADDING = 300
_LENS_POTENTIAL_ACCURACY = 0
_CACHE_PRECISION = 15
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")


def _normalise_value(value: Any) -> Any:
    """Return a cache-friendly representation of ``value``."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def _normalise_items(
    param_dict: Mapping[str, Any]
) -> tuple[tuple[str, Any], ...]:
    """Convert ``param_dict`` into a deterministic tuple of items."""

    normalised: list[tuple[str, Any]] = []
    for key in sorted(param_dict):
        normalised.append((str(key), _normalise_value(param_dict[key])))
    return tuple(normalised)


def _restore_dict(items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Rehydrate a mapping created by :func:`_normalise_items`."""

    restored: dict[str, Any] = {}
    for key, value in items:
        restored[str(key)] = value
    return restored


def _make_camb_params(
    param_dict: Mapping[str, Any], *, lmax: int | None = None
) -> camb.CAMBparams:
    """Return CAMB parameters mirroring the engine reference implementation."""

    params = camb.CAMBparams()
    cosmo_kwargs: dict[str, Any] = {}
    # CAMB insists on either ``H0`` or an angular scale parameter.  We mirror
    # the defaults used in ``_cached_cmb`` so helper calls remain backward
    # compatible when optional keys are omitted from ``param_dict``.
    cosmo_kwargs["H0"] = float(param_dict.get("H0", 67.5))
    cosmo_kwargs["ombh2"] = float(param_dict.get("ombh2", 0.022))
    cosmo_kwargs["omch2"] = float(param_dict.get("omch2", 0.12))
    if "tau" in param_dict:
        cosmo_kwargs["tau"] = float(param_dict["tau"])
    else:
        cosmo_kwargs["tau"] = float(0.06)
    if "omk" in param_dict:
        cosmo_kwargs["omk"] = float(param_dict["omk"])
    if "YHe" in param_dict:
        cosmo_kwargs["YHe"] = float(param_dict["YHe"])
    if "theta_H0_range" in param_dict:
        theta_range = param_dict["theta_H0_range"]
        cosmo_kwargs["theta_H0_range"] = tuple(
            float(value) for value in np.atleast_1d(theta_range)[:2]
        )

    # Translate high-level neutrino controls into the names CAMB expects.  The
    # helper accepts both the effective relativistic degrees of freedom and the
    # standard reference value so users can keep oscillation-motivated deltas
    # explicit in their parameter maps.
    if "Neff" in param_dict:
        cosmo_kwargs["nnu"] = float(param_dict["Neff"])
    if "standard_neutrino_neff" in param_dict:
        cosmo_kwargs["standard_neutrino_neff"] = float(
            param_dict["standard_neutrino_neff"]
        )
    if "num_massive_neutrinos" in param_dict:
        cosmo_kwargs["num_massive_neutrinos"] = int(
            float(param_dict["num_massive_neutrinos"])
        )
    if "neutrino_hierarchy" in param_dict:
        cosmo_kwargs["neutrino_hierarchy"] = param_dict["neutrino_hierarchy"]

    # The YAML layer lets models expose individual mass eigenstates via keys
    # such as ``mnu1`` and ``mnu2``.  CAMB only receives the summed mass, so we
    # aggregate the ordered entries before forwarding them.  When a direct
    # ``sum_mnu`` mapping is supplied it overrides the individual masses, while
    # ``mnu`` remains available for historical parameterisations.
    dynamic_mass_keys = [
        key for key in param_dict if _MNU_PATTERN.match(str(key))
    ]
    if dynamic_mass_keys:
        ordered = sorted(
            dynamic_mass_keys,
            key=lambda item: int(_MNU_PATTERN.match(str(item)).group(1)),
        )
        masses = [float(param_dict[key]) for key in ordered]
        cosmo_kwargs.setdefault("num_massive_neutrinos", len(masses))
        cosmo_kwargs["mnu"] = float(np.sum(masses))
    if "sum_mnu" in param_dict:
        cosmo_kwargs["mnu"] = float(param_dict["sum_mnu"])
    elif "mnu" in param_dict:
        cosmo_kwargs["mnu"] = float(param_dict["mnu"])

    params.set_cosmology(**cosmo_kwargs)
    if "omnuh2" in param_dict:
        params.omnuh2 = float(param_dict["omnuh2"])

    accuracy = getattr(params, "Accuracy", None)
    if accuracy is not None:
        if "AccuracyBoost" in param_dict:
            accuracy.AccuracyBoost = float(param_dict["AccuracyBoost"])
        if "lAccuracyBoost" in param_dict:
            accuracy.LAccuracyBoost = float(param_dict["lAccuracyBoost"])
        if "kAccuracyBoost" in param_dict:
            accuracy.KAccuracyBoost = float(param_dict["kAccuracyBoost"])

    if lmax is not None:
        params.set_for_lmax(
            int(lmax) + _LMAX_PADDING,
            lens_potential_accuracy=_LENS_POTENTIAL_ACCURACY,
        )

    power_kwargs: dict[str, Any] = {}
    if "As" in param_dict:
        power_kwargs["As"] = float(param_dict["As"])
    if "ns" in param_dict:
        power_kwargs["ns"] = float(param_dict["ns"])
    if "nrun" in param_dict:
        power_kwargs["nrun"] = float(param_dict["nrun"])
    if "nrunrun" in param_dict:
        power_kwargs["nrunrun"] = float(param_dict["nrunrun"])
    if "r" in param_dict:
        power_kwargs["r"] = float(param_dict["r"])
    if power_kwargs:
        params.InitPower.set_params(**power_kwargs)
    return params


@lru_cache(maxsize=128)
def _cached_cmb(
    key: tuple[str, tuple[tuple[str, Any], ...], int, tuple[str, ...]]
):
    """Return unlensed CAMB spectra for a given cache key."""

    _, items, lmax, spectra = key
    param_dict = _restore_dict(items)
    params = _make_camb_params(param_dict, lmax=int(lmax))
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


@lru_cache(maxsize=128)
def _cached_background(
    key: tuple[str, tuple[tuple[str, Any], ...], tuple[float, ...]]
) -> tuple[
    float,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return cached CAMB background observables for ``key``."""

    _, items, z_tuple = key
    param_dict = _restore_dict(items)
    params = _make_camb_params(param_dict, lmax=None)
    results = camb.get_results(params)
    derived = results.get_derived_params()
    rs_drag = float(derived.get("rdrag", float("nan")))

    z_arr = np.asarray(z_tuple, dtype=float)
    dm_vals: list[float] = []
    da_vals: list[float] = []
    hz_vals: list[float] = []
    for z_val in z_arr:
        dm_vals.append(float(results.comoving_radial_distance(float(z_val))))
        da_vals.append(float(results.angular_diameter_distance(float(z_val))))
        hz_vals.append(float(results.hubble_parameter(float(z_val))))

    dm = np.asarray(dm_vals, dtype=float)
    da = np.asarray(da_vals, dtype=float)
    hz = np.asarray(hz_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dh = np.where(np.abs(hz) > 1e-12, _C_LIGHT_KM_S / hz, np.nan)
    term = dm * dm
    term *= z_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        term = term * dh
    dv = np.full_like(term, np.nan, dtype=float)
    mask = np.isfinite(term) & (term >= 0.0)
    dv[mask] = np.power(term[mask], 1.0 / 3.0)
    zero = np.isfinite(term) & (z_arr == 0.0)
    dv[zero] = 0.0

    return (
        rs_drag,
        tuple(dm.tolist()),
        tuple(dh.tolist()),
        tuple(da.tolist()),
        tuple(dv.tolist()),
        tuple(hz.tolist()),
    )


def compute_camb_background_observables(
    param_dict: Mapping[str, Any], redshifts: Sequence[float]
) -> dict[str, np.ndarray]:
    """Return CAMB background quantities for ``redshifts``.

    The helper shares the same caching layer as the spectrum generator so
    BAO evaluations reuse cosmologies computed for the CMB likelihood.
    """

    z_arr = np.asarray(redshifts, dtype=float)
    z_tuple = tuple(
        float(f"{float(val):.{_CACHE_PRECISION}g}") for val in z_arr
    )
    items = _normalise_items(param_dict)
    rs_drag, dm, dh, da, dv, hz = _cached_background(
        ("background", items, z_tuple)
    )
    return {
        "rs_drag": float(rs_drag),
        "DM": np.asarray(dm, dtype=float),
        "DH": np.asarray(dh, dtype=float),
        "DA": np.asarray(da, dtype=float),
        "DV": np.asarray(dv, dtype=float),
        "Hz": np.asarray(hz, dtype=float),
        "z": np.asarray(z_tuple, dtype=float),
    }


def describe_camb_configuration() -> dict[str, Any]:
    """Return the default CAMB configuration used by the likelihood helpers."""

    params = camb.CAMBparams()
    accuracy = getattr(params, "Accuracy", None)
    accuracy_info: dict[str, float] = {}
    if accuracy is not None:
        accuracy_info = {
            "AccuracyBoost": float(getattr(accuracy, "AccuracyBoost", 1.0)),
            "LAccuracyBoost": float(getattr(accuracy, "LAccuracyBoost", 1.0)),
            "KAccuracyBoost": float(getattr(accuracy, "KAccuracyBoost", 1.0)),
        }

    return {
        "lmax_padding": _LMAX_PADDING,
        "lens_potential_accuracy": _LENS_POTENTIAL_ACCURACY,
        "reionization_model": "optical_depth_tau",
        # tau-based parameterisation
        "accuracy": accuracy_info,
    }


def compute_cmb_spectrum_from_dict(
    param_dict: Mapping[str, float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
) -> np.ndarray | Mapping[str, np.ndarray]:
    r"""Return theoretical :math:`D_\ell` spectra using CAMB with caching."""

    logger = logging.getLogger()
    try:
        items = _normalise_items(param_dict)
        lmax = int(np.max(list(ells)))
        cache_key = ("dict", items, lmax, tuple(sorted(spectra)))
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
    _ells: np.ndarray = field(init=False, repr=False)
    _observed: np.ndarray = field(init=False, repr=False)
    _cov_inv: np.ndarray | None = field(init=False, repr=False)
    _residual_buffer: np.ndarray = field(init=False, repr=False)
    _extra_params_cached: dict[str, float] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _setup_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Extract immutable arrays so log-likelihood evaluation stays lean."""

        df = self.data
        if df is None or df.empty:
            self._setup_error = "(cmb_like): CMB data is empty."
            self._ells = np.empty(0, dtype=int)
            self._observed = np.empty(0, dtype=float)
            self._cov_inv = None
            self._residual_buffer = np.empty(0, dtype=float)
            return

        self._ells = df["ell"].to_numpy(dtype=int, copy=True)
        self._observed = df["Dl_obs"].to_numpy(dtype=float, copy=True)
        if np.any(~np.isfinite(self._observed)):
            self._setup_error = (
                "(cmb_like): Observed spectrum contains non-finite values."
            )

        cov_attr = df.attrs.get("covariance_matrix_inv")
        self._cov_inv = (
            None if cov_attr is None else np.asarray(cov_attr, dtype=float)
        )
        if self._cov_inv is None:
            self._setup_error = (
                "(cmb_like): Missing inverse covariance matrix."
            )

        self._residual_buffer = np.empty_like(self._observed, dtype=float)

        if self.extra_params:
            cached: dict[str, float] = {}
            for key, value in self.extra_params.items():
                cached[str(key)] = float(value)
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
            camb_params = self.plugin.get_camb_params(params)
        except Exception:
            self._state = LikelihoodState()
            return float("-inf")

        if not isinstance(camb_params, Mapping):
            self._state = LikelihoodState()
            return float("-inf")

        params_dict = {
            str(key): float(val) for key, val in camb_params.items()
        }
        if self._extra_params_cached:
            params_dict.update(self._extra_params_cached)

        theory = compute_cmb_spectrum_from_dict(
            params_dict,
            self._ells,
            spectra=("TT",),
        )
        if not isinstance(theory, np.ndarray):
            theory = np.asarray(theory, dtype=float)
        if theory.shape != self._observed.shape or np.any(
            ~np.isfinite(theory)
        ):
            self._state = LikelihoodState()
            return float("-inf")

        np.subtract(
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
    "compute_cmb_spectrum_from_dict",
]
