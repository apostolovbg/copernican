"""Exact curved-sky CMB lensing remapping helpers.

The native declared CMB solver uses these utilities to remap unlensed
temperature and polarization spectra with a lensing-potential spectrum.
The implementation is adapted from CAMB's pure-Python correlation code.
"""

from __future__ import annotations

from typing import Iterable

import numpy
from scipy.special import lpn

_GAUSS_LEGENDRE_CACHE: dict[int, tuple[numpy.ndarray, numpy.ndarray]] = {}


def _validate_remapping_inputs(
    cls: numpy.ndarray,
    clpp: numpy.ndarray,
    *,
    lmax: int,
    lmax_lensed: int | None = None,
) -> None:
    """Validate finite, compatible surfaces before curved-sky remapping."""

    if cls.ndim != 2 or cls.shape[1] != 4:
        raise ValueError("Unlensed cls must have shape (ell, 4)")
    if clpp.ndim != 1:
        raise ValueError("Lensing clpp must be one-dimensional")
    if lmax < 2:
        raise ValueError("Lensing remapping requires lmax >= 2")
    if cls.shape[0] <= lmax or clpp.size <= lmax:
        raise ValueError(
            "Unlensed cls and lensing clpp must cover every ell through lmax"
        )
    if not numpy.all(numpy.isfinite(cls)):
        raise ValueError("Unlensed cls must contain only finite values")
    if not numpy.all(numpy.isfinite(clpp)):
        raise ValueError("Lensing clpp must contain only finite values")
    if lmax_lensed is not None and not 0 <= int(lmax_lensed) <= lmax:
        raise ValueError("lmax_lensed must lie between 0 and lmax")


def legendrep(degree: int, cosine_theta: float):
    """Return Legendre polynomials and derivatives up to ``degree``."""

    return lpn(int(degree), float(cosine_theta))


def _cached_gauss_legendre(
    npoints: int,
    cache: bool = True,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return Gauss-Legendre nodes and weights for ``npoints`` samples."""

    if cache and npoints in _GAUSS_LEGENDRE_CACHE:
        return _GAUSS_LEGENDRE_CACHE[npoints]
    xvals, weights = numpy.polynomial.legendre.leggauss(int(npoints))
    xvals = numpy.asarray(xvals, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    xvals.flags.writeable = False
    weights.flags.writeable = False
    if cache:
        _GAUSS_LEGENDRE_CACHE[npoints] = xvals, weights
    return xvals, weights


def legendre_funcs(
    lmax: int,
    x: float,
    modes: tuple[int, ...] = (0, 2),
    lfacs: numpy.ndarray | None = None,
    lfacs2: numpy.ndarray | None = None,
    lrootfacs: numpy.ndarray | None = None,
):
    """Return the Legendre and spin-weighted Wigner-d helper arrays."""

    allp, alldp = legendrep(int(lmax), float(x))
    fac1 = 1.0 - float(x)
    fac2 = 1.0 + float(x)
    res = []
    if 0 in modes:
        res.append((allp, alldp))

    if 1 in modes:
        lfacs1 = numpy.arange(1, int(lmax) + 1, dtype=float)
        lfacs1 *= 1.0 + lfacs1
        d11 = fac1 * alldp[1:] / lfacs1 + allp[1:]
        dm11 = fac2 * alldp[1:] / lfacs1 - allp[1:]
        res.append((d11, dm11))

    if 2 in modes:
        if lfacs is None:
            ell_values = numpy.arange(2, int(lmax) + 1, dtype=float)
            lfacs = ell_values * (ell_values + 1.0)
            lfacs2 = (ell_values + 2.0) * (ell_values - 1.0)
            lrootfacs = numpy.sqrt(lfacs * lfacs2)
        if lfacs is None or lfacs2 is None or lrootfacs is None:
            raise ValueError("Legendre helper factors are missing.")
        pvals = allp[2:]
        dpvals = alldp[2:]
        fac = fac1 / fac2
        d22 = (
            (((4.0 * x - 8.0) / fac2 + lfacs) * pvals)
            + 4.0 * fac * (fac2 + (x - 2.0) / lfacs) * dpvals
        ) / lfacs2
        if x > 0.998:
            d2m2 = numpy.empty(int(lmax) - 1)
            indser = int(numpy.sqrt((400.0 + 3.0 / (1.0 - x**2)) / 150.0)) - 1
            d2m2[indser:] = (
                (lfacs[indser:] - (4.0 * x + 8.0) / fac1) * pvals[indser:]
                + 4.0
                / fac
                * (-fac1 + (x + 2.0) / lfacs[indser:])
                * dpvals[indser:]
            ) / lfacs2[indser:]
            sin2 = 1.0 - x**2
            d2m2[:indser] = (
                lfacs[:indser]
                * lfacs2[:indser]
                * sin2**2
                / 7680.0
                * (20.0 + sin2 * (16.0 - lfacs[:indser]))
            )
        else:
            d2m2 = (
                ((lfacs - (4.0 * x + 8.0) / fac1) * pvals)
                + 4.0 / fac * (-fac1 + (x + 2.0) / lfacs) * dpvals
            ) / lfacs2
        d20 = (2.0 * x * dpvals - lfacs * pvals) / lrootfacs
        res.append((d20, d22, d2m2))

    return res


def lensed_correlations(
    cls: numpy.ndarray,
    clpp: numpy.ndarray,
    xvals: Iterable[float],
    weights: numpy.ndarray | None = None,
    lmax: int | None = None,
    delta: bool = False,
    theta_max: float | None = None,
    apodize_point_width: int = 10,
):
    """Return lensed correlation functions and optionally lensed spectra."""

    # The exact remapper can otherwise overflow on the large declared spectra
    # produced by the native solver before the final float cast.
    cls = numpy.asarray(cls, dtype=numpy.longdouble)
    clpp = numpy.asarray(clpp, dtype=numpy.longdouble)
    xvals = numpy.asarray(tuple(xvals), dtype=float)
    if lmax is None:
        lmax = int(cls.shape[0] - 1)
    _validate_remapping_inputs(cls, clpp, lmax=int(lmax))
    if xvals.ndim != 1 or xvals.size == 0:
        raise ValueError("Remapping quadrature nodes must be non-empty")
    if not numpy.all(numpy.isfinite(xvals)) or numpy.any(
        (xvals <= -1.0) | (xvals >= 1.0)
    ):
        raise ValueError(
            "Remapping quadrature nodes must lie strictly in (-1, 1)"
        )
    if numpy.any(numpy.diff(xvals) < 0.0):
        raise ValueError("Remapping quadrature nodes must be ordered")
    if weights is not None:
        weights = numpy.asarray(weights, dtype=numpy.longdouble)
        if weights.shape != xvals.shape or not numpy.all(
            numpy.isfinite(weights)
        ):
            raise ValueError("Remapping quadrature weights must match xvals")
    ell_values = numpy.arange(0, int(lmax) + 1, dtype=numpy.longdouble)
    ell_factors = ell_values * (ell_values + 1.0)
    ell_factors_all = ell_factors.copy()
    ell_factors[0] = 1.0
    cldd = clpp[1 : int(lmax) + 1] / ell_factors[1:]
    cphil3 = (2.0 * ell_values[1:] + 1.0) * cldd / 2.0
    facs = (
        (2.0 * ell_values + 1.0)
        / (4.0 * numpy.pi)
        * 2.0
        * numpy.pi
        / ell_factors
    )

    temperature_cls = numpy.asarray(
        facs * cls[: int(lmax) + 1, 0],
        dtype=numpy.longdouble,
    )
    electric_plus_magnetic_cls = numpy.asarray(
        facs[2:] * (cls[2 : int(lmax) + 1, 1] + cls[2 : int(lmax) + 1, 2]),
        dtype=numpy.longdouble,
    )
    electric_minus_magnetic_cls = numpy.asarray(
        facs[2:] * (cls[2 : int(lmax) + 1, 1] - cls[2 : int(lmax) + 1, 2]),
        dtype=numpy.longdouble,
    )
    temperature_polarization_cross_cls = numpy.asarray(
        facs[2:] * cls[2 : int(lmax) + 1, 3],
        dtype=numpy.longdouble,
    )

    ell_values = ell_values[2:]
    ell_factors = ell_factors[2:]
    ell_factors_squared = (ell_values + 2.0) * (ell_values - 1.0)
    ell_root_factors = numpy.sqrt(ell_factors * ell_factors_squared)
    root_factor_one = numpy.sqrt(ell_factors_squared)
    root_factor_two = numpy.sqrt(
        (ell_values[1:] + 3.0) * (ell_values[1:] - 2.0)
    )
    root_ratio = (
        ell_factors_squared[1:] / root_factor_one[1:] / root_factor_two
    )
    root_factor_three = numpy.sqrt(
        (ell_values[2:] - 3.0) * (ell_values[2:] + 4.0)
    )

    if weights is not None:
        lensedcls = numpy.zeros((int(lmax) + 1, 4), dtype=numpy.longdouble)
    delta_diff = 1 if delta else 0

    if theta_max is not None:
        xmin = numpy.cos(float(theta_max))
        imin = int(numpy.searchsorted(xvals, xmin))
    else:
        imin = 0

    corrs = numpy.zeros((len(xvals[imin:]), 4), dtype=numpy.longdouble)

    for i, x in enumerate(xvals[imin:]):
        x = numpy.longdouble(x)
        (
            (pvals, dpvals),
            (d11, dm11),
            (
                d20,
                d22,
                d2m2,
            ),
        ) = legendre_funcs(
            int(lmax),
            float(x),
            [0, 1, 2],
            ell_factors,
            ell_factors_squared,
            ell_root_factors,
        )
        pvals = numpy.asarray(pvals, dtype=numpy.longdouble)
        dpvals = numpy.asarray(dpvals, dtype=numpy.longdouble)
        d11 = numpy.asarray(d11, dtype=numpy.longdouble)
        dm11 = numpy.asarray(dm11, dtype=numpy.longdouble)
        d20 = numpy.asarray(d20, dtype=numpy.longdouble)
        d22 = numpy.asarray(d22, dtype=numpy.longdouble)
        d2m2 = numpy.asarray(d2m2, dtype=numpy.longdouble)
        sigma2 = numpy.dot(1.0 - d11, cphil3)
        cg2 = numpy.dot(dm11, cphil3)

        c2fac = ell_factors_all[1:] * cg2 / 2.0
        c2fac2 = c2fac[1:] ** 2
        fac = numpy.exp(-ell_factors_all * sigma2 / 2.0)
        difffac = fac - delta_diff
        f = temperature_cls * fac

        corrs[i, 0] = (
            numpy.dot(temperature_cls * difffac, pvals)
            + numpy.dot(f[1:], c2fac * (dm11 + c2fac * pvals[1:] / 4.0))
            + numpy.dot(f[2:], c2fac2 * d2m2) / 4.0
        )
        sine_theta = numpy.sqrt(1.0 - x**2)
        sine_factor = 4.0 / sine_theta
        one_minus_cosine = 1.0 - x
        one_plus_cosine = 1.0 + x
        d1m2 = (
            sine_theta
            / root_factor_one
            * (dpvals[2:] - 2.0 / one_minus_cosine * dm11[1:])
        )
        d12 = (
            sine_theta
            / root_factor_one
            * (dpvals[2:] - 2.0 / one_plus_cosine * d11[1:])
        )
        d1m3 = (
            -(x + 0.5) * sine_factor * d1m2[1:] / root_factor_two
            - root_ratio * dm11[2:]
        )
        d2m3 = (
            -one_plus_cosine * d2m2[1:] * sine_factor
            - root_factor_one[1:] * d1m2[1:]
        ) / root_factor_two
        d3m3 = (
            -(x + 1.5) * d2m3 * sine_factor - root_factor_one[1:] * d1m3
        ) / root_factor_two
        d13 = (x - 0.5) * sine_factor * d12[
            1:
        ] / root_factor_two - root_ratio * d11[2:]
        d04 = (
            (-ell_factors[2:] + (18.0 * x**2 + 6.0) / sine_theta**2) * d20[2:]
            - 6.0
            * x
            * ell_factors_squared[2:]
            * dpvals[4:]
            / ell_root_factors[2:]
        ) / (root_factor_two[1:] * root_factor_three)
        d2m4 = (
            -(6.0 * x + 4.0) / sine_theta * d2m3[1:]
            - root_factor_two[1:] * d2m2[2:]
        ) / root_factor_three
        d4m4 = (
            -7.0 / 5.0 * (ell_factors_squared[2:] - 6.0) * d2m2[2:]
            + 12.0
            / 5.0
            * (-ell_factors_squared[2:] + (9.0 * x + 26.0) / one_minus_cosine)
            * d3m3[1:]
        ) / (ell_factors_squared[2:] - 12.0)

        f = electric_plus_magnetic_cls * fac[2:]
        corrs[i, 1] = (
            numpy.dot(electric_plus_magnetic_cls * difffac[2:], d22)
            + numpy.dot(f[1:], c2fac[2:] * d13)
            + (numpy.dot(f, c2fac2 * d22) + numpy.dot(f[2:], c2fac2[2:] * d04))
            / 4.0
        )
        f = electric_minus_magnetic_cls * fac[2:]
        corrs[i, 2] = (
            numpy.dot(electric_minus_magnetic_cls * difffac[2:], d2m2)
            + (
                numpy.dot(f, c2fac[1:] * dm11[1:])
                + numpy.dot(f[1:], c2fac[2:] * d3m3)
            )
            / 2.0
            + (
                numpy.dot(f, c2fac2 * (2.0 * d2m2 + pvals[2:]))
                + numpy.dot(f[2:], c2fac2[2:] * d4m4)
            )
            / 8.0
        )
        f = temperature_polarization_cross_cls * fac[2:]
        corrs[i, 3] = (
            numpy.dot(temperature_polarization_cross_cls * difffac[2:], d20)
            + (
                numpy.dot(f, c2fac[1:] * d11[1:])
                + numpy.dot(f[1:], c2fac[2:] * d1m3)
            )
            / 2.0
            + (
                3.0 * numpy.dot(f, c2fac2 * d20)
                + numpy.dot(f[2:], c2fac2[2:] * d2m4)
            )
            / 8.0
        )
        if weights is not None:
            weight = numpy.longdouble(weights[i + imin])
            if theta_max is not None and i < apodize_point_width * 4:
                weight *= 1.0 - numpy.exp(
                    -(((i + 1.0) / apodize_point_width) ** 2) / 2.0
                )

            lensedcls[:, 0] += (weight * corrs[i, 0]) * pvals
            electric_component = (corrs[i, 1] * weight / 2.0) * d22
            magnetic_component = (corrs[i, 2] * weight / 2.0) * d2m2
            lensedcls[2:, 1] += electric_component + magnetic_component
            lensedcls[2:, 2] += electric_component - magnetic_component
            lensedcls[2:, 3] += (weight * corrs[i, 3]) * d20

    if weights is not None:
        lensedcls[1, :] *= 2.0
        lensedcls[2:, :] = (lensedcls[2:, :].T * ell_factors).T
        return corrs, lensedcls
    return corrs


def lensed_cls(
    cls: numpy.ndarray,
    clpp: numpy.ndarray,
    lmax: int | None = None,
    lmax_lensed: int | None = None,
    sampling_factor: float = 1.4,
    delta_cls: bool = False,
    theta_max: float = numpy.pi / 32.0,
    apodize_point_width: int = 10,
    leggaus: bool = True,
    cache: bool = True,
):
    """Return lensed power spectra for unlensed ``cls`` and lensing power."""

    cls = numpy.asarray(cls, dtype=numpy.longdouble)
    clpp = numpy.asarray(clpp, dtype=numpy.longdouble)
    if lmax is None:
        lmax = int(cls.shape[0] - 1)
    lmax = int(lmax)
    if not numpy.isfinite(float(sampling_factor)) or sampling_factor < 1.0:
        raise ValueError("sampling_factor must be finite and at least 1")
    _validate_remapping_inputs(
        cls,
        clpp,
        lmax=lmax,
        lmax_lensed=lmax_lensed,
    )
    npoints = int(sampling_factor * int(lmax)) + 1
    if leggaus:
        xvals, weights = _cached_gauss_legendre(npoints, cache=cache)
    else:
        theta = (
            numpy.arange(1, npoints + 1, dtype=numpy.longdouble)
            * numpy.longdouble(numpy.pi)
            / (npoints + 1.0)
        )
        xvals = numpy.cos(theta[::-1])
        weights = numpy.longdouble(numpy.pi) / npoints * numpy.sin(theta)
    _, lensedcls = lensed_correlations(
        cls,
        clpp,
        xvals,
        weights,
        lmax=lmax,
        delta=True,
        theta_max=theta_max,
        apodize_point_width=int(apodize_point_width * sampling_factor),
    )
    if not delta_cls:
        lensedcls += cls[: int(lmax) + 1, :]
    if lmax_lensed is not None:
        return lensedcls[: int(lmax_lensed) + 1, :]
    return lensedcls


__all__ = ["lensed_cls"]
