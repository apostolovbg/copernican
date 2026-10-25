"""Native effective H/He recombination and Compton thermal evolution.

The effective three-level atom, helium escape/continuum/triplet channels,
and hydrogen escape corrections follow Seager, Sasselov & Scott (1999),
Wong, Moss & Scott (2008), and the published RECFAST 1.5.2 prescription.
This independently integrated atomic adapter accepts the declared H(a);
it never calls an external Boltzmann solver or substitutes a cosmology.
"""

import math
from dataclasses import dataclass

import numpy
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

_BOLTZMANN = 1.380649e-23
_PLANCK = 6.62607015e-34
_LIGHT = 299792458.0
_ELECTRON_MASS = 9.1093837015e-31
_HYDROGEN_MASS = 1.673532838e-27
_HELIUM_HYDROGEN_MASS_RATIO = 3.971425883512839
_THOMSON = 6.6524587321e-29
_RADIATION_CONSTANT = 7.565733250033929e-16
_ENERGY_K_PER_WAVENUMBER = _PLANCK * _LIGHT / _BOLTZMANN
_SAHA_FACTOR = 2.0 * math.pi * _ELECTRON_MASS * _BOLTZMANN / _PLANCK**2


@dataclass(frozen=True)
class ThermalHistory:
    """Hydrogen ionization, helium electrons per H, and matter temperature."""

    hydrogen_fraction: numpy.ndarray
    helium_electrons: numpy.ndarray
    matter_temperature: numpy.ndarray


def _escape_probability(optical_depth):
    """Return the continuous Sobolev escape probability at zero depth."""

    depth = float(optical_depth)
    if not math.isfinite(depth) or depth < 0.0:
        raise ValueError("Line optical depth must be finite and nonnegative")
    return -math.expm1(-depth) / depth if depth else 1.0


def _thermal_derivative(
    scale_factor,
    state,
    *,
    hubble_si,
    hydrogen_today,
    temperature_today,
    helium_ratio,
):
    """Return coupled atomic and thermal derivatives with respect to ln a."""

    hydrogen = min(max(float(state[0]), 0.0), 1.0)
    helium = min(max(float(state[1]), 0.0), 1.0)
    ratio_temperature = max(float(state[2]), 1e-12)
    radiation_temperature = temperature_today / scale_factor
    temperature = max(ratio_temperature * radiation_temperature, 1e-6)
    number_hydrogen = hydrogen_today / scale_factor**3
    number_helium = helium_ratio * number_hydrogen
    electrons = hydrogen + helium_ratio * helium
    neutral_hydrogen = max(1.0 - hydrogen, 0.0)
    neutral_helium = max(1.0 - helium, 0.0)
    saha_prefactor = (_SAHA_FACTOR * temperature) ** 1.5

    # Hydrogen: detailed balance uses the same case-B coefficient in both
    # directions. The correction modifies the escape channel, not the seed.
    temperature_10k = temperature / 1e4
    alpha_hydrogen = (
        4.309e-19
        * temperature_10k**-0.6166
        / (1.0 + 0.6703 * temperature_10k**0.5300)
    )
    beta_hydrogen = (
        alpha_hydrogen
        * saha_prefactor
        * math.exp(
            -_ENERGY_K_PER_WAVENUMBER
            * (1.096787737e7 - 8.225916453e6)
            / temperature
        )
    )
    hydrogen_balance = (
        number_hydrogen * electrons * hydrogen * alpha_hydrogen
        - beta_hydrogen
        * neutral_hydrogen
        * math.exp(-_ENERGY_K_PER_WAVENUMBER * 8.225916453e6 / temperature)
    )
    log_radiation_redshift = math.log(radiation_temperature / 2.7255)
    escape_correction = (
        1.0
        - 0.14 * math.exp(-(((log_radiation_redshift - 7.28) / 0.18) ** 2))
        + 0.079 * math.exp(-(((log_radiation_redshift - 6.73) / 0.33) ** 2))
    )
    escape_hydrogen = escape_correction / (
        8.0 * math.pi * 8.225916453e6**3 * hubble_si
    )
    hydrogen_trapping = escape_hydrogen * number_hydrogen * neutral_hydrogen
    hydrogen_c = (1.0 + 8.2245809 * hydrogen_trapping) / (
        (1.0 + 8.2245809 * hydrogen_trapping) / 1.125
        + beta_hydrogen * hydrogen_trapping
    )
    if hydrogen > 0.985:
        hydrogen_c = 1.0
    derivative_hydrogen = -hydrogen_balance * hydrogen_c / hubble_si

    # Helium: singlet escape plus H continuum absorption, and the triplet
    # channel. Excited-state Boltzmann factors are divided out algebraically
    # rather than constructing enormous exp(+E/T) values after recombination.
    sqrt_low = math.sqrt(temperature / 3.0)
    sqrt_high = math.sqrt(temperature / 10**5.114)
    alpha_helium = 10**-16.744 / (
        sqrt_low
        * (1.0 + sqrt_low) ** (1.0 - 0.711)
        * (1.0 + sqrt_high) ** (1.0 + 0.711)
    )
    beta_helium = (
        4.0
        * alpha_helium
        * saha_prefactor
        * math.exp(
            -_ENERGY_K_PER_WAVENUMBER
            * (1.98310772e7 - 1.66277434e7)
            / temperature
        )
    )
    helium_escape = 1.0 / (8.0 * math.pi * 1.71134891e7**3 * hubble_si)
    channels_active = number_helium > 0.0 and 5e-9 <= helium <= 0.98
    triplet_c = 0.0
    if channels_active:
        singlet_depth = (
            1.798287e9 * helium_escape * 3.0 * number_helium * neutral_helium
        )
        singlet_rate = 1.798287e9 * _escape_probability(singlet_depth)
        doppler_ratio = math.sqrt(
            2.0
            * _BOLTZMANN
            * temperature
            / (_HYDROGEN_MASS * _HELIUM_HYDROGEN_MASS_RATIO * _LIGHT**2)
        )
        if hydrogen < 0.9999999:
            continuum_ratio = (
                3.0
                * 1.798287e9
                * helium_ratio
                * neutral_helium
                / (
                    math.sqrt(math.pi)
                    * 1.436289e-22
                    * 8.0
                    * math.pi
                    * _LIGHT
                    * 1.71134891e7**3
                    * doppler_ratio
                    * neutral_hydrogen
                )
            )
            singlet_rate += 1.798287e9 / (1.0 + 0.36 * continuum_ratio**0.86)
        helium_escape = 1.0 / (
            singlet_rate * 3.0 * number_helium * neutral_helium
        )
        triplet_depth = (
            177.58
            * number_helium
            * neutral_helium
            * 3.0
            / (8.0 * math.pi * hubble_si * 1.690871466e7**3)
        )
        triplet_rate = 177.58 * _escape_probability(triplet_depth)
        if hydrogen <= 0.99999:
            continuum_ratio = (
                3.0
                * 177.58
                * helium_ratio
                * neutral_helium
                / (
                    math.sqrt(math.pi)
                    * 1.484872e-22
                    * 8.0
                    * math.pi
                    * _LIGHT
                    * 1.690871466e7**3
                    * doppler_ratio
                    * neutral_hydrogen
                )
            )
            triplet_rate += 177.58 / (1.0 + 0.66 * continuum_ratio**0.9) / 3.0
        alpha_triplet = 10**-16.306 / (
            sqrt_low
            * (1.0 + sqrt_low) ** (1.0 - 0.761)
            * (1.0 + sqrt_high) ** (1.0 + 0.761)
        )
        beta_triplet = (
            alpha_triplet
            * saha_prefactor
            * 4.0
            / 3.0
            * math.exp(
                -_ENERGY_K_PER_WAVENUMBER * 3.8454693845e6 / temperature
            )
        )
        triplet_decay = triplet_rate * math.exp(
            -_ENERGY_K_PER_WAVENUMBER
            * (1.690871466e7 - 1.5985597526e7)
            / temperature
        )
        triplet_c = triplet_decay / max(beta_triplet + triplet_decay, 1e-300)
    excited_ratio = math.exp(
        -_ENERGY_K_PER_WAVENUMBER * (1.71134891e7 - 1.66277434e7) / temperature
    )
    helium_trapping = helium_escape * number_helium * neutral_helium
    helium_c = (excited_ratio + 51.3 * helium_trapping) / max(
        excited_ratio + (51.3 + beta_helium) * helium_trapping, 1e-300
    )
    derivative_helium = (
        -helium_c
        / hubble_si
        * (
            number_hydrogen * electrons * helium * alpha_helium
            - beta_helium
            * neutral_helium
            * math.exp(-_ENERGY_K_PER_WAVENUMBER * 1.66277434e7 / temperature)
        )
    )
    if channels_active:
        derivative_helium -= (
            triplet_c
            / hubble_si
            * (
                number_hydrogen * electrons * helium * alpha_triplet
                - 3.0
                * beta_triplet
                * neutral_helium
                * math.exp(
                    -_ENERGY_K_PER_WAVENUMBER * 1.5985597526e7 / temperature
                )
            )
        )
    if helium_ratio == 0.0:
        # A zero-abundance element has no atomic state to evolve.
        derivative_helium = 0.0
    compton_rate = (
        8.0
        * _THOMSON
        * _RADIATION_CONSTANT
        * radiation_temperature**4
        / (3.0 * _ELECTRON_MASS * _LIGHT)
        * electrons
        / (1.0 + helium_ratio + electrons)
    )
    derivative_temperature_ratio = (
        -ratio_temperature
        + compton_rate / hubble_si * (1.0 - ratio_temperature)
    )
    return numpy.asarray(
        (
            derivative_hydrogen,
            derivative_helium,
            derivative_temperature_ratio,
        )
    )


def evolve_thermal_history(
    scale_factors,
    hubble_rates_si,
    *,
    hydrogen_today,
    temperature_today,
    helium_ratio,
    initial_hydrogen,
    initial_helium,
    rtol=2e-8,
):
    """Integrate the coupled effective atom on a declared expansion grid."""

    scale_factors = numpy.asarray(scale_factors, dtype=float)
    hubble_rates_si = numpy.asarray(hubble_rates_si, dtype=float)
    if (
        scale_factors.ndim != 1
        or scale_factors.size < 2
        or hubble_rates_si.shape != scale_factors.shape
        or not numpy.all(numpy.isfinite(scale_factors))
        or not numpy.all(numpy.isfinite(hubble_rates_si))
        or numpy.any(scale_factors <= 0.0)
        or numpy.any(numpy.diff(scale_factors) <= 0.0)
        or numpy.any(hubble_rates_si <= 0.0)
    ):
        raise ValueError("Thermal evolution requires positive ordered grids")
    parameters = (
        hydrogen_today,
        temperature_today,
        helium_ratio,
        initial_hydrogen,
        initial_helium,
        rtol,
    )
    if (
        not all(math.isfinite(float(value)) for value in parameters)
        or hydrogen_today < 0.0
        or temperature_today <= 0.0
        or helium_ratio < 0.0
        or rtol <= 0.0
        or not 0.0 <= initial_hydrogen <= 1.0
        or not 0.0 <= initial_helium <= 1.0
    ):
        raise ValueError("Thermal evolution requires physical atomic inputs")
    log_a = numpy.log(scale_factors)
    expansion = PchipInterpolator(log_a, numpy.log(hubble_rates_si))

    def derivative(log_scale, state):
        """Evaluate atomic rates against the caller's physical expansion."""

        return _thermal_derivative(
            math.exp(log_scale),
            state,
            hubble_si=math.exp(float(expansion(log_scale))),
            hydrogen_today=hydrogen_today,
            temperature_today=temperature_today,
            helium_ratio=helium_ratio,
        )

    solution = solve_ivp(
        derivative,
        (log_a[0], log_a[-1]),
        (initial_hydrogen, initial_helium, 1.0),
        t_eval=log_a,
        method="BDF",
        rtol=rtol,
        atol=(1e-11, 1e-11, 1e-10),
    )
    if not solution.success or not numpy.all(numpy.isfinite(solution.y)):
        raise ValueError(
            "Coupled thermal evolution failed: " + solution.message
        )
    hydrogen, helium, ratio_temperature = solution.y
    if (
        numpy.any(hydrogen < -1e-7)
        or numpy.any(hydrogen > 1 + 1e-7)
        or numpy.any(helium < -1e-7)
        or numpy.any(helium > 1 + 1e-7)
        or numpy.any(ratio_temperature <= 0)
    ):
        raise ValueError("Coupled thermal evolution left physical bounds")
    return ThermalHistory(
        numpy.clip(hydrogen, 0.0, 1.0),
        helium_ratio * numpy.clip(helium, 0.0, 1.0),
        ratio_temperature * temperature_today / scale_factors,
    )
