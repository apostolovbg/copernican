# lcdm_model.py
"""
LCDM Model Plugin for the Copernican Suite.

DEV NOTE (v1.4rc6):
1.  API ALIGNMENT FIX: The primary distance modulus function has been updated
    to align with the v1.4rc engine API.
    - Renamed `get_distance_modulus_mu` to `get_distance_modulus`.
    - Signature changed to accept the nuisance parameter `M` and `**kwargs`
      to safely ignore extra parameters (like `Ok0`) passed by the engine.
    - Calculation now correctly uses the passed value of `M` instead of a
      hardcoded `+ 25`.
    This resolves the `TypeError` crash when using this model.

(Previous notes from v1.4rc preserved below)
...

DEV NOTE (v1.4g): Added newline at EOF for consistent file formatting.
DEV NOTE (scalar support): get_comoving_distance_Mpc now accepts scalar
redshift inputs to prevent integration errors.
"""

import numpy as np
from scipy.integrate import quad
import logging

# --- Model Metadata ---
# This dictionary is the standardized format for model information in v1.4+.
METADATA = {
    "model_name": "LambdaCDM",
    "model_description": "Standard flat Lambda Cold Matter model.",
    "equations": {
        "sne": [
            r"$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$",
            r"$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$",
            r"$\mu = 5 \log_{10}(d_L/1\,\mathrm{Mpc}) + M$"
        ],
        "bao": [
            r"$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$",
            r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
        ]
    },
    "parameters": {
        "H0": {
            "description": "Hubble Constant at z=0",
            "units": "km/s/Mpc",
            "latex": r"$H_0$",
            "role": "cosmological",
            "initial_guess": 70,
            "bounds": (50, 90)
        },
        "Omega_m0": {
            "description": "Matter Density Parameter at z=0",
            "units": None,
            "latex": r"$\Omega_{m0}$",
            "role": "cosmological",
            "initial_guess": 0.3,
            "bounds": (0.0, 1.0)
        },
        "Omega_b0": {
            "description": "Baryon Density Parameter at z=0",
            "units": None,
            "latex": r"$\Omega_{b0}$",
            "role": "cosmological",
            "initial_guess": 0.05,
            "bounds": (0.01, 0.1)
        },
        "M": {
            "description": "Absolute Magnitude of a Type Ia Supernova",
            "units": "magnitude",
            "latex": r"$M$",
            "role": "nuisance",
            "initial_guess": -19.3,
            "bounds": (-20, -18)
        },
        "alpha": {
            "description": "SNe Light Curve Stretch Nuisance Parameter",
            "units": None,
            "latex": r"$\alpha$",
            "role": "nuisance",
            "initial_guess": 0.14,
            "bounds": (0.0, 1.0)
        },
        "beta": {
            "description": "SNe Light Curve Color Nuisance Parameter",
            "units": None,
            "latex": r"$\beta$",
            "role": "nuisance",
            "initial_guess": 3.1,
            "bounds": (1.0, 5.0)
        }
    },
    "fixed_params": {
        "C_LIGHT_KM_S": 299792.458,
        "T_CMB0_K": 2.7255,
        "OMEGA_G_H2": 2.472e-5,
        "NEUTRINO_MASS_eV": 0.06
    }
}


# --- Physics Functions ---
# These functions implement the core logic of the LambdaCDM model.

def _get_derived_densities(H0, Omega_m0, Omega_b0):
    """
    Helper function to calculate derived cosmological densities.
    Returns h, Omega_Lambda0, omega_m_h2, omega_b_h2, omega_nu_h2.
    """
    if H0 <= 0: return (np.nan,) * 5
    h = H0 / 100.0
    Omega_Lambda0 = 1.0 - Omega_m0 # Assuming a flat universe
    omega_m_h2 = Omega_m0 * h**2
    omega_b_h2 = Omega_b0 * h**2
    # Neutrino density approximation
    omega_nu_h2 = METADATA['fixed_params']['NEUTRINO_MASS_eV'] / 93.14
    return h, Omega_Lambda0, omega_m_h2, omega_b_h2, omega_nu_h2

def get_Hz_per_Mpc(z_array, H0, Omega_m0):
    """
    Calculates the Hubble parameter H(z) in units of (km/s)/Mpc.
    This is the core equation for the expansion history.
    """
    z = np.asarray(z_array)
    _, Omega_Lambda0, _, _, _ = _get_derived_densities(H0, Omega_m0, 0) # Omega_b0 not needed here
    if np.isnan(Omega_Lambda0): return np.full_like(z, np.nan)
    
    term_m = Omega_m0 * (1 + z)**3
    term_lambda = Omega_Lambda0
    
    # Ensure terms inside sqrt are non-negative
    radicand = term_m + term_lambda
    with np.errstate(invalid='ignore'): # Ignore warnings for z < -1 if present
        return H0 * np.sqrt(radicand)

def get_comoving_distance_Mpc(z_array, H0, Omega_m0):
    """
    Calculates the line-of-sight comoving distance in Mpc.
    Integrates c/H(z) from 0 to z.
    """
    C_LIGHT = METADATA['fixed_params']['C_LIGHT_KM_S']
    integrand = lambda z_prime: C_LIGHT / get_Hz_per_Mpc(z_prime, H0, Omega_m0)

    # BUG FIX (v1.4b): Use a robust list comprehension instead of np.vectorize
    # This prevents NaN arrays when calculating smooth model lines for plotting.
    #
    # DEV NOTE (scalar support): v1.4g+ -- Accept scalar z by converting to at
    # least 1D before iteration. This resolves 'iteration over a 0-d array'
    # errors triggered when the engine calls model functions with a single
    # redshift value.
    z_vals = np.atleast_1d(z_array)
    try:
        # Perform integration for each z value in the input array.
        results = [quad(integrand, 0, z)[0] for z in z_vals]
        results = np.array(results)
        return results if results.size > 1 else results[0]
    except Exception as e:
        logging.error(f"Error during comoving distance integration: {e}")
        nan_array = np.full(z_vals.shape, np.nan)
        return nan_array if nan_array.size > 1 else np.nan

# --- Derived Cosmological Observables ---

def get_luminosity_distance_Mpc(z_array, H0, Omega_m0):
    """Calculates luminosity distance, d_L = d_C * (1+z), in Mpc."""
    d_c = get_comoving_distance_Mpc(z_array, H0, Omega_m0)
    return d_c * (1 + np.asarray(z_array))

def get_distance_modulus(z_array, H0, Omega_m0, M, **kwargs):
    """Calculates the distance modulus, mu = 5*log10(d_L/Mpc) + M."""
    # Pass only the needed cosmological params to the luminosity distance function.
    # The **kwargs will safely capture any other params like Omega_b0, Ok0.
    dl_mpc = get_luminosity_distance_Mpc(z_array, H0, Omega_m0)
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore log10(0) if dl is zero
        # Use the fitted nuisance parameter M, not a fixed value
        return 5 * np.log10(dl_mpc) + M

def get_angular_diameter_distance_Mpc(z_array, H0, Omega_m0, **kwargs):
    """Calculates angular diameter distance, d_A = d_C / (1+z), in Mpc."""
    d_c = get_comoving_distance_Mpc(z_array, H0, Omega_m0)
    return d_c / (1 + np.asarray(z_array))

def get_DV_Mpc(z_array, H0, Omega_m0, **kwargs):
    """
    Calculates the BAO volume-averaged distance, D_V.
    D_V = [ (1+z)^2 * D_A^2 * c*z / H(z) ]^(1/3)
    """
    da = get_angular_diameter_distance_Mpc(z_array, H0, Omega_m0)
    hz = get_Hz_per_Mpc(z_array, H0, Omega_m0)
    z = np.asarray(z_array)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_hz = np.where(hz > 1e-9, hz, np.nan)
        term = (1+z)**2 * da**2 * METADATA['fixed_params']["C_LIGHT_KM_S"] * z / safe_hz
    return np.power(term, 1/3, where=term>=0, out=np.full_like(z, np.nan))

def get_sound_horizon_rs_Mpc(H0, Omega_m0, Omega_b0):
    """
    Calculates the sound horizon at the drag epoch using the Eisenstein & Hu (1998)
    fitting formula. This is a standard, fast, and accurate approximation.
    """
    h, _, _, omega_m_h2, omega_b_h2, omega_nu_h2 = _get_derived_densities(H0, Omega_m0, Omega_b0)
    if np.isnan(h): return np.nan
    if not (omega_m_h2 > 1e-5 and omega_b_h2 > 1e-5): return np.nan

    T_CMB0 = METADATA['fixed_params']["T_CMB0_K"]
    om_r_h2 = METADATA['fixed_params']["OMEGA_G_H2"] + omega_nu_h2
    
    # Calculations from Eisenstein & Hu (1998), ApJ, 496, 605
    z_eq = 2.50e4 * omega_m_h2 * (T_CMB0/2.7)**-4
    k_eq = 7.46e-2 * omega_m_h2 * (T_CMB0/2.7)**-2
    
    b1 = 0.313 * (omega_m_h2)**-0.419 * (1 + 0.607 * (omega_m_h2)**0.674)
    b2 = 0.238 * (omega_m_h2)**0.223
    z_d = 1291 * ((omega_m_h2)**0.251 / (1 + 0.659 * (omega_m_h2)**0.828)) * (1 + b1 * (omega_b_h2)**b2)

    R_eq = 3.15e4 * omega_b_h2 * (T_CMB0/2.7)**-4 / z_eq
    R_d = 3.15e4 * omega_b_h2 * (T_CMB0/2.7)**-4 / z_d
    
    s = (2. / (3. * k_eq)) * np.sqrt(6. / R_eq) * np.log((np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq)))
    
    return s / h # Convert from Mpc/h to Mpc
