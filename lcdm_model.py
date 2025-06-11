# copernican_suite/lcdm_model.py
"""
LCDM Model Plugin for the Copernican Suite.
This version uses the standard SciPy/CPU backend with intelligent multiprocessing.
"""
# DEV NOTE (v1.3a): This file was updated to fix a critical bug in the
# get_comoving_distance_Mpc function. The multiprocessing implementation
# was reverted from a `pool.map` with `partial` to the more robust and
# stable `pool.starmap` with `zip` and `repeat`. This change was necessary
# to resolve an issue where the function would fail silently and return NaNs
# when calculating smooth curves for plotting, causing the BAO plot lines
# to disappear. This fix restores correct BAO plot generation.

import numpy as np
from scipy.integrate import quad
import logging
import multiprocessing as mp
from itertools import repeat

try:
    import psutil
    PHYSICAL_CORES = psutil.cpu_count(logical=False) or 2
except (ImportError, NotImplementedError):
    PHYSICAL_CORES = 2

# --- Model Metadata ---
MODEL_NAME = "LambdaCDM"
MODEL_DESCRIPTION = "Standard flat Lambda Cold Matter model."
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$",
    r"$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$",
    r"$\mu = 5 \log_{10}(d_L/1\,\mathrm{Mpc}) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$",
    r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
]
PARAMETER_NAMES = ["H0", "Omega_m0", "Omega_b0"]
PARAMETER_LATEX_NAMES = [r"$H_0$", r"$\Omega_{m0}$", r"$\Omega_{b0}$"]
PARAMETER_UNITS = ["km/s/Mpc", "", ""]
INITIAL_GUESSES = [67.7, 0.31, 0.0486]
PARAMETER_BOUNDS = [(50.0, 100.0), (0.05, 0.7), (0.01, 0.1)]
FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458, "MPC_PER_KM": 1.0 / (3.08567758e19), "T_CMB0_K": 2.7255,
    "N_EFF": 3.046, "OMEGA_G_H2": 2.47282e-5, "FLAT_UNIVERSE": True
}

def _get_derived_densities(H0, Omega_m0, Omega_b0):
    """Core helper to calculate derived density parameters."""
    if not (np.isfinite(H0) and H0 > 0 and np.isfinite(Omega_m0) and Omega_m0 >= 0 and np.isfinite(Omega_b0) and Omega_b0 >= 0 and Omega_b0 <= Omega_m0):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    h = H0 / 100.0
    omega_nu_h2 = FIXED_PARAMS["N_EFF"] * (7.0/8.0) * (4.0/11.0)**(4.0/3.0) * FIXED_PARAMS["OMEGA_G_H2"]
    Omega_r0 = (FIXED_PARAMS["OMEGA_G_H2"] + omega_nu_h2) / h**2
    Omega_k0 = 0.0
    Omega_L0 = 1.0 - Omega_m0 - Omega_r0
    if Omega_L0 < 0: return np.nan, np.nan, np.nan, np.nan, np.nan
    return h, Omega_r0, Omega_L0, Omega_k0, omega_nu_h2

def get_Hz_per_Mpc(z_array, H0, Omega_m0, Omega_b0):
    z_array = np.asarray(z_array)
    h, Omega_r0, Omega_L0, Omega_k0, _ = _get_derived_densities(H0, Omega_m0, Omega_b0)
    if np.isnan(h): return np.full_like(z_array, np.nan, dtype=float)
    one_plus_z = 1 + z_array
    Ez_sq = Omega_r0 * one_plus_z**4 + Omega_m0 * one_plus_z**3 + Omega_k0 * one_plus_z**2 + Omega_L0
    Hz = np.full_like(z_array, np.nan, dtype=float)
    valid_Ez_sq = np.isfinite(Ez_sq) & (Ez_sq >= 0)
    if np.any(valid_Ez_sq): Hz[valid_Ez_sq] = H0 * np.sqrt(Ez_sq[valid_Ez_sq])
    return Hz.item() if z_array.ndim == 0 else Hz

def _integrand_Dc(z_prime, H0, Omega_m0, Omega_b0):
    hz_val = get_Hz_per_Mpc(z_prime, H0, Omega_m0, Omega_b0)
    return FIXED_PARAMS["C_LIGHT_KM_S"] / hz_val if (np.isfinite(hz_val) and hz_val > 1e-9) else np.inf

def _worker_Dc_batch(z_batch, H0, Omega_m0, Omega_b0):
    """Calculates comoving distance for a batch of redshift values."""
    results = []
    for zi in z_batch:
        if abs(float(zi)) < 1e-9:
            results.append(0.0)
            continue
        try:
            # Perform the numerical integration for a single redshift value
            Dc_val, _ = quad(_integrand_Dc, 0, float(zi), args=(H0, Omega_m0, Omega_b0))
            results.append(Dc_val if np.isfinite(Dc_val) else np.nan)
        except Exception:
            results.append(np.nan)
    return results

def get_comoving_distance_Mpc(z_array, H0, Omega_m0, Omega_b0):
    """
    Calculates comoving distance, parallelized by batch processing
    across physical CPU cores using the stable starmap method.
    """
    z_array_np = np.asarray(z_array)
    original_shape = z_array_np.shape
    z_flat = z_array_np.flatten()
    if z_flat.size == 0: return np.array([]).reshape(original_shape)

    # Split the flattened array of redshifts into batches for each CPU core
    z_batches = np.array_split(z_flat, PHYSICAL_CORES)
    
    # Create argument packages for starmap. Each package is a tuple:
    # (z_batch, H0, Omega_m0, Omega_b0). 'repeat' ensures the same cosmo
    # parameters are passed along with each unique z_batch.
    args = zip(z_batches, repeat(H0), repeat(Omega_m0), repeat(Omega_b0))
    
    results_flat = []
    try:
        # Use a multiprocessing Pool to execute the worker function in parallel
        with mp.Pool(processes=PHYSICAL_CORES) as pool:
            # starmap is used because our worker function takes multiple arguments
            batch_results = pool.starmap(_worker_Dc_batch, args)
        # Combine the results from all batches into a single flat list
        results_flat = [item for sublist in batch_results for item in sublist]
    except (TypeError, AttributeError, EOFError, Exception) as e:
        # If multiprocessing fails for any reason, fall back to safe serial execution
        logging.getLogger().error(f"Multiprocessing failed with error: {e}. Falling back to serial execution.")
        results_flat = _worker_Dc_batch(z_flat, H0, Omega_m0, Omega_b0)
        
    # Reshape the flat list of results back to the original array shape
    results_Mpc = np.array(results_flat, dtype=float).reshape(original_shape)
    
    # If the input was a scalar, return a scalar
    return results_Mpc.item() if z_array_np.ndim == 0 else results_Mpc

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    dm = get_comoving_distance_Mpc(z_array, *cosmo_params)
    return dm * (1 + np.asarray(z_array))

def distance_modulus_model(z_array, *cosmo_params):
    """The standard, Scipy-based function for fitting and plotting."""
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5 * np.log10(dl_mpc) + 25.0
    mu[np.asarray(dl_mpc) <= 0] = np.nan
    return mu

def get_angular_diameter_distance_Mpc(z_array, *cosmo_params):
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    return dl_mpc / (1 + np.asarray(z_array))**2

def get_DV_Mpc(z_array, *cosmo_params):
    da = get_angular_diameter_distance_Mpc(z_array, *cosmo_params)
    hz = get_Hz_per_Mpc(z_array, *cosmo_params)
    z = np.asarray(z_array)
    # Use a 'where' clause to prevent division by zero or invalid operations
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_hz = np.where(hz > 1e-9, hz, np.nan)
        term = (1+z)**2 * da**2 * FIXED_PARAMS["C_LIGHT_KM_S"] * z / safe_hz
    return np.power(term, 1/3, where=term>=0, out=np.full_like(z, np.nan))

def get_sound_horizon_rs_Mpc(H0, Omega_m0, Omega_b0):
    """
    Calculates the sound horizon at the drag epoch using the Eisenstein & Hu (1998)
    fitting formula. This is a standard, fast, and accurate approximation.
    """
    h, _, _, _, omega_nu_h2 = _get_derived_densities(H0, Omega_m0, Omega_b0)
    if np.isnan(h): return np.nan
    om_m_h2, om_b_h2 = Omega_m0 * h**2, Omega_b0 * h**2
    if not (om_m_h2 > 1e-5 and om_b_h2 > 1e-5): return np.nan
    
    # Fitting formula from Eisenstein & Hu, ApJ, 496, 605 (1998)
    om_r_h2 = FIXED_PARAMS["OMEGA_G_H2"] + omega_nu_h2
    z_eq = 2.50e4 * om_m_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-4
    k_eq = 7.46e-2 * om_m_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-2

    b1 = 0.313 * om_m_h2**(-0.419) * (1 + 0.607 * om_m_h2**(0.674))
    b2 = 0.238 * om_m_h2**(0.223)
    z_d = 1291 * (om_m_h2**(0.251) / (1 + 0.659 * om_m_h2**(0.828))) * (1 + b1 * om_b_h2**b2)
    
    R_d = 31.5e3 * om_b_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-4 / (1 + z_d)
    R_eq = 31.5e3 * om_b_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-4 / (1 + z_eq)
    
    s_EH = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * \
           np.log((np.sqrt(1.0 + R_d) + np.sqrt(R_d + R_eq)) / (1.0 + np.sqrt(R_eq)))
           
    if not np.isfinite(s_EH) or h == 0: return np.nan
    return s_EH