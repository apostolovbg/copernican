# copernican_suite/lcdm_model.py
"""
LCDM Model Plugin for the Copernican Suite.
This version uses the standard SciPy/CPU backend with intelligent multiprocessing.
"""

import numpy as np
from scipy.integrate import quad
import logging
import multiprocessing as mp
from functools import partial

try:
    import psutil
    PHYSICAL_CORES = psutil.cpu_count(logical=False) or 2
except (ImportError, NotImplementedError):
    PHYSICAL_CORES = 2

# --- Model Metadata ---
MODEL_NAME = "LambdaCDM"
MODEL_DESCRIPTION = "Standard flat Lambda Cold Matter model."
# FIXED: Rewrote LaTeX strings to be standard strings with escaped backslashes
MODEL_EQUATIONS_LATEX_SN = [
    "$H(z) = H_0 \\sqrt{\\Omega_{m0}(1+z)^3 + \\Omega_{\\Lambda0}}$",
    "$d_L(z) = (1+z) \\int_0^z \\frac{c}{H(z')} dz'$",
    "$\\mu = 5 \\log_{10}(d_L/1\\,\\mathrm{Mpc}) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    "$D_A(z) = \\frac{1}{1+z} \\int_0^z \\frac{c}{H(z')} dz'$",
    "$D_V(z) = \\left[ (1+z)^2 D_A(z)^2 \\frac{cz}{H(z)} \\right]^{1/3}$"
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

# --- Standard Python/Scipy Functions ---
# ... (The rest of the file is unchanged from the previous version) ...
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
            Dc_val, _ = quad(_integrand_Dc, 0, float(zi), args=(H0, Omega_m0, Omega_b0))
            results.append(Dc_val if np.isfinite(Dc_val) else np.nan)
        except Exception:
            results.append(np.nan)
    return results

def get_comoving_distance_Mpc(z_array, H0, Omega_m0, Omega_b0):
    """
    Calculates comoving distance, parallelized by batch processing
    across physical CPU cores.
    """
    z_array_np = np.asarray(z_array)
    original_shape = z_array_np.shape
    z_flat = z_array_np.flatten()

    z_batches = np.array_split(z_flat, PHYSICAL_CORES)
    p_worker = partial(_worker_Dc_batch, H0=H0, Omega_m0=Omega_m0, Omega_b0=Omega_b0)
    
    results_flat = []
    try:
        with mp.Pool(processes=PHYSICAL_CORES) as pool:
            batch_results = pool.map(p_worker, z_batches)
        results_flat = [item for sublist in batch_results for item in sublist]
    except Exception as e:
        logging.getLogger().error(f"Multiprocessing failed with error: {e}. Falling back to serial execution.")
        results_flat = _worker_Dc_batch(z_flat, H0, Omega_m0, Omega_b0)
        
    results_Mpc = np.array(results_flat).reshape(original_shape)
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
    term = (1+z)**2 * da**2 * FIXED_PARAMS["C_LIGHT_KM_S"] * z / hz
    return np.power(term, 1/3, where=term>=0, out=np.full_like(z, np.nan))

def get_sound_horizon_rs_Mpc(H0, Omega_m0, Omega_b0):
    h, _, _, _, omega_nu_h2 = _get_derived_densities(H0, Omega_m0, Omega_b0)
    if np.isnan(h): return np.nan
    om_m_h2, om_b_h2 = Omega_m0 * h**2, Omega_b0 * h**2
    if not (om_m_h2 > 1e-5 and om_b_h2 > 1e-5): return np.nan
    om_r_h2 = FIXED_PARAMS["OMEGA_G_H2"] + omega_nu_h2
    z_eq = om_m_h2 / om_r_h2 - 1.0
    if z_eq < 0: return np.nan
    b1 = 0.313 * om_m_h2**(-0.419) * (1 + 0.607 * om_m_h2**(0.674))
    b2 = 0.238 * om_m_h2**(0.223)
    z_d = 1291 * (om_m_h2**(0.251) / (1 + 0.659 * om_m_h2**(0.828))) * (1 + b1 * om_b_h2**b2)
    R_d = 31.5e3 * om_b_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-4 / (1 + z_d)
    R_eq = 31.5e3 * om_b_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-4 / (1 + z_eq)
    s_EH = (2.0 / (3.0 * 7.46e-2 * om_m_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-2)) * \
           np.sqrt(6.0 / R_eq) * np.log((np.sqrt(1.0 + R_d) + np.sqrt(R_d + R_eq)) / (1.0 + np.sqrt(R_eq)))
    if not np.isfinite(s_EH) or h == 0: return np.nan
    return s_EH / h
"""

The template section at the end of the file remains the same as the previous turn, as it does not contain code that is actually run.
"""