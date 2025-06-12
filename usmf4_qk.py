# copernican_suite/usmf4_qk.py
"""
USMFv4 "Quantum Kinematic" Model Plugin for the Copernican Suite.
This version is based on the postulate that m(z) = m_0(1+z), which leads
to a matter density evolution of rho_m ~ (1+z)^4. This requires
numerical integration for distance measures.
"""
# DEV NOTE (v1.3.1): This file has been corrected to be fully compliant with
# the v1.3 Copernican Suite interface specification (doc.json).
# 1. Renamed 'PARAMETER_INITIAL_GUESSES' to 'INITIAL_GUESSES'.
# 2. Renamed 'PARAMETER_PRIORS' to 'PARAMETER_BOUNDS'.
# 3. Renamed function 'get_distance_modulus_mu' to 'distance_modulus_model'.
# 4. Upgraded the numerical integration to use the robust, suite-standard
#    multiprocessing implementation from lcdm_model.py for performance.
# 5. Replaced internal sound horizon calculation with the vetted lcdm_model.py
#    version for consistency and accuracy.
# DEV NOTE (v1.4g): Added newline at end of file to conform to coding
# style guidelines.

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

# --- Model Metadata (Corrected to match doc.json spec) ---
MODEL_NAME = "USMFv4_QuantumKinematic"
MODEL_DESCRIPTION = "Unified Shrinking Matter Framework (USMF) Version 4 - Quantum Kinematic"
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^4 + \Omega_{\Lambda0}}$",
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
INITIAL_GUESSES = [68.0, 0.3, 0.0486]
PARAMETER_BOUNDS = [(50, 80), (0.1, 0.5), (0.04, 0.06)]
FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458, "MPC_PER_KM": 1.0 / (3.08567758e19), "T_CMB0_K": 2.7255,
    "N_EFF": 3.046, "OMEGA_G_H2": 2.47282e-5, "FLAT_UNIVERSE": True
}

# --- Core Model Functions ---

def get_Hz_per_Mpc(z, *cosmo_params):
    """
    Calculates the Hubble parameter H(z) in km/s/Mpc.
    This is the unique core of the USMFv4 model, with rho_m ~ (1+z)^4.
    """
    H0, Omega_m0, _ = cosmo_params
    z = np.asarray(z)
    
    # Assuming a flat universe as is standard for this model type
    Omega_L0 = 1.0 - Omega_m0
    
    if not (np.isfinite(H0) and H0 > 0 and np.isfinite(Omega_m0) and Omega_m0 >= 0 and Omega_L0 >= 0):
         return np.full_like(z, np.nan, dtype=float)

    term_m = Omega_m0 * np.power(1 + z, 4)
    hz = H0 * np.sqrt(term_m + Omega_L0)
    return hz

def _integrand_Dc(z_prime, H0, Omega_m0, Omega_b0):
    """The integrand for the comoving distance calculation, 1/H(z)."""
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
    """Calculates comoving distance using the suite-standard multiprocessing numerical integrator."""
    z_array_np = np.asarray(z_array)
    original_shape = z_array_np.shape
    z_flat = z_array_np.flatten()
    if z_flat.size == 0: return np.array([]).reshape(original_shape)

    z_batches = np.array_split(z_flat, PHYSICAL_CORES)
    args = zip(z_batches, repeat(H0), repeat(Omega_m0), repeat(Omega_b0))
    
    results_flat = []
    try:
        with mp.Pool(processes=PHYSICAL_CORES) as pool:
            batch_results = pool.starmap(_worker_Dc_batch, args)
        results_flat = [item for sublist in batch_results for item in sublist]
    except Exception as e:
        logging.getLogger().error(f"Multiprocessing failed with error: {e}. Falling back to serial execution.")
        results_flat = _worker_Dc_batch(z_flat, H0, Omega_m0, Omega_b0)
        
    results_Mpc = np.array(results_flat, dtype=float).reshape(original_shape)
    return results_Mpc.item() if z_array_np.ndim == 0 else results_Mpc

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    """Calculates luminosity distance."""
    # This function is generic and relies on get_comoving_distance_Mpc
    dc = get_comoving_distance_Mpc(z_array, *cosmo_params)
    return dc * (1 + np.asarray(z_array))

def distance_modulus_model(z_array, *cosmo_params):
    """Calculates the distance modulus. Name is corrected to match spec."""
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5 * np.log10(dl_mpc) + 25.0
    mu[np.asarray(dl_mpc) <= 0] = np.nan
    return mu

def get_angular_diameter_distance_Mpc(z_array, *cosmo_params):
    """Calculates angular diameter distance."""
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    return dl_mpc / (1 + np.asarray(z_array))**2

def get_DV_Mpc(z_array, *cosmo_params):
    """Calculates the BAO observable DV in Mpc."""
    da = get_angular_diameter_distance_Mpc(z_array, *cosmo_params)
    hz = get_Hz_per_Mpc(z_array, *cosmo_params)
    z = np.asarray(z_array)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_hz = np.where(hz > 1e-9, hz, np.nan)
        term = (1+z)**2 * da**2 * FIXED_PARAMS["C_LIGHT_KM_S"] * z / safe_hz
    return np.power(term, 1/3, where=term>=0, out=np.full_like(z, np.nan))

def get_sound_horizon_rs_Mpc(H0, Omega_m0, Omega_b0):
    """Calculates the sound horizon using the standard Eisenstein & Hu formula for consistency."""
    h = H0 / 100.0
    om_m_h2, om_b_h2 = Omega_m0 * h**2, Omega_b0 * h**2
    if not (np.isfinite(h) and h > 0 and om_m_h2 > 1e-5 and om_b_h2 > 1e-5): return np.nan

    # Standard calculation from Eisenstein & Hu, ApJ, 496, 605 (1998)
    # This ensures a consistent ruler for testing the late-time H(z)
    omega_nu_h2 = FIXED_PARAMS["N_EFF"] * (7.0/8.0) * (4.0/11.0)**(4.0/3.0) * FIXED_PARAMS["OMEGA_G_H2"]
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
