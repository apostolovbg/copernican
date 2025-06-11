# copernican_suite/usmf4_qk.py
"""
USMFv4 "Quantum Kinematic" Model Plugin for the Copernican Suite.
This version is based on the postulate that m(z) = m_0(1+z), which leads
to a matter density evolution of rho_m ~ (1+z)^4. This requires
numerical integration for distance measures, similar to LambdaCDM.

CORRECTED: Removed internal multiprocessing to comply with suite architecture.
"""
import numpy as np
from scipy.integrate import quad
import logging

# --- Model Metadata ---
MODEL_NAME = "USMFv4_QuantumKinematic"
MODEL_DESCRIPTION = "Unified Shrinking Matter Framework (USMF) Version 4 - Quantum Kinematic"
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = H_0 \\sqrt{\\Omega_{m0}(1+z)^4 + \\Omega_{\\Lambda0}}$",
    r"$d_L(z) = (1+z) \\int_0^z \\frac{c}{H(z')} dz'$",
    r"$\\mu = 5 \\log_{10}(d_L/1\\,\\mathrm{Mpc}) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = \\frac{1}{1+z} \\int_0^z \\frac{c}{H(z')} dz'$",
    r"$D_V(z) = \\left[ (1+z)^2 D_A(z)^2 \\frac{cz}{H(z)} \\right]^{1/3}$"
]
PARAMETER_NAMES = ["H0", "Omega_m0", "Omega_b0"]
PARAMETER_LATEX_NAMES = ["$H_0$", "$\\Omega_{m0}$", "$\\Omega_{b0}$"]
PARAMETER_UNITS = ["km/s/Mpc", "", ""]
PARAMETER_PRIORS = [(50, 80), (0.1, 0.5), (0.04, 0.06)]
PARAMETER_INITIAL_GUESSES = [68.0, 0.3, 0.0486]

# --- Fixed Physical Constants ---
FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458,
    "T_CMB0_K": 2.7255,
    "OMEGA_G_H2": 2.47282e-5,
    "H0_FID_FOR_RS": 67.66
}

# --- Core Model Functions ---

def _get_derived_densities(H0, Omega_m0, Omega_b0):
    """Calculates derived density parameters."""
    if not (50 < H0 < 100 and 0 < Omega_m0 < 1 and 0 < Omega_b0 < 1):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    h = H0 / 100.0
    Omega_L0 = 1.0 - Omega_m0
    omega_nu_h2 = 0.00064
    return h, Omega_L0, Omega_m0 * h**2, Omega_b0 * h**2, omega_nu_h2

def get_Hz_per_Mpc(z, *cosmo_params):
    """
    Calculates the Hubble parameter H(z) in km/s/Mpc.
    This is the core of the USMFv4 model.
    """
    H0, Omega_m0, _ = cosmo_params
    z = np.asarray(z)
    Omega_L0 = 1.0 - Omega_m0
    if H0 <= 0 or Omega_m0 < 0 or Omega_L0 < 0:
        return np.full_like(z, np.nan)
    
    term_m = Omega_m0 * np.power(1 + z, 4)
    hz = H0 * np.sqrt(term_m + Omega_L0)
    return hz

def _integrand_comoving_dist(z, *cosmo_params):
    """The integrand for the comoving distance calculation."""
    hz = get_Hz_per_Mpc(z, *cosmo_params)
    return FIXED_PARAMS["C_LIGHT_KM_S"] / hz

def _integrator(func, z_array, cosmo_params):
    """Helper function to integrate over a z-array."""
    return np.array([quad(func, 0, z, args=cosmo_params)[0] for z in z_array])

def get_comoving_distance_Mpc(z_array, *cosmo_params):
    """Calculates comoving distance using numerical integration."""
    return _integrator(_integrand_comoving_dist, z_array, cosmo_params)

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    """Calculates luminosity distance."""
    dc = get_comoving_distance_Mpc(z_array, *cosmo_params)
    return dc * (1 + np.asarray(z_array))

def get_distance_modulus_mu(z_array, *cosmo_params):
    """Calculates the distance modulus."""
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    dl_mpc = np.asarray(dl_mpc)
    if np.any(dl_mpc <= 0): return np.nan
    mu = 5 * np.log10(dl_mpc) + 25
    return mu

def get_angular_diameter_distance_Mpc(z_array, *cosmo_params):
    """Calculates angular diameter distance."""
    dc = get_comoving_distance_Mpc(z_array, *cosmo_params)
    return dc / (1 + np.asarray(z_array))

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
    """
    Calculates the sound horizon at the drag epoch using the Eisenstein & Hu (1998)
    fitting formula.
    """
    h, _, _, _, omega_nu_h2 = _get_derived_densities(H0, Omega_m0, Omega_b0)
    if np.isnan(h): return np.nan
    om_m_h2, om_b_h2 = Omega_m0 * h**2, Omega_b0 * h**2
    if not (om_m_h2 > 1e-5 and om_b_h2 > 1e-5): return np.nan
    
    om_r_h2 = FIXED_PARAMS["OMEGA_G_H2"] + omega_nu_h2
    z_eq = 2.50e4 * om_m_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-4
    k_eq = 0.0746 * om_m_h2 * (FIXED_PARAMS["T_CMB0_K"]/2.7)**-2
    
    b1 = 0.313 * om_m_h2**-0.419 * (1 + 0.607 * om_m_h2**0.674)
    b2 = 0.238 * om_m_h2**0.223
    z_d = 1291 * (om_m_h2**0.251 / (1 + 0.659 * om_m_h2**0.828)) * (1 + b1 * om_b_h2**b2)
    
    R_d = (3 * om_b_h2) / (4 * om_r_h2 * (1+z_d))
    R_eq = (3 * om_b_h2) / (4 * om_r_h2 * z_eq)
    
    s_d = (2 / (3 * k_eq)) * np.sqrt(6 / R_eq) * np.log((np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq)))
    
    return s_d / h