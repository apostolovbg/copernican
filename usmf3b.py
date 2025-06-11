# copernican_suite/usmf3b.py
"""
Unified Shrinking Matter Framework (USMF) V3b "Kinematic" Model Plugin.
This version uses a fully analytic formulation for all distance and Hubble
calculations, making it extremely fast. It uses a standard Eisenstein & Hu
fitting formula for the sound horizon for BAO consistency.
"""

import numpy as np

# --- Model Metadata and Parameters ---
MODEL_NAME = "USMFv3b_Kinematic"
MODEL_DESCRIPTION = "Unified Shrinking Matter Framework (USMF) Version 3b - Kinematic."
MODEL_EQUATIONS_LATEX_SN = [
    r"$\alpha(t) = \left( t_0/t \right)^{p_{kin}}$",
    r"$r(z) = \frac{c \cdot t_0}{p_{kin}+1} \left[ 1 - (1+z)^{-\frac{p_{kin}+1}{p_{kin}}} \right]$",
    r"$d_L(z) = |r(z)| \cdot (1+z)^2 \cdot \frac{70.0}{H_A}$",
    r"$\mu = 5 \log_{10}(d_L) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = |r(z)| \cdot \frac{70.0}{H_A}$",
    r"$H_{USMF}(z) = \frac{p_{kin}}{t_0} (1+z)^{1/p_{kin}}$",
    r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
]
PARAMETER_NAMES = ["H_A", "t0_age_Gyr", "p_kin", "Omega_m0_fid", "Omega_b0_fid"]
PARAMETER_LATEX_NAMES = [r"$H_A$", r"$t_{0,age}$", r"$p_{kin}$", r"$\Omega_{m0,fid}$", r"$\Omega_{b0,fid}$"]
PARAMETER_UNITS = ["", "Gyr", "", "", ""]
INITIAL_GUESSES = [70.0, 14.0, 0.8, 0.31, 0.0486]
PARAMETER_BOUNDS = [(50.0, 100.0), (10.0, 20.0), (0.1, 2.0), (0.2, 0.4), (0.03, 0.07)]

FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458,
    "SECONDS_PER_GYR": 1e9 * 365.25 * 24 * 3600,
    "MPC_PER_KM": 1.0 / (3.08567758e19),
    "H0_FID_FOR_RS": 70.0 # A fixed H0 for calculating h in the r_s formula
}

# --- Core Analytic Functions ---

def _get_params(cosmo_params):
    """Helper to unpack parameters and return derived values."""
    H_A, t0_age_Gyr, p_kin, _, _ = cosmo_params
    t0_sec = t0_age_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]
    t0_Mpc = t0_sec * FIXED_PARAMS["C_LIGHT_KM_S"] * FIXED_PARAMS["MPC_PER_KM"]
    return H_A, t0_Mpc, p_kin

def get_comoving_distance_Mpc(z_array, *cosmo_params):
    """
    Calculates the comoving distance r(z) using the analytic solution.
    This distance is scaled by c, so it has units of Mpc.
    """
    z = np.asarray(z_array)
    _, t0_Mpc, p_kin = _get_params(cosmo_params)

    if p_kin <= 0 or t0_Mpc <= 0:
        return np.full_like(z, np.nan, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        exponent = -(p_kin + 1.0) / p_kin
        term_in_brackets = 1.0 - np.power(1.0 + z, exponent)
    
    r_mpc = (t0_Mpc / (p_kin + 1.0)) * term_in_brackets
    r_mpc[z < 0] = np.nan
    return r_mpc

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    """
    Calculates the luminosity distance dL, including the H_A scaling factor.
    """
    z = np.asarray(z_array)
    H_A, _, _ = _get_params(cosmo_params)

    if H_A <= 0:
        return np.full_like(z, np.nan, dtype=float)

    r_mpc = get_comoving_distance_Mpc(z_array, *cosmo_params)
    
    # The USMF dL definition includes a scaling factor relative to a H0=70 universe
    scaling_factor = 70.0 / H_A
    dl_mpc = np.abs(r_mpc) * np.power(1.0 + z, 2) * scaling_factor
    return dl_mpc

def distance_modulus_model(z_array, *cosmo_params):
    """
    Calculates the distance modulus mu for SNe fitting.
    """
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    # Ensure non-positive distances result in nan, not -inf.
    dl_mpc_safe = np.where(dl_mpc > 0, dl_mpc, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5.0 * np.log10(dl_mpc_safe) + 25.0
    return mu

def get_angular_diameter_distance_Mpc(z_array, *cosmo_params):
    """
    Calculates the angular diameter distance dA.
    """
    z = np.asarray(z_array)
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    # dA = dL / (1+z)^2
    da_mpc = dl_mpc / np.power(1.0 + z, 2)
    return da_mpc

def get_Hz_per_Mpc(z_array, *cosmo_params):
    """
    Calculates the effective Hubble parameter H(z) using the analytic solution.
    """
    z = np.asarray(z_array)
    _, t0_age_Gyr, p_kin, _, _ = cosmo_params
    t0_sec = t0_age_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]
    
    if p_kin <= 0 or t0_sec <= 0:
        return np.full_like(z, np.nan, dtype=float)
        
    with np.errstate(divide='ignore', invalid='ignore'):
        exponent = 1.0 / p_kin
        hz_sec_inv = (p_kin / t0_sec) * np.power(1.0 + z, exponent)
    
    # Convert from 1/s to km/s/Mpc
    km_per_mpc = 1.0 / FIXED_PARAMS["MPC_PER_KM"]
    hz_kms_mpc = hz_sec_inv * km_per_mpc
    hz_kms_mpc[z < 0] = np.nan
    return hz_kms_mpc

def get_DV_Mpc(z_array, *cosmo_params):
    """
    Calculates the BAO volume-averaged distance DV(z).
    """
    z = np.asarray(z_array)
    da = get_angular_diameter_distance_Mpc(z_array, *cosmo_params)
    hz = get_Hz_per_Mpc(z_array, *cosmo_params)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ensure hz is not zero to avoid division errors
        safe_hz = np.where(hz > 1e-9, hz, np.nan)
        term_in_bracket = np.power(1.0+z, 2) * np.power(da, 2) * FIXED_PARAMS["C_LIGHT_KM_S"] * z / safe_hz
    
    # Calculate power only where term is non-negative
    dv = np.power(term_in_bracket, 1.0/3.0, where=term_in_bracket >= 0, out=np.full_like(z, np.nan))
    return dv

def get_sound_horizon_rs_Mpc(*cosmo_params):
    """
    Calculates the sound horizon at the drag epoch using the Eisenstein & Hu (1998)
    fitting formula. This provides a physically-motivated, standard ruler for BAO analysis
    that is independent of the late-time USMF kinematics.
    
    This function uses the fiducial Omega_m0 and Omega_b0 parameters, which describe the
    properties of the early-universe plasma.
    """
    _, _, _, Omega_m0_fid, Omega_b0_fid = cosmo_params
    h_fid = FIXED_PARAMS["H0_FID_FOR_RS"] / 100.0
    
    om_m_h2 = Omega_m0_fid * h_fid**2
    om_b_h2 = Omega_b0_fid * h_fid**2
    
    if not (om_m_h2 > 1e-5 and om_b_h2 > 1e-5): return np.nan
    
    try:
        term1 = np.power(om_m_h2, 0.25351)
        term2 = np.power(om_b_h2, 0.12807)
        exponent_term = -72.3 * np.power(om_m_h2 - 0.14, 2)
        
        rs = 55.154 * np.exp(exponent_term) / (term1 * term2)
        
    except (ValueError, OverflowError):
        return np.nan
        
    return rs if np.isfinite(rs) and rs > 0 else np.nan