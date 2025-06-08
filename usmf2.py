# copernican_suite/usmf2.py
"""
Unified Shrinking Matter Framework (USMF) V2 Model Plugin for the Copernican Suite.
*** Optimized and Refactored for Stability and Speed ***
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# --- Model Metadata and Parameters ---
MODEL_NAME = "USMF_V2"
MODEL_DESCRIPTION = "Unified Shrinking Matter Framework (USMF) Version 2."
MODEL_EQUATIONS_LATEX_SN = [
    r"$1+z = \alpha(t_e) / \alpha(t_0)$",
    r"$r = \int_{t_e}^{t_0} \frac{c}{\alpha(t')} dt'$",
    r"$d_L(z) = |r| \cdot (1+z)^2 \cdot \frac{70.0}{H_A}$",
    r"$\mu = 5 \log_{10}(d_L) + 25$"
]
# --- BUG FIX: Replaced unsupported `\bigg|` with `\left. ... \right|` for Matplotlib compatibility ---
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = |r| \cdot \frac{70.0}{H_A}$",
    r"$H_{USMF}(z) = - \frac{1}{\alpha(t_e)} \left. \frac{d\alpha}{dt} \right|_{t_e}$",
    r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
]
PARAMETER_NAMES = ["H_A", "p_alpha", "k_exp", "s_exp", "t0_age_Gyr", "A_osc", "omega_osc", "ti_osc_Gyr", "phi_osc"]
PARAMETER_LATEX_NAMES = [r"$H_A$", r"$p_{\alpha}$", r"$k_{exp}$", r"$s_{exp}$", r"$t_{0,age}$", r"$A_{osc}$", r"$\omega_{osc}$", r"$t_{i,osc}$", r"$\phi_{osc}$"]
PARAMETER_UNITS = ["", "", "", "", "Gyr", "", "rad/log(time_ratio)", "Gyr", "rad"]
INITIAL_GUESSES = [77.111, 0.4313, -0.3268, 1.1038, 13.397, 0.0027088, 2.3969, 7.1399, 0.10905]
PARAMETER_BOUNDS = [(10.0, 200.0), (0.01, 3.0), (-5.0, 10.0), (0.1, 3.0), (1.0, 30.0), (0.0, 0.099), (0.1, 50.0), (0.1, 25.0), (-np.pi, np.pi)]

# --- Fixed Constants ---
FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458, "SECONDS_PER_GYR": 1e9 * 365.25 * 24 * 3600, "MPC_PER_KM": 1.0 / (3.08567758e19),
    "MPC_TO_KM": 3.08567758e19, "TINY_FLOAT": np.finfo(float).tiny,
    "USMF_EARLY_ALPHA_POWER_M": 0.5, "OMEGA_GAMMA_H2_PREFACTOR_FOR_RS": 2.47282e-5,
    "H0_FID_FOR_RS_PARAMS_KM_S_MPC": 67.7, "OMEGA_M0_FID_FOR_RS_PARAMS": 0.31, "OMEGA_B0_FID_FOR_RS_PARAMS": 0.0486,
}
_h_fid_for_rs = FIXED_PARAMS["H0_FID_FOR_RS_PARAMS_KM_S_MPC"] / 100.0
FIXED_PARAMS["OMEGA_B0_H2_EFF_FOR_RS"] = FIXED_PARAMS["OMEGA_B0_FID_FOR_RS_PARAMS"] * _h_fid_for_rs**2
FIXED_PARAMS["OMEGA_M0_H2_EFF_FOR_RS"] = FIXED_PARAMS["OMEGA_M0_FID_FOR_RS_PARAMS"] * _h_fid_for_rs**2

# --- Internal Calculation Class for Memoization ---
class USMF_Calculator:
    """A helper class to store memoized results for a single set of cosmological parameters."""
    def __init__(self, cosmo_params):
        p_alpha,k_exp,s_exp,t0_Gyr,A_osc,omega_osc,ti_Gyr,phi_osc = cosmo_params[1:9]
        self.t0_sec_calc = t0_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]
        ti_sec = ti_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]
        self.alpha_core_args = (self.t0_sec_calc,p_alpha,k_exp,s_exp,A_osc,omega_osc,ti_sec,phi_osc)
        self._t_from_z_cache = {}
        # Pre-calculate alpha at t0 for redshift calculations
        self.alpha_at_t0 = self._alpha_func(self.t0_sec_calc)

    def _alpha_func(self, t):
        """Calculates the USMF alpha(t). Returns NaN on math error."""
        t0,p_a,k,s,A,w,ti,phi = self.alpha_core_args
        if t <= 0: return np.nan
        ratio_t0 = t / t0
        try:
            term_power = np.power(ratio_t0, -p_a)
            term_exp = np.exp(k * (np.power(ratio_t0, s) - 1.0))
            # Protect against log(t/ti) domain error if t < ti
            if t < ti : osc_term = 0 # sin term is bounded, so log error is main issue
            else: osc_term = np.sin(w * np.log(t / ti) + phi)
            osc = 1.0 + A * osc_term
            return term_power * term_exp * osc
        except (ValueError, OverflowError): return np.nan
        
    def get_t_from_z(self, z):
        """Inverts the redshift equation to find emission time t_e from redshift z."""
        if z in self._t_from_z_cache: return self._t_from_z_cache[z]
        if abs(z) < 1e-9: self._t_from_z_cache[z] = self.t0_sec_calc; return self.t0_sec_calc
        
        if not np.isfinite(self.alpha_at_t0): return np.nan
        target_alpha = (1.0 + z) * self.alpha_at_t0
        
        def eq(t):
            val = self._alpha_func(t)
            return val - target_alpha if np.isfinite(val) else 1e30 # Return large val on failure
            
        try:
            # Set reasonable bounds for the root search
            low_b = FIXED_PARAMS["TINY_FLOAT"] * self.t0_sec_calc
            high_b = self.t0_sec_calc
            # Search for root using Brent's method
            sol, res = brentq(eq, low_b, high_b, xtol=1e-12 * self.t0_sec_calc, rtol=1e-12, full_output=True)
            if res.converged:
                self._t_from_z_cache[z] = sol
                return sol
        except (ValueError, RuntimeError):
            # brentq fails if the function doesn't cross zero within the bounds
            pass
        return np.nan

def _calculate_for_unique_z(z_array, single_value_calculator_func, *cosmo_params):
    """Optimization: calculates a value for unique redshifts only and maps back."""
    z_array = np.asarray(z_array)
    original_shape = z_array.shape
    if not z_array.shape: # Handle scalar input
        return single_value_calculator_func(z_array.item(), USMF_Calculator(cosmo_params))
    
    z_flat = z_array.flatten()
    unique_z, inverse_indices = np.unique(z_flat, return_inverse=True)
    
    # Instantiate the calculator once for the entire array
    calculator = USMF_Calculator(cosmo_params)
    
    # Calculate results only for the unique redshift values
    unique_results = np.array([single_value_calculator_func(zi, calculator) for zi in unique_z])
    
    # Map the unique results back to the original shape
    return unique_results[inverse_indices].reshape(original_shape)

def _r_Mpc_single(z, calculator):
    """Calculates comoving distance r for a single redshift z."""
    if abs(z) < 1e-9: return 0.0
    t_e = calculator.get_t_from_z(z)
    if not np.isfinite(t_e): return np.nan
    
    def r_integrand(t): 
        val = calculator._alpha_func(t)
        return FIXED_PARAMS["C_LIGHT_KM_S"] / val if (np.isfinite(val) and val > 1e-12) else np.inf
        
    try:
        # Integrate from emission time to observation time
        r_km, _ = quad(r_integrand, t_e, calculator.t0_sec_calc, epsabs=1e-9, epsrel=1e-9)
        return r_km * FIXED_PARAMS["MPC_PER_KM"]
    except Exception:
        return np.nan

def _Hz_single(z, calculator):
    """Calculates the USMF Hubble parameter H(z) for a single redshift."""
    t_e = calculator.get_t_from_z(z)
    if not np.isfinite(t_e): return np.nan
    
    t0,p,k,s,A,w,ti,phi = calculator.alpha_core_args
    alpha_at_te = calculator._alpha_func(t_e)
    if not (np.isfinite(alpha_at_te) and alpha_at_te > 1e-12 and t_e > 1e-12): return np.nan
    
    # Calculate d(alpha)/dt analytically
    osc_val = 1.0 + A * np.sin(w * np.log(t_e / ti) + phi) if t_e >= ti else 1.0
    if abs(osc_val) < 1e-9: return np.nan # Avoid division by zero
    
    d_log_alpha_dt = -p/t_e + (k*s/t_e)*np.power(t_e/t0,s)
    if t_e >= ti:
        d_log_alpha_dt += (A*w/t_e)*np.cos(w*np.log(t_e/ti)+phi)/osc_val
        
    d_alpha_dt_at_te = alpha_at_te * d_log_alpha_dt
    
    if np.isfinite(d_alpha_dt_at_te):
        # H_USMF = - (d(alpha)/dt) / alpha, converted to km/s/Mpc
        return (-d_alpha_dt_at_te / alpha_at_te) * FIXED_PARAMS["MPC_TO_KM"]
    
    return np.nan

# --- Main Model Functions Exposed to the Engine ---

def get_comoving_distance_Mpc(z_array, *cosmo_params):
    return _calculate_for_unique_z(z_array, _r_Mpc_single, *cosmo_params)

def get_Hz_per_Mpc(z_array, *cosmo_params):
    return _calculate_for_unique_z(z_array, _Hz_single, *cosmo_params)
    
def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    H_A = cosmo_params[0]; C_H = 70.0 / H_A
    r_Mpc = get_comoving_distance_Mpc(z_array, *cosmo_params)
    return np.abs(r_Mpc) * np.power(1 + np.asarray(z_array), 2) * C_H
    
def distance_modulus_model(z_array, *cosmo_params):
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    # Use errstate to prevent log10 domain warnings for non-positive dl_mpc
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5 * np.log10(dl_mpc) + 25.0
    # Explicitly set mu to nan where dl_mpc was not positive
    mu[np.asarray(dl_mpc) <= 0] = np.nan
    return mu

def get_angular_diameter_distance_Mpc(z_array, *cosmo_params):
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    return dl_mpc / np.power(1 + np.asarray(z_array), 2)

def get_DV_Mpc(z_array, *cosmo_params):
    da_mpc = get_angular_diameter_distance_Mpc(z_array, *cosmo_params)
    hz = get_Hz_per_Mpc(z_array, *cosmo_params)
    z = np.asarray(z_array)
    with np.errstate(divide='ignore', invalid='ignore'):
        term_in_bracket = (1+z)**2 * da_mpc**2 * FIXED_PARAMS["C_LIGHT_KM_S"] * z / hz
    return np.power(term_in_bracket, 1.0/3.0, where=term_in_bracket >= 0, out=np.full_like(z, np.nan, dtype=float))

def get_sound_horizon_rs_Mpc(*cosmo_params):
    """Calculates the sound horizon based on a simplified early-universe model."""
    t0_age_Gyr_fitted = cosmo_params[4]
    om_b_h2 = FIXED_PARAMS["OMEGA_B0_H2_EFF_FOR_RS"]
    om_m_h2 = FIXED_PARAMS["OMEGA_M0_H2_EFF_FOR_RS"]
    m_fixed = FIXED_PARAMS["USMF_EARLY_ALPHA_POWER_M"] # Assumed early universe power law
    
    def _zd(om_m, om_b):
        """Eisenstein & Hu 1998 fitting formula for z_drag."""
        try:
            g1 = 0.0783 * om_b**(-0.238) / (1.0 + 39.5 * om_b**0.763)
            g2 = 0.560 / (1.0 + 21.1 * om_b**1.81)
            return 1048.0 * np.power(1.0 + 0.00124 * om_b**(-0.738), -1.0) * (1.0 + g1 * om_m**g2)
        except (ValueError, OverflowError): return np.nan

    z_d = _zd(om_m_h2, om_b_h2)
    if not np.isfinite(z_d) or z_d <= 0: return np.nan
    
    def _Hz_early(z, t0, m):
        """Effective H(z) for the simplified early USMF model."""
        t0_sec = t0 * FIXED_PARAMS["SECONDS_PER_GYR"]
        return (m / t0_sec) * np.power(1.0 + z, 1.0 / m) * FIXED_PARAMS["MPC_TO_KM"]

    def _rs_integrand(z, t0, m, om_b):
        """Integrand for sound horizon: c_s(z) / H(z)."""
        hz = _Hz_early(z, t0, m)
        if not np.isfinite(hz) or hz <= 0: return np.inf
        R = (3. * om_b) / (4. * FIXED_PARAMS["OMEGA_GAMMA_H2_PREFACTOR_FOR_RS"] * (1+z))
        cs = FIXED_PARAMS["C_LIGHT_KM_S"] / np.sqrt(3. * (1+R))
        return cs / hz
        
    try:
        rs, _ = quad(_rs_integrand, z_d, np.inf, args=(t0_age_Gyr_fitted, m_fixed, om_b_h2))
        return rs if np.isfinite(rs) and rs > 0 else np.nan
    except Exception:
        return np.nan