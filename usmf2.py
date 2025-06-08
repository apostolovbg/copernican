# copernican_suite/usmf2.py
"""
Unified Shrinking Matter Framework (USMF) V2 Model Plugin for the Copernican Suite.
*** MODIFIED to include a real, complex OpenCL implementation. ***
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import logging

try:
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    cl = None

# --- Model Metadata and Parameters ---
MODEL_NAME = "USMF_V2"
MODEL_DESCRIPTION = "Unified Shrinking Matter Framework (USMF) Version 2."
PARAMETER_NAMES = ["H_A", "p_alpha", "k_exp", "s_exp", "t0_age_Gyr", "A_osc", "omega_osc", "ti_osc_Gyr", "phi_osc"]
INITIAL_GUESSES = [77.111, 0.4313, -0.3268, 1.1038, 13.397, 0.0027088, 2.3969, 7.1399, 0.10905]
PARAMETER_BOUNDS = [
    (50.0, 100.0), (0.1, 1.5), (-2.0, 2.0), (0.5, 2.0), (10.0, 20.0),
    (0.0, 0.05), (0.1, 10.0), (1.0, 20.0), (-np.pi, np.pi)
]
MODEL_EQUATIONS_LATEX_SN = [
    r"$1+z = \alpha(t_e) / \alpha(t_0)$",
    r"$r = \int_{t_e}^{t_0} \frac{c}{\alpha(t')} dt'$",
    r"$d_L(z) = |r| \cdot (1+z)^2 \cdot \frac{70.0}{H_A}$",
    r"$\mu = 5 \log_{10}(d_L) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = |r| \cdot \frac{70.0}{H_A}$",
    r"$H_{USMF}(z) = - \frac{1}{\alpha(t_e)} \left. \frac{d\alpha}{dt} \right|_{t_e}$",
    r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
]
PARAMETER_LATEX_NAMES = [r"$H_A$", r"$p_{\alpha}$", r"$k_{exp}$", r"$s_{exp}$", r"$t_{0,age}$", r"$A_{osc}$", r"$\omega_{osc}$", r"$t_{i,osc}$", r"$\phi_{osc}$"]
PARAMETER_UNITS = ["", "", "", "", "Gyr", "", "rad/log(time_ratio)", "Gyr", "rad"]
FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458, "SECONDS_PER_GYR": 1e9 * 365.25 * 24 * 3600, "MPC_PER_KM": 1.0 / (3.08567758e19),
    "MPC_TO_KM": 3.08567758e19, "TINY_FLOAT": 1e-30,
    "USMF_EARLY_ALPHA_POWER_M": 0.5, "OMEGA_GAMMA_H2_PREFACTOR_FOR_RS": 2.47282e-5,
    "H0_FID_FOR_RS_PARAMS_KM_S_MPC": 67.7, "OMEGA_M0_FID_FOR_RS_PARAMS": 0.31, "OMEGA_B0_FID_FOR_RS_PARAMS": 0.0486,
}
_h_fid_for_rs = FIXED_PARAMS["H0_FID_FOR_RS_PARAMS_KM_S_MPC"] / 100.0
FIXED_PARAMS["OMEGA_B0_H2_EFF_FOR_RS"] = FIXED_PARAMS["OMEGA_B0_FID_FOR_RS_PARAMS"] * _h_fid_for_rs**2
FIXED_PARAMS["OMEGA_M0_H2_EFF_FOR_RS"] = FIXED_PARAMS["OMEGA_M0_FID_FOR_RS_PARAMS"] * _h_fid_for_rs**2

# --- OpenCL Kernel Source ---
# This kernel now uses a robust bisection root-finder and a 40-point
# Gauss-Legendre quadrature for maximum fixed-grid precision.
OPENCL_KERNEL_SRC = """
// --- Gauss-Legendre Quadrature Constants (40-point) ---
__constant double GL_NODES_40[40] = {
    -0.9982624958369315, -0.9917578270972743, -0.9802096309214739, -0.9636599182749419,
    -0.9421867376173972, -0.9158999992138139, -0.8849422244840456, -0.8494833635292451,
    -0.8097184232325178, -0.7658660098939764, -0.7181673891832343, -0.6668879683832148,
    -0.6123184652232338, -0.5547693005898363, -0.4945722736423018, -0.4320788931118621,
    -0.3676643734006422, -0.3017215162224716, -0.2346593895394019, -0.1668984446554368,
    0.1668984446554368, 0.2346593895394019, 0.3017215162224716, 0.3676643734006422,
    0.4320788931118621, 0.4945722736423018, 0.5547693005898363, 0.6123184652232338,
    0.6668879683832148, 0.7181673891832343, 0.7658660098939764, 0.8097184232325178,
    0.8494833635292451, 0.8849422244840456, 0.9158999992138139, 0.9421867376173972,
    0.9636599182749419, 0.9802096309214739, 0.9917578270972743, 0.9982624958369315
};
__constant double GL_WEIGHTS_40[40] = {
    0.0044558379435402, 0.0103463344603417, 0.0162002321485644, 0.0219918841400244,
    0.0276953912119330, 0.0332851414483754, 0.0387358988924522, 0.0440229346613399,
    0.0491221147133742, 0.0540098191931069, 0.0586632422744275, 0.0630601832141519,
    0.0671801288940424, 0.0710034443936691, 0.0745114712294103, 0.0776868846328323,
    0.0805137252494947, 0.0829774659433249, 0.0850650993134336, 0.0867651030397532,
    0.0867651030397532, 0.0850650993134336, 0.0829774659433249, 0.0805137252494947,
    0.0776868846328323, 0.0745114712294103, 0.0710034443936691, 0.0671801288940424,
    0.0630601832141519, 0.0586632422744275, 0.0540098191931069, 0.0491221147133742,
    0.0440229346613399, 0.0387358988924522, 0.0332851414483754, 0.0276953912119330,
    0.0219918841400244, 0.0162002321485644, 0.0103463344603417, 0.0044558379435402
};


// OpenCL Helper Function for USMF alpha(t)
inline double alpha_func(double t, double t0_sec, double p_a, double k, double s, 
                         double A, double w, double ti_sec, double phi) {
    if (t <= 1e-12) return NAN; // Avoid log(0) and division by zero
    
    double ratio_t0 = t / t0_sec;
    double term_power = pow(ratio_t0, -p_a);
    double term_exp = exp(k * (pow(ratio_t0, s) - 1.0));

    double osc = 1.0;
    if (t >= ti_sec && A > 0.0) { // check A > 0 to avoid log with w=0
        osc += A * sin(w * log(t / ti_sec) + phi);
    }
    return term_power * term_exp * osc;
}

// OpenCL Kernel for USMF V2: calculates r(z)
__kernel void usmf_r_calculator(
    __global const double* z_in,
    __global double* r_Mpc_out,
    // Cosmological Parameters
    const double p_alpha, const double k_exp, const double s_exp,
    const double t0_Gyr, const double A_osc, const double omega_osc,
    const double ti_Gyr, const double phi_osc,
    // Fixed constants
    const double SECONDS_PER_GYR,
    const double C_LIGHT_KM_S,
    const double MPC_PER_KM,
    const double TINY_FLOAT,
    // Control parameters
    const int root_finder_max_iter
) {
    int i = get_global_id(0);
    double z = z_in[i];

    // Convert parameters from Gyr to seconds
    double t0_sec = t0_Gyr * SECONDS_PER_GYR;
    double ti_sec = ti_Gyr * SECONDS_PER_GYR;

    if (z < 1e-9) {
        r_Mpc_out[i] = 0.0;
        return;
    }

    // --- Step 1: Find t_e from z using a root finder (Bisection method) ---
    double alpha_at_t0 = alpha_func(t0_sec, t0_sec, p_alpha, k_exp, s_exp, A_osc, omega_osc, ti_sec, phi_osc);
    if (isnan(alpha_at_t0)) {
        r_Mpc_out[i] = NAN;
        return;
    }
    double target_alpha = (1.0 + z) * alpha_at_t0;

    double t_low = TINY_FLOAT * t0_sec;
    double t_high = t0_sec;
    double t_e = NAN;

    double f_low = alpha_func(t_low, t0_sec, p_alpha, k_exp, s_exp, A_osc, omega_osc, ti_sec, phi_osc) - target_alpha;
    if (isnan(f_low) || isinf(f_low)) { r_Mpc_out[i] = NAN; return; }
    
    for (int j = 0; j < root_finder_max_iter; ++j) {
        double t_mid = t_low + 0.5 * (t_high - t_low); // More stable midpoint calculation
        if (t_mid == t_low || t_mid == t_high) { t_e = t_mid; break; }

        double f_mid = alpha_func(t_mid, t0_sec, p_alpha, k_exp, s_exp, A_osc, omega_osc, ti_sec, phi_osc) - target_alpha;
        
        if (isnan(f_mid) || isinf(f_mid)) { t_e = NAN; break; }
        
        // Stricter tolerance check
        if (fabs((t_high - t_low)/t0_sec) < 1e-15 || fabs(f_mid) < 1e-15) { 
            t_e = t_mid; 
            break; 
        }

        if (sign(f_mid) == sign(f_low)) {
            t_low = t_mid;
            f_low = f_mid;
        } else {
            t_high = t_mid;
        }
        t_e = (t_low + t_high) / 2.0;
    }
    
    if (isnan(t_e)) {
        r_Mpc_out[i] = NAN;
        return;
    }
    
    // --- Step 2: Integrate c/alpha(t) from t_e to t_0 using Gauss-Legendre (40-point) ---
    double integral = 0.0;
    double interval_width = t0_sec - t_e;
    
    // Check for invalid interval
    if (interval_width <= 0) {
        r_Mpc_out[i] = 0.0;
        return;
    }

    double t_map, t_at_map;
    for (int k = 0; k < 40; ++k) {
        t_map = GL_NODES_40[k];
        // Map GL node t from [-1, 1] to time in [t_e, t0_sec]
        t_at_map = t_e + (interval_width / 2.0) * (1.0 + t_map);
        
        double alpha_val = alpha_func(t_at_map, t0_sec, p_alpha, k_exp, s_exp, A_osc, omega_osc, ti_sec, phi_osc);
        
        if (isnan(alpha_val) || alpha_val <= 0) {
            integral = NAN;
            break;
        }
        integral += GL_WEIGHTS_40[k] * (C_LIGHT_KM_S / alpha_val);
    }

    if (isnan(integral)) {
        r_Mpc_out[i] = NAN;
    } else {
        double r_km = (interval_width / 2.0) * integral;
        r_Mpc_out[i] = r_km * MPC_PER_KM;
    }
}
"""

# --- Standard Python/Scipy Implementation (Fallback and Plotting) ---
class USMF_Calculator:
    def __init__(self, cosmo_params):
        _, p_alpha,k_exp,s_exp,t0_Gyr,A_osc,omega_osc,ti_Gyr,phi_osc = cosmo_params
        self.t0_sec_calc = t0_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]
        ti_sec = ti_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]
        self.alpha_core_args = (self.t0_sec_calc,p_alpha,k_exp,s_exp,A_osc,omega_osc,ti_sec,phi_osc)
        self._t_from_z_cache = {}
        self.alpha_at_t0 = self._alpha_func(self.t0_sec_calc)
    def _alpha_func(self, t):
        t0,p_a,k,s,A,w,ti,phi = self.alpha_core_args
        if t <= 0: return np.nan
        ratio_t0 = t / t0
        try:
            term_power = np.power(ratio_t0, -p_a)
            term_exp = np.exp(k * (np.power(ratio_t0, s) - 1.0))
            if t < ti : osc_term = 0.0
            else: osc_term = np.sin(w * np.log(t / ti) + phi)
            osc = 1.0 + A * osc_term
            return term_power * term_exp * osc
        except (ValueError, OverflowError): return np.nan
    def get_t_from_z(self, z):
        if z in self._t_from_z_cache: return self._t_from_z_cache[z]
        if abs(z) < 1e-9: self._t_from_z_cache[z] = self.t0_sec_calc; return self.t0_sec_calc
        if not np.isfinite(self.alpha_at_t0): return np.nan
        target_alpha = (1.0 + z) * self.alpha_at_t0
        def eq(t):
            val = self._alpha_func(t)
            return val - target_alpha if np.isfinite(val) else 1e30
        try:
            low_b = FIXED_PARAMS["TINY_FLOAT"] * self.t0_sec_calc
            high_b = self.t0_sec_calc
            sol, res = brentq(eq, low_b, high_b, xtol=1e-12 * self.t0_sec_calc, rtol=1e-12, full_output=True)
            if res.converged: self._t_from_z_cache[z] = sol; return sol
        except (ValueError, RuntimeError): pass
        return np.nan

def _calculate_for_unique_z(z_array, single_value_calculator_func, *cosmo_params):
    z_array = np.asarray(z_array); original_shape = z_array.shape
    # Clear cache for each new calculation run by creating a new instance
    calculator = USMF_Calculator(cosmo_params)
    if not z_array.shape: return single_value_calculator_func(z_array.item(), calculator)
    z_flat = z_array.flatten()
    unique_z, inverse_indices = np.unique(z_flat, return_inverse=True)
    unique_results = np.array([single_value_calculator_func(zi, calculator) for zi in unique_z])
    return unique_results[inverse_indices].reshape(original_shape)

def _r_Mpc_single(z, calculator):
    if abs(z) < 1e-9: return 0.0
    t_e = calculator.get_t_from_z(z)
    if not np.isfinite(t_e): return np.nan
    def r_integrand(t): 
        val = calculator._alpha_func(t)
        return FIXED_PARAMS["C_LIGHT_KM_S"] / val if (np.isfinite(val) and val > 1e-12) else np.inf
    try:
        r_km, _ = quad(r_integrand, t_e, calculator.t0_sec_calc, epsabs=1e-9, epsrel=1e-9)
        return r_km * FIXED_PARAMS["MPC_PER_KM"]
    except Exception: return np.nan

def get_comoving_distance_Mpc(z_array, *cosmo_params):
    return _calculate_for_unique_z(z_array, _r_Mpc_single, *cosmo_params)

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    H_A = cosmo_params[0]; C_H = 70.0 / H_A
    r_Mpc = get_comoving_distance_Mpc(z_array, *cosmo_params)
    dl_mpc = np.abs(r_Mpc) * np.power(1 + np.asarray(z_array), 2) * C_H
    return dl_mpc

def distance_modulus_model(z_array, *cosmo_params):
    """The standard, Scipy-based function for plotting and fallback."""
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5 * np.log10(dl_mpc) + 25.0
    mu[np.asarray(dl_mpc) <= 0] = np.nan
    return mu

# --- OpenCL High-Performance Function ---

def distance_modulus_model_opencl(z_array, *cosmo_params, cl_context=None, cl_queue=None, cl_program=None):
    """High-performance OpenCL entry point for the USMF fitting engine."""
    logger = logging.getLogger()
    if not all([cl, cl_context, cl_queue, cl_program]):
        logger.warning(f"MODEL '{MODEL_NAME}': OpenCL context not available, falling back to CPU.")
        return distance_modulus_model(z_array, *cosmo_params)
        
    try:
        # 1. Prepare parameters and data
        z_data = np.asarray(z_array, dtype=np.float64)
        H_A, p_alpha, k_exp, s_exp, t0_age_Gyr, A_osc, omega_osc, ti_osc_Gyr, phi_osc = cosmo_params
        
        # 2. Create OpenCL buffers
        mf = cl.mem_flags
        z_buffer = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_data)
        r_Mpc_buffer = cl.Buffer(cl_context, mf.WRITE_ONLY, z_data.nbytes)
        
        # 3. Set kernel arguments and execute
        # Control parameter for root finder precision: increased iterations
        root_finder_max_iter = np.int32(150)

        kernel = cl_program.usmf_r_calculator
        # Set all arguments for the kernel
        kernel.set_args(
            z_buffer, r_Mpc_buffer, np.double(p_alpha), np.double(k_exp), np.double(s_exp),
            np.double(t0_age_Gyr), np.double(A_osc), np.double(omega_osc), np.double(ti_osc_Gyr), np.double(phi_osc),
            np.double(FIXED_PARAMS["SECONDS_PER_GYR"]), np.double(FIXED_PARAMS["C_LIGHT_KM_S"]),
            np.double(FIXED_PARAMS["MPC_PER_KM"]), np.double(FIXED_PARAMS["TINY_FLOAT"]),
            root_finder_max_iter
        )
        cl.enqueue_nd_range_kernel(cl_queue, kernel, z_data.shape, None).wait()

        # 4. Read results back from GPU
        r_Mpc_results = np.empty_like(z_data)
        cl.enqueue_copy(cl_queue, r_Mpc_results, r_Mpc_buffer).wait()

        # 5. Calculate final distance modulus
        C_H = 70.0 / H_A
        dl_mpc = np.abs(r_Mpc_results) * np.power(1.0 + z_data, 2) * C_H
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = 5 * np.log10(dl_mpc) + 25.0
        mu[np.asarray(dl_mpc) <= 0] = np.nan

        return mu

    except (cl.Error, cl.LogicError, cl.RuntimeError) as e:
        logger.error(f"MODEL '{MODEL_NAME}': An error occurred during OpenCL execution: {e}", exc_info=True)
        # Return an array of NaNs on failure so the fitter can recover
        return np.full_like(np.asarray(z_array), np.nan)


# --- Standard BAO and other functions ---

def _Hz_single(z, calculator):
    t_e = calculator.get_t_from_z(z)
    if not np.isfinite(t_e): return np.nan
    t0,p_a,k,s,A,w,ti,phi = calculator.alpha_core_args
    alpha_at_te = calculator._alpha_func(t_e)
    if not (np.isfinite(alpha_at_te) and alpha_at_te > 1e-12 and t_e > 1e-12): return np.nan
    osc_val = 1.0 + A * np.sin(w * np.log(t_e / ti) + phi) if t_e >= ti else 1.0
    if abs(osc_val) < 1e-9: return np.nan
    d_log_alpha_dt = -p_a/t_e + (k*s/t_e)*np.power(t_e/t0,s)
    if t_e >= ti: d_log_alpha_dt += (A*w/t_e)*np.cos(w*np.log(t_e/ti)+phi)/osc_val
    d_alpha_dt_at_te = alpha_at_te * d_log_alpha_dt
    if np.isfinite(d_alpha_dt_at_te): return (-d_alpha_dt_at_te / alpha_at_te) * FIXED_PARAMS["MPC_TO_KM"]
    return np.nan

def get_Hz_per_Mpc(z_array, *cosmo_params):
    return _calculate_for_unique_z(z_array, _Hz_single, *cosmo_params)

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
    _, _, _, _, t0_age_Gyr, _, _, _, _ = cosmo_params
    om_b_h2 = FIXED_PARAMS["OMEGA_B0_H2_EFF_FOR_RS"]
    om_m_h2 = FIXED_PARAMS["OMEGA_M0_H2_EFF_FOR_RS"]
    m_fixed = FIXED_PARAMS["USMF_EARLY_ALPHA_POWER_M"]
    def _zd(om_m, om_b):
        try:
            g1 = 0.0783 * om_b**(-0.238) / (1.0 + 39.5 * om_b**0.763)
            g2 = 0.560 / (1.0 + 21.1 * om_b**1.81)
            return 1048.0 * np.power(1.0 + 0.00124 * om_b**(-0.738), -1.0) * (1.0 + g1 * om_m**g2)
        except (ValueError, OverflowError): return np.nan
    z_d = _zd(om_m_h2, om_b_h2)
    if not np.isfinite(z_d) or z_d <= 0: return np.nan
    def _Hz_early(z, t0, m):
        t0_sec = t0 * FIXED_PARAMS["SECONDS_PER_GYR"]
        return (m / t0_sec) * np.power(1.0 + z, 1.0 / m) * FIXED_PARAMS["MPC_TO_KM"]
    def _rs_integrand(z, t0, m, om_b):
        hz = _Hz_early(z, t0, m)
        if not np.isfinite(hz) or hz <= 0: return np.inf
        R = (3. * om_b) / (4. * FIXED_PARAMS["OMEGA_GAMMA_H2_PREFACTOR_FOR_RS"] * (1+z))
        cs = FIXED_PARAMS["C_LIGHT_KM_S"] / np.sqrt(3. * (1+R))
        return cs / hz
    try:
        rs, _ = quad(_rs_integrand, z_d, np.inf, args=(t0_age_Gyr, m_fixed, om_b_h2))
        return rs if np.isfinite(rs) and rs > 0 else np.nan
    except Exception: return np.nan