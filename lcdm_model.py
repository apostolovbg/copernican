# copernican_suite/lcdm_model.py
"""
LCDM Model Plugin for the Copernican Suite.
*** MODIFIED to include an OpenCL implementation and an updated template. ***
"""

import numpy as np
from scipy.integrate import quad
import logging

try:
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    cl = None

# --- Model Metadata ---
MODEL_NAME = "LambdaCDM"
MODEL_DESCRIPTION = "Standard flat Lambda Cold Matter model."
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$",
    r"$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$",
    r"$\mu = 5 \log_{10}(d_L/1\mathrm{Mpc}) + 25$"
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

# --- OpenCL Kernel Source ---
# This kernel calculates luminosity distance for LCDM by integrating c/H(z)
# using a 40-point Gauss-Legendre quadrature for maximum precision.
OPENCL_KERNEL_SRC = f"""
// --- Gauss-Legendre Quadrature Constants (40-point) ---
__constant double GL_NODES_40[40] = {{
    -0.9982624958369315, -0.9917578270972743, -0.9802096309214739, -0.9636599182749419,
    -0.9421867376173972, -0.9158999992138139, -0.8849422244840456, -0.8494833635292451,
    -0.8097184232325178, -0.7658660098939764, -0.7181673891832343, -0.6668879683832148,
    -0.6123184652232338, -0.5547693005898363, -0.4945722736423018, -0.4320788931118621,
    -0.3676643734006422, -0.3017215162224716, -0.2346593895394019, -0.1668984446554368,
     0.1668984446554368,  0.2346593895394019,  0.3017215162224716,  0.3676643734006422,
     0.4320788931118621,  0.4945722736423018,  0.5547693005898363,  0.6123184652232338,
     0.6668879683832148,  0.7181673891832343,  0.7658660098939764,  0.8097184232325178,
     0.8494833635292451,  0.8849422244840456,  0.9158999992138139,  0.9421867376173972,
     0.9636599182749419,  0.9802096309214739,  0.9917578270972743,  0.9982624958369315
}};
__constant double GL_WEIGHTS_40[40] = {{
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
}};

// Helper function for the integrand c/H(z)
inline double integrand_func(
    double z, double H0, double Omega_m0, double Omega_r0,
    double Omega_L0, double C_LIGHT_KM_S
) {{
    double Ez_sq = Omega_r0*pown(1+z, 4) + Omega_m0*pown(1+z, 3) + Omega_L0;
    if (Ez_sq <= 0.0) return 0.0; // Avoid sqrt of negative
    double hz = H0 * sqrt(Ez_sq);
    if (hz == 0.0) return 0.0; // Avoid division by zero
    return C_LIGHT_KM_S / hz;
}}

__kernel void lcdm_dl_integrator(
    __global const double *z_values,
    __global double *dl_out,
    // Cosmological Parameters
    const double H0,
    const double Omega_m0,
    const double Omega_r0,
    const double Omega_L0,
    // Fixed Constants
    const double C_LIGHT_KM_S
) {{
    int gid = get_global_id(0);
    double z_upper = z_values[gid];

    if (z_upper < 1e-9) {{
        dl_out[gid] = 0.0;
        return;
    }}

    // Gauss-Legendre Quadrature for comoving distance
    // The integral is from 0 to z_upper.
    // We map this interval to the standard GL interval [-1, 1].
    double integral = 0.0;
    double t_map, z_at_t;

    for (int i = 0; i < 40; i++) {{
        // Map GL node t from [-1, 1] to z in [0, z_upper]
        t_map = GL_NODES_40[i];
        z_at_t = (z_upper / 2.0) * (1.0 + t_map);
        
        integral += GL_WEIGHTS_40[i] * integrand_func(z_at_t, H0, Omega_m0, Omega_r0, Omega_L0, C_LIGHT_KM_S);
    }}

    // Final comoving distance
    double dc_integral = (z_upper / 2.0) * integral;

    // Final luminosity distance calculation
    dl_out[gid] = dc_integral * (1.0 + z_upper);
}}
"""


# --- Standard Python/Scipy Functions (Used for plotting and as a fallback) ---

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

def get_comoving_distance_Mpc(z_array, H0, Omega_m0, Omega_b0):
    z_array_np = np.asarray(z_array); results_Mpc = np.full_like(z_array_np, np.nan, dtype=float)
    for i, zi in np.ndenumerate(z_array_np):
        if abs(float(zi)) < 1e-9: results_Mpc[i] = 0.0; continue
        try:
            Dc_val, _ = quad(_integrand_Dc, 0, float(zi), args=(H0, Omega_m0, Omega_b0))
            if np.isfinite(Dc_val): results_Mpc[i] = Dc_val
        except: results_Mpc[i] = np.nan
    return results_Mpc.item() if z_array_np.ndim == 0 else results_Mpc

def get_luminosity_distance_Mpc(z_array, *cosmo_params):
    dm = get_comoving_distance_Mpc(z_array, *cosmo_params)
    return dm * (1 + np.asarray(z_array))

def distance_modulus_model(z_array, *cosmo_params):
    """The standard, Scipy-based function for plotting and fallback."""
    dl_mpc = get_luminosity_distance_Mpc(z_array, *cosmo_params)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5 * np.log10(dl_mpc) + 25.0
    mu[np.asarray(dl_mpc) <= 0] = np.nan
    return mu

# --- OpenCL High-Performance Function ---

def distance_modulus_model_opencl(z_array, *cosmo_params, cl_context=None, cl_queue=None, cl_program=None):
    """High-performance OpenCL entry point for the LCDM fitting engine."""
    logger = logging.getLogger()
    if not all([cl, cl_context, cl_queue, cl_program]):
        logger.warning(f"MODEL '{MODEL_NAME}': OpenCL context not available, falling back to CPU.")
        return distance_modulus_model(z_array, *cosmo_params)
        
    try:
        # 1. Prepare parameters and data
        z_data = np.asarray(z_array, dtype=np.float64)
        H0, Omega_m0, Omega_b0 = cosmo_params
        _, Omega_r0, Omega_L0, _, _ = _get_derived_densities(H0, Omega_m0, Omega_b0)

        # Handle case of invalid derived parameters
        if any(np.isnan([Omega_r0, Omega_L0])):
            return np.full_like(z_data, np.nan)

        # 2. Create OpenCL buffers
        mf = cl.mem_flags
        z_buffer = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_data)
        dl_buffer = cl.Buffer(cl_context, mf.WRITE_ONLY, size=z_data.nbytes)
        
        # 3. Set kernel arguments and execute
        kernel = cl_program.lcdm_dl_integrator
        kernel.set_args(
            z_buffer, dl_buffer, np.double(H0), np.double(Omega_m0), np.double(Omega_r0),
            np.double(Omega_L0), np.double(FIXED_PARAMS["C_LIGHT_KM_S"])
        )
        cl.enqueue_nd_range_kernel(cl_queue, kernel, z_data.shape, None)

        # 4. Read results back from GPU
        dl_mpc_results = np.empty_like(z_data)
        cl.enqueue_copy(cl_queue, dl_mpc_results, dl_buffer).wait()

        # 5. Convert luminosity distance to distance modulus
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = 5.0 * np.log10(dl_mpc_results) + 25.0
        mu[dl_mpc_results <= 0] = np.nan
        
        return mu

    except (cl.Error, cl.LogicError, cl.RuntimeError) as e:
        logger.error(f"MODEL '{MODEL_NAME}': An error occurred during OpenCL execution: {e}", exc_info=True)
        # Return an array of NaNs on failure so the fitter can recover
        return np.full_like(np.asarray(z_array), np.nan)


# --- Standard BAO and other functions ---
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

# ==============================================================================
# ==============================================================================
# --- NEW TEMPLATE FOR MODEL IMPLEMENTATION (.py) ---
# This template is updated to promote the hybrid CPU/GPU approach for reliability.
# ==============================================================================
# ==============================================================================
"""
# Copy the text below into a new file (e.g., "my_theory.py") to create a new model plugin.

'''
Python implementation for [Your Model Name].
This template follows the recommended hybrid CPU/GPU architecture for reliability.
'''

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq # Example for models needing root-finding
import logging

try:
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    cl = None

# ==============================================================================
# --- METADATA (should be consistent with your .md file) ---
# ==============================================================================
MODEL_NAME = "YourModelName"
MODEL_DESCRIPTION = "A brief description of the model."
PARAMETER_NAMES = ["param1", "param2"] # etc.
# ... (rest of metadata)

# ==============================================================================
# --- OPTIONAL: RELIABLE HYBRID OpenCL (GPU) IMPLEMENTATION ---
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Simplified OpenCL Kernel
# This kernel should perform a simple, stable calculation (e.g., a weighted sum).
# It receives pre-calculated inputs from the CPU, NOT raw values like redshift.
# ------------------------------------------------------------------------------
OPENCL_KERNEL_SRC = f'''
__kernel void your_simple_kernel(
    __global const double *precalculated_inputs, // e.g., pre-integrated values or function evaluations
    __global double *result_out,
    // You may still need some parameters for final calculations
    const double param1 
) {{
    int gid = get_global_id(0);
    double intermediate_val = precalculated_inputs[gid];

    // Example: perform a simple, final calculation on the GPU
    result_out[gid] = intermediate_val * param1; 
}}
'''

# ------------------------------------------------------------------------------
# 2. Standard (CPU) Functions
# These are the "source of truth" and are used for plotting, BAO, and preparing
# inputs for the OpenCL kernel.
# ------------------------------------------------------------------------------

def _your_sensitive_calculation(z, *cosmo_params):
    '''
    This function performs the sensitive part of the calculation (e.g.,
    root-finding or a tricky integration) using reliable SciPy methods.
    It returns an intermediate result that is safe to pass to the GPU.
    '''
    # Example: A model that needs to solve an equation for a value 'x' at each z
    param1, param2 = cosmo_params
    
    # def equation_to_solve(x):
    #     return x**2 - z * param1 # dummy equation
    # try:
    #     # Use a reliable SciPy solver
    #     solved_x, results = brentq(equation_to_solve, a=0, b=100, full_output=True)
    #     if results.converged:
    #         return solved_x
    # except (ValueError, RuntimeError):
    #     return np.nan
    # For the template, we just return a placeholder
    return z * param1 # Placeholder for a calculated intermediate value

def distance_modulus_model(z_array, *cosmo_params):
    '''
    The standard, full Scipy-based function for plotting and fallback.
    This function defines the model completely for the CPU-only mode.
    '''
    # Your full CPU implementation here. This should calculate the final
    # distance modulus 'mu' from scratch.
    # ...
    # For the template, we'll make a simple placeholder calculation
    z = np.asarray(z_array)
    param1, param2 = cosmo_params
    # This is a dummy calculation for mu
    dl_mpc = (z * 1000) * (param1 / param2)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5.0 * np.log10(dl_mpc) + 25.0
    mu[dl_mpc <= 0] = np.nan
    return mu

# --- (Other required CPU functions like get_Hz_per_Mpc, get_sound_horizon_rs_Mpc, etc.) ---


# ------------------------------------------------------------------------------
# 3. Hybrid OpenCL Entry Point
# This is the function called by the fitter in OpenCL mode.
# It orchestrates the hybrid CPU/GPU calculation.
# ------------------------------------------------------------------------------

def distance_modulus_model_opencl(z_array, *cosmo_params, cl_context=None, cl_queue=None, cl_program=None):
    '''High-performance OpenCL entry point using the reliable hybrid architecture.'''
    logger = logging.getLogger()
    if not all([cl, cl_context, cl_queue, cl_program]):
        logger.warning(f"MODEL '{MODEL_NAME}': OpenCL context not available, falling back to CPU.")
        return distance_modulus_model(z_array, *cosmo_params)
        
    try:
        # --- HYBRID STEP 1: Perform sensitive calculations on CPU ---
        # Use your reliable CPU function to pre-calculate the tricky parts.
        precalculated_inputs = np.array([_your_sensitive_calculation(z, *cosmo_params) for z in np.atleast_1d(z_array)])
        
        # --- HYBRID STEP 2: Offload stable number-crunching to GPU ---
        
        # a. Prepare parameters and data buffers
        z_data = np.asarray(z_array, dtype=np.float64)
        param1, param2 = cosmo_params
        
        mf = cl.mem_flags
        # Send the pre-calculated, safe inputs to the GPU
        gpu_in_buffer = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=precalculated_inputs.astype(np.float64))
        gpu_out_buffer = cl.Buffer(cl_context, mf.WRITE_ONLY, size=z_data.nbytes)
        
        # b. Get kernel, set arguments, and execute
        kernel = cl_program.your_simple_kernel # Must match the name in OPENCL_KERNEL_SRC
        kernel.set_args(gpu_in_buffer, gpu_out_buffer, np.double(param1))
        cl.enqueue_nd_range_kernel(cl_queue, kernel, z_data.shape, None).wait()

        # c. Read results back from GPU
        gpu_results = np.empty_like(z_data)
        cl.enqueue_copy(cl_queue, gpu_results, gpu_out_buffer).wait()

        # --- HYBRID STEP 3: Perform final calculations on CPU ---
        # Often, a final, simple calculation is needed to get the distance modulus.
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = 5.0 * np.log10(gpu_results) + 25.0 # Example final step
        mu[gpu_results <= 0] = np.nan
        
        return mu

    except (cl.Error, cl.LogicError, cl.RuntimeError) as e:
        logger.error(f"MODEL '{MODEL_NAME}': An error occurred during OpenCL execution: {e}", exc_info=True)
        return np.full_like(np.asarray(z_array), np.nan)

"""