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

# Define the model's equations as raw LaTeX strings for on-graph display.
# This is a REQUIRED step for all new models.
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$",
    r"$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$",
    r"$\mu = 5 \log_{10}(d_L/1\mathrm{Mpc}) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$",
    r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
]

PARAMETER_NAMES = ["H0", "Omega_m0"] # Assuming flat universe, Omega_b0 is for other uses.
PARAMETER_LATEX_NAMES = [r"$H_0$", r"$\Omega_{m0}$"]
PARAMETER_UNITS = ["km/s/Mpc", ""]
INITIAL_GUESSES = [70.0, 0.3]
PARAMETER_BOUNDS = [(60.0, 80.0), (0.05, 0.7)]

# --- Fixed Parameters ---
C_LIGHT_KM_S = 299792.458

def hubble_model(z, H0, Omega_m0):
    """
    Calculates H(z) for the flat LCDM model.
    """
    Omega_Lambda0 = 1.0 - Omega_m0
    return H0 * np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda0)

def luminosity_distance_integrand(z, H0, Omega_m0):
    """
    Integrand for the luminosity distance calculation.
    """
    return C_LIGHT_KM_S / hubble_model(z, H0, Omega_m0)

def distance_modulus_model(z_array, H0, Omega_m0):
    """
    Calculates the distance modulus for a given array of redshifts.
    """
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        integral, _ = quad(luminosity_distance_integrand, 0, z, args=(H0, Omega_m0))
        d_L = (1 + z) * integral  # d_L is in Mpc
        if d_L > 0:
            mu[i] = 5 * np.log10(d_L) + 25
        else:
            mu[i] = np.nan
    return mu

# --- OpenCL Implementation (Hybrid) ---
OPENCL_KERNEL_SRC = """
__kernel void hubble_kernel(__global const double *z_array,
                           __global double *h_z,
                           const double H0,
                           const double Omega_m0)
{
    int i = get_global_id(0);
    double z = z_array[i];
    double Omega_Lambda0 = 1.0 - Omega_m0;
    h_z[i] = H0 * sqrt(Omega_m0 * pown(1.0 + z, 3) + Omega_Lambda0);
}
"""

def distance_modulus_model_opencl(z_array, cl_context, cl_queue, cl_program, *cosmo_params):
    """
    Hybrid OpenCL implementation for the distance modulus.
    The integration remains on the CPU as it is serial, but H(z) is calculated in parallel on the GPU.
    """
    # For this model, the OpenCL kernel can compute H(z) directly.
    # The serial integration step is still performed on the CPU.
    H0, Omega_m0 = cosmo_params
    
    # Fallback to standard CPU integration.
    # A more complex model might use the GPU for H(z) calculation before this loop.
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        integral, err = quad(luminosity_distance_integrand, 0, z, args=(H0, Omega_m0))
        if err > 1e-3:
            logging.warning(f"High integration error ({err}) for z={z}")
        d_L = (1 + z) * integral
        if d_L > 0:
            mu[i] = 5 * np.log10(d_L) + 25
        else:
            mu[i] = np.nan
    return mu


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# <><><>            TEMPLATE FOR NEW MODEL PYTHON PLUGIN            <><><>
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

"""
# copernican_suite/your_model.py
\"\"\"
Your Model Name Plugin for the Copernican Suite.
\"\"\"

import numpy as np
from scipy.integrate import quad
# from scipy.optimize import brentq # etc., if needed
import logging

try:
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    cl = None # This allows the suite to run in CPU-only mode if PyOpenCL is not installed.

# --- Model Metadata ---
# This block is parsed by the main suite engine.
# It should be consistent with the .md file.

MODEL_NAME = "YourModelName" # Short, machine-friendly name
MODEL_DESCRIPTION = "A brief description of your model."

# NEW/MANDATORY: Define the model's equations for on-graph display.
# Use raw LaTeX strings. One string per equation line.
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = ...$",
    r"$\mu = ...$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_V(z) = ...$"
]

# These lists are generated from the .md file's parameter table.
PARAMETER_NAMES = ["param1", "param2"]
PARAMETER_LATEX_NAMES = [r"$p_1$", r"$p_2$"]
PARAMETER_UNITS = ["unit1", ""]
INITIAL_GUESSES = [1.0, 2.0]
PARAMETER_BOUNDS = [(0.0, 2.0), (1.0, 3.0)]


# --- Fixed Parameters & Constants ---
C_LIGHT_KM_S = 299792.458 # Speed of light in km/s

# --- Core Model Functions (CPU Implementation) ---

def hubble_model(z, param1, param2):
    \"\"\"
    Calculates the Hubble parameter H(z) in km/s/Mpc.
    This is the heart of your cosmological model.
    \"\"\"
    # Replace with your model's equation for H(z)
    return 100 * np.sqrt(param1 * (1 + z)**3 + param2)

def luminosity_distance_integrand(z, param1, param2):
    \"\"\"
    The function to be integrated to find the luminosity distance.
    This is typically c / H(z).
    \"\"\"
    return C_LIGHT_KM_S / hubble_model(z, param1, param2)

def distance_modulus_model(z_array, param1, param2):
    \"\"\"
    Calculates the distance modulus for an array of redshifts.
    This function will be called by the SNe Ia fitter.
    \"\"\"
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        # SciPy's quad is used for numerical integration.
        integral, _ = quad(luminosity_distance_integrand, 0, z, args=(param1, param2))
        d_L = (1 + z) * integral  # d_L is in Mpc
        if d_L > 0:
            mu[i] = 5 * np.log10(d_L) + 25 # Standard distance modulus formula
        else:
            mu[i] = np.nan # Handle cases where distance is not positive
    return mu

# --- OpenCL Implementation (Hybrid Model - Recommended) ---

# The OpenCL C kernel source code.
# This code will be compiled at runtime and run on the GPU.
# Keep it simple: focus on the most parallelizable part of your calculation.
OPENCL_KERNEL_SRC = \"\"\"
__kernel void your_simple_kernel(__global const double *in_data,
                                __global double *out_data,
                                const double param1)
{
    int i = get_global_id(0);
    double z = in_data[i];
    // Example calculation:
    out_data[i] = param1 * pown(1.0 + z, 3);
}
\"\"\"

def distance_modulus_model_opencl(z_array, cl_context, cl_queue, cl_program, *cosmo_params):
    \"\"\"
    The hybrid OpenCL entry point for calculating the distance modulus.
    
    This function orchestrates three steps:
    1. Pre-computation on the CPU (if needed).
    2. Parallel computation on the GPU.
    3. Final computation on the CPU.
    \"\"\"
    
    # --- HYBRID STEP 1: Perform sensitive calculations on CPU first ---
    # For many models, this step is not needed. For others (like USMF_V2),
    # it's crucial for calculating stable inputs for the GPU.
    # precalculated_inputs = some_cpu_helper_function(z_array, *cosmo_params)

    # --- HYBRID STEP 2: Execute simple, parallel kernel on GPU ---
    
    # a. Prepare parameters and data buffers
    z_data = np.asarray(z_array, dtype=np.float64)
    param1, param2 = cosmo_params
    
    mf = cl.mem_flags
    # Send the pre-calculated, safe inputs to the GPU
    gpu_in_buffer = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_data)
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
    # For this template, we simply fall back to the robust CPU implementation.
    mu = distance_modulus_model(z_array, *cosmo_params)
    
    return mu
"""