# copernican_suite/usmf3.py
"""
Entangled-Geometrodynamic Model (EGM / USMF v3) Plugin for the Copernican Suite.
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
MODEL_NAME = "EGM_USMF_v3"
MODEL_DESCRIPTION = "Entangled-Geometrodynamic Model (USMF v3)."
PARAMETER_NAMES = ["H0", "Omega_m0", "Omega_phi0", "w_phi", "gamma_E"]
INITIAL_GUESSES = [70.0, 0.3, 0.7, -0.8, 0.05]
PARAMETER_BOUNDS = [(60.0, 80.0), (0.1, 0.5), (0.5, 0.9), (-1.5, -0.5), (0.0, 0.2)]

# --- Fixed Parameters ---
C_LIGHT_KM_S = 299792.458

def hubble_model(z, H0, Omega_m0, Omega_phi0, w_phi, gamma_E):
    """
    Calculates H(z) for the EGM model.
    The gamma_E parameter is included for future extensions,
    but in this simplified model, it implicitly defines Omega_phi0 and w_phi.
    """
    return H0 * np.sqrt(Omega_m0 * (1 + z)**3 + Omega_phi0 * (1 + z)**(3 * (1 + w_phi)))

def luminosity_distance_integrand(z, H0, Omega_m0, Omega_phi0, w_phi, gamma_E):
    """
    Integrand for the luminosity distance calculation.
    """
    return C_LIGHT_KM_S / hubble_model(z, H0, Omega_m0, Omega_phi0, w_phi, gamma_E)

def distance_modulus_model(z_array, H0, Omega_m0, Omega_phi0, w_phi, gamma_E):
    """
    Calculates the distance modulus for a given array of redshifts.
    """
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        integral, _ = quad(luminosity_distance_integrand, 0, z, args=(H0, Omega_m0, Omega_phi0, w_phi, gamma_E))
        d_L = (1 + z) * integral  # in Mpc
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
                           const double Omega_m0,
                           const double Omega_phi0,
                           const double w_phi)
{
    int i = get_global_id(0);
    double z = z_array[i];
    h_z[i] = H0 * sqrt(Omega_m0 * pown(1.0 + z, 3) + Omega_phi0 * pown(1.0 + z, 3 * (1 + w_phi)));
}
"""

def distance_modulus_model_opencl(z_array, cl_context, cl_queue, cl_program, *cosmo_params):
    """
    Hybrid OpenCL implementation for the distance modulus.
    """
    H0, Omega_m0, Omega_phi0, w_phi, gamma_E = cosmo_params
    z_data = np.asarray(z_array, dtype=np.float64)

    mf = cl.mem_flags
    gpu_z_buffer = cl.Buffer(cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_data)
    gpu_h_z_buffer = cl.Buffer(cl_context, mf.WRITE_ONLY, size=z_data.nbytes)

    kernel = cl_program.hubble_kernel
    kernel.set_args(gpu_z_buffer, gpu_h_z_buffer, np.double(H0), np.double(Omega_m0), np.double(Omega_phi0), np.double(w_phi))
    cl.enqueue_nd_range_kernel(cl_queue, kernel, z_data.shape, None).wait()

    h_z_gpu = np.empty_like(z_data)
    cl.enqueue_copy(cl_queue, h_z_gpu, gpu_h_z_buffer).wait()

    # CPU-based integration
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        # A simplified integration for the purpose of the template.
        # A more robust implementation would perform the integration on the GPU
        # or use a more sophisticated hybrid approach.
        def integrand(x):
            return C_LIGHT_KM_S / hubble_model(x, H0, Omega_m0, Omega_phi0, w_phi, gamma_E)

        integral, _ = quad(integrand, 0, z)
        d_L = (1 + z) * integral
        if d_L > 0:
            mu[i] = 5 * np.log10(d_L) + 25
        else:
            mu[i] = np.nan
    return mu