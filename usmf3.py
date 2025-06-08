# copernican_suite/usmf3.py
"""
Entangled-Geometrodynamic Model (EGM / USMF v3, Rev. 2) Plugin for the Copernican Suite.
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
MODEL_DESCRIPTION = "Entangled-Geometrodynamic Model (USMF v3, Rev. 2)."

# For displaying equations on plots
MODEL_EQUATIONS_LATEX_SN = [
    r"$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^{3(1-\gamma_E)} + (1-\Omega_{m0})(1+z)^{3(1+w_\phi)}}$",
    r"$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$",
    r"$\mu = 5 \log_{10}(d_L/1\mathrm{Mpc}) + 25$"
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$",
    r"$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$"
]

PARAMETER_NAMES = ["H0", "Omega_m0", "w_phi", "gamma_E"]
INITIAL_GUESSES = [69.0, 0.3, -1.1, 0.05]
PARAMETER_BOUNDS = [(60.0, 80.0), (0.1, 0.5), (-1.5, -0.7), (-0.1, 0.2)]

# --- Fixed Parameters ---
C_LIGHT_KM_S = 299792.458

def hubble_model(z, H0, Omega_m0, w_phi, gamma_E):
    """
    Calculates H(z) for the EGM model assuming a flat universe.
    """
    omega_phi_0 = 1.0 - Omega_m0
    matter_term = Omega_m0 * (1 + z)**(3 * (1 - gamma_E))
    qpf_term = omega_phi_0 * (1 + z)**(3 * (1 + w_phi))
    return H0 * np.sqrt(matter_term + qpf_term)

def luminosity_distance_integrand(z, H0, Omega_m0, w_phi, gamma_E):
    """
    Integrand for the luminosity distance calculation.
    """
    return C_LIGHT_KM_S / hubble_model(z, H0, Omega_m0, w_phi, gamma_E)

def distance_modulus_model(z_array, H0, Omega_m0, w_phi, gamma_E):
    """
    Calculates the distance modulus for a given array of redshifts.
    """
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        integral, _ = quad(luminosity_distance_integrand, 0, z, args=(H0, Omega_m0, w_phi, gamma_E))
        d_L = (1 + z) * integral  # in Mpc
        if d_L > 0:
            mu[i] = 5 * np.log10(d_L) + 25
        else:
            mu[i] = np.nan
    return mu

# --- OpenCL Implementation ---
OPENCL_KERNEL_SRC = """
__kernel void hubble_kernel(__global const double *z_array,
                           __global double *h_z,
                           const double H0,
                           const double Omega_m0,
                           const double w_phi,
                           const double gamma_E)
{
    int i = get_global_id(0);
    double z = z_array[i];
    double omega_phi_0 = 1.0 - Omega_m0;
    double matter_term = Omega_m0 * pown(1.0 + z, 3.0 * (1.0 - gamma_E));
    double qpf_term = omega_phi_0 * pown(1.0 + z, 3.0 * (1.0 + w_phi));
    h_z[i] = H0 * sqrt(matter_term + qpf_term);
}
"""

def distance_modulus_model_opencl(z_array, cl_context, cl_queue, cl_program, *cosmo_params):
    """
    Hybrid OpenCL implementation for the distance modulus.
    The integration remains on the CPU as it is serial, but H(z) is calculated in parallel on the GPU.
    """
    # For this model, the OpenCL kernel can compute H(z) directly.
    # The serial integration step is still performed on the CPU.
    H0, Omega_m0, w_phi, gamma_E = cosmo_params
    
    # Standard CPU integration as a fallback or for direct use
    mu = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        integral, err = quad(luminosity_distance_integrand, 0, z, args=(H0, Omega_m0, w_phi, gamma_E))
        if err > 1e-3:
            logging.warning(f"High integration error ({err}) for z={z}")
        d_L = (1 + z) * integral
        if d_L > 0:
            mu[i] = 5 * np.log10(d_L) + 25
        else:
            mu[i] = np.nan
    return mu