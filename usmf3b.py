# usmf3b.py
# "Unified Scalar Field with barotropic fluid" Model 3b Plugin

"""
DEV NOTE (v1.4rc): This model's code has not been changed, but its
functionality should now be restored due to the framework-level bugfixes
in the v1.4rc engine and plotter. The model's version is updated to
reflect its compatibility with the repaired Copernican Suite.
"""

import numpy as np

# --- Model Metadata ---
METADATA = {
    "model_name": "USMF3b",
    "model_description": "A 'toy model' of a flat universe with a unified scalar field behaving as both dark matter and dark energy, plus a baryonic matter component.",
    "parameters": {
        "H0": {
            "description": "Hubble Constant at z=0",
            "units": "km/s/Mpc",
            "latex": "$H_0$",
            "role": "cosmological",
            "initial_guess": 70,
            "bounds": (50, 90)
        },
        "Om0": {
            "description": "Baryonic Matter Density Parameter at z=0",
            "units": None,
            "latex": "$\\Omega_{b,0}$",
            "role": "cosmological",
            "initial_guess": 0.05,
            "bounds": (0.01, 0.1)
        },
        "M": {
            "description": "Absolute Magnitude of a Type Ia Supernova",
            "units": "magnitude",
            "latex": "$M$",
            "role": "nuisance",
            "initial_guess": -19.3,
            "bounds": (-20, -18)
        }
    }
}

# Define fixed physical constants used by the model
FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458,  # Speed of light in km/s
}


# --- Core Cosmological Functions ---

def get_Ez(z, Om0, Ok0):
    """
    Calculates the dimensionless Hubble parameter E(z) = H(z)/H0.
    For this model, the universe is assumed flat (Ok0=0), and the dark sector
    (dark matter + dark energy) is modeled as a single scalar field component.
    The total density parameter is Om0 (baryons) + (1-Om0) (scalar field).
    """
    # E(z)^2 = Om_baryon(1+z)^3 + (1-Om_baryon)(1+z)^(3/2)
    # The (1+z)^(3/2) term is a characteristic of this specific toy model's scalar field.
    return np.sqrt(Om0 * (1+z)**3 + (1 - Om0) * (1+z)**(3/2))


def get_Hz_per_Mpc(z, H0, Om0, Ok0):
    """
    Calculates the Hubble parameter H(z) in units of 1/Mpc.
    """
    return (H0 / FIXED_PARAMS["C_LIGHT_KM_S"]) * get_Ez(z, Om0, Ok0)


def get_comoving_distance_Mpc(z, H0, Om0, Ok0):
    """
    Calculates comoving distance. For this model, there's an analytic solution,
    so no integration is needed.
    """
    # Analytic solution for the integral of 1/E(z) for this model
    integral_val = (2 / (1 - Om0)) * (np.sqrt(Om0 * (1+z)**(3/2) + (1-Om0)) - np.sqrt(1))
    return (FIXED_PARAMS["C_LIGHT_KM_S"] / H0) * integral_val


# --- Derived Cosmological Distances & Observables ---

def get_angular_diameter_distance_Mpc(z, H0, Om0, Ok0):
    """
    Calculates the angular diameter distance in Mpc.
    """
    return get_comoving_distance_Mpc(z, H0, Om0, Ok0) / (1+z)


def get_luminosity_distance_Mpc(z, H0, Om0, Ok0):
    """
    Calculates the luminosity distance in Mpc.
    """
    return get_comoving_distance_Mpc(z, H0, Om0, Ok0) * (1+z)


def get_distance_modulus(z, H0, Om0, Ok0, M):
    """
    Calculates the distance modulus (mu).
    """
    dl_pc = get_luminosity_distance_Mpc(z, H0, Om0, Ok0) * 1e6
    return 5 * np.log10(dl_pc / 10) + M


def get_DV_Mpc(z, H0, Om0, Ok0):
    """
    Calculates the volume-averaged distance DV in Mpc. This is also analytic.
    DV = [ (c/H0) * z * (1+z)^2 * D_A(z)^2 / E(z) ] ^ (1/3)
    """
    # Since all components are numpy-friendly, this should vectorize safely.
    da = get_angular_diameter_distance_Mpc(z, H0, Om0, Ok0)
    ez = get_Ez(z, Om0, Ok0)
    
    # Handle z=0 case to prevent 0/0 if ez were also 0
    # Although for this model E(0)=1, this is good practice.
    term = np.zeros_like(z, dtype=float)
    non_zero_z = z > 0
    
    term[non_zero_z] = ( (FIXED_PARAMS["C_LIGHT_KM_S"] / H0) *
                         z[non_zero_z] * (1+z[non_zero_z])**2 * da[non_zero_z]**2 / ez[non_zero_z] )

    return np.power(term, 1/3)