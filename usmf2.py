# usmf2.py
"""
USMFv2 Model Plugin for the Copernican Suite.

DEV NOTE (v1.4rc6):
1.  API ALIGNMENT FIX: The primary distance modulus function has been updated
    to align with the v1.4rc engine API.
    - Renamed `get_distance_modulus_mu` to `get_distance_modulus`.
    - Signature changed to accept the nuisance parameter `M` and `**kwargs`
      to safely ignore extra parameters (like `Ok0`) passed by the engine.
    - Calculation now correctly uses the passed value of `M`.
    This resolves the `TypeError` crash when using this model.

(Previous notes from v1.4rc preserved below)
...
"""

import numpy as np
import logging

# --- Model Metadata ---
# This dictionary is the standardized format for model information in v1.4+.
METADATA = {
    "model_name": "USMFv2",
    "model_description": "Unified Shrinking Matter Framework v2. A cosmological model positing apparent expansion is an effect of matter shrinking over time, based on a linear shrinking law.",
    "equations": {
        "sne": [
            r"$\alpha(t) = 1 + \frac{H_A}{c}(t_0 - t)$",
            r"$1+z = \alpha(t_e)$",
            r"$d_L(z) = c \cdot (t_0-t_e) \cdot (1+z) \cdot \frac{70}{H_A}$",
            r"$\mu = 5 \log_{10}(d_L) + M$"
        ],
        "bao": [] # This version of the model is not designed for BAO analysis.
    },
    "parameters": {
        "H_A": {
            "description": "Apparent Hubble Constant, related to the shrinking rate",
            "units": "km/s/Mpc",
            "latex": r"$H_A$",
            "role": "cosmological",
            "initial_guess": 70.0,
            "bounds": (50.0, 100.0)
        },
        "t0_age_Gyr": {
            "description": "Age of the Universe at present time",
            "units": "Gyr",
            "latex": r"$t_{0,age}$",
            "role": "cosmological",
            "initial_guess": 14.0,
            "bounds": (10.0, 20.0)
        },
        "M": {
            "description": "Absolute Magnitude of a Type Ia Supernova",
            "units": "magnitude",
            "latex": r"$M$",
            "role": "nuisance",
            "initial_guess": -19.3,
            "bounds": (-20, -18)
        }
    },
    "fixed_params": {
        "GYR_TO_S": 3.15576e16,
        "C_LIGHT_KM_S": 299792.458,
        "MPC_TO_KM": 3.086e19
    }
}

# --- Physics Functions ---

def get_distance_modulus(z_array, H_A, t0_age_Gyr, M, **kwargs):
    """
    Calculates the distance modulus for the USMFv2 model.
    The formula for dL is derived from the linear shrinking law alpha(t).
    The **kwargs will safely capture any other params like Ok0.
    """
    z = np.asarray(z_array)
    C_LIGHT_KM_S = METADATA['fixed_params']['C_LIGHT_KM_S']
    GYR_S = METADATA['fixed_params']['GYR_TO_S']
    MPC_KM = METADATA['fixed_params']['MPC_TO_KM']
    
    # Convert t0 from Gyr to seconds for calculation
    t0_sec = t0_age_Gyr * GYR_S

    # Calculate time of emission (te) from redshift
    # From 1+z = alpha(te) = 1 + H_A/c * (t0 - te)
    # => z = H_A/c * (t0 - te)
    # => te = t0 - z * c / H_A
    try:
        with np.errstate(divide='ignore'):
            # This is actually t0-te, the lookback time in seconds
            lookback_time_sec = z * C_LIGHT_KM_S / H_A

        # Calculate luminosity distance (dL) in km
        # dL = c * (t0-te) * (1+z) * (70/H_A) -- The (70/H_A) factor is a normalization convention
        dL_km = C_LIGHT_KM_S * lookback_time_sec * (1 + z) * (70.0 / H_A)

        # The luminosity distance is in km, we need it in Mpc for mu calculation
        dL_Mpc = dL_km / MPC_KM

        # Calculate distance modulus using the fitted nuisance parameter M
        with np.errstate(divide='ignore', invalid='ignore'):
            # mu = 5 * log10(dL / 1 Mpc)
            mu = 5 * np.log10(dL_Mpc)
        
        return mu + M

    except (ValueError, FloatingPointError) as e:
        logging.error(f"Error in USMFv2 calculation for H_A={H_A}, t0={t0_age_Gyr}: {e}")
        return np.full_like(z, np.nan)

# NOTE: This model is not equipped for BAO analysis, so BAO-specific functions
# like get_DV_Mpc, get_sound_horizon_rs_Mpc, etc., are intentionally omitted.