# usmf3b.py
"""
USMFv3b "Kinematic" Model Plugin for the Copernican Suite

DEV NOTE (v1.4rc7):
This model has been completely rewritten to match the official specification
in the usmf3b.md documentation. It now implements the "Kinematic" USMF model.

1.  METADATA OVERHAUL: The parameters have been changed to H_A, t0_age_Gyr,
    p_kin, and fiducial parameters for BAO analysis, as specified in the
    markdown file.
2.  EQUATIONS REIMPLEMENTED: All physics functions have been replaced to
    implement the analytic equations for the Kinematic model (r(z), dL(z),
    DA(z), H(z), DV(z)).
3.  BAO SUPPORT: The model is now fully equipped for BAO analysis, including
    an implementation of the Eisenstein & Hu fitting formula for the sound
    horizon `rs` based on the model's fiducial parameters.
4.  API COMPLIANCE: All function signatures are compatible with the v1.4rc
    engine, accepting a nuisance `M` parameter and `**kwargs` where needed.

DEV NOTE (v1.4g): Newline appended at file end for style compliance.
"""

import numpy as np
import logging

# --- Model Metadata ---
METADATA = {
    "model_name": "USMF3b-Kinematic",
    "model_description": "A kinematic USMF model with a power-law shrinking index, providing analytic solutions for cosmological distances.",
    "equations": {
        "sne": [
            r"$r(z) = \frac{c t_0}{p+1} [ 1 - (1+z)^{-(p+1)/p} ]$",
            r"$d_L(z) = |r(z)|(1+z)^2 \frac{70}{H_A}$",
            r"$\mu = 5 \log_{10}(d_L) + M$"
        ],
        "bao": [
            r"$D_A(z) = |r(z)| \frac{70}{H_A}$",
            r"$H(z) = \frac{p_{kin}}{t_0} (1+z)^{1/p_{kin}}$",
            r"$D_V(z) = [ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} ]^{1/3}$"
        ]
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
        "p_kin": {
            "description": "Kinematic power-law index",
            "units": None,
            "latex": r"$p_{kin}$",
            "role": "cosmological",
            "initial_guess": 0.8,
            "bounds": (0.1, 2.0)
        },
        "Omega_m0_fid": {
            "description": "Fiducial Matter Density for rs calculation",
            "units": None,
            "latex": r"$\Omega_{m0,fid}$",
            "role": "cosmological", # Treated as cosmological for engine compatibility
            "initial_guess": 0.31,
            "bounds": (0.2, 0.4)
        },
        "Omega_b0_fid": {
            "description": "Fiducial Baryon Density for rs calculation",
            "units": None,
            "latex": r"$\Omega_{b0,fid}$",
            "role": "cosmological", # Treated as cosmological for engine compatibility
            "initial_guess": 0.0486,
            "bounds": (0.03, 0.07)
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
        "C_LIGHT_KM_S": 299792.458,
        "GYR_TO_S": 3.15576e16,
        "MPC_TO_KM": 3.08567758e19,
        "H0_fid": 70.0 # Fiducial Hubble constant for h in rs calculation
    }
}

# --- Core Physics Functions ---

def _get_rz_Mpc(z, t0_age_Gyr, p_kin):
    """
    Calculates the analytic comoving-like distance r(z) in Mpc.
    r(z) = [c*t0/(p+1)] * [1 - (1+z)^(-(p+1)/p)]
    """
    z = np.asarray(z)
    C_LIGHT_KM_S = METADATA['fixed_params']['C_LIGHT_KM_S']
    GYR_S = METADATA['fixed_params']['GYR_TO_S']
    MPC_KM = METADATA['fixed_params']['MPC_TO_KM']

    t0_s = t0_age_Gyr * GYR_S
    
    # Avoid division by zero if p_kin is close to 0
    if np.abs(p_kin) < 1e-9:
        return np.full_like(z, np.nan)
        
    exponent = -(p_kin + 1) / p_kin
    
    # Calculate r(z) in km
    rz_km = (C_LIGHT_KM_S * t0_s / (p_kin + 1)) * (1 - np.power(1 + z, exponent))
    
    # Convert from km to Mpc
    return rz_km / MPC_KM

# --- Derived Cosmological Observables ---

def get_luminosity_distance_Mpc(z, H_A, t0_age_Gyr, p_kin, **kwargs):
    """
    Calculates luminosity distance dL in Mpc.
    dL(z) = |r(z)| * (1+z)^2 * (70.0 / H_A)
    """
    z = np.asarray(z)
    rz = _get_rz_Mpc(z, t0_age_Gyr, p_kin)
    return np.abs(rz) * (1 + z)**2 * (70.0 / H_A)

def get_distance_modulus(z, M, H_A, t0_age_Gyr, p_kin, **kwargs):
    """
    Calculates the distance modulus mu.
    mu = 5 * log10(dL / 1 Mpc) + M
    """
    dl_mpc = get_luminosity_distance_Mpc(z, H_A=H_A, t0_age_Gyr=t0_age_Gyr, p_kin=p_kin)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 5 * np.log10(dl_mpc)
    return mu + M

def get_angular_diameter_distance_Mpc(z, H_A, t0_age_Gyr, p_kin, **kwargs):
    """
    Calculates angular diameter distance DA in Mpc.
    DA(z) = |r(z)| * (70.0 / H_A)
    """
    z = np.asarray(z)
    rz = _get_rz_Mpc(z, t0_age_Gyr, p_kin)
    return np.abs(rz) * (70.0 / H_A)

def get_Hz_per_Mpc(z, t0_age_Gyr, p_kin, **kwargs):
    """
    Calculates the Hubble parameter H(z) in units of km/s/Mpc.
    H_usmf(z) = (p_kin/t0) * (1+z)^(1/p_kin)
    """
    z = np.asarray(z)
    GYR_S = METADATA['fixed_params']['GYR_TO_S']
    MPC_KM = METADATA['fixed_params']['MPC_TO_KM']
    
    t0_s = t0_age_Gyr * GYR_S
    
    # Calculate H in units of 1/s
    Hz_per_s = (p_kin / t0_s) * np.power(1 + z, 1.0 / p_kin)
    
    # Convert from 1/s to km/s/Mpc
    return Hz_per_s * (MPC_KM / 1e19) # Correction for Mpc -> km factor

def get_DV_Mpc(z, H_A, t0_age_Gyr, p_kin, **kwargs):
    """
    Calculates the volume-averaged distance DV in Mpc.
    DV = [ (1+z)^2 * DA^2 * c*z / H(z) ] ^ (1/3)
    """
    z = np.asarray(z)
    C_LIGHT_KM_S = METADATA['fixed_params']['C_LIGHT_KM_S']
    
    da = get_angular_diameter_distance_Mpc(z, H_A=H_A, t0_age_Gyr=t0_age_Gyr, p_kin=p_kin)
    hz = get_Hz_per_Mpc(z, t0_age_Gyr=t0_age_Gyr, p_kin=p_kin)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_hz = np.where(hz > 1e-9, hz, np.nan)
        term = (1 + z)**2 * da**2 * C_LIGHT_KM_S * z / safe_hz
        
    return np.power(term, 1/3, where=term >= 0, out=np.full_like(z, np.nan))

def get_sound_horizon_rs_Mpc(Omega_m0_fid, Omega_b0_fid, **kwargs):
    """
    Calculates the sound horizon at the drag epoch using the Eisenstein & Hu (1998)
    fitting formula with this model's fiducial parameters.
    """
    # Use fiducial H0 to get fiducial h
    H0_fid = METADATA['fixed_params']['H0_fid']
    h_fid = H0_fid / 100.0
    
    omega_m_h2 = Omega_m0_fid * h_fid**2
    omega_b_h2 = Omega_b0_fid * h_fid**2
    
    # These constants are from the standard model, used here for the rs calculation
    T_CMB0 = 2.7255
    OMEGA_G_H2 = 2.472e-5
    NEUTRINO_MASS_eV = 0.06
    omega_nu_h2 = NEUTRINO_MASS_eV / 93.14
    om_r_h2 = OMEGA_G_H2 + omega_nu_h2
    
    if not (omega_m_h2 > 1e-5 and omega_b_h2 > 1e-5): return np.nan
    
    # Calculations from Eisenstein & Hu (1998), ApJ, 496, 605
    z_eq = 2.50e4 * omega_m_h2 * (T_CMB0/2.7)**-4
    k_eq = 7.46e-2 * omega_m_h2 * (T_CMB0/2.7)**-2
    
    b1 = 0.313 * (omega_m_h2)**-0.419 * (1 + 0.607 * (omega_m_h2)**0.674)
    b2 = 0.238 * (omega_m_h2)**0.223
    z_d = 1291 * ((omega_m_h2)**0.251 / (1 + 0.659 * (omega_m_h2)**0.828)) * (1 + b1 * (omega_b_h2)**b2)

    R_eq = 3.15e4 * omega_b_h2 * (T_CMB0/2.7)**-4 / z_eq
    R_d = 3.15e4 * omega_b_h2 * (T_CMB0/2.7)**-4 / z_d
    
    s = (2. / (3. * k_eq)) * np.sqrt(6. / R_eq) * np.log((np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq)))
    
    # E&H formula gives s in Mpc/h, so we convert to Mpc using fiducial h
    return s / h_fid
