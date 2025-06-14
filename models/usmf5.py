# usmf5.py
"""
Fixed-Size Filament Contraction Model (USMF) Version 5 Plugin for the
Copernican Suite.
"""
# DEV NOTE (v1.4.1): Refactored to comply with the plugin specification.
# Added required metadata variables, removed incorrect imports and
# renamed functions to the standard names expected by the engine.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import multiprocessing as mp


# Module‐level name for engine registration
MODEL_NAME = "Fixed-Size Filament Contraction Model (USMF) Version 5"
MODEL_DESCRIPTION = (
    "USMF version 5 assumes a fixed cosmic scale while local filaments "
    "contract over time. An early-universe exponent controls the BAO "
    "behavior and oscillatory terms encode quantum ripples."
)
MODEL_EQUATIONS_LATEX_SN = [
    r"$\alpha(t)$ piecewise as in Eq.(1)",
    r"$r(z)=\int_{t_e}^{t_0} \frac{c}{\alpha(t)} dt$",
    r"$d_L(z)=|r(z)|(1+z)^2\frac{70}{H_A}$",
    r"$\mu = 5\log_{10}(d_L)+25$",
]
MODEL_EQUATIONS_LATEX_BAO = [
    r"$D_A(z)=|r(z)|\frac{70}{H_A}$",
    r"$H(z)=-\frac{1}{\alpha}\frac{d\alpha}{dt}$",
    r"$D_V(z)=\left[(1+z)^2 D_A(z)^2 \frac{cz}{H(z)}\right]^{1/3}$",
    r"$r_s=\int_{z_d}^{\infty} \frac{c_s(z)}{H_{\rm early}(z)} dz$",
]

# Number of physical cores for multiprocessing
PHYSICAL_CORES = mp.cpu_count(logical=False) or 2

# ----------------------------------------------------------------------
# Parameter definitions (must match cosmo_model_usmf5.md)
# ----------------------------------------------------------------------
PARAMETER_NAMES = [
    "H_A", "p_alpha", "k_exp", "s_exp",
    "A_osc", "omega_osc", "t_i_osc", "phi_osc",
    "m_e", "t_eq", "delta", "n"
]
INITIAL_GUESSES = [
    70.0, 1.0, 0.1, 1.0,
    0.01, 5.0, 0.1, 0.0,
    0.5, 0.01, 0.1, 2.0
]
PARAMETER_BOUNDS = [
    (50.0, 90.0), (0.0, 5.0), (0.0, 1.0), (0.0, 5.0),
    (0.0, 0.1), (0.0, 20.0), (0.0, 1.0), (0.0, 2*np.pi),
    (0.0, 2.0), (1e-4, 0.1), (0.01, 1.0), (0.0, 4.0)
]
PARAMETER_LATEX_NAMES = [
    r"$H_A$", r"$p_\alpha$", r"$k_{\exp}$", r"$s_{\exp}$",
    r"$A_{\rm osc}$", r"$\omega_{\rm osc}$", r"$t_{i,\rm osc}$", r"$\phi_{\rm osc}$",
    r"$m_e$", r"$t_{\rm eq}$", r"$\Delta$", r"$n$"
]
PARAMETER_UNITS = [
    "km/s/Mpc", "", "", "", "", "", "Gyr", "rad", "", "Gyr", "Gyr", ""
]

FIXED_PARAMS = {
    "C_LIGHT_KM_S": 299792.458,
    "SECONDS_PER_GYR": 1e9 * 365.25 * 24 * 3600,
    "MPC_PER_KM": 1.0 / (3.08567758e19),
    "MPC_TO_KM": 3.08567758e19,
    "TINY_FLOAT": 1e-30,
    "AGE_GYR": 13.8,
    "OMEGA_GAMMA_H2_PREFACTOR_FOR_RS": 2.47282e-5,
    "OMEGA_B0_H2_EFF_FOR_RS": 0.0224,
    "ZD": 1059.0,
}

# ----------------------------------------------------------------------
# Core calculator: piecewise α(t), inversion t(z), distance integrals
# ----------------------------------------------------------------------
class USMF5Calculator:
    def __init__(self, cosmo_params):
        (self.H_A,
         self.p_alpha, self.k_exp, self.s_exp,
         self.A_osc, self.omega_osc, t_i_osc, self.phi_osc,
         self.m_e, t_eq, delta, self.n) = cosmo_params

        # Global age today in seconds
        self.t0_Gyr  = FIXED_PARAMS["AGE_GYR"]
        self.t0_sec  = self.t0_Gyr * FIXED_PARAMS["SECONDS_PER_GYR"]

        # Oscillation, equality and transition times/width in seconds
        self.t_i_sec  = t_i_osc * FIXED_PARAMS["SECONDS_PER_GYR"]
        self.t_eq_sec = t_eq    * FIXED_PARAMS["SECONDS_PER_GYR"]
        self.delta_sec= delta   * FIXED_PARAMS["SECONDS_PER_GYR"]

        # Cache for α(t₀) and t(z)
        self.alpha_t0 = self.alpha(self.t0_sec)
        self._t_cache = {}

    def alpha(self, t_sec):
        """Piecewise universal scale factor α(t)."""
        if t_sec <= 0:
            return np.nan
        ratio = t_sec / self.t0_sec
        # Early regime
        alpha_early = ratio ** (-self.m_e)
        # Late regime core
        term_power = ratio ** (-self.p_alpha)
        term_exp   = np.exp(self.k_exp * (ratio**self.s_exp - 1.0))
        osc = 0.0
        if t_sec >= self.t_i_sec:
            osc = np.sin(self.omega_osc * np.log(t_sec / self.t_i_sec) + self.phi_osc)
        alpha_late = term_power * term_exp * (1.0 + self.A_osc * osc)
        # Smooth transition
        x = (t_sec - self.t_eq_sec) / self.delta_sec
        S = 0.5 * (1.0 + np.tanh(x))
        return S * alpha_late + (1.0 - S) * alpha_early

    def get_t_from_z(self, z):
        """Invert 1+z = α(t_e)/α(t₀) → find t_e by root-finding."""
        if z in self._t_cache:
            return self._t_cache[z]
        if abs(z) < 1e-8:
            self._t_cache[z] = self.t0_sec
            return self.t0_sec
        target = (1.0 + z) * self.alpha_t0
        def f(t):
            return self.alpha(t) - target
        a = FIXED_PARAMS["TINY_FLOAT"] * self.t0_sec
        b = self.t0_sec
        try:
            t_root = brentq(f, a, b, maxiter=100, xtol=1e-6)
        except Exception:
            return np.nan
        self._t_cache[z] = t_root
        return t_root

# Helper for multiprocessing
def _worker(z_batch, cosmo_params, func_name):
    calc = USMF5Calculator(cosmo_params)
    fn = globals()[func_name]
    return [fn(z, calc) for z in z_batch]

def _parallelize(z_array, func_name, *params):
    z = np.asarray(z_array)
    if z.size == 1:
        return globals()[func_name](z.item(), USMF5Calculator(params))
    unique, inv = np.unique(z, return_inverse=True)
    batches = np.array_split(unique, PHYSICAL_CORES)
    args = [(batch, params, func_name) for batch in batches]
    try:
        with mp.Pool(PHYSICAL_CORES) as pool:
            results = pool.starmap(_worker, args)
        flat = np.concatenate(results)
    except:
        flat = []
        for batch in batches:
            flat.extend(_worker(batch, params, func_name))
        flat = np.array(flat)
    return flat[inv].reshape(z.shape)

# -----------------------------------------------------------------------------
# Single‐z core routines
# -----------------------------------------------------------------------------
def _r_Mpc_single(z, calc: USMF5Calculator):
    t_e = calc.get_t_from_z(z)
    if not np.isfinite(t_e):
        return np.nan
    def integrand(t):
        α = calc.alpha(t)
        return FIXED_PARAMS["C_LIGHT_KM_S"] / α if α > 0 else np.inf
    r_km, _ = quad(integrand, t_e, calc.t0_sec, epsrel=1e-6)
    return r_km * FIXED_PARAMS["MPC_PER_KM"]

def get_comoving_distance_Mpc(z_array, *params):
    return _parallelize(z_array, "_r_Mpc_single", *params)

def _DL_Mpc_single(z, calc: USMF5Calculator):
    r = _r_Mpc_single(z, calc)
    if not np.isfinite(r):
        return np.nan
    # scaling factor C_H = 70/H_A
    C_H = 70.0 / calc.H_A
    return abs(r) * (1+z)**2 * C_H

def get_luminosity_distance_Mpc(z_array, *params):
    return _parallelize(z_array, "_DL_Mpc_single", *params)

def _mu_single(z, calc: USMF5Calculator):
    DL = _DL_Mpc_single(z, calc)
    if not np.isfinite(DL) or DL <= 0:
        return np.nan
    return 5 * np.log10(DL) + 25

def distance_modulus_model(z_array, *params):
    """Return the distance modulus \(\mu\) for a redshift array."""
    return _parallelize(z_array, "_mu_single", *params)

def _Hz_single(z, calc: USMF5Calculator):
    t_e = calc.get_t_from_z(z)
    if not np.isfinite(t_e):
        return np.nan
    dt = 1e-3 * calc.t0_sec
    α1 = calc.alpha(t_e + dt)
    α2 = calc.alpha(t_e - dt)
    dadt = (α1 - α2) / (2 * dt)
    H_sec = -dadt / calc.alpha(t_e)
    # convert to km/s/Mpc
    return H_sec * FIXED_PARAMS["SECONDS_PER_GYR"] * (70.0 / calc.H_A)

def get_Hz_per_Mpc(z_array, *params):
    """Hubble parameter $H(z)$ in km/s/Mpc."""
    return _parallelize(z_array, "_Hz_single", *params)

def _DV_single(z, calc: USMF5Calculator):
    DA = _r_Mpc_single(z, calc) / (1+z)
    H_z = _Hz_single(z, calc)
    return ((1+z)**2 * DA**2 * (FIXED_PARAMS["C_LIGHT_KM_S"] * z / H_z))**(1/3)

def get_DV_Mpc(z_array, *params):
    """BAO volume distance $D_V(z)$ in Mpc."""
    return _parallelize(z_array, "_DV_single", *params)

def get_sound_horizon_rs_Mpc(*params):
    # Early-time H(z) ∝ (1+z)^{1/m_e}
    om_b = FIXED_PARAMS["OMEGA_B0_H2_EFF_FOR_RS"]
    z_d  = FIXED_PARAMS["ZD"]
    m_e  = params[8]
    def Hz_early(z):
        t0 = FIXED_PARAMS["AGE_GYR"]
        return (m_e / (t0 * FIXED_PARAMS["SECONDS_PER_GYR"])) * (1+z)**(1.0/m_e) * FIXED_PARAMS["MPC_TO_KM"]
    def integrand(z):
        R = (3 * om_b) / (4 * FIXED_PARAMS["OMEGA_GAMMA_H2_PREFACTOR_FOR_RS"] * (1+z))
        c_s = FIXED_PARAMS["C_LIGHT_KM_S"] / np.sqrt(3*(1+R))
        return c_s / Hz_early(z)
    rs, _ = quad(integrand, z_d, np.inf)
    return rs

# ----------------------------------------------------------------------
# Interface helpers required by the engine
# ----------------------------------------------------------------------
def get_angular_diameter_distance_Mpc(z_array, *params):
    z = np.asarray(z_array)
    # D_A = D_L / (1+z)^2, but D_L here is r*(1+z)^2*C_H -> so use our DL func
    DL = get_luminosity_distance_Mpc(z_array, *params)
    return DL / np.power(1+z, 2)

