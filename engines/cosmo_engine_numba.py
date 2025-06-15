"""Numba-accelerated engine for the Copernican Suite."""
# DEV NOTE (v1.5e): New optional backend using Numba. Wraps SciPy engine and
# accelerates BAO chi-squared calculation.

from numba import njit
import numpy as np
from . import cosmo_engine_1_4b as base_engine


@njit
def _chi2_loop(obs_vals, obs_err, pred_vals):
    chi2 = 0.0
    for i in range(len(obs_vals)):
        diff = obs_vals[i] - pred_vals[i]
        chi2 += (diff / obs_err[i]) ** 2
    return chi2


def fit_sne_parameters(sne_data_df, model_plugin):
    """Delegates SNe fitting to the base SciPy engine."""
    return base_engine.fit_sne_parameters(sne_data_df, model_plugin)


def calculate_bao_observables(bao_data_df, model_plugin, cosmo_params, z_smooth=None):
    """Delegates BAO prediction to the base SciPy engine."""
    return base_engine.calculate_bao_observables(bao_data_df, model_plugin, cosmo_params, z_smooth=z_smooth)


def chi_squared_bao(bao_data_df, model_plugin, cosmo_params, model_rs_Mpc):
    """Compute BAO chi-squared using a Numba-jitted loop."""
    pred_df, _, _ = calculate_bao_observables(bao_data_df, model_plugin, cosmo_params)
    if pred_df is None or pred_df.empty:
        return np.inf
    obs_vals = bao_data_df['value'].to_numpy(dtype=float)
    obs_err = bao_data_df['error'].to_numpy(dtype=float)
    pred_vals = pred_df['model_prediction'].to_numpy(dtype=float)
    if obs_vals.size != pred_vals.size:
        return np.inf
    return float(_chi2_loop(obs_vals, obs_err, pred_vals))

