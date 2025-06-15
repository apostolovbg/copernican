"""Interface between compiled models and computational engines."""
# DEV NOTE (v1.5e): Added validation helpers and unified execution wrapper
# for Phase 3 engine abstraction layer.

from typing import Callable, Dict, Any
import logging

REQUIRED_MODEL_KEYS = [
    'distance_modulus_model',
    'get_comoving_distance_Mpc',
    'get_luminosity_distance_Mpc',
    'get_angular_diameter_distance_Mpc',
    'get_Hz_per_Mpc',
    'get_DV_Mpc',
    'get_sound_horizon_rs_Mpc',
    'PARAMETER_NAMES',
    'INITIAL_GUESSES',
    'PARAMETER_BOUNDS',
    'FIXED_PARAMS',
    'MODEL_NAME'
]

REQUIRED_ENGINE_FUNCS = [
    'fit_sne_parameters',
    'calculate_bao_observables',
    'chi_squared_bao'
]


def validate_model(model: Dict[str, Any]) -> bool:
    """Ensure model dictionary exposes all required callables."""
    missing = [k for k in REQUIRED_MODEL_KEYS if k not in model]
    if missing:
        logging.getLogger().error(
            f"Model dictionary missing required keys: {missing}")
        return False
    for key in REQUIRED_MODEL_KEYS:
        if 'get_' in key or 'distance_modulus_model' in key:
            if not callable(model.get(key)):
                logging.getLogger().error(f"Model key '{key}' is not callable")
                return False
    return True


def validate_engine(engine_module) -> bool:
    """Verify engine module implements the minimal interface."""
    missing = [f for f in REQUIRED_ENGINE_FUNCS if not hasattr(engine_module, f)]
    if missing:
        logging.getLogger().error(
            f"Engine module missing required functions: {missing}")
        return False
    return True


def run_engine(engine_module, model_dict: Dict[str, Any], sne_data, bao_data,
               z_smooth=None) -> Dict[str, Any]:
    """Execute engine using validated model callables and data."""
    if not (validate_engine(engine_module) and validate_model(model_dict)):
        return {}

    sne_results = engine_module.fit_sne_parameters(sne_data, model_dict)
    cosmo_params = []
    if sne_results and sne_results.get('fitted_cosmological_params'):
        cosmo_params = list(sne_results['fitted_cosmological_params'].values())

    bao_pred, rs_Mpc, smooth = engine_module.calculate_bao_observables(
        bao_data, model_dict, cosmo_params, z_smooth=z_smooth)

    chi2_bao = engine_module.chi_squared_bao(
        bao_data, model_dict, cosmo_params, rs_Mpc) if bao_pred is not None else float('inf')

    return {
        'sne_fit_results': sne_results,
        'bao_predictions': bao_pred,
        'rs_Mpc': rs_Mpc,
        'chi2_bao': chi2_bao,
        'smooth_predictions': smooth
    }
