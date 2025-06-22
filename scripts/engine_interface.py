"""Interface to bridge generated model functions with existing engines."""

from types import SimpleNamespace
import inspect
import logging
import re

REQUIRED_FUNCTIONS = [
    "distance_modulus_model",
    "get_comoving_distance_Mpc",
    "get_luminosity_distance_Mpc",
    "get_angular_diameter_distance_Mpc",
    "get_Hz_per_Mpc",
    "get_DV_Mpc",
    "get_sound_horizon_rs_Mpc",
]

REQUIRED_ATTRIBUTES = [
    "MODEL_NAME",
    "MODEL_DESCRIPTION",
    "PARAMETER_NAMES",
    "PARAMETER_LATEX_NAMES",
    "PARAMETER_UNITS",
    "INITIAL_GUESSES",
    "PARAMETER_BOUNDS",
    "FIXED_PARAMS",
]


def build_plugin(model_data, func_dict):
    """Return an object mimicking the plugin interface."""
    plugin = SimpleNamespace()
    plugin.MODEL_NAME = model_data.get('model_name', 'GeneratedModel')
    plugin.MODEL_DESCRIPTION = model_data.get('description', '')
    plugin.PARAMETER_NAMES = [p['python_var'] for p in model_data['parameters']]
    plugin.PARAMETER_LATEX_NAMES = [p.get('latex_name', p['name']) for p in model_data['parameters']]
    plugin.PARAMETER_UNITS = [p.get('unit', '') for p in model_data['parameters']]
    plugin.INITIAL_GUESSES = [
        sum(p['bounds']) / 2.0 for p in model_data['parameters']
    ]
    plugin.PARAMETER_BOUNDS = [tuple(p['bounds']) for p in model_data['parameters']]
    plugin.FIXED_PARAMS = {}
    plugin.valid_for_distance_metrics = model_data.get('valid_for_distance_metrics', True)
    plugin.valid_for_bao = model_data.get('valid_for_bao', True)
    def sanitize_equation(eq_line: str) -> str:
        """Return a Matplotlib-friendly LaTeX string."""
        if not isinstance(eq_line, str):
            return ""
        eq = eq_line.strip()
        eq = re.sub(r'^\$+', '', eq)
        eq = re.sub(r'\$+$', '', eq)
        return f"${eq.strip()}$" if eq else ""

    eqs = model_data.get('equations', {})
    plugin.MODEL_EQUATIONS_LATEX_SN = [sanitize_equation(e) for e in eqs.get('sne', [])]
    plugin.MODEL_EQUATIONS_LATEX_BAO = [sanitize_equation(e) for e in eqs.get('bao', [])]
    if 'filename' in model_data:
        plugin.MODEL_FILENAME = model_data['filename']
    for name, func in func_dict.items():
        setattr(plugin, name, func)
    validate_plugin(plugin)
    return plugin


def validate_plugin(plugin):
    """Validate that ``plugin`` exposes the required attributes and functions."""
    logger = logging.getLogger()

    missing_attrs = [attr for attr in REQUIRED_ATTRIBUTES if not hasattr(plugin, attr)]
    if missing_attrs:
        logger.error(f"Plugin validation failed. Missing attributes: {missing_attrs}")
        return False

    required_funcs = REQUIRED_FUNCTIONS
    if getattr(plugin, 'valid_for_distance_metrics', True) is False:
        required_funcs = ['distance_modulus_model']
    elif getattr(plugin, 'valid_for_bao', True) is False:
        required_funcs = [
            'distance_modulus_model',
            'get_comoving_distance_Mpc',
            'get_luminosity_distance_Mpc',
            'get_angular_diameter_distance_Mpc',
            'get_Hz_per_Mpc',
            'get_DV_Mpc',
        ]

    for fname in required_funcs:
        func = getattr(plugin, fname, None)
        if not callable(func):
            logger.error(f"Plugin validation failed. Missing function '{fname}'.")
            return False
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            logger.error(f"Plugin validation failed. Unable to inspect '{fname}'.")
            return False
        if not sig.parameters:
            logger.error(f"Plugin validation failed. Function '{fname}' has no parameters.")
            return False

    return True
