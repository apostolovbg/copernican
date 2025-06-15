"""Interface to bridge generated model functions with existing engines."""
# DEV NOTE (v1.5a): Loads callables from the coder and presents them like a plugin.

from types import SimpleNamespace


def build_plugin(model_data, func_dict):
    """Return an object mimicking the plugin interface."""
    plugin = SimpleNamespace()
    plugin.MODEL_NAME = model_data.get('model_name', 'GeneratedModel')
    plugin.MODEL_DESCRIPTION = model_data.get('description', '')
    plugin.PARAMETER_NAMES = [p['python_var'] for p in model_data['parameters']]
    plugin.PARAMETER_LATEX_NAMES = [p.get('latex_name', p['name']) for p in model_data['parameters']]
    plugin.PARAMETER_UNITS = [p.get('unit', '') for p in model_data['parameters']]
    plugin.INITIAL_GUESSES = [p['initial_guess'] for p in model_data['parameters']]
    plugin.PARAMETER_BOUNDS = [tuple(p['bounds']) for p in model_data['parameters']]
    plugin.FIXED_PARAMS = {}
    for name, func in func_dict.items():
        setattr(plugin, name, func)
    return plugin
