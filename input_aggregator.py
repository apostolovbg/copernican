# input_aggregator.py
"""
This module is responsible for collecting all user inputs and data from various
sources and aggregating them into a single, standardized 'Job JSON' structure.
This structure is then passed to the computational engine.

DEV NOTE (v1.4rc):
1.  CRITICAL FIX: Corrected the dynamic module loading in `_load_model_plugin`.
    Previously, all model plugins were loaded with the same internal name
    ("model_plugin"), causing caching conflicts in Python that led to a
    crash when trying to access model `METADATA`. The loader now assigns a
    unique name to each plugin based on its filename, resolving the bug.
2.  LOGGING: Replaced internal `logger` calls with standard `print` statements
    to conform to the new verbatim console-mirroring log system in copernican.py.
"""

import os
import importlib.util
import sys

# --- Dynamic Module and Data Loading ---

def _load_model_plugin(model_path):
    """Dynamically loads a model plugin from a given .py file path."""
    try:
        # Generate a unique module name from the file path.
        # This is CRITICAL to prevent Python's import caching from causing
        # conflicts when loading multiple different model files.
        module_name = os.path.splitext(os.path.basename(model_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, model_path)
        model_plugin = importlib.util.module_from_spec(spec)
        # Add the module to sys.modules under its unique name before execution
        # to handle circular dependencies if they ever arise.
        sys.modules[module_name] = model_plugin
        spec.loader.exec_module(model_plugin)
        return model_plugin
    except Exception as e:
        print(f"ERROR: Failed to load model plugin from '{model_path}': {e}")
        return None

def _load_model_metadata(model_path):
    """Loads the METADATA dictionary from a model plugin."""
    model_plugin = _load_model_plugin(model_path)
    if model_plugin and hasattr(model_plugin, 'METADATA'):
        return model_plugin.METADATA
    print(f"ERROR: Failed to load or find METADATA in model plugin at '{model_path}'.")
    return None

def _load_data(data_info, parsers_dict, base_dir):
    """Loads dataset using the specified parser function."""
    if not data_info or 'path' not in data_info or 'parser' not in data_info:
        return {} # Return empty dict if no data is to be loaded

    path = data_info['path']
    parser_key = data_info['parser']
    parser_func = parsers_dict.get(parser_key, {}).get('function')
    
    if not parser_func:
        print(f"ERROR: Could not find the specified data parser '{parser_key}'.")
        return None

    full_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
    
    print(f"Loading data from '{os.path.basename(full_path)}' using parser '{parser_key}'...")
    loaded_data = parser_func(full_path)
    
    if loaded_data is None:
        print(f"ERROR: The parser '{parser_key}' failed to load data from '{full_path}'.")
        return None
        
    return loaded_data

# --- Main Aggregation Function ---

def build_job_json(run_id, engine_name, alt_model_path, sne_data_info, bao_data_info):
    """
    Constructs the main job JSON by collecting metadata and data.

    Returns:
        dict: The fully assembled job data structure, or None on failure.
    """
    # Dynamically import the data_loaders module to access its parser dictionaries
    # This avoids circular import issues at startup.
    import data_loaders
    
    print("Building job data structure...")
    
    # The base path for loading models is the directory of copernican.py
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    job_data = {
        'run_id': run_id,
        'engine': engine_name,
        'model1_metadata': None,
        'model2_metadata': None,
        'data': {}
    }

    # Load metadata for the two models
    # The 'test' keyword in the UI resolves to using lcdm_model.py for both.
    if alt_model_path.lower() == 'test':
        alt_model_path = 'lcdm_model.py'
        print("Test mode: Loading LCDM metadata for both models.")

    job_data['model1_metadata'] = _load_model_metadata(os.path.join(base_dir, "lcdm_model.py"))
    job_data['model2_metadata'] = _load_model_metadata(os.path.join(base_dir, alt_model_path))

    if not job_data['model1_metadata'] or not job_data['model2_metadata']:
        print("CRITICAL: Failed to load essential model metadata. Cannot proceed.")
        return None

    # Load the actual datasets
    sne_loaded_data = _load_data(sne_data_info, data_loaders.SNE_PARSERS, base_dir)
    if sne_loaded_data is None: # A hard failure in the parser
        return None
    job_data['data'].update(sne_loaded_data)

    bao_loaded_data = _load_data(bao_data_info, data_loaders.BAO_PARSERS, base_dir)
    if bao_loaded_data is None: # A hard failure in the parser
        return None
    job_data['data'].update(bao_loaded_data)

    # Final check to ensure we have at least SNe data
    if 'sne_data' not in job_data['data']:
        print("CRITICAL: Supernova data is required but was not loaded successfully. Cannot proceed.")
        return None
        
    print("Job data structure built successfully.")
    return job_data