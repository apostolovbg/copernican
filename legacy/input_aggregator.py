# input_aggregator.py
"""
This module is responsible for collecting all user inputs and data from various
sources and aggregating them into a single, standardized 'Job JSON' structure.
This structure is then passed to the computational engine.

DEV NOTE (v1.4rc6):
This module has been corrected to resolve the TypeError during parser execution.

1.  CRITICAL FIX (TypeError): The _load_data function was incorrectly passing
    the `base_dir` argument to all parsers, causing a crash for any parser
    not explicitly designed to receive it. The logic has been corrected to
    pass `base_dir` ONLY to the 'pantheon_plus_h2' parser.

2.  DIAGNOSTICS: Added verbose print statements to the _load_data function
    to make the parsing process more transparent, showing exactly which
    parser is being called and how.

DEV NOTE (v1.4g): Added a missing newline at the end of the file for
style consistency.
DEV NOTE (engine sync): The returned Job JSON now includes a new
`models` section with plugin paths so the engine can load model
plugins without errors.
DEV NOTE (file tracking): Data loaders now store the basename of the
source file in the Job JSON so output filenames reflect the input
datasets.
"""

import os
import importlib.util
import sys

# Deferred import to be called later
import data_loaders

# --- Dynamic Module and Data Loading ---

def _load_model_plugin(model_path):
    """Dynamically loads a model plugin from a given .py file path."""
    try:
        module_name = os.path.splitext(os.path.basename(model_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        model_plugin = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model_plugin
        spec.loader.exec_module(model_plugin)
        print(f"Successfully loaded model plugin: {model_path}")
        return model_plugin
    except Exception as e:
        print(f"CRITICAL: Failed to load model plugin from {model_path}: {e}")
        raise RuntimeError(f"Could not load model file {model_path}.") from e

def _load_model_metadata(model_path):
    """Loads only the METADATA dictionary from a model plugin."""
    print(f"Loading metadata from '{os.path.basename(model_path)}'...")
    try:
        plugin = _load_model_plugin(model_path)
        if hasattr(plugin, 'METADATA'):
            print("Metadata loaded successfully.")
            return plugin.METADATA
        else:
            print(f"Error: METADATA dictionary not found in {model_path}.")
            return None
    except Exception:
        return None

def _load_data(data_info, parsers_dict, base_dir):
    """
    Handles the loading of a single dataset (SNe or BAO) using the selected parser.
    Returns a dictionary containing the loaded data, or None on failure.
    """
    if not data_info or not data_info.get('filepath'):
        return {}

    filepath = data_info['filepath']
    parser_id = data_info['parser_id']
    parser_func = parsers_dict[parser_id]['function']
    data_type_key = data_info['type']

    print(f"Loading data from '{os.path.basename(filepath)}' using parser '{parser_id}'...")
    full_path = filepath if os.path.isabs(filepath) else os.path.join(base_dir, filepath)

    try:
        # --- DIAGNOSTIC & CRITICAL FIX (v1.4rc6) ---
        # Only pass `base_dir` to the specific parser that needs it.
        print(f"DIAGNOSTIC: Preparing to call parser '{parser_id}'.")
        if parser_id == 'pantheon_plus_h2':
            print(f"DIAGNOSTIC: Pantheon+ parser detected. Passing 'base_dir' for covariance matrix handling.")
            loaded_df = parser_func(full_path, base_dir=base_dir)
        else:
            print(f"DIAGNOSTIC: Standard parser detected. Calling without extra arguments.")
            loaded_df = parser_func(full_path)
        print(f"DIAGNOSTIC: Parser '{parser_id}' returned a result.")


        if loaded_df is None or loaded_df.empty:
            raise RuntimeError("Parser returned no data.")

        print(f"Successfully loaded {len(loaded_df)} data points from '{os.path.basename(filepath)}'.")

        # Include the source filename so the engine can construct
        # informative CSV and plot names later on.
        structured_data = {
            'dataframe': loaded_df,
            'parser_id': parser_id,
            'filepath': os.path.basename(filepath)
        }

        return {data_type_key: structured_data}

    except Exception as e:
        print(f"\nFATAL ERROR while loading data from '{os.path.basename(filepath)}'.")
        print(f"Parser '{parser_id}' failed with error: {e}")
        return None


# --- Main Aggregation Function ---

def build_job_json(engine_name, alt_model_path, sne_data_info, bao_data_info, base_dir, run_id):
    """
    The main function of this module. It orchestrates the collection of all
    data and metadata and assembles the final Job JSON.
    """
    print("\n--- Stage 2: Aggregating Job Data ---")
    print("Building job data structure...")

    job_data = {
        'run_id': run_id,
        'engine_name': engine_name,
        'model1_metadata': None,
        'model2_metadata': None,
        'data': {},
        'models': {}
    }

    if alt_model_path.lower() == 'test':
        alt_model_path = 'lcdm_model.py'
        print("Test mode: Loading LCDM metadata for both models.")

    job_data['model1_metadata'] = _load_model_metadata(os.path.join(base_dir, "lcdm_model.py"))
    job_data['model2_metadata'] = _load_model_metadata(os.path.join(base_dir, alt_model_path))
    job_data['models'] = {
        'model1': {'path': os.path.join(base_dir, "lcdm_model.py")},
        'model2': {'path': os.path.join(base_dir, alt_model_path)}
    }

    if not job_data['model1_metadata'] or not job_data['model2_metadata']:
        print("CRITICAL: Failed to load essential model metadata. Cannot proceed.")
        return None

    sne_loaded_data = _load_data(sne_data_info, data_loaders.SNE_PARSERS, base_dir)
    if sne_loaded_data is None:
        return None
    job_data['data'].update(sne_loaded_data)

    bao_loaded_data = _load_data(bao_data_info, data_loaders.BAO_PARSERS, base_dir)
    if bao_loaded_data is None:
        return None
    job_data['data'].update(bao_loaded_data)

    if 'sne_data' not in job_data['data'] and 'bao_data' not in job_data['data']:
        print("Error: No data was loaded. At least one dataset (SNe or BAO) is required.")
        return None

    print("Job data structure built successfully.")
    return job_data
