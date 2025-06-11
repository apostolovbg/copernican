# copernican_suite/input_aggregator.py
"""
DEV NOTE (v1.4b): This module is the "Assembler" of the Copernican Suite.
The `build_job_json` function has been updated to read the LaTeX mathematical
equations (`MODEL_EQUATIONS_LATEX_SN` and `MODEL_EQUATIONS_LATEX_BAO`) from
the model plugin files. This information is now included in the Job JSON.

This change is critical for the v1.4b plotting restoration, as it makes the
model's mathematical form available to the downstream output modules, which
is required to generate the detailed information boxes on the plots.
"""

import json
import importlib.util
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import the data loading functions from the existing module
import data_loaders

class CustomJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle special data types from NumPy and pandas,
    ensuring they can be successfully serialized into a JSON string. This is
    essential for packaging complex scientific data for the engine.
    """
    def default(self, obj):
        # Convert numpy arrays to nested lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Convert numpy integer types to standard Python int
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert numpy float types to standard Python float
        if isinstance(obj, np.floating):
            return float(obj)
        # Convert numpy bool types to standard Python bool
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Let the base class default method raise the TypeError for other types
        return super(CustomJSONEncoder, self).default(obj)

def _load_model_module(model_filepath):
    """
    Dynamically loads a Python model plugin from its file path using the
    `importlib` library. This allows the program to remain agnostic to the
    specific model being tested.

    NOTE ON FUTURE REFACTORING: This function is also present in the engine.
    In a future version, it could be moved to a `utils.py` file to avoid
    code duplication (adhering to the DRY - Don't Repeat Yourself - principle).

    Args:
        model_filepath (str): The path to the model's .py file.

    Returns:
        A Python module object, or None if loading fails.
    """
    logger = logging.getLogger()
    try:
        spec = importlib.util.spec_from_file_location(
            name=os.path.basename(model_filepath).replace('.py', ''),
            location=model_filepath
        )
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        logger.info(f"Successfully loaded model plugin: {model_filepath}")
        return model_module
    except Exception as e:
        logger.critical(f"Failed to load model plugin from {model_filepath}: {e}", exc_info=True)
        return None

def build_job_json(run_id, engine_name, model_filepath, sne_data_info, bao_data_info):
    """
    This is the main function of the module. It orchestrates the entire
    aggregation process.

    Workflow:
    1. Loads the procedural `.py` files for the models.
    2. Loads the observational data files using `data_loaders`.
    3. Assembles a master dictionary (`job_dict`) containing all information.
    4. Serializes this dictionary into the final "Job JSON" string.

    Args:
        run_id (str): A unique identifier for this analysis run.
        engine_name (str): The name of the selected cosmology engine version.
        model_filepath (str): Path to the alternative cosmological model's .py file.
        sne_data_info (dict): Dict containing 'path', 'format_key', and 'extra_args' for SNe.
        bao_data_info (dict): Dict containing 'path' and 'format_key' for BAO.

    Returns:
        str: A JSON string representing the complete analysis job, or None if failed.
    """
    logger = logging.getLogger()
    logger.info("--- Building Job JSON for Engine ---")

    # --- Step 1: Model Loading ---
    lcdm_model_module = _load_model_module('lcdm_model.py')
    alt_model_module = _load_model_module(model_filepath)
    if not lcdm_model_module or not alt_model_module:
        logger.error("Could not load one or both model modules. Aborting job build.")
        return None

    # --- Step 2: Observational Data Loading ---
    sne_df = data_loaders.load_sne_data(
        sne_data_info['path'],
        format_key=sne_data_info['format_key'],
        **sne_data_info.get('extra_args', {})
    )
    bao_df = data_loaders.load_bao_data(
        bao_data_info['path'],
        format_key=bao_data_info['format_key']
    ) if bao_data_info.get('path') else None

    if sne_df is None and bao_df is None:
        logger.error("Failed to load any observational data. Aborting job build.")
        return None

    # --- Step 3: Assemble the Job Dictionary ---
    # This dictionary is the heart of our DSL. Its structure is defined here
    # and serves as the contract between the aggregator and the engine.
    job_dict = {
        # METADATA: Information about the run itself.
        "metadata": {
            "run_id": run_id,
            "engine_name": engine_name,
            "creation_timestamp": datetime.now().isoformat(),
            "project_version": "1.4b" # Version updated to 1.4b
        },
        # MODELS: Defines the models to be compared.
        "models": {
            "lcdm": {
                "name": getattr(lcdm_model_module, 'MODEL_NAME', 'Unknown'),
                "parameters": getattr(lcdm_model_module, 'PARAMETER_NAMES', []),
                "initial_guesses": getattr(lcdm_model_module, 'INITIAL_GUESSES', []),
                "bounds": getattr(lcdm_model_module, 'PARAMETER_BOUNDS', []),
                "fixed_params": getattr(lcdm_model_module, 'FIXED_PARAMS', {}),
                # NEW (v1.4b): Add LaTeX equations for plotting later
                "equations_sn": getattr(lcdm_model_module, 'MODEL_EQUATIONS_LATEX_SN', []),
                "equations_bao": getattr(lcdm_model_module, 'MODEL_EQUATIONS_LATEX_BAO', [])
            },
            "alt_model": {
                "name": getattr(alt_model_module, 'MODEL_NAME', 'Unknown'),
                "filepath": model_filepath, # Essential for the engine to load the code
                "parameters": getattr(alt_model_module, 'PARAMETER_NAMES', []),
                "initial_guesses": getattr(alt_model_module, 'INITIAL_GUESSES', []),
                "bounds": getattr(alt_model_module, 'PARAMETER_BOUNDS', []),
                "fixed_params": getattr(alt_model_module, 'FIXED_PARAMS', {}),
                # NEW (v1.4b): Add LaTeX equations for plotting later
                "equations_sn": getattr(alt_model_module, 'MODEL_EQUATIONS_LATEX_SN', []),
                "equations_bao": getattr(alt_model_module, 'MODEL_EQUATIONS_LATEX_BAO', [])
            }
        },
        # DATASETS: Contains the actual observational data points and metadata.
        "datasets": {
            "sne_data": {
                "data": sne_df.to_dict(orient='split') if sne_df is not None else None,
                "attributes": sne_df.attrs if sne_df is not None else {}
            },
            "bao_data": {
                "data": bao_df.to_dict(orient='split') if bao_df is not None else None,
                "attributes": bao_df.attrs if bao_df is not None else {}
            }
        },
        # ENGINE_SETTINGS: Allows passing configuration options to the engine.
        "engine_settings": {
            "minimizer_method": "L-BFGS-B",
            "confidence_level": 0.95
        }
    }

    # --- Step 4: Serialize to JSON ---
    try:
        job_json = json.dumps(job_dict, cls=CustomJSONEncoder, indent=2)
        logger.info("Successfully built and serialized Job JSON.")
        return job_json
    except Exception as e:
        logger.critical(f"Failed to serialize job dictionary to JSON: {e}", exc_info=True)
        return None