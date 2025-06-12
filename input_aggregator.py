# input_aggregator.py
# Aggregates all user inputs and data into a single "Job JSON" structure.

"""
DEV NOTE (v1.4rc): Fixed a critical bug that caused a TypeError in the
engine. The `build_job_json` function was incorrectly returning a serialized
JSON string (`json.dumps`) instead of the raw Python dictionary. The fix is
to return the dictionary directly, ensuring the engine receives the data
structure it expects. The function name remains the same as it describes the
data structure being built.
"""

import logging
import os
import json
import importlib.util
import pandas as pd
import data_loaders

# --- Helper Functions ---

def _load_model_metadata(path):
    """Dynamically loads a model's METADATA dictionary from its .py file."""
    try:
        spec = importlib.util.spec_from_file_location("model_plugin", path)
        model_plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_plugin)
        return model_plugin.METADATA
    except Exception as e:
        logging.error(f"Failed to load metadata from model plugin at {path}: {e}")
        return None

def _load_data(data_info, parsers_registry):
    """Loads dataset using the appropriate parser and returns it as a dict."""
    path = data_info.get('path')
    format_key = data_info.get('format_key')
    extra_args = data_info.get('extra_args', {})
    
    if not all([path, format_key]):
        return None

    parser_info = parsers_registry.get(format_key)
    if not parser_info:
        logging.error(f"No parser found for format key '{format_key}'.")
        return None
        
    parser_func = parser_info['function']
    
    try:
        logging.info(f"Loading data from '{os.path.basename(path)}' using parser '{format_key}'...")
        df = parser_func(path, **extra_args)
        # Convert DataFrame to a serializable dictionary format ('split')
        return {
            "name": format_key,
            "dataframe": df.to_dict('split')
        }
    except Exception as e:
        logging.error(f"Failed to load or parse data from {path}: {e}", exc_info=True)
        return None

# --- Main Aggregation Function ---

def build_job_json(run_id, engine_name, alt_model_path, sne_data_info, bao_data_info):
    """
    Consolidates all configuration and data into a single Python dictionary.
    This dictionary is the "job" that will be passed to the engine.
    """
    logging.info("Building job data structure...")
    try:
        # Standard model is always LCDM
        lcdm_model_path = 'lcdm_model.py'
        lcdm_metadata = _load_model_metadata(lcdm_model_path)
        alt_metadata = _load_model_metadata(alt_model_path)
        
        # Load the actual data using the selected parsers
        sne_data = _load_data(sne_data_info, data_loaders.SNE_PARSERS)
        bao_data = _load_data(bao_data_info, data_loaders.BAO_PARSERS) if bao_data_info else {}

        if not all([lcdm_metadata, alt_metadata, sne_data]):
            logging.critical("Failed to load essential metadata or SNe data. Cannot proceed.")
            return None

        # Assemble the final job dictionary
        job_dict = {
            "run_id": run_id,
            "engine_name": engine_name,
            "models": {
                "model1": {
                    "path": lcdm_model_path,
                    "metadata": lcdm_metadata
                },
                "model2": {
                    "path": alt_model_path,
                    "metadata": alt_metadata
                }
            },
            "data": {
                "sne_data": sne_data,
                "bao_data": bao_data
            }
        }
        
        logging.info("Job data structure successfully aggregated.")
        # MODIFIED (v1.4rc): Return the raw dictionary, not a JSON string.
        return job_dict
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during job aggregation: {e}", exc_info=True)
        return None