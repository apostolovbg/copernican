"""Model Parser for Copernican Suite."""
# DEV NOTE (v1.5e): Added field validation and integration with error_handler.

import json
import os
from . import error_handler

# Required keys for the root of a model JSON file
REQUIRED_ROOT_FIELDS = ["model_name", "version", "date", "parameters", "equations"]
# Required keys for each parameter entry
REQUIRED_PARAMETER_FIELDS = ["name", "latex", "guess", "bounds", "unit"]


def validate_and_cache(json_path, cache_dir):
    """Validate a DSL file and write sanitized JSON to the cache."""
    try:
        with open(json_path, 'r', encoding='utf-8') as handle:
            model_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        error_handler.report_error(f"Failed to read '{json_path}': {exc}")
        return None

    missing = [f for f in REQUIRED_ROOT_FIELDS if f not in model_data]
    if missing:
        error_handler.report_error(
            f"Model file '{json_path}' missing required fields: {missing}")
        return None

    for idx, param in enumerate(model_data.get("parameters", [])):
        p_missing = [f for f in REQUIRED_PARAMETER_FIELDS if f not in param]
        if p_missing:
            error_handler.report_error(
                f"Parameter index {idx} missing fields: {p_missing}")
            return None

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(json_path))
    with open(cache_file, 'w', encoding='utf-8') as handle:
        json.dump(model_data, handle, indent=2)
    return cache_file
