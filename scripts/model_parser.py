"""Model parser for Copernican Suite JSON models."""
# DEV NOTE (v1.5f): Schema updated with optional CMB, gravitational wave, and
# standard siren fields for Phase 6. Writes validated JSON models to the cache
# directory and reports errors through ``error_handler``.
# DEV NOTE (v1.5f hotfix): Added optional ``abstract``, ``description`` and
# ``notes`` fields for human readability.

import json
from jsonschema import validate, ValidationError
from pathlib import Path
from . import error_handler

MODEL_SCHEMA = {
    "type": "object",
    "required": ["model_name", "version", "parameters", "equations"],
    "properties": {
        "model_name": {"type": "string"},
        "version": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "python_var", "initial_guess", "bounds"],
                "properties": {
                    "name": {"type": "string"},
                    "python_var": {"type": "string"},
                    "initial_guess": {"type": "number"},
                    "bounds": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"}
                    },
                    "unit": {"type": "string"},
                    "latex_name": {"type": "string"}
                }
            }
        },
        "equations": {"type": "object"},
        "cmb": {"type": "object"},
        "gravitational_waves": {"type": "object"},
        "standard_sirens": {"type": "object"},
        # Optional human-readable fields used by upcoming UI modules
        "abstract": {"type": "string"},
        "description": {"type": "string"},
        "notes": {"type": "string"}
    }
}


def parse_model_json(path, cache_dir):
    """Validate ``path`` and write cleaned JSON to ``cache_dir``.

    Parameters
    ----------
    path : str or Path
        Source JSON model file.
    cache_dir : str or Path
        Directory where the sanitized model will be stored.

    Returns
    -------
    str
        Path to the sanitized cache file.
    """
    path = Path(path)
    try:
        with path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        error_handler.report_error(f"Failed to read model JSON '{path}': {e}")
        raise

    try:
        validate(instance=data, schema=MODEL_SCHEMA)
    except ValidationError as e:
        error_handler.report_error(f"Model JSON validation error: {e.message}")
        raise ValueError(f"Model JSON validation error: {e.message}") from e

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cache_{path.name}"
    with cache_path.open("w") as f:
        json.dump(data, f, indent=2)
    return str(cache_path)
