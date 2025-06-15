"""Model parser for Copernican Suite JSON models."""
# DEV NOTE (v1.5a): New module to validate JSON model files as part of the JSON-based pipeline.

import json
from jsonschema import validate, ValidationError
from pathlib import Path

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
        "equations": {"type": "object"}
    }
}


def parse_model_json(path):
    """Validate and load a model JSON file."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    try:
        validate(instance=data, schema=MODEL_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Model JSON validation error: {e.message}") from e
    return data
