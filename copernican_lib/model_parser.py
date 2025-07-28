"""Model parser for Copernican Suite JSON models."""

# This module validates model definition files against a JSON schema and writes
# a sanitized copy to ``models/cache/``. The sanitized file is used by child
# processes so that validation only happens once in the main process.

import json
import re
from jsonschema import validate, ValidationError
from pathlib import Path
import multiprocessing as _mp
from . import error_handler
from . import latex_utils


def _sanitise_name_to_var(name: str) -> str:
    """Return a valid Python identifier derived from a LaTeX name."""
    return latex_utils.sanitize_name(name)

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
                "required": ["name", "bounds"],
                "properties": {
                    "name": {"type": "string"},
                    "python_var": {"type": "string"},
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
        "rs_expression": {"type": "string"},
        "cmb": {"type": "object"},
        "gravitational_waves": {"type": "object"},
        "standard_sirens": {"type": "object"},
        "predicts_bao": {"type": "boolean"},
        # Optional human-readable fields used by upcoming UI modules
        "abstract": {"type": "string"},
        "description": {"type": "string"},
        "notes": {"type": "string"}
    }
}


def parse_model_json(path, cache_dir):
    """Validate ``path`` and write cleaned JSON to ``cache_dir``.

    Validation is performed only in the main process. Worker processes simply
    read the sanitized file produced during program startup.

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
    def _escape_loose_backslashes(text: str) -> str:
        """Double unescaped backslashes so ``json.loads`` accepts LaTeX strings."""
        pattern = re.compile(r"(?<!\\)\\(?![\\\"/bfnrtu])")
        return pattern.sub(r"\\\\", text)

    try:
        raw_text = path.read_text()
        raw_text = _escape_loose_backslashes(raw_text)
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as e:
        error_handler.report_error(f"Failed to read model JSON '{path}': {e}")
        raise

    # Only validate in the main process to avoid random failures when
    # worker processes import this module under multiprocessing. The
    # sanitized file produced here is shared by child processes, so
    # repeated validation is unnecessary.
    if _mp.current_process().name == "MainProcess":
        try:
            validate(instance=data, schema=MODEL_SCHEMA)
        except ValidationError as e:
            error_handler.report_error(
                f"Model JSON validation error: {e.message}"
            )
            raise ValueError(
                f"Model JSON validation error: {e.message}"
            ) from e

    # Auto-generate missing python_var fields from LaTeX names or plain names
    used_vars = {p.get("python_var") for p in data.get("parameters", []) if p.get("python_var")}
    for param in data.get("parameters", []):
        if not param.get("python_var"):
            base = _sanitise_name_to_var(param.get("latex_name", param.get("name", "param")))
            candidate = base
            idx = 2
            while candidate in used_vars:
                candidate = f"{base}_{idx}"
                idx += 1
            param["python_var"] = candidate
            used_vars.add(candidate)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cache_{path.name}"
    def _wrap(expr: str | None) -> str | None:
        if expr is None:
            return expr
        expr = expr.strip()
        if not expr.startswith("$"):
            return f"$${expr}$$"
        return expr

    if isinstance(data.get("equations"), dict):
        for key, val in data["equations"].items():
            if isinstance(val, list):
                data["equations"][key] = [_wrap(v) for v in val]
            elif isinstance(val, str):
                data["equations"][key] = _wrap(val)
    if "Hz_expression" in data:
        data["Hz_expression"] = _wrap(data["Hz_expression"])
    if "rs_expression" in data:
        data["rs_expression"] = _wrap(data["rs_expression"])

    with cache_path.open("w") as f:
        json.dump(data, f, indent=2)
    return str(cache_path)
