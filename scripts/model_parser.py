"""Model Parser for Copernican Suite."""
# DEV NOTE (v1.5a): Initial skeleton implementing Phase 0.
import json
import os

def validate_and_cache(json_path, cache_dir):
    """Validate a DSL file and write sanitized JSON to the cache."""
    with open(json_path, 'r', encoding='utf-8') as handle:
        model_data = json.load(handle)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(json_path))
    with open(cache_file, 'w', encoding='utf-8') as handle:
        json.dump(model_data, handle, indent=2)
    return cache_file
