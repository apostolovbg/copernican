# utils.py
"""Common utility functions for the Copernican Suite."""
# These helpers are intentionally tiny but keep repetitive tasks such as
# timestamp generation and directory creation in one place.

import os
import time
import yaml
import logging
import numpy as np


def get_timestamp():
    """Generates a standardized timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def generate_filename(file_type, dataset_name, ext, model_name="", timestamp=None):
    """Generates a harmonized filename for all outputs.

    Parameters
    ----------
    file_type : str
        Short descriptor of the file's contents.
    dataset_name : str
        Human readable dataset identifier.
    ext : str
        File extension without the leading period.
    model_name : str, optional
        Name of the cosmological model, used when comparing multiple models.
    timestamp : str, optional
        Timestamp string applied to the filename. When ``None`` the current
        timestamp is generated.
    """
    sanitized_type = file_type.replace('_', '-').lower()
    sanitized_model = model_name.replace('_', '-').replace('.', '')
    sanitized_dataset = (
        dataset_name.replace('_', '-')
        .replace(' ', '')
        .replace('/', '-')
        .replace('.json', '')
        .replace('.dat', '')
    )
    base_name = (
        f"{sanitized_type}-{sanitized_model}-{sanitized_dataset}"
        if sanitized_model
        else f"{sanitized_type}-{sanitized_dataset}"
    )
    ts = timestamp or get_timestamp()
    return f"{base_name}_{ts}.{ext}"


def ensure_dir_exists(directory):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def load_metadata_from_dir(data_dir: str) -> dict:
    """Return dataset metadata from ``data_dir`` if available."""
    try:
        meta_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("metadata")
            and f.lower().endswith((".json", ".yml", ".yaml"))
        ]
        if meta_files:
            with open(os.path.join(data_dir, sorted(meta_files)[0]), "r") as fh:
                return yaml.safe_load(fh)
    except Exception:
        pass
    return {}


def set_random_seed(seed: int = 0) -> None:
    """Seed NumPy's global RNG and log the selected value."""
    np.random.seed(seed)
    logging.getLogger().info("Global RNG seed set to %s", seed)
