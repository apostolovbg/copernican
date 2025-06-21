# utils.py
"""Common utility functions for the Copernican Suite."""

import os
import time


def get_timestamp():
    """Generates a standardized timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def generate_filename(file_type, dataset_name, ext, model_name=""):
    """Generates a harmonized filename for all outputs."""
    sanitized_type = file_type.replace('_', '-').lower()
    sanitized_model = model_name.replace('_', '-').replace('.', '')
    sanitized_dataset = (
        dataset_name.replace('_', '-').replace(' ', '')
        .replace('.json', '').replace('.dat', '')
    )
    base_name = f"{sanitized_type}-{sanitized_model}-{sanitized_dataset}" if sanitized_model else f"{sanitized_type}-{sanitized_dataset}"
    return f"{base_name}_{get_timestamp()}.{ext}"


def ensure_dir_exists(directory):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
