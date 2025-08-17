# utils.py
"""Common utility functions for the Copernican Suite.

This module centralises a handful of small helpers used across the project
so that engines and parsers remain lightweight.  All dataset metadata and
tables are now provided in YAML format only; any legacy JSON handling has
been removed.  Functions here emphasise safe filename construction and
lightweight metadata parsing so that higher level modules can focus on
science logic rather than housekeeping.
"""
# These helpers are intentionally tiny but keep repetitive tasks such as
# timestamp generation and directory creation in one place.

import logging
import os
import time

import numpy as np
import yaml


def get_timestamp():
    """Generates a standardized timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def check_dataset_id(dataset_id: str) -> str:
    """Return ``dataset_id`` stripped of forbidden characters.

    The Copernican Suite expects ``dataset_id`` to be safe for file
    paths. Any unexpected characters such as spaces or path separators are
    dropped rather than replaced so that a slightly malformed identifier
    cannot corrupt directory structures.
    """

    forbidden = set(' \\/:*?"<>|')
    return "".join(ch for ch in dataset_id if ch not in forbidden)


def generate_filename(
    file_type,
    dataset_id,
    ext,
    model_name="",
    timestamp=None,
):
    """Generate a harmonized filename for all outputs.

    Parameters
    ----------
    file_type : str
        Short descriptor of the file's contents.
    dataset_id : str
        Identifier used in output filenames. It should already comply
        with :func:`check_dataset_id` rules.
    ext : str
        File extension without the leading period.
    model_name : str, optional
        Name of the cosmological model, used when comparing multiple
        models.
    timestamp : str, optional
        Timestamp string applied to the filename. When ``None`` the
        current timestamp is generated.
    """
    sanitized_type = file_type.replace("_", "-").lower()
    sanitized_model = model_name.replace("_", "-").replace(".", "")
    checked_id = check_dataset_id(dataset_id)
    base_name = (
        f"{sanitized_type}-{sanitized_model}-{checked_id}"
        if sanitized_model
        else f"{sanitized_type}-{checked_id}"
    )
    ts = timestamp or get_timestamp()
    return f"{base_name}_{ts}.{ext}"


def ensure_dir_exists(directory):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def load_metadata_from_dir(data_dir: str) -> dict:
    """Return dataset metadata from ``data_dir`` if available.

    The loader looks for a file starting with ``metadata`` and ending in
    ``.yml`` or ``.yaml``. JSON files were supported in earlier versions but
    have been removed to keep the format consistent across the project.
    Some legacy metadata files contain unquoted values with embedded colons,
    e.g. URLs. ``yaml.safe_load`` rejects such scalars so a simple
    line-by-line fallback parser is used when the strict loader fails.
    """
    try:
        # fmt: off
        meta_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("metadata")
            and f.lower().endswith((".yml", ".yaml"))
        ]
        # fmt: on
        if meta_files:
            path = os.path.join(data_dir, sorted(meta_files)[0])
            try:
                with open(path, "r") as fh:
                    return yaml.safe_load(fh)
            except Exception:
                # Fallback: parse ``key: value`` pairs manually so that
                # unquoted URLs or other colon-containing values do not break
                # metadata loading.
                meta = {}
                with open(path, "r") as fh:
                    for line in fh:
                        if ":" in line:
                            key, val = line.strip().split(":", 1)
                            meta[key.strip()] = val.strip()
                return meta
    except Exception:
        pass
    return {}


def set_random_seed(seed: int = 0) -> None:
    """Seed NumPy's global RNG and log the selected value.

    Engines call this helper so that optimisation results can be
    reproduced exactly when the same seed is provided.
    """
    np.random.seed(seed)
    logging.getLogger().info("Global RNG seed set to %s", seed)
