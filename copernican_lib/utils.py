# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# utils.py
"""Common utility functions for the Copernican Suite.

This module centralises a handful of small helpers used across the project
so that engines and parsers remain lightweight.  All dataset metadata and
tables are now provided in YAML format only; any legacy JSON handling has
been removed.  Functions here emphasise safe filename construction and
lightweight metadata parsing so that higher level modules can focus on
science logic rather than housekeeping.  Runtime timestamps are emitted in
Coordinated Universal Time (UTC) so logs, manifests and filenames remain
stable regardless of the host machine's locale.
"""
# These helpers are intentionally tiny but keep repetitive tasks such as
# timestamp generation and directory creation in one place.

import hashlib
import logging
import os
import random
from datetime import datetime, timezone

import numpy as np
import yaml


def get_utc_now() -> datetime:
    """Return the current UTC time with timezone information."""

    return datetime.now(timezone.utc)


def get_timestamp(now: datetime | None = None) -> str:
    """Generate a standardized UTC timestamp string.

    Parameters
    ----------
    now : :class:`datetime.datetime`, optional
        Explicit timestamp to convert.  When omitted the current UTC time is
        sampled.  Naive datetime objects are assumed to already represent UTC
        and will be tagged accordingly.
    """

    moment = now or get_utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y%m%d_%H%M%S")


def compute_sha256(path: str) -> str:
    """Return the SHA256 hex digest for the file at ``path``.

    The file is read in small chunks so that large datasets do not require
    excessive memory.  A hexadecimal string is returned to keep manifests
    human readable while still uniquely identifying file contents.
    """

    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


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
    timestamp_suffix = timestamp or get_timestamp()
    return f"{base_name}_{timestamp_suffix}.{ext}"


def ensure_dir_exists(directory):
    """Creates the specified directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def load_metadata_from_dir(data_dir: str) -> dict:
    """Return dataset metadata from ``data_dir`` if available.

    The loader searches for a file starting with ``metadata`` and ending in
    ``.yml`` or ``.yaml``. Metadata must be valid YAML; parse errors are
    surfaced to the caller so that malformed files do not silently pass
    through.
    """
    logger = logging.getLogger(__name__)
    try:
        # fmt: off
        meta_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("metadata")
            and f.lower().endswith((".yml", ".yaml"))
        ]
        # fmt: on
    except OSError as exc:
        logger.warning("Failed to list metadata in %s: %s", data_dir, exc)
        raise

    if not meta_files:
        return {}

    path = os.path.join(data_dir, sorted(meta_files)[0])
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to load metadata from %s: %s", path, exc)
        raise


CURRENT_SEED = 0


def set_random_seed(seed: int = 0) -> None:
    """Seed global RNGs and record the selected value.

    Engines call this helper so optimisation results can be reproduced
    when the same seed is provided.  The Python ``random`` module and
    optional engine libraries such as CAMB are seeded when available.
    The chosen seed is stored for later retrieval by
    :func:`get_random_seed` so the run manifest and logs can access it
    without threading the value through multiple functions.
    """

    global CURRENT_SEED
    CURRENT_SEED = seed
    np.random.seed(seed)
    random.seed(seed)
    logger = logging.getLogger()
    try:  # pragma: no cover - CAMB is optional
        import camb  # type: ignore

        if hasattr(camb, "set_random_seed"):
            camb.set_random_seed(seed)
            logger.info("CAMB RNG seed set to %s", seed)
    except Exception:
        logger.debug("CAMB RNG seeding unavailable", exc_info=True)
    logger.info("Global RNG seed set to %s", seed)


def get_random_seed() -> int:
    """Return the seed most recently passed to :func:`set_random_seed`."""

    return CURRENT_SEED
