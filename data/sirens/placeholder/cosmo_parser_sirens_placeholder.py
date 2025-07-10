"""Placeholder parser for future standard siren datasets."""

import logging
import os
from copernican_lib.data_loaders import register_siren_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "Standard Siren Placeholder")
DESCRIPTION = META.get(
    "description",
    "Placeholder standard siren parser.",
)

@register_siren_parser(DATASET_NAME, DESCRIPTION, data_dir=DATA_DIR)
def parse_siren_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    # Placeholder to keep the API consistent while actual siren datasets are
    # being prepared.
    logger = logging.getLogger()
    logger.info(f"Standard siren parser placeholder invoked in {data_dir}. Feature not implemented.")
    return None
