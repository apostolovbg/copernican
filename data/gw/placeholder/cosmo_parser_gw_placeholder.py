"""Placeholder parser for future gravitational wave datasets."""

import logging
import os
from copernican_lib.data_loaders import register_gw_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "GW Placeholder")
DESCRIPTION = META.get(
    "description",
    "Placeholder GW parser.",
)

@register_gw_parser(DATASET_NAME, DESCRIPTION, data_dir=DATA_DIR)
def parse_gw_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    # Gravitational wave support is under development. This stub allows the rest
    # of the framework to run even though no real data is parsed yet.
    logger = logging.getLogger()
    logger.info(f"GW parser placeholder invoked in {data_dir}. Feature not implemented.")
    return None
