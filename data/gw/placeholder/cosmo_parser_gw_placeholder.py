"""Placeholder parser for future gravitational wave datasets."""

import logging
import os
from copernican_lib.data_loaders import register_gw_parser

@register_gw_parser("gw_placeholder_v1", "Placeholder GW parser.", data_dir=os.path.dirname(__file__))
def parse_gw_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    # Gravitational wave support is under development. This stub allows the rest
    # of the framework to run even though no real data is parsed yet.
    logger = logging.getLogger()
    logger.info(f"GW parser placeholder invoked in {data_dir}. Feature not implemented.")
    return None
