import logging
import os
from scripts.data_loaders import register_gw_parser

@register_gw_parser("gw_placeholder_v1", "Placeholder GW parser.", data_dir=os.path.dirname(__file__))
def parse_gw_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"GW parser placeholder invoked in {data_dir}. Feature not implemented.")
    return None
