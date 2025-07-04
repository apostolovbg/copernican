import os
import logging
from scripts.data_loaders import register_cmb_parser

@register_cmb_parser("cmb_placeholder_v1", "Placeholder CMB parser.", data_dir=os.path.dirname(__file__))
def parse_cmb_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"CMB parser placeholder invoked in {data_dir}. Feature not implemented.")
    return None
