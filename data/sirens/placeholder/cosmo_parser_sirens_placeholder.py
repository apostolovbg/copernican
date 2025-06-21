import logging
import os
from scripts.data_loaders import register_siren_parser

@register_siren_parser("siren_placeholder_v1", "Placeholder standard siren parser.", data_dir=os.path.dirname(__file__))
def parse_siren_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"Standard siren parser placeholder invoked in {data_dir}. Feature not implemented.")
    return None
