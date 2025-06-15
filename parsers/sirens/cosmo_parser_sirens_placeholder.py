# DEV NOTE (v1.5f): Placeholder parser for future standard siren data formats.
import logging
from data_loaders import register_siren_parser

@register_siren_parser("siren_placeholder_v1", "Placeholder standard siren parser.")
def parse_siren_placeholder(filepath, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"Standard siren parser placeholder invoked for {filepath}. Feature not implemented.")
    return None
