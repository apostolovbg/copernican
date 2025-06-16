# DEV NOTE (v1.6a): Placeholder parser updated for explicit registration system.
import logging
from scripts.data_loaders import BaseParser, register_parser


def parse_cmb_placeholder(filepath, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"CMB parser placeholder invoked for {filepath}. Feature not implemented.")
    return None


class CMBPlaceholderParser(BaseParser):
    DATA_TYPE = "cmb"
    SOURCE_NAME = "placeholder"
    PARSER_NAME = "cmb_placeholder_v1"
    FILE_EXTENSIONS = []

    def parse(self, filepath, **kwargs):
        return parse_cmb_placeholder(filepath, **kwargs)


register_parser(
    data_type="cmb",
    source="placeholder",
    name="CMB Placeholder",
    parser=CMBPlaceholderParser()
)

