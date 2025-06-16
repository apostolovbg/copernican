# DEV NOTE (v1.6a): Placeholder parser updated for explicit registration system.
import logging
from scripts.data_loaders import BaseParser, register_parser


def parse_siren_placeholder(filepath, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"Standard siren parser placeholder invoked for {filepath}. Feature not implemented.")
    return None


class SirenPlaceholderParser(BaseParser):
    DATA_TYPE = "sirens"
    SOURCE_NAME = "placeholder"
    PARSER_NAME = "siren_placeholder_v1"
    FILE_EXTENSIONS = []

    def parse(self, filepath, **kwargs):
        return parse_siren_placeholder(filepath, **kwargs)


register_parser(
    data_type="sirens",
    source="placeholder",
    name="Siren Placeholder",
    parser=SirenPlaceholderParser()
)

