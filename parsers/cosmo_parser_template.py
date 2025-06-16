# DEV NOTE (v1.6a)
"""Parser template for new Copernican Suite data parsers.

This scaffold explains the required interface for parser plugins. Copy this file
into the appropriate subfolder under ``parsers/<data_type>/<source_name>/`` and
implement your parsing logic.

Attributes to define on your subclass:
- ``DATA_TYPE``: short name such as ``"sne"`` or ``"bao"``.
- ``SOURCE_NAME``: identifier for the dataset source (e.g., ``"pantheon"``).
- ``PARSER_NAME``: human friendly name for selection menus.
- ``FILE_EXTENSIONS``: list of supported file extensions.

Required methods:
- ``can_parse(self, filepath)`` -> bool
      Return ``True`` if the file appears compatible with this parser.
      The default implementation checks ``FILE_EXTENSIONS``.
- ``parse(self, filepath, **kwargs)`` -> pandas.DataFrame
      Perform the parsing and return a populated DataFrame with
      all expected columns. Any errors should raise ``Exception``
      with a clear message.
- ``get_extra_args(self, base_dir)`` (optional)
      Prompt the user for additional files or settings and return a
      dictionary of keyword arguments for ``parse``. Return ``None`` to
      cancel parsing.

Parsers register automatically when imported via the metaclass in
``scripts.data_loaders.BaseParser``. No manual registration code is
necessary. Simply place the module in the correct folder and ensure it
is importable.

This template itself is not imported by the suite.
"""

from scripts.data_loaders import BaseParser

class MyParser(BaseParser):
    """Short description of the parser."""
    DATA_TYPE = "example"
    SOURCE_NAME = "template"
    PARSER_NAME = "example_template_parser"
    FILE_EXTENSIONS = [".dat"]

    def can_parse(self, filepath):
        return super().can_parse(filepath)

    def parse(self, filepath, **kwargs):
        raise NotImplementedError("Implement parser logic here")
