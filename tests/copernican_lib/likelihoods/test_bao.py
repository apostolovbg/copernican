"""Unit tests for BAO helpers."""

from __future__ import annotations

import unittest

from copernican_lib.likelihoods import bao


class BAOPublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the BAO helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(bao.BAOLike))
        self.assertTrue(hasattr(bao, "BAOLike"))


if __name__ == "__main__":
    unittest.main()
