"""Smoke tests for copernican_lib.logger."""

import tempfile
import unittest
from pathlib import Path

from copernican_lib import logger as log_mod
from copernican_lib import logger as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.logger")


class TestProgramLogging(unittest.TestCase):
    """Exercise log rotation behavior."""

    def test_program_log_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = log_mod.setup_program_logging(
                log_dir=tmpdir,
                base_dir=str(tmpdir),
                rollover_mb=0.0001,
                backup_count=1,
            )
            prog_logger = log_mod.get_program_logger()
            message = "x" * 200
            for _ in range(200):
                prog_logger.info(message)
            primary = Path(log_path)
            rotated = primary.with_name(primary.name + ".1")
            self.assertTrue(primary.exists())
            self.assertTrue(rotated.exists())
            self.assertTrue(primary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
