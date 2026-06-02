"""Smoke tests for copernican.lib.logger."""

from __future__ import annotations

import builtins
import logging
import sys
import tempfile
import unittest
from pathlib import Path

from copernican.lib import logger as log_mod
from copernican.lib import logger as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.logger")


class TestLoggerSurface(unittest.TestCase):
    """Exercise the logger helper surface directly."""

    def test_surface_helpers_and_console_capture(self) -> None:
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_print = builtins.print
        original_input = builtins.input
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self.assertTrue(callable(log_mod.ensure_console_capture))
                self.assertIs(log_mod.get_logger(), root_logger)
                self.assertTrue(callable(log_mod.log_environment_info))
                self.assertTrue(callable(log_mod.setup_logging))
                self.assertTrue(callable(log_mod.setup_monitor_logging))
                self.assertFalse(hasattr(log_mod, "setup_program_logging"))
                self.assertFalse(hasattr(log_mod, "get_program_logger"))
                self.assertFalse(hasattr(log_mod, "get_program_log_path"))

                console_filter = log_mod._ConsoleFilter()
                record = logging.LogRecord(
                    "copernican",
                    logging.INFO,
                    "logger-test",
                    1,
                    "message",
                    (),
                    None,
                )
                self.assertEqual(console_filter.filter.__name__, "filter")
                self.assertTrue(console_filter.filter(record))
                record.console_capture = True
                self.assertFalse(console_filter.filter(record))

                builtins.input = lambda prompt="": "answer"
                log_path = log_mod.setup_logging(
                    log_dir=tmpdir,
                    base_dir=tmpdir,
                    log_tag="logger-test",
                )
                self.assertTrue(Path(log_path).exists())
                print("hello", flush=True)
                self.assertEqual(input("prompt: "), "answer")
                self.assertEqual(builtins.print.__name__, "print_patch")
                self.assertEqual(builtins.input.__name__, "input_patch")
                self.assertEqual(sys.stdout.write.__name__, "write")
                self.assertEqual(sys.stdout.flush.__name__, "flush")
                self.assertEqual(sys.stderr.write.__name__, "write")
                self.assertEqual(sys.stderr.flush.__name__, "flush")
                sys.stdout.flush()
                sys.stderr.flush()
                log_mod.ensure_console_capture(tmpdir)
                self.assertTrue(
                    getattr(builtins.print, "__copernican_patched__", False)
                )
                self.assertTrue(
                    getattr(builtins.input, "__copernican_patched__", False)
                )

                monitor_logger, monitor_path = log_mod.setup_monitor_logging(
                    log_dir=tmpdir,
                    log_tag="monitor-test",
                )
                self.assertEqual(monitor_logger.name, "copernican.gui.run")
                self.assertTrue(Path(monitor_path).exists())

                log_mod.log_environment_info(
                    target_logger=log_mod.get_logger(),
                    console=False,
                )
                self.assertIn(
                    "hello", Path(log_path).read_text(encoding="utf-8")
                )
                self.assertIn(
                    "prompt: answer",
                    Path(log_path).read_text(encoding="utf-8"),
                )
            finally:
                log_mod._close_handlers(root_logger)
                log_mod._close_handlers(
                    logging.getLogger("copernican.gui.run")
                )
                root_logger.setLevel(original_level)
                builtins.print = original_print
                builtins.input = original_input
                sys.stdout = original_stdout
                sys.stderr = original_stderr


if __name__ == "__main__":
    unittest.main()
