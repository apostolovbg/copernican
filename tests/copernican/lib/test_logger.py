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
                self.assertTrue(callable(log_mod.parse_worker_event))
                self.assertTrue(callable(log_mod.setup_logging))
                self.assertTrue(callable(log_mod.setup_monitor_logger))
                self.assertFalse(hasattr(log_mod, "setup_monitor_logging"))
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
                    base_dir=str(Path(tmpdir) / "repository"),
                    log_tag="logger-test",
                )
                self.assertTrue(Path(log_path).exists())
                self.assertEqual(
                    log_mod.setup_logging(
                        log_dir=tmpdir,
                        base_dir=str(Path(tmpdir) / "repository"),
                        log_tag="logger-test",
                    ),
                    log_path,
                )
                print("hello", flush=True)
                print("failure", file=sys.stderr, flush=True)
                self.assertEqual(input("prompt: "), "answer")
                self.assertEqual(builtins.print.__name__, "print_patch")
                self.assertEqual(builtins.input.__name__, "input_patch")
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
                log_mod.ensure_console_capture(tmpdir)
                self.assertTrue(
                    getattr(builtins.print, "__copernican_patched__", False)
                )
                self.assertTrue(
                    getattr(builtins.input, "__copernican_patched__", False)
                )

                monitor_logger = log_mod.setup_monitor_logger()
                self.assertEqual(monitor_logger.name, "copernican.gui.run")
                self.assertEqual(monitor_logger.handlers, [])

                log_mod.log_environment_info(
                    target_logger=log_mod.get_logger(),
                    console=False,
                )
                root_logger.warning("Output directory: %s", tmpdir)
                log_text = Path(log_path).read_text(encoding="utf-8")
                self.assertEqual(
                    log_text.count("Logging initialized with UTC"), 1
                )
                self.assertEqual(log_text.count("hello"), 1)
                self.assertEqual(log_text.count("failure"), 1)
                self.assertIn(
                    "prompt: answer",
                    log_text,
                )
                self.assertIn(f"Output directory: {tmpdir}", log_text)

                worker_formatter = log_mod._WorkerEventFormatter("%(message)s")
                worker_record = logging.LogRecord(
                    "copernican",
                    logging.ERROR,
                    "logger-test",
                    1,
                    "worker failed",
                    (),
                    None,
                )
                worker_line = worker_formatter.format(worker_record)
                self.assertEqual(
                    log_mod.parse_worker_event(worker_line),
                    (logging.ERROR, "worker failed"),
                )
                self.assertIsNone(log_mod.parse_worker_event("plain text"))
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
