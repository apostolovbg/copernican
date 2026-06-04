"""Smoke tests for copernican.lib.cli.menus."""

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from copernican.lib.cli import menus as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.cli.menus")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Assert the module keeps its exported CLI menu surface."""

    def test_public_symbols_remain_available(self) -> None:
        self.assertTrue(hasattr(module, "show_splash_screen"))
        self.assertTrue(hasattr(module, "select_seed"))
        self.assertTrue(hasattr(module, "select_from_list"))
        self.assertTrue(hasattr(module, "select_model_from_list"))
        self.assertTrue(hasattr(module, "normalise_failure_reasons"))
        self.assertTrue(hasattr(module, "prompt_stage1_retry"))

    def test_source_mentions_public_symbols(self) -> None:
        source = inspect.getsource(module)
        self.assertIn("show_splash_screen", source)
        self.assertIn("select_seed", source)
        self.assertIn("select_from_list", source)
        self.assertIn("select_model_from_list", source)
        self.assertIn("normalise_failure_reasons", source)
        self.assertIn("prompt_stage1_retry", source)

    def test_model_loader_prompt_accepts_a_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "custom_model.yaml"
            model_path.write_text(
                "model_name: Demo\nversion: '1.0'\n"
                "parameters: []\n"
                "equations: {}\n",
                encoding="utf-8",
            )
            with mock.patch(
                "copernican.lib.model_spec_validator.validate_and_cache_model",
                return_value=str(model_path),
            ):
                with mock.patch.object(
                    module.console,
                    "ask",
                    side_effect=["3", str(model_path)],
                ):
                    choice = module.select_from_list(
                        ["a", "b"],
                        "Select model",
                        allow_load_model=True,
                    )
        self.assertEqual(choice, str(model_path.resolve()))


if __name__ == "__main__":
    unittest.main()
