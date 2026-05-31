"""Unit tests for the Python language translator."""

import ast
import importlib.util
import unittest
from pathlib import Path

from devcovenant.core.translator import TranslatorDeclaration

_REPO_ROOT = Path(__file__).resolve().parents[5]
_TRANSLATOR_PATH = (
    _REPO_ROOT / "devcovenant/custom/profiles/python/python_translator.py"
)


def _load_translator_module():
    """Load the translator module from the profile directory."""
    spec = importlib.util.spec_from_file_location(
        "python_translator", _TRANSLATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _declaration() -> TranslatorDeclaration:
    """Build a translator declaration for this profile."""
    return TranslatorDeclaration(
        translator_id="python",
        profile_name="python",
        extensions=(".py",),
        can_handle_strategy="module_function",
        can_handle_entrypoint="python_translator.py:can_handle",
        translate_strategy="module_function",
        translate_entrypoint="python_translator.py:translate",
    )


class PythonTranslatorTests(unittest.TestCase):
    """Verify Python translation behavior."""

    def test_can_handle_suffix(self):
        """Declared suffixes are honored."""
        module = _load_translator_module()
        self.assertTrue(
            module.can_handle(
                path=Path("sample.py"), declaration=_declaration()
            )
        )

    def test_translate_emits_identifier_and_risk_facts(self):
        """Translation includes identifiers and risky patterns."""
        module = _load_translator_module()
        unit = module.translate(
            path=Path("sample.py"),
            source=(
                "# module docs\n"
                "def run(value):\n"
                "    return eval(value)\n"
            ),
            declaration=_declaration(),
        )
        names = {fact.name for fact in unit.identifier_facts}
        self.assertIn("run", names)
        self.assertTrue(unit.risk_facts)

    def test_translate_exposes_cached_visitor_contract(self):
        """The translator should keep one explicit Python-facts visitor."""
        module = _load_translator_module()
        self.assertTrue(hasattr(module, "translate"))
        self.assertTrue(hasattr(module, "translate_minimal"))
        self.assertTrue(hasattr(module, "_PythonFactsVisitor"))
        self.assertTrue(hasattr(module, "_PythonDocumentationFactsVisitor"))
        self.assertTrue(
            hasattr(module._PythonFactsVisitor, "visit_FunctionDef")
        )
        self.assertTrue(
            hasattr(module._PythonFactsVisitor, "visit_AsyncFunctionDef")
        )
        self.assertTrue(hasattr(module._PythonFactsVisitor, "visit_ClassDef"))
        self.assertTrue(hasattr(module._PythonFactsVisitor, "visit_Name"))

        visitor = module._PythonFactsVisitor(
            lines=[
                "# module docs",
                "class Widget:",
                "    pass",
                "def run(value):",
                "    return helper",
                "async def run_async(flag):",
                "    return flag",
            ]
        )
        visitor.visit(
            ast.parse(
                "class Widget:\n"
                "    pass\n"
                "def run(value):\n"
                "    return helper\n"
                "async def run_async(flag):\n"
                "    return flag\n"
            )
        )
        names = {fact.name for fact in visitor.identifiers}
        self.assertIn("Widget", names)
        self.assertIn("run", names)
        self.assertIn("run_async", names)
        self.assertIn("helper", names)

    def test_translate_minimal_emits_doc_facts_without_identifier_facts(self):
        """Minimal translation should keep doc facts without extra payload."""
        module = _load_translator_module()
        unit = module.translate_minimal(
            path=Path("sample.py"),
            source=(
                "# module docs\n"
                "class Widget:\n"
                '    """Widget docs."""\n'
                "    pass\n"
                "def run(value):\n"
                "    # explain run\n"
                "    return value\n"
            ),
            declaration=_declaration(),
        )
        self.assertTrue(unit.module_documented)
        self.assertFalse(unit.identifier_facts)
        self.assertFalse(unit.risk_facts)
        names = {fact.name for fact in unit.symbol_doc_facts}
        self.assertIn("Widget", names)
        self.assertIn("run", names)
        self.assertEqual(
            unit.test_name_templates,
            ("test_{stem}.py",),
        )


if __name__ == "__main__":
    unittest.main()
