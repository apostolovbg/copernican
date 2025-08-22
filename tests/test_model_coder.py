"""Security tests for ``model_coder`` expression handling."""

import unittest

import sympy as sp

from copernican_lib import model_coder


class TestModelCoderSecurity(unittest.TestCase):
    """Ensure potentially dangerous expressions are not executed."""

    def test_compile_sympy_expr_blocks_import(self):
        """``_compile_sympy_expr`` should deny access to ``__import__``."""
        z = sp.symbols("z")
        malicious = sp.Function("__import__")(sp.Symbol("os"))
        fn = model_coder._compile_sympy_expr(malicious, (z,))
        with self.assertRaises(NameError):
            fn(0)

    def test_safe_parse_expr_rejects_dunder(self):
        """Expressions containing ``__`` should be rejected outright."""
        with self.assertRaises(ValueError):
            model_coder._safe_parse_expr("__import__('os')", {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
