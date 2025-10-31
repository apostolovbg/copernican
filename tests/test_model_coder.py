# Copyright (c) 2025 Copernican Suite developers.
# Last Updated: 2025-10-31
# See LICENSE.md in the repository root for details.

"""Security tests for ``model_coder`` expression handling."""

import pickle
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

    def test_compile_sympy_expr_returns_picklable_callable(self):
        """Generated helpers should pickle under the spawn start method."""
        z = sp.symbols("z")
        expr = z + 1
        fn = model_coder._compile_sympy_expr(expr, (z,), name_hint="picklable")
        payload = pickle.dumps(fn)
        self.assertIsInstance(payload, bytes)
        restored = pickle.loads(payload)
        self.assertIsInstance(
            restored,
            model_coder._GeneratedCallable,
        )
        self.assertEqual(restored(1), 2)
        self.assertIsInstance(
            fn,
            model_coder._GeneratedCallable,
        )
        self.assertEqual(
            fn.python_function.__module__,
            "copernican_lib.model_coder",
        )
        self.assertTrue(hasattr(model_coder, fn.python_function.__name__))

    def test_compile_sympy_expr_integral_execution(self):
        """``_compile_sympy_expr`` should handle integrals safely."""
        z = sp.symbols("z")
        expr = sp.Integral(z, (z, 0, 1))  # integral of z from 0 to 1 = 0.5
        fn = model_coder._compile_sympy_expr(expr, (z,))
        self.assertAlmostEqual(fn(0), 0.5)
        self.assertEqual(
            fn.python_function.__globals__.get("__builtins__"),
            {},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
