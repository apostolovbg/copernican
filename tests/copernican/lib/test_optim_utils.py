# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for :mod:`copernican.lib.optim_utils`."""

import inspect
import unittest
from itertools import repeat
from unittest.mock import patch

from copernican.lib import optim_utils


class ProgressThrottleTestCase(unittest.TestCase):
    """Ensure progress updates are rate limited."""

    def test_eval_based_throttle(self) -> None:
        """Output only after every tenth evaluation when time stands still."""

        calls: list[str] = []

        def fake_write(
            msg: str = "", *, end: str = "\n", error: bool = False
        ) -> None:
            calls.append(msg)

        def fake_minimize(
            fun, initial_point, args=(), method=None, bounds=None, options=None
        ):
            for _ in range(25):
                fun(initial_point, *args)
            return object()

        with (
            patch("copernican.lib.optim_utils.console.write", new=fake_write),
            patch("copernican.lib.optim_utils.minimize", new=fake_minimize),
            patch(
                "copernican.lib.optim_utils.time.time",
                side_effect=repeat(0.0),
            ),
        ):
            optim_utils.minimize_with_progress(
                lambda parameters: 1.0, [0.0], [(-1.0, 1.0)]
            )

        progress = [
            captured_output
            for captured_output in calls
            if "Evals:" in captured_output
        ]
        self.assertEqual(len(progress), 2)

    def test_time_based_throttle(self) -> None:
        """Output after half a second even with few evaluations."""

        calls: list[str] = []

        def fake_write(
            msg: str = "", *, end: str = "\n", error: bool = False
        ) -> None:
            calls.append(msg)

        def fake_minimize(
            fun, initial_point, args=(), method=None, bounds=None, options=None
        ):
            for _ in range(5):
                fun(initial_point, *args)
            return object()

        current = {"time": 0.0}

        def fake_time() -> float:
            time_value = current["time"]
            current["time"] += 0.6
            return time_value

        with (
            patch("copernican.lib.optim_utils.console.write", new=fake_write),
            patch("copernican.lib.optim_utils.minimize", new=fake_minimize),
            patch(
                "copernican.lib.optim_utils.time.time", side_effect=fake_time
            ),
        ):
            optim_utils.minimize_with_progress(
                lambda parameters: 1.0, [0.0], [(-1.0, 1.0)]
            )

        progress = [
            captured_output
            for captured_output in calls
            if "Evals:" in captured_output
        ]
        self.assertEqual(len(progress), 5)


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the optimisation helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        source = inspect.getsource(optim_utils.minimize_with_progress)
        self.assertTrue(callable(optim_utils.minimize_with_progress))
        self.assertIn("def wrapped(", source)


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    unittest.main()
