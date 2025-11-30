# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for :mod:`copernican_lib.optim_utils`."""

import unittest
from itertools import repeat
from unittest.mock import patch

from copernican_lib import optim_utils

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
            fun, x0, args=(), method=None, bounds=None, options=None
        ):
            for _ in range(25):
                fun(x0, *args)
            return object()

        with (
            patch("copernican_lib.optim_utils.console.write", new=fake_write),
            patch("copernican_lib.optim_utils.minimize", new=fake_minimize),
            patch(
                "copernican_lib.optim_utils.time.time",
                side_effect=repeat(0.0),
            ),
        ):
            optim_utils.minimize_with_progress(
                lambda p: 1.0, [0.0], [(-1.0, 1.0)]
            )

        progress = [c for c in calls if "Evals:" in c]
        self.assertEqual(len(progress), 2)

    def test_time_based_throttle(self) -> None:
        """Output after half a second even with few evaluations."""

        calls: list[str] = []

        def fake_write(
            msg: str = "", *, end: str = "\n", error: bool = False
        ) -> None:
            calls.append(msg)

        def fake_minimize(
            fun, x0, args=(), method=None, bounds=None, options=None
        ):
            for _ in range(5):
                fun(x0, *args)
            return object()

        current = {"t": 0.0}

        def fake_time() -> float:
            t = current["t"]
            current["t"] += 0.6
            return t

        with (
            patch("copernican_lib.optim_utils.console.write", new=fake_write),
            patch("copernican_lib.optim_utils.minimize", new=fake_minimize),
            patch(
                "copernican_lib.optim_utils.time.time", side_effect=fake_time
            ),
        ):
            optim_utils.minimize_with_progress(
                lambda p: 1.0, [0.0], [(-1.0, 1.0)]
            )

        progress = [c for c in calls if "Evals:" in c]
        self.assertEqual(len(progress), 5)

if __name__ == "__main__":  # pragma: no cover - manual execution guard
    unittest.main()
