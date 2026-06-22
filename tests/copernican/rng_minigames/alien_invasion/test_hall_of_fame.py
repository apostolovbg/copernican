"""Smoke tests for copernican.rng_minigames.alien_invasion.hall_of_fame."""

import math
import tempfile
import unittest
from pathlib import Path

from copernican.rng_minigames.alien_invasion import hall_of_fame as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__,
            "copernican.rng_minigames.alien_invasion.hall_of_fame",
        )

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "HallOfFame"))
        self.assertTrue(hasattr(module.HallOfFame, "record"))
        self.assertTrue(hasattr(module.HallOfFame, "show"))

    def test_hall_of_fame_sorts_and_limits_entries(self) -> None:
        """The hall of fame should keep the fastest runs and persist them."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            board = module.HallOfFame(tmp_path, limit=3)
            board.record("NI", 15.0)
            board.record("AI", 6.0)
            board.record("NI", 9.0)
            board.record("NI", 30.0)

            restored = module.HallOfFame(tmp_path, limit=3)
            times_left = [entry["time_left"] for entry in restored.entries]
            initials = [entry["initials"] for entry in restored.entries]

            self.assertEqual(times_left, sorted(times_left, reverse=True))
            self.assertEqual(initials[0], "NI")
            self.assertEqual(len(times_left), 3)
            self.assertTrue(math.isclose(max(times_left), 30.0, abs_tol=0.01))


if __name__ == "__main__":
    unittest.main()
