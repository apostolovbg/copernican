"""Tests for the GUI progress state helper."""

import unittest
from pathlib import Path

from copernican_lib import progress_state


class TestProgressState(unittest.TestCase):
    """Exercise round-trip persistence for progress records."""

    def test_progress_state_round_trip(self) -> None:
        path = Path(self._tmp_dir.name) / "progress.json"
        payload = {"stage_label": "burn-in", "batch_percent": 5}
        progress_state.record_progress(path, payload)
        self.assertEqual(progress_state.load_progress(path), payload)
        updated = {"stage_label": "production", "walker_percent": 83}
        progress_state.record_progress(path, updated)
        self.assertEqual(progress_state.load_progress(path), updated)
        progress_state.clear_progress(path)
        self.assertIsNone(progress_state.load_progress(path))

    def setUp(self) -> None:
        import tempfile

        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
