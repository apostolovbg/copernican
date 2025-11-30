"""Tests for the GUI progress state helper."""

from pathlib import Path

from copernican_lib import progress_state


def test_progress_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    payload = {"stage_label": "burn-in", "batch_percent": 5}
    progress_state.record_progress(path, payload)
    assert progress_state.load_progress(path) == payload
    updated = {"stage_label": "production", "walker_percent": 83}
    progress_state.record_progress(path, updated)
    assert progress_state.load_progress(path) == updated
    progress_state.clear_progress(path)
    assert progress_state.load_progress(path) is None
