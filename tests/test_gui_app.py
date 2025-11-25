# Last Updated: 2025-11-25
"""Tests for the Tkinter GUI scaffold."""

import os
import tempfile

from copernican_lib import run_manifest
from copernican_lib.gui import CopernicanGUI, RunStatus


def test_builder_navigation_and_draft() -> None:
    gui = CopernicanGUI(render=False)
    assert gui.current_step_index == 0
    gui.next_step()
    assert gui.current_step_index == 1
    gui.jump_to_step(3)
    assert gui.current_step_index == 3
    draft = gui.save_draft()
    assert draft.completed_step == 3
    gui.previous_step()
    assert gui.current_step_index == 2
    gui.cancel_builder()
    assert gui.current_step_index == 0
    assert gui.draft.completed_step == 0


def test_run_monitor_lifecycle() -> None:
    gui = CopernicanGUI(render=False)
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"data")
        fh.flush()
        gui.register_dataset(dataset_id="ds", path=fh.name, name="Dataset")
    gui.selected_models.append("ModelA")
    gui.selected_engine = "engine"
    gui.draft.seed = "3"
    gui.confirm_start_run()
    assert gui.status is RunStatus.CONFIGURING
    assert gui.pending_manifest is not None
    assert gui.output_directory_prepared is False
    gui.start_run()
    assert gui.status is RunStatus.RUNNING
    assert gui.output_directory_prepared is True
    gui.update_progress(50)
    assert gui.progress == 50
    gui.pause_run()
    assert gui.status is RunStatus.PAUSED
    gui.cancel_run(disposition="archived")
    assert gui.pending_manifest["status"]["state"] == "cancelled"
    gui.stop_run(disposition="deleted")
    assert gui.status is RunStatus.ABORTED
    gui.update_progress(120)
    assert gui.status is RunStatus.IDLE
    assert gui.summary.output_links
    os.unlink(fh.name)


def test_manifest_import_export_round_trip() -> None:
    gui = CopernicanGUI(render=False)
    gui.draft.model = "ModelB"
    gui.draft.data = "Dataset"
    gui.draft.engine = "engine"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gui.export_manifest(tmpdir)
        loaded = run_manifest.load_manifest(path)
        assert loaded["selection"]["models"]
        imported = gui.import_manifest(path)
        assert imported["selection"]["engine"]["name"]
        assert gui.selected_models
