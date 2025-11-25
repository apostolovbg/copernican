# Last Updated: 2025-11-25
"""Tests for the Tkinter GUI scaffold."""

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
    gui.start_run()
    assert gui.status is RunStatus.RUNNING
    gui.update_progress(50)
    assert gui.progress == 50
    gui.cancel_run()
    assert gui.status is RunStatus.CANCELLED
    gui.stop_run()
    assert gui.status is RunStatus.PAUSED
    gui.update_progress(120)
    assert gui.status is RunStatus.IDLE
    assert gui.summary.output_links
