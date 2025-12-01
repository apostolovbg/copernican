"""Tests for the Tkinter GUI scaffold."""

import os
import tempfile
from pathlib import Path
from types import MethodType

from copernican_lib import run_manifest
from copernican_lib.gui import CopernicanGUI, RunStatus


def _prime_gui_selections(gui: CopernicanGUI) -> None:
    """Seed model and engine selections for tests that confirm runs."""

    model_entry = next(iter(gui.model_index.values()))
    engine_entry = next(iter(gui.engine_index.values()))
    gui.selected_models = [model_entry["id"]]
    gui._selected_model_entry = model_entry  # type: ignore[attr-defined]
    gui.selected_engine = engine_entry["id"]
    gui._selected_engine_entry = engine_entry  # type: ignore[attr-defined]


def _stub_worker_launch(gui: CopernicanGUI) -> None:
    """Avoid spawning the CLI worker during unit tests."""

    def _no_worker(self, *, config: dict) -> None:
        self._last_worker_config = config  # type: ignore[attr-defined]

    gui._launch_worker_process = MethodType(
        _no_worker,
        gui,
    )


def test_catalogue_metadata_and_filters() -> None:
    gui = CopernicanGUI(render=False)
    gui.refresh_inventory()
    assert gui.catalogue_index
    planck = gui.catalogue_index.get("planck_2018_lite")
    assert planck is not None
    assert planck["parser_trusted"]
    assert planck["metadata_digest"]
    filtered = gui.filter_catalogue(["cmb"])
    assert any(entry["id"] == "planck_2018_lite" for entry in filtered)
    contents = gui.view_metadata_file("planck_2018_lite")
    assert "dataset_name" in contents
    record = gui.revalidate_dataset("planck_2018_lite")
    assert record["hashes"]


def test_model_and_engine_metadata_actions() -> None:
    gui = CopernicanGUI(render=False)
    gui.refresh_inventory()
    assert gui.model_index
    model_entry = next(iter(gui.model_index.values()))
    assert model_entry["hash"]
    model_text = gui.view_metadata_file(model_entry["id"])
    assert model_text
    engine_entry = next(iter(gui.engine_index.values()))
    engine_text = gui.view_metadata_file(engine_entry["id"])
    assert engine_entry["hash"]
    assert engine_text
    assert gui.open_folder(Path(model_entry["path"]).parent.as_posix())


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
    _prime_gui_selections(gui)
    _stub_worker_launch(gui)
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"data")
        fh.flush()
        gui.register_dataset(dataset_id="ds", path=fh.name, name="Dataset")
    gui.draft.seed = "3"
    gui.confirm_start_run()
    assert gui.status is RunStatus.RUNNING
    assert gui.pending_manifest is not None
    assert gui.output_directory_prepared is True
    gui.update_progress(50)
    assert gui.progress == 50
    gui.pause_run()
    assert gui.status is RunStatus.RUNNING
    assert gui.alerts[-1].message.startswith("Pause/resume")
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
    gui.draft.walkers = "33"
    gui.draft.burn_in = "20"
    gui.draft.production_steps = "100"
    gui.draft.pool_size = "4"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gui.export_manifest(tmpdir)
        loaded = run_manifest.load_manifest(path)
        assert loaded["selection"]["models"]
        imported = gui.import_manifest(path)
        assert imported["selection"]["engine"]["name"]
        assert gui.selected_models
        assert gui.draft.walkers == "33"
        assert gui.draft.burn_in == "20"
        assert gui.draft.production_steps == "100"
        assert gui.draft.pool_size == "4"


def test_insert_manifest_from_builder_prepares_pending_manifest() -> None:
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    gui.draft.walkers = "48"
    gui.insert_manifest_from_builder()
    assert gui.pending_manifest is not None
    config = gui.pending_manifest.get("configuration", {})
    run_settings = config.get("run_settings", {})
    assert run_settings.get("n_walkers") == 48
    assert gui.summary.manifest_metadata


def test_duplicate_manifest_prefills_builder(tmp_path: Path) -> None:
    gui = CopernicanGUI(render=False)
    gui.selected_models = ["LambdaCDM"]
    gui.selected_engine = "cosmo_engine_mcmc"
    gui.draft.seed = "5"
    gui.register_dataset(
        dataset_id="planck_2018_lite",
        path="",
        name="Planck 2018 Lite",
    )
    manifest = gui._generate_manifest_snapshot()
    path = run_manifest.save_manifest(manifest, tmp_path)
    gui.duplicate_manifest_for_editing(path)
    assert "planck_2018_lite" in gui.draft.data
    assert gui.draft.plan.startswith("Duplicate & Edit")


def test_application_diagnostics_logging(tmp_path: Path) -> None:
    gui = CopernicanGUI(render=False)
    assert gui.application_log_path
    assert os.path.exists(gui.application_log_path)
    gui.create_toast("App diagnostics ready", severity="INFO", context="app")
    gui.set_diagnostics_filter("ERROR")
    gui.create_toast("App failure", severity="ERROR", context="app")
    filtered = gui.get_application_log_entries()
    assert filtered
    assert any(entry.severity == "ERROR" for entry in filtered)
    export_path = gui.export_application_logs(tmp_path)
    assert os.path.exists(export_path)


def test_run_log_confirmation_and_anchor_jump(tmp_path: Path) -> None:
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    _stub_worker_launch(gui)
    assert gui.run_log_path is None
    gui.confirm_start_run()
    assert gui.run_log_path is not None
    assert os.path.exists(gui.run_log_path)
    entries = gui.get_run_log_entries()
    assert entries
    assert entries[0].anchor.startswith("run-")
    gui.start_run()
    gui.update_progress(100)
    exported = gui.export_run_logs(tmp_path)
    assert os.path.exists(exported)
    if gui.alerts:
        anchor = gui.alerts[-1].anchor
        snippet = gui.jump_to_log_anchor(anchor)
        assert snippet is not None
