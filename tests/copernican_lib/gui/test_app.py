"""Tests for the Tkinter GUI scaffold."""

import os
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

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


def _tmp_path_or_default(tmp_path: Path | None) -> Path:
    """Return a usable temporary directory path for unittest methods."""

    if tmp_path is None:
        return Path(tempfile.mkdtemp())
    return tmp_path


class TestCopernicanGUI(unittest.TestCase):
    """Exercise the Tkinter GUI scaffold."""

    def test_catalogue_metadata_and_filters(self) -> None:
        _case_catalogue_metadata_and_filters(self)

    def test_model_and_engine_metadata_actions(self) -> None:
        _case_model_and_engine_metadata_actions(self)

    def test_builder_navigation_and_draft(self) -> None:
        _case_builder_navigation_and_draft(self)

    def test_builder_next_requires_all_pages_selected(self) -> None:
        _case_builder_next_requires_all_pages_selected(self)

    def test_builder_next_advances_when_pages_ready(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_builder_next_advances_when_pages_ready(self, tmp_path)

    def test_save_manifest_creates_new_config_folder(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_save_manifest_creates_new_config_folder(self, tmp_path)

    def test_clear_manifest_resets_state(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_clear_manifest_resets_state(self, tmp_path)

    def test_save_manifest_to_external(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_save_manifest_to_external(self, tmp_path)

    def test_auto_loads_saved_temp_manifest(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_auto_loads_saved_temp_manifest(self, tmp_path)

    def test_confirm_step_requires_saved_manifest(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_confirm_step_requires_saved_manifest(self, tmp_path)

    def test_confirm_start_run_renames_manifest(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_confirm_start_run_renames_manifest(self, tmp_path)

    def test_cancel_inactive_without_configuration(self) -> None:
        _case_cancel_inactive_without_configuration(self)

    def test_run_monitor_lifecycle(self) -> None:
        _case_run_monitor_lifecycle(self)

    def test_manifest_import_export_round_trip(self) -> None:
        _case_manifest_import_export_round_trip(self)

    def test_insert_manifest_from_builder_prepares_pending_manifest(
        self,
    ) -> None:
        _case_insert_manifest_from_builder_prepares_pending_manifest(self)

    def test_duplicate_manifest_prefills_builder(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_duplicate_manifest_prefills_builder(self, tmp_path)

    def test_application_diagnostics_logging(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_application_diagnostics_logging(self, tmp_path)

    def test_run_log_confirmation_and_anchor_jump(
        self,
        tmp_path: Path | None = None,
    ) -> None:
        _case_run_log_confirmation_and_anchor_jump(self, tmp_path)


def _case_catalogue_metadata_and_filters(self) -> None:
    gui = CopernicanGUI(render=False)
    gui.refresh_inventory()
    self.assertTrue(gui.catalogue_index)
    planck = gui.catalogue_index.get("planck_2018_lite")
    self.assertIsNotNone(planck)
    self.assertTrue(planck["parser_trusted"])
    self.assertTrue(planck["metadata_digest"])
    filtered = gui.filter_catalogue(["cmb"])
    self.assertTrue(
        any(entry["id"] == "planck_2018_lite" for entry in filtered)
    )
    contents = gui.view_metadata_file("planck_2018_lite")
    self.assertIn("dataset_name", contents)
    record = gui.revalidate_dataset("planck_2018_lite")
    self.assertTrue(record["hashes"])


def _case_model_and_engine_metadata_actions(self) -> None:
    gui = CopernicanGUI(render=False)
    gui.refresh_inventory()
    self.assertTrue(gui.model_index)
    model_entry = next(iter(gui.model_index.values()))
    self.assertTrue(model_entry["hash"])
    model_text = gui.view_metadata_file(model_entry["id"])
    self.assertTrue(model_text)
    engine_entry = next(iter(gui.engine_index.values()))
    engine_text = gui.view_metadata_file(engine_entry["id"])
    self.assertTrue(engine_entry["hash"])
    self.assertTrue(engine_text)
    self.assertTrue(
        gui.open_folder(Path(model_entry["path"]).parent.as_posix())
    )


def _case_builder_navigation_and_draft(self) -> None:
    gui = CopernicanGUI(render=False)
    self.assertEqual(gui.current_step_index, 0)
    gui.next_step()
    self.assertEqual(gui.current_step_index, 1)
    gui.jump_to_step(3)
    self.assertEqual(gui.current_step_index, 3)
    draft = gui.save_draft()
    self.assertEqual(draft.completed_step, 3)
    gui.previous_step()
    self.assertEqual(gui.current_step_index, 2)
    gui.cancel_builder()
    self.assertEqual(gui.current_step_index, 0)
    self.assertEqual(gui.draft.completed_step, 0)


def _case_builder_next_requires_all_pages_selected(self) -> None:
    gui = CopernicanGUI(render=False)
    if gui.builder_steps.index("Engine") != 3:
        self.skipTest("Expected engine step at index 3")
    gui.current_step_index = gui.builder_steps.index("Engine")
    starting_alerts = len(gui.alerts)
    gui._handle_builder_next()
    self.assertEqual(gui.current_step_index, gui.builder_steps.index("Engine"))
    self.assertEqual(len(gui.alerts), starting_alerts + 1)
    self.assertIn("Seed", gui.alerts[-1].message)


def _case_builder_next_advances_when_pages_ready(
    self,
    tmp_path: Path | None = None,
) -> None:
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    gui.draft.seed = "42"
    self.assertFalse(gui._builder_ready())
    tmp_path = _tmp_path_or_default(tmp_path)
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("data", encoding="utf-8")
    gui.register_dataset(
        dataset_id="ds",
        path=str(dataset_path),
        name="Dataset",
    )
    self.assertTrue(gui._builder_ready())
    gui.current_step_index = gui.builder_steps.index("Engine")
    gui._handle_builder_next()
    self.assertEqual(
        gui.current_step_index, gui.builder_steps.index("Engine") + 1
    )


def _prepare_gui_with_dataset(tmp_path: Path) -> CopernicanGUI:
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    gui.draft.seed = "5"
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("data", encoding="utf-8")
    gui.register_dataset(
        dataset_id="ds",
        path=str(dataset_path),
        name="Dataset",
    )
    return gui


def _case_save_manifest_creates_new_config_folder(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = _prepare_gui_with_dataset(tmp_path)
    workspace = gui._persist_manifest_workspace()
    self.assertIsNotNone(workspace)
    self.assertEqual(workspace.folder.name, "copernican_run_NEW_CONFIG")
    self.assertEqual(
        workspace.manifest_path.name, "run_manifest_NEW_CONFIG.yml"
    )
    self.assertTrue(workspace.manifest_path.exists())


def _case_clear_manifest_resets_state(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = _prepare_gui_with_dataset(tmp_path)
    workspace = gui._persist_manifest_workspace()
    self.assertIsNotNone(workspace)
    folder = workspace.folder
    gui._clear_manifest_configuration()
    self.assertIsNone(gui.manifest_workspace)
    self.assertFalse(folder.exists())
    self.assertEqual(gui.current_step_index, 0)
    self.assertEqual(gui.selected_models, [])
    self.assertEqual(gui.selected_engine, "")


def _case_save_manifest_to_external(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = _prepare_gui_with_dataset(tmp_path)
    target = tmp_path / "external_manifest.yml"
    gui._save_manifest_to_external_folder(output_path=str(target))
    exported = list(tmp_path.glob("external_manifest.yml"))
    self.assertTrue(exported)


def _case_auto_loads_saved_temp_manifest(
    self,
    tmp_path: Path | None = None,
) -> None:
    old_cwd = os.getcwd()
    tmp_path = _tmp_path_or_default(tmp_path)
    output_dir = tmp_path / "output"
    workspace = output_dir / "copernican_run_NEW_CONFIG"
    workspace.mkdir(parents=True)
    engine_module = SimpleNamespace(
        __name__="cosmo_engine_mcmc",
        ENGINE_VERSION="test",
    )
    model_plugin = SimpleNamespace(
        MODEL_NAME="TestModel",
        MODEL_FILENAME="test_model.yml",
        PARAMETER_NAMES=[],
        PARAMETER_PRIORS=[],
        valid_for_cmb=False,
    )
    manifest = run_manifest.build_manifest(
        models=[(model_plugin, "gui")],
        engine_module=engine_module,
        datasets=[
            {
                "id": "planck2020",
                "name": "Planck 2020",
                "version": "1.0",
                "path": "",
                "hashes": {},
                "independence": [],
                "type": "cmb",
                "license": "unspecified",
                "badges": [],
            }
        ],
        state="pending",
        output_policy="unprepared",
        configuration={
            "models": ["TestModel"],
            "engine": {"name": "cosmo_engine_mcmc", "version": "test"},
            "datasets": ["planck2020"],
        },
    )
    manifest.setdefault("confirmation", {})["seed"] = 0
    manifest_path = workspace / "run_manifest_NEW_CONFIG.yml"
    run_manifest.save_manifest(
        manifest,
        workspace,
        target_path=manifest_path,
    )
    try:
        os.chdir(tmp_path)
        gui = CopernicanGUI(render=False)
        self.assertIsNotNone(gui.manifest_workspace)
        self.assertIsNotNone(gui.pending_manifest)
        self.assertEqual(
            gui.manifest_workspace.manifest_path.resolve(),
            manifest_path.resolve(),
        )
        self.assertTrue(
            gui.summary.manifest_actions[-1].startswith(
                "Loaded saved manifest"
            )
        )
    finally:
        os.chdir(old_cwd)


def _case_confirm_step_requires_saved_manifest(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = CopernicanGUI(render=False)
    confirm_index = gui.builder_steps.index("Confirm")
    initial_alerts = len(gui.alerts)
    gui.jump_to_step(confirm_index)
    self.assertNotEqual(gui.current_step_index, confirm_index)
    self.assertEqual(len(gui.alerts), initial_alerts + 1)
    self.assertEqual(gui.alerts[-1].message, gui._MANIFEST_REQUIRED_MESSAGE)
    gui = _prepare_gui_with_dataset(tmp_path)
    gui._persist_manifest_workspace()
    gui.jump_to_step(confirm_index)
    self.assertEqual(gui.current_step_index, confirm_index)


def _case_confirm_start_run_renames_manifest(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = _prepare_gui_with_dataset(tmp_path)
    _stub_worker_launch(gui)
    workspace = gui._persist_manifest_workspace()
    self.assertIsNotNone(workspace)
    old_folder = workspace.folder
    gui.confirm_start_run()
    self.assertIsNotNone(gui.manifest_workspace)
    self.assertTrue(
        gui.manifest_workspace.folder.name.startswith("copernican-run_")
    )
    self.assertTrue(
        gui.manifest_workspace.manifest_path.name.startswith("run_manifest_")
    )
    self.assertFalse(old_folder.exists())


def _case_cancel_inactive_without_configuration(self) -> None:
    gui = CopernicanGUI(render=False)
    self.assertFalse(gui._has_configuration())
    gui.selected_models = ["LambdaCDM"]
    self.assertTrue(gui._has_configuration())


def _case_run_monitor_lifecycle(self) -> None:
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    _stub_worker_launch(gui)
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"data")
        fh.flush()
        gui.register_dataset(dataset_id="ds", path=fh.name, name="Dataset")
    gui.draft.seed = "3"
    gui._persist_manifest_workspace()
    gui.confirm_start_run()
    self.assertTrue(gui.status is RunStatus.RUNNING)
    self.assertIsNotNone(gui.pending_manifest)
    self.assertIs(gui.output_directory_prepared, True)
    gui.update_progress(50)
    self.assertEqual(gui.progress, 50)
    gui.pause_run()
    self.assertTrue(gui.status is RunStatus.RUNNING)
    self.assertTrue(gui.alerts[-1].message.startswith("Pause/resume"))
    gui.cancel_run(disposition="archived")
    self.assertEqual(gui.pending_manifest["status"]["state"], "cancelled")
    gui.stop_run(disposition="deleted")
    self.assertTrue(gui.status is RunStatus.ABORTED)
    gui.update_progress(120)
    self.assertTrue(gui.status is RunStatus.IDLE)
    self.assertTrue(gui.summary.output_links)
    os.unlink(fh.name)


def _case_manifest_import_export_round_trip(self) -> None:
    gui = CopernicanGUI(render=False)
    gui.draft.model = "ModelB"
    gui.draft.dataset = "Dataset"
    gui.draft.engine = "engine"
    gui.draft.walkers = "33"
    gui.draft.burn_in = "20"
    gui.draft.production_steps = "100"
    gui.draft.pool_size = "4"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gui.export_manifest(tmpdir)
        loaded = run_manifest.load_manifest(path)
        self.assertTrue(loaded["selection"]["models"])
        imported = gui.import_manifest(path)
        self.assertTrue(imported["selection"]["engine"]["name"])
        self.assertTrue(gui.selected_models)
        self.assertEqual(gui.draft.walkers, "33")
        self.assertEqual(gui.draft.burn_in, "20")
        self.assertEqual(gui.draft.production_steps, "100")
        self.assertEqual(gui.draft.pool_size, "4")


def _case_insert_manifest_from_builder_prepares_pending_manifest(self) -> None:
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    gui.draft.walkers = "48"
    gui.insert_manifest_from_builder()
    self.assertIsNotNone(gui.pending_manifest)
    config = gui.pending_manifest.get("configuration", {})
    run_settings = config.get("run_settings", {})
    self.assertEqual(run_settings.get("n_walkers"), 48)
    self.assertTrue(gui.summary.manifest_metadata)


def _case_duplicate_manifest_prefills_builder(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
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
    self.assertIn("planck_2018_lite", gui.draft.dataset)
    self.assertTrue(gui.draft.plan.startswith("Duplicate & Edit"))


def _case_application_diagnostics_logging(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = CopernicanGUI(render=False)
    self.assertTrue(gui.application_log_path)
    self.assertTrue(os.path.exists(gui.application_log_path))
    gui.create_toast("App diagnostics ready", severity="INFO", context="app")
    gui.set_diagnostics_filter("ERROR")
    gui.create_toast("App failure", severity="ERROR", context="app")
    filtered = gui.get_application_log_entries()
    self.assertTrue(filtered)
    self.assertTrue(any(entry.severity == "ERROR" for entry in filtered))
    export_path = gui.export_application_logs(tmp_path)
    self.assertTrue(os.path.exists(export_path))


def _case_run_log_confirmation_and_anchor_jump(
    self,
    tmp_path: Path | None = None,
) -> None:
    tmp_path = _tmp_path_or_default(tmp_path)
    gui = CopernicanGUI(render=False)
    _prime_gui_selections(gui)
    _stub_worker_launch(gui)
    self.assertIsNone(gui.run_log_path)
    gui._persist_manifest_workspace()
    gui.confirm_start_run()
    self.assertIsNotNone(gui.run_log_path)
    self.assertTrue(os.path.exists(gui.run_log_path))
    entries = gui.get_run_log_entries()
    self.assertTrue(entries)
    self.assertTrue(entries[0].anchor.startswith("run-"))
    gui.start_run()
    gui.update_progress(100)
    exported = gui.export_run_logs(tmp_path)
    self.assertTrue(os.path.exists(exported))
    if gui.alerts:
        anchor = gui.alerts[-1].anchor
        snippet = gui.jump_to_log_anchor(anchor)
        self.assertIsNotNone(snippet)
