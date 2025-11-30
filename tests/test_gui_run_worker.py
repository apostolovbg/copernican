"""Tests for the GUI run worker that launches the CLI pipeline."""

import json
from pathlib import Path

import copernican

from copernican_lib.cli import menus as cli_menus
from copernican_lib import dataset_registry
from copernican_lib.gui import run_worker


def test_patch_cli_runtime_enforces_gui_config() -> None:
    config = {
        "seed": 11,
        "model_filename": "cosmo_model_lcdm.yml",
        "engine_filename": "cosmo_engine_mcmc.py",
        "datasets": {
            "sne": "jla_2014",
            "bao": "boss_dr12_bao",
        },
        "sampling_plan": {"n_steps": 10, "n_walkers": 4},
    }
    patches = run_worker._patch_cli_runtime(config)
    try:
        assert cli_menus.select_seed() == 11
        model_choice = cli_menus.select_from_list(
            ["cosmo_model_lcdm.yml"], "Select cosmological model"
        )
        assert model_choice == "cosmo_model_lcdm.yml"
        engine_choice = cli_menus.select_from_list(
            ["cosmo_engine_mcmc.py"], "Select computation engine"
        )
        assert engine_choice == "cosmo_engine_mcmc.py"
        sne_id = dataset_registry.prompt_dataset_selection(
            {"jla_2014": {"dataset_name": "JLA"}}, "SNe"
        )
        assert sne_id == "jla_2014"
        fallback = dataset_registry.prompt_dataset_selection(
            {"planck_2018_lite": {"dataset_name": "Planck"}}, "CMB"
        )
        assert fallback == "planck_2018_lite"
        plan = copernican.prompt_sampling_configuration()
        assert plan == config["sampling_plan"]
    finally:
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)


def test_worker_main_loads_config_and_invokes_cli(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "seed": 3,
        "model_filename": "cosmo_model_lcdm.yml",
        "engine_filename": "cosmo_engine_mcmc.py",
        "datasets": {},
        "sampling_plan": {"n_steps": 5, "n_walkers": 2},
    }
    config_path = tmp_path / "worker_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run_called = {"count": 0}

    def _fake_main_workflow() -> None:
        run_called["count"] += 1

    monkeypatch.setattr(copernican, "main_workflow", _fake_main_workflow)
    exit_code = run_worker.main([str(config_path)])
    assert exit_code == 0
    assert run_called["count"] == 1
