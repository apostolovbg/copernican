from pathlib import Path

import pytest

from copernican_lib import run_executor


class FakeDataset:
    def __len__(self):
        return 5


@pytest.fixture(autouse=True)
def stub_console_output(monkeypatch):
    monkeypatch.setattr(run_executor.console_output, "write", lambda *_: None)


def test_execute_run_from_manifest_loads_datasets(monkeypatch, tmp_path):
    manifest = {
        "seed": 123,
        "selection": {
            "models": ["LambdaCDM"],
            "engine": {
                "name": "engines.cosmo_engine_mcmc",
                "version": "7.6.20",
            },
        },
        "datasets": {
            "sne/pantheon": {
                "name": "Pantheon",
                "type": "sne",
                "version": "1.0",
                "path": "/tmp",
            },
            "bao/bossdr12": {
                "name": "BOSS DR12",
                "type": "bao",
                "version": "1.0",
                "path": "/tmp",
            },
        },
        "configuration": {"run_settings": {"engine_kind": "mcmc"}},
    }
    loaded = []

    def fake_loader(*_, **__):
        loaded.append(True)
        return FakeDataset()

    monkeypatch.setattr(
        run_executor.dataset_registry,
        "load_sne_data",
        lambda **kwargs: fake_loader(),
    )
    monkeypatch.setattr(
        run_executor.dataset_registry,
        "load_bao_data",
        lambda **kwargs: fake_loader(),
    )
    progress_records = []

    def progress_callback(record):
        progress_records.append(record)

    run_executor.execute_run_from_manifest(
        manifest,
        script_dir=Path("."),
        output_root=tmp_path,
        progress_callback=progress_callback,
    )
    assert len(loaded) == 2
    assert (
        progress_records
        and progress_records[0]["status"] == "manifest_execution_started"
    )
