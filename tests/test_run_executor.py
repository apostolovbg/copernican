from pathlib import Path
from types import SimpleNamespace

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

    pipeline_calls = []

    def fake_pipeline(**kwargs):
        pipeline_calls.append(kwargs)

    monkeypatch.setattr(
        run_executor.run_pipeline,
        "execute_run_pipeline",
        fake_pipeline,
    )

    monkeypatch.setattr(
        run_executor,
        "_build_plugin_from_path",
        lambda path: SimpleNamespace(
            MODEL_NAME=path.stem,
            MODEL_FILENAME=path.name,
        ),
    )

    run_executor._PLUGIN_CACHE.clear()
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
    assert pipeline_calls
    sampling_plan = pipeline_calls[0]["sampling_plan"]
    assert sampling_plan["engine_kind"] == "mcmc"
    assert pipeline_calls[0]["display_progress"]
    assert pipeline_calls[0]["output_dir"] == str(tmp_path)


def test_execute_run_from_manifest_persists_manifest(monkeypatch, tmp_path):
    manifest = {
        "seed": 999,
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
        },
        "configuration": {"run_settings": {"engine_kind": "mcmc"}},
    }

    def fake_dataset(*_, **__):
        return FakeDataset()

    monkeypatch.setattr(
        run_executor.dataset_registry,
        "load_sne_data",
        lambda **kwargs: fake_dataset(),
    )
    monkeypatch.setattr(
        run_executor.dataset_registry,
        "load_bao_data",
        lambda **kwargs: fake_dataset(),
    )
    monkeypatch.setattr(
        run_executor.run_pipeline,
        "execute_run_pipeline",
        lambda **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        run_executor,
        "_build_plugin_from_path",
        lambda path: SimpleNamespace(
            MODEL_NAME=path.stem,
            MODEL_FILENAME=path.name,
        ),
    )
    monkeypatch.setattr(
        run_executor.utils,
        "get_timestamp",
        lambda: "20250101_000000",
    )

    run_executor._PLUGIN_CACHE.clear()
    run_executor.execute_run_from_manifest(
        manifest,
        script_dir=Path("."),
        output_root=tmp_path,
    )
    manifest_path = tmp_path / "run_manifest_20250101_000000.yml"
    assert manifest_path.exists()
    content = manifest_path.read_text(encoding="utf-8")
    assert "seed: 999" in content


def test_execute_run_from_manifest_sets_seed(monkeypatch, tmp_path):
    manifest = {
        "seed": 444,
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
        },
        "configuration": {"run_settings": {"engine_kind": "mcmc"}},
    }

    def fake_dataset(*_, **__):
        return FakeDataset()

    monkeypatch.setattr(
        run_executor.dataset_registry,
        "load_sne_data",
        lambda **kwargs: fake_dataset(),
    )
    monkeypatch.setattr(
        run_executor.dataset_registry,
        "load_bao_data",
        lambda **kwargs: fake_dataset(),
    )
    monkeypatch.setattr(
        run_executor.run_pipeline,
        "execute_run_pipeline",
        lambda **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        run_executor,
        "_build_plugin_from_path",
        lambda path: SimpleNamespace(
            MODEL_NAME=path.stem,
            MODEL_FILENAME=path.name,
        ),
    )
    seed_calls: list[int] = []

    def fake_set_random_seed(value):
        seed_calls.append(value)

    monkeypatch.setattr(
        run_executor.utils,
        "set_random_seed",
        fake_set_random_seed,
    )

    run_executor._PLUGIN_CACHE.clear()
    run_executor.execute_run_from_manifest(
        manifest,
        script_dir=Path("."),
        output_root=tmp_path,
    )
    assert seed_calls == [444]
