import pytest

from copernican_lib.run_config import (
    DatasetDescriptor,
    build_config_from_manifest,
)


@pytest.fixture
def simple_manifest():
    return {
        "seed": 42,
        "selection": {
            "models": ["LambdaCDM"],
            "engine": {
                "name": "engines.cosmo_engine_mcmc",
                "version": "7.6.20",
            },
        },
        "datasets": {
            "sne/pantheon": {
                "name": "Pantheon SNe",
                "type": "sne",
                "version": "1.0",
                "path": "/data/sne/pantheon",
                "hashes": {"data.csv": "abc123"},
                "independence": ["sne"],
            }
        },
        "configuration": {
            "run_settings": {"engine_kind": "mcmc", "n_steps": 200}
        },
    }


def test_build_config_from_manifest(simple_manifest):
    config = build_config_from_manifest(simple_manifest)
    assert config.seed == 42
    assert config.models == ["LambdaCDM"]
    assert config.engine.module_name == "engines.cosmo_engine_mcmc"
    assert config.engine.version == "7.6.20"
    assert config.run_settings.engine_kind == "mcmc"
    assert config.run_settings.settings["n_steps"] == 200
    assert len(config.datasets) == 1
    descriptor = config.datasets[0]
    assert isinstance(descriptor, DatasetDescriptor)
    assert descriptor.dataset_id == "sne/pantheon"
    assert descriptor.dataset_type == "sne"
    assert descriptor.version == "1.0"
