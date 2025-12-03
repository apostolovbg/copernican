"""Tests for the lightweight CLI helper utilities exposed in copernican."""

import os
import time

os.environ.setdefault("COPERNICAN_ALLOW_DIRECT", "1")

import copernican  # noqa: E402  pylint: disable=wrong-import-position


def test_catalogue_summary_reports_counts():
    summary = copernican._gather_catalogue_summary()  # pylint: disable=protected-access
    assert summary["dataset_count"] > 0
    assert summary["type_counter"]


def test_model_engine_summary_reports_counts():
    stats = copernican._gather_model_engine_summary()  # pylint: disable=protected-access
    assert stats["model_count"] > 0
    assert stats["engine_count"] > 0


def test_manifest_discovery_sorts_by_mtime(tmp_path):
    first = tmp_path / "copernican-run_20240101_000000"
    second = tmp_path / "copernican-run_20240102_000000"
    first.mkdir()
    second.mkdir()
    manifest_first = first / "run_manifest_20240101.yml"
    manifest_second = second / "run_manifest_20240102.yml"
    manifest_first.write_text("seed: 0\n", encoding="utf-8")
    manifest_second.write_text("seed: 1\n", encoding="utf-8")
    os.utime(manifest_first, (time.time() - 100, time.time() - 100))
    os.utime(manifest_second, (time.time(), time.time()))
    records = copernican._discover_manifest_records(  # pylint: disable=protected-access
        tmp_path
    )
    assert [directory.name for directory, _ in records] == [
        second.name,
        first.name,
    ]


def test_cli_revalidate_dataset_reports_missing():
    assert copernican._cli_revalidate_dataset("missing-dataset") is False  # pylint: disable=protected-access


def test_cli_revalidate_dataset_known_dataset():
    assert copernican._cli_revalidate_dataset(  # pylint: disable=protected-access
        "planck_2018_lite"
    )
