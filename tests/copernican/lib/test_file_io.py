"""Tests for retrying local file I/O helpers."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from copernican.lib import file_io


class FileIoTestCase(unittest.TestCase):
    """Exercise retry behavior for transient local file timeouts."""

    def test_read_text_retries_transient_timeout(self) -> None:
        """The helper should retry a timed-out read before failing."""

        with (
            mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=[
                    TimeoutError(errno.ETIMEDOUT, "Operation timed out"),
                    "resolved",
                ],
            ) as patched_read,
            mock.patch(
                "copernican.lib.file_io.time.sleep",
                autospec=True,
            ) as patched_sleep,
        ):
            result = file_io.read_text("README.md")

        self.assertEqual(result, "resolved")
        self.assertEqual(patched_read.call_count, 2)
        patched_sleep.assert_called_once()

    def test_write_text_retries_transient_replace_failure(self) -> None:
        """The helper should retry atomic replaces that time out once."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.txt"
            original_replace = os.replace
            replace_calls = {"count": 0}

            def _flaky_replace(src: str | Path, dst: str | Path) -> None:
                if replace_calls["count"] == 0:
                    replace_calls["count"] += 1
                    raise TimeoutError(
                        errno.ETIMEDOUT,
                        "Operation timed out",
                    )
                original_replace(src, dst)

            with (
                mock.patch(
                    "copernican.lib.file_io.os.replace",
                    side_effect=_flaky_replace,
                ),
                mock.patch(
                    "copernican.lib.file_io.time.sleep",
                    autospec=True,
                ) as patched_sleep,
            ):
                file_io.write_text(output_path, "stable", encoding="utf-8")

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "stable",
            )
            patched_sleep.assert_called_once()

    def test_timeout_classifier_matches_only_retryable_timeouts(self) -> None:
        """Only timeout-shaped local file errors should be classified."""

        timeout_error = TimeoutError(errno.ETIMEDOUT, "Operation timed out")
        timed_os_error = OSError(errno.ETIMEDOUT, "Operation timed out")
        missing_file_error = FileNotFoundError(errno.ENOENT, "missing")

        self.assertTrue(file_io.is_transient_file_timeout(timeout_error))
        self.assertTrue(file_io.is_transient_file_timeout(timed_os_error))
        self.assertFalse(file_io.is_transient_file_timeout(missing_file_error))

    def test_read_helpers_cover_bytes_json_and_yaml(self) -> None:
        """Structured read helpers should decode persisted content."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bytes_path = tmp_path / "payload.bin"
            json_path = tmp_path / "payload.json"
            yaml_path = tmp_path / "payload.yml"

            bytes_path.write_bytes(b"abc")
            json_path.write_text('{"alpha": 1}\n', encoding="utf-8")
            yaml_path.write_text("alpha: 1\nbeta:\n  - 2\n", encoding="utf-8")

            self.assertEqual(file_io.read_bytes(bytes_path), b"abc")
            self.assertEqual(file_io.read_json(json_path), {"alpha": 1})
            self.assertEqual(
                file_io.read_yaml(yaml_path),
                {"alpha": 1, "beta": [2]},
            )

    def test_write_helpers_cover_bytes_json_and_yaml(self) -> None:
        """Structured write helpers should persist retrievable content."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bytes_path = tmp_path / "payload.bin"
            json_path = tmp_path / "payload.json"
            yaml_path = tmp_path / "payload.yml"

            self.assertEqual(
                file_io.write_bytes(bytes_path, b"xyz"),
                bytes_path,
            )
            self.assertEqual(
                file_io.write_json(json_path, {"beta": 2}),
                json_path,
            )
            self.assertEqual(
                file_io.write_yaml(
                    yaml_path,
                    {"gamma": ["delta"]},
                    sort_keys=False,
                    allow_unicode=True,
                ),
                yaml_path,
            )
            self.assertEqual(bytes_path.read_bytes(), b"xyz")
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"beta": 2},
            )
            self.assertEqual(
                yaml.safe_load(yaml_path.read_text(encoding="utf-8")),
                {"gamma": ["delta"]},
            )
