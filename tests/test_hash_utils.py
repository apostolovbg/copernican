"""Tests for :mod:`copernican_lib.hash_utils`.

The tests patch network access so no real HTTP requests are performed."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from copernican_lib import hash_utils


def test_update_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure missing hashes are inserted and existing ones kept."""
    req = tmp_path / "req.lock"
    req.write_text("pkg==1.0 \\\n    --hash=sha256:aaa\n")

    cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    sample = {
        "urls": [
            {
                "filename": f"pkg-1.0-{cp_tag}-{cp_tag}-win_amd64.whl",
                "digests": {"sha256": "aaa"},
            },
            {
                "filename": f"pkg-1.0-{cp_tag}-{cp_tag}-macosx_11_0_arm64.whl",
                "digests": {"sha256": "bbb"},
            },
        ]
    }

    class Dummy:
        def __enter__(self) -> io.BytesIO:  # pragma: no cover - trivial
            return io.BytesIO(json.dumps(sample).encode())

        def __exit__(self, *exc: object) -> None:  # pragma: no cover - trivial
            pass

    monkeypatch.setattr(hash_utils, "urlopen", lambda req: Dummy())

    changed = hash_utils.update_hashes(req, ["pkg"])
    assert changed
    text = req.read_text().splitlines()
    assert text[1].endswith("\\")
    assert text[2] == "    --hash=sha256:bbb"


def test_universal2_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recognise macOS ``universal2`` wheels."""
    req = tmp_path / "req.lock"
    req.write_text("pkg==1.0 \\\n" "    --hash=sha256:aaa\n")

    cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    sample = {
        "urls": [
            {
                "filename": f"pkg-1.0-{cp_tag}-{cp_tag}-win_amd64.whl",
                "digests": {"sha256": "aaa"},
            },
            {
                "filename": (
                    f"pkg-1.0-{cp_tag}-{cp_tag}-macosx_10_13_universal2.whl"
                ),
                "digests": {"sha256": "ccc"},
            },
        ]
    }

    class Dummy:
        def __enter__(self) -> io.BytesIO:  # pragma: no cover - trivial
            return io.BytesIO(json.dumps(sample).encode())

        def __exit__(self, *exc: object) -> None:  # pragma: no cover - trivial
            pass

    monkeypatch.setattr(hash_utils, "urlopen", lambda req: Dummy())

    changed = hash_utils.update_hashes(req, ["pkg"])
    assert changed
    text = req.read_text().splitlines()
    assert text[1].endswith("\\")
    assert text[2] == "    --hash=sha256:ccc"


def test_abi3_wheels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pick up wheels built against Python's stable ``abi3`` interface."""

    req = tmp_path / "req.lock"
    req.write_text("pkg==1.0 \\\n" "    --hash=sha256:aaa\n")

    sample = {
        "urls": [
            {
                "filename": "pkg-1.0-cp39-abi3-macosx_11_0_arm64.whl",
                "digests": {"sha256": "ddd"},
            },
            {
                "filename": "pkg-1.0-cp39-abi3-win_amd64.whl",
                "digests": {"sha256": "eee"},
            },
        ]
    }

    class Dummy:
        def __enter__(self) -> io.BytesIO:  # pragma: no cover - trivial
            return io.BytesIO(json.dumps(sample).encode())

        def __exit__(self, *exc: object) -> None:  # pragma: no cover - trivial
            pass

    monkeypatch.setattr(hash_utils, "urlopen", lambda req: Dummy())

    changed = hash_utils.update_hashes(req, ["pkg"])
    assert changed
    text = req.read_text().splitlines()
    assert text[1].endswith("\\")
    hashes = {line.strip(" \\") for line in text[1:3]}
    assert hashes == {"--hash=sha256:ddd", "--hash=sha256:eee"}


def test_interpreter_specific_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handle wheels built for the running interpreter."""

    req = tmp_path / "req.lock"
    req.write_text("pkg==1.0 \\\n" "    --hash=sha256:aaa\n")

    cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    sample = {
        "urls": [
            {
                "filename": (
                    f"pkg-1.0-{cp_tag}-{cp_tag}-manylinux_2_17_x86_64.whl"
                ),
                "digests": {"sha256": "fff"},
            }
        ]
    }

    class Dummy:
        def __enter__(self) -> io.BytesIO:  # pragma: no cover - trivial
            return io.BytesIO(json.dumps(sample).encode())

        def __exit__(self, *exc: object) -> None:  # pragma: no cover - trivial
            pass

    monkeypatch.setattr(hash_utils, "urlopen", lambda req: Dummy())

    changed = hash_utils.update_hashes(req, ["pkg"])
    assert changed
    text = req.read_text().splitlines()
    assert text[0].endswith("\\")
    assert text[1] == "    --hash=sha256:fff"
