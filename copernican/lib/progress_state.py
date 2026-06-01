"""Utilities to share sampler progress between CLI runs and the GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def record_progress(path: str | Path, payload: Mapping[str, object]) -> None:
    """Persist ``payload`` to ``path`` so external monitors can read it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(target)


def load_progress(path: str | Path) -> dict | None:
    """Return the latest progress payload stored at ``path``."""

    target = Path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clear_progress(path: str | Path) -> None:
    """Remove the stored progress payload at ``path`` if present."""

    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
