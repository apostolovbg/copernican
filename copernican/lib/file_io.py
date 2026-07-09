"""Retry local file operations that can transiently time out."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

_ResultT = TypeVar("_ResultT")
_TRANSIENT_TIMEOUT_ERRNOS = frozenset(
    {
        code
        for code in (getattr(errno, "ETIMEDOUT", None), 60)
        if code is not None
    }
)
_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4)


def is_transient_file_timeout(exc: BaseException) -> bool:
    """Return whether ``exc`` is a retryable local file timeout."""

    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_TIMEOUT_ERRNOS


def _run_with_retries(operation: Callable[[], _ResultT]) -> _ResultT:
    """Run ``operation`` again when local file I/O times out transiently."""

    last_attempt_index = len(_RETRY_DELAYS_SECONDS)
    for attempt_index in range(last_attempt_index + 1):
        try:
            return operation()
        except OSError as exc:
            if (
                not is_transient_file_timeout(exc)
                or attempt_index == last_attempt_index
            ):
                raise
            time.sleep(_RETRY_DELAYS_SECONDS[attempt_index])
    raise RuntimeError("unreachable retry loop exit")


def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read text from ``path`` with retries for transient timeouts."""

    resolved_path = Path(path)
    return _run_with_retries(
        lambda: resolved_path.read_text(encoding=encoding)
    )


def read_bytes(path: str | Path) -> bytes:
    """Read bytes from ``path`` with retries for transient timeouts."""

    resolved_path = Path(path)
    return _run_with_retries(resolved_path.read_bytes)


def _write_bytes_atomic(path: Path, payload: bytes) -> Path:
    """Persist ``payload`` to ``path`` through one atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)

    def _write_once() -> Path:
        """Write one temporary payload file and atomically replace ``path``."""

        file_descriptor, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except OSError:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        return path

    return _run_with_retries(_write_once)


def write_text(
    path: str | Path, content: str, *, encoding: str = "utf-8"
) -> Path:
    """Write ``content`` to ``path`` through an atomic replace."""

    return _write_bytes_atomic(Path(path), content.encode(encoding))


def write_bytes(
    path: str | Path,
    content: bytes | bytearray | memoryview,
) -> Path:
    """Write binary ``content`` to ``path`` through an atomic replace."""

    return _write_bytes_atomic(Path(path), bytes(content))


def read_json(path: str | Path, *, encoding: str = "utf-8") -> Any:
    """Load JSON content from ``path`` through the retrying text reader."""

    return json.loads(read_text(path, encoding=encoding))


def write_json(
    path: str | Path,
    content: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    trailing_newline: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """Serialize ``content`` as JSON and write it atomically."""

    payload = json.dumps(content, indent=indent, ensure_ascii=ensure_ascii)
    if trailing_newline:
        payload += "\n"
    return write_text(path, payload, encoding=encoding)


def read_yaml(path: str | Path, *, encoding: str = "utf-8") -> Any:
    """Load YAML content from ``path`` through the retrying text reader."""

    return yaml.safe_load(read_text(path, encoding=encoding))


def write_yaml(
    path: str | Path,
    content: Any,
    *,
    sort_keys: bool = True,
    allow_unicode: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Serialize ``content`` as YAML and write it atomically."""

    payload = yaml.safe_dump(
        content,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )
    return write_text(path, payload, encoding=encoding)


__all__ = [
    "is_transient_file_timeout",
    "read_bytes",
    "read_json",
    "read_text",
    "read_yaml",
    "write_bytes",
    "write_json",
    "write_text",
    "write_yaml",
]
