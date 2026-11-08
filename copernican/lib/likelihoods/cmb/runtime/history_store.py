"""Retain exact source histories for one request and its refinements.

Memory pressure spills raw arrays to request-local files. Numerical bounds
and convergence certificates are never stored or borrowed by this owner.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any, Iterator, Mapping
from zipfile import BadZipFile

import numpy

_NATIVE_HISTORY_RESOURCES: ContextVar[list | None] = ContextVar(
    "native_history_resources", default=None
)
NATIVE_HISTORY_MEMORY_CELLS = 16_000_000


def with_native_history_storage(function):
    """Own full-state buffers through projection and error cleanup."""

    @wraps(function)
    def execute(*args, **kwargs):
        """Release every mapping after consumers have copied their results."""

        if _NATIVE_HISTORY_RESOURCES.get() is not None:
            return function(*args, **kwargs)
        resources = []
        token = _NATIVE_HISTORY_RESOURCES.set(resources)
        try:
            return function(*args, **kwargs)
        finally:
            for array, stream in reversed(resources):
                array._mmap.close()
                stream.close()
            _NATIVE_HISTORY_RESOURCES.reset(token)

    return execute


def allocate_native_history(shape: tuple[int, ...]) -> numpy.ndarray:
    """Store every native state, mapping large histories instead of splitting.

    Only the saved trajectory is file backed. Live RK states, equations,
    stages, substeps and boundary/collision callbacks are unchanged. The
    operating system's file cache is separate from this ndarray budget.
    """

    resources = _NATIVE_HISTORY_RESOURCES.get()
    if resources is None or numpy.prod(shape) <= NATIVE_HISTORY_MEMORY_CELLS:
        return numpy.empty(shape, dtype=float)
    stream = TemporaryFile(prefix="ccmbs-native-")
    try:
        history = numpy.memmap(stream, dtype=float, mode="w+", shape=shape)
    except (OSError, ValueError):
        stream.close()
        raise
    resources.append((history, stream))
    return history


@dataclass(frozen=True)
class _SpilledSourceHistory:
    """Describe an integrity-checked array bundle owned by one request."""

    path: Path
    digest: str
    names: tuple[str, ...]


class RequestSourceHistoryStore:
    """Own compatible raw histories until all nested refinements finish."""

    def __init__(self, *, max_bytes: int):
        """Use a memory budget without imposing a numerical work limit."""

        if int(max_bytes) < 0:
            raise ValueError(
                "Source-history memory budget must be nonnegative"
            )
        self.max_bytes = int(max_bytes)
        self._entries: dict[Any, Any] = {}
        self._sizes: dict[Any, int] = {}
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._directory: TemporaryDirectory | None = None
        self._sequence = 0
        self.hits = 0
        self.misses = 0
        self.spills = 0
        self.loads = 0
        self.invalidations = 0

    def set(self, key: Any, sources: Mapping[str, numpy.ndarray]) -> None:
        """Retain an exact source bundle, spilling older bundles if needed."""

        arrays = {
            str(name): numpy.array(values, dtype=float, copy=True)
            for name, values in sources.items()
        }
        if not arrays or any(
            values.ndim != 1
            or values.size == 0
            or not numpy.all(numpy.isfinite(values))
            for values in arrays.values()
        ):
            raise ValueError(
                "Retained source histories must be finite vectors"
            )
        if len({values.shape for values in arrays.values()}) != 1:
            raise ValueError("Retained source histories must share a grid")
        for values in arrays.values():
            values.setflags(write=False)
        previous = self._entries.pop(key, None)
        if isinstance(previous, _SpilledSourceHistory):
            previous.path.unlink(missing_ok=True)
        self._resident_bytes -= self._sizes.pop(key, 0)
        self._entries[key] = arrays
        size = sum(int(values.nbytes) for values in arrays.values())
        self._sizes[key] = size
        self._resident_bytes += size
        self._peak_resident_bytes = max(
            self._peak_resident_bytes, self._resident_bytes
        )
        while self._resident_bytes > self.max_bytes:
            resident_key = next(iter(self._sizes))
            resident = self._entries[resident_key]
            if self._directory is None:
                self._directory = TemporaryDirectory(prefix="ccmbs-history-")
            path = Path(self._directory.name) / f"{self._sequence}.npz"
            self._sequence += 1
            names = tuple(resident)
            # Archive keys are controlled positional names; declared source
            # names never become paths or arguments to the archive writer.
            numpy.savez(path, *(resident[name] for name in names))
            self._entries[resident_key] = _SpilledSourceHistory(
                path, hashlib.sha256(path.read_bytes()).hexdigest(), names
            )
            self._resident_bytes -= self._sizes.pop(resident_key)
            self.spills += 1

    def get(self, key: Any) -> Mapping[str, numpy.ndarray] | None:
        """Read an exact bundle; corrupt spill contents cause a cache miss."""

        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if isinstance(entry, _SpilledSourceHistory):
            try:
                if hashlib.sha256(entry.path.read_bytes()).hexdigest() != (
                    entry.digest
                ):
                    raise ValueError("Retained source-history digest changed")
                with numpy.load(entry.path, allow_pickle=False) as archive:
                    expected = {
                        f"arr_{index}" for index in range(len(entry.names))
                    }
                    if set(archive.files) != expected:
                        raise ValueError(
                            "Retained source-history fields changed"
                        )
                    sources = {
                        name: numpy.asarray(
                            archive[f"arr_{index}"], dtype=float
                        )
                        for index, name in enumerate(entry.names)
                    }
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                entry.path.unlink(missing_ok=True)
                self._entries.pop(key, None)
                self.invalidations += 1
                self.misses += 1
                return None
            self.loads += 1
        else:
            sources = entry
        for values in sources.values():
            values.setflags(write=False)
        self.hits += 1
        return dict(sources)

    def snapshot(self) -> dict[str, int]:
        """Expose retained work and memory accounting without local paths."""

        return {
            "entries": len(self._entries),
            "resident_bytes": self._resident_bytes,
            "peak_resident_bytes": self._peak_resident_bytes,
            "memory_budget_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "spills": self.spills,
            "loads": self.loads,
            "invalidations": self.invalidations,
        }

    def clear(self) -> None:
        """Release all retained arrays and request-local spill files."""

        self._entries.clear()
        self._sizes.clear()
        self._resident_bytes = 0
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None


_ACTIVE_SOURCE_HISTORY_STORE: ContextVar[RequestSourceHistoryStore | None] = (
    ContextVar("ccmbs_request_source_history_store", default=None)
)


def active_source_history_store() -> RequestSourceHistoryStore | None:
    """Return this execution context's request-owned raw history store."""

    return _ACTIVE_SOURCE_HISTORY_STORE.get()


@contextmanager
def retain_source_histories(
    *, max_bytes: int
) -> Iterator[RequestSourceHistoryStore]:
    """Retain nested refinement work and always release it at request exit."""

    current = active_source_history_store()
    if current is not None:
        yield current
        return
    store = RequestSourceHistoryStore(max_bytes=max_bytes)
    token = _ACTIVE_SOURCE_HISTORY_STORE.set(store)
    try:
        yield store
    finally:
        _ACTIVE_SOURCE_HISTORY_STORE.reset(token)
        store.clear()
