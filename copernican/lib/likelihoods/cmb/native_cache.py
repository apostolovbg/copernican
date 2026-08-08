"""Internal cache ownership for the native declared-graph CMB path."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any, Generic, Mapping, TypeVar

import numpy

from copernican.lib.cmb_identity import NATIVE_CMB_ENGINE_ID

_CacheValue = TypeVar("_CacheValue")


@dataclass(frozen=True, slots=True)
class _NativeCacheSnapshot:
    """One immutable snapshot of one bounded native cache."""

    entries: int
    limit: int
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True, slots=True)
class NativeRuntimeCacheIdentity:
    """Name the static and request-specific portions of a runtime cache key."""

    contract_static: Any
    cosmology_static: Any
    request_specific: Any
    execution_engine: str = field(default=NATIVE_CMB_ENGINE_ID, init=False)


class _BoundedCacheStore(Generic[_CacheValue]):
    """Keep a bounded LRU cache with explicit lifecycle accounting."""

    def __init__(self, *, limit: int, max_bytes: int | None = None) -> None:
        """Initialize one bounded cache with the requested entry limit."""

        self.limit = int(limit)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._store: OrderedDict[Any, _CacheValue] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any) -> _CacheValue | None:
        """Return the cached value for ``key`` and update hit counters."""

        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: Any, value: _CacheValue) -> None:
        """Store ``value`` under ``key`` and evict stale entries if needed."""

        value_bytes = _numpy_payload_bytes(value)
        if self.max_bytes is not None and value_bytes > self.max_bytes:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._bytes -= _numpy_payload_bytes(self._store[key])
        self._store[key] = value
        self._bytes += value_bytes
        while len(self._store) > self.limit or (
            self.max_bytes is not None and self._bytes > self.max_bytes
        ):
            _, stale_value = self._store.popitem(last=False)
            self._bytes -= _numpy_payload_bytes(stale_value)
            self.evictions += 1

    def clear(self) -> None:
        """Remove every cached entry and reset cache accounting counters."""

        self._store.clear()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def snapshot(self) -> dict[str, int]:
        """Return entry counts and accounting counters for this cache."""

        snapshot = _NativeCacheSnapshot(
            entries=len(self._store),
            limit=self.limit,
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
        )
        return {
            "entries": snapshot.entries,
            "limit": snapshot.limit,
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "evictions": snapshot.evictions,
        }

    def prune(self, predicate) -> None:
        """Drop cached keys that match the supplied eviction predicate."""

        stale_keys = [key for key in self._store if predicate(key)]
        for key in stale_keys:
            stale_value = self._store.pop(key, None)
            if stale_value is not None:
                self._bytes -= _numpy_payload_bytes(stale_value)


_DECLARED_SYMBOL_PLAN_CACHE = _BoundedCacheStore(limit=256)
_DECLARED_GRAPH_EXECUTION_PLAN_CACHE = _BoundedCacheStore(limit=256)
_DECLARED_MOMENTUM_GRID_CACHE = _BoundedCacheStore(limit=128)
_NATIVE_CMB_BACKGROUND_CACHE = _BoundedCacheStore(limit=64)
_NATIVE_CMB_SPECTRUM_CACHE = _BoundedCacheStore(limit=64)
_NATIVE_CMB_BESSEL_INPUT_CACHE = _BoundedCacheStore(limit=512)
_NATIVE_CMB_BESSEL_VALUE_CACHE = _BoundedCacheStore(limit=4096)
_NATIVE_CMB_BESSEL_BATCH_CACHE = _BoundedCacheStore(
    limit=512,
    max_bytes=32 * 1024 * 1024,
)
_PROJECTION_BATCH_CACHE_MAX_BYTES = 8 * 1024 * 1024
_NATIVE_PERFORMANCE_PHASE_SECONDS: dict[str, float] = {}
_NATIVE_PERFORMANCE_REQUESTS = 0
_NATIVE_PERFORMANCE_CACHE_HITS = 0


def _numpy_payload_bytes(value: Any) -> int:
    """Return the array storage held by one cache value when measurable."""

    if isinstance(value, numpy.ndarray):
        return int(value.nbytes)
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if dataclass_fields is None:
        return 0
    return sum(
        _numpy_payload_bytes(getattr(value, field.name))
        for field in fields(value)
    )


def get_declared_symbol_plan(cache_key: Any):
    """Return one cached declared-symbol evaluation plan when present."""

    return _DECLARED_SYMBOL_PLAN_CACHE.get(cache_key)


def set_declared_symbol_plan(cache_key: Any, compiled_plan: Any) -> None:
    """Store one compiled declared-symbol evaluation plan."""

    _DECLARED_SYMBOL_PLAN_CACHE.set(cache_key, compiled_plan)


def get_declared_graph_execution_plan(cache_key: Any):
    """Return one cached declared-graph execution plan when present."""

    return _DECLARED_GRAPH_EXECUTION_PLAN_CACHE.get(cache_key)


def set_declared_graph_execution_plan(
    cache_key: Any,
    compiled_plan: Any,
) -> None:
    """Store one declared-graph execution plan."""

    _DECLARED_GRAPH_EXECUTION_PLAN_CACHE.set(cache_key, compiled_plan)


def get_declared_momentum_grid(cache_key: Any):
    """Return one cached momentum-grid runtime bundle when present."""

    return _DECLARED_MOMENTUM_GRID_CACHE.get(cache_key)


def set_declared_momentum_grid(cache_key: Any, runtime_bundle: Any) -> None:
    """Store one prepared momentum-grid runtime bundle."""

    _DECLARED_MOMENTUM_GRID_CACHE.set(cache_key, runtime_bundle)


def get_native_cmb_background(cache_key: Any):
    """Return one cached native background payload when present."""

    return _NATIVE_CMB_BACKGROUND_CACHE.get(cache_key)


def set_native_cmb_background(cache_key: Any, background_data: Any) -> None:
    """Store one native background payload."""

    _NATIVE_CMB_BACKGROUND_CACHE.set(cache_key, background_data)


def get_native_cmb_spectrum(cache_key: Any):
    """Return one cached native spectrum payload when present."""

    return _NATIVE_CMB_SPECTRUM_CACHE.get(cache_key)


def set_native_cmb_spectrum(cache_key: Any, spectrum_data: Any) -> None:
    """Store one native spectrum payload."""

    _NATIVE_CMB_SPECTRUM_CACHE.set(cache_key, spectrum_data)


def get_bessel_inputs(x_signature: str) -> numpy.ndarray | None:
    """Return one cached x-grid keyed by its stable signature."""

    return _NATIVE_CMB_BESSEL_INPUT_CACHE.get(x_signature)


def store_bessel_inputs(x_signature: str, x_values: numpy.ndarray) -> None:
    """Store one x-grid and prune dependent Bessel caches on eviction."""

    before_keys = set(_NATIVE_CMB_BESSEL_INPUT_CACHE._store)
    _NATIVE_CMB_BESSEL_INPUT_CACHE.set(x_signature, x_values)
    after_keys = set(_NATIVE_CMB_BESSEL_INPUT_CACHE._store)
    evicted_signatures = before_keys - after_keys
    for stale_signature in evicted_signatures:
        _NATIVE_CMB_BESSEL_VALUE_CACHE.prune(
            lambda key, stale_signature=stale_signature: (
                key[1] == stale_signature
            )
        )
        _NATIVE_CMB_BESSEL_BATCH_CACHE.prune(
            lambda key, stale_signature=stale_signature: (
                key[1] == stale_signature
            )
        )


def get_bessel_values(cache_key: Any):
    """Return one cached spherical-Bessel pair when present."""

    return _NATIVE_CMB_BESSEL_VALUE_CACHE.get(cache_key)


def set_bessel_values(cache_key: Any, values: Any) -> None:
    """Store one spherical-Bessel pair."""

    _NATIVE_CMB_BESSEL_VALUE_CACHE.set(cache_key, values)


def get_declared_projection_kernel_batch(cache_key: Any):
    """Return one cached ell-batched kernel pack when present."""

    return _NATIVE_CMB_BESSEL_BATCH_CACHE.get(cache_key)


def set_declared_projection_kernel_batch(cache_key: Any, batch: Any) -> None:
    """Store one ell-batched kernel pack."""

    if _numpy_payload_bytes(batch) > _PROJECTION_BATCH_CACHE_MAX_BYTES:
        return
    _NATIVE_CMB_BESSEL_BATCH_CACHE.set(cache_key, batch)


def record_native_cmb_performance(
    phase_seconds: Mapping[str, float],
    *,
    cache_hit: bool = False,
) -> None:
    """Record one native request's phase timings and cache outcome."""

    global _NATIVE_PERFORMANCE_REQUESTS
    global _NATIVE_PERFORMANCE_CACHE_HITS
    _NATIVE_PERFORMANCE_REQUESTS += 1
    if cache_hit:
        _NATIVE_PERFORMANCE_CACHE_HITS += 1
    for phase_name, elapsed in phase_seconds.items():
        value = float(elapsed)
        if value < 0.0 or value != value:
            raise ValueError("Native performance phase time must be finite")
        _NATIVE_PERFORMANCE_PHASE_SECONDS[str(phase_name)] = (
            _NATIVE_PERFORMANCE_PHASE_SECONDS.get(str(phase_name), 0.0) + value
        )


def record_native_cmb_phase(name: str, elapsed_seconds: float) -> None:
    """Record one native phase measured outside spectrum projection."""

    value = float(elapsed_seconds)
    if value < 0.0 or value != value:
        raise ValueError("Native performance phase time must be finite")
    phase_name = str(name)
    _NATIVE_PERFORMANCE_PHASE_SECONDS[phase_name] = (
        _NATIVE_PERFORMANCE_PHASE_SECONDS.get(phase_name, 0.0) + value
    )


def native_cmb_performance_stats() -> dict[str, Any]:
    """Return aggregate phase timings and native cache-hit accounting."""

    return {
        "requests": int(_NATIVE_PERFORMANCE_REQUESTS),
        "cache_hits": int(_NATIVE_PERFORMANCE_CACHE_HITS),
        "phase_seconds": dict(_NATIVE_PERFORMANCE_PHASE_SECONDS),
    }


def clear_native_cmb_caches() -> None:
    """Clear every bounded cache used by the native declared CMB path."""

    global _NATIVE_PERFORMANCE_REQUESTS
    global _NATIVE_PERFORMANCE_CACHE_HITS

    for cache in (
        _DECLARED_SYMBOL_PLAN_CACHE,
        _DECLARED_GRAPH_EXECUTION_PLAN_CACHE,
        _DECLARED_MOMENTUM_GRID_CACHE,
        _NATIVE_CMB_BACKGROUND_CACHE,
        _NATIVE_CMB_SPECTRUM_CACHE,
        _NATIVE_CMB_BESSEL_INPUT_CACHE,
        _NATIVE_CMB_BESSEL_VALUE_CACHE,
        _NATIVE_CMB_BESSEL_BATCH_CACHE,
    ):
        cache.clear()
    _NATIVE_PERFORMANCE_PHASE_SECONDS.clear()
    _NATIVE_PERFORMANCE_REQUESTS = 0
    _NATIVE_PERFORMANCE_CACHE_HITS = 0


def native_cmb_cache_stats() -> dict[str, dict[str, int]]:
    """Return entry, limit, and hit/miss counters for native CMB caches."""

    return {
        "declared_symbol_plan": _DECLARED_SYMBOL_PLAN_CACHE.snapshot(),
        "declared_graph_execution_plan": (
            _DECLARED_GRAPH_EXECUTION_PLAN_CACHE.snapshot()
        ),
        "declared_momentum_grid": (_DECLARED_MOMENTUM_GRID_CACHE.snapshot()),
        "native_background": _NATIVE_CMB_BACKGROUND_CACHE.snapshot(),
        "native_spectrum": _NATIVE_CMB_SPECTRUM_CACHE.snapshot(),
        "bessel_inputs": _NATIVE_CMB_BESSEL_INPUT_CACHE.snapshot(),
        "bessel_values": _NATIVE_CMB_BESSEL_VALUE_CACHE.snapshot(),
        "declared_projection_kernel_batch": (
            _NATIVE_CMB_BESSEL_BATCH_CACHE.snapshot()
        ),
    }
