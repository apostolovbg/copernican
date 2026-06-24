"""Internal cache ownership for the native declared-graph CMB path."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy

_CacheValue = TypeVar("_CacheValue")


@dataclass(frozen=True, slots=True)
class _NativeCacheSnapshot:
    """One immutable snapshot of one bounded native cache."""

    entries: int
    limit: int
    hits: int
    misses: int
    evictions: int


class _BoundedCacheStore(Generic[_CacheValue]):
    """Keep a bounded LRU cache with explicit lifecycle accounting."""

    def __init__(self, *, limit: int) -> None:
        """Initialize one bounded cache with the requested entry limit."""

        self.limit = int(limit)
        self._store: OrderedDict[Any, _CacheValue] = OrderedDict()
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

        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self.limit:
            self._store.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        """Remove every cached entry and reset cache accounting counters."""

        self._store.clear()
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
            self._store.pop(key, None)


_DECLARED_SYMBOL_PLAN_CACHE = _BoundedCacheStore(limit=256)
_DECLARED_GRAPH_EXECUTION_PLAN_CACHE = _BoundedCacheStore(limit=256)
_DECLARED_MOMENTUM_GRID_CACHE = _BoundedCacheStore(limit=128)
_CUSTOM_CMB_BACKGROUND_CACHE = _BoundedCacheStore(limit=64)
_CUSTOM_CMB_SPECTRUM_CACHE = _BoundedCacheStore(limit=64)
_CUSTOM_CMB_BESSEL_INPUT_CACHE = _BoundedCacheStore(limit=512)
_CUSTOM_CMB_BESSEL_VALUE_CACHE = _BoundedCacheStore(limit=4096)
_CUSTOM_CMB_BESSEL_BATCH_CACHE = _BoundedCacheStore(limit=512)


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


def get_custom_cmb_background(cache_key: Any):
    """Return one cached native background payload when present."""

    return _CUSTOM_CMB_BACKGROUND_CACHE.get(cache_key)


def set_custom_cmb_background(cache_key: Any, background_data: Any) -> None:
    """Store one native background payload."""

    _CUSTOM_CMB_BACKGROUND_CACHE.set(cache_key, background_data)


def get_custom_cmb_spectrum(cache_key: Any):
    """Return one cached native spectrum payload when present."""

    return _CUSTOM_CMB_SPECTRUM_CACHE.get(cache_key)


def set_custom_cmb_spectrum(cache_key: Any, spectrum_data: Any) -> None:
    """Store one native spectrum payload."""

    _CUSTOM_CMB_SPECTRUM_CACHE.set(cache_key, spectrum_data)


def get_bessel_inputs(x_signature: str) -> numpy.ndarray | None:
    """Return one cached x-grid keyed by its stable signature."""

    return _CUSTOM_CMB_BESSEL_INPUT_CACHE.get(x_signature)


def store_bessel_inputs(x_signature: str, x_values: numpy.ndarray) -> None:
    """Store one x-grid and prune dependent Bessel caches on eviction."""

    before_keys = set(_CUSTOM_CMB_BESSEL_INPUT_CACHE._store)
    _CUSTOM_CMB_BESSEL_INPUT_CACHE.set(x_signature, x_values)
    after_keys = set(_CUSTOM_CMB_BESSEL_INPUT_CACHE._store)
    evicted_signatures = before_keys - after_keys
    for stale_signature in evicted_signatures:
        _CUSTOM_CMB_BESSEL_VALUE_CACHE.prune(
            lambda key, stale_signature=stale_signature: (
                key[1] == stale_signature
            )
        )
        _CUSTOM_CMB_BESSEL_BATCH_CACHE.prune(
            lambda key, stale_signature=stale_signature: (
                key[1] == stale_signature
            )
        )


def get_bessel_values(cache_key: Any):
    """Return one cached spherical-Bessel pair when present."""

    return _CUSTOM_CMB_BESSEL_VALUE_CACHE.get(cache_key)


def set_bessel_values(cache_key: Any, values: Any) -> None:
    """Store one spherical-Bessel pair."""

    _CUSTOM_CMB_BESSEL_VALUE_CACHE.set(cache_key, values)


def get_declared_projection_kernel_batch(cache_key: Any):
    """Return one cached ell-batched kernel pack when present."""

    return _CUSTOM_CMB_BESSEL_BATCH_CACHE.get(cache_key)


def set_declared_projection_kernel_batch(cache_key: Any, batch: Any) -> None:
    """Store one ell-batched kernel pack."""

    _CUSTOM_CMB_BESSEL_BATCH_CACHE.set(cache_key, batch)


def clear_native_cmb_caches() -> None:
    """Clear every bounded cache used by the native declared CMB path."""

    for cache in (
        _DECLARED_SYMBOL_PLAN_CACHE,
        _DECLARED_GRAPH_EXECUTION_PLAN_CACHE,
        _DECLARED_MOMENTUM_GRID_CACHE,
        _CUSTOM_CMB_BACKGROUND_CACHE,
        _CUSTOM_CMB_SPECTRUM_CACHE,
        _CUSTOM_CMB_BESSEL_INPUT_CACHE,
        _CUSTOM_CMB_BESSEL_VALUE_CACHE,
        _CUSTOM_CMB_BESSEL_BATCH_CACHE,
    ):
        cache.clear()


def native_cmb_cache_stats() -> dict[str, dict[str, int]]:
    """Return entry, limit, and hit/miss counters for native CMB caches."""

    return {
        "declared_symbol_plan": _DECLARED_SYMBOL_PLAN_CACHE.snapshot(),
        "declared_graph_execution_plan": (
            _DECLARED_GRAPH_EXECUTION_PLAN_CACHE.snapshot()
        ),
        "declared_momentum_grid": (_DECLARED_MOMENTUM_GRID_CACHE.snapshot()),
        "custom_background": _CUSTOM_CMB_BACKGROUND_CACHE.snapshot(),
        "custom_spectrum": _CUSTOM_CMB_SPECTRUM_CACHE.snapshot(),
        "bessel_inputs": _CUSTOM_CMB_BESSEL_INPUT_CACHE.snapshot(),
        "bessel_values": _CUSTOM_CMB_BESSEL_VALUE_CACHE.snapshot(),
        "declared_projection_kernel_batch": (
            _CUSTOM_CMB_BESSEL_BATCH_CACHE.snapshot()
        ),
    }
