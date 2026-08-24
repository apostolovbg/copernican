"""Internal cache ownership for the declared-graph CMB path."""

from __future__ import annotations

import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field, fields
from typing import Any, Generic, Mapping, TypeVar

import numpy

from copernican.lib.cmb_identity import CCMBS_ID

_CacheValue = TypeVar("_CacheValue")


@dataclass(frozen=True, slots=True)
class _CacheSnapshot:
    """One immutable snapshot of one bounded declared cache."""

    entries: int
    limit: int
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True, slots=True)
class RuntimeCacheIdentity:
    """Name the static and request-specific portions of a runtime cache key."""

    contract_static: Any
    model_static: Any
    request_specific: Any
    execution_solver: str = field(default=CCMBS_ID, init=False)


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

        snapshot = _CacheSnapshot(
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
_CMB_RUNTIME_ASSET_CACHE = _BoundedCacheStore(limit=32)
_DECLARED_MOMENTUM_TOPOLOGY_CACHE = _BoundedCacheStore(limit=128)
_DECLARED_MOMENTUM_GRID_CACHE = _BoundedCacheStore(limit=128)
_CMB_BACKGROUND_CACHE = _BoundedCacheStore(limit=64)
_CMB_REIONIZATION_CALIBRATION_SEED_CACHE = _BoundedCacheStore(limit=128)
_CMB_SPECTRUM_CACHE = _BoundedCacheStore(limit=64)
_CMB_TRANSFER_CACHE = _BoundedCacheStore(
    limit=16,
    max_bytes=64 * 1024 * 1024,
)
_CMB_SOURCE_HISTORY_CACHE = _BoundedCacheStore(
    limit=128,
    max_bytes=128 * 1024 * 1024,
)
_CMB_BESSEL_INPUT_CACHE = _BoundedCacheStore(limit=512)
_CMB_BESSEL_VALUE_CACHE = _BoundedCacheStore(limit=4096)
_CMB_BESSEL_BATCH_CACHE = _BoundedCacheStore(
    limit=512,
    max_bytes=32 * 1024 * 1024,
)
_PROJECTION_BATCH_CACHE_MAX_BYTES = 8 * 1024 * 1024
_CMB_PERFORMANCE_PHASE_SECONDS: dict[str, float] = {}
_CMB_PERFORMANCE_REQUESTS = 0
_CMB_PERFORMANCE_CACHE_HITS = 0
_CMB_PERFORMANCE_FAILURES = 0
_CMB_PERFORMANCE_RECORDS: deque[dict[str, Any]] = deque(maxlen=256)
_LAST_CMB_REQUEST_IDENTITY: RuntimeCacheIdentity | None = None


def _numpy_payload_bytes(value: Any) -> int:
    """Return the array storage held by one cache value when measurable."""

    if isinstance(value, numpy.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_numpy_payload_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_numpy_payload_bytes(item) for item in value)
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


def get_runtime_assets(cache_key: Any):
    """Return one process-local immutable runtime asset bundle."""

    return _CMB_RUNTIME_ASSET_CACHE.get(cache_key)


def set_runtime_assets(cache_key: Any, runtime_assets: Any) -> None:
    """Store one process-local immutable runtime asset bundle."""

    _CMB_RUNTIME_ASSET_CACHE.set(cache_key, runtime_assets)


def get_declared_momentum_topology(cache_key: Any):
    """Return one parameter-independent momentum quadrature topology."""

    return _DECLARED_MOMENTUM_TOPOLOGY_CACHE.get(cache_key)


def set_declared_momentum_topology(cache_key: Any, topology: Any) -> None:
    """Store one parameter-independent momentum quadrature topology."""

    _DECLARED_MOMENTUM_TOPOLOGY_CACHE.set(cache_key, topology)


def get_declared_momentum_grid(cache_key: Any):
    """Return one cached momentum-grid runtime bundle when present."""

    return _DECLARED_MOMENTUM_GRID_CACHE.get(cache_key)


def set_declared_momentum_grid(cache_key: Any, runtime_bundle: Any) -> None:
    """Store one prepared momentum-grid runtime bundle."""

    _DECLARED_MOMENTUM_GRID_CACHE.set(cache_key, runtime_bundle)


def get_cmb_background(cache_key: Any):
    """Return one cached declared background payload when present."""

    return _CMB_BACKGROUND_CACHE.get(cache_key)


def set_cmb_background(cache_key: Any, background_data: Any) -> None:
    """Store one declared background payload."""

    _CMB_BACKGROUND_CACHE.set(cache_key, background_data)


def get_reionization_calibration_seed(cache_key: Any) -> float | None:
    """Return one bounded warm-start seed for optical-depth calibration."""

    cached = _CMB_REIONIZATION_CALIBRATION_SEED_CACHE.get(cache_key)
    return None if cached is None else float(cached)


def set_reionization_calibration_seed(
    cache_key: Any,
    calibration_value: float,
) -> None:
    """Store a finite calibration seed without treating it as a result."""

    value = float(calibration_value)
    if not numpy.isfinite(value):
        raise ValueError("Reionization calibration seed must be finite")
    _CMB_REIONIZATION_CALIBRATION_SEED_CACHE.set(cache_key, value)


def get_cmb_spectrum(cache_key: Any):
    """Return one cached declared spectrum payload when present."""

    cached = _CMB_SPECTRUM_CACHE.get(cache_key)
    if cached is not None:
        remember_cmb_request_identity(cache_key)
    return cached


def set_cmb_spectrum(cache_key: Any, spectrum_data: Any) -> None:
    """Store one declared spectrum payload."""

    _CMB_SPECTRUM_CACHE.set(cache_key, spectrum_data)
    remember_cmb_request_identity(cache_key)


def get_cmb_transfer(cache_key: Any):
    """Return one cached transfer-product payload when present."""

    return _CMB_TRANSFER_CACHE.get(cache_key)


def set_cmb_transfer(cache_key: Any, transfer_data: Any) -> None:
    """Store one bounded transfer-product payload."""

    _CMB_TRANSFER_CACHE.set(cache_key, transfer_data)


def get_cmb_source_history(cache_key: Any):
    """Return source arrays cached for one exact parameter/grid identity."""

    return _CMB_SOURCE_HISTORY_CACHE.get(cache_key)


def set_cmb_source_history(cache_key: Any, source_data: Any) -> None:
    """Store immutable source arrays for one exact parameter/grid identity."""

    _CMB_SOURCE_HISTORY_CACHE.set(cache_key, source_data)


def remember_cmb_request_identity(cache_key: Any) -> None:
    """Record the most recent declared spectrum request shape."""

    global _LAST_CMB_REQUEST_IDENTITY
    if not isinstance(cache_key, RuntimeCacheIdentity):
        return
    _LAST_CMB_REQUEST_IDENTITY = cache_key


def latest_cmb_request_identity() -> RuntimeCacheIdentity | None:
    """Return the most recent declared spectrum request identity."""

    return _LAST_CMB_REQUEST_IDENTITY


def get_bessel_inputs(x_signature: str) -> numpy.ndarray | None:
    """Return one cached x-grid keyed by its stable signature."""

    return _CMB_BESSEL_INPUT_CACHE.get(x_signature)


def store_bessel_inputs(x_signature: str, x_values: numpy.ndarray) -> None:
    """Store one x-grid and prune dependent Bessel caches on eviction."""

    before_keys = set(_CMB_BESSEL_INPUT_CACHE._store)
    _CMB_BESSEL_INPUT_CACHE.set(x_signature, x_values)
    after_keys = set(_CMB_BESSEL_INPUT_CACHE._store)
    evicted_signatures = before_keys - after_keys
    for stale_signature in evicted_signatures:
        _CMB_BESSEL_VALUE_CACHE.prune(
            lambda key, stale_signature=stale_signature: (
                key[1] == stale_signature
            )
        )
        _CMB_BESSEL_BATCH_CACHE.prune(
            lambda key, stale_signature=stale_signature: (
                key[1] == stale_signature
            )
        )


def get_bessel_values(cache_key: Any):
    """Return one cached spherical-Bessel pair when present."""

    return _CMB_BESSEL_VALUE_CACHE.get(cache_key)


def set_bessel_values(cache_key: Any, values: Any) -> None:
    """Store one spherical-Bessel pair."""

    _CMB_BESSEL_VALUE_CACHE.set(cache_key, values)


def get_declared_projection_kernel_batch(cache_key: Any):
    """Return one cached ell-batched kernel pack when present."""

    return _CMB_BESSEL_BATCH_CACHE.get(cache_key)


def set_declared_projection_kernel_batch(cache_key: Any, batch: Any) -> None:
    """Store one ell-batched kernel pack."""

    if _numpy_payload_bytes(batch) > _PROJECTION_BATCH_CACHE_MAX_BYTES:
        return
    _CMB_BESSEL_BATCH_CACHE.set(cache_key, batch)


def record_cmb_performance(
    phase_seconds: Mapping[str, float],
    *,
    cache_hit: bool = False,
    workload: str = "full_spectrum",
    cache_state: str | None = None,
    outcome: str = "success",
    stop_phase: str | None = None,
    work_units: Mapping[str, int] | None = None,
    failure: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Record one declared request's phase timings and cache outcome."""

    global _CMB_PERFORMANCE_REQUESTS
    global _CMB_PERFORMANCE_CACHE_HITS
    global _CMB_PERFORMANCE_FAILURES
    _CMB_PERFORMANCE_REQUESTS += 1
    if cache_hit:
        _CMB_PERFORMANCE_CACHE_HITS += 1
    normalized_outcome = str(outcome).strip().lower()
    if normalized_outcome not in {"success", "failure"}:
        raise ValueError(
            "Declared performance outcome must be success or failure"
        )
    if normalized_outcome == "failure":
        _CMB_PERFORMANCE_FAILURES += 1
    normalized_cache_state = (
        "exact_cache_hit" if cache_hit else str(cache_state or "cold")
    )
    if normalized_cache_state not in {"cold", "warm", "exact_cache_hit"}:
        raise ValueError("Declared performance cache state is invalid")
    normalized_phases: dict[str, float] = {}
    for phase_name, elapsed in phase_seconds.items():
        value = float(elapsed)
        if value < 0.0 or value != value:
            raise ValueError("Declared performance phase time must be finite")
        name = str(phase_name)
        normalized_phases[name] = value
        _CMB_PERFORMANCE_PHASE_SECONDS[name] = (
            _CMB_PERFORMANCE_PHASE_SECONDS.get(name, 0.0) + value
        )
    record = {
        "request_index": int(_CMB_PERFORMANCE_REQUESTS),
        "workload": str(workload),
        "cache_state": normalized_cache_state,
        "outcome": normalized_outcome,
        "stop_phase": None if stop_phase is None else str(stop_phase),
        "phase_seconds": normalized_phases,
        "work_units": {
            str(name): int(value) for name, value in (work_units or {}).items()
        },
        "failure": None if failure is None else dict(failure),
        "context": dict(context or {}),
    }
    _CMB_PERFORMANCE_RECORDS.append(record)
    return dict(record)


def record_cmb_phase(name: str, elapsed_seconds: float) -> None:
    """Record one declared phase measured outside spectrum projection."""

    value = float(elapsed_seconds)
    if value < 0.0 or value != value:
        raise ValueError("Declared performance phase time must be finite")
    phase_name = str(name)
    _CMB_PERFORMANCE_PHASE_SECONDS[phase_name] = (
        _CMB_PERFORMANCE_PHASE_SECONDS.get(phase_name, 0.0) + value
    )


def extend_latest_cmb_request_phase(
    name: str,
    elapsed_seconds: float,
) -> None:
    """Append one post-projection phase to the latest local request."""

    record_cmb_phase(name, elapsed_seconds)
    if not _CMB_PERFORMANCE_RECORDS:
        return
    value = float(elapsed_seconds)
    latest = _CMB_PERFORMANCE_RECORDS[-1]
    phases = latest["phase_seconds"]
    phase_name = str(name)
    phases[phase_name] = float(phases.get(phase_name, 0.0)) + value
    phases["total_seconds"] = float(phases.get("total_seconds", 0.0)) + value


def fail_latest_cmb_request(
    failure: Mapping[str, Any],
    *,
    stop_phase: str | None = None,
) -> None:
    """Mark the latest local request as failed after output assembly."""

    global _CMB_PERFORMANCE_FAILURES
    if not _CMB_PERFORMANCE_RECORDS:
        return
    latest = _CMB_PERFORMANCE_RECORDS[-1]
    if latest["outcome"] != "failure":
        _CMB_PERFORMANCE_FAILURES += 1
    latest["outcome"] = "failure"
    latest["failure"] = dict(failure)
    if stop_phase is not None:
        latest["stop_phase"] = str(stop_phase)


def latest_cmb_performance_record() -> Mapping[str, Any] | None:
    """Return the latest process-local declared request record."""

    if not _CMB_PERFORMANCE_RECORDS:
        return None
    latest = _CMB_PERFORMANCE_RECORDS[-1]
    return {
        **latest,
        "phase_seconds": dict(latest["phase_seconds"]),
        "work_units": dict(latest["work_units"]),
        "context": dict(latest["context"]),
        "failure": (
            None if latest["failure"] is None else dict(latest["failure"])
        ),
    }


def cmb_performance_quantiles(
    *,
    workload: str,
    cache_state: str,
) -> dict[str, float | int | str]:
    """Return deterministic success-time quantiles for one workload state."""

    normalized_workload = str(workload)
    normalized_cache_state = str(cache_state).strip().lower()
    if normalized_cache_state not in {"cold", "warm", "exact_cache_hit"}:
        raise ValueError("Declared performance cache state is invalid")
    elapsed_samples = []
    for record in _CMB_PERFORMANCE_RECORDS:
        if record["outcome"] != "success":
            continue
        if record["workload"] != normalized_workload:
            continue
        if record["cache_state"] != normalized_cache_state:
            continue
        phase_seconds = record["phase_seconds"]
        elapsed_samples.append(
            float(
                phase_seconds.get(
                    "total_seconds",
                    sum(float(value) for value in phase_seconds.values()),
                )
            )
        )
    if not elapsed_samples:
        raise ValueError(
            "No successful declared performance records match the requested "
            "workload and cache state"
        )
    values = numpy.asarray(elapsed_samples, dtype=float)
    quantiles = numpy.quantile(values, (0.5, 0.95), method="linear")
    return {
        "workload": normalized_workload,
        "cache_state": normalized_cache_state,
        "sample_count": int(values.size),
        "median_seconds": float(quantiles[0]),
        "p95_seconds": float(quantiles[1]),
        "minimum_seconds": float(numpy.min(values)),
        "maximum_seconds": float(numpy.max(values)),
    }


def cmb_performance_stats() -> dict[str, Any]:
    """Return aggregate phase timings and declared cache-hit accounting."""

    return {
        "requests": int(_CMB_PERFORMANCE_REQUESTS),
        "cache_hits": int(_CMB_PERFORMANCE_CACHE_HITS),
        "failures": int(_CMB_PERFORMANCE_FAILURES),
        "phase_seconds": dict(_CMB_PERFORMANCE_PHASE_SECONDS),
        "recent_requests": tuple(
            {
                **record,
                "phase_seconds": dict(record["phase_seconds"]),
                "work_units": dict(record["work_units"]),
                "context": dict(record["context"]),
                "failure": (
                    None
                    if record["failure"] is None
                    else dict(record["failure"])
                ),
            }
            for record in _CMB_PERFORMANCE_RECORDS
        ),
    }


def clear_cmb_result_caches() -> None:
    """Clear complete declared spectrum results without dropping structure."""

    _CMB_SPECTRUM_CACHE.clear()


def clear_cmb_parameter_caches() -> None:
    """Clear parameter-bound data while retaining structural compilation."""

    for cache in (
        _DECLARED_MOMENTUM_GRID_CACHE,
        _CMB_BACKGROUND_CACHE,
        _CMB_REIONIZATION_CALIBRATION_SEED_CACHE,
        _CMB_SPECTRUM_CACHE,
        _CMB_TRANSFER_CACHE,
        _CMB_SOURCE_HISTORY_CACHE,
        _CMB_BESSEL_INPUT_CACHE,
        _CMB_BESSEL_VALUE_CACHE,
        _CMB_BESSEL_BATCH_CACHE,
    ):
        cache.clear()


def clear_cmb_caches() -> None:
    """Clear every bounded cache used by the declared CMB path."""

    global _CMB_PERFORMANCE_REQUESTS
    global _CMB_PERFORMANCE_CACHE_HITS
    global _CMB_PERFORMANCE_FAILURES
    global _LAST_CMB_REQUEST_IDENTITY

    for cache in (
        _DECLARED_SYMBOL_PLAN_CACHE,
        _DECLARED_GRAPH_EXECUTION_PLAN_CACHE,
        _CMB_RUNTIME_ASSET_CACHE,
        _DECLARED_MOMENTUM_TOPOLOGY_CACHE,
        _DECLARED_MOMENTUM_GRID_CACHE,
        _CMB_BACKGROUND_CACHE,
        _CMB_REIONIZATION_CALIBRATION_SEED_CACHE,
        _CMB_SPECTRUM_CACHE,
        _CMB_TRANSFER_CACHE,
        _CMB_SOURCE_HISTORY_CACHE,
        _CMB_BESSEL_INPUT_CACHE,
        _CMB_BESSEL_VALUE_CACHE,
        _CMB_BESSEL_BATCH_CACHE,
    ):
        cache.clear()
    _CMB_PERFORMANCE_PHASE_SECONDS.clear()
    _CMB_PERFORMANCE_RECORDS.clear()
    _CMB_PERFORMANCE_REQUESTS = 0
    _CMB_PERFORMANCE_CACHE_HITS = 0
    _CMB_PERFORMANCE_FAILURES = 0
    _LAST_CMB_REQUEST_IDENTITY = None


def cmb_cache_stats() -> dict[str, dict[str, int]]:
    """Return entry, limit, and hit/miss counters for declared CMB caches."""

    return {
        "declared_symbol_plan": _DECLARED_SYMBOL_PLAN_CACHE.snapshot(),
        "declared_graph_execution_plan": (
            _DECLARED_GRAPH_EXECUTION_PLAN_CACHE.snapshot()
        ),
        "runtime_assets": _CMB_RUNTIME_ASSET_CACHE.snapshot(),
        "declared_momentum_topology": (
            _DECLARED_MOMENTUM_TOPOLOGY_CACHE.snapshot()
        ),
        "declared_momentum_grid": (_DECLARED_MOMENTUM_GRID_CACHE.snapshot()),
        "background": _CMB_BACKGROUND_CACHE.snapshot(),
        "reionization_calibration_seed": (
            _CMB_REIONIZATION_CALIBRATION_SEED_CACHE.snapshot()
        ),
        "declared_spectrum": _CMB_SPECTRUM_CACHE.snapshot(),
        "declared_transfer": _CMB_TRANSFER_CACHE.snapshot(),
        "source_history": _CMB_SOURCE_HISTORY_CACHE.snapshot(),
        "bessel_inputs": _CMB_BESSEL_INPUT_CACHE.snapshot(),
        "bessel_values": _CMB_BESSEL_VALUE_CACHE.snapshot(),
        "declared_projection_kernel_batch": (
            _CMB_BESSEL_BATCH_CACHE.snapshot()
        ),
    }


def cmb_cache_inventory() -> dict[str, Mapping[str, Any]]:
    """Describe cache ownership and invalidation class for diagnostics."""

    categories = {
        "declared_symbol_plan": "structural",
        "declared_graph_execution_plan": "structural",
        "runtime_assets": "structural",
        "declared_momentum_topology": "structural",
        "declared_momentum_grid": "parameter",
        "background": "parameter",
        "reionization_calibration_seed": "parameter",
        "bessel_inputs": "parameter",
        "bessel_values": "parameter",
        "declared_projection_kernel_batch": "parameter",
        "declared_spectrum": "result",
        "declared_transfer": "parameter",
        "source_history": "parameter",
    }
    snapshots = cmb_cache_stats()
    return {
        name: {
            "category": categories[name],
            "owner_pid": os.getpid(),
            "bounded": True,
            "limit": int(snapshot["limit"]),
        }
        for name, snapshot in snapshots.items()
    }
