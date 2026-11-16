"""Durable, identity-bound raw histories for completion attempts.

Only completed numerical arrays are reusable. Convergence assessments and
certificates are always recomputed by the current solver.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy

from . import evidence_assets

_ACTIVE = ContextVar("ccmbs_durable_histories", default=None)


def _add_key_component(digest, value):
    """Include type and length so distinct values cannot concatenate."""

    if isinstance(value, numpy.generic):
        value = value.item()
    if value is None:
        tag, payload = b"n", b""
    elif isinstance(value, bool):
        tag, payload = b"b", bytes([value])
    elif isinstance(value, int):
        tag, payload = b"i", str(value).encode("ascii")
    elif isinstance(value, float):
        tag, payload = b"f", struct.pack("!d", value)
    elif isinstance(value, str):
        tag, payload = b"s", value.encode("utf-8")
    elif isinstance(value, bytes):
        tag, payload = b"y", value
    elif isinstance(value, tuple):
        digest.update(b"t" + struct.pack("!Q", len(value)))
        for item in value:
            _add_key_component(digest, item)
        return
    else:
        raise TypeError("unsupported durable history identity value")
    digest.update(tag + struct.pack("!Q", len(payload)) + payload)


def _key_digest(key):
    """Hash supported exact cache-key values with unambiguous type tags."""

    digest = hashlib.sha256()
    _add_key_component(digest, key)
    return digest.hexdigest()


def _visit_product(digest, value):
    """Hash every typed array, mapping and sequence without coercing data."""

    if isinstance(value, numpy.ndarray):
        if value.dtype.kind not in "biufc":
            raise ValueError("unsupported durable product array")
        _add_key_component(digest, ("array", value.dtype.str, value.shape))
        digest.update(numpy.ascontiguousarray(value).tobytes())
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("durable product keys must be strings")
        _add_key_component(digest, ("mapping", len(value)))
        for key in sorted(value):
            _add_key_component(digest, key)
            _visit_product(digest, value[key])
    elif isinstance(value, (tuple, list)):
        _add_key_component(digest, ("sequence", len(value)))
        for item in value:
            _visit_product(digest, item)
    else:
        _add_key_component(digest, value)


def product_fingerprint(product):
    """Identify actual arrays and metadata used to produce a calculation."""

    digest = hashlib.sha256()
    _visit_product(digest, product)
    return digest.hexdigest()


class DurableHistories:
    """Publish completed history bundles under exact producer identities."""

    def __init__(self, root, identity, producer):
        """Bind one attempt to its physics, numerical plan and provenance."""

        self.root = Path(root)
        self.identity = dict(identity)
        if set(self.identity) != {"physical", "numerical"} or any(
            not isinstance(value, str) or len(value) != 64
            for value in self.identity.values()
        ):
            raise ValueError("durable history identity is incomplete")
        self.producer = str(producer)
        self.scope = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.hits = 0
        self.writes = 0
        self.product_hits = 0
        self.product_writes = 0
        self._verified_histories = {}

    def _stem(self, key):
        """Keep filenames independent of user declarations and local paths."""

        return f"resume/{self.scope}/{_key_digest(key)}"

    def get_product(self, key):
        """Read a linked product tree under its exact dependency identity."""

        stem = self._stem(key)
        pointer = evidence_assets.contained_path(
            self.root, stem + ".product-index.json"
        )
        if not pointer.exists():
            return None
        index = json.loads(pointer.read_text(encoding="utf-8"))
        key_digest = _key_digest(key)
        if (
            index.get("schema_version") != 1
            or index.get("identity") != self.identity
            or index.get("key_sha256") != key_digest
        ):
            raise ValueError("durable product identity differs")
        payload = evidence_assets.read_array_products(
            self.root, index["artifact"], materialize=True
        )
        if (
            not isinstance(payload, dict)
            or payload.get("identity") != self.identity
            or payload.get("key_sha256") != key_digest
            or payload.get("producer") != index.get("producer")
            or "product" not in payload
        ):
            raise ValueError("durable product link differs")
        self.product_hits += 1
        return payload["product"]

    def set_product(self, key, product):
        """Atomically publish a completed tree with immutable provenance."""

        hits = self.product_hits
        existing = self.get_product(key)
        self.product_hits = hits
        if existing is not None:
            return
        stem = self._stem(key)
        key_digest = _key_digest(key)
        artifact = evidence_assets.write_array_products(
            self.root,
            stem + ".product-" + uuid.uuid4().hex,
            {
                "identity": self.identity,
                "key_sha256": key_digest,
                "producer": self.producer,
                "product": product,
            },
        )
        evidence_assets.write_json_artifact(
            self.root,
            stem + ".product-index.json",
            {
                "schema_version": 1,
                "identity": self.identity,
                "key_sha256": key_digest,
                "producer": self.producer,
                "artifact": artifact,
            },
        )
        self.product_writes += 1

    def get(self, key):
        """Verify the linked artifact before returning actual raw arrays."""

        stem = self._stem(key)
        pointer = evidence_assets.contained_path(
            self.root, stem + ".index.json"
        )
        if not pointer.exists():
            return None
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("identity") != self.identity
            or payload.get("key_sha256") != _key_digest(key)
        ):
            raise ValueError("durable history identity differs")
        retained = evidence_assets.read_array_products(
            self.root, payload["artifact"], materialize=True
        )
        if (
            not isinstance(retained, dict)
            or retained.get("identity") != self.identity
            or retained.get("key_sha256") != _key_digest(key)
            or retained.get("producer") != payload.get("producer")
        ):
            raise ValueError("durable history link differs")
        products = retained.get("histories")
        if (
            not isinstance(products, dict)
            or not products
            or any(
                not isinstance(name, str)
                or not isinstance(values, numpy.ndarray)
                or values.ndim != 1
                or not values.size
                or not numpy.all(numpy.isfinite(values))
                for name, values in products.items()
            )
            or len({values.shape for values in products.values()}) != 1
        ):
            raise ValueError("durable history payload is invalid")
        if (
            isinstance(key, tuple)
            and key
            and key[0] == "native_evolution_history"
        ):
            if (
                "eta" not in products
                or len(products) < 2
                or any(
                    name != "eta" and not name.startswith("state:")
                    for name in products
                )
                or not numpy.all(numpy.diff(products["eta"]) > 0)
            ):
                raise ValueError("durable native history is invalid")
        self.hits += 1
        self._verified_histories[_key_digest(key)] = key
        return products

    def verified_history_token(self, key):
        """Return a token only for a history validated in this attempt."""

        token = _key_digest(key)
        return token if token in self._verified_histories else None

    def verifies_history_wave(self, token, wave):
        """Bind an observed wave to its verified raw-history identity."""

        key = self._verified_histories.get(token)
        return bool(isinstance(key, tuple) and key and key[-1] == float(wave))

    def set(self, key, products):
        """Publish arrays before an atomic index with original provenance."""

        stem = self._stem(key)
        hits = self.hits
        existing = self.get(key)
        self.hits = hits
        if (
            existing is not None
            and set(existing) == set(products)
            and all(
                numpy.array_equal(existing[name], products[name])
                for name in existing
            )
        ):
            self._verified_histories[_key_digest(key)] = key
            return
        content = hashlib.sha256()
        for name, values in sorted(products.items()):
            array = numpy.asarray(values)
            content.update(repr((name, array.dtype.str, array.shape)).encode())
            content.update(array.tobytes())
        # The product manifest is immutable too: an interruption while
        # publishing a replacement must not invalidate the previous index.
        product_stem = stem + ".product-" + content.hexdigest()
        artifact = evidence_assets.write_array_products(
            self.root,
            product_stem,
            {
                "identity": self.identity,
                "key_sha256": _key_digest(key),
                "producer": self.producer,
                "histories": products,
            },
        )
        evidence_assets.write_json_artifact(
            self.root,
            stem + ".index.json",
            {
                "schema_version": 1,
                "identity": self.identity,
                "key_sha256": _key_digest(key),
                "producer": self.producer,
                "artifact": artifact,
            },
        )
        self.writes += 1
        self._verified_histories[_key_digest(key)] = key


class RKCheckpoint:
    """Persist completed RK intervals as immutable, identity-bound chunks."""

    def __init__(
        self, durable, key, *, interval_seconds=30.0, stats_fields=None
    ):
        """Bind a grouped trajectory to its exact schedule and producer."""

        self.durable = durable
        self.key = key
        self.stats_fields = frozenset(
            stats_fields
            if stats_fields is not None
            else (
                "rk_stage_count",
                "total_substep_count",
                "maximum_substeps",
                "split_interval_count",
                "split_group_count",
                "completed_work",
            )
        )
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds < 0 or not numpy.isfinite(
            self.interval_seconds
        ):
            raise ValueError(
                "Checkpoint interval must be finite and nonnegative"
            )
        self.stem = durable._stem(("rk_checkpoint", key))
        self.key_sha256 = _key_digest(key)
        self.entries = []
        self.saved_nodes = 0
        self.restored_intervals = 0
        self.restored_rk_stages = 0
        self.published_chunks = 0
        self.last_save = time.monotonic()

    def restore(self, histories):
        """Validate and restore a complete, contiguous saved history prefix."""

        pointer = evidence_assets.contained_path(
            self.durable.root, self.stem + ".rk-index.json"
        )
        if not pointer.exists():
            return None
        index = json.loads(pointer.read_text(encoding="utf-8"))
        if (
            index.get("schema_version") != 1
            or index.get("identity") != self.durable.identity
            or index.get("key_sha256") != self.key_sha256
            or not isinstance(index.get("chunks"), list)
            or not index["chunks"]
        ):
            raise ValueError("RK checkpoint identity or index differs")
        next_node = 0
        last_stats = None
        for entry in index["chunks"]:
            if not isinstance(entry, dict):
                raise ValueError("RK checkpoint chunk link is invalid")
            start, stop = entry.get("start"), entry.get("stop")
            if (
                not isinstance(start, int)
                or not isinstance(stop, int)
                or start != next_node
                or stop <= start
                or stop > histories.shape[1]
            ):
                raise ValueError("RK checkpoint chunks are not contiguous")
            payload = evidence_assets.read_array_products(
                self.durable.root, entry["artifact"], materialize=True
            )
            values = payload.get("histories")
            if (
                payload.get("identity") != self.durable.identity
                or payload.get("key_sha256") != self.key_sha256
                or payload.get("start") != start
                or payload.get("stop") != stop
                or not isinstance(values, numpy.ndarray)
                or values.shape
                != (histories.shape[0], stop - start, histories.shape[2])
                or not numpy.all(numpy.isfinite(values))
                or not isinstance(payload.get("stats"), dict)
            ):
                raise ValueError("RK checkpoint payload differs")
            histories[:, start:stop, :] = values
            last_stats = payload["stats"]
            next_node = stop
        if set(last_stats) != self.stats_fields or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in last_stats.values()
        ):
            raise ValueError("RK checkpoint work counters are invalid")
        self.entries = list(index["chunks"])
        self.saved_nodes = next_node
        self.restored_intervals = next_node - 1
        self.restored_rk_stages = last_stats.get("rk_stage_count", 0)
        return next_node - 1, dict(last_stats)

    def record(self, histories, completed_interval, stats):
        """Publish only newly completed nodes when the time budget elapses."""

        now = time.monotonic()
        stop = int(completed_interval) + 1
        if (
            stop <= self.saved_nodes
            or now - self.last_save < self.interval_seconds
        ):
            return
        start = self.saved_nodes
        artifact = evidence_assets.write_array_products(
            self.durable.root,
            self.stem + ".rk-chunk-" + uuid.uuid4().hex,
            {
                "identity": self.durable.identity,
                "key_sha256": self.key_sha256,
                "producer": self.durable.producer,
                "start": start,
                "stop": stop,
                "histories": numpy.asarray(histories[:, start:stop, :]),
                "stats": stats,
            },
        )
        entries = [
            *self.entries,
            {"start": start, "stop": stop, "artifact": artifact},
        ]
        evidence_assets.write_json_artifact(
            self.durable.root,
            self.stem + ".rk-index.json",
            {
                "schema_version": 1,
                "identity": self.durable.identity,
                "key_sha256": self.key_sha256,
                "chunks": entries,
            },
        )
        self.entries = entries
        self.saved_nodes = stop
        self.published_chunks += 1
        self.last_save = time.monotonic()


class BDFCheckpoint:
    """Persist accepted BDF steps and incremental sampled histories."""

    def __init__(self, durable, key, *, interval_seconds=30.0):
        """Bind a trajectory to its complete solver and physical identity."""

        self.durable = durable
        self.key = key
        self.interval_seconds = float(interval_seconds)
        if (
            not numpy.isfinite(self.interval_seconds)
            or self.interval_seconds < 0
        ):
            raise ValueError(
                "Checkpoint interval must be finite and nonnegative"
            )
        self.stem = durable._stem(("bdf_checkpoint", key))
        self.key_sha256 = _key_digest(key)
        self.entries = []
        self.saved_nodes = 0
        self.last_save = time.monotonic()
        self.restored_steps = 0
        self.restored_nfev = 0
        self.restored_samples = 0
        self.published_chunks = 0

    def due(self):
        """Return whether another accepted step should be persisted."""

        return time.monotonic() - self.last_save >= self.interval_seconds

    def restore(self, history):
        """Validate an exact prefix and return its latest solver snapshot."""

        pointer = evidence_assets.contained_path(
            self.durable.root, self.stem + ".bdf-index.json"
        )
        if not pointer.exists():
            return None
        index = json.loads(pointer.read_text(encoding="utf-8"))
        if (
            not isinstance(index, dict)
            or index.get("schema_version") != 1
            or index.get("identity") != self.durable.identity
            or index.get("key_sha256") != self.key_sha256
            or not isinstance(index.get("chunks"), list)
            or not index["chunks"]
        ):
            raise ValueError("BDF checkpoint identity or index differs")
        if not isinstance(history, numpy.ndarray) or history.ndim != 2:
            raise ValueError("BDF checkpoint history shape is invalid")
        next_node = 0
        previous_steps = 0
        previous_t = float("-inf")
        last_state = None
        for entry in index["chunks"]:
            if not isinstance(entry, dict):
                raise ValueError("BDF checkpoint chunk link is invalid")
            start, stop = entry.get("start"), entry.get("stop")
            if (
                not isinstance(start, int)
                or not isinstance(stop, int)
                or start != next_node
                or stop < start
                or stop > history.shape[1]
            ):
                raise ValueError("BDF checkpoint chunks are not contiguous")
            payload = evidence_assets.read_array_products(
                self.durable.root, entry["artifact"], materialize=True
            )
            if not isinstance(payload, dict):
                raise ValueError("BDF checkpoint payload differs")
            values = payload.get("history")
            state = payload.get("state")
            if (
                payload.get("identity") != self.durable.identity
                or payload.get("key_sha256") != self.key_sha256
                or payload.get("producer") != entry.get("producer")
                or payload.get("start") != start
                or payload.get("stop") != stop
                or not isinstance(values, numpy.ndarray)
                or values.shape != (history.shape[0], stop - start)
                or not numpy.all(numpy.isfinite(values))
                or not isinstance(state, dict)
            ):
                raise ValueError("BDF checkpoint payload differs")
            steps, epoch = state.get("accepted_steps"), state.get("t")
            if (
                not isinstance(steps, int)
                or isinstance(steps, bool)
                or steps <= previous_steps
                or not isinstance(epoch, (int, float))
                or not numpy.isfinite(epoch)
                or epoch <= previous_t
                or state.get("evaluated") != stop
            ):
                raise ValueError("BDF checkpoint progression is invalid")
            history[:, start:stop] = values
            next_node, previous_steps, previous_t = stop, steps, epoch
            last_state = state
        self.entries = list(index["chunks"])
        self.saved_nodes = next_node
        self.restored_steps = last_state["accepted_steps"]
        nfev = last_state.get("nfev")
        if not isinstance(nfev, int) or isinstance(nfev, bool) or nfev < 0:
            raise ValueError("BDF checkpoint work counters are invalid")
        self.restored_nfev = nfev
        self.restored_samples = next_node
        return last_state

    def record(self, history, evaluated, state):
        """Atomically publish new samples and one completed solver step."""

        now = time.monotonic()
        stop = int(evaluated)
        if stop < self.saved_nodes or stop > history.shape[1]:
            raise ValueError("BDF checkpoint sample count is invalid")
        if now - self.last_save < self.interval_seconds:
            return
        start = self.saved_nodes
        artifact = evidence_assets.write_array_products(
            self.durable.root,
            self.stem + ".bdf-chunk-" + uuid.uuid4().hex,
            {
                "identity": self.durable.identity,
                "key_sha256": self.key_sha256,
                "producer": self.durable.producer,
                "start": start,
                "stop": stop,
                "history": numpy.asarray(history[:, start:stop]),
                "state": state,
            },
        )
        entries = [
            *self.entries,
            {
                "start": start,
                "stop": stop,
                "producer": self.durable.producer,
                "artifact": artifact,
            },
        ]
        evidence_assets.write_json_artifact(
            self.durable.root,
            self.stem + ".bdf-index.json",
            {
                "schema_version": 1,
                "identity": self.durable.identity,
                "key_sha256": self.key_sha256,
                "chunks": entries,
            },
        )
        self.entries = entries
        self.saved_nodes = stop
        self.published_chunks += 1
        self.last_save = time.monotonic()


def active_durable_histories():
    """Return the explicitly scoped completion-attempt history store."""

    return _ACTIVE.get()


@contextmanager
def retain_durable_histories(root, identity, producer):
    """Enable restart-safe reuse within this completion attempt."""

    store = DurableHistories(root, identity, producer)
    token = _ACTIVE.set(store)
    try:
        yield store
    finally:
        _ACTIVE.reset(token)
