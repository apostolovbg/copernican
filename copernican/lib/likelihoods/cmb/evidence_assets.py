"""Durable, integrity-checked numerical products for scientific evidence.

These helpers store products for the existing completion runner. They make
no scientific decision: the numerical and independent physical comparisons
remain owned by the solver and its diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy

from .contracts import _jsonable


@dataclass(frozen=True)
class RetainedArrayInfo:
    """Describe verified binary content, distinct from user mapping values."""

    dtype: str
    shape: tuple[int, ...]
    sha256: str


def _file_sha256(path):
    """Hash large numerical products without reading them all into RAM."""

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained_path(root, relative):
    """Resolve one portable artifact path without following escaped links."""

    base = Path(root).resolve()
    relative = Path(relative)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or "\\" in relative.as_posix()
        or ":" in relative.as_posix()
    ):
        raise ValueError("artifact path must be relative and contained")
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("artifact path escapes its evidence root")
    return target


def _atomic_write(path, writer):
    """Publish a flushed complete file, leaving an older artifact intact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".pending-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_artifact(root, relative, payload):
    """Write one JSON payload and return its portable content identity."""

    path = contained_path(root, relative)
    contents = (
        json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(contents))
    return {
        "path": path.relative_to(Path(root).resolve()).as_posix(),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def read_json_artifact(root, entry):
    """Verify both a JSON artifact's location and its actual bytes."""

    path = contained_path(root, entry.get("path", ""))
    if _file_sha256(path) != entry.get("sha256"):
        raise ValueError("artifact content digest is invalid")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be a mapping")
    return payload


def write_array_products(root, relative_stem, products):
    """Persist full arrays without expanding native histories into JSON."""

    arrays = {}
    descriptions = {}

    def encode(value):
        """Keep physical mapping keys distinct from typed array links."""

        if isinstance(value, numpy.ndarray):
            array = numpy.asarray(value)
            if array.dtype.kind not in "biufc" or not numpy.all(
                numpy.isfinite(array)
            ):
                raise ValueError("retained numerical arrays must be finite")
            name = f"array_{len(arrays)}"
            arrays[name] = array
            descriptions[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            return {"kind": "array", "key": name}
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError(
                    "numerical product mapping keys must be strings"
                )
            return {
                "kind": "mapping",
                "items": {
                    str(key): encode(item) for key, item in value.items()
                },
            }
        if isinstance(value, (tuple, list)):
            return {
                "kind": "sequence",
                "items": [encode(item) for item in value],
            }
        return {"kind": "scalar", "value": _jsonable(value)}

    tree = encode(products)
    # Publish immutable binary content before its manifest. Replacing a
    # preceding binary first would invalidate an older complete manifest
    # if the producer stopped between the two writes.
    stem = contained_path(root, relative_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=stem.parent) as archive:
        numpy.savez_compressed(archive, **arrays)
        archive.seek(0)
        digest = hashlib.file_digest(archive, "sha256").hexdigest()
        relative = f"{relative_stem}-{digest}.npz"
        binary = contained_path(root, relative)
        archive.seek(0)
        _atomic_write(
            binary, lambda stream: shutil.copyfileobj(archive, stream)
        )
    binary_identity = {
        "path": relative,
        "sha256": digest,
    }
    return write_json_artifact(
        root,
        str(relative_stem) + ".json",
        {
            "schema_version": 1,
            "binary": binary_identity,
            "arrays": descriptions,
            "products": tree,
        },
    )


def read_array_products(root, entry, *, materialize=False, array_paths=()):
    """Verify each saved array, using bounded memory during ordinary readback.

    Validation visits arrays separately. Rendering or a small consumer may
    explicitly materialize them; normal verification retains only metadata.
    """

    manifest = read_json_artifact(root, entry)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported numerical product schema")
    binary = contained_path(root, manifest["binary"]["path"])
    if _file_sha256(binary) != manifest["binary"]["sha256"]:
        raise ValueError("numerical product binary digest differs")
    loaded = {}
    selected = set()
    for path in array_paths:
        node = manifest["products"]
        for name in path:
            if (
                not isinstance(node, dict)
                or node.get("kind") != "mapping"
                or not isinstance(node.get("items"), dict)
                or name not in node["items"]
            ):
                raise ValueError("selected numerical array path is missing")
            node = node["items"][name]
        if not isinstance(node, dict) or node.get("kind") != "array":
            raise ValueError("selected numerical product is not an array")
        selected.add(node["key"])
    with numpy.load(binary, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["arrays"]):
            raise ValueError("numerical array inventory differs")
        for name, description in manifest["arrays"].items():
            array = archive[name]
            if (
                array.dtype.str != description["dtype"]
                or list(array.shape) != description["shape"]
                or array.dtype.kind not in "biufc"
                or not numpy.all(numpy.isfinite(array))
                or hashlib.sha256(array.tobytes()).hexdigest()
                != description["sha256"]
            ):
                raise ValueError(f"retained numerical array differs: {name}")
            if materialize or name in selected:
                array.setflags(write=False)
                loaded[name] = array

    def decode(value):
        """Rebuild a validated typed tree without treating physics as links."""

        if not isinstance(value, dict):
            raise ValueError("numerical product node is malformed")
        kind = value.get("kind")
        if kind == "array" and set(value) == {"kind", "key"}:
            name = value["key"]
            if name not in manifest["arrays"]:
                raise ValueError("product references a missing array")
            if name in loaded:
                return loaded[name]
            description = manifest["arrays"][name]
            return RetainedArrayInfo(
                dtype=description["dtype"],
                shape=tuple(description["shape"]),
                sha256=description["sha256"],
            )
        if kind == "mapping" and set(value) == {"kind", "items"}:
            return {
                name: decode(item) for name, item in value["items"].items()
            }
        if kind == "sequence" and set(value) == {"kind", "items"}:
            return [decode(item) for item in value["items"]]
        if kind == "scalar" and set(value) == {"kind", "value"}:
            return value["value"]
        raise ValueError("numerical product node has an invalid kind")

    return decode(manifest["products"])
