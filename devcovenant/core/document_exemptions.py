"""Managed document exemption fingerprints and normalization."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

DEFAULT_MANAGED_MARKERS = (
    ("<!-- DEVCOV:BEGIN -->", "<!-- DEVCOV:END -->"),
    ("<!-- DEVCOV-WORKFLOW:BEGIN -->", "<!-- DEVCOV-WORKFLOW:END -->"),
    ("<!-- DEVCOV-POLICIES:BEGIN -->", "<!-- DEVCOV-POLICIES:END -->"),
)
DEFAULT_HEADER_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})
DEFAULT_HEADER_KEYS = frozenset(
    {
        "last updated",
        "project version",
        "project stage",
        "versioning mode",
        "project codename",
        "build identity",
        "devcovenant version",
    }
)
DEFAULT_HEADER_SCAN_LINES = 4
EMPTY_MANAGED_MARKER_SIGNATURE = hashlib.sha256(b"").hexdigest()


def _hash_lines(lines: list[str]) -> str:
    """Return a deterministic SHA-256 digest for normalized text lines."""
    normalized = "\n".join(line.rstrip() for line in lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _managed_ranges(
    content: str,
    *,
    managed_markers: tuple[tuple[str, str], ...] = DEFAULT_MANAGED_MARKERS,
) -> list[tuple[int, int]]:
    """Return line ranges covered by managed block markers."""
    ranges: list[tuple[int, int]] = []
    lines = content.splitlines()
    for begin_marker, end_marker in managed_markers:
        start_line: int | None = None
        for index, line in enumerate(lines, start=1):
            if start_line is None and begin_marker in line:
                start_line = index
            elif start_line is not None and end_marker in line:
                ranges.append((start_line, index))
                start_line = None
    return ranges


def _is_header_key_line(line: str, header_keys: set[str]) -> bool:
    """Return True when a line starts with an allowed header key."""
    match = re.match(
        r"^\*{0,2}\s*([a-z][a-z \-]+?)\s*:\*{0,2}\s*",
        line.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    return match.group(1).strip().lower() in header_keys


def _document_header_ranges(
    content: str,
    *,
    header_keys: set[str],
    header_scan_lines: int,
) -> list[tuple[int, int]]:
    """Return top-of-file header line ranges eligible for changelog skips."""
    lines = content.splitlines()
    ranges: list[tuple[int, int]] = []
    for line_number, line in enumerate(lines[:header_scan_lines], start=1):
        if _is_header_key_line(line, header_keys):
            ranges.append((line_number, line_number))
    return ranges


def _line_in_ranges(line_number: int, ranges: list[tuple[int, int]]) -> bool:
    """Return True when one line number falls inside any exempt range."""
    for start, end in ranges:
        if start <= line_number <= end:
            return True
    return False


def managed_marker_signature(
    content: str,
    *,
    managed_markers: tuple[tuple[str, str], ...] = DEFAULT_MANAGED_MARKERS,
) -> str:
    """Return a deterministic managed-marker sequence signature."""
    tokens: list[str] = []
    for line in content.splitlines():
        for marker_index, (begin_marker, end_marker) in enumerate(
            managed_markers
        ):
            if begin_marker in line:
                tokens.append(f"{marker_index}:begin")
            if end_marker in line:
                tokens.append(f"{marker_index}:end")
    return _hash_lines(tokens)


def non_exempt_content_hash(
    content: str,
    relative_path: str,
    *,
    header_doc_suffixes: set[str],
    header_keys: set[str],
    header_scan_lines: int,
    managed_markers: tuple[tuple[str, str], ...] = DEFAULT_MANAGED_MARKERS,
) -> str:
    """Return a hash for lines outside managed/header exempt ranges."""
    ranges = _managed_ranges(content, managed_markers=managed_markers)
    suffix = Path(relative_path).suffix.lower()
    if suffix in header_doc_suffixes:
        ranges.extend(
            _document_header_ranges(
                content,
                header_keys=header_keys,
                header_scan_lines=header_scan_lines,
            )
        )
    lines = content.splitlines()
    visible = [
        line
        for line_number, line in enumerate(lines, start=1)
        if not _line_in_ranges(line_number, ranges)
    ]
    return _hash_lines(visible)


def document_exemption_fingerprint_for_path(
    repo_root: Path,
    relative_path: str,
    *,
    header_doc_suffixes: set[str],
    header_keys: set[str],
    header_scan_lines: int,
    managed_markers: tuple[tuple[str, str], ...] = DEFAULT_MANAGED_MARKERS,
) -> dict[str, str] | None:
    """Return one exemption-fingerprint entry for a repository file path."""
    path = repo_root / relative_path
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return {
        "non_exempt_content_sha256": non_exempt_content_hash(
            content,
            relative_path,
            header_doc_suffixes=header_doc_suffixes,
            header_keys=header_keys,
            header_scan_lines=header_scan_lines,
            managed_markers=managed_markers,
        ),
        "managed_marker_signature": managed_marker_signature(
            content,
            managed_markers=managed_markers,
        ),
    }


def normalize_document_exemption_entry(
    raw_entry: object,
    *,
    relative_path: str,
) -> dict[str, str]:
    """Validate one exemption-fingerprint entry loaded from gate status."""
    if not isinstance(raw_entry, dict):
        raise ValueError(
            "Invalid gate status payload: "
            "`document_exemption_baseline` entries must be "
            "mappings."
        )
    non_exempt_hash = str(
        raw_entry.get("non_exempt_content_sha256", "")
    ).strip()
    marker_signature = str(
        raw_entry.get("managed_marker_signature", "")
    ).strip()
    if not non_exempt_hash or not marker_signature:
        raise ValueError(
            "Invalid gate status payload: "
            "`document_exemption_baseline` entries must define "
            "`non_exempt_content_sha256` and `managed_marker_signature` "
            f"(path: {relative_path})."
        )
    return {
        "non_exempt_content_sha256": non_exempt_hash,
        "managed_marker_signature": marker_signature,
    }


__all__ = [
    "DEFAULT_HEADER_DOC_SUFFIXES",
    "DEFAULT_HEADER_KEYS",
    "DEFAULT_HEADER_SCAN_LINES",
    "DEFAULT_MANAGED_MARKERS",
    "EMPTY_MANAGED_MARKER_SIGNATURE",
    "document_exemption_fingerprint_for_path",
    "managed_marker_signature",
    "non_exempt_content_hash",
    "normalize_document_exemption_entry",
]
