"""Dependency hash utilities for the Copernican Suite.

This module automates recreation of platform-specific wheel hashes for
``requirements.lock`` so ``pip`` can verify downloads on Linux, macOS and
Windows.  The helper downloads metadata from the PyPI JSON API and inserts
missing ``--hash=sha256`` lines for any wheel files not already listed.

Keeping this process in code eliminates the repeated manual steps that led
to past CI failures when new platform wheels were published after the
lock file was generated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence
from urllib.request import Request, urlopen


def fetch_wheel_hashes(package: str, version: str) -> set[str]:
    """Return hashes for all wheel files of a package release.

    Parameters
    ----------
    package:
        Package name as published on PyPI.
    version:
        Exact package version to inspect.
    """
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    req = Request(url, headers={"User-Agent": "copernican-suite"})
    with urlopen(
        req
    ) as resp:  # pragma: no cover - exercised in tests via mock
        data = json.load(resp)

    def wanted(filename: str) -> bool:
        """Determine whether ``filename`` targets our supported platforms."""

        if not filename.endswith(".whl"):
            return False

        # Wheel filenames are ``{dist}-{ver}-{py}-{abi}-{plat}.whl`` with
        # hyphens only occurring in the ``{dist}`` segment.  Splitting from the
        # right avoids mis-parsing distributions containing hyphens.
        parts = filename.rsplit("-", 3)
        if len(parts) != 4:
            return False
        py_tag, abi_tag, plat = parts[1], parts[2], parts[3][:-4]

        # Accept wheels built for the running interpreter version, pure Python
        # wheels (``py3``) and stable ABI builds (``abi3``).  Stable ABI wheels
        # such as ``cp39-abi3`` remain compatible across Python 3 releases and
        # previously slipped through the filters, leading to missing hashes for
        # packages like ``pyerfa``.
        cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if py_tag != cp_tag and abi_tag != "abi3" and py_tag != "py3":
            return False

        if plat.startswith("win_amd64"):
            return True
        if "manylinux" in plat and ("x86_64" in plat or "aarch64" in plat):
            return True
        if plat.startswith("macosx") and (
            "x86_64" in plat or "arm64" in plat or "universal2" in plat
        ):
            # ``universal2`` wheels embed both x86_64 and arm64 binaries.
            return True
        return False

    return {
        file["digests"]["sha256"]
        for file in data.get("urls", [])
        if wanted(file["filename"])
    }


def _hash_block(lines: Sequence[str], start: int) -> tuple[int, set[str]]:
    """Return end index and hashes for the block starting at ``start``.

    The function scans the ``lines`` sequence starting one line below the
    requirement declaration and collects every ``--hash=sha256`` entry.
    ``start`` must point to the requirement line itself.
    """
    hashes: set[str] = set()
    end = start + 1
    while end < len(lines) and lines[end].lstrip().startswith("--hash="):
        # ``--hash=sha256:<digest>`` may optionally end with a trailing ``\``.
        digest = lines[end].split(":", 1)[1].split()[0].rstrip("\\")
        hashes.add(digest)
        end += 1
    return end, hashes


def update_hashes(path: Path, packages: Iterable[str] | None = None) -> bool:
    """Insert missing wheel hashes into ``path``.

    Parameters
    ----------
    path:
        Location of the requirements lock file.
    packages:
        Optional iterable restricting which packages to refresh.  When omitted
        all packages found in the file are processed.

    Returns
    -------
    bool
        ``True`` when the file was modified.
    """
    lines = path.read_text().splitlines()
    pkg_pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<ver>[^\\s]+) \\"
    )
    targets = {p.lower() for p in packages} if packages else None
    i = 0
    changed = False
    while i < len(lines):
        match = pkg_pattern.match(lines[i])
        if not match:
            i += 1
            continue
        name = match.group("name")
        if targets and name.lower() not in targets:
            i += 1
            continue
        version = match.group("ver")
        end, existing = _hash_block(lines, i)
        needed = fetch_wheel_hashes(name, version)
        if not needed:
            i = end
            continue
        if existing != needed:
            merged = sorted(needed)
            block = [f"    --hash=sha256:{h} \\" for h in merged[:-1]]
            block.append(f"    --hash=sha256:{merged[-1]}")
            lines[i + 1 : end] = block  # noqa: E203
            delta = len(block) - (end - (i + 1))
            end += delta
            changed = True
        i = end
    if changed:
        path.write_text("\n".join(lines) + "\n")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Refresh wheel hashes for the provided lock file.

    This command updates ``requirements.lock`` in place.  It exits with ``0``
    even when modifications occur so pre-commit can rerun automatically.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("requirements.lock")
    )
    parser.add_argument("packages", nargs="*")
    args = parser.parse_args(argv)
    update_hashes(args.path, args.packages or None)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
