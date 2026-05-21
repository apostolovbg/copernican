"""Script entrypoint for the custom command."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devcovenant.custom import main, run

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
