"""Console-script entrypoint for the Copernican package."""

from __future__ import annotations

from collections.abc import Iterable

from copernican.workflow import main as workflow_main


def main(argv: Iterable[str] | None = None) -> int:
    """Run the package workflow through the published CLI entrypoint."""

    return workflow_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
