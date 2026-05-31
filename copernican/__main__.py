"""Module entrypoint for `python -m copernican`."""

import sys

from . import main as package_main


def main() -> int:
    """Run the package workflow entrypoint."""

    return package_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
