"""Module entrypoint for `python -m copernican`."""

from .cli import main as package_main


def main() -> int:
    """Run the package workflow entrypoint."""

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
