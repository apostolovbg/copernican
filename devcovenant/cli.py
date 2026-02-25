"""
Command-line interface for devcovenant.
"""

import argparse
import sys
from pathlib import Path

from .engine import DevCovenantEngine


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DevCovenant - Self-enforcing policy system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=["check", "sync", "test", "update-hashes"],
        help="Command to run",
    )

    parser.add_argument(
        "--mode",
        choices=["startup", "lint", "pre-commit", "normal"],
        default="normal",
        help="Check mode (default: normal)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix violations when possible",
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )

    args = parser.parse_args()

    # Initialize engine
    engine = DevCovenantEngine(repo_root=args.repo)

    # Execute command
    if args.command == "check":
        result = engine.check(mode=args.mode, apply_fixes=args.fix)

        # Exit with error code if blocked
        if result.should_block or result.has_sync_issues():
            sys.exit(1)
        else:
            sys.exit(0)

    elif args.command == "sync":
        # Force a sync check
        result = engine.check(mode="startup")
        if result.has_sync_issues():
            print(
                "\n⚠️  Policy sync issues detected. "
                "Please update policy scripts."
            )
            sys.exit(1)
        else:
            print("\n✅ All policies are in sync!")
            sys.exit(0)

    elif args.command == "test":
        # Run devcovenant's own tests
        import subprocess

        try:
            result = subprocess.run(
                ["pytest", "devcovenant/tests/", "-v"],
                cwd=args.repo,
                check=True,
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            sys.exit(1)

    elif args.command == "update-hashes":
        # Update policy script hashes in registry.json
        from .update_hashes import update_registry_hashes

        result = update_registry_hashes(args.repo)
        sys.exit(result)


if __name__ == "__main__":
    main()
