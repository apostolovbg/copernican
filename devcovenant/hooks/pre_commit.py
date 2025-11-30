#!/usr/bin/env python3
"""
Pre-commit hook for devcovenant.

This script runs devcovenant checks before allowing a commit.
Install by adding to .pre-commit-config.yaml or copying to
.git/hooks/pre-commit
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import devcovenant
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from devcovenant.engine import DevCovenantEngine  # noqa: E402


def main():
    """Run devcovenant pre-commit checks."""
    print("Running DevCovenant pre-commit checks...\n")

    try:
        engine = DevCovenantEngine(repo_root=repo_root)
        result = engine.check(mode="pre-commit")

        if result.has_sync_issues():
            print("\n❌ PRE-COMMIT BLOCKED: Policy sync required")
            print("Please update policy scripts before committing.")
            return 1

        if result.should_block:
            print("\n❌ PRE-COMMIT BLOCKED: Policy violations detected")
            print("Please fix the violations above before committing.")
            return 1

        print("\n✅ DevCovenant checks passed!")
        return 0

    except Exception as e:
        print(f"\n⚠️  DevCovenant check failed with error: {e}")
        print("Proceeding with commit (devcovenant is not blocking)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
