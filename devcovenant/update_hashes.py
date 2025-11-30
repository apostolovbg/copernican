"""Update DevCovenant policy script hashes in registry.json.

This utility automatically computes SHA256 hashes for all policy scripts
combined with their policy text from AGENTS.md and updates the registry.json
file.
"""

import sys
from pathlib import Path

from .parser import PolicyParser
from .registry import PolicyRegistry


def update_registry_hashes(repo_root: Path | None = None) -> int:
    """Update all policy script hashes in registry.json.

    Args:
        repo_root: Repository root path (defaults to script parent directory)

    Returns:
        0 on success, 1 on error
    """
    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    agents_md_path = repo_root / "AGENTS.md"
    registry_path = repo_root / "devcovenant" / "registry.json"

    if not agents_md_path.exists():
        print(
            f"Error: AGENTS.md not found at {agents_md_path}",
            file=sys.stderr,
        )
        return 1

    if not registry_path.exists():
        print(
            f"Error: Registry not found at {registry_path}",
            file=sys.stderr,
        )
        return 1

    # Parse policies from AGENTS.md
    parser = PolicyParser(agents_md_path)
    policies = parser.parse_agents_md()

    # Load registry
    registry = PolicyRegistry(registry_path, repo_root)

    # Update each policy's hash
    updated = 0
    for policy in policies:
        # Skip deleted or deprecated policies
        if policy.status in ["deleted", "deprecated"]:
            continue

        # Determine script path
        script_name = policy.policy_id.replace("-", "_")
        script_path = (
            repo_root / "devcovenant" / "policy_scripts" / f"{script_name}.py"
        )

        if not script_path.exists():
            print(
                f"Warning: Policy script not found: {script_path}",
                file=sys.stderr,
            )
            continue

        # Update hash using the correct calculation (policy text + script)
        registry.update_policy_hash(
            policy.policy_id, policy.description, script_path
        )
        updated += 1
        print(f"Updated {policy.policy_id}: {script_path.name}")

    if updated == 0:
        print("All policy hashes are up to date.")
        return 0

    print(f"\nUpdated {updated} policy hash(es) in registry.json")
    return 0


def main() -> int:
    """CLI entry point."""
    return update_registry_hashes()


if __name__ == "__main__":
    sys.exit(main())
