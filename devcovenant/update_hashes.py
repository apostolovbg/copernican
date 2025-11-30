"""Update DevCovenant policy script hashes in registry.json.

This utility automatically computes SHA256 hashes for all policy scripts
with line-ending normalization and updates the registry.json file.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def compute_hash(file_path: Path) -> str:
    """Compute SHA256 hash with line-ending normalization.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal SHA256 digest
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk.replace(b"\r\n", b"\n"))
    return hasher.hexdigest()

def update_registry_hashes(repo_root: Path | None = None) -> int:
    """Update all policy script hashes in registry.json.

    Args:
        repo_root: Repository root path (defaults to script parent directory)

    Returns:
        0 on success, 1 on error
    """
    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    registry_path = repo_root / "devcovenant" / "registry.json"

    if not registry_path.exists():
        print(f"Error: Registry not found at {registry_path}", file=sys.stderr)
        return 1

    # Load current registry
    with open(registry_path) as f:
        registry = json.load(f)

    # Update timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    # Update each policy's hash
    updated = 0
    for policy_id, policy_data in registry.get("policies", {}).items():
        script_path_str = policy_data.get("script_path")
        if not script_path_str:
            continue

        script_path = repo_root / script_path_str
        if not script_path.exists():
            print(
                f"Warning: Policy script not found: {script_path}",
                file=sys.stderr,
            )
            continue

        old_hash = policy_data.get("hash", "")
        new_hash = compute_hash(script_path)

        if old_hash != new_hash:
            policy_data["hash"] = new_hash
            policy_data["last_updated"] = timestamp
            updated += 1
            print(f"Updated {policy_id}: {script_path.name}")

    if updated == 0:
        print("All policy hashes are up to date.")
        return 0

    # Write updated registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")

    print(f"\nUpdated {updated} policy hash(es) in registry.json")
    return 0

def main() -> int:
    """CLI entry point."""
    return update_registry_hashes()

if __name__ == "__main__":
    sys.exit(main())
