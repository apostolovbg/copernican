"""
Registry for tracking policy hashes and sync status.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .parser import PolicyDefinition


@dataclass
class PolicySyncIssue:
    """
    Represents a policy that is out of sync with its script.

    Attributes:
        policy_id: ID of the policy
        policy_text: Current policy text from AGENTS.md
        policy_hash: Hash of current policy text
        script_path: Path to the policy script
        script_exists: Whether the script exists
        issue_type: Type of sync issue
            (hash_mismatch, script_missing, new_policy)
        current_hash: Current hash from registry (if any)
    """

    policy_id: str
    policy_text: str
    policy_hash: str
    script_path: Path
    script_exists: bool
    issue_type: str
    current_hash: Optional[str] = None


class PolicyRegistry:
    """
    Manages the policy registry (tracking hashes and sync status).
    """

    def __init__(self, registry_path: Path, repo_root: Path):
        """
        Initialize the registry.

        Args:
            registry_path: Path to registry.json
            repo_root: Root directory of the repository
        """
        self.registry_path = registry_path
        self.repo_root = repo_root
        self._data: Dict = {}
        self.load()

    def load(self):
        """Load the registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"policies": {}, "metadata": {"version": "1.0.0"}}

    def save(self):
        """Save the registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def calculate_full_hash(
        self, policy_text: str, script_content: str
    ) -> str:
        """
        Calculate the full hash (policy text + script content).

        Args:
            policy_text: The policy description from AGENTS.md
            script_content: The Python script content

        Returns:
            SHA256 hash of combined content
        """
        # Normalize both
        normalized_policy = policy_text.strip()
        normalized_script = script_content.strip()

        # Combine
        combined = f"{normalized_policy}\n---\n{normalized_script}"

        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def check_policy_sync(
        self, policies: List[PolicyDefinition]
    ) -> List[PolicySyncIssue]:
        """
        Check which policies are out of sync with their scripts.

        Args:
            policies: List of policies from AGENTS.md

        Returns:
            List of PolicySyncIssue objects for policies that need updating
        """
        issues = []

        for policy in policies:
            # Skip deleted or deprecated policies
            if policy.status in ["deleted", "deprecated"]:
                continue

            # Determine script path
            # Convert hyphens to underscores for Python module names
            script_name = policy.policy_id.replace("-", "_")
            script_path = (
                self.repo_root
                / "devcovenant"
                / "policy_scripts"
                / f"{script_name}.py"
            )

            # Check if script exists
            script_exists = script_path.exists()

            # Get current hash from registry
            current_hash = None
            if policy.policy_id in self._data.get("policies", {}):
                current_hash = self._data["policies"][policy.policy_id].get(
                    "hash"
                )

            # Determine if there's an issue
            issue_type = None

            if policy.status == "new" or not script_exists:
                if not script_exists:
                    issue_type = "script_missing"
                else:
                    issue_type = "new_policy"
                issues.append(
                    PolicySyncIssue(
                        policy_id=policy.policy_id,
                        policy_text=policy.description,
                        policy_hash="",
                        script_path=script_path,
                        script_exists=script_exists,
                        issue_type=issue_type,
                        current_hash=current_hash,
                    )
                )
                continue

            # If policy is marked as updated, it needs sync
            if policy.updated:
                issue_type = "hash_mismatch"
                issues.append(
                    PolicySyncIssue(
                        policy_id=policy.policy_id,
                        policy_text=policy.description,
                        policy_hash="",
                        script_path=script_path,
                        script_exists=script_exists,
                        issue_type=issue_type,
                        current_hash=current_hash,
                    )
                )
                continue

            # Calculate current hash if script exists
            if script_exists:
                with open(script_path, "r", encoding="utf-8") as f:
                    script_content = f.read()

                calculated_hash = self.calculate_full_hash(
                    policy.description, script_content
                )

                # Compare with stored hash
                if current_hash and calculated_hash != current_hash:
                    issue_type = "hash_mismatch"
                    issues.append(
                        PolicySyncIssue(
                            policy_id=policy.policy_id,
                            policy_text=policy.description,
                            policy_hash=calculated_hash,
                            script_path=script_path,
                            script_exists=script_exists,
                            issue_type=issue_type,
                            current_hash=current_hash,
                        )
                    )

        return issues

    def update_policy_hash(
        self, policy_id: str, policy_text: str, script_path: Path
    ):
        """
        Update the hash for a policy after its script has been updated.

        Args:
            policy_id: ID of the policy
            policy_text: Current policy text
            script_path: Path to the policy script
        """
        # Read script
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()

        # Calculate new hash
        new_hash = self.calculate_full_hash(policy_text, script_content)

        # Update registry
        if "policies" not in self._data:
            self._data["policies"] = {}

        self._data["policies"][policy_id] = {
            "hash": new_hash,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "script_path": str(script_path.relative_to(self.repo_root)),
        }

        self.save()

    def get_policy_hash(self, policy_id: str) -> Optional[str]:
        """
        Get the stored hash for a policy.

        Args:
            policy_id: ID of the policy

        Returns:
            Hash string or None if not found
        """
        return self._data.get("policies", {}).get(policy_id, {}).get("hash")
