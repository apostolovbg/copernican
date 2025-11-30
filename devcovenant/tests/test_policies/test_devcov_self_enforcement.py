"""
Tests for devcov-self-enforcement policy.
"""

import tempfile
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.devcov_self_enforcement import (
    DevCovenantSelfEnforcementCheck,
)


def test_policy_with_test_passes():
    """Test that policy scripts with tests pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create devcovenant structure
        (repo_root / "devcovenant" / "policy_scripts").mkdir(parents=True)
        (repo_root / "devcovenant" / "tests" / "test_policies").mkdir(
            parents=True
        )

        # Create a policy script
        policy_script = (
            repo_root / "devcovenant" / "policy_scripts" / "test-policy.py"
        )
        policy_script.write_text("# Test policy")

        # Create corresponding test
        test_file = (
            repo_root
            / "devcovenant"
            / "tests"
            / "test_policies"
            / "test_test-policy.py"
        )
        test_file.write_text("# Test")

        checker = DevCovenantSelfEnforcementCheck()
        context = CheckContext(repo_root=repo_root, all_files=[policy_script])
        violations = checker.check(context)

        assert len(violations) == 0


def test_policy_without_test_fails():
    """Test that policy scripts without tests fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create devcovenant structure
        (repo_root / "devcovenant" / "policy_scripts").mkdir(parents=True)
        (repo_root / "devcovenant" / "tests" / "test_policies").mkdir(
            parents=True
        )

        # Create a policy script WITHOUT test
        policy_script = (
            repo_root / "devcovenant" / "policy_scripts" / "test-policy.py"
        )
        policy_script.write_text("# Test policy")

        checker = DevCovenantSelfEnforcementCheck()
        context = CheckContext(repo_root=repo_root, all_files=[policy_script])
        violations = checker.check(context)

        assert len(violations) >= 1
        assert violations[0].policy_id == "devcov-self-enforcement"
