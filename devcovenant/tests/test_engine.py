"""
Tests for the devcovenant engine.
"""

import tempfile
from pathlib import Path

from devcovenant.engine import DevCovenantEngine


def test_engine_initialization():
    """Test that the engine initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir).resolve()

        # Create minimal structure
        (repo_root / "devcovenant").mkdir()
        (repo_root / "AGENTS.md").write_text("# Test")

        engine = DevCovenantEngine(repo_root=repo_root)

        assert engine.repo_root == repo_root
        assert engine.agents_md_path.exists()


def test_engine_check_no_violations():
    """Test engine check with no violations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create structure
        devcov_dir = repo_root / "devcovenant"
        devcov_dir.mkdir()
        (devcov_dir / "policy_scripts").mkdir()
        config_text = "engine:\n  fail_threshold: error"
        (devcov_dir / "config.yaml").write_text(config_text)

        # Create AGENTS.md with no policies
        agents_text = "# Development Guide\n\nNo policies yet."
        (repo_root / "AGENTS.md").write_text(agents_text)

        engine = DevCovenantEngine(repo_root=repo_root)
        result = engine.check(mode="normal")

        # Should have no violations and not block
        assert result.should_block is False
