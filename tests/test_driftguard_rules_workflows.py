from pathlib import Path

from driftguard.rules import RuleContext
from driftguard.rules.workflows import FullTestSuiteInCIRule
from driftguard.spec import DriftConfig, DriftGuardSpec, SurfaceSpec


def _context(tmp_path: Path) -> RuleContext:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    policy_doc = tmp_path / "DRIFTGUARD.md"
    policy_doc.touch()

    spec = DriftGuardSpec(
        version=1,
        project="Tests",
        rulesets={"workflow-discipline": "hard"},
        surfaces={
            "ci-workflows": SurfaceSpec(
                name="ci-workflows",
                include=[".github/workflows/ci.yml", "DRIFTGUARD.md"],
                exclude=[],
                rules=[FullTestSuiteInCIRule.name],
            )
        },
        drift=DriftConfig(metrics={}),
    )

    return RuleContext(
        repo_root=tmp_path, spec=spec, scope="repo", mode="full"
    )


def test_full_suite_rule_requires_ci_pytest_and_policy_doc(
    tmp_path: Path,
) -> None:
    """The rule should flag missing pytest runs and absent policy wording."""

    context = _context(tmp_path)
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.write_text("name: CI\nrun: echo nope\n", encoding="utf-8")
    policy_doc = tmp_path / "DRIFTGUARD.md"
    policy_doc.write_text(
        "Policy doc without test reminder.\n", encoding="utf-8"
    )

    violations = FullTestSuiteInCIRule().check(context)

    messages = [violation.message for violation in violations]
    assert any("pytest invocation" in message for message in messages)
    assert any("unittest discover" in message for message in messages)
    assert any("before every commit" in message for message in messages)


def test_full_suite_rule_accepts_pytest_and_policy_doc(tmp_path: Path) -> None:
    """Pytest coverage in CI and documented expectations should pass."""

    context = _context(tmp_path)
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.write_text(
        """name: CI
jobs:
  tests:
    steps:
      - name: Unit tests
        run: python -m unittest discover -v
      - name: Pytest
        run: python -m pytest -q
      - name: Run DriftGuard policy check
        run: >-
          python -m driftguard.cli check --scope=repo --mode=full --repo-root .
""",
        encoding="utf-8",
    )
    policy_doc = tmp_path / "DRIFTGUARD.md"
    policy_doc.write_text(
        "Run the full program unit test suite in /tests before every commit "
        "and every task using both python -m pytest -q and python -m unittest "
        "discover -v.\n",
        encoding="utf-8",
    )

    violations = FullTestSuiteInCIRule().check(context)

    assert not violations


def test_full_suite_rule_requires_driftguard_after_tests(
    tmp_path: Path,
) -> None:
    """DriftGuard must run after the test steps."""

    context = _context(tmp_path)
    workflow_path = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow_path.write_text(
        """name: CI
jobs:
  tests:
    steps:
      - name: Run DriftGuard policy check
        run: >-
          python -m driftguard.cli check --scope=repo --mode=full --repo-root .
      - name: Unit tests
        run: python -m unittest discover -v
      - name: Pytest
        run: python -m pytest -q
""",
        encoding="utf-8",
    )
    policy_doc = tmp_path / "DRIFTGUARD.md"
    policy_doc.write_text(
        "Run the full program unit test suite in /tests before every commit "
        "and every task using both python -m pytest -q and python -m unittest "
        "discover -v.\n",
        encoding="utf-8",
    )

    violations = FullTestSuiteInCIRule().check(context)

    assert any(
        "after the Pytest and Unit tests" in v.message for v in violations
    )
