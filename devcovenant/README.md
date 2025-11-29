# DevCovenant - Self-Enforcing Policy System

DevCovenant is an autonomous policy enforcement system for the Copernican Suite.
It maintains consistency between human-readable policies in AGENTS.md and
automated Python checks.

## How It Works

1. **Policies are defined** in `AGENTS.md` using a structured format
2. **Python scripts** in `policy_scripts/` check compliance
3. **Hash verification** ensures policies and scripts stay in sync
4. **AI automatically updates** scripts when policies change
5. **Pre-commit hooks** enforce policies before commits

## Directory Structure

```
devcovenant/
├── __init__.py              # Package initialization
├── base.py                  # Base classes (PolicyCheck, PolicyFixer, etc.)
├── engine.py                # Main orchestration engine
├── parser.py                # Parse AGENTS.md for policy definitions
├── registry.py              # Track policy hashes and sync status
├── cli.py                   # Command-line interface
├── config.yaml              # Configuration
├── registry.json            # Policy hash registry (auto-generated)
├── policy_scripts/          # Policy check implementations
│   ├── changelog_coverage.py
│   ├── no_git_conflict_markers.py
│   ├── line_length_limit.py
│   └── last_updated_placement.py
├── fixers/                  # Automated policy fixers
│   └── last_updated_placement.py
├── tests/                   # Tests for devcovenant
│   ├── test_parser.py
│   ├── test_engine.py
│   └── test_policies/
└── hooks/                   # Git hook integration
    └── pre_commit.py
```

## Usage

### For AI Agents

When starting work on the repository:

```bash
python devcovenant_check.py check --mode=startup
```

This will:
- Check if any policies have been updated
- Report which policy scripts need to be updated
- Block until sync issues are resolved

### For Developers

Run all checks:
```bash
python devcovenant_check.py check
```

Run checks with auto-fix:
```bash
python devcovenant_check.py check --fix
```

Check policy sync status:
```bash
python devcovenant_check.py sync
```

Run tests:
```bash
python devcovenant_check.py test
```

### As Part of Lint

Add to your lint command:
```bash
python devcovenant_check.py check --mode=lint
```

### Pre-commit Hook

The pre-commit hook automatically runs before each commit. To install:

```bash
chmod +x devcovenant/hooks/pre_commit.py
ln -s ../../devcovenant/hooks/pre_commit.py .git/hooks/pre-commit
```

Or add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: devcovenant
        name: DevCovenant Policy Checks
        entry: python devcovenant/hooks/pre_commit.py
        language: system
        pass_filenames: false
```

## Writing Policy Scripts

Policy scripts must:
1. Inherit from `PolicyCheck` base class
2. Set `policy_id` to match the ID in AGENTS.md
3. Implement `check(context) -> List[Violation]` method
4. Have corresponding tests in `tests/test_policies/`

Example:
```python
from devcovenant.base import CheckContext, PolicyCheck, Violation

class MyPolicyCheck(PolicyCheck):
    policy_id = "my-policy"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        violations = []
        # Check logic here
        return violations
```

## Writing Fixers

Fixers are optional. They must:
1. Inherit from `PolicyFixer` base class
2. Set `policy_id` to match the policy
3. Implement `can_fix(violation) -> bool`
4. Implement `fix(violation) -> FixResult`

Example:
```python
from devcovenant.base import FixResult, PolicyFixer, Violation

class MyPolicyFixer(PolicyFixer):
    policy_id = "my-policy"

    def can_fix(self, violation: Violation) -> bool:
        return violation.policy_id == self.policy_id

    def fix(self, violation: Violation) -> FixResult:
        # Fix logic here
        return FixResult(success=True, message="Fixed!")
```

## Configuration

Edit `devcovenant/config.yaml` to configure:

- `master_update`: Allow AI to update policy scripts (default: true)
- `fix_threshold`: Minimum severity to auto-fix (default: warning)
- `fail_threshold`: Minimum severity to block commit (default: error)
- `auto_fix_enabled`: Enable auto-fixers (default: true)

## Severity Levels

- **critical**: Always blocks, must be fixed immediately
- **error**: Blocks at 'error' threshold or higher
- **warning**: Blocks at 'warning' threshold or higher
- **info**: Informational only, never blocks

## Policy Status Values

- **new**: Policy is new, script needs to be created
- **active**: Policy is active and enforced
- **updated**: Policy has been updated, script needs sync
- **deprecated**: Policy is deprecated, warnings only
- **deleted**: Policy is deleted, script should be removed

## Self-Enforcement

DevCovenant enforces policies on itself:
- All policy scripts must have tests
- Tests must achieve 80% coverage
- Scripts must follow the PolicyCheck interface
- All code must pass lint checks

## Troubleshooting

**"Policy sync required" message:**
- A policy in AGENTS.md has been updated
- Update the corresponding script in `policy_scripts/`
- Run tests to verify
- Re-run devcovenant to update hash

**"Script missing" error:**
- Create the policy script using the template above
- Add tests
- Run devcovenant again

**Hash mismatch:**
- Policy text or script has changed
- If intentional, verify both are correct
- Re-run devcovenant to recalculate hash

## Development

Run all devcovenant tests:
```bash
pytest devcovenant/tests/ -v
```

Run tests for a specific policy:
```bash
pytest devcovenant/tests/test_policies/test_<policy_id>.py -v
```

Check coverage:
```bash
pytest devcovenant/tests/ --cov=devcovenant --cov-report=html
```
