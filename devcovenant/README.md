# DevCovenant - Self-Enforcing Policy System

**Version:** 1.0.0 **Status:** Production Ready **License:** MIT (when
standalone)

DevCovenant is an autonomous, AI-driven policy enforcement system that
maintains perfect consistency between human-readable policies and automated
compliance checks. Originally developed for the Copernican Suite, it's designed
to be a standalone system that can be integrated into any repository.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Architecture](#architecture)
4. [Installation & Integration](#installation--integration)
5. [Usage](#usage)
   - [For AI Agents](#for-ai-agents)
   - [For Human Developers](#for-human-developers)
   - [Integration Points](#integration-points)
6. [Policy Definition Format](#policy-definition-format)
7. [Writing Policy Scripts](#writing-policy-scripts)
8. [Writing Fixers](#writing-fixers)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)
13. [Extending DevCovenant](#extending-devcovenant)
14. [Contributing](#contributing)

---

## Overview

### The Problem

Development policies are typically documented in one place (README,
CONTRIBUTING.md, etc.) but enforced separately through linters, pre-commit
hooks, and CI checks. This creates several issues:

- **Drift**: Policy documentation and enforcement logic diverge over time
- **Manual Sync**: Developers must manually update enforcement scripts when
  policies change
- **Inconsistency**: Different tools enforce different interpretations of the
  same policy
- **Discovery**: New contributors struggle to find and understand all policies

### The Solution

DevCovenant solves this by making policies **self-enforcing**:

1. **Single Source of Truth**: Policies are defined in plain English in your
   main documentation file (AGENTS.md, CONTRIBUTING.md, etc.)
2. **Structured Metadata**: Each policy has machine-readable metadata
   (severity, status, auto-fix capability)
3. **Automated Sync**: AI agents automatically generate and update enforcement
   scripts from policy text
4. **Hash Verification**: Cryptographic hashes ensure policies and scripts stay
   in sync
5. **Continuous Enforcement**: Pre-commit hooks, lint, and CI all use the same
   policy engine

### Key Benefits

- ✅ **Zero Drift**: Policies and enforcement are always synchronized
- ✅ **AI-Maintained**: Policy scripts update automatically when policies change
- ✅ **Developer-Friendly**: Clear, actionable error messages guide fixes
- ✅ **Self-Documenting**: Every policy is documented where it's defined
- ✅ **Flexible**: Policies can warn, block, or auto-fix
- ✅ **Extensible**: Easy to add new policies and fixers
- ✅ **Portable**: Can be integrated into any repository

---

## Key Concepts

### Policy Definition

A **policy** is a development rule defined in structured format within your
documentation:

```markdown
## Policy: No Hardcoded Secrets

```policy-def
id: no-hardcoded-secrets status: active severity: critical auto_fix: false
updated: false applies_to: *.py,*.js,*.yml
```

Never commit secrets, API keys, passwords, or tokens to the repository.
Use environment variables or secret management services instead.

Examples of violations:
- `API_KEY = "sk_live_1234..."`
- `password = "admin123"`

Fix by using environment variables:
- `API_KEY = os.getenv("API_KEY")`

---
```

### Policy Script

A **policy script** is a Python module that checks code for compliance:

```python
from devcovenant.base import CheckContext, PolicyCheck, Violation

class NoHardcodedSecretsCheck(PolicyCheck):
    policy_id = "no-hardcoded-secrets"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        violations = []
        # Check for patterns like API_KEY = "..."
        # Return violations found
        return violations
```

### Hash Registry

DevCovenant maintains a **registry** that stores cryptographic hashes of:
- Policy text (from documentation)
- Policy script (Python implementation)

When the hash of policy text changes but the script hash hasn't, DevCovenant
detects this mismatch and alerts the AI to update the script.

### Severity Levels

Policies have severity levels that control when they block operations:

| Severity   | Description                 | Blocks At |
|------------|-----------------------------|-----------|
| `critical` | Must fix immediately        | Always    |
| `error`    | Blocks at error threshold   | error+    |
| `warning`  | Blocks at warning threshold | warning+  |
| `info`     | Informational only          | Never     |

`error+` covers error, warning and info runs. `warning+` covers warning and
info runs.

### Status Values

Policies have status values that control their lifecycle:

| Status       | Description                 | AI Action          |
|--------------|-----------------------------|--------------------|
| `new`        | Policy is newly added       | Create script/tests |
| `active`     | Policy is enforced          | None               |
| `updated`    | Policy text has changed     | Update script/tests |
| `deprecated` | Policy is being phased out  | None (warn only)   |
| `deleted`    | Policy has been removed     | Remove script/tests |

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENTS.md (or similar)                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Policy Definitions (Human-Readable)               │    │
│  │  • Plain English descriptions                       │    │
│  │  • Machine-readable metadata                        │    │
│  │  • Examples and guidance                            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    PolicyParser                             │
│  • Extracts policy definitions                              │
│  • Parses metadata blocks                                   │
│  • Calculates policy text hashes                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    PolicyRegistry                           │
│  • Tracks policy-script hashes                              │
│  • Detects sync mismatches                                  │
│  • Maintains audit trail                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   DevCovenantEngine                         │
│  • Orchestrates policy checks                               │
│  • Reports violations                                       │
│  • Manages auto-fixing                                      │
│  • Determines block/pass status                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────┬──────────────────────┬──────────────┐
│   Policy Scripts     │      Fixers          │   Tests      │
│  • Check compliance  │  • Auto-fix issues   │  • Verify    │
│  • Return violations │  • Return results    │  • Coverage  │
└──────────────────────┴──────────────────────┴──────────────┘
```

### Directory Structure

```
devcovenant/
├── __init__.py              # Package initialization
├── README.md                # This file
├── config.yaml              # Engine + policy configuration
│                            # (incl. ignore globs)
├── registry.json            # Policy hash registry (auto-generated)
│
├── base.py                  # Base classes and data structures
│   ├── PolicyCheck          # Base class for policy checks
│   ├── PolicyFixer          # Base class for fixers
│   ├── CheckContext         # Context passed to checks
│   ├── Violation            # Represents a policy violation
│   └── FixResult            # Result of a fix operation
│
├── parser.py                # Parse AGENTS.md for policy definitions
│   └── PolicyParser         # Extracts and parses policies
│
├── registry.py              # Track policy hashes and sync status
│   └── PolicyRegistry       # Manages hash registry
│
├── engine.py                # Main orchestration engine
│   └── DevCovenantEngine    # Coordinates all operations
│
├── cli.py                   # Command-line interface
│   └── main()               # CLI entry point
│
├── policy_scripts/          # Policy check implementations
│   ├── __init__.py
│   ├── changelog-coverage.py
│   ├── no-git-conflict-markers.py
│   ├── line-length-limit.py
│   ├── last-updated-placement.py
│   └── devcov-self-enforcement.py
│
├── fixers/                  # Automated policy fixers
│   ├── __init__.py
│   └── last-updated-placement.py
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_parser.py       # Parser tests
│   ├── test_engine.py       # Engine tests
│   ├── test_registry.py     # Registry tests
│   ├── test_policies/       # Policy-specific tests
│   │   ├── test_changelog-coverage.py
│   │   └── test_no-git-conflict-markers.py
│   └── fixtures/            # Test data and fixtures
│
└── hooks/                   # Git hook integration
    └── pre_commit.py        # Pre-commit hook script
```

---

## Installation & Integration

### Standalone Installation

1. **Copy DevCovenant to your repository:**

```bash
# From the source repository
cp -r devcovenant/ /path/to/your/repo/

# Or clone as a submodule
git submodule add https://github.com/your-org/devcovenant.git
```

2. **Install dependencies:**

```bash
pip install pyyaml  # For config parsing
```

3. **Create wrapper script:**

```bash
# Create devcovenant_check.py in your repo root
cat > devcovenant_check.py << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from devcovenant.cli import main

if __name__ == "__main__":
    main()
EOF

chmod +x devcovenant_check.py
```

### Integration Steps

#### 1. Define Your Policy Documentation File

DevCovenant reads policies from a markdown file (typically `AGENTS.md`,
`CONTRIBUTING.md`, or `POLICIES.md`). Update `devcovenant/parser.py` if using a
different filename:

```python
# In parser.py
self.agents_md_path = repo_root / "CONTRIBUTING.md"  # Change as needed
```

Or pass it as a parameter:

```python
parser = PolicyParser(Path("CONTRIBUTING.md"))
```

#### 2. Add Policy Definitions

Add policy definitions to your documentation file using the structured format:

```markdown
## Development Policies

## Policy: Your Policy Name

```policy-def
id: your-policy-id status: active severity: error auto_fix: false updated:
false applies_to: *.py
```

Policy description here...

---
```

#### 3. Initialize Registry

Run DevCovenant for the first time to initialize the registry:

```bash
python devcovenant_check.py sync
```

#### 4. Set Up Pre-commit Hook

**Option A: Manual installation**

```bash
chmod +x devcovenant/hooks/pre_commit.py
ln -s ../../devcovenant/hooks/pre_commit.py .git/hooks/pre-commit
```

**Option B: Using pre-commit framework**

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: devcovenant
        name: DevCovenant Policy Checks
        entry: python devcovenant_check.py check --mode=pre-commit
        language: system
        pass_filenames: false
        always_run: true
```

#### 5. Integrate with CI/CD

Add to your CI workflow (GitHub Actions example):

```yaml
# .github/workflows/ci.yml
- name: Check DevCovenant Policies
  run: |
    python devcovenant_check.py check --mode=lint
```

---

## Usage

### For AI Agents

AI agents should run DevCovenant at specific points in their workflow.

#### At the Start of Every Work Session

**REQUIRED**: Before beginning any work on the repository:

```bash
python devcovenant_check.py check --mode=startup
```

This ensures:
- All policies are synchronized with their scripts
- Any updated policies trigger script updates
- The AI is aware of all current policies

**What happens:**
- DevCovenant parses all policy definitions
- Checks for hash mismatches (policy updated but script hasn't been)
- Reports sync issues with clear instructions
- AI updates any out-of-sync scripts BEFORE proceeding with user's request

**Example workflow:**

```bash
# 1. AI starts work session
$ git status
$ cat AGENTS.md  # Read policies (standard practice)
$ python devcovenant_check.py check --mode=startup

# 2. DevCovenant detects updated policy
🔄 POLICY SYNC REQUIRED

Policy 'no-hardcoded-secrets' has been updated.
The policy script is out of sync and must be updated FIRST.

📋 Updated Policy Definition:
[policy text shown]

🎯 Action Required:
1. Update: devcovenant/policy_scripts/no-hardcoded-secrets.py
2. Implement the policy above
3. Add/update tests
4. Run tests
5. Re-run devcovenant

⚠️  Complete this BEFORE working on user's request.

# 3. AI updates the script
$ vi devcovenant/policy_scripts/no-hardcoded-secrets.py
$ vi devcovenant/tests/test_policies/test_no-hardcoded-secrets.py
$ pytest devcovenant/tests/test_policies/test_no-hardcoded-secrets.py -v
$ python devcovenant_check.py check --mode=startup

# 4. Now hash matches, proceed with user's request
✅ All policies are in sync!
```

#### Before Committing Code

DevCovenant runs automatically via pre-commit hook, but AI can also run
manually:

```bash
python devcovenant_check.py check --mode=pre-commit
```

This checks only changed files for faster performance.

#### At the End of a Work Session

**RECOMMENDED**: Before finishing work:

```bash
python devcovenant_check.py check --mode=lint
```

This performs a full check of all files to ensure nothing was missed.

### For Human Developers

#### Quick Start

```bash
# Check all policies
python devcovenant_check.py check

# Check and auto-fix violations
python devcovenant_check.py check --fix

# Check only policy sync status
python devcovenant_check.py sync

# Run devcovenant's own tests
python devcovenant_check.py test
```

#### Common Workflows

**Before starting work:**

```bash
python devcovenant_check.py sync
```

**During development:**

```bash
# Run checks frequently
python devcovenant_check.py check

# Auto-fix when possible
python devcovenant_check.py check --fix
```

**Before committing:**

Pre-commit hook runs automatically, but you can run manually:

```bash
python devcovenant_check.py check --mode=pre-commit
```

**After updating a policy:**

```bash
# 1. Edit AGENTS.md, set updated: true
# 2. Update the corresponding script
# 3. Update tests
# 4. Run tests
pytest devcovenant/tests/test_policies/test_<policy_id>.py -v

# 5. Re-run devcovenant (hash updates automatically)
python devcovenant_check.py sync
```

### Integration Points

#### As Part of Lint

Add to your existing lint script:

```bash
#!/bin/bash
# lint.sh

python devcovenant_check.py check --mode=lint
ruff check .
mypy .
pytest --quick
```

#### In CI/CD

```bash
# Run in CI with strict mode
python devcovenant_check.py check --mode=lint

# Exit code 0 = pass, 1 = violations found
```

#### In IDE/Editor

Configure your IDE to run DevCovenant on save or as a task:

**VS Code** (`tasks.json`):

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "DevCovenant Check",
      "type": "shell",
      "command": "python devcovenant_check.py check",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

---

## Policy Definition Format

### Complete Format Specification

```markdown
## Policy: [Human-Readable Name]

```policy-def
id: [unique-identifier-kebab-case] status:
[new|active|updated|deprecated|deleted] severity: [critical|error|warning|info]
auto_fix: [true|false] updated: [true|false] applies_to: [file-pattern]
(optional) hash: [sha256-hash] (optional, auto-maintained)
```

[Detailed policy description in plain English]

**Why this policy exists:**
[Rationale and motivation]

**Examples of violations:**
- [Example 1]
- [Example 2]

**How to fix:**
- [Fix approach 1]
- [Fix approach 2]

**Exceptions:**
- [When this policy doesn't apply]

---
```

### Field Descriptions

- **`id`** *(required)* — Unique identifier in kebab-case (for example,
  `no-hardcoded-secrets`).
- **`status`** *(required)* — Lifecycle flag: `new`, `active`, `updated`,
  `deprecated` or `deleted`.
- **`severity`** *(required)* — Enforcement tier: `critical`, `error`,
  `warning` or `info`.
- **`auto_fix`** *(required)* — Whether the policy offers an auto-fix
  helper (`true` or `false`).
- **`updated`** *(required)* — Set to `true` whenever the policy text changes
  so the AI knows to refresh the script and tests.
- **`applies_to`** *(optional)* — Glob or path expression defining the files
  to check (for example, `*.py` or `src/**/*.js`).
- **`hash`** *(optional)* — SHA256 hash of the policy text plus the script,
  maintained by DevCovenant.
- **`enforcement`** *(optional)* — `active` for blocking checks or
  `fiducial` for informational reminders.
- **`waiver`** *(optional)* — `true` when deviations can be recorded in
  `.devcovenant/waivers/<policy-id>.txt`.

### Metadata-driven options

Any extra `key: value` pairs that appear inside the `policy-def` code fence
become structured policy options. Comma-separated values (for example,
`required_commands: pytest,python -m unittest discover`) are parsed into lists,
booleans recognize `true`/`false`, and numeric strings become integers/floats.
DevCovenant feeds those options into each policy check via
`PolicyCheck.get_option("key", default)`. Repository-level overrides defined in
`devcovenant/config.yaml` still win over metadata, so AGENTS.md describes the
defaults while `config.yaml` captures local deviations.

Fiducial policies emit informational reminders without blocking commits (you
can promote them to `active` enforcement once the reminders are addressed).
Waiver-enabled policies (e.g., `read-only-directories`) expect the agent to add
the approved exceptions to the matching file under `.devcovenant/waivers/`
before editing the protected paths. The read-only directory patterns themselves
now live in the policy metadata (e.g., `protected_globs`, `exempt_globs`), so
AGENTS.md stays the single source for scope defaults while waivers cover one-
off edits.

The `docstring-and-comment-coverage` policy always scans any `.py` file outside
`tests/` (in addition to the staged files), so running DevCovenant in `lint` or
`startup` mode inspects the entire workspace for missing docstrings/comments.

#### Standard selector metadata

Policies filtering files by path, suffix or glob should reuse the shared
selector keys:

- `include_suffixes`, `include_prefixes`, `include_globs` — positive filters.
- `exclude_suffixes`, `exclude_prefixes`, `exclude_globs` — negative filters.
- `force_include_globs` — overrides exclusions for specific patterns.
- `watch_files`, `watch_dirs` — explicit per-file/per-directory watch lists.
- `protected_globs` / `exempt_globs`, `guarded_paths`, `user_visible_dirs` —
  policy-specific selectors that still follow the same matching semantics for
  read-only, security or documentation scopes.

Each entry accepts either a comma-separated string (e.g.,
`include_suffixes: .py,.md`) or a YAML list. Paths and globs use forward
slashes so the metadata reads consistently across platforms. The new
`devcovenant/selectors.py` helper exposes a `SelectorSet` that policies can
instantiate via `SelectorSet.from_policy(self)` to evaluate all selectors with
uniform precedence rules (force-include > exclude > include). Tests in
`devcovenant/tests/test_selectors.py` demonstrate the expected behavior. Future
policies should default to these keys so contributors can reuse the same
configuration surface everywhere.

### Example Policies

**Critical Policy (Always Blocks):**

```markdown
## Policy: No Secrets in Code

```policy-def
id: no-secrets-in-code status: active severity: critical auto_fix: false
updated: false applies_to: *
```

Never commit secrets, API keys, passwords, or tokens.

**Why:** Exposed secrets can lead to security breaches.

**Examples of violations:**
- `API_KEY = "sk_live_1234567890"`
- `password = "admin123"`

**How to fix:**
Use environment variables:
- `API_KEY = os.getenv("API_KEY")`

---
```

**Warning Policy (Auto-fixable):**

```markdown
## Policy: Trailing Whitespace

```policy-def
id: no-trailing-whitespace status: active severity: warning auto_fix: true
updated: false applies_to: *.py,*.js,*.md
```

Remove trailing whitespace from lines.

**Why:** Trailing whitespace causes unnecessary diff noise.

**How to fix:**
Auto-fix available: `python devcovenant_check.py check --fix`

---
```

---

## Writing Policy Scripts

### Basic Template

```python
"""
Policy: [Policy Name]

[Brief description of what this policy checks]
"""

from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

class [PolicyName]Check(PolicyCheck):
    """
    [Detailed docstring explaining what this policy checks for]

    This policy ensures that [specific requirement].

    Examples of violations:
    - [Example 1]
    - [Example 2]

    Examples of compliant code:
    - [Example 1]
    - [Example 2]
    """

    policy_id = "[policy-id-from-AGENTS.md]"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check files for policy compliance.

        Args:
            context: Check context containing:
                - repo_root: Repository root directory
                - changed_files: List of changed files (pre-commit mode)
                - all_files: List of all files (lint mode)
                - mode: Check mode (startup, lint, pre-commit, normal)

        Returns:
            List of Violation objects (empty if no violations)
        """
        violations = []

        # Determine which files to check
        files_to_check = (
            context.changed_files
            if context.changed_files
            else context.all_files
        )

        for file_path in files_to_check:
            # Apply file filtering based on policy
            if not self._should_check_file(file_path):
                continue

            # Perform checks
            file_violations = self._check_file(file_path, context)
            violations.extend(file_violations)

        return violations

    def _should_check_file(self, file_path: Path) -> bool:
        """
        Determine if this file should be checked.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be checked
        """
        # Example: Only check Python files
        return file_path.suffix == ".py"

    def _check_file(
        self, file_path: Path, context: CheckContext
    ) -> List[Violation]:
        """
        Check a single file for violations.

        Args:
            file_path: Path to the file
            context: Check context

        Returns:
            List of violations found in this file
        """
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Perform actual checks here
            # Example: Check for a specific pattern
            if "FORBIDDEN_PATTERN" in content:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=file_path,
                        line_number=None,  # Set if known
                        message="Forbidden pattern detected",
                        suggestion=(
                            "Remove the forbidden pattern and use an "
                            "approved alternative"
                        ),
                        can_auto_fix=False,
                    )
                )

        except Exception as e:
            # Handle errors gracefully
            pass

        return violations
```

Policy checks automatically receive any metadata and configuration overrides
defined for their policy. Retrieve them with
`self.get_option("option-name", default)`—metadata from `AGENTS.md` supplies
the defaults, while `devcovenant/config.yaml` entries override those values
when present. This keeps policy logic focused on validation instead of
file-path plumbing.

### Advanced Example: Line-by-Line Checking

```python
from devcovenant.base import CheckContext, PolicyCheck, Violation

class NoTodoCommentsCheck(PolicyCheck):
    """Check for TODO comments in production code."""

    policy_id = "no-todo-comments"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        violations = []

        files_to_check = context.changed_files or context.all_files

        for file_path in files_to_check:
            if not file_path.suffix in [".py", ".js", ".ts"]:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, start=1):
                    if "TODO" in line or "FIXME" in line:
                        violations.append(
                            Violation(
                                policy_id=self.policy_id,
                                severity="warning",
                                file_path=file_path,
                                line_number=line_num,
                                message=(
                                    f"TODO/FIXME comment found: {line.strip()}"
                                ),
                                suggestion=(
                                    "Create a GitHub issue and reference it "
                                    "in the comment"
                                ),
                                can_auto_fix=False,
                            )
                        )
            except Exception:
                pass

        return violations
```

### Testing Policy Scripts

Every policy script must have corresponding tests:

```python
# devcovenant/tests/test_policies/test_no-todo-comments.py

import tempfile
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.no_todo_comments import NoTodoCommentsCheck

def test_detects_todo_comments():
    """Test that TODO comments are detected."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("# TODO: Fix this later\ndef foo():\n    pass\n")
        temp_path = Path(f.name)

    try:
        checker = NoTodoCommentsCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) == 1
        assert violations[0].policy_id == "no-todo-comments"
        assert violations[0].line_number == 1
    finally:
        temp_path.unlink()

def test_clean_file_passes():
    """Test that files without TODOs pass."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("def foo():\n    return 42\n")
        temp_path = Path(f.name)

    try:
        checker = NoTodoCommentsCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) == 0
    finally:
        temp_path.unlink()
```

---

## Writing Fixers

Fixers are optional components that automatically fix policy violations.

### Fixer Template

```python
"""
Fixer for [Policy Name]

[Brief description of what this fixer does]
"""

from pathlib import Path

from devcovenant.base import FixResult, PolicyFixer, Violation

class [PolicyName]Fixer(PolicyFixer):
    """
    Automatically fix [policy name] violations.

    This fixer [what it does and how].
    """

    policy_id = "[policy-id]"

    def can_fix(self, violation: Violation) -> bool:
        """
        Determine if this violation can be fixed.

        Args:
            violation: The violation to check

        Returns:
            True if this fixer can handle this violation
        """
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
            and violation.can_auto_fix
        )

    def fix(self, violation: Violation) -> FixResult:
        """
        Fix the violation.

        Args:
            violation: The violation to fix

        Returns:
            FixResult indicating success/failure and what was changed
        """
        if not violation.file_path:
            return FixResult(success=False, message="No file path provided")

        try:
            # Read the file
            with open(violation.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Apply fix
            original_content = content
            content = self._apply_fix(content, violation)

            # Write back if changed
            if content != original_content:
                with open(violation.file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return FixResult(
                    success=True,
                    message=(
                        f"Fixed {violation.policy_id} in "
                        f"{violation.file_path}"
                    ),
                    files_modified=[violation.file_path],
                )
            else:
                return FixResult(success=True, message="No changes needed")

        except Exception as e:
            return FixResult(success=False, message=f"Fix failed: {e}")

    def _apply_fix(self, content: str, violation: Violation) -> str:
        """
        Apply the actual fix to the content.

        Args:
            content: File content
            violation: The violation

        Returns:
            Fixed content
        """
        # Implement fix logic here
        return content
```

### Example: Trailing Whitespace Fixer

```python
import re

from devcovenant.base import FixResult, PolicyFixer, Violation

class TrailingWhitespaceFixer(PolicyFixer):
    """Remove trailing whitespace from lines."""

    policy_id = "no-trailing-whitespace"

    def can_fix(self, violation: Violation) -> bool:
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
        )

    def fix(self, violation: Violation) -> FixResult:
        if not violation.file_path:
            return FixResult(success=False, message="No file path")

        try:
            with open(violation.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove trailing whitespace from each line
            fixed_lines = [
                re.sub(r"\s+$", "", line, flags=re.MULTILINE) for line in lines
            ]

            # Write back
            with open(violation.file_path, "w", encoding="utf-8") as f:
                f.writelines(fixed_lines)

            return FixResult(
                success=True,
                message=(
                    "Removed trailing whitespace from "
                    f"{violation.file_path}"
                ),
                files_modified=[violation.file_path],
            )

        except Exception as e:
            return FixResult(success=False, message=f"Failed: {e}")
```

### Bundled Auto-Fixers

`devcovenant check --fix` now loads every fixer under `devcovenant/fixers/`
and re-runs the policy suite after applying changes. The following fixers ship
with the suite:

- `no-future-dates` – rewrites future timestamps to the current UTC date.
- `raw-string-escapes` – double-escapes backslashes in offending literals.
- `start-script-parity` – mirrors the edited launcher into the other
  `start.*` entry points.
- `dependency-license-sync` – appends dated notes to
  `THIRD_PARTY_LICENSES.md` and touches the `licenses/` directory whenever
  dependencies change.
- `start-script-guardrails` – injects the canonical `pkg_notice` /
  `sudo -k` snippets (or `PKG_NOTICE` on Windows) so password prompts remain.
- `last-updated-placement` – strips stray “Last Updated” lines from files that
  are not on the allowlist.

Policies declaring `auto_fix: true` in `AGENTS.md` should expose enough context
via the `Violation.context` dictionary for their fixers to work without
inspecting git metadata.

---

## Configuration

### Configuration File

Edit `devcovenant/config.yaml` to tune DevCovenant for your repository. The
file now exposes **global paths**, **engine options** and **per-policy blocks**
so every formerly hard-coded identifier can be overridden:

```yaml
paths:
  policy_definitions: AGENTS.md
  registry_file: devcovenant/registry.json

engine:
  master_update: true
  fix_threshold: warning
  fail_threshold: error
  auto_fix_enabled: true
  parallel_checks: true
  verbose: true
  file_suffixes:
    - .py
    - .md
    - .yml
    - .yaml

self_enforcement:
  enabled: true
  policy_prefix: "devcov-"

hooks:
  pre_commit: true
  pre_push: false

reporting:
  show_policy_links: true
  audit_trail: true
  use_colors: true

policies:
  devflow-run-gates:
    test_status_file: devcovenant/test_status.json
    required_commands:
      - pytest
      - python -m unittest discover
    code_extensions:
      - .py
      - .pyi
  changelog-coverage:
    main_changelog: CHANGELOG.md
    skipped_files:
      - CHANGELOG.md
    collections:
      - prefix: rng_minigames/
        changelog: rng_minigames/CHANGELOG.md
        exclusive: true
  # ...see below for every policy-specific option
```

### Configuration Options Explained

- **`master_update`** *(default `true`)* — Allows the AI to update scripts
  automatically.
- **`fix_threshold`** *(default `warning`)* — Auto-fixes issues at this
  severity and above.
- **`fail_threshold`** *(default `error`)* — Blocks runs when violations at
  this level (or worse) appear.
- **`auto_fix_enabled`** *(default `true`)* — Toggles all auto-fixers.
- **`parallel_checks`** *(default `true`)* — Runs policy checks in parallel
  for faster feedback.
- **`file_suffixes`** *(default `[.py, .md, .yml, .yaml]`)* — Controls which
  files are scanned when building the repository inventory for policy checks.
- **`verbose`** *(default `true`)* — Prints detailed progress messages.
- **`self_enforcement`** *(default `true`)* — Enables DevCovenant's
  self-checks.
- **`pre_commit`** *(default `true`)* — Installs the pre-commit hook.
- **`show_policy_links`** *(default `true`)* — Adds documentation links to
  violation messages.
- **`audit_trail`** *(default `true`)* — Tracks policy updates in the registry.
- **`use_colors`** *(default `true`)* — Emits ANSI color codes in the CLI
  output.
- **`paths.policy_definitions`** *(default `AGENTS.md`)* — Points the parser to
  your canonical policy document.
- **`paths.registry_file`** *(default `devcovenant/registry.json`)* — Allows
  relocating the hash registry (for monorepos or workspace layouts).
- **`policies.<policy-id>`** — Every policy can be tuned without editing the
  Python script (see below).

### Per-policy options

Each policy inherits defaults declared next to its `policy-def` block in
`AGENTS.md`. Those metadata entries become runtime options exposed through
`PolicyCheck.get_option()`, and any overrides placed under
`policies.<policy-id>` in `devcovenant/config.yaml` take precedence. The list
below documents the recognized keys for each policy so you know what can be
tuned without editing Python scripts:

- **`changelog-coverage`**  
  - `main_changelog`: root changelog path.  
  - `skipped_files`: filenames ignored by the policy.  
  - `collections`: list describing additional changelog partitions. When using
    metadata, encode each entry as `prefix:changelog:exclusive` (for example,
    `rng_minigames/:rng_minigames/CHANGELOG.md:true`).
- **`documentation-growth-tracking`**  
  - `user_visible_dirs` / `user_visible_files`: directories and files that
    trigger documentation reminders.
- **`dependency-license-sync`**  
  - `dependency_files`: manifest files guarded by the policy.  
  - `third_party_file`: consolidated license table.  
  - `licenses_dir`: directory containing per-package licenses.  
  - `report_heading`: heading text that marks the “License Report” section.
- **`line-length-limit`**  
  - `max_length`: column limit (defaults to 79).  
  - `include_suffixes`: file extensions that must obey the limit (for example,
    `.py`, `.md`, `.rst`, `.txt`).  
- `exclude_prefixes`: directory prefixes that remain exempt (vendored trees,
    archived datasets, etc.).  
  - `force_include_globs`: glob patterns to re-include even when they live
    under a skipped prefix (the default brings `data/**/cosmo_parser_*.py`
    back into scope).
- **`docstring-and-comment-coverage`**  
  - `include_suffixes`: language extensions that should carry doc coverage
    (default: `.py`).  
- `exclude_prefixes`: relative path prefixes that are exempt (vendored code).  
  - `skip_components`: path components (such as `tests`) that should be ignored
    regardless of depth.
- **`devflow-run-gates`**  
  - `test_status_file`: JSON file storing the last test run (defaults to
    `devcovenant/test_status.json`).  
  - `required_commands`: lower-case strings that must appear in
    `commands`.  
  - `code_extensions`: extensions considered “code changes” for enforcing test
    runs.
- **`test-status-tracking`**  
  - `test_status_file`: mirrors the gate above.  
  - `watched_roots` / `watched_files`: modifications here require a fresh
    recorded test run.
- **`managed-venv`**  
  - `expected_virtualenvs`: list of repository-relative directories that host
    valid virtual environments (default: `.venv`).
- **`new-modules-need-tests`**  
  - `module_roots`: directories whose Python modules must ship with tests.  
  - `tests_root`: location that should change whenever modules are added or
    removed.
- **`no-print-in-library`**  
  - `target_roots`: runtime packages to scan for `print()` calls.  
  - `vendor_paths`: prefixes that should be ignored (vendored dependencies).  
  - `allowed_files`: explicit exceptions (e.g., a console output helper).
- **`read-only-directories`**  
  - `protected_globs`: gitignore-style globs that stay read-only.  
  - `exempt_globs`: glob patterns that remain editable even when they live
    under a protected tree (dataset parsers by default).
- **`security-compliance-notes`**  
  - `guarded_paths`: directories/files representing security-critical assets.  
  - `log_path`: markdown file that must record any guarded edits.
- **`start-script-guardrails`**  
  - `scripts`: list of `{path, required}` patterns that enforce sudo prompts,
    package manager notices, etc. When editing via metadata, encode entries as
    `path:snippet|snippet` and separate scripts with `;`.
- **`start-script-parity`**  
  - `scripts`: list of launcher names that must evolve together.
- **`version-sync`**  
  - `version_file`, `readme_file`, `citation_file`,
    `pyproject_file`: the documents compared for consistency.  
  - `runtime_entrypoints`: top-level Python files that should not hard-code the
    suite version.  
  - `runtime_roots`: package directories scanned for hard-coded versions.
- **`semantic-version-scope`**  
  - `version_file`, `changelog_file`, `ignored_prefixes`, `override_file`:
    fine-tune the SemVer scope policy for any repo layout.
- **`changelog-coverage`** (collections) and the other policies follow the same
  pattern: add a block under `policies.<policy-id>` and override whichever keys
  you need.

### Global Ignore List

Repository-wide exclusions now live under `ignore.patterns` in
`devcovenant/config.yaml`. Each entry accepts `.gitignore` syntax (one pattern
per line, `#` for comments) and is read by `CheckContext` before any policy
runs. Paths matching a pattern are removed from both `changed_files` and
`all_files`, so every policy automatically skips vendored code, generated
artifacts or other shared exclusions. Update the configuration whenever a new
global exclusion is required instead of duplicating logic inside each policy
script.

---

## Migration Playbook

DevCovenant ships inside Copernican, but the entire system is portable. Use
this playbook to install it elsewhere without re-writing any scripts:

1. **Copy the engine** – Add the `devcovenant/` directory to the destination
   repository (or install it as a package), commit
   `devcovenant/config.yaml`, and wire the CLI into your tooling (`python
   devcovenant/cli.py` or `pre-commit`).
2. **Point at your policies** – Update `paths.policy_definitions` to reference
   the host project’s canonical policy document (for example
   `CONTRIBUTING.md`) and relocate `paths.registry_file` as needed.
3. **Describe scopes declaratively** – Define each policy’s selectors inside
   its `policy-def` block using the standard keys (`include_*`, `exclude_*`,
   `force_include_globs`, `watch_files`, `protected_globs`, etc.). These
   defaults document the intended scope for humans and become runtime options
   for the engine.
4. **Override via config** – For repository-specific tweaks, add entries under
   `policies.<policy-id>` inside `devcovenant/config.yaml`. Config overrides
   always beat metadata, so forks can retarget a policy without editing the
   base document.
5. **Set global ignores** – Move any former `.gitignore`-style waivers into
   `ignore.patterns` so `CheckContext` prunes them before checks run.
6. **Regenerate the registry** – Run `pre-commit run --all-files` or `python
   devcovenant_check.py check --mode=startup` to produce the policy hashes,
   then update or add tests (see `devcovenant/tests/`) so the selectors and
   new scopes are covered.

With the metadata, config overrides and selector helpers aligned, DevCovenant
behaves exactly the same on any repository—the only customization lives in
declarative metadata rather than hard-coded constants.

### Semantic Version Scope Markers

Releases that bump `copernican_lib/VERSION` must also tag the newest changelog
section with `[semver:patch]`, `[semver:minor]` or `[semver:major]`. The
`semantic-version-scope` policy compares those markers to the difference
between the newest two `## Version` headers and blocks when the bump is smaller
than the declared scope. Use patch for backwards-compatible fixes, minor for
backwards-compatible features and major for breaking changes. Scope checks
ignore changes scoped entirely to `devcovenant/` or `rng_minigames/`, and
exceptional cases may specify `override=<level>` in
`.devcovenant/waivers/semantic-version-scope.txt`. Remove the override file as
soon as the release lands so future bumps use the changelog markers again. When
a changelog release is recorded, bump `copernican_lib/VERSION` in the same
commit and keep every `[semver:*]` tag in that release block at the same scope;
mixed scopes or changelog-only scope changes now trigger a policy violation.

---

## Testing

### Running Tests

```bash
# Run all devcovenant tests
pytest devcovenant/tests/ -v

# Run tests for a specific policy
pytest devcovenant/tests/test_policies/test_<policy_id>.py -v

# Run with coverage
pytest devcovenant/tests/ --cov=devcovenant --cov-report=html

# Run tests and show detailed output
pytest devcovenant/tests/ -v -s
```

### Test Requirements

All policy scripts must have tests that:

1. **Test positive cases** (code that should pass)
2. **Test negative cases** (code that should violate)
3. **Test edge cases** (boundary conditions)
4. **Achieve ≥80% coverage**

### Test Structure

```
devcovenant/tests/
├── __init__.py
├── test_parser.py          # Parser tests
├── test_engine.py          # Engine tests
├── test_registry.py        # Registry tests
├── test_policies/          # Policy-specific tests
│   ├── test_<policy_id>.py
│   └── ...
└── fixtures/               # Test data
    ├── sample_policy.md
    └── ...
```

---

## Troubleshooting

### Common Issues

#### "Policy sync required" Message

**Symptom:**
```
🔄 POLICY SYNC REQUIRED

Policy 'my-policy' has been updated.
The policy script is out of sync and must be updated FIRST.
```

**Cause:** Policy text in AGENTS.md was changed, but the script hasn't been
updated yet.

**Solution:**
1. Update the script in `devcovenant/policy_scripts/<policy_id>.py`
2. Update tests if needed
3. Run tests: `pytest devcovenant/tests/test_policies/test_<policy_id>.py -v`
4. Re-run DevCovenant: `python devcovenant_check.py sync`

#### "Script missing" Error

**Symptom:**
```
Policy 'new-policy' requires attention.
Issue: Script Missing
```

**Cause:** New policy was added to AGENTS.md but no script exists yet.

**Solution:**
1. Create script: `devcovenant/policy_scripts/<policy_id>.py`
2. Create tests: `devcovenant/tests/test_policies/test_<policy_id>.py`
3. Run tests
4. Re-run DevCovenant

#### Hash Mismatch

**Symptom:**
```
Hash mismatch detected for policy '<policy_id>'
```

**Cause:** Either policy text or script changed without updating the other.

**Solution:**
1. Review both policy text and script
2. Ensure they match
3. Run tests
4. Re-run DevCovenant (hash recalculates automatically)

#### Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'devcovenant'
```

**Cause:** DevCovenant not in Python path.

**Solution:**
```bash
# Run from repository root
cd /path/to/repo
python devcovenant_check.py check

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/repo
```

---

## Best Practices

### For Policy Authors

1. **Be Specific**: Write clear, unambiguous policy descriptions
2. **Provide Examples**: Include both violations and fixes
3. **Explain Why**: Document the rationale for each policy
4. **Set Appropriate Severity**: Reserve `critical` for security/data loss
   issues
5. **Enable Auto-fix When Possible**: Makes compliance easier
6. **Update Tests**: Always update tests when changing policies

### For Script Developers

1. **Follow the Template**: Use the provided policy script template
2. **Handle Errors Gracefully**: Don't crash on unexpected files
3. **Optimize Performance**: Only check relevant files
4. **Provide Clear Messages**: Help developers understand and fix violations
5. **Test Thoroughly**: Cover edge cases and error conditions
6. **Document Well**: Include comprehensive docstrings

### For AI Agents

1. **Always Run at Startup**: Check policies before starting work
2. **Prioritize Sync Issues**: Fix policy sync before user's request
3. **Run Comprehensive Tests**: Verify scripts work correctly
4. **Update Hashes**: Let DevCovenant recalculate hashes automatically
5. **Check Before Committing**: Run pre-commit check

### For Repository Maintainers

1. **Review Policy Changes**: Ensure policies align with project goals
2. **Monitor Violation Trends**: Identify common issues
3. **Adjust Thresholds**: Tune severity levels based on impact
4. **Document Exceptions**: Note when policies don't apply
5. **Keep Registry Clean**: Periodically review and update policies

---

## Extending DevCovenant

### Adding New Check Modes

To add a new check mode (e.g., `ci` mode):

1. **Update CLI** (`cli.py`):
```python
parser.add_argument(
    "--mode",
    choices=["startup", "lint", "pre-commit", "ci", "normal"],
    default="normal",
)
```

2. **Update Engine** (`engine.py`):
```python
def check(self, mode: str = "normal"):
    # Add mode-specific logic
    if mode == "ci":
        # CI-specific behavior
        pass
```

### Adding New Severity Levels

To add a new severity level (e.g., `style`):

1. **Update Documentation**: Add to severity level table
2. **Update Engine**: Add to severity mapping
3. **Update Config Schema**: Add as valid option

### Creating Plugins

DevCovenant can be extended with plugins:

```python
# devcovenant/plugins/my_plugin.py

class MyPlugin:
    def on_policy_check_start(self, policy_id):
        """Called before checking a policy."""
        pass

    def on_violation_found(self, violation):
        """Called when a violation is found."""
        pass
```

---

## Contributing

Contributions to DevCovenant are welcome!

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/devcovenant.git
cd devcovenant

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black ruff mypy

# Run tests
pytest tests/ -v
```

### Contribution Guidelines

1. **Follow Existing Patterns**: Match the style of existing code
2. **Add Tests**: All new features must include tests
3. **Update Documentation**: Keep README.md current
4. **Run Checks**: Ensure all tests and lint checks pass
5. **Write Clear Commits**: Use descriptive commit messages

### Roadmap

Future enhancements planned:

- [ ] Web dashboard for policy visualization
- [ ] Integration with more CI/CD systems
- [ ] Policy templates library
- [ ] Machine learning-based policy suggestions
- [ ] Multi-repository policy synchronization
- [ ] Browser extension for GitHub integration
- [ ] Slack/Discord notifications for violations
- [ ] Policy performance metrics and analytics

---

## License

DevCovenant is released under the MIT License (when standalone).

See LICENSE file for details.

---

## Support

For issues, questions, or contributions:

- **GitHub Issues**: https://github.com/your-org/devcovenant/issues
- **Documentation**: https://devcovenant.readthedocs.io
- **Email**: support@devcovenant.dev

---

**DevCovenant** - Making policies self-enforcing, one repository at a time.
