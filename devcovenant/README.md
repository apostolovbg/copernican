# DevCovenant - Self-Enforcing Policy System

**Version:** 1.0.0
**Status:** Production Ready
**License:** MIT (when standalone)

DevCovenant is an autonomous, AI-driven policy enforcement system that maintains perfect consistency between human-readable policies and automated compliance checks. Originally developed for the Copernican Suite, it's designed to be a standalone system that can be integrated into any repository.

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

Development policies are typically documented in one place (README, CONTRIBUTING.md, etc.) but enforced separately through linters, pre-commit hooks, and CI checks. This creates several issues:

- **Drift**: Policy documentation and enforcement logic diverge over time
- **Manual Sync**: Developers must manually update enforcement scripts when policies change
- **Inconsistency**: Different tools enforce different interpretations of the same policy
- **Discovery**: New contributors struggle to find and understand all policies

### The Solution

DevCovenant solves this by making policies **self-enforcing**:

1. **Single Source of Truth**: Policies are defined in plain English in your main documentation file (AGENTS.md, CONTRIBUTING.md, etc.)
2. **Structured Metadata**: Each policy has machine-readable metadata (severity, status, auto-fix capability)
3. **Automated Sync**: AI agents automatically generate and update enforcement scripts from policy text
4. **Hash Verification**: Cryptographic hashes ensure policies and scripts stay in sync
5. **Continuous Enforcement**: Pre-commit hooks, lint, and CI all use the same policy engine

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

A **policy** is a development rule defined in structured format within your documentation:

```markdown
## Policy: No Hardcoded Secrets

```policy-def
id: no-hardcoded-secrets
status: active
severity: critical
auto_fix: false
updated: false
applies_to: *.py,*.js,*.yml
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

When the hash of policy text changes but the script hash hasn't, DevCovenant detects this mismatch and alerts the AI to update the script.

### Severity Levels

Policies have severity levels that control when they block operations:

| Severity   | Description                        | Blocks At           |
|------------|------------------------------------|---------------------|
| `critical` | Must fix immediately, always blocks | Always              |
| `error`    | Should fix, blocks at error threshold | error, warning, info |
| `warning`  | Should address, blocks at warning threshold | warning, info       |
| `info`     | Informational only                 | Never               |

### Status Values

Policies have status values that control their lifecycle:

| Status       | Description                           | AI Action Required     |
|--------------|---------------------------------------|------------------------|
| `new`        | Policy is newly added                 | Create script & tests  |
| `active`     | Policy is active and enforced         | None                   |
| `updated`    | Policy text has changed               | Update script & tests  |
| `deprecated` | Policy is being phased out            | None (warnings only)   |
| `deleted`    | Policy has been removed               | Delete script & tests  |

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
├── config.yaml              # Configuration settings
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

DevCovenant reads policies from a markdown file (typically `AGENTS.md`, `CONTRIBUTING.md`, or `POLICIES.md`). Update `devcovenant/parser.py` if using a different filename:

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
id: your-policy-id
status: active
severity: error
auto_fix: false
updated: false
applies_to: *.py
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

DevCovenant runs automatically via pre-commit hook, but AI can also run manually:

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
id: [unique-identifier-kebab-case]
status: [new|active|updated|deprecated|deleted]
severity: [critical|error|warning|info]
auto_fix: [true|false]
updated: [true|false]
applies_to: [file-pattern] (optional)
hash: [sha256-hash] (optional, auto-maintained)
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

| Field        | Required | Description                                                                 |
|--------------|----------|-----------------------------------------------------------------------------|
| `id`         | Yes      | Unique identifier in kebab-case (e.g., `no-hardcoded-secrets`)              |
| `status`     | Yes      | Lifecycle status: `new`, `active`, `updated`, `deprecated`, `deleted`       |
| `severity`   | Yes      | Enforcement level: `critical`, `error`, `warning`, `info`                   |
| `auto_fix`   | Yes      | Whether auto-fixing is available: `true` or `false`                         |
| `updated`    | Yes      | Set to `true` when policy text changes, triggers AI script update           |
| `applies_to` | No       | File pattern (glob) this policy applies to, e.g., `*.py`, `src/**/*.js`     |
| `hash`       | No       | SHA256 hash of policy + script, auto-maintained by DevCovenant              |

### Example Policies

**Critical Policy (Always Blocks):**

```markdown
## Policy: No Secrets in Code

```policy-def
id: no-secrets-in-code
status: active
severity: critical
auto_fix: false
updated: false
applies_to: *
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
id: no-trailing-whitespace
status: active
severity: warning
auto_fix: true
updated: false
applies_to: *.py,*.js,*.md
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
        files_to_check = context.changed_files if context.changed_files else context.all_files

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

    def _check_file(self, file_path: Path, context: CheckContext) -> List[Violation]:
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
                        suggestion="Remove the forbidden pattern and use approved alternative",
                        can_auto_fix=False,
                    )
                )

        except Exception as e:
            # Handle errors gracefully
            pass

        return violations
```

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
                                message=f"TODO/FIXME comment found: {line.strip()}",
                                suggestion="Create a GitHub issue and reference it in the comment",
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# TODO: Fix this later\ndef foo():\n    pass\n")
        temp_path = Path(f.name)

    try:
        checker = NoTodoCommentsCheck()
        context = CheckContext(repo_root=temp_path.parent, all_files=[temp_path])
        violations = checker.check(context)

        assert len(violations) == 1
        assert violations[0].policy_id == "no-todo-comments"
        assert violations[0].line_number == 1
    finally:
        temp_path.unlink()

def test_clean_file_passes():
    """Test that files without TODOs pass."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 42\n")
        temp_path = Path(f.name)

    try:
        checker = NoTodoCommentsCheck()
        context = CheckContext(repo_root=temp_path.parent, all_files=[temp_path])
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
                    message=f"Fixed {violation.policy_id} in {violation.file_path}",
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
        return violation.policy_id == self.policy_id and violation.file_path is not None

    def fix(self, violation: Violation) -> FixResult:
        if not violation.file_path:
            return FixResult(success=False, message="No file path")

        try:
            with open(violation.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove trailing whitespace from each line
            fixed_lines = [re.sub(r"\s+$", "", line, flags=re.MULTILINE) for line in lines]

            # Write back
            with open(violation.file_path, "w", encoding="utf-8") as f:
                f.writelines(fixed_lines)

            return FixResult(
                success=True,
                message=f"Removed trailing whitespace from {violation.file_path}",
                files_modified=[violation.file_path],
            )

        except Exception as e:
            return FixResult(success=False, message=f"Failed: {e}")
```

---

## Configuration

### Configuration File

Edit `devcovenant/config.yaml`:

```yaml
# DevCovenant Configuration

engine:
  # Allow AI to automatically update policy scripts when hash mismatches detected
  master_update: true

  # Fix violations at this severity level and above before commit
  # Options: critical, error, warning, info
  fix_threshold: warning

  # Block commit if violations at this severity level or above
  # Options: critical, error, warning, info
  fail_threshold: error

  # Enable auto-fixers when available
  auto_fix_enabled: true

  # Run policy checks in parallel for performance
  parallel_checks: true

  # Verbose output for AI guidance
  verbose: true

self_enforcement:
  # devcovenant enforces its own policies on itself
  enabled: true

  # Prefix for devcovenant's own policies in AGENTS.md
  policy_prefix: "devcov-"

hooks:
  # Enable pre-commit hook
  pre_commit: true

  # Enable pre-push hook (optional)
  pre_push: false

reporting:
  # Show links to policy documentation in violation messages
  show_policy_links: true

  # Maintain audit trail of policy updates
  audit_trail: true

  # Use colored output (ANSI colors)
  use_colors: true
```

### Configuration Options Explained

| Option              | Default   | Description                                                    |
|---------------------|-----------|----------------------------------------------------------------|
| `master_update`     | `true`    | Allow AI to update scripts automatically                       |
| `fix_threshold`     | `warning` | Auto-fix violations at this level and above                    |
| `fail_threshold`    | `error`   | Block operations on violations at this level and above         |
| `auto_fix_enabled`  | `true`    | Enable auto-fixers globally                                    |
| `parallel_checks`   | `true`    | Run policy checks in parallel (faster)                         |
| `verbose`           | `true`    | Detailed output messages                                       |
| `self_enforcement`  | `true`    | DevCovenant checks itself                                      |
| `pre_commit`        | `true`    | Enable pre-commit hook                                         |
| `show_policy_links` | `true`    | Include links to policy docs in violation messages             |
| `audit_trail`       | `true`    | Track policy update history                                    |
| `use_colors`        | `true`    | ANSI color codes in output                                     |

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

**Cause:** Policy text in AGENTS.md was changed, but the script hasn't been updated yet.

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
4. **Set Appropriate Severity**: Reserve `critical` for security/data loss issues
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
