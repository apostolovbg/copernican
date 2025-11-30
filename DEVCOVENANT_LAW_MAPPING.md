# DevCovenant Policy-to-Law Mapping

This document tracks the transition from numbered Development Laws to self-enforcing DevCovenant policies.

## Overview

The Copernican Suite is transitioning from a numbered list of "Development Laws" to a modern **DevCovenant** self-governing policy system. DevCovenant policies are:
- **Self-enforcing**: Automatically checked by scripts
- **Self-documenting**: Policy text and enforcement code stay in sync
- **AI-maintained**: Hash verification ensures policies and scripts match

## Laws Converted to DevCovenant Policies

The following laws have been successfully converted to DevCovenant policies and **should be removed from the numbered law list**:

| Original Law # | Law Description | DevCovenant Policy | Status |
|----------------|-----------------|-------------------|---------|
| 1 | Summarize every change in CHANGELOG.md | `changelog-coverage` | ✅ Active in AGENTS.md |
| 4 | Refresh documentation and Last Updated markers on allowlisted surfaces | `last-updated-placement` | ✅ Active in AGENTS.md |
| 7 | Follow the Versioning Policy | `version-sync` | ⚠️ Implemented but NOT in AGENTS.md |
| 8 | Never insert Git conflict markers | `no-git-conflict-markers` | ✅ Active in AGENTS.md |
| 15 | Keep individual lines under 79 characters | `line-length-limit` | ✅ Active in AGENTS.md |
| 20 | Add tests alongside new functionality | `new-modules-need-tests` | ⚠️ Implemented but NOT in AGENTS.md |
| 24 | Validate every timestamp before recording it | `no-future-dates` | ⚠️ Implemented but NOT in AGENTS.md |

## Additional Policies Not Mapped to Laws

These policies exist but don't directly correspond to existing laws:

| Policy ID | Description | Purpose |
|-----------|-------------|---------|
| `devcov-self-enforcement` | DevCovenant enforces its own policies | Meta-policy for system integrity |
| `no-print-in-library` | Prevent direct print() in library modules | Project-specific code quality |

## Laws Remaining as Manual Guidelines

These laws remain as human-readable guidelines and have NOT been converted to automated policies:

| Law # | Description | Reason Not Automated |
|-------|-------------|---------------------|
| 2 | Comment the code extensively | Requires human judgment on quality |
| 3 | Keep comments synchronized with code | Difficult to automate meaningfully |
| 5 | Keep AGENTS.md as canonical law source | Meta-guideline |
| 6 | Treat /data as read-only | Could be automated (future policy) |
| 9 | Re-read laws at start of development session | AI workflow guideline |
| 10 | Document every module, function and class | Requires judgment on adequacy |
| 11 | Use concise, descriptive names | Requires semantic understanding |
| 12 | Use raw strings or escape backslashes | Could be automated (future policy) |
| 13 | Run pre-commit before committing | Enforced by git hooks (different system) |
| 14 | Do not redistribute or assert patent claims | Legal/license requirement |
| 16 | Treat documentation refresh as integral | Workflow guideline |
| 17 | Commit changes only after all tests pass | Enforced by git hooks |
| 18 | Treat start.* launchers equally | Design principle |
| 19 | Follow compliance and security requirements | High-level principle |
| 21 | Audit licenses for new dependencies | Could be automated (future policy) |
| 22 | Run suite through managed virtual environment | Workflow guideline |
| 23 | Refresh dependencies when packages change | Workflow guideline |
| 25 | Preserve human-authored edits | Development principle |

## Actions Required

### 1. Add Missing Policies to AGENTS.md

The following policies are implemented but not documented in AGENTS.md:
- `version-sync`
- `no-future-dates`
- `new-modules-need-tests`
- `no-print-in-library`

### 2. Remove Redundant Laws from AGENTS.md

Remove laws #1, #4, #7, #8, #15, #20, and #24 from the numbered list.

### 3. Renumber Remaining Laws

After removal, renumber the remaining laws sequentially (1-18).

### 4. Update Cross-References

Update any references to law numbers elsewhere in documentation.

## Benefits of DevCovenant Over Numbered Laws

1. **Automated Enforcement**: Policies are checked automatically on every commit
2. **No Drift**: Policy text and enforcement code must stay synchronized
3. **Better Developer Experience**: Clear, actionable error messages
4. **Auto-Fixing**: Some policies can automatically fix violations
5. **Graduated Severity**: Policies can be critical, error, warning, or info
6. **AI-Maintained**: Hash verification ensures consistency
7. **Extensible**: Easy to add new policies as scripts

## Future Policy Candidates

Laws that could potentially be automated in the future:
- Law 6: Treat /data as read-only → `data-directory-immutability`
- Law 12: Escape backslashes → `valid-escape-sequences`
- Law 21: Audit licenses → `license-compatibility`

---

**Migration Status**: In Progress
**Policies in AGENTS.md**: 5
**Policies Implemented**: 9
**Laws Remaining**: 18 (after cleanup)
