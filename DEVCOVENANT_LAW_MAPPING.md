# DevCovenant Policy-to-Law Mapping

This document tracks the transition from the numbered Development Laws to the
self-enforcing **DevCovenant** policy system. It is the canonical place to
record which bespoke laws have been deprecated in favor of automated checks
and which remain manual reminders. Whenever new policies are introduced or laws
are retired, update this file alongside `AGENTS.md` so readers can trace the
history.

## Overview

The Copernican Suite now relies on DevCovenant to enforce every policy listed in
`AGENTS.md`. Policies are:

- **Self-enforcing**: Code and documentation must stay in sync via hashes.
- **AI-maintained**: Hash mismatches trigger actionable guidance for script
  updates.
- **Transparent**: Each policy is documented next to the human-readable rule.
- **Audit-ready**: Logs and manifests show which policies were checked.

Laws that are backed by policies are tracked below. Any remaining numbered law
continues to describe a manual expectation for contributors.

## Policies Replacing Development Laws

| Policy ID | Description | Superseded Law(s) | Documented In | Notes |
|-----------|-------------|-------------------|--------------|-------|
| `changelog-coverage` | Requires every touched file to be enumerated in `CHANGELOG.md`. | Law 1 (Summarize changes in the changelog) and Law 11 (Treat documentation refresh as integral). | `AGENTS.md#policy-changelog-coverage` | Blocks commits when files are missing from the changelog summary. |
| `last-updated-placement` | Restricts `Last Updated` headers to allowlisted surfaces. | Law 4 (Refresh documentation and `Last Updated` markers on allowlisted surfaces) and Law 11. | `AGENTS.md#policy-last-updated-marker-placement` | Provides an auto-fixer for stray markers (`--fix`). |
| `version-sync` | Keeps `copernican_lib/VERSION`, `README.md`, `pyproject.toml` and `CITATION.cff` aligned. | Law 7 (Follow the versioning policy). | `AGENTS.md#policy-version-synchronization` | Prevents drift between runtime metadata and docs. |
| `no-future-dates` | Bans `Last Updated` or date fields set in the future. | Law 24 (Validate timestamps before recording them). | `AGENTS.md#policy-no-future-dates` | Ensures changelog entries and version markers use current UTC dates. |
| `no-git-conflict-markers` | Detects `<<<<<<<`, `=======` and `>>>>>>>`. | Law 8 (Never insert Git conflict markers). | `AGENTS.md#policy-no-git-conflict-markers` | Runs on the entire repo tree (excluding ignored directories). |
| `line-length-limit` | Enforces the 79-character line budget in Python code. | Law 15 (Keep individual lines under 79 characters). | `AGENTS.md#policy-line-length-limit` | Emits warnings only for `.py` files. |
| `new-modules-need-tests` | Requires tests whenever new modules appear in `copernican_lib/` or `engines/`. | Law 20 (Add tests alongside new functionality). | `AGENTS.md#policy-new-modules-need-tests` | Scans the Git status to determine added modules. |

## Additional Policies (Not Derived from Numbered Laws)

- `devcov-self-enforcement` documentation and tests ensure DevCovenant enforces its own policies (`AGENTS.md#policy-devcov-self-enforcement`).
- `no-print-in-library` forbids bare `print()` calls inside `copernican_lib/` and `engines/` so output streams remain centralized (`AGENTS.md#policy-no-print-in-library`).

## Manual Laws Still in AGENTS

After the deprecations above, the remaining numbered laws in `AGENTS.md` are
still manual guidelines that require human judgment (for example, law 1 now
remains focused on descriptive comments rather than the changelog). When new
policies cover a manual law, update both this file and the AGENTS entry to mark
it as deprecated. Law 11 ("Treat documentation refresh as integral") is still
active because no policy governs the broader content expansion requirements it
describes; the policies above cover metadata hygiene and changelog reporting but
do not replace the obligation to extend the relevant prose whenever substantive
functionality or configuration shifts occur.

## Deprecated Law References

- **Law 11**: "Treat documentation refresh as integral to every task..." is now
  partially enforced by `changelog-coverage`, `last-updated-placement`,
  `version-sync` and `no-future-dates`. The law text remains in `AGENTS.md` and contributors should follow both it and the related policy.
- **Law 1**, **Law 4**, **Law 7**, **Law 8**, **Law 15**, **Law 20** and **Law 24**: The
  original wording for these laws has been retired in favor of the policies listed
  in the table above. See the row for each policy to understand the replacement.

## Status Summary

- Policies documented in `AGENTS.md`: **9** (all listed above).
- Policies implemented but without a numbered-law counterpart: `devcov-self-enforcement`, `no-print-in-library`.
- Deprecated laws maintained here: Law 11 plus the historical Law 1, 4, 7, 8, 15, 20 and 24 entries.

## Future Policy Candidates

The remaining laws that still require manual attention can become policies in
the future. Candidate policies include:

1. **Law 6** (`/data` immutability) → potential `data-directory-immutability` policy.
2. **Law 12** (escape sequences) → potential `valid-escape-sequences` policy.
3. **Law 21** (license audits) → potential `license-compatibility` policy.

Update this file when a candidate becomes reality so the numbering stays
accurate and the mapping reflects every policy-to-law transition.
