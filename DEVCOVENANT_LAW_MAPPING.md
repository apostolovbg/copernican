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
| `version-sync` | Keeps `copernican_lib/VERSION`, `README.md`, `pyproject.toml` and `CITATION.cff` aligned on the same SemVer string and ensures it strictly increases compared to the previous commit. | Law 7 (Follow the versioning policy). | `AGENTS.md#policy-version-synchronization` | Prevents drift between runtime metadata and docs while flagging invalid SemVer values or non-forward bumps. |
| `no-future-dates` | Bans `Last Updated` or date fields set in the future. | Law 24 (Validate timestamps before recording them). | `AGENTS.md#policy-no-future-dates` | Ensures changelog entries and version markers use current UTC dates. |
| `no-git-conflict-markers` | Detects `<<<<<<<`, `=======` and `>>>>>>>`. | Law 8 (Never insert Git conflict markers). | `AGENTS.md#policy-no-git-conflict-markers` | Runs on the entire repo tree (excluding ignored directories). |
| `line-length-limit` | Enforces the 79-character line budget in Python code. | Law 15 (Keep individual lines under 79 characters). | `AGENTS.md#policy-line-length-limit` | Emits warnings only for `.py` files. |
| `new-modules-need-tests` | Requires tests whenever new modules appear in `copernican_lib/` or `engines/`. | Law 20 (Add tests alongside new functionality). | `AGENTS.md#policy-new-modules-need-tests` | Scans the Git status to determine added modules. |
| `read-only-directories` | Guards the pattern list under `devcovenant/read_only_directories.txt`, blocking edits to datasets or parser files unless a waiver exists. | Law 4 (Treat `/data` as read-only). | `AGENTS.md#policy-read-only-directories` | Waivers live under `.devcovenant/waivers/read-only-directories.txt` and the patterns are refreshed every run. |
| `docstring-and-comment-coverage` | Warns when any non-test Python module lacks docstrings or nearby explanatory comments; the check now inspects every matching `.py` (info-level reminders) so coverage gaps in untouched files are surfaced before escalation. | Law 6 (Document every module, function and class with clear "what" and "why" explanations). | `AGENTS.md#policy-docstring-and-comment-coverage` | Both short docstrings and descriptive pre-definition comments satisfy the check; vendor/test code is ignored. |
| `dependency-license-sync` | Requires simultaneous updates to dependency inputs, `THIRD_PARTY_LICENSES.md` (including a `## License Report` section) and `licenses/`. | Laws 15 and 17 (Audit licenses for new dependencies; refresh dependencies after package changes). | `AGENTS.md#policy-dependency-license-sync` | The report must mention each touched dependency file so reviewers can confirm the coverage. |
| `documentation-growth-tracking` | Reminds contributors to grow README/AGENTS/docs when user-facing files change so the corpus strictly expands. | Law 11 (Treat documentation refresh as integral to every task). | `AGENTS.md#policy-documentation-growth-tracking` | Runs as an active info-level policy that highlights the rule whenever user-visible components move. |

## Additional Policies (Not Derived from Numbered Laws)

- `devcov-self-enforcement` documentation and tests ensure DevCovenant enforces its own policies (`AGENTS.md#policy-devcov-self-enforcement`).
- `no-print-in-library` forbids bare `print()` calls inside `copernican_lib/` and `engines/` so output streams remain centralized (`AGENTS.md#policy-no-print-in-library`).

## Manual Laws Still in AGENTS

After the deprecations above, the remaining numbered laws in `AGENTS.md` are
still manual guidelines that require human judgment. When new policies cover a
manual law, update both this file and the AGENTS entry to mark it as deprecated.
The new policies now cover data immutability, docstrings/comments,
documentation growth, and license auditing so the manual list focuses on the
broader development discipline items outlined in the law section.

## Deprecated Law References

- **Law 11**: “Treat documentation refresh as integral…” is now enforced by
  `documentation-growth-tracking`, which issues reminders when user-visible files
  change so the corpus strictly grows alongside semantic updates.
- **Law 4**: “Treat `/data` as read-only” is enforced by `read-only-directories`
  (with waivers under `.devcovenant/waivers/read-only-directories.txt`), and
  **Law 6** (“Document every module, function and class…”) is enforced by
  `docstring-and-comment-coverage`.
- **Law 15** and **Law 17** (license audits and dependency refreshes) are now
  consolidated under `dependency-license-sync`, which requires `THIRD_PARTY_LICENSES.md`,
  `licenses/*`, and the dependency inputs to change in lockstep with a `## License Report`.
- **Law 1**, **Law 7**, **Law 8**, **Law 20** and **Law 24** have already been retired in favor of the policies in the table above (`changelog-coverage`, `last-updated-placement`, `version-sync`, `line-length-limit`, `new-modules-need-tests` and its companions).

## Status Summary

- Policies documented in `AGENTS.md`: **13** (the eleven numbered policies above plus `devcov-self-enforcement` and `no-print-in-library`).
- Policies implemented but without a numbered-law counterpart: `devcov-self-enforcement`, `no-print-in-library`.
- Deprecated laws maintained here: Laws 1, 4, 6, 7, 8, 11, 15, 17, 20 and 24.

## Future Policy Candidates

When one of the current manual laws becomes automatable, update this file. The
escape-sequence law (use raw strings or escape backslashes explicitly) still
requires tooling support, so a `valid-escape-sequences` policy remains a strong
candidate.

Update this file when a candidate becomes reality so the numbering stays
accurate and the mapping reflects every policy-to-law transition.
