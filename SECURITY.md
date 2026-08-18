# Security Policy
**Doc ID:** SECURITY
**Doc Type:** repo-security
**Project Version:** 12.0.26
**Last Updated:** 2026-08-18
**DevCovenant Version:** 1.0.1b6

## Table of Contents
- [Overview](#overview)
- [Supported Surfaces](#supported-surfaces)
- [How To Report](#how-to-report)
- [Response Expectations](#response-expectations)

## Overview
Security in Copernican is handled as part of repository governance, not as an
afterthought. The repository tracks its trusted datasets, package metadata,
runtime dependencies, and GUI launch surfaces through DevCovenant so the same
rules govern local work, managed runs, and mirrored documentation.

The intent of this file is to make the reporting path explicit and keep the
project safe to use without exposing private details in issues or review
threads. If you suspect a vulnerability, a data-integrity problem, or a
malicious change in the dependency surface, report it through the process
below instead of opening a public discussion with exploit details.

## Supported Surfaces
The security posture for this repository covers the pieces that can affect
execution, reproducibility, or data trust:

- the managed environment in `.venv`
- the package dependency surface in `pyproject.toml`
- the workspace dependency lockfiles
- the bundled runtime packages in `copernican/samplers/`,
  `copernican/models/`, and `copernican/validation/`
- the dataset registry and parser trust checks
- the GUI and CLI entry points
- the generated documentation that explains the supported workflow
- the home-folder validation marker at `~/VALIDATION.md`

Those areas are treated as first-class project surfaces because a change in
any one of them can alter how Copernican executes or what data it trusts.

The package runtime surface excludes scientific-reference solvers. CAMB is
locked and licensed only by the repository workspace for independent tests;
it is absent from default wheel metadata, package runtime locks, and installed
package license assets. Package-isolation tests reject attempts to import CAMB
or CLASS during native CMB execution.

## How To Report
Include enough detail to let maintainers reproduce the issue without guessing.
The most useful reports include:

- the affected file or command
- the Python version and operating system
- the exact step that exposed the issue
- a minimal reproduction sequence
- any logs, manifests, or stack traces that are safe to share

Do not include secrets, tokens, private dataset material, or exploit
payloads. If the report is sensitive, contact the maintainers privately using
the repository's preferred issue or security channel.

## Response Expectations
Maintainers should acknowledge the report, reproduce it if possible, and then
either patch the issue or explain why the behavior is expected. When a fix is
needed, the documentation should be updated alongside the code so the next
operator sees the same rule set the maintainers used to resolve the problem.

If the report affects the dependency surface, the lockfiles and the generated
license artifacts should be refreshed together so the repository stays
consistent after the fix lands.
