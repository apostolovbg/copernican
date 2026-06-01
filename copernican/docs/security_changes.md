# Security Changes Log

Document each security or compliance-related update you make for visibility.
When you modify launchers, security helpers, or any guarded subsystem, add a
short entry describing the change, the files touched, and any new mitigations
so reviewers know the latest requirements were considered.

## Table of Contents

- [Overview](#overview)
- [Recent Changes](#recent-changes)
- [Practical Rule](#practical-rule)

## Overview

Use this page to record security or compliance-related updates that affect the
governance surface of the repository. Keep entries brief, factual, and tied to
the files that changed so reviewers can audit the latest security context
without reading the entire changelog.

## Recent Changes

- 2025-12-28: Realigned the security-compliance notes and security scanner
  with the latest DevCovenant instructions so guarded security-helper edits
  continue to require explicit documentation before they pass
  (devcovenant/policy_scripts/security_compliance_notes.py,
  devcovenant/policy_scripts/security_scanner.py).

## Practical Rule

When a security-sensitive file changes, log the change here as well as in the
main changelog if the broader release history should capture it.
