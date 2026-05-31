# Security Changes Log

Document each security or compliance-related update you make for visibility.
When you modify launchers, security helpers, or any guarded subsystem, add a
short entry describing the change, the files touched, and any new mitigations
so reviewers know the latest requirements were considered.

- 2025-12-28: Realigned the security-compliance notes and security scanner
  with the latest DevCovenant instructions so guarded security-helper edits
  continue to require explicit documentation before they pass
  (devcovenant/policy_scripts/security_compliance_notes.py,
  devcovenant/policy_scripts/security_scanner.py).
