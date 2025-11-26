# DriftGuard Development Notes
**Last Updated:** 2025-11-26

- Keep `DRIFTGUARD.md` and `driftguard/repo_policy.yml` in sync. Any change to
  either must be mirrored in the other file and reflected in the DriftGuard
  enforcement code and tests; policy text is the human source of truth and must
  always align with the YAML and enforcement logic.
- DriftGuard must stay decoupled from Copernican-specific modules; avoid imports
  from outside the `driftguard/` package.
- Run the **full** DriftGuard check (not fast mode) before every commit after
  completing the test suite. DriftGuard must enforce its own policy and exit
  non-zero on violations.
