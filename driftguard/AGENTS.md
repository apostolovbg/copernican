# DriftGuard Development Notes
**Last Updated:** 2025-11-26

- Keep `DRIFTGUARD.md` and `driftguard/repo_policy.yml` in sync. Any change to
  either must be mirrored in the other file and reflected in the DriftGuard
  enforcement code and tests.
- DriftGuard must stay decoupled from Copernican-specific modules; avoid imports
  from outside the `driftguard/` package.
