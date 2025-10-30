# Documentation Policy

**Version:** 1.0
**Last Updated:** 2025-10-30

The Copernican Suite treats documentation as a first-class component of the
project. Every development task must include a documentation refresh that does
more than fix typos.

## Required Steps
- Update `README.md`, `AGENTS.md` and any affected files under `docs/`.
- Expand the scope or depth of documentation so it grows alongside the code.
- Synchronise version strings and `Last Updated` fields across all documents.
- Record the changes in `CHANGELOG.md`.
- Document user-facing console behaviour changes, including logging output and
  progress indicators, so users understand new interactions. The retirement of
  the combined optimiser in favour of the verbose MCMC progress reporting is an
  example that must always be highlighted in end-user docs.
- Highlight shared module refactors—such as the introduction of
  `copernican_lib.statistics`—in the API and design documentation so future
  tasks build atop the central helpers instead of re-creating legacy links.
- Call out sampler stability updates (for example walker reseeding to remove
  emcee warnings) so run logs and archived outputs stay interpretable for
  researchers auditing LCDM self-tests.

Adhering to this policy keeps the suite's knowledge base accurate and protects
our intellectual property by clearly documenting provenance and intent.
