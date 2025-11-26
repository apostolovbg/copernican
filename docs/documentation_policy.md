**Last Updated:** 2025-11-26
# Documentation Policy
**Version:** 1.0

The Copernican Suite treats documentation as a first-class component of the
project. Every development task must include a documentation refresh that does
more than fix typos.

## Required Steps
- Update `README.md`, `AGENTS.md` and any affected files under `docs/`.
- Expand the scope or depth of documentation so it grows alongside the code.
- Synchronise version strings and `Last Updated` fields across all documents.
- Keep every `Last Updated` marker within the first three lines of its file
  and record only the ISO calendar date (YYYY-MM-DD) with no time component.
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
- Treat every model `description` block as a manuscript-length document.
  Expand it to at least ten pages of theory, derivations and observational
  context whenever a model changes, and bump the model's internal `version`
  even if the suite version stays fixed. Only `cosmo_model_lcdm.yml` is
  mandatory; all other models evolve as their documentation grows.

Adhering to this policy keeps the suite's knowledge base accurate and protects
our intellectual property by clearly documenting provenance and intent.

DriftGuard now anchors these expectations in `driftguard/repo_policy.yml`. The
doc surfaces defined there mirror the Last Updated and date checks enforced by the
metadata rules so documentation drift is visible to contributors before code
reaches review.
The metadata surface now also validates citation YAML structure, synchronises
release versions across `README.md`, `CITATION.cff` and
`copernican_lib/VERSION`, and insists on changelog coverage whenever other
files change.
