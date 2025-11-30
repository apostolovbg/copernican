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

## Law 11 and Documentation Growth
Law 11 in [`AGENTS.md`](../AGENTS.md) enshrines the expectation that every
meaningful change enlarges the written record. Extending this policy means you
must enrich `README.md`, `AGENTS.md` and the relevant `docs/` pages with
additional paragraphs, images or usage notes describing the new behaviour.
- Add an explanatory section in `README.md` that places the feature in the
  broader workflow, mentions the affected launchers or modules, and links to
  the new `docs/` pages that catalogue the change.
- Increase the depth of `docs/` entries, for example by appending a new
  subsection for the updated start scripts or a call-out in the GUI overview.
- When a launcher, menu or parser changes, create or expand a `docs/`
  reference (such as the new `docs/launcher_gui.md`) so operators know how to
  reach the GUI without delving into shell scripts and so the Law 11 story
  tracks the behaviour in both prose and code.

Adhering to this policy keeps the suite's knowledge base accurate and protects
our intellectual property by clearly documenting provenance and intent.
