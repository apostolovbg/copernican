# devcovenant
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.3
**Last Updated:** 2026-05-23
**DevCovenant Version:** 1.0.1b5

<!-- DEVCOV:BEGIN -->

<!-- DEVCOV:END -->

![DevCovenant banner](https://raw.githubusercontent.com/apostolovbg/devcovenant/main/devcovenant/docs/banner.png)

## Overview
DevCovenant is a repository governance framework.
It gives a repository an explicit, executable contract for how work happens,
what stays in sync, and what evidence development must leave behind.

In plain language, repository governance means the repository itself owns the
rules for how it is developed and maintained.
That includes the work sequence, the active policy set, the generated and
managed files, the dependency surfaces, and the run evidence.

DevCovenant turns that contract into CLI-enforced behavior.
Instead of leaving workflow law in scattered docs, shell snippets, CI glue, and
private AI prompts, it keeps the contract inside the repository and makes it
executable.

In practice, DevCovenant gives a repository four things:

1. A required workflow.

   The standard work slice is `gate --open`, edit, `gate --verify`, `run`,
   `gate --close`.

2. Executable policy rules.

   Policies are declared in project metadata, surfaced in `AGENTS.md`, and
   enforced by the CLI instead of living only as prose.

3. Managed repository surfaces.

   DevCovenant can keep selected docs, config sections, generated files,
   workflow files, and policy blocks aligned.

4. Run evidence.

   Each governed command writes summaries and logs so operators can inspect
   what happened instead of guessing.

## Why Repositories Need It
Most repositories do not fail because a formatter was missing.
They fail in more expensive ways:

- people skip or reorder required steps
- CI and local behavior drift apart
- docs stop matching what the code and workflow actually do
- generated or managed files are edited by hand and quietly drift
- dependency surfaces look current until security or platform-specific failures
  appear later
- humans and AI tools follow different invisible rules in the same repository

DevCovenant exists to make those failures visible, repeatable, and owned by the
repository itself.

## Why It Is Different
DevCovenant is not just another checker bundle.

1. It is not a linter.

   A linter tells you a file is wrong.
   DevCovenant tells you the repository contract is drifting, which owned
   surfaces are involved, and which workflow step has to re-establish trust.

2. It is not an IDE plugin.

   The contract does not live in one editor setup.
   It lives in the repository and follows the repository everywhere.

3. It is not a model wrapper or prompt pack.

   DevCovenant does not depend on one AI account, one assistant, or one hidden
   system prompt.

4. It is repository-level and tool-agnostic.

   If a team uses ChatGPT, Claude, Codex, Cursor, Copilot, Vim, VS Code,
   PyCharm, terminals, or no AI at all, the governed repository contract stays
   the same.

5. Custom governance is first-class.

   Builtin policies and profiles are the shipped baseline, not the limit.
   A repository can define and own its own governance model.

## Where It Came From
DevCovenant was born inside Copernican.
That repository already had written development law in `AGENTS.md`, but prose
alone was not enough.
The rules were real, yet too much of the operational contract still lived in
habit, manual interpretation, and scattered tooling.

DevCovenant came out of that pressure:
repository law had to stop being only documentation and become executable,
repo-owned behavior.

## Metadata And Layered Ownership
The metadata system is not decorative YAML.
It is the authoring surface that tells DevCovenant what belongs together and
who owns what.

That layered ownership model is one of the main reasons DevCovenant reduces
drift instead of just adding ceremony.
Instead of throwing everything into one prose document, the system separates:

- builtin policies and profiles
- repo-owned custom policies and profiles
- config overlays and overrides
- managed documents and generated artifacts
- tracked registry state
- runtime registry state

That separation matters.
It makes it harder to edit the wrong layer by accident, and it makes it less
likely that an AI tool will grab the canonical law surface and rewrite it just
because it sees text that looks editable.

It also makes the system more extensible over time.
Metadata describes ownership, selectors, managed surfaces, runtime actions,
autofix surfaces, blueprint materialization, dependency surfaces, and workflow
composition.

## Proof It Is Real
DevCovenant is not a README-only concept.

- It is dogfooded on itself: the DevCovenant repository is developed under
  DevCovenant governance.
- It already runs on multiple downstream repositories with different stack
  shapes.
- It already governs more than code style: workflow law, changelog discipline,
  managed docs, dependency surfaces, security scanning, version propagation,
  trust docs, and customization flow.

That matters because it means the framework is being tested under real
maintenance pressure, not only in toy examples.

## Fast Evaluation
DevCovenant is worth your attention if most of the following sound familiar:

- you want humans and AI tools to follow the same repository-owned rules
- you are tired of CI doing something different from local workflow
- your repository has docs, generated files, or dependency artifacts that drift
- you want repo-specific governance, not just a fixed builtin rule bundle
- you need evidence and logs for workflow steps, not only pass/fail signals

If that sounds right, use the route map below.
If you are specifically evaluating extensibility, skim
[Custom Governance](#custom-governance) and then run `devcovenant demo`.

## Quick Start
For most users, the right first path is an isolated machine install with
`pipx`, followed by activation inside the repository you want to govern.
If you want a terse static reminder of that sequence, run
`devcovenant quickstart`.
If you want a disposable guided evaluation repo, run `devcovenant demo`.
The README and the demo are the primary onboarding path; quickstart is
only a reminder.

```bash
pipx install devcovenant
devcovenant --version
cd your-repo
devcovenant install
# review devcovenant/config.yaml
# set install.config_reviewed: true
devcovenant deploy
# prepare the environment declared by the active profile stack
python3 -m pip install -r requirements.lock
devcovenant gate --open
# make your edits
devcovenant gate --verify
devcovenant run
devcovenant gate --close
```

What those steps mean:

1. `pipx install devcovenant` installs the CLI in its own machine-level
   application environment.

2. `devcovenant install` adds DevCovenant to the target repository and writes
   the initial `devcovenant/config.yaml`.

3. The config review is the human decision point.

   Start with `project-governance`, `developer_mode`, and `profiles.active`.
   Builtin `devcovuser` is the always-active baseline for every DevCovenant
   user repo.
   In most downstream repositories, keep `devcovuser` active and add a
   repo-owned `userproject` custom profile when the repository needs its own
   starter layer; Python repositories often add a custom `python` profile
   too.
   The copy-ready bootstrap template starts at
   `devcovenant/builtin/profiles/userproject/` and becomes repo-owned when
   copied to `devcovenant/custom/profiles/userproject/`.

4. `devcovenant deploy` writes the managed docs, generated files, and other
   owned outputs for the active profile stack.

5. Prepare the environment declared by the active profile stack before the
   first gate cycle.

   That might be a local `.venv`, a system interpreter, a bench-managed
   environment, a container-managed environment, or another declared layout.

6. The first full gate cycle proves the reviewed setup actually works.

Use a source checkout instead of `pipx` only when you intentionally need local
source behavior rather than the installed CLI.
If the console script is unavailable there, use `python3 -m devcovenant ...`.
On Windows, `py -m devcovenant ...` is the common equivalent form.

If you want the deeper doc set, use the map below instead of hunting through
the tree.

## Docs Map
Use the shorter map below instead of treating the README as the whole manual.

- "How do I install it and get through first setup?"

  Read [installation.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/installation.md).

- "What does the daily workflow look like?"

  Read [workflow.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/workflow.md).

- "How does config and metadata actually work?"

  Read [config.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/config.md).

- "How do I create or tune repository governance?"

  Read [policies.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/policies.md) and
  [profiles.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/profiles.md).

- "How do I make my first custom profile or policy?"

  Read [customization.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/customization.md).

- "What does refresh own?"

  Read [refresh.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/refresh.md).

- "How is the system structured internally?"

  Read [architecture.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/architecture.md).

- "What registry and runtime state does DevCovenant track?"

  Read [registry.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/registry.md).

- "What do I do when something fails?"

  Read [troubleshooting.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/troubleshooting.md).

## Workflow
The standard repository workflow is:

```bash
devcovenant gate --open
# edit files and clear complaints while working
devcovenant gate --verify
devcovenant run
devcovenant gate --close
```

In `engine.tests_output_mode: normal`, test progress stays visible during
workflow runs while ordinary command output remains concise.

Use the commands this way:

- `check`

  Read-only audit.
  It inspects the repository and writes logs, but it does not open or close a
  gate session.

- `gate --open`

  Opens a work session and records the starting state for the slice.

- `gate --verify`

  Required pre-run check.
  It catches hook changes and DevCovenant-managed mutations before `run`.

- `run`

  Runs the declared workflow steps in order and records the results.

- `gate --close`

  Runs the closing pre-commit pass and closes the session when the required
  evidence is fresh and passing.

When a command prints `Run logs: ...`, start with `summary.txt`.
If that is not enough, inspect `tail.txt`, then `stdout.log` and `stderr.log`.

## Custom Governance
Custom governance is one of DevCovenant's main differentiators.
The framework is built so a repository can define and own its own law instead
of waiting for the core project to ship one more builtin.

The normal downstream shape keeps builtin `devcovuser` active and adds
repo-owned custom profiles when the repository needs its own reusable
behavior. `userproject` is the copy-ready bootstrap template for that first
repo-owned custom profile, and Python repositories often add a custom
`python` or `python_venv` profile for environment ownership. That split keeps
builtin layers generic and puts the real project law in
repository-owned metadata.

If you want the guided first custom path, read
[customization.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/customization.md).
It walks through the first custom profile path, the first custom policy path,
shadowing and materialization, and which layer to edit and why.

A custom governance stack can combine:

- custom policies under `devcovenant/custom/policies/`
- custom profiles under `devcovenant/custom/profiles/`
- builtin shadow copies promoted through
  `devcovenant custom --policy <policy-id> --do|--undo` and
  `devcovenant custom --profile <profile-name> --do|--undo`
- selector-role scopes for different file families inside one policy
- translators for language-aware checks
- runtime actions, autofix helpers, and policy-born commands
- managed docs, workflow fragments, dependency surfaces, and environment
  contracts

The key point is that metadata is the authoring API.
It describes what a policy or profile owns, what it may update, what files and
surfaces it governs, where its tests materialize, and how it composes with the
active repository stack.

Customization is override-based by design:

- a same-id custom policy fully shadows the builtin policy with that id
- a same-name custom profile fully shadows the builtin profile with that name
- when a custom entry shadows a builtin one, the builtin entry is ignored

For the deeper authoring model, go straight to
[policies.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/policies.md),
[profiles.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/profiles.md), and
[config.md](https://github.com/apostolovbg/devcovenant/blob/main/devcovenant/docs/config.md).

## License
DevCovenant is released under the MIT License.
See [LICENSE](https://github.com/apostolovbg/devcovenant/blob/main/LICENSE)
and
[licenses/THIRD_PARTY_LICENSES.md](https://github.com/apostolovbg/devcovenant/blob/main/licenses/THIRD_PARTY_LICENSES.md).
The published package includes the same license and compliance files under
`devcovenant/licenses/`.
