# DevCovenant Waivers

Some policies support temporary exceptions via waiver files stored under
`.devcovenant/waivers/`. Each waiver file is named after the policy ID (e.g.
`read-only-directories.txt`) and lists the paths or glob patterns that are
temporarily exempted.

### Read-only directories waiver

- Create `.devcovenant/waivers/read-only-directories.txt` with newline-
  delimited paths relative to the repository root or gitignore-style globs.
- The read-only-directories policy accepts these entries while the waiver file
  exists so you can stage approved dataset or parser updates.
- When you finish the waiver work, remove or shrink the waiver file so the
  policy reverts to protecting the directories again.

Waivers are intended for one-off maintenance work. Always document the reason
and duration inside the waiver file to keep builds and auditors informed.
