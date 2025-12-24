# RNG Mini-games

`rng_minigames/` is a portable bundle of deterministic random-number games.
Each title records player interactions, hashes the payload and emits a seed via
`MinigameContext.set_seed`. Because the project is intentionally “marsupial,”
other applications can copy the directory verbatim, run the registry and pick up
the same playful seed-forging experience Copernican offers inside its GUI.

## Architecture

- Every game lives inside `rng_minigames/<id>/` with a `metadata.json` descriptor
  and a launcher module.
- `registry.json` caches hashes for every descriptor/launcher pair. Call
  `rng_minigames.refresh_registry()` after adding or editing a game so hosts can
  verify integrity before loading it.
- `rng_minigames.api.MinigameContext` packages the host’s callbacks
  (`set_seed`, `notify`, optional Tk root and `render` flag). Launchers call the
  provided functions when they complete or want to display toast messages.
- `rng_minigames.load_launcher(game_id)` automatically reloads the module each
  time it is invoked, making iterative development as simple as editing the file
  and reopening the mini-game window. No GUI restart is required.

Each title only writes data inside its own folder—Alien Invasion persists state
inside `_storage/`, for instance—so vendoring the directory preserves both the
code _and_ accumulated scores.

## Available Games

- [Emoji Meteors](emoji_meteors/README.md) – pet five falling animals; the order
  and elapsed time define the seed.
- [Constellation](constellation/README.md) – connect ten stars to weave a
  constellation whose path becomes the seed payload.
- [Alien Invasion](alien_invasion/README.md) – defend Earth, store Neutron
  charges and optionally let the autopilot learn from every run.

Every folder ships with a README describing controls, accessibility notes,
seed-generation logic and any configuration files specific to that title.

## Configuration & Storage

- Alien Invasion exposes `alien_invasion/ai_settings.yml` (autopilot tunables)
  plus `_storage/game_settings.yml` (shield counts, motion limits, debris
  behaviour, etc.). Each file is generated on the first launch, ignored by Git
  and reused between runs so user tweaks survive pulls; delete either to
  regenerate fresh defaults. Both are hot-reloaded whenever the game launches,
  and the AI template already declares a five-layer network (`40,32,24,15,12`)
  so deeper brains appear automatically.
- Sample data and persistent state (AI weights, hall-of-fame entries) live in
  `_storage/` under each game so the bundle remains self-contained. Alien
  Invasion creates its AI state and hall-of-fame YAML files the first time it
  runs; deleting either resets the corresponding data on the next launch.
- Stateless titles such as Emoji Meteors and Constellation rely only on the
  shared API and metadata.

## Documentation & Changelog

- High-level architecture and embedding instructions live in this README.
- Each mini-game folder provides its own README with gameplay and configuration
  details.
- RNG-specific history is recorded in `rng_minigames/CHANGELOG.md`. Copernican’s
  root changelog only tracks non-RNG work; note every RNG edit in the local
  changelog so downstream users can audit the bundle independently.

## Embedding Checklist

1. Vendor the entire `rng_minigames/` directory into your repository.
2. Call `rng_minigames.load_registry()` to enumerate available games.
3. Instantiate `rng_minigames.MinigameContext` with `set_seed`, `notify` and an
   optional Tk root (or `render=False` for headless environments).
4. Load a launcher via `rng_minigames.load_launcher(descriptor.game_id)` and hand the
   context to it.
5. Offer a **Refresh** button that calls `rng_minigames.refresh_registry()` so
   new games or metadata edits are recognised without restarting your UI.

## Adding a Game

1. Create `rng_minigames/<game_id>/` with `metadata.json`, the launcher module
   and any supporting assets.
2. Update or add tests under `rng_minigames/tests/` if the new code requires
   coverage.
3. Run `rng_minigames.refresh_registry()` to regenerate `registry.json`.
4. Document the gameplay in `rng_minigames/<game_id>/README.md`.
5. Log the change in both `rng_minigames/CHANGELOG.md` and, if relevant, the
   host application’s changelog.

Follow this blueprint and RNG Mini-games will remain easy to drop into any GUI
that needs deterministic yet entertaining seed generators.
