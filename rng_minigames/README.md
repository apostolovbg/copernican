# RNG Mini-games

`rng_minigames/` is a self-contained collection of playful random-number
generators. Each game produces a deterministic seed by capturing player
interactions, hashing the result and calling a shared `MinigameContext`. Because
the project ships as a marsupial folder, other applications can copy
`rng_minigames/` verbatim, point their GUI at the registry and immediately gain
the same seed-forging experience Copernican uses inside its Run Builder.

- Every mini-game lives in its own directory with a `metadata.json` file and a
  launcher module (for example, `emoji_meteors/metadata.json` and
  `emoji_meteors/game.py`).
- `registry.json` caches the discovered games. Calling
  `rng_minigames.refresh_registry()` recomputes hashes and rewrites the file,
  while `rng_minigames.load_registry()` returns the current descriptors.
- `rng_minigames.api.MinigameContext` packages the host callbacks:
  `set_seed(value)`, `notify(message, severity)` plus the `render` flag and the
  optional Tk root window. Launchers call `context.set_seed` when finished so
  hosts can update their manifests.
- Seed consumers such as Copernican build their UI dynamically by iterating the
  descriptors and calling `rng_minigames.load_launcher(game_id)` on demand. The
  loader automatically reloads the module each time, so editing a game’s code
  and closing its window is enough to test changes—no full application restart
  required.

Each game only writes local data inside its folder—Alien Invasion, for example,
stores AI progress inside `alien_invasion/_storage/`. Copying the directory into
another repository preserves both the code and any accumulated scores without
depending on Copernican’s `.cache` structure.

## Emoji Meteors
- **Flow**: click five falling animal emojis to "pet" them. Each selection
  enlarges the emoji and the captured combination is hashed with the time spent
  in the game.
- **Extras**: Cute Enough confirms the generated seed, Try Again refreshes the
  sky and Cancel closes the window. The mini-game falls back to a deterministic
  random selection when GUI rendering is disabled.

## Constellation
- **Flow**: select ten stars on a dense field to weave a constellation. Each
  pick draws lines between the stars and the final path plus completion time
  determine the seed.
- **Controls**: Left-click to add a star, right-click to remove one, and watch
  the “Stars connected” counter climb toward ten. Try Again clears the canvas,
  Cancel exits, and “Ad astra!” confirms once all ten stars are connected.

## Alien Invasion
- **Flow**: future Earth pilots fend off four stacked rows of 16 invaders, a
  support row of eight, and a roving general. Move the ship with the mouse,
  left-click to fire lasers and right-click to launch stored space charges. A
  capsule loads automatically the moment it touches your ship, so you only need
  to focus on dodging and firing. Every destroyed ship records its ID; the kill
  order plus total completion time forge the seed.
- **Gameplay notes**:
  - Your ship now carries a 50-hit shield. Each hit revives the most recently
    destroyed invader, and if the shield collapses you must restart via Try
    Again.
  - The bold status line tracks your 50-hit shield alongside the general’s
    shield, the Neutron charge stockpile and the live run timer so you always
    know when either side is close to collapse and how long the attempt took.
  - Space charges are rare (only one capsule drops at a time). Touching one
    stores it automatically (up to three in reserve), and launching it sends a
    slow-moving blue bomb that always triggers a massive chain explosion. Every
    ship caught in that blast sheds a storm of debris that can knock you out of
    position, so expect the skyline to rain shrapnel after a good hit.
  - The lowermost Colonels wear heavy plating: they need five regular hits (or a
    single space charge) before they explode and their bright shield border
    thins as you chip away at it. The full-width row above them holds Majors
    with one shield pip, and both ranks fire more frequently than the
    Lieutenant rows stacked overhead.
  - The roaming general now patrols a dedicated horizontal rail, picking new
    waypoints across the skyline and dashing away the moment you fly beneath
    him. He never leaves the field, yet keeps firing at your position until his
    20-hit shield collapses.
  - Darts can intercept each other mid-air, lasers vaporise debris, and the
    skyline includes pine clusters, hills, cities and moonlit gradients so the
    battlefield evolves as you play.
  - The mini-game ends once all invaders are down and the Use Seed button
    becomes available.
  - A **Let AI take care** button spins up the `AlienInvasionAI` helper. It
    honours the same 0.1 s firing cooldown, respects the player speed cap and
    learns from every run by updating `_storage/alien_invasion_ai_state.yml`.
    Toggle it off at any time to resume manual control; even one-off autopilot
    sorties contribute to training.
  - When your shield collapses the game now auto-resets after a short explosion
    animation; pressing **Reset** respawns immediately, and continuous **Let AI
    learn** loops skip the animation altogether so the AI can keep iterating.
  - **Let AI learn** keeps launching AI-controlled runs back-to-back; wins still
    land in the hall of fame and every finished run bumps the on-screen AI games
    counter plus the “Everybody lives/Everybody dies” digits that track how many
    worlds the autopilot has saved or doomed. Use **Let AI forget** to wipe
    `alien_invasion/_storage/alien_invasion_ai_state.yml` via a Wipe/Pardon
    dialog.
  - **Pause/Resume** freezes the action mid-run so you can grab a capsule or
    step away without losing progress, and the hall-of-fame button opens the
    `_storage/alien_invasion_hof.yml` scoreboard from the same window.

### Configuring Alien Invasion

Alien Invasion exposes two YAML files alongside its code so downstream
applications can rebalance the encounter without editing Python:

- `alien_invasion/ai_settings.yml` tweaks the autopilot—exploration rate,
  learning speed multiplier, time-pressure curve, kill rewards and respawn
  penalties. Editing the file takes effect on the next launch because the
  settings loader runs every time the module is imported.
- `alien_invasion/game_settings.yml` defines gameplay knobs: player/general
  shields, general/max player speed limits, motion acceleration and snap
  tolerances, Neutron charge capacity plus the explosion/debris behaviour (frame
  cadence, shard volume, debris damage and more). Update the file to try out new
  pacing without recompiling anything.
- Global explosion settings now control both the player animation and the
  charge-triggered chain reactions; `player_explosion.hold_seconds` sets how long
  the defeat animation is shown before the auto-reset kicks in.
- `debris.damages_all` toggles whether shrapnel from any source can hurt the
  invading fleet as well as the player—flip it on for full friendly fire.

# Embedding the Mini-games

1. Vendor the entire `rng_minigames/` directory into your repository.
2. Call `rng_minigames.load_registry()` to retrieve the list of available games.
3. When a user launches a game, instantiate `rng_minigames.MinigameContext` with
   your `set_seed` and `notify` callbacks plus any GUI toolkit handles.
4. Pass that context object to the launcher returned by
   `rng_minigames.load_launcher(descriptor.id)`.
5. Offer a **Refresh** button that calls `rng_minigames.refresh_registry()` so
   new games or metadata changes become available without restarting the host.

# Adding a New Game

1. Create a subdirectory (`rng_minigames/<game_id>/`) containing:
   - `metadata.json` with `id`, `name`, `module`, `callable`, and `description`.
   - A module exporting the callable declared in the metadata.
   - Optional storage directories for AI brains, hall-of-fame files, etc.
2. Implement the launcher so it accepts a single `MinigameContext`.
3. Run `python - <<'PY'` sample? or simply call `rng_minigames.refresh_registry()`
   to rebuild the hashes.
4. Add or update tests under `rng_minigames/tests/`.
5. Update this README (and any embedding host documentation) with the gameplay
   summary and new ID.

Following this pattern keeps RNG Mini-games portable—drop the folder into any
Python project, refresh the registry and hook the launchers into your UI.
