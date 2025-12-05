# RNG Mini-games (relocated)

The mini-game documentation now lives under `rng_minigames/README.md` so the
entire RNG toolkit can be dropped into other repositories. Please consult that
file for current gameplay descriptions, the launcher API and contribution
guidelines.

## Quick reference

- `rng_minigames/README.md` – architectural overview, embedding guidance and
  registry notes.
- `rng_minigames/emoji_meteors/README.md`, `rng_minigames/constellation/README.md`
  and `rng_minigames/alien_invasion/README.md` – game-specific docs that describe
  controls, accessibility considerations and configuration files.
- `rng_minigames/alien_invasion/ai_settings.yml` – tune the autopilot’s
  exploration rate, hidden-layer widths (comma-separated list for multi-layer
  brains), history window, kill rewards, time-pressure curve and learning speed
  multiplier without touching Python code. The file is generated on first
  launch with the default `40,32,24,15,12` stack, ignored by Git and reused
  until you delete it to regenerate the defaults.
- `rng_minigames/alien_invasion/_storage/game_settings.yml` – adjust gameplay
  knobs such as player/general shields, movement acceleration and snap
  tolerances, Neutron charge capacity, explosion cadence and debris behaviour.
  The file is generated on first launch; delete it to regenerate the defaults.
  Update-and-save is all that’s required; the launcher reloads the settings on
  every run. Global explosion parameters now govern both player and enemy
  blasts, while `player_explosion.hold_seconds` sets the defeat animation delay
  before an auto-reset and `debris.damages_all` enables or disables
  friendly-fire shrapnel.
- `rng_minigames/CHANGELOG.md` – dedicated history for everything that changes
  inside the RNG bundle.
