# RNG Mini-games (relocated)

The mini-game documentation now lives under `rng_minigames/README.md` so the
entire RNG toolkit can be dropped into other repositories. Please consult that
file for current gameplay descriptions, the launcher API and contribution
guidelines.

## Quick reference

- `rng_minigames/README.md` – authoritative guide for every mini-game plus the
  embedding API.
- `rng_minigames/alien_invasion/ai_settings.yml` – tune the autopilot’s
  exploration rate, kill rewards, time-pressure curve and learning speed
  multiplier without touching Python code.
- `rng_minigames/alien_invasion/game_settings.yml` – adjust gameplay knobs such
  as player/general shields, movement acceleration and snap tolerances, Neutron
  charge capacity, explosion cadence and debris behaviour. Update-and-save is
  all that’s required; the launcher reloads the settings on every run. Global
  explosion parameters now govern both player and enemy blasts, while
  `player_explosion.hold_seconds` sets the defeat animation delay before an
  auto-reset and `debris.damages_all` enables or disables friendly-fire shrapnel.
