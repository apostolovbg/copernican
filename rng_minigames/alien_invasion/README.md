# Alien Invasion

Alien Invasion is a frenetic homage to classic fixed shooters. You defend future
Earth by piloting a hovercraft across the bottom of the screen, dodging barrages
from four lieutenant rows, a shielded line of majors, the heavy colonels and a
perpetually evasive general. Every destroyed ship records its identifier, and
the finalized kill order plus the run duration are hashed into the manifest
seed. Like every title in `rng_minigames/`, the game remains deterministic for a
given sequence of inputs so reproducibility never suffers.

## Controls and Flow

- **Movement** – Move the cursor horizontally; the ship chases your pointer
  using a gentle acceleration/deceleration curve (think “car on ice”) capped at
  twice the general’s max speed. The craft returns to centre when the game
  resets.
- **Fire** – Left-click to shoot. A short built-in cooldown mirrors the CLI
  engine so humans and the autopilot share the same capabilities.
- **Space charges** – Right-click (or Ctrl-click) to expend a stored Neutron
  charge. Capsules fall at random; touching one automatically stores it (up to
  three). Launched charges crawl toward the fleet and detonate in a chain
  reaction that rains debris across the skyline.
- **Status strip** – The bold line above the controls tracks your 50-hit shield,
  the general’s 20-hit shield, current charges and the countdown timer
  (five-minute limit). A secondary line prints current actions (“Capsule stored”,
  “Fleet neutralised!” etc.) so you never wonder what just happened.
- **Buttons** – Use Seed commits the current kill order, Reset restarts
  immediately, Cancel closes the window, Pause halts the simulation, Let AI take
  care hands control to the autopilot and Let AI learn launches back-to-back AI
  sorties. A Hall of Fame button opens `_storage/alien_invasion_hof.yml` inside
  the GUI.

If your shield collapses the craft explodes for the configured hold duration,
after which the game auto-resets (continuous AI learning skips the animation so
training remains fast).

## Gameplay Notes

- Colonels (bottom row) require five hits or one space charge and display a
  thick shield border that thins as they weaken. Majors (the row above) carry a
  single pip, and the lieutenant rows have no armour at all.
- The general can only die after the main fleet is gone or after you land 20
  direct hits. He darts across a dedicated rail, alternates between pressure
  and patrol modes and fires straight down when you linger beneath him.
- Enemy and player darts can destroy each other. Debris harms the player but
  never the invaders unless `debris.damages_all` is enabled in `game_settings`.
- Shooting stars, city skylines, pine clusters and rolling hills are cosmetic,
  but they reinforce the feeling that you are defending Earth.

## Autopilot and Learning Modes

`AlienInvasionAI` lives in `alien_invasion/ai_agent.py` and persists its state in
`_storage/alien_invasion_ai_state.yml`. “Let AI take care” starts a single
autopilot run, while “Let AI learn” loops indefinitely, restarting a new sortie
the moment the previous one ends. “Let AI forget” deletes the stored weights so
the helper relearns from scratch. All autopilot attempts honor the same movement
constraints that human players face, and every stitched run updates both the
hall-of-fame board and the AI statistics banner (runs trained, win rate, average
kills and edge discipline).

## AI Settings Reference (`ai_settings.yml`)

All knobs are hot-reloaded when the game launches, so editing the YAML and
reopening the window is sufficient to try new behaviours.

- `run_duration_seconds` – Hard cap on each sortie. The countdown timer and time
  pressure logic use this value.
- `learning_speed` – Time multiplier applied when “Let AI learn” is active. The
  GUI exposes the same value via the spinbox so you can accelerate or slow down
  learning in real time.
- `exploration_rate` – Probability (0–1) that the AI experiments with random
  movement/shoot/charge decisions instead of following the neural network’s
  recommendation. Higher values help the brain escape local optima (for example,
  edge camping).
- `hidden_units` – Width of the neural network’s hidden layer. Increasing this
  value adds more neurons and yields a richer policy at the cost of additional
  training time and a heavier state file.
- `history_limit` – Number of recorded decision samples the trainer replays when
  adjusting the network after each run. Lower values focus on the most recent
  behaviour, while higher values preserve long-term context.
- `time_pressure` (`base`, `scale`, `exponent`, `fallback`) – Controls how
  urgently the AI acts as the countdown approaches zero. `base` is the minimum
  pressure, `scale` sets how much extra pressure can accumulate, `exponent`
  controls the curve (higher values delay the ramp-up) and `fallback` is used
  whenever the game cannot compute a real fraction (for example, in headless
  mode).
- `kill_reward` (`base`, `general_bonus`, `increment`, `max_increment`) –
  Describes the reinforcement signal applied whenever the AI destroys a ship.
  The general bonus increases the reward for finishing the flagship, and each
  kill adds `increment` until the cumulative bonus reaches `max_increment`.
- `respawn_penalty` (`lieutenant`, `major`, `colonel`) – Deducted whenever a
  destroyed ship respawns (for example, after the player gets hit). Use higher
  values to encourage evasive behaviour.

Tuning these values lets you experiment with aggressive or cautious play styles
without touching Python code. Document every adjustment in
`rng_minigames/CHANGELOG.md` so downstream teams know why the AI behaves
differently.

## Gameplay Settings (`game_settings.yml`)

- `player` / `general` blocks control shield counts and descriptive metadata
  shown in the status bar.
- `player_motion` defines acceleration, deceleration, snap error and maximum
  speed, keeping the feel consistent across hosts.
- `charges`, `explosion`, `player_explosion` and `debris` determine capsule
  capacity, shard counts, animation cadence and whether shrapnel hurts invaders.
  Increasing `violence_scale` or shard counts dramatically changes the look of a
  space charge detonation, so keep the skyline readability in mind.

Each parameter is read on launch, making Alien Invasion just as configurable as
the CLI engines.

## Storage

- `_storage/alien_invasion_ai_state.yml` – Autopilot weights, run counts and
  historical metrics.
- `_storage/alien_invasion_hof.yml` – Hall-of-fame entries (only the 10 fastest
  runs are kept).

Deleting either file resets the corresponding data. Both live alongside the game
so the `rng_minigames/` folder stays self-contained when vendored into other
projects.
