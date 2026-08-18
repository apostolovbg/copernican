# Alien Invasion

Alien Invasion is a frenetic homage to classic fixed shooters. You defend
future Earth by piloting a hovercraft across the bottom of the screen,
dodging barrages from four lieutenant rows, a shielded line of majors, the
heavy colonels and a perpetually evasive general. Every destroyed ship
records its identifier, and the finalized kill order plus the run duration are
hashed into the manifest seed. Like every title in
`copernican/rng_minigames/`, the game remains deterministic for a given
sequence of inputs so reproducibility never suffers.

## Overview
This title combines mouse-driven combat, persistent AI training and runtime
storage under `_storage/`. The launcher can run human play, autopilot and
learning loops without changing the deterministic seed contract.

## Controls and Flow

- **Movement** – Move the cursor horizontally; the ship chases your pointer
  using a gentle acceleration/deceleration curve (think “car on ice”) capped at
  twice the general’s max speed. The craft returns to centre when the game
  resets.
- The acceleration/deceleration curve now feels tighter so the ship responds
  immediately to your mouse movements instead of wobbling.
- **Fire** – Left-click to shoot. A short built-in cooldown mirrors the CLI
  firing mode so humans and the autopilot share the same capabilities.
- **Space charges** – Right-click (or Ctrl-click) to expend a stored Neutron
  charge. Capsules fall at random; touching one automatically stores it (up to
  three). Launched charges crawl toward the fleet and detonate in a chain
  reaction that rains debris across the skyline.
- **Status strip** – The bold line above the controls tracks your 50-hit
  shield, the general’s 20-hit shield, current charges and the countdown timer
  (five-minute limit). A secondary line prints current actions (“Capsule
  stored”, “Fleet neutralised!” etc.) so you never wonder what just happened.
- **Buttons** – Use Seed commits the current kill order, Reset restarts
  immediately, Cancel closes the window, Pause halts the simulation, Let AI
  take care hands control to the autopilot and Let AI learn launches back-to-
  back AI sorties. A Hall of Fame button opens
  `_storage/alien_invasion_hof.yml` inside the GUI.

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
- The general now pauses between barrages, so you can dive under him to draw
  shots, and once every other enemy is gone his shield drops to a single dart
  so you can finish him cleanly.
- Enemy and player darts can destroy each other. Debris harms the player but
  never the invaders unless `debris.damages_all` is enabled in
  `_storage/game_settings.yml`.
- The general now retreats all the way to the opposite rail whenever you camp
  near a corner so you get a breather instead of staying pinned.
- The general now drifts toward the opposite rail whenever you linger near a
  corner, giving you a little breathing space instead of pinning you by the
  wall.
- Shooting stars, city skylines, pine clusters and rolling hills are cosmetic,
  but they reinforce the feeling that you are defending Earth.

## Autopilot and Learning Modes

`AlienInvasionAI` lives in `alien_invasion/ai_agent.py` and persists its state
in `_storage/alien_invasion_ai_state.yml`. The file is created automatically on
the first launch (and ignored by Git) so every install keeps its own pilot
history. “Let AI take care” starts a single autopilot run, while “Let AI learn”
loops indefinitely, restarting a new sortie the moment the previous one ends.
“Let AI forget” deletes the stored weights so the helper relearns from scratch.
All autopilot attempts honor the same movement constraints that human players
face, and every stitched run updates both the hall-of-fame board and the AI
statistics banner (runs trained, win rate, average kills and edge discipline).
The default brain now ships with a five-layer stack (40,32,24,15,12 neurons) so
fresh installs immediately enjoy a deeper, more aggressive pilot.

## AI Settings Reference (`ai_settings.yml`)

Copernican writes `ai_settings.yml` next to the game the first time you open
the window. Deleting it regenerates the default template; otherwise the file is
reused between runs so your preferred configuration sticks even after pulling
updates.

All knobs are hot-reloaded when the game launches, so editing the YAML and
reopening the window is sufficient to try new behaviours.

- `run_duration_seconds` – Hard cap on each sortie. The countdown timer and
  time pressure logic use this value.
- `learning_speed` – Time multiplier applied when “Let AI learn” is active. The
  GUI exposes the same value via the spinbox so you can accelerate or slow down
  learning in real time.
- `exploration_rate` – Probability (0–1) that the AI experiments with random
  movement/shoot/charge decisions instead of following the neural network’s
  recommendation. Higher values help the brain escape local optima (for
  example, edge camping).
- `hidden_units` – Comma-separated list or YAML array describing each hidden
  layer. The default stack is `40,32,24,15,12`, giving the pilot five layers of
  progressively narrower neurons. Add/remove entries to reshape the network;
  wider/deeper layouts amplify learning capacity at the cost of training time.
- `history_limit` – Number of recorded decision samples the trainer replays
  when adjusting the network after each run. Lower values focus on the most
  recent behaviour, while higher values preserve long-term context.
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
- `edge_penalty_multiplier` – Scales how harshly the autoscaler punishes
  excursions near the screen edges; higher numbers make edge-camping more
  expensive so the AI spends more time defending the centre line. The default
  factor is now tuned so corner hugging still hurts but clean runs can recover.
- `edge_streak_scale` – Applies a stacking penalty whenever the AI stays near
  the edge for multiple frames; staying pinned in the corner rapidly saturates
  the extra cost.
- `edge_streak_decay` – Controls how quickly the accumulated streak penalty
  unwinds once the AI leaves the wall; higher values let you clean the streak
  faster so the brain can bounce back between runs.
- `kill_time_bonus` – Parameters (`multiplier`, `exponent`) that apply an
  exponential reward when the pilot kills more enemies in less time so fast,
  aggressive sessions get amplified.
- `kill_drought_penalty` – Applies a growing penalty whenever a run ends with
  only a few kills for the time spent, so time-wasting sessions feel worse than
  they did before.
- `initial_weights` – Starting aggression/caution/charge weights. The defaults
  now bias the brain toward aggression from the very first run so it doesn’t
  retreat into cowardice before it learns to kill.
- `win_bonus` – Manual adjustments applied after every successful run so
  aggression/charge gets a boost and caution drops slightly, reinforcing wins.
- `loss_caution_cap` – Prevents a single loss from inflating caution
  indefinitely.

Tuning these values lets you experiment with aggressive or cautious play
styles without touching Python code. The bundled defaults now bias toward
aggressive, fast-learning behavior (high learning speed, steeper kill
rewards, softer respawn penalties, and a heavy edge penalty) so the pilot
quickly becomes a challenging, human-beating opponent. Document every
adjustment in `copernican/rng_minigames/CHANGELOG.md` so downstream teams know
why the AI behaves differently.

## Gameplay Settings (`_storage/game_settings.yml`)

Copernican writes `game_settings.yml` into `_storage/` alongside the other
runtime files. Deleting it regenerates the defaults on the next launch.

- `player` / `general` blocks control shield counts and descriptive metadata
  shown in the status bar.
- `player_motion` defines acceleration, deceleration, snap error and maximum
  speed, keeping the feel consistent across hosts.
- `charges`, `explosion`, `player_explosion` and `debris` determine capsule
  capacity, shard counts, animation cadence and whether shrapnel hurts
  invaders. Increasing `violence_scale` or shard counts dramatically changes
  the look of a space charge detonation, so keep the skyline readability in
  mind.

Each parameter is read on launch, making Alien Invasion just as configurable
as the command-line gameplay workflows.

## Storage

- `_storage/alien_invasion_ai_state.yml` – Autopilot weights, run counts and
  historical metrics.
- `_storage/alien_invasion_hof.yml` – Hall-of-fame entries (only the 10 fastest
  runs are kept).
- `_storage/ai_learning_stats.yml` – Cumulative training stats (runs, wins,
  losses, kill averages, edge discipline) that persist across sessions unless
  you reset the AI helper.
- `_storage/game_settings.yml` – User-tuned gameplay knobs (shields, debris,
  motion parameters, etc.).

Deleting any of these files resets the corresponding data. The AI state file
is deleted automatically when you choose **Let AI forget**, the hall of fame
entry list is safe to clear whenever you want a fresh scoreboard, and
`game_settings` regenerates with defaults whenever it is removed. All runtime
data lives inside `_storage/` so the `copernican/rng_minigames/` folder stays
self-contained when vendored.
