# Mini-game Seed Tools
Last Updated: 2025-12-03

Copernican's Run Builder offers several playful mini-games for generating
random seeds without leaving the GUI. Each game produces a deterministic seed by
capturing the player's interactions and recording the result in the manifest.
All mini-game code now lives in `copernican_lib/gui/minigames/` (one module per
game) and the Run Builder calls their exported `launch_*` helpers from the Seed
page. Cancel is available on every mini-game so you can exit without altering
the draft. The Default, Random timestamp and mini-game buttons are stacked
vertically in the builder so each option is reachable with sequential
keyboard focus.

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
  - You have unlimited lives. When hit, a previously destroyed invader
    regenerates until the grid is full again.
  - Space charges drop slowly; touching one stores it instantly so you can
    right-click to launch a bomb that may trigger a short chain reaction.
  - The roaming general now fires a rapid barrage every few hundred
    milliseconds, so stay mobile whenever the flagship is on screen.
  - The mini-game ends once all invaders are down and the Use Seed button
    becomes available.

## Adding or Updating Mini-games
- Place new mini-game helpers inside `copernican_lib/gui/minigames/` and expose a
  `launch_*` function that accepts `(CopernicanGUI, tk.StringVar)`. Follow the
  existing modal structure (instructions, status labels, Try Again / Cancel /
  confirm buttons plus a headless fallback).
- Update this document, `README.md`, and `AGENTS.md` whenever new mini-games or
  flows are introduced so the behaviour stays discoverable.
