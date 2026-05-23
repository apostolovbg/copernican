# Emoji Meteors

Emoji Meteors is the quickest way to capture a playful Copernican seed. Five
animal emojis fall across a twilight sky; you “pet” them by clicking directly
on each sprite. The exact order of the pets, the time you needed to collect the
set and the emoji identities form the payload that is hashed into the final
seed. Because the mini-game runs inside the deterministic `rng_minigames/`
framework, two identical play sessions always produce the same result.

## Overview
This title is intentionally short and accessible so a user can collect a seed
quickly. The interaction model stays simple, but the payload remains fully
deterministic for repeatable runs.

## Controls

- **Pet the animals** – Left-click the meteor you want to claim. The emoji
  enlarges for visual confirmation and is appended to the selection list.
- **Cute Enough** – Commits the current selection, hashes the payload and calls
  `context.set_seed(...)`.
- **Try Again** – Restarts the shower without closing the window so you can
  gather a fresh combination.
- **Cancel** – Closes the mini-game without altering the seed.

The window tracks how many animals are left to pet, highlights collected ones
and disables Cute Enough until all five selections are made. Keyboard focus
cycles through the buttons in top-to-bottom order so the mini-game remains
accessible. When `render=False` (for example, during headless system tests) the
launcher silently generates a deterministic fallback seed instead of showing
the Tk window.

## Tips

- Meteor trails are intentionally wide so you can click anywhere on the sprite.
- There is no penalty for missing a meteor; just wait for another and continue.
- Try Again is faster than cancelling when you want multiple seeds in a row.

For high-level embedding guidance see `rng_minigames/README.md`; the minigame
loads on demand whenever a host calls `load_launcher("emoji_meteors")`.
