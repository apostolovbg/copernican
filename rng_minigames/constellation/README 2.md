# Constellation

Constellation asks you to sketch a ten-star pattern on a dense celestial map.
Every star you select is connected with smooth lines so you can literally watch
your constellation form. Once the tenth star is in place the game hashes the
path, the click order and the elapsed time to produce a deterministic random
seed. The mechanic mirrors Emoji Meteors’ determinism while rewarding a steadier
pace.

## Controls

- **Left-click** to select a star. The halo brightens to confirm the choice and
  the running counter increments.
- **Right-click** (or Ctrl-click) to undo the most recent selection if you want
  to adjust the pattern.
- **Ad astra!** finalises the constellation and emits the seed once ten stars
  are selected.
- **Try Again** clears the canvas so you can draw a new pattern immediately.
- **Cancel** closes the window without touching the manifest seed.

The status label always reports “Stars connected: X/10” so you know how far you
are from completion. Like the rest of `rng_minigames/`, the launcher reloads
this module on demand, so editing the code and reopening the window is enough to
test new visual tweaks.

## Tips

- Stars can be anywhere on the field; there is no requirement to form a convex
  shape.
- Undo operations only remove the most recent selection, keeping the UX
  predictable.
- Time is part of the hashed payload, so slow, deliberate runs generate
  different seeds from frantic ones even if you pick the same stars.

See the top-level `rng_minigames/README.md` for details about the registry,
hot-reload workflow and host integration.
