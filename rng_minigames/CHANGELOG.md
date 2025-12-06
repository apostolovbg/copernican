# RNG Mini-games Changelog

-## Version 11.0.1
- 2025-12-06: Ensured `rng_minigames/registry.json` ends with a newline and
            matched the registry checksum so the RNG bundle stays parsable
            (`rng_minigames/registry.json`, `rng_minigames/CHANGELOG.md`).
- 2025-12-06: Guarded the Alien Invasion AI reward sum so failing runs no longer
            reference an uninitialised value, keeping the test harness stable
            (`rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Turned Alien Invasion's AI settings, hall-of-fame and neural state
            files into runtime artifacts so every install keeps its own
            configuration, auto-generated defaults appear on first launch and
            Let AI forget rewrites the state file immediately (`.gitignore`,
            `rng_minigames/alien_invasion/_storage/.gitkeep`,
            `rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/hall_of_fame.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/README.md`, `docs/minigames.md`,
            `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/_storage/alien_invasion_ai_state.yml`,
            `rng_minigames/alien_invasion/_storage/alien_invasion_hof.yml`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Cleaned up lingering lint issues by splitting the explosion
            helpers' `nonlocal` declarations, rewrapping the auxiliary
            docstrings, and shortening the hall-of-fame labels so the
            pre-commit suite (black/flake8) ran cleanly
            (`rng_minigames/alien_invasion/game.py`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/hall_of_fame.py`,
            `rng_minigames/tests/test_registry_and_ai.py`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Ignored the entire `_storage/` folder, removed the placeholder
            `.gitkeep`, relocated `game_settings.yml` into `_storage/` and
            taught the loaders to regenerate AI/game configs on first launch
            while **Let AI forget** now removes the saved state file entirely
            (`rng_minigames/.gitignore`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/game_config.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/README.md`, `docs/minigames.md`,
            `rng_minigames/alien_invasion/game_settings.yml`,
            `rng_minigames/alien_invasion/_storage/.gitkeep`,
            `rng_minigames/alien_invasion/ai_settings.yml`).
- 2025-12-05: Pushed a heavier edge discipline penalty into the autopilot
            (`ai_settings.yml`, `rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_agent.py`, `rng_minigames/alien_invasion/game.py`,
            `rng_minigames/alien_invasion/README.md`, `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Made the general back off to the opposite flank whenever you camp
            near the edge so it doesn’t squeeze you into the corner
            (`rng_minigames/alien_invasion/game.py`, `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: The general now flips into a retreat mode when you hug a corner,
            lurching all the way to the far rail before resuming patrol so you
            get some breathing room (`rng_minigames/alien_invasion/game.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Raised the edge-camping penalty multiplier to 6× and added a
            probabilistic retreat fallback so the general only retreats when
            you are putting him under too much pressure (`rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/game.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Re-tuned every AI runtime default (learning speed, exploration,
            rewards, penalties, edge multiplier) so new installs boot straight
            into a fast-learning, aggressive pilot that cements its strategy
            quickly (`rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Made corner camping unbearable by raising the edge multiplier to
            12×, introducing an escalating streak penalty, and emphasizing the
            new parameters in the docs (`rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Tuned the edge penalty defaults back toward the playable range while adding
            a configurable decay so the AI can recover faster after a few clean runs
            (`rng_minigames/alien_invasion/ai_config.py`, `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/ai_agent.py`, `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Added initial weight, win bonus and caution cap tuning plus story in the docs
            so the pilot launches confidently, boosts aggression on wins and never
            gets stuck as a coward after an edge-heavy loss
            (`rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Added the `kill_time_bonus` multiplier/exponent so rapid kills
            now produce an exponential reward, encouraging the AI to be
            aggressive and finish enemies quickly (`rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Introduced `kill_drought_penalty`, a configurable stick that
            punishes low-kill/long-duration runs so wasting time without scoring
            is a sure way to boost caution (`rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/ai_settings.yml`,
            `rng_minigames/alien_invasion/ai_agent.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Added persistent learning stats, the kill meter and auto-reset so
            small status counters now reflect the all-time history stored in
            `_storage/ai_learning_stats.yml` and the counter resets whenever
            the AI forgets (`rng_minigames/alien_invasion/game.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: General barrages now rest between volleys and his shield drops to
            1 when he is the last enemy alive, and the ship’s motion defaults
            were tuned for a tighter, wobble-free feel so you can dive in for a
            shot (`rng_minigames/alien_invasion/game.py`,
            `rng_minigames/alien_invasion/README.md`, `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Restored the stacked Alien Invasion pilot defaults to a
            five-layer `40,32,24,15,12` network so freshly generated AI config
            files immediately train deeper brains and documented how to supply
            multi-layer lists (`rng_minigames/alien_invasion/ai_config.py`,
            `rng_minigames/alien_invasion/README.md`,
            `rng_minigames/README.md`, `docs/minigames.md`,
            `rng_minigames/CHANGELOG.md`).
- 2025-12-05: Removed stray Finder " 2" copies from the RNG docs and promoted
            the Alien Invasion README duplicate to the canonical filename so
            the bundle only ships one copy of each reference (rng_minigames/
            CHANGELOG.md, rng_minigames/CHANGELOG 2.md,
            rng_minigames/README 2.md,
            rng_minigames/constellation/README 2.md,
            rng_minigames/alien_invasion/README 2.md,
            rng_minigames/alien_invasion/README.md).
- 2025-12-05: Raised the Alien Invasion learning-speed cap to 60x while keeping
            the default 10x multiplier and tuned the fast-learning scheduler so
            continuous training sessions can run up to sixty times faster
            without overwhelming Tk's event loop (rng_minigames/alien_invasion/
            game.py, rng_minigames/CHANGELOG.md).
- 2025-12-05: Let the Alien Invasion autopilot declare up to ten comma-
            separated hidden layer sizes (1–64 neurons each) and upgraded the
            neural trainer to honor multi-layer stacks so downstream configs can
            describe richer brains directly in `ai_settings.yml`
            (rng_minigames/alien_invasion/ai_agent.py,
            rng_minigames/alien_invasion/README.md,
            rng_minigames/tests/test_registry_and_ai.py,
            rng_minigames/CHANGELOG.md).
- 2025-12-05: Updated the default Alien Invasion pilot to a layered
            configuration (40,32,24,15,12 neurons) so fresh installs immediately
            train deeper brains without editing YAML by hand
            (rng_minigames/alien_invasion/ai_settings.yml,
            rng_minigames/CHANGELOG.md).
- 2025-12-05: Created the dedicated RNG changelog and broke documentation out
            into per-game README files so the marsupial project can be vendored
            independently (rng_minigames/CHANGELOG.md, rng_minigames/README.md,
            rng_minigames/emoji_meteors/README.md,
            rng_minigames/constellation/README.md,
            rng_minigames/alien_invasion/README.md).
- 2025-12-05: Exposed the Alien Invasion AI’s hidden-layer width and training
            history depth through `ai_settings.yml`, taught the neural helper to
            rebuild its network when those knobs change and documented the new
            configuration points for downstream projects (rng_minigames/
            alien_invasion/ai_agent.py, rng_minigames/alien_invasion/
            ai_config.py, rng_minigames/alien_invasion/ai_settings.yml,
            rng_minigames/README.md, docs/minigames.md, CHANGELOG.md).
- 2025-12-04: Wired Alien Invasion’s shields, motion limits and the shared
             explosion/debris behaviour to `game_settings.yml`, reintroduced a
             gentle “car on ice” player movement model, added defeat auto-reset
             timing (with instant restarts during AI learning) and documented
             the new `player_explosion`/`debris` knobs so downstream apps can
             rebalance without code edits (rng_minigames/alien_invasion/game.py,
             rng_minigames/alien_invasion/game_settings.yml,
             rng_minigames/README.md, docs/minigames.md, CHANGELOG.md).
- 2025-12-04: Removed the stray rightmost craft from the staggered second row in
             Alien Invasion so the formation stays centered and no ships spawn
             beyond the intended flight lane (rng_minigames/alien_invasion/
             game.py, CHANGELOG.md).
- 2025-12-04: Rewired the Alien Invasion AI incentives so victories boost
             aggression/charge far more than defeats while tracking the
             “Everybody lives/dies” counts that show up in the stats banner
             (rng_minigames/alien_invasion/ai_agent.py,
             rng_minigames/alien_invasion/game.py, rng_minigames/README.md,
             CHANGELOG.md).
- 2025-12-04: Fixed the Alien Invasion regressions introduced by the accelerated
             learning mode: shooting stars now pause during Let AI learn runs,
             the scheduler no longer crashes before enemies spawn and stars are
             cleared when toggling modes (rng_minigames/alien_invasion/game.py,
             CHANGELOG.md).
- 2025-12-04: Updated the RNG mini-game launcher to reload modules on every run
             so editing a game no longer requires restarting the GUI. The Seed
             page now imports each mini-game on demand without caching, and the
             documentation explains the hot-reload behaviour (copernican_lib/
             gui/app.py, rng_minigames/registry.py, README.md, AGENTS.md,
             rng_minigames/README.md, CHANGELOG.md).
- 2025-12-04: Ensured Alien Invasion's AI records every autopilot session
             (single-run or continuous learning) and clamped the AI pilot to the
             same speed limit as human players so training remains fair
             (rng_minigames/alien_invasion/game.py,
             rng_minigames/README.md, README.md, AGENTS.md, CHANGELOG.md).
- 2025-12-04: Added in-window Pause/Resume controls, continuous **Let AI learn**
             loops, a **Let AI forget** dialog, an AI games counter and smarter
             victory/defeat handling so practice runs reset automatically while
             hall-of-fame and seed workflows stay intact (rng_minigames/
             alien_invasion/game.py, rng_minigames/README.md, README.md,
             AGENTS.md, docs/gui_guide.md).
- 2025-12-04: Extended the Alien Invasion AI helper with a forget/reset API and
             expanded the regression tests to cover the wipe behaviour
             (copernican_lib/gui/minigames/alien_invasion/ai_agent.py,
             tests/test_minigames_modules.py, CHANGELOG.md).
- 2025-12-04: Added the Alien Invasion autopilot toggle, hall-of-fame modal and
             live runtime counter plus refreshed the README/AGENTS/gui guide
             docs so the AI helper, cache files and scoreboard are documented
             properly (copernican_lib/gui/minigames/alien_invasion/game.py,
             docs/minigames.md, README.md, AGENTS.md, docs/gui_guide.md,
             CHANGELOG.md).
- 2025-12-04: Introduced regression tests that exercise the Alien Invasion AI
             brain and hall-of-fame persistence so the new mini-game modules
             stay covered (tests/test_minigames_modules.py, CHANGELOG.md).
- 2025-12-04: Extended the Alien Invasion backdrop with continuous sky fill,
             guaranteed twin skyline valleys and multiple pine forests so each
             playfield shows at least two cities and ridge clusters hiding
             their trunks below the hills (copernican_lib/gui/minigames/
             alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Refined the Alien Invasion skyline with symmetric lit windows,
             denser clustered pines, scattered tiny circular bushes tucked
             general movement boundaries so the flagship stays on-screen
             (copernican_lib/gui/minigames/alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Made the general’s flight path more evasive and staggered the
             second invader row (offset half a ship with one fewer craft) so
             barrages can’t pin the flagship or wipe columns in straight lines
             (copernican_lib/gui/minigames/alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Added a 20-hit player shield with matching status display, enabled
             dart-vs-dart interceptions plus debris-clearing shots, and renamed
             the counter line to highlight Neutron charge stockpiles
             (copernican_lib/gui/minigames/alien_invasion.py,
             docs/minigames.md, CHANGELOG.md).
- 2025-12-04: Rebuilt the general’s movement with a horizontal-rail AI so the
             flagship glides from edge to edge, dodges players proactively and
             never clips outside the playfield (copernican_lib/gui/minigames/
             alien_invasion.py, docs/minigames.md, CHANGELOG.md).
- 2025-12-04: Deferred mini-game imports until players open them so GUI launch
             times stay snappy even as the games grow (copernican_lib/gui/app.py,
             CHANGELOG.md).
- 2025-12-04: Toughened Alien Invasion again by standardising the staggered
             second row (15 evenly spaced ships), doubling Neutron capsule
             drops, bumping the pilot shield to 50 HP, biasing revived fleets to
             the far side, and prioritising lieutenant respawns before higher
             ranks so heavy cruisers return only after the fodder is restored
             (copernican_lib/gui/minigames/alien_invasion.py,
             docs/minigames.md, CHANGELOG.md).
- 2025-12-04: Fixed the Alien Invasion regression that paused all timers by
             referencing an undefined margin constant when spawning the general
             (copernican_lib/gui/minigames/alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Locked Alien Invasion ship placement to their original slots so
             respawns no longer add extra columns or shift rows after hits
             (copernican_lib/gui/minigames/alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Patched the Alien Invasion respawn helper so it no longer references
             the local ``start_y`` variable outside its scope, preventing Tk
             crashes during shield hits (copernican_lib/gui/minigames/
             alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Updated the Alien Invasion victory logic to recalc total enemies
             after the staggered formations are built, allowing the game to end
             once the last ship is destroyed (copernican_lib/gui/minigames/
             alien_invasion.py, CHANGELOG.md).
- 2025-12-04: Fixed the alien spawn regression introduced by the new row
             staggering (the non-general clamp now runs after the general flag
             is set) so enemies no longer vanish when the game loads
             (copernican_lib/gui/minigames/alien_invasion.py, CHANGELOG.md).
- 2025-12-03: Added the Emoji meteors seed generator so the Run Builder can
             derive playful random seeds from animal emoji trios alongside the
             timestamp helper (copernican_lib/gui/app.py, CHANGELOG.md).
- 2025-12-03: Expanded Emoji meteors into an interactive canvas with larger
             falling animals, “Cute enough”/“Try again” controls, and an enlarged
             selection preview so seed picking feels like a mini-game
             (copernican_lib/gui/app.py, CHANGELOG.md).
- 2025-12-03: Widened the Emoji meteors playfield, doubled the animal count and
             moved the instructions near the selection controls so players pet
             three animals before confirming the seed (copernican_lib/gui/app.py,
             CHANGELOG.md).
- 2025-12-03: Added the Constellation and Alien Invasion mini-games,
             refreshed Emoji Meteors to require five selections, added Cancel
             buttons to every mini-game, and documented the system under
             docs/minigames.md with README/AGENTS cross-references
             (copernican_lib/gui/app.py, docs/minigames.md, README.md, AGENTS.md,
             CHANGELOG.md).
- 2025-12-03: Stacked the seed helper buttons vertically, moved every
             mini-game into `copernican_lib/gui/minigames/` modules and updated
             the README, GUI guide and mini-game documentation so contributors
             know where the helpers live (copernican_lib/gui/app.py,
             copernican_lib/gui/minigames/__init__.py,
             copernican_lib/gui/minigames/emoji_meteors.py,
             copernican_lib/gui/minigames/constellation_connect.py,
             copernican_lib/gui/minigames/alien_invasion.py,
             docs/minigames.md, docs/gui_guide.md, README.md, AGENTS.md,
             CHANGELOG.md).
- 2025-12-03: Added two more invader rows, randomized shooter selection,
             automatic capsule pickup and refreshed instructions for the Alien
             Invasion seed mini-game so charges are easier to collect and attack
             patterns less predictable (copernican_lib/gui/minigames/alien_invasion.py,
             docs/minigames.md, CHANGELOG.md).
- 2025-12-03: Fixed the Alien Invasion timer bug, centered every formation and
             gave the general a dedicated rapid-fire cycle so the flagship
             peppers players with constant shots (copernican_lib/gui/minigames/alien_invasion.py,
             docs/minigames.md, CHANGELOG.md).
- 2025-12-03: Made Alien Invasion’s space charges rarer with guaranteed
             explosions, added falling debris hazards, prevented launching more
             than one charge at a time, and gave the general a 50-hit shield so
             he only falls once the fleet is gone (copernican_lib/gui/minigames/alien_invasion.py,
             docs/minigames.md, CHANGELOG.md).
- 2025-12-03: Improved space-charge launching (supporting trackpad bindings),
             halved the general’s average fire rate while making his barrages and
             movement erratic, reskinned the battlefield with a moonlit gradient
             sky plus hills, shrank all ships, added charge/general counters and
             heavy-plated bottom-row cruisers that take five hits to defeat
             (copernican_lib/gui/minigames/alien_invasion.py, docs/minigames.md,
             CHANGELOG.md).
- 2025-12-03: Documented the GUI mini-games in the contributor guide and GUI
             guide so both humans and AI helpers know where Emoji Meteors,
             Constellation and Alien Invasion live and where to extend them
             (AGENTS.md, docs/gui_guide.md, docs/minigames.md, CHANGELOG.md).
